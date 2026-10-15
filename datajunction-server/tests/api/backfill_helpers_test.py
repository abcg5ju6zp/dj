"""
Unit tests for the backfill scheduling/reconciliation endpoint helpers.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from datajunction_server.api.materializations import (
    _reconcile_active_run,
    _schedule_backfill,
)
from datajunction_server.database.backfill import Backfill
from datajunction_server.database.materialization import Materialization
from datajunction_server.errors import DJBackfillScheduleUncertainException
from datajunction_server.models.materialization import MaterializationInfo
from datajunction_server.models.node_type import NodeType
from datajunction_server.models.partition import BackfillStatus, PartitionBackfill


@pytest.mark.asyncio
async def test_schedule_uncertain_without_workflow_name_stays_pending():
    """
    A legacy materialization with no recorded workflow name cannot reconcile a
    lost response: the run stays PENDING for explicit recovery and nothing is
    queried or launched implicitly.
    """
    backfill = Backfill(
        id=1,
        workflow_name=None,
        idempotency_key="bf-x",
        status=BackfillStatus.PENDING.value,
    )
    materialization = Materialization(name="m", workflow_names=None)
    job = MagicMock()
    job.run_backfill.side_effect = DJBackfillScheduleUncertainException(message="slow")
    clazz = MagicMock(return_value=job)
    query_service_client = MagicMock()

    info = await _schedule_backfill(
        backfill,
        materialization,
        [],
        clazz,
        query_service_client,
        None,
    )
    assert info == MaterializationInfo(output_tables=[], urls=[])
    assert backfill.status == BackfillStatus.PENDING.value
    query_service_client.get_workflow_status.assert_not_called()


@pytest.mark.asyncio
async def test_reconcile_active_run_without_name_is_noop():
    """No pinned name means there is nothing to reconcile against."""
    session = AsyncMock()
    backfill = Backfill(
        workflow_name=None,
        status=BackfillStatus.RUNNING.value,
    )
    materialization = Materialization(name="m", workflow_names=None)
    query_service_client = MagicMock()

    await _reconcile_active_run(
        session,
        backfill,
        materialization,
        query_service_client,
        None,
    )
    query_service_client.get_workflow_status.assert_not_called()
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_reconcile_pending_run_resubmits_when_workflow_absent(mocker):
    """
    A PENDING run whose named workflow the service does not know is scheduled
    once; adoption of the new answer marks it running.
    """

    class FakeJob:
        def run_backfill(
            self,
            materialization,
            partitions,
            query_service_client,
            **kwargs,
        ):
            return query_service_client.run_backfill(
                "default.hard_hat",
                "v1.0",
                NodeType.DIMENSION,
                materialization.name,
                partitions,
                **kwargs,
            )

    mocker.patch(
        "datajunction_server.materialization.jobs.materialization_job"
        ".MaterializationJob.__subclasses__",
        return_value=[FakeJob],
    )

    session = AsyncMock()
    backfill = Backfill(
        id=3,
        workflow_name="wf.backfill",
        idempotency_key="bf-3",
        status=BackfillStatus.PENDING.value,
        spec=[
            PartitionBackfill(
                column_name="birth_date",
                range=["20230101", "20230101"],
            ).model_dump(),
        ],
    )
    materialization = Materialization(
        name="spark_sql__incremental_time__birth_date",
        job="FakeJob",
        workflow_names=["wf.main", "wf.backfill"],
    )
    query_service_client = MagicMock()
    query_service_client.get_workflow_status.return_value = None
    query_service_client.run_backfill.return_value = MaterializationInfo(
        urls=["http://fake.url/retry"],
        output_tables=[],
    )

    await _reconcile_active_run(
        session,
        backfill,
        materialization,
        query_service_client,
        None,
    )
    assert query_service_client.run_backfill.call_count == 1
    assert backfill.status == BackfillStatus.RUNNING.value
    assert backfill.urls == ["http://fake.url/retry"]
    session.commit.assert_awaited_once()
