"""
End-to-end tests for generation-tracked, idempotent node materialization backfills.
"""

from unittest import mock

import pytest
from httpx import AsyncClient

from datajunction_server.errors import DJBackfillScheduleUncertainException
from datajunction_server.models.access import ResourceAction, ResourceType
from datajunction_server.models.materialization import MaterializationInfo
from datajunction_server.models.partition import (
    BackfillStatus,
    Granularity,
    WorkflowRunState,
    WorkflowStatusInfo,
)
from datajunction_server.service_clients import QueryServiceClient
from tests.authz import VALIDATOR_AUTH_SERVICE, RecordingAuthorizationService

MAT_POST = {
    "job": "spark_sql",
    "strategy": "incremental_time",
    "config": {},
    "schedule": "0 * * * *",
}


@pytest.fixture(autouse=True)
def workflow_mocks(
    module__query_service_client: QueryServiceClient,
    mocker,
):
    """Pin named workflows on schedule and reset run/status mocks per test."""
    module__query_service_client.materialize.return_value = MaterializationInfo(
        urls=["http://fake.url/job"],
        output_tables=["default.roads.scheduled_output"],
        workflow_names=[
            "dj.nodematerialization.main",
            "dj.nodematerialization.backfill",
        ],
    )
    module__query_service_client.run_backfill.reset_mock()
    module__query_service_client.run_backfill.side_effect = None
    module__query_service_client.run_backfill.return_value = MaterializationInfo(
        urls=["http://fake.url/backfill"],
        output_tables=[],
    )
    mocker.patch.object(
        module__query_service_client,
        "get_workflow_status",
        new=mock.MagicMock(
            return_value=WorkflowStatusInfo(
                workflow_name="dj.nodematerialization.backfill",
                state=WorkflowRunState.RUNNING,
            ),
        ),
    )
    yield module__query_service_client


@pytest.fixture
def set_temporal_column():
    async def _set(client: AsyncClient, node_name: str, column: str):
        response = await client.post(
            f"/nodes/{node_name}/columns/{column}/partition",
            json={
                "type_": "temporal",
                "granularity": Granularity.DAY,
                "format": "yyyyMMdd",
            },
        )
        assert response.status_code in (200, 201)

    return _set


async def _setup_materialization(
    client: AsyncClient,
    node_name: str,
    column: str,
    set_temporal_column,
    *,
    config=None,
):
    await set_temporal_column(client, node_name, column)
    payload = dict(MAT_POST)
    payload["config"] = config or {}
    response = await client.post(
        f"/nodes/{node_name}/materialization/",
        json=payload,
    )
    assert response.status_code in (200, 201), response.text
    return f"spark_sql__incremental_time__{column}"


@pytest.mark.asyncio
async def test_backfill_pins_generation_and_retries_adopt_the_run(
    module__client_with_roads: AsyncClient,
    module__query_service_client: QueryServiceClient,
    set_temporal_column,
):
    """
    The run records version/generation/range/plan/workflow/output; an identical
    retry adopts it and never launches a second workflow.
    """
    node = "default.hard_hat"
    mat_name = await _setup_materialization(
        module__client_with_roads,
        node,
        "birth_date",
        set_temporal_column,
    )
    node_version = (await module__client_with_roads.get(f"/nodes/{node}/")).json()[
        "version"
    ]

    response = await module__client_with_roads.post(
        f"/nodes/{node}/materializations/{mat_name}/backfill",
        json=[{"column_name": "birth_date", "range": ["20230101", "20230103"]}],
    )
    assert response.status_code == 201
    assert response.json() == {
        "output_tables": [],
        "urls": ["http://fake.url/backfill"],
        "workflow_names": ["dj.nodematerialization.backfill"],
    }
    assert module__query_service_client.run_backfill.call_count == 1
    kwargs = module__query_service_client.run_backfill.call_args.kwargs
    assert kwargs["workflow_name"] == "dj.nodematerialization.backfill"
    assert kwargs["idempotency_key"].startswith("bf-")

    listed = (
        await module__client_with_roads.get(
            f"/nodes/{node}/materializations/{mat_name}/backfills/",
        )
    ).json()
    assert len(listed) == 1
    run = listed[0]
    assert run["status"] == BackfillStatus.RUNNING
    assert run["generation"] == 1
    assert run["node_version"] == node_version
    assert run["workflow_name"] == "dj.nodematerialization.backfill"
    assert run["temporal_range"] == ["20230101", "20230103"]
    assert run["planned_partitions"] == [
        "birth_date=20230101",
        "birth_date=20230102",
        "birth_date=20230103",
    ]

    # An identical retry reconciles against the live workflow and schedules
    # nothing new.
    response = await module__client_with_roads.post(
        f"/nodes/{node}/materializations/{mat_name}/backfill",
        json=[{"column_name": "birth_date", "range": ["20230101", "20230103"]}],
    )
    assert response.status_code == 201
    assert module__query_service_client.run_backfill.call_count == 1
    assert module__query_service_client.get_workflow_status.call_count == 1

    detail = (
        await module__client_with_roads.get(
            f"/nodes/{node}/materializations/{mat_name}/backfills/{run['id']}",
        )
    ).json()
    assert detail["id"] == run["id"]
    assert detail["status"] == BackfillStatus.RUNNING


@pytest.mark.asyncio
async def test_overlapping_same_generation_refused_disjoint_range_accepted(
    module__client_with_roads: AsyncClient,
    module__query_service_client: QueryServiceClient,
    set_temporal_column,
):
    node = "default.hard_hat_2"
    mat_name = await _setup_materialization(
        module__client_with_roads,
        node,
        "birth_date",
        set_temporal_column,
    )

    response = await module__client_with_roads.post(
        f"/nodes/{node}/materializations/{mat_name}/backfill",
        json=[{"column_name": "birth_date", "range": ["20230101", "20230131"]}],
    )
    assert response.status_code == 201

    response = await module__client_with_roads.post(
        f"/nodes/{node}/materializations/{mat_name}/backfill",
        json=[{"column_name": "birth_date", "range": ["20230115", "20230215"]}],
    )
    assert response.status_code == 409
    assert "overlapping" in response.json()["message"]

    response = await module__client_with_roads.post(
        f"/nodes/{node}/materializations/{mat_name}/backfill",
        json=[{"column_name": "birth_date", "range": ["20230301", "20230305"]}],
    )
    assert response.status_code == 201
    assert module__query_service_client.run_backfill.call_count == 2


@pytest.mark.asyncio
async def test_generation_bump_supersedes_old_run_and_refuses_its_confirmations(
    module__client_with_roads: AsyncClient,
    module__query_service_client: QueryServiceClient,
    set_temporal_column,
):
    node = "default.hard_hat_to_delete"
    mat_name = await _setup_materialization(
        module__client_with_roads,
        node,
        "birth_date",
        set_temporal_column,
    )

    response = await module__client_with_roads.post(
        f"/nodes/{node}/materializations/{mat_name}/backfill",
        json=[{"column_name": "birth_date", "range": ["20230101", "20230131"]}],
    )
    assert response.status_code == 201
    old_id = (
        await module__client_with_roads.get(
            f"/nodes/{node}/materializations/{mat_name}/backfills/",
        )
    ).json()[0]["id"]

    # Redefine the materialization: generation advances.
    response = await module__client_with_roads.post(
        f"/nodes/{node}/materialization/",
        json={
            "job": "spark_sql",
            "strategy": "incremental_time",
            "config": {"lookback_window": "3 DAY"},
            "schedule": "0 * * * *",
        },
    )
    assert response.status_code in (200, 201), response.text

    # A confirmation from the old generation cannot publish onto the new one.
    response = await module__client_with_roads.post(
        f"/nodes/{node}/materializations/{mat_name}/backfills/{old_id}/completions",
        json={"partitions": [{"values": ["20230101"]}]},
    )
    assert response.status_code == 409

    # A new overlapping run supersedes the old one instead of conflicting.
    response = await module__client_with_roads.post(
        f"/nodes/{node}/materializations/{mat_name}/backfill",
        json=[{"column_name": "birth_date", "range": ["20230115", "20230215"]}],
    )
    assert response.status_code == 201
    runs = (
        await module__client_with_roads.get(
            f"/nodes/{node}/materializations/{mat_name}/backfills/",
        )
    ).json()
    by_id = {run["id"]: run for run in runs}
    assert by_id[old_id]["status"] == BackfillStatus.SUPERSEDED
    new_run = next(run for run in runs if run["status"] == BackfillStatus.RUNNING)
    assert new_run["generation"] == 2

    # One confirmed, one failed: availability covers only the confirmed day.
    response = await module__client_with_roads.post(
        f"/nodes/{node}/materializations/{mat_name}/backfills/"
        f"{new_run['id']}/completions",
        json={
            "partitions": [
                {
                    "values": ["20230115"],
                    "catalog": "default",
                    "schema": "roads",
                    "table": "hard_hat_to_delete_materialized",
                    "valid_through_ts": 1673827199,
                },
                {"values": ["20230116"], "succeeded": False},
            ],
        },
    )
    assert response.status_code == 200, response.text
    updated = response.json()
    assert updated["confirmed_partitions"] == ["birth_date=20230115"]
    assert updated["failed_partitions"] == ["birth_date=20230116"]
    assert updated["published_partitions"] == ["birth_date=20230115"]

    availability = (
        await module__client_with_roads.get(f"/nodes/{node}/availability/")
    ).json()
    current = next(
        state
        for state in availability
        if state["table"] == "hard_hat_to_delete_materialized"
    )
    assert current["min_temporal_partition"] == ["20230115"]
    assert current["max_temporal_partition"] == ["20230115"]

    # Recovery cross-checks availability records: the confirmed day is not
    # republished and nothing is recomputed.
    response = await module__client_with_roads.post(
        f"/nodes/{node}/materializations/{mat_name}/backfills/{new_run['id']}/recover",
    )
    assert response.status_code == 200
    recovered = response.json()
    assert recovered["published_partitions"] == ["birth_date=20230115"]
    assert recovered["confirmed_partitions"] == ["birth_date=20230115"]
    # The original submission is the only workflow the run ever launched.
    assert module__query_service_client.run_backfill.call_count == 2

    # Another overlapping submission walks past the superseded run and still
    # conflicts with the active generation-2 run.
    response = await module__client_with_roads.post(
        f"/nodes/{node}/materializations/{mat_name}/backfill",
        json=[{"column_name": "birth_date", "range": ["20230120", "20230125"]}],
    )
    assert response.status_code == 409


@pytest.mark.asyncio
async def test_backfill_endpoints_reject_unknown_materialization(
    module__client_with_roads: AsyncClient,
):
    response = await module__client_with_roads.get(
        "/nodes/default.hard_hat/materializations/no_such_mat/backfills/",
    )
    assert response.status_code == 404


async def _run_id_for_range(client, node, mat_name, temporal_range):
    runs = (
        await client.get(
            f"/nodes/{node}/materializations/{mat_name}/backfills/",
        )
    ).json()
    return next(run["id"] for run in runs if run["temporal_range"] == temporal_range)


@pytest.mark.asyncio
async def test_lost_scheduling_response_queries_workflow_before_resubmitting(
    module__client_with_roads: AsyncClient,
    module__query_service_client: QueryServiceClient,
    set_temporal_column,
):
    node = "default.repair_order"
    mat_name = await _setup_materialization(
        module__client_with_roads,
        node,
        "order_date",
        set_temporal_column,
    )
    qs = module__query_service_client

    # Timeout, but the original workflow exists: adopt it, do not resubmit.
    qs.run_backfill.side_effect = [
        DJBackfillScheduleUncertainException(message="timed out"),
    ]
    qs.get_workflow_status.return_value = WorkflowStatusInfo(
        workflow_name="dj.nodematerialization.backfill",
        state=WorkflowRunState.RUNNING,
        urls=["http://fake.url/adopted"],
        output_tables=["default.roads.repair_order_m"],
    )
    response = await module__client_with_roads.post(
        f"/nodes/{node}/materializations/{mat_name}/backfill",
        json=[{"column_name": "order_date", "range": ["20230101", "20230103"]}],
    )
    assert response.status_code == 201
    assert response.json()["urls"] == ["http://fake.url/adopted"]
    assert qs.run_backfill.call_count == 1
    assert qs.get_workflow_status.call_count == 1

    # Duplicate request while the workflow runs schedules nothing new.
    response = await module__client_with_roads.post(
        f"/nodes/{node}/materializations/{mat_name}/backfill",
        json=[{"column_name": "order_date", "range": ["20230101", "20230103"]}],
    )
    assert response.status_code == 201
    assert qs.run_backfill.call_count == 1

    # Timeout and the workflow genuinely does not exist: one resubmit.
    qs.run_backfill.side_effect = [
        DJBackfillScheduleUncertainException(message="timed out"),
        MaterializationInfo(urls=["http://fake.url/retry"], output_tables=[]),
    ]
    qs.get_workflow_status.return_value = None
    response = await module__client_with_roads.post(
        f"/nodes/{node}/materializations/{mat_name}/backfill",
        json=[{"column_name": "order_date", "range": ["20230301", "20230303"]}],
    )
    assert response.status_code == 201
    assert response.json()["urls"] == ["http://fake.url/retry"]
    assert qs.run_backfill.call_count == 3
    assert qs.get_workflow_status.call_count == 3


@pytest.mark.asyncio
async def test_pending_recover_resubmits_once_and_missing_workflow_marks_failed(
    module__client_with_roads: AsyncClient,
    module__query_service_client: QueryServiceClient,
    set_temporal_column,
):
    node = "default.repair_order"
    mat_name = "spark_sql__incremental_time__order_date"
    await _setup_materialization(
        module__client_with_roads,
        node,
        "order_date",
        set_temporal_column,
    )
    qs = module__query_service_client

    # Both the scheduling call and the immediate resubmit lose their answer:
    # the run stays PENDING.
    qs.run_backfill.side_effect = [
        DJBackfillScheduleUncertainException(message="timed out"),
        DJBackfillScheduleUncertainException(message="timed out again"),
    ]
    qs.get_workflow_status.return_value = None
    response = await module__client_with_roads.post(
        f"/nodes/{node}/materializations/{mat_name}/backfill",
        json=[{"column_name": "order_date", "range": ["20230601", "20230602"]}],
    )
    assert response.status_code == 504
    june_id = await _run_id_for_range(
        module__client_with_roads,
        node,
        mat_name,
        ["20230601", "20230602"],
    )
    detail = (
        await module__client_with_roads.get(
            f"/nodes/{node}/materializations/{mat_name}/backfills/{june_id}",
        )
    ).json()
    assert detail["status"] == BackfillStatus.PENDING

    # A recovery whose resubmission also loses its answer leaves the run
    # PENDING without raising.
    qs.run_backfill.side_effect = [
        DJBackfillScheduleUncertainException(message="still gone"),
    ]
    response = await module__client_with_roads.post(
        f"/nodes/{node}/materializations/{mat_name}/backfills/{june_id}/recover",
    )
    assert response.status_code == 200
    assert response.json()["status"] == BackfillStatus.PENDING

    # The next recovery, once the service answers, schedules it exactly once.
    qs.run_backfill.side_effect = None
    qs.run_backfill.return_value = MaterializationInfo(
        urls=["http://fake.url/recovered"],
        output_tables=[],
    )
    response = await module__client_with_roads.post(
        f"/nodes/{node}/materializations/{mat_name}/backfills/{june_id}/recover",
    )
    assert response.status_code == 200
    assert response.json()["status"] == BackfillStatus.RUNNING

    # A running workflow the service no longer knows leaves unconfirmed
    # partitions marked failed rather than ever covering the range.
    qs.get_workflow_status.return_value = None
    response = await module__client_with_roads.post(
        f"/nodes/{node}/materializations/{mat_name}/backfill",
        json=[{"column_name": "order_date", "range": ["20230701", "20230702"]}],
    )
    assert response.status_code == 201
    july_id = await _run_id_for_range(
        module__client_with_roads,
        node,
        mat_name,
        ["20230701", "20230702"],
    )
    response = await module__client_with_roads.post(
        f"/nodes/{node}/materializations/{mat_name}/backfills/{july_id}/recover",
    )
    assert response.status_code == 200
    recovered = response.json()
    assert recovered["status"] == BackfillStatus.FAILED
    assert set(recovered["failed_partitions"]) == {
        "order_date=20230701",
        "order_date=20230702",
    }
    assert recovered["published_partitions"] == []


@pytest.mark.asyncio
async def test_rejected_scheduling_can_be_retried_on_the_same_run(
    module__client_with_roads: AsyncClient,
    module__query_service_client: QueryServiceClient,
    set_temporal_column,
):
    node = "default.repair_order"
    mat_name = "spark_sql__incremental_time__order_date"
    await _setup_materialization(
        module__client_with_roads,
        node,
        "order_date",
        set_temporal_column,
    )
    qs = module__query_service_client

    # The query service rejects (for example: no backfill workflow): the
    # historical empty result is returned and the run is recorded FAILED.
    qs.run_backfill.return_value = MaterializationInfo(output_tables=[], urls=[])
    response = await module__client_with_roads.post(
        f"/nodes/{node}/materializations/{mat_name}/backfill",
        json=[{"column_name": "order_date", "range": ["20230801", "20230801"]}],
    )
    assert response.status_code == 201
    assert response.json() == {
        "output_tables": [],
        "urls": [],
        "workflow_names": [],
    }
    august = await _run_id_for_range(
        module__client_with_roads,
        node,
        mat_name,
        ["20230801", "20230801"],
    )
    assert (
        await module__client_with_roads.get(
            f"/nodes/{node}/materializations/{mat_name}/backfills/{august}",
        )
    ).json()["status"] == BackfillStatus.FAILED

    # Once the workflow exists, the same request retries that record.
    qs.run_backfill.return_value = MaterializationInfo(
        urls=["http://fake.url/backfill"],
        output_tables=[],
    )
    response = await module__client_with_roads.post(
        f"/nodes/{node}/materializations/{mat_name}/backfill",
        json=[{"column_name": "order_date", "range": ["20230801", "20230801"]}],
    )
    assert response.status_code == 201
    assert response.json()["urls"] == ["http://fake.url/backfill"]
    assert (
        await module__client_with_roads.get(
            f"/nodes/{node}/materializations/{mat_name}/backfills/{august}",
        )
    ).json()["status"] == BackfillStatus.RUNNING


@pytest.mark.asyncio
async def test_recover_adopts_finished_workflow_and_publishes_without_rerun(
    module__client_with_roads: AsyncClient,
    module__query_service_client: QueryServiceClient,
    set_temporal_column,
):
    """
    Recovery combines the Backfill row, external workflow state and
    availability records: finished partitions are adopted and published, and
    the workflow is never submitted again. The disjoint May range keeps this
    run apart from the timeout test's January/March runs on the same node.
    """
    node = "default.repair_order"
    mat_name = "spark_sql__incremental_time__order_date"
    # The materialization (and order_date partition) already exists from the
    # timeout test; the POST is idempotent when the config is unchanged.
    await _setup_materialization(
        module__client_with_roads,
        node,
        "order_date",
        set_temporal_column,
    )
    qs = module__query_service_client

    response = await module__client_with_roads.post(
        f"/nodes/{node}/materializations/{mat_name}/backfill",
        json=[{"column_name": "order_date", "range": ["20230501", "20230503"]}],
    )
    assert response.status_code == 201
    run_id = (
        await module__client_with_roads.get(
            f"/nodes/{node}/materializations/{mat_name}/backfills/",
        )
    ).json()
    run_id = next(
        run["id"] for run in run_id if run["temporal_range"] == ["20230501", "20230503"]
    )

    planned = [
        "order_date=20230501",
        "order_date=20230502",
        "order_date=20230503",
    ]

    # Simulate the crash window: the workflow finished, and only the first
    # day's availability landed before the restart.
    response = await module__client_with_roads.post(
        f"/data/{node}/availability/",
        json={
            "catalog": "default",
            "schema_": "roads",
            "table": "repair_order_m",
            "valid_through_ts": 1672531200,
            "categorical_partitions": [],
            "temporal_partitions": ["order_date"],
            "min_temporal_partition": ["20230501"],
            "max_temporal_partition": ["20230501"],
            "partitions": [
                {
                    "value": [],
                    "min_temporal_partition": ["20230501"],
                    "max_temporal_partition": ["20230501"],
                    "valid_through_ts": 1672531200,
                },
            ],
        },
    )
    assert response.status_code == 200, response.text

    # The workflow reports all three days and the output it writes to:
    # recovery adopts them, keeps day 1's availability as-is and publishes
    # only days 2 and 3 -- no resubmission, no recomputation.
    qs.get_workflow_status.return_value = WorkflowStatusInfo(
        workflow_name="dj.nodematerialization.backfill",
        state=WorkflowRunState.SUCCEEDED,
        output_tables=["default.roads.repair_order_m"],
        completed_partitions=planned,
        failed_partitions=[],
    )
    before_runs = qs.run_backfill.call_count
    response = await module__client_with_roads.post(
        f"/nodes/{node}/materializations/{mat_name}/backfills/{run_id}/recover",
    )
    assert response.status_code == 200, response.text
    recovered = response.json()
    assert recovered["status"] == BackfillStatus.COMPLETE
    assert recovered["confirmed_partitions"] == planned
    assert recovered["published_partitions"] == planned
    assert qs.run_backfill.call_count == before_runs

    availability = (
        await module__client_with_roads.get(f"/nodes/{node}/availability/")
    ).json()
    current = next(
        state for state in availability if state["table"] == "repair_order_m"
    )
    assert current["min_temporal_partition"] == ["20230501"]
    assert current["max_temporal_partition"] == ["20230503"]

    # A second recovery is publish-only and workflow-free -- even once the
    # workflow itself is gone, a COMPLETE run neither resubmits nor regresses.
    qs.get_workflow_status.return_value = None
    response = await module__client_with_roads.post(
        f"/nodes/{node}/materializations/{mat_name}/backfills/{run_id}/recover",
    )
    assert response.status_code == 200
    assert response.json()["status"] == BackfillStatus.COMPLETE
    assert qs.run_backfill.call_count == before_runs

    # A completed run no longer accepts partition outcomes.
    response = await module__client_with_roads.post(
        f"/nodes/{node}/materializations/{mat_name}/backfills/{run_id}/completions",
        json={"partitions": [{"values": ["20230501"]}]},
    )
    assert response.status_code == 409

    # But submitting the same completed plan again answers idempotently with
    # the recorded run and schedules nothing.
    response = await module__client_with_roads.post(
        f"/nodes/{node}/materializations/{mat_name}/backfill",
        json=[{"column_name": "order_date", "range": ["20230501", "20230503"]}],
    )
    assert response.status_code == 201
    assert qs.run_backfill.call_count == before_runs


@pytest.mark.asyncio
async def test_pending_recovery_without_workflow_name_resubmits(
    module__client_with_roads: AsyncClient,
    module__query_service_client: QueryServiceClient,
    module__session,
    set_temporal_column,
):
    """
    A materialization predating workflow-name recording (or a run whose name
    was never pinned) is recovered by scheduling directly: no status lookup is
    attempted and exactly one submission happens.
    """
    from sqlalchemy import update

    from datajunction_server.database.backfill import Backfill as BackfillRow

    node = "default.repair_order"
    mat_name = "spark_sql__incremental_time__order_date"
    await _setup_materialization(
        module__client_with_roads,
        node,
        "order_date",
        set_temporal_column,
    )
    qs = module__query_service_client
    response = await module__client_with_roads.post(
        f"/nodes/{node}/materializations/{mat_name}/backfill",
        json=[{"column_name": "order_date", "range": ["20230901", "20230902"]}],
    )
    assert response.status_code == 201
    run_id = await _run_id_for_range(
        module__client_with_roads,
        node,
        mat_name,
        ["20230901", "20230902"],
    )

    # Simulate a crash on a legacy row: nothing about the workflow was pinned.
    await module__session.execute(
        update(BackfillRow)
        .where(BackfillRow.id == run_id)
        .values(workflow_name=None, status=BackfillStatus.PENDING.value),
    )
    await module__session.commit()

    before = qs.run_backfill.call_count
    response = await module__client_with_roads.post(
        f"/nodes/{node}/materializations/{mat_name}/backfills/{run_id}/recover",
    )
    assert response.status_code == 200
    assert response.json()["status"] == BackfillStatus.RUNNING
    assert qs.run_backfill.call_count == before + 1
    # No status lookup is attempted without a pinned name.
    assert qs.get_workflow_status.call_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "suffix,body",
    [
        ("/backfills/1/completions", {"partitions": []}),
        ("/backfills/1/recover", None),
    ],
)
async def test_backfill_run_mutation_denies_without_write(
    module__client_with_roads: AsyncClient,
    mocker,
    suffix,
    body,
):
    """Generation-tracked run mutations are governed by WRITE on the node."""
    service = RecordingAuthorizationService(denied=ResourceAction.WRITE)
    mocker.patch(VALIDATOR_AUTH_SERVICE, lambda: service)
    base = "/nodes/enforce_test/materializations/enforce_test"
    response = await module__client_with_roads.post(f"{base}{suffix}", json=body)
    assert response.status_code == 403
    checked = {
        (request.verb, request.access_object.resource_type, request.access_object.name)
        for request in service.requests
    }
    assert (ResourceAction.WRITE, ResourceType.NODE, "enforce_test") in checked
