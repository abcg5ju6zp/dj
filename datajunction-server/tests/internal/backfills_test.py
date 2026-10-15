"""
Tests for generation-tracked backfill orchestration helpers.
"""

import types
from unittest import mock

import pytest

from datajunction_server.database.availabilitystate import AvailabilityState
from datajunction_server.database.backfill import Backfill
from datajunction_server.database.column import Column
from datajunction_server.database.materialization import Materialization
from datajunction_server.database.node import Node, NodeRevision
from datajunction_server.database.partition import Partition
from datajunction_server.errors import (
    DJBackfillConflictException,
    DJInvalidInputException,
)
from datajunction_server.internal import backfills
from datajunction_server.models.materialization import MaterializationInfo
from datajunction_server.models.node_type import NodeType
from datajunction_server.models.partition import (
    BackfillPartitionCompletion,
    BackfillStatus,
    Granularity,
    PartitionBackfill,
    PartitionType,
    WorkflowRunState,
    WorkflowStatusInfo,
)
from datajunction_server.sql.parsing.types import StringType, TimestampType


def _revision(
    *,
    temporal: tuple[str, str] | None = ("birth_date", "yyyyMMdd"),
    categorical: str | None = None,
    node_type: NodeType = NodeType.DIMENSION,
) -> NodeRevision:
    columns = []
    if temporal:
        columns.append(
            Column(
                name=temporal[0],
                display_name=temporal[0],
                type=StringType(),
                partition=Partition(
                    type_=PartitionType.TEMPORAL,
                    granularity=Granularity.DAY,
                    format=temporal[1],
                ),
            ),
        )
    if categorical:
        columns.append(
            Column(
                name=categorical,
                display_name=categorical,
                type=StringType(),
                partition=Partition(type_=PartitionType.CATEGORICAL),
            ),
        )
    return NodeRevision(
        name="default.hard_hat",
        type=node_type,
        columns=columns,
    )


def test_expand_plan_enumerates_temporal_range_inclusive():
    """The concrete plan fixes every partition coordinate scheduling saw."""
    revision = _revision()
    keys, temporal_range = backfills.expand_plan(
        revision,
        [
            {
                "column_name": "birth_date",
                "range": ["20230101", "20230103"],
                "values": None,
            }
        ],
    )
    assert keys == [
        "birth_date=20230101",
        "birth_date=20230102",
        "birth_date=20230103",
    ]
    assert temporal_range == ["20230101", "20230103"]


def test_expand_plan_cross_product_with_categorical():
    """Categorical values fan the temporal range out per category."""
    revision = _revision(categorical="country")
    keys, _ = backfills.expand_plan(
        revision,
        [
            {
                "column_name": "birth_date",
                "range": ["20230101", "20230101"],
                "values": None,
            },
            {"column_name": "country", "values": ["US", "DE"], "range": None},
        ],
    )
    assert keys == [
        "birth_date=20230101|country=US",
        "birth_date=20230101|country=DE",
    ]


def test_expand_plan_opaque_unit_when_format_cannot_enumerate():
    """An unparseable range stays one opaque unit rather than exploding."""
    revision = _revision(temporal=("birth_date", None))
    keys, _ = backfills.expand_plan(
        revision,
        [{"column_name": "birth_date", "range": ["x", "y"], "values": None}],
    )
    assert keys == ["birth_date=[x..y]"]


def test_idempotency_key_is_stable_and_generation_scoped():
    """Retry hashes identically; a new generation or plan is a different run."""
    canonical = backfills.canonical_spec(
        [
            PartitionBackfill(
                column_name="birth_date",
                range=["20230101", "20230201"],
            ),
        ],
    )
    first = backfills.make_idempotency_key(7, 1, canonical)
    reordered = backfills.make_idempotency_key(7, 1, list(reversed(canonical)))
    assert first == reordered
    newer_generation = backfills.make_idempotency_key(7, 2, canonical)
    assert newer_generation != first


def test_specs_overlap_detects_temporal_intersection():
    revision = _revision()
    assert backfills.specs_overlap(
        revision,
        [
            {
                "column_name": "birth_date",
                "range": ["20230101", "20230201"],
                "values": None,
            }
        ],
        [
            {
                "column_name": "birth_date",
                "range": ["20230115", "20230301"],
                "values": None,
            }
        ],
    )
    assert not backfills.specs_overlap(
        revision,
        [
            {
                "column_name": "birth_date",
                "range": ["20230101", "20230131"],
                "values": None,
            }
        ],
        [
            {
                "column_name": "birth_date",
                "range": ["20230201", "20230301"],
                "values": None,
            }
        ],
    )


def test_specs_overlap_respects_disjoint_categoricals():
    revision = _revision(categorical="country")
    assert not backfills.specs_overlap(
        revision,
        [{"column_name": "country", "values": ["US"], "range": None}],
        [{"column_name": "country", "values": ["DE"], "range": None}],
    )
    assert backfills.specs_overlap(
        revision,
        [{"column_name": "country", "values": ["US", "DE"], "range": None}],
        [{"column_name": "country", "values": ["DE"], "range": None}],
    )


def test_record_completions_partitions_the_plan():
    """Only planned coordinates are accepted; successes and failures partition."""
    revision = _revision()
    backfill = Backfill(
        planned_partitions=["birth_date=20230101", "birth_date=20230102"],
        confirmed_partitions=[],
        failed_partitions=[],
        published_partitions=[],
    )
    backfills.record_completions(
        revision,
        backfill,
        [
            BackfillPartitionCompletion(values=["20230101"]),
            BackfillPartitionCompletion(values=["20230102"], succeeded=False),
        ],
    )
    assert backfill.confirmed_partitions == ["birth_date=20230101"]
    assert backfill.failed_partitions == ["birth_date=20230102"]

    with pytest.raises(DJInvalidInputException):
        backfills.record_completions(
            revision,
            backfill,
            [BackfillPartitionCompletion(values=["20990101"])],
        )


def test_finalize_status_requires_publication():
    """Confirmed-but-unpublished is not COMPLETE; a failed run stays partial."""
    backfill = Backfill(
        status=BackfillStatus.RUNNING.value,
        planned_partitions=["a=1", "a=2"],
        confirmed_partitions=["a=1"],
        failed_partitions=[],
        published_partitions=[],
    )
    assert backfills.finalize_status(backfill) == BackfillStatus.RUNNING

    backfill.published_partitions = ["a=1"]
    backfill.failed_partitions = ["a=2"]
    assert backfills.finalize_status(backfill) == BackfillStatus.PARTIALLY_COMPLETE

    backfill.failed_partitions = []
    backfill.confirmed_partitions = ["a=1", "a=2"]
    backfill.published_partitions = ["a=1", "a=2"]
    assert backfills.finalize_status(backfill) == BackfillStatus.COMPLETE


def test_finalize_status_failed_workflow_marks_unaccounted_partial():
    backfill = Backfill(
        status=BackfillStatus.RUNNING.value,
        planned_partitions=["a=1", "a=2"],
        confirmed_partitions=["a=1"],
        failed_partitions=[],
        published_partitions=["a=1"],
    )
    assert (
        backfills.finalize_status(backfill, WorkflowRunState.FAILED)
        == BackfillStatus.PARTIALLY_COMPLETE
    )


def test_ensure_generation_current_rejects_old_runs():
    backfill = Backfill(generation=1, status=BackfillStatus.RUNNING.value)
    materialization = Materialization(name="m", generation=2)
    with pytest.raises(DJBackfillConflictException):
        backfills.ensure_generation_current(backfill, materialization)


def test_adopt_workflow_progress_keeps_to_plan():
    backfill = Backfill(
        workflow_name="wf",
        planned_partitions=["a=1", "a=2"],
        confirmed_partitions=[],
        failed_partitions=[],
    )
    backfills.adopt_workflow_progress(
        backfill,
        WorkflowStatusInfo(
            workflow_name="wf",
            state=WorkflowRunState.RUNNING,
            completed_partitions=["a=1", "a=99"],
            failed_partitions=["a=2", "a=88"],
        ),
    )
    assert backfill.confirmed_partitions == ["a=1"]
    assert backfill.failed_partitions == ["a=2"]


@pytest.mark.asyncio
async def test_publish_and_recovery_does_not_republish_or_recompute(
    session,
    current_user,
):
    """
    A failed partition must not publish whole-range availability; a partition
    already covered by availability records is published neither again nor
    recomputed.
    """
    revision = _revision()
    revision.name = "default.generation_backfill_ut"
    revision.version = "v1.0"
    revision.created_by_id = current_user.id
    revision.node = Node(
        name=revision.name,
        type=revision.type,
        current_version="v1.0",
        created_by_id=current_user.id,
    )
    materialization = Materialization(
        node_revision=revision,
        name="spark_sql__incremental_time__birth_date",
        schedule="@daily",
        config={},
        generation=1,
    )
    session.add(materialization)
    await session.commit()

    backfill = Backfill(
        materialization=materialization,
        spec=[
            {
                "column_name": "birth_date",
                "range": ["20230101", "20230102"],
                "values": None,
            },
        ],
        generation=1,
        materialization_version="v1.0",
        workflow_name="dj.hard_hat.backfill",
        planned_partitions=["birth_date=20230101", "birth_date=20230102"],
        confirmed_partitions=["birth_date=20230101", "birth_date=20230102"],
        failed_partitions=[],
        published_partitions=[],
        output_tables=[],
    )
    session.add(backfill)
    await session.commit()

    await session.refresh(revision, ["availability"])

    # Only one of the two partitions gets confirmed through the API path; the
    # other failed, so availability must cover the one day only.
    backfill.failed_partitions = ["birth_date=20230102"]
    backfill.confirmed_partitions = ["birth_date=20230101"]
    due = await backfills.publish_due_partitions(
        session,
        revision,
        backfill,
        [
            BackfillPartitionCompletion(
                values=["20230101"],
                catalog="default",
                schema_="roads",
                table="hard_hat_materialized",
                valid_through_ts=1672617599,
            ),
        ],
    )
    assert due == ["birth_date=20230101"]
    await session.commit()
    await session.refresh(revision, ["availability"])

    availability = revision.availability
    assert availability is not None
    assert availability.catalog == "default"
    assert availability.schema_ == "roads"
    assert availability.table == "hard_hat_materialized"
    assert availability.min_temporal_partition == ["20230101"]
    assert availability.max_temporal_partition == ["20230101"]

    # The availability record already covers the confirmed day, so a second
    # publish pass republishes nothing and leaves the confirmed set untouched.
    second_due = await backfills.publish_due_partitions(session, revision, backfill)
    assert second_due == []
    assert backfill.confirmed_partitions == ["birth_date=20230101"]
    assert backfills.availability_covers_key(
        availability,
        "birth_date=20230101",
        revision,
    )
    assert not backfills.availability_covers_key(
        availability,
        "birth_date=20230102",
        revision,
    )


# ---------------------------------------------------------------------------
# Enumeration edge cases
# ---------------------------------------------------------------------------


def test_enumerate_temporal_separator_and_hour_format():
    assert backfills.enumerate_temporal_values(
        "yyyy-MM-dd HH",
        Granularity.HOUR,
        "2023-01-01 00",
        "2023-01-01 02",
    ) == ["2023-01-01 00", "2023-01-01 01", "2023-01-01 02"]


def test_enumerate_temporal_month_quarter_year():
    assert backfills.enumerate_temporal_values(
        "yyyyMM",
        Granularity.MONTH,
        "202301",
        "202303",
    ) == ["202301", "202302", "202303"]
    assert backfills.enumerate_temporal_values(
        "yyyyMMdd",
        Granularity.QUARTER,
        "20230101",
        "20230701",
    ) == ["20230101", "20230401", "20230701"]
    assert backfills.enumerate_temporal_values(
        "yyyy",
        Granularity.YEAR,
        "2023",
        "2024",
    ) == ["2023", "2024"]


def test_enumerate_temporal_rejects_bad_inputs():
    assert (
        backfills.enumerate_temporal_values(
            "yyyyMMdd",
            Granularity.DAY,
            "20230103",
            "20230101",
        )
        is None
    )
    assert backfills.enumerate_temporal_values(None, Granularity.DAY, "a", "b") is None
    assert (
        backfills.enumerate_temporal_values(
            "yyyyMMdd",
            Granularity.DAY,
            "x",
            "y",
        )
        is None
    )


def test_enumerate_temporal_caps_runaway_ranges():
    assert (
        backfills.enumerate_temporal_values(
            "yyyyMMddHHmmss",
            Granularity.SECOND,
            "20200101000000",
            "20300101000000",
        )
        is None
    )


def test_expand_plan_treats_categorical_range_as_bounds():
    revision = _revision(temporal=None, categorical="country")
    keys, temporal_range = backfills.expand_plan(
        revision,
        [{"column_name": "country", "range": ["US", "DE"], "values": None}],
    )
    assert keys == ["country=US", "country=DE"]
    assert temporal_range is None


def test_parse_coordinate_key_ignores_segments_without_value():
    assert backfills.parse_coordinate_key("a=1|marker") == {"a": "1"}


def test_expand_plan_entry_without_values_or_range_is_skipped():
    revision = _revision()
    keys, _ = backfills.expand_plan(
        revision,
        [{"column_name": "birth_date", "values": None, "range": None}],
    )
    assert keys == []


def test_expand_plan_skips_unmentioned_partition_columns_and_empty_plan():
    revision = _revision(categorical="country")
    keys, _ = backfills.expand_plan(
        revision,
        [{"column_name": "birth_date", "range": ["20230101", "20230101"], "values": None}],
    )
    assert keys == ["birth_date=20230101"]
    assert backfills.expand_plan(revision, []) == ([], None)


def test_bounds_overlap_unbounded_entry_covers_everything():
    revision = _revision_with_temporal("yyyyMMdd")
    assert backfills.specs_overlap(
        revision,
        [{"column_name": "birth_date", "values": None, "range": None}],
        [{"column_name": "birth_date", "range": ["20230101", "20230101"], "values": None}],
    )


def test_valid_through_ts_unparseable_value_returns_none():
    column = _revision_with_temporal("yyyyMMdd").columns[0]
    assert backfills._valid_through_ts(column, "not-a-date", None) is None


def test_availability_covers_extra_temporal_axis_is_skipped():
    revision = _revision()
    revision.columns.append(
        Column(
            name="other_date",
            display_name="other_date",
            type=StringType(),
            partition=Partition(
                type_=PartitionType.TEMPORAL,
                granularity=Granularity.DAY,
                format="yyyyMMdd",
            ),
        ),
    )
    availability = AvailabilityState(
        catalog="default",
        table="t",
        valid_through_ts=0,
        temporal_partitions=["birth_date", "other_date"],
        partitions=[
            {
                "value": [],
                "min_temporal_partition": ["20230101", "20230201"],
                "max_temporal_partition": ["20230103", "20230203"],
                "valid_through_ts": None,
            },
        ],
    )
    # The coordinate only names birth_date; the record's other axis is ignored.
    assert backfills.availability_covers_key(
        availability,
        "birth_date=20230102",
        revision,
    )
    assert not backfills.availability_covers_key(
        availability,
        "birth_date=20230105",
        revision,
    )


# ---------------------------------------------------------------------------
# Overlap edge cases
# ---------------------------------------------------------------------------


def _revision_with_temporal(format_: str | None):
    return _revision(temporal=("birth_date", format_))


def test_specs_overlap_uses_values_bounds():
    revision = _revision_with_temporal("yyyyMMdd")
    assert backfills.specs_overlap(
        revision,
        [{"column_name": "birth_date", "values": ["20230101", "20230105"], "range": None}],
        [{"column_name": "birth_date", "values": ["20230105"], "range": None}],
    )
    assert not backfills.specs_overlap(
        revision,
        [{"column_name": "birth_date", "values": ["20230101"], "range": None}],
        [{"column_name": "birth_date", "values": ["20230201"], "range": None}],
    )


def test_specs_overlap_unbounded_axis_covers_all():
    revision = _revision_with_temporal("yyyyMMdd")
    assert backfills.specs_overlap(revision, [], [])


def test_specs_overlap_numeric_format_fallback():
    revision = _revision_with_temporal(None)
    assert backfills.specs_overlap(
        revision,
        [{"column_name": "birth_date", "range": [1, 3], "values": None}],
        [{"column_name": "birth_date", "range": [3, 5], "values": None}],
    )
    assert not backfills.specs_overlap(
        revision,
        [{"column_name": "birth_date", "range": [1, 2], "values": None}],
        [{"column_name": "birth_date", "range": [3, 5], "values": None}],
    )


def test_specs_overlap_raw_string_fallback():
    revision = _revision_with_temporal("yyyyMMdd")
    assert not backfills.specs_overlap(
        revision,
        [{"column_name": "birth_date", "range": ["a", "b"], "values": None}],
        [{"column_name": "birth_date", "range": ["c", "d"], "values": None}],
    )
    assert backfills.specs_overlap(
        revision,
        [{"column_name": "birth_date", "range": ["a", "c"], "values": None}],
        [{"column_name": "birth_date", "range": ["b", "d"], "values": None}],
    )


# ---------------------------------------------------------------------------
# Confirmation validation and status derivation
# ---------------------------------------------------------------------------


def test_assignment_from_values_requires_planned_arity():
    revision = _revision()
    backfill = Backfill(planned_partitions=["birth_date=20230101"])
    with pytest.raises(DJInvalidInputException):
        backfills.assignment_from_values(
            revision,
            backfill.planned_partitions,
            ["20230101", "US"],
        )


def test_finalize_status_sticky_terminal_states():
    backfill = Backfill(
        status=BackfillStatus.FAILED.value,
        planned_partitions=["a=1"],
    )
    assert backfills.finalize_status(backfill) == BackfillStatus.FAILED
    backfill.status = BackfillStatus.SUPERSEDED.value
    assert backfills.finalize_status(backfill) == BackfillStatus.SUPERSEDED


def test_finalize_status_failed_workflow_with_nothing_confirmed():
    backfill = Backfill(
        status=BackfillStatus.RUNNING.value,
        planned_partitions=["a=1", "a=2"],
        confirmed_partitions=[],
        failed_partitions=["a=1", "a=2"],
        published_partitions=[],
    )
    assert (
        backfills.finalize_status(backfill, WorkflowRunState.FAILED)
        == BackfillStatus.FAILED
    )


def test_finalize_status_succeeded_with_partial_reporting_stays_running():
    backfill = Backfill(
        status=BackfillStatus.RUNNING.value,
        planned_partitions=["a=1", "a=2"],
        confirmed_partitions=["a=1"],
        failed_partitions=[],
        published_partitions=["a=1"],
    )
    assert (
        backfills.finalize_status(backfill, WorkflowRunState.SUCCEEDED)
        == BackfillStatus.RUNNING
    )


def test_is_generation_current():
    materialization = Materialization(name="m", generation=2)
    assert backfills.is_generation_current(
        Backfill(generation=2),
        materialization,
    )
    assert not backfills.is_generation_current(
        Backfill(generation=1),
        materialization,
    )


# ---------------------------------------------------------------------------
# Adoption / location / coverage helpers
# ---------------------------------------------------------------------------


def test_adopt_workflow_info_dedupes_and_picks_names():
    backfill = Backfill(urls=["u0"], output_tables=["t0"], workflow_name=None)
    backfills.adopt_workflow_info(
        backfill,
        MaterializationInfo(
            urls=["u0", "u1"],
            output_tables=["t0", "t1"],
            workflow_names=["wf.main", "wf.backfill"],
        ),
    )
    assert backfill.urls == ["u0", "u1"]
    assert backfill.output_tables == ["t0", "t1"]
    assert backfill.workflow_name == "wf.backfill"

    # A second adoption dedupes again; a name already pinned is never replaced.
    backfills.adopt_workflow_info(
        backfill,
        MaterializationInfo(
            urls=["u1"],
            output_tables=["t1"],
            workflow_names=["other.main"],
        ),
    )
    assert backfill.urls == ["u0", "u1"]
    assert backfill.workflow_name == "wf.backfill"


def test_adopt_workflow_info_fallback_name():
    backfill = Backfill(workflow_name=None)
    backfills.adopt_workflow_info(
        backfill,
        MaterializationInfo(urls=[], output_tables=[], workflow_names=["only.name"]),
    )
    assert backfill.workflow_name == "only.name"


def test_adopt_workflow_progress_without_pinned_plan_accepts_all():
    backfill = Backfill(
        planned_partitions=[],
        confirmed_partitions=[],
        failed_partitions=[],
    )
    backfills.adopt_workflow_progress(
        backfill,
        WorkflowStatusInfo(
            workflow_name="wf",
            completed_partitions=["a=1"],
            failed_partitions=["a=2", "a=3"],
        ),
    )
    assert backfill.confirmed_partitions == ["a=1"]
    assert backfill.failed_partitions == ["a=2", "a=3"]


def test_valid_through_ts_branches():
    day_column = _revision().columns[0]
    assert backfills._valid_through_ts(day_column, "20230101", 42) == 42
    computed = backfills._valid_through_ts(day_column, "20230101", None)
    assert computed is not None and computed > 1672531200

    month_column = Column(
        name="m",
        type=TimestampType(),
        partition=Partition(
            type_=PartitionType.TEMPORAL,
            granularity=Granularity.MONTH,
            format="yyyyMM",
        ),
    )
    assert backfills._valid_through_ts(month_column, "202301", None) is not None

    unformatted = Column(
        name="u",
        type=StringType(),
        partition=Partition(type_=PartitionType.TEMPORAL, format=None),
    )
    assert backfills._valid_through_ts(unformatted, "123", None) == 123
    assert backfills._valid_through_ts(unformatted, "abc", None) is None


def _revision_with_availability(availability):
    revision = _revision(categorical="country")
    revision.availability = availability
    return revision


def test_availability_covers_key_temporal_range_and_categorical():
    availability = AvailabilityState(
        catalog="default",
        table="t",
        valid_through_ts=0,
        categorical_partitions=["country"],
        temporal_partitions=["birth_date"],
        partitions=[
            {
                "value": ["US"],
                "min_temporal_partition": ["20230101"],
                "max_temporal_partition": ["20230103"],
                "valid_through_ts": None,
            },
        ],
    )
    revision = _revision_with_availability(availability)
    assert backfills.availability_covers_key(
        availability,
        "birth_date=20230102|country=US",
        revision,
    )
    assert not backfills.availability_covers_key(
        availability,
        "birth_date=20230102|country=DE",
        revision,
    )
    assert not backfills.availability_covers_key(
        availability,
        "birth_date=20230105|country=US",
        revision,
    )

    short = AvailabilityState(
        catalog="default",
        table="t",
        valid_through_ts=0,
        categorical_partitions=["country"],
        partitions=[
            {
                "value": [None],
                "min_temporal_partition": [],
                "max_temporal_partition": [],
                "valid_through_ts": None,
            },
        ],
    )
    assert not backfills.availability_covers_key(
        short,
        "birth_date=20230101",
        revision,
    )


def test_location_for_sources_precedence():
    completion = BackfillPartitionCompletion(
        values=["20230101"],
        catalog="reported",
        schema_="roads",
        table="reported_t",
        url="u",
        valid_through_ts=1,
    )
    revision = types.SimpleNamespace(availability=None)
    backfill = Backfill(output_tables=[])
    assert backfills._location_for(revision, backfill, [completion]) == (
        "reported",
        "roads",
        "reported_t",
        "u",
        1,
    )

    existing = types.SimpleNamespace(
        catalog="old",
        schema_="old_s",
        table="old_t",
        url="old_u",
    )
    assert backfills._location_for(
        types.SimpleNamespace(availability=existing),
        Backfill(output_tables=[]),
        [],
    ) == ("old", "old_s", "old_t", "old_u", None)

    assert backfills._location_for(
        types.SimpleNamespace(availability=None),
        Backfill(output_tables=["a.b.c"]),
        [],
    ) == ("a", "b", "c", None, None)
    assert backfills._location_for(
        types.SimpleNamespace(availability=None),
        Backfill(output_tables=["c"]),
        [],
    ) == ("default", None, "c", None, None)

    with pytest.raises(DJInvalidInputException):
        backfills._location_for(
            types.SimpleNamespace(availability=None),
            Backfill(output_tables=[]),
            [],
        )


async def test_reconcile_requires_a_workflow_name_to_act():
    qs = mock.MagicMock()
    qs.get_workflow_status.return_value = None
    materialization = Materialization(name="m", workflow_names=None)

    # No pinned name: nothing can be queried, nothing is launched.
    status, info = await backfills.reconcile_scheduling(
        Backfill(workflow_name=None, spec=[]),
        qs,
        materialization,
        allow_resubmit=True,
    )
    assert (status, info) == (None, None)
    assert qs.get_workflow_status.call_count == 0

    # Name pinned but the service does not know it and resubmit is disallowed
    # (a recovery decides that separately).
    status, info = await backfills.reconcile_scheduling(
        Backfill(workflow_name="wf", spec=[]),
        qs,
        materialization,
        allow_resubmit=False,
    )
    assert (status, info) == (None, None)
    assert qs.get_workflow_status.call_count == 1


async def test_get_backfill_missing_and_scope(session):
    with pytest.raises(DJInvalidInputException):
        await backfills.get_backfill(session, 999_999_999)


async def test_publishing_a_second_day_merges_into_existing_state(
    session,
    current_user,
):
    """Same-table availability from a later confirmation merges, not replaces."""
    revision = _revision()
    revision.name = "default.generation_backfill_merge_ut"
    revision.version = "v1.0"
    revision.created_by_id = current_user.id
    revision.node = Node(
        name=revision.name,
        type=revision.type,
        current_version="v1.0",
        created_by_id=current_user.id,
    )
    materialization = Materialization(
        node_revision=revision,
        name="spark_sql__incremental_time__birth_date",
        schedule="@daily",
        config={},
    )
    session.add(materialization)
    await session.commit()

    backfill = Backfill(
        materialization=materialization,
        spec=[
            {"column_name": "birth_date", "range": ["20230101", "20230102"], "values": None},
        ],
        planned_partitions=["birth_date=20230101", "birth_date=20230102"],
        confirmed_partitions=["birth_date=20230101"],
        output_tables=[],
    )
    session.add(backfill)
    await session.commit()
    await session.refresh(revision, ["availability"])

    location = dict(
        catalog="default",
        schema_="roads",
        table="merge_ut",
    )
    await backfills.publish_due_partitions(
        session,
        revision,
        backfill,
        [BackfillPartitionCompletion(values=["20230101"], **location)],
    )
    await session.commit()
    await session.refresh(revision, ["availability"])
    assert revision.availability.max_temporal_partition == ["20230101"]

    backfill.confirmed_partitions = ["birth_date=20230101", "birth_date=20230102"]
    await backfills.publish_due_partitions(
        session,
        revision,
        backfill,
        [BackfillPartitionCompletion(values=["20230102"], **location)],
    )
    await session.commit()
    await session.refresh(revision, ["availability"])
    assert revision.availability.min_temporal_partition == ["20230101"]
    assert revision.availability.max_temporal_partition == ["20230102"]
