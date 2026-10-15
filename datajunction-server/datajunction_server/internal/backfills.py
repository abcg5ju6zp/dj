"""
Generation-tracked backfill orchestration.

A backfill pins one immutable snapshot -- materialization version and
generation, requested time range, concrete partition plan, external workflow
name and output identifier -- so that scheduling, running, partition
confirmation and availability publishing all advance the same record.

Everything here is deliberately workflow-service agnostic: it talks to the
external service through ``QueryServiceClient`` and keeps all durable progress
on the ``Backfill`` row plus the node revision's availability records.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime, timedelta
from itertools import product
from typing import TYPE_CHECKING, Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from datajunction_server.database.availabilitystate import AvailabilityState
from datajunction_server.database.backfill import Backfill
from datajunction_server.database.materialization import Materialization
from datajunction_server.errors import (
    DJBackfillConflictException,
    DJInvalidInputException,
)
from datajunction_server.models.materialization import MaterializationInfo
from datajunction_server.models.node import (
    AvailabilityStateBase,
    PartitionAvailability,
)
from datajunction_server.models.node_type import NodeType
from datajunction_server.models.partition import (
    BackfillPartitionCompletion,
    BackfillStatus,
    BackfillStatusInfo,
    Granularity,
    PartitionBackfill,
    WorkflowRunState,
    WorkflowStatusInfo,
)
from datajunction_server.naming import amenable_name

if TYPE_CHECKING:
    from datajunction_server.database.column import Column
    from datajunction_server.database.node import NodeRevision
    from datajunction_server.service_clients import QueryServiceClient

_logger = logging.getLogger(__name__)

# Keep range expansion bounded -- a granularity mismatch must never enumerate
# forever.
MAX_ENUMERATED_PARTITIONS = 10000

_GRANULARITY_STEP = {
    Granularity.SECOND: timedelta(seconds=1),
    Granularity.MINUTE: timedelta(minutes=1),
    Granularity.HOUR: timedelta(hours=1),
    Granularity.DAY: timedelta(days=1),
    Granularity.WEEK: timedelta(weeks=1),
}


# ---------------------------------------------------------------------------
# Plan normalisation
# ---------------------------------------------------------------------------


def canonical_spec(partitions: list[PartitionBackfill]) -> list[dict[str, Any]]:
    """
    Serialise the requested partition plan into a form that is stable under
    reordering, so a retried request hashes to the same idempotency key.
    """
    return sorted(
        (
            {
                "column_name": partition.column_name,
                "values": sorted(partition.values) if partition.values else None,
                "range": list(partition.range) if partition.range else None,
            }
            for partition in partitions
        ),
        key=lambda entry: entry["column_name"],
    )


def make_idempotency_key(
    materialization_id: int,
    generation: int,
    spec: list[dict[str, Any]],
) -> str:
    """Deterministic identity of one (materialization, generation, plan) run."""
    digest = hashlib.sha256(
        json.dumps(spec, sort_keys=True, default=str).encode(),
    ).hexdigest()
    return f"bf-{materialization_id}-g{generation}-{digest[:16]}"


def partition_column_name(column: Column, revision: NodeRevision) -> str:
    """The amenable name the backfill spec uses for a partition column."""
    raw = column.cube_element_name if revision.type == NodeType.CUBE else column.name
    return amenable_name(raw)


def _partition_columns(revision: NodeRevision) -> tuple[list[Column], list[Column]]:
    """Categorical then temporal partition columns, in node order."""
    temporal = revision.temporal_partition_columns()
    categorical = revision.categorical_partition_columns()
    return categorical, temporal


def _entries_by_column(
    spec: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {entry["column_name"]: entry for entry in spec if entry.get("column_name")}


def _strptime_pattern(format_: str | None) -> str | None:
    """Translate a Java token format to a strptime pattern, separators kept."""
    if not format_:
        return None
    tokens = {
        "yyyy": "%Y",
        "MM": "%m",
        "dd": "%d",
        "HH": "%H",
        "mm": "%M",
        "ss": "%S",
    }
    pattern = format_
    for token, replacement in tokens.items():
        pattern = pattern.replace(token, replacement)
    return pattern


def enumerate_temporal_values(
    format_: str | None,
    granularity: str | None,
    start_raw: Any,
    end_raw: Any,
) -> list[str] | None:
    """
    Enumerate the concrete partition values a temporal range covers, rendered
    in the column format. Bounds are inclusive.

    Returns None when the format/granularity cannot be enumerated (e.g. a
    monthly token on a format strptime cannot parse); callers then plan the
    whole range as a single opaque unit.
    """
    pattern = _strptime_pattern(format_)
    if pattern is None or "%" not in pattern:
        return None
    try:
        start = datetime.strptime(str(start_raw), pattern)
        end = datetime.strptime(str(end_raw), pattern)
    except (TypeError, ValueError):
        return None
    if end < start:
        return None

    step = _GRANULARITY_STEP.get(granularity) if granularity else timedelta(days=1)

    def add_step(moment: datetime) -> datetime:
        if granularity == Granularity.MONTH:
            month = moment.month + 1
            year = moment.year + (month - 1) // 12
            return moment.replace(year=year, month=(month - 1) % 12 + 1)
        if granularity == Granularity.QUARTER:
            month = moment.month + 3
            year = moment.year + (month - 1) // 12
            return moment.replace(year=year, month=(month - 1) % 12 + 1)
        if granularity == Granularity.YEAR:
            return moment.replace(year=moment.year + 1)
        return moment + (step or timedelta(days=1))

    values: list[str] = []
    current = start
    while current <= end:
        values.append(current.strftime(pattern))
        if len(values) > MAX_ENUMERATED_PARTITIONS:
            return None
        current = add_step(current)
    return values


def coordinate_key(assignment: dict[str, str]) -> str:
    """Stable key for one concrete partition coordinate."""
    return "|".join(f"{column}={value}" for column, value in sorted(assignment.items()))


def parse_coordinate_key(key: str) -> dict[str, str]:
    """Reverse of :func:`coordinate_key`."""
    assignment: dict[str, str] = {}
    for segment in key.split("|"):
        if "=" in segment:
            column, value = segment.split("=", 1)
            assignment[column] = value
    return assignment


def expand_plan(
    revision: NodeRevision,
    spec: list[dict[str, Any]],
) -> tuple[list[str], list[str] | None]:
    """
    Expand the request into concrete partition coordinates the workflow will
    run and report back, plus the pinned primary temporal interval.

    The plan is fixed at submission and never re-derived, so a retried
    confirmation or a recovery after restart measures progress against exactly
    these coordinates.
    """
    categorical, temporal = _partition_columns(revision)
    by_column = _entries_by_column(spec)
    columns_in_order = categorical + temporal

    axes: list[tuple[str, list[str]]] = []
    temporal_range: list[str] | None = None
    for column in columns_in_order:
        name = partition_column_name(column, revision)
        entry = by_column.get(name)
        if not entry:
            continue
        if entry.get("values"):
            axes.append((name, [str(value) for value in entry["values"]]))
            continue
        if entry.get("range"):
            start, end = entry["range"][0], entry["range"][-1]
            if column in temporal:
                enumerated = enumerate_temporal_values(
                    column.partition.format if column.partition else None,
                    column.partition.granularity if column.partition else None,
                    start,
                    end,
                )
                if enumerated is not None:
                    axes.append((name, enumerated))
                else:
                    axes.append((name, [f"[{start}..{end}]"]))
                temporal_range = [str(start), str(end)]
            else:
                # A "range" on a categorical column is not meaningful; take
                # the bounds as the two coordinates.
                axes.append((name, [str(start), str(end)]))

    if not axes:
        return [], temporal_range

    names = [name for name, _ in axes]
    keys = [
        coordinate_key(dict(zip(names, combo, strict=True)))
        for combo in product(*(values for _, values in axes))
    ]
    return keys, temporal_range


def specs_overlap(
    revision: NodeRevision,
    spec_a: list[dict[str, Any]],
    spec_b: list[dict[str, Any]],
) -> bool:
    """
    Whether two plans against the same materialization cover a common
    partition. Temporal intervals are compared on the same partition column;
    explicitly enumerated categorical value sets must intersect.
    """
    _, temporal = _partition_columns(revision)
    a = _entries_by_column(spec_a)
    b = _entries_by_column(spec_b)

    for column in temporal:
        name = partition_column_name(column, revision)
        entry_a = a.get(name)
        entry_b = b.get(name)
        if entry_a and entry_b and not _bounds_overlap(column, entry_a, entry_b):
            return False

    categorical, _ = _partition_columns(revision)
    for column in categorical:
        name = partition_column_name(column, revision)
        values_a = set((a.get(name) or {}).get("values") or [])
        values_b = set((b.get(name) or {}).get("values") or [])
        if values_a and values_b and values_a.isdisjoint(values_b):
            return False
    return True


def _entry_bounds(column: Column, entry: dict[str, Any]) -> tuple[Any, Any] | None:
    if entry.get("range"):
        return entry["range"][0], entry["range"][-1]
    values = entry.get("values")
    if values:
        return min(values), max(values)
    return None


def _bounds_overlap(
    column: Column,
    entry_a: dict[str, Any],
    entry_b: dict[str, Any],
) -> bool:
    bounds_a = _entry_bounds(column, entry_a)
    bounds_b = _entry_bounds(column, entry_b)
    if not bounds_a or not bounds_b:
        return True  # an un-bounded axis covers everything on it

    format_ = column.partition.format if column.partition else None
    pattern = _strptime_pattern(format_)
    points: list[Any] = [*bounds_a, *bounds_b]
    try:
        if pattern:
            parsed = [datetime.strptime(str(point), pattern) for point in points]
        else:
            parsed = [float(point) for point in points]
    except (TypeError, ValueError):
        # Uninterpretable formats: fall back to raw ordering, declining to
        # declare non-overlap unless the raw strings clearly separate.
        return not (bounds_a[1] < bounds_b[0] or bounds_b[1] < bounds_a[0])
    start_a, end_a, start_b, end_b = parsed
    return start_a <= end_b and start_b <= end_a


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def planned_columns_in_order(revision: NodeRevision) -> list[str]:
    """Column coordinates are reported categorical-first, then temporal."""
    categorical, temporal = _partition_columns(revision)
    return [
        partition_column_name(column, revision) for column in categorical + temporal
    ]


def assignment_from_values(
    revision: NodeRevision,
    planned_keys: list[str],
    values: list[str],
) -> dict[str, str]:
    """
    Map a reported coordinate (values in categorical-then-temporal order) onto
    the columns this plan actually uses.
    """
    involved: set[str] = set()
    for key in planned_keys:
        involved.update(parse_coordinate_key(key))
    # Preserve node ordering rather than the set's iteration order.
    ordered = [name for name in planned_columns_in_order(revision) if name in involved]
    if len(values) != len(ordered):
        raise DJInvalidInputException(
            message=(
                f"Partition coordinate {values!r} must name exactly the planned "
                f"partition columns {ordered!r}."
            ),
        )
    return dict(zip(ordered, (str(value) for value in values), strict=True))


async def get_backfill(
    session: AsyncSession,
    backfill_id: int,
    materialization_id: int | None = None,
) -> Backfill:
    """Load a backfill, optionally scoped to its materialization."""
    statement = select(Backfill).where(Backfill.id == backfill_id)
    if materialization_id is not None:
        statement = statement.where(
            Backfill.materialization_id == materialization_id,
        )
    backfill = (await session.execute(statement)).scalar_one_or_none()
    if backfill is None:
        raise DJInvalidInputException(
            message=f"Backfill {backfill_id} not found.",
            http_status_code=404,
        )
    return backfill


def is_generation_current(backfill: Backfill, materialization: Materialization) -> bool:
    """Whether the run still belongs to the materialization's current definition."""
    return backfill.generation == (materialization.generation or 1)


def ensure_generation_current(
    backfill: Backfill,
    materialization: Materialization,
) -> None:
    """
    Reject work from a run pinned to an older generation: its definition was
    updated and it must not confirm partitions or publish onto the replacement.
    """
    if not is_generation_current(backfill, materialization):
        raise DJBackfillConflictException(
            message=(
                f"Backfill {backfill.id} belongs to generation "
                f"{backfill.generation} of materialization "
                f"{materialization.name}, but the current definition is "
                f"generation {materialization.generation}. Confirmations from "
                "the superseded run are refused."
            ),
        )


def finalize_status(
    backfill: Backfill,
    workflow_state: WorkflowRunState | None = None,
) -> BackfillStatus:
    """
    Lifecycle state from recorded progress, publication progress and
    (optionally) the workflow's reported state.

    A run only reaches COMPLETE once every planned partition was confirmed and
    the confirmed partitions are published; a failed partition makes the
    terminal state PARTIALLY_COMPLETE instead.
    """
    current = backfill.state
    if current in (BackfillStatus.FAILED, BackfillStatus.SUPERSEDED):
        return current
    planned = set(backfill.planned_partitions or [])
    confirmed = set(backfill.confirmed_partitions or [])
    failed = set(backfill.failed_partitions or [])
    published = set(backfill.published_partitions or [])
    accounted = confirmed | failed
    all_accounted = bool(planned) and planned.issubset(accounted)
    all_published = confirmed.issubset(published)

    if all_accounted and all_published and confirmed:
        return (
            BackfillStatus.COMPLETE if not failed else BackfillStatus.PARTIALLY_COMPLETE
        )
    if workflow_state is not None and workflow_state.is_terminal:
        # The workflow stopped with the plan unfinished: unaccounted partitions
        # are failed, never silently treated as available.
        if all_accounted and all_published and not confirmed:
            return BackfillStatus.FAILED
        if workflow_state == WorkflowRunState.SUCCEEDED and not all_accounted:
            # A successful workflow that did not report every coordinate leaves
            # the run in flight from DJ's point of view.
            return BackfillStatus.RUNNING
        return BackfillStatus.PARTIALLY_COMPLETE
    if confirmed or failed:
        return BackfillStatus.RUNNING
    return (
        BackfillStatus.PENDING
        if current == BackfillStatus.PENDING
        else BackfillStatus.RUNNING
    )


def adopt_workflow_info(
    backfill: Backfill,
    info: MaterializationInfo | WorkflowStatusInfo,
) -> None:
    """Record the identifiers the workflow service answered with."""
    urls = list(getattr(info, "urls", None) or [])
    output_tables = list(getattr(info, "output_tables", None) or [])
    if urls:
        current_urls = backfill.urls or []
        merged = [
            *current_urls,
            *(url for url in urls if url not in current_urls),
        ]
        backfill.urls = merged
    if output_tables:
        existing = backfill.output_tables or []
        backfill.output_tables = [
            *existing,
            *(table for table in output_tables if table not in existing),
        ]
    names = getattr(info, "workflow_names", None)
    if names and not backfill.workflow_name:
        backfill.workflow_name = next(
            (name for name in names if name.endswith(".backfill")),
            names[0],
        )


# ---------------------------------------------------------------------------
# Partition confirmation and availability publication
# ---------------------------------------------------------------------------


def _valid_through_ts(
    column: Column,
    value: str,
    supplied: int | None,
) -> int | None:
    if supplied is not None:
        return supplied
    pattern = _strptime_pattern(column.partition.format if column.partition else None)
    if pattern:
        try:
            moment = datetime.strptime(str(value), pattern)
        except (TypeError, ValueError):
            moment = None
        if moment is not None:
            granularity = column.partition.granularity if column.partition else None
            step = (
                _GRANULARITY_STEP.get(granularity) if granularity else timedelta(days=1)
            )
            if granularity in (
                Granularity.MONTH,
                Granularity.QUARTER,
                Granularity.YEAR,
            ):
                step = timedelta(days=1)
            return int(
                (moment + step - timedelta(seconds=1)).replace(tzinfo=UTC).timestamp()
            )
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _availability_partition(
    revision: NodeRevision,
    assignment: dict[str, str],
    valid_through_ts: int | None,
) -> PartitionAvailability:
    categorical, temporal = _partition_columns(revision)
    cat_names = [partition_column_name(column, revision) for column in categorical]
    temporal_names = [partition_column_name(column, revision) for column in temporal]
    values = [assignment[name] if name in assignment else None for name in cat_names]
    temporal_values = [
        assignment[name] for name in temporal_names if name in assignment
    ]
    return PartitionAvailability(
        value=values,
        min_temporal_partition=temporal_values,
        max_temporal_partition=list(temporal_values),
        valid_through_ts=valid_through_ts,
    )


def availability_covers_key(
    availability: AvailabilityState | None,
    key: str,
    revision: NodeRevision,
) -> bool:
    """
    Whether a recorded availability state already covers the partition. Used on
    recovery so a confirmation whose publish crashed after the commit is not
    published twice and never recomputed.
    """
    if availability is None or not availability.partitions:
        return False
    assignment = parse_coordinate_key(key)
    categorical, temporal = _partition_columns(revision)
    cat_names = [partition_column_name(column, revision) for column in categorical]
    temporal_names = [partition_column_name(column, revision) for column in temporal]
    wanted_cats = [
        assignment[name] if name in assignment else None for name in cat_names
    ]
    wanted_temporal = {
        name: assignment[name] for name in temporal_names if name in assignment
    }
    for record in availability.partitions:
        record = record if isinstance(record, dict) else record.model_dump()
        if list(record.get("value") or []) != wanted_cats:
            continue
        minima = record.get("min_temporal_partition") or []
        maxima = record.get("max_temporal_partition") or []
        covered = True
        for index, name in enumerate(temporal_names):
            if name not in wanted_temporal:
                continue
            if index >= len(minima) or index >= len(maxima):
                covered = False
                break
            if not (str(minima[index]) <= wanted_temporal[name] <= str(maxima[index])):
                covered = False
                break
        if covered:
            return True
    return False


def record_completions(
    revision: NodeRevision,
    backfill: Backfill,
    completions: list[BackfillPartitionCompletion],
) -> tuple[set[str], set[str]]:
    """
    Apply partition outcomes onto the run's progress. Returns the confirmed
    and failed key sets. Does not publish availability and does not touch the
    database -- the caller commits first, so a later publish failure only needs
    the publish retried.
    """
    planned = set(backfill.planned_partitions or [])
    confirmed = set(backfill.confirmed_partitions or [])
    failed = set(backfill.failed_partitions or [])
    published = set(backfill.published_partitions or [])

    for completion in completions:
        assignment = assignment_from_values(
            revision,
            backfill.planned_partitions or [],
            completion.values,
        )
        key = coordinate_key(assignment)
        if key not in planned:
            raise DJInvalidInputException(
                message=(
                    f"Partition {key} is not part of backfill {backfill.id}'s "
                    "pinned plan; refusing the confirmation."
                ),
            )
        if completion.succeeded:
            failed.discard(key)
            confirmed.add(key)
        else:
            confirmed.discard(key)
            published.discard(key)
            failed.add(key)

    backfill.confirmed_partitions = sorted(confirmed)
    backfill.failed_partitions = sorted(failed)
    backfill.published_partitions = sorted(published)
    return confirmed, failed


def _location_for(
    revision: NodeRevision,
    backfill: Backfill,
    completions: list[BackfillPartitionCompletion],
) -> tuple[str, str | None, str, str | None, int | None]:
    """Resolve the output location confirmed partitions were written to."""
    existing = revision.availability
    located = next(
        (
            completion
            for completion in completions
            if completion.succeeded and completion.catalog and completion.table
        ),
        None,
    )
    if located:
        return (
            located.catalog,
            located.schema_,
            located.table,
            located.url,
            located.valid_through_ts,
        )
    if existing is not None:
        return (
            existing.catalog,
            existing.schema_,
            existing.table,
            existing.url,
            None,
        )
    if backfill.output_tables:
        parts = (backfill.output_tables[0]).split(".")
        if len(parts) >= 3:
            return parts[0], parts[1], parts[-1], None, None
        return "default", None, parts[-1], None, None
    raise DJInvalidInputException(
        message=(
            f"Cannot publish availability for backfill {backfill.id}: no output "
            "catalog/schema/table has been reported for the confirmed "
            "partitions and none was pinned at scheduling time."
        ),
    )


async def publish_due_partitions(
    session: AsyncSession,
    revision: NodeRevision,
    backfill: Backfill,
    completions: list[BackfillPartitionCompletion] | None = None,
) -> list[str]:
    """
    Publish availability for exactly the confirmed partitions not yet
    published.

    A single failed partition only narrows the published set -- this never
    publishes a range covering a failed partition. Published coordinates are
    recorded separately so a write that fails here is retried as a publish
    only; completed partitions are never recomputed or rerun.
    """
    confirmed = set(backfill.confirmed_partitions or [])
    published = set(backfill.published_partitions or [])
    completions = completions or []

    due = [
        key
        for key in sorted(confirmed)
        if key not in published
        and not availability_covers_key(revision.availability, key, revision)
    ]
    if not due:
        backfill.published_partitions = sorted(published | confirmed)
        return []

    catalog, schema_, table, url, default_valid_through = _location_for(
        revision,
        backfill,
        completions,
    )
    output_identifier = (
        f"{catalog}.{schema_}.{table}" if schema_ else f"{catalog}.{table}"
    )
    if output_identifier not in (backfill.output_tables or []):
        backfill.output_tables = [*(backfill.output_tables or []), output_identifier]

    categorical, temporal = _partition_columns(revision)
    cat_names = [partition_column_name(column, revision) for column in categorical]
    temporal_cols = {
        partition_column_name(column, revision): column for column in temporal
    }

    supplied_valid_through: dict[str, int | None] = {
        coordinate_key(
            assignment_from_values(
                revision,
                backfill.planned_partitions or [],
                completion.values,
            )
        ): completion.valid_through_ts
        for completion in completions
        if completion.succeeded
    }

    partition_availabilities = []
    for key in due:
        assignment = parse_coordinate_key(key)
        valid_through = supplied_valid_through.get(key, default_valid_through)
        if valid_through is None:
            temporal_value = next(
                (value for name, value in assignment.items() if name in temporal_cols),
                None,
            )
            temporal_column = next(
                (col for name, col in temporal_cols.items() if name in assignment),
                None,
            )
            valid_through = (
                _valid_through_ts(temporal_column, temporal_value, None)
                if temporal_column and temporal_value is not None
                else None
            )
        partition_availabilities.append(
            _availability_partition(revision, assignment, valid_through),
        )

    min_temporal = [
        min(values)
        for values in zip(
            *(
                partition.min_temporal_partition or []
                for partition in partition_availabilities
            )
        )
    ]
    max_temporal = [
        max(values)
        for values in zip(
            *(
                partition.max_temporal_partition or []
                for partition in partition_availabilities
            )
        )
    ]
    valid_through_values = [
        partition.valid_through_ts
        for partition in partition_availabilities
        if partition.valid_through_ts is not None
    ]

    new_state = AvailabilityStateBase(
        catalog=catalog,
        schema_=schema_,
        table=table,
        valid_through_ts=max(valid_through_values) if valid_through_values else 0,
        url=url,
        categorical_partitions=cat_names,
        temporal_partitions=list(temporal_cols),
        min_temporal_partition=min_temporal,
        max_temporal_partition=max_temporal,
        partitions=partition_availabilities,
    )

    # Mirror the availability POST: same-table state is merged into, then a new
    # row replaces the revision's state; the join swaps accordingly.
    old = revision.availability
    if old and old.catalog == catalog and old.schema_ == schema_ and old.table == table:
        new_state.merge(
            AvailabilityStateBase.model_validate(old),
        )

    availability = AvailabilityState(
        catalog=new_state.catalog,
        schema_=new_state.schema_,
        table=new_state.table,
        valid_through_ts=new_state.valid_through_ts,
        url=new_state.url,
        min_temporal_partition=[
            str(part) for part in new_state.min_temporal_partition or []
        ],
        max_temporal_partition=[
            str(part) for part in new_state.max_temporal_partition or []
        ],
        partitions=[
            partition if isinstance(partition, dict) else partition.model_dump()
            for partition in new_state.partitions  # type: ignore
        ],
        categorical_partitions=new_state.categorical_partitions,
        temporal_partitions=new_state.temporal_partitions,
        links=new_state.links,
    )
    revision.availability = availability
    session.add(availability)
    session.add(revision)

    backfill.published_partitions = sorted(published | set(due))
    return due


# ---------------------------------------------------------------------------
# Workflow reconciliation / recovery
# ---------------------------------------------------------------------------


async def reconcile_scheduling(
    backfill: Backfill,
    query_service_client: QueryServiceClient,
    materialization: Materialization,
    *,
    allow_resubmit: bool,
    request_headers: dict[str, str] | None = None,
) -> tuple[WorkflowStatusInfo | None, MaterializationInfo | None]:
    """
    Decide what became of a scheduling call whose answer is missing.

    The original workflow is always queried first. Only a workflow service
    that genuinely does not know the name permits a resubmit; finding the
    workflow (in any state) adopts it and launches nothing.

    Returns ``(workflow_status, resubmitted_info)``: exactly one is non-None
    when the run is now known, both None when it cannot be reconciled.
    """
    workflow_name = backfill.workflow_name or materialization.backfill_workflow
    if not workflow_name:
        return None, None

    status = query_service_client.get_workflow_status(
        workflow_name,
        request_headers=request_headers,
    )
    if status is not None:
        adopt_workflow_info(backfill, status)
        return status, None

    if not allow_resubmit:
        return None, None

    from datajunction_server.materialization.jobs import MaterializationJob

    partitions = [PartitionBackfill(**entry) for entry in backfill.spec or []]
    job_class = next(
        (
            cls
            for cls in MaterializationJob.__subclasses__()
            if cls.__name__ == materialization.job
        ),
        None,
    )
    if job_class is None:  # pragma: no cover
        return None, None
    info = job_class().run_backfill(
        materialization,
        partitions,
        query_service_client,
        workflow_name=workflow_name,
        idempotency_key=backfill.idempotency_key,
        request_headers=request_headers,
    )
    adopt_workflow_info(backfill, info)
    return None, info


def adopt_workflow_progress(
    backfill: Backfill,
    status: WorkflowStatusInfo,
) -> None:
    """
    Merge the partition outcomes the workflow itself reports into the run.
    Only coordinates belonging to the pinned plan are accepted.
    """
    planned = set(backfill.planned_partitions or [])
    confirmed = set(backfill.confirmed_partitions or [])
    failed = set(backfill.failed_partitions or [])
    for key in status.completed_partitions or []:
        if not planned or key in planned:
            confirmed.add(key)
            failed.discard(key)
    for key in status.failed_partitions or []:
        if not planned or key in planned:
            failed.add(key)
            confirmed.discard(key)
    backfill.confirmed_partitions = sorted(confirmed)
    backfill.failed_partitions = sorted(failed)
    adopt_workflow_info(backfill, status)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def to_status_info(
    backfill: Backfill,
    materialization: Materialization,
    node_name: str,
    node_version: str,
) -> BackfillStatusInfo:
    """API-facing view of a run."""
    return BackfillStatusInfo(
        id=backfill.id,
        materialization_id=materialization.id,
        materialization_name=materialization.name,
        node_name=node_name,
        node_version=node_version,
        generation=backfill.generation,
        status=BackfillStatus(backfill.status),
        workflow_name=backfill.workflow_name,
        idempotency_key=backfill.idempotency_key,
        temporal_range=backfill.temporal_range,
        spec=[PartitionBackfill(**entry) for entry in backfill.spec or []],
        planned_partitions=list(backfill.planned_partitions or []),
        confirmed_partitions=list(backfill.confirmed_partitions or []),
        failed_partitions=list(backfill.failed_partitions or []),
        published_partitions=list(backfill.published_partitions or []),
        output_tables=list(backfill.output_tables or []),
        urls=list(backfill.urls or []),
    )
