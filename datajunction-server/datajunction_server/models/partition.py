"""Partition-related models."""

import re
from datetime import date, datetime

from pydantic import ConfigDict, Field
from pydantic.main import BaseModel

from datajunction_server.enum import StrEnum


class PartitionType(StrEnum):
    """
    Partition type.

    A partition can be temporal or categorical
    """

    TEMPORAL = "temporal"
    CATEGORICAL = "categorical"


class Granularity(StrEnum):
    """
    Time dimension granularity.
    """

    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


class PartitionInput(BaseModel):
    """
    Expected settings for specifying a partition column
    """

    type_: PartitionType

    #
    # Temporal partitions will additionally have the following properties:
    #
    # Timestamp granularity
    granularity: Granularity | None = None
    # Timestamp format
    format: str | None = None


class PartitionBackfill(BaseModel):
    """
    Used for setting backfilled values
    """

    column_name: str

    # Backfilled values and range. Most temporal partitions will just use `range`, but some may
    # optionally use `values` to specify specific values
    # Ex: values: [20230901]
    #     range: [20230901, 20231001]
    values: list | None = None
    range: list | None = None

    model_config = ConfigDict(from_attributes=True)


class PartitionOutput(BaseModel):
    """
    Output for partition
    """

    type_: PartitionType
    format: str | None = None
    granularity: str | None = None
    expression: str | None = None

    model_config = ConfigDict(from_attributes=True)


class PartitionColumnOutput(BaseModel):
    """
    Output for partition columns
    """

    name: str
    type_: PartitionType
    format: str | None = None
    expression: str | None = None


class BackfillOutput(BaseModel):
    """
    Output model for backfills
    """

    spec: list[PartitionBackfill] | None = None
    urls: list[str] | None = None

    model_config = ConfigDict(from_attributes=True)


class BackfillStatus(StrEnum):
    """
    Lifecycle of a generation-tracked backfill run.

    PENDING: the run was recorded but the external workflow's scheduling
    response was never confirmed (lost response, timeout or crash).
    RUNNING: the workflow was accepted and not every planned partition is
    accounted for yet.
    PARTIALLY_COMPLETE: the workflow is done but at least one planned partition
    failed; availability is published for the confirmed partitions only.
    COMPLETE: every planned partition was confirmed and published.
    FAILED: scheduling was rejected outright (e.g. the backfill workflow does
    not exist); nothing was launched.
    SUPERSEDED: a backfill of a newer materialization generation took over the
    overlapping range; this run must no longer publish anything.
    """

    PENDING = "pending"
    RUNNING = "running"
    PARTIALLY_COMPLETE = "partially_complete"
    COMPLETE = "complete"
    FAILED = "failed"
    SUPERSEDED = "superseded"

    @property
    def is_active(self) -> bool:
        """Whether a run in this state may still make progress or be retried."""
        return self in (
            BackfillStatus.PENDING,
            BackfillStatus.RUNNING,
            BackfillStatus.PARTIALLY_COMPLETE,
        )


class WorkflowRunState(StrEnum):
    """
    State of an external workflow run, as reported by the workflow service.
    """

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"

    @property
    def is_terminal(self) -> bool:
        """Whether the workflow has stopped."""
        return self in (
            WorkflowRunState.SUCCEEDED,
            WorkflowRunState.FAILED,
            WorkflowRunState.CANCELLED,
        )


class WorkflowStatusInfo(BaseModel):
    """
    Status of an external workflow run, used to reconcile a backfill after a
    lost scheduling response or a service restart.
    """

    workflow_name: str
    state: WorkflowRunState = WorkflowRunState.UNKNOWN
    urls: list[str] = Field(default_factory=list)
    output_tables: list[str] = Field(default_factory=list)
    # Concrete partition coordinates (same keys as ``Backfill.planned_partitions``)
    completed_partitions: list[str] = Field(default_factory=list)
    failed_partitions: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")


class BackfillPartitionCompletion(BaseModel):
    """
    Outcome of one concrete planned partition.

    ``values`` is the partition coordinate the workflow ran for, e.g.
    ``["20230101"]`` for a single temporal partition or
    ``["US", "20230101"]`` for a categorical + temporal one. The catalog /
    schema / table identify the output the partition was written to and are
    only required with the first confirmed partition of a run that has no
    output table pinned yet.
    """

    values: list[str]
    succeeded: bool = True
    catalog: str | None = None
    schema_: str | None = Field(default=None, alias="schema")
    table: str | None = None
    url: str | None = None
    valid_through_ts: int | None = None

    model_config = ConfigDict(populate_by_name=True)


class BackfillCompletionInput(BaseModel):
    """
    Partition-level confirmations posted by the external workflow.

    Availability is published for exactly the confirmed partitions reported
    here; a run that reports a failure must never publish a state covering the
    whole requested range.
    """

    partitions: list[BackfillPartitionCompletion]


class BackfillStatusInfo(BaseModel):
    """
    Status of a generation-tracked backfill run.
    """

    id: int
    materialization_id: int
    materialization_name: str
    node_name: str
    node_version: str
    generation: int
    status: BackfillStatus
    workflow_name: str | None = None
    idempotency_key: str | None = None
    temporal_range: list[str] | None = None
    spec: list[PartitionBackfill] = Field(default_factory=list)
    planned_partitions: list[str] = Field(default_factory=list)
    confirmed_partitions: list[str] = Field(default_factory=list)
    failed_partitions: list[str] = Field(default_factory=list)
    published_partitions: list[str] = Field(default_factory=list)
    output_tables: list[str] = Field(default_factory=list)
    urls: list[str] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


# Java-style temporal format tokens, most to least significant. A format built
# only from these, in this order, renders to an integer whose ordering is
# chronological -- which is what makes comparing two such values meaningful.
PARTITION_FORMAT_TOKENS = {
    "yyyy": "%Y",
    "MM": "%m",
    "dd": "%d",
    "HH": "%H",
    "mm": "%M",
    "ss": "%S",
}
_TOKEN_PATTERN = re.compile("|".join(PARTITION_FORMAT_TOKENS))


def render_partition_value(moment: datetime, format_: str | None) -> int | None:
    """
    Render a point in time as an integer in a partition format.

    Only a format built purely from tokens in descending order of significance
    renders: anything with separators (``yyyy-MM-dd``) can't produce an integer,
    and anything out of order (``ddMMyyyy``) produces one whose ordering isn't
    chronological. Both yield None so the caller declines to judge.
    """
    if not format_:
        return None
    tokens = _TOKEN_PATTERN.findall(format_)
    if _TOKEN_PATTERN.sub("", format_) != "":
        return None
    order = list(PARTITION_FORMAT_TOKENS)
    if [order.index(token) for token in tokens] != sorted(
        order.index(token) for token in tokens
    ):
        return None
    return int(
        moment.strftime(
            _TOKEN_PATTERN.sub(lambda m: PARTITION_FORMAT_TOKENS[m.group()], format_),
        ),
    )


def parse_partition_value(value: str, format_: str | None) -> date | None:
    """
    Read a partition value back as the day it names.

    The reverse of ``render_partition_value``, and unlike it happy with
    separators: ``yyyy-MM-dd`` renders no integer but still names a day. None
    when there is no format, or the value does not match it.
    """
    if not format_:
        return None
    pattern = _TOKEN_PATTERN.sub(
        lambda m: PARTITION_FORMAT_TOKENS[m.group()],
        format_,
    )
    try:
        return datetime.strptime(value, pattern).date()
    except (TypeError, ValueError):
        return None
