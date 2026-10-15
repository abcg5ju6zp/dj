"""Backfill database schema."""

from datetime import UTC, datetime
from functools import partial
from typing import TYPE_CHECKING

import sqlalchemy as sa
from sqlalchemy import JSON, BigInteger, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from datajunction_server.database.base import Base
from datajunction_server.models.partition import BackfillStatus, PartitionBackfill
from datajunction_server.typing import UTCDatetime

if TYPE_CHECKING:
    from datajunction_server.database.materialization import Materialization


class Backfill(Base):  # type: ignore
    """
    A backfill run is linked to a materialization config, where users provide the range
    (of a temporal partition) to backfill for the node.

    Generation-tracked fields pin everything scheduling, partition confirmation and
    availability publishing share to one immutable snapshot: the materialization
    version and generation, the requested time range, the concrete partition plan,
    the external workflow name and the output identifier the workflow writes to.
    """

    __tablename__ = "backfill"
    __table_args__ = (
        sa.Index("ix_backfill_materialization_status", "materialization_id", "status"),
    )

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
    )

    # The column reference that this backfill is defined on
    materialization_id: Mapped[int] = mapped_column(
        ForeignKey(
            "materialization.id",
            name="fk_backfill_materialization_id_materialization",
            ondelete="CASCADE",
        ),
    )
    materialization: Mapped["Materialization"] = relationship(
        back_populates="backfills",
        primaryjoin="Materialization.id==Backfill.materialization_id",
    )

    # Backfilled values and range
    spec: Mapped[list[PartitionBackfill]] = mapped_column(
        JSON,
        default=[],
    )

    urls: Mapped[list[str] | None] = mapped_column(
        JSON,
        default=[],
    )

    # Deterministic dedupe key. Two submissions carrying the same key are the
    # same run, so a retried request after a lost scheduling response adopts the
    # recorded run instead of launching a second workflow. Null for rows written
    # before generation tracking existed.
    idempotency_key: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
        unique=True,
    )

    # Node revision version this backfill was requested against. Pinned so a run
    # left in flight across a redefinition keeps publishing to its own revision.
    materialization_version: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
    )

    # Materialization generation pinned at submission time. Confirmations from a
    # run older than the current generation are refused: an updated definition
    # owns the range now.
    generation: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        server_default="1",
    )

    # External workflow the run belongs to, when the materialization has one
    # recorded. Null only for materializations that predate workflow name
    # recording, for which the query service derives a name itself.
    workflow_name: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
    )

    # Output identifiers (fully-qualified tables) the workflow writes to.
    output_tables: Mapped[list[str]] = mapped_column(
        JSON,
        default=[],
        server_default="[]",
    )

    # Pinned temporal interval [start, end], taken from the temporal partition
    # part of the spec. Used for overlapping-range arbitration.
    temporal_range: Mapped[list[str] | None] = mapped_column(
        JSON,
        nullable=True,
    )

    # Concrete partition coordinates derived from the spec, e.g.
    # "birth_date=20230101". The plan is fixed at submission, so progress is
    # always measured against the same set scheduling saw.
    planned_partitions: Mapped[list[str]] = mapped_column(
        JSON,
        default=[],
        server_default="[]",
    )
    confirmed_partitions: Mapped[list[str]] = mapped_column(
        JSON,
        default=[],
        server_default="[]",
    )
    failed_partitions: Mapped[list[str]] = mapped_column(
        JSON,
        default=[],
        server_default="[]",
    )
    # Partitions whose availability has already been published. Kept apart from
    # confirmed partitions so a failed availability write is retried as a
    # publish only, never by recomputing or rerunning a finished partition.
    published_partitions: Mapped[list[str]] = mapped_column(
        JSON,
        default=[],
        server_default="[]",
    )

    status: Mapped[str] = mapped_column(
        String,
        nullable=False,
        default=BackfillStatus.PENDING.value,
        server_default=BackfillStatus.PENDING.value,
    )

    created_at: Mapped[UTCDatetime] = mapped_column(
        DateTime(timezone=True),
        default=partial(datetime.now, UTC),
    )
    updated_at: Mapped[UTCDatetime] = mapped_column(
        DateTime(timezone=True),
        default=partial(datetime.now, UTC),
        onupdate=partial(datetime.now, UTC),
    )

    @property
    def state(self) -> BackfillStatus:
        """The run's lifecycle state."""
        return BackfillStatus(self.status)
