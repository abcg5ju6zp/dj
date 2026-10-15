"""backfill generation tracking

Revision ID: bf0001gen
Revises: tbl0001idx
Create Date: 2026-10-15 00:00:00.000000+00:00

Pins every backfill to the materialization generation, version, time range,
partition plan, workflow name and output identifier it was scheduled against,
and tracks per-partition confirmation and publication progress.
"""

import sqlalchemy as sa
from alembic import op

revision = "bf0001gen"
down_revision = "tbl0001idx"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "materialization",
        sa.Column(
            "generation",
            sa.Integer(),
            nullable=False,
            server_default="1",
        ),
    )

    op.add_column(
        "backfill",
        sa.Column(
            "idempotency_key",
            sa.String(),
            nullable=True,
        ),
    )
    op.add_column(
        "backfill",
        sa.Column(
            "materialization_version",
            sa.String(),
            nullable=True,
        ),
    )
    op.add_column(
        "backfill",
        sa.Column(
            "generation",
            sa.Integer(),
            nullable=False,
            server_default="1",
        ),
    )
    op.add_column(
        "backfill",
        sa.Column(
            "workflow_name",
            sa.String(),
            nullable=True,
        ),
    )
    op.add_column(
        "backfill",
        sa.Column(
            "output_tables",
            sa.JSON(),
            nullable=False,
            server_default="[]",
        ),
    )
    op.add_column(
        "backfill",
        sa.Column(
            "temporal_range",
            sa.JSON(),
            nullable=True,
        ),
    )
    op.add_column(
        "backfill",
        sa.Column(
            "planned_partitions",
            sa.JSON(),
            nullable=False,
            server_default="[]",
        ),
    )
    op.add_column(
        "backfill",
        sa.Column(
            "confirmed_partitions",
            sa.JSON(),
            nullable=False,
            server_default="[]",
        ),
    )
    op.add_column(
        "backfill",
        sa.Column(
            "failed_partitions",
            sa.JSON(),
            nullable=False,
            server_default="[]",
        ),
    )
    op.add_column(
        "backfill",
        sa.Column(
            "published_partitions",
            sa.JSON(),
            nullable=False,
            server_default="[]",
        ),
    )
    op.add_column(
        "backfill",
        sa.Column(
            "status",
            sa.String(),
            nullable=False,
            server_default="pending",
        ),
    )
    op.add_column(
        "backfill",
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )
    op.add_column(
        "backfill",
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )
    op.create_unique_constraint(
        "uq_backfill_idempotency_key",
        "backfill",
        ["idempotency_key"],
    )
    op.create_index(
        "ix_backfill_materialization_status",
        "backfill",
        ["materialization_id", "status"],
    )
    # Rows written before generation tracking are fire-and-forget records; mark
    # them complete so recovery never mistakes them for unscheduled runs.
    op.execute("UPDATE backfill SET status = 'complete'")


def downgrade():
    op.drop_index(
        "ix_backfill_materialization_status",
        table_name="backfill",
    )
    op.drop_constraint(
        "uq_backfill_idempotency_key",
        "backfill",
        type_="unique",
    )
    op.drop_column("backfill", "updated_at")
    op.drop_column("backfill", "created_at")
    op.drop_column("backfill", "status")
    op.drop_column("backfill", "published_partitions")
    op.drop_column("backfill", "failed_partitions")
    op.drop_column("backfill", "confirmed_partitions")
    op.drop_column("backfill", "planned_partitions")
    op.drop_column("backfill", "temporal_range")
    op.drop_column("backfill", "output_tables")
    op.drop_column("backfill", "workflow_name")
    op.drop_column("backfill", "generation")
    op.drop_column("backfill", "materialization_version")
    op.drop_column("backfill", "idempotency_key")
    op.drop_column("materialization", "generation")
