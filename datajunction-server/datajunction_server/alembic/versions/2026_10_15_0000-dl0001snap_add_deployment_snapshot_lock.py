"""add snapshot/progress/lock fields to deployments and managed namespace lock table

Revision ID: dl0001snap
Revises: tbl0001idx
Create Date: 2026-10-15 00:00:00.000000+00:00
"""
# pylint: disable=no-member, invalid-name, missing-function-docstring, unused-import, no-name-in-module

import sqlalchemy as sa
import sqlalchemy_utils
from alembic import op

revision = "dl0001snap"
down_revision = "tbl0001idx"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        # New terminal deployment statuses. Enum labels store member names.
        op.execute(
            "ALTER TYPE deploymentstatus ADD VALUE IF NOT EXISTS 'STALE'",
        )
        op.execute(
            "ALTER TYPE deploymentstatus ADD VALUE IF NOT EXISTS 'CANCELLED'",
        )

    op.add_column(
        "deployments",
        sa.Column("idempotency_key", sa.String(), nullable=True),
    )
    op.create_unique_constraint(
        "uq_deployments_idempotency_key",
        "deployments",
        ["idempotency_key"],
    )
    op.add_column("deployments", sa.Column("snapshot", sa.JSON(), nullable=True))
    op.add_column("deployments", sa.Column("progress", sa.JSON(), nullable=True))
    op.add_column(
        "deployments",
        sa.Column("lock_token", sa.Integer(), nullable=True),
    )
    op.add_column(
        "deployments",
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "deployments",
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_table(
        "deploymentnamespacelock",
        sa.Column("namespace", sa.String(), nullable=False),
        sa.Column(
            "owner_deployment_id",
            sqlalchemy_utils.types.uuid.UUIDType(),
            nullable=False,
        ),
        sa.Column("fencing_token", sa.Integer(), nullable=False),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["owner_deployment_id"],
            ["deployments.uuid"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("namespace"),
    )


def downgrade():
    op.drop_table("deploymentnamespacelock")
    op.drop_column("deployments", "finished_at")
    op.drop_column("deployments", "started_at")
    op.drop_column("deployments", "lock_token")
    op.drop_column("deployments", "progress")
    op.drop_column("deployments", "snapshot")
    op.drop_constraint(
        "uq_deployments_idempotency_key",
        "deployments",
        type_="unique",
    )
    op.drop_column("deployments", "idempotency_key")
    # Postgres offers no DROP VALUE for enums; STALE/CANCELLED stay as labels.
