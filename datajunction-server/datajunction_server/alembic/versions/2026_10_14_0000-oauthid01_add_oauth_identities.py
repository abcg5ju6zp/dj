"""
Add oauth_identities bindings

Revision ID: oauthid01
Revises: tbl0001idx
Create Date: 2026-10-14 00:00:00.000000+00:00

Binds OAuth logins to local principals by the immutable (provider, subject)
pair instead of mutable emails/logins. No data is backfilled: the provider
subjects were never stored, so legacy OAuth users are bound lazily on their
next login through the one-time, uniqueness-proven claim path.
"""
# pylint: disable=no-member, invalid-name, missing-function-docstring, unused-import, no-name-in-module

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "oauthid01"
down_revision = "tbl0001idx"
branch_labels = None
depends_on = None


def upgrade():
    # Reuse the existing postgres ``oauthprovider`` enum type (created in the
    # initial migration and switched to upper-case labels in 20f060b02772);
    # only GITHUB/GOOGLE rows are ever written here.
    provider_enum = sa.Enum(
        "BASIC",
        "GITHUB",
        "GOOGLE",
        name="oauthprovider",
        create_type=False,
    )
    op.create_table(
        "oauth_identities",
        sa.Column(
            "id",
            sa.BigInteger().with_variant(sa.Integer(), "sqlite"),
            nullable=False,
        ),
        sa.Column("provider", provider_enum, nullable=False),
        sa.Column("subject", sa.String(length=256), nullable=False),
        sa.Column(
            "user_id",
            sa.BigInteger().with_variant(sa.Integer(), "sqlite"),
            nullable=False,
        ),
        sa.Column(
            "provider_username",
            sa.String(length=256),
            nullable=True,
        ),
        sa.Column("email", sa.String(), nullable=True),
        sa.Column("name", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_oauth_identities_user_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "provider",
            "subject",
            name="uq_oauth_identities_provider_subject",
        ),
    )
    op.create_index(
        "idx_oauth_identities_user_id",
        "oauth_identities",
        ["user_id"],
        unique=False,
    )


def downgrade():
    op.drop_index(
        "idx_oauth_identities_user_id",
        table_name="oauth_identities",
    )
    op.drop_table("oauth_identities")
