"""OAuth identity bindings.

An :class:`OAuthIdentity` binds an external, provider-issued identity
(``(provider, subject)``) to a single local principal (:class:`User`).

The subject is the immutable identifier the provider assigns (Google's OIDC
``sub`` claim, GitHub's numeric user ``id``). Mutable profile fields such as
the provider login, email or display name are *not* part of the binding and
are only kept as a last-seen snapshot. All local ownership, membership and
authorization relations continue to hang off ``users.id``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from functools import partial

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from datajunction_server.database.base import Base
from datajunction_server.database.user import OAuthProvider, PROVIDER_ENUM
from datajunction_server.typing import UTCDatetime


class OAuthIdentity(Base):
    """
    An immutable binding between an OAuth ``(provider, subject)`` pair and a
    local user principal.
    """

    __tablename__ = "oauth_identities"
    __table_args__ = (
        UniqueConstraint(
            "provider",
            "subject",
            name="uq_oauth_identities_provider_subject",
        ),
        Index("idx_oauth_identities_user_id", "user_id"),
    )

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
    )
    provider: Mapped[OAuthProvider] = mapped_column(PROVIDER_ENUM)
    # Provider-issued, immutable subject identifier (Google ``sub`` / GitHub id)
    subject: Mapped[str] = mapped_column(String(256))
    user_id: Mapped[int] = mapped_column(
        ForeignKey(
            "users.id",
            name="fk_oauth_identities_user_id",
            ondelete="CASCADE",
        ),
    )

    # Last-seen mutable provider profile (display information only; never
    # used to resolve which principal is authenticating).
    provider_username: Mapped[str | None] = mapped_column(String(256), nullable=True)
    email: Mapped[str | None] = mapped_column(String, nullable=True)
    name: Mapped[str | None] = mapped_column(String, nullable=True)

    created_at: Mapped[UTCDatetime | None] = mapped_column(
        DateTime(timezone=True),
        insert_default=partial(datetime.now, UTC),
        nullable=True,
    )
    updated_at: Mapped[UTCDatetime | None] = mapped_column(
        DateTime(timezone=True),
        insert_default=partial(datetime.now, UTC),
        onupdate=partial(datetime.now, UTC),
        nullable=True,
    )
