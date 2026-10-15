"""Durable, owner-aware deployment lock for managed namespaces."""

from datetime import UTC, datetime
from functools import partial
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy_utils import UUIDType

from datajunction_server.database.base import Base
from datajunction_server.typing import UTCDatetime


class DeploymentNamespaceLock(Base):
    """
    An exclusive lock on a managed namespace, held by at most one deployment.

    Unlike a transaction-scoped advisory lock, this row is durable: it survives
    statement/transaction boundaries for the whole (possibly resumed) async
    execution. Ownership carries a strictly increasing ``fencing_token`` so a
    deployment whose lease expired and was fenced off can neither write
    progress nor release the new owner's lock — every mutation compares
    ``(owner_deployment_id, fencing_token)`` (compare-and-set).

    A dead owner (process restart, lost task) stops heart-beating
    ``lease_expires_at``; once the lease is in the past another deployment may
    take over with the next token.
    """

    __tablename__ = "deploymentnamespacelock"

    namespace: Mapped[str] = mapped_column(String, primary_key=True)
    owner_deployment_id: Mapped[UUID] = mapped_column(
        UUIDType(),
        ForeignKey("deployments.uuid", ondelete="CASCADE"),
        nullable=False,
    )
    fencing_token: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    lease_expires_at: Mapped[UTCDatetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
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
