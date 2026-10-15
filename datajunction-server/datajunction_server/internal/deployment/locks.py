"""
Durable managed-namespace deployment locks with fencing tokens.

The lock is a row in ``deploymentnamespacelock`` (one per namespace), not a
transaction-scoped advisory lock: an async deployment spans multiple
transactions and may be resumed after a process restart. Ownership carries a
strictly increasing fencing token; every heartbeat, progress write and release
is a compare-and-set on ``(owner, token)``, so a deployment that was fenced off
after its lease expired can neither keep mutating state nor release the new
owner's lock.

Contention policy is fail-fast: an actively held lock makes a second create
fail with 409. A holder that reached a terminal status or let its lease expire
can be taken over with the next token.

PostgreSQL serializes the read-decide-write with a transaction-scoped advisory
lock; SQLite (used by lightweight local checks) relies on the primary key and a
single insert retry, with equivalent semantics.
"""

import logging
from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from datajunction_server.database.deployment import Deployment
from datajunction_server.database.deployment_lock import DeploymentNamespaceLock
from datajunction_server.errors import DJDeploymentConflictException
from datajunction_server.models.deployment import DeploymentStatus
from datajunction_server.utils import get_settings

logger = logging.getLogger(__name__)

_ADVISORY_KEY_PREFIX = "dj:deploylock:"


class _ConcurrentInsert(Exception):
    """Another session inserted the lock row between our read and insert."""


def _now() -> datetime:
    return datetime.now(UTC)


def _new_expiry(lease_seconds: int, now: datetime | None = None) -> datetime:
    return (now or _now()) + timedelta(seconds=lease_seconds)


def _is_postgres(session: AsyncSession) -> bool:
    return session.get_bind().dialect.name == "postgresql"


async def _serialize_namespace(session: AsyncSession, namespace: str) -> None:
    """Serialize acquire decisions for one namespace for this transaction."""
    if _is_postgres(session):
        await session.execute(
            text("SELECT pg_advisory_xact_lock(hashtext(:key))"),
            {"key": _ADVISORY_KEY_PREFIX + namespace},
        )


async def _owner_status(
    session: AsyncSession,
    owner_id: UUID,
) -> str | None:
    result = await session.execute(
        select(Deployment.status).where(Deployment.uuid == owner_id),
    )
    return result.scalar_one_or_none()


_TERMINAL_STATUS_VALUES = {status.value for status in DeploymentStatus.terminal()}


async def get_lock(
    session: AsyncSession,
    namespace: str,
) -> DeploymentNamespaceLock | None:
    """Return the current lock row for a namespace, if any."""
    return (
        await session.execute(
            select(DeploymentNamespaceLock).where(
                DeploymentNamespaceLock.namespace == namespace,
            ),
        )
    ).scalar_one_or_none()


async def _decide(
    session: AsyncSession,
    namespace: str,
    deployment_uuid: UUID,
    lease_seconds: int,
) -> int:
    """Read-own-decide-write. Caller must hold the advisory serialization lock
    on postgres. Returns the fencing token. Must run inside the caller's
    transaction; the caller commits to make ownership durable."""
    now = _now()
    row = (
        await session.execute(
            select(DeploymentNamespaceLock)
            .where(DeploymentNamespaceLock.namespace == namespace)
            .with_for_update(),
        )
    ).scalar_one_or_none()

    if row is None:
        new_row = DeploymentNamespaceLock(
            namespace=namespace,
            owner_deployment_id=deployment_uuid,
            fencing_token=1,
            lease_expires_at=_new_expiry(lease_seconds, now),
        )
        try:
            # Nested savepoint so a racing insert (SQLite, no advisory lock)
            # rolls back just the INSERT and lets us re-run the decision.
            async with session.begin_nested():
                session.add(new_row)
                await session.flush()
        except IntegrityError as exc:
            raise _ConcurrentInsert(namespace) from exc
        return 1

    if row.owner_deployment_id == deployment_uuid:
        # Idempotent re-acquire (resume): same owner keeps its token.
        row.lease_expires_at = _new_expiry(lease_seconds, now)
        await session.flush()
        return row.fencing_token

    status = await _owner_status(session, row.owner_deployment_id)
    status_value = status.value if hasattr(status, "value") else status
    terminal = status_value in _TERMINAL_STATUS_VALUES
    expired = row.lease_expires_at <= now
    if not (terminal or expired):
        raise DJDeploymentConflictException(
            message=(
                f"Namespace '{namespace}' is locked by an active deployment "
                f"'{row.owner_deployment_id}' until {row.lease_expires_at.isoformat()}. "
                "Wait for it to finish, cancel it, or retry after the lease "
                "expires."
            ),
            owner_deployment_id=str(row.owner_deployment_id),
            lease_expires_at=row.lease_expires_at.isoformat(),
        )

    reason = "terminal status" if terminal else "expired lease"
    logger.info(
        "Taking over deployment lock on %s from %s (token %d, %s) for %s",
        namespace,
        row.owner_deployment_id,
        row.fencing_token,
        reason,
        deployment_uuid,
    )
    row.owner_deployment_id = deployment_uuid
    row.fencing_token = row.fencing_token + 1
    row.lease_expires_at = _new_expiry(lease_seconds, now)
    await session.flush()
    return row.fencing_token


async def acquire_lock(
    session: AsyncSession,
    namespace: str,
    deployment_id: "str | UUID",
    lease_seconds: int | None = None,
) -> int:
    """
    Acquire the exclusive deployment lock for a managed namespace.

    Returns the fencing token. Raises DJDeploymentConflictException (409) when
    an active deployment holds the lock. The decision is committed by the
    caller, not here, so it stays atomic with creating the deployment row.
    """
    deployment_uuid = deployment_id if isinstance(deployment_id, UUID) else UUID(
        str(deployment_id),
    )
    lease_seconds = (
        lease_seconds
        if lease_seconds is not None
        else get_settings().deployment_lock_lease_seconds
    )
    await _serialize_namespace(session, namespace)
    try:
        return await _decide(session, namespace, deployment_uuid, lease_seconds)
    except _ConcurrentInsert:
        await session.rollback()
        await _serialize_namespace(session, namespace)
        return await _decide(session, namespace, deployment_uuid, lease_seconds)


async def renew_lock(
    session: AsyncSession,
    namespace: str,
    deployment_id: "str | UUID",
    fencing_token: int,
    lease_seconds: int | None = None,
) -> bool:
    """
    Heartbeat: extend the lease, but only while still the owner with the same
    fencing token. Returns False when fenced (lease was taken by a newer
    deployment) — the caller must stop.
    """
    deployment_uuid = deployment_id if isinstance(deployment_id, UUID) else UUID(
        str(deployment_id),
    )
    lease_seconds = (
        lease_seconds
        if lease_seconds is not None
        else get_settings().deployment_lock_lease_seconds
    )
    result = await session.execute(
        update(DeploymentNamespaceLock)
        .where(
            DeploymentNamespaceLock.namespace == namespace,
            DeploymentNamespaceLock.owner_deployment_id == deployment_uuid,
            DeploymentNamespaceLock.fencing_token == fencing_token,
        )
        .values(lease_expires_at=_new_expiry(lease_seconds)),
    )
    return (result.rowcount or 0) > 0


async def release_lock(
    session: AsyncSession,
    namespace: str,
    deployment_id: "str | UUID",
    fencing_token: int,
) -> bool:
    """
    Release the lock if still owned by this deployment/token. A stale owner
    (older fencing token, or a deployment that was cancelled/failed after a
    newer version took over) performs a no-op. Returns whether a row was
    deleted.
    """
    from sqlalchemy import delete

    deployment_uuid = deployment_id if isinstance(deployment_id, UUID) else UUID(
        str(deployment_id),
    )
    result = await session.execute(
        delete(DeploymentNamespaceLock).where(
            DeploymentNamespaceLock.namespace == namespace,
            DeploymentNamespaceLock.owner_deployment_id == deployment_uuid,
            DeploymentNamespaceLock.fencing_token == fencing_token,
        ),
    )
    return (result.rowcount or 0) > 0


async def is_active_owner(
    session: AsyncSession,
    namespace: str,
    deployment_id: "str | UUID",
    fencing_token: int,
) -> bool:
    """Whether this deployment/token still owns a non-expired lock."""
    deployment_uuid = deployment_id if isinstance(deployment_id, UUID) else UUID(
        str(deployment_id),
    )
    row = await get_lock(session, namespace)
    return (
        row is not None
        and row.owner_deployment_id == deployment_uuid
        and row.fencing_token == fencing_token
        and row.lease_expires_at > _now()
    )
