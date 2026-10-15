"""
Unit tests for the durable managed-namespace deployment lock (fencing tokens,
lease takeover, compare-and-set heartbeat/release).
"""

from datetime import UTC, datetime, timedelta

import pytest

from datajunction_server.database.deployment import Deployment
from datajunction_server.database.user import OAuthProvider, User
from datajunction_server.errors import DJDeploymentConflictException
from datajunction_server.internal.deployment import locks
from datajunction_server.models.deployment import DeploymentStatus


async def _make_deployment(session, suffix: str) -> Deployment:
    user = User(
        username=f"locker_{suffix}",
        email=f"locker_{suffix}@example.com",
        oauth_provider=OAuthProvider.BASIC,
    )
    session.add(user)
    await session.flush()
    deployment = Deployment(
        namespace=f"ns_{suffix}",
        spec={"namespace": f"ns_{suffix}"},
        results={},
        status=DeploymentStatus.PENDING,
        created_by_id=user.id,
    )
    session.add(deployment)
    await session.commit()
    return deployment


async def test_first_acquire_returns_token_one(clean_session):
    deployment = await _make_deployment(clean_session, "a")
    token = await locks.acquire_lock(
        clean_session,
        "shared.ns",
        deployment.uuid,
        lease_seconds=60,
    )
    assert token == 1
    await clean_session.commit()
    row = await locks.get_lock(clean_session, "shared.ns")
    assert row is not None
    assert row.owner_deployment_id == deployment.uuid
    assert row.fencing_token == 1
    assert row.lease_expires_at > datetime.now(UTC)


async def test_contended_lock_raises_409_with_owner(clean_session):
    d1 = await _make_deployment(clean_session, "c1")
    d2 = await _make_deployment(clean_session, "c2")
    await locks.acquire_lock(clean_session, "ns_contended", d1.uuid, 60)
    await clean_session.commit()

    with pytest.raises(DJDeploymentConflictException) as exc:
        await locks.acquire_lock(clean_session, "ns_contended", d2.uuid, 60)
    assert exc.value.owner_deployment_id == str(d1.uuid)
    assert exc.value.http_status_code == 409


async def test_heartbeat_and_release_are_fenced(clean_session):
    d1 = await _make_deployment(clean_session, "f1")
    d2 = await _make_deployment(clean_session, "f2")
    token = await locks.acquire_lock(clean_session, "ns_fence", d1.uuid, 60)
    await clean_session.commit()

    # Old/wrong token cannot heartbeat or release.
    assert not await locks.renew_lock(
        clean_session,
        "ns_fence",
        d1.uuid,
        token + 99,
    )
    assert not await locks.release_lock(
        clean_session,
        "ns_fence",
        d1.uuid,
        token + 99,
    )
    row = await locks.get_lock(clean_session, "ns_fence")
    assert row.owner_deployment_id == d1.uuid

    assert await locks.renew_lock(
        clean_session,
        "ns_fence",
        d1.uuid,
        token,
    )
    assert await locks.is_active_owner(
        clean_session,
        "ns_fence",
        d1.uuid,
        token,
    )

    # d2 takes over after lease expiry: backdate the lease directly to
    # simulate a dead owner whose heartbeat has stopped.
    row = await locks.get_lock(clean_session, "ns_fence")
    row.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
    await clean_session.commit()

    new_token = await locks.acquire_lock(clean_session, "ns_fence", d2.uuid, 60)
    assert new_token == token + 1
    await clean_session.commit()

    # d1's stale token is now powerless; d2 owns the lock.
    assert not await locks.renew_lock(
        clean_session,
        "ns_fence",
        d1.uuid,
        token,
    )
    assert not await locks.release_lock(
        clean_session,
        "ns_fence",
        d1.uuid,
        token,
    )
    row = await locks.get_lock(clean_session, "ns_fence")
    assert row.owner_deployment_id == d2.uuid
    assert row.fencing_token == 2
    assert await locks.is_active_owner(
        clean_session,
        "ns_fence",
        d2.uuid,
        new_token,
    )

    # New owner releases normally.
    assert await locks.release_lock(
        clean_session,
        "ns_fence",
        d2.uuid,
        new_token,
    )
    assert await locks.get_lock(clean_session, "ns_fence") is None


async def test_terminal_holder_is_takeover_immediately(clean_session):
    d1 = await _make_deployment(clean_session, "t1")
    d2 = await _make_deployment(clean_session, "t2")
    await locks.acquire_lock(clean_session, "ns_terminal", d1.uuid, 600)
    await clean_session.commit()

    d1.status = DeploymentStatus.FAILED
    await clean_session.commit()

    # Lease is still valid, but a terminal owner yields immediately.
    new_token = await locks.acquire_lock(
        clean_session,
        "ns_terminal",
        d2.uuid,
        600,
    )
    assert new_token == 2
    await clean_session.commit()
    row = await locks.get_lock(clean_session, "ns_terminal")
    assert row.owner_deployment_id == d2.uuid


async def test_same_owner_reacquire_keeps_token(clean_session):
    d1 = await _make_deployment(clean_session, "r1")
    first = await locks.acquire_lock(clean_session, "ns_same", d1.uuid, 60)
    await clean_session.commit()
    second = await locks.acquire_lock(clean_session, "ns_same", d1.uuid, 60)
    assert first == second == 1


async def test_stale_owner_cannot_release_newer_lock(clean_session):
    """AC-16: cancelled/failed old deployment cannot unlock a later version."""
    d1 = await _make_deployment(clean_session, "o1")
    d2 = await _make_deployment(clean_session, "o2")
    token1 = await locks.acquire_lock(clean_session, "ns_old", d1.uuid, -60)
    await clean_session.commit()
    token2 = await locks.acquire_lock(clean_session, "ns_old", d2.uuid, 60)
    await clean_session.commit()
    assert token2 > token1

    d1.status = DeploymentStatus.CANCELLED
    await clean_session.commit()
    assert not await locks.release_lock(
        clean_session,
        "ns_old",
        d1.uuid,
        token1,
    )
    row = await locks.get_lock(clean_session, "ns_old")
    assert row.owner_deployment_id == d2.uuid and row.fencing_token == token2
