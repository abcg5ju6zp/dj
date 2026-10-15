"""
Tests for binding OAuth logins to local principals via the immutable
(provider, subject) key, including:

* first-login provisioning and profile-change refreshes (same principal),
* one-time claiming of legacy same-provider users,
* local-account / cross-provider / ambiguous / already-bound conflicts,
* duplicate callback idempotency and concurrent convergence.
"""

import asyncio

import pytest
from pytest_mock import MockerFixture
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from datajunction_server.database.oauth import OAuthIdentity
from datajunction_server.database.user import OAuthProvider, User
from datajunction_server.errors import (
    DJAuthenticationException,
    DJInternalErrorException,
    DJOAuthIdentityConflictException,
)
from datajunction_server.internal.access.authentication.identities import (
    OAuthConflictReason,
    OAuthProfile,
    OAuthResolutionStatus,
    _claim_or_create,
    _is_strong_match,
    resolve_oauth_principal,
)


def gh_profile(subject="gh-sub-1", username="alice", **overrides) -> OAuthProfile:
    defaults = {"email": "alice@example.com", "name": "Alice"}
    defaults.update(overrides)
    return OAuthProfile(
        provider=OAuthProvider.GITHUB,
        subject=subject,
        username=username,
        **defaults,
    )


def google_profile(
    subject="g-sub-1", email="alice@example.com", **overrides
) -> OAuthProfile:
    defaults = {"name": "Alice", "email_verified": True}
    defaults.update(overrides)
    return OAuthProfile(
        provider=OAuthProvider.GOOGLE,
        subject=subject,
        username=defaults.get("username", email),
        email=email,
        name=defaults["name"],
        email_verified=defaults["email_verified"],
    )


async def _add_user(
    session: AsyncSession,
    username: str,
    provider: OAuthProvider,
    email: str | None = None,
    name: str | None = None,
) -> User:
    user = User(
        username=username,
        password="hashed",
        email=email,
        name=name,
        oauth_provider=provider,
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user


async def _add_identity(
    session: AsyncSession,
    user: User,
    provider: OAuthProvider,
    subject: str,
    username: str | None = None,
) -> OAuthIdentity:
    identity = OAuthIdentity(
        provider=provider,
        subject=subject,
        user_id=user.id,
        provider_username=username or user.username,
        email=user.email,
        name=user.name,
    )
    session.add(identity)
    await session.commit()
    await session.refresh(identity)
    return identity


async def _identity_count(session: AsyncSession) -> int:
    return (
        await session.execute(select(func.count()).select_from(OAuthIdentity))
    ).scalar_one()


async def _user_count(session: AsyncSession, username: str) -> int:
    return (
        await session.execute(
            select(func.count()).select_from(User).where(User.username == username),
        )
    ).scalar_one()


async def _get_identity(
    session: AsyncSession,
    provider: OAuthProvider,
    subject: str,
) -> OAuthIdentity | None:
    return (
        await session.execute(
            select(OAuthIdentity).where(
                OAuthIdentity.provider == provider,
                OAuthIdentity.subject == subject,
            ),
        )
    ).scalar_one_or_none()


@pytest.mark.asyncio
async def test_first_login_creates_principal_and_binding(session: AsyncSession):
    """
    A brand new (provider, subject) provisions exactly one user and one
    binding, bound by the immutable subject.
    """
    resolution = await resolve_oauth_principal(session, gh_profile())
    assert resolution.status == OAuthResolutionStatus.CREATED
    identity = await _get_identity(session, OAuthProvider.GITHUB, "gh-sub-1")
    assert identity is not None
    assert identity.user_id == resolution.user.id
    assert identity.subject == "gh-sub-1"
    assert identity.provider_username == "alice"
    assert resolution.user.oauth_provider == OAuthProvider.GITHUB
    assert await _identity_count(session) == 1


@pytest.mark.asyncio
async def test_profile_change_keeps_principal_and_updates_display(
    session: AsyncSession,
):
    """
    After a provider changes login/email/name, the subject resolves to the
    same local principal; only display fields change. The local username
    (session handle) is untouched.
    """
    first = await resolve_oauth_principal(session, gh_profile())
    original_id = first.user.id
    original_username = first.user.username

    second = await resolve_oauth_principal(
        session,
        gh_profile(
            username="alice-renamed",
            email="alice@newdomain.com",
            name="Alice Renamed",
        ),
    )
    assert second.status == OAuthResolutionStatus.BOUND
    assert second.user.id == original_id
    assert second.user.username == original_username
    assert second.user.email == "alice@newdomain.com"
    assert second.user.name == "Alice Renamed"
    assert second.identity.provider_username == "alice-renamed"
    assert await _identity_count(session) == 1
    assert await _user_count(session, original_username) == 1
    assert await _user_count(session, "alice-renamed") == 0


@pytest.mark.asyncio
async def test_bound_login_ignores_username_collision_with_other_user(
    session: AsyncSession,
):
    """
    Once bound, a subject always logs into its bound principal even if its new
    login happens to equal another account's username. The binding wins.
    """
    await resolve_oauth_principal(session, gh_profile(username="alice"))
    await _add_user(
        session,
        username="alice-renamed",
        provider=OAuthProvider.BASIC,
        email="other@example.com",
    )
    resolution = await resolve_oauth_principal(
        session,
        gh_profile(username="alice-renamed"),
    )
    assert resolution.status == OAuthResolutionStatus.BOUND
    assert resolution.user.username == "alice"
    assert await _identity_count(session) == 1


@pytest.mark.asyncio
async def test_conflict_with_basic_account(session: AsyncSession):
    """
    A first login whose login matches a password-based local account is
    refused with an actionable conflict; nothing is bound or created.
    """
    await _add_user(
        session,
        username="alice",
        provider=OAuthProvider.BASIC,
        email="alice@example.com",
    )
    with pytest.raises(DJOAuthIdentityConflictException) as exc_info:
        await resolve_oauth_principal(session, gh_profile())
    assert exc_info.value.http_status_code == 409
    error = exc_info.value.errors[0]
    assert error.debug["reason"] == OAuthConflictReason.LOCAL_ACCOUNT
    assert await _identity_count(session) == 0
    assert await _user_count(session, "alice") == 1


@pytest.mark.asyncio
async def test_conflict_with_account_from_another_provider(session: AsyncSession):
    """
    A profile matching an account that belongs to another provider is not
    auto-merged.
    """
    await _add_user(
        session,
        username="sam@example.com",
        provider=OAuthProvider.GOOGLE,
        email="sam@example.com",
    )
    with pytest.raises(DJOAuthIdentityConflictException) as exc_info:
        await resolve_oauth_principal(
            session,
            gh_profile(username="sam@example.com", email="sam@example.com"),
        )
    assert (
        exc_info.value.errors[0].debug["reason"]
        == OAuthConflictReason.DIFFERENT_PROVIDER
    )
    assert await _identity_count(session) == 0


@pytest.mark.asyncio
async def test_legacy_github_user_is_claimed_once(session: AsyncSession):
    """
    A legacy GitHub user (oauth_provider=github, no binding) whose username is
    the old GitHub login is claimed once, uniquely, and thereafter bound.
    """
    legacy = await _add_user(
        session,
        username="oldlogin",
        provider=OAuthProvider.GITHUB,
        email="old@example.com",
        name="Old",
    )
    resolution = await resolve_oauth_principal(
        session,
        gh_profile(
            username="oldlogin",
            email="new@example.com",
            name="New Name",
        ),
    )
    assert resolution.status == OAuthResolutionStatus.CLAIMED
    assert resolution.user.id == legacy.id
    # Display info refreshed but principal/username preserved.
    assert resolution.user.username == "oldlogin"
    assert resolution.user.email == "new@example.com"
    identity = await _get_identity(session, OAuthProvider.GITHUB, "gh-sub-1")
    assert identity is not None
    assert identity.user_id == legacy.id

    # Next login with the same subject is the normal bound path.
    again = await resolve_oauth_principal(
        session,
        gh_profile(username="newlogin", email="new@example.com"),
    )
    assert again.status == OAuthResolutionStatus.BOUND
    assert again.user.id == legacy.id
    assert await _identity_count(session) == 1


@pytest.mark.asyncio
async def test_weak_email_only_match_stays_unbound(session: AsyncSession):
    """
    A new subject whose only overlap with a legacy same-provider user is the
    (unverified) email cannot be proven to own that account: it stays
    unbound, no second principal is created, and the login is refused.
    """
    await _add_user(
        session,
        username="someoneelse",
        provider=OAuthProvider.GITHUB,
        email="shared@example.com",
    )
    with pytest.raises(DJOAuthIdentityConflictException) as exc_info:
        await resolve_oauth_principal(
            session,
            gh_profile(
                subject="gh-sub-new",
                username="brandnew",
                email="shared@example.com",
            ),
        )
    assert (
        exc_info.value.errors[0].debug["reason"]
        == OAuthConflictReason.AMBIGUOUS_PROFILE
    )
    assert await _identity_count(session) == 0
    assert await _user_count(session, "brandnew") == 0


@pytest.mark.asyncio
async def test_legacy_google_claim_requires_verified_email(session: AsyncSession):
    """
    Legacy Google users are only claimed when the provider asserts the email.
    """
    legacy = await _add_user(
        session,
        username="g@example.com",
        provider=OAuthProvider.GOOGLE,
        email="g@example.com",
    )
    with pytest.raises(DJOAuthIdentityConflictException) as exc_info:
        await resolve_oauth_principal(
            session,
            google_profile(email="g@example.com", email_verified=False),
        )
    assert (
        exc_info.value.errors[0].debug["reason"]
        == OAuthConflictReason.AMBIGUOUS_PROFILE
    )
    assert await _identity_count(session) == 0

    resolution = await resolve_oauth_principal(
        session,
        google_profile(email="g@example.com", email_verified=True),
    )
    assert resolution.status == OAuthResolutionStatus.CLAIMED
    assert resolution.user.id == legacy.id
    assert await _identity_count(session) == 1


@pytest.mark.asyncio
async def test_legacy_user_bound_to_other_subject_is_not_claimable(
    session: AsyncSession,
):
    """
    If the strongly-matching legacy user already has a binding for a different
    subject, auto-claim is refused.
    """
    legacy = await _add_user(
        session,
        username="takenlogin",
        provider=OAuthProvider.GITHUB,
        email="t@example.com",
    )
    await _add_identity(
        session,
        legacy,
        OAuthProvider.GITHUB,
        subject="original-subject",
        username="takenlogin",
    )
    with pytest.raises(DJOAuthIdentityConflictException) as exc_info:
        await resolve_oauth_principal(
            session,
            gh_profile(subject="different-subject", username="takenlogin"),
        )
    assert exc_info.value.errors[0].debug["reason"] == OAuthConflictReason.ALREADY_BOUND
    assert await _identity_count(session) == 1


@pytest.mark.asyncio
async def test_email_collision_with_local_account_blocks_new_principal(
    session: AsyncSession,
):
    """
    Even a distinct username must not silently create a second principal when
    the provider email matches an existing local account.
    """
    await _add_user(
        session,
        username="basicguy",
        provider=OAuthProvider.BASIC,
        email="Sam@Example.com",
    )
    with pytest.raises(DJOAuthIdentityConflictException) as exc_info:
        await resolve_oauth_principal(
            session,
            google_profile(email="sam@example.com"),
        )
    assert exc_info.value.errors[0].debug["reason"] == OAuthConflictReason.LOCAL_ACCOUNT
    assert await _user_count(session, "sam@example.com") == 0
    assert await _identity_count(session) == 0


@pytest.mark.asyncio
async def test_duplicate_callback_is_idempotent(session: AsyncSession):
    """
    Two sequential callbacks with the same code outcome (same subject) converge
    to one user and one binding.
    """
    first = await resolve_oauth_principal(session, gh_profile())
    second = await resolve_oauth_principal(session, gh_profile())
    assert second.status == OAuthResolutionStatus.BOUND
    assert first.user.id == second.user.id
    assert await _identity_count(session) == 1
    assert await _user_count(session, "alice") == 1


@pytest.mark.asyncio
async def test_concurrent_first_logins_converge(session: AsyncSession):
    """
    Concurrent callbacks for a new subject converge to a single
    principal/binding instead of creating duplicates or failing.
    """
    profile = gh_profile(subject="gh-race", username="racer")
    second_session = AsyncSession(session.bind, expire_on_commit=False)
    try:
        first, second = await asyncio.gather(
            resolve_oauth_principal(session, profile),
            resolve_oauth_principal(second_session, profile),
        )
        # Capture while both sessions are still open.
        assert first.user.id == second.user.id
        assert await _identity_count(session) == 1
        assert await _user_count(session, "racer") == 1
    finally:
        await second_session.close()


@pytest.mark.asyncio
async def test_concurrent_different_subjects_same_login_do_not_merge(
    session: AsyncSession,
):
    """
    Two different provider subjects presenting the same login cannot both
    claim it: one binds, the other gets a conflict rather than a duplicate or
    a wrong-account session.
    """
    second_session = AsyncSession(session.bind, expire_on_commit=False)
    try:
        results = await asyncio.gather(
            resolve_oauth_principal(
                session,
                gh_profile(subject="subject-a", username="recycled"),
            ),
            resolve_oauth_principal(
                second_session,
                gh_profile(subject="subject-b", username="recycled"),
            ),
            return_exceptions=True,
        )
    finally:
        await second_session.close()

    bound = [r for r in results if not isinstance(r, Exception)]
    conflicts = [r for r in results if isinstance(r, Exception)]
    assert len(bound) == 1
    assert len(conflicts) == 1
    assert isinstance(conflicts[0], DJOAuthIdentityConflictException)
    assert await _identity_count(session) == 1
    assert await _user_count(session, "recycled") == 1


@pytest.mark.asyncio
async def test_first_login_without_email_still_provisions(session: AsyncSession):
    """
    GitHub emails can be absent; the binding/provisioning must still work and
    candidate matching must not compare against a null email.
    """
    resolution = await resolve_oauth_principal(
        session,
        gh_profile(subject="gh-noemail", username="noemail", email=None),
    )
    assert resolution.status == OAuthResolutionStatus.CREATED
    assert resolution.user.email is None
    assert resolution.identity.email is None


@pytest.mark.asyncio
async def test_bound_refresh_keeps_fields_when_provider_omits_them(
    session: AsyncSession,
):
    """
    A callback that omits email/name must not blank out previously known
    display fields.
    """
    await resolve_oauth_principal(
        session,
        gh_profile(subject="gh-keep", username="keep", email="keep@example.com"),
    )
    resolution = await resolve_oauth_principal(
        session,
        gh_profile(
            subject="gh-keep",
            username="keep",
            email=None,
            name=None,
        ),
    )
    assert resolution.status == OAuthResolutionStatus.BOUND
    assert resolution.user.email == "keep@example.com"
    assert resolution.identity.email == "keep@example.com"


def test_strong_match_rules():
    """
    Strong matches are same-provider, same-local-username, and verified email
    for Google.
    """
    google_legacy = User(
        username="g@example.com",
        oauth_provider=OAuthProvider.GOOGLE,
    )
    github_legacy = User(
        username="login",
        oauth_provider=OAuthProvider.GITHUB,
    )
    assert _is_strong_match(
        github_legacy,
        gh_profile(username="login"),
    )
    assert not _is_strong_match(
        github_legacy,
        gh_profile(username="other"),
    )
    assert not _is_strong_match(
        github_legacy,
        OAuthProfile(OAuthProvider.GOOGLE, "x", username="login"),
    )
    assert _is_strong_match(
        google_legacy,
        google_profile(email="g@example.com", email_verified=True),
    )
    assert not _is_strong_match(
        google_legacy,
        google_profile(email="g@example.com", email_verified=False),
    )


@pytest.mark.asyncio
async def test_rejects_non_oauth_provider(session: AsyncSession):
    with pytest.raises(DJInternalErrorException):
        await resolve_oauth_principal(
            session,
            OAuthProfile(OAuthProvider.BASIC, "x", username="basic"),
        )


@pytest.mark.asyncio
async def test_rejects_missing_subject(session: AsyncSession):
    with pytest.raises(DJAuthenticationException):
        await resolve_oauth_principal(
            session,
            OAuthProfile(OAuthProvider.GITHUB, "", username="nosub"),
        )


@pytest.mark.asyncio
async def test_rejects_missing_username_on_provision(session: AsyncSession):
    with pytest.raises(DJAuthenticationException):
        await resolve_oauth_principal(
            session,
            OAuthProfile(OAuthProvider.GITHUB, "s", username=None),
        )


@pytest.mark.asyncio
async def test_claim_or_create_converges_when_binding_committed_during_wait(
    session: AsyncSession,
):
    """
    If an identity for the same subject is committed between the bound lookup
    and the candidate lookup (row-lock wait), resolution converges to BOUND
    instead of flagging the candidate as already bound.
    """
    user = await _add_user(
        session,
        username="justbound",
        provider=OAuthProvider.GITHUB,
    )
    await _add_identity(
        session,
        user,
        OAuthProvider.GITHUB,
        subject="late-subject",
        username="justbound",
    )
    # Bypasses the top-level bound lookup to simulate the interleaving.
    resolution = await _claim_or_create(
        session,
        gh_profile(subject="late-subject", username="justbound"),
    )
    assert resolution.status == OAuthResolutionStatus.BOUND
    assert resolution.user.id == user.id


@pytest.mark.asyncio
async def test_double_integrity_error_is_not_retried_forever(
    session: AsyncSession,
    mocker: MockerFixture,
):
    """
    If the race persists after one retry, the integrity error surfaces instead
    of looping.
    """
    error = IntegrityError("insert", {}, RuntimeError("duplicate key"))
    mocker.patch(
        "datajunction_server.internal.access.authentication.identities._claim_or_create",
        new=mocker.AsyncMock(side_effect=error),
    )
    with pytest.raises(IntegrityError):
        await resolve_oauth_principal(session, gh_profile())
