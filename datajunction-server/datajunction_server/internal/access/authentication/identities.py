"""
Resolve an authenticated OAuth identity into a local principal.

The binding key is ``(provider, subject)`` where ``subject`` is the immutable
identifier issued by the provider (Google's OIDC ``sub``, GitHub's numeric
user id). Provider logins, emails and display names are mutable and are only
used to:

* refresh display fields on an already bound principal, and
* decide whether a *legacy* (pre-binding) OAuth user can be claimed once,
  uniquely, or whether the login conflicts with an existing account.

``resolve_oauth_principal`` commits one atomic transaction per callback:
duplicate callbacks, concurrent first logins and retries after a lost
response all end with exactly one binding and one principal, or a
machine-readable conflict and no session.
"""

from __future__ import annotations

import logging
import secrets
from collections import defaultdict
from dataclasses import dataclass

from sqlalchemy import func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from datajunction_server.database.oauth import OAuthIdentity
from datajunction_server.database.user import OAuthProvider, User
from datajunction_server.enum import StrEnum
from datajunction_server.errors import (
    DJAuthenticationException,
    DJError,
    DJInternalErrorException,
    DJOAuthIdentityConflictException,
    ErrorCode,
)
from datajunction_server.internal.access.authentication.basic import get_password_hash

_logger = logging.getLogger(__name__)

OAUTH_PROVIDERS = (OAuthProvider.GITHUB, OAuthProvider.GOOGLE)


class OAuthResolutionStatus(StrEnum):
    """
    How the OAuth login was resolved to a local principal.
    """

    # A brand new principal (and its binding) was provisioned.
    CREATED = "created"
    # The provider/subject was already bound; display fields were refreshed.
    BOUND = "bound"
    # A legacy, unbound OAuth user was proven to be the same principal and the
    # binding was established once.
    CLAIMED = "claimed"


class OAuthConflictReason(StrEnum):
    """
    Machine-readable reasons for :class:`DJOAuthIdentityConflictException`.
    Callers (UI/CLI) use these to decide which remediation flow to offer.
    """

    # The login/email matches a password-based local account (including groups
    # and service accounts); OAuth must not log into it automatically.
    LOCAL_ACCOUNT = "local_account"
    # The profile matches an account provisioned through another provider.
    DIFFERENT_PROVIDER = "different_provider"
    # The matched account is already bound to a different provider subject.
    ALREADY_BOUND = "already_bound"
    # Existing records match, but ownership cannot be proven uniquely; they
    # stay unbound and no new principal is created.
    AMBIGUOUS_PROFILE = "ambiguous_profile"


@dataclass(frozen=True)
class OAuthProfile:
    """
    Normalized profile returned by an OAuth provider for a login callback.
    """

    provider: OAuthProvider
    # Immutable provider-issued subject identifier.
    subject: str
    # Provider handle (GitHub login, Google email address).
    username: str | None
    email: str | None = None
    name: str | None = None
    email_verified: bool = False


@dataclass
class OAuthResolution:
    """
    The result of resolving an OAuth login.
    """

    user: User
    identity: OAuthIdentity
    status: OAuthResolutionStatus


def _conflict(reason: OAuthConflictReason, profile: OAuthProfile) -> None:
    messages = {
        OAuthConflictReason.LOCAL_ACCOUNT: (
            "This login matches a local account that signs in with a password. "
            "Log in with that account or ask an administrator to link the "
            "OAuth identity."
        ),
        OAuthConflictReason.DIFFERENT_PROVIDER: (
            "This login matches an account that already signs in through a "
            "different provider. Log in with that provider or ask an "
            "administrator to link these identities."
        ),
        OAuthConflictReason.ALREADY_BOUND: (
            "The matching account is already linked to a different OAuth "
            "identity. Ask an administrator to review the account links."
        ),
        OAuthConflictReason.AMBIGUOUS_PROFILE: (
            "This login matches existing accounts, but the match cannot be "
            "proven uniquely. Ask an administrator to link the account "
            "manually."
        ),
    }
    _logger.warning(
        "OAuth identity conflict for provider=%s subject=%s reason=%s",
        profile.provider,
        profile.subject,
        reason,
    )
    raise DJOAuthIdentityConflictException(
        errors=[
            DJError(
                code=ErrorCode.OAUTH_IDENTITY_CONFLICT,
                message=messages[reason],
                debug={
                    "reason": reason.value,
                    "provider": profile.provider.value,
                },
            ),
        ],
    )


def _is_strong_match(user: User, profile: OAuthProfile) -> bool:
    """
    Whether a legacy OAuth user can be proven, uniquely, to be the same
    principal as the provider profile.

    The local username for legacy OAuth users was seeded from the provider's
    immutable-at-the-time handle: GitHub login for GitHub users, the email
    address for Google users. For Google we additionally require the provider
    to assert the email is verified, since otherwise anyone can complete the
    consent screen with an unverified address.

    Email-only matches are deliberately *not* strong enough: GitHub emails are
    optional/unverified and addresses may be reused.
    """
    if user.oauth_provider != profile.provider:
        return False
    if user.username != profile.username:
        return False
    # For Google the provider must assert the email; this is vacuous for
    # GitHub, whose strong key is the login.
    return not (profile.provider == OAuthProvider.GOOGLE and not profile.email_verified)


def _apply_display_fields(user: User, profile: OAuthProfile) -> None:
    """
    Copy mutable display fields onto the principal. The local username (the
    stable handle referenced by sessions) and the user id never change.
    """
    if profile.email is not None:
        user.email = profile.email
    if profile.name is not None:
        user.name = profile.name


def _snapshot_fields(identity: OAuthIdentity, profile: OAuthProfile) -> None:
    identity.provider_username = profile.username
    if profile.email is not None:
        identity.email = profile.email
    if profile.name is not None:
        identity.name = profile.name


async def _find_bound(
    session: AsyncSession,
    profile: OAuthProfile,
) -> OAuthResolution | None:
    """
    Find the binding for ``(provider, subject)``. The row is locked on
    Postgres so concurrent/duplicate callbacks for the same provider subject
    are serialized through the same binding.
    """
    statement = (
        select(OAuthIdentity, User)
        .join(User, User.id == OAuthIdentity.user_id)
        .where(
            OAuthIdentity.provider == profile.provider,
            OAuthIdentity.subject == profile.subject,
        )
        .with_for_update(of=OAuthIdentity)
    )
    row = (await session.execute(statement)).one_or_none()
    if row is None:
        return None
    identity, user = row
    return OAuthResolution(user, identity, OAuthResolutionStatus.BOUND)


async def _find_candidate_users(
    session: AsyncSession,
    profile: OAuthProfile,
) -> list[User]:
    """
    Existing principals whose username or (case-insensitive) email matches
    the provider profile. Rows are locked on Postgres so two different
    provider subjects can't race to claim the same legacy account.
    """
    conditions = [User.username == profile.username]
    if profile.email:
        conditions.append(func.lower(User.email) == func.lower(profile.email))
    statement = select(User).where(or_(*conditions)).with_for_update(of=User)
    result = await session.execute(statement)
    return list(result.scalars().unique().all())


async def _identities_for_users(
    session: AsyncSession,
    user_ids: list[int],
) -> dict[int, list[OAuthIdentity]]:
    if not user_ids:  # pragma: no cover - callers always pass candidates
        return {}
    result = await session.execute(
        select(OAuthIdentity).where(OAuthIdentity.user_id.in_(user_ids)),
    )
    grouped: dict[int, list[OAuthIdentity]] = defaultdict(list)
    for identity in result.scalars().all():
        grouped[identity.user_id].append(identity)
    return grouped


async def _create_principal(
    session: AsyncSession,
    profile: OAuthProfile,
) -> OAuthResolution:
    if not profile.username:
        raise DJAuthenticationException(
            f"OAuth provider {profile.provider} did not return a username",
        )
    user = User(
        username=profile.username,
        password=get_password_hash(secrets.token_urlsafe(13)),
        email=profile.email,
        name=profile.name,
        oauth_provider=profile.provider,
    )
    session.add(user)
    # Flush the user first so the binding row can reference its id; both rows
    # commit together or roll back together.
    await session.flush()
    identity = OAuthIdentity(
        provider=profile.provider,
        subject=profile.subject,
        user_id=user.id,
        provider_username=profile.username,
        email=profile.email,
        name=profile.name,
    )
    session.add(identity)
    await session.flush()
    return OAuthResolution(user, identity, OAuthResolutionStatus.CREATED)


async def _claim_legacy_user(
    session: AsyncSession,
    user: User,
    profile: OAuthProfile,
) -> OAuthResolution:
    _apply_display_fields(user, profile)
    identity = OAuthIdentity(
        provider=profile.provider,
        subject=profile.subject,
        user_id=user.id,
        provider_username=profile.username,
        email=profile.email,
        name=profile.name,
    )
    session.add(identity)
    await session.flush()
    return OAuthResolution(user, identity, OAuthResolutionStatus.CLAIMED)


async def _claim_or_create(
    session: AsyncSession,
    profile: OAuthProfile,
) -> OAuthResolution:
    candidates = await _find_candidate_users(session, profile)
    if candidates:
        identities = await _identities_for_users(
            session,
            [candidate.id for candidate in candidates],
        )
        # A concurrent first login may have committed a binding for this same
        # (provider, subject) while the candidate row lock was held. Converge
        # to that binding instead of flagging the user as already bound.
        for candidate in candidates:
            matching = next(
                (
                    identity
                    for identity in identities.get(candidate.id, [])
                    if identity.provider == profile.provider
                    and identity.subject == profile.subject
                ),
                None,
            )
            if matching is not None:
                _apply_display_fields(candidate, profile)
                _snapshot_fields(matching, profile)
                await session.flush()
                return OAuthResolution(
                    candidate,
                    matching,
                    OAuthResolutionStatus.BOUND,
                )
        claim_target: User | None = None
        first_blocker: OAuthConflictReason | None = None
        for candidate in candidates:
            if candidate.oauth_provider != profile.provider:
                reason = (
                    OAuthConflictReason.LOCAL_ACCOUNT
                    if candidate.oauth_provider == OAuthProvider.BASIC
                    else OAuthConflictReason.DIFFERENT_PROVIDER
                )
            elif identities.get(candidate.id):
                reason = OAuthConflictReason.ALREADY_BOUND
            elif _is_strong_match(candidate, profile):
                claim_target = candidate
                continue
            else:
                reason = OAuthConflictReason.AMBIGUOUS_PROFILE
            first_blocker = first_blocker or reason

        if first_blocker is not None:
            _conflict(first_blocker, profile)
        if claim_target is not None:
            _logger.info(
                "One-time claim of legacy user id=%s by %s/%s",
                claim_target.id,
                profile.provider,
                profile.subject,
            )
            return await _claim_legacy_user(session, claim_target, profile)
        # All candidates were blockers; ``_conflict`` raises, so this is
        # unreachable unless the classification changes.
        raise DJInternalErrorException(  # pragma: no cover
            "OAuth conflict classification produced no resolution",
        )
    return await _create_principal(session, profile)


async def resolve_oauth_principal(
    session: AsyncSession,
    profile: OAuthProfile,
    *,
    _is_retry: bool = False,
) -> OAuthResolution:
    """
    Resolve an authenticated OAuth profile into a local principal.

    Returns the bound principal for existing ``(provider, subject)`` bindings
    (refreshing display fields), claims a uniquely-proven legacy same-provider
    user once, or provisions a new principal when nothing matches. Raises
    :class:`DJOAuthIdentityConflictException` on local-account,
    cross-provider, already-bound or ambiguous matches.

    The lookup, binding/claim/provisioning and display refresh land in a
    single committed transaction; callers issue the session cookie only after
    it returns. If a concurrent first login wins the unique-constraint race,
    the transaction is rolled back and resolution is retried once so
    duplicate callbacks converge to the same binding instead of failing.
    """
    if profile.provider not in OAUTH_PROVIDERS:
        raise DJInternalErrorException(
            f"Cannot resolve an OAuth identity for provider {profile.provider}",
        )
    if not profile.subject:
        raise DJAuthenticationException(
            f"OAuth provider {profile.provider} did not return a subject",
        )

    bound = await _find_bound(session, profile)
    if bound is not None:
        _apply_display_fields(bound.user, profile)
        _snapshot_fields(bound.identity, profile)
        # The binding lookup/profile refresh is the single transactional
        # unit; the caller only issues the session cookie after it commits.
        await session.commit()
        return bound

    try:
        resolution = await _claim_or_create(session, profile)
    except IntegrityError:
        if _is_retry:
            raise
        _logger.warning(
            "Concurrent OAuth binding race for %s/%s; rolling back and re-converging",
            profile.provider,
            profile.subject,
        )
        await session.rollback()
        return await resolve_oauth_principal(session, profile, _is_retry=True)
    # Principal + binding (or one-time legacy claim + binding) land together.
    await session.commit()
    return resolution
