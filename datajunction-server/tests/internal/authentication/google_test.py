"""
Tests for the Google OAuth token callback router.

The OIDC token verification/network is mocked; the router is invoked directly
with a Starlette request. These tests cover principal resolution keyed on the
immutable ``sub`` claim, legacy one-time claim, and conflicts.
"""

from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import MagicMock

import google_auth_oauthlib.flow
import pytest
from pytest_mock import MockerFixture
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.requests import Request

from datajunction_server.constants import AUTH_COOKIE
from datajunction_server.database.user import OAuthProvider, User
from datajunction_server.errors import (
    DJAuthenticationException,
    DJOAuthIdentityConflictException,
)
from datajunction_server.internal.access.authentication import google as google_internal
from datajunction_server.internal.access.authentication.tokens import decode_token
from datajunction_server.utils import get_settings


def _auth_cookie(response) -> str:
    for header in response.headers.getlist("set-cookie"):
        name, _, value = header.partition("=")
        if name == AUTH_COOKIE:
            return value.split(";", 1)[0]
    raise AssertionError(f"No {AUTH_COOKIE} cookie in response")


def _request() -> Request:
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "server": ("localhost", 8000),
        "root_path": "",
        "path": "/google/token/",
        "raw_path": b"/google/token/",
        "query_string": b"code=the-code",
        "headers": [],
    }
    return Request(scope)


def _patch_google(mocker: MockerFixture, idinfo: dict) -> None:
    mocker.patch(
        "datajunction_server.api.access.authentication.google.exchange_google_credentials",
        MagicMock(return_value=SimpleNamespace(_id_token="id-token")),
    )
    mocker.patch(
        "google.oauth2.id_token.verify_oauth2_token",
        MagicMock(return_value=idinfo),
    )


@pytest.mark.asyncio
async def test_google_callback_claims_legacy_user_then_binds(
    session: AsyncSession,
    mocker: MockerFixture,
):
    """
    A legacy Google user (username/email) is claimed once from the verified
    sub; subsequent email/name changes resolve to the same principal.
    """
    from datajunction_server.api.access.authentication import google as google_api

    legacy = User(
        username="g@example.com",
        password="hashed",
        email="g@example.com",
        name="Old",
        oauth_provider=OAuthProvider.GOOGLE,
    )
    session.add(legacy)
    await session.commit()
    await session.refresh(legacy)

    settings = get_settings()
    mocker.patch.object(settings, "google_oauth_client_secret_file", "/tmp/fake.json")
    _patch_google(
        mocker,
        {
            "sub": "g-sub-9",
            "email": "g@example.com",
            "name": "New Name",
            "email_verified": True,
        },
    )

    response = await google_api.get_access_token(
        _request(),
        session=session,
        settings=settings,
    )
    # RedirectResponse defaults to 307, matching the pre-existing callback.
    assert response.status_code == HTTPStatus.TEMPORARY_REDIRECT
    claims = decode_token(_auth_cookie(response))
    assert claims["uid"] == legacy.id
    assert claims["username"] == "g@example.com"

    # Email changes at the provider; sub stays the same.
    _patch_google(
        mocker,
        {
            "sub": "g-sub-9",
            "email": "g@newdomain.com",
            "name": "Newer Name",
            "email_verified": True,
        },
    )
    response = await google_api.get_access_token(
        _request(),
        session=session,
        settings=settings,
    )
    claims = decode_token(_auth_cookie(response))
    assert claims["uid"] == legacy.id
    assert claims["username"] == "g@example.com"  # local handle never changes
    refreshed = (
        await session.execute(select(User).where(User.id == legacy.id))
    ).scalar_one()
    assert refreshed.email == "g@newdomain.com"
    assert refreshed.name == "Newer Name"


@pytest.mark.asyncio
async def test_google_callback_conflict_issues_no_session(
    session: AsyncSession,
    mocker: MockerFixture,
):
    """
    A verified Google email matching a local password account is a conflict,
    not an auto-merge, and no session is issued.
    """
    from datajunction_server.api.access.authentication import google as google_api

    local = User(
        username="localguy",
        password="hashed",
        email="g@example.com",
        name="Local",
        oauth_provider=OAuthProvider.BASIC,
    )
    session.add(local)
    await session.commit()

    settings = get_settings()
    mocker.patch.object(settings, "google_oauth_client_secret_file", "/tmp/fake.json")
    _patch_google(
        mocker,
        {
            "sub": "g-sub-x",
            "email": "g@example.com",
            "name": "Google",
            "email_verified": True,
        },
    )

    with pytest.raises(DJOAuthIdentityConflictException) as exc_info:
        await google_api.get_access_token(
            _request(),
            session=session,
            settings=settings,
        )
    assert exc_info.value.http_status_code == HTTPStatus.CONFLICT
    assert exc_info.value.errors[0].debug["reason"] == "local_account"


def test_google_login_redirects_to_authorize_url(mocker: MockerFixture):
    from datajunction_server.api.access.authentication import google as google_api

    mocker.patch(
        "datajunction_server.api.access.authentication.google.get_authorize_url",
        return_value="https://accounts.google.com/auth",
    )
    response = google_api.login(target="/home")
    assert response.status_code == HTTPStatus.FOUND
    assert response.headers["location"] == "https://accounts.google.com/auth"


def _fake_flow_settings(mocker: MockerFixture, secret_file: str | None):
    fake_flow = MagicMock()
    fake_flow.authorization_url.return_value = (
        "https://accounts.google.com/o/auth?x=1",
        "state",
    )
    fake_flow.credentials = "CREDS"
    mocker.patch.object(
        google_auth_oauthlib.flow.Flow,
        "from_client_secrets_file",
        return_value=fake_flow,
    )
    mocker.patch(
        "datajunction_server.internal.access.authentication.google.get_settings",
        return_value=SimpleNamespace(
            google_oauth_client_secret_file=secret_file,
            url="http://localhost:8000/",
        ),
    )
    return fake_flow


def test_get_authorize_url_uses_fresh_flow(mocker: MockerFixture):
    fake_flow = _fake_flow_settings(mocker, "/tmp/fake.json")
    url = google_internal.get_authorize_url(state="target")
    assert url == "https://accounts.google.com/o/auth?x=1"
    kwargs = google_auth_oauthlib.flow.Flow.from_client_secrets_file.call_args.kwargs
    assert kwargs["redirect_uri"] == "http://localhost:8000/google/token/"
    fake_flow.authorization_url.assert_called_once()


def test_google_flow_requires_configuration(mocker: MockerFixture):
    _fake_flow_settings(mocker, None)
    with pytest.raises(DJAuthenticationException):
        google_internal.get_authorize_url()
    with pytest.raises(DJAuthenticationException):
        google_internal.exchange_google_credentials("http://localhost/code")


def test_exchange_google_credentials_returns_flow_credentials(mocker: MockerFixture):
    fake_flow = _fake_flow_settings(mocker, "/tmp/fake.json")
    credentials = google_internal.exchange_google_credentials(
        "http://localhost/code",
    )
    assert credentials == "CREDS"
    fake_flow.fetch_token.assert_called_once_with(
        authorization_response="http://localhost/code",
    )


@pytest.mark.asyncio
async def test_google_callback_propagates_provider_error(
    session: AsyncSession,
):
    from datajunction_server.api.access.authentication import google as google_api

    with pytest.raises(DJAuthenticationException):
        await google_api.get_access_token(
            _request(),
            error="access_denied",
            session=session,
            settings=get_settings(),
        )


@pytest.mark.asyncio
async def test_google_callback_requires_configuration(session: AsyncSession):
    from datajunction_server.api.access.authentication import google as google_api

    settings = get_settings()
    assert settings.google_oauth_client_secret_file is None
    with pytest.raises(DJAuthenticationException):
        await google_api.get_access_token(
            _request(),
            session=session,
            settings=settings,
        )


@pytest.mark.asyncio
async def test_google_callback_rejects_invalid_id_token(
    session: AsyncSession,
    mocker: MockerFixture,
):
    from datajunction_server.api.access.authentication import google as google_api

    settings = get_settings()
    mocker.patch.object(settings, "google_oauth_client_secret_file", "/tmp/fake.json")
    mocker.patch(
        "datajunction_server.api.access.authentication.google.exchange_google_credentials",
        MagicMock(return_value=SimpleNamespace(_id_token="bad")),
    )
    mocker.patch(
        "google.oauth2.id_token.verify_oauth2_token",
        MagicMock(side_effect=ValueError("invalid audience")),
    )
    with pytest.raises(DJAuthenticationException) as exc_info:
        await google_api.get_access_token(
            _request(),
            session=session,
            settings=settings,
        )
    assert exc_info.value.http_status_code == HTTPStatus.UNAUTHORIZED
