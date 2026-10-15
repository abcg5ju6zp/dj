"""
Tests for GitHub OAuth authorize URL and the token callback router.
"""

from http import HTTPStatus

import httpx
import pytest
from pytest_mock import MockerFixture
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from datajunction_server.constants import AUTH_COOKIE
from datajunction_server.database.user import OAuthProvider, User
from datajunction_server.errors import (
    DJAuthenticationException,
    DJOAuthIdentityConflictException,
)
from datajunction_server.internal.access.authentication import github
from datajunction_server.internal.access.authentication.identities import (
    OAuthProfile,
)
from datajunction_server.internal.access.authentication.tokens import decode_token
from datajunction_server.utils import get_settings


def _patch_httpx(mocker: MockerFixture, handler) -> None:
    # Capture the unpatched class first: ``github.httpx`` is the global module,
    # so patching its ``AsyncClient`` attribute would otherwise make the
    # factory recurse into the mock.
    original_client = httpx.AsyncClient
    mocker.patch(
        "datajunction_server.internal.access.authentication.github.httpx.AsyncClient",
        side_effect=lambda *args, **kwargs: original_client(
            transport=httpx.MockTransport(handler),
        ),
    )


def _auth_cookie(response) -> str:
    for header in response.headers.getlist("set-cookie"):
        name, _, value = header.partition("=")
        if name == AUTH_COOKIE:
            return value.split(";", 1)[0]
    raise AssertionError(f"No {AUTH_COOKIE} cookie in response")


def test_get_authorize_url():
    """
    Test generating a GitHub OAuth authorize url for a GitHub app client ID
    """
    assert github.get_authorize_url("foo") == (
        "https://github.com/login/oauth/authorize?"
        "client_id=foo&scope=read:user&redirect_uri="
        "http://localhost:8000/github/token/"
    )


@pytest.mark.asyncio
async def test_github_callback_issues_session_for_bound_subject(
    session: AsyncSession,
    mocker: MockerFixture,
):
    """
    The callback binds by immutable GitHub id and issues a cookie carrying the
    local principal id; a later login with a changed login updates the same
    principal and re-issues a session for it.
    """
    from datajunction_server.api.access.authentication import github as github_api

    mocker.patch(
        "datajunction_server.internal.access.authentication.github.exchange_github_code",
        new=mocker.AsyncMock(return_value="access-token"),
    )
    mocker.patch(
        "datajunction_server.internal.access.authentication.github.fetch_github_profile",
        new=mocker.AsyncMock(
            return_value=OAuthProfile(
                provider=OAuthProvider.GITHUB,
                subject="4242",
                username="octocat",
                email="octo@example.com",
                name="Octo Cat",
            ),
        ),
    )

    response = await github_api.get_access_token(
        code="the-code",
        session=session,
        settings=get_settings(),
    )
    # RedirectResponse defaults to 307, matching the pre-existing callback.
    assert response.status_code == HTTPStatus.TEMPORARY_REDIRECT
    cookie = _auth_cookie(response)
    user = (
        await session.execute(select(User).where(User.username == "octocat"))
    ).scalar_one()
    claims = decode_token(cookie)
    assert claims["uid"] == user.id
    assert claims["username"] == "octocat"

    # Provider profile changes the login; the subject is unchanged.
    mocker.patch(
        "datajunction_server.internal.access.authentication.github.fetch_github_profile",
        new=mocker.AsyncMock(
            return_value=OAuthProfile(
                provider=OAuthProvider.GITHUB,
                subject="4242",
                username="octocat-renamed",
                email="octo@new.com",
                name="Octo Cat II",
            ),
        ),
    )
    response = await github_api.get_access_token(
        code="the-code-2",
        session=session,
        settings=get_settings(),
    )
    claims = decode_token(_auth_cookie(response))
    assert claims["uid"] == user.id  # same principal despite the rename
    assert claims["username"] == "octocat"  # local handle never changes


@pytest.mark.asyncio
async def test_github_callback_conflict_issues_no_session(
    session: AsyncSession,
    mocker: MockerFixture,
):
    """
    When the GitHub login collides with a local password account the callback
    raises the conflict result and no cookie/session is produced.
    """
    from datajunction_server.api.access.authentication import github as github_api

    local_user = User(
        username="octocat",
        password="hashed",
        email="local@example.com",
        name="Local",
        oauth_provider=OAuthProvider.BASIC,
    )
    session.add(local_user)
    await session.commit()

    mocker.patch(
        "datajunction_server.internal.access.authentication.github.exchange_github_code",
        new=mocker.AsyncMock(return_value="access-token"),
    )
    mocker.patch(
        "datajunction_server.internal.access.authentication.github.fetch_github_profile",
        new=mocker.AsyncMock(
            return_value=OAuthProfile(
                provider=OAuthProvider.GITHUB,
                subject="9999",
                username="octocat",
                email="local@example.com",
            ),
        ),
    )

    with pytest.raises(DJOAuthIdentityConflictException) as exc_info:
        await github_api.get_access_token(
            code="the-code",
            session=session,
            settings=get_settings(),
        )
    assert exc_info.value.http_status_code == HTTPStatus.CONFLICT
    assert exc_info.value.errors[0].debug["reason"] == "local_account"


@pytest.mark.asyncio
async def test_exchange_code_and_profile_normalized(mocker: MockerFixture):
    """
    The GitHub helpers use the immutable numeric id as subject and pass the
    login/email/name through as mutable profile fields.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/login/oauth/access_token":
            return httpx.Response(200, json={"access_token": "tok"})
        return httpx.Response(
            200,
            json={
                "id": 12345,
                "login": "octocat",
                "email": "o@example.com",
                "name": "Octo",
            },
        )

    _patch_httpx(mocker, handler)
    token = await github.exchange_github_code("code", get_settings())
    assert token == "tok"
    profile = await github.fetch_github_profile(token)
    assert profile.provider == OAuthProvider.GITHUB
    assert profile.subject == "12345"
    assert profile.username == "octocat"
    assert profile.email == "o@example.com"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"error": "bad_verification_code"},
        {"something_else": True},
    ],
)
async def test_exchange_code_failures(
    mocker: MockerFixture,
    payload,
):
    _patch_httpx(
        mocker,
        lambda request: httpx.Response(200, json=payload),
    )
    with pytest.raises(DJAuthenticationException):
        await github.exchange_github_code("code", get_settings())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status_code,payload",
    [
        (401, {"message": "Bad credentials"}),
        (500, {}),
        (200, {"message": "Bad credentials"}),
        (200, {"login": "octocat"}),  # missing id
        (200, {"id": 1}),  # missing login
    ],
)
async def test_fetch_profile_failures(
    mocker: MockerFixture,
    status_code,
    payload,
):
    _patch_httpx(
        mocker,
        lambda request: httpx.Response(status_code, json=payload),
    )
    with pytest.raises(DJAuthenticationException):
        await github.fetch_github_profile("tok")
