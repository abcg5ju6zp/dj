"""
Test internal http authentication logic
"""

import asyncio
from datetime import timedelta
from unittest.mock import MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from datajunction_server.database.user import OAuthProvider, User
from datajunction_server.errors import DJException
from datajunction_server.internal.access.authentication.http import DJHTTPBearer
from datajunction_server.internal.access.authentication.tokens import create_token
from datajunction_server.models.user import UserOutput
from datajunction_server.utils import get_settings


def test_dj_http_bearer_raise_when_unauthenticated():
    """
    Test raising when no cookie or auth headers are provided
    """
    bearer = DJHTTPBearer()
    request = MagicMock()
    request.cookies.get.return_value = None
    request.headers.get.return_value = None
    with pytest.raises(DJException) as exc_info:
        asyncio.run(bearer(request))
    assert "Not authenticated" in str(exc_info.value)


def test_dj_http_bearer_raise_with_empty_bearer_token():
    """
    Test raising when the authorization header has an empty bearer token
    """
    bearer = DJHTTPBearer()
    request = MagicMock()
    request.cookies.get.return_value = None
    request.headers.get.return_value = "Bearer "
    with pytest.raises(DJException) as exc_info:
        asyncio.run(bearer(request))
    assert "Not authenticated" in str(exc_info.value)


def test_dj_http_bearer_raise_with_unsupported_scheme(jwt_token):
    """
    Test raising when the authorization header scheme is not a supported one
    """
    bearer = DJHTTPBearer()
    request = MagicMock()
    request.cookies.get.return_value = None
    request.headers.get.return_value = f"Foo {jwt_token}"
    with pytest.raises(DJException) as exc_info:
        asyncio.run(bearer(request))
    assert "Invalid authentication credentials" in str(exc_info.value)


def test_dj_http_bearer_raise_with_non_jwt_token():
    """
    Test raising when the token can't be parsed as a JWT
    """
    bearer = DJHTTPBearer()
    request = MagicMock()
    request.cookies.get.return_value = None
    request.headers.get.return_value = "Foo NotAJWT"
    with pytest.raises(DJException) as exc_info:
        asyncio.run(bearer(request))
    assert "Invalid authentication credentials" in str(exc_info.value)


def test_dj_http_bearer_w_cookie(
    jwt_token: str,
    session: AsyncSession,
    current_user: User,
):
    """
    Test using the DJHTTPBearer middleware with a cookie
    """
    bearer = DJHTTPBearer()
    request = MagicMock()
    request.cookies.get.return_value = jwt_token

    asyncio.run(bearer(request, session))
    assert UserOutput.model_validate(request.state.user).model_dump() == {
        "id": 1,
        "username": "dj",
        "email": "dj@datajunction.io",
        "name": "DJ",
        "oauth_provider": OAuthProvider.BASIC,
        "is_admin": False,
        "last_viewed_notifications_at": None,
    }


def test_dj_http_bearer_w_auth_headers(
    jwt_token: str,
    session: AsyncSession,
    current_user: User,
):
    """
    Test using the DJHTTPBearer middleware with an authorization header
    """
    bearer = DJHTTPBearer()
    request = MagicMock()
    request.cookies.get.return_value = None
    request.headers.get.return_value = f"Bearer {jwt_token}"

    asyncio.run(bearer(request, session))
    assert UserOutput.model_validate(request.state.user).model_dump() == {
        "id": 1,
        "username": "dj",
        "email": "dj@datajunction.io",
        "name": "DJ",
        "oauth_provider": OAuthProvider.BASIC,
        "is_admin": False,
        "last_viewed_notifications_at": None,
    }


def test_raise_on_non_jwt_cookie():
    """
    Test using the DJHTTPBearer middleware with a cookie
    """
    bearer = DJHTTPBearer()
    request = MagicMock()
    request.cookies.get.return_value = "NotAJWT"
    with pytest.raises(DJException) as exc_info:
        asyncio.run(bearer(request))
    assert "Cannot decode authorization token" in str(exc_info.value)


async def test_uid_token_resolves_after_username_rename(session: AsyncSession):
    """
    A token carrying the immutable uid keeps pointing at the same principal
    even after the (provider or local) username has changed.
    """
    user = User(
        username="oauth_user",
        password="x",
        email="oauth@example.com",
        name="Before",
        oauth_provider=OAuthProvider.GITHUB,
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)

    settings = get_settings()
    token = create_token(
        {"uid": user.id, "username": "oauth_user"},
        secret=settings.secret,
        iss=settings.url,
        expires_delta=timedelta(hours=24),
    )

    # The provider profile changed the handle; local username is renamed.
    user.username = "oauth_user_renamed"
    await session.commit()

    bearer = DJHTTPBearer()
    request = MagicMock()
    request.cookies.get.return_value = token
    await bearer(request, session)
    assert request.state.user.id == user.id
    assert request.state.user.username == "oauth_user_renamed"


async def test_uid_token_for_missing_principal_is_rejected(session: AsyncSession):
    """
    A uid that no longer exists is an authentication failure, not a fallback
    to some other lookup.
    """
    settings = get_settings()
    token = create_token(
        {"uid": 999_999_999, "username": "ghost"},
        secret=settings.secret,
        iss=settings.url,
        expires_delta=timedelta(hours=24),
    )
    bearer = DJHTTPBearer()
    request = MagicMock()
    request.cookies.get.return_value = None
    request.headers.get.return_value = f"Bearer {token}"
    with pytest.raises(DJException) as exc_info:
        await bearer(request, session)
    assert "not found" in str(exc_info.value)
