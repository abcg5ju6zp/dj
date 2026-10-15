"""
Google OAuth Router
"""

import logging
from datetime import timedelta
from http import HTTPStatus
from urllib.parse import urljoin, urlparse

import google.auth.transport.requests
import requests
from fastapi import APIRouter, Depends, Request
from google.oauth2 import id_token
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool
from starlette.responses import RedirectResponse

from datajunction_server.constants import AUTH_COOKIE, LOGGED_IN_FLAG_COOKIE
from datajunction_server.database.user import OAuthProvider
from datajunction_server.errors import DJAuthenticationException, DJError, ErrorCode
from datajunction_server.internal.access.authentication.google import (
    exchange_google_credentials,
    get_authorize_url,
)
from datajunction_server.internal.access.authentication.identities import (
    OAuthProfile,
    resolve_oauth_principal,
)
from datajunction_server.internal.access.authentication.tokens import create_token
from datajunction_server.utils import Settings, get_session, get_settings

_logger = logging.getLogger(__name__)
router = APIRouter(tags=["Google OAuth"])


@router.get("/google/login/", status_code=HTTPStatus.FOUND)
def login(target: str | None = None):
    """
    Login using Google OAuth
    """
    return RedirectResponse(
        url=get_authorize_url(state=target),
        status_code=HTTPStatus.FOUND,
    )


def _verify_google_id_token(credentials, audience: str) -> dict:
    """
    Verify the provider's OIDC id token synchronously (blocking google-auth call).
    """
    request_session = requests.session()
    token_request = google.auth.transport.requests.Request(session=request_session)
    return id_token.verify_oauth2_token(
        id_token=credentials._id_token,
        request=token_request,
        audience=audience,
    )


@router.get("/google/token/")
async def get_access_token(
    request: Request,
    state: str | None = None,
    error: str | None = None,
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
):
    """
    Perform a token exchange, exchanging a google auth code for a google access
    token. The verified OIDC id token's immutable ``sub`` claim (not the
    email/name) determines which local principal is authenticated. Binding
    lookup/creation and the profile refresh commit as one transaction before
    the session cookie is issued.
    """
    if error:
        raise DJAuthenticationException(
            f"Ran into an error during Google auth: {error}",
        )
    if not settings.google_oauth_client_secret_file:
        raise DJAuthenticationException("Google OAuth is not configured")
    hostname = urlparse(settings.url).hostname
    url = str(request.url)
    try:
        credentials = await run_in_threadpool(exchange_google_credentials, url)
        user_data = await run_in_threadpool(
            _verify_google_id_token,
            credentials,
            settings.google_oauth_client_id,
        )
    except ValueError as exc:
        raise DJAuthenticationException(
            http_status_code=HTTPStatus.UNAUTHORIZED,
            errors=[
                DJError(
                    code=ErrorCode.OAUTH_ERROR,
                    message=f"Invalid Google id token: {exc}",
                ),
            ],
        ) from exc

    profile = OAuthProfile(
        provider=OAuthProvider.GOOGLE,
        subject=user_data.get("sub") or "",
        username=user_data.get("email"),
        email=user_data.get("email"),
        name=user_data.get("name"),
        email_verified=bool(user_data.get("email_verified", False)),
    )
    # Resolves and commits atomically; the cookie is only issued afterwards.
    resolution = await resolve_oauth_principal(session, profile)
    _logger.info(
        "Google login resolved as %s for provider subject %s",
        resolution.status,
        profile.subject,
    )
    response = RedirectResponse(
        url=urljoin(settings.frontend_host, state or ""),
    )
    response.set_cookie(
        AUTH_COOKIE,
        create_token(
            {"uid": resolution.user.id, "username": resolution.user.username},
            secret=settings.secret,
            iss=settings.url,
            expires_delta=timedelta(days=365),
        ),
        httponly=True,
        samesite="none",
        secure=True,
        domain=hostname,
    )
    response.set_cookie(
        LOGGED_IN_FLAG_COOKIE,
        "true",
        samesite="none",
        secure=True,
        domain=hostname,
    )
    return response
