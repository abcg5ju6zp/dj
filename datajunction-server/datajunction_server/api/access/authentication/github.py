"""
GitHub OAuth Authentication Router
"""

import logging
from datetime import timedelta
from http import HTTPStatus

from fastapi import APIRouter, Depends
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession

from datajunction_server.constants import AUTH_COOKIE, LOGGED_IN_FLAG_COOKIE
from datajunction_server.errors import (
    DJConfigurationException,
    DJError,
    ErrorCode,
)
from datajunction_server.internal.access.authentication import github
from datajunction_server.internal.access.authentication.identities import (
    resolve_oauth_principal,
)
from datajunction_server.internal.access.authentication.tokens import create_token
from datajunction_server.utils import Settings, get_session, get_settings

_logger = logging.getLogger(__name__)
router = APIRouter(tags=["GitHub OAuth2"])


@router.get("/github/login/", status_code=HTTPStatus.FOUND)
def login() -> RedirectResponse:  # pragma: no cover
    """
    Login
    """
    settings = get_settings()
    if not settings.github_oauth_client_id:
        raise DJConfigurationException(
            http_status_code=HTTPStatus.NOT_IMPLEMENTED,
            errors=[
                DJError(
                    code=ErrorCode.OAUTH_ERROR,
                    message="GITHUB_OAUTH_CLIENT_ID is not set",
                ),
            ],
        )
    return RedirectResponse(
        url=github.get_authorize_url(oauth_client_id=settings.github_oauth_client_id),
        status_code=HTTPStatus.FOUND,
    )


@router.get("/github/token/")
async def get_access_token(
    code: str,
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> RedirectResponse:
    """
    Resolve a GitHub OAuth callback.

    The provider-issued, immutable user id (not the login or email) determines
    which local principal is authenticated. The binding lookup/creation and
    the profile refresh commit as one transaction; only then is the session
    cookie issued, so duplicate or lost callbacks can never create a second
    binding/principal or issue a session for the wrong account.
    """
    access_token = await github.exchange_github_code(code, settings)
    profile = await github.fetch_github_profile(access_token)
    # Resolves and commits atomically; the cookie is only issued afterwards.
    resolution = await resolve_oauth_principal(session, profile)
    _logger.info(
        "GitHub login resolved as %s for provider subject %s",
        resolution.status,
        profile.subject,
    )
    response = RedirectResponse(url=settings.frontend_host)
    response.set_cookie(
        AUTH_COOKIE,
        create_token(
            {"uid": resolution.user.id, "username": resolution.user.username},
            secret=settings.secret,
            iss=settings.url,
            expires_delta=timedelta(days=365),
        ),
        httponly=True,
    )
    response.set_cookie(
        LOGGED_IN_FLAG_COOKIE,
        "true",
    )
    return response
