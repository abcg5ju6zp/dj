"""
GitHub OAuth helper functions
"""

import logging
from urllib.parse import urljoin

import httpx

from datajunction_server.config import Settings
from datajunction_server.database.user import OAuthProvider
from datajunction_server.errors import DJAuthenticationException, DJError, ErrorCode
from datajunction_server.internal.access.authentication.identities import OAuthProfile
from datajunction_server.utils import get_settings

_logger = logging.getLogger(__name__)

GITHUB_ACCESS_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_USER_URL = "https://api.github.com/user"
REQUEST_TIMEOUT = 10


def get_authorize_url(oauth_client_id: str) -> str:
    """
    Get the authorize url for a GitHub OAuth app
    """
    settings = get_settings()
    redirect_uri = urljoin(settings.url, "/github/token/")
    return (
        f"https://github.com/login/oauth/authorize?client_id={oauth_client_id}"
        f"&scope=read:user&redirect_uri={redirect_uri}"
    )


async def exchange_github_code(
    code: str,
    settings: Settings | None = None,
) -> str:
    """
    Exchange an authorization code for a GitHub access token
    """
    settings = settings or get_settings()
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        response = await client.post(
            GITHUB_ACCESS_TOKEN_URL,
            params={
                "client_id": settings.github_oauth_client_id,
                "client_secret": settings.github_oauth_client_secret,
                "code": code,
            },
            headers={"Accept": "application/json"},
        )
    access_data = response.json()
    if "error" in access_data:
        raise DJAuthenticationException(
            errors=[
                DJError(
                    code=ErrorCode.OAUTH_ERROR,
                    message=(
                        "Received an error from the GitHub authorization "
                        f"server: {access_data['error']}"
                    ),
                ),
            ],
        )
    access_token = access_data.get("access_token")
    if not access_token:
        message = "No user access token retrieved from GitHub OAuth API"
        _logger.error(message)
        raise DJAuthenticationException(
            errors=[DJError(message=message, code=ErrorCode.OAUTH_ERROR)],
        )
    return access_token


async def fetch_github_profile(access_token: str) -> OAuthProfile:
    """
    Fetch the GitHub user's profile and normalize it. The immutable binding
    key is GitHub's numeric user ``id`` (never reused); ``login``/``email``/
    ``name`` are mutable display fields.
    """
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        response = await client.get(
            GITHUB_USER_URL,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {access_token}",
            },
        )
    if response.status_code != httpx.codes.OK:
        raise DJAuthenticationException(
            errors=[
                DJError(
                    code=ErrorCode.OAUTH_ERROR,
                    message=(
                        "Could not retrieve user using the GitHub provided "
                        f"access token ({response.status_code}): {response.text}"
                    ),
                ),
            ],
        )
    user_data = response.json()
    if "message" in user_data and user_data["message"] == "Bad credentials":
        raise DJAuthenticationException(
            "Cannot authorize user via GitHub, bad credentials",
        )
    if user_data.get("id") is None or not user_data.get("login"):
        raise DJAuthenticationException(
            "GitHub did not return the user id and login",
        )
    return OAuthProfile(
        provider=OAuthProvider.GITHUB,
        subject=str(user_data["id"]),
        username=user_data["login"],
        email=user_data.get("email"),
        name=user_data.get("name"),
    )
