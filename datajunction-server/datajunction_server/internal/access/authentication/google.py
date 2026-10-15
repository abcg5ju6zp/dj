"""
Google OAuth helper functions
"""

import logging
from urllib.parse import urljoin

import google_auth_oauthlib.flow
from google.auth.external_account_authorized_user import Credentials

from datajunction_server.errors import DJAuthenticationException
from datajunction_server.utils import get_settings

_logger = logging.getLogger(__name__)

SCOPES = [
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/userinfo.email",
    "openid",
]


def _build_flow(
    redirect_uri: str | None = None,
) -> google_auth_oauthlib.flow.Flow:
    """
    Build a fresh OAuth flow. A fresh flow per request avoids the shared mutable
    state of a module-global flow, which lets concurrent callbacks corrupt each
    other's token exchange.
    """
    settings = get_settings()
    if not settings.google_oauth_client_secret_file:
        raise DJAuthenticationException("Google OAuth is not configured")
    return google_auth_oauthlib.flow.Flow.from_client_secrets_file(
        settings.google_oauth_client_secret_file,
        scopes=SCOPES,
        redirect_uri=redirect_uri
        or urljoin(settings.url, "/google/token/"),
    )


def get_authorize_url(state: str | None = None) -> str:
    """
    Get the authorize url for a Google OAuth app
    """
    flow = _build_flow()
    authorization_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
        state=state,
    )
    return authorization_url


def exchange_google_credentials(
    authorization_response_url: str,
) -> Credentials:
    """
    Exchange an authorization code for Google credentials.
    """
    flow = _build_flow()
    flow.fetch_token(authorization_response=authorization_response_url)
    return flow.credentials
