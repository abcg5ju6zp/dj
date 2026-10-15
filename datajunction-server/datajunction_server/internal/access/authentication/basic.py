"""
Basic OAuth and JWT helper functions
"""

import logging

from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import noload, selectinload
from sqlalchemy.sql.base import ExecutableOption

from datajunction_server.database.rbac import Role, RoleAssignment
from datajunction_server.database.user import User
from datajunction_server.errors import DJAuthenticationException, DJError, ErrorCode

_logger = logging.getLogger(__name__)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def validate_password_hash(plain_password, hashed_password) -> bool:
    """
    Verify a plain-text password against a hashed password
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password) -> str:
    """
    Returns a hashed version of a plain-text password
    """
    return pwd_context.hash(password)


def _authentication_options() -> list[ExecutableOption]:
    """
    Eager-loading options for a user resolved during authentication.
    """
    from datajunction_server.database.group_member import GroupMember

    return [
        # Load user's direct role assignments
        selectinload(User.role_assignments)
        .selectinload(RoleAssignment.role)
        .selectinload(Role.scopes),
        # Load user's group memberships and the groups' role assignments
        selectinload(User.member_of)
        .selectinload(GroupMember.group)
        .selectinload(User.role_assignments)
        .selectinload(RoleAssignment.role)
        .selectinload(Role.scopes),
        # Prevent loading unnecessary data during authentication
        noload(User.notification_preferences),
        noload(User.created_collections),
        noload(User.created_tags),
        noload(User.created_nodes),
        noload(User.owned_nodes),
    ]


async def get_user(
    username: str,
    session: AsyncSession,
    options: list[ExecutableOption] | None = None,
) -> User:
    """
    Get a DJ user by username.

    Username lookup is retained as a fallback for sessions issued before
    sessions carried the immutable user id; new logins resolve by id so a
    profile rename never changes which principal a session points to.
    """
    user = await User.get_by_username(
        session=session,
        username=username,
        options=options or _authentication_options(),
    )
    if not user:
        raise DJAuthenticationException(
            errors=[
                DJError(
                    message=f"User {username} not found",
                    code=ErrorCode.USER_NOT_FOUND,
                ),
            ],
        )
    return user


async def get_user_by_id(
    user_id: int,
    session: AsyncSession,
    options: list[ExecutableOption] | None = None,
) -> User:
    """
    Get a DJ principal by its immutable id. This is the authoritative path
    for resolving a session: a provider-side (or admin) username/email change
    leaves the id untouched, so existing sessions keep pointing at the same
    principal.
    """
    user = await User.get_by_id(
        session=session,
        user_id=user_id,
        options=options or _authentication_options(),
    )
    if not user:
        raise DJAuthenticationException(
            errors=[
                DJError(
                    message=f"User {user_id} not found",
                    code=ErrorCode.USER_NOT_FOUND,
                ),
            ],
        )
    return user


async def validate_user_password(
    username: str,
    password: str,
    session: AsyncSession,
) -> User:
    """
    Get a DJ user and verify that the provided password matches the hashed password
    """
    user = await get_user(username=username, session=session)
    if not validate_password_hash(password, user.password):
        raise DJAuthenticationException(
            errors=[
                DJError(
                    message=f"Invalid password for user {username}",
                    code=ErrorCode.INVALID_LOGIN_CREDENTIALS,
                ),
            ],
        )
    return user
