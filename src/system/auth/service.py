import logging
from typing import List, Optional

from sqlmodel import select

from settings import settings
from system.auth.db import get_session, init_db
from system.auth.models import User
from system.auth.security import hash_password, verify_password

logger = logging.getLogger(__name__)


def get_user(username: str) -> Optional[User]:
    with get_session() as s:
        return s.get(User, username)


def create_user(username: str, password: str, role: str = "user") -> User:
    with get_session() as s:
        if s.get(User, username):
            raise ValueError("user already exists")
        user = User(username=username, password_hash=hash_password(password), role=role)
        s.add(user)
        s.commit()
        s.refresh(user)
        return user


def authenticate_user(username: str, password: str) -> Optional[User]:
    user = get_user(username)
    if not user or not user.is_active:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user


def list_users() -> List[User]:
    with get_session() as s:
        return list(s.exec(select(User)).all())


def set_active(username: str, active: bool) -> Optional[User]:
    with get_session() as s:
        user = s.get(User, username)
        if not user:
            return None
        user.is_active = active
        s.add(user)
        s.commit()
        s.refresh(user)
        return user


def ensure_admin() -> None:
    """Create the env-configured admin on startup, if set and absent."""
    init_db()
    if not settings.ADMIN_PASSWORD:
        logger.info("ADMIN_PASSWORD not set — skipping admin bootstrap")
        return
    if get_user(settings.ADMIN_USERNAME):
        return
    create_user(settings.ADMIN_USERNAME, settings.ADMIN_PASSWORD, role="admin")
    logger.info("Bootstrapped admin user %r", settings.ADMIN_USERNAME)
