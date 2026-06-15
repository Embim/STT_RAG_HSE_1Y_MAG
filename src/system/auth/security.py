from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

from settings import settings


def _bcrypt_bytes(password: str) -> bytes:
    # bcrypt only uses the first 72 bytes; bcrypt>=5 raises on longer input,
    # so truncate consistently for both hashing and verification.
    return password.encode("utf-8")[:72]


def hash_password(password: str) -> str:
    return bcrypt.hashpw(_bcrypt_bytes(password), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(_bcrypt_bytes(password), password_hash.encode("utf-8"))
    except ValueError:
        return False


def create_access_token(username: str, role: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": username,
        "role": role,
        "iat": now,
        "exp": now + timedelta(minutes=settings.JWT_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    return jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
