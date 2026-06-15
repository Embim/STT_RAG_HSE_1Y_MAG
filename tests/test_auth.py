import os
os.environ.setdefault("LLM_MODEL", "test/model")
os.environ.setdefault("LLM_API_KEY_1", "x")
os.environ.setdefault("LLM_API_KEY_2", "x")
os.environ.setdefault("LLM_API_KEY_3", "x")

import importlib
import pytest


@pytest.fixture
def auth(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTH_DB_PATH", str(tmp_path / "users.db"))
    from settings import settings as s
    s.AUTH_DB_PATH = str(tmp_path / "users.db")
    import system.auth.db as db
    db._engine = None
    import system.auth.service as service
    importlib.reload(service)
    db.init_db()
    return service


def test_hash_roundtrip():
    from system.auth.security import hash_password, verify_password
    h = hash_password("secret123")
    assert h != "secret123"
    assert verify_password("secret123", h)
    assert not verify_password("wrong", h)


def test_token_roundtrip():
    from system.auth.security import create_access_token, decode_token
    payload = decode_token(create_access_token("alice", "admin"))
    assert payload["sub"] == "alice" and payload["role"] == "admin"


def test_create_and_authenticate(auth):
    auth.create_user("bob", "pw", role="user")
    assert auth.authenticate_user("bob", "pw") is not None
    assert auth.authenticate_user("bob", "bad") is None
    assert auth.authenticate_user("ghost", "pw") is None


def test_duplicate_user_rejected(auth):
    auth.create_user("dup", "pw")
    with pytest.raises(ValueError):
        auth.create_user("dup", "pw2")


def test_inactive_user_cannot_authenticate(auth):
    auth.create_user("carol", "pw")
    auth.set_active("carol", False)
    assert auth.authenticate_user("carol", "pw") is None


def test_long_password_does_not_crash(auth):
    pw = "a" * 100  # > 72 bytes
    auth.create_user("longpw", pw)
    assert auth.authenticate_user("longpw", pw) is not None
