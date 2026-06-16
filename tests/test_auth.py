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


def _client(auth):
    from fastapi.testclient import TestClient
    from unittest.mock import patch
    import importlib
    with patch("system.rag.vectore_store.weaviate.connect_to_local"):
        import api.main as main
        importlib.reload(main)
        return TestClient(main.app)


def test_login_and_protected_route(auth):
    auth.create_user("alice", "pw", role="admin")
    c = _client(auth)
    assert c.post("/forward", json={"question": "x", "use_rewrite": False}).status_code == 401
    assert c.post("/auth/login", json={"username": "alice", "password": "bad"}).status_code == 401
    r = c.post("/auth/login", json={"username": "alice", "password": "pw"})
    assert r.status_code == 200
    token = r.json()["access_token"]
    assert c.get("/auth/me", headers={"Authorization": f"Bearer {token}"}).json()["role"] == "admin"


def test_admin_only(auth):
    auth.create_user("admin1", "pw", role="admin")
    auth.create_user("user1", "pw", role="user")
    c = _client(auth)
    admin_tok = c.post("/auth/login", json={"username": "admin1", "password": "pw"}).json()["access_token"]
    user_tok = c.post("/auth/login", json={"username": "user1", "password": "pw"}).json()["access_token"]
    assert c.post("/admin/users", json={"username": "z", "password": "p"},
                  headers={"Authorization": f"Bearer {user_tok}"}).status_code == 403
    assert c.post("/admin/users", json={"username": "z", "password": "p"},
                  headers={"Authorization": f"Bearer {admin_tok}"}).status_code == 200
