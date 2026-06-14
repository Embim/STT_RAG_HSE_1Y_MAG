import os
# settings.py reads these at import; provide dummies so import never needs a real .env.
os.environ.setdefault("LLM_MODEL", "test/model")
os.environ.setdefault("LLM_API_KEY_1", "x")
os.environ.setdefault("LLM_API_KEY_2", "x")
os.environ.setdefault("LLM_API_KEY_3", "x")

import importlib
from unittest.mock import patch

import pytest


@pytest.fixture
def svc_module():
    """Reload llm_services under a patched Weaviate connect, and reset the lazy
    cache afterwards so the mock-backed manager can't leak into other tests."""
    with patch("system.rag.vectore_store.weaviate.connect_to_local") as connect:
        import system.llm.llm_services as svc
        importlib.reload(svc)
        try:
            yield svc, connect
        finally:
            svc._CHAT_VECTORE_STORE_MANAGER = None


def test_import_does_not_connect(svc_module):
    _svc, connect = svc_module
    assert connect.call_count == 0


def test_getter_connects_once_and_caches(svc_module):
    svc, connect = svc_module
    mgr1 = svc.get_chat_vectore_store_manager()
    assert connect.call_count == 1
    mgr2 = svc.get_chat_vectore_store_manager()
    assert connect.call_count == 1
    assert mgr1 is mgr2
