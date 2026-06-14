import os
# settings.py reads these at import; provide dummies so import never needs a real .env.
os.environ.setdefault("LLM_MODEL", "test/model")
os.environ.setdefault("LLM_API_KEY_1", "x")
os.environ.setdefault("LLM_API_KEY_2", "x")
os.environ.setdefault("LLM_API_KEY_3", "x")

import importlib
from unittest.mock import patch


def test_import_does_not_connect_and_getter_is_cached():
    with patch("system.rag.vectore_store.weaviate.connect_to_local") as connect:
        import system.llm.llm_services as svc
        importlib.reload(svc)
        assert connect.call_count == 0          # import must NOT connect
        mgr1 = svc.get_chat_vectore_store_manager()
        assert connect.call_count == 1          # connects on first use
        mgr2 = svc.get_chat_vectore_store_manager()
        assert connect.call_count == 1          # cached
        assert mgr1 is mgr2
