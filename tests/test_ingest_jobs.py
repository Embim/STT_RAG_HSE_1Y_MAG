import os
os.environ.setdefault("LLM_MODEL", "test/model")
os.environ.setdefault("LLM_API_KEY_1", "x")
os.environ.setdefault("LLM_API_KEY_2", "x")
os.environ.setdefault("LLM_API_KEY_3", "x")

import pytest


def test_ssrf_validator_allows_youtube():
    from api.schemas import IngestRequest
    assert IngestRequest(url="https://www.youtube.com/watch?v=abc", use_rewrite=False) is not None
    assert IngestRequest(url="https://youtu.be/abc") is not None


@pytest.mark.parametrize("bad", [
    "http://169.254.169.254/latest/meta-data",
    "file:///etc/passwd",
    "https://evil.example.com/x",
    "http://localhost:8080/",
])
def test_ssrf_validator_rejects(bad):
    from pydantic import ValidationError
    from api.schemas import IngestRequest
    with pytest.raises(ValidationError):
        IngestRequest(url=bad)


def test_job_store_lifecycle():
    import system.ingest_jobs as j
    jid = j.create_job()
    assert j.get_job(jid)["status"] == "queued"
    j.update_job(jid, status="running", progress=50.0)
    assert j.get_job(jid)["status"] == "running"
    assert j.get_job(jid)["progress"] == 50.0
    assert j.get_job("nope") is None


@pytest.mark.asyncio
async def test_youtube_job_runner_done(monkeypatch):
    from unittest.mock import patch
    import importlib
    with patch("system.rag.vectore_store.weaviate.connect_to_local"):
        import api.main as main
        importlib.reload(main)

    import system.ingest_jobs as j

    async def fake_process_youtube(**kwargs):
        cb = kwargs.get("progress_cb")
        if cb:
            cb({"total_items": 1, "done_items": 1, "progress": 100.0, "current_item": "X"})
        return {"ingested_count": 1, "error_count": 0,
                "items": [{"status": "ok", "title": "X"}], "errors": []}

    monkeypatch.setattr(main, "process_youtube", fake_process_youtube)
    from api.schemas import IngestRequest
    jid = j.create_job()
    await main._run_youtube_job(jid, IngestRequest(url="https://youtu.be/abc"))
    job = j.get_job(jid)
    assert job["status"] == "done"
    assert job["done_items"] == 1
    assert job["items"][0]["title"] == "X"


@pytest.mark.asyncio
async def test_youtube_job_runner_error(monkeypatch):
    from unittest.mock import patch
    import importlib
    with patch("system.rag.vectore_store.weaviate.connect_to_local"):
        import api.main as main
        importlib.reload(main)

    import system.ingest_jobs as j

    async def boom(**kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(main, "process_youtube", boom)
    from api.schemas import IngestRequest
    jid = j.create_job()
    await main._run_youtube_job(jid, IngestRequest(url="https://youtu.be/abc"))
    assert j.get_job(jid)["status"] == "error"
    assert "kaboom" in j.get_job(jid)["detail"]
