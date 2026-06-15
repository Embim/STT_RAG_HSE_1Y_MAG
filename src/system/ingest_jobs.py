import time
import uuid
from typing import Any, Dict, Optional

_JOBS: Dict[str, Dict[str, Any]] = {}
_MAX_JOBS = 50


def create_job() -> str:
    job_id = uuid.uuid4().hex[:12]
    _JOBS[job_id] = {
        "job_id": job_id, "status": "queued", "progress": 0.0,
        "total_items": 0, "done_items": 0, "current_item": None,
        "items": [], "errors": [], "error_count": 0, "detail": None,
        "created_at": time.time(),
    }
    _prune()
    return job_id


def update_job(job_id: str, **fields: Any) -> None:
    job = _JOBS.get(job_id)
    if job is not None:
        job.update(fields)


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    return _JOBS.get(job_id)


def _prune() -> None:
    if len(_JOBS) <= _MAX_JOBS:
        return
    for k in sorted(_JOBS, key=lambda j: _JOBS[j]["created_at"])[: len(_JOBS) - _MAX_JOBS]:
        _JOBS.pop(k, None)
