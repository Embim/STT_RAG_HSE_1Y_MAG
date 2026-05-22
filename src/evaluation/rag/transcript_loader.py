"""Read transcripts from disk into a uniform shape for downstream consumers.

Accepts both .txt (plain text) and .json (with `text` or `segments`).
Files shorter than 200 chars are skipped — too short to seed RAGAS testset
generation usefully.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


_MIN_CHARS = 200


def _extract_text(path: Path) -> str:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            if payload.get("text"):
                return str(payload["text"])
            segments = payload.get("segments") or []
            if segments:
                return " ".join(str(s.get("text", "")).strip() for s in segments)
        return ""
    return path.read_text(encoding="utf-8")


def load_transcripts(src: Path) -> List[Dict[str, Any]]:
    """Return [{"source": str, "text": str}, ...] from src (file or directory)."""
    if src.is_file():
        files = [src]
    else:
        files = []
        for ext in (".json", ".txt"):
            files.extend(sorted(src.rglob(f"*{ext}")))

    docs: List[Dict[str, Any]] = []
    for path in files:
        try:
            text = _extract_text(path)
        except Exception as e:
            logger.warning("Skip %s: %s", path, e)
            continue
        if not text or len(text.strip()) < _MIN_CHARS:
            continue
        docs.append({"source": str(path), "text": text})
    logger.info("Loaded %d transcripts from %s", len(docs), src)
    return docs


def to_langchain_documents(docs: List[Dict[str, Any]]):
    """Wrap loaded transcripts as langchain Document objects."""
    from langchain_core.documents import Document
    return [
        Document(page_content=d["text"], metadata={"source": d["source"]})
        for d in docs
    ]
