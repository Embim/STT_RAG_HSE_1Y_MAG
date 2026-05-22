"""Synthesize a Q&A testset via ragas.testset.TestsetGenerator."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from evaluation.paths import ensure_dirs
from evaluation.rag.llm import get_embeddings, get_llm, require_eval_extra
from evaluation.rag.transcript_loader import load_transcripts, to_langchain_documents

logger = logging.getLogger(__name__)


def _coerce_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    try:
        return [str(v) for v in value]
    except TypeError:
        return [str(value)]


def generate_testset(
    transcripts_dir: Path,
    output_path: Path,
    size: int,
) -> Path:
    """Read transcripts, generate `size` Q&A items, write JSON, return path."""
    require_eval_extra()
    from ragas.testset import TestsetGenerator

    ensure_dirs()
    docs = load_transcripts(transcripts_dir)
    if not docs:
        raise SystemExit(f"No usable transcripts found under {transcripts_dir}")

    lc_docs = to_langchain_documents(docs)
    generator = TestsetGenerator(llm=get_llm(), embedding_model=get_embeddings())
    logger.info("Generating %d Q&A samples (this calls the LLM)...", size)
    testset = generator.generate_with_langchain_docs(lc_docs, testset_size=size)

    df = testset.to_pandas()
    items: List[Dict[str, Any]] = []
    for i, row in df.iterrows():
        items.append({
            "item_id": f"auto_{i:04d}",
            "question": str(row.get("user_input") or row.get("question") or ""),
            "reference_answer": str(row.get("reference") or row.get("ground_truth") or ""),
            "reference_contexts": _coerce_list(row.get("reference_contexts")),
            "metadata": {
                "auto_generated": True,
                "verified": False,
                "synthesizer": str(row.get("synthesizer_name", "")),
            },
        })

    payload = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "transcripts_dir": str(transcripts_dir),
        "size": len(items),
        "items": items,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Wrote %d items to %s", len(items), output_path)
    return output_path
