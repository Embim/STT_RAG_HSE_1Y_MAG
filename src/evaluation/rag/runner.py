"""CLI: run the production RAG pipeline on a curated testset and score it.

    python -m evaluation.rag.runner \
        --testset data/eval/testsets/ragas_v1.json \
        --run-name baseline_$(date +%s) \
        --langfuse-dataset stt-rag-qa-v1
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

from settings import settings
from system.rag.vectore_store import VectorStoreManager
from evaluation.paths import ensure_dirs
from evaluation.rag.metrics import evaluate_samples
from evaluation.rag.pipeline_runner import run_pipeline_for_items
from evaluation.rag.scoring import write_metric_rows
from evaluation.rag.testset_loader import load_testset

logger = logging.getLogger(__name__)


async def _run(args) -> Path:
    ensure_dirs()
    items = load_testset(Path(args.testset))
    if args.max_items is not None:
        items = items[: args.max_items]
    logger.info("Loaded %d items from %s", len(items), args.testset)

    vsm = (
        VectorStoreManager(collection_name=args.collection)
        if args.collection else VectorStoreManager()
    )
    run_name = args.run_name or f"rag_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    samples = await run_pipeline_for_items(
        items,
        vector_store_manager=vsm,
        top_k=args.top_k,
        similarity_threshold=args.similarity_threshold,
        use_rewrite=not args.no_rewrite,
        langfuse_dataset=args.langfuse_dataset,
        run_name=run_name,
    )

    metric_rows = await asyncio.to_thread(evaluate_samples, samples)
    out_path = write_metric_rows(
        metric_rows,
        run_name=run_name,
        collection=args.collection or settings.WEAVIATE_COLLECTION_NAME,
    )
    logger.info("RAG eval CSV: %s", out_path)
    return out_path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="evaluation.rag.runner")
    p.add_argument("--testset", required=True, help="Path to curated testset JSON")
    p.add_argument("--run-name", default=None)
    p.add_argument("--collection", default=None,
                   help="Override Weaviate collection (e.g. LectureChunks_eval_qwen3_asr)")
    p.add_argument("--top-k", type=int, default=settings.K)
    p.add_argument("--similarity-threshold", type=float,
                   default=settings.DEFAULT_SIMILARITY_THRESHOLD)
    p.add_argument("--no-rewrite", action="store_true")
    p.add_argument("--max-items", type=int, default=None)
    p.add_argument("--langfuse-dataset", default=None)
    return p


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_parser().parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
