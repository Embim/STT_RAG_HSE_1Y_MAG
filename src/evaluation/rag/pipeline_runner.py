"""Run the production RAG pipeline against a list of testset items.

Returns EvalSample list ready to feed into RAGAS evaluation. Side effect:
when Langfuse is enabled and `langfuse_dataset` is provided, links each
trace to the dataset item under `run_name`.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from system.rag.pipeline import run as rag_run
from system.rag.vectore_store import VectorStoreManager
from system.tracing import get_client
from evaluation.rag.eval_sample import EvalSample

logger = logging.getLogger(__name__)


def _coerce_contexts(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def _is_real_langfuse(client) -> bool:
    return client.__class__.__name__ != "_NoopLangfuseClient"


async def run_pipeline_for_items(
    items: List[Dict[str, Any]],
    *,
    vector_store_manager: VectorStoreManager,
    top_k: int,
    similarity_threshold: float,
    use_rewrite: bool,
    langfuse_dataset: Optional[str],
    run_name: str,
) -> List[EvalSample]:
    samples: List[EvalSample] = []
    langfuse = get_client()
    is_real_lf = _is_real_langfuse(langfuse)

    for item in items:
        question = item.get("question", "").strip()
        if not question:
            logger.warning("Skipping item %s: empty question", item.get("item_id"))
            continue
        try:
            result = await rag_run(
                question=question,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                use_rewrite=use_rewrite,
                vector_store_manager=vector_store_manager,
            )
        except Exception as e:
            logger.error("Pipeline failed for %s: %s", item.get("item_id"), e)
            continue
        if not result:
            continue

        retrieved = result.get("retrieved_documents") or []
        contexts = [d.get("text", "") for d in retrieved if d.get("text")]
        trace_id = result.get("trace_id")

        samples.append(EvalSample(
            item_id=str(item.get("item_id")),
            question=question,
            answer=str(result.get("answer", "")),
            contexts=contexts,
            reference_answer=item.get("reference_answer") or None,
            reference_contexts=_coerce_contexts(item.get("reference_contexts")) or None,
            trace_id=trace_id,
        ))

        if is_real_lf and langfuse_dataset and trace_id:
            try:
                langfuse.create_dataset_run_item(
                    dataset_name=langfuse_dataset,
                    dataset_item_id=item.get("item_id"),
                    run_name=run_name,
                    trace_id=trace_id,
                )
            except Exception as e:
                logger.debug("dataset run linking failed: %s", e)

    return samples
