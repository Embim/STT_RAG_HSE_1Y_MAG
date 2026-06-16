"""RAGAS metric set + evaluate(samples) helper."""
from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Sequence

from system.tracing import get_client, observe
from evaluation.rag.eval_sample import EvalSample
from evaluation.rag.llm import get_embeddings, get_llm, require_eval_extra

logger = logging.getLogger(__name__)


def _item_span(item_id: str, question: str):
    """Open a Langfuse span scoped to one testset item.

    Falls back to a no-op context manager when Langfuse is disabled or the
    SDK doesn't expose `start_as_current_span` — keeps the path safe for
    smoke tests / offline runs.
    """
    client = get_client()
    if not hasattr(client, "start_as_current_span"):
        return nullcontext()
    try:
        return client.start_as_current_span(
            name=f"item-{item_id}",
            metadata={"item_id": item_id, "question": question[:200]},
        )
    except Exception:
        return nullcontext()


def default_metrics(*, reference_based: bool = True) -> List[Any]:
    """RAGAS metrics to compute by default.

    Reference-free are always included; reference-based are added when at
    least some samples carry `reference_answer`.
    """
    require_eval_extra()
    from ragas.metrics import (
        Faithfulness,
        ResponseRelevancy,
        LLMContextPrecisionWithoutReference,
    )

    metrics: List[Any] = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithoutReference(),
    ]
    if reference_based:
        from ragas.metrics import (
            LLMContextPrecisionWithReference,
            LLMContextRecall,
            AnswerCorrectness,
            AnswerSimilarity,
        )
        metrics.extend([
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
            AnswerCorrectness(),
            AnswerSimilarity(),
        ])
    return metrics


def _samples_to_eval_dataset(samples: Sequence[EvalSample]):
    require_eval_extra()
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample

    out = []
    for s in samples:
        kwargs: Dict[str, Any] = {
            "user_input": s.question,
            "response": s.answer,
            "retrieved_contexts": list(s.contexts),
        }
        if s.reference_answer:
            kwargs["reference"] = s.reference_answer
        if s.reference_contexts:
            kwargs["reference_contexts"] = list(s.reference_contexts)
        out.append(SingleTurnSample(**kwargs))
    return EvaluationDataset(samples=out)


@observe(name="ragas-evaluate")
def evaluate_samples(
    samples: Sequence[EvalSample],
    *,
    metrics: Optional[List[Any]] = None,
    reference_based: bool = True,
) -> List[Dict[str, Any]]:
    """Run RAGAS evaluation, return [{item_id, metric, value}, ...].

    Iterates per-sample so that every judge LLM call becomes nested under a
    Langfuse span named `item-<item_id>`. Trades batch concurrency for
    legibility in the UI — you can click any item span and see exactly which
    judge calls (faithfulness, answer_relevancy, ...) ran for that question.

    A trailing row with item_id="__corpus__" carries the mean over items.
    """
    require_eval_extra()
    from ragas import evaluate

    if not samples:
        return []

    has_refs = any(s.reference_answer for s in samples)
    metrics = metrics or default_metrics(reference_based=reference_based and has_refs)
    llm = get_llm()
    embeddings = get_embeddings()

    logger.info("Running RAGAS on %d samples with %d metrics", len(samples), len(metrics))
    rows: List[Dict[str, Any]] = []
    per_metric_values: Dict[str, List[float]] = {}

    for i, sample in enumerate(samples):
        with _item_span(sample.item_id, sample.question):
            single_ds = _samples_to_eval_dataset([sample])
            try:
                result = evaluate(
                    dataset=single_ds,
                    metrics=metrics,
                    llm=llm,
                    embeddings=embeddings,
                    show_progress=False,
                )
            except Exception as e:
                logger.error("RAGAS failed on item %s: %s", sample.item_id, e)
                continue
            df = result.to_pandas()
            metric_cols = [c for c in df.columns if c not in {
                "user_input", "response", "retrieved_contexts",
                "reference", "reference_contexts",
            }]
            for col in metric_cols:
                value = df.iloc[0][col]
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                rows.append({
                    "item_id": sample.item_id,
                    "metric": col,
                    "value": value,
                    "trace_id": sample.trace_id,
                })
                per_metric_values.setdefault(col, []).append(value)
        logger.info(
            "RAGAS item %d/%d done (%s)", i + 1, len(samples), sample.item_id
        )

    for col, values in per_metric_values.items():
        if not values:
            continue
        rows.append({
            "item_id": "__corpus__",
            "metric": col,
            "value": sum(values) / len(values),
        })
    return rows
