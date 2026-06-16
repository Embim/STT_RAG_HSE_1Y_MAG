"""Run an ASR backend over a stream of samples and compute metrics.

Three independent sinks, все fail-soft:

  - **CSV** (always) — long-format строки в data/eval/results/<run>_<ts>.csv,
    one row per (sample, metric).
  - **Langfuse** (когда `langfuse_dataset` задан) — span-tree: один outer
    "evaluator"-span обнимает весь прогон; каждый sample = inner "span"
    с input={audio,ref}/output={hyp} и attached scores на каждую метрику;
    в конце corpus scores attach'аются к outer-span'у. Dataset item'ы
    linkаются к sample trace_id, что даёт сравнение моделей в Langfuse UI
    через "Runs" view.
  - **MLflow** (когда MLFLOW_TRACKING_URI задан) — один run на benchmark,
    в нём log_metric для corpus_{wer,cer,mer,wil} + log_artifact CSV.

Caller передаёт iterator `{id, audio_path, reference, duration}`.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, List, Optional

from evaluation.asr.backends.base import ASRBackend
from evaluation.asr.metrics import compute_corpus, compute_pair
from evaluation.asr.model_registry import log_asr_model_to_registry
from evaluation.reporting import csv_export, langfuse_export
from processing.progress_tracker import tracked_run
from settings import settings

logger = logging.getLogger(__name__)


async def run_benchmark(
    backend: ASRBackend,
    *,
    samples: Iterable[dict],
    benchmark_label: str,
    run_name: str,
    langfuse_dataset: Optional[str],
) -> Path:
    """Iterate samples, transcribe, compute WER/CER, persist all sinks."""
    if langfuse_dataset:
        langfuse_export.ensure_dataset(
            langfuse_dataset,
            description=f"ASR benchmark {benchmark_label}",
        )

    references: List[str] = []
    hypotheses: List[str] = []
    rows: List[dict] = []

    asr_url = getattr(backend, "base_url", "") or ""
    asr_model_id = getattr(backend, "model_id", "") or ""

    with tracked_run(
        title=run_name,
        kind="asr_eval",
        params={
            "backend_name": backend.name,
            "asr_model_id": asr_model_id,
            "asr_url": asr_url,
            "benchmark_label": benchmark_label,
            "langfuse_dataset": langfuse_dataset or "",
        },
        tags={"benchmark": benchmark_label},
    ) as mlrun, langfuse_export.start_run_span(
        name=run_name,
        metadata={
            "backend_name": backend.name,
            "asr_model_id": asr_model_id,
            "asr_url": asr_url,
            "benchmark_label": benchmark_label,
            "langfuse_dataset": langfuse_dataset or "",
        },
    ):
        for sample in samples:
            sample_id = sample["id"]
            duration = sample.get("duration")
            ref = sample["reference"]

            # Каждый sample — отдельный inner span с input/output и scores.
            # Если Langfuse выключен, start_item_span yield'ит None и всё внутри
            # работает без spans (CSV/MLflow продолжают писаться нормально).
            with langfuse_export.start_item_span(
                name=sample_id,
                audio_path=sample["audio_path"],
                reference=ref,
                extra_input={"duration": duration} if duration is not None else None,
                extra_metadata={
                    "benchmark_label": benchmark_label,
                    "asr_model_id": asr_model_id,
                    "backend_name": backend.name,
                },
            ):
                try:
                    result = await backend.transcribe(sample["audio_path"])
                except Exception as e:
                    logger.error("Transcription failed for %s: %s", sample_id, e)
                    continue

                hyp = str(result.get("text", "")).strip()
                # Записать hypothesis в output активного span'а — видно в UI
                # рядом с input/reference.
                langfuse_export.update_current_output({"hypothesis": hyp})

                per_sample = compute_pair(ref, hyp)
                references.append(ref)
                hypotheses.append(hyp)

                # Sample scores → к текущему item-span'у (auto-attach через
                # OTel context — никаких trace_id руками передавать не надо).
                for metric, value in per_sample.items():
                    rows.append({
                        "model": backend.name,
                        "item_id": sample_id,
                        "metric": metric,
                        "value": value,
                        "extra": json.dumps(
                            {
                                "reference": ref,
                                "hypothesis": hyp,
                                "duration": duration,
                                "asr_url": asr_url,
                                "asr_model_id": asr_model_id,
                            },
                            ensure_ascii=False,
                        ),
                    })
                    langfuse_export.score_current(
                        name=f"asr_{metric}",
                        value=value,
                        comment=f"{backend.name}/{sample_id}",
                    )

                # Dataset item linkается к этому trace (source_trace_id),
                # чтобы в Langfuse UI на странице item был "Source trace".
                # ID санитайзится внутри add_dataset_item (`:` → `_`).
                if langfuse_dataset:
                    langfuse_export.add_dataset_item(
                        langfuse_dataset,
                        item_id=f"{benchmark_label}::{sample_id}",
                        input_payload={"audio_path": sample["audio_path"]},
                        expected_output={"text": ref},
                        metadata={"duration": duration} if duration is not None else None,
                        source_trace_id=langfuse_export.current_trace_id(),
                    )

                logger.info(
                    "[%s] %s: WER=%.3f CER=%.3f",
                    backend.name, sample_id, per_sample["wer"], per_sample["cer"],
                )

        # Corpus rollup. После закрытия последнего sample-span'а текущий
        # активный span = outer evaluator-span; score_current() прикрепит
        # corpus scores именно к нему. В Langfuse UI они отображаются как
        # "Run-level scores".
        corpus = compute_corpus(references, hypotheses)
        logger.info(
            "[%s] CORPUS over %d samples: WER=%.4f CER=%.4f",
            backend.name, corpus.get("n", 0),
            corpus.get("wer", float("nan")), corpus.get("cer", float("nan")),
        )
        for metric in ("wer", "cer", "mer", "wil"):
            rows.append({
                "model": backend.name,
                "item_id": "__corpus__",
                "metric": metric,
                "value": corpus.get(metric, float("nan")),
                "extra": json.dumps({
                    "n": corpus.get("n", 0),
                    "asr_url": asr_url,
                    "asr_model_id": asr_model_id,
                }),
            })
            langfuse_export.score_current(
                name=f"asr_corpus_{metric}",
                value=corpus.get(metric, float("nan")),
                comment=f"{backend.name} on {benchmark_label} (n={corpus.get('n', 0)})",
            )

        # MLflow corpus rollup.
        mlrun.log_metric("n_samples", corpus.get("n", 0))
        for metric in ("wer", "cer", "mer", "wil"):
            mlrun.log_metric(f"corpus_{metric}", float(corpus.get(metric, float("nan"))))

        out_path = csv_export.write_rows(run_name, rows, kind="asr")
        mlrun.log_artifact(str(out_path))
        log_asr_model_to_registry(
            backend,
            registered_model_name=settings.MLFLOW_ASR_REGISTERED_MODEL_NAME,
            alias=settings.MLFLOW_ASR_MODEL_ALIAS,
            run_id=mlrun.run_id,
            tags={
                "backend_name": backend.name,
                "asr_model_id": asr_model_id,
                "asr_url": asr_url,
                "asr_endpoint": getattr(backend, "endpoint", ""),
                "asr_language": getattr(backend, "language", ""),
                "benchmark_label": benchmark_label,
                "run_name": run_name,
                "corpus_n": str(corpus.get("n", 0)),
                "corpus_wer": str(corpus.get("wer", "")),
                "corpus_cer": str(corpus.get("cer", "")),
                "corpus_mer": str(corpus.get("mer", "")),
                "corpus_wil": str(corpus.get("wil", "")),
            },
        )
        langfuse_export.flush()
        logger.info("ASR benchmark CSV: %s", out_path)
        return out_path
