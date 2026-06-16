"""ASR runner CLI — picks one of the run modes.

Workflow for comparing models:
    1. Set WHISPER_URL / ASR_NAME / ASR_MODEL_ID in .env, run once.
    2. Edit .env to point at another container, run again.
    3. Diff the two CSVs in data/eval/results/ (or compare in Langfuse).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from evaluation.asr.backends.http_asr import HttpASRBackend
from evaluation.asr.benchmark_runner import run_benchmark
from evaluation.asr.batch_transcribe import transcribe_dir
from evaluation.asr.benchmarks import iter_benchmark
from evaluation.asr.local_benchmark import iter_local_benchmark
from evaluation.paths import RESULTS_DIR, ensure_dirs
from evaluation.reporting.asr_report import generate_report

logger = logging.getLogger(__name__)


def _check_self_evaluation_leak(manifest_path: Path, backend_name: str) -> None:
    """Warn loudly if we're about to evaluate the same model that produced the references."""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.debug("Cannot peek manifest %s: %s", manifest_path, e)
        return
    ref_source = manifest.get("reference_source")
    if not ref_source:
        return
    if ref_source == backend_name:
        logger.warning(
            "==================================================================\n"
            "  SELF-EVALUATION LEAK: backend %r is also the reference source.\n"
            "  WER will be ~0 by construction — the model is being compared\n"
            "  to its own previous output. To get meaningful numbers, either:\n"
            "    - run a different ASR model (set ASR_NAME/WHISPER_URL),\n"
            "    - or curate references manually and use --only-verified.\n"
            "==================================================================",
            backend_name,
        )
    else:
        logger.info(
            "Reference produced by %r; evaluating %r against it (relative comparison).",
            ref_source, backend_name,
        )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="evaluation.asr.runner")
    p.add_argument("--run-name", default=None,
                   help="Identifier for CSV/Langfuse run (default: asr_{ASR_NAME})")
    p.add_argument("--benchmark", default=None,
                   help="HF dataset name (default: settings.ASR_BENCHMARK)")
    p.add_argument("--lang", default=None,
                   help="HF dataset config (default: settings.ASR_BENCHMARK_LANG)")
    p.add_argument("--split", default=None)
    p.add_argument("--max-samples", type=int, default=50)
    p.add_argument("--local-benchmark", default=None,
                   help="Path to a local manifest.json (built with `evaluation.asr.benchmark`); "
                        "overrides --benchmark/--lang/--split when set")
    p.add_argument("--only-verified", action="store_true",
                   help="Local benchmark only: skip items where verified=false")
    p.add_argument("--langfuse-dataset", default=None,
                   help="Optional Langfuse dataset name to attach items to")
    p.add_argument("--transcribe-dir", default=None,
                   help="If set, transcribe every audio in this dir; skip metrics")
    p.add_argument("--out-dir", default=None,
                   help="Where transcribe-dir mode writes JSONs")
    return p


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_parser().parse_args()
    ensure_dirs()
    backend = HttpASRBackend()
    logger.info(
        "ASR backend: name=%s url=%s model=%s lang=%s",
        backend.name, backend.base_url, backend.model_id, backend.language,
    )

    if args.transcribe_dir:
        asyncio.run(transcribe_dir(
            backend,
            src_dir=Path(args.transcribe_dir),
            out_dir=Path(args.out_dir) if args.out_dir else None,
        ))
        return

    run_name = args.run_name or f"asr_{backend.name}"
    if args.local_benchmark:
        manifest_path = Path(args.local_benchmark)
        _check_self_evaluation_leak(manifest_path, backend.name)
        samples = iter_local_benchmark(
            manifest_path,
            max_samples=args.max_samples,
            only_verified=args.only_verified,
        )
        label = f"local:{manifest_path.parent.name}"
    else:
        samples = iter_benchmark(
            name=args.benchmark, lang=args.lang, split=args.split,
            max_samples=args.max_samples,
        )
        label = args.benchmark or "default"

    csv_path: Path | None = asyncio.run(run_benchmark(
        backend,
        samples=samples,
        benchmark_label=label,
        run_name=run_name,
        langfuse_dataset=args.langfuse_dataset,
    ))

    # Авто-генерация отчёта по ИМЕННО ЭТОМУ прогону.
    # Подпапка называется так же, как CSV: data/eval/outputs/<csv_stem>/.
    # Это даёт 1-к-1 соответствие между CSV в data/eval/results/ и report
    # bundle в data/eval/outputs/, плюс история сохраняется автоматически
    # (каждый прогон → своя папка, не перезаписывается).
    #
    # Для сравнительного отчёта по нескольким прогонам/моделям используй
    # scripts/asr_eval_report.py с явным --out:
    #   python scripts/asr_eval_report.py --out data/eval/outputs/compare_v1 \
    #       data/eval/results/asr_*.csv
    #
    # Не блокируем eval если report-генерация падает — это побочный артефакт.
    if csv_path is not None:
        repo_root = Path(__file__).resolve().parents[3]
        out_dir = repo_root / "data" / "eval" / "outputs" / csv_path.stem
        try:
            html = generate_report([csv_path], out_dir)
            logger.info("Report bundle ready: %s", html)
        except Exception as e:
            logger.warning("Auto-report generation failed: %s "
                           "(eval CSV is still at %s)", e, csv_path)


if __name__ == "__main__":
    main()
