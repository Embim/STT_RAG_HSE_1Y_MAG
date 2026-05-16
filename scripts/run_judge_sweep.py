"""Sweep judge.runner across N ASR backends под одним MLflow batch run.

Workflow:
  1. Открывает parent batch run в experiment `stt-rag-judge`
     (kind=judge_sweep, tag batch_name=<...>).
  2. Для каждой пары `--asr NAME LECTURES_DIR` дёргает
     `evaluation.judge.runner` с `--mlflow-parent-run-id=<batch_run_id>`,
     дочерний run автоматически линкуется к parent через tag
     parent_batch_run_id (см. progress_tracker.tracked_run).
  3. По мере прогона parent run обновляется метриками `n_done`,
     `progress_pct`, `eta_minutes` через MlflowClient — видно прогресс
     в UI без F5.
  4. По завершении строит сравнительный HTML leaderboard через
     `evaluation.reporting.judge_report.generate_report(all_csvs, ...)`
     и заливает его в parent run как `leaderboard.html`. Plus финализирует
     parent (tags n_succeeded / n_failed_final / wall_minutes).

Пример (4-way сравнение на CNN лекции):
    python scripts/run_judge_sweep.py \\
        --prompt-version v11 \\
        --batch-name cnn_v11_4way \\
        --asr whisper-large-v3-turbo  data/transcripts/cnn_whisper \\
        --asr qwen3-asr-1.7b          data/transcripts/cnn_qwen3 \\
        --asr vibevoice-bnb4          data/transcripts/cnn_vibevoice \\
        --asr parakeet-tdt-0.6b-v3    data/transcripts/cnn_parakeet

После прогона в MLflow UI открыть batch run и:
  - в Artifacts посмотреть leaderboard.html (Q-Score сравнение всех 4);
  - в Tags увидеть batch_name + n_succeeded;
  - кликнуть в "Child runs" чтобы зайти в индивидуальные прогоны.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from evaluation.judge.runner import _build_parser as judge_parser, _run as judge_run  # noqa: E402
from evaluation.reporting.judge_report import generate_report  # noqa: E402
from processing.progress_tracker import (  # noqa: E402
    finalize_batch_run,
    open_batch_run,
)
from settings import settings  # noqa: E402

logger = logging.getLogger(__name__)


def _latest(results_dir: Path, run_name: str, suffix: str) -> Path | None:
    """Найти самый свежий артефакт judge runner-а (CSV/JSONL) для этого run_name.

    judge_<run-name>_<YYYYMMDD_HHMMSS>.<suffix>
    """
    candidates = sorted(results_dir.glob(f"judge_{run_name}_*.{suffix}"))
    return candidates[-1] if candidates else None


async def _run_one(asr_name: str, lectures_dir: Path, *,
                   prompt_version: str, run_name: str,
                   parent_run_id: str | None,
                   judge_url: str | None,
                   skip_healthcheck: bool,
                   nested: bool) -> int:
    """Прогнать один backend, вернуть rc."""
    argv: List[str] = [
        "--source", "lecture",
        "--lectures", str(lectures_dir),
        "--prompt-version", prompt_version,
        "--asr-name", asr_name,
        "--run-name", run_name,
    ]
    if parent_run_id:
        argv += ["--mlflow-parent-run-id", parent_run_id]
    if judge_url:
        argv += ["--judge-url", judge_url]
    if skip_healthcheck:
        argv += ["--skip-healthcheck"]
    if nested:
        argv += ["--mlflow-nested"]

    args = judge_parser().parse_args(argv)
    return await judge_run(args)


def _log_leaderboard_to_parent(batch_handle, leaderboard_html: Path) -> None:
    """Залить leaderboard.html в parent batch run.

    batch_handle._client — это MlflowClient, у которого log_artifact работает
    с явным run_id (в отличие от глобального mlflow.log_artifact, которому
    нужен активный run-контекст, а у нас parent run "висит" без контекста).
    """
    try:
        client = batch_handle._client
        client.log_artifact(batch_handle.run_id, str(leaderboard_html))
        logger.info("Leaderboard uploaded to parent batch run: %s",
                    leaderboard_html.name)
    except Exception as e:
        logger.warning("Failed to log leaderboard to parent: %s", e)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--asr", action="append", nargs=2, required=True,
        metavar=("NAME", "LECTURES"),
        help="Пара (ASR-имя, путь к транскриптам). Повторять для каждого backend-а.",
    )
    p.add_argument("--prompt-version", default="v11")
    p.add_argument("--batch-name", required=True,
                   help="Имя parent batch run-а в MLflow.")
    p.add_argument(
        "--out-dir", default="data/eval/outputs",
        help="Куда положить leaderboard HTML (относительно репо корня).",
    )
    p.add_argument("--judge-url", default=None,
                   help="Override settings.JUDGE_URL для всех children.")
    p.add_argument("--skip-healthcheck", action="store_true",
                   help="Прокидывается в дочерние judge runner-ы.")
    p.add_argument("--mlflow-nested", action="store_true",
                   help="Открыть parent batch как active run и каждый child "
                        "judge-прогон как nested под ним. В MLflow UI parent "
                        "будет с разворачивающимся деревом из 4 children. "
                        "Без этого флага (default) — parent детачится, дети "
                        "линкуются через tag parent_batch_run_id (плоский вид).")
    args = p.parse_args()

    asr_configs: List[Tuple[str, Path]] = [
        (name, Path(path)) for name, path in args.asr
    ]
    n = len(asr_configs)
    logger.info("Sweep: %d backend(s), prompt=%s, batch=%s",
                n, args.prompt_version, args.batch_name)

    # Сразу выставляем experiment, чтобы parent batch попал в нужный.
    prev_experiment = os.environ.get("MLFLOW_EXPERIMENT_NAME")
    os.environ["MLFLOW_EXPERIMENT_NAME"] = settings.MLFLOW_JUDGE_EXPERIMENT

    batch = open_batch_run(
        name=args.batch_name,
        kind="judge_sweep",
        n_jobs=n,
        params={
            "prompt_version": args.prompt_version,
            "n_backends": n,
            "backends": ",".join(name for name, _ in asr_configs),
        },
        nested=args.mlflow_nested,
    )
    parent_run_id = batch.run_id if batch is not None else None
    if parent_run_id:
        logger.info("MLflow parent batch run: %s "
                    "(experiment=%s)", parent_run_id,
                    settings.MLFLOW_JUDGE_EXPERIMENT)
    else:
        logger.warning("MLflow disabled — sweep продолжится, но без UI трекинга")

    started = time.time()
    csv_paths: List[Path] = []
    jsonl_paths: List[Path] = []
    n_succeeded = 0
    n_failed = 0
    results_dir = REPO_ROOT / "data" / "eval" / "results"

    for i, (asr_name, lectures_dir) in enumerate(asr_configs, 1):
        child_run_name = f"{args.batch_name}_{asr_name}"
        sep = "═" * 80
        print()
        print(sep)
        print(f"  [{i}/{n}] judge backend: {asr_name}")
        print(f"          lectures:     {lectures_dir}")
        print(f"          run_name:     {child_run_name}")
        print(sep, flush=True)
        logger.info("[%d/%d] running judge: asr=%s, lectures=%s",
                    i, n, asr_name, lectures_dir)
        try:
            rc = asyncio.run(_run_one(
                asr_name, lectures_dir,
                prompt_version=args.prompt_version,
                run_name=child_run_name,
                parent_run_id=parent_run_id,
                judge_url=args.judge_url,
                skip_healthcheck=args.skip_healthcheck,
                nested=args.mlflow_nested,
            ))
        except Exception as e:
            logger.exception("Child run failed for %s: %s", asr_name, e)
            rc = 1

        if rc == 0:
            n_succeeded += 1
            csv = _latest(results_dir, child_run_name, "csv")
            jsonl = _latest(results_dir, child_run_name, "jsonl")
            if csv is not None:
                csv_paths.append(csv)
            if jsonl is not None:
                jsonl_paths.append(jsonl)
        else:
            n_failed += 1

        if batch is not None:
            batch.update_progress(done=i, failed=n_failed, step=i)

    # Сравнительный leaderboard HTML (тот же generate_report что у
    # scripts/judge_compare_asr.py — единая точка правды).
    if csv_paths:
        out_dir = REPO_ROOT / args.out_dir / f"{args.batch_name}_leaderboard"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            html_path = generate_report(
                csv_paths, out_dir, jsonl_paths=jsonl_paths,
            )
            # Переименуем report.html → leaderboard.html для ясности артефакта.
            leaderboard = out_dir / "leaderboard.html"
            if html_path.exists() and html_path != leaderboard:
                leaderboard.write_bytes(html_path.read_bytes())
            logger.info("Leaderboard ready: %s", leaderboard)
            if batch is not None:
                _log_leaderboard_to_parent(batch, leaderboard)
        except Exception as e:
            logger.warning("Leaderboard generation failed: %s", e)

    if batch is not None:
        finalize_batch_run(batch, n_succeeded=n_succeeded, n_failed=n_failed)

    elapsed_min = (time.time() - started) / 60
    print()
    print("=" * 60)
    print(f"Sweep: {args.batch_name}")
    print(f"  succeeded: {n_succeeded}/{n}")
    print(f"  failed:    {n_failed}")
    print(f"  wall:      {elapsed_min:.1f} min")
    if parent_run_id:
        print(f"  parent run id: {parent_run_id}")
    print("=" * 60)

    # Восстановим experiment env, как делает judge.runner у себя.
    if prev_experiment is None:
        os.environ.pop("MLFLOW_EXPERIMENT_NAME", None)
    else:
        os.environ["MLFLOW_EXPERIMENT_NAME"] = prev_experiment

    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
