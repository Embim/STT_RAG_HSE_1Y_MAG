"""Benchmark DAG: JSON+MP3 → локальный ASR-бенчмарк (manifest + аудио чанки).

Два режима, выбираются по содержимому `dag_run.conf`:

  1. Single mode — `conf = {"json_path": "...", "audio_path": "...", "title": "..."}`.
     Обрабатывается одна пара. Используется когда триггерится извне (REST API,
     ручной trigger из UI с params).

  2. Batch mode — `conf` пустой или без `json_path`. Сканирует
     `data/transcripts/*.json` + `data/audio/*.mp3`, находит пары без уже
     собранного бенчмарка в `data/eval/benchmarks/local/<slug>/`, обрабатывает
     все за один DAG-run. Используется кнопкой «Собрать датасеты» в Streamlit
     или ручным триггером без params.

Tasks:
    1. plan_jobs       — формирует список (json, audio, title) пар к обработке
    2. build_and_clean — для каждой пары: build_asr_benchmark + cleanup_benchmark
    3. report_stats    — пушит сводки в MLflow (по одному run на пару)

`benchmark_dag` НЕ триггерит eval_dag — eval запускается отдельно (по
расписанию или вручную).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import pendulum

from airflow import DAG
from airflow.operators.python import PythonOperator

PROJECT_ROOT = Path("/opt/airflow/project")
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

DEFAULT_ARGS = {
    "owner": "stt-rag",
    "depends_on_past": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=2),
}


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)[:60]


def _plan_jobs(**context) -> list[dict]:
    """Возвращает список пар к обработке.

    Если в conf есть json_path/audio_path — обрабатываем только эту пару.
    Иначе сканируем data/transcripts + data/audio и берём всё, для чего
    в data/eval/benchmarks/local/ ещё нет директории.

    Также открывает parent MLflow batch run; его run_id уходит в xcom для
    дочерних задач, чтобы они могли обновлять aggregate progress.
    """
    from processing.progress_tracker import open_batch_run

    conf = (context.get("dag_run") and context["dag_run"].conf) or {}

    # Single mode
    if conf.get("json_path") and conf.get("audio_path"):
        title = conf.get("title") or Path(conf["json_path"]).stem
        jobs = [{
            "json_path": conf["json_path"],
            "audio_path": conf["audio_path"],
            "title": title,
            "slug": _slug(title),
        }]
        print(f"Single-mode: 1 job for {title}")
    else:
        # Batch mode
        transcripts_dir = PROJECT_ROOT / "data" / "transcripts"
        audio_dir = PROJECT_ROOT / "data" / "audio"
        benchmarks_dir = PROJECT_ROOT / "data" / "eval" / "benchmarks" / "local"

        json_files = list(transcripts_dir.glob("*.json"))
        print(f"Batch-mode: scanning {len(json_files)} JSON files in {transcripts_dir}")

        jobs = []
        for j in sorted(json_files):
            title = j.stem
            slug = _slug(title)
            if (benchmarks_dir / slug).exists():
                print(f"  - skip {title}: benchmark already built")
                continue
            audio_path = None
            for ext in (".mp3", ".mp4", ".wav", ".m4a", ".flac", ".ogg"):
                candidate = audio_dir / f"{title}{ext}"
                if candidate.exists():
                    audio_path = candidate
                    break
            if audio_path is None:
                print(f"  - skip {title}: no matching audio file in {audio_dir}")
                continue
            jobs.append({
                "json_path": str(j),
                "audio_path": str(audio_path),
                "title": title,
                "slug": slug,
            })

        print(f"Batch-mode: {len(jobs)} jobs queued")

    context["ti"].xcom_push(key="jobs", value=jobs)

    # Open MLflow batch run (skipped silently if MLflow disabled).
    batch = open_batch_run(
        name=f"benchmark-batch-{context['ts_nodash']}",
        kind="batch_benchmark",
        n_jobs=len(jobs),
        params={"dag_run_id": context["dag_run"].run_id},
    )
    if batch is not None:
        context["ti"].xcom_push(key="batch_run_id", value=batch.run_id)
        print(f"MLflow batch run id: {batch.run_id}")
    else:
        context["ti"].xcom_push(key="batch_run_id", value=None)

    return jobs


def _build_one_job(job: dict, idx: int, total: int) -> dict:
    """Run build_asr_benchmark + cleanup_benchmark for one (json, audio) pair."""
    benchmarks_dir = PROJECT_ROOT / "data" / "eval" / "benchmarks" / "local"
    out_dir = benchmarks_dir / job["slug"]
    print(f"=== [{idx}/{total}] {job['title']} → {out_dir} ===")
    try:
        subprocess.run(
            [
                "python", str(PROJECT_ROOT / "scripts" / "build_asr_benchmark.py"),
                "from-json",
                "--json", job["json_path"],
                "--audio", job["audio_path"],
                "--output", str(out_dir),
                "--merge-target-duration", "15",
                "--max-duration", "25",
                "--min-chars", "100",
            ],
            check=True, cwd=PROJECT_ROOT,
        )
        subprocess.run(
            [
                "python", str(PROJECT_ROOT / "scripts" / "cleanup_benchmark.py"),
                str(out_dir),
            ],
            check=True, cwd=PROJECT_ROOT,
        )
        return {**job, "benchmark_dir": str(out_dir), "status": "ok"}
    except subprocess.CalledProcessError as e:
        print(f"  FAILED: {e}")
        return {**job, "benchmark_dir": str(out_dir), "status": "failed", "error": str(e)}


def _build_and_clean(**context) -> list[dict]:
    """Для каждой пары запускаем build_asr_benchmark + cleanup_benchmark.

    После каждого успешного/проваленного job-а обновляется aggregate progress
    на parent batch run в MLflow (если включён).
    """
    from processing.progress_tracker import reattach_batch

    jobs: list[dict] = context["ti"].xcom_pull(key="jobs", task_ids="plan_jobs") or []
    batch_run_id: str | None = context["ti"].xcom_pull(key="batch_run_id", task_ids="plan_jobs")
    if not jobs:
        print("No jobs to process")
        return []

    benchmarks_dir = PROJECT_ROOT / "data" / "eval" / "benchmarks" / "local"
    benchmarks_dir.mkdir(parents=True, exist_ok=True)

    batch = reattach_batch(batch_run_id, n_jobs=len(jobs)) if batch_run_id else None

    results: list[dict] = []
    n_ok = n_failed = 0
    for i, job in enumerate(jobs, 1):
        result = _build_one_job(job, i, len(jobs))
        results.append(result)
        if result["status"] == "ok":
            n_ok += 1
        else:
            n_failed += 1
        if batch:
            batch.update_progress(done=n_ok + n_failed, failed=n_failed, step=i)

    context["ti"].xcom_push(key="results", value=results)
    print(f"Done: {n_ok}/{len(results)} OK, {n_failed} failed")
    return results


def _report_stats(**context) -> None:
    """Per-benchmark MLflow runs + finalize parent batch run."""
    from processing.progress_tracker import (
        finalize_batch_run, reattach_batch, tracked_run,
    )

    results: list[dict] = context["ti"].xcom_pull(key="results", task_ids="build_and_clean") or []
    batch_run_id: str | None = context["ti"].xcom_pull(key="batch_run_id", task_ids="plan_jobs")

    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_failed = sum(1 for r in results if r["status"] != "ok")

    for r in results:
        if r.get("status") != "ok":
            continue
        manifest_path = Path(r["benchmark_dir"]) / "manifest.json"
        if not manifest_path.exists():
            continue
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        items = m.get("items", [])
        durs = [it["duration"] for it in items]
        chars = [len(it["reference"]) for it in items]

        # tracked_run = single run without GPU sidecar (benchmark is ffmpeg-bound,
        # not GPU-bound). Cross-reference: tag = source_file_name → найти все runs
        # связанные с этим видео в MLflow поиском по тегу.
        with tracked_run(
            title=f"benchmark:{r['title']}",
            kind="benchmark",
            params={
                "benchmark_dir": r["benchmark_dir"],
                "source_json": r["json_path"],
                "source_audio": r["audio_path"],
            },
            tags={"source_file_name": Path(r["json_path"]).stem},
            parent_run_id=batch_run_id,
        ) as run:
            run.log_metric("n_items", len(items))
            run.log_metric("total_minutes", sum(durs) / 60 if durs else 0)
            run.log_metric("avg_chars", (sum(chars) / len(chars)) if chars else 0)
            run.log_metric("min_duration_sec", min(durs) if durs else 0)
            run.log_metric("max_duration_sec", max(durs) if durs else 0)
            run.log_artifact(str(manifest_path))

    # Close out the parent batch run with final tags + status=FINISHED
    if batch_run_id:
        handle = reattach_batch(batch_run_id, n_jobs=len(results))
        finalize_batch_run(handle, n_succeeded=n_ok, n_failed=n_failed)


with DAG(
    dag_id="benchmark_dag",
    default_args=DEFAULT_ARGS,
    description="JSON+MP3 → локальный ASR-бенчмарк (single или batch mode)",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    tags=["stt-rag", "benchmark"],
) as dag:

    plan_jobs = PythonOperator(
        task_id="plan_jobs",
        python_callable=_plan_jobs,
    )

    build_and_clean = PythonOperator(
        task_id="build_and_clean",
        python_callable=_build_and_clean,
    )

    report_stats = PythonOperator(
        task_id="report_stats",
        python_callable=_report_stats,
    )

    plan_jobs >> build_and_clean >> report_stats
