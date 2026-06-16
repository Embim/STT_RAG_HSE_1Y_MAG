"""Full Benchmark DAG: end-to-end сравнение 4 ASR на одной лекции через LLM-судью.

Заменяет `scripts/full_asr_benchmark.sh`. Что делает:
  1. Для каждого из 4 ASR backend (whisper, qwen3-asr, vibevoice, parakeet):
     up → wait healthy → transcribe полную лекцию → JSON-файл → down.
  2. up asr-judge → wait → sweep по 4 транскриптам под одним parent batch run
     (использует evaluation.judge.runner с тегом parent_batch_run_id).
  3. Собрать cross-ASR HTML leaderboard.

Цена: ~2-3 часа wallclock (vibevoice transcribe ~43 мин, judge sweep ~2 ч).

Trigger params:
  - audio        : путь к mp4/mp3 лекции (required)
  - lecture_id   : str — prefix для имён транскриптов и batch run
                   (e.g. "cnn_31_01_26")
  - prompt_version : str (default "v15")
  - backends     : list (default: 4 ASR — whisper/qwen3/vibevoice/parakeet)

Tasks:
  - transcribe_each — последовательно по backend: up → transcribe → down
  - judge_sweep     — триггерит judge_sweep_dag с готовыми транскриптами
  - leaderboard_link — печатает URL leaderboard'а в логи
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

import pendulum

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

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

DEFAULT_BACKENDS = [
    {"profile": "asr-whisper", "container": "asr-whisper",
     "asr_name": "whisper-large-v3-turbo", "chunk_minutes": 5},
    {"profile": "asr-qwen3", "container": "asr-qwen3",
     "asr_name": "qwen3-asr-1.7b", "chunk_minutes": 5},
    {"profile": "asr-vibevoice", "container": "asr-vibevoice",
     "asr_name": "vibevoice-bnb4", "chunk_minutes": 5},
    {"profile": "asr-parakeet", "container": "asr-parakeet",
     "asr_name": "parakeet-tdt-0.6b-v3", "chunk_minutes": 5},
]


def _get_conf(context) -> dict:
    return (context.get("dag_run") and context["dag_run"].conf) or {}


def _wait_container_healthy(container: str, timeout_sec: int = 600) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        out = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Health.Status}}", container],
            capture_output=True, text=True,
        )
        status = out.stdout.strip()
        if status == "healthy":
            print(f"  {container}: healthy")
            return
        print(f"  {container}: {status or 'unknown'} (waiting)")
        time.sleep(10)
    raise TimeoutError(f"{container} not healthy after {timeout_sec}s")


def _transcribe_each(**context) -> list[dict]:
    """Последовательно по 4 backend: up → transcribe lecture → down."""
    conf = _get_conf(context)
    audio = conf.get("audio")
    lecture_id = conf.get("lecture_id")
    if not audio or not Path(audio).exists():
        raise FileNotFoundError(f"audio file required and must exist: {audio}")
    if not lecture_id:
        raise ValueError("lecture_id required (e.g. 'cnn_31_01_26')")
    backends = conf.get("backends") or DEFAULT_BACKENDS

    transcripts_dir = PROJECT_ROOT / "data" / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, b in enumerate(backends, 1):
        slug = b["asr_name"].split("-")[0]  # whisper / qwen3 / vibevoice / parakeet
        output = transcripts_dir / f"{lecture_id}_{slug}.json"
        print(f"\n=== [{i}/{len(backends)}] {b['asr_name']} → {output.name} ===")

        subprocess.run(["docker", "compose", "--profile", b["profile"], "up", "-d"],
                       cwd=str(PROJECT_ROOT), check=True)
        _wait_container_healthy(b["container"])

        # Use existing CLI: scripts/transcribe_long_audio.py
        cmd = [
            "python", "scripts/transcribe_long_audio.py",
            "--input", str(audio),
            "--output", str(output),
            "--asr-name", b["asr_name"],
            "--asr-url", f"http://{b['container']}:8000",
            "--chunk-minutes", str(b.get("chunk_minutes", 5)),
        ]
        env = {**os.environ, "PYTHONPATH": str(PROJECT_SRC)}
        rc = subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT)).returncode
        results.append({"asr_name": b["asr_name"], "output": str(output), "rc": rc})

        # Освобождаем GPU перед следующим
        subprocess.run(["docker", "compose", "--profile", b["profile"], "down"],
                       cwd=str(PROJECT_ROOT), check=False)

    return results


def _build_sweep_conf(**context) -> dict:
    """Подготовить conf для триггера judge_sweep_dag."""
    ti = context["ti"]
    conf = _get_conf(context)
    transcripts = ti.xcom_pull(task_ids="transcribe_each") or []
    valid = [t for t in transcripts if t["rc"] == 0 and Path(t["output"]).exists()]
    if not valid:
        raise RuntimeError("No transcripts produced — sweep canceled")

    lecture_id = conf["lecture_id"]
    prompt_version = conf.get("prompt_version", "v15")
    sweep_conf = {
        "batch_name": f"{lecture_id}_{prompt_version}_4way",
        "prompt_version": prompt_version,
        "asr_configs": [
            {"name": t["asr_name"], "lectures": t["output"]} for t in valid
        ],
    }
    print(f"Triggering judge_sweep with {len(valid)} transcripts")
    return sweep_conf


def _summary(**context) -> None:
    ti = context["ti"]
    sweep_conf = ti.xcom_pull(task_ids="build_sweep_conf")
    batch_name = sweep_conf["batch_name"]
    leaderboard = (PROJECT_ROOT / "data" / "eval" / "outputs"
                   / f"{batch_name}_leaderboard" / "leaderboard.html")
    print("=" * 70)
    print(f"End-to-end benchmark done: {batch_name}")
    print(f"  Leaderboard HTML: {leaderboard}")
    print(f"  MLflow UI:        http://localhost:5000/#/experiments")
    print(f"  Langfuse UI:      http://localhost:3000")
    print("=" * 70)


with DAG(
    dag_id="full_benchmark_dag",
    default_args=DEFAULT_ARGS,
    description="End-to-end: 4 ASR transcribe → judge sweep → leaderboard",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    tags=["stt-rag", "asr", "judge", "full"],
    params={
        "lecture_id": "cnn_31_01_26",
        "audio": "/opt/airflow/project/data/audio/cnn_lecture.mp3",
        "prompt_version": "v15",
    },
) as dag:

    transcribe_each = PythonOperator(
        task_id="transcribe_each", python_callable=_transcribe_each,
    )

    build_sweep_conf = PythonOperator(
        task_id="build_sweep_conf", python_callable=_build_sweep_conf,
    )

    trigger_sweep = TriggerDagRunOperator(
        task_id="trigger_judge_sweep",
        trigger_dag_id="judge_sweep_dag",
        conf="{{ ti.xcom_pull(task_ids='build_sweep_conf') }}",
        wait_for_completion=True,
        poke_interval=60,
        reset_dag_run=True,
    )

    summary = PythonOperator(task_id="summary", python_callable=_summary)

    transcribe_each >> build_sweep_conf >> trigger_sweep >> summary
