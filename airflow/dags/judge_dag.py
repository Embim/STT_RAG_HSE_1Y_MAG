"""Judge DAG: один прогон LLM-судьи на одной лекции (или CSV-бенчмарке).

Заменяет ручной вызов `scripts/run_judge.py` / `python -m evaluation.judge.runner`.
Используй когда нужно прогнать конкретный транскрипт и не хочется руками
держать командную строку.

Trigger params:
  - source       : "lecture" | "benchmark" (required)
  - lectures     : path to .json or dir (требуется для source=lecture)
  - csvs         : list of paths к asr CSV (требуется для source=benchmark)
  - asr_name     : человеко-читаемая метка ASR (для leaderboard'а)
  - prompt_version : дефолт "v15"
  - run_name     : опционально (default judge_<version>_<source>)
  - max_chunks_per_item : опционально (для smoke прогонов на N чанках)

Tasks:
  1. precheck     — sanity check params + наличие asr-judge healthy
  2. run_judge    — subprocess `python -m evaluation.judge.runner ...`
  3. summary      — пушит ссылку на report.html / MLflow run в xcom

Retries=0 потому что прогон судьи дорогой (50-200 sec/chunk × N chunks),
лучше fail-fast чем повторять впустую.
"""
from __future__ import annotations

import os
import shlex
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


def _get_conf(context) -> dict:
    return (context.get("dag_run") and context["dag_run"].conf) or {}


def _precheck(**context) -> None:
    conf = _get_conf(context)
    source = conf.get("source")
    if source not in ("lecture", "benchmark"):
        raise ValueError(
            f"--source must be 'lecture' or 'benchmark', got {source!r}. "
            f"Set via dag_run.conf, e.g. {{'source': 'lecture', 'lectures': '...'}}"
        )
    if source == "lecture":
        if not conf.get("lectures"):
            raise ValueError("source=lecture requires 'lectures' (path to .json or dir)")
        p = Path(conf["lectures"])
        if not p.exists():
            raise FileNotFoundError(f"lectures path not found: {p}")
    elif source == "benchmark":
        csvs = conf.get("csvs") or []
        if not csvs:
            raise ValueError("source=benchmark requires 'csvs' (list of asr CSV paths)")
        for c in csvs:
            if not Path(c).exists():
                raise FileNotFoundError(f"benchmark csv not found: {c}")

    # Sanity check: asr-judge должен быть healthy. Если down — fail fast.
    rc = subprocess.run(
        ["curl", "-fsS", "http://asr-judge:8000/v1/models"],
        capture_output=True, timeout=10,
    ).returncode
    if rc != 0:
        raise RuntimeError(
            "asr-judge container не отвечает на /v1/models. "
            "Подними: `docker compose --profile asr-judge up -d`"
        )
    print("Precheck OK: source/paths/asr-judge healthy")


def _run_judge(**context) -> str:
    conf = _get_conf(context)
    source = conf["source"]
    ts = context["ts_nodash"]
    prompt_version = conf.get("prompt_version", "v15")
    asr_name = conf.get("asr_name", "")
    default_run = f"judge_{prompt_version}_{source}_{ts}"
    run_name = conf.get("run_name") or default_run

    cmd = [
        "python", "-m", "evaluation.judge.runner",
        "--source", source,
        "--prompt-version", prompt_version,
        "--run-name", run_name,
        "--judge-url", "http://asr-judge:8000",
    ]
    if source == "lecture":
        cmd += ["--lectures", str(conf["lectures"])]
    else:
        for c in conf["csvs"]:
            cmd += ["--csv", str(c)]
    if asr_name:
        cmd += ["--asr-name", asr_name]
    if conf.get("max_chunks_per_item"):
        cmd += ["--max-chunks-per-item", str(conf["max_chunks_per_item"])]

    env = {**os.environ, "PYTHONPATH": str(PROJECT_SRC)}
    print(f"Running: {' '.join(shlex.quote(x) for x in cmd)}")
    rc = subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT)).returncode
    if rc != 0:
        raise RuntimeError(f"judge runner exited with code {rc}")

    context["ti"].xcom_push(key="run_name", value=run_name)
    context["ti"].xcom_push(
        key="csv_glob",
        value=str(PROJECT_ROOT / "data" / "eval" / "results" / f"judge_{run_name}_*.csv"),
    )
    return run_name


def _summary(**context) -> None:
    ti = context["ti"]
    run_name = ti.xcom_pull(task_ids="run_judge", key="run_name")
    csv_glob = ti.xcom_pull(task_ids="run_judge", key="csv_glob")
    print("=" * 60)
    print(f"Judge run finished: {run_name}")
    print(f"  CSV pattern: {csv_glob}")
    print(f"  MLflow UI:   http://localhost:5000/#/experiments")
    print(f"  Langfuse:    http://localhost:3000")
    print("=" * 60)


with DAG(
    dag_id="judge_dag",
    default_args=DEFAULT_ARGS,
    description="Один прогон LLM-судьи на лекции или CSV-бенчмарке",
    schedule=None,  # only manual trigger
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    tags=["stt-rag", "judge"],
    params={
        "source": "lecture",
        "lectures": "/opt/airflow/project/data/transcripts/cnn_qwen3_full.json",
        "asr_name": "qwen3-asr-1.7b",
        "prompt_version": "v15",
    },
) as dag:

    precheck = PythonOperator(task_id="precheck", python_callable=_precheck)
    run_judge = PythonOperator(task_id="run_judge", python_callable=_run_judge)
    summary = PythonOperator(task_id="summary", python_callable=_summary)

    precheck >> run_judge >> summary
