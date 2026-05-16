"""Judge Sweep DAG: прогон LLM-судьи на N ASR backend под одним parent batch run.

Заменяет `scripts/run_judge_sweep.py`. Преимущество DAG-формы:
  - parent batch run открывается в отдельной task (видно сразу в MLflow UI),
    дочерним прогонам передаётся run_id через xcom;
  - каждый backend = отдельный task с динамическим mapping (`expand`) —
    Airflow UI показывает 4 квадратика, можно ретраить только failed;
  - leaderboard собирается отдельным task на финале;
  - параметры (asr_configs, prompt_version) передаются через `dag_run.conf`.

Trigger params:
  - asr_configs : list of {name, lectures} — пары ASR-метка + путь к транскрипту
                  (required, минимум 1 запись)
  - prompt_version : str (default "v15")
  - batch_name  : str (required) — имя parent batch run в MLflow

Tasks:
  1. open_batch       — открывает parent batch run в MLflow, kладёт run_id в xcom
  2. judge_backend    — dynamic task per asr_config, передаёт parent_run_id
  3. build_leaderboard — сборка cross-ASR HTML из всех CSV
  4. finalize_batch   — упаковывает parent run с n_succeeded/n_failed_final

Пример trigger conf:
    {
      "batch_name": "cnn_v15_4way",
      "prompt_version": "v15",
      "asr_configs": [
        {"name": "whisper-large-v3-turbo",
         "lectures": "/opt/airflow/project/data/transcripts/cnn_whisper.json"},
        {"name": "qwen3-asr-1.7b",
         "lectures": "/opt/airflow/project/data/transcripts/cnn_qwen3_full.json"},
        {"name": "vibevoice-bnb4",
         "lectures": "/opt/airflow/project/data/transcripts/cnn_vibevoice_full.json"},
        {"name": "parakeet-tdt-0.6b-v3",
         "lectures": "/opt/airflow/project/data/transcripts/cnn_parakeet_full.json"}
      ]
    }
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

import pendulum

from airflow import DAG
from airflow.decorators import task
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


def _open_batch(**context) -> str:
    """Открыть parent batch run в MLflow + вернуть его run_id через xcom."""
    from processing.progress_tracker import open_batch_run

    conf = _get_conf(context)
    batch_name = conf.get("batch_name")
    if not batch_name:
        raise ValueError("batch_name required in dag_run.conf")
    n_backends = len(conf.get("asr_configs", []))
    if n_backends < 1:
        raise ValueError("asr_configs required (list of {name, lectures})")

    # Override MLFLOW_EXPERIMENT_NAME для judge runs
    os.environ["MLFLOW_EXPERIMENT_NAME"] = os.environ.get(
        "MLFLOW_JUDGE_EXPERIMENT", "stt-rag-judge"
    )

    handle = open_batch_run(
        name=batch_name,
        kind="judge_sweep",
        n_jobs=n_backends,
        params={
            "prompt_version": conf.get("prompt_version", "v15"),
            "n_backends": n_backends,
            "backends": ",".join(c["name"] for c in conf["asr_configs"]),
            "triggered_by": "airflow",
        },
    )
    if handle is None:
        # MLflow disabled — продолжаем без parent run
        print("WARNING: MLflow disabled, children won't be linked to batch")
        return ""
    print(f"Opened MLflow batch run: {handle.run_id}")
    context["ti"].xcom_push(key="started_at", value=time.time())
    return handle.run_id


@task(task_id="judge_backend", retries=0)
def judge_backend(asr_config: dict, parent_run_id: str, **context) -> dict:
    """Один прогон judge на одном ASR backend. Динамически expand'ится по asr_configs."""
    conf = _get_conf(context)
    batch_name = conf["batch_name"]
    prompt_version = conf.get("prompt_version", "v15")
    asr_name = asr_config["name"]
    lectures = asr_config["lectures"]
    child_run_name = f"{batch_name}_{asr_name}"

    cmd = [
        "python", "-m", "evaluation.judge.runner",
        "--source", "lecture",
        "--lectures", str(lectures),
        "--prompt-version", prompt_version,
        "--asr-name", asr_name,
        "--run-name", child_run_name,
        "--judge-url", "http://asr-judge:8000",
    ]
    if parent_run_id:
        cmd += ["--mlflow-parent-run-id", parent_run_id]
        # В Airflow tasks live в разных Python processes (subprocess), поэтому
        # nested mode НЕ используется — линкуем через tag parent_batch_run_id.

    env = {**os.environ, "PYTHONPATH": str(PROJECT_SRC)}
    print(f"Running: {' '.join(shlex.quote(x) for x in cmd)}")
    rc = subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT)).returncode

    results_dir = PROJECT_ROOT / "data" / "eval" / "results"
    csvs = sorted(results_dir.glob(f"judge_{child_run_name}_*.csv"))
    jsonls = sorted(results_dir.glob(f"judge_{child_run_name}_*.jsonl"))
    return {
        "asr_name": asr_name,
        "rc": rc,
        "run_name": child_run_name,
        "csv": str(csvs[-1]) if csvs else None,
        "jsonl": str(jsonls[-1]) if jsonls else None,
    }


def _build_leaderboard(**context) -> str:
    """Сборка cross-ASR leaderboard через evaluation.reporting.judge_report."""
    from evaluation.reporting.judge_report import generate_report

    ti = context["ti"]
    parent_run_id = ti.xcom_pull(task_ids="open_batch")
    # collect from all dynamic-mapped instances
    results = ti.xcom_pull(task_ids="judge_backend") or []
    if not isinstance(results, list):
        results = [results]

    csv_paths = [Path(r["csv"]) for r in results if r.get("csv")]
    jsonl_paths = [Path(r["jsonl"]) for r in results if r.get("jsonl")]
    if not csv_paths:
        print("WARNING: no CSVs to build leaderboard from")
        return ""

    conf = _get_conf(context)
    batch_name = conf["batch_name"]
    out_dir = PROJECT_ROOT / "data" / "eval" / "outputs" / f"{batch_name}_leaderboard"
    out_dir.mkdir(parents=True, exist_ok=True)
    html = generate_report(csv_paths, out_dir, jsonl_paths=jsonl_paths)
    # rename report.html → leaderboard.html for clarity in artifacts list
    leaderboard = out_dir / "leaderboard.html"
    if html.exists() and html != leaderboard:
        leaderboard.write_bytes(html.read_bytes())

    # Attach to parent batch run as artifact
    if parent_run_id:
        try:
            import mlflow
            mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI",
                                                    "http://mlflow-server:5000"))
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            client.log_artifact(parent_run_id, str(leaderboard))
            print(f"Uploaded leaderboard.html to parent run {parent_run_id}")
        except Exception as e:
            print(f"WARNING: failed to attach leaderboard to parent: {e}")

    return str(leaderboard)


def _finalize_batch(**context) -> None:
    """Закрыть parent batch run с n_succeeded/n_failed_final."""
    from processing.progress_tracker import finalize_batch_run, reattach_batch

    ti = context["ti"]
    parent_run_id = ti.xcom_pull(task_ids="open_batch")
    results = ti.xcom_pull(task_ids="judge_backend") or []
    if not isinstance(results, list):
        results = [results]
    n_succeeded = sum(1 for r in results if r.get("rc") == 0)
    n_failed = sum(1 for r in results if r.get("rc") != 0)

    if not parent_run_id:
        print(f"Sweep done (no MLflow): succeeded={n_succeeded}, failed={n_failed}")
        return

    handle = reattach_batch(parent_run_id, len(results))
    if handle is None:
        print("WARNING: cannot reattach to batch run")
        return
    finalize_batch_run(handle, n_succeeded=n_succeeded, n_failed=n_failed)
    print(f"Batch finalized: succeeded={n_succeeded}, failed={n_failed}")


with DAG(
    dag_id="judge_sweep_dag",
    default_args=DEFAULT_ARGS,
    description="Sweep LLM-судьи на N ASR backend под одним parent batch run",
    schedule=None,  # only manual
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    tags=["stt-rag", "judge", "sweep"],
    params={
        "batch_name": "cnn_v15_4way",
        "prompt_version": "v15",
        "asr_configs": [
            {"name": "whisper-large-v3-turbo",
             "lectures": "/opt/airflow/project/data/transcripts/cnn_whisper.json"},
            {"name": "qwen3-asr-1.7b",
             "lectures": "/opt/airflow/project/data/transcripts/cnn_qwen3_full.json"},
            {"name": "vibevoice-bnb4",
             "lectures": "/opt/airflow/project/data/transcripts/cnn_vibevoice_full.json"},
            {"name": "parakeet-tdt-0.6b-v3",
             "lectures": "/opt/airflow/project/data/transcripts/cnn_parakeet_full.json"},
        ],
    },
) as dag:

    open_batch = PythonOperator(task_id="open_batch", python_callable=_open_batch)
    build_leaderboard = PythonOperator(
        task_id="build_leaderboard", python_callable=_build_leaderboard,
    )
    finalize_batch = PythonOperator(
        task_id="finalize_batch", python_callable=_finalize_batch,
        trigger_rule="all_done",  # finalize даже если children failed
    )

    # Dynamic task mapping: один judge_backend per asr_config из conf
    # Используем XComArg для динамического expand'а через partial+expand паттерн.
    # `dag_run.conf['asr_configs']` доступен только в runtime, поэтому
    # делаем mapper task который возвращает list.
    def _asr_configs(**context):
        return _get_conf(context).get("asr_configs", [])

    list_configs = PythonOperator(
        task_id="list_asr_configs", python_callable=_asr_configs,
    )

    backends = judge_backend.partial(
        parent_run_id="{{ ti.xcom_pull(task_ids='open_batch') }}",
    ).expand(asr_config=list_configs.output)

    open_batch >> list_configs >> backends >> build_leaderboard >> finalize_batch
