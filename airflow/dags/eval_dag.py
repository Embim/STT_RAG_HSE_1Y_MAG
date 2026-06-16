"""Eval DAG: ASR runner на манифестах + RAG runner на тестсете.

Может запускаться:
    - вручную с params {"benchmark_dir": ".../local/<slug>"} (одиночный bench)
    - вручную без params (прогоняет все манифесты в data/eval/benchmarks/local/*)
    - по расписанию `@daily` (все манифесты + актуальный testset)

Tasks:
    1. asr_eval — один прогон evaluation.asr.runner на каждый манифест
    2. rag_eval — один прогон evaluation.rag.runner на курированный testset
    3. comparison_report — собирает агрегаты последних N runs из CSV,
       прикладывает markdown к MLflow run
"""
from __future__ import annotations

import json
import os
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


def _resolve_benchmark_dirs(context) -> list[Path]:
    conf = (context.get("dag_run") and context["dag_run"].conf) or {}
    single = conf.get("benchmark_dir")
    if single:
        return [Path(single)]
    base = PROJECT_ROOT / "data" / "eval" / "benchmarks" / "local"
    return sorted([p for p in base.iterdir() if (p / "manifest.json").exists()])


def _asr_eval(**context) -> list[str]:
    dirs = _resolve_benchmark_dirs(context)
    if not dirs:
        print("No benchmarks to evaluate")
        return []

    csv_paths: list[str] = []
    env = {**os.environ, "PYTHONPATH": str(PROJECT_SRC)}
    for d in dirs:
        manifest = d / "manifest.json"
        run_name = f"asr_{d.name}_{context['ts_nodash']}"
        print(f"=== ASR eval on {d.name} ===")
        cmd = [
            "python", "-m", "evaluation.asr.runner",
            "--local-benchmark", str(manifest),
            "--max-samples", "50",
            "--run-name", run_name,
            "--langfuse-dataset", f"asr-{d.name}",
        ]
        subprocess.run(cmd, check=False, env=env, cwd=PROJECT_ROOT)
        csv_paths.append(str(d / "manifest.json"))
    context["ti"].xcom_push(key="benchmark_dirs", value=[str(d) for d in dirs])
    return [str(d) for d in dirs]


def _rag_eval(**context) -> str:
    testset_dir = PROJECT_ROOT / "data" / "eval" / "testsets"
    if not testset_dir.exists():
        print("No testsets dir, skipping RAG eval")
        return ""
    candidates = sorted(testset_dir.glob("*.json"))
    if not candidates:
        print("No testsets, skipping RAG eval")
        return ""
    testset = candidates[-1]  # newest

    run_name = f"rag_{testset.stem}_{context['ts_nodash']}"
    env = {**os.environ, "PYTHONPATH": str(PROJECT_SRC)}
    cmd = [
        "python", "-m", "evaluation.rag.runner",
        "--testset", str(testset),
        "--run-name", run_name,
        "--langfuse-dataset", f"rag-{testset.stem}",
    ]
    subprocess.run(cmd, check=False, env=env, cwd=PROJECT_ROOT)
    return str(testset)


def _comparison_report(**context) -> None:
    """Собрать markdown по последним результатам, прикрепить к MLflow."""
    from processing.progress_tracker import video_run

    results_dir = PROJECT_ROOT / "data" / "eval" / "results"
    csvs = sorted(results_dir.glob("*.csv")) if results_dir.exists() else []
    recent = csvs[-10:]

    lines = ["# Eval comparison report", ""]
    lines.append(f"Запущен в {context['ts']}")
    lines.append(f"Найдено CSV-файлов: {len(csvs)} (показано последние {len(recent)})")
    lines.append("")
    for c in recent:
        lines.append(f"- `{c.name}` ({c.stat().st_size // 1024} KB)")

    report_path = results_dir / f"_comparison_{context['ts_nodash']}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")

    with video_run(title=f"eval-summary-{context['ts_nodash']}") as run:
        run.set_tag("kind", "eval-summary")
        run.log_param("n_csv_files", len(csvs))
        run.log_artifact(str(report_path))


with DAG(
    dag_id="eval_dag",
    default_args=DEFAULT_ARGS,
    description="ASR runner на бенчмарках + RAG runner на testset",
    schedule="0 3 * * *",  # каждый день в 03:00 UTC
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    tags=["stt-rag", "eval"],
) as dag:

    asr_eval = PythonOperator(
        task_id="asr_eval",
        python_callable=_asr_eval,
    )

    rag_eval = PythonOperator(
        task_id="rag_eval",
        python_callable=_rag_eval,
    )

    comparison_report = PythonOperator(
        task_id="comparison_report",
        python_callable=_comparison_report,
    )

    [asr_eval, rag_eval] >> comparison_report
