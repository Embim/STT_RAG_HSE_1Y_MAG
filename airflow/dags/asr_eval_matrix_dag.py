"""ASR Eval Matrix DAG: прогон N ASR backend × M manifests с переключением контейнеров.

Заменяет `scripts/run_asr_eval_matrix.py`. Зачем DAG vs CLI:
  - matrix-режим неудобно держать в shell (один backend на VRAM, нужно
    последовательно up→eval→down, чтобы 16GB-карта не OOM-нула);
  - в Airflow это chain тасков с clean retry;
  - результат — таблица из (backend, manifest) → WER/CER в MLflow + CSV.

Trigger params:
  - backends    : list of backend dict {profile, container, asr_name,
                  asr_model_id, endpoint, asr_url_port}
                  default — все 5 (whisper, qwen3, vibevoice, parakeet, phi4)
  - manifests   : list of paths to manifest.json
                  default — все в data/eval/benchmarks/local/*/manifest.json
  - max_samples : int (default 50)
  - skip_down   : bool (default false) — не делать down после последнего

Tasks:
  - plan         — формирует matrix задач
  - matrix_run   — по очереди для каждого backend: up → wait → eval each manifest → down
  - aggregate    — собирает агрегированный CSV/MD по всем (backend, manifest)

Из-за того что Airflow tasks по дефолту parallel, а ASR контейнеры
эксклюзивные по VRAM — используем последовательный chain через
PythonOperator c циклом внутри (не dynamic mapping).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
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

# Дефолтная таблица ASR backend'ов. Должна соответствовать docker-compose
# profiles + container names + endpoint contract.
DEFAULT_BACKENDS = [
    {"profile": "asr-whisper", "container": "asr-whisper",
     "asr_name": "whisper-large-v3-turbo", "asr_model_id": "whisper-1",
     "endpoint": "transcription"},
    {"profile": "asr-qwen3", "container": "asr-qwen3",
     "asr_name": "qwen3-asr-1.7b", "asr_model_id": "Qwen/Qwen3-ASR-1.7B",
     "endpoint": "chat"},
    {"profile": "asr-vibevoice", "container": "asr-vibevoice",
     "asr_name": "vibevoice-bnb4", "asr_model_id": "microsoft/VibeVoice-ASR-HF",
     "endpoint": "transcription"},
    {"profile": "asr-parakeet", "container": "asr-parakeet",
     "asr_name": "parakeet-tdt-0.6b-v3", "asr_model_id": "nvidia/parakeet-tdt-0.6b-v3",
     "endpoint": "transcription"},
]


def _get_conf(context) -> dict:
    return (context.get("dag_run") and context["dag_run"].conf) or {}


def _docker(*args: str, check: bool = True) -> int:
    """Run `docker ...` from host filesystem. Airflow container должен иметь
    mount /var/run/docker.sock + docker CLI установлен (см. airflow/Dockerfile)."""
    print(f"$ docker {' '.join(args)}")
    rc = subprocess.run(["docker", *args]).returncode
    if check and rc != 0:
        raise RuntimeError(f"docker {' '.join(args)} → exit {rc}")
    return rc


def _wait_container_healthy(container: str, timeout_sec: int = 600) -> None:
    """Polling docker inspect пока health=healthy или таймаут."""
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


def _plan(**context) -> dict:
    """Сформировать matrix задач: какие backends × какие manifests."""
    conf = _get_conf(context)
    backends = conf.get("backends") or DEFAULT_BACKENDS
    manifests = conf.get("manifests")
    if not manifests:
        base = PROJECT_ROOT / "data" / "eval" / "benchmarks" / "local"
        if not base.exists():
            raise FileNotFoundError(
                f"{base} not found and no `manifests` param given"
            )
        manifests = [str(p / "manifest.json")
                     for p in sorted(base.iterdir())
                     if (p / "manifest.json").exists()]
    if not manifests:
        raise ValueError("No manifests to evaluate")

    plan = {
        "backends": backends,
        "manifests": manifests,
        "max_samples": int(conf.get("max_samples", 50)),
        "skip_down": bool(conf.get("skip_down", False)),
    }
    print(f"Plan: {len(backends)} backends × {len(manifests)} manifests")
    for b in backends:
        print(f"  - {b['asr_name']}  ({b['container']})")
    for m in manifests:
        print(f"  - {Path(m).parent.name}")
    return plan


def _matrix_run(**context) -> list[dict]:
    """Последовательно по backend'ам: up → wait → eval each manifest → down."""
    ti = context["ti"]
    plan = ti.xcom_pull(task_ids="plan")
    backends = plan["backends"]
    manifests = plan["manifests"]
    max_samples = plan["max_samples"]
    skip_down = plan["skip_down"]
    ts = context["ts_nodash"]

    env = {**os.environ, "PYTHONPATH": str(PROJECT_SRC)}
    results = []

    for i, b in enumerate(backends, 1):
        print(f"\n=== [{i}/{len(backends)}] {b['asr_name']} ({b['container']}) ===")
        _docker("compose", "--profile", b["profile"], "up", "-d",
                cwd=None if hasattr(subprocess, "_n") else None)  # cwd handled below
        # docker-compose требует cwd=PROJECT_ROOT чтобы найти docker-compose.yml
        # Подменяем через override
        subprocess.run(["docker", "compose", "--profile", b["profile"], "up", "-d"],
                       cwd=str(PROJECT_ROOT), check=True)
        _wait_container_healthy(b["container"])

        # Прогон evaluation.asr.runner на каждом manifest для этого backend
        for m in manifests:
            slug = Path(m).parent.name
            run_name = f"asr_matrix_{b['asr_name']}_{slug}_{ts}"
            cmd = [
                "python", "-m", "evaluation.asr.runner",
                "--local-benchmark", m,
                "--max-samples", str(max_samples),
                "--run-name", run_name,
                "--langfuse-dataset", f"asr-matrix-{slug}",
            ]
            # ASR_NAME / ASR_MODEL_ID / WHISPER_URL читаются из env settings
            env["ASR_NAME"] = b["asr_name"]
            env["ASR_MODEL_ID"] = b["asr_model_id"]
            env["ASR_ENDPOINT"] = b["endpoint"]
            env["WHISPER_URL"] = f"http://{b['container']}:8000"

            print(f"  ▶ eval on {slug}")
            rc = subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT)).returncode
            results.append({
                "backend": b["asr_name"], "manifest": slug,
                "run_name": run_name, "rc": rc,
            })

        # Освобождаем GPU перед следующим backend
        if not (skip_down and i == len(backends)):
            subprocess.run(["docker", "compose", "--profile", b["profile"], "down"],
                           cwd=str(PROJECT_ROOT), check=False)

    return results


def _aggregate(**context) -> str:
    """Простая сводка результатов в markdown + MLflow."""
    from processing.progress_tracker import tracked_run

    ti = context["ti"]
    results = ti.xcom_pull(task_ids="matrix_run") or []
    n_ok = sum(1 for r in results if r["rc"] == 0)
    n_fail = sum(1 for r in results if r["rc"] != 0)

    md = ["# ASR Matrix Run Summary", ""]
    md.append(f"Triggered at {context['ts']}")
    md.append(f"Total cells: {len(results)} ({n_ok} ok, {n_fail} failed)")
    md.append("")
    md.append("| Backend | Manifest | Run name | RC |")
    md.append("|---|---|---|---|")
    for r in results:
        md.append(f"| {r['backend']} | {r['manifest']} | `{r['run_name']}` | {r['rc']} |")

    report = PROJECT_ROOT / "data" / "eval" / "results" / f"_matrix_{context['ts_nodash']}.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(md), encoding="utf-8")

    with tracked_run(title=f"asr-matrix-{context['ts_nodash']}",
                     kind="asr-matrix-summary",
                     params={"n_cells": len(results), "n_ok": n_ok, "n_fail": n_fail}) as run:
        run.log_artifact(str(report))
    return str(report)


with DAG(
    dag_id="asr_eval_matrix_dag",
    default_args=DEFAULT_ARGS,
    description="ASR eval matrix: N backends × M manifests с переключением VRAM",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    tags=["stt-rag", "asr", "matrix"],
    params={"max_samples": 50, "skip_down": False},
) as dag:

    plan = PythonOperator(task_id="plan", python_callable=_plan)
    matrix_run = PythonOperator(task_id="matrix_run", python_callable=_matrix_run)
    aggregate = PythonOperator(
        task_id="aggregate", python_callable=_aggregate,
        trigger_rule="all_done",
    )

    plan >> matrix_run >> aggregate
