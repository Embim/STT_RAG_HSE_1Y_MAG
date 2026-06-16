"""Ingest DAG: видео → audio → транскрипт+JSON+TXT → ингест в Weaviate.

Каждый прогон обрабатывает одно видео. Прогресс конкретного видео виден в
MLflow в реальном времени (см. `src/processing/progress_tracker.py`).

Сборка ASR-датасета НЕ запускается автоматически — это отдельный benchmark_dag,
который дёргается вручную (через UI Airflow или кнопку «Собрать датасеты» в
Streamlit). Так разделена «дешёвая часть» (транскрибация) и «дорогая по диску»
(ffmpeg-нарезка на чанки бенчмарка).

Триггеры:
    - manual: с params {"video_path": "/opt/airflow/project/data/video/<file>.mp4"}

Tasks:
    1. extract_audio: ffmpeg вытаскивает .mp3 рядом с видео
    2. transcribe_and_export: вызывает наш processor.transcribe_and_ingest
       с export_json=True, keep_audio=True, chunk_minutes из settings
"""
from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pendulum

from airflow import DAG
from airflow.operators.python import PythonOperator

# Mounted in docker-compose: ./src → /opt/airflow/project/src
PROJECT_SRC = Path("/opt/airflow/project/src")
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

DEFAULT_ARGS = {
    "owner": "stt-rag",
    "depends_on_past": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}


def _extract_audio(**context) -> str:
    """ffmpeg-extract audio from the video provided via dag_run.conf or sensor."""
    import subprocess

    conf = context.get("dag_run").conf or {}
    video_path = conf.get("video_path")
    if not video_path:
        # Fall back to whatever the sensor matched (single newest file)
        from glob import glob
        candidates = sorted(glob("/opt/airflow/project/data/video/incoming/*.mp4"))
        if not candidates:
            raise ValueError("No video_path in dag_run.conf and no files in incoming/")
        video_path = candidates[-1]

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    audio_path = video_path.with_suffix(".mp3")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000",
        "-codec:a", "libmp3lame", "-q:a", "2", str(audio_path),
    ]
    subprocess.run(cmd, check=True)
    print(f"Extracted audio: {audio_path}")

    context["ti"].xcom_push(key="audio_path", value=str(audio_path))
    context["ti"].xcom_push(key="video_path", value=str(video_path))
    context["ti"].xcom_push(key="title", value=video_path.stem)
    return str(audio_path)


def _transcribe_and_export(**context) -> dict:
    """Run our existing transcribe_and_ingest with MLflow instrumentation."""
    from downloader.processor import transcribe_and_ingest

    ti = context["ti"]
    audio_path = ti.xcom_pull(key="audio_path", task_ids="extract_audio")
    title = ti.xcom_pull(key="title", task_ids="extract_audio")
    if not audio_path or not title:
        raise ValueError("Missing audio_path/title from extract_audio xcom")

    # Stable doc_id from filename so re-ingestion overwrites the same Weaviate uuid.
    import hashlib
    doc_id = hashlib.sha1(title.encode("utf-8")).hexdigest()[:16]

    result = asyncio.run(transcribe_and_ingest(
        audio_path=audio_path,
        doc_id=doc_id,
        title=title,
        export_txt=True,
        export_json=True,
        keep_audio=True,
        source_url=None,
        source_file_name=Path(audio_path).name,
    ))

    # Surface the JSON path so the downstream DAG knows what to process.
    ti.xcom_push(key="json_path", value=result.get("json_path"))
    ti.xcom_push(key="audio_kept_path", value=result.get("audio_path"))
    return result


with DAG(
    dag_id="ingest_dag",
    default_args=DEFAULT_ARGS,
    description="Видео → аудио → транскрипт+JSON+TXT → ингест в Weaviate",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    tags=["stt-rag", "ingest"],
) as dag:

    extract_audio = PythonOperator(
        task_id="extract_audio",
        python_callable=_extract_audio,
    )

    transcribe_and_export = PythonOperator(
        task_id="transcribe_and_export",
        python_callable=_transcribe_and_export,
    )

    extract_audio >> transcribe_and_export
