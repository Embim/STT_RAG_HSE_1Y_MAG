import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable, List
from urllib.parse import urlparse

from fastapi import UploadFile

from downloader.transcriber import transcribe
from downloader.ingest import ingest_json_to_vector_store
from downloader.sources.youtube import download_audio, download_video, get_playlist_urls
from downloader.sources.local_audio import extract_audio_from_video
from processing.chunked_transcribe import ffprobe_metadata, transcribe_chunked
from processing.progress_tracker import video_run
from processing.quality_signals import compute as compute_quality_signals
from settings import settings

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
AUDIO_DIR = DATA_DIR / "audio"
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def _safe_filename(title: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", title)


async def prepare_audio(url: str, keep_video: bool) -> tuple[str, str, str]:
    """Download audio (and optionally keep video) from a YouTube URL.

    Returns:
        (audio_path, doc_id, title)
    """
    if keep_video:
        video_path = await asyncio.to_thread(download_video, url)
        local_result = await asyncio.to_thread(extract_audio_from_video, video_path)
        return local_result.audio_path, local_result.file_id, local_result.title
    else:
        dr = await asyncio.to_thread(download_audio, url)
        return dr.audio_path, dr.video_id, dr.title


async def process_youtube(
    url: str,
    keep_video: bool,
    export_txt: bool,
    export_json: bool = False,
    keep_audio: bool = False,
    progress_cb: Callable[[dict], None] | None = None,
) -> dict:
    """Download, transcribe and ingest a YouTube URL (video or playlist).

    Detects automatically: if path is /playlist — processes all videos,
    otherwise treats as a single video.

    Videos are processed with limited concurrency (settings.INGEST_CONCURRENCY)
    so that downloading the next video overlaps with transcribing the current one.
    """
    is_playlist = urlparse(url).path == "/playlist"
    video_urls = await asyncio.to_thread(get_playlist_urls, url) if is_playlist else [url]
    logger.info("%s detected: %d video(s)", "Playlist" if is_playlist else "Video", len(video_urls))

    total = len(video_urls)
    fractions = {u: 0.0 for u in video_urls}

    def _report(current: str | None = None) -> None:
        if progress_cb:
            overall = 100.0 * sum(fractions.values()) / max(1, total)
            done = sum(1 for f in fractions.values() if f >= 1.0)
            progress_cb({"total_items": total, "done_items": done,
                         "progress": round(overall, 1), "current_item": current})

    _report()

    sem = asyncio.Semaphore(settings.INGEST_CONCURRENCY)

    async def _process_one(video_url: str) -> dict:
        async with sem:
            audio_path, doc_id, title = await prepare_audio(video_url, keep_video)

            def _chunk(idx: int, total_chunks: int, extra: dict) -> None:
                fractions[video_url] = idx / max(1, total_chunks)
                _report(title)

            item = await transcribe_and_ingest(
                audio_path=audio_path,
                doc_id=doc_id,
                title=title,
                export_txt=export_txt,
                export_json=export_json,
                keep_audio=keep_audio,
                source_url=video_url,
                progress_cb=_chunk,
            )
            fractions[video_url] = 1.0
            _report(title)
            item["video_id"] = item.pop("doc_id")
            return item

    results = await asyncio.gather(
        *[_process_one(u) for u in video_urls],
        return_exceptions=True,
    )

    items, errors = [], []
    for video_url, result in zip(video_urls, results):
        if isinstance(result, Exception):
            logger.error("Failed to process %s: %s", video_url, result)
            errors.append({"url": video_url, "error": str(result)})
        else:
            items.append(result)

    return {"ingested_count": len(items), "error_count": len(errors), "items": items, "errors": errors}


def _save_uploads_sync(files, work_dir):
    """(used by the sync wrapper) — write UploadFiles to work_dir, return [(path, filename)]."""
    saved = []
    for f in files:
        name = Path(f.filename).name
        p = Path(work_dir) / name
        # NOTE: sync wrapper path; the async API endpoint streams uploads itself.
        saved.append((str(p), name))
    return saved


async def process_saved_uploads(
    saved: list[tuple[str, str]],
    export_txt: bool,
    export_json: bool = False,
    keep_audio: bool = False,
    progress_cb: Callable[[dict], None] | None = None,
) -> dict:
    """Transcribe+ingest already-saved upload files. `saved` = [(path, original_filename)]."""
    items, errors = [], []
    total = len(saved)
    fractions = {p: 0.0 for p, _ in saved}

    def _report(current: str | None = None) -> None:
        if progress_cb:
            overall = 100.0 * sum(fractions.values()) / max(1, total)
            done = sum(1 for f in fractions.values() if f >= 1.0)
            progress_cb({"total_items": total, "done_items": done,
                         "progress": round(overall, 1), "current_item": current})

    _report()
    work_dir = tempfile.mkdtemp()
    try:
        for path, filename in saved:
            try:
                local_result = await asyncio.to_thread(extract_audio_from_video, path, work_dir)

                def _chunk(idx: int, total_chunks: int, extra: dict, _p: str = path) -> None:
                    fractions[_p] = idx / max(1, total_chunks)
                    _report(filename)

                item = await transcribe_and_ingest(
                    audio_path=local_result.audio_path, doc_id=local_result.file_id,
                    title=local_result.title, export_txt=export_txt, export_json=export_json,
                    keep_audio=keep_audio, source_url=None, source_file_name=filename,
                    progress_cb=_chunk,
                )
                item["file_id"] = item.pop("doc_id")
                items.append(item)
            except Exception as e:
                logger.error("Failed to process file %s: %s", filename, e, exc_info=True)
                errors.append({"filename": filename, "error": str(e)})
            finally:
                fractions[path] = 1.0
                _report(filename)
                try:
                    Path(path).unlink(missing_ok=True)
                except OSError:
                    pass
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    return {"ingested_count": len(items), "error_count": len(errors), "items": items, "errors": errors}


async def process_uploaded_files(
    files: List[UploadFile],
    export_txt: bool,
    export_json: bool = False,
    keep_audio: bool = False,
) -> dict:
    """Save uploaded files, extract audio, transcribe and ingest each one."""
    tmp_dir = tempfile.mkdtemp()
    saved: list[tuple[str, str]] = []
    try:
        for upload_file in files:
            name = Path(upload_file.filename).name
            dest = str(Path(tmp_dir) / name)
            data = await upload_file.read()
            Path(dest).write_bytes(data)
            saved.append((dest, upload_file.filename))
        return await process_saved_uploads(
            saved, export_txt=export_txt, export_json=export_json, keep_audio=keep_audio,
        )
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


async def transcribe_and_ingest(
    audio_path: str,
    doc_id: str,
    title: str,
    export_txt: bool,
    source_url: str | None,
    source_file_name: str | None = None,
    export_json: bool = False,
    keep_audio: bool = False,
    parent_batch_run_id: str | None = None,
    progress_cb: Callable[[int, int, dict], None] | None = None,
) -> dict:
    """Transcribe audio, ingest into vector store, optionally persist artifacts.

    Optional artifacts (each writes to a fixed location and returns its path
    in the result dict):
        - export_txt   → data/transcripts/<safe_title>.txt
        - export_json  → data/transcripts/<safe_title>.json (text + segments)
        - keep_audio   → data/audio/<safe_title><orig_ext>
    """
    safe_title = _safe_filename(title)
    audio_src = Path(audio_path)
    saved_audio_path: Path | None = None
    audio_size_mb = audio_src.stat().st_size / 1024 / 1024 if audio_src.exists() else 0.0
    chunk_minutes = int(settings.TRANSCRIBE_CHUNK_MINUTES or 0)
    started_at = time.time()

    media_meta = ffprobe_metadata(audio_src)
    extra_params: dict = {}
    if media_meta:
        # Round duration so MLflow UI stays compact.
        extra_params["audio_duration_sec"] = round(media_meta.get("duration_sec", 0.0), 1)
        if media_meta.get("audio_codec"):
            extra_params["audio_codec"] = media_meta["audio_codec"]
        if media_meta.get("container_format"):
            extra_params["container_format"] = media_meta["container_format"]
        if media_meta.get("bitrate_kbps"):
            extra_params["bitrate_kbps"] = media_meta["bitrate_kbps"]
        if media_meta.get("audio_sample_rate"):
            extra_params["audio_sample_rate"] = media_meta["audio_sample_rate"]
        if media_meta.get("audio_channels"):
            extra_params["audio_channels"] = media_meta["audio_channels"]

    with video_run(
        title=title,
        source_file_name=source_file_name,
        asr_name=settings.ASR_NAME,
        audio_size_mb=audio_size_mb,
        chunk_minutes=chunk_minutes,
        extra_params=extra_params,
        parent_run_id=parent_batch_run_id,
    ) as run:
        if keep_audio and audio_src.exists():
            saved_audio_path = AUDIO_DIR / f"{safe_title}{audio_src.suffix}"
            try:
                shutil.copy2(audio_src, saved_audio_path)
                logger.info("Audio saved: %s", saved_audio_path)
            except OSError as e:
                logger.warning("Failed to keep audio %s: %s", saved_audio_path, e)
                saved_audio_path = None

        logger.info("Transcribing: %s (%s)", title, audio_path)
        try:
            def _on_chunk(idx: int, total: int, extra: dict) -> None:
                pct = 100.0 * idx / max(1, total)
                run.log_progress(pct, step=idx)
                # Per-chunk telemetry — track speed/throughput across the file.
                if extra:
                    if "chunk_time_sec" in extra:
                        run.log_metric("chunk_time_sec", extra["chunk_time_sec"], step=idx)
                    if "chunk_chars" in extra:
                        run.log_metric("chunk_chars", extra["chunk_chars"], step=idx)
                    if "chunk_chars_per_sec" in extra:
                        run.log_metric("chunk_chars_per_sec", extra["chunk_chars_per_sec"], step=idx)
                if progress_cb:
                    progress_cb(idx, total, extra)

            run.log_progress(0.0, step=0)
            if chunk_minutes > 0:
                transcript = await transcribe_chunked(
                    audio_path,
                    chunk_minutes=chunk_minutes,
                    progress_cb=_on_chunk,
                )
            else:
                transcript = await transcribe(audio_path)
                run.log_progress(100.0, step=1)
                if progress_cb:
                    progress_cb(1, 1, {})
            text = transcript["text"]
            segments = transcript.get("segments", [])
            logger.info("Transcription done: %s — %d chars, %d segments", title, len(text), len(segments))
            run.log_metric("total_chars", len(text))
            run.log_metric("total_segments", len(segments))
            run.log_metric("transcribe_wall_sec", time.time() - started_at)

            # Quality signals — flag suspicious transcripts (loops, low diversity, etc.)
            qs = compute_quality_signals(
                text=text,
                segments=segments,
                audio_duration_sec=media_meta.get("duration_sec", 0.0) if media_meta else 0.0,
                language_detected=transcript.get("language"),
            )
            for k, v in qs.items():
                if isinstance(v, bool):
                    run.set_tag(k, str(v).lower())
                elif isinstance(v, (int, float)):
                    run.log_metric(k, float(v))
                else:
                    run.set_tag(k, str(v))
            if qs.get("quality_warn"):
                run.set_tag("quality_warn", "true")
                logger.warning(
                    "Quality warning for %s: loops=%s unique_word_ratio=%.2f lowercase_pct=%.2f",
                    title, qs.get("n_loop_segments"), qs.get("unique_word_ratio"), qs.get("lowercase_segment_pct"),
                )
        finally:
            try:
                os.remove(audio_path)
            except OSError:
                pass

        logger.info("Ingesting into vector store: %s", title)
        ingest_started = time.time()
        await ingest_json_to_vector_store(
            [
                {
                    "hash": doc_id,
                    "text": text,
                    "segments": segments,
                    "title": title,
                    "source_url": source_url,
                    "source_file_name": source_file_name,
                }
            ]
        )
        run.log_metric("ingest_wall_sec", time.time() - ingest_started)
        logger.info("Vector store ingest complete: %s", title)

        txt_path: Path | None = None
        if export_txt:
            txt_path = TRANSCRIPTS_DIR / f"{safe_title}.txt"
            txt_path.write_text(text, encoding="utf-8")
            logger.info("Transcript saved: %s", txt_path)
            run.log_artifact(str(txt_path))

        json_path: Path | None = None
        if export_json:
            json_path = TRANSCRIPTS_DIR / f"{safe_title}.json"
            json_payload = {
                "doc_id": doc_id,
                "title": title,
                "source_url": source_url,
                "source_file_name": source_file_name,
                "asr_name": settings.ASR_NAME,
                "language": transcript.get("language"),
                "text": text,
                "segments": segments,
            }
            json_path.write_text(
                json.dumps(json_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info("Transcript JSON saved: %s", json_path)
            run.log_artifact(str(json_path))

        # Disk usage — sum of all artifacts we wrote out to data/
        bytes_written = 0
        for p in (txt_path, json_path, saved_audio_path):
            if p and p.exists():
                bytes_written += p.stat().st_size
        if bytes_written:
            run.log_metric("bytes_written", bytes_written)
            run.log_metric("mb_written", bytes_written / 1024 / 1024)

        run.log_metric("total_wall_sec", time.time() - started_at)
        # Parent batch run aggregates are updated by the batch coordinator
        # (e.g. benchmark_dag._build_and_clean), not by individual child jobs —
        # see `processing.progress_tracker.open_batch_run`.

    result = {"status": "ok", "doc_id": doc_id, "title": title}
    if source_file_name:
        result["source_file_name"] = source_file_name
    if export_txt:
        result["transcript"] = text
        result["txt_path"] = str(txt_path) if txt_path else None
    if export_json:
        result["json_path"] = str(json_path) if json_path else None
    if keep_audio and saved_audio_path:
        result["audio_path"] = str(saved_audio_path)
    return result
