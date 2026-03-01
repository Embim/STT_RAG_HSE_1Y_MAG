import asyncio
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import List
from urllib.parse import urlparse

from fastapi import UploadFile

from downloader.transcriber import transcribe
from downloader.ingest import ingest_json_to_vector_store
from downloader.youtube import download_audio, download_video, get_playlist_urls
from downloader.local_audio import extract_audio_from_video

logger = logging.getLogger(__name__)

TRANSCRIPTS_DIR = Path(__file__).resolve().parent.parent / "data" / "transcripts"
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)


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


async def process_youtube(url: str, keep_video: bool, export_txt: bool) -> dict:
    """Download, transcribe and ingest a YouTube URL (video or playlist).

    Detects automatically: if path is /playlist — processes all videos,
    otherwise treats as a single video.
    """
    is_playlist = urlparse(url).path == "/playlist"
    video_urls = await asyncio.to_thread(get_playlist_urls, url) if is_playlist else [url]
    logger.info("%s detected: %d video(s)", "Playlist" if is_playlist else "Video", len(video_urls))

    items, errors = [], []
    for video_url in video_urls:
        try:
            audio_path, doc_id, title = await prepare_audio(video_url, keep_video)
            item = await transcribe_and_ingest(audio_path=audio_path, doc_id=doc_id, title=title, export_txt=export_txt)
            item["video_id"] = item.pop("doc_id")
            items.append(item)
        except Exception as e:
            logger.error("Failed to process %s: %s", video_url, e)
            errors.append({"url": video_url, "error": str(e)})

    return {"ingested_count": len(items), "error_count": len(errors), "items": items, "errors": errors}


async def process_uploaded_files(files: List[UploadFile], export_txt: bool) -> dict:
    """Save uploaded files, extract audio, transcribe and ingest each one."""
    work_dir = tempfile.mkdtemp()
    items, errors = [], []

    try:
        for upload_file in files:
            raw_path = Path(work_dir) / upload_file.filename
            try:
                raw_path.write_bytes(await upload_file.read())
                local_result = await asyncio.to_thread(extract_audio_from_video, str(raw_path), work_dir)
                try:
                    raw_path.unlink(missing_ok=True)
                except OSError:
                    pass

                item = await transcribe_and_ingest(
                    audio_path=local_result.audio_path,
                    doc_id=local_result.file_id,
                    title=local_result.title,
                    export_txt=export_txt,
                )
                item["file_id"] = item.pop("doc_id")
                items.append(item)
            except Exception as e:
                logger.error("Failed to process file %s: %s", upload_file.filename, e, exc_info=True)
                errors.append({"filename": upload_file.filename, "error": str(e)})
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    return {"ingested_count": len(items), "error_count": len(errors), "items": items, "errors": errors}


async def transcribe_and_ingest(audio_path: str, doc_id: str, title: str, export_txt: bool) -> dict:
    """Transcribe audio, ingest into vector store, optionally save transcript to disk."""
    logger.info("Transcribing: %s (%s)", title, audio_path)
    try:
        text = await transcribe(audio_path)
        logger.info("Transcription done: %s — %d chars", title, len(text))
    finally:
        try:
            os.remove(audio_path)
        except OSError:
            pass

    logger.info("Ingesting into vector store: %s", title)
    await ingest_json_to_vector_store([{"hash": doc_id, "text": text}])
    logger.info("Vector store ingest complete: %s", title)

    if export_txt:
        safe_title = re.sub(r'[\\/*?:"<>|]', "_", title)
        txt_path = TRANSCRIPTS_DIR / f"{safe_title}.txt"
        txt_path.write_text(text, encoding="utf-8")
        logger.info("Transcript saved: %s", txt_path)

    result = {"status": "ok", "doc_id": doc_id, "title": title}
    if export_txt:
        result["transcript"] = text
    return result
