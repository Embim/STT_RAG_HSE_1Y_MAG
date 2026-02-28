import asyncio
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from system.rag.pipeline import run
from system.llm.llm_services import CHAT_VECTORE_STORE_MANAGER
from downloader.youtube import download_audio, download_video, get_playlist_urls
from downloader.local_audio import extract_audio_from_video
from downloader.transcriber import transcribe
from downloader.ingest import ingest_json_to_vector_store

TRANSCRIPTS_DIR = Path(__file__).resolve().parent.parent / "data" / "transcripts"
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

(Path(__file__).resolve().parent.parent / "logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).resolve().parent.parent / "logs" / "api.log", encoding="utf-8"),
    ],
)

app = FastAPI(title="DS Navigator API")
logger = logging.getLogger(__name__)


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Welcome to DS Navigator API",
        "documentation": "/docs"
    }


@app.get("/health", status_code=status.HTTP_200_OK, tags=["Health"])
async def healthcheck():
    return {
        "status": "ok"
    }

@app.get("/check-vdb", tags=["Health"])
async def check_vdb():
    try:
        collection = CHAT_VECTORE_STORE_MANAGER.vector_store._collection
        count = collection.count()

        return {
            "status": "ok",
            "documents_in_vdb": count
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="vector db is not available"
        )


class ForwardRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Вопрос пользователя")
    top_k: Optional[int] = Field(None, ge=1, le=10, description="Количество возвращаемых результатов")
    similarity_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Порог схожести")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={"detail": "bad request"},
    )


class IngestRequest(BaseModel):
    url: str = Field(..., description="YouTube video URL")
    export_txt: bool = Field(False, description="Include transcript text in response")
    keep_video: bool = Field(False, description="Download and save full video to data/video/")


async def _transcribe_and_ingest(audio_path: str, doc_id: str, title: str, export_txt: bool, delete_audio: bool = True) -> dict:
    """Transcribe audio, ingest into vector store, and optionally return transcript."""
    logger.info("Transcribing: %s (%s)", title, audio_path)
    try:
        text = await transcribe(audio_path)
        logger.info("Transcription done: %s — %d chars", title, len(text))
    finally:
        if delete_audio:
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


@app.post("/ingest", tags=["Ingest"])
async def ingest(req: IngestRequest):
    logger.info("Ingest request: %s (keep_video=%s)", req.url, req.keep_video)

    if req.keep_video:
        try:
            video_path = await asyncio.to_thread(download_video, req.url)
        except Exception as e:
            logger.error("Video download failed for %s: %s", req.url, e)
            raise HTTPException(status_code=422, detail="не удалось скачать видео")

        logger.info("Extracting audio from video: %s", video_path)
        try:
            local_result = await asyncio.to_thread(extract_audio_from_video, video_path)
        except Exception as e:
            logger.error("Audio extraction failed for %s: %s", video_path, e)
            raise HTTPException(status_code=422, detail="не удалось извлечь аудио из видео")

        audio_path = local_result.audio_path
        video_id = local_result.file_id
        title = local_result.title
    else:
        try:
            download_result = await asyncio.to_thread(download_audio, req.url)
        except Exception as e:
            logger.error("Download failed for %s: %s", req.url, e)
            raise HTTPException(status_code=422, detail="не удалось скачать видео")

        audio_path = download_result.audio_path
        video_id = download_result.video_id
        title = download_result.title

    logger.info("Download complete: [%s] %s", video_id, title)
    logger.info("Starting transcription: %s", audio_path)
    try:
        result = await _transcribe_and_ingest(
            audio_path=audio_path,
            doc_id=video_id,
            title=title,
            export_txt=req.export_txt,
            delete_audio=True,
        )
    except Exception as e:
        logger.error("Transcription failed for [%s] %s: %s", video_id, title, e)
        raise HTTPException(status_code=502, detail="сервис транскрибации недоступен")

    logger.info("Ingest complete: [%s] %s", video_id, title)

    result["video_id"] = result.pop("doc_id")
    return result


class PlaylistIngestRequest(BaseModel):
    url: str = Field(..., description="YouTube playlist or video URL")
    export_txt: bool = Field(False, description="Include transcript text in response items")
    keep_video: bool = Field(False, description="Download and save full video to data/video/")


@app.post("/ingest-playlist", tags=["Ingest"])
async def ingest_playlist(req: PlaylistIngestRequest):
    try:
        video_urls = await asyncio.to_thread(get_playlist_urls, req.url)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"не удалось загрузить плейлист: {e}")

    logger.info("Playlist has %d videos", len(video_urls))
    items = []
    errors = []

    for video_url in video_urls:
        try:
            if req.keep_video:
                video_path = await asyncio.to_thread(download_video, video_url)
                local_result = await asyncio.to_thread(extract_audio_from_video, video_path)
                audio_path = local_result.audio_path
                video_id = local_result.file_id
                title = local_result.title
            else:
                dr = await asyncio.to_thread(download_audio, video_url)
                audio_path = dr.audio_path
                video_id = dr.video_id
                title = dr.title

            item = await _transcribe_and_ingest(
                audio_path=audio_path,
                doc_id=video_id,
                title=title,
                export_txt=req.export_txt,
                delete_audio=True,
            )
            item["video_id"] = item.pop("doc_id")
            items.append(item)
        except Exception as e:
            logger.error("Failed to process %s: %s", video_url, e)
            errors.append({"url": video_url, "error": str(e)})

    return {
        "ingested_count": len(items),
        "error_count": len(errors),
        "items": items,
        "errors": errors,
    }


@app.post("/ingest-upload", tags=["Ingest"])
async def ingest_upload(
    files: List[UploadFile] = File(...),
    export_txt: bool = False,
):
    work_dir = tempfile.mkdtemp()
    items = []
    errors = []

    try:
        for upload_file in files:
            raw_path = Path(work_dir) / upload_file.filename
            try:
                raw_path.write_bytes(await upload_file.read())
                local_result = await asyncio.to_thread(
                    extract_audio_from_video, str(raw_path), work_dir
                )
                try:
                    raw_path.unlink(missing_ok=True)
                except OSError:
                    pass

                item = await _transcribe_and_ingest(
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

    return {
        "ingested_count": len(items),
        "error_count": len(errors),
        "items": items,
        "errors": errors,
    }


@app.post("/forward", tags=["Usage"])
async def forward(req: ForwardRequest):
    try:
        result = await run(
            question=req.question,
            top_k=req.top_k,
            similarity_threshold=req.similarity_threshold
        )

        return result
    except Exception as e:
        logger.exception("Error in /forward: %s", e)
        raise HTTPException(
            status_code=403,
            detail="модель не смогла обработать данные"
        )
