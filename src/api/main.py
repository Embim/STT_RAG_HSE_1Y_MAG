import asyncio
import logging
import os

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional

from system.rag.pipeline import run
from system.llm.llm_services import CHAT_VECTORE_STORE_MANAGER
from downloader.youtube import download_audio
from downloader.transcriber import transcribe
from downloader.ingest import ingest_json_to_vector_store

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


@app.post("/ingest", tags=["Ingest"])
async def ingest(req: IngestRequest):
    try:
        download_result = await asyncio.to_thread(download_audio, req.url)
    except Exception:
        raise HTTPException(status_code=422, detail="не удалось скачать видео")

    try:
        text = await transcribe(download_result.audio_path)
    except Exception:
        raise HTTPException(status_code=502, detail="сервис транскрибации недоступен")
    finally:
        os.remove(download_result.audio_path)

    await ingest_json_to_vector_store([{"hash": download_result.video_id, "text": text}])

    return {
        "status": "ok",
        "video_id": download_result.video_id,
        "title": download_result.title,
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