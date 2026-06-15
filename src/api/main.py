import asyncio
import logging
import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import colorlog
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from settings import settings
from system.rag.pipeline import run
from system.llm.llm_services import get_chat_vectore_store_manager
from system.tracing import flush as langfuse_flush
from downloader.processor import process_youtube, process_saved_uploads
from api.schemas import ForwardRequest, IngestRequest, EmbeddingLocateRequest
from system.embedding_map import build_map, locate as locate_in_map, reset_cache as reset_map_cache
from system.auth.deps import get_current_user
from system.auth.models import User
from system.auth.service import ensure_admin
from api.auth_routes import router as auth_router
from system.ingest_jobs import create_job, update_job, get_job
from system.asr_models import ASR_MODELS, DEFAULT_ASR_MODEL, get_model, list_models
from system.asr_manager import asr_manager

MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB per file

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Console handler — цветной по уровню
_console = colorlog.StreamHandler()
_console.setFormatter(colorlog.ColoredFormatter(
    fmt="%(log_color)s%(asctime)s %(levelname)-8s%(reset)s %(blue)s%(name)s%(reset)s: %(message)s",
    datefmt="%H:%M:%S",
    log_colors={
        "DEBUG":    "cyan",
        "INFO":     "green",
        "WARNING":  "yellow",
        "ERROR":    "red",
        "CRITICAL": "bold_red",
    },
))

# File handler — plain text, полный timestamp
_file = logging.FileHandler(LOG_DIR / "api.log", encoding="utf-8")
_file.setFormatter(logging.Formatter(
    fmt="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))

logging.basicConfig(level=logging.INFO, handlers=[_console, _file])


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_admin()
    yield
    langfuse_flush()


app = FastAPI(title="DS Navigator API", lifespan=lifespan)
app.include_router(auth_router)
logger = logging.getLogger(__name__)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=400, content={"detail": "bad request"})


@app.get("/api", tags=["Root"])
async def root():
    return {"message": "Welcome to DS Navigator API", "documentation": "/docs"}


@app.get("/health", status_code=status.HTTP_200_OK, tags=["Health"])
async def healthcheck():
    return {"status": "ok"}


@app.get("/check-vdb", tags=["Health"])
async def check_vdb(user: User = Depends(get_current_user)):
    try:
        collection = get_chat_vectore_store_manager().collection
        count = collection.aggregate.over_all(total_count=True).total_count
        return {"status": "ok", "documents_in_vdb": count}
    except Exception:
        raise HTTPException(status_code=500, detail="vector db is not available")


async def _run_youtube_job(job_id: str, req: IngestRequest) -> None:
    update_job(job_id, status="running")
    try:
        model = get_model(req.asr_model)
        async with asr_manager.session(
            req.asr_model, status_cb=lambda m: update_job(job_id, detail=m)
        ) as backend:
            res = await process_youtube(
                url=req.url, export_txt=req.export_txt,
                export_json=req.export_json, keep_audio=req.keep_audio,
                use_ocr=req.use_ocr, backend=backend, asr_label=model.name,
                progress_cb=lambda u: update_job(job_id, **u),
            )
        update_job(job_id, status="done", progress=100.0, items=res["items"],
                   errors=res["errors"], error_count=res["error_count"],
                   done_items=res["ingested_count"],
                   total_items=res["ingested_count"] + res["error_count"])
        reset_map_cache()  # в БД новые чанки → карта тем устарела
    except Exception as e:
        logger.exception("ingest job %s failed: %s", job_id, e)
        update_job(job_id, status="error", detail=str(e))


async def _run_upload_job(job_id, saved, tmp_dir, export_txt, export_json, keep_audio, use_ocr, asr_model) -> None:
    update_job(job_id, status="running")
    try:
        model = get_model(asr_model)
        async with asr_manager.session(
            asr_model, status_cb=lambda m: update_job(job_id, detail=m)
        ) as backend:
            res = await process_saved_uploads(
                saved, export_txt=export_txt, export_json=export_json, keep_audio=keep_audio,
                use_ocr=use_ocr, backend=backend, asr_label=model.name,
                progress_cb=lambda u: update_job(job_id, **u),
            )
        update_job(job_id, status="done", progress=100.0, items=res["items"],
                   errors=res["errors"], error_count=res["error_count"],
                   done_items=res["ingested_count"],
                   total_items=res["ingested_count"] + res["error_count"])
        reset_map_cache()  # в БД новые чанки → карта тем устарела
    except Exception as e:
        logger.exception("upload job %s failed: %s", job_id, e)
        update_job(job_id, status="error", detail=str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/ingest", status_code=202, tags=["Ingest"])
async def ingest(req: IngestRequest, user: User = Depends(get_current_user)):
    job_id = create_job()
    asyncio.create_task(_run_youtube_job(job_id, req))
    return {"job_id": job_id, "status": "queued"}


@app.post("/ingest-upload", status_code=202, tags=["Ingest"])
async def ingest_upload(
    files: List[UploadFile] = File(...),
    export_txt: bool = False,
    export_json: bool = False,
    keep_audio: bool = False,
    use_ocr: bool = False,
    asr_model: str = DEFAULT_ASR_MODEL,
    user: User = Depends(get_current_user),
):
    model = ASR_MODELS.get(asr_model)
    if model is None or not model.available:
        raise HTTPException(status_code=400, detail=f"invalid asr_model '{asr_model}'")
    tmp_dir = tempfile.mkdtemp(prefix="ingest_up_")
    saved: list[tuple[str, str]] = []
    for f in files:
        name = Path(f.filename or "").name or f"upload_{len(saved)}"   # None/empty/path-traversal safe
        dest = str(Path(tmp_dir) / name)
        size = 0
        with open(dest, "wb") as out:
            while True:
                chunk = await f.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    out.close()
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                    raise HTTPException(status_code=413, detail="file too large")
                out.write(chunk)
        saved.append((dest, name))
    job_id = create_job()
    asyncio.create_task(_run_upload_job(job_id, saved, tmp_dir, export_txt, export_json, keep_audio, use_ocr, asr_model))
    return {"job_id": job_id, "status": "queued"}


@app.get("/ingest-status/{job_id}", tags=["Ingest"])
async def ingest_status(job_id: str, user: User = Depends(get_current_user)):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return {k: v for k, v in job.items() if k != "created_at"}


@app.get("/asr-models", tags=["Ingest"])
async def asr_models(user: User = Depends(get_current_user)):
    """Каталог ASR-моделей для выпадающего списка на фронте.

    `active` — какая модель сейчас поднята на GPU (если включён авто-свап;
    иначе null). `autoswap` — включён ли авто-свап контейнеров.
    """
    return {
        "models": list_models(),
        "default": DEFAULT_ASR_MODEL,
        "active": await asr_manager.active_model_key(),
        "autoswap": settings.ASR_AUTOSWAP_ENABLED,
    }


@app.get("/embedding-map", tags=["Map"])
async def embedding_map(force: bool = False, user: User = Depends(get_current_user)):
    """3D-карта эмбеддингов («облако тем»): точки-чанки + авто-темы.

    Тяжёлый расчёт (UMAP/KMeans) идёт в threadpool, чтобы не блокировать loop;
    результат кэшируется до изменения корпуса (или force=true). Принудительная
    пересборка дорогая, поэтому force доступен только админам.
    """
    if force and user.role != "admin":
        force = False
    try:
        return await asyncio.to_thread(build_map, force)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ModuleNotFoundError as e:
        raise HTTPException(status_code=501, detail=f"требуется scikit-learn: {e}")


@app.post("/embedding-map/locate", tags=["Map"])
async def embedding_map_locate(req: EmbeddingLocateRequest, user: User = Depends(get_current_user)):
    """Спроецировать запрос в карту: id ближайших чанков + маркер запроса."""
    try:
        return await asyncio.to_thread(locate_in_map, req.question, req.top_k)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ModuleNotFoundError as e:
        raise HTTPException(status_code=501, detail=f"требуется scikit-learn: {e}")


@app.post("/forward", tags=["Usage"])
async def forward(req: ForwardRequest, user: User = Depends(get_current_user)):
    try:
        return await run(
            question=req.question,
            top_k=req.top_k,
            similarity_threshold=req.similarity_threshold,
            use_rewrite=req.use_rewrite,
            source_file_name=req.source_file_name,
            source_title=req.source_title,
        )
    except Exception as e:
        logger.exception("Error in /forward: %s", e)
        raise HTTPException(status_code=403, detail="модель не смогла обработать данные")


@app.get("/source-files", tags=["Usage"])
async def source_files(user: User = Depends(get_current_user)):
    try:
        files = get_chat_vectore_store_manager().list_source_titles()
        return {"files": files}
    except Exception as e:
        logger.exception("Error in /source-files: %s", e)
        raise HTTPException(status_code=500, detail="не удалось получить список файлов")


# ── Статический фронт (SPA) ──────────────────────────────────────────
# nginx раздаёт Angular-бандл в проде → SERVE_SPA=false в api-контейнере.
# По умолчанию "true", чтобы локальный dev (uvicorn без nginx) всё ещё
# отдавал src/web/index.html. Mount добавлен ПОСЛЕ всех API-роутов, поэтому
# /forward, /docs и т.п. матчатся раньше catch-all "/".
WEB_DIR = Path(__file__).resolve().parent.parent / "web"
_serve_spa = os.getenv("SERVE_SPA", "true").lower() not in ("false", "0", "no")
if _serve_spa and WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
    logger.info("Serving SPA from %s at /", WEB_DIR)
else:
    logger.info("SPA serving disabled (SERVE_SPA=%s) — API-only mode", os.getenv("SERVE_SPA"))
