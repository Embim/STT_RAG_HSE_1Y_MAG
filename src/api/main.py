import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import colorlog
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from system.rag.pipeline import run
from system.llm.llm_services import get_chat_vectore_store_manager
from system.tracing import flush as langfuse_flush
from downloader.processor import process_youtube, process_uploaded_files
from api.schemas import ForwardRequest, IngestRequest

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
    yield
    langfuse_flush()


app = FastAPI(title="DS Navigator API", lifespan=lifespan)
logger = logging.getLogger(__name__)


# ── Access gate ──────────────────────────────────────────────────────
# Когда задан DEMO_ACCESS_TOKEN — дорогие/абьюзо-опасные ручки требуют
# заголовок X-Demo-Token (или ?key=...). Нужно при публичной выдаче через
# туннель: /forward жжёт OpenRouter-кредиты, ingest качает+транскрибирует
# по запросу. Пусто (дефолт) → гейт выключен, локальная разработка как была.
DEMO_ACCESS_TOKEN = os.getenv("DEMO_ACCESS_TOKEN", "")
_PROTECTED_PATHS = {"/forward", "/ingest", "/ingest-upload"}


@app.middleware("http")
async def access_gate(request: Request, call_next):
    if DEMO_ACCESS_TOKEN and request.url.path in _PROTECTED_PATHS:
        provided = request.headers.get("X-Demo-Token") or request.query_params.get("key")
        if provided != DEMO_ACCESS_TOKEN:
            return JSONResponse(status_code=401, content={"detail": "unauthorized"})
    return await call_next(request)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=400, content={"detail": "bad request"})


@app.get("/api", tags=["Root"])
async def root():
    return {"message": "Welcome to DS Navigator API", "documentation": "/docs"}


@app.get("/auth-check", tags=["Health"])
async def auth_check(request: Request):
    """Фронт дёргает это, чтобы проверить введённый ключ перед стартом.
    Если гейт выключен — всегда ok. Если включён — проверяет X-Demo-Token."""
    if not DEMO_ACCESS_TOKEN:
        return {"gate": False, "ok": True}
    provided = request.headers.get("X-Demo-Token") or request.query_params.get("key")
    if provided != DEMO_ACCESS_TOKEN:
        raise HTTPException(status_code=401, detail="unauthorized")
    return {"gate": True, "ok": True}


@app.get("/health", status_code=status.HTTP_200_OK, tags=["Health"])
async def healthcheck():
    return {"status": "ok"}


@app.get("/check-vdb", tags=["Health"])
async def check_vdb():
    try:
        collection = get_chat_vectore_store_manager().collection
        count = collection.aggregate.over_all(total_count=True).total_count
        return {"status": "ok", "documents_in_vdb": count}
    except Exception:
        raise HTTPException(status_code=500, detail="vector db is not available")


@app.post("/ingest", tags=["Ingest"])
async def ingest(req: IngestRequest):
    logger.info(
        "Ingest request: %s (keep_video=%s, keep_audio=%s, export_txt=%s, export_json=%s)",
        req.url, req.keep_video, req.keep_audio, req.export_txt, req.export_json,
    )
    try:
        return await process_youtube(
            url=req.url,
            keep_video=req.keep_video,
            export_txt=req.export_txt,
            export_json=req.export_json,
            keep_audio=req.keep_audio,
        )
    except Exception as e:
        logger.error("Ingest failed for %s: %s", req.url, e)
        raise HTTPException(status_code=422, detail="не удалось обработать URL")


@app.post("/ingest-upload", tags=["Ingest"])
async def ingest_upload(
    files: List[UploadFile] = File(...),
    export_txt: bool = False,
    export_json: bool = False,
    keep_audio: bool = False,
):
    return await process_uploaded_files(
        files=files,
        export_txt=export_txt,
        export_json=export_json,
        keep_audio=keep_audio,
    )


@app.post("/forward", tags=["Usage"])
async def forward(req: ForwardRequest):
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
async def source_files():
    try:
        files = get_chat_vectore_store_manager().list_source_titles()
        return {"files": files}
    except Exception as e:
        logger.exception("Error in /source-files: %s", e)
        raise HTTPException(status_code=500, detail="не удалось получить список файлов")


# ── Статический фронт (SPA) ──────────────────────────────────────────
# Раздаём кастомный фронт по "/" тем же origin'ом, что и API → CORS не
# нужен, один туннель отдаёт и сайт, и ручки. Mount добавлен ПОСЛЕ всех
# API-роутов, поэтому /forward, /docs и т.п. матчатся раньше catch-all "/".
WEB_DIR = Path(__file__).resolve().parent.parent / "web"
if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
    logger.info("Serving SPA from %s at /", WEB_DIR)
else:
    logger.warning("Web dir %s not found — SPA not served (API-only mode)", WEB_DIR)
