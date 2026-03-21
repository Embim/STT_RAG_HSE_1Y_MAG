import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import colorlog
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from system.rag.pipeline import run
from system.llm.llm_services import CHAT_VECTORE_STORE_MANAGER
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


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=400, content={"detail": "bad request"})


@app.get("/", tags=["Root"])
async def root():
    return {"message": "Welcome to DS Navigator API", "documentation": "/docs"}


@app.get("/health", status_code=status.HTTP_200_OK, tags=["Health"])
async def healthcheck():
    return {"status": "ok"}


@app.get("/check-vdb", tags=["Health"])
async def check_vdb():
    try:
        collection = CHAT_VECTORE_STORE_MANAGER.collection
        count = collection.aggregate.over_all(total_count=True).total_count
        return {"status": "ok", "documents_in_vdb": count}
    except Exception:
        raise HTTPException(status_code=500, detail="vector db is not available")


@app.post("/ingest", tags=["Ingest"])
async def ingest(req: IngestRequest):
    logger.info("Ingest request: %s (keep_video=%s)", req.url, req.keep_video)
    try:
        return await process_youtube(req.url, req.keep_video, req.export_txt)
    except Exception as e:
        logger.error("Ingest failed for %s: %s", req.url, e)
        raise HTTPException(status_code=422, detail="не удалось обработать URL")


@app.post("/ingest-upload", tags=["Ingest"])
async def ingest_upload(files: List[UploadFile] = File(...), export_txt: bool = False):
    return await process_uploaded_files(files, export_txt)


@app.post("/forward", tags=["Usage"])
async def forward(req: ForwardRequest):
    try:
        return await run(question=req.question, top_k=req.top_k, similarity_threshold=req.similarity_threshold)
    except Exception as e:
        logger.exception("Error in /forward: %s", e)
        raise HTTPException(status_code=403, detail="модель не смогла обработать данные")
