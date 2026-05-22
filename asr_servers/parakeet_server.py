"""OpenAI-compatible HTTP wrapper for NVIDIA Parakeet-TDT-0.6B-v3.

Тонкая FastAPI-обёртка вокруг `nemo.collections.asr.models.ASRModel`, чтобы
parakeet встал в общий ряд ASR-backend'ов на /v1/audio/transcriptions.

Parakeet, в отличие от Canary-Qwen, авто-детектирует язык per-utterance и
не умеет code-switching — это **русский-only baseline**, годный для
сравнения с CS-моделями (Qwen3-ASR, Voxtral), а не основной backend на
лекциях с англицизмами.

Запуск (через docker compose --profile asr-parakeet):
    NEMO_MODEL=nvidia/parakeet-tdt-0.6b-v3 ASR_PORT=8000 python server.py
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("parakeet-server")

MODEL_NAME = os.environ.get("NEMO_MODEL", "nvidia/parakeet-tdt-0.6b-v3")
PORT = int(os.environ.get("ASR_PORT", 8000))
HOST = os.environ.get("ASR_HOST", "0.0.0.0")

logger.info("Loading NeMo model: %s", MODEL_NAME)
import nemo.collections.asr as nemo_asr  # noqa: E402

_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME).to(_device).eval()
logger.info("Model loaded on %s", _device)

app = FastAPI(title="parakeet-asr", version="1.0")


@app.get("/v1/models")
def list_models() -> dict:
    return {"data": [{"id": MODEL_NAME, "object": "model"}]}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_NAME, "device": _device}


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(default=MODEL_NAME),
    # `language` принимается, но Parakeet сам детектит — параметр игнорируется
    # на уровне модели; вернётся как часть verbose_json для совместимости.
    language: str = Form(default="ru"),
    prompt: str | None = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
) -> JSONResponse:
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        results = _model.transcribe([tmp_path])
        # Parakeet возвращает Hypothesis-объекты с .text
        text = results[0].text if hasattr(results[0], "text") else str(results[0])
        logger.info("transcribed %d chars from %s", len(text), file.filename)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    payload: dict = {"text": text}
    if response_format == "verbose_json":
        payload.update({"language": language, "duration": 0.0, "segments": []})
    return JSONResponse(payload)


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
