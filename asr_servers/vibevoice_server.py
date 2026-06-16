"""OpenAI-compatible HTTP wrapper for microsoft/VibeVoice-ASR-HF.

Используем HF-вариант VibeVoice через `transformers` напрямую (не vLLM —
там сломан pipeline). API специфический для VibeVoice:
  - VibeVoiceAsrForConditionalGeneration (не AutoModelForSpeechSeq2Seq)
  - processor.apply_transcription_request(audio=, prompt=) (не __call__)
  - модель ждёт 24kHz audio (acoustic_tokenizer работает на 24kHz)
  - decode с return_format='transcription_only' даёт чистый текст без
    спикер-меток и таймстемпов

Hotwords/context передаём через `prompt` — это родной механизм VibeVoice
для подачи domain-glossary терминов («About VibeVoice», список англицизмов,
имена собственные). У нас мапится в OpenAI-API поле `prompt`.

Запуск (через docker compose --profile asr-vibevoice):
    HF_MODEL=microsoft/VibeVoice-ASR-HF ASR_PORT=8000 python server.py
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("vibevoice-server")

MODEL_NAME = os.environ.get("HF_MODEL", "microsoft/VibeVoice-ASR-HF")
PORT = int(os.environ.get("ASR_PORT", 8000))
HOST = os.environ.get("ASR_HOST", "0.0.0.0")
QUANT = os.environ.get("VIBEVOICE_QUANT", "bnb4")  # bnb4 | bnb8 | none
# VibeVoice acoustic_tokenizer работает на 24kHz. Если входной WAV/OGG в
# другой частоте — resample через librosa.
TARGET_SR = 24000

logger.info("Loading %s (quant=%s, target_sr=%d)", MODEL_NAME, QUANT, TARGET_SR)

from transformers import (  # noqa: E402
    AutoProcessor,
    BitsAndBytesConfig,
    VibeVoiceAsrForConditionalGeneration,
)

_quant_kwargs: dict = {}
if QUANT == "bnb4":
    _quant_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
elif QUANT == "bnb8":
    _quant_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
else:
    _quant_kwargs["dtype"] = torch.bfloat16

_processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
_model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    device_map="auto",
    **_quant_kwargs,
).eval()
logger.info("Model loaded; device: %s, dtype: %s",
            getattr(_model, "device", "?"), getattr(_model, "dtype", "?"))

app = FastAPI(title="vibevoice-asr", version="2.0")


@app.get("/v1/models")
def list_models() -> dict:
    return {"data": [{"id": MODEL_NAME, "object": "model"}]}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_NAME, "quant": QUANT}


def _load_audio_24k(path: Path):
    """Загрузить аудио файл и привести к 24kHz mono float32."""
    import soundfile as sf
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        # Lazy import librosa — он нужен только если sample rate отличается
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    return audio


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(default=MODEL_NAME),
    language: str = Form(default="ru"),
    # OpenAI `prompt` → VibeVoice `prompt` (context для лучшей точности
    # на доменных терминах, code-switching, именах собственных).
    prompt: Optional[str] = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
) -> JSONResponse:
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        audio = _load_audio_24k(tmp_path)
        # apply_transcription_request формирует chat-template с audio token
        # и опциональным prompt'ом, кладёт всё нужное для model.generate().
        kwargs: dict = {"audio": audio}
        if prompt:
            kwargs["prompt"] = prompt
        inputs = _processor.apply_transcription_request(**kwargs)
        inputs = inputs.to(_model.device, _model.dtype)

        with torch.inference_mode():
            output_ids = _model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
            )
        # Срезаем prompt-токены, остаётся только сгенерированное.
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        # `transcription_only` убирает JSON-структуру с Speaker/Start/End,
        # отдаёт чистый текст — это и есть наш hypothesis для WER.
        text = _processor.decode(generated_ids,
                                 return_format="transcription_only")[0]
        text = text.strip()
        logger.info("transcribed %d chars (prompt=%r) from %s",
                    len(text), prompt or "", file.filename)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass

    payload: dict = {"text": text}
    if response_format == "verbose_json":
        payload.update({
            "language": language,
            "duration": float(len(audio) / TARGET_SR),
            "segments": [],
        })
    return JSONResponse(payload)


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
