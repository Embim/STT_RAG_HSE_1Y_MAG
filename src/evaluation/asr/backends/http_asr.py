"""Single HTTP ASR backend that talks to any OpenAI-compatible transcription
container (faster-whisper-server, WhisperX-server, a self-hosted Qwen3-ASR
wrapper, the OpenAI cloud API, ...).

Switching models is just .env: bring up a different container, point
ASR_BASE_URL at it, set ASR_NAME for reporting.

Two endpoint modes (controlled by `ASR_ENDPOINT` env / `endpoint` arg):

  - `transcription` (default) — POST `{base_url}/v1/audio/transcriptions`
    with multipart form {file, model, response_format, language}. Шаблон
    faster-whisper-server / api.openai.com / большинство vLLM-моделей.

  - `chat` — POST `{base_url}/v1/chat/completions` с аудио как
    data:audio/<mime>;base64,<...> внутри user-message content. Требуется
    для Qwen3-ASR и других моделей с собственным внутренним prompt
    template (model emits `language Russian<asr_text>...`). Распарсим
    output, извлечём чистый transcript.
"""
from __future__ import annotations

import base64
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from settings import settings
from evaluation.asr.backends.base import TranscriptionResult

logger = logging.getLogger(__name__)

# Соответствие расширения файла → MIME-тип, ожидаемый vLLM audio_url decoder.
_MIME_BY_EXT = {
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".webm": "audio/webm",
}


def _extract_segments(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    for segment in payload.get("segments", []):
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        segments.append(
            {
                "text": text,
                "start": float(segment.get("start", 0.0) or 0.0),
                "end": float(segment.get("end", 0.0) or 0.0),
            }
        )
    return segments


# Qwen3-ASR emits:
#   "language Russian<asr_text>Здесь распознанный текст.</asr_text>"
# (closing tag иногда отсутствует — модель просто хитом EOS обрывает).
_QWEN_ASR_PATTERN = re.compile(
    r"<asr_text>(?P<text>.*?)(?:</asr_text>|$)",
    re.DOTALL,
)


def _parse_chat_asr_output(raw: str) -> str:
    """Из ответа chat-endpoint выдрать чистый transcript.

    Поддерживает Qwen3-ASR формат `language XXX<asr_text>...</asr_text>`,
    плюс гладко падает на остальные модели (просто возвращает raw, если
    маркеров нет).
    """
    m = _QWEN_ASR_PATTERN.search(raw)
    if m:
        return m.group("text").strip()
    return raw.strip()


class HttpASRBackend:
    def __init__(
        self,
        base_url: Optional[str] = None,
        model_id: Optional[str] = None,
        language: Optional[str] = None,
        name: Optional[str] = None,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
    ) -> None:
        self.base_url = (base_url or settings.WHISPER_URL).rstrip("/")
        self.model_id = model_id or settings.ASR_MODEL_ID
        self.language = language or settings.ASR_LANGUAGE
        # `name` is a human label that lands in CSV / Langfuse — keep it
        # concrete (e.g. "qwen3_asr_1.7b") so cross-run comparisons are
        # legible. Falls back to model_id.
        self.name = name or settings.ASR_NAME or self.model_id
        self.api_key = api_key  # optional Bearer token (e.g. for OpenAI cloud)
        # `transcription` (default) или `chat` — управляется env ASR_ENDPOINT.
        # Для Qwen3-ASR ставим chat, потому что у их `/v1/audio/transcriptions`
        # сломан pipeline в vLLM 0.20.2, а /v1/chat/completions работает.
        self.endpoint = (endpoint or settings.ASR_ENDPOINT or "transcription").lower()

    async def transcribe(self, audio_path: str) -> TranscriptionResult:
        path = Path(audio_path)
        size_mb = path.stat().st_size / 1024 / 1024
        logger.info(
            "[%s] %s (%.1f MB) -> %s (model=%s, lang=%s, endpoint=%s)",
            self.name, path.name, size_mb, self.base_url, self.model_id, self.language, self.endpoint,
        )
        if self.endpoint == "chat":
            return await self._transcribe_chat(path)
        return await self._transcribe_transcription(path)

    async def _transcribe_transcription(self, path: Path) -> TranscriptionResult:
        """Стандартный OpenAI /v1/audio/transcriptions endpoint."""
        timeout = httpx.Timeout(connect=10.0, read=3600.0, write=300.0, pool=10.0)
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else None
        # response_format=json — universally supported. verbose_json дал бы
        # segments, но vLLM-бэкенды отклоняют: 'Currently do not support
        # verbose_json for <model>'.
        data: Dict[str, Any] = {
            "model": self.model_id,
            "response_format": "json",
        }
        if self.language:
            data["language"] = self.language

        mime = _MIME_BY_EXT.get(path.suffix.lower(), "application/octet-stream")
        async with httpx.AsyncClient(timeout=timeout) as client:
            with open(path, "rb") as f:
                response = await client.post(
                    f"{self.base_url}/v1/audio/transcriptions",
                    headers=headers,
                    files={"file": (path.name, f, mime)},
                    data=data,
                )
                response.raise_for_status()
        payload = response.json()
        text = str(payload.get("text", ""))
        segments = _extract_segments(payload)
        logger.info("[%s] %d chars, %d segments", self.name, len(text), len(segments))
        return {"text": text, "segments": segments, "language": self.language or ""}

    async def _transcribe_chat(self, path: Path) -> TranscriptionResult:
        """vLLM /v1/chat/completions с аудио как data: URI base64.

        Используется для Qwen3-ASR и любых других audio-LLM, у которых
        transcription endpoint сломан или возвращает пустоту. Передаём
        ОРИГИНАЛЬНЫЙ файл без перекодирования — Qwen3-ASR сильно деградирует
        качество распознавания на конвертированном WAV.
        """
        timeout = httpx.Timeout(connect=10.0, read=3600.0, write=300.0, pool=10.0)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        mime = _MIME_BY_EXT.get(path.suffix.lower(), "application/octet-stream")
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        data_uri = f"data:{mime};base64,{b64}"

        body: Dict[str, Any] = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio_url", "audio_url": {"url": data_uri}},
                    ],
                }
            ],
            "max_tokens": 2048,
            "temperature": 0,
        }
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{self.base_url}/v1/chat/completions",
                headers=headers,
                json=body,
            )
            response.raise_for_status()
        payload = response.json()
        try:
            raw = payload["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            logger.error("[%s] malformed chat response: %s", self.name, e)
            raw = ""
        text = _parse_chat_asr_output(raw)
        logger.info("[%s] %d chars (chat endpoint)", self.name, len(text))
        return {"text": text, "segments": [], "language": self.language or ""}
