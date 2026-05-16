"""Production transcription entry point.

Thin shim over the universal HTTP ASR backend. To swap ASR models:
1. Bring up a different OpenAI-compatible transcription container.
2. Point WHISPER_URL / ASR_MODEL_ID / ASR_NAME at it in .env.
3. Restart the API. No code change needed.
"""
import logging
from typing import Any, Dict

from evaluation.asr.backends.http_asr import HttpASRBackend

logger = logging.getLogger(__name__)


async def transcribe(audio_path: str) -> Dict[str, Any]:
    """Transcribe an audio file via the configured ASR endpoint.

    Returns a dict with at least:
        - "text": full transcript
        - "segments": list of {text, start, end} (may be empty if backend doesn't expose them)
    """
    backend = HttpASRBackend()
    logger.info("Transcribing %s with %s", audio_path, backend.name)
    return await backend.transcribe(audio_path)
