"""Production transcription entry point.

Thin shim over the universal HTTP ASR backend. To swap ASR models:
1. Bring up a different OpenAI-compatible transcription container.
2. Point WHISPER_URL / ASR_MODEL_ID / ASR_NAME at it in .env.
3. Restart the API. No code change needed.
"""
import logging
from typing import Any, Dict, Optional

from evaluation.asr.backends.http_asr import HttpASRBackend

logger = logging.getLogger(__name__)


async def transcribe(
    audio_path: str, backend: Optional[HttpASRBackend] = None
) -> Dict[str, Any]:
    """Transcribe an audio file via the configured ASR endpoint.

    Args:
        audio_path: path to the media file.
        backend: optional pre-configured backend (e.g. one the ASR manager
            built for the user-selected model). Defaults to a backend from
            settings (WHISPER_URL / ASR_NAME).

    Returns a dict with at least:
        - "text": full transcript
        - "segments": list of {text, start, end} (may be empty if backend doesn't expose them)
    """
    backend = backend or HttpASRBackend()
    logger.info("Transcribing %s with %s", audio_path, backend.name)
    return await backend.transcribe(audio_path)
