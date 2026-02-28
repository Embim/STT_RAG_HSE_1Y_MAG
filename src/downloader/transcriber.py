import logging
from pathlib import Path

import httpx

from settings import settings

logger = logging.getLogger(__name__)


async def transcribe(audio_path: str) -> str:
    """Transcribe an audio file via the Whisper HTTP API.

    Uses the OpenAI-compatible endpoint:
        POST {WHISPER_URL}/v1/audio/transcriptions

    Args:
        audio_path: Absolute path to the audio file (mp3).

    Returns:
        Transcribed text.
    """
    path = Path(audio_path)
    file_size_mb = path.stat().st_size / 1024 / 1024
    logger.info("Sending to Whisper: %s (%.1f MB)", path.name, file_size_mb)
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=3600.0, write=300.0, pool=10.0)) as client:
        with open(path, "rb") as f:
            response = await client.post(
                f"{settings.WHISPER_URL}/v1/audio/transcriptions",
                files={"file": (path.name, f, "audio/mpeg")},
                data={"model": "whisper-1", "response_format": "text"},
            )
            response.raise_for_status()

    logger.info("Whisper response: %d chars for %s", len(response.text), path.name)
    return response.text
