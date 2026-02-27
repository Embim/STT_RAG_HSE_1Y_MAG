from pathlib import Path

import httpx

from settings import settings


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
    print(path)
    async with httpx.AsyncClient(timeout=600.0) as client:
        with open(path, "rb") as f:
            response = await client.post(
                f"{settings.WHISPER_URL}/v1/audio/transcriptions",
                files={"file": (path.name, f, "audio/mpeg")},
                data={"model": "whisper-1", "response_format": "text"},
            )
            response.raise_for_status()

    return response.text
