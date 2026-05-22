"""Common protocol for ASR backends."""
from __future__ import annotations

from typing import Any, Dict, List, Protocol, TypedDict, runtime_checkable


class Segment(TypedDict, total=False):
    text: str
    start: float
    end: float


class TranscriptionResult(TypedDict, total=False):
    text: str
    segments: List[Segment]
    language: str


@runtime_checkable
class ASRBackend(Protocol):
    """Async transcription backend with a stable identifier.

    Implementations must populate at least `text` in the returned dict.
    `segments` is optional but strongly recommended (the ingest pipeline uses
    them to keep timestamps in vector store metadata).
    """

    name: str

    async def transcribe(self, audio_path: str) -> TranscriptionResult: ...
