"""Transcribe every audio file in a directory and write per-file JSONs."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from evaluation.asr.backends.base import ASRBackend
from evaluation.paths import RESULTS_DIR

logger = logging.getLogger(__name__)


_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm"}


async def transcribe_dir(
    backend: ASRBackend,
    *,
    src_dir: Path,
    out_dir: Optional[Path],
) -> Path:
    """For every audio in src_dir write a JSON to out_dir matching the
    data/transcripts schema (text + segments + metadata).

    Skips files that already have a JSON in out_dir, so this can resume.
    """
    out_dir = out_dir or (RESULTS_DIR.parent / "transcripts_eval" / backend.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_files = [p for p in sorted(src_dir.rglob("*")) if p.suffix.lower() in _AUDIO_EXTS]
    if not audio_files:
        logger.warning("No audio files found in %s", src_dir)
        return out_dir

    logger.info("Transcribing %d files with %s -> %s", len(audio_files), backend.name, out_dir)
    for audio in audio_files:
        target = out_dir / (audio.stem + ".json")
        if target.exists():
            logger.info("Skip (exists): %s", target.name)
            continue
        try:
            result = await backend.transcribe(str(audio))
        except Exception as e:
            logger.error("Transcribe failed for %s: %s", audio.name, e)
            continue
        payload = {
            "source_file_name": audio.name,
            "asr_name": backend.name,
            "asr_model_id": getattr(backend, "model_id", None),
            "asr_url": getattr(backend, "base_url", None),
            "language": result.get("language", getattr(backend, "language", None)),
            "text": result.get("text", ""),
            "segments": result.get("segments", []),
            "processed_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("Wrote %s (%d chars)", target.name, len(payload["text"]))
    return out_dir
