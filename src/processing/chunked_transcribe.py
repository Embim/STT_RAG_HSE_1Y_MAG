"""Chunked transcription wrapper around the existing ASR backend.

The default `transcribe(audio_path)` sends the whole file to the whisper
container in one request, which means progress is binary: pending → done
after N minutes. For real-time progress in MLflow we need *checkpoints*.

This module slices the input file into N-minute pieces with ffmpeg, sends
each one to the configured backend, then concatenates the results. After
each chunk the caller's progress callback fires, so MLflow gets a fresh
`progress_pct` metric every N minutes of audio (which is one HTTP call to
whisper, ~30-60 sec of wall time).

Output schema is the same as `evaluation.asr.backends.base.TranscriptionResult`
(text + segments + language). Segments are time-shifted by the chunk's start
offset so the global timeline is correct.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from evaluation.asr.backends.base import TranscriptionResult
from evaluation.asr.backends.http_asr import HttpASRBackend

logger = logging.getLogger(__name__)


def _ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on PATH; install it or fall back to non-chunked transcribe"
        )


def _ffprobe_duration(audio_path: Path) -> float:
    """Return audio duration in seconds via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def ffprobe_metadata(media_path) -> Dict[str, Any]:
    """Extract container/codec/bitrate metadata via ffprobe.

    Returns a flat dict ready to push to MLflow as params:
        - duration_sec
        - container_format (e.g. "mp3", "mov,mp4,m4a,3gp,3g2,mj2")
        - bitrate_kbps
        - audio_codec / audio_sample_rate / audio_channels
        - video_codec (empty if no video stream)

    Returns empty dict on failure (ffprobe missing, unreadable file).
    """
    import json
    p = Path(media_path)
    if not p.exists():
        return {}
    if shutil.which("ffprobe") is None:
        return {}
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(p)],
            capture_output=True, text=True, check=True, timeout=15,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {}
    try:
        data = json.loads(out.stdout)
    except json.JSONDecodeError:
        return {}
    fmt = data.get("format", {}) or {}
    streams = data.get("streams", []) or []
    audio = next((s for s in streams if s.get("codec_type") == "audio"), {}) or {}
    video = next((s for s in streams if s.get("codec_type") == "video"), {}) or {}
    return {
        "duration_sec": float(fmt.get("duration") or 0.0),
        "container_format": str(fmt.get("format_name") or ""),
        "bitrate_kbps": int(int(fmt.get("bit_rate") or 0) / 1000),
        "audio_codec": str(audio.get("codec_name") or ""),
        "audio_sample_rate": int(audio.get("sample_rate") or 0),
        "audio_channels": int(audio.get("channels") or 0),
        "video_codec": str(video.get("codec_name") or ""),
    }


def _ffmpeg_cut(src: Path, start: float, duration: float, dst: Path) -> None:
    """Extract [start, start+duration] from src into dst as 16kHz mono mp3."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start:.3f}",
        "-i", str(src),
        "-t", f"{duration:.3f}",
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-codec:a", "libmp3lame",
        "-q:a", "2",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


ProgressCb = Callable[[int, int, Dict[str, Any]], None]
"""Callback signature: (chunk_idx_1based, total_chunks, extra).

`extra` is a dict with per-chunk telemetry caller may push to MLflow:
    - chunk_time_sec: wall time spent transcribing this chunk
    - chunk_chars: characters returned for this chunk
    - chunk_chars_per_sec: throughput on this chunk's audio
    - chunk_audio_sec: duration of audio in this chunk
"""


async def transcribe_chunked(
    audio_path: str,
    *,
    chunk_minutes: int = 5,
    progress_cb: Optional[ProgressCb] = None,
    backend: Optional[HttpASRBackend] = None,
) -> TranscriptionResult:
    """Slice → transcribe each chunk → merge.

    Args:
        audio_path: Path to a media file (mp3/mp4/wav/...).
        chunk_minutes: Length of each chunk in minutes. <= 0 means "no chunking,
            use the backend on the whole file as before".
        progress_cb: Called after each chunk is transcribed with
            (chunk_idx, total_chunks). Use to update MLflow progress.
        backend: Optional HttpASRBackend instance. Defaults to one constructed
            from settings (i.e. the configured WHISPER_URL/ASR_NAME).

    Returns:
        Same shape as `HttpASRBackend.transcribe`: dict with `text`, `segments`,
        `language`. Segments' start/end are in the global audio timeline.
    """
    backend = backend or HttpASRBackend()
    src = Path(audio_path)

    if chunk_minutes <= 0:
        # Caller asked for no chunking — let the backend handle the file as-is.
        if progress_cb:
            progress_cb(0, 1, {})
        t0 = time.time()
        result = await backend.transcribe(audio_path)
        if progress_cb:
            progress_cb(1, 1, {
                "chunk_time_sec": time.time() - t0,
                "chunk_chars": len(str(result.get("text", ""))),
            })
        return result

    _ensure_ffmpeg()
    total_dur = _ffprobe_duration(src)
    chunk_sec = chunk_minutes * 60
    n_chunks = max(1, int((total_dur + chunk_sec - 1) // chunk_sec))
    logger.info(
        "Chunked transcribe: %s (%.1fs total) → %d chunks of %ds",
        src.name, total_dur, n_chunks, chunk_sec,
    )

    if progress_cb:
        progress_cb(0, n_chunks, {})

    full_text_parts: List[str] = []
    full_segments: List[Dict[str, Any]] = []
    language: Optional[str] = None

    with tempfile.TemporaryDirectory(prefix="stt_chunks_") as tmp:
        tmp_dir = Path(tmp)
        for i in range(n_chunks):
            start = i * chunk_sec
            length = min(chunk_sec, total_dur - start)
            if length <= 0.05:
                break
            chunk_path = tmp_dir / f"chunk_{i:03d}.mp3"
            _ffmpeg_cut(src, start, length, chunk_path)

            t0 = time.time()
            try:
                chunk_result = await backend.transcribe(str(chunk_path))
            except Exception as e:
                logger.error("Chunk %d failed: %s", i, e)
                if progress_cb:
                    progress_cb(i + 1, n_chunks, {
                        "chunk_time_sec": time.time() - t0,
                        "chunk_chars": 0,
                        "chunk_audio_sec": length,
                        "chunk_failed": True,
                    })
                continue

            chunk_text = str(chunk_result.get("text", "")).strip()
            if chunk_text:
                full_text_parts.append(chunk_text)
            for seg in chunk_result.get("segments", []) or []:
                seg_text = str(seg.get("text", "")).strip()
                if not seg_text:
                    continue
                full_segments.append({
                    "text": seg_text,
                    "start": float(seg.get("start", 0.0) or 0.0) + start,
                    "end": float(seg.get("end", 0.0) or 0.0) + start,
                })
            if language is None:
                language = chunk_result.get("language") or None

            if progress_cb:
                elapsed = time.time() - t0
                progress_cb(i + 1, n_chunks, {
                    "chunk_time_sec": elapsed,
                    "chunk_chars": len(chunk_text),
                    "chunk_audio_sec": length,
                    "chunk_chars_per_sec": (len(chunk_text) / length) if length > 0 else 0.0,
                })

    return {
        "text": " ".join(full_text_parts),
        "segments": full_segments,
        "language": language or "",
    }
