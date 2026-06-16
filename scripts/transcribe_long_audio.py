"""Транскрибировать длинное аудио через любой /v1/audio/transcriptions backend.

Чанкует входное аудио ffmpeg-ом на 5-минутные куски, отправляет последовательно
на ASR-сервер, склеивает результат. Сохраняет в формате совместимом с
data/transcripts/*.json (text, segments, language, asr_name).

Использование:
    python scripts/transcribe_long_audio.py \\
        --input "E:/video_pipeline/audio/lecture.mp4" \\
        --output "data/transcripts/cnn_qwen3.json" \\
        --asr-name qwen3-asr-1.7b \\
        --asr-url http://localhost:8000 \\
        --chunk-minutes 5
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _ffmpeg_split(src: Path, dst_dir: Path, chunk_seconds: int) -> list[Path]:
    """Раскидать аудио на куски через ffmpeg segment-mux (без перекодирования
    если возможно — извлекаем аудио в wav 16k mono чтобы было совместимо
    с любым ASR backend)."""
    out_pattern = str(dst_dir / "chunk_%03d.wav")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-vn",                       # no video
        "-ac", "1",                  # mono
        "-ar", "16000",              # 16 kHz (стандарт для ASR)
        "-f", "segment",
        "-segment_time", str(chunk_seconds),
        "-c:a", "pcm_s16le",         # uncompressed wav
        out_pattern,
    ]
    logger.info("ffmpeg split: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    chunks = sorted(dst_dir.glob("chunk_*.wav"))
    logger.info("Split into %d chunks", len(chunks))
    return chunks


def _transcribe_one(
    chunk: Path, *, url: str, model: str, language: str, timeout: int = 600,
    proxies: dict | None = None, response_format: str = "verbose_json",
) -> dict:
    """POST /v1/audio/transcriptions.

    Сначала пробуем verbose_json (даёт segments с тайм-кодами).
    Если сервер отвечает 400 ("not supported") — фолбэчим на json
    (только text — Qwen3-ASR vLLM 0.20.2 так умеет).
    """
    def _post(fmt: str):
        with open(chunk, "rb") as f:
            files = {"file": (chunk.name, f, "audio/wav")}
            data = {"model": model, "language": language, "response_format": fmt}
            return requests.post(
                f"{url.rstrip('/')}/v1/audio/transcriptions",
                files=files, data=data, timeout=timeout, proxies=proxies or {},
            )

    r = _post(response_format)
    if r.status_code == 400 and "verbose_json" in r.text and response_format == "verbose_json":
        logger.debug("Server doesn't support verbose_json, falling back to json")
        r = _post("json")
    r.raise_for_status()
    return r.json()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True,
                   help="Путь к audio/video файлу (любой формат, ffmpeg извлечёт аудио)")
    p.add_argument("--output", type=Path, required=True,
                   help="Куда писать JSON-транскрипт")
    p.add_argument("--asr-name", required=True,
                   help="Метка ASR-системы (попадает в JSON-поле asr_name)")
    p.add_argument("--asr-url", default="http://127.0.0.1:8000",
                   help="Base URL ASR-сервера (127.0.0.1 чтобы обойти прокси)")
    p.add_argument("--model", default=None,
                   help="model param в request (autodetect через /v1/models)")
    p.add_argument("--language", default="ru")
    p.add_argument("--chunk-minutes", type=int, default=5,
                   help="Длина одного чанка в минутах (default 5)")
    p.add_argument("--keep-chunks", action="store_true",
                   help="Не удалять временные wav-чанки после прогона")
    p.add_argument("--max-chunks", type=int, default=None,
                   help="Транскрибировать только первые N чанков (pilot mode)")
    args = p.parse_args()

    if not args.input.exists():
        logger.error("Input not found: %s", args.input)
        return 2

    # Autodetect model name через /v1/models
    if not args.model:
        try:
            resp = requests.get(f"{args.asr_url}/v1/models", timeout=10, proxies={})
            args.model = resp.json()["data"][0]["id"]
            logger.info("Autodetected model: %s", args.model)
        except Exception as e:
            logger.error("Не удалось получить /v1/models: %s. Передай --model явно.", e)
            return 3

    tmp_dir = Path(tempfile.mkdtemp(prefix="asr_chunks_"))
    logger.info("Chunks dir: %s", tmp_dir)

    try:
        chunks = _ffmpeg_split(
            args.input, tmp_dir, chunk_seconds=args.chunk_minutes * 60
        )
        if not chunks:
            logger.error("ffmpeg вернул 0 чанков")
            return 4
        if args.max_chunks is not None and args.max_chunks > 0:
            chunks = chunks[:args.max_chunks]
            logger.info("Pilot mode: первые %d чанков", len(chunks))

        all_segments: list[dict] = []
        all_text_parts: list[str] = []
        total_duration = 0.0
        started = time.monotonic()

        for i, chunk in enumerate(chunks):
            t0 = time.monotonic()
            logger.info(
                "[%d/%d] Transcribing %s (%.1f MB)...",
                i + 1, len(chunks), chunk.name,
                chunk.stat().st_size / 1024 / 1024,
            )
            try:
                result = _transcribe_one(
                    chunk,
                    url=args.asr_url,
                    model=args.model,
                    language=args.language,
                )
            except Exception as e:
                logger.error("[%d/%d] FAILED: %s", i + 1, len(chunks), e)
                continue
            chunk_text = (result.get("text") or "").strip()
            chunk_segs = result.get("segments") or []
            chunk_dur = float(result.get("duration") or args.chunk_minutes * 60)

            # сдвинуть таймкоды сегментов по абсолютному оффсету
            offset = total_duration
            for s in chunk_segs:
                if "start" in s:
                    s["start"] = float(s["start"]) + offset
                if "end" in s:
                    s["end"] = float(s["end"]) + offset
                s["chunk_index"] = i

            all_segments.extend(chunk_segs)
            if chunk_text:
                all_text_parts.append(chunk_text)
            total_duration += chunk_dur

            elapsed = time.monotonic() - t0
            cumulative = time.monotonic() - started
            eta = (cumulative / (i + 1)) * (len(chunks) - i - 1)
            logger.info(
                "[%d/%d] %d chars in %.1fs (cum %.1fs, ETA ~%.0fs)",
                i + 1, len(chunks), len(chunk_text), elapsed, cumulative, eta,
            )

        combined_text = " ".join(all_text_parts).strip()
        out_obj = {
            "asr_name": args.asr_name,
            "asr_model_id": args.model,
            "source_audio": str(args.input),
            "language": args.language,
            "duration": total_duration,
            "text": combined_text,
            "segments": all_segments,
            "n_chunks": len(chunks),
            "chunk_minutes": args.chunk_minutes,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(out_obj, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        wall = time.monotonic() - started
        logger.info(
            "DONE: %d chars, %d segments, %.1f min audio, %.1fs wallclock (%.2fx RT)",
            len(combined_text), len(all_segments), total_duration / 60,
            wall, total_duration / wall,
        )
        logger.info("Wrote: %s", args.output)
        return 0
    finally:
        if not args.keep_chunks:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
