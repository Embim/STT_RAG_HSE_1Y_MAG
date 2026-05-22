"""Сборщик локального ASR-бенчмарка из своих лекций.

Скрипт нужен один раз — чтобы подготовить пары (аудио, эталонный текст), на
которых потом ASR-модели будут гоняться через `evaluation.asr.runner`. Лежит
вне пакета `src/evaluation`, потому что это setup-утилита, а не часть runtime.

Результат работы — папка с `manifest.json` и подпапкой `audio/` внутри. На
эту папку дальше указываешь runner'у:

    python -m evaluation.asr.runner --local-benchmark <output>/manifest.json

== Режим 1: авто-нарезка по сегментам JSON-транскрипта ==

Если у тебя уже есть JSON-транскрипт лекции с сегментами и таймкодами (формат
faster-whisper-server: {"segments": [{"text", "start", "end"}, ...]}) — `ffmpeg`
сам нарежет исходное видео/аудио по сегментам, а тексты из JSON станут
reference'ами. Никакой ручной разметки не нужно, кроме опциональной правки
manifest'а перед прогоном (если в JSON есть галлюцинации старой ASR — открой
`manifest.json`, поправь подозрительные `reference`, поставь `verified: true`
на тех, что проверил, и запускай runner с `--only-verified`).

    python scripts/build_asr_benchmark.py from-json \
        --json E:/video_pipeline/transcripts/lecture_01.json \
        --audio E:/video_pipeline/audio/lecture_01.mp3 \
        --output data/eval/benchmarks/local/lecture_01 \
        --max-segments 30

Опции `--min-duration`, `--max-duration` (секунды) и `--min-chars` отсекают
слишком короткие/длинные сегменты. `--max-segments N` ограничивает размер
бенчмарка первыми N подходящими сегментами.

== Режим 2: уже порезанная папка + CSV с подписями ==

Если ты сам нарезал аудио (например, в Audacity на естественных границах
фраз) и подписал каждый кусок — собери CSV с двумя колонками `filename` и
`reference`, и подложи папку с самими аудиофайлами. Все элементы манифеста
автоматически получают `verified: true`.

    python scripts/build_asr_benchmark.py from-folder \
        --audio-dir /path/to/cut/audio \
        --references-csv /path/to/refs.csv \
        --output data/eval/benchmarks/local/manual_v1

== Требования ==

Для режима `from-json` нужен `ffmpeg` в PATH. Режим `from-folder` ничего
внешнего не требует. На вход принимаются mp3, wav, m4a, flac, ogg, webm,
mp4, mov, mkv — всё, что умеет ffmpeg.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm", ".mp4", ".mov", ".mkv"}
_SLUG_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _slug(s: str) -> str:
    return _SLUG_RE.sub("_", s)[:40]


def _ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH; install it to use from-json mode")


def _resolve_audio(audio_dir: Path, fname: str) -> Path | None:
    """Find an audio file by name. If `fname` has an extension and exists, use it.
    Otherwise glob for `<fname>.*` and pick the first audio file we know of.
    """
    direct = (audio_dir / fname).resolve()
    if direct.exists() and direct.suffix.lower() in _AUDIO_EXTS:
        return direct
    for ext in _AUDIO_EXTS:
        candidate = (audio_dir / f"{fname}{ext}").resolve()
        if candidate.exists():
            return candidate
    # Last-resort glob (e.g. fname might already include a partial path)
    for cand in sorted(audio_dir.glob(f"{fname}.*")):
        if cand.suffix.lower() in _AUDIO_EXTS:
            return cand.resolve()
    return None


def _ffmpeg_cut(src: Path, start: float, end: float, dst: Path, sample_rate: int = 16000) -> None:
    duration = max(0.0, end - start)
    if duration <= 0.05:
        raise ValueError(f"degenerate segment {start}->{end}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start:.3f}",
        "-i", str(src),
        "-t", f"{duration:.3f}",
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-acodec", "pcm_s16le",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def _merge_segments(
    segments: List[Dict[str, Any]],
    *,
    target_duration: float,
    max_duration: float,
) -> List[Dict[str, Any]]:
    """Склеивает соседние Whisper-сегменты в более длинные блоки.

    Логика: набираем сегменты в текущий блок, пока он не достигнет
    `target_duration`. Если последний добавленный сегмент кончается на
    знак препинания — закрываем блок (мысль завершена). Если перевалил за
    `max_duration` — закрываем принудительно. Между блоками gap >2 сек тоже
    форсит закрытие.
    """
    out: List[Dict[str, Any]] = []
    cur_text: List[str] = []
    cur_start: Optional[float] = None
    cur_end: Optional[float] = None
    prev_end: Optional[float] = None

    def flush() -> None:
        nonlocal cur_text, cur_start, cur_end
        if cur_text and cur_start is not None and cur_end is not None:
            out.append({
                "text": " ".join(cur_text).strip(),
                "start": cur_start,
                "end": cur_end,
            })
        cur_text = []
        cur_start = None
        cur_end = None

    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        start = float(seg.get("start", 0.0) or 0.0)
        end = float(seg.get("end", start) or start)

        # gap > 2 сек = разрыв в речи, закрываем блок
        if prev_end is not None and (start - prev_end) > 2.0:
            flush()

        if cur_start is None:
            cur_start = start
        cur_text.append(text)
        cur_end = end

        block_dur = cur_end - cur_start
        ends_clean = text.rstrip()[-1:] in ".!?…"
        if (block_dur >= target_duration and ends_clean) or block_dur >= max_duration:
            flush()
        prev_end = end

    flush()
    return out


def build_from_segmented_json(
    json_path: Path,
    audio_path: Path,
    output_dir: Path,
    *,
    min_duration: float = 2.0,
    max_duration: float = 25.0,
    min_chars: int = 20,
    max_segments: Optional[int] = None,
    reference_source: Optional[str] = None,
    merge_target_duration: Optional[float] = None,
) -> Path:
    _ensure_ffmpeg()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    segments = payload.get("segments") or []
    if not segments:
        raise SystemExit(f"No segments in {json_path}")
    # If the JSON itself declares which ASR produced it, use that as the
    # reference_source unless the caller overrode it via CLI.
    ref_source = reference_source or payload.get("asr_name") or "unknown"

    # Если задан merge_target_duration, склеиваем мелкие сегменты в осмысленные
    # блоки заданной длительности.
    if merge_target_duration is not None:
        segments = _merge_segments(
            segments,
            target_duration=merge_target_duration,
            max_duration=max_duration,
        )

    output_dir = output_dir.resolve()
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    items: List[Dict[str, Any]] = []
    source_id = _slug(json_path.stem)
    for idx, seg in enumerate(segments):
        text = str(seg.get("text", "")).strip()
        start = float(seg.get("start", 0.0) or 0.0)
        end = float(seg.get("end", start) or start)
        duration = end - start
        if not text or len(text) < min_chars:
            continue
        if duration < min_duration or duration > max_duration:
            continue
        seg_id = f"{source_id}_seg_{idx:05d}"
        wav_path = audio_dir / f"{seg_id}.wav"
        try:
            _ffmpeg_cut(audio_path, start, end, wav_path)
        except subprocess.CalledProcessError as e:
            logger.warning("ffmpeg failed on %s [%.2f-%.2f]: %s", seg_id, start, end, e)
            continue
        items.append({
            "id": seg_id,
            "audio_path": wav_path.relative_to(output_dir).as_posix(),
            "reference": text,
            "duration": round(duration, 3),
            "verified": False,
        })
        if max_segments is not None and len(items) >= max_segments:
            break

    manifest = {
        "source": json_path.stem,
        "audio_source": str(audio_path),
        "reference_source": ref_source,
        "size": len(items),
        "items": items,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        "Wrote manifest %s with %d items (reference_source=%s)",
        manifest_path, len(items), ref_source,
    )
    return manifest_path


def build_from_folder(
    audio_dir: Path,
    references_csv: Path,
    output_dir: Path,
    *,
    reference_source: Optional[str] = None,
) -> Path:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    items: List[Dict[str, Any]] = []
    with open(references_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = (row.get("filename") or "").strip()
            ref = (row.get("reference") or "").strip()
            if not fname or not ref:
                continue
            audio_path = _resolve_audio(audio_dir, fname)
            if audio_path is None:
                logger.warning("Skip missing audio for %r in %s", fname, audio_dir)
                continue
            items.append({
                "id": _slug(audio_path.stem),
                "audio_path": str(audio_path),
                "reference": ref,
                "duration": 0.0,
                "verified": True,
            })

    manifest = {
        "source": str(audio_dir),
        "audio_source": str(audio_dir),
        "reference_source": reference_source or "manual",
        "size": len(items),
        "items": items,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        "Wrote manifest %s with %d items (reference_source=%s)",
        manifest_path, len(items), manifest["reference_source"],
    )
    return manifest_path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scripts/build_asr_benchmark.py")
    sub = p.add_subparsers(dest="mode", required=True)

    j = sub.add_parser("from-json", help="ffmpeg-cut audio by segments[] in a JSON transcript")
    j.add_argument("--json", required=True)
    j.add_argument("--audio", required=True)
    j.add_argument("--output", required=True)
    j.add_argument("--min-duration", type=float, default=2.0)
    j.add_argument("--max-duration", type=float, default=25.0)
    j.add_argument("--min-chars", type=int, default=20)
    j.add_argument("--max-segments", type=int, default=None)
    j.add_argument("--merge-target-duration", type=float, default=None,
                   help="Если задано, склеиваем подряд идущие Whisper-сегменты в блоки "
                        "указанной длительности (секунд). Хорошее значение 12-18 сек: "
                        "получаются осмысленные фразы, а не обрывки по 2-5 секунд.")
    j.add_argument("--reference-source", default=None,
                   help="Override which ASR produced these references "
                        "(by default read from JSON's 'asr_name' field). "
                        "Used by the runner to detect self-evaluation leaks.")

    f = sub.add_parser("from-folder", help="Bundle pre-cut audio + references CSV")
    f.add_argument("--audio-dir", required=True)
    f.add_argument("--references-csv", required=True)
    f.add_argument("--output", required=True)
    f.add_argument("--reference-source", default="manual",
                   help="Label for who produced the references (default: 'manual')")
    return p


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_parser().parse_args()
    if args.mode == "from-json":
        build_from_segmented_json(
            json_path=Path(args.json),
            audio_path=Path(args.audio),
            output_dir=Path(args.output),
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            min_chars=args.min_chars,
            max_segments=args.max_segments,
            reference_source=args.reference_source,
            merge_target_duration=args.merge_target_duration,
        )
    elif args.mode == "from-folder":
        build_from_folder(
            audio_dir=Path(args.audio_dir),
            references_csv=Path(args.references_csv),
            output_dir=Path(args.output),
            reference_source=args.reference_source,
        )


if __name__ == "__main__":
    main()
