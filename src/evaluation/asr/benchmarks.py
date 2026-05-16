"""Public ASR benchmark loaders backed by HuggingFace `datasets`.

We materialize each sample to a file on disk so that any ASR backend (HTTP or
in-process) can read it the same way. Files are cached under
`data/eval/benchmarks/{benchmark}/{lang}/`.

Yielded sample shape:
    {
        "id": str,           # stable id
        "audio_path": str,   # absolute path to .wav
        "reference": str,    # human-verified transcript
        "duration": float,   # seconds (best-effort)
    }
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional

from settings import settings
from evaluation.paths import BENCHMARK_DIR, ensure_dirs

logger = logging.getLogger(__name__)


def _slug(name: str) -> str:
    return name.replace("/", "__")


def _write_wav(path: Path, array, sample_rate: int) -> None:
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), array, sample_rate)


def iter_benchmark(
    name: Optional[str] = None,
    lang: Optional[str] = None,
    split: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Iterator[dict]:
    """Stream samples from a HuggingFace ASR dataset.

    Defaults come from settings (Common Voice ru test). For FLEURS pass
    name="google/fleurs", lang="ru_ru".
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "`datasets` is not installed. Run: uv sync --extra eval"
        ) from e

    ensure_dirs()
    name = name or settings.ASR_BENCHMARK
    lang = lang or settings.ASR_BENCHMARK_LANG
    split = split or settings.ASR_BENCHMARK_SPLIT

    cache_dir = BENCHMARK_DIR / _slug(name) / lang
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading benchmark %s lang=%s split=%s", name, lang, split)
    ds = load_dataset(name, lang, split=split, streaming=True)

    text_field_candidates = ("sentence", "text", "transcription", "raw_transcription")

    for idx, item in enumerate(ds):
        if max_samples is not None and idx >= max_samples:
            break
        audio = item.get("audio") or {}
        array = audio.get("array")
        sample_rate = int(audio.get("sampling_rate") or 16000)
        if array is None or len(array) == 0:
            continue
        ref = ""
        for f in text_field_candidates:
            if item.get(f):
                ref = str(item[f])
                break
        if not ref:
            continue

        sample_id = str(item.get("client_id") or item.get("id") or f"sample_{idx:05d}")
        # HF audio paths can collide across shards; suffix with idx
        audio_path = cache_dir / f"{idx:05d}_{_safe_filename(sample_id)}.wav"
        if not audio_path.exists():
            _write_wav(audio_path, array, sample_rate)

        yield {
            "id": sample_id,
            "audio_path": str(audio_path),
            "reference": ref.strip(),
            "duration": float(len(array)) / float(sample_rate),
        }


_FNAME_RE = None


def _safe_filename(s: str) -> str:
    global _FNAME_RE
    if _FNAME_RE is None:
        import re
        _FNAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
    return _FNAME_RE.sub("_", s)[:80]
