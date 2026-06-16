"""Iterate samples from a local ASR benchmark manifest.

The manifest is produced by the one-off helper at `scripts/build_asr_benchmark.py`
(either by ffmpeg-cutting a video by JSON segments or by bundling a pre-cut
folder). Shape:

    {
        "source": "<lecture name>",
        "items": [
            {
                "id": "lec01_seg_0042",
                "audio_path": "audio/lec01_seg_0042.wav",
                "reference": "Полностью прописанный текст этого фрагмента.",
                "duration": 9.7,
                "verified": false
            },
            ...
        ]
    }

Same yield shape as `evaluation.asr.benchmarks.iter_benchmark` so the runner
treats local and HuggingFace benchmarks uniformly.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


def iter_local_benchmark(
    manifest_path: Path,
    *,
    max_samples: Optional[int] = None,
    only_verified: bool = False,
) -> Iterator[dict]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = payload.get("items", payload if isinstance(payload, list) else [])
    if only_verified:
        items = [it for it in items if it.get("verified")]
    logger.info(
        "Local benchmark %s: %d items (only_verified=%s)",
        manifest_path, len(items), only_verified,
    )
    base_dir = manifest_path.parent
    for idx, item in enumerate(items):
        if max_samples is not None and idx >= max_samples:
            break
        audio_rel = item.get("audio_path", "")
        if not audio_rel:
            continue
        audio_path = Path(audio_rel)
        if not audio_path.is_absolute():
            audio_path = (base_dir / audio_rel).resolve()
        if not audio_path.exists():
            logger.warning("Audio missing, skip: %s", audio_path)
            continue
        ref = str(item.get("reference", "")).strip()
        if not ref:
            continue
        yield {
            "id": str(item.get("id", f"sample_{idx:05d}")),
            "audio_path": str(audio_path),
            "reference": ref,
            "duration": float(item.get("duration") or 0.0),
        }
