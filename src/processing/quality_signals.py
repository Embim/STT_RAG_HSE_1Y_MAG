"""Quality heuristics on Whisper transcript output.

Same family of signals used by `scripts/cleanup_benchmark.py`, but here
they're computed at *ingest time* — so we can spot bad transcripts in the
MLflow run BEFORE building benchmarks. The output is a dict with metric
values + a boolean `quality_warn` flag, ready to push to MLflow.

Heuristics:

  - n_loop_segments: how many segments look like Whisper-loop artifacts
    (one word > 35% of segment, trigram repeats > 3x, "А. А. А." pattern)
  - unique_word_ratio: unique tokens / total tokens (low = repetitive speech
    or stuck loop)
  - lowercase_segment_pct: fraction of segments starting with a lowercase
    letter (Whisper sometimes loses capitalization mid-recording)
  - avg_chars_per_sec: rough speech density. <5 — many silences or music,
    >25 — very dense (or hallucinated speech)
  - language_detected: passthrough from transcript

`quality_warn=True` if any of these breach thresholds (see compute()).
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional

WORD_RE = re.compile(r"[А-Яа-яёЁA-Za-z]+")
SHORT_PHRASE_REPEAT_RE = re.compile(r"(\b\w{1,5}\.\s*){4,}")


def _is_looped_segment(text: str) -> bool:
    words = WORD_RE.findall(text.lower())
    if len(words) < 4:
        return False
    counts = Counter(words)
    most_word, most_count = counts.most_common(1)[0]
    if most_count / len(words) > 0.35 and len(words) >= 8:
        return True
    trigrams = [" ".join(words[i:i + 3]) for i in range(len(words) - 2)]
    if trigrams:
        tg_count = Counter(trigrams).most_common(1)[0][1]
        if tg_count > 3:
            return True
    if SHORT_PHRASE_REPEAT_RE.search(text):
        return True
    return False


def compute(
    *,
    text: str,
    segments: List[Dict[str, Any]],
    audio_duration_sec: float = 0.0,
    language_detected: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute quality signals from a finished transcript.

    Returns a flat dict with metric values (floats/ints) plus `quality_warn`
    bool tag. Safe on empty input — returns zeros.
    """
    n_segments = len(segments)
    if not text and not segments:
        return {
            "n_loop_segments": 0,
            "loop_segment_pct": 0.0,
            "unique_word_ratio": 0.0,
            "lowercase_segment_pct": 0.0,
            "avg_chars_per_sec": 0.0,
            "language_detected": language_detected or "",
            "quality_warn": False,
        }

    n_loops = sum(1 for s in segments if _is_looped_segment(str(s.get("text", ""))))
    loop_pct = n_loops / n_segments if n_segments else 0.0

    n_lower = sum(
        1 for s in segments
        if (str(s.get("text", "")).strip()[:1] or "Я").islower()
    )
    lower_pct = n_lower / n_segments if n_segments else 0.0

    all_words = WORD_RE.findall(text.lower())
    unique_ratio = (len(set(all_words)) / len(all_words)) if all_words else 0.0

    chars_per_sec = (len(text) / audio_duration_sec) if audio_duration_sec > 0 else 0.0

    # Composite warning flag
    warn = (
        loop_pct > 0.05            # > 5% сегментов выглядят как лупы
        or unique_ratio < 0.20     # < 20% уникальных слов
        or lower_pct > 0.30        # > 30% сегментов в lowercase
        or chars_per_sec > 25.0    # подозрительно плотно
        or chars_per_sec < 2.0     # подозрительно редко (если duration > 0)
    )

    return {
        "n_loop_segments": n_loops,
        "loop_segment_pct": round(loop_pct, 4),
        "unique_word_ratio": round(unique_ratio, 4),
        "lowercase_segment_pct": round(lower_pct, 4),
        "avg_chars_per_sec": round(chars_per_sec, 2),
        "language_detected": language_detected or "",
        "quality_warn": bool(warn),
    }
