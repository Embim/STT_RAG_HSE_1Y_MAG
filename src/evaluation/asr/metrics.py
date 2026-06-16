"""WER/CER/MER/WIL metrics on normalized text via jiwer."""
from __future__ import annotations

from typing import Dict, Sequence

try:
    import jiwer
except ImportError:  # pragma: no cover — eval extra not installed
    jiwer = None

from evaluation.asr.text_norm import normalize_ru


def _ensure_jiwer() -> None:
    if jiwer is None:
        raise RuntimeError("jiwer is not installed. Run: uv sync --extra eval")


def compute_pair(reference: str, hypothesis: str) -> Dict[str, float]:
    """Per-sample WER/CER/MER/WIL on normalized text."""
    _ensure_jiwer()
    ref = normalize_ru(reference)
    hyp = normalize_ru(hypothesis)
    if not ref:
        return {"wer": float("nan"), "cer": float("nan"), "mer": float("nan"), "wil": float("nan")}
    return {
        "wer": float(jiwer.wer(ref, hyp)),
        "cer": float(jiwer.cer(ref, hyp)),
        "mer": float(jiwer.mer(ref, hyp)),
        "wil": float(jiwer.wil(ref, hyp)),
    }


def compute_corpus(
    references: Sequence[str], hypotheses: Sequence[str]
) -> Dict[str, float]:
    """Aggregate WER/CER over a corpus.

    jiwer aggregates by total edits / total ref length (the conventional
    reporting style for ASR) — not the per-sample mean.
    """
    _ensure_jiwer()
    if len(references) != len(hypotheses):
        raise ValueError(
            f"refs/hyps length mismatch: {len(references)} vs {len(hypotheses)}"
        )
    refs = [normalize_ru(r) for r in references]
    hyps = [normalize_ru(h) for h in hypotheses]
    pairs = [(r, h) for r, h in zip(refs, hyps) if r]
    if not pairs:
        return {"wer": float("nan"), "cer": float("nan"), "mer": float("nan"), "wil": float("nan"), "n": 0}
    refs_f, hyps_f = zip(*pairs)
    return {
        "wer": float(jiwer.wer(list(refs_f), list(hyps_f))),
        "cer": float(jiwer.cer(list(refs_f), list(hyps_f))),
        "mer": float(jiwer.mer(list(refs_f), list(hyps_f))),
        "wil": float(jiwer.wil(list(refs_f), list(hyps_f))),
        "n": len(pairs),
    }
