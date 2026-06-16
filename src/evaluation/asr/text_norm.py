"""Russian-aware text normalization used before WER/CER comparison."""
from __future__ import annotations

import re
import unicodedata


_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_MULTISPACE_RE = re.compile(r"\s+")
_FILLER_TOKENS = {
    "<eot>", "<eos>", "<pad>", "<unk>", "<s>", "</s>",
    "[bos]", "[eos]",
}


def normalize_ru(text: str) -> str:
    """Normalize Russian transcription text for fair WER/CER comparison.

    - Unicode NFKC
    - lowercase
    - ё → е (Whisper inconsistently emits both forms)
    - drop ASR special tokens
    - drop punctuation
    - collapse whitespace
    """
    if text is None:
        return ""
    s = unicodedata.normalize("NFKC", text).lower()
    for tok in _FILLER_TOKENS:
        s = s.replace(tok, " ")
    s = s.replace("ё", "е")
    s = _PUNCT_RE.sub(" ", s)
    s = _MULTISPACE_RE.sub(" ", s).strip()
    return s
