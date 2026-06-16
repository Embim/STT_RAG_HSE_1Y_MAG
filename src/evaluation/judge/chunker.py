"""Разбиение длинных транскрипций на чанки, влезающие в окно судьи.

Длинная лекция (~136K chars) не помещается в 16K-токенный контекст Qwen3.
Режем на ~6K-char чанки, СТАРАЯСЬ резать по границам предложений (между
точкой/?/! и пробелом), иначе судья получит обрезанную фразу и пометит её
как `gap` — а это артефакт чанкинга, не ASR-ошибка.

С overlap'ом не работаем (в отличие от RAG ingest): в каждом чанке judge
ищет ошибки независимо, дубликат findings на стыках — реальная проблема,
overlap эту проблему создал бы вместо того чтобы её решить.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable, List

logger = logging.getLogger(__name__)

# Граница предложения: точка/?/! + пробельный символ (включая \n). Достаточно
# хорошо для русского. Для аббревиатур типа "т.е." это даст ложный split,
# но это нестрашно — judge всё равно справится с обрывком в начале чанка.
_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class TextChunk:
    """Один чанк — то, что отправляется в один LLM-вызов."""
    chunk_id: str        # стабильный id вида "<source_id>_chunk_<n>"
    text: str            # чистый текст для судьи
    char_start: int      # офсет в исходном тексте (для дебага и привязки)
    char_end: int


def chunk_by_sentences(
    text: str,
    *,
    source_id: str,
    max_chars: int = 6000,
) -> List[TextChunk]:
    """Разрезать текст на чанки по границам предложений.

    Алгоритм:
      1. Сплитим текст по `_SENT_BOUNDARY` на предложения.
      2. Жадно набираем предложения в текущий чанк, пока не превысим max_chars.
      3. Если ОДНО предложение длиннее max_chars (бывает в плохих транскриптах
         без пунктуации), режем его руками по словам.

    Args:
        text: исходный текст (полная транскрипция или сегмент).
        source_id: префикс для chunk_id, чтобы при сборе findings из нескольких
            источников было видно откуда что (например "lecture_cnn", "csv_row42").
        max_chars: верхняя граница длины чанка. По дефолту совпадает с
            settings.JUDGE_MAX_INPUT_CHARS, но передаём явно ради тестов.
    """
    if not text.strip():
        return []

    sentences = _SENT_BOUNDARY.split(text)
    # Восстановим терминаторы. _SENT_BOUNDARY режет ПОСЛЕ знака, поэтому
    # каждое предложение кроме последнего теряет пробел — добавим обратно.
    chunks: List[TextChunk] = []
    buf: List[str] = []
    buf_len = 0
    chunk_start = 0
    cursor = 0
    n = 0

    def flush() -> None:
        nonlocal buf, buf_len, chunk_start, n
        if not buf:
            return
        chunk_text = " ".join(buf).strip()
        if chunk_text:
            chunks.append(TextChunk(
                chunk_id=f"{source_id}_chunk_{n:03d}",
                text=chunk_text,
                char_start=chunk_start,
                char_end=chunk_start + len(chunk_text),
            ))
            n += 1
        buf = []
        buf_len = 0

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        slen = len(sent) + 1  # +1 за разделяющий пробел

        # Случай 1: одно предложение длиннее лимита — режем по словам.
        if slen > max_chars:
            flush()
            chunk_start = cursor
            words = sent.split()
            word_buf: List[str] = []
            word_len = 0
            for w in words:
                if word_len + len(w) + 1 > max_chars and word_buf:
                    buf = word_buf
                    buf_len = word_len
                    flush()
                    chunk_start = cursor
                    word_buf = [w]
                    word_len = len(w)
                else:
                    word_buf.append(w)
                    word_len += len(w) + 1
            if word_buf:
                buf = word_buf
                buf_len = word_len
                flush()
            cursor += len(sent) + 1
            chunk_start = cursor
            continue

        # Случай 2: предложение влезает само, но добавляется в текущий чанк только
        # если не превышает лимит вместе с уже набранным.
        if buf_len + slen > max_chars and buf:
            flush()
            chunk_start = cursor
        buf.append(sent)
        buf_len += slen
        cursor += slen

    flush()
    logger.info(
        "Chunked %s: %d chars → %d chunks (max=%d)",
        source_id, len(text), len(chunks), max_chars,
    )
    return chunks


def total_chars(chunks: Iterable[TextChunk]) -> int:
    return sum(len(c.text) for c in chunks)
