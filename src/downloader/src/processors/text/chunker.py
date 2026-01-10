"""
Text chunking processor.

Разбивает текстовые сегменты на чанки оптимального размера для эмбеддинга.
"""

from typing import Dict, Any, List
import logging

from ...core.interfaces import BaseChunker
from ...core.models import TextSegment, TextChunk
from ...core.registry import register_chunker

logger = logging.getLogger(__name__)


@register_chunker
class TextChunker(BaseChunker):
    """
    Стандартный чанкер для разбиения текста.

    Объединяет короткие сегменты и разбивает длинные,
    чтобы получить чанки оптимального размера для эмбеддинга.
    """

    chunker_name = "text"

    def __init__(
        self,
        min_length: int = 50,
        max_length: int = 2000,
        merge_short: bool = True,
    ):
        """
        Инициализация чанкера.

        Args:
            min_length: Минимальная длина чанка в символах
            max_length: Максимальная длина чанка в символах
            merge_short: Объединять короткие сегменты
        """
        self.min_length = min_length
        self.max_length = max_length
        self.merge_short = merge_short

    def chunk(
        self,
        segments: List[TextSegment],
        min_length: int = None,
        max_length: int = None,
    ) -> List[TextChunk]:
        """
        Разбивает сегменты на чанки.

        Args:
            segments: Список сегментов для разбиения
            min_length: Минимальная длина (переопределяет настройку)
            max_length: Максимальная длина (переопределяет настройку)

        Returns:
            Список чанков
        """
        if not segments:
            return []

        min_len = min_length or self.min_length
        max_len = max_length or self.max_length

        # Фильтруем пустые сегменты заранее
        non_empty_segments = [s for s in segments if s.text.strip()]
        if not non_empty_segments:
            logger.warning("All segments are empty after filtering")
            return []

        chunks = []
        current_text = ""
        current_start = non_empty_segments[0].start
        current_end = non_empty_segments[0].end
        current_segment_ids = []

        for idx, segment in enumerate(non_empty_segments):
            segment_text = segment.text.strip()

            # Проверяем, не превысит ли добавление сегмента максимальную длину
            potential_length = len(current_text) + len(segment_text) + 1  # +1 для пробела

            if potential_length > max_len and current_text:
                # Создаём чанк из накопленного текста
                if len(current_text) >= min_len:
                    chunk = TextChunk(
                        chunk_id=len(chunks),
                        text=current_text.strip(),
                        start_position=current_start,
                        end_position=current_end,
                        segment_ids=current_segment_ids.copy(),
                    )
                    chunks.append(chunk)

                # Начинаем новый чанк
                current_text = segment_text
                current_start = segment.start
                current_end = segment.end
                current_segment_ids = [segment.id]
            else:
                # Добавляем сегмент к текущему чанку
                if current_text:
                    current_text += " " + segment_text
                else:
                    current_text = segment_text
                current_end = segment.end
                current_segment_ids.append(segment.id)

        # Добавляем последний чанк
        if current_text and len(current_text) >= min_len and current_segment_ids:
            chunk = TextChunk(
                chunk_id=len(chunks),
                text=current_text.strip(),
                start_position=current_start,
                end_position=current_end,
                segment_ids=current_segment_ids,
            )
            chunks.append(chunk)

        logger.debug(f"Created {len(chunks)} chunks from {len(non_empty_segments)} non-empty segments")
        return chunks


@register_chunker
class SentenceChunker(BaseChunker):
    """
    Чанкер на основе предложений.

    Разбивает текст по предложениям, группируя их до достижения
    максимального размера.
    """

    chunker_name = "sentence"

    def __init__(
        self,
        min_length: int = 100,
        max_length: int = 1500,
    ):
        self.min_length = min_length
        self.max_length = max_length

    def chunk(
        self,
        segments: List[TextSegment],
        min_length: int = None,
        max_length: int = None,
    ) -> List[TextChunk]:
        """Разбивает на чанки по предложениям."""
        if not segments:
            return []

        min_len = min_length or self.min_length
        max_len = max_length or self.max_length

        # Объединяем все сегменты в один текст
        full_text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())

        # Разбиваем на предложения
        sentences = self._split_sentences(full_text)

        chunks = []
        current_text = ""
        current_start = segments[0].start if segments else 0.0

        # Рассчитываем примерную скорость текста (символов в секунду)
        total_duration = segments[-1].end - segments[0].start if segments else 1.0
        chars_per_second = len(full_text) / max(total_duration, 1.0)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            potential_length = len(current_text) + len(sentence) + 1

            if potential_length > max_len and current_text:
                # Создаём чанк
                if len(current_text) >= min_len:
                    # Примерно оцениваем end_position
                    duration = len(current_text) / max(chars_per_second, 1.0)
                    chunk = TextChunk(
                        chunk_id=len(chunks),
                        text=current_text.strip(),
                        start_position=current_start,
                        end_position=current_start + duration,
                        segment_ids=[],
                    )
                    chunks.append(chunk)
                    current_start = current_start + duration

                current_text = sentence
            else:
                if current_text:
                    current_text += " " + sentence
                else:
                    current_text = sentence

        # Последний чанк
        if current_text and len(current_text) >= min_len:
            chunk = TextChunk(
                chunk_id=len(chunks),
                text=current_text.strip(),
                start_position=current_start,
                end_position=segments[-1].end if segments else current_start,
                segment_ids=[],
            )
            chunks.append(chunk)

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Разбивает текст на предложения."""
        import re

        # Простое разбиение по точкам, вопросительным и восклицательным знакам
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


@register_chunker
class FixedSizeChunker(BaseChunker):
    """
    Чанкер с фиксированным размером.

    Разбивает текст на чанки фиксированного размера с перекрытием.
    """

    chunker_name = "fixed_size"

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 200,
    ):
        """
        Args:
            chunk_size: Размер чанка в символах
            overlap: Размер перекрытия между чанками
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(
        self,
        segments: List[TextSegment],
        min_length: int = None,
        max_length: int = None,
    ) -> List[TextChunk]:
        """Разбивает на чанки фиксированного размера."""
        if not segments:
            return []

        chunk_size = max_length or self.chunk_size
        overlap = min(self.overlap, chunk_size // 2)

        # Объединяем все сегменты
        full_text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())

        if not full_text:
            return []

        chunks = []
        start = 0
        text_length = len(full_text)

        # Рассчитываем скорость текста
        total_duration = segments[-1].end - segments[0].start if segments else 1.0
        chars_per_second = text_length / max(total_duration, 1.0)
        base_start = segments[0].start if segments else 0.0

        while start < text_length:
            end = min(start + chunk_size, text_length)

            # Пытаемся найти границу слова
            if end < text_length:
                space_pos = full_text.rfind(" ", start, end)
                if space_pos > start:
                    end = space_pos

            chunk_text = full_text[start:end].strip()

            if chunk_text:
                # Оцениваем позиции по времени
                time_start = base_start + (start / max(chars_per_second, 1.0))
                time_end = base_start + (end / max(chars_per_second, 1.0))

                chunk = TextChunk(
                    chunk_id=len(chunks),
                    text=chunk_text,
                    start_position=time_start,
                    end_position=time_end,
                    segment_ids=[],
                )
                chunks.append(chunk)

            start = end - overlap
            if start <= 0 or end >= text_length:
                break

        return chunks
