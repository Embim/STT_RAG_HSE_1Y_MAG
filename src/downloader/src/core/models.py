"""
Data models for the pipeline.

Универсальные модели данных, используемые всеми компонентами системы.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
from pathlib import Path
import uuid
from datetime import datetime


class ContentType(Enum):
    """Типы контента, поддерживаемые системой."""
    AUDIO = "audio"
    VIDEO = "video"
    IMAGE = "image"
    PDF = "pdf"
    DOCX = "docx"
    TEXT = "text"
    HTML = "html"


class SourceType(Enum):
    """Типы источников данных."""
    YOUTUBE = "youtube"
    RUTUBE = "rutube"
    LOCAL_FILE = "local_file"
    TELEGRAM = "telegram"
    RSS = "rss"
    WEB = "web"


class ProcessingStage(Enum):
    """Этапы обработки контента."""
    PENDING = "pending"
    DOWNLOADED = "downloaded"
    PROCESSED = "processed"
    CHUNKED = "chunked"
    EMBEDDED = "embedded"
    STORED = "stored"
    FAILED = "failed"


@dataclass
class TextSegment:
    """
    Сегмент текста с информацией о позиции.

    Для аудио/видео: start/end - время в секундах
    Для документов: start/end - номер страницы или параграфа
    """
    id: int
    text: str
    start: float = 0.0
    end: float = 0.0
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.text = self.text.strip()


@dataclass
class TextChunk:
    """
    Чанк текста, готовый для эмбеддинга.

    Объединяет несколько сегментов в оптимальный для эмбеддинга размер.
    """
    chunk_id: int
    text: str
    start_position: float
    end_position: float
    segment_ids: List[int] = field(default_factory=list)
    embedding: Optional[List[float]] = None

    def __post_init__(self):
        self.text = self.text.strip()

    @property
    def has_embedding(self) -> bool:
        """Проверяет, есть ли эмбеддинг у чанка."""
        return self.embedding is not None and len(self.embedding) > 0


@dataclass
class ContentItem:
    """
    Универсальный контейнер для любого контента.

    Это основная структура данных, которая проходит через весь pipeline.
    Содержит как исходные данные, так и результаты обработки.
    """
    # Идентификация
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""  # ID из источника (video_id, file_path, message_id)

    # Метаданные контента
    content_type: ContentType = ContentType.TEXT
    source_type: SourceType = SourceType.LOCAL_FILE
    title: str = ""
    author: str = ""
    url: Optional[str] = None

    # Пути к файлам
    source_path: Optional[Path] = None  # Оригинальный файл (аудио, PDF, изображение)
    processed_path: Optional[Path] = None  # Обработанный файл

    # Извлечённый контент
    raw_text: str = ""
    segments: List[TextSegment] = field(default_factory=list)
    chunks: List[TextChunk] = field(default_factory=list)

    # Дополнительные метаданные
    duration: Optional[float] = None  # Для аудио/видео
    language: str = "auto"
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Состояние обработки
    processing_stage: ProcessingStage = ProcessingStage.PENDING
    errors: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        # Конвертируем Path из строки если нужно
        if self.source_path and isinstance(self.source_path, str):
            self.source_path = Path(self.source_path)
        if self.processed_path and isinstance(self.processed_path, str):
            self.processed_path = Path(self.processed_path)

    def add_error(self, error: str) -> None:
        """Добавляет ошибку в список."""
        self.errors.append(error)
        self.updated_at = datetime.now()

    def update_stage(self, stage: ProcessingStage) -> None:
        """Обновляет этап обработки."""
        self.processing_stage = stage
        self.updated_at = datetime.now()

    @property
    def is_failed(self) -> bool:
        """Проверяет, завершилась ли обработка с ошибкой."""
        return self.processing_stage == ProcessingStage.FAILED or len(self.errors) > 0

    @property
    def full_text(self) -> str:
        """Возвращает полный текст из всех сегментов."""
        if self.raw_text:
            return self.raw_text
        return " ".join(seg.text for seg in self.segments if seg.text)

    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует в словарь для сериализации."""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "content_type": self.content_type.value,
            "source_type": self.source_type.value,
            "title": self.title,
            "author": self.author,
            "url": self.url,
            "source_path": str(self.source_path) if self.source_path else None,
            "duration": self.duration,
            "language": self.language,
            "segments_count": len(self.segments),
            "chunks_count": len(self.chunks),
            "processing_stage": self.processing_stage.value,
            "errors": self.errors,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class ProcessingResult:
    """Результат обработки одного ContentItem."""
    item: ContentItem
    success: bool
    chunks_created: int = 0
    embeddings_generated: int = 0
    stored_count: int = 0
    processing_time: float = 0.0
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует в словарь."""
        return {
            "item_id": self.item.id,
            "source_id": self.item.source_id,
            "success": self.success,
            "chunks_created": self.chunks_created,
            "embeddings_generated": self.embeddings_generated,
            "stored_count": self.stored_count,
            "processing_time": self.processing_time,
            "error_message": self.error_message,
        }


@dataclass
class PipelineStats:
    """Статистика работы pipeline."""
    total_items: int = 0
    processed: int = 0
    succeeded: int = 0
    failed: int = 0
    chunks_created: int = 0
    embeddings_generated: int = 0
    stored: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    @property
    def duration_seconds(self) -> float:
        """Возвращает время работы в секундах."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

    @property
    def success_rate(self) -> float:
        """Возвращает процент успешных обработок."""
        if self.processed == 0:
            return 0.0
        return (self.succeeded / self.processed) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует в словарь."""
        return {
            "total_items": self.total_items,
            "processed": self.processed,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "success_rate": f"{self.success_rate:.1f}%",
            "chunks_created": self.chunks_created,
            "embeddings_generated": self.embeddings_generated,
            "stored": self.stored,
            "duration_seconds": self.duration_seconds,
        }
