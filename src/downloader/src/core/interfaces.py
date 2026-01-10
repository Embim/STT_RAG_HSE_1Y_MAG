"""
Abstract base classes for pipeline components.

Все плагины (источники, процессоры, эмбеддеры, хранилища) наследуются от этих классов.
Это обеспечивает единый интерфейс и возможность расширения без изменения существующего кода.
"""

from abc import ABC, abstractmethod
from typing import List, Iterator, Optional, Dict, Any
import logging

from .models import ContentItem, ContentType, TextSegment, TextChunk

logger = logging.getLogger(__name__)


class BaseSource(ABC):
    """
    Абстрактный базовый класс для источников данных.

    Реализует Open/Closed Principle:
    - Open for extension (новые источники через наследование)
    - Closed for modification (базовый класс стабилен)

    Примеры источников: YouTube, RuTube, локальные файлы, Telegram, RSS.
    """

    # Метаданные класса для регистрации в реестре
    source_name: str = "base"
    supported_content_types: List[ContentType] = []

    @abstractmethod
    def fetch(self, **kwargs) -> Iterator[ContentItem]:
        """
        Получает контент из источника.

        Args:
            **kwargs: Параметры специфичные для источника
                      (urls для YouTube, paths для локальных файлов и т.д.)

        Yields:
            ContentItem объекты готовые для обработки
        """
        pass

    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Проверяет корректность конфигурации источника.

        Args:
            config: Словарь с параметрами конфигурации

        Returns:
            True если конфигурация валидна
        """
        pass

    def setup(self, config: Dict[str, Any]) -> None:
        """
        Инициализация источника (опционально).

        Args:
            config: Параметры конфигурации
        """
        pass

    def teardown(self) -> None:
        """Освобождение ресурсов (опционально)."""
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """Возвращает метаданные источника."""
        return {
            "name": self.source_name,
            "supported_types": [ct.value for ct in self.supported_content_types],
        }


class BaseProcessor(ABC):
    """
    Абстрактный базовый класс для процессоров контента.

    Процессоры трансформируют контент (аудио -> текст, изображение -> текст и т.д.).
    Каждый процессор обрабатывает определённые типы контента.
    """

    processor_name: str = "base"
    input_types: List[ContentType] = []
    output_type: ContentType = ContentType.TEXT

    @abstractmethod
    def process(self, item: ContentItem) -> ContentItem:
        """
        Обрабатывает ContentItem.

        Args:
            item: ContentItem для обработки

        Returns:
            Обработанный ContentItem с извлечённым текстом/сегментами
        """
        pass

    @abstractmethod
    def can_process(self, item: ContentItem) -> bool:
        """
        Проверяет, может ли процессор обработать данный item.

        Args:
            item: ContentItem для проверки

        Returns:
            True если процессор может обработать item
        """
        pass

    def setup(self, config: Dict[str, Any]) -> None:
        """
        Инициализация процессора (загрузка моделей и т.д.).

        Args:
            config: Параметры конфигурации
        """
        pass

    def teardown(self) -> None:
        """Освобождение ресурсов."""
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """Возвращает метаданные процессора."""
        return {
            "name": self.processor_name,
            "input_types": [ct.value for ct in self.input_types],
            "output_type": self.output_type.value,
        }


class BaseChunker(ABC):
    """
    Абстрактный базовый класс для стратегий разбиения текста на чанки.

    Чанкеры разбивают сегменты на оптимальные для эмбеддинга части.
    """

    chunker_name: str = "base"

    @abstractmethod
    def chunk(
        self,
        segments: List[TextSegment],
        min_length: int = 50,
        max_length: int = 2000,
    ) -> List[TextChunk]:
        """
        Разбивает сегменты на чанки.

        Args:
            segments: Список сегментов для разбиения
            min_length: Минимальная длина чанка в символах
            max_length: Максимальная длина чанка в символах

        Returns:
            Список чанков готовых для эмбеддинга
        """
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """Возвращает метаданные чанкера."""
        return {"name": self.chunker_name}


class BaseEmbedder(ABC):
    """
    Абстрактный базовый класс для генераторов эмбеддингов.

    Эмбеддеры преобразуют текст в векторные представления.
    """

    embedder_name: str = "base"
    embedding_dimension: int = 0

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Генерирует эмбеддинги для списка текстов.

        Args:
            texts: Список текстов для эмбеддинга

        Returns:
            Список векторов (эмбеддингов)
        """
        pass

    @abstractmethod
    def embed_single(self, text: str) -> List[float]:
        """
        Генерирует эмбеддинг для одного текста.

        Args:
            text: Текст для эмбеддинга

        Returns:
            Вектор эмбеддинга
        """
        pass

    def setup(self, config: Dict[str, Any]) -> None:
        """Инициализация модели."""
        pass

    def teardown(self) -> None:
        """Освобождение ресурсов."""
        pass

    def clear_cache(self) -> None:
        """Очистка кеша (для GPU)."""
        pass

    def get_dimension(self) -> int:
        """Возвращает размерность эмбеддинга."""
        return self.embedding_dimension

    def get_metadata(self) -> Dict[str, Any]:
        """Возвращает метаданные эмбеддера."""
        return {
            "name": self.embedder_name,
            "dimension": self.embedding_dimension,
        }


class BaseStore(ABC):
    """
    Абстрактный базовый класс для хранилищ векторов.

    Хранилища сохраняют эмбеддинги и обеспечивают семантический поиск.
    """

    store_name: str = "base"

    @abstractmethod
    def add(
        self,
        chunks: List[TextChunk],
        metadata: Dict[str, Any],
    ) -> int:
        """
        Добавляет чанки с эмбеддингами в хранилище.

        Args:
            chunks: Список чанков с эмбеддингами
            metadata: Общие метаданные (source_id, title и т.д.)

        Returns:
            Количество успешно добавленных записей
        """
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Ищет похожие записи по эмбеддингу.

        Args:
            query_embedding: Вектор запроса
            limit: Максимальное количество результатов
            filters: Дополнительные фильтры (по source_type, language и т.д.)

        Returns:
            Список найденных записей с метаданными и score
        """
        pass

    @abstractmethod
    def delete(self, source_id: str) -> int:
        """
        Удаляет все записи для данного source_id.

        Args:
            source_id: Идентификатор источника

        Returns:
            Количество удалённых записей
        """
        pass

    def connect(self) -> None:
        """Устанавливает соединение с хранилищем."""
        pass

    def close(self) -> None:
        """Закрывает соединение."""
        pass

    def count(self) -> int:
        """Возвращает общее количество записей."""
        return 0

    def get_metadata(self) -> Dict[str, Any]:
        """Возвращает метаданные хранилища."""
        return {"name": self.store_name}
