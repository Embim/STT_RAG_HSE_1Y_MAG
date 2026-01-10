"""
Sentence Transformer embedder.

Генерирует эмбеддинги с помощью sentence-transformers.
"""

from typing import Dict, Any, List
import logging
import time

import torch
import numpy as np
from sentence_transformers import SentenceTransformer

from ..core.interfaces import BaseEmbedder
from ..core.registry import register_embedder

logger = logging.getLogger(__name__)


@register_embedder
class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Эмбеддер на основе sentence-transformers.

    Поддерживает различные модели:
    - ai-forever/FRIDA (768 dim) - для русского языка, требует префиксы
    - ai-forever/sbert_large_nlu_ru (1024 dim) - для русского языка
    - intfloat/multilingual-e5-large (1024 dim) - multilingual
    - BAAI/bge-m3 (1024 dim) - multilingual

    Для FRIDA используйте:
    - embed_query() для поисковых запросов
    - embed_documents() для документов/чанков
    """

    embedder_name = "sentence_transformer"
    embedding_dimension = 1024  # Будет обновлено после загрузки модели

    def __init__(
        self,
        model_name: str = "ai-forever/FRIDA",
        device: str = "cuda",
        batch_size: int = 64,
        max_length: int = 512,
        normalize: bool = True,
        use_fp16: bool = True,
        query_prefix: str = "search_query: ",
        document_prefix: str = "search_document: ",
    ):
        """
        Инициализация эмбеддера.

        Args:
            model_name: Название модели на HuggingFace
            device: Устройство (cuda, cpu)
            batch_size: Размер батча для обработки
            max_length: Максимальная длина токенов
            normalize: Нормализовать эмбеддинги
            use_fp16: Использовать FP16 для экономии памяти
            query_prefix: Префикс для поисковых запросов (для FRIDA)
            document_prefix: Префикс для документов (для FRIDA)
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize = normalize
        self.use_fp16 = use_fp16
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.model = None
        self._is_setup = False

    def setup(self, config: Dict[str, Any]) -> None:
        """Загружает модель."""
        if self._is_setup:
            return

        logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
        start_time = time.time()

        self.model = SentenceTransformer(
            self.model_name,
            device=self.device,
        )

        # Устанавливаем максимальную длину
        self.model.max_seq_length = self.max_length

        # Переводим в FP16 для экономии памяти
        if self.use_fp16 and self.device == "cuda":
            self.model.half()

        # Обновляем размерность эмбеддинга
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()

        load_time = time.time() - start_time
        logger.info(
            f"Embedding model loaded in {load_time:.2f}s, "
            f"dimension: {self.embedding_dimension}"
        )
        self._is_setup = True

    def teardown(self) -> None:
        """Освобождает ресурсы."""
        if self.model:
            del self.model
            self.model = None
        self.clear_cache()
        self._is_setup = False

    def clear_cache(self) -> None:
        """Очищает кеш GPU."""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            logger.debug("GPU cache cleared")

    def embed(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """
        Генерирует эмбеддинги для списка текстов.

        Args:
            texts: Список текстов
            is_query: True для поисковых запросов, False для документов

        Returns:
            Список эмбеддингов (списков float)
        """
        if not texts:
            return []

        if not self._is_setup:
            self.setup({})

        # Добавляем префикс в зависимости от типа (для FRIDA и подобных моделей)
        prefix = self.query_prefix if is_query else self.document_prefix
        if prefix:
            texts = [prefix + text for text in texts]

        logger.debug(f"Encoding {len(texts)} texts in batches of {self.batch_size}")
        start_time = time.time()

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        total_time = time.time() - start_time
        avg_time = total_time / len(texts)

        logger.debug(
            f"Encoded {len(texts)} texts in {total_time:.2f}s "
            f"({avg_time*1000:.2f}ms per text)"
        )

        # Конвертируем numpy arrays в списки
        return [emb.tolist() for emb in embeddings]

    def embed_query(self, texts: List[str]) -> List[List[float]]:
        """
        Генерирует эмбеддинги для поисковых запросов.

        Args:
            texts: Список запросов

        Returns:
            Список эмбеддингов
        """
        return self.embed(texts, is_query=True)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Генерирует эмбеддинги для документов.

        Args:
            texts: Список документов

        Returns:
            Список эмбеддингов
        """
        return self.embed(texts, is_query=False)

    def embed_single(self, text: str, is_query: bool = False) -> List[float]:
        """
        Генерирует эмбеддинг для одного текста.

        Args:
            text: Текст
            is_query: True для поискового запроса, False для документа

        Returns:
            Эмбеддинг (список float)
        """
        if not self._is_setup:
            self.setup({})

        # Добавляем префикс
        prefix = self.query_prefix if is_query else self.document_prefix
        if prefix:
            text = prefix + text

        embedding = self.model.encode(
            text,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return embedding.tolist()

    def similarity(
        self,
        embedding1: List[float],
        embedding2: List[float],
    ) -> float:
        """
        Вычисляет косинусное сходство между двумя эмбеддингами.

        Args:
            embedding1: Первый эмбеддинг
            embedding2: Второй эмбеддинг

        Returns:
            Косинусное сходство (от -1 до 1)
        """
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)

        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)
