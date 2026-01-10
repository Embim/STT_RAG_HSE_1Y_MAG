"""
Weaviate vector store.

Хранит эмбеддинги в Weaviate для семантического поиска.
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import MetadataQuery

from ..core.interfaces import BaseStore
from ..core.models import TextChunk
from ..core.registry import register_store

logger = logging.getLogger(__name__)


@register_store
class WeaviateStore(BaseStore):
    """
    Хранилище векторов на базе Weaviate.

    Поддерживает семантический поиск по эмбеддингам
    с фильтрацией по метаданным.
    """

    store_name = "weaviate"

    def __init__(
        self,
        url: str = "http://localhost:8080",
        collection_name: str = "ContentChunks",
        vector_dimension: int = 1024,
        batch_size: int = 100,
    ):
        """
        Инициализация хранилища.

        Args:
            url: URL Weaviate сервера
            collection_name: Имя коллекции
            vector_dimension: Размерность векторов
            batch_size: Размер батча для загрузки
        """
        self.url = url
        self.collection_name = collection_name
        self.vector_dimension = vector_dimension
        self.batch_size = batch_size
        self.client = None
        self.collection = None

    def connect(self) -> None:
        """Устанавливает соединение с Weaviate."""
        try:
            self.client = weaviate.connect_to_local(
                host=self.url.replace("http://", "").replace("https://", "").split(":")[0],
                port=int(self.url.split(":")[-1]) if ":" in self.url else 8080,
            )

            # Создаём коллекцию если не существует
            self._ensure_collection()

            logger.info(f"Connected to Weaviate at {self.url}")

        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {e}")
            raise

    def close(self) -> None:
        """Закрывает соединение."""
        if self.client:
            self.client.close()
            self.client = None
            self.collection = None
            logger.debug("Weaviate connection closed")

    def _ensure_collection(self) -> None:
        """Создаёт коллекцию если не существует."""
        if not self.client:
            return

        try:
            if self.client.collections.exists(self.collection_name):
                self.collection = self.client.collections.get(self.collection_name)
                logger.debug(f"Using existing collection: {self.collection_name}")
            else:
                self.collection = self.client.collections.create(
                    name=self.collection_name,
                    vectorizer_config=Configure.Vectorizer.none(),
                    properties=[
                        Property(name="source_id", data_type=DataType.TEXT),
                        Property(name="source_type", data_type=DataType.TEXT),
                        Property(name="title", data_type=DataType.TEXT),
                        Property(name="author", data_type=DataType.TEXT),
                        Property(name="chunk_index", data_type=DataType.INT),
                        Property(name="start_position", data_type=DataType.NUMBER),
                        Property(name="end_position", data_type=DataType.NUMBER),
                        Property(name="text", data_type=DataType.TEXT),
                        Property(name="language", data_type=DataType.TEXT),
                        Property(name="url", data_type=DataType.TEXT),
                        Property(name="processed_at", data_type=DataType.DATE),
                    ],
                )
                logger.info(f"Created collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
            raise

    def add(
        self,
        chunks: List[TextChunk],
        metadata: Dict[str, Any],
    ) -> int:
        """
        Добавляет чанки в хранилище.

        Args:
            chunks: Список чанков с эмбеддингами
            metadata: Общие метаданные (source_id, title и т.д.)

        Returns:
            Количество добавленных записей
        """
        if not self.collection:
            self.connect()

        if not chunks:
            return 0

        added_count = 0

        try:
            with self.collection.batch.dynamic() as batch:
                for chunk in chunks:
                    if not chunk.embedding:
                        logger.warning(f"Chunk {chunk.chunk_id} has no embedding, skipping")
                        continue

                    properties = {
                        "source_id": metadata.get("source_id", ""),
                        "source_type": metadata.get("source_type", ""),
                        "title": metadata.get("title", ""),
                        "author": metadata.get("author", ""),
                        "chunk_index": chunk.chunk_id,
                        "start_position": chunk.start_position,
                        "end_position": chunk.end_position,
                        "text": chunk.text,
                        "language": metadata.get("language", "auto"),
                        "url": metadata.get("url", ""),
                        "processed_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    }

                    batch.add_object(
                        properties=properties,
                        vector=chunk.embedding,
                    )
                    added_count += 1

            logger.debug(f"Added {added_count} chunks to Weaviate")
            return added_count

        except Exception as e:
            logger.error(f"Failed to add chunks to Weaviate: {e}")
            raise

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
            filters: Фильтры (source_type, language и т.д.)

        Returns:
            Список найденных записей
        """
        if not self.collection:
            self.connect()

        try:
            # Строим запрос
            query = self.collection.query.near_vector(
                near_vector=query_embedding,
                limit=limit,
                return_metadata=MetadataQuery(distance=True),
            )

            results = []
            for obj in query.objects:
                result = {
                    "id": str(obj.uuid),
                    "score": 1 - obj.metadata.distance if obj.metadata.distance else 0,
                    "distance": obj.metadata.distance,
                    **obj.properties,
                }
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def delete(self, source_id: str) -> int:
        """
        Удаляет все записи для данного source_id.

        Args:
            source_id: Идентификатор источника

        Returns:
            Количество удалённых записей
        """
        if not self.collection:
            self.connect()

        try:
            # Находим все объекты с данным source_id
            from weaviate.classes.query import Filter

            result = self.collection.query.fetch_objects(
                filters=Filter.by_property("source_id").equal(source_id),
                limit=10000,
            )

            deleted = 0
            for obj in result.objects:
                self.collection.data.delete_by_id(obj.uuid)
                deleted += 1

            logger.info(f"Deleted {deleted} objects for source_id: {source_id}")
            return deleted

        except Exception as e:
            logger.error(f"Failed to delete objects: {e}")
            return 0

    def count(self) -> int:
        """Возвращает общее количество записей."""
        if not self.collection:
            self.connect()

        try:
            result = self.collection.aggregate.over_all(total_count=True)
            return result.total_count
        except Exception as e:
            logger.error(f"Failed to count objects: {e}")
            return 0
