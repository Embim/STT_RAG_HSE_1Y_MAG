"""Адаптер для использования VectorStoreManager в InputPipline"""
import json
from typing import List, Dict, Any
from src.system.rag.vectore_store import VectorStoreManager
from src.downloader.src.core.models import TextChunk


class VectorStoreAdapter:
    """Адаптирует VectorStoreManager для InputPipline"""

    def __init__(self, collection_name: str = "youtube_lectures"):
        self.store = VectorStoreManager(
            persist_directory=collection_name,
            host="localhost",
            port=8080
        )

    def connect(self):
        """Подключение уже выполнено в __init__"""
        print(f"✅ Подключен к Weaviate коллекции: {self.store.collection_name}")

    async def add_chunks(self, chunks: List[TextChunk], metadata: Dict[str, Any]):
        """Добавить чанки в векторную БД"""
        texts = []
        metadatas = []
        ids = []

        for i, chunk in enumerate(chunks):
            # Текст чанка
            texts.append(chunk.text)

            # Расширенная metadata с InputPipline данными
            chunk_metadata = {
                "source_id": metadata.get("source_id", ""),
                "source_type": metadata.get("source_type", ""),
                "title": metadata.get("title", ""),
                "author": metadata.get("author", ""),
                "url": metadata.get("url", ""),
                "chunk_id": i,
                "start_position": chunk.start_position if hasattr(chunk, 'start_position') else 0,
                "end_position": chunk.end_position if hasattr(chunk, 'end_position') else 0,
            }

            metadatas.append(chunk_metadata)

            # ID = source_id + chunk_index
            chunk_id = f"{metadata.get('source_id', 'unknown')}_{i}"
            ids.append(chunk_id)

        # Добавить в векторную БД (эмбеддинги создадутся автоматически через Weaviate text2vec-openai/Infinity)
        await self.store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        print(f"✅ Добавлено {len(chunks)} чанков в Weaviate")

    def close(self):
        """Закрыть подключение"""
        self.store.close()
