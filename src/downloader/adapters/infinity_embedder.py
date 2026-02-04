"""Адаптер для использования Infinity/FRIDA embeddings в InputPipline"""
import requests
import numpy as np
from typing import List
from src.downloader.src.core.interfaces import BaseEmbedder


class InfinityEmbedder(BaseEmbedder):
    """Embedder через Infinity API (FRIDA модель)"""

    name = "infinity"

    def __init__(self, url: str = "http://localhost:7997", batch_size: int = 32):
        self.url = url
        self.batch_size = batch_size
        self.endpoint = f"{url}/v1/embeddings"

    def setup(self, config: dict):
        """Проверить подключение к Infinity"""
        try:
            response = requests.post(
                self.endpoint,
                json={"input": "test"},
                timeout=5
            )
            response.raise_for_status()
            print(f"✅ Infinity доступен на {self.url}")
        except Exception as e:
            raise ConnectionError(f"Не могу подключиться к Infinity: {e}")

    def embed(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """Создать эмбеддинги через Infinity API"""
        embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            response = requests.post(
                self.endpoint,
                json={"input": batch},
                timeout=60
            )
            response.raise_for_status()

            # Infinity возвращает: {"data": [{"embedding": [...]}, ...]}
            batch_embeddings = [item["embedding"] for item in response.json()["data"]]
            embeddings.extend(batch_embeddings)

        return embeddings

    def embed_single(self, text: str, is_query: bool = False) -> List[float]:
        """Эмбеддинг одного текста"""
        return self.embed([text], is_query=is_query)[0]

    def clear_cache(self):
        """Очистка кеша (не требуется для Infinity)"""
        pass

    def teardown(self):
        """Нет необходимости в cleanup"""
        pass
