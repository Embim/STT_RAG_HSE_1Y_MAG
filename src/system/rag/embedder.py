import logging

import requests
from settings import settings

logger = logging.getLogger(__name__)


class LocalEmbedder:
    def __init__(self):
        self.url = settings.EMBEDDING_URL
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        logger.debug("Embedding %d documents", len(texts))
        response = requests.post(self.url, json={"input": texts})
        response.raise_for_status()
        data = response.json()
        logger.debug("Embeddings received: %d vectors", len(data["data"]))
        return [item["embedding"] for item in data["data"]]

    def embed_query(self, text: str) -> list[float]:
        logger.debug("Embedding query (%d chars)", len(text))
        response = requests.post(self.url, json={"input": text})
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"]