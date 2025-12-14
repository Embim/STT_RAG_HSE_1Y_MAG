import asyncio
from logging import getLogger
from typing import Dict, Iterable, List, Optional
from langchain_chroma import Chroma
from langchain_core.documents import Document
from system.rag.embedder import LocalEmbedder

logger = getLogger()


class VectorStoreManager:
    BATCH_SIZE = 100
    MAX_RETRIES = 3

    def __init__(self, persist_directory):
        self.embeddings = LocalEmbedder()
        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )

    async def add_texts(
            self,
            texts: Iterable[str],
            metadatas: Optional[List[Dict]] = None,
            ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Create embeddings."""
        logger.info("Start add texts in vector db.")
        texts_list = list(texts)
        len_texts_list = len(texts_list)
        all_ids = []
        i = 0

        current_batch_size = min(self.BATCH_SIZE, len_texts_list - i)
        while i < len_texts_list:
            try:
                chunk_texts = texts_list[i : i + current_batch_size]
                chunk_metas = metadatas[i : i + current_batch_size] if metadatas is not None else None
                chunk_ids = ids[i : i + current_batch_size] if ids is not None else None

                result_ids = await self.vector_store.aadd_texts(
                    chunk_texts, metadatas=chunk_metas, ids=chunk_ids
                )
                logger.info("Create embeddings chunk from %d to %d texts.", i, i + current_batch_size)
                all_ids.extend(result_ids)
                i += current_batch_size
            except ValueError as e:
                msg = str(e).lower()
                if (
                    "max batch size" in msg
                    or "cannot submit more than" in msg
                    or "exceeds maximum batch size" in msg
                ) and current_batch_size != 0:
                    current_batch_size = current_batch_size // 2
                    logger.warning(
                        "Max batch size exceeded. Trying decrease batch size - %d",
                        current_batch_size
                    )
                    continue
                logger.error(
                    "Unexpected ValueError: %s", e
                )
                raise
        logger.info("Create embeddings - success.")
        return all_ids


    async def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        """Search relevance top-K docs in vector DB."""
        return await self.vector_store.asimilarity_search_with_relevance_scores(
            query, k=k
        )