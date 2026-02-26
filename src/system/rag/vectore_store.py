import asyncio
from logging import getLogger
from typing import Dict, Iterable, List, Optional
from pathlib import Path
import atexit
import uuid
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import MetadataQuery
from langchain_core.documents import Document

logger = getLogger()


class VectorStoreManager:
    BATCH_SIZE = 100

    def __init__(self, persist_directory, host: str = "localhost", port: int = 8080):
        self.collection_name = Path(persist_directory).name.replace("-", "_").replace(".", "_")
        self.client = weaviate.connect_to_local(host=host, port=port)
        atexit.register(self.close)
        
        if not self.client.collections.exists(self.collection_name):
            self.client.collections.create(
                name=self.collection_name,
                vector_config=Configure.Vectorizer.text2vec_openai(
                    model="ai-forever/FRIDA",
                    base_url="http://localhost:7997/v1",
                    vectorize_collection_name=False,
                ),
                properties=[
                    Property(name="text", data_type=DataType.TEXT),
                    Property(name="metadata", data_type=DataType.TEXT),
                ]
            )
        
        self.collection = self.client.collections.get(self.collection_name)

    def _hash_to_uuid(self, hash_str: str) -> str:
        """Преобразует hash в детерминированный UUID."""
        # Используем UUID5 для генерации детерминированного UUID из hash
        namespace = uuid.UUID('00000000-0000-0000-0000-000000000000')
        return str(uuid.uuid5(namespace, hash_str))

    async def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Create embeddings."""
        logger.info("Start add texts in vector db.")
        texts_list = list(texts)
        all_ids = []
        
        for i, text in enumerate(texts_list):
            properties = {
                "text": text,
                "metadata": str(metadatas[i]) if metadatas else "{}"
            }
            
            # Преобразуем hash в UUID если ids передан
            obj_uuid = None
            if ids:
                obj_uuid = self._hash_to_uuid(ids[i])
            
            uuid_result = self.collection.data.insert(
                properties=properties,
                uuid=obj_uuid
            )
            all_ids.append(str(uuid_result))
            
            if (i + 1) % self.BATCH_SIZE == 0:
                logger.info("Processed %d texts.", i + 1)
        
        logger.info("Create embeddings - success.")
        return all_ids

    async def search(
        self, query: str, k: int, similarity_threshold: float
    ) -> list[tuple[Document, float]]:
        """Search relevance top-K docs in vector DB."""
        response = self.collection.query.near_text(
            query=query,
            limit=k,
            return_metadata=MetadataQuery(distance=True)
        )

        results = []
        for obj in response.objects:
            similarity_score = 1 - (obj.metadata.distance or 0)
            
            if similarity_score >= similarity_threshold:
                doc = Document(
                    page_content=obj.properties["text"],
                    metadata=eval(obj.properties.get("metadata", "{}"))
                )
                results.append((doc, similarity_score))

        return results

    def close(self):
        """Закрыть соединение с Weaviate."""
        if hasattr(self, 'client'):
            try:
                self.client.close()
            except:
                pass