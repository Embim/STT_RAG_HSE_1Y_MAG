import ast
import asyncio
import logging
from typing import Dict, Iterable, List, Optional
import atexit
import uuid
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import Filter, MetadataQuery
from langchain_core.documents import Document
from settings import settings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    BATCH_SIZE = 100
    TOP_LOG_CANDIDATES = 3

    def __init__(self):
        self.collection_name = settings.WEAVIATE_COLLECTION_NAME
        logger.info("Connecting to Weaviate at %s:%s", settings.WEAVIATE_HOST, settings.WEAVIATE_PORT)
        self.client = weaviate.connect_to_local(host=settings.WEAVIATE_HOST, port=settings.WEAVIATE_PORT)
        atexit.register(self.close)

        if not self.client.collections.exists(self.collection_name):
            logger.info("Collection %r not found, creating...", self.collection_name)
            self.client.collections.create(
                name=self.collection_name,
                vectorizer_config=Configure.Vectorizer.text2vec_openai(
                    model="ai-forever/FRIDA",
                    base_url=settings.WEAVIATE_VECTORIZER_BASE_URL,
                    vectorize_collection_name=False,
                ),
                properties=[
                    Property(name="text", data_type=DataType.TEXT),
                    Property(name="metadata", data_type=DataType.TEXT),
                ]
            )
        
        self.collection = self.client.collections.get(self.collection_name)
        logger.info("VectorStoreManager ready: collection=%r", self.collection_name)

    def _hash_to_uuid(self, hash_str: str) -> str:
        """Преобразует hash в детерминированный UUID."""
        # Используем UUID5 для генерации детерминированного UUID из hash
        namespace = uuid.UUID('00000000-0000-0000-0000-000000000000')
        return str(uuid.uuid5(namespace, hash_str))

    @staticmethod
    def _safe_parse_metadata(raw_metadata: str) -> Dict:
        try:
            return ast.literal_eval(raw_metadata) if raw_metadata else {}
        except (ValueError, SyntaxError):
            logger.warning("Failed to parse metadata, using empty dict")
            return {}

    def list_source_titles(self, limit: int = 5000) -> List[str]:
        """Return sorted unique source titles from metadata."""
        response = self.collection.query.fetch_objects(
            limit=limit,
            return_properties=["metadata"],
        )
        titles = set()
        for obj in response.objects:
            metadata = self._safe_parse_metadata(obj.properties.get("metadata", "{}"))
            source_title = metadata.get("title") or metadata.get("source_file_name")
            if source_title:
                titles.add(str(source_title))
        return sorted(titles)

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
            
            try:
                uuid_result = self.collection.data.insert(
                    properties=properties,
                    uuid=obj_uuid
                )
                all_ids.append(str(uuid_result))
            except Exception as e:
                if "already exists" in str(e):
                    logger.warning("Skipping duplicate object: %s", obj_uuid)
                    all_ids.append(str(obj_uuid))
                else:
                    raise
            
            if (i + 1) % self.BATCH_SIZE == 0:
                logger.info("Processed %d texts.", i + 1)
        
        logger.info("Create embeddings - success.")
        return all_ids

    async def search(
        self,
        query: str,
        k: int,
        similarity_threshold: float,
        source_file_name: str | None = None,
        source_title: str | None = None,
    ) -> list[tuple[Document, float]]:
        """Search relevance top-K docs in vector DB."""
        candidate_limit = k
        if source_file_name or source_title:
            candidate_limit = max(k * 20, 200)

        logger.info(
            "VDB search: query=%r, k=%d, threshold=%.2f, source_file_name=%r, source_title=%r, candidate_limit=%d",
            query[:60],
            k,
            similarity_threshold,
            source_file_name,
            source_title,
            candidate_limit,
        )
        response = self.collection.query.near_text(
            query=query,
            limit=candidate_limit,
            return_metadata=MetadataQuery(distance=True),
            filters=(
                Filter.by_property("metadata").like("*title*")
                if source_title
                else (Filter.by_property("metadata").like("*source_file_name*") if source_file_name else None)
            ),
        )

        results = []
        logged_candidates = 0
        for idx, obj in enumerate(response.objects):
            similarity_score = 1 - (obj.metadata.distance or 0)
            metadata = self._safe_parse_metadata(obj.properties.get("metadata", "{}"))

            if source_file_name and metadata.get("source_file_name") != source_file_name:
                continue
            if source_title and metadata.get("title") != source_title:
                continue

            if logged_candidates < self.TOP_LOG_CANDIDATES:
                logger.info(
                    "VDB top candidate #%d: score=%.4f distance=%.4f hash=%s chunk=%s text=%r",
                    logged_candidates + 1,
                    similarity_score,
                    obj.metadata.distance or 0.0,
                    metadata.get("hash"),
                    metadata.get("chunk_index"),
                    obj.properties.get("text", ""),
                )
                logged_candidates += 1
            
            if similarity_score >= similarity_threshold:
                doc = Document(
                    page_content=obj.properties["text"],
                    metadata=metadata
                )
                results.append((doc, similarity_score))
                if len(results) >= k:
                    break

        return results

    def close(self):
        """Закрыть соединение с Weaviate."""
        if hasattr(self, 'client'):
            try:
                self.client.close()
            except:
                pass