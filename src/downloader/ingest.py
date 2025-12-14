import asyncio
from typing import List, Dict
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from system.llm.llm_services import chat_vector_store_manager
from settings import settings
from logging import getLogger

logger = getLogger()


def create_documents_from_json(data: Dict[str, str]) -> List[Document]:
    """Разбивает текст на чанки и создает документы."""
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_text(data["text"])
    
    documents = [
        Document(
            page_content=chunk,
            metadata={
                "hash": data["hash"],
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
        )
        for i, chunk in enumerate(chunks)
    ]
    
    return documents


async def ingest_json_to_vector_store(data_list: List[Dict[str, str]]):
    """Загружает JSON данные в векторную БД."""
    
    all_docs: List[Document] = []
    
    for data in data_list:
        docs = create_documents_from_json(data)
        all_docs.extend(docs)
    
    if not all_docs:
        logger.info("No data to ingest.")
        return
    
    texts = [doc.page_content for doc in all_docs]
    metas = [doc.metadata for doc in all_docs]
    
    await chat_vector_store_manager.add_texts(texts=texts, metadatas=metas)
    logger.info(f"Ingested {len(all_docs)} chunks from {len(data_list)} documents")


# Пример использования
async def main():
    json_data = [
        {
            "hash": "abc123",
            "text": "Model Context Protocol: универсальный мост между LLM и внешним миром"
        },
        {
            "hash": "def456", 
            "text": "Еще один документ с текстом..."
        }
    ]
    
    await ingest_json_to_vector_store(json_data)


if __name__ == "__main__":
    asyncio.run(main())