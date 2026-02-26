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
    ids = [f"{meta['hash']}_{meta['chunk_index']}" for meta in metas]
    
    await chat_vector_store_manager.add_texts(texts=texts, metadatas=metas, ids=ids)
    logger.info(f"Ingested {len(all_docs)} chunks from {len(data_list)} documents")


# Пример использования
async def main():
    json_data = [
        {
            "hash": "pol123",
            "text": "Поляризация — свойство поперечных волн (например, световых) иметь определенную ориентацию колебаний в пространстве."
        },
        {
            "hash": "usk456",
            "text": "Ускорение — векторная физическая величина, характеризующая быстроту изменения скорости тела."
        },
        {
            "hash": "ind789",
            "text": "Индуктивность — физическая величина, характеризующая способность проводника или цепи накапливать энергию в магнитном поле при протекании тока."
        },
        {
            "hash": "fot012",
            "text": "Фотон — элементарная частица, квант электромагнитного излучения (например, света), не имеющая массы покоя."
        },
        {
            "hash": "adi345",
            "text": "Адиабатный процесс — процесс, происходящий в системе без теплообмена с окружающей средой."
        },
        {
            "hash": "dif678",
            "text": "Дифракция — явление огибания волнами препятствий или отклонения от прямолинейного распространения при прохождении через узкие щели."
        },
        {
            "hash": "pro901",
            "text": "Проводник — вещество, среда или материал, хорошо проводящие электрический ток."
        },
        {
            "hash": "vnu234",
            "text": "Внутренняя энергия — сумма кинетической энергии хаотического движения частиц и потенциальной энергии их взаимодействия"
        },
        {
            "hash": "ine567",
            "text": "Инерция — свойство тела сохранять состояние покоя или равномерного прямолинейного движения при отсутствии внешних воздействий"
        },
        {
            "hash": "sup890",
            "text": "Суперпозиция — способность квантовой системы находиться одновременно в нескольких состояниях, пока не произведено измерение (классический пример — кот Шрёдингера)."
        }
    ]
    
    await ingest_json_to_vector_store(json_data)


if __name__ == "__main__":
    asyncio.run(main())