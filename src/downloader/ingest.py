import argparse
import asyncio
import hashlib
from pathlib import Path
from typing import Any, Dict, List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from system.llm.llm_services import get_chat_vectore_store_manager
from settings import settings
import logging

logger = logging.getLogger(__name__)


def _create_documents_from_text(data: Dict[str, Any]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = text_splitter.split_text(data["text"])
    ocr_segments = data.get("ocr_segments", [])
    
    # Распределяем OCR-сегменты по чанкам.
    # Так как у ASR-текста нет таймкодов, мы можем лишь грубо сопоставить OCR-сегменты
    # по проценту продвижения по чанкам.
    total_ocr_segments = len(ocr_segments)
    total_chunks = len(chunks)
    
    documents = []
    for i, chunk in enumerate(chunks):
        current_ocr_parts = []
        # Этот блок выполнится только если в data пришли ocr_segments (то есть стоял флаг в UI)
        if total_ocr_segments > 0 and total_chunks > 0:
            # Находим долю этого чанка от общего количества
            start_ratio = i / total_chunks
            end_ratio = (i + 1) / total_chunks
            
            start_idx = int(start_ratio * total_ocr_segments)
            end_idx = int(end_ratio * total_ocr_segments)
            
            for ocr_seg in ocr_segments[start_idx:end_idx]:
                clean_ocr = str(ocr_seg.get("text", "")).replace("[ВИЗУАЛЬНЫЙ ТЕКСТ НА ЭКРАНЕ:", "").replace("]", "").strip()
                if clean_ocr:
                    current_ocr_parts.append(clean_ocr)
        
        # Если OCR не запускался (или на слайдах не было текста),
        # то current_ocr_parts останется пустым, и в метаданные запишется пустая строка "".
        documents.append(
            Document(
                page_content=chunk,
                metadata={
                    "hash": data["hash"],
                    "chunk_index": i,
                    "total_chunks": total_chunks,
                    "title": data.get("title"),
                    "source_url": data.get("source_url"),
                    "source_file_name": data.get("source_file_name"),
                    "ocr_text": " | ".join(current_ocr_parts) if current_ocr_parts else "",
                }
            )
        )

    return documents


def _create_documents_from_segments(data: Dict[str, Any]) -> List[Document]:
    segments = data.get("segments", [])
    if not segments:
        return _create_documents_from_text(data)

    chunks: List[Dict[str, Any]] = []
    current_text_parts: List[str] = []
    current_ocr_parts: List[str] = []
    current_start: float | None = None
    current_end: float | None = None
    current_size = 0

    for segment in segments:
        segment_text = str(segment.get("text", "")).strip()
        if not segment_text:
            continue
            
        segment_start = float(segment.get("start", 0.0) or 0.0)
        segment_end = float(segment.get("end", segment_start) or segment_start)
        
        is_ocr = "[ВИЗУАЛЬНЫЙ ТЕКСТ НА ЭКРАНЕ:" in segment_text
        
        if is_ocr:
            clean_ocr = segment_text.replace("[ВИЗУАЛЬНЫЙ ТЕКСТ НА ЭКРАНЕ:", "").replace("]", "").strip()
            if clean_ocr:
                current_ocr_parts.append(clean_ocr)
            continue

        next_size = current_size + len(segment_text) + (1 if current_text_parts else 0)
        if current_text_parts and next_size > settings.CHUNK_SIZE:
            chunks.append(
                {
                    "text": " ".join(current_text_parts),
                    "ocr_text": " | ".join(current_ocr_parts) if current_ocr_parts else "",
                    "start_sec": current_start,
                    "end_sec": current_end,
                }
            )
            current_text_parts = [segment_text]
            current_ocr_parts = []
            current_start = segment_start
            current_end = segment_end
            current_size = len(segment_text)
            continue

        if not current_text_parts:
            current_start = segment_start
        current_text_parts.append(segment_text)
        current_end = segment_end
        current_size = next_size

    if current_text_parts:
        chunks.append(
            {
                "text": " ".join(current_text_parts),
                "ocr_text": " | ".join(current_ocr_parts) if current_ocr_parts else "",
                "start_sec": current_start,
                "end_sec": current_end,
            }
        )

    documents = [
        Document(
            page_content=chunk["text"],
            metadata={
                "hash": data["hash"],
                "chunk_index": idx,
                "total_chunks": len(chunks),
                "start_sec": chunk["start_sec"],
                "end_sec": chunk["end_sec"],
                "ocr_text": chunk["ocr_text"],
                "title": data.get("title"),
                "source_url": data.get("source_url"),
                "source_file_name": data.get("source_file_name"),
            },
        )
        for idx, chunk in enumerate(chunks)
    ]
    return documents


def create_documents_from_json(data: Dict[str, Any]) -> List[Document]:
    """Создает документы для ingest; при наличии segments сохраняет таймстемпы."""
    if data.get("segments"):
        return _create_documents_from_segments(data)
    return _create_documents_from_text(data)


async def ingest_json_to_vector_store(data_list: List[Dict[str, Any]]):
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
    
    await get_chat_vectore_store_manager().add_texts(texts=texts, metadatas=metas, ids=ids)
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


def _load_txt_dir(directory: Path) -> List[Dict[str, Any]]:
    """Read every .txt under directory into the ingest dict shape."""
    files = sorted(directory.rglob("*.txt"))
    items: List[Dict[str, Any]] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            continue
        doc_hash = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:16]
        items.append({
            "hash": doc_hash,
            "text": text,
            "title": path.stem,
            "source_file_name": path.name,
        })
    return items


async def ingest_txt_dir(directory: Path) -> None:
    items = _load_txt_dir(directory)
    if not items:
        logger.warning("No .txt files in %s", directory)
        return
    logger.info("Ingesting %d .txt files from %s", len(items), directory)
    await ingest_json_to_vector_store(items)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="downloader.ingest")
    parser.add_argument("--from-dir", default=None,
                        help="Ingest all .txt files under this directory; runs the demo data if omitted")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.from_dir:
        asyncio.run(ingest_txt_dir(Path(args.from_dir)))
    else:
        asyncio.run(main())