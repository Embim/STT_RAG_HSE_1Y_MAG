from settings import settings
from system.llm.llm_services import chat_vector_store_manager
from system.rag.vectore_store import VectorStoreManager
from system.rag.question_rewriter import rewrite
from system.rag.retriver import retrieve
from system.rag.answer import generate_answer
from system.exceptions import LLMPermissionDeniedError

from openai import PermissionDeniedError


def format_timestamp(seconds: float) -> str:
    """Форматировать секунды в HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


async def run(
        question: str,
        top_k: int = settings.K,
        similarity_threshold: float = settings.DEFAULT_SIMILARITY_THRESHOLD,
        vector_store_manager: VectorStoreManager = chat_vector_store_manager,
):
    try:
        #TODO logs, history
        # rewritten_question = await rewrite(question)
        print('rewritten_question', question)
        context = await retrieve(
            vector_store_manager=vector_store_manager,
            query=question,
            k = top_k,
            similarity_threshold=similarity_threshold
        )
        answer = await generate_answer(
            question_rewritten = question,
            final_rag_content=context
        )

        # Парсить metadata для sources
        sources = []
        for doc, score in context:
            try:
                # metadata хранится как строка, нужно преобразовать
                metadata = eval(doc.metadata) if isinstance(doc.metadata, str) else doc.metadata

                # Проверить тип источника
                source_info = {
                    "name": metadata.get("title", "Документ"),
                    "url": metadata.get("url", ""),
                    "timestamp": format_timestamp(metadata.get("start_position", 0)),
                    "score": round(score, 3)
                }
                sources.append(source_info)
            except Exception as e:
                # Fallback на простой формат если metadata не парсится
                sources.append({
                    "name": "Документ",
                    "timestamp": "00:00:00",
                    "score": round(score, 3)
                })

        final_json = {
            "answer": answer,
            "sources": sources
        }

        return final_json, context
    
    except PermissionDeniedError as e:

        print(e)

