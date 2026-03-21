import logging

from settings import settings
from system.llm.llm_services import CHAT_VECTORE_STORE_MANAGER
from system.rag.vectore_store import VectorStoreManager
from system.rag.question_rewriter import rewrite
from system.rag.retriver import retrieve
from system.rag.answer import generate_answer
from system.exceptions import LLMPermissionDeniedError
from system.tracing import observe

from openai import PermissionDeniedError

logger = logging.getLogger(__name__)


@observe(name="rag-pipeline")
async def run(
        question: str,
        top_k: int = settings.K,
        similarity_threshold: float = settings.DEFAULT_SIMILARITY_THRESHOLD,
        vector_store_manager: VectorStoreManager = CHAT_VECTORE_STORE_MANAGER,
):
    try:
        logger.info("RAG query: %r (top_k=%d, threshold=%.2f)", question, top_k, similarity_threshold)
        context = await retrieve(
            vector_store_manager=vector_store_manager,
            query=question,
            k=top_k,
            similarity_threshold=similarity_threshold
        )
        logger.info("Retrieved context (%d chars)", len(context))
        answer = await generate_answer(
            question_rewritten=question,
            final_rag_content=context
        )
        logger.info("Answer generated (%d chars)", len(answer))
        return {"answer": answer, "context": context}

    except PermissionDeniedError as e:
        logger.error("LLM permission denied: %s", e)

