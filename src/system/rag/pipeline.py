import logging

from settings import settings
from system.llm.llm_services import CHAT_VECTORE_STORE_MANAGER
from system.rag.vectore_store import VectorStoreManager
from system.rag.question_rewriter import rewrite
from system.rag.retriver import retrieve
from system.rag.answer import generate_answer
from system.exceptions import LLMPermissionDeniedError
from system.tracing import get_client, observe

from openai import PermissionDeniedError

logger = logging.getLogger(__name__)


@observe(name="rag-pipeline")
async def run(
        question: str,
        top_k: int = settings.K,
        similarity_threshold: float = settings.DEFAULT_SIMILARITY_THRESHOLD,
        use_rewrite: bool = True,
        source_file_name: str | None = None,
        source_title: str | None = None,
        vector_store_manager: VectorStoreManager = CHAT_VECTORE_STORE_MANAGER,
):
    try:
        logger.info(
            "RAG query: %r (top_k=%d, threshold=%.2f, rewrite=%s, source_file_name=%r, source_title=%r)",
            question,
            top_k,
            similarity_threshold,
            use_rewrite,
            source_file_name,
            source_title,
        )
        langfuse = get_client()
        retrieval_query = question
        if use_rewrite:
            try:
                retrieval_query = await rewrite(question)
            except Exception as rewrite_error:
                logger.warning("Rewrite failed, using original question: %s", rewrite_error)
                retrieval_query = question

        langfuse.update_current_trace(
            tags=["rag"],
            metadata={
                "top_k": top_k,
                "threshold": similarity_threshold,
                "use_rewrite": use_rewrite,
                "retrieval_query": retrieval_query,
                "source_file_name": source_file_name,
                "source_title": source_title,
            },
        )
        context = await retrieve(
            vector_store_manager=vector_store_manager,
            query=retrieval_query,
            k=top_k,
            similarity_threshold=similarity_threshold,
            source_file_name=source_file_name,
            source_title=source_title,
        )
        logger.info("Retrieved context (%d chars)", len(context))
        answer = await generate_answer(
            question_rewritten=question,
            final_rag_content=context
        )
        logger.info("Answer generated (%d chars)", len(answer))
        return {
            "answer": answer,
            "context": context,
            "retrieval_query": retrieval_query,
            "rewrite_applied": use_rewrite and retrieval_query != question,
            "source_file_name": source_file_name,
            "source_title": source_title,
        }

    except PermissionDeniedError as e:
        logger.error("LLM permission denied: %s", e)

