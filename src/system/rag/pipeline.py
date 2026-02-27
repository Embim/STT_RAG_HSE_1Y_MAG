from settings import settings
from system.llm.llm_services import CHAT_VECTORE_STORE_MANAGER
from system.rag.vectore_store import VectorStoreManager
from system.rag.question_rewriter import rewrite
from system.rag.retriver import retrieve
from system.rag.answer import generate_answer
from system.exceptions import LLMPermissionDeniedError

from openai import PermissionDeniedError


async def run(
        question: str,
        top_k: int = settings.K,
        similarity_threshold: float = settings.DEFAULT_SIMILARITY_THRESHOLD,
        vector_store_manager: VectorStoreManager = CHAT_VECTORE_STORE_MANAGER,
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
        final_json = {
            "answer": answer,
            "context": context
        }

        return final_json
    
    except PermissionDeniedError as e:

        print(e)

