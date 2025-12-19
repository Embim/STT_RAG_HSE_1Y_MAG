from settings import settings
from system.llm.llm_services import chat_vector_store_manager
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
        vector_store_manager: VectorStoreManager = chat_vector_store_manager,
):
    try:
        #TODO logs, history
        rewritten_question = await rewrite(question)
        print('rewritten_question', rewritten_question)
        context = await retrieve(
            vector_store_manager=vector_store_manager,
            query=rewritten_question,
            k = top_k,
            similarity_threshold=similarity_threshold
        )
        print('context', context)
        answer = await generate_answer(
            question_rewritten = rewritten_question,
            final_rag_content=context
        )
        final_json = {
            "answer": answer,
            "sources": [{"name": "Empty_for_now", "timestamp": '00:11:22'}]
        }

        return final_json
    
    except PermissionDeniedError as e:

        print(e)

