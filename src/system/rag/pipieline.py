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
        vector_store_manager: VectorStoreManager = chat_vector_store_manager
):
    try:
        #TODO logs, history
        # rewritten_question = await rewrite(question)
        print('rewritten_question', question)
        context = await retrieve(
            vector_store_manager,
            question,
            k = settings.K
        )
        answer = await generate_answer(
            question_rewritten = question,
            final_rag_content=context
        )

        return answer
    
    except PermissionDeniedError as e:

        print(e)

