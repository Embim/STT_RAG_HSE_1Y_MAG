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
        rewritten_question = await rewrite(question)
        print('rewritten_question', rewritten_question)
        context = await retrieve(
            vector_store_manager,
            rewritten_question,
            k = settings.K
        )
        print('context', context)
        answer = await generate_answer(
            question_rewritten = rewritten_question,
            final_rag_content=context
        )

        return answer
    
    except PermissionDeniedError as e:

        print(e)

