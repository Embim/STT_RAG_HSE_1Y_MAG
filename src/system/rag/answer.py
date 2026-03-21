import logging
from typing import Dict, Iterable, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langfuse import get_client

from system.llm.llm_services import LLM_GENERATE_ANSWER
from system.prompts import ANSWER_SYSTEM_PROMPT, FINAL_ANSWER_CONTEXT_SYSTEM
from system.tracing import observe
from settings import settings

logger = logging.getLogger(__name__)

def _build_anser_messages(
    question_rewritten: str,
    final_rag_content: str,
) -> List[BaseMessage]:
    messages =[]

    messages.append(SystemMessage(content=ANSWER_SYSTEM_PROMPT, role='system'))

    context_system = FINAL_ANSWER_CONTEXT_SYSTEM.format(final_rag_content=final_rag_content)
    messages.append(SystemMessage(content=context_system, role='system'))

    messages.append(HumanMessage(content=question_rewritten, role='user'))

    return messages

@observe(name="answer-generation")
async def generate_answer(
    question_rewritten: str,
    final_rag_content: str
) -> str:
    logger.info("Generating answer (context=%d chars)", len(final_rag_content))
    messages = _build_anser_messages(
        question_rewritten=question_rewritten,
        final_rag_content=final_rag_content,
    )
    langfuse = get_client()
    langfuse.update_current_span(
        metadata={
            "rag_context": final_rag_content,
            "question": question_rewritten,
        }
    )
    answer = await LLM_GENERATE_ANSWER.chat(messages)
    logger.info("Answer generated (%d chars)", len(answer))
    return answer

