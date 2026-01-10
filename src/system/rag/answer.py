from typing import Dict, Iterable, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage

from system.llm.llm_services import llm_generate_asnwer
from system.prompts import ANSWER_SYSTEM_PROMPT, FINAL_ANSWER_CONTEXT_SYSTEM
from settings import settings

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

async def generate_answer(
    question_rewritten: str,
    final_rag_content: str
) -> str:
    messages = _build_anser_messages(
        question_rewritten=question_rewritten,
        final_rag_content=final_rag_content
    )

    return await llm_generate_asnwer.chat(messages)

