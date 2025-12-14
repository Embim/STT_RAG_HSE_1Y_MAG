from typing import Any, Dict, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage

from system.prompts import REWRITE_PROMPT, SYSTEM_NOTE_FOR_LLM
from system.llm.llm_services import llm_rewrite
from settings import settings


async def rewrite(question: str) -> str:

    prompt = REWRITE_PROMPT.format(question=question)

    return await llm_rewrite.chat(
        [
            SystemMessage(content=SYSTEM_NOTE_FOR_LLM, role='system'),
            HumanMessage(content=prompt, role='user')
        ]
    )