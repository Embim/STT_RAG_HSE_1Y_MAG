import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from system.prompts import REWRITE_PROMPT, SYSTEM_NOTE_FOR_LLM
from system.llm.llm_services import LLM_REWRITE
from settings import settings

logger = logging.getLogger(__name__)


async def rewrite(question: str) -> str:
    logger.info("Rewriting question: %r", question)
    prompt = REWRITE_PROMPT.format(question=question)
    result = await LLM_REWRITE.chat([
        SystemMessage(content=SYSTEM_NOTE_FOR_LLM, role="system"),
        HumanMessage(content=prompt, role="user"),
    ])
    logger.info("Rewritten: %r", result)
    return result

