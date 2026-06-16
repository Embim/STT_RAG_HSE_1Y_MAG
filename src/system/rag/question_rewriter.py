import logging

from langchain_core.messages import HumanMessage, SystemMessage

from system.prompts import REWRITE_PROMPT, SYSTEM_NOTE_FOR_LLM
from system.llm.llm_services import LLM_REWRITE

logger = logging.getLogger(__name__)


def _normalize_query(text: str) -> str:
    return " ".join(text.replace("\n", " ").split()).strip().strip('"').strip("'")


def _is_noisy_rewrite(original: str, rewritten: str) -> bool:
    original_words = max(1, len(original.split()))
    rewritten_words = len(rewritten.split())
    max_words = max(12, int(original_words * 1.6))

    looks_like_list = ":" in rewritten and rewritten.count(",") >= 2
    too_long = rewritten_words > max_words
    empty_or_same = not rewritten or rewritten == original
    return looks_like_list or too_long or empty_or_same


async def rewrite(question: str) -> str:
    logger.info("Rewriting question: %r", question)
    prompt = REWRITE_PROMPT.format(question=question)
    result = await LLM_REWRITE.chat([
        SystemMessage(content=SYSTEM_NOTE_FOR_LLM, role="system"),
        HumanMessage(content=prompt, role="user"),
    ])
    question_normalized = _normalize_query(question)
    rewritten = _normalize_query(result)

    if _is_noisy_rewrite(question_normalized, rewritten):
        logger.info("Rewrite rejected as noisy; using original question")
        return question_normalized

    logger.info("Rewritten: %r", rewritten)
    return rewritten

