import logging
from typing import Optional, Dict, Any

from openai import AsyncOpenAI
from settings import settings
from system.rag.vectore_store import VectorStoreManager
from system.tracing import get_client, observe

logger = logging.getLogger(__name__)


def init_vectore_store_manager() -> VectorStoreManager:
    return VectorStoreManager()


class OpenRouterClient:
    def __init__(
        self,
        model: str = settings.LLM_MODEL,
        api_key: str = settings.LLM_API_KEY_3,
        base_url: str = settings.LLM_BASE_URL
    ):
        self.model = model
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )

    @observe(name="llm-call", capture_input=False, capture_output=False)
    async def chat(self, messages: list, **kwargs) -> str:
        logger.debug("LLM request: model=%s, messages=%d", self.model, len(messages))
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            **kwargs
        )
        content = response.choices[0].message.content
        if response.usage:
            langfuse = get_client()
            question_content = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])
            prompt_content = messages[1].content if len(messages) > 1 and hasattr(messages[1], "content") else ""
            langfuse.update_current_generation(
                model=self.model,
                usage_details={
                    "input": response.usage.prompt_tokens,
                    "output": response.usage.completion_tokens,
                    "total": response.usage.total_tokens,
                },
                input=question_content,
                output=content,
                metadata={"prompt": prompt_content},
            )
        logger.debug("LLM response: %d chars", len(content))
        return content

# # Использование
LLM_REWRITE = OpenRouterClient()
LLM_GENERATE_ANSWER = OpenRouterClient()

CHAT_VECTORE_STORE_MANAGER = init_vectore_store_manager()