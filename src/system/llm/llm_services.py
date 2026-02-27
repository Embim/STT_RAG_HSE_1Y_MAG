from typing import Optional, Dict, Any
from system.rag.vectore_store import VectorStoreManager
from openai import AsyncOpenAI
from settings import settings


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

    async def chat(self, messages: list, **kwargs) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            **kwargs
        )
        return response.choices[0].message.content

# # Использование
LLM_REWRITE = OpenRouterClient()
LLM_GENERATE_ANSWER = OpenRouterClient()

CHAT_VECTORE_STORE_MANAGER = init_vectore_store_manager()