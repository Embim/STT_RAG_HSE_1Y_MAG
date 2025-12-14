from typing import Optional, Dict, Any
from system.rag.vectore_store import VectorStoreManager
from openai import AsyncOpenAI
from settings import settings


def init_vectore_store_manager(persist_dir_path:str) -> VectorStoreManager:
    return VectorStoreManager(persist_dir_path)


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
llm_rewrite = OpenRouterClient()
llm_generate_asnwer = OpenRouterClient()

chat_vector_store_manager = init_vectore_store_manager(settings.VECTORE_STORE_DIR)