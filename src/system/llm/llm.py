from typing import Optional, Dict, Any
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("MODEL")
OR_API_KEY = os.environ.get('API_KEY_1')

class OpenRouterClient:
    def __init__(
        self,
        model: str = MODEL,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1"
    ):
        self.model = model
        self.client = OpenAI(
            api_key=OR_API_KEY,
            base_url=base_url
        )
    
    def chat(self, messages: list, **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        return response.choices[0].message.content

# # Использование
# llm = OpenRouterClient()
# response = llm.chat([{"role": "user", "content": "Hello!"}])