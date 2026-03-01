from pathlib import Path
from pydantic_settings import BaseSettings

ENV_PATH = Path(__file__).resolve().parent.parent / '.env'

class Settings(BaseSettings):

    WEAVIATE_HOST: str = "localhost"
    WEAVIATE_PORT: int = 8080
    WEAVIATE_COLLECTION_NAME: str = "LectureChunks"

    EMBEDDING_URL: str = 'http://localhost:7997/v1/embeddings'
    WEAVIATE_VECTORIZER_BASE_URL: str = 'http://localhost:7997'
    WHISPER_URL: str = 'http://localhost:8000'

    CHUNK_SIZE: int =  800
    CHUNK_OVERLAP: int = 125

    K: int = 5
    DEFAULT_SIMILARITY_THRESHOLD: float = 0.0
    HYBRID_SEARCH_ALPHA: float = 0.7

    HUGGINGFACE_CACHE: str = ""

    LLM_MODEL: str
    LLM_API_KEY_1: str
    LLM_API_KEY_2: str
    LLM_API_KEY_3: str
    LLM_BASE_URL: str = "https://openrouter.ai/api/v1"
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 2000

    class Config:
        env_file = ENV_PATH
        env_file_encoding = 'utf-8'

settings = Settings()