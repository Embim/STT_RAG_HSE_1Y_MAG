from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    PROJECT_ROOT: str = '/home/kovynev-sergey/Documents/projects/STT_RAG_HSE_1Y_MAG/'
    DATA_DIR: str = PROJECT_ROOT + "data"
    TRANSCRIPTS_DIR: str = DATA_DIR + "/transcripts"
    VECTORE_STORE_DIR: str = DATA_DIR + '/vectore_store'

    WEAVIATE_URL: str = "http://localhost:8080"
    WEAVIATE_COLLECTION_NAME: str = "LectureChunks"

    EMBEDDING_URL: str = 'http://localhost:7997/embeddings'

    CHUNK_SIZE: int =  800
    CHUNK_OVERLAP: int = 125

    K: int = 5
    DEFAULT_SIMILARITY_THRESHOLD: float = 0.7
    HYBRID_SEARCH_ALPHA: float = 0.7

    LLM_MODEL: str
    LLM_API_KEY_1: str
    LLM_API_KEY_2: str
    LLM_API_KEY_3: str
    LLM_BASE_URL: str = "https://openrouter.ai/api/v1"
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 2000

    class Config:
        env_file = '/home/kovynev-sergey/Documents/projects/STT_RAG_HSE_1Y_MAG/.env'
        env_file_encoding = 'utf-8'

settings = Settings()