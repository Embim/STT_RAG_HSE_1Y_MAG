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

    INGEST_CONCURRENCY: int = 3

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

    # Langfuse observability
    LANGFUSE_ENABLED: bool = True
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_BASE_URL: str = "http://localhost:3000"

    # MLflow — process tracking + experiment runs.
    # Если MLFLOW_TRACKING_URI пуст — инструментация работает в noop-режиме.
    MLFLOW_TRACKING_URI: str = "http://localhost:5001"
    MLFLOW_REGISTRY_URI: str = ""
    MLFLOW_S3_ENDPOINT_URL: str = "http://localhost:9002"
    MLFLOW_ARTIFACT_BUCKET: str = "mlflow-artifacts"
    MLFLOW_EXPERIMENT_NAME: str = "stt-rag-ingest"
    MLFLOW_ASR_REGISTERED_MODEL_NAME: str = "stt-rag-asr"
    MLFLOW_ASR_MODEL_ALIAS: str = "candidate"
    # Отдельный experiment для judge-прогонов: чтобы ingest-runs (с GPU/CPU
    # сайдкаром каждые 5 сек) не смешивались в одном UI list view с
    # judge-runs (короткие, без сайдкара, по 1 на ASR-backend).
    MLFLOW_JUDGE_EXPERIMENT: str = "stt-rag-judge"
    AWS_ACCESS_KEY_ID: str = "mlflow"
    AWS_SECRET_ACCESS_KEY: str = "mlflow_secret"
    # Чанкинг аудио перед транскрипцией: 0 = без чанкинга, иначе минут на чанк.
    # Нужен для real-time прогресса в MLflow (после каждого чанка обновляется log_metric).
    # 5 мин = умеренная гранулярность (12 точек/час), и достаточный контекст
    # для long-context backends типа vibevoice/qwen3 (которые на коротких чанках
    # теряют свою главную фичу — cross-segment understanding/diarization).
    TRANSCRIBE_CHUNK_MINUTES: int = 5

    # Evaluation
    # ASR — single OpenAI-compatible HTTP endpoint. Swap models by pointing
    # WHISPER_URL at a different container and updating ASR_NAME / ASR_MODEL_ID.
    ASR_NAME: str = "faster_whisper_large_v3_turbo"
    ASR_MODEL_ID: str = "whisper-1"
    ASR_LANGUAGE: str = "ru"
    # `transcription` (default) — стандартный /v1/audio/transcriptions путь.
    # `chat` — /v1/chat/completions с audio как data URI base64. Использовать
    # для Qwen3-ASR (у которого transcription endpoint в vLLM 0.20.2 сломан).
    ASR_ENDPOINT: str = "transcription"
    ASR_BENCHMARK: str = "mozilla-foundation/common_voice_17_0"
    ASR_BENCHMARK_LANG: str = "ru"
    ASR_BENCHMARK_SPLIT: str = "test"
    # RAGAS
    RAGAS_JUDGE_MODEL: str = ""
    RAGAS_TESTSET_SIZE: int = 30
    EVAL_DATA_DIR: str = "data/eval"
    WEAVIATE_EVAL_COLLECTION_PREFIX: str = "LectureChunks_eval_"

    # ──────────────────────────────────────────────────────────────
    # Transcription Judge (Qwen3.5-9B GGUF Q4 через llama.cpp server)
    # Локальный профиль docker compose `asr-judge` на :8002,
    # OpenAI-compatible /v1/chat/completions. Используется модулем
    # `evaluation.judge.*` для разбора транскрипций на 4 типа ошибок:
    # terminology / hallucination / grammar / gap.
    #
    # Backend = llama.cpp/server-cuda (не vLLM!) — потому что GGUF
    # формат родной для llama.cpp. llama-server c --reasoning-format
    # deepseek + --jinja отдаёт `reasoning_content` отдельным полем
    # в response, и наш JudgeClient ровно его и читает.
    #
    # JUDGE_MODEL_REPO/JUDGE_MODEL_FILE — параметры HF auto-download
    # на старте контейнера (передаются как --hf-repo / --hf-file).
    # JUDGE_MODEL_ID — alias модели, которым server'у её именовать в
    # /v1/models и который мы кладём в request.model. По дефолту короткий
    # "qwen3.5-9b-q4" — попадает в CSV колонку model для агрегации.
    #
    # JUDGE_REASONING=True → клиент ждёт reasoning_content; на не-thinking
    # моделях выключай, иначе будет debug warning о пустом reasoning.
    #
    # Бюджет контекста (llama-server --ctx-size=32768 c q8_0 KV, влезает в 16 GB):
    #   system prompt          ~700 tok
    #   user template wrapper  ~100 tok
    #   transcript (4000 ch)   ~2000 tok        ← v11: половина от v10
    #   reasoning Qwen3.5      до 20000 tok (с большим запасом)
    #   output JSON            ~2000 tok
    #   ────────────────────   ~25000 tok → влезает в 32K с буфером
    #
    # Окно расширено до 32K через q8_0 KV cache (~520 MB вместо ~1 GB в f16).
    # JUDGE_MAX_INPUT_CHARS=4000 — снизили в 2× после прогонов v10: на больших
    # 8K-чанках Qwen3.5-9B иногда уходил в reasoning-loop и не успевал закрыть
    # финальный JSON в пределах max_tokens. Короче чанк ⇒ меньше кандидатов
    # ⇒ меньше шансов на петлю.
    # JUDGE_MAX_OUTPUT_TOKENS=24000 — длинный reasoning без обрезок.
    # ──────────────────────────────────────────────────────────────
    JUDGE_URL: str = "http://localhost:8002"
    JUDGE_MODEL_REPO: str = "unsloth/Qwen3.5-9B-GGUF"
    JUDGE_MODEL_FILE: str = "Qwen3.5-9B-Q4_K_M.gguf"
    JUDGE_MODEL_ID: str = "qwen3.5-9b-q4"
    JUDGE_NAME: str = "qwen3_5_9b_q4"
    JUDGE_REASONING: bool = True
    JUDGE_TEMPERATURE: float = 0.2
    JUDGE_MAX_OUTPUT_TOKENS: int = 24000
    JUDGE_MAX_INPUT_CHARS: int = 4000
    JUDGE_PROMPT_DIR: str = "prompts/judge"

    # ── Auth (Phase 2) ──────────────────────────────────────────────
    JWT_SECRET: str = "dev-insecure-change-me"   # MUST override in .env for prod
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 720                # 12h
    ADMIN_USERNAME: str = "admin"
    ADMIN_PASSWORD: str = ""                      # set in .env -> admin auto-created on startup
    AUTH_DB_PATH: str = ""                        # empty -> <repo>/auth/users.db

    class Config:
        env_file = ENV_PATH
        env_file_encoding = 'utf-8'
        # Игнорировать переменные из .env, которые нужны Docker-контейнерам,
        # но не Python-коду (например HF_TOKEN для скачивания моделей в
        # vibevoice-контейнере).
        extra = 'ignore'

settings = Settings()
