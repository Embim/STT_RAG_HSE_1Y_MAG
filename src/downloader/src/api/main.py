"""
FastAPI application.

Главное приложение для REST API.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from .routes import search, status
from ..core.config import Config, ConfigLoader
from ..core.registry import PluginRegistry
from ..embedders.sentence_transformer import SentenceTransformerEmbedder
from ..stores.weaviate import WeaviateStore

logger = logging.getLogger(__name__)

# Глобальные компоненты
embedder = None
store = None
config = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager для FastAPI."""
    global embedder, store, config

    logger.info("Starting API server...")

    # Загружаем конфигурацию
    config = ConfigLoader.load("config/config.yaml")

    # Инициализируем реестр плагинов
    registry = PluginRegistry()
    registry.discover_all()

    # Инициализируем эмбеддер
    embedder = SentenceTransformerEmbedder(
        model_name=config.embedder_model,
        device=config.embedder_device,
        batch_size=config.embedder_batch_size,
        max_length=config.embedder_max_length,
        normalize=config.embedder_normalize,
        use_fp16=config.embedder_use_fp16,
    )
    embedder.setup({})

    # Инициализируем хранилище
    store = WeaviateStore(
        url=config.store_url,
        collection_name=config.store_collection,
        vector_dimension=config.store_vector_dimension,
    )
    store.connect()

    # Инициализируем роуты
    search.init_search(embedder, store)
    status.init_status(store, config)

    logger.info("API server started")

    yield

    # Shutdown
    logger.info("Shutting down API server...")

    if store:
        store.close()
    if embedder:
        embedder.teardown()

    logger.info("API server stopped")


# Создаём приложение
app = FastAPI(
    title="Video Pipeline API",
    description="REST API для семантического поиска по обработанному контенту",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключаем роуты
app.include_router(search.router)
app.include_router(status.router)


@app.get("/")
async def root():
    """Корневой эндпоинт."""
    return {
        "name": "Video Pipeline API",
        "version": "2.0.0",
        "docs": "/docs",
    }


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Запускает сервер."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
