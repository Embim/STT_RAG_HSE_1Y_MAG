"""
Status API routes.

Эндпоинты для проверки статуса системы.
"""

from fastapi import APIRouter
import time
import logging

from ..schemas import StatusResponse, PluginsResponse, SourceInfo, ProcessorInfo
from ...core.registry import PluginRegistry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/status", tags=["status"])

# Время запуска
start_time = time.time()

# Глобальные ссылки
store = None
config = None


def init_status(st, cfg):
    """Инициализирует компоненты статуса."""
    global store, config
    store = st
    config = cfg


@router.get("", response_model=StatusResponse)
async def get_status():
    """Возвращает статус системы."""
    registry = PluginRegistry()

    store_count = 0
    if store:
        try:
            store_count = store.count()
        except Exception:
            pass

    return StatusResponse(
        status="running",
        version="2.0.0",
        uptime=time.time() - start_time,
        sources=registry.list_sources(),
        processors=registry.list_processors(),
        embedders=registry.list_embedders(),
        stores=registry.list_stores(),
        store_count=store_count,
    )


@router.get("/plugins", response_model=PluginsResponse)
async def get_plugins():
    """Возвращает список доступных плагинов."""
    registry = PluginRegistry()

    sources = []
    for name in registry.list_sources():
        source_class = registry.get_source(name)
        if source_class:
            sources.append(SourceInfo(
                name=name,
                supported_types=[ct.value for ct in source_class.supported_content_types],
            ))

    processors = []
    for name in registry.list_processors():
        proc_class = registry.get_processor(name)
        if proc_class:
            processors.append(ProcessorInfo(
                name=name,
                input_types=[ct.value for ct in proc_class.input_types],
                output_type=proc_class.output_type.value,
            ))

    return PluginsResponse(
        sources=sources,
        processors=processors,
    )


@router.get("/health")
async def health_check():
    """Проверка здоровья сервиса."""
    return {"status": "healthy"}
