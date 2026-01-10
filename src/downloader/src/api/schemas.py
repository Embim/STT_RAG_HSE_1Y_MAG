"""
Pydantic schemas for API.

Модели запросов и ответов для FastAPI.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


# === Запросы ===

class SearchRequest(BaseModel):
    """Запрос на поиск."""
    query: str = Field(..., description="Поисковый запрос")
    limit: int = Field(10, ge=1, le=100, description="Количество результатов")
    filters: Optional[Dict[str, Any]] = Field(None, description="Фильтры")


class IngestRequest(BaseModel):
    """Запрос на загрузку данных."""
    source: str = Field(..., description="Имя источника (youtube, local_files)")
    params: Dict[str, Any] = Field(default_factory=dict, description="Параметры источника")


# === Ответы ===

class SearchResult(BaseModel):
    """Один результат поиска."""
    id: str
    score: float
    text: str
    source_id: str
    source_type: str
    title: str
    author: str
    chunk_index: int
    start_position: float
    end_position: float
    url: Optional[str] = None


class SearchResponse(BaseModel):
    """Ответ на поиск."""
    query: str
    results: List[SearchResult]
    total: int
    processing_time: float


class IngestResponse(BaseModel):
    """Ответ на загрузку данных."""
    success: bool
    message: str
    stats: Dict[str, Any]


class StatusResponse(BaseModel):
    """Статус системы."""
    status: str
    version: str
    uptime: float
    sources: List[str]
    processors: List[str]
    embedders: List[str]
    stores: List[str]
    store_count: int


class SourceInfo(BaseModel):
    """Информация об источнике."""
    name: str
    supported_types: List[str]


class ProcessorInfo(BaseModel):
    """Информация о процессоре."""
    name: str
    input_types: List[str]
    output_type: str


class PluginsResponse(BaseModel):
    """Список доступных плагинов."""
    sources: List[SourceInfo]
    processors: List[ProcessorInfo]


class ErrorResponse(BaseModel):
    """Ответ с ошибкой."""
    error: str
    detail: Optional[str] = None
