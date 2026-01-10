"""
Search API routes.

Эндпоинты для семантического поиска.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import time
import logging

from ..schemas import SearchRequest, SearchResponse, SearchResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["search"])


# Глобальные ссылки на компоненты (инициализируются в main.py)
embedder = None
store = None


def init_search(emb, st):
    """Инициализирует компоненты поиска."""
    global embedder, store
    embedder = emb
    store = st


@router.get("", response_model=SearchResponse)
async def search(
    q: str = Query(..., description="Поисковый запрос"),
    limit: int = Query(10, ge=1, le=100, description="Количество результатов"),
):
    """
    Семантический поиск по базе.

    Генерирует эмбеддинг для запроса и ищет похожие чанки.
    """
    if not embedder or not store:
        raise HTTPException(status_code=503, detail="Search not initialized")

    start_time = time.time()

    try:
        # Генерируем эмбеддинг для запроса (is_query=True для FRIDA)
        query_embedding = embedder.embed_single(q, is_query=True)

        # Ищем в хранилище
        raw_results = store.search(query_embedding, limit=limit)

        # Форматируем результаты
        results = []
        for r in raw_results:
            result = SearchResult(
                id=r.get("id", ""),
                score=r.get("score", 0),
                text=r.get("text", ""),
                source_id=r.get("source_id", ""),
                source_type=r.get("source_type", ""),
                title=r.get("title", ""),
                author=r.get("author", ""),
                chunk_index=r.get("chunk_index", 0),
                start_position=r.get("start_position", 0),
                end_position=r.get("end_position", 0),
                url=r.get("url"),
            )
            results.append(result)

        processing_time = time.time() - start_time

        return SearchResponse(
            query=q,
            results=results,
            total=len(results),
            processing_time=processing_time,
        )

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("", response_model=SearchResponse)
async def search_post(request: SearchRequest):
    """
    Семантический поиск (POST версия).

    Позволяет передать фильтры в теле запроса.
    """
    if not embedder or not store:
        raise HTTPException(status_code=503, detail="Search not initialized")

    start_time = time.time()

    try:
        query_embedding = embedder.embed_single(request.query, is_query=True)
        raw_results = store.search(
            query_embedding,
            limit=request.limit,
            filters=request.filters,
        )

        results = []
        for r in raw_results:
            result = SearchResult(
                id=r.get("id", ""),
                score=r.get("score", 0),
                text=r.get("text", ""),
                source_id=r.get("source_id", ""),
                source_type=r.get("source_type", ""),
                title=r.get("title", ""),
                author=r.get("author", ""),
                chunk_index=r.get("chunk_index", 0),
                start_position=r.get("start_position", 0),
                end_position=r.get("end_position", 0),
                url=r.get("url"),
            )
            results.append(result)

        processing_time = time.time() - start_time

        return SearchResponse(
            query=request.query,
            results=results,
            total=len(results),
            processing_time=processing_time,
        )

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
