import logging
from typing import Any, List

from system.rag.vectore_store import VectorStoreManager
from system.tracing import observe

logger = logging.getLogger(__name__)


def _format_ts(seconds: Any) -> str | None:
    if seconds is None:
        return None
    try:
        total = max(0, int(float(seconds)))
    except (TypeError, ValueError):
        return None
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _prepare_docs(pairs) -> list:
    docs = []

    for doc, _score in pairs:
        full_context = doc.metadata.get('full_context', doc.page_content)
        doc.page_content = full_context
        docs.append(doc)
    return docs
 
def _pack_results(
    docs: List[Any]
) -> str:
    blocks: List[str] = []
    for doc in docs:
        start_ts = _format_ts(doc.metadata.get("start_sec"))
        end_ts = _format_ts(doc.metadata.get("end_sec"))
        source_url = doc.metadata.get("source_url")
        source_file_name = doc.metadata.get("source_file_name")
        title = doc.metadata.get("title")

        header_parts: List[str] = []
        if start_ts and end_ts:
            header_parts.append(f"{start_ts}-{end_ts}")
        source_label = source_file_name or title
        if source_label:
            header_parts.append(str(source_label))
        if source_url:
            header_parts.append(str(source_url))

        if header_parts:
            blocks.append(f"[{' | '.join(header_parts)}]\n{doc.page_content}")
        else:
            blocks.append(doc.page_content)

    return "\n\n".join(blocks)


@observe(name="vector-retrieval")
async def retrieve(
    vector_store_manager: VectorStoreManager,
    query: str,
    k: int,
    similarity_threshold: float,
    source_file_name: str | None = None,
    source_title: str | None = None,
) -> str:
    pairs = await vector_store_manager.search(
        query,
        k=k,
        similarity_threshold=similarity_threshold,
        source_file_name=source_file_name,
        source_title=source_title,
    )
    logger.info(
        "Retrieved %d/%d chunks (threshold=%.2f, source_file_name=%r, source_title=%r)",
        len(pairs),
        k,
        similarity_threshold,
        source_file_name,
        source_title,
    )
    return _pack_results(_prepare_docs(pairs))