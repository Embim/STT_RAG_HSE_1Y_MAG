from typing import Dict, Any, List

from system.prompts import ALL_INFORMATION_FOR_ANSWER
from system.rag.vectore_store import VectorStoreManager

def _prepare_docs(pairs) -> list:
    docs = []

    for doc, _score in pairs:
        full_context = doc.metadata.get('full_context', doc.page_content)
        doc.page_content = full_context
        docs.append(doc)
    return docs
 
def _pack_results(
    docs: List[Any]
) -> Dict[str, Any]:

    text = "\n\n".join([doc.page_content for doc in docs])

    all_info_for_answer = ALL_INFORMATION_FOR_ANSWER.format(context = text)

    return all_info_for_answer


async def retrieve(
    vector_store_manager: VectorStoreManager, 
    query: str,
    k: int
) -> Dict[str, Any]:
    
    pairs = await vector_store_manager.search(query, k=k)

    result = _pack_results(_prepare_docs(pairs))

    return result