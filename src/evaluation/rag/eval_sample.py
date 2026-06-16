"""Single shape passed between RAG eval components."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class EvalSample:
    """One Q&A item ready for RAGAS scoring.

    `contexts` are the chunks the live retriever returned; `reference_*` are
    the human-curated gold for reference-based metrics. `trace_id` is the
    Langfuse trace id of the pipeline.run that produced this sample, so we
    can attach metric scores back to the same trace in the UI.
    """
    item_id: str
    question: str
    answer: str
    contexts: List[str]
    reference_answer: Optional[str] = None
    reference_contexts: Optional[List[str]] = None
    trace_id: Optional[str] = None
