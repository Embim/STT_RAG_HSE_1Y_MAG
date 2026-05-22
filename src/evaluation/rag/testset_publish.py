"""Publish a curated testset JSON to a Langfuse Dataset."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from evaluation.reporting import langfuse_export

logger = logging.getLogger(__name__)


def push_to_langfuse(testset_json: Path, dataset_name: str) -> None:
    """Create the dataset (idempotent) and upload every item from testset_json."""
    payload = json.loads(testset_json.read_text(encoding="utf-8"))
    items = payload.get("items", [])
    langfuse_export.ensure_dataset(dataset_name, description=f"From {testset_json.name}")
    for item in items:
        langfuse_export.add_dataset_item(
            dataset_name,
            item_id=item["item_id"],
            input_payload={"question": item["question"]},
            expected_output={
                "reference_answer": item.get("reference_answer", ""),
                "reference_contexts": item.get("reference_contexts", []),
            },
            metadata=item.get("metadata", {}),
        )
    langfuse_export.flush()
    logger.info("Pushed %d items to Langfuse dataset %s", len(items), dataset_name)
