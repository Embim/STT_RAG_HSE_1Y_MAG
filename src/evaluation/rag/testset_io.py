"""Round-trip a testset between JSON (canonical) and CSV (Excel-editable).

The CSV is for humans to curate (mark verified=TRUE, fix wording, drop
hallucinated questions). reference_contexts are joined with `\n---\n` so they
fit a single CSV cell.
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

CTX_SEP = "\n---\n"
CSV_FIELDS = [
    "item_id",
    "question",
    "reference_answer",
    "reference_contexts",
    "verified",
    "auto_generated",
    "synthesizer",
    "notes",
]


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "верно", "да"}


def json_to_csv(json_path: Path, csv_path: Path | None = None) -> Path:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    items = payload.get("items", payload if isinstance(payload, list) else [])
    csv_path = csv_path or json_path.with_suffix(".csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for item in items:
            md = item.get("metadata") or {}
            writer.writerow({
                "item_id": item.get("item_id", ""),
                "question": item.get("question", ""),
                "reference_answer": item.get("reference_answer", ""),
                "reference_contexts": CTX_SEP.join(item.get("reference_contexts") or []),
                "verified": "TRUE" if md.get("verified") else "FALSE",
                "auto_generated": "TRUE" if md.get("auto_generated") else "FALSE",
                "synthesizer": md.get("synthesizer", ""),
                "notes": md.get("notes", ""),
            })
    logger.info("Wrote %s (%d rows)", csv_path, len(items))
    return csv_path


def csv_to_json(
    csv_path: Path,
    json_path: Path | None = None,
    *,
    only_verified: bool = True,
) -> Path:
    rows: List[Dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            verified = _truthy(row.get("verified", ""))
            if only_verified and not verified:
                continue
            ctxs = [
                c for c in (row.get("reference_contexts") or "").split(CTX_SEP)
                if c.strip()
            ]
            rows.append({
                "item_id": row["item_id"],
                "question": row.get("question", ""),
                "reference_answer": row.get("reference_answer", ""),
                "reference_contexts": ctxs,
                "metadata": {
                    "verified": verified,
                    "auto_generated": _truthy(row.get("auto_generated", "")),
                    "synthesizer": row.get("synthesizer", ""),
                    "notes": row.get("notes", ""),
                },
            })

    json_path = json_path or csv_path.with_suffix(".json")
    payload = {"size": len(rows), "items": rows, "source_csv": str(csv_path)}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %s (%d items)", json_path, len(rows))
    return json_path
