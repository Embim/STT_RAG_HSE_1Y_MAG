"""Convert RAGAS metric rows into three sinks: CSV + Langfuse Scores + MLflow."""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List

from evaluation.reporting import csv_export, langfuse_export
from processing.progress_tracker import tracked_run


def write_metric_rows(
    metric_rows: Iterable[Dict[str, Any]],
    *,
    run_name: str,
    collection: str,
):
    """Persist per-item metric rows to CSV, Langfuse, and MLflow.

    Returns the CSV path. All three sinks are independent and fail-soft.
    """
    csv_rows: List[Dict[str, Any]] = []
    rows = list(metric_rows)  # iterate twice (Langfuse + MLflow corpus aggregation)

    with tracked_run(
        title=run_name,
        kind="rag_eval",
        params={"collection": collection},
    ) as mlrun:
        for r in rows:
            csv_rows.append({
                "model": run_name,
                "item_id": r["item_id"],
                "metric": r["metric"],
                "value": r["value"],
                "extra": json.dumps({"collection": collection}),
            })
            is_corpus = r["item_id"] == "__corpus__"
            score_name = ("ragas_corpus_" if is_corpus else "ragas_") + r["metric"]
            # Langfuse: per-item scores attach to the pipeline trace; corpus stand-alone.
            langfuse_export.push_score(
                name=score_name,
                value=float(r["value"]),
                trace_id=r.get("trace_id") if not is_corpus else None,
                comment=f"{run_name}/{r['item_id']}",
            )
            # MLflow: log only corpus aggregates as run-level metrics (per-item
            # would explode the metric count and isn't useful for model comparison).
            if is_corpus:
                try:
                    mlrun.log_metric(f"corpus_{r['metric']}", float(r["value"]))
                except Exception:
                    pass

        # MLflow: aggregate counts for navigation
        mlrun.log_metric(
            "n_items",
            sum(1 for r in rows if r["item_id"] != "__corpus__")
            // max(1, len({r["metric"] for r in rows if r["item_id"] != "__corpus__"})),
        )

        out_path = csv_export.write_rows(run_name, csv_rows, kind="rag")
        mlrun.log_artifact(str(out_path))
        langfuse_export.flush()
        return out_path
