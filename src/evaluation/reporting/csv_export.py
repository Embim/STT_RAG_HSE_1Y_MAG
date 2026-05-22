"""Append metric rows to a CSV file in a single canonical schema."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

from evaluation.paths import RESULTS_DIR, ensure_dirs

CANONICAL_FIELDS: List[str] = [
    "ts",
    "run_name",
    "kind",          # "asr" or "rag"
    "model",         # backend name or RAG run id
    "item_id",
    "metric",
    "value",
    "extra",         # free-form json string
]


def write_rows(
    run_name: str,
    rows: Iterable[Dict[str, Any]],
    *,
    kind: str,
    out_path: Path | None = None,
) -> Path:
    """Persist evaluation rows to data/eval/results/{run_name}.csv.

    Each input row should contain at least: model, item_id, metric, value.
    Missing fields are filled with defaults; unknown fields go into `extra`.
    """
    ensure_dirs()
    if out_path is None:
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = RESULTS_DIR / f"{kind}_{_slug(run_name)}_{ts}.csv"

    is_new = not out_path.exists()
    # utf-8-sig пишет BOM в начале файла — это сигнал Excel/Windows-приложениям
    # открывать в UTF-8, а не в локальной CP1251. Без BOM кириллица в CSV
    # отображается как «РЎРѕСЃС‚РѕРёС‚» (двойная декодировка).
    with open(out_path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CANONICAL_FIELDS)
        if is_new:
            writer.writeheader()
        for raw in rows:
            row = _normalize(raw, run_name=run_name, kind=kind)
            writer.writerow(row)
    return out_path


def _normalize(raw: Dict[str, Any], *, run_name: str, kind: str) -> Dict[str, Any]:
    out = {f: "" for f in CANONICAL_FIELDS}
    out["ts"] = raw.get("ts") or datetime.now(tz=timezone.utc).isoformat()
    out["run_name"] = run_name
    out["kind"] = kind
    for k in ("model", "item_id", "metric", "value", "extra"):
        if k in raw and raw[k] is not None:
            out[k] = raw[k]
    return out


def _slug(s: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)[:80]
