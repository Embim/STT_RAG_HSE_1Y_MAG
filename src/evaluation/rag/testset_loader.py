"""Load a curated testset JSON file into a list of items."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def load_testset(path: Path) -> List[Dict[str, Any]]:
    """Return the items list from a testset JSON, supporting both shapes.

    - {"items": [...]}  (canonical)
    - [...]             (bare list — older curate output)
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    return payload.get("items", [])
