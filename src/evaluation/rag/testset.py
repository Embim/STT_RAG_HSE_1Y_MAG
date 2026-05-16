"""CLI: generate a Q&A testset from transcripts."""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

from settings import settings
from evaluation.paths import TESTSET_DIR
from evaluation.rag.testset_generator import generate_testset


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    p = argparse.ArgumentParser(prog="evaluation.rag.testset")
    p.add_argument("--transcripts", required=True, help="Path to transcript file or directory")
    p.add_argument("--output", default=None, help="Output JSON path (default: testsets/ragas_<ts>.json)")
    p.add_argument("--size", type=int, default=settings.RAGAS_TESTSET_SIZE)
    args = p.parse_args()

    out = Path(args.output) if args.output else (
        TESTSET_DIR / f"ragas_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    )
    generate_testset(Path(args.transcripts), out, size=args.size)


if __name__ == "__main__":
    main()
