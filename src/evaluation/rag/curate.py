"""CLI: round-trip testset JSON ↔ CSV and optionally publish to Langfuse.

    python -m evaluation.rag.curate --json testset.json --to-csv
    python -m evaluation.rag.curate --csv  testset.csv  --to-json [--push-langfuse-dataset NAME]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from evaluation.rag.testset_io import csv_to_json, json_to_csv
from evaluation.rag.testset_publish import push_to_langfuse


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    p = argparse.ArgumentParser(prog="evaluation.rag.curate")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--json", dest="json_path", help="Convert this JSON to CSV")
    src.add_argument("--csv", dest="csv_path", help="Convert this CSV back to JSON")
    p.add_argument("--to-csv", action="store_true")
    p.add_argument("--to-json", action="store_true")
    p.add_argument("--include-rejected", action="store_true",
                   help="Keep items with verified=FALSE when CSV→JSON")
    p.add_argument("--push-langfuse-dataset", default=None,
                   help="After CSV→JSON, push items to a Langfuse dataset of this name")
    args = p.parse_args()

    if args.json_path and args.to_csv:
        json_to_csv(Path(args.json_path))
    elif args.csv_path and args.to_json:
        out = csv_to_json(Path(args.csv_path), only_verified=not args.include_rejected)
        if args.push_langfuse_dataset:
            push_to_langfuse(out, args.push_langfuse_dataset)
    else:
        p.error("specify --to-csv with --json or --to-json with --csv")


if __name__ == "__main__":
    main()
