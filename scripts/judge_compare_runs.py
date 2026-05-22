"""Сравнить два прогона judge на детерминизм.

Берёт два JSONL-сайдкара, по каждому chunk_id выводит:
  - overlap evidence (intersection / max set size)
  - delta completion_tokens, elapsed_sec
  - какие findings есть только в одном из прогонов

Использование:
    python scripts/judge_compare_runs.py \\
        data/eval/results/judge_v9_run1_*.jsonl \\
        data/eval/results/judge_v9_run2_*.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def _load(p: Path) -> dict:
    out = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            chunk = r["chunk_id"].split("_chunk_")[-1]
            out[chunk] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run1", type=Path, help="JSONL первого прогона")
    ap.add_argument("run2", type=Path, help="JSONL второго прогона")
    args = ap.parse_args()

    a = _load(args.run1)
    b = _load(args.run2)

    print(f"Run 1: {args.run1.name}")
    print(f"Run 2: {args.run2.name}")
    print()

    total_overlap = 0
    total_max = 0
    identical_chunks = 0

    for chunk in sorted(set(a) | set(b)):
        ra = a.get(chunk, {})
        rb = b.get(chunk, {})
        fa = ra.get("findings", []) or []
        fb = rb.get("findings", []) or []
        evs_a = set(f.get("evidence", "") for f in fa)
        evs_b = set(f.get("evidence", "") for f in fb)
        common = evs_a & evs_b
        only_a = evs_a - evs_b
        only_b = evs_b - evs_a
        max_set = max(len(evs_a), len(evs_b))
        if max_set > 0:
            total_overlap += len(common)
            total_max += max_set

        ct_a = ra.get("completion_tokens", 0)
        ct_b = rb.get("completion_tokens", 0)
        el_a = ra.get("elapsed_sec", 0)
        el_b = rb.get("elapsed_sec", 0)
        same_tokens = ct_a == ct_b
        same_evidence = evs_a == evs_b
        if same_evidence and same_tokens:
            identical_chunks += 1

        flag = "✓ IDENTICAL" if (same_evidence and same_tokens) else "✗ DIFFERS"
        print(f"=== chunk #{chunk}  {flag} ===")
        print(f"  Run1: findings={len(fa)}  completion_tok={ct_a}  elapsed={el_a:.1f}s")
        print(f"  Run2: findings={len(fb)}  completion_tok={ct_b}  elapsed={el_b:.1f}s")
        print(f"  Findings overlap: {len(common)}/{max_set} "
              f"({100*len(common)/max(1, max_set):.0f}%)")
        if only_a:
            print(f"  ONLY in Run 1 ({len(only_a)}):")
            for e in sorted(only_a)[:5]:
                print(f"    - {e!r}")
        if only_b:
            print(f"  ONLY in Run 2 ({len(only_b)}):")
            for e in sorted(only_b)[:5]:
                print(f"    - {e!r}")
        print()

    print("=" * 60)
    overlap_pct = 100 * total_overlap / max(1, total_max)
    print(f"TOTAL: {total_overlap}/{total_max} findings overlap "
          f"({overlap_pct:.1f}%)")
    print(f"Identical chunks (evidence + tokens): "
          f"{identical_chunks}/{len(set(a) | set(b))}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
