"""Filesystem paths used across evaluation."""
from __future__ import annotations

from pathlib import Path

from settings import settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_ROOT = PROJECT_ROOT / settings.EVAL_DATA_DIR
BENCHMARK_DIR = EVAL_ROOT / "benchmarks"
TESTSET_DIR = EVAL_ROOT / "testsets"
RESULTS_DIR = EVAL_ROOT / "results"


def ensure_dirs() -> None:
    for d in (EVAL_ROOT, BENCHMARK_DIR, TESTSET_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
