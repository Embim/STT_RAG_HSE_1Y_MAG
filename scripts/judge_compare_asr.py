"""Сравнить несколько ASR-моделей через LLM-judge на одной лекции.

Workflow:
  1. Один и тот же `.mp3/ogg/wav` транскрибируется N разными ASR-системами.
     Получаются N JSON-транскриптов с разными `text`/`segments`, но одним и
     тем же audio_path. Имя ASR кладётся в JSON-поле `asr_name` (или
     `asr_backend`/`model`/`backend`), либо передаётся CLI флагом
     `--asr-name <name>` при запуске судьи.
  2. На каждом транскрипте независимо прогоняется судья (одинаковый prompt,
     одинаковый seed, фиксированный набор настроек) — получается N CSV+JSONL.
  3. Этот скрипт объединяет все N CSV в один сравнительный отчёт. На выходе
     HTML с leaderboard'ом по Q-Score и per-item breakdown.

Использование:
    # Прогнать судью на двух транскриптах:
    python scripts/run_judge.py \\
        --transcript data/transcripts/cnn_whisper.json \\
        --prompt-version v10 --asr-name whisper-large-v3-turbo \\
        --run-name cnn_whisper

    python scripts/run_judge.py \\
        --transcript data/transcripts/cnn_qwen3.json \\
        --prompt-version v10 --asr-name qwen3-asr-1.7b \\
        --run-name cnn_qwen3

    # Сравнительный отчёт:
    python scripts/judge_compare_asr.py \\
        data/eval/results/judge_cnn_whisper_*.csv \\
        data/eval/results/judge_cnn_qwen3_*.csv \\
        --out data/eval/outputs/asr_compare_cnn

Открыть `data/eval/outputs/asr_compare_cnn/report.html` — там leaderboard
ASR-моделей по Q-Score, per-type density, per-item breakdown.
"""
from __future__ import annotations

import argparse
import glob
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from evaluation.reporting.judge_report import generate_report  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _expand(args: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw in args:
        matched = glob.glob(raw)
        if matched:
            out.extend(Path(m) for m in matched)
        else:
            out.append(Path(raw))
    return [p for p in out if p.exists()]


def _auto_jsonl_for(csv_paths: list[Path]) -> list[Path]:
    """Найти JSONL-сайдкары рядом с CSV.

    Имена не совпадают полностью: CSV пишется в конце (с timestamp окончания),
    JSONL — в начале (с timestamp старта). Совпадает префикс `judge_<run>_`.
    Берём все JSONL чей префикс совпадает до последнего `_<14digits>.csv`.
    """
    import re
    out: list[Path] = []
    ts_suffix = re.compile(r"_\d{8}_\d{6}$")
    for csv in csv_paths:
        same_stem = csv.with_suffix(".jsonl")
        if same_stem.exists():
            out.append(same_stem)
            continue
        # Попробовать без timestamp-суффикса
        stem = csv.stem
        prefix = ts_suffix.sub("", stem)
        if prefix == stem:
            continue
        candidates = sorted(csv.parent.glob(f"{prefix}_*.jsonl"))
        if candidates:
            out.append(candidates[-1])  # самый свежий
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csvs", nargs="+",
                   help="CSV-файлы прогонов разных ASR (можно glob)")
    p.add_argument("--out", type=Path, required=True,
                   help="Output папка для отчёта (HTML + metrics.json + report.csv)")
    p.add_argument("--jsonl", nargs="*", default=None,
                   help="JSONL сайдкары. По умолчанию ищем рядом с CSV.")
    args = p.parse_args()

    csv_paths = _expand(args.csvs)
    if len(csv_paths) < 2:
        logger.warning(
            "Похоже у вас только %d CSV. Сравнительный отчёт особо смысла "
            "не имеет — нужно ≥2 разных прогона с разными --asr-name.",
            len(csv_paths),
        )
    logger.info("Loading %d CSV file(s):", len(csv_paths))
    for c in csv_paths:
        logger.info("  %s", c)

    jsonl_paths = _expand(args.jsonl) if args.jsonl else _auto_jsonl_for(csv_paths)
    if jsonl_paths:
        logger.info("Found %d JSONL sidecar(s)", len(jsonl_paths))

    html = generate_report(csv_paths, args.out, jsonl_paths=jsonl_paths)
    logger.info("Done. Open %s in browser.", html)
    return 0


if __name__ == "__main__":
    sys.exit(main())
