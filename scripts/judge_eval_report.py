"""CLI для перегенерации judge-eval отчёта (metrics.json + report.csv + report.html).

В обычном workflow отчёт **создаётся автоматически** в конце прогона
`evaluation.judge.runner` (в `data/eval/outputs/<csv_stem>/`). Этот скрипт
нужен если хочешь:
  - пересобрать сравнительный отчёт по нескольким CSV (разные prompt-версии
    или разные модели-судьи);
  - сохранить снапшот в отдельную папку через --out;
  - перегенерировать после правки порогов/стилизации в judge_report.py.

Запуск:
    # Дефолт — один прогон, отчёт рядом с CSV в data/eval/outputs/<csv_stem>/
    python scripts/judge_eval_report.py \\
        data/eval/results/judge_judge_v1_cnn_20260514_*.csv

    # Сравнительный отчёт по нескольким версиям промпта
    python scripts/judge_eval_report.py \\
        --out data/eval/outputs/judge_compare_v1_v2 \\
        data/eval/results/judge_judge_v1_*.csv \\
        data/eval/results/judge_judge_v2_*.csv

    # С указанием JSONL для ссылки в HTML (auto-detect если рядом с CSV
    # лежит файл с тем же stem'ом)
    python scripts/judge_eval_report.py \\
        --jsonl data/eval/results/judge_judge_v1_cnn_20260514_*.jsonl \\
        data/eval/results/judge_judge_v1_cnn_20260514_*.csv
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


def _expand(raw_list: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw in raw_list:
        matched = glob.glob(raw)
        if matched:
            out.extend(Path(m) for m in matched)
        else:
            out.append(Path(raw))
    return [p for p in out if p.exists()]


def _auto_jsonl_for(csv_paths: list[Path]) -> list[Path]:
    """Найти JSONL-сайдкары рядом с CSV.

    Имена отличаются timestamp-суффиксом — CSV пишется в конце прогона,
    JSONL в начале. Совпадает префикс `judge_<run-name>_`.
    """
    import re
    ts_suffix = re.compile(r"_\d{8}_\d{6}$")
    out: list[Path] = []
    for csv in csv_paths:
        same_stem = csv.with_suffix(".jsonl")
        if same_stem.exists():
            out.append(same_stem)
            continue
        prefix = ts_suffix.sub("", csv.stem)
        if prefix == csv.stem:
            continue
        candidates = sorted(csv.parent.glob(f"{prefix}_*.jsonl"))
        if candidates:
            out.append(candidates[-1])
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+",
                   help="CSV-файлы из data/eval/results/ (можно glob)")
    p.add_argument("--out", type=Path, default=None,
                   help="Output папка (по умолчанию data/eval/outputs/<csv_stem>/ "
                        "если один CSV, иначе data/eval/outputs/judge_combined/)")
    p.add_argument("--jsonl", nargs="*", default=None,
                   help="Параллельные JSONL-сайдкары для упоминания в HTML. "
                        "Если не указан, ищем рядом с CSV по тому же stem'у.")
    args = p.parse_args()

    csv_paths = _expand(args.inputs)
    if not csv_paths:
        logger.error("No input CSV found")
        return 2
    logger.info("Loading %d CSV file(s):", len(csv_paths))
    for pth in csv_paths:
        logger.info("  %s", pth)

    jsonl_paths: list[Path] | None
    if args.jsonl is not None:
        jsonl_paths = _expand(args.jsonl)
    else:
        jsonl_paths = _auto_jsonl_for(csv_paths)
    if jsonl_paths:
        logger.info("Linking %d JSONL sidecar(s) in HTML", len(jsonl_paths))

    if args.out is not None:
        out_dir = args.out
    elif len(csv_paths) == 1:
        out_dir = REPO_ROOT / "data" / "eval" / "outputs" / csv_paths[0].stem
    else:
        out_dir = REPO_ROOT / "data" / "eval" / "outputs" / "judge_combined"

    html = generate_report(csv_paths, out_dir, jsonl_paths=jsonl_paths)
    logger.info("Done. Open %s in browser.", html)
    return 0


if __name__ == "__main__":
    sys.exit(main())
