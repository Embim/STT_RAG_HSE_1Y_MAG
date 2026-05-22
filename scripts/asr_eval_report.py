"""CLI для перегенерации ASR-eval отчёта.

В обычном workflow отчёт **создаётся автоматически** в конце прогона
`evaluation.asr.runner` (в `data/eval/outputs/`). Этот скрипт нужен
только если хочешь:
  - пересобрать отчёт с другим набором CSV (например, сравнительный)
  - сохранить снапшот в отдельную папку через --out
  - перегенерировать после правки порогов/стилизации в asr_report.py

По умолчанию пишет в `data/eval/outputs/` — три файла там перезаписываются
каждый прогон.

Запуск:
    # Дефолт — перезаписать data/eval/outputs/ из всех CSV
    python scripts/asr_eval_report.py data/eval/results/asr_*.csv

    # Снапшот в отдельную папку (например baseline до эксперимента)
    python scripts/asr_eval_report.py --out data/eval/outputs_baseline_v1 \
        data/eval/results/asr_*.csv
"""
from __future__ import annotations

import argparse
import glob
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from evaluation.reporting.asr_report import generate_report  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+",
                   help="CSV-файлы из data/eval/results/ (можно glob)")
    p.add_argument("--out", type=Path, default=None,
                   help="Output папка (по умолчанию data/eval/outputs)")
    args = p.parse_args()

    paths: list[Path] = []
    for raw in args.inputs:
        matched = glob.glob(raw)
        paths.extend(Path(m) for m in matched) if matched else paths.append(Path(raw))
    paths = [pth for pth in paths if pth.exists()]
    if not paths:
        logger.error("No input CSV found")
        return 2
    logger.info("Loading %d CSV file(s):", len(paths))
    for pth in paths:
        logger.info("  %s", pth)

    out_dir = args.out or (REPO_ROOT / "data" / "eval" / "outputs")
    html = generate_report(paths, out_dir)
    logger.info("Done. Open %s in browser.", html)
    return 0


if __name__ == "__main__":
    sys.exit(main())
