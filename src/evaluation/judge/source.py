"""Загрузка входных данных для судьи. Два режима источника:

  - `benchmark` — построчное чтение CSV в data/eval/results/asr_*.csv.
    Каждая строка содержит `reference` и `hypothesis` (в JSON-поле extra).
    Группируем строки по item_id (1 строка → много метрик, но reference+
    hypothesis одинаковые). На выходе — JudgeItem с обоими текстами.

  - `lecture` — JSON-транскрипция из data/transcripts/. Reference отсутствует
    (это то, что мы хотим оценить с judge как раз потому что нет ground truth).
    На выходе — JudgeItem с большим `hypothesis` и `reference=None`.

JudgeItem дальше уходит в chunker → reviewer.
"""
from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JudgeItem:
    """Один input для судьи: что-то, что надо проверить.

    Атрибуты:
        item_id: стабильный id (стек: source_kind + источник + локальный id).
        hypothesis: текст для разбора (ASR-вывод).
        reference: эталонный текст или None для reference-free режима.
        metadata: source-specific поля (model, duration, audio_path, ...).
    """
    item_id: str
    hypothesis: str
    reference: Optional[str] = None
    metadata: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────
# Benchmark source: data/eval/results/asr_*.csv
# ─────────────────────────────────────────────────────────────────

# Длинные имена runs могут содержать `:`/пробелы/кириллицу — оставляем как
# есть в item_id, judge_item_id сделаем безопасным позже.
_SAFE_ITEM_ID = re.compile(r"[^A-Za-z0-9._-]")


def _safe(raw: str) -> str:
    return _SAFE_ITEM_ID.sub("_", raw)


def iter_benchmark_csv(
    csv_paths: Iterable[Path],
    *,
    skip_corpus: bool = True,
    models_filter: Optional[List[str]] = None,
) -> Iterator[JudgeItem]:
    """Прочитать ASR-eval CSV и сгруппировать строки по (model, item_id).

    Каждая строка CSV — одна метрика (wer/cer/mer/wil) с дублированными
    reference+hypothesis в JSON-extra. Берём из каждой группы по одной строке.

    Args:
        csv_paths: пути к asr_*.csv (data/eval/results/).
        skip_corpus: пропускать ли строки с item_id="__corpus__" (агрегаты).
        models_filter: оставить только эти модели (например ["whisper_large_v3_turbo"]),
            None = все.
    """
    seen: set[tuple[str, str]] = set()  # (model, item_id) уже выдали

    for csv_path in csv_paths:
        if not csv_path.exists():
            logger.warning("CSV not found: %s", csv_path)
            continue
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                model = row.get("model", "")
                local_id = row.get("item_id", "")
                if not model or not local_id:
                    continue
                if skip_corpus and local_id == "__corpus__":
                    continue
                if models_filter and model not in models_filter:
                    continue
                key = (model, local_id)
                if key in seen:
                    continue
                seen.add(key)

                # extra — JSON-строка с reference/hypothesis/duration/asr_url.
                try:
                    extra = json.loads(row.get("extra", "") or "{}")
                except json.JSONDecodeError as e:
                    logger.warning(
                        "Bad extra json in %s row item_id=%s: %s",
                        csv_path.name, local_id, e,
                    )
                    continue
                hyp = str(extra.get("hypothesis", "")).strip()
                ref = str(extra.get("reference", "")).strip()
                if not hyp:
                    logger.debug("Skipping %s/%s: empty hypothesis", model, local_id)
                    continue

                composed_id = _safe(f"benchmark__{model}__{local_id}")
                yield JudgeItem(
                    item_id=composed_id,
                    hypothesis=hyp,
                    reference=ref or None,
                    metadata={
                        "source_kind": "benchmark",
                        "csv_path": str(csv_path),
                        "model": model,
                        "original_item_id": local_id,
                        "duration": extra.get("duration"),
                        "asr_url": extra.get("asr_url", ""),
                        "asr_model_id": extra.get("asr_model_id", ""),
                    },
                )


# ─────────────────────────────────────────────────────────────────
# Lecture source: data/transcripts/*.json (полные лекции, no reference)
# ─────────────────────────────────────────────────────────────────


def iter_lecture_jsons(
    paths: Iterable[Path],
    *,
    use_text_field: bool = True,
    asr_name_override: Optional[str] = None,
) -> Iterator[JudgeItem]:
    """Прочитать транскрипции лекций и вернуть по одному JudgeItem на файл.

    Args:
        paths: пути к .json (`data/transcripts/...`).
        use_text_field: если True, берём `text` (склеенный) — это полная
            лекция одним стрингом, дальше её рубит chunker. Если False —
            склеиваем `segments[*].text` (то же самое, но честнее по таймингам).
        asr_name_override: записать в metadata.model имя ASR-системы которая
            создала транскрипт. Нужно для сравнения ASR через judge:
            прогон judge на N разных транскриптах одной и той же лекции с
            разным asr_name даст N столбцов в leaderboard. Если в JSON
            самой лекции уже есть поле `asr_name`/`backend`/`model` — оно
            переопределяет CLI override.
    """
    for path in paths:
        if not path.exists():
            logger.warning("Lecture JSON not found: %s", path)
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.warning("Bad JSON in %s: %s", path, e)
            continue

        if use_text_field and data.get("text"):
            text = str(data["text"])
        else:
            text = " ".join(
                str(s.get("text", "")).strip()
                for s in data.get("segments", [])
                if s.get("text")
            )
        text = text.strip()
        if not text:
            logger.warning("Empty transcript: %s", path)
            continue

        doc_id = str(data.get("doc_id") or path.stem)
        title = str(data.get("title") or path.stem)
        # ASR-имя: приоритет JSON-meta → CLI override → пусто.
        asr_name = (
            data.get("asr_name") or data.get("asr_backend")
            or data.get("model") or data.get("backend")
            or asr_name_override or ""
        )
        composed_id = _safe(f"lecture__{path.stem}__{doc_id[:12]}")

        yield JudgeItem(
            item_id=composed_id,
            hypothesis=text,
            reference=None,
            metadata={
                "source_kind": "lecture",
                # `model` поле — стандарт для попадания в leaderboard (отчёт
                # ищет именно его в extra.asr_model → переименован в
                # extra.asr_model на стороне reviewer).
                "model": asr_name,
                "json_path": str(path),
                "doc_id": doc_id,
                "title": title,
                "language": data.get("language", ""),
                "n_segments": len(data.get("segments", [])),
                "n_chars": len(text),
            },
        )


def discover_csv(paths: Iterable[str]) -> List[Path]:
    """Распаковать список аргументов (могут быть как файлы так и директории)."""
    out: List[Path] = []
    for arg in paths:
        p = Path(arg)
        if p.is_dir():
            out.extend(sorted(p.glob("asr_*.csv")))
        elif p.is_file():
            out.append(p)
        else:
            # glob-like с подстановкой через shell у нас и так разобран, но
            # на всякий случай поддержим явный wildcard.
            base = p.parent if p.parent.parts else Path(".")
            out.extend(sorted(base.glob(p.name)))
    return out


def discover_lectures(paths: Iterable[str]) -> List[Path]:
    out: List[Path] = []
    for arg in paths:
        p = Path(arg)
        if p.is_dir():
            out.extend(sorted(p.glob("*.json")))
        elif p.is_file() and p.suffix == ".json":
            out.append(p)
    return out
