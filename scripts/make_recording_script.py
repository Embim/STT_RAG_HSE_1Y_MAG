"""Подготовка скрипта для голосовой записи друзьями.

Чтобы собрать честный human-recorded ASR-бенчмарк (где reference написан до
записи и не зависит от модели), нужно дать людям список разумно длинных
связных фраз для начитки. Этот скрипт берёт JSON-транскрипты или manifest'ы,
**склеивает соседние сегменты** в осмысленные блоки 10-20 секунд звучания,
фильтрует мусор (filler-слова в начале, оборванные на полуслове, без
пунктуации, и т.д.), и выдаёт два файла:

    script.txt   — для человека: пронумерованный список фраз, который друг
                   читает по одной, отправляя голосовое за каждой
    script.csv   — для нашего pipeline'а: filename + reference, скармливается
                   в `scripts/build_asr_benchmark.py from-folder`

Дальше:

    # 1. Сгенерировать (один раз):
    uv run python scripts/make_recording_script.py \
        --from-json data/transcripts/*.json \
        --count 50 \
        --output data/eval/recordings/shared

    # 2. Отправить script.txt друзьям. Они записывают голосовые в Telegram
    #    по одной фразе на сообщение, в порядке нумерации.

    # 3. Скачать .ogg, разложить под именами 001.ogg, 002.ogg, ...

    # 4. Собрать манифест:
    uv run python scripts/build_asr_benchmark.py from-folder \
        --audio-dir       data/eval/recordings/aleks/audio \
        --references-csv  data/eval/recordings/shared/script.csv \
        --output          data/eval/benchmarks/local/human_aleks \
        --reference-source "human:aleks"
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Слова, с которых не должна начинаться фраза для начитки — будет звучать
# обрывочно без контекста.
_BAD_START = {
    "вот", "ну", "там", "то", "это", "значит", "потому", "поэтому", "и",
    "а", "но", "да", "или", "также", "чтобы", "что", "как", "когда", "где",
    "если", "кто", "тогда", "так", "тут", "здесь", "вообще", "просто",
    "хорошо", "ладно", "да-да", "угу", "нет", "его", "её", "их", "его",
    "ему", "ей", "им", "ого", "хм", "уф",
}

_FILLER_RE = re.compile(
    r"\b(вот|ну|там|то\s+есть|значит|короче|типа|в\s+общем|в\s+целом|как-то|"
    r"например|собственно|на\s+самом\s+деле|в\s+принципе|допустим)\b",
    flags=re.IGNORECASE,
)


def _load_segments_from_manifest(manifest_path: Path) -> List[Dict[str, Any]]:
    """Из manifest.json достаём reference + duration как один сегмент."""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    out = []
    for it in payload.get("items", []):
        out.append({
            "text": it["reference"],
            "start": 0.0,
            "end": it.get("duration", 0.0),
            "source": payload.get("source", str(manifest_path)),
        })
    return out


def _load_segments_from_json_transcript(json_path: Path) -> List[Dict[str, Any]]:
    """Из JSON-транскрипта (формат faster-whisper-server) — все сегменты."""
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    segments = payload.get("segments", [])
    src_label = payload.get("title") or json_path.stem
    out = []
    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        start = float(seg.get("start", 0.0) or 0.0)
        end = float(seg.get("end", start) or start)
        out.append({"text": text, "start": start, "end": end, "source": src_label})
    return out


def _merge_neighbours(
    segments: List[Dict[str, Any]],
    *,
    target_duration: float,
    max_duration: float,
    max_chars: int,
) -> List[Dict[str, Any]]:
    """Склеиваем соседние сегменты в один блок, пока он не достигнет цели.

    Ключевая идея: один сегмент — короткая обрывочная фраза. Склеив 3-5
    подряд идущих, получаем осмысленный кусок длительностью ~15 секунд.
    Склеиваем только в пределах одного источника (не через границу файлов).
    """
    merged: List[Dict[str, Any]] = []
    cur_text: List[str] = []
    cur_start: Optional[float] = None
    cur_end: Optional[float] = None
    cur_source: Optional[str] = None

    def flush():
        nonlocal cur_text, cur_start, cur_end, cur_source
        if cur_text and cur_start is not None and cur_end is not None:
            merged.append({
                "text": " ".join(cur_text).strip(),
                "start": cur_start,
                "end": cur_end,
                "source": cur_source or "",
            })
        cur_text = []
        cur_start = None
        cur_end = None
        cur_source = None

    prev_end: Optional[float] = None
    for seg in segments:
        # граница источника или большой gap = flush
        same_source = cur_source is None or cur_source == seg["source"]
        gap_ok = prev_end is None or (seg["start"] - prev_end) < 2.0
        if not same_source or not gap_ok:
            flush()

        if cur_start is None:
            cur_start = seg["start"]
            cur_source = seg["source"]
        cur_text.append(seg["text"])
        cur_end = seg["end"]

        block_dur = cur_end - cur_start
        block_chars = sum(len(t) for t in cur_text) + max(0, len(cur_text) - 1)

        # Достигли цели — flush, если в конце уже знак препинания
        last_chunk = cur_text[-1].rstrip()
        ends_clean = last_chunk and last_chunk[-1] in ".!?…"
        too_long = block_dur >= max_duration or block_chars >= max_chars
        target_reached = block_dur >= target_duration and ends_clean
        if too_long or target_reached:
            flush()
        prev_end = seg["end"]

    flush()
    return merged


def _quality_filter(
    blocks: List[Dict[str, Any]],
    *,
    min_chars: int,
    max_chars: int,
    min_duration: float,
    max_duration: float,
) -> List[Dict[str, Any]]:
    """Эвристический отбор осмысленных кусков.

    Работаем на склеенных блоках (после `_merge_neighbours`). Пропускаем:
      - те, что не вписались в длину/длительность
      - которые начинаются с filler-слова (оборвыш без контекста)
      - без точки/?!… в конце (мысль не закончена)
      - где доля filler-слов слишком высокая (бессмысленный поток)
      - без длинных содержательных слов (>=8 букв) — обычно это пустая болтовня
    """
    out = []
    seen = set()
    for b in blocks:
        text = b["text"].strip()
        # схлопываем повторяющиеся пробелы
        text = re.sub(r"\s+", " ", text)
        # схлопываем дубли запятых ("вот, вот, " → ", ")
        text = re.sub(r"(,\s*){2,}", ", ", text)
        text = text.strip(", ")

        duration = b["end"] - b["start"]
        if not (min_chars <= len(text) <= max_chars):
            continue
        if duration > 0 and not (min_duration <= duration <= max_duration):
            continue

        # начинается с приличного слова
        first_word = re.split(r"\s+", text, maxsplit=1)[0].strip(".,;:!?…«»\"'()[]").lower()
        if first_word in _BAD_START:
            continue
        # должна начинаться с большой буквы (в русской раскладке)
        if not text[:1].isupper():
            continue
        # заканчивается знаком препинания (не оборвыш)
        if text[-1] not in ".!?…":
            continue
        # есть хотя бы одно длинное слово (содержательный термин)
        if not any(len(w) >= 8 for w in re.findall(r"[А-Яа-яёЁA-Za-z]+", text)):
            continue
        # не слишком много filler'ов
        words = re.findall(r"[А-Яа-яёЁA-Za-z]+", text)
        if not words:
            continue
        filler_count = len(_FILLER_RE.findall(text))
        if filler_count / max(1, len(words)) > 0.18:
            continue

        norm = text.lower()
        if norm in seen:
            continue
        seen.add(norm)
        out.append({**b, "text": text})
    return out


def make_script(
    sources: List[Path],
    *,
    output_dir: Path,
    count: int,
    target_duration: float = 12.0,
    min_duration: float = 8.0,
    max_duration: float = 22.0,
    min_chars: int = 120,
    max_chars: int = 380,
    seed: int = 42,
) -> tuple[Path, Path]:
    """Главный pipeline: source → segments → merged blocks → filtered → sample."""
    all_segments: List[Dict[str, Any]] = []
    for src in sources:
        if src.name == "manifest.json":
            all_segments.extend(_load_segments_from_manifest(src))
        else:
            all_segments.extend(_load_segments_from_json_transcript(src))

    if not all_segments:
        raise SystemExit(f"Не нашёл сегментов в {sources}")

    merged = _merge_neighbours(
        all_segments,
        target_duration=target_duration,
        max_duration=max_duration,
        max_chars=max_chars,
    )
    logger.info("После склейки соседних сегментов: %d блоков", len(merged))

    filtered = _quality_filter(
        merged,
        min_chars=min_chars, max_chars=max_chars,
        min_duration=min_duration, max_duration=max_duration,
    )
    logger.info("После фильтра качества: %d блоков", len(filtered))

    if not filtered:
        raise SystemExit("После фильтрации не осталось фраз. Расширь диапазоны.")

    # стараемся брать из разных источников равномерно
    rng = random.Random(seed)
    rng.shuffle(filtered)
    by_source: Dict[str, List[Dict[str, Any]]] = {}
    for f in filtered:
        by_source.setdefault(f["source"], []).append(f)
    chosen: List[Dict[str, Any]] = []
    while len(chosen) < count and any(by_source.values()):
        for src in list(by_source.keys()):
            if not by_source[src]:
                continue
            chosen.append(by_source[src].pop(0))
            if len(chosen) >= count:
                break

    if len(chosen) < count:
        logger.warning("Запрошено %d фраз, доступно только %d", count, len(chosen))

    output_dir.mkdir(parents=True, exist_ok=True)

    txt_path = output_dir / "script.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(
            "Скрипт для записи голосовых сообщений\n"
            "=====================================\n"
            "Запиши по одному голосовому на каждую фразу, в порядке нумерации\n"
            "(001, 002, ...). Читай естественно, как в обычном разговоре,\n"
            "не нужно стараться \"чисто\" артикулировать. Если ошибся —\n"
            "переотправь голосовое для этого номера, остальные не трогай.\n"
            "После всех записей пришли заархивированной папкой или просто\n"
            "по очереди.\n\n"
        )
        for i, p in enumerate(chosen, start=1):
            dur = p["end"] - p["start"] if p["end"] > 0 else 0.0
            f.write(f"{i:03d}.  ({dur:.0f} сек)  {p['text']}\n\n")
    logger.info("Wrote %s (%d phrases)", txt_path, len(chosen))

    csv_path = output_dir / "script.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "reference"])
        writer.writeheader()
        for i, p in enumerate(chosen, start=1):
            writer.writerow({"filename": f"{i:03d}", "reference": p["text"]})
    logger.info("Wrote %s", csv_path)

    return txt_path, csv_path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scripts/make_recording_script.py")
    p.add_argument("--from-manifest", action="append", default=[],
                   help="Путь к manifest.json (можно указать несколько раз). "
                        "Сегменты в манифесте уже короткие — склейка слабее.")
    p.add_argument("--from-json", action="append", default=[],
                   help="Путь к JSON-транскрипту с segments[] (можно несколько раз). "
                        "Предпочтительнее: больше сырых сегментов для умной склейки.")
    p.add_argument("--output", required=True, help="Папка для script.txt и script.csv")
    p.add_argument("--count", type=int, default=50)
    p.add_argument("--target-duration", type=float, default=12.0,
                   help="Целевая длительность блока, секунд (default: 12).")
    p.add_argument("--min-duration", type=float, default=8.0)
    p.add_argument("--max-duration", type=float, default=22.0)
    p.add_argument("--min-chars", type=int, default=120)
    p.add_argument("--max-chars", type=int, default=380)
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args()
    sources = [Path(s) for s in (args.from_manifest + args.from_json)]
    if not sources:
        raise SystemExit("Укажи --from-manifest или --from-json")
    make_script(
        sources,
        output_dir=Path(args.output),
        count=args.count,
        target_duration=args.target_duration,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
