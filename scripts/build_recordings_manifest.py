"""Сборка ASR-бенчмарка из записанных друзьями голосовых.

Структура входа:
    data/eval/recordings/<benchmark>/
        script.txt                 # 50 пронумерованных фраз 001..050
        raw_telegram/
            <speaker_slug>/
                001.ogg            # i-й файл = i-я фраза по порядку записи
                ...
                050.ogg

На выходе пишет рядом со script.txt:
    manifest.json                  # combined: 4 диктора × 50 фраз = 200 items
    manifest_<speaker>.json        # per-speaker, по 50 items

Каждый item:
    {
        "id":          "<speaker>_<phrase_id>",
        "audio_path":  "raw_telegram/<speaker>/NNN.ogg",  # relative to manifest
        "reference":   "<текст фразы из script.txt>",
        "duration":    <секунды через ffprobe>,
        "verified":    true,            # тексты курированы вручную
        "speaker":     "<speaker_slug>",
        "phrase_id":   "NNN"
    }

Запуск (из корня репо):

    python scripts/build_recordings_manifest.py \
        --recordings-dir data/eval/recordings/shared_dl_ml \
        --speakers german izbrannoe nikolay_moroz sergey_kovynev

Дефолтные параметры заточены под shared_dl_ml — можно запускать без аргументов.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

PHRASE_RE = re.compile(r"^\s*(\d{3})\.\s+(.+?)\s*$")


def parse_script(path: Path) -> Dict[str, str]:
    """Прочитать script.txt → {"001": "текст фразы", ...}.

    Каждая фраза — одна строка, начинающаяся с "NNN. ". Пустые строки и
    заголовок игнорируются.
    """
    phrases: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = PHRASE_RE.match(line)
        if m:
            phrases[m.group(1)] = m.group(2)
    if not phrases:
        raise SystemExit(f"No phrases parsed from {path}")
    return phrases


def ffprobe_duration(audio: Path) -> float:
    """Длительность в секундах через ffprobe; 0.0 если не получилось."""
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(audio),
            ],
            check=True, capture_output=True, text=True,
        )
        return round(float(out.stdout.strip()), 3)
    except (subprocess.CalledProcessError, ValueError) as e:
        logger.warning("ffprobe failed on %s: %s", audio, e)
        return 0.0


def build_speaker_items(
    speaker: str,
    speaker_dir: Path,
    phrases: Dict[str, str],
    recordings_root: Path,
) -> List[dict]:
    """Собрать items для одного диктора."""
    items: List[dict] = []
    files = sorted(speaker_dir.glob("*.ogg"))
    if not files:
        logger.warning("No .ogg files in %s, skipping", speaker_dir)
        return items
    for f in files:
        phrase_id = f.stem  # "001"
        ref = phrases.get(phrase_id)
        if ref is None:
            logger.warning("No phrase %s in script for %s/%s", phrase_id, speaker, f.name)
            continue
        rel_audio = f.relative_to(recordings_root).as_posix()
        items.append({
            "id": f"{speaker}_{phrase_id}",
            "audio_path": rel_audio,
            "reference": ref,
            "duration": ffprobe_duration(f),
            "verified": True,
            "speaker": speaker,
            "phrase_id": phrase_id,
        })
    return items


def write_manifest(path: Path, items: List[dict], *, source: str, ref_source: str) -> None:
    payload = {
        "source": source,
        "reference_source": ref_source,
        "size": len(items),
        "items": items,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %s with %d items", path, len(items))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    repo_root = Path(__file__).resolve().parent.parent
    default_dir = repo_root / "data" / "eval" / "recordings" / "shared_dl_ml"

    parser = argparse.ArgumentParser(prog="build_recordings_manifest.py")
    parser.add_argument(
        "--recordings-dir", type=Path, default=default_dir,
        help="Папка с script.txt и raw_telegram/<speaker>/*.ogg внутри",
    )
    parser.add_argument(
        "--speakers", nargs="+",
        default=["german", "izbrannoe", "nikolay_moroz", "sergey_kovynev"],
        help="Имена подпапок в raw_telegram/",
    )
    parser.add_argument(
        "--script", type=Path, default=None,
        help="Путь к script.txt (default: <recordings-dir>/script.txt)",
    )
    args = parser.parse_args()

    rec_dir: Path = args.recordings_dir.resolve()
    script_path: Path = (args.script or (rec_dir / "script.txt")).resolve()
    raw_dir = rec_dir / "raw_telegram"

    phrases = parse_script(script_path)
    logger.info("Parsed %d phrases from %s", len(phrases), script_path)

    combined: List[dict] = []
    for speaker in args.speakers:
        speaker_dir = raw_dir / speaker
        if not speaker_dir.is_dir():
            logger.warning("Speaker dir missing: %s", speaker_dir)
            continue
        items = build_speaker_items(speaker, speaker_dir, phrases, rec_dir)
        write_manifest(
            rec_dir / f"manifest_{speaker}.json",
            items,
            source=f"shared_dl_ml/{speaker}",
            ref_source="manual_script_v1",
        )
        combined.extend(items)

    write_manifest(
        rec_dir / "manifest.json",
        combined,
        source="shared_dl_ml",
        ref_source="manual_script_v1",
    )


if __name__ == "__main__":
    main()
