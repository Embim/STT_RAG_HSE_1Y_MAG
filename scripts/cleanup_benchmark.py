"""Постобработка локального ASR-бенчмарка: чистит шум в reference-текстах
и удаляет осиротевшие wav-файлы.

Применяется к манифесту, который уже собран `build_asr_benchmark.py from-json
--merge-target-duration N`. Делает в одном проходе:

  1. Schmlonk-cleanup: схлопывает многоточия, лишние пробелы, дедуплицирует
     повторяющиеся слова подряд (артефакт склейки соседних Whisper-сегментов).
  2. Substitutions: словарь типичных Whisper-ошибок на русско-английском
     code-switching домене (Wi-Fi, accuracy, Colab, ResNet и т.п.).
  3. Loop removal: детектит и удаляет сегменты, в которых Whisper зациклился
     (одно слово > 35% или триграмма повторяется 4+ раз).
  4. Choppy removal: удаляет сегменты, состоящие из коротких 1-2-словных
     псевдо-предложений (Whisper нарезал паузы/раздумья на куски).

Удаляет соответствующие .wav из подпапки `audio/`. Сохраняет статистику и
заметку в поле `cleaning_note` манифеста.

    python scripts/cleanup_benchmark.py data/eval/benchmarks/local/dl_cnn
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


_SUBS = [
    (r'\bАгураси\b',         'accuracy'),
    (r'\bагураси\b',         'accuracy'),
    (r'\bвай[\s-]?фай\b',    'Wi-Fi'),
    (r'\bфлэттен\b',         'flatten'),
    (r'\bфлеттен\b',         'flatten'),
    (r'\bколлаб\b',          'Colab'),
    (r'\bколаб\b',           'Colab'),
    (r'\bКоллаб\b',          'Colab'),
    (r'\bКалаб\b',           'Colab'),
    (r'\bКалабе\b',          'Colab'),
    (r'\bзафокатил',         'зафакапил'),
    (r'\bзафокапил',         'зафакапил'),
    (r'\bимэдж[\s-]?нет\b',  'ImageNet'),
    (r'\bимеджнет\b',        'ImageNet'),
    (r'\bимаджнет\b',        'ImageNet'),
    (r'\bдроп[\s-]?аут\b',   'dropout'),
    (r'\bдропаут\b',         'dropout'),
    (r'\bбатч[\s-]?норм',    'batch norm'),
    (r'\bпайторч\b',         'PyTorch'),
    (r'\bтензорфлоу\b',      'TensorFlow'),
    (r'\bкагл\b',            'Kaggle'),
    (r'\bкагле\b',           'Kaggle'),
    (r'\bконсорс\b',         'open source'),
    (r'\bконсорсный\b',      'опенсорсный'),
    (r'\bРесНет\b',          'ResNet'),
    (r'\bрезнет\b',          'ResNet'),
    (r'\bВГГ\b',             'VGG'),
    (r'\bЭлекс[Нн]ет\b',     'AlexNet'),
    (r'\bАлекс[Нн]ет\b',     'AlexNet'),
    # RecSys-специфичные
    (r'\bэксис\b',           'RecSys'),
    (r'\bРексис\b',          'RecSys'),
    (r'\bРэксис\b',          'RecSys'),
    (r'\bТебанке\b',         'Т-Банке'),
    (r'\bТебан\b',           'Т-Банк'),
    (r'\bТибан\b',           'Т-Банк'),
    (r'\bклубарн',           'коллаборативн'),
    (r'\bКлубарн',           'Коллаборативн'),
    # ML-специфичные
    (r'\bкатбуст\b',         'CatBoost'),
    (r'\bкэтбуст\b',         'CatBoost'),
    (r'\bлайтгбм\b',         'LightGBM'),
    (r'\bэксгибуст\b',       'XGBoost'),
    (r'\bхгбуст\b',          'XGBoost'),
    (r'\bадабуст\b',         'AdaBoost'),
    (r'\bбустинг\b',         'boosting'),
    (r'\bбэггинг\b',         'bagging'),
    (r'\bжини\b',             'Джини'),
]

_DOTS_RE = re.compile(r'\.{2,}')
_SPACE_RE = re.compile(r'\s+')
_DOT_SPACE_RE = re.compile(r'\s+([.,!?…])')
_REPEAT_DOT_RE = re.compile(r'(\.\s*){2,}')
_DUP_WORD_RE = re.compile(r'\b(\S+)\s+\1\b')

_WORD_RE = re.compile(r'[А-Яа-яёЁA-Za-z]+')
_SHORT_PHRASE_REPEAT_RE = re.compile(r'(\b\w{1,5}\.\s*){4,}')


def schlonk_cleanup(text: str) -> str:
    text = _DOTS_RE.sub('.', text)
    text = _DOT_SPACE_RE.sub(r'\1', text)
    text = _REPEAT_DOT_RE.sub('. ', text)
    text = _SPACE_RE.sub(' ', text).strip()
    text = _DUP_WORD_RE.sub(r'\1', text)
    return text


def apply_subs(text: str) -> str:
    for pat, repl in _SUBS:
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)
    return text


def is_looped(text: str) -> bool:
    words = _WORD_RE.findall(text.lower())
    if len(words) < 4:
        return False
    counts = Counter(words)
    most_word, most_count = counts.most_common(1)[0]
    if most_count / len(words) > 0.35 and len(words) >= 8:
        return True
    trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
    if trigrams:
        tg_count = Counter(trigrams).most_common(1)[0][1]
        if tg_count > 3:
            return True
    if _SHORT_PHRASE_REPEAT_RE.search(text):
        return True
    return False


def is_choppy(text: str) -> bool:
    sents = [s.strip() for s in re.split(r'[.!?…]+', text) if s.strip()]
    if len(sents) < 4:
        return False
    short = sum(1 for s in sents if len(_WORD_RE.findall(s)) <= 2)
    return short / len(sents) > 0.5


def cleanup(benchmark_dir: Path, *, min_chars: int = 80) -> Dict:
    mp = benchmark_dir / 'manifest.json'
    if not mp.exists():
        raise SystemExit(f"manifest.json not found in {benchmark_dir}")
    m = json.loads(mp.read_text(encoding='utf-8'))
    items = m.get('items', [])
    n_orig = len(items)

    # Pass 1: schlonk-cleanup + substitutions on every item
    n_subs = 0
    for it in items:
        before = it['reference']
        cleaned = apply_subs(schlonk_cleanup(before))
        if cleaned != before:
            n_subs += 1
        it['reference'] = cleaned

    # Pass 2: filter — minimum length, no loops, no choppy
    kept: List[Dict] = []
    removed: List[Tuple[Dict, str]] = []
    for it in items:
        ref = it['reference']
        if len(ref) < min_chars:
            removed.append((it, 'too_short'))
            continue
        if is_looped(ref):
            removed.append((it, 'loop'))
            continue
        if is_choppy(ref):
            removed.append((it, 'choppy'))
            continue
        kept.append(it)

    # Pass 3: delete orphaned wavs
    n_audio_removed = 0
    for it, _ in removed:
        ap = benchmark_dir / it['audio_path']
        if ap.exists():
            try:
                ap.unlink()
                n_audio_removed += 1
            except OSError:
                pass

    # Save
    m['items'] = kept
    m['size'] = len(kept)
    note = m.get('cleaning_note', '')
    by_reason = Counter(r for _, r in removed)
    m['cleaning_note'] = (
        (note + ' | ' if note else '') +
        f'Cleanup: {n_orig}->{len(kept)} '
        f'({len(removed)} removed: {dict(by_reason)}; subs in {n_subs} items).'
    )
    mp.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding='utf-8')

    return {
        'orig': n_orig,
        'kept': len(kept),
        'removed': len(removed),
        'by_reason': dict(by_reason),
        'subs': n_subs,
        'audio_removed': n_audio_removed,
        'duration_min': sum(i['duration'] for i in kept) / 60,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    p = argparse.ArgumentParser(prog='scripts/cleanup_benchmark.py')
    p.add_argument('benchmark_dir', help='Path to a benchmark directory (with manifest.json)')
    p.add_argument('--min-chars', type=int, default=80,
                   help='Drop items with reference shorter than this (default: 80)')
    args = p.parse_args()

    stats = cleanup(Path(args.benchmark_dir), min_chars=args.min_chars)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
