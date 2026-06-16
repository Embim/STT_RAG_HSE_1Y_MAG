"""ASR model registry — каталог доступных моделей распознавания речи.

Единый источник правды о том, какие ASR-бэкенды есть в проекте, как к ним
обращаться (model_id / endpoint), сколько они едят VRAM и когда какую лучше
выбирать. Используется тремя местами:

  - `system.asr_manager` — чтобы знать, какой docker-контейнер поднимать под
    выбранную пользователем модель и как обращаться к нему по HTTP.
  - `api.schemas.IngestRequest` — валидация поля `asr_model`.
  - `GET /asr-models` + Angular-фронт — выпадающий список с подсказками
    «когда какую использовать».

Соответствует профилям `asr-*` в docker-compose.yml и матрице
`scripts/run_asr_eval_matrix.py`. Подробный ресёрч — `docs/asr_backends.md`.

Все бэкенды OpenAI-совместимы и слушают один и тот же порт 8000 с сетевым
алиасом `asr` (эксклюзивны на одной GPU). Поэтому при выборе модели меняется
только `model_id`/`endpoint`/`name`, а `base_url` (WHISPER_URL) остаётся
прежним — за алиасом стоит тот контейнер, который сейчас поднят.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List


@dataclass(frozen=True)
class AsrModel:
    """Описание одного ASR-бэкенда.

    Args:
        key: стабильный идентификатор для API/фронта (`whisper`, `qwen3`, ...).
        profile: имя docker-compose профиля (`asr-whisper`).
        container: имя контейнера (`docker start`/`docker inspect`).
        model_id: значение поля `model` в OpenAI-запросе к бэкенду.
        name: человекочитаемая метка для MLflow/CSV (`qwen3_asr_1.7b`).
        endpoint: `transcription` (/v1/audio/transcriptions) или `chat`
            (/v1/chat/completions с аудио как data-URI — для Qwen3/Phi-4).
        language: язык по умолчанию (`ru`).
        label: отображаемое имя в UI.
        vram_gb: примерный объём VRAM.
        code_switch: краткое описание поддержки ru-en code-switching (для UI).
        when: подсказка «когда выбирать» (RU, user-facing).
        healthy_timeout: сколько ждать healthy после старта контейнера (сек) —
            cold start varies от ~1 мин (whisper) до ~30 мин (vibevoice).
        available: False → модель в каталоге, но временно нерабочая (phi4).
        recommended: дефолтный выбор для лекций с англицизмами.
    """
    key: str
    profile: str
    container: str
    model_id: str
    name: str
    endpoint: str
    language: str
    label: str
    vram_gb: float
    code_switch: str
    when: str
    healthy_timeout: int
    available: bool = True
    recommended: bool = False


# Порядок = порядок в выпадающем списке (рекомендованная — первой).
_MODELS: List[AsrModel] = [
    AsrModel(
        key="qwen3",
        profile="asr-qwen3",
        container="asr-qwen3",
        model_id="Qwen/Qwen3-ASR-1.7B",
        name="qwen3_asr_1.7b",
        endpoint="chat",
        language="ru",
        label="Qwen3-ASR 1.7B",
        vram_gb=4.0,
        code_switch="нативно",
        when=(
            "Рекомендуется для лекций с англицизмами. LLM-декодер с нативным "
            "ru-en code-switching: «attention», «embedding», «LightGBM» остаются "
            "латиницей, а не транслитерируются. Лучший баланс качества и VRAM "
            "для лекций по ML/DL."
        ),
        healthy_timeout=600,
        recommended=True,
    ),
    AsrModel(
        key="whisper",
        profile="asr-whisper",
        container="asr-whisper",
        model_id="deepdml/faster-whisper-large-v3-turbo-ct2",
        name="whisper_large_v3_turbo",
        endpoint="transcription",
        language="ru",
        label="Whisper large-v3 turbo",
        vram_gb=2.0,
        code_switch="через initial_prompt",
        when=(
            "Быстрый универсальный baseline и самый лёгкий прогрев (~1 мин). "
            "Английские термины тянет через подсказку-биас, но иногда всё же "
            "транслитерирует в кириллицу. Бери, когда важна скорость, а "
            "англицизмов в лекции немного."
        ),
        healthy_timeout=300,
    ),
    AsrModel(
        key="vibevoice",
        profile="asr-vibevoice",
        container="asr-vibevoice",
        model_id="microsoft/VibeVoice-ASR-HF",
        name="vibevoice_asr_bnb4",
        endpoint="transcription",
        language="ru",
        label="VibeVoice-ASR (4-bit)",
        vram_gb=6.0,
        code_switch="нативно + hotwords + диаризация",
        when=(
            "Максимальное качество для лекций с несколькими спикерами: нативный "
            "code-switching, hotwords (глоссарий терминов) и диаризация. Самая "
            "тяжёлая (~6 ГБ) и самый долгий прогрев (до ~30 мин при первом "
            "запуске). Примечание: транскрипция идёт чанками (см. настройку "
            "TRANSCRIBE_CHUNK_MINUTES), а не одним 60-мин проходом."
        ),
        healthy_timeout=1800,
    ),
    AsrModel(
        key="parakeet",
        profile="asr-parakeet",
        container="asr-parakeet",
        model_id="nvidia/parakeet-tdt-0.6b-v3",
        name="parakeet_tdt_v3",
        endpoint="transcription",
        language="ru",
        label="Parakeet-TDT 0.6B v3",
        vram_gb=2.0,
        code_switch="нет (только ru)",
        when=(
            "Чисто русский baseline без code-switching — фиксирует один язык на "
            "фразу. Подходит для лекций без англицизмов или как контрольная "
            "точка для сравнения; англоязычные термины запишет кириллицей."
        ),
        healthy_timeout=600,
    ),
    AsrModel(
        key="phi4",
        profile="asr-phi4-nvfp4",
        container="asr-phi4-nvfp4",
        model_id="nvidia/Phi-4-multimodal-instruct-NVFP4",
        name="phi4_mm_nvfp4",
        endpoint="chat",
        language="ru",
        label="Phi-4 Multimodal (NVFP4)",
        vram_gb=4.0,
        code_switch="через промпт",
        when=(
            "Мультимодальная LLM (FP4) с поддержкой глоссария через "
            "system-prompt. Сейчас недоступна: сломан путь /v1/chat в "
            "trtllm-serve 1.2.0rc6 (issue #14100/#14125)."
        ),
        healthy_timeout=1200,
        available=False,
    ),
]

ASR_MODELS: Dict[str, AsrModel] = {m.key: m for m in _MODELS}

# Дефолт совпадает с профилем `full` в docker-compose (поднимает asr-whisper)
# и со старым поведением (settings.ASR_NAME = whisper_large_v3_turbo).
DEFAULT_ASR_MODEL = "whisper"


def get_model(key: str) -> AsrModel:
    """Вернуть модель по ключу или бросить ValueError со списком доступных."""
    model = ASR_MODELS.get(key)
    if model is None:
        raise ValueError(
            f"unknown asr_model '{key}'; available: {', '.join(ASR_MODELS)}"
        )
    return model


def list_models() -> List[dict]:
    """Каталог для `GET /asr-models` — без внутренних docker-полей."""
    out = []
    for m in _MODELS:
        d = asdict(m)
        # profile/container/healthy_timeout — внутренняя кухня, наружу не нужны.
        for internal in ("profile", "container", "healthy_timeout"):
            d.pop(internal, None)
        out.append(d)
    return out
