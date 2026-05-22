"""Push evaluation results to Langfuse Datasets + Scores.

Уровень: Langfuse SDK v4.x (current). API в v4 принципиально span-based, а не
"create_score standalone" как в v2/v3. Эти 4 факта про v4 диктуют форму этого
модуля:

  1. `create_score(...)` БЕЗ trace_id/observation_id → API возвращает
     400 "Bad request". Все scores ДОЛЖНЫ быть прикреплены к span'у.
  2. `start_as_current_observation(as_type=..., ...)` — единственный
     поддерживаемый способ создать новый span. Возвращает context manager.
  3. Внутри активного span'а `score_current_span(name=, value=)` сам
     находит current trace_id/observation_id из OpenTelemetry context.
  4. `create_dataset_item(id=...)` PATCH'ит item по URL-path, и если в id
     есть ':' (например `local:shared_dl_ml::german_003`), сервер не может
     распарсить URL → возвращает 404 "Dataset item ... not found".
     Поэтому ID мы санитайзим перед каждым вызовом.

Defensive: если Langfuse disabled (noop client) или SDK call падает — log
warning и продолжаем, чтобы CSV всё равно строился.
"""
from __future__ import annotations

import contextlib
import logging
import re
from typing import Any, Dict, Iterator, Optional

from system.tracing import get_client

logger = logging.getLogger(__name__)

# Langfuse server PATCH'ит item через URL path `/api/v1/datasets/{name}/items/{id}`.
# Двоеточия + ещё какие-то URL-reserved символы ломают parsing на стороне
# сервера. Sanitize заранее. Этим же фильтром санитайзим имена span'ов /
# benchmark labels — пусть будет consistent.
_ID_UNSAFE_RE = re.compile(r"[^A-Za-z0-9._-]")


def _safe_id(raw: str) -> str:
    """Заменить URL-небезопасные символы на `_` для использования в Langfuse IDs."""
    return _ID_UNSAFE_RE.sub("_", raw)


def _client_or_none():
    client = get_client()
    # Noop client дёргать нет смысла — он возвращает None из всего.
    if client.__class__.__name__ == "_NoopLangfuseClient":
        return None
    return client


def ensure_dataset(name: str, description: str = "") -> Optional[str]:
    """Создать dataset (idempotent — повторный create на duplicate возвращает ok)."""
    client = _client_or_none()
    if client is None:
        logger.info("Langfuse disabled; skipping dataset %r", name)
        return None
    try:
        client.create_dataset(name=name, description=description)
    except Exception as e:
        # SDK кидает на duplicates — это норм.
        logger.debug("create_dataset(%r) raised (ok if exists): %s", name, e)
    return name


@contextlib.contextmanager
def start_run_span(
    *,
    name: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Iterator[Optional[Any]]:
    """Open-level span охватывающий ВЕСЬ benchmark run.

    Args:
        name: человеко-читаемое имя run'а в Langfuse UI
        metadata: backend name, url, model_id, benchmark_label, ...

    Использование:
        with langfuse_export.start_run_span(name="asr_whisper", metadata={...}):
            for sample in samples:
                ...   # item-spans + scores создаются внутри

        Корпусные scores (`asr_corpus_wer`) можно лить через
        `score_current(...)` пока активен внешний span.

    Если Langfuse disabled — yield None, всё внутри работает без span'а.
    """
    client = _client_or_none()
    if client is None:
        yield None
        return
    try:
        ctx = client.start_as_current_observation(
            name=_safe_id(name),
            as_type="evaluator",   # семантический тип "evaluation run"
            metadata=metadata or {},
        )
    except Exception as e:
        logger.warning("Langfuse start_run_span(%s) failed: %s — continuing without span", name, e)
        yield None
        return
    with ctx as span:
        try:
            yield span
        finally:
            pass


@contextlib.contextmanager
def start_item_span(
    *,
    name: str,
    audio_path: str,
    reference: str,
    hypothesis: str = "",
    extra_input: Optional[Dict[str, Any]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Iterator[Optional[Any]]:
    """Inner span на ОДИН sample (1 audio = 1 transcription = 1 set of scores).

    Создаётся как ребёнок текущего span'а (run-level), благодаря чему
    `score_current_span(...)` внутри прикрепит scores именно к этому item'у
    (видно в Langfuse UI как scores на каждом sample отдельно).

    Args:
        name: id sample'а (будет санитайзено)
        audio_path: путь к аудио (для input)
        reference: ground truth текст
        hypothesis: предсказанный текст backend'а (можно set'ить после
            transcribe через span.update(output={...}))
        extra_input: добавочные поля в input (duration, asr_url, ...)
        extra_metadata: добавочные поля в metadata
    """
    client = _client_or_none()
    if client is None:
        yield None
        return

    in_payload: Dict[str, Any] = {"audio_path": audio_path, "reference": reference}
    if extra_input:
        in_payload.update(extra_input)
    out_payload: Dict[str, Any] = {"hypothesis": hypothesis} if hypothesis else {}

    try:
        ctx = client.start_as_current_observation(
            name=_safe_id(name),
            as_type="span",
            input=in_payload,
            output=out_payload or None,
            metadata=extra_metadata or {},
        )
    except Exception as e:
        logger.warning("Langfuse start_item_span(%s) failed: %s — continuing", name, e)
        yield None
        return
    with ctx as span:
        try:
            yield span
        finally:
            pass


@contextlib.contextmanager
def start_judge_chunk_span(
    *,
    chunk_id: str,
    chunk_text: str,
    reference: Optional[str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Iterator[Optional[Any]]:
    """Inner span на один chunk LLM-judge'а.

    Семантически отличается от `start_item_span` тем, что у судьи нет audio
    pipeline — на вход транскрипт и опциональный reference, на выход findings.
    Используем as_type="generation" чтобы Langfuse UI показывал input/output
    как LLM-call (с подсветкой токенов).
    """
    client = _client_or_none()
    if client is None:
        yield None
        return

    in_payload: Dict[str, Any] = {"chunk_text": chunk_text}
    if reference:
        in_payload["reference"] = reference

    try:
        ctx = client.start_as_current_observation(
            name=_safe_id(chunk_id),
            as_type="generation",
            input=in_payload,
            metadata=extra_metadata or {},
        )
    except Exception as e:
        logger.warning(
            "Langfuse start_judge_chunk_span(%s) failed: %s — continuing",
            chunk_id, e,
        )
        yield None
        return
    with ctx as span:
        try:
            yield span
        finally:
            pass


def update_current_output(output: Dict[str, Any]) -> None:
    """Update output of currently-active span (для записи hypothesis после transcribe)."""
    client = _client_or_none()
    if client is None:
        return
    try:
        client.update_current_span(output=output)
    except Exception as e:
        logger.debug("update_current_span failed: %s", e)


def score_current(
    *,
    name: str,
    value: float,
    comment: Optional[str] = None,
    data_type: str = "NUMERIC",
) -> None:
    """Attach score к currently-active span (auto-resolves trace_id).

    Безопасно вне span'а — silently no-op (если current span отсутствует,
    SDK сам не падает, но и nothing полезного не делает).
    """
    client = _client_or_none()
    if client is None:
        return
    try:
        client.score_current_span(
            name=_safe_id(name),
            value=value,
            comment=comment,
            data_type=data_type,
        )
    except Exception as e:
        logger.warning("score_current(%s=%s) failed: %s", name, value, e)


def add_dataset_item(
    dataset_name: str,
    *,
    item_id: str,
    input_payload: Dict[str, Any],
    expected_output: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    source_trace_id: Optional[str] = None,
) -> None:
    """Создать dataset_item, прилинкованный к source trace (если есть).

    Если source_trace_id задан → linkнем item к этому trace (в UI на странице
    item видно "Source trace" с переходом). Если None → SDK по-старому,
    standalone item.

    item_id принудительно санитайзится — `:` и прочие URL-небезопасные
    символы заменяются на `_` чтобы избежать 404 от REST API.
    """
    client = _client_or_none()
    if client is None:
        return
    safe_id = _safe_id(item_id)
    try:
        client.create_dataset_item(
            dataset_name=dataset_name,
            id=safe_id,
            input=input_payload,
            expected_output=expected_output,
            metadata=metadata or {},
            source_trace_id=source_trace_id,
        )
    except Exception as e:
        logger.warning("create_dataset_item failed for %s/%s: %s", dataset_name, safe_id, e)


def current_trace_id() -> Optional[str]:
    """Получить trace_id текущего active span (для source-link в dataset_item)."""
    client = _client_or_none()
    if client is None:
        return None
    try:
        return client.get_current_trace_id()
    except Exception as e:
        logger.debug("get_current_trace_id failed: %s", e)
        return None


def flush() -> None:
    client = _client_or_none()
    if client is None:
        return
    try:
        client.flush()
    except Exception as e:
        logger.warning("Langfuse flush failed: %s", e)


# ---- Backwards-compat shim ----------------------------------------------
# Старый код может звать push_score(name=, value=). Мы это поддерживаем как
# alias на score_current — но БЕЗ trace_id/observation_id это no-op (warning).
def push_score(
    *,
    name: str,
    value: float,
    trace_id: Optional[str] = None,
    observation_id: Optional[str] = None,
    comment: Optional[str] = None,
    data_type: str = "NUMERIC",
) -> None:
    """[DEPRECATED] Use start_item_span(...) + score_current(...) instead.

    Если передан trace_id или observation_id — пробуем создать score через
    legacy API (`create_score`). Если нет — пробуем score_current() (требует
    активного span'а). Иначе silently skip.
    """
    client = _client_or_none()
    if client is None:
        return
    if trace_id or observation_id:
        try:
            kwargs: Dict[str, Any] = {"name": _safe_id(name), "value": value, "data_type": data_type}
            if trace_id:
                kwargs["trace_id"] = trace_id
            if observation_id:
                kwargs["observation_id"] = observation_id
            if comment:
                kwargs["comment"] = comment
            client.create_score(**kwargs)
            return
        except Exception as e:
            logger.warning("create_score(%s=%s) failed: %s", name, value, e)
            return
    # Без trace_id — пробуем повесить на текущий span (если есть).
    score_current(name=name, value=value, comment=comment, data_type=data_type)
