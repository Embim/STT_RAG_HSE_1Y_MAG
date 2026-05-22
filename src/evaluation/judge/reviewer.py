"""Оркестратор разбора: source-item → chunker → judge.chat → parse + write sinks.

Контракт промпта v1 — модель возвращает строгий JSON `{"findings": [...]}`.
Парсер устойчив к лёгким префиксам/суффиксам (markdown ```json```, лишние
пояснения) — режет от первой `{` до последней `}`, потом json.loads. Если
не парсится — записываем "raw" findings со специальным error_type="parse_error",
чтобы человек мог разглядеть проблему в JSONL-сайдкаре и поправить промпт.

Sinks:
  - CSV (canonical schema): per-chunk row для каждого error_type, value=count.
    Используется существующий evaluation.reporting.csv_export.write_rows.
  - JSONL sidecar: полный findings + reasoning + raw response. Это главный
    артефакт для тюнинга — открываешь и видишь как модель думает.
  - Console: краткий лог по каждому чанку (счётчики ошибок).
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from tqdm import tqdm

from evaluation.judge.chunker import TextChunk, chunk_by_sentences
from evaluation.judge.client import JudgeClient, JudgeResponse
from evaluation.judge.prompt_loader import JudgePrompt
from evaluation.judge.source import JudgeItem
from evaluation.paths import RESULTS_DIR, ensure_dirs
from evaluation.reporting import csv_export, langfuse_export
from processing.progress_tracker import mlflow_span
from settings import settings

logger = logging.getLogger(__name__)

# Жадно вырезаем JSON между первой `{` и последней `}` — обходит markdown
# обёртки ```json...``` и любые комментарии модели до/после.
_JSON_GREEDY = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class Finding:
    """Один find — нормализованное представление того что вернула модель."""
    error_type: str
    severity: str
    evidence: str
    suggestion: str = ""
    confidence: float = 0.0
    explanation: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ChunkReview:
    """Полный результат разбора одного чанка."""
    chunk_id: str
    item_id: str
    findings: List[Finding] = field(default_factory=list)
    reasoning: str = ""
    raw_content: str = ""
    parse_error: Optional[str] = None
    elapsed_sec: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Подразбивка completion на reasoning + финальный output (v11+).
    # reasoning_tokens — содержимое <think>...</think>; output_tokens — финальный
    # content/JSON. completion_tokens == reasoning_tokens + output_tokens.
    reasoning_tokens: int = 0
    output_tokens: int = 0
    # Скорости стадий из timings блока llama.cpp.
    # prefill (input) обычно ~1000+ t/s, decode (reasoning+output) ~80-100 t/s.
    # Хранятся отдельно потому что префилл и декод физически разные стадии.
    prefill_ms: float = 0.0
    prefill_tps: float = 0.0
    decode_ms: float = 0.0
    decode_tps: float = 0.0
    # SGR (v7+): структурированный анализ кандидатов до фильтрации
    # (analysis.candidates + analysis.summary). None если промпт не v7.
    analysis: Optional[Dict[str, Any]] = None


# Severity → числовая шкала для CSV-метрик. Удобно потом усреднять/сравнивать.
_SEVERITY_SCORE = {"low": 1.0, "medium": 2.0, "high": 3.0}


def parse_findings(
    raw_content: str,
    *,
    allowed_error_types: List[str],
    allowed_severity: List[str],
) -> tuple[List[Finding], Optional[str], Optional[Dict[str, Any]]]:
    """Из текста модели извлечь список Finding'ов + SGR analysis (если есть).

    Returns:
        (findings, parse_error, analysis).
        parse_error=None если всё ок, иначе строка с диагностикой.
        analysis=None для старых промптов (v1-v6), dict для v7+ с SGR-схемой
        (`{"candidates": [...], "summary": "..."}`).
    """
    if not raw_content.strip():
        return [], "empty content", None

    m = _JSON_GREEDY.search(raw_content)
    if not m:
        return [], "no JSON object found in content", None

    candidate = m.group(0)
    try:
        data = json.loads(candidate)
        recovery_note = None
    except json.JSONDecodeError as primary_err:
        # Retry: модель часто упирается в max_tokens посреди генерации
        # последнего finding — JSON получается оборванным. Пытаемся
        # отрезать незавершённый хвост и достроить закрытие.
        repaired = _repair_truncated_findings_json(raw_content)
        if repaired is None:
            return [], f"JSONDecodeError: {primary_err}", None
        try:
            data = json.loads(repaired)
            recovery_note = f"recovered from truncation ({primary_err})"
            logger.info("parse_findings: recovered truncated JSON")
        except json.JSONDecodeError as e2:
            return ([], f"JSONDecodeError: {primary_err}; repair also failed: {e2}",
                    None)

    if not isinstance(data, dict) or "findings" not in data:
        return [], "missing top-level `findings` array", None
    raw_findings = data["findings"]
    if not isinstance(raw_findings, list):
        return [], "`findings` is not a list", None

    # SGR (v7+): извлекаем analysis блок если есть.
    analysis = data.get("analysis") if isinstance(data.get("analysis"), dict) else None

    out: List[Finding] = []
    bad_types: List[str] = []
    for f in raw_findings:
        if not isinstance(f, dict):
            continue
        etype = str(f.get("error_type", "")).strip().lower()
        if etype not in allowed_error_types:
            bad_types.append(etype)
            continue
        severity = str(f.get("severity", "")).strip().lower()
        if severity not in allowed_severity:
            severity = "medium"  # дефолт, чтобы не выкидывать findings из-за severity
        try:
            confidence = float(f.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        out.append(Finding(
            error_type=etype,
            severity=severity,
            evidence=str(f.get("evidence", "")).strip(),
            suggestion=str(f.get("suggestion", "")).strip(),
            confidence=max(0.0, min(1.0, confidence)),
            explanation=str(f.get("explanation", "")).strip(),
        ))
    err = None
    if bad_types:
        err = f"dropped findings with disallowed error_type: {set(bad_types)}"
    if recovery_note:
        # Не маскируем восстановление как success — записываем в parse_error
        # чтобы при анализе видеть какие чанки были обрезаны и сколько
        # findings потеряно.
        err = (f"{err}; {recovery_note}" if err else recovery_note)
    return out, err, analysis


def _repair_truncated_findings_json(raw: str) -> Optional[str]:
    """Восстановить оборванный по max_tokens JSON вида {"findings": [...]}.

    Стратегия:
      1. Найти начало `{"findings"` (или `{"findings`).
      2. Идти по строке, считать `{` и `[`, отслеживать внутри ли мы строки.
      3. На первом `{` верхнего уровня НЕЗАВЕРШЁННОГО finding (там где
         кончился raw_content без закрытия) — обрезать до запятой перед ним.
      4. Достроить `]}` чтобы получить валидный `{"findings": [...]}`.

    Возвращает строку-кандидата или None если структура не распознана.
    """
    start = raw.find('"findings"')
    if start == -1:
        return None
    # Найти открывающий { перед "findings"
    brace_pos = raw.rfind("{", 0, start)
    if brace_pos == -1:
        return None
    # Найти открывающий [ массива findings ПОСЛЕ "findings"
    bracket_pos = raw.find("[", start)
    if bracket_pos == -1:
        return None

    # Парсим посимвольно, отслеживая глубину объектов внутри массива.
    # Цель: найти позицию последнего полностью завершённого `}` верхнего
    # уровня внутри массива (т.е. конец предыдущего finding'а).
    s = raw
    i = bracket_pos + 1
    n = len(s)
    in_string = False
    escape = False
    depth = 0  # глубина {} внутри массива
    last_complete_obj_end = -1  # позиция сразу ПОСЛЕ закрывающей `}` последнего объекта на верхнем уровне массива

    while i < n:
        c = s[i]
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
        else:
            if c == '"':
                in_string = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    last_complete_obj_end = i + 1
            elif c == "]" and depth == 0:
                # Закрытие массива найдено — но раз мы зашли в repair,
                # значит дальше всё-таки сломан JSON. Доверяем последнему
                # полностью закрытому объекту.
                break
        i += 1

    if last_complete_obj_end == -1:
        # Ни одного полного finding'а не сгенерировано до обрыва.
        # Возвращаем пустой массив, чтобы не терять диагностику.
        return s[brace_pos:bracket_pos + 1] + "]}"

    return s[brace_pos:last_complete_obj_end] + "]}"


async def review_chunk(
    chunk: TextChunk,
    item: JudgeItem,
    *,
    client: JudgeClient,
    prompt: JudgePrompt,
) -> ChunkReview:
    """Один LLM-вызов + парсинг."""
    started = time.monotonic()
    # Reference (если есть) подсовываем только в первом чанке, чтобы не дублировать
    # один и тот же эталон во все промпты — судья делает корпусную сверку, а на
    # уровне отдельных чанков reference обычно мешает (фрагмент эталона ≠ фрагмент
    # гипотезы на одинаковых смещениях, особенно для длинных лекций).
    reference = item.reference if (item.reference and item.metadata.get("source_kind") == "benchmark") else None
    messages = prompt.render_messages(
        transcript_text=chunk.text,
        reference=reference,
        context_label=chunk.chunk_id,
    )
    hints = prompt.model_hints or {}
    temperature = float(hints.get("temperature", settings.JUDGE_TEMPERATURE))
    max_tokens = int(hints.get("max_tokens", settings.JUDGE_MAX_OUTPUT_TOKENS))
    enable_thinking = bool(hints.get("enable_thinking", True))

    # Опциональные sampling/anti-loop параметры из YAML model_hints.
    # Если не указаны — оставляем None (бэкенд использует дефолт).
    def _f(k):  # safe float or None
        v = hints.get(k)
        return float(v) if v is not None else None

    def _i(k):
        v = hints.get(k)
        return int(v) if v is not None else None

    try:
        resp: JudgeResponse = await client.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            top_p=_f("top_p"),
            top_k=_i("top_k"),
            min_p=_f("min_p"),
            frequency_penalty=_f("frequency_penalty"),
            presence_penalty=_f("presence_penalty"),
            repetition_penalty=_f("repetition_penalty"),
            seed=_i("seed"),
            response_format=hints.get("response_format"),
        )
    except Exception as e:
        logger.error("Judge call failed for %s: %s", chunk.chunk_id, e)
        return ChunkReview(
            chunk_id=chunk.chunk_id,
            item_id=item.item_id,
            parse_error=f"http error: {e}",
            elapsed_sec=time.monotonic() - started,
        )

    findings, parse_error, analysis = parse_findings(
        resp.content,
        allowed_error_types=prompt.allowed_error_types,
        allowed_severity=prompt.allowed_severity,
    )
    return ChunkReview(
        chunk_id=chunk.chunk_id,
        item_id=item.item_id,
        findings=findings,
        reasoning=resp.reasoning,
        raw_content=resp.content,
        parse_error=parse_error,
        elapsed_sec=time.monotonic() - started,
        prompt_tokens=resp.prompt_tokens,
        completion_tokens=resp.completion_tokens,
        reasoning_tokens=resp.reasoning_tokens,
        output_tokens=resp.output_tokens,
        prefill_ms=resp.prefill_ms,
        prefill_tps=resp.prefill_tps,
        decode_ms=resp.decode_ms,
        decode_tps=resp.decode_tps,
        analysis=analysis,
    )


def _csv_rows_for_chunk(
    review: ChunkReview,
    item: JudgeItem,
    *,
    allowed_error_types: List[str],
    judge_name: str,
    prompt_version: str,
) -> List[Dict[str, Any]]:
    """CSV-строки для одного чанка: count + avg_severity на каждый error_type.

    canonical schema csv_export.write_rows ждёт {model, item_id, metric, value, extra}.
    Мы:
      - кодируем judge_name в `model` (чтобы можно было сравнивать разных судей);
      - chunk_id в `item_id`;
      - метрики: count_<error_type>, sev_avg_<error_type>, total_findings.
      - в `extra` кладём json с findings (для отладки прямо в CSV) +
        token usage + parse_error.
    """
    counts = {et: 0 for et in allowed_error_types}
    sev_sums = {et: 0.0 for et in allowed_error_types}
    for f in review.findings:
        if f.error_type in counts:
            counts[f.error_type] += 1
            sev_sums[f.error_type] += _SEVERITY_SCORE.get(f.severity, 0.0)

    extra = {
        "source_kind": item.metadata.get("source_kind", ""),
        "asr_model": item.metadata.get("model", ""),
        "asr_model_id": item.metadata.get("asr_model_id", ""),
        "original_item_id": item.metadata.get("original_item_id", ""),
        "title": item.metadata.get("title", ""),
        "n_findings": len(review.findings),
        "prompt_tokens": review.prompt_tokens,
        "completion_tokens": review.completion_tokens,
        # v11+ per-stage breakdown: см. ChunkReview docstring.
        "reasoning_tokens": review.reasoning_tokens,
        "output_tokens": review.output_tokens,
        "prefill_ms": round(review.prefill_ms, 1),
        "prefill_tps": round(review.prefill_tps, 1),
        "decode_ms": round(review.decode_ms, 1),
        "decode_tps": round(review.decode_tps, 1),
        "elapsed_sec": round(review.elapsed_sec, 2),
        "parse_error": review.parse_error or "",
        "findings": [f.as_dict() for f in review.findings],
        "prompt_version": prompt_version,
    }
    extra_json = json.dumps(extra, ensure_ascii=False)

    rows: List[Dict[str, Any]] = []
    for et in allowed_error_types:
        rows.append({
            "model": judge_name,
            "item_id": review.chunk_id,
            "metric": f"count_{et}",
            "value": float(counts[et]),
            "extra": extra_json,
        })
        avg = sev_sums[et] / counts[et] if counts[et] else 0.0
        rows.append({
            "model": judge_name,
            "item_id": review.chunk_id,
            "metric": f"sev_avg_{et}",
            "value": round(avg, 3),
            "extra": extra_json,
        })
    rows.append({
        "model": judge_name,
        "item_id": review.chunk_id,
        "metric": "n_findings_total",
        "value": float(len(review.findings)),
        "extra": extra_json,
    })
    return rows


@dataclass
class ReviewRun:
    """Возвращается из review_items — путь к CSV + JSONL + агрегаты."""
    csv_path: Optional[Path]
    jsonl_path: Path
    n_items: int
    n_chunks: int
    n_findings: int
    counts_by_type: Dict[str, int]


async def review_items(
    items: Iterable[JudgeItem],
    *,
    client: JudgeClient,
    prompt: JudgePrompt,
    run_name: str,
    judge_name: str = "",
    max_chars_per_chunk: Optional[int] = None,
    max_chunks_per_item: Optional[int] = None,
    mlrun: Any = None,
) -> ReviewRun:
    """Главный entry-point: проитерировать items, разбить на чанки, разобрать.

    Args:
        items: iterable JudgeItem (из source.iter_*)
        client: настроенный JudgeClient
        prompt: загруженный JudgePrompt
        run_name: попадает в имена CSV/JSONL
        judge_name: метка модели-судьи (default — settings.JUDGE_NAME)
        max_chars_per_chunk: override settings.JUDGE_MAX_INPUT_CHARS
        max_chunks_per_item: остановиться после N чанков каждого item — для
            быстрой проверки промпта на 2-3 чанках без ожидания всех 17.
        mlrun: optional MLflow handle (`_ActiveRun` или `_NoopRun` из
            `processing.progress_tracker`). Если задан, per-chunk метрики
            льются как time-series со `step=chunk_idx` для time-line view
            в MLflow UI. None / noop = ничего не пишем, остальная логика
            работает идентично.
    """
    ensure_dirs()
    judge_name = judge_name or settings.JUDGE_NAME
    max_chars = max_chars_per_chunk or settings.JUDGE_MAX_INPUT_CHARS

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    jsonl_path = RESULTS_DIR / f"judge_{_slug(run_name)}_{ts}.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    counts_by_type: Dict[str, int] = {et: 0 for et in prompt.allowed_error_types}
    n_items = 0
    n_chunks = 0
    n_findings = 0

    # Открываем JSONL в режиме append — пишем сразу после каждого чанка чтобы
    # при падении посередине прогона у нас уже были partial результаты.
    with open(jsonl_path, "a", encoding="utf-8") as jsonl_f:
        for item in items:
            n_items += 1
            chunks = chunk_by_sentences(
                item.hypothesis, source_id=item.item_id, max_chars=max_chars,
            )
            total_chunks_in_item = len(chunks)
            if max_chunks_per_item is not None and max_chunks_per_item > 0:
                chunks = chunks[:max_chunks_per_item]
            logger.info(
                "[%s] %d chunks (of %d, ref=%s, source=%s)",
                item.item_id, len(chunks), total_chunks_in_item,
                bool(item.reference), item.metadata.get("source_kind", ""),
            )

            # tqdm-бар для visual прогресса по чанкам этого item. desc — короткий
            # tag (ASR-модель + первые 12 символов doc_id), postfix обновляется
            # каждый чанк с актуальной статистикой.
            asr_tag = (item.metadata.get("model")
                       or item.metadata.get("asr_model_id")
                       or "judge")[:20]
            doc_short = item.item_id.split("__")[-1][:12]
            pbar = tqdm(
                chunks,
                desc=f"{asr_tag:<20}│{doc_short}",
                unit="chunk",
                ncols=120,
                leave=True,
                # Длинный bar_format — видим bar / %% / count / elapsed<eta / rate
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}",
            )
            findings_so_far = 0
            parse_err_so_far = 0
            for chunk in pbar:
                n_chunks += 1
                # Tracing: одинаковые spans в Langfuse + MLflow. Открыты СРАЗУ
                # вокруг review_chunk, чтобы каждый LLM-call судьи попадал в
                # observability с input=chunk_text и output=findings/reasoning.
                # Outer span (run-level) открывает runner.py — наши chunk-spans
                # auto-attachатся к нему через OpenTelemetry / MLflow context.
                lf_metadata = {
                    "chunk_id": chunk.chunk_id,
                    "item_id": item.item_id,
                    "asr_model": item.metadata.get("model", ""),
                    "prompt_version": prompt.version,
                    "source_kind": item.metadata.get("source_kind", ""),
                }
                mlf_inputs = {
                    "chunk_text": chunk.text[:2000],  # обрезаем превью для UI
                    "chunk_chars": len(chunk.text),
                    "reference": (item.reference[:500]
                                  if item.reference else None),
                }
                with langfuse_export.start_judge_chunk_span(
                    chunk_id=chunk.chunk_id,
                    chunk_text=chunk.text,
                    reference=item.reference if (
                        item.reference
                        and item.metadata.get("source_kind") == "benchmark"
                    ) else None,
                    extra_metadata=lf_metadata,
                ), mlflow_span(
                    chunk.chunk_id,
                    span_type="CHAT_MODEL",
                    inputs=mlf_inputs,
                    attributes={
                        "asr_model": item.metadata.get("model", ""),
                        "prompt_version": prompt.version,
                    },
                ) as ml_span:
                    review = await review_chunk(
                        chunk, item, client=client, prompt=prompt,
                    )
                    # Outputs обновляем после chat call — теперь известны
                    # findings + reasoning + tokens.
                    chunk_outputs = {
                        "n_findings": len(review.findings),
                        "findings": [f.as_dict() for f in review.findings],
                        "reasoning_preview": (review.reasoning[:1000]
                                              if review.reasoning else ""),
                        "parse_error": review.parse_error or "",
                        "prompt_tokens": review.prompt_tokens,
                        "reasoning_tokens": review.reasoning_tokens,
                        "output_tokens": review.output_tokens,
                        "elapsed_sec": round(review.elapsed_sec, 2),
                    }
                    langfuse_export.update_current_output(chunk_outputs)
                    try:
                        ml_span.set_outputs(chunk_outputs)
                    except Exception:
                        pass

                    # Scores на текущий chunk-span (Langfuse): per-chunk
                    # метрики удобно смотреть в UI прямо рядом с reasoning.
                    langfuse_export.score_current(
                        name="n_findings", value=float(len(review.findings)),
                    )
                    if review.prefill_tps > 0:
                        langfuse_export.score_current(
                            name="prefill_tps", value=float(review.prefill_tps),
                        )
                    if review.decode_tps > 0:
                        langfuse_export.score_current(
                            name="decode_tps", value=float(review.decode_tps),
                        )
                    langfuse_export.score_current(
                        name="reasoning_tokens",
                        value=float(review.reasoning_tokens),
                    )
                    langfuse_export.score_current(
                        name="output_tokens", value=float(review.output_tokens),
                    )
                    langfuse_export.score_current(
                        name="parse_error",
                        value=1.0 if review.parse_error else 0.0,
                    )
                    type_counts_lf: Dict[str, int] = {}
                    for f in review.findings:
                        type_counts_lf[f.error_type] = (
                            type_counts_lf.get(f.error_type, 0) + 1
                        )
                    for et in prompt.allowed_error_types:
                        cnt = type_counts_lf.get(et, 0)
                        if cnt > 0:
                            langfuse_export.score_current(
                                name=f"count_{et}", value=float(cnt),
                            )

                n_findings += len(review.findings)
                for f in review.findings:
                    counts_by_type[f.error_type] = counts_by_type.get(f.error_type, 0) + 1

                # MLflow per-chunk timeline. step=глобальный index чанка в
                # прогоне, чтобы в UI каждая лекция и item шли подряд во времени.
                # Все вызовы fail-soft через _ActiveRun, исключения не пробросятся.
                if mlrun is not None:
                    step = n_chunks - 1
                    # progress_pct — линейная метрика 0..100 per item. На графике
                    # MLflow Metrics tab видна как растущая прямая, обновляется
                    # каждый чанк. Для single-item lecture-mode (наш типовой
                    # случай) — монотонно 0→100. Для multi-item benchmark —
                    # сбрасывается на каждом новом item (sawtooth).
                    # pbar.n ещё не инкрементирован (tqdm делает update после
                    # выхода из тела loop'а), поэтому (n + 1) — текущий index.
                    item_chunk_idx = pbar.n
                    item_progress = 100.0 * (item_chunk_idx + 1) / max(1, len(chunks))
                    mlrun.log_metric("progress_pct", item_progress, step=step)
                    mlrun.log_metric("n_findings", float(len(review.findings)), step=step)
                    mlrun.log_metric("prompt_tokens", float(review.prompt_tokens), step=step)
                    mlrun.log_metric("reasoning_tokens", float(review.reasoning_tokens), step=step)
                    mlrun.log_metric("output_tokens", float(review.output_tokens), step=step)
                    mlrun.log_metric("completion_tokens", float(review.completion_tokens), step=step)
                    mlrun.log_metric("elapsed_sec", float(review.elapsed_sec), step=step)
                    if review.prefill_tps > 0:
                        mlrun.log_metric("prefill_tps", float(review.prefill_tps), step=step)
                    if review.decode_tps > 0:
                        mlrun.log_metric("decode_tps", float(review.decode_tps), step=step)
                    mlrun.log_metric("chunk_chars", float(len(chunk.text)), step=step)
                    mlrun.log_metric("parse_error", 1.0 if review.parse_error else 0.0, step=step)
                    # Per-type findings count — отдельные оси в UI.
                    type_counts: Dict[str, int] = {}
                    for f in review.findings:
                        type_counts[f.error_type] = type_counts.get(f.error_type, 0) + 1
                    for et in prompt.allowed_error_types:
                        mlrun.log_metric(
                            f"count_{et}", float(type_counts.get(et, 0)), step=step,
                        )

                rows = _csv_rows_for_chunk(
                    review, item,
                    allowed_error_types=prompt.allowed_error_types,
                    judge_name=judge_name,
                    prompt_version=prompt.version,
                )
                all_rows.extend(rows)

                jsonl_record = {
                    "chunk_id": review.chunk_id,
                    "item_id": review.item_id,
                    "source_kind": item.metadata.get("source_kind", ""),
                    "asr_model": item.metadata.get("model", ""),
                    "title": item.metadata.get("title", ""),
                    "chunk_text": chunk.text,
                    "reference": item.reference,
                    "reasoning": review.reasoning,
                    "raw_content": review.raw_content,
                    "findings": [f.as_dict() for f in review.findings],
                    # SGR analysis (v7+): candidates до фильтрации + summary.
                    # Главный инструмент тюнинга — видно какие кандидаты модель
                    # отбросила через whitelist/literary/confidence гейты.
                    "analysis": review.analysis,
                    "parse_error": review.parse_error,
                    "elapsed_sec": review.elapsed_sec,
                    "prompt_tokens": review.prompt_tokens,
                    "completion_tokens": review.completion_tokens,
                    "reasoning_tokens": review.reasoning_tokens,
                    "output_tokens": review.output_tokens,
                    "prefill_ms": review.prefill_ms,
                    "prefill_tps": review.prefill_tps,
                    "decode_ms": review.decode_ms,
                    "decode_tps": review.decode_tps,
                    "judge_name": judge_name,
                    "prompt_version": prompt.version,
                }
                jsonl_f.write(json.dumps(jsonl_record, ensure_ascii=False) + "\n")
                jsonl_f.flush()

                # Лог per-chunk идёт через tqdm.write чтобы прогресс-бар
                # оставался прибит к низу терминала. logger.info НЕ зовём —
                # иначе stderr-handler перебьёт bar.
                if review.parse_error:
                    parse_err_so_far += 1
                    tqdm.write(
                        f"  ⚠ [{review.chunk_id}] parse_error={review.parse_error}; "
                        f"raw={review.raw_content[:120]!r}"
                    )
                findings_so_far += len(review.findings)
                summary = ",".join(
                    f"{f.error_type[:4]}/{f.severity[:1]}"
                    for f in review.findings[:5]
                )
                tqdm.write(
                    f"  • [{review.chunk_id}] {len(review.findings)} findings "
                    f"({summary or '-'}) "
                    f"in={review.prompt_tokens}t "
                    f"reason={review.reasoning_tokens}t "
                    f"out={review.output_tokens}t "
                    f"{review.elapsed_sec:.1f}s"
                )
                pbar.set_postfix({
                    "find": findings_so_far,
                    "err": parse_err_so_far,
                    "in":  f"{review.prompt_tokens}t",
                    "rsn": f"{review.reasoning_tokens}t",
                    "out": f"{review.output_tokens}t",
                    "dec": f"{review.decode_tps:.0f}t/s" if review.decode_tps else "—",
                })

    csv_path: Optional[Path] = None
    if all_rows:
        csv_path = csv_export.write_rows(run_name, all_rows, kind="judge")

    logger.info(
        "Run done: %d items, %d chunks, %d findings. CSV=%s, JSONL=%s",
        n_items, n_chunks, n_findings, csv_path, jsonl_path,
    )
    return ReviewRun(
        csv_path=csv_path,
        jsonl_path=jsonl_path,
        n_items=n_items,
        n_chunks=n_chunks,
        n_findings=n_findings,
        counts_by_type=counts_by_type,
    )


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)[:80]
