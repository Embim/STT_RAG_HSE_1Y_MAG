"""Генерация LLM-judge eval отчёта (metrics.json + report.csv + report.html).

Аналог `asr_report.py`, но для метрик count_<type>/sev_avg_<type>/n_findings_total.

Главная фишка HTML-отчёта: per-chunk блоки с **самими findings и reasoning'ом**
от модели — то ради чего весь LLM-judge затевался. Каждый чанк = карточка:
  - короткий header (chunk #N, кол-во findings, time, parse_error если есть)
  - текст чанка (collapsed, чтобы не загромождать)
  - таблица findings: type/severity/evidence/suggestion/confidence/explanation
  - reasoning модели (collapsed) — её внутреннее `<think>...` рассуждение,
    то самое что нужно читать для тюнинга промпта
  - raw_content (collapsed) — сырой ответ до парсинга

Вызывается двумя путями:
  - автоматически из `evaluation.judge.runner` после прогона
  - вручную через `scripts/judge_eval_report.py`
"""
from __future__ import annotations

import html as _html
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

ERROR_TYPES = ("terminology", "hallucination", "grammar", "gap")
COUNT_COLS = [f"count_{t}" for t in ERROR_TYPES]
SEV_COLS = [f"sev_avg_{t}" for t in ERROR_TYPES]
DENSITY_COLS = [f"density_{t}" for t in ERROR_TYPES]
SEVERITY_SCORE = {"low": 1.0, "medium": 2.0, "high": 3.0}

# Q-Score калибровка: max(0, 100 - WED * SCALE).
# WED = weighted_error_density (severity-weighted, per 1000 chars).
# На наших данных среднее WED у whisper-large-v3-turbo ~3-5, у мелких моделей
# 8-15. SCALE=8 даёт удобный спред:
#   WED=0   → Q=100 (идеально)
#   WED=2   → Q=84  (отлично, мало ошибок)
#   WED=5   → Q=60  (приемлемо)
#   WED=8   → Q=36  (плохо)
#   WED=12+ → Q=0   (катастрофа)
# Эту константу можно подкрутить если на твоей корпусе все модели в одну кучу.
QSCORE_SCALE = 8.0

# Регекс для отделения parent item_id от chunk_id (`_chunk_NNN` суффикс).
_CHUNK_SUFFIX = re.compile(r"_chunk_(\d+)$")

# Цвета для error_type badge'ей.
_TYPE_COLORS = {
    "terminology": "#3b82f6",     # blue
    "hallucination": "#ef4444",   # red
    "grammar": "#f59e0b",         # amber
    "gap": "#8b5cf6",             # violet
}
_SEVERITY_COLORS = {
    "low": "#22c55e",      # green
    "medium": "#f59e0b",   # amber
    "high": "#ef4444",     # red
}


def _parent_and_idx(chunk_id: str) -> tuple[str, int]:
    """Разрезать chunk_id на (parent_id, index). По умолчанию idx=0."""
    m = _CHUNK_SUFFIX.search(chunk_id)
    if m:
        return chunk_id[:m.start()], int(m.group(1))
    return chunk_id, 0


def _load_long_csv(paths: list[Path]) -> pd.DataFrame:
    dfs = [pd.read_csv(p, encoding="utf-8-sig") for p in paths]
    df = pd.concat(dfs, ignore_index=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def _load_jsonl(paths: list[Path]) -> Dict[tuple[str, str], dict]:
    """Загрузить все JSONL сайдкары → словарь {(model, chunk_id): record}.

    Из JSONL берём то чего нет в CSV: chunk_text, reasoning, raw_content,
    список findings полностью.
    """
    out: Dict[tuple[str, str], dict] = {}
    for p in paths:
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (r.get("judge_name", ""), r.get("chunk_id", ""))
                out[key] = r
    return out


def _parse_extra(extra_raw: object) -> dict:
    if not isinstance(extra_raw, str) or not extra_raw.strip():
        return {}
    try:
        return json.loads(extra_raw)
    except Exception:
        return {}


def _chars_and_weighted_from_jsonl(
    jsonl_records: Dict[tuple[str, str], dict],
) -> Dict[tuple[str, str], tuple[int, float]]:
    """Для каждого (model, chunk_id) вернуть (n_chars, weighted_severity_sum).

    n_chars — длина чанка (нормализатор для density).
    weighted = sum(SEVERITY_SCORE[f.severity] for f in findings).
    """
    out = {}
    for key, r in jsonl_records.items():
        chars = len(r.get("chunk_text", "") or "")
        weighted = sum(
            SEVERITY_SCORE.get(str(f.get("severity", "")).lower(), 0.0)
            for f in (r.get("findings") or [])
            if isinstance(f, dict)
        )
        out[key] = (chars, weighted)
    return out


def _wide_per_chunk(
    df_long: pd.DataFrame,
    jsonl_records: Optional[Dict[tuple[str, str], dict]] = None,
) -> pd.DataFrame:
    df = df_long.copy()
    extras = df["extra"].apply(_parse_extra)
    df["title"] = extras.apply(lambda d: d.get("title", "") or "")
    df["asr_model"] = extras.apply(lambda d: d.get("asr_model", "") or "")
    df["source_kind"] = extras.apply(lambda d: d.get("source_kind", "") or "")
    df["prompt_version"] = extras.apply(lambda d: d.get("prompt_version", "") or "")
    df["parse_error"] = extras.apply(lambda d: d.get("parse_error", "") or "")
    df["elapsed_sec"] = extras.apply(lambda d: float(d.get("elapsed_sec", 0.0) or 0.0))
    df["prompt_tokens"] = extras.apply(lambda d: int(d.get("prompt_tokens", 0) or 0))
    df["completion_tokens"] = extras.apply(lambda d: int(d.get("completion_tokens", 0) or 0))
    # v11+ per-stage breakdown (reasoning + output = completion).
    # На старых CSV без этих полей: reasoning_tokens=0, output_tokens=completion.
    df["reasoning_tokens"] = extras.apply(
        lambda d: int(d.get("reasoning_tokens", 0) or 0)
    )
    df["output_tokens"] = extras.apply(
        lambda d: int(d.get("output_tokens", 0) or 0)
    )
    df["output_tokens"] = df.apply(
        lambda r: r["output_tokens"] if r["output_tokens"] > 0
        else max(0, r["completion_tokens"] - r["reasoning_tokens"]),
        axis=1,
    )
    df["prefill_ms"] = extras.apply(lambda d: float(d.get("prefill_ms", 0.0) or 0.0))
    df["prefill_tps"] = extras.apply(lambda d: float(d.get("prefill_tps", 0.0) or 0.0))
    df["decode_ms"] = extras.apply(lambda d: float(d.get("decode_ms", 0.0) or 0.0))
    df["decode_tps"] = extras.apply(lambda d: float(d.get("decode_tps", 0.0) or 0.0))
    # t/s output (decode): completion_tokens / elapsed_sec. Это то что обычно
    # репортится для LLM-серверов как "скорость генерации". prefill (input)
    # отдельно не считаем — обычно быстрый, не лимитирующий.
    # Если есть нативный decode_tps из timings — используем его, иначе fallback.
    df["tokens_per_sec"] = df.apply(
        lambda r: r["decode_tps"] if r["decode_tps"] > 0
        else ((r["completion_tokens"] / r["elapsed_sec"]) if r["elapsed_sec"] > 0 else 0.0),
        axis=1,
    )
    df["findings_from_csv"] = extras.apply(lambda d: d.get("findings", []))
    df["original_item_id"] = extras.apply(lambda d: d.get("original_item_id", "") or "")
    df["chunk_id"] = df["item_id"]
    df["parent_item_id"] = df.apply(
        lambda r: r["original_item_id"] or _parent_and_idx(str(r["chunk_id"]))[0],
        axis=1,
    )
    df["chunk_idx"] = df["chunk_id"].apply(lambda c: _parent_and_idx(str(c))[1])

    # ЭФФЕКТИВНЫЙ КЛЮЧ ГРУППИРОВКИ для сравнения ASR.
    # В CSV-колонке `model` сидит ИМЯ СУДЬИ (всегда одно, e.g. "qwen3_5_9b_q4").
    # В extra.asr_model — ИМЯ ASR-модели (whisper/qwen3-asr/...).
    # Для leaderboard нам нужен asr_model, иначе всё схлопнется в одну строку.
    # Fallback на model (= judge) если asr_model пустой — это случай когда
    # judge юзается для тюнинга промпта, а не для сравнения ASR.
    df["asr_or_judge"] = df.apply(
        lambda r: r["asr_model"] if r["asr_model"] else r["model"], axis=1
    )

    # n_chars + weighted_severity per (model, chunk_id) из JSONL.
    jsonl_records = jsonl_records or {}
    chars_map = _chars_and_weighted_from_jsonl(jsonl_records)
    df["n_chars"] = df.apply(
        lambda r: chars_map.get((r["model"], r["chunk_id"]), (0, 0.0))[0], axis=1
    )
    df["weighted_severity"] = df.apply(
        lambda r: chars_map.get((r["model"], r["chunk_id"]), (0, 0.0))[1], axis=1
    )

    pivot = df.pivot_table(
        index=["model", "chunk_id"],
        columns="metric",
        values="value",
        aggfunc="first",
    ).reset_index()

    meta_cols = ["title", "asr_model", "source_kind", "prompt_version",
                 "parse_error", "elapsed_sec", "prompt_tokens",
                 "completion_tokens", "reasoning_tokens", "output_tokens",
                 "prefill_ms", "prefill_tps", "decode_ms", "decode_tps",
                 "tokens_per_sec", "findings_from_csv",
                 "parent_item_id", "chunk_idx", "n_chars", "weighted_severity",
                 "asr_or_judge"]
    meta = (
        df.groupby(["model", "chunk_id"], as_index=False)[meta_cols]
        .agg("first")
    )
    wide = pivot.merge(meta, on=["model", "chunk_id"], how="left")

    for col in COUNT_COLS + SEV_COLS + ["n_findings_total"]:
        if col not in wide.columns:
            wide[col] = 0.0

    # Per-chunk density: findings на 1000 символов транскрипта.
    # Если n_chars=0 (JSONL не приложен) → density тоже 0.
    def _safe_div(a, b):
        return (a / b * 1000.0) if b > 0 else 0.0

    wide["errors_per_1k_chars"] = wide.apply(
        lambda r: _safe_div(r["n_findings_total"], r["n_chars"]), axis=1
    )
    wide["weighted_error_density"] = wide.apply(
        lambda r: _safe_div(r["weighted_severity"], r["n_chars"]), axis=1
    )
    for et in ERROR_TYPES:
        wide[f"density_{et}"] = wide.apply(
            lambda r, et=et: _safe_div(r[f"count_{et}"], r["n_chars"]), axis=1
        )

    return wide.sort_values(
        ["model", "parent_item_id", "chunk_idx"]
    ).reset_index(drop=True)


def _aggregate_metrics(wide: pd.DataFrame) -> dict:
    out: dict = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "n_models": int(wide["asr_or_judge"].nunique()),
        "n_items": int(wide.groupby(["asr_or_judge", "parent_item_id"]).ngroups),
        "n_chunks": int(len(wide)),
        "n_findings_total": int(wide["n_findings_total"].sum()),
        "models": sorted(wide["asr_or_judge"].unique().tolist()),
        "judges": sorted(wide["model"].unique().tolist()),
        "prompt_versions": sorted(
            [v for v in wide["prompt_version"].unique().tolist() if v]
        ),
    }

    overall: dict[str, dict] = {}
    for model, gm in wide.groupby("asr_or_judge"):
        parse_err = (gm["parse_error"].astype(str).str.strip() != "").mean()
        total_completion = int(gm["completion_tokens"].sum())
        total_prompt = int(gm["prompt_tokens"].sum())
        total_elapsed = float(gm["elapsed_sec"].sum())
        corpus_tps = (total_completion / total_elapsed) if total_elapsed > 0 else 0.0

        # Корпусные density-метрики: суммарно по всем чанкам этого ASR-прогона.
        # Это нормированные метрики качества ASR с точки зрения судьи.
        # Корпусный density = total_findings * 1000 / total_chars (weighted
        # average, учитывает реальный вклад каждого чанка).
        total_chars = int(gm["n_chars"].sum())
        total_findings = int(gm["n_findings_total"].sum())
        total_weighted = float(gm["weighted_severity"].sum())

        epk = (total_findings * 1000.0 / total_chars) if total_chars > 0 else 0.0
        wed = (total_weighted * 1000.0 / total_chars) if total_chars > 0 else 0.0
        q_score = max(0.0, 100.0 - wed * QSCORE_SCALE)

        # Type-specific density.
        type_density = {}
        for et in ERROR_TYPES:
            type_count = int(gm[f"count_{et}"].sum())
            type_density[f"density_{et}"] = (
                float(round(type_count * 1000.0 / total_chars, 3))
                if total_chars > 0 else 0.0
            )

        # v11+ per-stage breakdown — медианы более устойчивы к loop-чанкам
        # (4-5 чанков с reasoning=20K тянут mean в небо, медиана их игнорирует).
        total_reasoning = int(gm["reasoning_tokens"].sum())
        total_output = int(gm["output_tokens"].sum())
        # decode tps: для корпуса считаем по сумме decode_tokens / сумму decode_ms,
        # это эквивалент weighted average. Если timings нет (decode_ms=0) —
        # фолбэк на mean от per-chunk tokens_per_sec.
        total_decode_ms = float(gm["decode_ms"].sum())
        if total_decode_ms > 0:
            decode_tps_corpus = total_completion * 1000.0 / total_decode_ms
        else:
            decode_tps_corpus = corpus_tps
        # prefill tps: input_tokens / prefill_ms. На свежем llama.cpp есть всегда.
        total_prefill_ms = float(gm["prefill_ms"].sum())
        prefill_tps_corpus = (
            total_prompt * 1000.0 / total_prefill_ms
            if total_prefill_ms > 0 else 0.0
        )

        row = {
            "n_chunks": int(len(gm)),
            "n_chars_total": total_chars,
            "n_findings_total": total_findings,
            "weighted_severity_total": float(round(total_weighted, 1)),
            # ── ГЛАВНЫЕ метрики для сравнения ASR ──────────────────────
            "judge_quality_score": float(round(q_score, 1)),  # 0..100, выше = лучше
            "weighted_error_density": float(round(wed, 3)),    # severity-weighted per 1k chars
            "errors_per_1k_chars": float(round(epk, 3)),       # raw findings per 1k chars
            # ──────────────────────────────────────────────────────────
            "findings_per_chunk": float(round(gm["n_findings_total"].mean(), 3)),
            "parse_error_rate": float(round(parse_err, 3)),
            "elapsed_sec_total": float(round(total_elapsed, 1)),
            "elapsed_sec_per_chunk": float(round(gm["elapsed_sec"].mean(), 1)),
            "prompt_tokens_total": total_prompt,
            "completion_tokens_total": total_completion,
            "tokens_per_sec_corpus": float(round(corpus_tps, 2)),
            "tokens_per_sec_chunk_avg": float(round(gm["tokens_per_sec"].mean(), 2)),
            # ── Per-stage breakdown (v11+) ────────────────────────────
            "reasoning_tokens_total": total_reasoning,
            "output_tokens_total": total_output,
            "prompt_tokens_median": float(round(gm["prompt_tokens"].median(), 1)),
            "reasoning_tokens_median": float(round(gm["reasoning_tokens"].median(), 1)),
            "output_tokens_median": float(round(gm["output_tokens"].median(), 1)),
            "prefill_tps_corpus": float(round(prefill_tps_corpus, 1)),
            "decode_tps_corpus": float(round(decode_tps_corpus, 2)),
        }
        row.update(type_density)
        for c in COUNT_COLS:
            row[c] = int(gm[c].sum())
        for s in SEV_COLS:
            non_zero = gm[s][gm[s] > 0]
            row[s] = float(round(non_zero.mean(), 3)) if len(non_zero) else 0.0
        overall[model] = row
    out["overall_corpus"] = overall

    per_item: dict[str, dict] = {}
    for model, gm in wide.groupby("asr_or_judge"):
        per_item[model] = {}
        for parent, gi in gm.groupby("parent_item_id"):
            item_chars = int(gi["n_chars"].sum())
            item_findings = int(gi["n_findings_total"].sum())
            item_weighted = float(gi["weighted_severity"].sum())
            item_wed = (item_weighted * 1000.0 / item_chars) if item_chars > 0 else 0.0
            item_q = max(0.0, 100.0 - item_wed * QSCORE_SCALE)
            row = {
                "title": gi["title"].iloc[0] or "",
                "asr_model": gi["asr_model"].iloc[0] or "",
                "source_kind": gi["source_kind"].iloc[0] or "",
                "n_chunks": int(len(gi)),
                "n_chars": item_chars,
                "n_findings": item_findings,
                "judge_quality_score": float(round(item_q, 1)),
                "weighted_error_density": float(round(item_wed, 3)),
                "errors_per_1k_chars": float(round(
                    (item_findings * 1000.0 / item_chars) if item_chars > 0 else 0.0,
                    3,
                )),
                "findings_per_chunk": float(round(gi["n_findings_total"].mean(), 2)),
            }
            for c in COUNT_COLS:
                row[c] = int(gi[c].sum())
            per_item[model][parent or "<unknown>"] = row
    out["per_item"] = per_item
    return out


def _esc(s: object) -> str:
    """HTML-escape с None-safe и truncation guard."""
    if s is None:
        return ""
    return _html.escape(str(s))


def _badge(text: str, color: str) -> str:
    """Цветной inline-badge."""
    return (
        f'<span style="background:{color};color:#fff;padding:2px 8px;'
        f'border-radius:10px;font-size:11px;font-weight:600;'
        f'white-space:nowrap;">{_esc(text)}</span>'
    )


def _render_findings_table(findings: list[dict]) -> str:
    """Таблица findings внутри одного чанка."""
    if not findings:
        return '<p class="empty">— findings нет —</p>'
    rows = []
    for f in findings:
        etype = str(f.get("error_type", "")).lower()
        sev = str(f.get("severity", "")).lower()
        evidence = _esc(f.get("evidence", ""))
        suggestion = _esc(f.get("suggestion", ""))
        conf = f.get("confidence", 0)
        try:
            conf_s = f"{float(conf):.2f}"
        except (ValueError, TypeError):
            conf_s = "—"
        expl = _esc(f.get("explanation", ""))
        rows.append(f"""
        <tr>
          <td>{_badge(etype, _TYPE_COLORS.get(etype, '#666'))}</td>
          <td>{_badge(sev, _SEVERITY_COLORS.get(sev, '#666'))}</td>
          <td class="evidence"><code>{evidence}</code></td>
          <td class="suggestion">{suggestion or '<span class="empty">—</span>'}</td>
          <td class="conf">{conf_s}</td>
          <td class="expl">{expl}</td>
        </tr>""")
    return f"""
    <table class="findings">
      <thead>
        <tr>
          <th>type</th><th>severity</th>
          <th>evidence (цитата из транскрипции)</th>
          <th>suggestion (как должно быть)</th>
          <th>conf</th><th>explanation</th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>"""


def _render_chunk_block(
    chunk_row: pd.Series,
    jsonl_record: Optional[dict],
) -> str:
    """Карточка одного чанка: header + findings table + collapsible reasoning/raw/text."""
    chunk_idx = int(chunk_row["chunk_idx"])
    n_findings = int(chunk_row["n_findings_total"])
    elapsed = float(chunk_row["elapsed_sec"])
    tps = float(chunk_row.get("tokens_per_sec", 0.0))
    completion_tok = int(chunk_row.get("completion_tokens", 0))
    # v11+ per-stage breakdown.
    prompt_tok = int(chunk_row.get("prompt_tokens", 0))
    reasoning_tok = int(chunk_row.get("reasoning_tokens", 0))
    output_tok = int(chunk_row.get("output_tokens", 0))
    prefill_tps = float(chunk_row.get("prefill_tps", 0.0))
    decode_tps = float(chunk_row.get("decode_tps", 0.0))
    # decode_tps==0 на старых CSV (до v11) — фолбэк на tps (= completion/elapsed).
    decode_tps_display = decode_tps if decode_tps > 0 else tps
    parse_err = str(chunk_row["parse_error"]).strip()
    parse_err_html = (
        f'<span class="parse-error">parse_error: {_esc(parse_err)}</span>'
        if parse_err else ""
    )

    # Findings — приоритет JSONL (полные), fallback на CSV-extra.
    findings: list[dict] = []
    if jsonl_record and isinstance(jsonl_record.get("findings"), list):
        findings = jsonl_record["findings"]
    else:
        raw = chunk_row.get("findings_from_csv")
        if isinstance(raw, list):
            findings = raw

    # Тексты только из JSONL (в CSV их нет — слишком большие).
    chunk_text = jsonl_record.get("chunk_text", "") if jsonl_record else ""
    reasoning = jsonl_record.get("reasoning", "") if jsonl_record else ""
    raw_content = jsonl_record.get("raw_content", "") if jsonl_record else ""
    analysis = jsonl_record.get("analysis") if jsonl_record else None

    # Type-by-type счётчики для header'а
    type_chips = []
    for et in ERROR_TYPES:
        cnt = int(chunk_row.get(f"count_{et}", 0))
        if cnt > 0:
            type_chips.append(
                f'<span class="chip" style="background:{_TYPE_COLORS[et]}22;'
                f'color:{_TYPE_COLORS[et]};border:1px solid {_TYPE_COLORS[et]}55;">'
                f'{et}: {cnt}</span>'
            )
    chips_html = " ".join(type_chips) if type_chips else (
        '<span class="chip empty-chip">no findings</span>'
    )

    header_status = "ok" if n_findings > 0 and not parse_err else (
        "warn" if parse_err else "clean"
    )

    # Collapsible блоки — детали скрываем под <details>
    text_preview = chunk_text[:240] + ("…" if len(chunk_text) > 240 else "")

    analysis_block = ""
    if analysis and isinstance(analysis, dict):
        candidates = analysis.get("candidates", []) or []
        summary = analysis.get("summary", "")
        cand_rows = []
        for c in candidates:
            if not isinstance(c, dict):
                continue
            keep = c.get("keep_as_finding", False)
            wl = c.get("is_in_whitelist", False)
            lit = c.get("is_literary_device", False)
            conf = c.get("confidence_estimate", 0)
            try:
                conf_s = f"{float(conf):.2f}"
            except (ValueError, TypeError):
                conf_s = "—"
            row_class = "candidate-keep" if keep else "candidate-drop"
            kept_icon = "✓" if keep else "✗"
            cand_rows.append(f"""
            <tr class="{row_class}">
              <td>{kept_icon}</td>
              <td><code>{_esc(c.get('evidence', ''))}</code></td>
              <td>{_esc(c.get('my_guess', '')) or '<span class="empty">—</span>'}</td>
              <td>{'✓' if wl else '·'}</td>
              <td>{'✓' if lit else '·'}</td>
              <td class="conf">{conf_s}</td>
            </tr>""")
        analysis_block = f"""
      <details class="analysis" open>
        <summary>🔬 SGR Analysis: {len(candidates)} candidates → {sum(1 for c in candidates if isinstance(c, dict) and c.get('keep_as_finding'))} findings</summary>
        <table class="candidates">
          <thead>
            <tr>
              <th>keep</th><th>evidence (кандидат)</th>
              <th>my_guess</th><th>whitelist?</th>
              <th>literary?</th><th>conf</th>
            </tr>
          </thead>
          <tbody>{''.join(cand_rows)}</tbody>
        </table>
        <p class="analysis-summary">📝 <i>{_esc(summary)}</i></p>
      </details>"""

    reasoning_block = ""
    if reasoning:
        reasoning_block = f"""
      <details class="reasoning">
        <summary>🧠 Reasoning модели ({len(reasoning)} chars)</summary>
        <pre>{_esc(reasoning)}</pre>
      </details>"""

    raw_block = ""
    if raw_content:
        raw_block = f"""
      <details class="raw">
        <summary>📝 Raw response ({len(raw_content)} chars)</summary>
        <pre>{_esc(raw_content)}</pre>
      </details>"""

    full_text_block = ""
    if chunk_text and len(chunk_text) > 240:
        full_text_block = f"""
      <details class="chunk-text">
        <summary>📄 Полный текст чанка ({len(chunk_text)} chars)</summary>
        <pre>{_esc(chunk_text)}</pre>
      </details>"""

    # Per-stage breakdown row (v11+).
    # 3 счётчика токенов (input/reasoning/output) + 2 скорости (prefill/decode).
    # На старых CSV (v10−) reasoning_tokens=0 — стадия reasoning не подсвечивается.
    prefill_speed_html = (
        f' <span class="stage-tps" title="prefill speed">@ {prefill_tps:.0f} t/s</span>'
        if prefill_tps > 0 else ""
    )
    decode_speed_html = (
        f' <span class="stage-tps" title="decode speed (reasoning+output идут одним потоком)">'
        f'@ {decode_tps_display:.0f} t/s</span>'
        if decode_tps_display > 0 else ""
    )
    stage_row = f"""
      <div class="stage-row" title="completion = reasoning + output">
        <span class="stage stage-input">
          <b>input</b> {prompt_tok:,}t{prefill_speed_html}
        </span>
        <span class="stage stage-reasoning">
          <b>reasoning</b> {reasoning_tok:,}t{decode_speed_html}
        </span>
        <span class="stage stage-output">
          <b>output</b> {output_tok:,}t{decode_speed_html}
        </span>
      </div>"""

    return f"""
    <div class="chunk chunk-{header_status}">
      <div class="chunk-header">
        <span class="chunk-num">#{chunk_idx:03d}</span>
        <span class="findings-count">{n_findings} findings</span>
        <span class="elapsed" title="elapsed wallclock time">{elapsed:.1f}s</span>
        {parse_err_html}
        <div class="chips">{chips_html}</div>
      </div>
      {stage_row}
      <div class="chunk-text-preview">{_esc(text_preview)}</div>
      {_render_findings_table(findings)}
      {analysis_block}
      {reasoning_block}
      {raw_block}
      {full_text_block}
    </div>"""


def _render_html(
    wide: pd.DataFrame,
    agg: dict,
    out_path: Path,
    *,
    jsonl_records: Dict[tuple[str, str], dict],
    jsonl_paths: list[Path] | None = None,
) -> None:
    n_chunks_w_findings = int((wide["n_findings_total"] > 0).sum())
    prompt_str = ", ".join(agg["prompt_versions"]) or "—"

    # Корпусные summary cards
    def _q_color(q: float) -> str:
        """Цвет Q-Score: красный → жёлтый → зелёный."""
        if q >= 80:
            return "#22c55e"  # green
        if q >= 60:
            return "#84cc16"  # lime
        if q >= 40:
            return "#f59e0b"  # amber
        if q >= 20:
            return "#f97316"  # orange
        return "#ef4444"      # red

    cards = []
    for model, mvals in agg["overall_corpus"].items():
        tps_corpus = mvals.get("tokens_per_sec_corpus", 0.0)
        total_compl = mvals.get("completion_tokens_total", 0)
        q_score = mvals.get("judge_quality_score", 0.0)
        wed = mvals.get("weighted_error_density", 0.0)
        epk = mvals.get("errors_per_1k_chars", 0.0)
        n_chars = mvals.get("n_chars_total", 0)
        # v11+ per-stage breakdown.
        prompt_med = mvals.get("prompt_tokens_median", 0.0)
        reasoning_med = mvals.get("reasoning_tokens_median", 0.0)
        output_med = mvals.get("output_tokens_median", 0.0)
        prefill_tps = mvals.get("prefill_tps_corpus", 0.0)
        decode_tps = mvals.get("decode_tps_corpus", 0.0)
        stage_corpus_html = ""
        if reasoning_med > 0 or output_med > 0:
            prefill_chip = (
                f"<div><b>{prefill_tps:.0f}</b><span>prefill t/s</span></div>"
                if prefill_tps > 0 else ""
            )
            decode_chip = (
                f"<div><b>{decode_tps:.1f}</b><span>decode t/s</span></div>"
                if decode_tps > 0 else ""
            )
            stage_corpus_html = f"""
          <div class="stage-row-corpus" title="median tokens per chunk + speeds">
            <div><b>{prompt_med:.0f}</b><span>median input</span></div>
            <div><b>{reasoning_med:.0f}</b><span>median reasoning</span></div>
            <div><b>{output_med:.0f}</b><span>median output</span></div>
            {prefill_chip}
            {decode_chip}
          </div>"""
        cards.append(f"""
        <div class="card">
          <div class="card-title">{_esc(model)}</div>
          <div class="qscore-row">
            <div class="qscore" style="color:{_q_color(q_score)}">
              <span class="qscore-value">{q_score:.0f}</span>
              <span class="qscore-label">Q-Score / 100</span>
            </div>
            <div class="qscore-supporting">
              <div><b>{wed:.2f}</b><span title="severity-weighted findings per 1000 chars">WED</span></div>
              <div><b>{epk:.2f}</b><span title="raw findings per 1000 chars">err/1k</span></div>
              <div><b>{n_chars:,}</b><span>chars</span></div>
            </div>
          </div>
          <div class="card-stats">
            <div><b>{mvals['n_chunks']}</b><span>chunks</span></div>
            <div><b>{mvals['n_findings_total']}</b><span>findings</span></div>
            <div><b>{mvals['findings_per_chunk']:.2f}</b><span>per chunk</span></div>
            <div class="{'bad' if mvals['parse_error_rate'] > 0 else ''}">
              <b>{mvals['parse_error_rate']*100:.1f}%</b><span>parse err</span>
            </div>
            <div><b>{mvals['elapsed_sec_per_chunk']:.1f}s</b><span>avg/chunk</span></div>
            <div title="output decode rate">
              <b>{tps_corpus:.1f}</b><span>tok/s</span>
            </div>
          </div>
          <div class="card-types">
            <span style="color:{_TYPE_COLORS['terminology']}">
              terminology: <b>{mvals['count_terminology']}</b>
              (density {mvals.get('density_terminology', 0):.2f})</span>
            <span style="color:{_TYPE_COLORS['hallucination']}">
              hallucination: <b>{mvals['count_hallucination']}</b>
              (density {mvals.get('density_hallucination', 0):.2f})</span>
            <span style="color:{_TYPE_COLORS['grammar']}">
              grammar: <b>{mvals['count_grammar']}</b>
              (density {mvals.get('density_grammar', 0):.2f})</span>
            <span style="color:{_TYPE_COLORS['gap']}">
              gap: <b>{mvals['count_gap']}</b>
              (density {mvals.get('density_gap', 0):.2f})</span>
          </div>{stage_corpus_html}
        </div>""")

    # Per-item секции
    item_sections = []
    for (model, parent_id), gm in wide.groupby(["asr_or_judge", "parent_item_id"]):
        title = gm["title"].iloc[0] or parent_id
        n_findings = int(gm["n_findings_total"].sum())
        chunks_html = []
        for _, row in gm.iterrows():
            # jsonl_records ключ был (judge_name, chunk_id) — для lookup'а
            # используем реальное имя судьи (колонка `model`), не asr_or_judge.
            key = (row["model"], row["chunk_id"])
            chunks_html.append(_render_chunk_block(row, jsonl_records.get(key)))
        item_sections.append(f"""
        <section class="item">
          <h2 class="item-title">
            <span class="model-tag">{_esc(model)}</span>
            {_esc(title)}
            <span class="item-summary">{len(gm)} chunks · {n_findings} findings</span>
          </h2>
          <div class="chunks">{''.join(chunks_html)}</div>
        </section>""")

    jsonl_links_html = ""
    if jsonl_paths:
        jsonl_links_html = (
            "<p class='meta'>JSONL sidecars (полный сырой output модели): "
            + " · ".join(f"<code>{_esc(p)}</code>" for p in jsonl_paths)
            + "</p>"
        )

    # Сравнительная панель если моделей ≥2 (= сравнение ASR через judge).
    comparison_html = ""
    if len(agg["overall_corpus"]) >= 2:
        ranked = sorted(
            agg["overall_corpus"].items(),
            key=lambda kv: kv[1].get("judge_quality_score", 0),
            reverse=True,
        )
        rows = []
        for rank, (model, mv) in enumerate(ranked, 1):
            q = mv.get("judge_quality_score", 0.0)
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"
            rows.append(f"""
            <tr>
              <td class="rank">{medal}</td>
              <td class="asr-name">{_esc(model)}</td>
              <td class="qscore-cell" style="background:{_q_color(q)}22;color:{_q_color(q)}"><b>{q:.1f}</b></td>
              <td>{mv.get('weighted_error_density', 0):.2f}</td>
              <td>{mv.get('errors_per_1k_chars', 0):.2f}</td>
              <td>{mv.get('density_terminology', 0):.2f}</td>
              <td>{mv.get('density_hallucination', 0):.2f}</td>
              <td>{mv.get('density_grammar', 0):.2f}</td>
              <td>{mv.get('density_gap', 0):.2f}</td>
              <td>{mv['n_chars_total']:,}</td>
            </tr>""")
        comparison_html = f"""
        <section class="comparison">
          <h2>🏆 ASR Comparison (через LLM-judge)</h2>
          <p class="meta">
            Ранжировано по <b>Q-Score</b> (severity-weighted findings per 1000 chars,
            нормализовано в шкалу 0..100). <b>Выше = лучше.</b><br>
            WED = weighted error density (low=1, medium=2, high=3, на 1000 chars).
            err/1k = raw findings count per 1000 chars.
          </p>
          <table class="leaderboard">
            <thead><tr>
              <th>rank</th><th>ASR model</th><th>Q-Score</th>
              <th>WED</th><th>err/1k</th>
              <th>term</th><th>halluc</th><th>gram</th><th>gap</th>
              <th>chars</th>
            </tr></thead>
            <tbody>{''.join(rows)}</tbody>
          </table>
        </section>"""

    css = """
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  margin: 0; padding: 24px; color: #1a1a1a; background: #fafafa;
  max-width: 1400px; margin: 0 auto;
}
h1 { margin: 0 0 8px; font-size: 24px; }
h2.item-title {
  margin: 32px 0 16px; padding: 12px 16px;
  background: #1f2937; color: #fff; border-radius: 8px;
  font-size: 16px; display: flex; align-items: center; gap: 12px;
}
.model-tag {
  background: #3b82f6; padding: 3px 10px; border-radius: 6px;
  font-size: 12px; font-weight: 600;
}
.item-summary {
  margin-left: auto; opacity: 0.7; font-size: 13px; font-weight: 400;
}
.meta { color: #6b7280; font-size: 13px; margin: 4px 0; }
.cards {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 12px; margin: 16px 0 24px;
}
.card {
  background: #fff; border-radius: 8px; padding: 16px;
  box-shadow: 0 1px 3px rgba(0,0,0,.06);
}
.card-title { font-weight: 600; margin-bottom: 12px; color: #1f2937; }
.card-stats { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 10px; }
.card-stats > div { display: flex; flex-direction: column; }
.card-stats b { font-size: 18px; }
.card-stats span { font-size: 11px; color: #6b7280; }
.card-stats .bad b { color: #ef4444; }
.card-types { font-size: 12px; display: flex; gap: 14px; flex-wrap: wrap; }
.chunk {
  background: #fff; border-radius: 8px; margin-bottom: 10px;
  padding: 12px 14px; border-left: 4px solid #e5e7eb;
}
.chunk-clean { border-left-color: #22c55e; }
.chunk-ok { border-left-color: #3b82f6; }
.chunk-warn { border-left-color: #ef4444; background: #fef2f2; }
.chunk-header {
  display: flex; align-items: center; gap: 12px;
  flex-wrap: wrap; margin-bottom: 8px;
}
.chunk-num { font-weight: 700; font-size: 13px; color: #6b7280; }
.findings-count {
  background: #f3f4f6; padding: 2px 8px; border-radius: 4px;
  font-size: 12px; font-weight: 600;
}
.elapsed { color: #9ca3af; font-size: 11px; }
.tps {
  background: #ede9fe; color: #6d28d9;
  padding: 2px 8px; border-radius: 4px;
  font-size: 11px; font-weight: 600;
}
.tokens {
  color: #6b7280; font-size: 11px; font-family: monospace;
}
.stage-row {
  display: flex; gap: 8px; flex-wrap: wrap;
  margin: 4px 0 10px; padding: 6px 8px;
  background: #f9fafb; border-radius: 4px;
  font-size: 11px; font-family: 'SF Mono', Menlo, monospace;
}
.stage {
  padding: 2px 8px; border-radius: 4px;
  background: #fff; border: 1px solid #e5e7eb;
  color: #374151;
}
.stage b { font-weight: 600; margin-right: 4px; text-transform: uppercase;
           letter-spacing: 0.4px; font-size: 10px; }
.stage-input b   { color: #2563eb; }
.stage-reasoning b { color: #7c3aed; }
.stage-output b  { color: #059669; }
.stage-tps { color: #6b7280; margin-left: 4px; font-size: 10px; }
.stage-row-corpus {
  display: flex; gap: 12px; flex-wrap: wrap; margin-top: 6px;
  font-size: 11px;
}
.stage-row-corpus > div { display: flex; flex-direction: column; }
.stage-row-corpus b { font-size: 14px; font-family: 'SF Mono', monospace; }
.stage-row-corpus span { font-size: 10px; color: #6b7280;
                          text-transform: uppercase; letter-spacing: 0.4px; }
.parse-error {
  background: #fee2e2; color: #991b1b;
  padding: 2px 8px; border-radius: 4px;
  font-size: 11px; font-family: monospace;
}
.chips { margin-left: auto; display: flex; gap: 6px; flex-wrap: wrap; }
.chip {
  padding: 2px 8px; border-radius: 4px;
  font-size: 11px; font-weight: 600;
}
.empty-chip { background: #f3f4f6; color: #9ca3af; }
.chunk-text-preview {
  background: #f9fafb; padding: 8px 10px; border-radius: 4px;
  font-size: 12px; color: #4b5563; margin-bottom: 10px;
  font-style: italic; line-height: 1.5;
}
table.findings {
  border-collapse: collapse; width: 100%; margin: 6px 0;
  font-size: 12px; background: #fff;
}
table.findings th, table.findings td {
  border: 1px solid #e5e7eb; padding: 6px 10px;
  text-align: left; vertical-align: top;
}
table.findings th {
  background: #f9fafb; font-weight: 600; color: #374151;
  font-size: 11px; text-transform: uppercase; letter-spacing: 0.4px;
}
.evidence code {
  background: #fef3c7; padding: 2px 5px; border-radius: 3px;
  font-family: 'SF Mono', Menlo, monospace; font-size: 12px;
  color: #92400e; white-space: pre-wrap;
}
.suggestion {
  color: #047857; font-weight: 500;
}
.conf { text-align: right; font-family: monospace; }
.expl { color: #4b5563; font-size: 11px; }
.empty { color: #9ca3af; font-style: italic; }
details { margin-top: 8px; }
details summary {
  cursor: pointer; padding: 4px 8px; background: #f3f4f6;
  border-radius: 4px; font-size: 12px; color: #4b5563;
  user-select: none;
}
details[open] summary { background: #e5e7eb; }
details pre {
  background: #1f2937; color: #e5e7eb; padding: 12px;
  border-radius: 4px; margin-top: 6px;
  font-size: 11px; line-height: 1.5;
  white-space: pre-wrap; word-break: break-word;
  max-height: 400px; overflow-y: auto;
}
table.candidates {
  border-collapse: collapse; width: 100%; margin-top: 8px;
  font-size: 11px; background: #fff;
}
table.candidates th, table.candidates td {
  border: 1px solid #e5e7eb; padding: 4px 8px;
  vertical-align: middle;
}
table.candidates th {
  background: #f9fafb; font-weight: 600; color: #374151;
  font-size: 10px; text-transform: uppercase; letter-spacing: 0.3px;
}
table.candidates tr.candidate-keep td { background: #ecfdf5; }
table.candidates tr.candidate-drop td { background: #f9fafb; color: #6b7280; }
table.candidates code {
  background: #fef3c7; padding: 1px 4px; border-radius: 3px;
  font-family: 'SF Mono', Menlo, monospace; font-size: 11px;
  color: #92400e;
}
.analysis-summary {
  margin-top: 8px; padding: 6px 10px;
  background: #f3f4f6; border-radius: 4px;
  font-size: 12px; color: #4b5563;
}
.qscore-row {
  display: flex; align-items: center; gap: 16px;
  padding: 8px 0; margin-bottom: 8px;
  border-bottom: 1px dashed #e5e7eb;
}
.qscore { display: flex; flex-direction: column; align-items: center; }
.qscore-value {
  font-size: 38px; font-weight: 800; line-height: 1;
  font-family: 'SF Mono', Menlo, monospace;
}
.qscore-label {
  font-size: 10px; color: #6b7280; text-transform: uppercase;
  letter-spacing: 0.5px; margin-top: 2px;
}
.qscore-supporting {
  display: flex; gap: 14px;
}
.qscore-supporting > div {
  display: flex; flex-direction: column; align-items: flex-start;
}
.qscore-supporting b { font-size: 16px; }
.qscore-supporting span {
  font-size: 10px; color: #6b7280; text-transform: uppercase;
  letter-spacing: 0.4px;
}
section.comparison {
  background: #fff; border-radius: 8px; padding: 16px;
  margin: 16px 0; box-shadow: 0 1px 3px rgba(0,0,0,.06);
}
section.comparison h2 { margin: 0 0 8px; }
table.leaderboard {
  border-collapse: collapse; width: 100%; margin-top: 12px;
  font-size: 13px;
}
table.leaderboard th, table.leaderboard td {
  border: 1px solid #e5e7eb; padding: 8px 12px;
  text-align: center;
}
table.leaderboard th {
  background: #f9fafb; font-weight: 600;
  font-size: 11px; text-transform: uppercase; letter-spacing: 0.4px;
}
table.leaderboard td.rank { font-size: 18px; }
table.leaderboard td.asr-name {
  text-align: left; font-weight: 600; font-family: 'SF Mono', monospace;
}
table.leaderboard td.qscore-cell {
  font-size: 18px; font-weight: 800;
}
"""

    html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Judge eval — {prompt_str}</title>
<style>{css}</style></head><body>
<h1>LLM-Judge Eval Report</h1>
<p class="meta">
  <b>Generated:</b> {agg['generated_at']}<br>
  <b>Models:</b> {_esc(', '.join(agg['models']))} ·
  <b>Prompt:</b> {_esc(prompt_str)} ·
  <b>Items:</b> {agg['n_items']} ·
  <b>Chunks:</b> {agg['n_chunks']} ({n_chunks_w_findings} с findings) ·
  <b>Total findings:</b> {agg['n_findings_total']}
</p>
{jsonl_links_html}

{comparison_html}

<div class="cards">{''.join(cards)}</div>

{''.join(item_sections)}

</body></html>"""
    out_path.write_text(html, encoding="utf-8")


def generate_report(
    csv_paths: list[Path],
    out_dir: Path,
    *,
    jsonl_paths: list[Path] | None = None,
) -> Path:
    """Сгенерировать metrics.json + report.csv + report.html.

    Args:
        csv_paths: long-format CSV (любое количество).
        out_dir: куда писать (создаётся при необходимости).
        jsonl_paths: параллельные JSONL сайдкары — из них HTML-отчёт
            подтягивает chunk_text/reasoning/raw_content/findings. Без них
            HTML тоже сгенерируется, но без рассуждений модели — что
            прибивает главный сценарий использования.

    Returns:
        Path к report.html.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_long = _load_long_csv([Path(p) for p in csv_paths])
    if df_long.empty:
        logger.warning("Empty CSV input — nothing to report")
        return out_dir / "report.html"

    jsonl_records = _load_jsonl([Path(p) for p in (jsonl_paths or [])])
    wide = _wide_per_chunk(df_long, jsonl_records=jsonl_records)
    agg = _aggregate_metrics(wide)

    (out_dir / "metrics.json").write_text(
        json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # CSV без findings_from_csv (он в JSONL и в HTML), без chunk_idx (служебный).
    csv_view_cols = [c for c in wide.columns
                     if c not in ("chunk_idx", "findings_from_csv")]
    wide[csv_view_cols].to_csv(
        out_dir / "report.csv", index=False, encoding="utf-8-sig",
    )
    _render_html(wide, agg, out_dir / "report.html",
                 jsonl_records=jsonl_records, jsonl_paths=jsonl_paths)

    logger.info("Wrote judge report bundle to %s", out_dir)
    return out_dir / "report.html"
