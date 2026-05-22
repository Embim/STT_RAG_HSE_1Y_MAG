"""Генерация ASR-eval отчёта (metrics.json + report.csv + report.html).

Эта логика вызывается двумя путями:
  - из `evaluation.asr.runner` после прогона eval (auto-report)
  - из `scripts/asr_eval_report.py` отдельно (если нужно перегенерировать)

Принимает long-format CSV из `csv_export.write_rows()` (где каждая строка =
(model, item_id, metric, value, extra)) и пишет в `out_dir` три файла в
стиле RAGAS-репорта.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

METRICS = ("wer", "cer", "mer", "wil")

# Пороги для цветовой шкалы в HTML. WER<0.05 = «отлично», >0.20 = «плохо».
# Зелёный → жёлто-зелёный → жёлтый → красный.
_THRESHOLDS = {
    "wer": (0.05, 0.10, 0.20),
    "cer": (0.03, 0.05, 0.10),
    "mer": (0.05, 0.10, 0.20),
    "wil": (0.10, 0.20, 0.40),
}


def _load_long_csv(paths: list[Path]) -> pd.DataFrame:
    """Загрузить и сконкатенировать несколько long-format CSV."""
    dfs = [pd.read_csv(p, encoding="utf-8-sig") for p in paths]
    df = pd.concat(dfs, ignore_index=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def _parse_extra(extra_raw: str) -> dict:
    if not isinstance(extra_raw, str) or not extra_raw.strip():
        return {}
    try:
        return json.loads(extra_raw)
    except Exception:
        return {}


def _wide_per_item(df: pd.DataFrame) -> pd.DataFrame:
    """Long → wide: 1 строка на (model, item_id), колонки = метрики."""
    real = df[df["item_id"] != "__corpus__"].copy()
    extras = real["extra"].apply(_parse_extra)
    real["reference"] = extras.apply(lambda d: d.get("reference", ""))
    real["hypothesis"] = extras.apply(lambda d: d.get("hypothesis", ""))
    real["duration"] = extras.apply(lambda d: d.get("duration", 0.0))
    real["speaker"] = real["item_id"].str.replace(r"_\d+$", "", regex=True)
    real["phrase_id"] = real["item_id"].str.extract(r"_(\d+)$").astype(int, errors="ignore")

    pivot = real.pivot_table(
        index=["model", "item_id", "speaker", "phrase_id"],
        columns="metric",
        values="value",
        aggfunc="first",
    ).reset_index()

    meta = (
        real.groupby(["model", "item_id"], as_index=False)
        .agg(reference=("reference", "first"),
             hypothesis=("hypothesis", "first"),
             duration=("duration", "first"))
    )
    wide = pivot.merge(meta, on=["model", "item_id"], how="left")
    front = ["model", "speaker", "phrase_id", "item_id",
             "reference", "hypothesis", "duration"]
    rest = [c for c in METRICS if c in wide.columns]
    return wide[front + rest].sort_values(
        ["model", "speaker", "phrase_id"]
    ).reset_index(drop=True)


def _aggregate_metrics(df_long: pd.DataFrame, wide: pd.DataFrame) -> dict:
    out: dict = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "n_items": int(wide["item_id"].nunique()),
        "models": sorted(wide["model"].unique().tolist()),
    }
    corpus = df_long[df_long["item_id"] == "__corpus__"]
    overall: dict[str, dict[str, float]] = {}
    for model, g in corpus.groupby("model"):
        overall[model] = {
            m: float(g.loc[g["metric"] == m, "value"].iloc[0])
            for m in METRICS if (g["metric"] == m).any()
        }
    out["overall_corpus"] = overall

    per_speaker: dict[str, dict[str, dict[str, float]]] = {}
    for model, gm in wide.groupby("model"):
        per_speaker[model] = {}
        for speaker, gs in gm.groupby("speaker"):
            per_speaker[model][speaker] = {
                "n": int(len(gs)),
                **{m: float(gs[m].mean()) for m in METRICS if m in gs.columns},
            }
    out["per_speaker_mean"] = per_speaker
    return out


def _color_for(metric: str, value: float) -> str:
    if pd.isna(value):
        return ""
    good, mid, bad = _THRESHOLDS[metric]
    if value <= good:
        return "background-color: rgb(80,200,80)"
    if value <= mid:
        return "background-color: rgb(180,215,80)"
    if value <= bad:
        return "background-color: rgb(235,200,80)"
    return "background-color: rgb(220,80,80)"


def _render_html(wide: pd.DataFrame, agg: dict, out_path: Path) -> None:
    display_cols = ["model", "speaker", "phrase_id", "reference", "hypothesis",
                    "duration", *[m for m in METRICS if m in wide.columns]]
    view = wide[display_cols].copy()

    def color_row(row):
        styles = [""] * len(row)
        for i, col in enumerate(row.index):
            if col in METRICS:
                styles[i] = _color_for(col, row[col])
        return styles

    styler = view.style.apply(color_row, axis=1).format({
        "duration": "{:.1f}",
        **{m: "{:.3f}" for m in METRICS if m in view.columns},
    })

    def trunc(s, n=120):
        return s if not isinstance(s, str) else (s[:n] + "…" if len(s) > n else s)
    styler = styler.format({"reference": trunc, "hypothesis": trunc}, na_rep="")
    table_html = styler.to_html()

    def _fmt(v):
        return f"{v:.4f}" if isinstance(v, (int, float)) and not pd.isna(v) else "—"

    sections = ["<h2>Corpus aggregates</h2>"]
    corpus_rows = []
    for model, mvals in agg["overall_corpus"].items():
        corpus_rows.append(
            "<tr><td>" + model + "</td>" +
            "".join(f"<td>{_fmt(mvals.get(m))}</td>" for m in METRICS) +
            "</tr>"
        )
    sections.append(
        "<table><thead><tr><th>model</th>" +
        "".join(f"<th>{m}</th>" for m in METRICS) +
        "</tr></thead><tbody>" + "".join(corpus_rows) + "</tbody></table>"
    )

    sections.append("<h2>Per-speaker (mean across phrases)</h2>")
    sp_rows = []
    for model, by_sp in agg["per_speaker_mean"].items():
        for speaker, v in by_sp.items():
            sp_rows.append(
                f"<tr><td>{model}</td><td>{speaker}</td><td>{v['n']}</td>" +
                "".join(f"<td>{_fmt(v.get(m))}</td>" for m in METRICS) +
                "</tr>"
            )
    sections.append(
        "<table><thead><tr><th>model</th><th>speaker</th><th>n</th>" +
        "".join(f"<th>{m}</th>" for m in METRICS) +
        "</tr></thead><tbody>" + "".join(sp_rows) + "</tbody></table>"
    )

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>ASR eval report</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 24px; color: #222; }}
table {{ border-collapse: collapse; margin: 12px 0; }}
table, th, td {{ border: 1px solid #ddd; padding: 6px 10px; }}
th {{ background: #f5f5f5; text-align: left; }}
h1, h2 {{ margin: 12px 0; }}
.meta {{ color: #666; }}
</style></head><body>
<h1>ASR Eval — {agg['generated_at']}</h1>
<p class="meta">n_items={agg['n_items']}, models={', '.join(agg['models'])}</p>
{''.join(sections)}
<h2>Per-question</h2>
{table_html}
</body></html>"""
    out_path.write_text(html, encoding="utf-8")


def generate_report(csv_paths: list[Path], out_dir: Path) -> Path:
    """Сгенерировать metrics.json + report.csv + report.html.

    Args:
        csv_paths: long-format CSV (любое количество). Если передано несколько
            файлов от разных моделей — получится сравнительный отчёт.
        out_dir: куда писать (создаётся при необходимости). Содержимое
            перезаписывается.

    Returns:
        Path к report.html (для логирования / открытия в браузере).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_long = _load_long_csv([Path(p) for p in csv_paths])
    wide = _wide_per_item(df_long)
    agg = _aggregate_metrics(df_long, wide)

    (out_dir / "metrics.json").write_text(
        json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    wide.to_csv(out_dir / "report.csv", index=False, encoding="utf-8-sig")
    _render_html(wide, agg, out_dir / "report.html")

    logger.info("Wrote report bundle to %s", out_dir)
    return out_dir / "report.html"
