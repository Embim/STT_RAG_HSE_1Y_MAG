"""CLI для запуска LLM-судьи по транскрипциям.

Два режима:

  # 1. Benchmark — построчная сверка reference vs hypothesis для каждой модели
  python -m evaluation.judge.runner \
      --source benchmark \
      --csv data/eval/results/asr_whisper_large_v3_turbo__manifest_20260513_201929.csv \
      --csv data/eval/results/asr_qwen3_shared_dl_ml_20260513_185547.csv \
      --prompt-version v1 \
      --run-name judge_pilot_4backends \
      --max-items 10

  # 2. Lecture — reference-free разбор полных лекций
  python -m evaluation.judge.runner \
      --source lecture \
      --lectures data/transcripts \
      --prompt-version v1 \
      --run-name judge_recsys_lectures

  # 3. Список доступных версий промпта
  python -m evaluation.judge.runner --list-prompts

Артефакты:
  - data/eval/results/judge_<run-name>_<ts>.csv  — canonical schema с
    count_<type>, sev_avg_<type>, n_findings_total per chunk;
  - data/eval/results/judge_<run-name>_<ts>.jsonl — полный fidelity-сайдкар
    с reasoning, raw content, findings, prompt_version. Открыть → читать
    reasoning → править prompts/judge/transcription_review_v2.yaml.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import re
import sys
from pathlib import Path

from evaluation.judge.client import JudgeClient
from evaluation.judge.prompt_loader import list_versions, load_prompt
from evaluation.judge.reviewer import review_items
from evaluation.judge.source import (
    discover_csv,
    discover_lectures,
    iter_benchmark_csv,
    iter_lecture_jsons,
)
from evaluation.reporting import langfuse_export
from evaluation.reporting.judge_report import generate_report
from processing.progress_tracker import mlflow_span, tracked_run
from settings import settings

logger = logging.getLogger(__name__)


# MLflow ограничивает имена метрик символами `[A-Za-z0-9_./\- ]`. ASR имена
# у нас типа "qwen3-asr-1.7b" или "faster-whisper-large-v3-turbo" — `-` и `.`
# разрешены, но мы для надёжности нормализуем в snake_case.
_METRIC_SLUG = re.compile(r"[^A-Za-z0-9_]+")


def _slug_metric(name: str) -> str:
    return _METRIC_SLUG.sub("_", name).strip("_")


def _is_nan(v: object) -> bool:
    return isinstance(v, float) and math.isnan(v)


def _log_findings_table(mlrun, jsonl_path) -> None:
    """Собрать DataFrame со всеми findings из JSONL и залить через log_table.

    Колонки: chunk_id, chunk_idx, asr_model, error_type, severity, evidence,
    suggestion, confidence, explanation. Если pandas не установлен — silent
    skip (всё равно у нас pandas в eval extras обязателен).
    """
    if jsonl_path is None or not Path(jsonl_path).exists():
        return
    try:
        import pandas as pd
    except ImportError:
        logger.debug("pandas not available, skipping log_table")
        return

    rows = []
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                chunk_id = rec.get("chunk_id", "")
                # Извлекаем chunk_idx из суффикса `_chunk_NNN`.
                idx = 0
                if "_chunk_" in chunk_id:
                    try:
                        idx = int(chunk_id.rsplit("_chunk_", 1)[1])
                    except (ValueError, IndexError):
                        pass
                asr_model = rec.get("asr_model", "")
                for finding in (rec.get("findings") or []):
                    if not isinstance(finding, dict):
                        continue
                    rows.append({
                        "chunk_id": chunk_id,
                        "chunk_idx": idx,
                        "asr_model": asr_model,
                        "error_type": finding.get("error_type", ""),
                        "severity": finding.get("severity", ""),
                        "evidence": finding.get("evidence", ""),
                        "suggestion": finding.get("suggestion", ""),
                        "confidence": finding.get("confidence", 0.0),
                        "explanation": finding.get("explanation", ""),
                    })
    except Exception as e:
        logger.warning("Failed to read JSONL for findings table: %s", e)
        return

    if not rows:
        return
    df = pd.DataFrame(rows).sort_values(["asr_model", "chunk_idx", "error_type"])
    mlrun.log_table(df, artifact_file="findings.json")
    logger.info("Logged findings table to MLflow (%d findings)", len(df))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evaluation.judge.runner",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--source", choices=["benchmark", "lecture"], default=None,
        help="Тип входа: benchmark (CSV из data/eval/results/) или lecture (JSON из data/transcripts/).",
    )
    p.add_argument(
        "--csv", action="append", default=[],
        help="Пути к ASR-eval CSV или директория. Можно повторять --csv N раз.",
    )
    p.add_argument(
        "--lectures", action="append", default=[],
        help="Пути к транскрипциям лекций или директория. Можно повторять.",
    )
    p.add_argument(
        "--prompt-version", default="v1",
        help="Версия промпта (prompts/judge/transcription_review_<version>.yaml)",
    )
    p.add_argument("--prompt-name", default="transcription_review")
    p.add_argument("--list-prompts", action="store_true",
                   help="Показать список доступных версий и выйти")
    p.add_argument("--run-name", default=None,
                   help="Метка для CSV/JSONL имени (default: judge_<version>_<source>)")
    p.add_argument("--max-items", type=int, default=None,
                   help="Ограничить количество JudgeItem (для отладки)")
    p.add_argument("--max-chunks-per-item", type=int, default=None,
                   help="Остановиться после N чанков каждого item — для pilot-прогона "
                        "промпта на 2-3 чанках вместо полной лекции")
    p.add_argument("--max-chars", type=int, default=None,
                   help="Override settings.JUDGE_MAX_INPUT_CHARS")
    p.add_argument("--models", nargs="+", default=None,
                   help="benchmark mode: только эти ASR-модели (фильтр по полю `model`)")
    p.add_argument("--asr-name", default=None,
                   help="lecture mode: имя ASR-системы которая создала эти "
                        "транскрипты (например 'whisper-large-v3-turbo'). "
                        "Попадает в metadata.model и группирует findings в "
                        "leaderboard. Без него все прогоны схлопываются в одну "
                        "группу с пустым asr_model и сравнение невозможно.")
    p.add_argument("--judge-url", default=None,
                   help="Override settings.JUDGE_URL (по умолчанию http://localhost:8002)")
    p.add_argument("--judge-model", default=None,
                   help="Override settings.JUDGE_MODEL_ID (по умолчанию Qwen/Qwen3-8B-FP8)")
    p.add_argument("--skip-healthcheck", action="store_true",
                   help="Не делать GET /v1/models перед стартом (нужно для smoke без судьи)")
    p.add_argument("--mlflow-parent-run-id", default=None,
                   help="Если задан, этот judge-прогон линкуется к parent batch-run "
                        "в MLflow через тэг parent_batch_run_id. Используется "
                        "scripts/run_judge_sweep.py при прогоне 4 backend-ов под "
                        "одним зонтиком.")
    p.add_argument("--mlflow-nested", action="store_true",
                   help="Открыть judge-run как nested run (для sweep'а в одном "
                        "процессе с открытым parent batch run). В UI отобразится "
                        "как drilldown child этого parent'а. Требует уже активного "
                        "parent run в этом же процессе — иначе start_run упадёт.")
    return p


async def _run(args: argparse.Namespace) -> int:
    if args.list_prompts:
        vers = list_versions(name=args.prompt_name)
        if not vers:
            print(f"No prompts found for name={args.prompt_name!r} in "
                  f"{settings.JUDGE_PROMPT_DIR}")
            return 1
        print("Available prompt versions:")
        for v in vers:
            print(f"  --prompt-version {v}")
        return 0

    if not args.source:
        print("--source is required (benchmark|lecture). See --help.", file=sys.stderr)
        return 2

    prompt = load_prompt(args.prompt_version, name=args.prompt_name)
    logger.info(
        "Loaded prompt %s version=%s from %s",
        prompt.name, prompt.version, prompt.source_path,
    )

    client = JudgeClient(base_url=args.judge_url, model_id=args.judge_model)
    if not args.skip_healthcheck:
        ok = await client.health()
        if not ok:
            logger.error(
                "Judge unreachable at %s — поднимай профиль:\n"
                "    docker compose --profile asr-judge up -d",
                client.base_url,
            )
            return 3
        logger.info("Judge healthy at %s (model=%s)", client.base_url, client.model_id)

    # Источник
    if args.source == "benchmark":
        csv_paths = discover_csv(args.csv) if args.csv else []
        if not csv_paths:
            print("--source benchmark needs at least one --csv path", file=sys.stderr)
            return 2
        items_iter = iter_benchmark_csv(csv_paths, models_filter=args.models)
        default_run = f"judge_{args.prompt_version}_benchmark"
    else:
        lecture_paths = discover_lectures(args.lectures) if args.lectures else []
        if not lecture_paths:
            print("--source lecture needs at least one --lectures path", file=sys.stderr)
            return 2
        items_iter = iter_lecture_jsons(lecture_paths, asr_name_override=args.asr_name)
        default_run = f"judge_{args.prompt_version}_lecture"

    if args.max_items is not None:
        items_iter = _take(items_iter, args.max_items)

    run_name = args.run_name or default_run
    logger.info("Run name: %s", run_name)

    # MLflow tracking — fail-soft. Один run = один прогон judge.runner.
    # Per-chunk метрики льются time-series'ом внутри review_items (step=idx).
    # Корпусные числа и report.html логируем после прогона.
    mlflow_params = {
        "judge_name": settings.JUDGE_NAME,
        "judge_model_id": settings.JUDGE_MODEL_ID,
        "judge_url": client.base_url,
        "prompt_version": prompt.version,
        "prompt_name": prompt.name,
        "source": args.source,
        "max_input_chars": args.max_chars or settings.JUDGE_MAX_INPUT_CHARS,
        "max_chunks_per_item": args.max_chunks_per_item or 0,
        "asr_name": args.asr_name or "",
        # YAML model_hints — короткие ключи как params (числа и булы).
        "judge_temperature": prompt.model_hints.get("temperature",
                                                    settings.JUDGE_TEMPERATURE),
        "judge_max_tokens": prompt.model_hints.get("max_tokens",
                                                    settings.JUDGE_MAX_OUTPUT_TOKENS),
        "judge_repetition_penalty": prompt.model_hints.get("repetition_penalty", 0),
        "judge_frequency_penalty": prompt.model_hints.get("frequency_penalty", 0),
        "judge_presence_penalty": prompt.model_hints.get("presence_penalty", 0),
        "judge_seed": prompt.model_hints.get("seed", 0),
    }
    mlflow_tags = {
        "kind": "judge_eval",
        "prompt_version": prompt.version,
        "judge_name": settings.JUDGE_NAME,
        "asr_name": args.asr_name or "(none)",
        "source": args.source,
    }

    # Override experiment name только на время этого вызова — не трогаем
    # глобальный settings.MLFLOW_EXPERIMENT_NAME (он для ingest-прогонов).
    prev_experiment = os.environ.get("MLFLOW_EXPERIMENT_NAME")
    os.environ["MLFLOW_EXPERIMENT_NAME"] = settings.MLFLOW_JUDGE_EXPERIMENT

    report_html: Path | None = None
    try:
        # Outer Langfuse + MLflow Tracing spans обнимают весь прогон.
        # Per-chunk inner spans открываются в reviewer.review_items и
        # автоматически attach'атся к этим outer'ам через OTel / MLflow context.
        outer_lf_metadata = {
            "judge_name": settings.JUDGE_NAME,
            "judge_url": client.base_url,
            "prompt_version": prompt.version,
            "asr_name": args.asr_name or "",
            "source": args.source,
            "max_input_chars": args.max_chars or settings.JUDGE_MAX_INPUT_CHARS,
        }
        outer_mlf_inputs = {
            "judge_model_id": settings.JUDGE_MODEL_ID,
            "prompt_version": prompt.version,
            "asr_name": args.asr_name or "",
        }

        with tracked_run(
            title=run_name,
            kind="judge_eval",
            params=mlflow_params,
            tags=mlflow_tags,
            parent_run_id=args.mlflow_parent_run_id,
            nested=args.mlflow_nested,
        ) as mlrun, langfuse_export.start_run_span(
            name=run_name,
            metadata=outer_lf_metadata,
        ), mlflow_span(
            run_name,
            span_type="AGENT",
            inputs=outer_mlf_inputs,
            attributes={"kind": "judge_eval", "prompt_version": prompt.version},
        ) as outer_ml_span:
            result = await review_items(
                items_iter,
                client=client,
                prompt=prompt,
                run_name=run_name,
                max_chars_per_chunk=args.max_chars,
                max_chunks_per_item=args.max_chunks_per_item,
                mlrun=mlrun,
            )

            # Авто-генерация отчёта по этому прогону — metrics.json + report.csv +
            # report.html в data/eval/outputs/<csv_stem>/. Та же логика как у ASR
            # runner-а: 1-к-1 соответствие CSV ↔ папка с отчётом, история сохраняется
            # автоматически.
            #
            # Не блокируем return code если report-генерация падает: CSV+JSONL уже
            # на диске, отчёт — побочный артефакт, который можно перегенерировать
            # вручную через scripts/judge_eval_report.py.
            metrics_json_path: Path | None = None
            if result.csv_path is not None:
                repo_root = Path(__file__).resolve().parents[3]
                out_dir = repo_root / "data" / "eval" / "outputs" / result.csv_path.stem
                try:
                    report_html = generate_report(
                        [result.csv_path], out_dir,
                        jsonl_paths=[result.jsonl_path] if result.jsonl_path else None,
                    )
                    metrics_json_path = out_dir / "metrics.json"
                    logger.info("Report bundle ready: %s", report_html)
                except Exception as e:
                    logger.warning("Auto-report generation failed: %s "
                                   "(CSV+JSONL всё ещё на диске)", e)

            # MLflow: корпусные метрики из metrics.json + report.html как артефакт.
            # Делаем ПОСЛЕ генерации отчёта, чтобы потерять только эти числа в
            # случае сбоя report, но не метрики прогона (они уже залиты per-chunk).
            if metrics_json_path is not None and metrics_json_path.exists():
                try:
                    metrics_dict = json.loads(metrics_json_path.read_text("utf-8"))
                    corpus = metrics_dict.get("overall_corpus", {})
                    # На случай если внутри несколько ASR-моделей в одном CSV —
                    # ключ метрики получает префикс `<asr>__`. В типовом случае
                    # (один backend на прогон) префикс совпадает у всех метрик
                    # и в UI читается легко.
                    multi = len(corpus) > 1
                    for asr_model, mv in corpus.items():
                        prefix = (f"{_slug_metric(asr_model)}__"
                                  if multi else "")
                        for k, v in mv.items():
                            if isinstance(v, (int, float)) and not _is_nan(v):
                                mlrun.log_metric(f"corpus_{prefix}{k}", float(v))
                    mlrun.set_tag("n_chunks", str(metrics_dict.get("n_chunks", 0)))
                    mlrun.set_tag("n_items", str(metrics_dict.get("n_items", 0)))
                except Exception as e:
                    logger.warning("MLflow corpus rollup failed: %s", e)

            # Артефакты: HTML отчёт, исходники (CSV/JSONL/metrics.json) и сам
            # prompt YAML. Кладём всё на root run-а, видно прямо в Artifacts вкладке.
            # JSONL обычно 10-20 MB на корпус — не критично для S3 MinIO.
            for artifact in [
                report_html,
                result.csv_path,
                result.jsonl_path,
                metrics_json_path,
                prompt.source_path,
            ]:
                if artifact is not None and Path(artifact).exists():
                    mlrun.log_artifact(str(artifact))

            # findings table — pandas DataFrame со всеми findings из JSONL,
            # рендерится в MLflow UI как sortable table. Позволяет смотреть
            # evidence/suggestion прямо в браузере, без открытия JSONL.
            _log_findings_table(mlrun, result.jsonl_path)

            # Corpus scores на outer Langfuse span — мы всё ещё внутри
            # start_run_span контекста, поэтому score_current прикрепит
            # их к evaluator-span'у (видно в Langfuse UI как run-level scores).
            # Делаем здесь, не в reviewer, потому что corpus метрики живут в
            # metrics.json, который генерится после review_items.
            if metrics_json_path is not None and metrics_json_path.exists():
                try:
                    metrics_dict = json.loads(metrics_json_path.read_text("utf-8"))
                    corpus = metrics_dict.get("overall_corpus", {})
                    for asr_model, mv in corpus.items():
                        suffix = f"__{_slug_metric(asr_model)}" if len(corpus) > 1 else ""
                        # Самые ценные числа: Q-Score + density + parse_err_rate.
                        for k in (
                            "judge_quality_score",
                            "weighted_error_density",
                            "errors_per_1k_chars",
                            "findings_per_chunk",
                            "parse_error_rate",
                            "prefill_tps_corpus",
                            "decode_tps_corpus",
                        ):
                            v = mv.get(k)
                            if isinstance(v, (int, float)) and not _is_nan(v):
                                langfuse_export.score_current(
                                    name=f"{k}{suffix}", value=float(v),
                                    comment=asr_model,
                                )
                    # Outer MLflow span — output = краткое корпусное саммари.
                    try:
                        outer_ml_span.set_outputs({
                            "n_chunks": metrics_dict.get("n_chunks", 0),
                            "n_findings_total": metrics_dict.get("n_findings_total", 0),
                            "corpus": corpus,
                        })
                    except Exception:
                        pass
                except Exception as e:
                    logger.warning("Langfuse corpus scores failed: %s", e)

            # Flush Langfuse, чтобы все span/scores ушли на сервер до того,
            # как event loop закроется (иначе SDK выдаёт warning о background
            # threads не завершившихся при exit).
            try:
                langfuse_export.flush()
            except Exception as e:
                logger.debug("Langfuse flush failed: %s", e)
    finally:
        if prev_experiment is None:
            os.environ.pop("MLFLOW_EXPERIMENT_NAME", None)
        else:
            os.environ["MLFLOW_EXPERIMENT_NAME"] = prev_experiment

    # Финальное саммари — чтобы было видно в stdout без открывания CSV.
    print()
    print("=" * 60)
    print(f"Run: {run_name}")
    print(f"  items:    {result.n_items}")
    print(f"  chunks:   {result.n_chunks}")
    print(f"  findings: {result.n_findings}")
    if result.counts_by_type:
        print("  by type:")
        for et, n in sorted(result.counts_by_type.items()):
            print(f"    {et:14s} {n}")
    print(f"  CSV:    {result.csv_path}")
    print(f"  JSONL:  {result.jsonl_path}")
    if report_html is not None:
        print(f"  Report: {report_html}")
    print("=" * 60)
    return 0


def _take(iterable, n: int):
    for i, x in enumerate(iterable):
        if i >= n:
            return
        yield x


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_parser().parse_args()
    rc = asyncio.run(_run(args))
    sys.exit(rc)


if __name__ == "__main__":
    main()
