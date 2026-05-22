# airflow/dags

7 DAG'ов под Airflow Web UI на http://localhost:8080 (admin/admin
если поднят профиль `airflow`).

## DAG'и по слоям пайплайна

### Pre-existing (data prep)

| DAG | Trigger | Что делает |
|---|---|---|
| **ingest_dag** | manual (params: `video_path`) | ffmpeg → audio → ASR → JSON+TXT → Weaviate ingest |
| **benchmark_dag** | manual (params: single или batch) | JSON+MP3 → локальный ASR-бенчмарк (manifest + chunks) |
| **eval_dag** | `@daily 03:00 UTC` + manual | ASR runner на manifests + RAG runner на testset + summary |

### Новые (бывшие scripts/, мигрированы в Airflow)

| DAG | Trigger | Заменяет | Зачем |
|---|---|---|---|
| **judge_dag** | manual | `scripts/run_judge.py` | Один прогон LLM-judge с retry + MLflow tracking + xcom |
| **judge_sweep_dag** | manual | `scripts/run_judge_sweep.py` | 4 ASR backend под parent batch + leaderboard, динамический task mapping |
| **asr_eval_matrix_dag** | manual | `scripts/run_asr_eval_matrix.py` | N backend × M manifest с переключением VRAM (up→eval→down chain) |
| **full_benchmark_dag** | manual | `scripts/full_asr_benchmark.sh` | End-to-end: 4 ASR transcribe → judge sweep → leaderboard. Триггерит judge_sweep_dag внутри. |

## Зависимости DAG'ов

```
ingest_dag                                       (manual on new video)
                ↓
            data/transcripts/*.json
                ↓
benchmark_dag                                    (manual on new manifest)
                ↓
            data/eval/benchmarks/local/*
                ↓
eval_dag (daily) ── ASR runner ── RAG runner
                                       ↓
asr_eval_matrix_dag                              (manual, цикл по backend'ам)
                ↓
            CSV в data/eval/results/
                ↓
judge_dag (single) или judge_sweep_dag (4-way)   (manual)
                ↓
            leaderboard.html
                ↑
full_benchmark_dag триггерит judge_sweep_dag     (end-to-end one-click)
```

## Типичные сценарии

### A. Сравнить 4 ASR на готовых транскриптах (быстро)

Если транскрипты уже лежат в `data/transcripts/`:

```
trigger judge_sweep_dag with conf:
{
  "batch_name": "cnn_v15_4way",
  "prompt_version": "v15",
  "asr_configs": [
    {"name": "whisper-large-v3-turbo", "lectures": "/opt/airflow/project/data/transcripts/cnn_whisper.json"},
    {"name": "qwen3-asr-1.7b",         "lectures": "/opt/airflow/project/data/transcripts/cnn_qwen3_full.json"},
    {"name": "vibevoice-bnb4",         "lectures": "/opt/airflow/project/data/transcripts/cnn_vibevoice_full.json"},
    {"name": "parakeet-tdt-0.6b-v3",   "lectures": "/opt/airflow/project/data/transcripts/cnn_parakeet_full.json"}
  ]
}
```

### B. Полный прогон от mp3 до leaderboard

Если только аудио:

```
trigger full_benchmark_dag with conf:
{
  "audio": "/opt/airflow/project/data/audio/lecture.mp3",
  "lecture_id": "lecture_31_01_26",
  "prompt_version": "v15"
}
```

Внутри: транскрибирует 4 раза + триггерит judge_sweep_dag + ждёт его. Wallclock ~2-3 ч.

### C. ASR-метрики (WER/CER) без судьи

```
trigger asr_eval_matrix_dag with conf:
{
  "max_samples": 50,
  "manifests": [
    "/opt/airflow/project/data/eval/benchmarks/local/recsys_01/manifest.json"
  ]
}
```

Дефолтно прогоняет все 4 ASR backend по всем manifest'ам в `data/eval/benchmarks/local/`.

## Зачем DAG vs CLI

| Что | scripts/ CLI | Airflow DAG |
|---|---|---|
| Quick one-off | ✓ | overhead |
| Retry при failure | manual | ✓ автоматический |
| Manual schedule | cron сам | ✓ schedule= в DAG |
| Progress UI | tqdm в терминале | ✓ Web UI + Graph view |
| MLflow nesting | через CLI flag | ✓ через xcom |
| Cleanup on fail | manual | ✓ trigger_rule="all_done" |
| Audit trail | по логам | ✓ run history в DB |

Скрипты в `scripts/` **остались** — для разовых ad-hoc прогонов и для
случая когда Airflow не поднят. DAG'и дёргают те же `evaluation.*`
модули через subprocess.
