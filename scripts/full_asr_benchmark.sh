#!/bin/bash
# Полный прогон сравнения 4 ASR на 3-часовой CNN лекции через LLM-judge.
#
# Шаги:
#   1. Опустить asr-judge
#   2. Транскрибировать через qwen3-asr-1.7b → cnn_qwen3.json
#   3. Опустить qwen3, поднять vibevoice → cnn_vibevoice.json
#   4. Опустить vibevoice, поднять parakeet → cnn_parakeet.json
#   5. Опустить parakeet, поднять judge
#   6. Прогон judge v10 на каждом из 4 транскриптов (whisper, qwen3, vibevoice, parakeet)
#   7. Сгенерировать 4-way leaderboard
#
# Использование: запускается в фоне, логирует в stdout, в конце пишет leaderboard.
# Ожидаемое wallclock ~2 часа (vibevoice самый долгий, 43 минуты transcribe).

set -e  # fail fast

AUDIO="E:/video_pipeline/audio/Глубинное обучение Свёрточные нейронные сети (CNN) (31.01.26).mp4"
REPO=$(pwd)
TS=$(date +%Y%m%d_%H%M%S)

echo "=== START $(date) ==="

cleanup_running() {
  local profile=$1
  echo "[$(date +%H:%M:%S)] Stopping $profile ..."
  docker compose --profile "$profile" down >/dev/null 2>&1 || true
}

wait_http() {
  local url=$1
  local label=$2
  echo "[$(date +%H:%M:%S)] Waiting for $label HTTP ready..."
  until curl -fsS "$url" > /dev/null 2>&1; do sleep 8; done
  echo "[$(date +%H:%M:%S)] $label READY"
}

run_transcribe() {
  local asr_name=$1
  local output=$2
  echo "[$(date +%H:%M:%S)] === Transcribing $asr_name → $output (full lecture, no --max-chunks) ==="
  python -X utf8 scripts/transcribe_long_audio.py \
    --input "$AUDIO" \
    --output "$output" \
    --asr-name "$asr_name" \
    --asr-url http://127.0.0.1:8000 \
    --chunk-minutes 5
}

run_judge() {
  local transcript=$1
  local asr_name=$2
  local run_name=$3
  echo "[$(date +%H:%M:%S)] === Judge on $asr_name (run=$run_name) ==="
  python -X utf8 scripts/run_judge.py \
    --transcript "$transcript" \
    --prompt-version v10 \
    --asr-name "$asr_name" \
    --run-name "$run_name" \
    --no-up
}

# ─── Phase 1: ensure starting state ─────────────────────────────
echo "[$(date +%H:%M:%S)] Phase 0: prep — opt out of asr-judge"
cleanup_running asr-judge

# ─── Phase 2: qwen3-asr ─────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Phase 1: qwen3-asr"
docker compose --profile asr-qwen3 up -d
wait_http "http://127.0.0.1:8000/v1/models" "qwen3-asr"
run_transcribe "qwen3-asr-1.7b" "data/transcripts/cnn_qwen3_full.json"
cleanup_running asr-qwen3

# ─── Phase 3: vibevoice ─────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Phase 2: vibevoice"
docker compose --profile asr-vibevoice up -d
wait_http "http://127.0.0.1:8000/v1/models" "vibevoice"
run_transcribe "vibevoice-bnb4" "data/transcripts/cnn_vibevoice_full.json"
cleanup_running asr-vibevoice

# ─── Phase 4: parakeet ──────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Phase 3: parakeet"
docker compose --profile asr-parakeet up -d
wait_http "http://127.0.0.1:8000/v1/models" "parakeet"
run_transcribe "parakeet-tdt-0.6b-v3" "data/transcripts/cnn_parakeet_full.json"
cleanup_running asr-parakeet

# ─── Phase 5: judge ─────────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Phase 4: judge (full lecture, all chunks)"
docker compose --profile asr-judge up -d
wait_http "http://127.0.0.1:8002/v1/models" "judge"

run_judge "data/transcripts/cnn_whisper.json"        "whisper-large-v3-turbo"  "asr_full_whisper_$TS"
run_judge "data/transcripts/cnn_qwen3_full.json"     "qwen3-asr-1.7b"          "asr_full_qwen3_$TS"
run_judge "data/transcripts/cnn_vibevoice_full.json" "vibevoice-bnb4"          "asr_full_vibevoice_$TS"
run_judge "data/transcripts/cnn_parakeet_full.json"  "parakeet-tdt-0.6b-v3"    "asr_full_parakeet_$TS"

# ─── Phase 6: compare ───────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Phase 5: compare report"
python -X utf8 scripts/judge_compare_asr.py \
  "data/eval/results/judge_asr_full_whisper_${TS}_"*.csv \
  "data/eval/results/judge_asr_full_qwen3_${TS}_"*.csv \
  "data/eval/results/judge_asr_full_vibevoice_${TS}_"*.csv \
  "data/eval/results/judge_asr_full_parakeet_${TS}_"*.csv \
  --out "data/eval/outputs/asr_compare_cnn_full_$TS"

echo ""
echo "=== DONE $(date) ==="
echo "Leaderboard: data/eval/outputs/asr_compare_cnn_full_$TS/report.html"
echo ""
python -X utf8 -c "
import json
import sys
from pathlib import Path
metrics = sorted(Path('data/eval/outputs').glob('asr_compare_cnn_full_*/metrics.json'))[-1]
m = json.loads(metrics.read_text(encoding='utf-8'))
print(f'{\"rank\":>4} {\"ASR\":<28} {\"Q-Score\":>8} {\"WED\":>6} {\"err/1k\":>8} {\"term\":>5} {\"halluc\":>7} {\"gram\":>5} {\"gap\":>4} {\"chars\":>8}')
print('-' * 96)
for rank, (asr, mv) in enumerate(sorted(m['overall_corpus'].items(), key=lambda x: -x[1]['judge_quality_score']), 1):
    medal = ['🥇','🥈','🥉'][rank-1] if rank <= 3 else f' {rank}'
    print(f'{medal:>4} {asr:<28} {mv[\"judge_quality_score\"]:>8.1f} {mv[\"weighted_error_density\"]:>6.2f} {mv[\"errors_per_1k_chars\"]:>8.2f} {mv[\"count_terminology\"]:>5} {mv[\"count_hallucination\"]:>7} {mv[\"count_grammar\"]:>5} {mv[\"count_gap\"]:>4} {mv[\"n_chars_total\"]:>8,}')
"
