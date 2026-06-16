"""Прогон ASR eval-матрицы по всем backends × выбранным manifests.

Что делает:
    1. Для каждого backend (asr-whisper, asr-qwen3, ...) по очереди:
       up → wait healthy → запустить evaluation.asr.runner на каждом
       manifest'е → down → следующий backend.
    2. Между backend'ами GPU освобождается (down контейнера), потом
       стартует следующий — так на 5080 16 GB можно прогнать все 5
       моделей без OOM.
    3. CSV-результаты падают в data/eval/results/<run-name>_<ts>.csv,
       Langfuse получает per-item scores, MLflow — corpus aggregates.

Запуск из корня репо:

    # Все 5 backends × combined manifest
    python scripts/run_asr_eval_matrix.py

    # Только конкретные backends, конкретные manifests
    python scripts/run_asr_eval_matrix.py \
        --backends asr-whisper asr-qwen3 \
        --manifests data/eval/recordings/shared_dl_ml/manifest_german.json \
                    data/eval/recordings/shared_dl_ml/manifest_izbrannoe.json

    # Только показать план, не запускать
    python scripts/run_asr_eval_matrix.py --dry-run

Перед запуском убедись, что профили уже прогреты через
scripts/preload_asr_backends.py (иначе первый запуск каждой модели
включает download весов с HF, и таймаут лимита может не хватить).
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Backend:
    """ASR backend spec для матрицы прогона.

    Args:
        profile: имя docker-compose профиля (asr-whisper / asr-qwen3 / ...)
        container: имя контейнера (нужно для docker inspect health)
        asr_name: человекочитаемая метка для CSV / Langfuse / MLflow
        asr_model_id: model id, который backend репортит как `model`
        endpoint: 'transcription' (default OpenAI /v1/audio/transcriptions)
            или 'chat' (/v1/chat/completions с audio как data URI base64).
            Qwen3-ASR требует 'chat' — у /v1/audio/transcriptions в vLLM
            0.20.2 сломан pipeline (возвращает пустой text).
        healthy_timeout: сколько ждать healthy после up -d (сек)
    """
    profile: str
    container: str
    asr_name: str
    asr_model_id: str
    healthy_timeout: int
    endpoint: str = "transcription"


# Список backends — соответствует тому, что в docker-compose.yml.
BACKENDS: list[Backend] = [
    Backend(
        profile="asr-whisper",
        container="asr-whisper",
        asr_name="whisper_large_v3_turbo",
        asr_model_id="deepdml/faster-whisper-large-v3-turbo-ct2",
        # Image fedirz/faster-whisper-server делает `uv pip install -e` при
        # старте → тянет hatchling из pypi. На медленном DNS / прогретом
        # кеше cold-start ~2-3 минуты. 120s было мало.
        healthy_timeout=300,
    ),
    Backend(
        profile="asr-qwen3",
        container="asr-qwen3",
        asr_name="qwen3_asr_1.7b",
        asr_model_id="Qwen/Qwen3-ASR-1.7B",
        endpoint="chat",  # transcription endpoint сломан в vLLM 0.20.2
        healthy_timeout=600,
    ),
    Backend(
        profile="asr-parakeet",
        container="asr-parakeet",
        asr_name="parakeet_tdt_v3",
        asr_model_id="nvidia/parakeet-tdt-0.6b-v3",
        healthy_timeout=600,
    ),
    Backend(
        profile="asr-phi4-nvfp4",
        container="asr-phi4-nvfp4",
        asr_name="phi4_mm_nvfp4",
        asr_model_id="nvidia/Phi-4-multimodal-instruct-NVFP4",
        # trtllm-serve не имплементирует /v1/audio/transcriptions для Phi-4
        # (возвращает 404). Используем /v1/chat/completions с audio как
        # data URI — как у Qwen3-ASR.
        # ВАЖНО: в trtllm-serve 1.2.0rc6 этот путь сломан в 2 местах:
        #   - data:/file:// URI в audio_url не декодируются (issue #14100)
        #   - KV cache block reuse ломает 2-й multimodal request (issue #14125)
        # Backend оставлен в матрице чтобы можно было быстро вернуть после
        # того, как NVIDIA выпустит fix (PR #14010 + multimodal-aware
        # KV cache default). Сейчас при запуске даст ошибку — это ожидаемо.
        endpoint="chat",
        healthy_timeout=1200,
    ),
    Backend(
        profile="asr-vibevoice",
        container="asr-vibevoice",
        asr_name="vibevoice_asr_bnb4",
        asr_model_id="microsoft/VibeVoice-ASR-HF",
        healthy_timeout=1800,
    ),
]

# Manifests, которые проходим по дефолту. Combined даёт средний WER, per-speaker
# вытягивает scoreboard по дикторам (важно: speaker-conditional variance).
DEFAULT_MANIFESTS: list[Path] = [
    REPO_ROOT / "data/eval/recordings/shared_dl_ml/manifest.json",
]


def run(cmd: list[str], *, check: bool = True, env: dict | None = None) -> int:
    logger.info("$ %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=check)
    return proc.returncode


def stop_running_asr_containers() -> None:
    """Гасит любые asr-* контейнеры, которые сейчас запущены.

    Нужно перед началом матрицы — иначе если предыдущий смок-тест оставил
    backend на :8000, наш `up -d` упадёт с 'port is already allocated'.
    """
    res = subprocess.run(
        ["docker", "ps", "--filter", "name=^asr-", "--format", "{{.Names}}"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    names = [n.strip() for n in res.stdout.splitlines() if n.strip()]
    if not names:
        return
    logger.info("Pre-flight: stopping running ASR containers: %s", ", ".join(names))
    subprocess.run(["docker", "stop", *names],
                   capture_output=True, cwd=REPO_ROOT)
    subprocess.run(["docker", "rm", *names],
                   capture_output=True, cwd=REPO_ROOT)


def health_status(container: str) -> str:
    res = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Health.Status}}", container],
        capture_output=True, text=True,
    )
    return res.stdout.strip() if res.returncode == 0 else "missing"


def wait_healthy(backend: Backend) -> bool:
    deadline = time.monotonic() + backend.healthy_timeout
    last = ""
    while time.monotonic() < deadline:
        status = health_status(backend.container)
        if status != last:
            elapsed = int(backend.healthy_timeout - (deadline - time.monotonic()))
            logger.info("[%s] %s (after %ds)", backend.profile, status, elapsed)
            last = status
        if status == "healthy":
            return True
        if status == "unhealthy":
            logger.error("[%s] unhealthy — bailing on this backend", backend.profile)
            return False
        time.sleep(10)
    logger.warning("[%s] not healthy within %ds", backend.profile, backend.healthy_timeout)
    return False


def run_eval(backend: Backend, manifest: Path) -> bool:
    """Запустить evaluation.asr.runner на одном manifest. True если RC=0."""
    if not manifest.exists():
        logger.error("Manifest not found: %s", manifest)
        return False

    manifest_label = manifest.stem  # e.g. manifest_german
    run_name = f"{backend.asr_name}__{manifest_label}"
    # ASR_NAME / ASR_MODEL_ID → попадают в CSV-строки и Langfuse scores
    # как идентификатор модели. WHISPER_URL по дефолту = http://localhost:8000
    # — то, что наш профиль выдаёт наружу.
    # PYTHONPATH=src нужен потому что проект использует src/-layout и не
    # установлен через pip install -e — модуль `evaluation` живёт в src/.
    src_path = str(REPO_ROOT / "src")
    existing_pp = os.environ.get("PYTHONPATH", "")
    new_pp = src_path if not existing_pp else f"{src_path}{os.pathsep}{existing_pp}"
    env = {
        **os.environ,
        "ASR_NAME": backend.asr_name,
        "ASR_MODEL_ID": backend.asr_model_id,
        "ASR_ENDPOINT": backend.endpoint,
        "PYTHONPATH": new_pp,
    }
    cmd = [
        sys.executable, "-m", "evaluation.asr.runner",
        "--local-benchmark", str(manifest),
        "--run-name", run_name,
        "--langfuse-dataset", manifest_label,
        "--max-samples", "9999",
    ]
    try:
        run(cmd, env=env, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error("[%s × %s] eval failed: %s", backend.asr_name, manifest_label, e)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--backends", nargs="+", default=None,
        help="Профили из BACKENDS (по умолчанию все 5)",
    )
    parser.add_argument(
        "--manifests", nargs="+", type=Path, default=None,
        help="Пути к manifest.json (по умолчанию combined manifest)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Только показать план",
    )
    args = parser.parse_args()

    backends = BACKENDS
    if args.backends:
        wanted = set(args.backends)
        backends = [b for b in BACKENDS if b.profile in wanted]
        missing = wanted - {b.profile for b in BACKENDS}
        if missing:
            logger.error("Unknown profiles: %s", ", ".join(missing))
            return 2

    manifests = args.manifests or DEFAULT_MANIFESTS
    manifests = [m if m.is_absolute() else (REPO_ROOT / m).resolve() for m in manifests]

    plan = [(b, m) for b in backends for m in manifests]
    logger.info("Plan: %d eval runs (%d backends × %d manifests)",
                len(plan), len(backends), len(manifests))
    for b, m in plan:
        logger.info("  %s × %s", b.asr_name, m.name)

    if args.dry_run:
        return 0

    # Pre-flight: погасить любые asr-* контейнеры, оставшиеся от прошлых
    # ручных запусков. Иначе наш первый `up -d` упадёт на конфликте :8000.
    stop_running_asr_containers()

    n_ok = n_fail = 0
    current_profile: str | None = None
    for backend, manifest in plan:
        # Если backend сменился — погасить старый, поднять новый
        if current_profile != backend.profile:
            if current_profile:
                logger.info("=" * 60)
                logger.info("Stopping %s", current_profile)
                logger.info("=" * 60)
                run(["docker", "compose", "--profile", current_profile, "down"],
                    check=False)
            logger.info("=" * 60)
            logger.info("Starting %s", backend.profile)
            logger.info("=" * 60)
            run(["docker", "compose", "--profile", backend.profile, "up", "-d"])
            if not wait_healthy(backend):
                logger.warning("Skipping all manifests for %s", backend.profile)
                run(["docker", "compose", "--profile", backend.profile, "down"],
                    check=False)
                current_profile = None
                n_fail += 1
                continue
            current_profile = backend.profile

        logger.info("=" * 60)
        logger.info("Eval: %s × %s", backend.asr_name, manifest.name)
        logger.info("=" * 60)
        if run_eval(backend, manifest):
            n_ok += 1
        else:
            n_fail += 1

    # Погасить последний profile
    if current_profile:
        run(["docker", "compose", "--profile", current_profile, "down"], check=False)

    logger.info("=" * 60)
    logger.info("DONE — %d ok, %d failed (out of %d)", n_ok, n_fail, len(plan))
    logger.info("CSV: data/eval/results/")
    logger.info("=" * 60)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
