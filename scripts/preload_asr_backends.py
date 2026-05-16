"""Скачивает все ASR-образы и веса моделей в ./hf_cache.

Один скрипт на Windows и Linux (вся логика через subprocess + docker compose).

Что делает:
    1. docker compose pull — тянет все 8 ASR-образов параллельно
    2. Для каждого профиля по очереди: up → ждёт healthy (= модель скачана и
       обслуживает /v1/models) → down. На GPU одновременно держится только
       одна модель.

После прогона все веса в ./hf_cache, повторные запуски — это только load из
файла в VRAM (10-30 сек, не 4 минуты).

Запуск из корня репозитория:

    python scripts/preload_asr_backends.py                # все 8 профилей
    python scripts/preload_asr_backends.py --only asr-qwen3 asr-whisper
    python scripts/preload_asr_backends.py --skip-pull    # пропустить pull образов
    python scripts/preload_asr_backends.py --dry-run      # только показать план

Замечания:
    - Voxtral-профили требуют HF_TOKEN в .env и принятых условий на HF.
    - Сетевой сбой посередине безопасен: повторный запуск докачает
      недостающее (HF кэширует по hash).
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent

# (profile, container_name, healthy_timeout_sec)
# Таймаут с запасом на ПЕРВОЕ скачивание модели из HF + cold start vLLM.
# Повторные запуски укладываются в 30 сек, поэтому таймауты выбраны под
# первый раз.
BACKENDS: list[tuple[str, str, int]] = [
    ("asr-whisper",    "asr-whisper",     600),
    ("asr-qwen3",      "asr-qwen3",       600),
    ("asr-parakeet",   "asr-parakeet",    600),
    ("asr-phi4-nvfp4", "asr-phi4-nvfp4", 1200),
    ("asr-vibevoice",  "asr-vibevoice",  1500),
]

# Внешние артефакты, которые нужны конкретным профилям. Если папка отсутствует,
# она клонируется до прогрева соответствующего профиля. Сейчас пусто:
# vibevoice больше не зависит от vendor/VibeVoice (используем transformers).
EXTERNAL_REPOS: dict[str, tuple[str, str]] = {}


def run(cmd: list[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    """Логирующая обёртка над subprocess.run. На Windows и Linux ведёт себя одинаково."""
    logger.info("$ %s", " ".join(cmd))
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=check,
        capture_output=capture,
        text=True,
    )


def pull_images(profiles: list[str]) -> list[str]:
    """Тянет образы для всех профилей. Возвращает список профилей, у которых
    pull провалился (не найден тег, нет доступа и т.д.) — их вызывающий код
    может пропустить на этапе прогрева.

    Стратегия: сначала пытаемся одной командой (Docker реюзит слои между
    общими образами). Если она падает — повторяем по одному профилю, чтобы
    точно понять, у кого именно сломан образ, и не блокировать остальные.
    """
    bulk = ["docker", "compose"]
    for p in profiles:
        bulk += ["--profile", p]
    bulk.append("pull")
    res = subprocess.run(bulk, cwd=REPO_ROOT)
    if res.returncode == 0:
        return []

    logger.warning("Bulk pull failed (exit %d); falling back to per-profile pull "
                   "to identify which image is broken", res.returncode)
    failed: list[str] = []
    for p in profiles:
        cmd = ["docker", "compose", "--profile", p, "pull"]
        logger.info("$ %s", " ".join(cmd))
        r = subprocess.run(cmd, cwd=REPO_ROOT)
        if r.returncode != 0:
            logger.error("Pull failed for profile %s", p)
            failed.append(p)
    return failed


def health_status(container: str) -> str:
    """Возвращает 'healthy' | 'starting' | 'unhealthy' | 'missing' | 'no-healthcheck'."""
    res = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Health.Status}}", container],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        return "missing"
    status = res.stdout.strip()
    return status or "no-healthcheck"


def ensure_external_repo(profile: str) -> bool:
    """Если профилю нужен внешний git-репо (см. EXTERNAL_REPOS), клонировать.
    Возвращает True если репо доступно (либо склонировали, либо уже было).

    Сейчас EXTERNAL_REPOS пустой — оставлено как hook, если в будущем
    добавим backend, требующий внешних артефактов (например, custom plugin).
    """
    spec = EXTERNAL_REPOS.get(profile)
    if not spec:
        return True
    url, rel_path = spec
    target = REPO_ROOT / rel_path
    if target.exists() and (target / ".git").exists():
        return True
    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Cloning %s into %s (one-time setup for %s)", url, target, profile)
    try:
        run(["git", "clone", "--depth", "1", url, str(target)])
        return True
    except subprocess.CalledProcessError as e:
        logger.error("Failed to clone %s: %s", url, e)
        return False


def warm_one(profile: str, container: str, timeout_sec: int) -> bool:
    """Поднимает профиль, ждёт healthy, гасит. True если успел до таймаута."""
    if not ensure_external_repo(profile):
        logger.error("[%s] external repo missing, cannot warm", profile)
        return False
    logger.info("=" * 60)
    logger.info("Warming %s (timeout %ds)", profile, timeout_sec)
    logger.info("=" * 60)

    try:
        run(["docker", "compose", "--profile", profile, "up", "-d"])
    except subprocess.CalledProcessError as e:
        logger.error("Failed to start %s: %s", profile, e)
        return False

    deadline = time.monotonic() + timeout_sec
    last_status = ""
    success = False
    while time.monotonic() < deadline:
        status = health_status(container)
        if status != last_status:
            elapsed = int(timeout_sec - (deadline - time.monotonic()))
            logger.info("[%s] status=%s (after %ds)", profile, status, elapsed)
            last_status = status
        if status == "healthy":
            success = True
            break
        if status == "unhealthy":
            logger.error("[%s] reported unhealthy — see logs", profile)
            break
        time.sleep(10)

    if not success:
        logger.warning("[%s] not healthy within %ds, last status=%s",
                       profile, timeout_sec, last_status)
        # Печатаем хвост логов контейнера, чтобы пользователь видел, что пошло не так.
        try:
            res = subprocess.run(
                ["docker", "logs", "--tail", "30", container],
                capture_output=True, text=True,
            )
            tail = (res.stdout or "") + (res.stderr or "")
            if tail.strip():
                logger.warning("Last 30 log lines from %s:\n%s", container, tail)
        except Exception:
            pass

    # В любом случае гасим контейнер, чтобы освободить GPU для следующего профиля.
    try:
        run(["docker", "compose", "--profile", profile, "down"], check=False)
    except subprocess.CalledProcessError:
        pass

    return success


def cache_size_human() -> str:
    """Прикинуть размер ./hf_cache. Кроссплатформенно: Path.rglob + sum st_size."""
    cache = REPO_ROOT / "hf_cache"
    if not cache.exists():
        return "missing"
    total = 0
    for p in cache.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if total < 1024:
            return f"{total:.1f} {unit}"
        total /= 1024
    return f"{total:.1f} PB"


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="preload_asr_backends.py",
        description="Скачивает образы и веса моделей для всех ASR-профилей.",
    )
    parser.add_argument(
        "--only", nargs="+", metavar="PROFILE",
        help="Прогреть только указанные профили (по умолчанию — все 8).",
    )
    parser.add_argument(
        "--skip-pull", action="store_true",
        help="Пропустить docker compose pull (только прогрев весов).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Только показать план, ничего не запускать.",
    )
    args = parser.parse_args()

    if args.only:
        wanted = set(args.only)
        backends = [b for b in BACKENDS if b[0] in wanted]
        missing = wanted - {b[0] for b in BACKENDS}
        if missing:
            logger.error("Unknown profile(s): %s", ", ".join(missing))
            return 2
    else:
        backends = list(BACKENDS)

    profiles = [b[0] for b in backends]

    logger.info("Will warm %d backends: %s", len(backends), ", ".join(profiles))
    if args.dry_run:
        return 0

    failed_pull: list[str] = []
    if not args.skip_pull:
        logger.info("=" * 60)
        logger.info("Pulling docker images")
        logger.info("=" * 60)
        failed_pull = pull_images(profiles)
        if failed_pull:
            logger.warning("Skipping warm-up for profiles with failed pull: %s",
                           ", ".join(failed_pull))

    n_ok = n_fail = n_skip = 0
    for profile, container, timeout_sec in backends:
        if profile in failed_pull:
            n_skip += 1
            continue
        if warm_one(profile, container, timeout_sec):
            n_ok += 1
        else:
            n_fail += 1

    logger.info("=" * 60)
    logger.info("DONE — %d ok, %d failed, %d skipped (pull error). Cache size: %s",
                n_ok, n_fail, n_skip, cache_size_human())
    if failed_pull:
        logger.info("Skipped profiles (image not pullable): %s", ", ".join(failed_pull))
    logger.info("=" * 60)
    return 0 if (n_fail == 0 and n_skip == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
