"""Прогон LLM-судьи на одной лекции/папке с оркестрацией контейнеров.

Что делает (в порядке):
    1. PRE-FLIGHT: гасит все запущенные asr-* контейнеры (КРОМЕ asr-judge),
       чтобы освободить GPU 1 — судья и ASR-бэкенды живут на одной 5080
       и конкурируют за VRAM.
    2. UP: запускает профиль asr-judge (docker compose --profile asr-judge up -d).
       Первый старт качает ~5.5 GB GGUF из HF (3-5 мин на нормальном канале).
    3. WAIT: ждёт health=healthy (до 10 минут).
    4. RUN: вызывает `python -m evaluation.judge.runner --source lecture ...`.
    5. POST: оставляет судью UP по умолчанию, чтобы можно было итерировать
       промпт через веб-морду на http://localhost:8002 без рестарта.
       Флаг --down-after гасит контейнер после прогона.

Примеры:

    # Smoke на одной лекции, только первые 3 чанка (быстрая проверка промпта)
    python scripts/run_judge.py \
        --transcript "data/transcripts/Глубинное обучение Свёрточные нейронные сети (CNN) (31.01.26).json" \
        --max-chunks-per-item 3 \
        --prompt-version v1

    # Полный прогон на одной лекции
    python scripts/run_judge.py \
        --transcript "data/transcripts/Глубинное обучение Свёрточные нейронные сети (CNN) (31.01.26).json" \
        --run-name judge_cnn_v1

    # Прогон по всем лекциям в папке, погасить судью после
    python scripts/run_judge.py \
        --lectures-dir data/transcripts \
        --down-after

    # Только показать план, ничего не запускать
    python scripts/run_judge.py --transcript ... --dry-run
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
JUDGE_PROFILE = "asr-judge"
JUDGE_CONTAINER = "asr-judge"
HEALTHY_TIMEOUT_SEC = 600  # 10 мин на cold start (HF download + llama.cpp init)


def run(cmd: list[str], *, check: bool = True, env: dict | None = None) -> int:
    logger.info("$ %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=check)
    return proc.returncode


def stop_competing_asr_containers() -> None:
    """Гасит все asr-* контейнеры КРОМЕ asr-judge.

    Судья на GPU 1, ASR-бэкенды тоже на GPU 1 — конкуренция за VRAM
    приведёт к OOM или OOM-kill одного из них.
    """
    res = subprocess.run(
        ["docker", "ps", "--filter", "name=^asr-", "--format", "{{.Names}}"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    names = [n.strip() for n in res.stdout.splitlines()
             if n.strip() and n.strip() != JUDGE_CONTAINER]
    if not names:
        logger.info("No competing ASR containers running — good")
        return
    logger.info("Stopping competing ASR containers (GPU sharing): %s", ", ".join(names))
    subprocess.run(["docker", "stop", *names], capture_output=True, cwd=REPO_ROOT)
    # rm чтобы при следующем up -d они не подцепились (профили в compose
    # не пересекаются, но руками поднятые контейнеры могут).
    subprocess.run(["docker", "rm", *names], capture_output=True, cwd=REPO_ROOT)


def judge_status() -> str:
    """missing / created / running / healthy / unhealthy / starting."""
    res = subprocess.run(
        ["docker", "inspect", "--format",
         "{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}",
         JUDGE_CONTAINER],
        capture_output=True, text=True,
    )
    return res.stdout.strip() if res.returncode == 0 else "missing"


def wait_judge_healthy(timeout_sec: int = HEALTHY_TIMEOUT_SEC) -> bool:
    """Дождаться готовности судьи в ДВА этапа:

    1. Docker healthcheck → healthy (или хотя бы TCP-порт :8002 слушает).
       Это значит llama-server открыл сокет, но **модель ещё может грузиться
       с диска** (mmap warmup, ~30-60 сек после открытия порта).
    2. HTTP GET /v1/models возвращает 200 — это значит llama-server
       полностью инициализирован и готов принимать /chat/completions.

    Без этапа 2 первый запрос в eval-runner получит ConnectionError/HTTPError
    («Judge unreachable») и упадёт, хотя контейнер «здоров».

    `unhealthy` НЕ заставляет нас сразу бейлиться — это может быть
    транзиентное состояние (особенно если healthcheck-команда отсутствует
    в образе и Docker мгновенно проставляет unhealthy с exit 127). Пробуем
    до самого конца timeout.
    """
    deadline = time.monotonic() + timeout_sec
    last = ""
    started_at = time.monotonic()

    # Этап 1: docker healthy ИЛИ TCP-порт открыт
    tcp_open = False
    while time.monotonic() < deadline:
        status = judge_status()
        if status != last:
            elapsed = int(time.monotonic() - started_at)
            logger.info("[%s] %s (after %ds)", JUDGE_PROFILE, status, elapsed)
            last = status
        if status == "healthy" or _judge_port_open():
            tcp_open = True
            break
        time.sleep(5)
    if not tcp_open:
        logger.warning("[%s] not healthy within %ds", JUDGE_PROFILE, timeout_sec)
        return False

    # Этап 2: HTTP GET /v1/models возвращает 200
    logger.info(
        "[%s] TCP up, waiting for HTTP /v1/models (model loading from disk)...",
        JUDGE_PROFILE,
    )
    http_attempts = 0
    while time.monotonic() < deadline:
        http_attempts += 1
        if _judge_http_ready():
            elapsed = int(time.monotonic() - started_at)
            logger.info(
                "[%s] HTTP /v1/models OK after %ds (%d attempts) — ready",
                JUDGE_PROFILE, elapsed, http_attempts,
            )
            return True
        if http_attempts % 6 == 0:  # каждые ~30 сек
            elapsed = int(time.monotonic() - started_at)
            logger.info(
                "[%s] still loading... (%ds elapsed; last_err=%s)",
                JUDGE_PROFILE, elapsed, _HTTP_LAST_ERR or "n/a",
            )
        time.sleep(5)
    logger.warning("[%s] HTTP not ready within %ds", JUDGE_PROFILE, timeout_sec)
    return False


def _judge_port_open(host: str = "localhost", port: int = 8002) -> bool:
    """Открыт ли TCP-порт судьи на хосте? Запасной парашют к docker health."""
    import socket
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


_HTTP_LAST_ERR = ""


def _judge_http_ready(url: str = "http://127.0.0.1:8002/v1/models") -> bool:
    """GET /v1/models вернёт 200 только когда модель загружена и сервер
    обрабатывает API-запросы (не просто слушает TCP).

    127.0.0.1 (не localhost) — на Windows DNS lookup `localhost` иногда
    сначала пробует IPv6 ::1 и медленно фолбэчится на IPv4; явный IP
    обходит это.

    ProxyHandler({}) — критично! Если у пользователя HTTP_PROXY/HTTPS_PROXY
    в env (корпоративный прокси, VPN-клиент, Clash, и т.п.), urllib
    отправит запрос через прокси, который вернёт 502 для localhost.
    Пустой ProxyHandler полностью отключает прокси для этого запроса.
    """
    global _HTTP_LAST_ERR
    import urllib.request
    try:
        # Билдим opener БЕЗ прокси-хендлера из окружения.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url, timeout=5) as resp:
            ok = resp.status == 200
            if ok:
                _HTTP_LAST_ERR = ""
            return ok
    except Exception as e:  # noqa: BLE001 — широко чтобы залогировать что угодно
        _HTTP_LAST_ERR = f"{type(e).__name__}: {e}"
        return False


def run_judge_eval(args: argparse.Namespace) -> bool:
    """Запустить `python -m evaluation.judge.runner` с нужными флагами.

    Возвращает True если RC=0.
    """
    src_path = str(REPO_ROOT / "src")
    existing_pp = os.environ.get("PYTHONPATH", "")
    new_pp = src_path if not existing_pp else f"{src_path}{os.pathsep}{existing_pp}"
    env = {**os.environ, "PYTHONPATH": new_pp}

    cmd: list[str] = [
        sys.executable, "-m", "evaluation.judge.runner",
        "--source", "lecture",
        "--prompt-version", args.prompt_version,
    ]
    if args.transcript:
        cmd.extend(["--lectures", str(args.transcript)])
    if args.lectures_dir:
        cmd.extend(["--lectures", str(args.lectures_dir)])
    if args.run_name:
        cmd.extend(["--run-name", args.run_name])
    if args.max_items is not None:
        cmd.extend(["--max-items", str(args.max_items)])
    if args.max_chunks_per_item is not None:
        cmd.extend(["--max-chunks-per-item", str(args.max_chunks_per_item)])
    if args.max_chars is not None:
        cmd.extend(["--max-chars", str(args.max_chars)])
    if args.asr_name:
        cmd.extend(["--asr-name", args.asr_name])

    try:
        run(cmd, env=env, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error("Judge eval failed: %s", e)
        return False


def _default_run_name(args: argparse.Namespace) -> str | None:
    """Авто-сгенерировать имя run-а если не указано — из имени файла лекции."""
    if args.run_name:
        return args.run_name
    if args.transcript:
        stem = Path(args.transcript).stem
        # Сократим до ASCII-friendly, чтобы CSV-имя файла было читаемое
        import re
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)[:60]
        return f"judge_{args.prompt_version}_{slug}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--transcript", type=Path, default=None,
                     help="Путь к одному JSON-транскрипту лекции")
    src.add_argument("--lectures-dir", type=Path, default=None,
                     help="Папка со множеством *.json транскрипций")

    parser.add_argument("--prompt-version", default="v1",
                        help="Версия промпта (prompts/judge/transcription_review_<ver>.yaml)")
    parser.add_argument("--run-name", default=None,
                        help="Метка для CSV/JSONL (default: judge_<ver>_<stem>)")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--max-chunks-per-item", type=int, default=None,
                        help="Pilot mode: остановиться после N чанков каждой лекции")
    parser.add_argument("--max-chars", type=int, default=None,
                        help="Override settings.JUDGE_MAX_INPUT_CHARS")
    parser.add_argument("--asr-name", default=None,
                        help="Имя ASR-системы создавшей транскрипт (например "
                             "'whisper-large-v3-turbo' или 'qwen3-asr'). "
                             "Нужно для сравнения ASR через judge: leaderboard "
                             "группирует findings по этому полю. По умолчанию "
                             "пусто и все прогоны лекций неразличимы.")

    parser.add_argument("--healthy-timeout", type=int, default=HEALTHY_TIMEOUT_SEC,
                        help="Сколько секунд ждать healthy после up -d (default 600)")
    parser.add_argument("--skip-asr-stop", action="store_true",
                        help="Не гасить ASR-контейнеры (если уверен что GPU не пересекается)")
    parser.add_argument("--down-after", action="store_true",
                        help="Погасить asr-judge после прогона (default: оставить UP "
                             "для интерактивного debug через http://localhost:8002)")
    parser.add_argument("--no-up", action="store_true",
                        help="Считать что asr-judge уже запущен — не делать up/wait")
    parser.add_argument("--dry-run", action="store_true",
                        help="Показать что будет сделано, ничего не запускать")
    args = parser.parse_args()

    # Валидация transcript path
    if args.transcript and not args.transcript.exists():
        logger.error("Transcript not found: %s", args.transcript)
        return 2
    if args.lectures_dir and not args.lectures_dir.exists():
        logger.error("Lectures dir not found: %s", args.lectures_dir)
        return 2

    args.run_name = _default_run_name(args)

    # План
    logger.info("=" * 60)
    logger.info("PLAN")
    logger.info("=" * 60)
    logger.info("  prompt:      %s", args.prompt_version)
    logger.info("  run_name:    %s", args.run_name)
    if args.transcript:
        logger.info("  transcript:  %s", args.transcript)
    else:
        logger.info("  lectures:    %s", args.lectures_dir)
    logger.info("  max-items:   %s", args.max_items)
    logger.info("  max-chunks:  %s", args.max_chunks_per_item)
    logger.info("  stop ASR:    %s", not args.skip_asr_stop)
    logger.info("  up judge:    %s", not args.no_up)
    logger.info("  down after:  %s", args.down_after)
    if args.dry_run:
        logger.info("DRY-RUN: exiting without doing anything")
        return 0

    # 1. Гасим конкурентов на GPU 1
    if not args.skip_asr_stop:
        stop_competing_asr_containers()

    # 2. Поднимаем судью (если ещё не запущен)
    if not args.no_up:
        logger.info("=" * 60)
        logger.info("Starting asr-judge profile")
        logger.info("=" * 60)
        run(["docker", "compose", "--profile", JUDGE_PROFILE, "up", "-d"])
        if not wait_judge_healthy(args.healthy_timeout):
            logger.error("Judge не поднялся за %ds — abort. Логи: docker logs %s",
                         args.healthy_timeout, JUDGE_CONTAINER)
            return 3
    else:
        status = judge_status()
        if status != "healthy":
            logger.warning("--no-up задан, но judge сейчас %s (а не healthy). "
                           "Прогон может упасть на первом запросе.", status)

    # 3. Прогон
    logger.info("=" * 60)
    logger.info("Eval: %s", args.run_name)
    logger.info("=" * 60)
    ok = run_judge_eval(args)

    # 4. Down-after если попросили
    if args.down_after:
        logger.info("=" * 60)
        logger.info("Stopping asr-judge")
        logger.info("=" * 60)
        run(["docker", "compose", "--profile", JUDGE_PROFILE, "down"], check=False)
    else:
        logger.info("=" * 60)
        logger.info("asr-judge остался UP — http://localhost:8002 для интерактива.")
        logger.info("Чтобы погасить: docker compose --profile %s down", JUDGE_PROFILE)
        logger.info("=" * 60)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
