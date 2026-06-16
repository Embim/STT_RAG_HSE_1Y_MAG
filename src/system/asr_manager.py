"""ASR GPU manager — переключение ASR-моделей под одну GPU.

Все ASR-бэкенды (`asr-whisper`, `asr-qwen3`, ...) делят один порт 8000 и
сетевой алиас `asr`, поэтому на одной видеокарте может работать только ОДИН
за раз. Когда пользователь выбирает модель при загрузке лекции, менеджер:

  1. Берёт глобальный `asyncio.Lock` — на всё время транскрипции, чтобы
     параллельная загрузка не «выдернула» GPU из-под текущей (job'ы встают
     в очередь, а не дерутся за видеокарту).
  2. Если выбранная модель не та, что сейчас поднята — гасит текущий
     `asr-*` контейнер и стартует нужный (`docker stop` / `docker start`),
     затем ждёт healthy.
  3. Отдаёт настроенный `HttpASRBackend` (правильные model_id/endpoint/name).

Управление docker идёт через смонтированный сокет `/var/run/docker.sock`
и обычный `docker` CLI. Чтобы не превратить публичный API в дыру, менеджер
дёргает ТОЛЬКО контейнеры из белого списка (`asr-*` из реестра) и только
командами start/stop/inspect/ps — никакого произвольного docker exec.

Авто-свап включается флагом `ASR_AUTOSWAP_ENABLED` (по умолчанию ВЫКЛ). Когда
выключен — менеджер просто строит backend под выбранную модель и обращается
к тому, что уже поднято за алиасом `asr` (поведение как раньше, без docker).

Контейнеры должны быть СОЗДАНЫ заранее (один раз при деплое):

    docker compose --profile asr-whisper --profile asr-qwen3 \\
        --profile asr-parakeet --profile asr-vibevoice create

После этого `docker start <container>` поднимает их мгновенно (веса уже
в кеше). Если контейнер не создан — менеджер бросит понятную ошибку.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager
from typing import Callable, List, Optional

from settings import settings
from system.asr_models import ASR_MODELS, DEFAULT_ASR_MODEL, AsrModel, get_model
from evaluation.asr.backends.http_asr import HttpASRBackend

logger = logging.getLogger(__name__)

StatusCb = Callable[[str], None]


class AsrSwapError(RuntimeError):
    """Не удалось переключить/поднять ASR-контейнер."""


def _notify(cb: Optional[StatusCb], msg: str) -> None:
    logger.info("[asr-manager] %s", msg)
    if cb:
        try:
            cb(msg)
        except Exception:  # noqa: BLE001 — статус не должен ронять job
            logger.debug("status_cb raised", exc_info=True)


class AsrManager:
    """Сериализует доступ к GPU и свапит ASR-контейнеры по запросу."""

    def __init__(self) -> None:
        # Лениво создаём Lock при первом использовании, чтобы он привязался
        # к текущему event loop (важно для тестов с отдельными циклами).
        self._lock: Optional[asyncio.Lock] = None
        # Ключ модели, которую мы последний раз успешно подняли (под lock).
        # Источник правды для GET /asr-models — без гонки с docker ps в середине свапа.
        self._active_key: Optional[str] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    # ── docker helpers ────────────────────────────────────────────────
    async def _docker(self, *args: str, timeout: float = 60.0) -> tuple[int, str, str]:
        """Запустить `docker <args>`; вернуть (rc, stdout, stderr)."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as e:
            # `docker` нет в PATH / сокет не смонтирован — авто-свап включили без
            # docker-доступа. Не роняем процесс непонятным OSError.
            raise AsrSwapError(
                f"docker CLI недоступен ({e}); смонтирован ли /var/run/docker.sock "
                "и стоит ли docker в образе? (см. docker-compose.autoswap.yml)"
            ) from e
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            # Дожать kill, чтобы reaped зомби и закрылись pipe'ы (без ResourceWarning).
            with contextlib.suppress(Exception):
                await proc.wait()
            raise AsrSwapError(f"docker {args[0]} timed out after {timeout}s")
        return proc.returncode or 0, out.decode(errors="replace"), err.decode(errors="replace")

    async def _running_asr_containers(self) -> List[str]:
        """Имена запущенных `asr-*` контейнеров (источник правды о том, что на GPU).

        ВНИМАНИЕ: docker-фильтр `name=^asr-` ловит ЛЮБОЙ контейнер с таким
        префиксом, включая чужой `asr-judge` (LLM-судья). Чистку до «наших»
        моделей делает `_ensure_active` через белый список реестра.
        """
        rc, out, err = await self._docker(
            "ps", "--filter", "name=^asr-", "--format", "{{.Names}}"
        )
        if rc != 0:
            logger.error("`docker ps` failed: %s", (err.strip() or out.strip()))
            raise AsrSwapError("не удалось получить список ASR-контейнеров (docker ps)")
        return [n.strip() for n in out.splitlines() if n.strip()]

    async def _health(self, container: str) -> str:
        """Health-статус контейнера: healthy/unhealthy/starting/none/missing."""
        rc, out, _ = await self._docker(
            "inspect", "--format", "{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}",
            container,
        )
        if rc != 0:
            return "missing"
        return out.strip() or "none"

    async def _wait_healthy(self, model: AsrModel, status_cb: Optional[StatusCb]) -> None:
        """Поллить health контейнера до healthy / таймаута."""
        loop = asyncio.get_event_loop()
        deadline = loop.time() + model.healthy_timeout
        last = ""
        while loop.time() < deadline:
            status = await self._health(model.container)
            if status == "healthy":
                _notify(status_cb, f"модель {model.label} готова")
                return
            if status == "none":
                # У контейнера нет healthcheck — считаем готовым после старта.
                _notify(status_cb, f"модель {model.label} запущена (без healthcheck)")
                return
            if status == "unhealthy":
                raise AsrSwapError(f"{model.container} стал unhealthy при старте")
            if status == "missing":
                raise AsrSwapError(f"{model.container} исчез во время прогрева")
            if status != last:
                left = int(deadline - loop.time())
                _notify(status_cb, f"прогрев {model.label}… (осталось до {left}s)")
                last = status
            await asyncio.sleep(5)
        raise AsrSwapError(
            f"{model.container} не стал healthy за {model.healthy_timeout}s"
        )

    async def _ensure_active(self, model: AsrModel, status_cb: Optional[StatusCb]) -> None:
        """Гарантировать, что под алиасом `asr` поднята именно `model`."""
        # Белый список — ТОЛЬКО контейнеры наших ASR-моделей из реестра. Гасим
        # лишь их; чужие `asr-*` (например `asr-judge` — LLM-судья) не трогаем,
        # даже если они попали под docker-фильтр `name=^asr-`.
        known = {m.container for m in _iter_models()}
        running = await self._running_asr_containers()
        # Пока свап не подтверждён — не утверждаем, какая модель активна
        # (если start упадёт после stop, active_model_key честно перейдёт на
        # детект по docker ps вместо устаревшего ключа).
        self._active_key = None
        for name in running:
            if name == model.container or name not in known:
                continue
            _notify(status_cb, f"останавливаю {name}…")
            rc, _, err = await self._docker("stop", name, timeout=120.0)
            if rc != 0:
                logger.error("docker stop %s failed: %s", name, err.strip())
                raise AsrSwapError(f"не удалось остановить {name}")
        if model.container not in running:
            _notify(status_cb, f"переключаюсь на {model.label}…")
            rc, _, err = await self._docker("start", model.container, timeout=120.0)
            if rc != 0:
                logger.error("docker start %s failed: %s", model.container, err.strip())
                raise AsrSwapError(
                    f"не удалось запустить {model.label}. Контейнер создан? "
                    f"(docker compose --profile {model.profile} create)"
                )
            await self._wait_healthy(model, status_cb)
        self._active_key = model.key

    # ── публичный API ─────────────────────────────────────────────────
    @asynccontextmanager
    async def session(self, model_key: str, *, status_cb: Optional[StatusCb] = None):
        """Контекст транскрипции выбранной моделью.

        Держит глобальный lock на всё время блока, при необходимости свапит
        контейнер и отдаёт настроенный `HttpASRBackend`.

        Raises:
            ValueError: неизвестный ключ модели.
            AsrSwapError: модель помечена недоступной, или свап не удался.
        """
        model = get_model(model_key)
        if not model.available:
            raise AsrSwapError(
                f"модель {model.label} сейчас недоступна: {model.code_switch}"
            )
        lock = self._get_lock()
        # На одной GPU транскрипции строго последовательны. Если lock занят —
        # покажем в статусе задания, что мы в очереди, а не «зависли».
        if lock.locked():
            _notify(status_cb, "ожидание очереди: GPU занята другой транскрипцией…")
        async with lock:
            if settings.ASR_AUTOSWAP_ENABLED:
                await self._ensure_active(model, status_cb)
            else:
                _notify(status_cb, f"модель: {model.label} (авто-свап выключен)")
            backend = HttpASRBackend(
                base_url=settings.WHISPER_URL,
                model_id=model.model_id,
                language=model.language,
                name=model.name,
                endpoint=model.endpoint,
            )
            yield backend

    async def active_model_key(self) -> Optional[str]:
        """Какой ключ модели сейчас обслуживается (для GET /asr-models).

        - Авто-свап ВЫКЛ: за алиасом `asr` всегда одна модель — её задаёт
          ASR_ACTIVE_MODEL (по умолчанию реестровый дефолт).
        - Авто-свап ВКЛ: сначала отдаём последнюю успешно поднятую под lock
          (без гонки с docker ps в середине свапа), иначе детектим по docker ps.
        """
        if not settings.ASR_AUTOSWAP_ENABLED:
            key = settings.ASR_ACTIVE_MODEL or DEFAULT_ASR_MODEL
            return key if key in ASR_MODELS else DEFAULT_ASR_MODEL
        if self._active_key:
            return self._active_key
        try:
            running = await self._running_asr_containers()
        except AsrSwapError:
            return None
        by_container = {m.container: m.key for m in _iter_models()}
        for name in running:
            if name in by_container:
                return by_container[name]
        return None


def _iter_models():
    return ASR_MODELS.values()


# Singleton — один менеджер на процесс (один GPU).
asr_manager = AsrManager()
