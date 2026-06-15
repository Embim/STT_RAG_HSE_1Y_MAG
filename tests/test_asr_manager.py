import os
os.environ.setdefault("LLM_MODEL", "test/model")
os.environ.setdefault("LLM_API_KEY_1", "x")
os.environ.setdefault("LLM_API_KEY_2", "x")
os.environ.setdefault("LLM_API_KEY_3", "x")

import pytest


class FakeDocker:
    """Записывает вызовы `docker` и отдаёт заранее заданные ответы."""

    def __init__(self, ps_output="", health_seq=None):
        self.calls = []
        self.ps_output = ps_output
        self.health_seq = list(health_seq or [])

    async def __call__(self, *args, timeout=60.0):
        self.calls.append(args)
        cmd = args[0]
        if cmd == "ps":
            return (0, self.ps_output, "")
        if cmd == "inspect":
            status = self.health_seq.pop(0) if self.health_seq else "healthy"
            return (0, status, "")
        if cmd in ("stop", "start"):
            return (0, "", "")
        return (0, "", "")

    def cmds(self):
        return [c[0] for c in self.calls]


def _mgr():
    from system.asr_manager import AsrManager
    return AsrManager()


@pytest.mark.asyncio
async def test_session_autoswap_off_builds_backend_no_docker(monkeypatch):
    from settings import settings
    from system.asr_models import ASR_MODELS
    monkeypatch.setattr(settings, "ASR_AUTOSWAP_ENABLED", False)
    mgr = _mgr()
    fake = FakeDocker()
    mgr._docker = fake  # type: ignore[method-assign]

    async with mgr.session("qwen3") as backend:
        assert backend.model_id == ASR_MODELS["qwen3"].model_id
        assert backend.endpoint == "chat"
        assert backend.name == ASR_MODELS["qwen3"].name
    assert fake.calls == []  # никаких docker-команд при выключенном свапе


@pytest.mark.asyncio
async def test_session_unavailable_model_raises(monkeypatch):
    from settings import settings
    from system.asr_manager import AsrSwapError
    monkeypatch.setattr(settings, "ASR_AUTOSWAP_ENABLED", False)
    mgr = _mgr()
    with pytest.raises(AsrSwapError):
        async with mgr.session("phi4"):
            pass


@pytest.mark.asyncio
async def test_session_unknown_model_raises(monkeypatch):
    mgr = _mgr()
    with pytest.raises(ValueError):
        async with mgr.session("nope"):
            pass


@pytest.mark.asyncio
async def test_ensure_active_noop_when_already_running(monkeypatch):
    from settings import settings
    from system.asr_models import get_model
    monkeypatch.setattr(settings, "ASR_AUTOSWAP_ENABLED", True)
    mgr = _mgr()
    fake = FakeDocker(ps_output="asr-qwen3\n")
    mgr._docker = fake  # type: ignore[method-assign]

    async with mgr.session("qwen3"):
        pass
    # только `ps`, без stop/start
    assert "stop" not in fake.cmds()
    assert "start" not in fake.cmds()


@pytest.mark.asyncio
async def test_ensure_active_swaps_and_waits_healthy(monkeypatch):
    from settings import settings
    monkeypatch.setattr(settings, "ASR_AUTOSWAP_ENABLED", True)
    mgr = _mgr()
    # сейчас поднят whisper, просим qwen3 → stop whisper, start qwen3, healthy
    fake = FakeDocker(ps_output="asr-whisper\n", health_seq=["starting", "healthy"])
    mgr._docker = fake  # type: ignore[method-assign]

    statuses = []
    async with mgr.session("qwen3", status_cb=statuses.append):
        pass

    cmds = fake.cmds()
    assert "stop" in cmds and "start" in cmds and "inspect" in cmds
    # остановили именно whisper, запустили qwen3
    assert ("stop", "asr-whisper") in fake.calls
    assert ("start", "asr-qwen3") in fake.calls
    assert any("готова" in s or "qwen3" in s.lower() or "Qwen" in s for s in statuses)
    # успешный свап запоминает активную модель (без гонки с docker ps)
    assert await mgr.active_model_key() == "qwen3"


@pytest.mark.asyncio
async def test_ensure_active_does_not_stop_foreign_asr_container(monkeypatch):
    """asr-judge (LLM-судья) НЕ в реестре → свап не должен его гасить."""
    from settings import settings
    monkeypatch.setattr(settings, "ASR_AUTOSWAP_ENABLED", True)
    mgr = _mgr()
    # рядом крутится чужой asr-judge + наш asr-whisper; просим qwen3
    fake = FakeDocker(ps_output="asr-judge\nasr-whisper\n", health_seq=["healthy"])
    mgr._docker = fake  # type: ignore[method-assign]

    async with mgr.session("qwen3"):
        pass

    assert ("stop", "asr-judge") not in fake.calls   # чужой контейнер не трогаем
    assert ("stop", "asr-whisper") in fake.calls      # наш — гасим
    assert ("start", "asr-qwen3") in fake.calls


@pytest.mark.asyncio
async def test_wait_healthy_timeout_raises(monkeypatch):
    from system.asr_manager import AsrManager, AsrSwapError
    from system.asr_models import AsrModel
    mgr = AsrManager()
    # timeout=0 → дедлайн в прошлом, цикл не крутится, сразу ошибка (без sleep)
    model = AsrModel(
        key="t", profile="asr-t", container="asr-t", model_id="m", name="t",
        endpoint="transcription", language="ru", label="T", vram_gb=1.0,
        code_switch="-", when="-", healthy_timeout=0,
    )

    async def always_starting(container):
        return "starting"
    mgr._health = always_starting  # type: ignore[method-assign]

    with pytest.raises(AsrSwapError):
        await mgr._wait_healthy(model, None)


@pytest.mark.asyncio
async def test_unhealthy_container_raises(monkeypatch):
    from system.asr_manager import AsrManager, AsrSwapError
    from system.asr_models import get_model
    mgr = AsrManager()

    async def unhealthy(container):
        return "unhealthy"
    mgr._health = unhealthy  # type: ignore[method-assign]

    with pytest.raises(AsrSwapError):
        await mgr._wait_healthy(get_model("qwen3"), None)


@pytest.mark.asyncio
async def test_active_model_key_maps_running_container(monkeypatch):
    from settings import settings
    monkeypatch.setattr(settings, "ASR_AUTOSWAP_ENABLED", True)
    mgr = _mgr()
    fake = FakeDocker(ps_output="asr-vibevoice\n")
    mgr._docker = fake  # type: ignore[method-assign]
    assert await mgr.active_model_key() == "vibevoice"


@pytest.mark.asyncio
async def test_active_model_key_reports_configured_when_autoswap_off(monkeypatch):
    from settings import settings
    from system.asr_models import DEFAULT_ASR_MODEL
    monkeypatch.setattr(settings, "ASR_AUTOSWAP_ENABLED", False)
    # пусто → реестровый дефолт
    monkeypatch.setattr(settings, "ASR_ACTIVE_MODEL", "")
    mgr = _mgr()
    assert await mgr.active_model_key() == DEFAULT_ASR_MODEL
    # явный ASR_ACTIVE_MODEL уважается
    monkeypatch.setattr(settings, "ASR_ACTIVE_MODEL", "qwen3")
    assert await mgr.active_model_key() == "qwen3"
    # мусорное значение → фолбэк на дефолт
    monkeypatch.setattr(settings, "ASR_ACTIVE_MODEL", "garbage")
    assert await mgr.active_model_key() == DEFAULT_ASR_MODEL
