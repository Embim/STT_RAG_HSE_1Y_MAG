import os
os.environ.setdefault("LLM_MODEL", "test/model")
os.environ.setdefault("LLM_API_KEY_1", "x")
os.environ.setdefault("LLM_API_KEY_2", "x")
os.environ.setdefault("LLM_API_KEY_3", "x")

import pytest


def test_registry_has_expected_models():
    from system.asr_models import ASR_MODELS, DEFAULT_ASR_MODEL
    for key in ("whisper", "qwen3", "parakeet", "vibevoice", "phi4"):
        assert key in ASR_MODELS
    assert DEFAULT_ASR_MODEL in ASR_MODELS
    # дефолт обязан быть рабочим
    assert ASR_MODELS[DEFAULT_ASR_MODEL].available


def test_exactly_one_recommended_and_phi4_unavailable():
    from system.asr_models import ASR_MODELS
    recommended = [m for m in ASR_MODELS.values() if m.recommended]
    assert len(recommended) == 1
    assert recommended[0].key == "qwen3"
    assert ASR_MODELS["phi4"].available is False


def test_get_model_unknown_raises():
    from system.asr_models import get_model
    with pytest.raises(ValueError):
        get_model("does-not-exist")


def test_list_models_hides_internal_fields():
    from system.asr_models import list_models
    rows = list_models()
    assert rows, "catalog must not be empty"
    for row in rows:
        # внутренняя docker-кухня наружу не уходит
        assert "profile" not in row
        assert "container" not in row
        assert "healthy_timeout" not in row
        # а пользовательские поля — есть
        for field in ("key", "label", "when", "code_switch", "vram_gb", "available", "recommended"):
            assert field in row


def test_schema_accepts_known_rejects_unknown_and_unavailable():
    from pydantic import ValidationError
    from api.schemas import IngestRequest

    ok = IngestRequest(url="https://youtu.be/abc", asr_model="qwen3")
    assert ok.asr_model == "qwen3"

    # дефолт подставляется
    assert IngestRequest(url="https://youtu.be/abc").asr_model == "whisper"

    with pytest.raises(ValidationError):
        IngestRequest(url="https://youtu.be/abc", asr_model="nope")
    with pytest.raises(ValidationError):
        IngestRequest(url="https://youtu.be/abc", asr_model="phi4")  # unavailable
