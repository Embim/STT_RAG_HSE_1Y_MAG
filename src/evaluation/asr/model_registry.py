"""Minimal MLflow Model Registry integration for ASR HTTP backends."""
from __future__ import annotations

import base64
import logging
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable

import requests

logger = logging.getLogger(__name__)

try:
    from mlflow.pyfunc import PythonModel
except ImportError:
    PythonModel = object

_MIME_BY_EXT = {
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".webm": "audio/webm",
}

_QWEN_ASR_PATTERN = re.compile(
    r"<asr_text>(?P<text>.*?)(?:</asr_text>|$)",
    re.DOTALL,
)


def _parse_chat_asr_output(raw: str) -> str:
    match = _QWEN_ASR_PATTERN.search(raw)
    return match.group("text").strip() if match else raw.strip()


def _iter_audio_paths(model_input: Any) -> list[str]:
    if hasattr(model_input, "columns"):
        if "audio_path" in model_input.columns:
            return [str(v) for v in model_input["audio_path"].tolist()]
        if len(model_input.columns) == 1:
            return [str(v) for v in model_input.iloc[:, 0].tolist()]
    if isinstance(model_input, dict):
        value = model_input.get("audio_path") or model_input.get("audio_paths")
        if isinstance(value, (list, tuple)):
            return [str(v) for v in value]
        if value is not None:
            return [str(value)]
    if isinstance(model_input, (list, tuple)):
        return [str(v) for v in model_input]
    return [str(model_input)]


class HttpASRPyfuncModel(PythonModel):
    """Tiny pyfunc wrapper around the same OpenAI-compatible ASR endpoint."""

    def __init__(self, config: dict[str, str]) -> None:
        self.config = config

    def predict(self, context, model_input, params=None):  # noqa: ANN001
        rows = []
        for audio_path in _iter_audio_paths(model_input):
            text = self._transcribe(audio_path)
            rows.append({
                "audio_path": audio_path,
                "text": text,
                "model": self.config.get("name", ""),
                "model_id": self.config.get("model_id", ""),
            })
        return rows

    def _transcribe(self, audio_path: str) -> str:
        endpoint = (self.config.get("endpoint") or "transcription").lower()
        if endpoint == "chat":
            return self._transcribe_chat(audio_path)
        return self._transcribe_transcription(audio_path)

    def _transcribe_transcription(self, audio_path: str) -> str:
        path = Path(audio_path)
        mime = _MIME_BY_EXT.get(path.suffix.lower(), "application/octet-stream")
        data: dict[str, Any] = {
            "model": self.config.get("model_id") or "whisper-1",
            "response_format": "json",
        }
        language = self.config.get("language")
        if language:
            data["language"] = language
        headers = self._auth_headers()
        with path.open("rb") as audio_file:
            response = requests.post(
                f"{self.config['base_url']}/v1/audio/transcriptions",
                headers=headers or None,
                files={"file": (path.name, audio_file, mime)},
                data=data,
                timeout=(10, 3600),
            )
        response.raise_for_status()
        return str(response.json().get("text", ""))

    def _transcribe_chat(self, audio_path: str) -> str:
        path = Path(audio_path)
        mime = _MIME_BY_EXT.get(path.suffix.lower(), "application/octet-stream")
        with path.open("rb") as audio_file:
            b64 = base64.b64encode(audio_file.read()).decode("ascii")
        headers = {"Content-Type": "application/json", **self._auth_headers()}
        body = {
            "model": self.config.get("model_id") or "whisper-1",
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "audio_url",
                    "audio_url": {"url": f"data:{mime};base64,{b64}"},
                }],
            }],
            "max_tokens": 2048,
            "temperature": 0,
        }
        response = requests.post(
            f"{self.config['base_url']}/v1/chat/completions",
            headers=headers,
            json=body,
            timeout=(10, 3600),
        )
        response.raise_for_status()
        payload = response.json()
        raw = payload.get("choices", [{}])[0].get("message", {}).get("content", "")
        return _parse_chat_asr_output(str(raw or ""))

    def _auth_headers(self) -> dict[str, str]:
        api_key = self.config.get("api_key")
        return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def log_asr_model_to_registry(
    backend: Any,
    *,
    registered_model_name: str,
    alias: str = "",
    run_id: str | None = None,
    tags: dict[str, str] | None = None,
) -> str | None:
    """Log the current ASR endpoint as an MLflow pyfunc model and register it."""
    if not registered_model_name:
        return None
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError:
        logger.info("MLflow is not installed; ASR model registration skipped")
        return None

    config = {
        "name": getattr(backend, "name", ""),
        "base_url": getattr(backend, "base_url", ""),
        "model_id": getattr(backend, "model_id", ""),
        "language": getattr(backend, "language", ""),
        "endpoint": getattr(backend, "endpoint", "transcription"),
        "api_key": getattr(backend, "api_key", "") or "",
    }

    artifact_path = "asr_model"
    try:
        with tempfile.TemporaryDirectory(prefix="asr-mlflow-model-") as tmp_dir:
            model_dir = Path(tmp_dir) / artifact_path
            mlflow.pyfunc.save_model(
                path=str(model_dir),
                python_model=HttpASRPyfuncModel(config),
                pip_requirements=["requests>=2.32.0"],
            )
            mlflow.log_artifacts(str(model_dir), artifact_path=artifact_path)
            source = mlflow.get_artifact_uri(artifact_path)
    except Exception as e:
        logger.warning("ASR model artifact logging failed: %s", e)
        return None

    client = MlflowClient()
    try:
        try:
            client.create_registered_model(registered_model_name)
        except Exception:
            pass
        model_version = client.create_model_version(
            name=registered_model_name,
            source=source,
            run_id=run_id,
            tags=tags or {},
        )
        version = str(model_version.version)
    except Exception as e:
        logger.warning("ASR model version creation failed: %s", e)
        return None

    try:
        for key, value in (tags or {}).items():
            client.set_model_version_tag(registered_model_name, version, key, str(value))
        if alias:
            client.set_registered_model_alias(registered_model_name, alias, version)
        logger.info(
            "ASR model registered as %s v%s%s",
            registered_model_name,
            version,
            f" (@{alias})" if alias else "",
        )
    except Exception as e:
        logger.warning("ASR model registry metadata update failed: %s", e)
    return version


def _find_logged_model_version(
    client: Any,
    *,
    registered_model_name: str,
    run_id: str | None,
    source: str,
) -> str | None:
    try:
        versions: Iterable[Any] = client.search_model_versions(
            f"name = '{registered_model_name}'"
        )
    except Exception as e:
        logger.debug("Cannot search registered model versions: %s", e)
        return None

    candidates = []
    for version in versions:
        if run_id and getattr(version, "run_id", None) == run_id:
            candidates.append(version)
        elif source and str(getattr(version, "source", "")).endswith(source):
            candidates.append(version)
    if not candidates:
        return None
    latest = max(candidates, key=lambda item: int(getattr(item, "version", 0)))
    return str(latest.version)
