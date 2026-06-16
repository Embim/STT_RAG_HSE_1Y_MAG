"""Загрузка YAML-промптов из prompts/judge/.

Минималистичный template engine: только `str.format(**vars)`. Никаких jinja —
yaml-промпты должны быть прозрачные, чтобы их можно было редактировать в
обычном текстовом редакторе и легко делать diff между версиями.

Идентификация версии:
  --prompt-version v1   → prompts/judge/transcription_review_v1.yaml
  --prompt-version v2   → prompts/judge/transcription_review_v2.yaml
И т.д. Имя файла фиксировано: transcription_review_<version>.yaml.

Зачем yaml а не py:
  - можно дать промпт на правку человеку без python-окружения;
  - git diff между версиями читабельный;
  - model_hints и allowed_* живут в одном файле с самим промптом —
    нельзя случайно поменять промпт и забыть обновить enum типов ошибок.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml

from settings import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JudgePrompt:
    """Распарсенный YAML-промпт.

    Используется reviewer'ом так:
      msgs = prompt.render_messages(transcript_text=..., reference="...", context_label=...)
      → list для chat completion
    """
    version: str
    name: str
    system_prompt: str
    user_template: str
    allowed_error_types: List[str]
    allowed_severity: List[str]
    model_hints: Dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None

    def render_messages(
        self,
        *,
        transcript_text: str,
        reference: str | None = None,
        context_label: str = "",
    ) -> List[Dict[str, str]]:
        """Заполнить шаблоны и вернуть messages в формате OpenAI chat API."""
        reference_block = ""
        if reference:
            reference_block = (
                f"Эталонный текст (то, что должно было быть распознано):\n"
                f"---\n{reference}\n---\n\n"
            )
        # str.format с пропуском неизвестных полей мы НЕ хотим — пусть кидает
        # KeyError если кто-то добавит {newvar} в yaml и забудет передать.
        user_msg = self.user_template.format(
            transcript_text=transcript_text,
            reference_block=reference_block,
            context_label=context_label or "(unspecified)",
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_msg},
        ]


def _prompts_root() -> Path:
    """Корень с YAML-файлами (`prompts/judge/` по дефолту, override через env)."""
    p = Path(settings.JUDGE_PROMPT_DIR)
    if not p.is_absolute():
        # repo_root = .../STT_RAG_HSE_1Y_MAG; settings.py живёт в src/
        repo_root = Path(__file__).resolve().parents[3]
        p = repo_root / p
    return p


def load_prompt(version: str, *, name: str = "transcription_review") -> JudgePrompt:
    """Прочитать prompts/judge/<name>_<version>.yaml и вернуть JudgePrompt.

    Raises:
        FileNotFoundError если yaml не найден.
        ValueError если в yaml отсутствуют обязательные ключи.
    """
    root = _prompts_root()
    path = root / f"{name}_{version}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {path}. Available: "
            f"{[p.name for p in root.glob('*.yaml')]}"
        )
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    required = ("version", "name", "system_prompt", "user_template")
    for key in required:
        if key not in raw:
            raise ValueError(f"{path}: missing required key {key!r}")

    return JudgePrompt(
        version=str(raw["version"]),
        name=str(raw["name"]),
        system_prompt=str(raw["system_prompt"]),
        user_template=str(raw["user_template"]),
        allowed_error_types=list(raw.get("allowed_error_types", [])),
        allowed_severity=list(raw.get("allowed_severity", ["low", "medium", "high"])),
        model_hints=dict(raw.get("model_hints", {})),
        source_path=path,
    )


def list_versions(*, name: str = "transcription_review") -> List[str]:
    """Все версии промпта в prompts/judge/ — для CLI hints."""
    root = _prompts_root()
    if not root.exists():
        return []
    prefix = f"{name}_"
    return sorted(
        p.stem[len(prefix):]
        for p in root.glob(f"{prefix}*.yaml")
        if p.stem.startswith(prefix)
    )
