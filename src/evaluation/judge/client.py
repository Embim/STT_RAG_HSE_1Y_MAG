"""HTTP-клиент к Qwen3-8B судье (vLLM /v1/chat/completions).

API: OpenAI-compatible chat completions, но с расширением vLLM —
`response.choices[0].message.reasoning_content` (заполняется когда сервер
запущен с `--reasoning-parser deepseek_r1` или `qwen3`). Туда попадает
содержимое `<think>...</think>` блока Qwen3 ОТДЕЛЬНО от финального content.

Почему это важно для нас:
  - в `content` мы ждём строгий JSON с findings;
  - в `reasoning_content` лежит как модель пришла к этому JSON;
  - тюнинг промпта = смотрим где reasoning расходится с тем что мы хотим,
    правим system_prompt чтобы подтолкнуть модель в нужном направлении.

Если сервер запущен БЕЗ reasoning-parser, `reasoning_content` будет None,
а thinking-блок останется в `content` как `<think>...</think>...JSON`. На этот
случай в `parse_response()` есть fallback-парсер.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from settings import settings

logger = logging.getLogger(__name__)

# Fallback: вытащить <think>...</think> из content, если reasoning parser
# на сервере не настроен. Жадный match чтобы захватить весь блок.
_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL)


@dataclass
class JudgeResponse:
    """Ответ от судьи после парсинга.

    Поля по стадиям генерации (v11+):
      - prompt_tokens   : размер input (prefill).
      - reasoning_tokens: размер <think>-блока (часть decode).
      - output_tokens   : размер финального content/JSON (вторая часть decode).
        Связь: completion_tokens == reasoning_tokens + output_tokens.
      - prefill_ms / prefill_tps: время и скорость prefill-стадии.
      - decode_ms  / decode_tps : время и скорость decode-стадии (reasoning+content
        идут одним потоком с одинаковой скоростью, разделить честно без streaming
        нельзя — поэтому tps один на оба decode-куска).

    Источники полей в llama.cpp /v1/chat/completions:
      - usage.prompt_tokens
      - usage.completion_tokens
      - usage.completion_tokens_details.reasoning_tokens (новый OpenAI стиль,
        в свежем llama.cpp есть; если нет — пытаемся оценить по символам).
      - timings.prompt_ms, timings.prompt_per_second  (llama.cpp native)
      - timings.predicted_ms, timings.predicted_per_second
    """
    content: str               # финальный текст (по идее — JSON с findings)
    reasoning: str             # содержимое <think>...</think>, может быть ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    output_tokens: int = 0
    prefill_ms: float = 0.0
    prefill_tps: float = 0.0
    decode_ms: float = 0.0
    decode_tps: float = 0.0
    raw_response: Optional[Dict[str, Any]] = None


class JudgeClient:
    """Async-клиент к vLLM-судье.

    Args:
        base_url: URL контейнера (по умолчанию settings.JUDGE_URL = http://localhost:8002)
        model_id: имя модели (для поля `model` в request body)
        api_key: опциональный Bearer token. У локального vLLM обычно не нужен.
        reasoning: ждать ли отдельное поле reasoning_content в ответе.
            False = не делаем warning если поля нет (полезно для не-thinking моделей).
        timeout_sec: read timeout. На 16k input + 4k output Qwen3-8B FP8 на 5080
            генерирует ~30-90 сек. Ставим 600 с запасом, sidecar-killer не нужен.
    """

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
        reasoning: Optional[bool] = None,
        timeout_sec: float = 600.0,
    ) -> None:
        self.base_url = (base_url or settings.JUDGE_URL).rstrip("/")
        self.model_id = model_id or settings.JUDGE_MODEL_ID
        self.api_key = api_key
        self.reasoning = settings.JUDGE_REASONING if reasoning is None else reasoning
        self._timeout = httpx.Timeout(connect=10.0, read=timeout_sec, write=60.0, pool=10.0)

    async def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        enable_thinking: bool = True,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        min_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> JudgeResponse:
        """Один запрос к /v1/chat/completions.

        Args:
            messages: OpenAI chat messages.
            temperature: 0.2 для воспроизводимости findings.
            max_tokens: верхняя граница генерации (включает reasoning + content).
            enable_thinking: Qwen3.5 chat_template_kwargs.enable_thinking. False
                выключает `<think>` блок целиком — модель сразу выдаёт content.
                Полезно если модель уходит в reasoning-loop на сложных задачах.
            top_p, top_k, min_p: nucleus / top-k / min-p sampling. None = не передавать.
            frequency_penalty: штраф за повтор уже сгенерированных токенов
                пропорционально их частоте. 0.3-0.7 борется с loop'ами.
            presence_penalty: штраф за повтор токенов, появлявшихся хотя бы раз.
            repetition_penalty: множитель к логиту повторяющихся токенов (llama.cpp
                native). 1.1-1.3 типично; > 1 = меньше повторов.
            response_format: OpenAI-style формат вывода. Для llama.cpp поддерживается
                `{"type": "json_object"}` (грубый mode) и
                `{"type": "json_schema", "json_schema": {...}}` (точная схема через
                GBNF под капотом).
            extra_body: любые дополнительные поля.
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        body: Dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            # vLLM/llama.cpp расширение: прокидываем в Qwen3 chat template.
            # На моделях без template-arg это поле просто игнорируется.
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        # Опциональные sampling-параметры — добавляем только если переданы.
        # Это нужно потому что у разных бэкендов разные дефолты, и
        # передача 0/None по-разному интерпретируется.
        if top_p is not None:
            body["top_p"] = top_p
        if top_k is not None:
            body["top_k"] = top_k
        if min_p is not None:
            body["min_p"] = min_p
        if frequency_penalty is not None:
            body["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            body["presence_penalty"] = presence_penalty
        if repetition_penalty is not None:
            body["repetition_penalty"] = repetition_penalty
        if seed is not None:
            # llama.cpp принимает seed в JSON. Фиксирует RNG для всех
            # стохастических операций (sampling, mirostat, и т.п.).
            body["seed"] = seed
        if response_format is not None:
            body["response_format"] = response_format
        if extra_body:
            body.update(extra_body)

        # trust_env=False — критично! Если у пользователя стоит HTTP_PROXY
        # / HTTPS_PROXY (корпоративный прокси, VPN-клиент, Clash, и т.п.),
        # httpx по умолчанию пропустит туда запрос, и localhost:8002
        # вернёт 502 Bad Gateway (прокси не знает про локальный llama-server).
        # Браузер обычно настроен на bypass для localhost, потому UI работает
        # а Python-клиент — нет.
        async with httpx.AsyncClient(timeout=self._timeout, trust_env=False) as client:
            response = await client.post(
                f"{self.base_url}/v1/chat/completions",
                headers=headers,
                json=body,
            )
            response.raise_for_status()
            payload = response.json()

        return self._parse(payload)

    def _parse(self, payload: Dict[str, Any]) -> JudgeResponse:
        try:
            choice = payload["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as e:
            logger.error("Malformed chat response from %s: %s", self.base_url, e)
            return JudgeResponse(content="", reasoning="", raw_response=payload)

        content = str(message.get("content") or "")
        # vLLM/llama.cpp с --reasoning-parser выкладывает в отдельное поле.
        reasoning_field = message.get("reasoning_content")
        reasoning = str(reasoning_field) if reasoning_field else ""

        # ВАЖНО: llama.cpp `--reasoning-format deepseek` имеет баг — обрезает
        # reasoning_content на первом backtick (`) в потоке. Если модель в
        # рассуждении использует markdown-бэктики (а Qwen3.5 любит, особенно
        # когда в промпте упоминаются `<think>` теги), reasoning_content
        # приходит обрезанным, а ОСТАЛЬНОЙ reasoning утекает в content вместе
        # с финальным JSON.
        #
        # Признак leak'а: reasoning_content < 500 chars, в content есть
        # большой текст ДО начала настоящего JSON `{"findings"`.
        #
        # Стратегия восстановления: ищем якорь `{"findings"` (с возможными
        # пробелами после `{`). Всё до этой позиции = leak reasoning;
        # всё начиная с неё = настоящий JSON. Это надёжнее чем regex
        # `<think>...</think>` (модель в leak пишет примеры этих тегов
        # из system_prompt'а и путает greedy/non-greedy матчи).
        json_anchor = re.search(r'\{\s*"findings"', content)
        if json_anchor and json_anchor.start() > 200 and len(reasoning) < 500:
            leak = content[:json_anchor.start()].rstrip(" \t\n`,")
            if leak.strip():
                prefix = (
                    "[Reasoning leaked into content "
                    "(llama.cpp backtick-truncation bug):]"
                )
                if reasoning:
                    reasoning = f"{reasoning}\n\n{prefix}\n{leak}"
                else:
                    reasoning = f"{prefix}\n{leak}"
                content = content[json_anchor.start():].strip()
        elif "<think>" in content:
            # Fallback на старую логику только если якорь не нашёлся.
            m = _THINK_BLOCK.search(content)
            if m:
                extracted = m.group(1).strip()
                if len(extracted) > len(reasoning):
                    reasoning = extracted
                content = _THINK_BLOCK.sub("", content, count=1).strip()
        if self.reasoning and not reasoning:
            logger.debug(
                "Judge reasoning expected but absent. Server без "
                "--reasoning-parser? Или модель не сгенерировала reasoning."
            )

        usage = payload.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)

        # Reasoning токены — отдельным полем (свежий llama.cpp / OpenAI o1-style).
        # Если поля нет — оценим по длине reasoning-строки (русский ≈ 3.5 char/tok
        # для Qwen3.5; для нашей задачи tps-метрика и так грубая оценка).
        details = usage.get("completion_tokens_details") or {}
        reasoning_tokens = int(details.get("reasoning_tokens", 0) or 0)
        if reasoning_tokens == 0 and reasoning:
            reasoning_tokens = max(1, len(reasoning) // 3)  # rough fallback
            reasoning_tokens = min(reasoning_tokens, completion_tokens)
        output_tokens = max(0, completion_tokens - reasoning_tokens)

        # Timings — llama.cpp native блок. У vLLM/OpenAI его нет, тогда оставляем 0
        # и потребитель сам посчитает tps через elapsed_sec на стороне reviewer.
        timings = payload.get("timings") or {}
        prefill_ms = float(timings.get("prompt_ms", 0.0) or 0.0)
        prefill_tps = float(timings.get("prompt_per_second", 0.0) or 0.0)
        decode_ms = float(timings.get("predicted_ms", 0.0) or 0.0)
        decode_tps = float(timings.get("predicted_per_second", 0.0) or 0.0)

        return JudgeResponse(
            content=content.strip(),
            reasoning=reasoning.strip(),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            output_tokens=output_tokens,
            prefill_ms=prefill_ms,
            prefill_tps=prefill_tps,
            decode_ms=decode_ms,
            decode_tps=decode_tps,
            raw_response=payload,
        )

    async def health(self) -> bool:
        """Проверка что vLLM поднят и отвечает на /v1/models.

        trust_env=False — см. комментарий в chat(): без него корпоративный
        прокси перехватит запрос к localhost и вернёт 502.
        """
        try:
            async with httpx.AsyncClient(timeout=10.0, trust_env=False) as client:
                resp = await client.get(f"{self.base_url}/v1/models")
                return resp.status_code == 200
        except Exception as e:
            logger.debug("Judge health check failed: %s", e)
            return False
