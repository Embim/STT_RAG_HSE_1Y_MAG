"""OpenRouter API client для генерации резюме встреч."""
import os
from typing import Dict, List, Optional
import requests
import json
from dotenv import load_dotenv

# Загружаем переменные из .env файла
load_dotenv()


class OpenRouterClient:
    """Клиент для работы с OpenRouter API (Nvidia Nemotron)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "openai/gpt-oss-120b:free",
        base_url: str = "https://openrouter.ai/api/v1"
    ):
        """
        Инициализация клиента OpenRouter.

        Args:
            api_key: API ключ OpenRouter (если None, берется из env OPENROUTER_API_KEY)
            model: Модель для использования
            base_url: Base URL для OpenRouter API
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key not provided. Set OPENROUTER_API_KEY env variable or pass api_key parameter.")

        self.model = model
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def generate_meeting_summary(
        self,
        transcript: str,
        temperature: float = 0.3,
        max_tokens: int = 4000
    ) -> Dict:
        """
        Генерирует резюме встречи из транскрипции в формате JSON с bullet points.

        Args:
            transcript: Транскрипция встречи
            temperature: Температура генерации (0-1)
            max_tokens: Максимальное количество токенов

        Returns:
            Dict с резюме встречи в формате JSON
        """
        system_prompt = """Ты эксперт по составлению протоколов деловых встреч. Твоя задача - создать развернутый протокол встречи в формате связного текста.

Создай протокол в следующем формате JSON:
{
    "title": "Протокол встречи по [основная тема]",
    "protocol": "Полный текст протокола в виде связных абзацев"
}

Структура текста протокола:
1. ПЕРВЫЙ АБЗАЦ - Основная задача и цель встречи. Опиши главную задачу, которая обсуждалась, её важность и ожидаемый результат. Это должен быть развернутый абзац на 3-5 предложений.

2. СЛЕДУЮЩИЕ АБЗАЦЫ - Детальное описание обсуждаемых вопросов:
   - Каждый крупный вопрос или тема обсуждения - отдельный абзац
   - Опиши контекст, детали, технические подробности
   - Укажи мнения участников, если они упоминались
   - Абзацы должны быть связными и плавно переходить друг в друга

3. АБЗАЦ О РАСПРЕДЕЛЕНИИ ЗАДАЧ - Кто за что отвечает:
   - "[Имя] отвечает за [задача]. Её/его задачи включают [детальное описание]..."
   - Для каждого участника с задачами - отдельное предложение или группа предложений
   - Укажи сроки если они были названы

4. ТЕХНИЧЕСКИЕ ДЕТАЛИ - Если обсуждались технические аспекты:
   - Структура данных, архитектура, форматы
   - Конкретные примеры и требования
   - Все в виде связного текста, не списками

5. БИЗНЕС-ЦЕЛИ И РЕЗУЛЬТАТЫ - Зачем всё это нужно:
   - Какую бизнес-задачу решает
   - Какие выгоды получит пользователь
   - Как это улучшит текущую ситуацию

ВАЖНО:
- Весь текст должен быть связным, как настоящий протокол встречи
- Используй полные развернутые предложения
- Никаких маркированных списков или нумерации внутри текста протокола
- Абзацы должны плавно переходить друг в друга
- Пиши профессиональным деловым языком
- Сохраняй все упомянутые имена, цифры, даты, технические термины"""

        user_prompt = f"""Проанализируй следующую транскрипцию встречи и создай структурированное резюме в формате JSON:

Транскрипция:
{transcript}

ВАЖНО: Верни ТОЛЬКО валидный JSON объект. Текст в поле "protocol" должен содержать символы переноса строки как \\n (экранированные), а НЕ реальные переносы строк. Никаких дополнительных комментариев или markdown."""

        try:
            # Формируем запрос к API
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"}
            }

            # Отправляем запрос
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()

            # Парсим ответ
            response_data = response.json()
            content = response_data["choices"][0]["message"]["content"]

            # Очищаем от возможных markdown блоков и лишних пробелов
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]  # Убираем ```json
            if content.startswith("```"):
                content = content[3:]   # Убираем ```
            if content.endswith("```"):
                content = content[:-3]  # Убираем ```
            content = content.strip()

            # Пытаемся распарсить напрямую
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # Если не получилось, пробуем более мягкий парсинг
                # Ищем JSON объект в тексте
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    # Пытаемся распарсить найденный JSON
                    return json.loads(json_str)
                else:
                    raise  # Пробрасываем исключение дальше

        except json.JSONDecodeError as e:
            # Если не удалось распарсить JSON, пытаемся вернуть хотя бы сырой ответ
            raw_content = content if 'content' in locals() else "No content"

            # Попытка найти JSON в ответе (если он обернут в текст)
            try:
                # Ищем JSON объект в тексте
                import re
                json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            except:
                pass

            return {
                "error": "Failed to parse JSON",
                "raw_response": raw_content,
                "parse_error": str(e),
                "title": "Ошибка парсинга",
                "summary": f"Не удалось распарсить ответ модели. Сырой ответ: {raw_content[:500]}"
            }
        except requests.exceptions.RequestException as e:
            return {
                "error": "API request failed",
                "details": str(e)
            }
        except Exception as e:
            return {
                "error": "Unexpected error",
                "details": str(e)
            }

    def summarize_with_custom_prompt(
        self,
        transcript: str,
        custom_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 4000,
        json_mode: bool = True
    ) -> str:
        """
        Генерирует резюме с кастомным промптом.

        Args:
            transcript: Транскрипция
            custom_prompt: Кастомный промпт
            temperature: Температура генерации
            max_tokens: Максимальное количество токенов
            json_mode: Использовать ли JSON режим

        Returns:
            Сгенерированное резюме
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": custom_prompt},
                {"role": "user", "content": transcript}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()

            response_data = response.json()
            return response_data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error: {str(e)}"
