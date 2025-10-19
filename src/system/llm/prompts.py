"""
Prompts module for RAG system.

Contains prompt templates for generating answers based on retrieved chunks
from knowledge base (lecture/talk transcriptions).

Example usage:
    from src.system.llm.prompts import RAGPrompts

    prompts = RAGPrompts()
    system_prompt = prompts.get_system_prompt()
    user_prompt = prompts.format_user_query(
        query="What are transformers?",
        context=retrieved_chunks
    )
"""

from typing import List, Dict, Any


class RAGPrompts:
    """
    Class for working with RAG system prompts.

    Generates system and user prompts for LLM,
    formats context from vector DB with timestamps and links.

    Used in engine.py for generating answers with source citations.
    """

    @staticmethod
    def get_system_prompt() -> str:
        """
        Returns system prompt for LLM.

        Defines assistant role as DS content knowledge base navigator.
        Instructs to always provide source links and timestamps.

        Returns:
            str: System prompt with instructions for LLM

        Example:
            system_msg = RAGPrompts.get_system_prompt()
            messages = [{"role": "system", "content": system_msg}]
        """
        return """Ты DS Navigator - интеллектуальный ассистент для навигации по базе знаний образовательного контента по Data Science, Machine Learning и AI.

Твои задачи:
1. Отвечать на вопросы пользователя на основе предоставленного контекста из транскрипций лекций, докладов и подкастов
2. ВСЕГДА указывать источники с конкретными таймкодами
3. Давать точные, информативные ответы с цитированием источников
4. Если в контексте нет релевантной информации - честно об этом говорить

Формат ссылок на источники:
- Указывай название источника и таймкод в формате [Источник: название | Время: MM:SS]
- Если несколько источников подтверждают информацию - перечисли все

Стиль ответа:
- Структурированный, разбитый на абзацы
- Технически точный, но понятный
- С примерами из контекста, если доступны
- На русском языке"""

    @staticmethod
    def format_user_query(query: str, context: List[Dict[str, Any]]) -> str:
        """
        Formats user query with relevant context added.

        Takes chunks found via vector search and formats them
        with metadata (source, timestamp) for passing to LLM.

        Args:
            query (str): User question
            context (List[Dict[str, Any]]): List of found chunks from DB.
                Each element should contain:
                - text (str): Transcription chunk text
                - source (str): Source name (video/podcast)
                - timestamp (str or int): Timecode in seconds or MM:SS format
                - score (float, optional): Chunk relevance

        Returns:
            str: Formatted prompt with context and question

        Example:
            context = [
                {
                    "text": "Transformers use attention mechanism...",
                    "source": "Stanford CS224N Lecture",
                    "timestamp": 1520,
                    "score": 0.89
                }
            ]
            prompt = RAGPrompts.format_user_query(
                "What are transformers?",
                context
            )
        """
        if not context:
            return f"""Вопрос пользователя: {query}

Контекст: (контекст не найден)

Сообщи пользователю, что по данному запросу не найдена релевантная информация в базе знаний."""

        # Format context
        formatted_context = []
        for idx, chunk in enumerate(context, 1):
            text = chunk.get("text", "")
            source = chunk.get("source", "Unknown source")
            timestamp = chunk.get("timestamp", 0)
            score = chunk.get("score", 0.0)

            # Convert timestamp to readable format
            time_str = RAGPrompts._format_timestamp(timestamp)

            chunk_text = f"""Фрагмент {idx} (релевантность: {score:.2f}):
Источник: {source}
Таймкод: {time_str}
Текст: {text}
---"""
            formatted_context.append(chunk_text)

        context_block = "\n\n".join(formatted_context)

        return f"""Вопрос пользователя: {query}

Контекст из базы знаний:

{context_block}

На основе предоставленного контекста дай развернутый ответ на вопрос пользователя.
ОБЯЗАТЕЛЬНО указывай источники и таймкоды, откуда взята информация."""

    @staticmethod
    def _format_timestamp(timestamp: Any) -> str:
        """
        Converts timestamp to readable format MM:SS or HH:MM:SS.

        Args:
            timestamp: Timestamp in seconds (int/float) or string

        Returns:
            str: Formatted timestamp

        Example:
            _format_timestamp(90) -> "01:30"
            _format_timestamp(3665) -> "01:01:05"
            _format_timestamp("01:30") -> "01:30"
        """
        # If already string - return as is
        if isinstance(timestamp, str):
            return timestamp

        # Convert seconds
        try:
            seconds = int(timestamp)
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            secs = seconds % 60

            if hours > 0:
                return f"{hours:02d}:{minutes:02d}:{secs:02d}"
            else:
                return f"{minutes:02d}:{secs:02d}"
        except (ValueError, TypeError):
            return "00:00"

