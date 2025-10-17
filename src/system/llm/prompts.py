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
        return """You are DS Navigator, an intelligent assistant for navigating educational content knowledge base on Data Science, Machine Learning and AI.

Your tasks:
1. Answer user questions based on provided context from lecture, talk and podcast transcriptions
2. ALWAYS cite sources with specific timestamps
3. Give accurate, informative answers with citations
4. If there's no relevant info in context - honestly say so

Source link format:
- Indicate source name and timestamp in format [Source: name | Time: MM:SS]
- If multiple sources confirm the info - list all

Answer style:
- Structured, split into paragraphs
- Technically accurate but understandable
- With examples from context if available
- In Russian language (content is mostly in Russian)"""

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
            return f"""User question: {query}

Context: (no context found)

Tell the user that no relevant information was found in the knowledge base for this query."""

        # Format context
        formatted_context = []
        for idx, chunk in enumerate(context, 1):
            text = chunk.get("text", "")
            source = chunk.get("source", "Unknown source")
            timestamp = chunk.get("timestamp", 0)
            score = chunk.get("score", 0.0)

            # Convert timestamp to readable format
            time_str = RAGPrompts._format_timestamp(timestamp)

            chunk_text = f"""Chunk {idx} (relevance: {score:.2f}):
Source: {source}
Timestamp: {time_str}
Text: {text}
---"""
            formatted_context.append(chunk_text)

        context_block = "\n\n".join(formatted_context)

        return f"""User question: {query}

Context from knowledge base:

{context_block}

Based on the provided context, give a detailed answer to the user's question.
ALWAYS cite sources and timestamps where the information came from."""

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

    @staticmethod
    def format_standalone_query(query: str, chat_history: List[Dict[str, str]]) -> str:
        """
        Converts user query to standalone query considering chat history.

        Used to improve vector search quality when user asks
        follow-up questions ("what about X?", "tell me more").

        Args:
            query (str): Current user question
            chat_history (List[Dict[str, str]]): Conversation history.
                Format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

        Returns:
            str: Standalone query for vector search

        Example:
            history = [
                {"role": "user", "content": "What is BERT?"},
                {"role": "assistant", "content": "BERT is a transformer model..."}
            ]
            query = "What are its drawbacks?"
            standalone = format_standalone_query(query, history)
            # Result: "What are the drawbacks of BERT model?"
        """
        if not chat_history:
            return query

        # Take last N messages for context
        recent_history = chat_history[-4:]  # Last 2 question-answer pairs

        history_text = "\n".join([
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in recent_history
        ])

        prompt = f"""Dialog history:
{history_text}

New user question: {query}

Convert this question into a complete standalone query that can be used for search without history context.
Return ONLY the reformulated question, without additional explanations."""

        return prompt
