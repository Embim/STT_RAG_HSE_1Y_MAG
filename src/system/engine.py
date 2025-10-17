"""
RAG engine for processing user queries.

Implements pipeline: user query -> vector search -> answer generation with LLM.
Integrates vector DB (ChromaDB/FAISS) and LLM for generating answers with source citations.

Example usage:
    from src.system.engine import RAGEngine

    engine = RAGEngine()
    response = engine.query("What is attention mechanism?")
    print(response["answer"])  # LLM answer
    print(response["sources"])  # Sources with timestamps
"""

from typing import List, Dict, Any, Optional
from src.system.llm.llm import OpenRouterClient
from src.system.llm.prompts import RAGPrompts


class RAGEngine:
    """
    Main engine for RAG search and answer generation.

    Combines vector search through transcription database and answer generation via LLM.
    Supports dialog history for contextual follow-up questions.

    Works with vector DB (ChromaDB/FAISS) to find relevant chunks
    and uses LLM via OpenRouter API to generate answers.

    Attributes:
        llm (OpenRouterClient): Client for working with LLM
        prompts (RAGPrompts): Prompt generator
        vector_db: Vector database (stub for MVP)
        chat_history (List): Dialog history for contextual questions
    """

    def __init__(
        self,
        model: str = "anthropic/claude-3.5-sonnet",
        api_key: Optional[str] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ):
        """
        Initialize RAG engine.

        Args:
            model (str): LLM model for answer generation
            api_key (Optional[str]): OpenRouter API key (if not in env)
            top_k (int): Number of chunks to retrieve from vector DB
            similarity_threshold (float): Chunk relevance threshold (0-1)

        Example:
            engine = RAGEngine(
                model="anthropic/claude-3.5-sonnet",
                top_k=5,
                similarity_threshold=0.7
            )
        """
        self.llm = OpenRouterClient(model=model, api_key=api_key)
        self.prompts = RAGPrompts()
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.chat_history = []

        # TODO: Integration with real vector DB (ChromaDB/FAISS)
        # For MVP stage using mock data
        self.vector_db = None

    def query(
        self,
        user_query: str,
        use_chat_history: bool = True,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Process user query through RAG pipeline.

        Full cycle:
        1. (Optional) Convert query considering dialog history
        2. Perform vector search in transcription DB
        3. Format found context
        4. Generate answer via LLM with source citations
        5. Save to dialog history

        Args:
            user_query (str): User question
            use_chat_history (bool): Whether to use history for context
            stream (bool): Stream response (for future implementation)

        Returns:
            Dict[str, Any]: Dictionary with answer and metadata:
                - answer (str): Generated LLM answer
                - sources (List[Dict]): Used sources with timestamps
                - query (str): Original query
                - retrieved_chunks (List[Dict]): Found chunks

        Example:
            result = engine.query("Explain transformer attention principle")
            print(result["answer"])
            for source in result["sources"]:
                print(f"{source['name']} - {source['timestamp']}")
        """
        # Step 1: Convert query considering history (if needed)
        search_query = user_query
        if use_chat_history and self.chat_history:
            search_query = self._contextualize_query(user_query)

        # Step 2: Vector search for relevant chunks
        retrieved_chunks = self._retrieve_context(search_query)

        # Step 3: Format prompt with context
        user_prompt = self.prompts.format_user_query(user_query, retrieved_chunks)
        system_prompt = self.prompts.get_system_prompt()

        # Step 4: Generate answer via LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        answer = self.llm.chat(messages, temperature=0.7, max_tokens=2000)

        # Step 5: Extract sources from context
        sources = self._extract_sources(retrieved_chunks)

        # Save to history
        self.chat_history.append({"role": "user", "content": user_query})
        self.chat_history.append({"role": "assistant", "content": answer})

        return {
            "answer": answer,
            "sources": sources,
            "query": user_query,
            "retrieved_chunks": retrieved_chunks
        }

    def _contextualize_query(self, query: str) -> str:
        """
        Convert query to standalone considering dialog history.

        Uses LLM to reformulate follow-up questions
        ("tell me more", "what about X?") into complete standalone queries.

        Args:
            query (str): Current user query

        Returns:
            str: Standalone query for vector search

        Example:
            History: "What is BERT?" -> "BERT is a model..."
            Query: "What are the drawbacks?"
            Result: "What are the drawbacks of BERT model?"
        """
        prompt = self.prompts.format_standalone_query(query, self.chat_history)

        messages = [{"role": "user", "content": prompt}]
        standalone_query = self.llm.chat(messages, temperature=0.3, max_tokens=200)

        return standalone_query.strip()

    def _retrieve_context(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform vector search for relevant chunks.

        For MVP stage returns mock data.
        In production will integrate with ChromaDB/FAISS.

        Args:
            query (str): Search query

        Returns:
            List[Dict[str, Any]]: List of found chunks with metadata

        TODO: Integrate real vector DB:
            - Embeddings via OpenAI/HuggingFace
            - ChromaDB for storage and search
            - Filtering by similarity_threshold
        """
        # Mock data for MVP
        mock_chunks = [
            {
                "text": "Attention mechanism in transformers allows the model to dynamically "
                        "weight the importance of different parts of the input sequence. "
                        "This is a key difference from recurrent networks.",
                "source": "Stanford CS224N Lecture - Transformers",
                "timestamp": 1520,
                "score": 0.92
            },
            {
                "text": "Self-attention computes three matrices: Query, Key and Value. "
                        "Then applies formula attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V",
                "source": "Stanford CS224N Lecture - Transformers",
                "timestamp": 1680,
                "score": 0.88
            },
            {
                "text": "Multi-head attention allows the model to jointly attend to information "
                        "from different representation subspaces at different positions.",
                "source": "PyData Talk - Attention is All You Need",
                "timestamp": 420,
                "score": 0.85
            }
        ]

        # Filter by relevance threshold
        filtered_chunks = [
            chunk for chunk in mock_chunks
            if chunk["score"] >= self.similarity_threshold
        ]

        # Return top_k results
        return filtered_chunks[:self.top_k]

    def _extract_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Extract unique sources from found chunks.

        Forms list of sources with metadata for displaying to user.

        Args:
            chunks (List[Dict]): Found chunks with metadata

        Returns:
            List[Dict[str, str]]: List of sources:
                - name (str): Source name
                - timestamp (str): Timestamp
                - url (str, optional): Link to video/podcast

        Example:
            sources = [
                {
                    "name": "Stanford CS224N Lecture",
                    "timestamp": "25:20",
                    "url": "https://youtube.com/watch?v=..."
                }
            ]
        """
        sources = []
        seen_sources = set()

        for chunk in chunks:
            source_name = chunk.get("source", "Unknown source")
            timestamp = chunk.get("timestamp", 0)
            time_str = self.prompts._format_timestamp(timestamp)

            # Create unique key
            source_key = f"{source_name}|{timestamp}"

            if source_key not in seen_sources:
                seen_sources.add(source_key)
                sources.append({
                    "name": source_name,
                    "timestamp": time_str,
                    # TODO: Add URL from DB
                    # "url": chunk.get("url", "")
                })

        return sources

    def clear_history(self):
        """
        Clear dialog history.

        Used to start new session or reset context.

        Example:
            engine.clear_history()
        """
        self.chat_history = []

    def get_history(self) -> List[Dict[str, str]]:
        """
        Return current dialog history.

        Returns:
            List[Dict[str, str]]: Message history

        Example:
            history = engine.get_history()
            for msg in history:
                print(f"{msg['role']}: {msg['content']}")
        """
        return self.chat_history


# Usage example for testing
if __name__ == "__main__":
    # Initialize engine
    engine = RAGEngine()

    # Simple query
    result = engine.query("What is attention mechanism in transformers?")

    print("=== Answer ===")
    print(result["answer"])
    print("\n=== Sources ===")
    for source in result["sources"]:
        print(f"- {source['name']} [{source['timestamp']}]")
