from typing import List, Dict, Any, Optional
from src.system.llm.llm import OpenRouterClient
from src.system.llm.prompts import RAGPrompts


class RAGEngine:
    """
    Main engine for RAG search and answer generation.

    Combines vector search through transcription database and answer generation via LLM.

    Works with vector DB (ChromaDB/FAISS) to find relevant chunks
    and uses LLM via OpenRouter API to generate answers.

    Attributes:
        llm (OpenRouterClient): Client for working with LLM
        prompts (RAGPrompts): Prompt generator
        vector_db: Vector database (stub for MVP)
    """

    def __init__(
        self,
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
        """
        self.llm = OpenRouterClient()
        self.prompts = RAGPrompts()
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

        # TODO: Integration with real vector DB (ChromaDB/FAISS)
        # For MVP stage using mock data
        self.vector_db = None

    def query(
        self,
        user_query: str
    ) -> Dict[str, Any]:
        """
        Process user query through RAG pipeline.

        Full cycle:
        1. Perform vector search in transcription DB
        2. Format found context
        3. Generate answer via LLM with source citations

        Args:
            user_query (str): User question

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
        # Step 1: Vector search for relevant chunks
        retrieved_chunks = self._retrieve_context(user_query)

        # Step 2: Format prompt with context
        user_prompt = self.prompts.format_user_query(user_query, retrieved_chunks)
        system_prompt = self.prompts.get_system_prompt()

        # Step 3: Generate answer via LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        answer = self.llm.chat(messages, temperature=0.7, max_tokens=2000)

        # Step 4: Extract sources from context
        sources = self._extract_sources(retrieved_chunks)

        return {
            "answer": answer,
            "sources": sources,
            "query": user_query,
            "retrieved_chunks": retrieved_chunks
        }

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
