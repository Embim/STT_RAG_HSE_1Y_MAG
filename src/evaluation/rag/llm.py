"""RAGAS judge LLM and embeddings wired to the project's OpenRouter + FRIDA."""
from __future__ import annotations

from settings import settings


def require_eval_extra() -> None:
    """Raise a friendly error if the optional `eval` extra isn't installed."""
    try:
        import ragas  # noqa: F401
        import langchain_openai  # noqa: F401
        import datasets  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "RAG evaluation requires the `eval` extra. Run: uv sync --extra eval"
        ) from e


def _langfuse_callbacks() -> list:
    """Return [langfuse.langchain.CallbackHandler()] when Langfuse is on, else [].

    The handler intercepts every langchain LLM call (i.e. every RAGAS judge
    invocation) and emits it as a Langfuse generation, so judge prompts and
    responses become visible in the UI alongside the rest of the trace.
    """
    try:
        from system.tracing import _LANGFUSE_ACTIVE  # type: ignore
    except Exception:
        _LANGFUSE_ACTIVE = False
    if not _LANGFUSE_ACTIVE:
        return []
    try:
        from langfuse.langchain import CallbackHandler
        return [CallbackHandler()]
    except Exception:
        return []


def get_llm():
    """Build the langchain ChatOpenAI used by RAGAS as judge.

    Uses the same OpenRouter credentials as the production RAG pipeline so
    there are no extra secrets to manage. Langfuse callback is attached so
    every judge call becomes a tracked generation in the UI.
    """
    require_eval_extra()
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    model = settings.RAGAS_JUDGE_MODEL or settings.LLM_MODEL
    chat = ChatOpenAI(
        model=model,
        api_key=settings.LLM_API_KEY_3,
        base_url=settings.LLM_BASE_URL,
        temperature=0.0,
        max_tokens=settings.LLM_MAX_TOKENS,
        callbacks=_langfuse_callbacks(),
    )
    return LangchainLLMWrapper(chat)


def get_embeddings():
    """Build the embeddings client RAGAS uses for similarity-based metrics.

    Points at the same Infinity (FRIDA) endpoint as the production retriever.
    """
    require_eval_extra()
    from langchain_openai import OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    base_url = settings.EMBEDDING_URL.rstrip("/")
    if base_url.endswith("/v1/embeddings"):
        base_url = base_url[: -len("/embeddings")]
    elif not base_url.endswith("/v1"):
        base_url = base_url + "/v1"

    emb = OpenAIEmbeddings(
        model="ai-forever/FRIDA",
        base_url=base_url,
        api_key="sk-local-frida",
        check_embedding_ctx_length=False,
    )
    # OpenAIEmbeddings doesn't support callbacks; embedding calls won't show up
    # individually in Langfuse. Acceptable — what matters is judge LLM calls.
    return LangchainEmbeddingsWrapper(emb)
