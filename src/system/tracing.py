import logging
import os
import sys

from settings import settings

logger = logging.getLogger(__name__)

os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.LANGFUSE_PUBLIC_KEY)
os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.LANGFUSE_SECRET_KEY)
os.environ.setdefault("LANGFUSE_HOST", settings.LANGFUSE_BASE_URL)
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

_LANGFUSE_ACTIVE = False


class _NoopLangfuseClient:
    def update_current_trace(self, **kwargs) -> None:
        return None

    def update_current_generation(self, **kwargs) -> None:
        return None

    def update_current_span(self, **kwargs) -> None:
        return None

    def flush(self) -> None:
        return None


_langfuse_client = _NoopLangfuseClient()

if settings.LANGFUSE_ENABLED and sys.version_info < (3, 14):
    try:
        from langfuse import get_client as _langfuse_get_client, observe as _langfuse_observe

        _langfuse_client = _langfuse_get_client()
        _LANGFUSE_ACTIVE = True
        logger.info("Langfuse enabled: %s", settings.LANGFUSE_BASE_URL)
    except Exception as exc:
        logger.warning("Langfuse disabled (import/runtime error): %s", exc)
else:
    if settings.LANGFUSE_ENABLED and sys.version_info >= (3, 14):
        logger.warning("Langfuse disabled on Python %s.%s (use Python <=3.13).", sys.version_info.major, sys.version_info.minor)
    else:
        logger.info("Langfuse disabled by config.")


def observe(*args, **kwargs):
    if _LANGFUSE_ACTIVE:
        return _langfuse_observe(*args, **kwargs)

    def _decorator(func):
        return func

    return _decorator


def get_client():
    return _langfuse_client


def flush() -> None:
    try:
        _langfuse_client.flush()
    except Exception as exc:
        logger.warning("Langfuse flush error: %s", exc)
