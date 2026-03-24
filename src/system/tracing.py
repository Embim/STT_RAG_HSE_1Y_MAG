import logging
import os

from langfuse import get_client, observe
from settings import settings

logger = logging.getLogger(__name__)

os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.LANGFUSE_PUBLIC_KEY)
os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.LANGFUSE_SECRET_KEY)
os.environ.setdefault("LANGFUSE_HOST", settings.LANGFUSE_BASE_URL)
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

_langfuse_client = get_client()
logger.info("Langfuse enabled: %s", settings.LANGFUSE_BASE_URL)


def flush() -> None:
    try:
        _langfuse_client.flush()
    except Exception as exc:
        logger.warning("Langfuse flush error: %s", exc)
