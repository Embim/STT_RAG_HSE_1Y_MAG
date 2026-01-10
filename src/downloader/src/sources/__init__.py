"""Data sources - источники данных."""

from .youtube import YouTubeSource
from .local_files import LocalFileSource

__all__ = ["YouTubeSource", "LocalFileSource"]
