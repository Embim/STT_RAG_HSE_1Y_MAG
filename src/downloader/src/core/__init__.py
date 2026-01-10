"""Core module - ядро системы."""

from .models import ContentType, SourceType, ContentItem, TextSegment, TextChunk
from .interfaces import BaseSource, BaseProcessor, BaseChunker, BaseEmbedder, BaseStore
from .registry import PluginRegistry, register_source, register_processor, register_embedder, register_store
from .config import Config, ConfigLoader

__all__ = [
    # Models
    "ContentType",
    "SourceType",
    "ContentItem",
    "TextSegment",
    "TextChunk",
    # Interfaces
    "BaseSource",
    "BaseProcessor",
    "BaseChunker",
    "BaseEmbedder",
    "BaseStore",
    # Registry
    "PluginRegistry",
    "register_source",
    "register_processor",
    "register_embedder",
    "register_store",
    # Config
    "Config",
    "ConfigLoader",
]
