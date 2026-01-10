"""
Local files source plugin.

Загружает файлы с локального диска для обработки.
"""

from typing import Iterator, Dict, Any, List, Optional
from pathlib import Path
import logging
import mimetypes

from ..core.interfaces import BaseSource
from ..core.models import ContentItem, ContentType, SourceType, ProcessingStage
from ..core.registry import register_source

logger = logging.getLogger(__name__)


# Маппинг расширений файлов на типы контента
EXTENSION_TO_CONTENT_TYPE = {
    # Аудио
    ".mp3": ContentType.AUDIO,
    ".wav": ContentType.AUDIO,
    ".m4a": ContentType.AUDIO,
    ".flac": ContentType.AUDIO,
    ".ogg": ContentType.AUDIO,
    ".wma": ContentType.AUDIO,
    # Видео
    ".mp4": ContentType.VIDEO,
    ".mkv": ContentType.VIDEO,
    ".avi": ContentType.VIDEO,
    ".mov": ContentType.VIDEO,
    ".webm": ContentType.VIDEO,
    ".flv": ContentType.VIDEO,
    # Документы
    ".pdf": ContentType.PDF,
    ".docx": ContentType.DOCX,
    ".doc": ContentType.DOCX,
    # Текст
    ".txt": ContentType.TEXT,
    ".md": ContentType.TEXT,
    ".rst": ContentType.TEXT,
    ".csv": ContentType.TEXT,
    ".json": ContentType.TEXT,
    # HTML
    ".html": ContentType.HTML,
    ".htm": ContentType.HTML,
    # Изображения
    ".png": ContentType.IMAGE,
    ".jpg": ContentType.IMAGE,
    ".jpeg": ContentType.IMAGE,
    ".gif": ContentType.IMAGE,
    ".bmp": ContentType.IMAGE,
    ".webp": ContentType.IMAGE,
    ".tiff": ContentType.IMAGE,
}


@register_source
class LocalFileSource(BaseSource):
    """
    Источник данных из локальной файловой системы.

    Поддерживает различные типы файлов:
    - Аудио/Видео (mp3, wav, mp4, mkv и т.д.)
    - Документы (PDF, DOCX)
    - Текстовые файлы (txt, md)
    - Изображения (png, jpg и т.д.)
    """

    source_name = "local_files"
    supported_content_types = list(set(EXTENSION_TO_CONTENT_TYPE.values()))

    def __init__(
        self,
        base_path: str = ".",
        extensions: List[str] = None,
        recursive: bool = True,
    ):
        """
        Инициализация источника.

        Args:
            base_path: Базовый путь для поиска файлов
            extensions: Список расширений для фильтрации (например [".pdf", ".docx"])
            recursive: Искать рекурсивно в поддиректориях
        """
        self.base_path = Path(base_path)
        self.extensions = extensions
        self.recursive = recursive

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Проверяет валидность конфигурации."""
        return "paths" in config or "path" in config

    def fetch(
        self,
        paths: List[str] = None,
        extensions: List[str] = None,
        recursive: bool = None,
        skip_paths: List[str] = None,
        **kwargs,
    ) -> Iterator[ContentItem]:
        """
        Получает файлы из локальной файловой системы.

        Args:
            paths: Список путей к файлам или директориям
            extensions: Фильтр по расширениям (переопределяет настройку из __init__)
            recursive: Рекурсивный поиск (переопределяет настройку из __init__)
            skip_paths: Пути для пропуска

        Yields:
            ContentItem объекты для каждого файла
        """
        paths = paths or kwargs.get("path", [str(self.base_path)])
        if isinstance(paths, str):
            paths = [paths]

        extensions = extensions or self.extensions
        if extensions:
            extensions = [ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions]

        recursive = recursive if recursive is not None else self.recursive
        skip_paths = set(skip_paths or [])

        for path_str in paths:
            path = Path(path_str)

            if path.is_file():
                # Одиночный файл
                if str(path.absolute()) not in skip_paths:
                    item = self._create_item(path, extensions)
                    if item:
                        yield item

            elif path.is_dir():
                # Директория
                glob_pattern = "**/*" if recursive else "*"
                for file_path in path.glob(glob_pattern):
                    if file_path.is_file():
                        if str(file_path.absolute()) in skip_paths:
                            continue
                        item = self._create_item(file_path, extensions)
                        if item:
                            yield item
            else:
                logger.warning(f"Path not found: {path}")

    def _create_item(
        self,
        path: Path,
        extensions_filter: List[str] = None,
    ) -> Optional[ContentItem]:
        """
        Создаёт ContentItem из файла.

        Args:
            path: Путь к файлу
            extensions_filter: Список разрешённых расширений

        Returns:
            ContentItem или None если файл не поддерживается
        """
        suffix = path.suffix.lower()

        # Фильтр по расширениям
        if extensions_filter and suffix not in extensions_filter:
            return None

        # Определяем тип контента
        content_type = EXTENSION_TO_CONTENT_TYPE.get(suffix)
        if content_type is None:
            logger.debug(f"Unsupported file type: {path}")
            return None

        try:
            stat = path.stat()
            file_size = stat.st_size
            modified_time = stat.st_mtime
        except Exception as e:
            logger.warning(f"Cannot read file stats: {path}: {e}")
            return None

        return ContentItem(
            source_id=str(path.absolute()),
            content_type=content_type,
            source_type=SourceType.LOCAL_FILE,
            title=path.stem,
            author="local",
            source_path=path,
            processing_stage=ProcessingStage.PENDING,
            metadata={
                "file_size": file_size,
                "modified_time": modified_time,
                "extension": suffix,
                "parent_dir": str(path.parent),
            },
        )


def get_content_type_for_extension(extension: str) -> Optional[ContentType]:
    """
    Возвращает тип контента для расширения файла.

    Args:
        extension: Расширение файла (с или без точки)

    Returns:
        ContentType или None
    """
    if not extension.startswith("."):
        extension = f".{extension}"
    return EXTENSION_TO_CONTENT_TYPE.get(extension.lower())


def get_supported_extensions() -> List[str]:
    """Возвращает список всех поддерживаемых расширений."""
    return list(EXTENSION_TO_CONTENT_TYPE.keys())
