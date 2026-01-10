"""
File utilities.

Функции для работы с файлами: удаление, архивирование, проверки.
"""

import os
import zipfile
from pathlib import Path
from typing import List
import logging
from datetime import datetime
import shutil

logger = logging.getLogger(__name__)


def cleanup_file(file_path: str, missing_ok: bool = True) -> bool:
    """
    Удаляет файл.

    Args:
        file_path: Путь к файлу
        missing_ok: Не выбрасывать ошибку если файл не существует

    Returns:
        True если файл был удалён
    """
    path = Path(file_path)

    if not path.exists():
        if missing_ok:
            return False
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        path.unlink()
        logger.debug(f"Deleted file: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to delete file {file_path}: {e}")
        return False


def cleanup_directory(
    directory: str,
    pattern: str = "*",
    recursive: bool = False,
) -> int:
    """
    Удаляет файлы из директории по паттерну.

    Args:
        directory: Путь к директории
        pattern: Glob паттерн для файлов
        recursive: Рекурсивно удалять файлы

    Returns:
        Количество удалённых файлов
    """
    dir_path = Path(directory)

    if not dir_path.exists():
        logger.warning(f"Directory not found: {directory}")
        return 0

    count = 0
    glob_method = dir_path.rglob if recursive else dir_path.glob

    for file_path in glob_method(pattern):
        if file_path.is_file():
            if cleanup_file(str(file_path)):
                count += 1

    logger.info(f"Cleaned up {count} files from {directory}")
    return count


def create_archive(
    files: List[str],
    archive_path: str,
    compression: int = zipfile.ZIP_DEFLATED,
    delete_originals: bool = False,
) -> bool:
    """
    Создаёт ZIP архив из списка файлов.

    Args:
        files: Список путей к файлам
        archive_path: Путь к создаваемому архиву
        compression: Тип сжатия
        delete_originals: Удалить оригиналы после архивирования

    Returns:
        True если архив создан успешно
    """
    archive = Path(archive_path)
    archive.parent.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(archive, "w", compression=compression) as zf:
            for file_path in files:
                path = Path(file_path)
                if path.exists() and path.is_file():
                    arcname = path.name
                    zf.write(file_path, arcname=arcname)
                    logger.debug(f"Added {file_path} to archive")
                else:
                    logger.warning(f"File not found, skipping: {file_path}")

        logger.info(f"Created archive: {archive_path} with {len(files)} files")

        if delete_originals:
            for file_path in files:
                cleanup_file(file_path)

        return True

    except Exception as e:
        logger.error(f"Failed to create archive {archive_path}: {e}")
        return False


def batch_archive_transcripts(
    transcript_dir: str,
    archive_dir: str,
    batch_size: int = 10,
    delete_originals: bool = False,
) -> List[str]:
    """
    Архивирует транскрипты батчами.

    Args:
        transcript_dir: Директория с транскриптами
        archive_dir: Директория для архивов
        batch_size: Количество файлов в одном архиве
        delete_originals: Удалить оригиналы после архивирования

    Returns:
        Список путей к созданным архивам
    """
    transcript_path = Path(transcript_dir)
    archive_path = Path(archive_dir)
    archive_path.mkdir(parents=True, exist_ok=True)

    # Получаем все файлы транскриптов
    txt_files = sorted(transcript_path.glob("*.txt"))
    json_files = sorted(transcript_path.glob("*.json"))
    all_files = list(txt_files) + list(json_files)

    if not all_files:
        logger.warning(f"No transcript files found in {transcript_dir}")
        return []

    archives_created = []

    # Батчуем файлы
    for i in range(0, len(all_files), batch_size):
        batch = all_files[i:i + batch_size]
        batch_files = [str(f) for f in batch]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"transcripts_batch_{i // batch_size + 1}_{timestamp}.zip"
        archive_full_path = str(archive_path / archive_name)

        if create_archive(batch_files, archive_full_path, delete_originals=delete_originals):
            archives_created.append(archive_full_path)

    logger.info(f"Created {len(archives_created)} archives from {len(all_files)} files")
    return archives_created


def get_file_size_mb(file_path: str) -> float:
    """Возвращает размер файла в МБ."""
    path = Path(file_path)
    if not path.exists():
        return 0.0

    size_bytes = path.stat().st_size
    return round(size_bytes / (1024 * 1024), 2)


def get_directory_size_mb(directory: str) -> float:
    """Возвращает общий размер директории в МБ."""
    dir_path = Path(directory)
    if not dir_path.exists():
        return 0.0

    total_size = sum(f.stat().st_size for f in dir_path.rglob("*") if f.is_file())
    return round(total_size / (1024 * 1024), 2)


def ensure_disk_space(path: str, required_gb: float = 5.0) -> bool:
    """
    Проверяет наличие свободного места на диске.

    Args:
        path: Путь к директории для проверки
        required_gb: Требуемое место в ГБ

    Returns:
        True если места достаточно
    """
    try:
        stat = shutil.disk_usage(path)
        free_gb = stat.free / (1024 ** 3)

        if free_gb < required_gb:
            logger.warning(
                f"Low disk space on {path}: {free_gb:.2f}GB free, "
                f"{required_gb:.2f}GB required"
            )
            return False

        return True

    except Exception as e:
        logger.error(f"Failed to check disk space: {e}")
        return False
