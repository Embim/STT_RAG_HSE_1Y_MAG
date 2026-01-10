"""
Processing tracker.

Отслеживает обработанные элементы и ошибки.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set
from pathlib import Path
from datetime import datetime
import json
import logging
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class ProcessedRecord:
    """Запись об обработанном элементе."""
    source_id: str
    title: str
    author: str
    chunks_count: int
    duration: Optional[float]
    processed_at: str
    status: str = "completed"


@dataclass
class ErrorRecord:
    """Запись об ошибке."""
    source_id: str
    stage: str
    error_type: str
    error_message: str
    timestamp: str


class ProcessingTracker:
    """
    Трекер обработки.

    Сохраняет информацию об обработанных элементах и ошибках.
    Используется для идемпотентности - пропуска уже обработанных элементов.
    """

    def __init__(
        self,
        processed_journal_path: str = "processed.json",
        error_journal_path: str = "errors.json",
    ):
        """
        Инициализация трекера.

        Args:
            processed_journal_path: Путь к журналу обработанных
            error_journal_path: Путь к журналу ошибок
        """
        self.processed_journal_path = Path(processed_journal_path)
        self.error_journal_path = Path(error_journal_path)

        self._processed: Dict[str, ProcessedRecord] = {}
        self._errors: List[ErrorRecord] = []
        self._lock = Lock()

        self._load()

    def _load(self) -> None:
        """Загружает журналы из файлов."""
        # Загружаем обработанные
        if self.processed_journal_path.exists():
            try:
                with open(self.processed_journal_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                    # Поддержка старого формата (dict) и нового (list)
                    if isinstance(data, dict):
                        # Старый формат: {video_id: {...}, ...}
                        for source_id, item in data.items():
                            record = ProcessedRecord(
                                source_id=item.get("video_id", source_id),
                                title=item.get("title", ""),
                                author=item.get("uploader", ""),
                                chunks_count=item.get("chunks_count", 0),
                                duration=item.get("duration"),
                                processed_at=item.get("processed_at", ""),
                                status=item.get("status", "completed"),
                            )
                            self._processed[record.source_id] = record
                    elif isinstance(data, list):
                        # Новый формат: [{...}, {...}, ...]
                        for item in data:
                            record = ProcessedRecord(**item)
                            self._processed[record.source_id] = record

                logger.debug(f"Loaded {len(self._processed)} processed records")
            except Exception as e:
                logger.warning(f"Failed to load processed journal: {e}")

        # Загружаем ошибки
        if self.error_journal_path.exists():
            try:
                with open(self.error_journal_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._errors = [ErrorRecord(**item) for item in data]
                logger.debug(f"Loaded {len(self._errors)} error records")
            except Exception as e:
                logger.warning(f"Failed to load error journal: {e}")

    def _save_processed(self) -> None:
        """Сохраняет журнал обработанных."""
        self.processed_journal_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.processed_journal_path, "w", encoding="utf-8") as f:
            data = [asdict(record) for record in self._processed.values()]
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _save_errors(self) -> None:
        """Сохраняет журнал ошибок."""
        self.error_journal_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.error_journal_path, "w", encoding="utf-8") as f:
            data = [asdict(record) for record in self._errors]
            json.dump(data, f, ensure_ascii=False, indent=2)

    def is_processed(self, source_id: str) -> bool:
        """Проверяет, был ли элемент уже обработан."""
        with self._lock:
            return source_id in self._processed

    def get_processed_ids(self) -> Set[str]:
        """Возвращает множество ID обработанных элементов."""
        with self._lock:
            return set(self._processed.keys())

    def mark_processed(
        self,
        source_id: str,
        title: str = "",
        author: str = "",
        chunks_count: int = 0,
        duration: Optional[float] = None,
        status: str = "completed",
    ) -> None:
        """
        Отмечает элемент как обработанный.

        Args:
            source_id: ID источника
            title: Заголовок
            author: Автор
            chunks_count: Количество чанков
            duration: Длительность (для аудио/видео)
            status: Статус обработки
        """
        with self._lock:
            record = ProcessedRecord(
                source_id=source_id,
                title=title,
                author=author,
                chunks_count=chunks_count,
                duration=duration,
                processed_at=datetime.now().isoformat(),
                status=status,
            )
            self._processed[source_id] = record
            self._save_processed()

        logger.debug(f"Marked as processed: {source_id}")

    def log_error(
        self,
        source_id: str,
        stage: str,
        error_type: str,
        error_message: str,
    ) -> None:
        """
        Логирует ошибку.

        Args:
            source_id: ID источника
            stage: Этап обработки
            error_type: Тип ошибки
            error_message: Сообщение об ошибке
        """
        with self._lock:
            record = ErrorRecord(
                source_id=source_id,
                stage=stage,
                error_type=error_type,
                error_message=error_message,
                timestamp=datetime.now().isoformat(),
            )
            self._errors.append(record)
            self._save_errors()

        logger.debug(f"Logged error for {source_id}: {error_type}")

    def get_statistics(self) -> Dict[str, int]:
        """Возвращает статистику."""
        with self._lock:
            completed = sum(1 for r in self._processed.values() if r.status == "completed")
            failed = sum(1 for r in self._processed.values() if r.status == "failed")
            total_chunks = sum(r.chunks_count for r in self._processed.values())

            return {
                "total_processed": len(self._processed),
                "completed": completed,
                "failed": failed,
                "total_chunks": total_chunks,
                "total_errors": len(self._errors),
            }

    def clear(self) -> None:
        """Очищает журналы."""
        with self._lock:
            self._processed.clear()
            self._errors.clear()

            if self.processed_journal_path.exists():
                self.processed_journal_path.unlink()
            if self.error_journal_path.exists():
                self.error_journal_path.unlink()

        logger.info("Cleared processing journals")
