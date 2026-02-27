"""Утилиты для обработки файлов встреч и генерации резюме."""
import logging
from pathlib import Path
from typing import Dict, Optional, Union
import json
from dotenv import load_dotenv

# Загружаем переменные из .env файла
load_dotenv()

# Импорты из существующего pipeline
import sys
from pathlib import Path as PathLib
project_root = PathLib(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.downloader.src.processors.audio.whisper import WhisperProcessor
from src.downloader.adapters.openrouter_client import OpenRouterClient

logger = logging.getLogger(__name__)


class MeetingProcessor:
    """Процессор для обработки файлов встреч и генерации резюме."""

    def __init__(
        self,
        openrouter_api_key: Optional[str] = None,
        whisper_model: str = "openai/whisper-large-v3-turbo",
        whisper_device: str = "cuda",
        nemotron_model: str = "nvidia/llama-3.1-nemotron-70b-instruct"
    ):
        """
        Инициализация процессора встреч.

        Args:
            openrouter_api_key: API ключ OpenRouter
            whisper_model: Модель Whisper для транскрипции
            whisper_device: Устройство для Whisper (cuda/cpu)
            nemotron_model: Модель Nemotron через OpenRouter
        """
        # Инициализация Whisper для транскрипции
        self.whisper = WhisperProcessor(
            model_name=whisper_model,
            device=whisper_device,
            compute_type="float16" if whisper_device == "cuda" else "float32",
            language="ru"  # Можно изменить на нужный язык
        )

        # Инициализация OpenRouter клиента
        self.openrouter = OpenRouterClient(
            api_key=openrouter_api_key,
            model=nemotron_model
        )

        self._whisper_setup = False

    def process_audio_file(
        self,
        file_path: Union[str, Path],
        save_transcript: bool = True,
        save_summary: bool = True,
        output_dir: Optional[Union[str, Path]] = None
    ) -> Dict:
        """
        Обрабатывает аудио/видео файл: транскрибирует и генерирует резюме.

        Args:
            file_path: Путь к аудио/видео файлу
            save_transcript: Сохранить ли транскрипцию в файл
            save_summary: Сохранить ли резюме в файл
            output_dir: Директория для сохранения результатов (по умолчанию рядом с файлом)

        Returns:
            Dict с результатами обработки:
            {
                "file_path": str,
                "transcript": str,
                "summary": dict,
                "transcript_file": str (если сохранен),
                "summary_file": str (если сохранен)
            }
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")

        logger.info(f"Обработка файла: {file_path}")

        # Определяем output директорию
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = file_path.parent

        # Шаг 1: Транскрипция аудио
        logger.info("Транскрибирование аудио...")
        transcript = self._transcribe_audio(file_path)

        # Сохранение транскрипции
        transcript_file = None
        if save_transcript:
            transcript_file = output_dir / f"{file_path.stem}_transcript.txt"
            with open(transcript_file, "w", encoding="utf-8") as f:
                f.write(transcript)
            logger.info(f"Транскрипция сохранена: {transcript_file}")

        # Шаг 2: Генерация резюме через OpenRouter
        logger.info("Генерация резюме встречи через Nvidia Nemotron...")
        summary = self.openrouter.generate_meeting_summary(transcript)

        # Сохранение резюме
        summary_file = None
        if save_summary:
            summary_file = output_dir / f"{file_path.stem}_summary.json"
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            logger.info(f"Резюме сохранено: {summary_file}")

        result = {
            "file_path": str(file_path),
            "transcript": transcript,
            "summary": summary,
        }

        if transcript_file:
            result["transcript_file"] = str(transcript_file)
        if summary_file:
            result["summary_file"] = str(summary_file)

        logger.info("Обработка завершена успешно!")
        return result

    def _transcribe_audio(self, file_path: Path) -> str:
        """Транскрибирует аудио файл."""
        from src.downloader.src.core.models import ContentItem, ContentType

        # Setup Whisper если еще не сделано
        if not self._whisper_setup:
            self.whisper.setup({})
            self._whisper_setup = True

        # Создаем ContentItem для обработки
        item = ContentItem(
            id=file_path.stem,
            source_path=file_path,
            content_type=ContentType.AUDIO
        )

        # Обрабатываем через Whisper
        processed_item = self.whisper.process(item)

        # Возвращаем текст
        return processed_item.raw_text

    def process_text_file(
        self,
        file_path: Union[str, Path],
        save_summary: bool = True,
        output_dir: Optional[Union[str, Path]] = None
    ) -> Dict:
        """
        Обрабатывает текстовый файл с готовой транскрипцией.

        Args:
            file_path: Путь к текстовому файлу
            save_summary: Сохранить ли резюме в файл
            output_dir: Директория для сохранения результатов

        Returns:
            Dict с результатами
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")

        logger.info(f"Обработка текстового файла: {file_path}")

        # Читаем транскрипцию
        with open(file_path, "r", encoding="utf-8") as f:
            transcript = f.read()

        # Определяем output директорию
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = file_path.parent

        # Генерация резюме
        logger.info("Генерация резюме встречи через Nvidia Nemotron...")
        summary = self.openrouter.generate_meeting_summary(transcript)

        # Сохранение резюме
        summary_file = None
        if save_summary:
            summary_file = output_dir / f"{file_path.stem}_summary.json"
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            logger.info(f"Резюме сохранено: {summary_file}")

        result = {
            "file_path": str(file_path),
            "transcript": transcript,
            "summary": summary,
        }

        if summary_file:
            result["summary_file"] = str(summary_file)

        logger.info("Обработка завершена успешно!")
        return result

    def cleanup(self):
        """Освобождает ресурсы."""
        if self._whisper_setup:
            self.whisper.teardown()
            self._whisper_setup = False


def process_meeting_file(
    file_path: Union[str, Path],
    openrouter_api_key: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    save_files: bool = True
) -> Dict:
    """
    Удобная функция для обработки одного файла встречи.

    Args:
        file_path: Путь к аудио/видео/текстовому файлу
        openrouter_api_key: API ключ OpenRouter
        output_dir: Директория для сохранения результатов
        save_files: Сохранять ли результаты в файлы

    Returns:
        Dict с результатами обработки
    """
    processor = MeetingProcessor(openrouter_api_key=openrouter_api_key)

    try:
        file_path = Path(file_path)

        # Определяем тип файла
        audio_extensions = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".mp4", ".avi", ".mkv"}
        text_extensions = {".txt", ".text"}

        if file_path.suffix.lower() in audio_extensions:
            result = processor.process_audio_file(
                file_path=file_path,
                save_transcript=save_files,
                save_summary=save_files,
                output_dir=output_dir
            )
        elif file_path.suffix.lower() in text_extensions:
            result = processor.process_text_file(
                file_path=file_path,
                save_summary=save_files,
                output_dir=output_dir
            )
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {file_path.suffix}")

        return result
    finally:
        processor.cleanup()
