#!/usr/bin/env python3
"""
Скрипт для обработки файлов встреч и генерации резюме.

Использование:
    # Обработка аудио файла:
    python process_meeting.py path/to/audio.mp3

    # Обработка текстового файла с транскрипцией:
    python process_meeting.py path/to/transcript.txt

    # С указанием output директории:
    python process_meeting.py path/to/audio.mp3 --output-dir ./results

Требуется установленная переменная окружения:
    OPENROUTER_API_KEY - API ключ OpenRouter
"""

import sys
import argparse
import logging
from pathlib import Path
import json
import os
from dotenv import load_dotenv

# Загружаем переменные из .env файла
load_dotenv()

# Добавляем корень проекта в sys.path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.downloader.utils.meeting_processor import process_meeting_file


def setup_logging():
    """Настройка логирования."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def print_summary(summary: dict):
    """Красиво выводит протокол встречи."""
    print("\n" + "="*80)
    print("ПРОТОКОЛ ВСТРЕЧИ")
    print("="*80 + "\n")

    if "error" in summary:
        print(f"❌ Ошибка: {summary.get('error')}")
        print(f"Детали: {summary.get('details', summary.get('raw_response', 'N/A'))}")
        return

    # Заголовок
    if "title" in summary:
        print(f"{summary['title']}")
        print("=" * len(summary['title']))
        print()

    # Основной текст протокола
    if "protocol" in summary:
        # Разбиваем на абзацы и выводим
        paragraphs = summary["protocol"].split("\n\n")
        for paragraph in paragraphs:
            if paragraph.strip():
                # Форматируем абзац для консоли (80 символов в строке)
                import textwrap
                wrapped = textwrap.fill(paragraph.strip(), width=80)
                print(wrapped)
                print()  # Пустая строка между абзацами
    else:
        # Fallback на старый формат
        if "summary" in summary:
            print(summary["summary"])
            print()

    print("="*80)


def main():
    """Главная функция."""
    parser = argparse.ArgumentParser(
        description="Обработка файлов встреч и генерация резюме через Nvidia Nemotron"
    )
    parser.add_argument(
        "file_path",
        type=str,
        help="Путь к аудио/видео/текстовому файлу встречи"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Директория для сохранения результатов (по умолчанию рядом с исходным файлом)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Не сохранять результаты в файлы, только вывести в консоль"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenRouter API ключ (по умолчанию из env OPENROUTER_API_KEY)"
    )

    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    # Проверяем API ключ
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("❌ OpenRouter API ключ не найден!")
        logger.error("Установите переменную окружения OPENROUTER_API_KEY или передайте --api-key")
        sys.exit(1)

    # Проверяем файл
    file_path = Path(args.file_path)
    if not file_path.exists():
        logger.error(f"❌ Файл не найден: {file_path}")
        sys.exit(1)

    try:
        logger.info(f"🚀 Начало обработки файла: {file_path}")

        # Обрабатываем файл
        result = process_meeting_file(
            file_path=file_path,
            openrouter_api_key=api_key,
            output_dir=args.output_dir,
            save_files=not args.no_save
        )

        # Выводим резюме
        print_summary(result["summary"])

        # Информация о сохраненных файлах
        if not args.no_save:
            print("\n📁 Сохраненные файлы:")
            if "transcript_file" in result:
                print(f"  📄 Транскрипция: {result['transcript_file']}")
            if "summary_file" in result:
                print(f"  📊 Резюме: {result['summary_file']}")

        logger.info("✅ Обработка завершена успешно!")

    except Exception as e:
        logger.error(f"❌ Ошибка при обработке: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
