#!/usr/bin/env python3
"""
Пример использования системы обработки файлов встреч.

Демонстрирует различные способы использования API.
"""

import sys
from pathlib import Path
import json

# Добавляем корень проекта в sys.path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def example_1_simple():
    """Пример 1: Простая обработка одного файла."""
    print("="*80)
    print("ПРИМЕР 1: Простая обработка одного файла")
    print("="*80)

    from src.downloader.utils.meeting_processor import process_meeting_file

    # Обработка аудио файла
    result = process_meeting_file(
        file_path="path/to/meeting.mp3",
        # openrouter_api_key будет взят из env OPENROUTER_API_KEY
        output_dir="./results",
        save_files=True
    )

    print(f"✅ Файл обработан: {result['file_path']}")
    print(f"📄 Транскрипция сохранена: {result.get('transcript_file')}")
    print(f"📊 Резюме сохранено: {result.get('summary_file')}")

    # Доступ к резюме
    summary = result["summary"]
    print(f"\n📋 Название встречи: {summary.get('title')}")
    print(f"🔑 Ключевых пунктов: {len(summary.get('key_points', []))}")
    print(f"✅ Принятых решений: {len(summary.get('decisions', []))}")
    print(f"📌 Задач: {len(summary.get('action_items', []))}")


def example_2_advanced():
    """Пример 2: Расширенное использование с настройками."""
    print("\n" + "="*80)
    print("ПРИМЕР 2: Расширенное использование")
    print("="*80)

    from src.downloader.utils.meeting_processor import MeetingProcessor

    # Создаем процессор с кастомными настройками
    processor = MeetingProcessor(
        openrouter_api_key="your_key_here",
        whisper_model="openai/whisper-large-v3-turbo",
        whisper_device="cuda",  # или "cpu" если нет GPU
        nemotron_model="nvidia/llama-3.1-nemotron-70b-instruct"
    )

    try:
        # Обработка аудио файла
        result1 = processor.process_audio_file(
            file_path="meeting1.mp3",
            save_transcript=True,
            save_summary=True,
            output_dir="./results"
        )

        print(f"✅ Файл 1 обработан: {result1['file_path']}")

        # Обработка готовой транскрипции
        result2 = processor.process_text_file(
            file_path="transcript2.txt",
            save_summary=True,
            output_dir="./results"
        )

        print(f"✅ Файл 2 обработан: {result2['file_path']}")

    finally:
        # Обязательно очищаем ресурсы
        processor.cleanup()


def example_3_custom_prompt():
    """Пример 3: Использование кастомного промпта."""
    print("\n" + "="*80)
    print("ПРИМЕР 3: Кастомный промпт")
    print("="*80)

    from src.downloader.adapters.openrouter_client import OpenRouterClient

    client = OpenRouterClient(
        api_key="your_key_here",
        model="nvidia/llama-3.1-nemotron-70b-instruct"
    )

    # Пример транскрипции
    transcript = """
    [Начало встречи]
    Менеджер: Добрый день, коллеги. Сегодня обсудим запуск нового продукта.
    ...
    """

    # Кастомный промпт для извлечения только задач
    custom_prompt = """
    Ты эксперт по анализу встреч. Извлеки из транскрипции все задачи и поручения.

    Верни результат в формате JSON:
    {
        "tasks": [
            {
                "description": "Описание задачи",
                "assignee": "Ответственный",
                "priority": "Приоритет (высокий/средний/низкий)",
                "deadline": "Срок"
            }
        ]
    }
    """

    result = client.summarize_with_custom_prompt(
        transcript=transcript,
        custom_prompt=custom_prompt,
        temperature=0.3,
        json_mode=True
    )

    print("📌 Извлеченные задачи:")
    print(result)


def example_4_batch_processing():
    """Пример 4: Массовая обработка файлов."""
    print("\n" + "="*80)
    print("ПРИМЕР 4: Массовая обработка файлов")
    print("="*80)

    from src.downloader.utils.meeting_processor import MeetingProcessor
    from pathlib import Path

    # Список файлов для обработки
    files = [
        "meetings/meeting_2024_01_15.mp3",
        "meetings/meeting_2024_01_22.mp3",
        "meetings/meeting_2024_01_29.mp3",
    ]

    processor = MeetingProcessor()

    try:
        results = []

        for file_path in files:
            if not Path(file_path).exists():
                print(f"⚠️ Файл не найден: {file_path}")
                continue

            print(f"\n🔄 Обработка: {file_path}")

            result = processor.process_audio_file(
                file_path=file_path,
                save_transcript=True,
                save_summary=True,
                output_dir="./results"
            )

            results.append(result)
            print(f"✅ Готово: {result['summary'].get('title', 'Без названия')}")

        # Сводная статистика
        print(f"\n📊 Обработано файлов: {len(results)}")

        total_decisions = sum(
            len(r['summary'].get('decisions', []))
            for r in results
        )
        print(f"✅ Всего решений: {total_decisions}")

        total_tasks = sum(
            len(r['summary'].get('action_items', []))
            for r in results
        )
        print(f"📌 Всего задач: {total_tasks}")

    finally:
        processor.cleanup()


def example_5_export_to_markdown():
    """Пример 5: Экспорт резюме в Markdown."""
    print("\n" + "="*80)
    print("ПРИМЕР 5: Экспорт в Markdown")
    print("="*80)

    from src.downloader.utils.meeting_processor import process_meeting_file

    result = process_meeting_file(
        file_path="meeting.mp3",
        save_files=False  # Не сохраняем JSON
    )

    summary = result["summary"]

    # Генерируем Markdown
    markdown = f"""# {summary.get('title', 'Резюме встречи')}

## 📅 Информация

- **Дата:** {summary.get('date', 'Не указана')}
- **Участники:** {', '.join(summary.get('participants', ['Не указаны']))}

## 📝 Краткое резюме

{summary.get('summary', '')}

## 🔑 Ключевые пункты

"""

    for i, point in enumerate(summary.get('key_points', []), 1):
        markdown += f"{i}. {point}\n\n"

    markdown += "## ✅ Принятые решения\n\n"
    for i, decision in enumerate(summary.get('decisions', []), 1):
        markdown += f"{i}. {decision}\n\n"

    markdown += "## 📌 Задачи и поручения\n\n"
    for i, item in enumerate(summary.get('action_items', []), 1):
        markdown += f"{i}. {item['task']}\n"
        if item.get('assignee'):
            markdown += f"   - **Ответственный:** {item['assignee']}\n"
        if item.get('deadline'):
            markdown += f"   - **Срок:** {item['deadline']}\n"
        markdown += "\n"

    markdown += "## ➡️ Следующие шаги\n\n"
    for i, step in enumerate(summary.get('next_steps', []), 1):
        markdown += f"{i}. {step}\n\n"

    # Сохраняем в файл
    output_file = "meeting_summary.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"✅ Резюме экспортировано в Markdown: {output_file}")


def main():
    """Главная функция с выбором примера."""
    print("\n🎙️ ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ СИСТЕМЫ ОБРАБОТКИ ВСТРЕЧ\n")

    examples = {
        "1": ("Простая обработка одного файла", example_1_simple),
        "2": ("Расширенное использование", example_2_advanced),
        "3": ("Кастомный промпт", example_3_custom_prompt),
        "4": ("Массовая обработка файлов", example_4_batch_processing),
        "5": ("Экспорт в Markdown", example_5_export_to_markdown),
    }

    print("Выберите пример для запуска:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")

    choice = input("\nВаш выбор (1-5): ").strip()

    if choice in examples:
        name, func = examples[choice]
        print(f"\n▶️ Запуск примера: {name}\n")

        # ВАЖНО: Это только демонстрационный код
        # В реальном использовании замените пути к файлам на существующие
        print("⚠️ ВНИМАНИЕ: Это демонстрационный код.")
        print("Замените пути к файлам на реальные перед запуском.\n")

        try:
            func()
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")
            print("Проверьте что:")
            print("  1. Указаны корректные пути к файлам")
            print("  2. Установлена переменная OPENROUTER_API_KEY")
            print("  3. Все зависимости установлены")

    else:
        print("❌ Некорректный выбор")


if __name__ == "__main__":
    main()
