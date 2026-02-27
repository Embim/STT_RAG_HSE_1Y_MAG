# 🚀 Быстрый старт: Обработка файлов встреч

## Шаг 1: Настройка API ключа

1. Получите API ключ на https://openrouter.ai/
2. Создайте файл `.env` в корне проекта:

```bash
# Скопируйте пример
copy .env.example .env
```

3. Откройте `.env` и добавьте ваш API ключ:

```env
OPENROUTER_API_KEY=sk-or-v1-ваш-ключ-здесь
```

**Готово!** Файл `.env` автоматически загрузится при запуске.

## Шаг 2: Выберите способ использования

### Вариант A: Web UI (самый простой) 🌐

```bash
streamlit run meeting_ui.py
```

1. Откроется браузер
2. Загрузите файл встречи (mp3, mp4, txt и т.д.)
3. Нажмите "Обработать файл"
4. Получите резюме и скачайте результаты

### Вариант B: Командная строка 💻

```bash
python process_meeting.py "path/to/meeting.mp3"
```

Результаты сохранятся рядом с исходным файлом:
- `meeting_transcript.txt` - транскрипция
- `meeting_summary.json` - резюме

### Вариант C: Из Python кода 🐍

```python
from src.downloader.utils.meeting_processor import process_meeting_file

# API ключ автоматически загрузится из .env
result = process_meeting_file("meeting.mp3")

# Получаем резюме
summary = result["summary"]
print(summary["title"])
print(summary["key_points"])
```

## Пример использования

```bash
# 1. Создаем .env файл
echo OPENROUTER_API_KEY=sk-or-v1-your-key > .env

# 2. Запускаем UI
streamlit run meeting_ui.py

# ИЛИ обрабатываем файл напрямую
python process_meeting.py "E:\ToConvect\audio1242902869.m4a"
```

## Что получите на выходе

JSON резюме встречи:
```json
{
  "title": "Название встречи",
  "summary": "Краткое резюме",
  "key_points": [
    "Развернутый абзац с описанием ключевого пункта 1...",
    "Развернутый абзац с описанием ключевого пункта 2..."
  ],
  "decisions": [
    "Развернутый абзац с описанием принятого решения..."
  ],
  "action_items": [
    {
      "task": "Описание задачи в виде абзаца...",
      "assignee": "Иван Иванов",
      "deadline": "2026-02-10"
    }
  ],
  "next_steps": [
    "Описание следующего шага в виде абзаца..."
  ]
}
```

## Поддерживаемые форматы

- **Аудио:** mp3, wav, m4a, flac, ogg
- **Видео:** mp4, avi, mkv
- **Текст:** txt (готовая транскрипция)

## Требования

- ✅ Python 3.10+ (уже установлен)
- ✅ Все зависимости из requirements.txt (уже установлены)
- ✅ CUDA GPU для быстрой транскрипции (у вас RTX 5080 ✅)
- ✅ OpenRouter API ключ (нужно получить)

## Решение проблем

### "OPENROUTER_API_KEY not found"
Проверьте что файл `.env` существует и содержит ключ:
```bash
type .env
```

### Медленная обработка
Убедитесь что используется GPU:
- Infinity и Weaviate должны быть запущены (для pipeline)
- Whisper автоматически использует CUDA если доступно

### Ошибка подключения к OpenRouter
- Проверьте что API ключ корректный
- Проверьте интернет соединение
- Убедитесь что на балансе OpenRouter есть средства

## Дополнительная информация

Полная документация: `MEETING_PROCESSOR_README.md`
Примеры кода: `example_meeting_processing.py`
