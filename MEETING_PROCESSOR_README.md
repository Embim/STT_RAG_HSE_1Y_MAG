# Обработка файлов встреч с генерацией резюме

Система автоматической обработки аудио/видео/текстовых файлов встреч с генерацией структурированного резюме через Nvidia Nemotron 70B.

## 🎯 Возможности

- ✅ Автоматическая транскрипция аудио/видео через Whisper Large V3 Turbo
- ✅ Генерация структурированного резюме через Nvidia Nemotron 70B (OpenRouter)
- ✅ Поддержка русского языка
- ✅ GPU ускорение (CUDA)
- ✅ Web UI на Streamlit
- ✅ CLI интерфейс
- ✅ Программный API

## 📋 Требования

- Python 3.10+
- CUDA (для GPU ускорения Whisper)
- OpenRouter API ключ

## 🚀 Установка

Все зависимости уже установлены в проекте. Требуется только API ключ OpenRouter.

### Получение OpenRouter API ключа

1. Зарегистрируйтесь на https://openrouter.ai/
2. Получите API ключ в разделе "Keys"
3. Создайте файл `.env` в корне проекта:

```bash
# Скопируйте файл-пример
cp .env.example .env

# Отредактируйте .env и добавьте свой API ключ
# OPENROUTER_API_KEY=your_api_key_here
```

Содержимое `.env` файла:
```env
OPENROUTER_API_KEY=your_actual_api_key_here
```

**Важно:** Файл `.env` содержит секретные данные. Добавьте его в `.gitignore` чтобы не закоммитить в git!

## 💻 Использование

### 1. Web UI (Streamlit)

Самый простой способ - через web интерфейс:

```bash
streamlit run meeting_ui.py
```

Откроется браузер с интерфейсом для загрузки файлов:
- Загрузите аудио/видео/текстовый файл
- Нажмите "Обработать файл"
- Получите структурированное резюме
- Скачайте результаты

### 2. CLI интерфейс

Обработка через командную строку:

```bash
# Обработка аудио файла
python process_meeting.py path/to/audio.mp3

# Обработка текстового файла с транскрипцией
python process_meeting.py path/to/transcript.txt

# С указанием output директории
python process_meeting.py path/to/audio.mp3 --output-dir ./results

# Без сохранения файлов (только вывод в консоль)
python process_meeting.py path/to/audio.mp3 --no-save
```

### 3. Программный API

Использование в коде:

```python
from src.downloader.utils.meeting_processor import process_meeting_file

# Обработка файла
result = process_meeting_file(
    file_path="path/to/meeting.mp3",
    openrouter_api_key="your_api_key",  # Или None для чтения из env
    output_dir="./results",
    save_files=True
)

# Доступ к результатам
transcript = result["transcript"]  # Текст транскрипции
summary = result["summary"]        # JSON с резюме
```

### Расширенное использование

```python
from src.downloader.utils.meeting_processor import MeetingProcessor

# Создание процессора
processor = MeetingProcessor(
    openrouter_api_key="your_key",
    whisper_model="openai/whisper-large-v3-turbo",
    whisper_device="cuda",  # или "cpu"
    nemotron_model="nvidia/llama-3.1-nemotron-70b-instruct"
)

# Обработка аудио
result = processor.process_audio_file(
    file_path="meeting.mp3",
    save_transcript=True,
    save_summary=True,
    output_dir="./results"
)

# Обработка готовой транскрипции
result = processor.process_text_file(
    file_path="transcript.txt",
    save_summary=True,
    output_dir="./results"
)

# Очистка ресурсов
processor.cleanup()
```

## 📊 Формат результата

Резюме генерируется в формате JSON со следующей структурой:

```json
{
  "title": "Краткое название встречи",
  "date": "Дата встречи",
  "participants": ["Участник 1", "Участник 2"],
  "summary": "Краткое общее резюме встречи в виде абзаца",
  "key_points": [
    "Ключевой пункт 1 в виде развернутого абзаца с полным описанием",
    "Ключевой пункт 2 в виде развернутого абзаца с полным описанием"
  ],
  "decisions": [
    "Принятое решение 1 в виде развернутого абзаца с контекстом",
    "Принятое решение 2 в виде развернутого абзаца с контекстом"
  ],
  "action_items": [
    {
      "task": "Описание задачи в виде развернутого абзаца",
      "assignee": "Ответственный",
      "deadline": "Срок выполнения"
    }
  ],
  "next_steps": [
    "Следующий шаг 1 в виде развернутого абзаца",
    "Следующий шаг 2 в виде развернутого абзаца"
  ]
}
```

**Важно:** Все пункты представлены в виде **развернутых абзацев сплошного текста**, а не коротких фраз.

## 📁 Поддерживаемые форматы

### Аудио/Видео
- MP3, WAV, M4A, FLAC, OGG
- MP4, AVI, MKV

### Текст
- TXT (готовая транскрипция)

## 🤖 Используемые модели

- **Транскрипция:** OpenAI Whisper Large V3 Turbo (локально на GPU)
- **Генерация резюме:** Nvidia Llama 3.1 Nemotron 70B Instruct (через OpenRouter API)

## 💡 Примеры

### Пример 1: Быстрая обработка одного файла

```bash
python process_meeting.py meeting_recording.mp3
```

Результат:
- `meeting_recording_transcript.txt` - транскрипция
- `meeting_recording_summary.json` - резюме в JSON

### Пример 2: Web интерфейс для нескольких файлов

```bash
streamlit run meeting_ui.py
```

### Пример 3: Обработка готовой транскрипции

```python
from src.downloader.utils.meeting_processor import process_meeting_file

result = process_meeting_file("transcript.txt")
print(result["summary"]["key_points"])
```

## 🔧 Настройка

### Изменение языка транскрипции

По умолчанию используется русский язык. Для изменения:

```python
processor = MeetingProcessor()
processor.whisper.language = "en"  # Английский
```

### Использование CPU вместо GPU

```python
processor = MeetingProcessor(whisper_device="cpu")
```

### Кастомный промпт для резюме

```python
from src.downloader.adapters.openrouter_client import OpenRouterClient

client = OpenRouterClient()
result = client.summarize_with_custom_prompt(
    transcript="текст транскрипции",
    custom_prompt="Ваш кастомный промпт",
    json_mode=True
)
```

## 📝 Структура файлов

```
├── src/downloader/
│   ├── adapters/
│   │   ├── openrouter_client.py     # Клиент OpenRouter API
│   │   └── infinity_embedder.py     # (существующий)
│   └── utils/
│       └── meeting_processor.py     # Основная логика обработки
├── process_meeting.py               # CLI скрипт
├── meeting_ui.py                    # Streamlit UI
└── MEETING_PROCESSOR_README.md      # Эта инструкция
```

## ⚠️ Ограничения

- Для транскрипции аудио требуется CUDA GPU (или используйте CPU режим, будет медленнее)
- OpenRouter API - платный сервис (но дешевле прямого использования Nemotron)
- Качество резюме зависит от качества транскрипции

## 🆘 Решение проблем

### Ошибка "CUDA not available"

Используйте CPU режим:
```python
processor = MeetingProcessor(whisper_device="cpu")
```

### Ошибка "OpenRouter API key not found"

Установите переменную окружения:
```bash
set OPENROUTER_API_KEY=your_key
```

### Медленная транскрипция

- Убедитесь что используется GPU (cuda)
- Проверьте что установлена CUDA версия PyTorch
- Для больших файлов обработка может занять несколько минут

## 📞 Поддержка

При возникновении проблем проверьте:
1. API ключ OpenRouter установлен
2. CUDA доступна (для GPU режима)
3. Все зависимости установлены из requirements.txt
