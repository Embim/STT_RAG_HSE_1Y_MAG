# 📝 Инструкция по настройке обработки встреч

## ✅ Что уже сделано

Вся система для обработки файлов встреч уже реализована и готова к использованию:

1. ✅ **OpenRouter клиент** - интеграция с Nvidia Nemotron 30B (БЕСПЛАТНАЯ модель!)
2. ✅ **Обработчик встреч** - автоматическая транскрипция + генерация резюме
3. ✅ **Web UI** - удобный интерфейс на Streamlit для загрузки файлов
4. ✅ **CLI** - скрипт для обработки из командной строки
5. ✅ **Программный API** - можно использовать в своем коде

## 🔑 Единственный шаг настройки: API ключ

### 1. Получите бесплатный API ключ OpenRouter

1. Перейдите на https://openrouter.ai/
2. Зарегистрируйтесь (можно через Google/GitHub)
3. Перейдите в раздел "Keys": https://openrouter.ai/settings/keys
4. Нажмите "Create Key"
5. Скопируйте ключ (начинается с `sk-or-v1-...`)

**Важно:** Используется **БЕСПЛАТНАЯ** модель `nvidia/nemotron-3-nano-30b-a3b:free`, так что никаких затрат!

### 2. Создайте файл .env

В корне проекта создайте файл `.env`:

```bash
# Windows CMD
copy .env.example .env

# PowerShell
Copy-Item .env.example .env
```

### 3. Добавьте ключ в .env

Откройте файл `.env` в любом текстовом редакторе и вставьте ваш ключ:

```env
OPENROUTER_API_KEY=sk-or-v1-ваш-ключ-здесь
```

**Готово!** Больше ничего настраивать не нужно.

## 🚀 Использование

### Способ 1: Web интерфейс (рекомендуется для UI)

```bash
streamlit run meeting_ui.py
```

Откроется браузер с интерфейсом:
1. Загрузите файл встречи (аудио/видео/текст)
2. Нажмите "Обработать файл"
3. Получите структурированное резюме
4. Скачайте результаты (транскрипцию и JSON с резюме)

### Способ 2: Командная строка

```bash
python process_meeting.py "путь/к/файлу.mp3"
```

Результат сохранится автоматически:
- `файл_transcript.txt` - полная транскрипция
- `файл_summary.json` - структурированное резюме

### Способ 3: Программный вызов из кода

```python
from src.downloader.utils.meeting_processor import process_meeting_file

# API ключ автоматически загружается из .env
result = process_meeting_file("meeting.mp3")

# Доступ к результатам
transcript = result["transcript"]
summary = result["summary"]

# Вывод резюме
print(f"Название: {summary['title']}")
for point in summary["key_points"]:
    print(f"- {point}")
```

## 📊 Что получите на выходе

Структурированное JSON резюме со следующими секциями:

- **title** - название встречи
- **date** - дата (если упомянута)
- **participants** - список участников
- **summary** - краткое резюме (2-3 предложения)
- **key_points** - ключевые пункты обсуждения (развернутые абзацы)
- **decisions** - принятые решения (развернутые абзацы)
- **action_items** - задачи с исполнителями и сроками
- **next_steps** - следующие шаги (развернутые абзацы)

**Важно:** Все пункты в формате развернутых абзацев сплошного текста, а не коротких фраз!

## 📁 Созданные файлы

```
проект/
├── .env                                    # ← СОЗДАЙТЕ ЭТОТ ФАЙЛ с API ключом
├── .env.example                            # ← Пример для создания .env
├── meeting_ui.py                           # Web UI на Streamlit
├── process_meeting.py                      # CLI скрипт
├── example_meeting_processing.py           # Примеры использования API
├── QUICKSTART_MEETING.md                   # Краткая инструкция
├── MEETING_PROCESSOR_README.md             # Полная документация
└── src/downloader/
    ├── adapters/
    │   └── openrouter_client.py           # Клиент OpenRouter API
    └── utils/
        └── meeting_processor.py           # Основная логика
```

## 🎯 Поддерживаемые форматы файлов

### Аудио
- MP3, WAV, M4A, FLAC, OGG

### Видео
- MP4, AVI, MKV

### Текст
- TXT (готовая транскрипция, пропускает этап распознавания)

## 💡 Примеры использования

### Пример 1: Обработка аудио записи встречи

```bash
python process_meeting.py "E:\ToConvect\audio1242902869.m4a"
```

### Пример 2: Запуск Web UI

```bash
streamlit run meeting_ui.py
```

### Пример 3: Обработка готовой транскрипции

```bash
python process_meeting.py "transcript.txt"
```

### Пример 4: Использование в коде

```python
from src.downloader.utils import process_meeting_file

# Обработка файла
result = process_meeting_file(
    file_path="meeting.mp3",
    output_dir="./results",  # Куда сохранить результаты
    save_files=True           # Сохранить на диск
)

# Вывод ключевых пунктов
for i, point in enumerate(result["summary"]["key_points"], 1):
    print(f"{i}. {point}\n")
```

## 🔧 Продвинутые настройки

### Изменение модели

По умолчанию используется бесплатная модель. Для других моделей:

```python
from src.downloader.adapters.openrouter_client import OpenRouterClient

client = OpenRouterClient(
    model="nvidia/llama-3.1-nemotron-70b-instruct"  # Более мощная, но платная
)
```

### Использование CPU вместо GPU (для Whisper)

```python
from src.downloader.utils.meeting_processor import MeetingProcessor

processor = MeetingProcessor(whisper_device="cpu")
```

### Кастомный промпт

```python
from src.downloader.adapters.openrouter_client import OpenRouterClient

client = OpenRouterClient()
result = client.summarize_with_custom_prompt(
    transcript="текст встречи",
    custom_prompt="Извлеки только задачи и сроки в формате JSON",
    json_mode=True
)
```

## ⚠️ Частые вопросы

### Q: Нужно ли платить за OpenRouter?
**A:** Нет! Используется бесплатная модель `nvidia/nemotron-3-nano-30b-a3b:free`. Даже регистрация на OpenRouter бесплатная.

### Q: Как узнать, что .env файл загружен?
**A:** При запуске скрипта, если ключ не найден, вы увидите ошибку `OPENROUTER_API_KEY not found`. Если такой ошибки нет - всё ОК!

### Q: Можно ли обрабатывать несколько файлов сразу?
**A:** Да! Смотрите пример в `example_meeting_processing.py` (Пример 4: Массовая обработка).

### Q: Поддерживается ли русский язык?
**A:** Да! И Whisper, и Nemotron отлично работают с русским языком.

### Q: Как долго обрабатывается файл?
**A:**
- Транскрипция: ~1 минута на 10 минут аудио (с GPU)
- Генерация резюме: ~10-30 секунд (через API)
- Итого: ~1-2 минуты на 10 минут встречи

## 📞 Что делать если не работает?

1. **Проверьте .env файл:**
   ```bash
   type .env
   ```
   Должен показать ваш API ключ.

2. **Проверьте что ключ правильный:**
   - Начинается с `sk-or-v1-`
   - Скопирован полностью без пробелов

3. **Проверьте что Weaviate и Infinity запущены** (для основного pipeline):
   ```bash
   docker ps
   ```
   Должны быть запущены контейнеры `weaviate` и `infinity`.

4. **Для обработки встреч Weaviate/Infinity НЕ нужны** - работает автономно!

## 🎉 Готово к использованию!

После создания `.env` файла с API ключом, всё готово к работе:

```bash
# Запустите Web UI
streamlit run meeting_ui.py

# ИЛИ обработайте файл напрямую
python process_meeting.py "your_meeting.mp3"
```

Удачи! 🚀
