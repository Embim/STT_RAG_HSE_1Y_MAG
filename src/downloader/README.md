# InputPipline - Модуль обработки контента

Интегрированный модуль для обработки YouTube видео и файлов в основной RAG проект.

**Pipeline**: YouTube/файлы → Whisper → чанкинг → Infinity embeddings → Weaviate → RAG система

## Особенности интеграции

- Использует **Infinity API** (FRIDA модель) для эмбеддингов
- Использует **VectorStoreManager** из основного проекта
- Единая **Weaviate БД** для всех данных
- Интеграция с **Streamlit UI** для загрузки видео
- Автоматическое сохранение транскриптов

## Требования

- Python 3.10+
- CUDA GPU (рекомендуется 16GB+ VRAM для Whisper)
- Docker (для Weaviate и Infinity)
- FFmpeg

## Установка

Из корня проекта:

```bash
# Установить зависимости
pip install -r requirements.txt

# Запустить сервисы
docker-compose -f docker-compose-weaviate.yml up -d
docker-compose -f docker-compose-infinity.yml up -d

# Проверить статус
cd src/downloader
python main.py status
```

## Использование

### CLI (из src/downloader/)

```bash
# Обработка YouTube видео
python main.py process --source youtube --urls "https://youtube.com/watch?v=VIDEO_ID"

# Обработка нескольких видео
python main.py process --source youtube --urls "https://youtube.com/watch?v=ID1" "https://youtube.com/watch?v=ID2"

# Обработка локальных файлов
python main.py process --source local_files --paths "E:/documents"
python main.py process --source local_files --paths "E:/docs" --extensions ".pdf,.docx"

# Проверка статуса
python main.py status
```

### Через Streamlit UI (рекомендуется)

```bash
# Из корня проекта
streamlit run src/app/ui.py
```

В UI:
1. В sidebar найдите раздел "Загрузить YouTube видео"
2. Вставьте URL видео
3. Нажмите "Обработать видео"
4. Дождитесь завершения (2-10 минут в зависимости от длины)
5. Задавайте вопросы о содержимом через RAG

## Конфигурация

`config/config.yaml`:

```yaml
paths:
  base_dir: "E:/video_pipeline"           # Базовая директория для данных
  downloads_dir: "downloads"              # Скачанные файлы
  transcripts_dir: "transcripts"          # Транскрипции (TXT/JSON)
  logs_dir: "logs"                        # Логи

processors:
  whisper:
    enabled: true
    params:
      model_name: "openai/whisper-large-v3-turbo"
      device: cuda                        # Или "cpu" если нет GPU
      language: ru                        # Язык транскрипции
      batch_size: 16

chunking:
  min_length: 100                         # Минимальная длина чанка
  max_length: 800                         # Максимальная длина чанка

embedder:
  batch_size: 32                          # Размер батча для Infinity

store:
  upload_to_store: true                   # Автоматически загружать в Weaviate
  collection: "youtube_lectures"          # Имя коллекции

system:
  continue_on_error: true                 # Продолжать при ошибках
  gpu_cache_clear_interval: 10            # Очистка GPU кеша каждые N видео
```

### Интеграция с основным проектом

InputPipline автоматически использует:
- **Infinity API**: `http://localhost:7997` (из docker-compose-infinity.yml)
- **Weaviate**: `http://localhost:8080` (из docker-compose-weaviate.yml)
- **VectorStoreManager**: из `src/system/rag/vectore_store.py`

Адаптеры находятся в `src/downloader/adapters/`:
- `infinity_embedder.py` - использует Infinity вместо sentence-transformers
- `vector_store_adapter.py` - использует VectorStoreManager вместо прямого Weaviate клиента

## Структура модуля

```
src/downloader/
├── adapters/                          # Адаптеры для интеграции
│   ├── infinity_embedder.py          # Infinity API embedder
│   └── vector_store_adapter.py       # VectorStoreManager adapter
├── src/
│   ├── core/                         # Ядро pipeline
│   │   ├── config.py                 # Загрузка конфигурации
│   │   ├── pipeline.py               # Оркестрация обработки
│   │   ├── models.py                 # Модели данных
│   │   ├── interfaces.py             # Базовые интерфейсы
│   │   └── registry.py               # Регистрация плагинов
│   ├── sources/                      # Источники данных
│   │   ├── youtube.py                # YouTube downloader
│   │   └── local_files.py            # Локальные файлы
│   └── processors/                   # Процессоры
│       ├── audio/whisper.py          # Whisper транскрипция
│       └── text/chunker.py           # Чанкинг текста
├── config/
│   └── config.yaml                   # Конфигурация
├── main.py                           # CLI интерфейс
└── README.md                         # Документация

Интеграция с основным проектом:
- src/system/rag/vectore_store.py     # VectorStoreManager (используется через adapter)
- src/app/ui.py                       # Streamlit UI с панелью загрузки
- docker-compose-weaviate.yml         # Weaviate БД
- docker-compose-infinity.yml         # Infinity embeddings
```

## Выходные данные

### Транскрипты
- **Путь**: `E:/video_pipeline/transcripts/`
- **Форматы**:
  - `.txt` - текстовый формат с таймкодами
  - `.json` - структурированный формат с метаданными

### Логи
- **Путь**: `E:/video_pipeline/logs/pipeline.log`
- **Содержит**: статистику обработки, ошибки, warnings

### Векторная БД
- **Weaviate коллекция**: `youtube_lectures`
- **Метаданные чанков**:
  - `source_id` - ID видео
  - `title` - название видео
  - `author` - автор/канал
  - `url` - ссылка на видео
  - `start_position`, `end_position` - таймкоды в секундах
  - `chunk_id` - номер чанка

### Временные файлы
- **Путь**: `E:/video_pipeline/downloads/`
- **Автоудаление**: после успешной обработки аудио удаляется

## Архитектура интеграции

```
YouTube видео
    ↓
[YouTube Source] - скачивает видео
    ↓
[Whisper Processor] - транскрибирует аудио
    ↓
[Text Chunker] - разбивает на чанки
    ↓
[Infinity Embedder] - создает эмбеддинги через Infinity API
    ↓
[VectorStore Adapter] - сохраняет в Weaviate через VectorStoreManager
    ↓
Weaviate БД (youtube_lectures)
    ↓
[RAG Pipeline] - отвечает на вопросы с таймкодами
    ↓
Streamlit UI
```

## Расширение

См. [docs/extending-pipeline.md](docs/extending-pipeline.md) для:
- Добавления новых источников данных
- Создания кастомных процессоров
- Интеграции других embedding моделей
