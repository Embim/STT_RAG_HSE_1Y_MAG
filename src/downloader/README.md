# Video Pipeline

RAG-pipeline: YouTube/файлы → Whisper → embeddings → Weaviate → семантический поиск.

## Требования

- Python 3.10+
- CUDA GPU (16GB+ VRAM)
- Docker
- FFmpeg

## Установка

```bash
pip install -r requirements.txt
docker compose up -d
python main.py status
```

## CLI

```bash
# Обработка YouTube
python main.py process --source youtube --urls "https://youtube.com/watch?v=ID"
python main.py process --source youtube --urls "https://youtube.com/@channel/videos"

# Обработка файлов
python main.py process --source local_files --paths "E:/documents"
python main.py process --source local_files --paths "E:/docs" --extensions ".pdf,.docx"

# Поиск
python main.py search "запрос" --limit 10

# API сервер
python main.py serve --port 8000
```

## API

```
GET /search?q=текст&limit=10    Семантический поиск
GET /status                      Статус системы
GET /docs                        Swagger
```

## Конфигурация

`config/config.yaml`:

```yaml
paths:
  base_dir: "E:/video_pipeline"

processors:
  whisper:
    params:
      model_name: "openai/whisper-large-v3-turbo"
      device: cuda
      language: ru

embedder:
  model: "ai-forever/sbert_large_nlu_ru"
  device: cuda

store:
  url: "http://localhost:8080"
  collection: "VideoChunk"
```

## Структура

```
src/
├── core/           # Pipeline, модели, интерфейсы
├── sources/        # youtube.py, local_files.py
├── processors/     # audio/whisper.py, text/chunker.py
├── embedders/      # sentence_transformer.py
├── stores/         # weaviate.py
└── api/            # FastAPI сервер
```

## Расширение

См. [docs/extending-pipeline.md](docs/extending-pipeline.md) — добавление новых источников, процессоров, типов данных.

## Выходные данные

- `transcripts/` — TXT и JSON транскрипции
- `processed.json` — журнал обработанных
- `logs/pipeline.log` — логи
- Weaviate — векторная БД для поиска
