# Быстрая справка RAG Pipeline

## Установка и запуск

### 1. Запуск Weaviate
```bash
# Вариант 1: Docker
docker run -d -p 8080:8080 -p 50051:50051 semitechnologies/weaviate:latest

# Вариант 2: Docker Compose
docker-compose up -d

# Проверка
curl http://localhost:8080/v1/meta
```

### 2. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 3. Индексация лекций
```bash
# Автоматический запуск (Windows)
quickstart.bat

# Автоматический запуск (Linux/Mac)
chmod +x quickstart.sh && ./quickstart.sh

# Ручной запуск
python rag_pipeline.py
```

## Основные команды

### Интерактивный поиск
```bash
python interactive_search.py
```

### Статистика базы
```bash
python stats.py
```

### Примеры использования
```bash
python examples.py
```

## Использование в коде

### Простой поиск
```python
from rag_pipeline import EmbeddingModel, WeaviateIndexer, RAGRetriever

embedding_model = EmbeddingModel()
indexer = WeaviateIndexer()
indexer.connect()

retriever = RAGRetriever(indexer, embedding_model)
results = retriever.retrieve("ваш запрос", top_k=5)
retriever.display_results(results)

indexer.close()
```

### Обработка новых лекций
```python
from rag_pipeline import LectureProcessor, TextChunker, EmbeddingModel

processor = LectureProcessor("путь/к/папке")
lectures = processor.load_lectures()

chunker = TextChunker(chunk_size=500, overlap=50)
chunks = chunker.process_lectures(lectures)

embedding_model = EmbeddingModel()
chunks_with_embeddings = embedding_model.encode_chunks(chunks)
```

### Индексация в Weaviate
```python
from rag_pipeline import WeaviateIndexer

indexer = WeaviateIndexer()
indexer.connect()
indexer.create_schema()
indexer.index_chunks(chunks_with_embeddings)
indexer.close()
```

## Настройки

### Изменение параметров chunking
В [rag_pipeline.py](rag_pipeline.py):
```python
CHUNK_SIZE = 500   # слов в чанке
OVERLAP = 50       # перекрытие в словах
```

### Изменение модели эмбеддингов
```python
model = SentenceTransformer("другая/модель")
```

### Изменение URL Weaviate
```python
indexer = WeaviateIndexer(url="http://другой-хост:8080")
```

## Частые проблемы

### Weaviate не отвечает
```bash
# Проверить статус
docker ps

# Перезапустить
docker restart weaviate
```

### Out of Memory
Уменьшите batch_size в коде:
```python
chunks_with_embeddings = embedding_model.encode_chunks(chunks, batch_size=16)
```

### Медленная работа
1. Используйте GPU (если доступен)
2. Установите flash-attention: `pip install flash-attn`
3. Увеличьте batch_size

## Полезные ссылки

- [README.md](README.md) - полная документация
- [examples.py](examples.py) - примеры использования
- [rag_pipeline.py](rag_pipeline.py) - исходный код
- Weaviate Dashboard: http://localhost:8080

## Структура данных

### JSON лекции
```json
{
  "hash": "...",
  "text": "полный текст лекции",
  "segments": [...]
}
```

### Чанк
```python
{
  "text": "...",
  "chunk_id": 0,
  "lecture_id": "lecture_0",
  "embedding": [...]  # вектор 768D
}
```

## Команды Docker

```bash
# Запуск
docker-compose up -d

# Остановка
docker-compose down

# Остановка с удалением данных
docker-compose down -v

# Логи
docker-compose logs -f

# Статус
docker-compose ps
```
