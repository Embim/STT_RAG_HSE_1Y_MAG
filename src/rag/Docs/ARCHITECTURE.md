# Архитектура RAG Pipeline

## Общая схема системы

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RAG PIPELINE                                │
└─────────────────────────────────────────────────────────────────────┘

                        ┌──────────────────┐
                        │  Входные данные  │
                        │  JSON файлы      │
                        │  (40 лекций)     │
                        └────────┬─────────┘
                                 │
                                 │ load_lectures()
                                 ▼
                        ┌──────────────────┐
                        │ LectureProcessor │
                        │                  │
                        │ - hash           │
                        │ - text           │
                        │ - segments       │
                        └────────┬─────────┘
                                 │
                                 │ chunk_text()
                                 ▼
                        ┌──────────────────┐
                        │   TextChunker    │
                        │                  │
                        │ - chunk_size: 500│
                        │ - overlap: 50    │
                        └────────┬─────────┘
                                 │
                                 │ encode_chunks()
                                 ▼
                        ┌──────────────────┐
                        │ EmbeddingModel   │
                        │                  │
                        │ Qwen3-Embedding  │
                        │ 768 dimensions   │
                        └────────┬─────────┘
                                 │
                                 │ index_chunks()
                                 ▼
                        ┌──────────────────┐
                        │ WeaviateIndexer  │
                        │                  │
                        │ LectureChunks    │
                        │ Collection       │
                        └────────┬─────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
         ┌──────────────────┐      ┌──────────────────┐
         │  RAGRetriever    │      │    Weaviate      │
         │                  │◄─────┤   Vector DB      │
         │ retrieve()       │      │  localhost:8080  │
         └──────────────────┘      └──────────────────┘
                    │
                    │
                    ▼
         ┌──────────────────┐
         │  Search Results  │
         │  Top-K chunks    │
         └──────────────────┘
```

## Поток данных

### 1. Загрузка и обработка

```
transcripted_text/
├── lecture_0.json ─┐
├── lecture_1.json ─┤
├── ...            ─┤──► LectureProcessor ──► List[Dict]
├── lecture_38.json ─┤                         [
├── lecture_39.json ─┘                           {
                                                    "hash": "...",
                                                    "text": "...",
                                                    "lecture_id": "..."
                                                  }
                                                ]
```

### 2. Chunking

```
Исходный текст (50,000 слов)
         │
         ▼
┌────────────────────┐
│   TextChunker      │
│                    │
│  chunk_size = 500  │
│  overlap = 50      │
└─────────┬──────────┘
          │
          ▼
    Chunk 0: [0:500]
    Chunk 1: [450:950]   ◄─── overlap 50 слов
    Chunk 2: [900:1400]
    ...
    Chunk N: [49500:50000]

Итого: ~110 чанков на лекцию
```

### 3. Векторизация

```
Текстовый чанк:
"Чем мы будем с вами заниматься? Мне нравится приводить..."
                    │
                    ▼
         ┌──────────────────┐
         │ SentenceTransformer
         │ Qwen3-Embedding  │
         └─────────┬────────┘
                   │
                   ▼
        Embedding вектор:
        [0.123, -0.456, 0.789, ..., 0.234]
                768 чисел
```

### 4. Индексация в Weaviate

```
        Чанк с эмбеддингом
                │
                ▼
     ┌──────────────────────┐
     │  Weaviate Object     │
     ├──────────────────────┤
     │ Properties:          │
     │  - text              │
     │  - lecture_id        │
     │  - chunk_id          │
     │  - word_start        │
     │  - word_end          │
     │                      │
     │ Vector:              │
     │  [768 dimensions]    │
     └──────────────────────┘
                │
                ▼
        Сохранено в БД
```

### 5. Поиск

```
Запрос пользователя:
"Что такое машинное обучение?"
        │
        ▼
┌───────────────────┐
│ Векторизация     │
│ (с prompt="query")│
└────────┬──────────┘
         │
         ▼
  Query vector [768D]
         │
         ▼
┌────────────────────┐
│ Weaviate           │
│ near_vector()      │
│ Cosine similarity  │
└────────┬───────────┘
         │
         ▼
Top-K похожих чанков
[
  {similarity: 0.89, text: "..."},
  {similarity: 0.85, text: "..."},
  ...
]
```

## Классы и их ответственности

### LectureProcessor
```python
class LectureProcessor:
    """
    Ответственность:
    - Поиск JSON файлов
    - Чтение и парсинг
    - Валидация структуры
    - Извлечение метаданных
    """

    load_lectures() → List[Dict]
```

### TextChunker
```python
class TextChunker:
    """
    Ответственность:
    - Разбиение текста на слова
    - Создание чанков с overlap
    - Добавление метаданных позиций
    - Обработка batch лекций
    """

    chunk_text(text, metadata) → List[Dict]
    process_lectures(lectures) → List[Dict]
```

### EmbeddingModel
```python
class EmbeddingModel:
    """
    Ответственность:
    - Загрузка модели Qwen
    - Batch векторизация
    - Конвертация в numpy
    - Progress tracking
    """

    encode_chunks(chunks, batch_size) → List[Dict]
```

### WeaviateIndexer
```python
class WeaviateIndexer:
    """
    Ответственность:
    - Подключение к Weaviate
    - Создание/удаление schema
    - Batch индексация
    - Управление коллекциями
    """

    connect()
    create_schema(vector_dim)
    index_chunks(chunks, batch_size)
    close()
```

### RAGRetriever
```python
class RAGRetriever:
    """
    Ответственность:
    - Векторизация запросов
    - Поиск по векторам
    - Форматирование результатов
    - Отображение результатов
    """

    retrieve(query, top_k) → List[Dict]
    display_results(results)
```

## Схема базы данных Weaviate

```
Collection: LectureChunks
├── Properties:
│   ├── text (text)           - текст чанка
│   ├── lecture_id (text)     - ID лекции
│   ├── lecture_hash (text)   - хеш лекции
│   ├── filename (text)       - имя файла
│   ├── chunk_id (int)        - номер чанка
│   ├── word_start (int)      - начальная позиция
│   ├── word_end (int)        - конечная позиция
│   └── total_words (int)     - всего слов в лекции
│
└── Vector:
    └── embedding (768D float) - вектор эмбеддинга
```

## Индексы и поиск

### Векторный индекс (HNSW)
```
Weaviate использует HNSW (Hierarchical Navigable Small World)
для быстрого приближенного поиска ближайших соседей

         Layer 2:  ○────○
                   │    │
         Layer 1:  ○─○──○─○
                   │ │  │ │
         Layer 0:  ○─○─○○─○─○─○─○

Сложность: O(log N) вместо O(N)
```

### Процесс поиска
```
1. Запрос → Вектор
2. HNSW навигация по графу
3. Вычисление cosine similarity
4. Ранжирование результатов
5. Возврат Top-K
```

## Метрики и производительность

### Chunking
- Входные данные: 40 лекций × ~50,000 слов = 2,000,000 слов
- Выходные данные: ~4,400 чанков
- Скорость: ~200,000 слов/сек

### Векторизация
- CPU: ~100 чанков/сек
- GPU: ~500 чанков/сек
- Batch size: 32 (оптимально)

### Индексация
- Скорость: ~500 объектов/сек
- Размер индекса: ~15 MB для 4,400 чанков

### Поиск
- Латентность: <100ms для Top-10
- Throughput: >100 запросов/сек

## Масштабируемость

### Текущий размер
```
40 лекций → 4,400 чанков → 15 MB индекс
```

### Проекция на 1000 лекций
```
1,000 лекций → 110,000 чанков → 375 MB индекс
Время индексации: ~3-5 минут
Поиск: по-прежнему <100ms
```

### Ограничения
- RAM: 8 GB достаточно до 100,000 чанков
- Disk: ~1 GB на 100,000 чанков
- CPU: bottleneck на векторизации
- GPU: решает проблему векторизации

## Точки расширения

### 1. Pre-processing
```
JSON → [Cleaning] → [Tokenization] → Chunks
         ↑
    Можно добавить:
    - Удаление stopwords
    - Стемминг/лемматизация
    - Нормализация текста
```

### 2. Chunking strategies
```
Current: Fixed window

Alternatives:
- Semantic chunking (по смыслу)
- Sentence-based (по предложениям)
- Paragraph-based (по параграфам)
- Sliding window with semantic boundaries
```

### 3. Post-processing
```
Search Results → [Re-ranking] → [Filtering] → Final Results
                      ↑
                 Можно добавить:
                 - Cross-encoder re-ranking
                 - Diversity filtering
                 - Metadata filtering
```

## Диаграмма зависимостей

```
rag_pipeline.py
    │
    ├── sentence_transformers
    │   └── torch
    │       └── numpy
    │
    ├── weaviate-client
    │   └── grpc
    │
    └── tqdm

interactive_search.py
    └── rag_pipeline

stats.py
    └── weaviate-client

examples.py
    ├── rag_pipeline
    └── scikit-learn
```

## Deployment опции

### 1. Local Development
```
Python + Weaviate (Docker)
├── Простая настройка
├── Быстрая разработка
└── Ограниченная масштабируемость
```

### 2. Production (Single Server)
```
VM/Server
├── Docker Compose
├── Nginx reverse proxy
├── Monitoring (Prometheus)
└── Backup scripts
```

### 3. Production (Distributed)
```
Kubernetes Cluster
├── Weaviate StatefulSet
├── Python API Deployment
├── Load Balancer
└── Distributed monitoring
```
