# RAG Pipeline для Лекций

Полнофункциональный пайплайн для обработки транскрибированных лекций с использованием RAG (Retrieval-Augmented Generation).

## Описание

Пайплайн выполняет следующие шаги:

1. **Загрузка данных** - чтение JSON файлов из папки `transcripted_text`
2. **Chunking** - разбиение текста на чанки (фрагменты) с перекрытием
3. **Векторизация** - преобразование текста в векторы с помощью модели `Qwen/Qwen3-Embedding-0.6B`
4. **Индексация** - сохранение векторов в Weaviate (векторная база данных)
5. **Retrieval** - поиск релевантных фрагментов по запросу

## Структура проекта

```
.
├── transcripted_text/        # Папка с JSON файлами лекций
│   ├── lecture_0.json
│   ├── lecture_1.json
│   └── ...
├── rag_pipeline.py           # Основной пайплайн
├── interactive_search.py     # Интерактивный поиск
├── stats.py                  # Статистика по базе
├── RAG.py                    # Пример работы с эмбеддингами
├── requirements.txt          # Зависимости
├── docker-compose.yml        # Конфигурация Docker Compose
├── quickstart.bat            # Быстрый старт (Windows)
├── quickstart.sh             # Быстрый старт (Linux/Mac)
└── README.md                 # Этот файл
```

## Установка

### 1. Установка Python зависимостей

```bash
pip install -r requirements.txt
```

### 2. Установка и запуск Weaviate

Weaviate - это векторная база данных для хранения эмбеддингов.

**Вариант 1: Docker (рекомендуется)**

```bash
docker run -d \
  -p 8080:8080 \
  -p 50051:50051 \
  --name weaviate \
  semitechnologies/weaviate:latest
```

**Вариант 2: Docker Compose**

Создайте файл `docker-compose.yml`:

```yaml
version: '3.4'
services:
  weaviate:
    image: semitechnologies/weaviate:latest
    ports:
      - "8080:8080"
      - "50051:50051"
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'none'
      ENABLE_MODULES: ''
      CLUSTER_HOSTNAME: 'node1'
```

Затем запустите:

```bash
docker-compose up -d
```

### 3. Проверка установки Weaviate

Откройте в браузере: http://localhost:8080/v1/meta

Вы должны увидеть JSON с информацией о Weaviate.

## Использование

### Быстрый старт

Для быстрого запуска используйте готовые скрипты:

**Windows:**
```bash
quickstart.bat
```

**Linux/Mac:**
```bash
chmod +x quickstart.sh
./quickstart.sh
```

Эти скрипты автоматически:
- Проверят, что Weaviate запущен
- Установят необходимые зависимости
- Запустят индексацию

### Запуск полного пайплайна

Основной скрипт обрабатывает все лекции и индексирует их в Weaviate:

```bash
python rag_pipeline.py
```

### Интерактивный поиск

После индексации можно использовать интерактивный поиск:

```bash
python interactive_search.py
```

Пример работы:
```
🔍 Ваш запрос: Что такое машинное обучение?

Поиск...

================================================================================
Найдено 5 релевантных фрагментов:
================================================================================

#1 | Сходство: 0.8765 | lecture_0.json (chunk 2)
--------------------------------------------------------------------------------
Чем мы будем с вами заниматься? Мне нравится приводить такой пример, который
объясняет вообще суть машинного обучения...
```

### Статистика по базе

Для просмотра статистики по проиндексированным лекциям:

```bash
python stats.py
```

Выводит:
- Общее количество чанков
- Количество уникальных лекций
- Топ лекций по количеству чанков
- Статистику по размерам чанков

### Этапы работы пайплайна

1. **Загрузка лекций** - загружаются все файлы `lecture_*.json` из папки `transcripted_text`
2. **Разбиение на чанки** - каждая лекция разбивается на фрагменты по 500 слов с перекрытием в 50 слов
3. **Векторизация** - каждый чанк преобразуется в вектор размерностью 768 (для модели Qwen3-Embedding-0.6B)
4. **Индексация** - векторы загружаются в Weaviate
5. **Демонстрация поиска** - выполняются тестовые запросы

### Примеры запросов

После запуска пайплайна автоматически выполняются тестовые запросы:

- "Что такое машинное обучение?"
- "Какие бывают типы признаков?"
- "Расскажи про регрессию"

## Настройка параметров

В файле [rag_pipeline.py](rag_pipeline.py) можно изменить следующие параметры:

```python
# Путь к папке с транскрипциями
TRANSCRIPTS_FOLDER = r"c:\Users\petrc\OneDrive\Мага\Годовой проект\transcripted_text"

# Размер чанка в словах
CHUNK_SIZE = 500

# Перекрытие между чанками в словах
OVERLAP = 50

# URL для подключения к Weaviate
WEAVIATE_URL = "http://localhost:8080"
```

## Использование в своем коде

### Пример 1: Простой поиск

```python
from rag_pipeline import EmbeddingModel, WeaviateIndexer, RAGRetriever

# Инициализация
embedding_model = EmbeddingModel()
indexer = WeaviateIndexer(url="http://localhost:8080")
indexer.connect()

retriever = RAGRetriever(indexer, embedding_model)

# Поиск
results = retriever.retrieve("Что такое нейронные сети?", top_k=5)
retriever.display_results(results)

# Закрытие соединения
indexer.close()
```

### Пример 2: Обработка новых лекций

```python
from rag_pipeline import LectureProcessor, TextChunker, EmbeddingModel, WeaviateIndexer

# Загрузка
processor = LectureProcessor("путь/к/папке")
lectures = processor.load_lectures()

# Разбиение на чанки
chunker = TextChunker(chunk_size=500, overlap=50)
chunks = chunker.process_lectures(lectures)

# Векторизация
embedding_model = EmbeddingModel()
chunks_with_embeddings = embedding_model.encode_chunks(chunks)

# Индексация
indexer = WeaviateIndexer()
indexer.connect()
indexer.create_schema()
indexer.index_chunks(chunks_with_embeddings)
indexer.close()
```

## Архитектура

### Классы

1. **LectureProcessor** - загрузка лекций из JSON файлов
2. **TextChunker** - разбиение текста на чанки с перекрытием
3. **EmbeddingModel** - векторизация текста через `sentence-transformers`
4. **WeaviateIndexer** - работа с Weaviate (создание схемы, индексация)
5. **RAGRetriever** - поиск релевантных чанков

### Формат данных

**Входной JSON (lecture_N.json):**
```json
{
  "hash": "QVs8QjuAN74",
  "text": "Полный текст лекции...",
  "segments": [...]
}
```

**Чанк (внутреннее представление):**
```python
{
  "text": "Текст фрагмента...",
  "chunk_id": 0,
  "lecture_id": "lecture_0",
  "lecture_hash": "QVs8QjuAN74",
  "filename": "lecture_0.json",
  "word_start": 0,
  "word_end": 500,
  "total_words": 5000,
  "embedding": [0.123, -0.456, ...]  # вектор размерности 768
}
```

## Модель эмбеддингов

Используется модель **Qwen/Qwen3-Embedding-0.6B**:

- Размерность векторов: 768
- Поддержка русского языка
- Специальный промпт для запросов (`prompt_name="query"`)

### Оптимизация производительности

Для ускорения работы модели можно использовать flash attention:

```python
model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-0.6B",
    model_kwargs={
        "attn_implementation": "flash_attention_2",
        "device_map": "auto"
    },
    tokenizer_kwargs={"padding_side": "left"}
)
```

Для этого нужно установить:
```bash
pip install flash-attn
```

## Troubleshooting

### Ошибка подключения к Weaviate

```
Ошибка подключения к Weaviate: ...
```

**Решение:**
- Убедитесь, что Weaviate запущен: `docker ps`
- Проверьте доступность: http://localhost:8080/v1/meta

### Out of Memory при векторизации

```
RuntimeError: CUDA out of memory
```

**Решение:**
- Уменьшите `batch_size` в `encode_chunks()`
- Используйте CPU вместо GPU: установите `device='cpu'` в SentenceTransformer

### Медленная работа

**Решение:**
- Используйте GPU если доступен
- Установите flash-attention (см. выше)
- Увеличьте `batch_size` для векторизации

## Дополнительные возможности

### Изменение стратегии chunking

Можно использовать различные стратегии разбиения:

1. **По предложениям** - использовать NLTK или spaCy для разбиения по предложениям
2. **По семантике** - использовать semantic chunking
3. **По fixed tokens** - разбиение по токенам вместо слов

### Добавление метаданных

В схему Weaviate можно добавить дополнительные поля:

```python
{"name": "speaker", "dataType": ["text"]},
{"name": "timestamp", "dataType": ["date"]},
{"name": "topic", "dataType": ["text"]},
```

### Гибридный поиск

Weaviate поддерживает гибридный поиск (векторный + BM25):

```python
results = collection.query.hybrid(
    query="запрос",
    limit=5,
    alpha=0.75  # баланс между векторным (1.0) и keyword (0.0)
)
```

## Лицензия

MIT License

## Контакты

Для вопросов и предложений создайте issue в репозитории.
