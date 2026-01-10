# Расширение Video Pipeline

Руководство по добавлению новых типов данных, источников и процессоров.

---

## Архитектура

```
Источник (Source) → Процессор (Processor) → Чанкер → Эмбеддер → Хранилище
     ↓                      ↓
 ContentItem          TextSegment[]
```

**Source** — получает данные (YouTube, локальные файлы, API)
**Processor** — извлекает текст (Whisper, PDF, OCR)
**Chunker** — разбивает на части
**Embedder** — генерирует векторы
**Store** — сохраняет в БД

Плагины регистрируются автоматически через декораторы.

---

## 1. Добавление нового процессора

Примеры: OCR для изображений, извлечение текста из Excel, парсинг HTML.

### Шаг 1. Создать файл

Путь: `src/processors/<категория>/<name>.py`

Категории:
- `audio/` — аудио/видео
- `text/` — текстовые форматы
- `vision/` — документы, изображения

### Шаг 2. Структура класса

```python
from ...core.interfaces import BaseProcessor
from ...core.models import ContentItem, ContentType, TextSegment, ProcessingStage
from ...core.registry import register_processor

@register_processor
class MyProcessor(BaseProcessor):
    processor_name = "my_processor"           # Уникальное имя
    input_types = [ContentType.IMAGE]         # Какие типы принимает
    output_type = ContentType.TEXT            # Что выдаёт

    def __init__(self, param1: str = "default"):
        self.param1 = param1
        self._is_setup = False

    def setup(self, config: dict) -> None:
        if self._is_setup:
            return
        # Загрузка моделей, инициализация
        self._is_setup = True

    def teardown(self) -> None:
        # Очистка ресурсов
        self._is_setup = False

    def can_process(self, item: ContentItem) -> bool:
        return item.content_type in self.input_types

    def process(self, item: ContentItem) -> ContentItem:
        if not self._is_setup:
            self.setup({})

        # Обработка item.source_path
        # ...

        item.segments = [
            TextSegment(id=0, text="...", start=0.0, end=1.0)
        ]
        item.raw_text = "..."
        item.processing_stage = ProcessingStage.PROCESSED
        return item
```

### Шаг 3. Конфигурация

В `config/config.yaml`:

```yaml
processors:
  my_processor:
    enabled: true
    params:
      param1: "value"
```

### Шаг 4. Проверка

```bash
python main.py status
# Должен показать my_processor в списке Processors
```

---

## 2. Добавление нового источника данных

Примеры: RuTube, Telegram, Google Drive, S3.

### Шаг 1. Создать файл

Путь: `src/sources/<name>.py`

### Шаг 2. Структура класса

```python
from typing import Iterator
from pathlib import Path

from ..core.interfaces import BaseSource
from ..core.models import ContentItem, ContentType, SourceType, ProcessingStage
from ..core.registry import register_source

@register_source
class MySource(BaseSource):
    source_name = "my_source"
    supported_content_types = [ContentType.VIDEO, ContentType.AUDIO]

    def __init__(self, output_dir: str, option1: str = "default"):
        self.output_dir = Path(output_dir)
        self.option1 = option1
        self.total_count = 0  # Для progress bar

    def validate_config(self, config: dict) -> bool:
        return "urls" in config or "paths" in config

    def fetch(self, urls: list = None, **kwargs) -> Iterator[ContentItem]:
        urls = urls or []
        self.total_count = len(urls)

        for url in urls:
            # Скачивание/получение файла
            file_path = self._download(url)

            yield ContentItem(
                source_id="unique_id",
                content_type=ContentType.VIDEO,
                source_type=SourceType.LOCAL_FILE,  # или новый тип
                title="Title",
                author="Author",
                url=url,
                source_path=file_path,
                processing_stage=ProcessingStage.DOWNLOADED,
            )

    def _download(self, url: str) -> Path:
        # Логика скачивания
        pass
```

### Шаг 3. Конфигурация

```yaml
sources:
  my_source:
    enabled: true
    params:
      output_dir: "E:/pipeline/downloads"
      option1: "value"
```

### Шаг 4. CLI

```bash
python main.py process --source my_source --urls "url1,url2"
```

---

## 3. Добавление нового типа контента

Когда существующие типы не подходят (IMAGE, AUDIO, VIDEO, PDF, TEXT).

### Шаг 1. Расширить enum

В `src/core/models.py`:

```python
class ContentType(Enum):
    AUDIO = "audio"
    VIDEO = "video"
    TEXT = "text"
    PDF = "pdf"
    IMAGE = "image"
    EXCEL = "excel"      # Новый тип
    HTML = "html"        # Новый тип
```

### Шаг 2. Источник должен создавать ContentItem с новым типом

```python
yield ContentItem(
    content_type=ContentType.EXCEL,
    ...
)
```

### Шаг 3. Процессор должен принимать новый тип

```python
class ExcelProcessor(BaseProcessor):
    input_types = [ContentType.EXCEL]
```

---

## 4. Добавление нового хранилища

Примеры: Qdrant, Pinecone, ChromaDB, PostgreSQL+pgvector.

### Шаг 1. Создать файл

Путь: `src/stores/<name>.py`

### Шаг 2. Структура класса

```python
from typing import List, Dict
from ..core.interfaces import BaseStore
from ..core.models import TextChunk
from ..core.registry import register_store

@register_store
class MyStore(BaseStore):
    store_name = "my_store"

    def __init__(self, url: str, collection_name: str, **kwargs):
        self.url = url
        self.collection_name = collection_name
        self.client = None

    def connect(self) -> None:
        # Подключение к БД
        pass

    def close(self) -> None:
        # Закрытие соединения
        pass

    def add(self, chunks: List[TextChunk], metadata: Dict) -> int:
        # Сохранение чанков с эмбеддингами
        # Вернуть количество сохранённых
        return len(chunks)

    def search(self, query_embedding: List[float], limit: int) -> List[Dict]:
        # Поиск по вектору
        return []

    def count(self) -> int:
        # Количество записей
        return 0

    def delete(self, source_id: str) -> int:
        # Удаление по source_id
        return 0
```

### Шаг 3. Конфигурация

```yaml
store:
  type: my_store
  url: "http://localhost:6333"
  collection: "chunks"
```

---

## 5. Добавление нового эмбеддера

Примеры: OpenAI Embeddings, Cohere, локальная модель.

### Шаг 1. Создать файл

Путь: `src/embedders/<name>.py`

### Шаг 2. Структура класса

```python
from typing import List
from ..core.interfaces import BaseEmbedder
from ..core.registry import register_embedder

@register_embedder
class MyEmbedder(BaseEmbedder):
    embedder_name = "my_embedder"

    def __init__(self, model_name: str, api_key: str = None, **kwargs):
        self.model_name = model_name
        self.api_key = api_key
        self._is_setup = False

    def setup(self, config: dict) -> None:
        if self._is_setup:
            return
        # Инициализация клиента/модели
        self._is_setup = True

    def teardown(self) -> None:
        self._is_setup = False

    def embed(self, texts: List[str]) -> List[List[float]]:
        # Генерация эмбеддингов
        return [[0.0] * 1024 for _ in texts]

    def embed_single(self, text: str) -> List[float]:
        return self.embed([text])[0]

    def clear_cache(self) -> None:
        pass
```

### Шаг 3. Конфигурация

```yaml
embedder:
  type: my_embedder
  model: "text-embedding-3-small"
  api_key: "${OPENAI_API_KEY}"
```

---

## 6. Расширение LocalFileSource

Для поддержки новых расширений файлов без создания нового источника.

### Шаг 1. Добавить расширения в конфиг

```yaml
sources:
  local_files:
    params:
      extensions: [".pdf", ".docx", ".xlsx", ".html", ".jpg", ".png"]
```

### Шаг 2. Маппинг расширений на ContentType

В `src/sources/local_files.py` добавить в `EXTENSION_MAP`:

```python
EXTENSION_MAP = {
    ".pdf": ContentType.PDF,
    ".jpg": ContentType.IMAGE,
    ".png": ContentType.IMAGE,
    ".xlsx": ContentType.EXCEL,  # Новый тип
}
```

### Шаг 3. Создать процессор для нового типа

См. раздел 1.

---

## Чеклист интеграции

### Новый процессор
- [ ] Файл в `src/processors/<category>/<name>.py`
- [ ] Декоратор `@register_processor`
- [ ] Атрибуты: `processor_name`, `input_types`, `output_type`
- [ ] Методы: `setup()`, `teardown()`, `can_process()`, `process()`
- [ ] Секция в `config.yaml`
- [ ] Проверка: `python main.py status`

### Новый источник
- [ ] Файл в `src/sources/<name>.py`
- [ ] Декоратор `@register_source`
- [ ] Атрибуты: `source_name`, `supported_content_types`
- [ ] Методы: `validate_config()`, `fetch()`
- [ ] Секция в `config.yaml`
- [ ] Поддержка в CLI (`main.py`)

### Новый тип контента
- [ ] Значение в `ContentType` enum
- [ ] Маппинг в источнике
- [ ] Процессор с `input_types = [ContentType.NEW]`

---

## Ключевые файлы

| Файл | Назначение |
|------|------------|
| `src/core/models.py` | ContentItem, ContentType, TextSegment |
| `src/core/interfaces.py` | Базовые классы (ABC) |
| `src/core/registry.py` | Реестр плагинов, декораторы |
| `src/core/pipeline.py` | Оркестратор обработки |
| `config/config.yaml` | Конфигурация всех компонентов |

---

## Примеры реализации

| Компонент | Файл | Описание |
|-----------|------|----------|
| Процессор аудио | `src/processors/audio/whisper.py` | ML-модель, GPU, timestamps |
| Процессор PDF | `src/processors/vision/pdf.py` | PyMuPDF, постраничная сегментация |
| Источник YouTube | `src/sources/youtube.py` | yt-dlp, retry, metadata |
| Источник файлов | `src/sources/local_files.py` | Рекурсивный обход, фильтрация |
| Хранилище | `src/stores/weaviate.py` | Batch insert, search |
| Эмбеддер | `src/embedders/sentence_transformer.py` | GPU, batching |
