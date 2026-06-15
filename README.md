# STT_RAG_HSE_1Y_MAG
# Audio2RAG (DS Navigator)

## Состав веселых людей

* Ковынев Сергей Сергеевич	
* Чернов Петр Болеславович	
* Наумов Герман Константинович	
* Мороз Николай Сергеевич	

## Творческий руководитель 
Петр Гринберг

## General Idea
> **Коротко:** офлайн‑инжест лекций/докладов (YouTube, конференции, подкасты) → качественная транскрибация с диаризацией и шумоподавлением + **извлечение текста с видео (OCR)** → умная пост‑обработка (суммаризация, топики, ключи, Q&A) → разметка по таймкодам → индексирование в векторной БД → **RAG‑поиск** с ответами LLM + **кликабельные ссылки на источники и таймкоды** в ответах.
>
> **Ценность:** не «ещё один сервис транскрибации», а **навигатор знаний по DS‑домену** с доказательными ответами и маршрутизацией к первоисточнику.

## Текущий статус (что уже работает)

✅ **Рабочее:**
- RAG pipeline с переформулировкой вопросов
- Семантический поиск по векторной БД (текст из аудио + текст на экране)
- Распознавание текста с видео (OCR через EasyOCR)
- Генерация ответов через LLM (OpenRouter)
- Эмбеддинг модель FRIDA (локально через Infinity)
- Асинхронная архитектура

⚠️ **Временные костыли:**
- Только семантический поиск (гибридный в TODO)
- Логирование минимальное

🚧 **В разработке:**
- Расширение ASR/RAG eval (см. [src/evaluation/README.md](src/evaluation/README.md))

---

## Где живут эксперименты и raw данные

Этот репо — **только код + финальный (v15) judge-промпт + финальный
sweep**. Всё что exploratory — 15 версий промпта, 56 директорий
отчётов, 91 CSV/JSONL прогона, 1 GB локальных ASR-бенчмарков, 2.4 GB
исходного audio/video, анонимизированные записи друзей — лежит в
**сестринском репо** [`../STT_RAG_HSE_1Y_MAG_DATA`](../STT_RAG_HSE_1Y_MAG_DATA/).

Главное чтиво из архива — [`docs/judge_evolution.md`](../STT_RAG_HSE_1Y_MAG_DATA/docs/judge_evolution.md):
narrative эволюции judge-промпта v1→v15 с 5 ключевыми уроками
(server-side fix > prompt engineering, ловушки strict json_schema
с reasoning, и т.п.).

---

## Быстрый старт

### 1. Установка зависимостей

Проект полностью на `uv` — он сам создаёт `.venv/`, тянет нужную версию Python (>=3.12) и ставит зависимости из `pyproject.toml`/`uv.lock`. Команды одинаковые на Linux/macOS и на Windows.

```bash
# Клонируем репо
git clone <repo-url>
cd STT_RAG_HSE_1Y_MAG

# Установить uv (если ещё не стоит)
# Linux / macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell):
#   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Создать окружение и поставить зависимости
uv sync

# Опционально: eval-харнес (ragas, jiwer, datasets, langchain-openai, ...)
uv sync --extra eval
```

**Windows (PowerShell):**
```powershell
# 1.2. Установить uv
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# После установки перезапустите терминал, чтобы uv появился в PATH

# 1.3. Создать окружение и установить зависимости (Python 3.12/3.13)
uv sync --python 3.13

# 1.4. Активировать окружение
.venv\Scripts\activate
```

### 2. Настройка окружения

Создай `.env` файл в корне проекта:
```env
# LLM (OpenRouter)
LLM_API_KEY_3=your_openrouter_key
LLM_MODEL=openai/gpt-4o-mini
LLM_BASE_URL=https://openrouter.ai/api/v1

# Прочие настройки
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=2000
```

### 3. Поднимаем docker с вбд, эмбедингами и виспером (можно выборочно через профайлы)
```bash
docker compose --profile full --profile langfuse up -d
```

Проверяем, что работает:
```bash
curl http://localhost:7997/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input":"тест"}'
```

### 4. Тестируем RAG - смотрим что модель отвечает (секунд 7)
```bash
cd src
uv run python -m test
```

### 6. Поднимаем FastAPI - (можно проверить ручки)
```bash
uv run uvicorn api.main:app --host 0.0.0.0 --port 8001
```

### 7. Запускаем ui - streamlit на локал хосте
```bash
uv run streamlit run app/ui.py
```

### 8. Загружаем данные в векторную БД
```bash
uv run python -m downloader.ingest                              # пример из main()
uv run python -m downloader.ingest --from-dir data/transcripts/recsys   # любая папка с .txt
```

Без флагов запустится пример из `main()`. Для произвольной папки с `.txt` используй `--from-dir <path>`. Для произвольной структуры данных можно отредактировать `src/downloader/ingest.py`:
```python
json_data = [
    {
        "hash": "unique_id_1",
        "text": "Текст вашей лекции..."
    }
]
```

### 9. Eval (опционально)

Подробности про оценку качества ASR и RAG — в [src/evaluation/README.md](src/evaluation/README.md). Кратко:
```bash
uv sync --extra eval
uv run python -m evaluation.asr.runner --max-samples 50 --langfuse-dataset asr-cv-ru
uv run python -m evaluation.rag.runner --testset data/eval/testsets/recsys_v1.json \
    --run-name baseline --langfuse-dataset stt-rag-recsys-v1
```

---

## Архитектура проекта
```
src/
├── app/
│   └── ui.py                # Streamlit интерфейс
├── api/
│   └── main.py              # FastAPI endpoints
├── downloader/
│   ├── ingest.py            # Загрузка данных в Weaviate (--from-dir для папки .txt)
│   ├── transcriber.py       # Тонкий shim над HttpASRBackend
│   ├── processor.py         # YouTube/upload → транскрибация → ингест
│   └── ocr.py               # Распознавание визуального текста (EasyOCR)
├── system/
│   ├── llm/
│   │   └── llm_services.py  # OpenRouter клиент + Langfuse generation
│   ├── rag/
│   │   ├── pipeline.py      # Главный RAG pipeline (rewrite→retrieve→answer)
│   │   ├── embedder.py      # Локальный эмбеддер (Infinity, FRIDA)
│   │   ├── retriver.py      # Поиск в Weaviate (возвращает текст + Document'ы)
│   │   ├── answer.py        # Генерация ответа
│   │   ├── question_rewriter.py  # Переформулировка вопроса
│   │   └── vectore_store.py # Менеджер Weaviate (поддержка кастомных коллекций)
│   ├── prompts.py           # Системные промпты
│   ├── tracing.py           # Langfuse @observe + noop-fallback
│   └── exceptions.py
├── evaluation/              # ASR + RAG eval-харнес (см. свой README)
│   ├── asr/                 # WER/CER, бенчмарки, runner
│   ├── rag/                 # RAGAS метрики, тестсеты, runner
│   └── reporting/           # CSV + Langfuse Datasets/Scores
├── data/
│   └── eval/                # Только финальный v15 sweep (CSV/JSONL/HTML).
│                            # Старые версии (v1-v14), бенчмарки, raw audio/video,
│                            # все 56 outputs и 91 results — в archive репо
│                            # `STT_RAG_HSE_1Y_MAG_DATA` (sister directory).
├── settings.py              # Конфиг из .env
└── test.py                  # Быстрый тест RAG

scripts/
└── build_asr_benchmark.py   # Сборка локального ASR-бенчмарка (one-off)

docker-compose.yml           # Weaviate + Infinity + faster-whisper + Langfuse
pyproject.toml               # Зависимости (uv)
```

## TO-DO
- гибридный поиск (BM25 + dense)
- больше ASR-моделей в eval (Qwen3-ASR FastAPI-обёртка)
- расширить тестсет за пределы 3 RecSys-лекций
- логирование как у взрослых 