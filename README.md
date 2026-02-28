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
> **Коротко:** офлайн‑инжест лекций/докладов (YouTube, конференции, подкасты) → качественная транскрибация с диаризацией и шумоподавлением → умная пост‑обработка (суммаризация, топики, ключи, Q&A) → разметка по таймкодам → индексирование в векторной БД → **RAG‑поиск** с ответами LLM + **кликабельные ссылки на источники и таймкоды** в ответах.
>
> **Ценность:** не «ещё один сервис транскрибации», а **навигатор знаний по DS‑домену** с доказательными ответами и маршрутизацией к первоисточнику.

## Текущий статус (что уже работает)

✅ **Рабочее:**
- RAG pipeline с переформулировкой вопросов
- Семантический поиск по векторной БД
- Генерация ответов через LLM (OpenRouter)
- Эмбеддинг модель FRIDA (локально через Infinity)
- Асинхронная архитектура

⚠️ **Временные костыли:**
- ChromaDB вместо нормальной векторки
- Только семантический поиск (гибридный в TODO)
- Простая структура данных (text + hash)
- Логирование минимальное

🚧 **В разработке:**
- YouTube downloader
- Whisper транскрибация
- Таймкоды и привязка к видео
- Нормальная БД (Weaviate/Qdrant)

---

## Быстрый старт

### 1. Установка зависимостей
```bash
# Клонируем репо
git clone 
cd STT_RAG_HSE_1Y_MAG

# 1.2. Установить uv (если не установлен)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 1.3. Создать окружение и установить зависимости
uv sync

# 1.4. Активировать окружение
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
# 1.2. Установить uv
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# После установки перезапустите терминал, чтобы uv появился в PATH

# 1.3. Создать окружение и установить зависимости
uv sync

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
docker compose --profile full up -d
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

python -m test
```

### 6. Поднимаем FastAPI - (можно проверить ручки)
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8001
```

### 7. Запускаем ui - streamlit на локал хосте
```bash
streamlit run app/ui.py
 ```

### 8. Загружаем данные в векторную БД
```bash
python -m downloader.ingest
```

Это запустит пример загрузки. Для своих данных отредактируй `src/downloader/ingest.py`:
```python
json_data = [
    {
        "hash": "unique_id_1",
        "text": "Текст вашей лекции..."
    }
]
```
---

## Архитектура проекта
```
src/
├── app/
│   └── ui.py              # Streamlit интерфейс (WIP)
├── downloader/
│   └── ingest.py          # Загрузка данных в векторную БД
├── system/
│   ├── llm/
│   │   └── llm_services.py   # Клиент OpenRouter
│   └── rag/
│       ├── pipeline.py       # Главный RAG pipeline
│       ├── embedder.py       # Локальный эмбеддер (Infinity)
│       ├── retriver.py       # Поиск по векторной БД
│       ├── answer.py         # Генерация ответов
│       ├── question_rewriter.py  # Переформулировка вопросов
│       └── vectore_store.py  # Менеджер векторной БД
├── prompts.py             # Системные промпты
├── settings.py            # Конфиг из .env
└── test.py               # Быстрый тест

data/
├── transcripts/          # Транскрибированные тексты
├── vectore_store/        # ChromaDB данные
└── infinity_data/        # Кэш эмбеддинг модели

docker-compose.yaml       # Infinity контейнер
requirements.txt          # Python зависимости
```

## TO-DO
- настроить so и stream режим
- переехать на milvus контейнер
- подключить трассировку через langfuse
- собрать датасет и ввыбрать домен
- выбрать метрики и оценить качество через эксперимент в langfuse
- улучшить поиск (сохранять версии и метрики)
- добавить логи 