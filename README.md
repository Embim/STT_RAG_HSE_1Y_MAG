# STT_RAG_HSE_1Y_MAG
# Audio2RAG (DS Navigator)

## Состав веселых людей

* Ковынев Сергей Сергеевич	
* Чернов Петр Болеславович	
* Наумов Герман Константинович	
* Мороз Николай Сергеевич	

## Творческий руководитель 
Петр Гринберг

## А так же
ChatGPT & Claude 

## General Idea
> **Коротко:** офлайн‑инжест лекций/докладов (YouTube, конференции, подкасты) → качественная транскрибация с диаризацией и шумоподавлением → умная пост‑обработка (суммаризация, топики, ключи, Q&A) → разметка по таймкодам → индексирование в векторной БД → **RAG‑поиск** с ответами LLM + **кликабельные ссылки на источники и таймкоды** в ответах.
>
> **Ценность:** не «ещё один сервис транскрибации», а **навигатор знаний по DS‑домену** с доказательными ответами и маршрутизацией к первоисточнику.

## Quick Start (MVP)

### Setup
```bash
# Install dependencies
pip install streamlit openai python-dotenv

# Configure API key
echo "OPENROUTER_API_KEY=your_key" > .env

# Launch Streamlit interface
streamlit run src/app/ui.py
```

### Project Structure
```
src/
├── app/
│   ├── ui.py          # Streamlit web interface
│   └── README.md      # Interface documentation
├── system/
│   ├── engine.py      # RAG pipeline
│   └── llm/
│       ├── llm.py     # LLM client (OpenRouter)
│       └── prompts.py # Prompt templates
└── downloader/        # Media ingestion (TBD)
```

### Components

- **RAG Engine** (`src/system/engine.py`) - Main query processing pipeline
- **LLM Integration** (`src/system/llm/`) - OpenRouter client + prompt management
- **Web Interface** (`src/app/ui.py`) - Streamlit chat UI with source citations

### Current Status: MVP
Uses mock data for demonstration. Next steps:
- Vector DB integration (ChromaDB/FAISS)
- Real transcription data
- YouTube downloader
- Timestamp-linked video navigation
