# Streamlit Interface for DS Navigator

Web interface for interacting with Audio2RAG system.

## Launch Application

### 1. Ensure dependencies are installed

```bash
pip install streamlit openai python-dotenv
```

### 2. Configure environment variables

Create `.env` file in project root:

```bash
OPENROUTER_API_KEY=your_api_key_here
```

### 3. Run Streamlit

From project root:

```bash
streamlit run src/app/ui.py
```

Or from app folder:

```bash
cd src/app
streamlit run ui.py
```

## Usage

1. Open browser (usually auto-opens at `http://localhost:8501`)
2. Enter DS/ML/AI question in input field
3. Receive answer with source citations and timestamps
4. In sidebar you can configure:
   - Number of retrieved sources (top_k)
   - Relevance threshold (similarity_threshold)
   - Clear chat history

## Features

- Chat interface with dialog history
- Answer generation via LLM with RAG
- Source display with timestamps
- Real-time RAG parameter configuration
- Contextual follow-up questions

## Status

**MVP version** - uses mock data to demonstrate functionality.

In future will integrate real vector DB with transcriptions.
