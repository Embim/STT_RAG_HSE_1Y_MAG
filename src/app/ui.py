"""
Streamlit interface for DS Navigator (Audio2RAG).

Simple web interface for interacting with RAG system.
Allows asking questions and receiving answers with source citations and timestamps.

Launch:
    streamlit run src/app/ui.py

Example usage:
    1. Launch application
    2. Enter question in text field
    3. Receive answer with sources
    4. Navigate to timestamps (future feature)
"""

import streamlit as st
import sys
import os
import asyncio
import requests

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# from src.system.engine import RAGEngine
from system.rag.pipeline import run

# def init_session_state():
#     """
#     Initialize Streamlit session state.

#     Creates variables for storing:
#     - engine: RAG engine instance

#     Called once on page load.
#     """
#     if "engine" not in st.session_state:
#         st.session_state.engine = RAGEngine()


def display_message(role: str, content: str):
    """
    Display one message in chat.

    Args:
        role (str): Sender role ("user" or "assistant")
        content (str): Message text

    Example:
        display_message("user", "What are transformers?")
        display_message("assistant", "Transformers are an architecture...")
    """
    with st.chat_message(role):
        st.markdown(content)


def display_sources(sources: list):
    """
    Display list of sources as expander.

    Shows sources with timestamps for current answer.
    Uses expander for compact display.

    Args:
        sources (list): List of sources with metadata
            [{"name": str, "timestamp": str, "url": str (optional)}]

    Example:
        sources = [
            {"name": "CS224N Lecture", "timestamp": "25:20"},
            {"name": "PyData Talk", "timestamp": "07:00"}
        ]
        display_sources(sources)
    """
    if sources:
        with st.expander("📚 Sources", expanded=False):
            for idx, source in enumerate(sources, 1):
                st.markdown(
                    f"**{idx}.** {source['name']} - `{source['timestamp']}`"
                )


def main():
    """
    Main Streamlit application function.

    Configures page, creates interface and processes user queries.
    Implements simple query interface with source display.

    Main elements:
    - Title and description
    - Sidebar with settings
    - Input field for questions
    - Answer display with sources
    """
    # Page configuration
    st.set_page_config(
        page_title="DS Navigator",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize state
    # init_session_state()

    # Header
    st.title("🎓 DS Navigator - Audio2RAG")
    st.markdown(
        "Intelligent navigator for educational content knowledge base "
        "on Data Science, ML and AI"
    )

    # Sidebar with settings
    with st.sidebar:
        st.header("⚙️ Settings")

        # System information
        st.markdown("### 📊 System Status")
        st.info("✅ System ready")

        # RAG settings
        st.markdown("### 🔧 RAG Parameters")

        top_k = st.slider(
            "Number of sources",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of chunks to retrieve from knowledge base"
        )

        similarity_threshold = st.slider(
            "Relevance threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Minimum chunk relevance (0-1)"
        )

        # Update engine parameters
        # st.session_state.engine.top_k = top_k
        # st.session_state.engine.similarity_threshold = similarity_threshold

        # Project information
        st.markdown("---")
        st.markdown("### ℹ️ About Project")
        st.markdown(
            """
            **DS Navigator** - system for navigating
            educational content using RAG.

            **Features:**
            - Search through lecture transcriptions
            - Answers with source citations
            - Timestamps for navigating to moments

            **Status:** MVP
            """
        )

    # Question input field
    if prompt := st.chat_input("Ask question about DS, ML or AI..."):
        # Display user question
        display_message("user", prompt)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching knowledge base..."):
                try:
                    # Query RAG engine
                    # result = st.session_state.engine.query(prompt)
                    # result = run(question=prompt, top_k=top_k, similarity_threshold=similarity_threshold)
                    resp = requests.post(
                        "http://localhost:8000/forward",
                        json={
                            "question": prompt,
                            "top_k": top_k,
                            "similarity_threshold": similarity_threshold,
                        },
                        timeout=60,
                    )
                    print("Response status code:", resp.status_code)
                    resp.raise_for_status()  # Raise error for bad status
                    result = resp.json()
                    print("Result type:", type(result))
                    print("Result:", result) 
                    # Display answer
                    st.markdown(result["answer"])

                    # Display sources
                    if "sources" in result:
                        display_sources(result["sources"])

                except Exception as e:
                    error_msg = f"❌ Error processing query: {str(e)}"
                    st.error(error_msg)


if __name__ == "__main__":
    main()
