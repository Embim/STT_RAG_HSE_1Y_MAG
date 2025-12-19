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

# Add project root to PYTHONPATH for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.system.rag.pipieline import run


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
                    result = asyncio.run(run(prompt))

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
