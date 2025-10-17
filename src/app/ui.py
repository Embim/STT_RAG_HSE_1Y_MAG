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

# Add project root to PYTHONPATH for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.system.engine import RAGEngine


def init_session_state():
    """
    Initialize Streamlit session state.

    Creates variables for storing:
    - engine: RAG engine instance
    - messages: Message history for chat display
    - sources_history: Source history for each answer

    Called once on page load.
    """
    if "engine" not in st.session_state:
        st.session_state.engine = RAGEngine()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "sources_history" not in st.session_state:
        st.session_state.sources_history = []


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
                # TODO: Add clickable links to video with timestamps
                # if "url" in source:
                #     st.markdown(f"[🔗 Go to moment]({source['url']})")


def clear_chat():
    """
    Clear chat history and reset RAG engine.

    Used by "Clear chat" button in sidebar.
    Removes all messages and resets dialog context.
    """
    st.session_state.messages = []
    st.session_state.sources_history = []
    st.session_state.engine.clear_history()


def main():
    """
    Main Streamlit application function.

    Configures page, creates interface and processes user queries.
    Implements chat interface with history support and source display.

    Main elements:
    - Title and description
    - Sidebar with settings
    - Chat for entering questions
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
    init_session_state()

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

        # Statistics
        st.markdown("### 📈 Statistics")
        st.metric("Messages in history", len(st.session_state.messages))

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
        st.session_state.engine.top_k = top_k
        st.session_state.engine.similarity_threshold = similarity_threshold

        # Control buttons
        st.markdown("### 🎛️ Control")

        if st.button("🗑️ Clear chat", use_container_width=True):
            clear_chat()
            st.rerun()

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

    # Display message history
    for idx, message in enumerate(st.session_state.messages):
        display_message(message["role"], message["content"])

        # Display sources for assistant answers
        if message["role"] == "assistant" and idx < len(st.session_state.sources_history):
            display_sources(st.session_state.sources_history[idx // 2])

    # Question input field
    if prompt := st.chat_input("Ask question about DS, ML or AI..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_message("user", prompt)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching knowledge base..."):
                try:
                    # Query RAG engine
                    result = st.session_state.engine.query(prompt)

                    # Display answer
                    st.markdown(result["answer"])

                    # Save to history
                    st.session_state.messages.append(
                        {"role": "assistant", "content": result["answer"]}
                    )
                    st.session_state.sources_history.append(result["sources"])

                    # Display sources
                    display_sources(result["sources"])

                except Exception as e:
                    error_msg = f"❌ Error processing query: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )


if __name__ == "__main__":
    main()
