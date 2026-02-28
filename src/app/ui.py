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


# def display_sources(sources: list):
#     """
#     Display list of sources as expander.

#     Shows sources with timestamps for current answer.
#     Uses expander for compact display.

#     Args:
#         sources (list): List of sources with metadata
#             [{"name": str, "timestamp": str, "url": str (optional)}]

#     Example:
#         sources = [
#             {"name": "CS224N Lecture", "timestamp": "25:20"},
#             {"name": "PyData Talk", "timestamp": "07:00"}
#         ]
#         display_sources(sources)
#     """
#     if sources:
#         with st.expander("📚 Sources", expanded=False):
#             for idx, source in enumerate(sources, 1):
#                 st.markdown(
#                     f"**{idx}.** {source['name']} - `{source['timestamp']}`"
#                 )


def display_context(context: str):
    if not context:
        return
    with st.expander("📄 Retrieved context", expanded=False):
        st.text(context)


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

        # Ingest section with tabs
        st.markdown("---")
        st.markdown("### 📥 Add Lecture")

        tab_video, tab_playlist, tab_upload = st.tabs(["Video", "Playlist", "Upload"])

        # --- Tab 1: Single Video ---
        with tab_video:
            yt_url = st.text_input(
                "YouTube URL",
                placeholder="https://www.youtube.com/watch?v=...",
                label_visibility="collapsed",
                key="yt_single_url",
            )
            export_single = st.checkbox("Export transcript as TXT", key="export_single")
            keep_video_single = st.checkbox("Download & keep full video", key="keep_video_single")

            if st.button("Load Video", use_container_width=True, disabled=not yt_url, key="btn_single"):
                with st.spinner("⏳ Downloading and transcribing..."):
                    try:
                        resp = requests.post(
                            "http://localhost:8001/ingest",
                            json={"url": yt_url, "export_txt": export_single, "keep_video": keep_video_single},
                            timeout=600,
                        )
                        resp.raise_for_status()
                        data = resp.json()
                        st.success(f"✅ Added: {data['title']}")
                        if export_single and "transcript" in data:
                            st.session_state["last_transcript"] = (data["title"], data["transcript"])
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

            if st.session_state.get("last_transcript"):
                title, transcript = st.session_state["last_transcript"]
                st.download_button(
                    label="Download transcript (.txt)",
                    data=transcript,
                    file_name=f"{title}.txt",
                    mime="text/plain",
                    key="dl_single",
                )

        # --- Tab 2: Playlist ---
        with tab_playlist:
            pl_url = st.text_input(
                "Playlist URL",
                placeholder="https://www.youtube.com/playlist?list=...",
                label_visibility="collapsed",
                key="yt_playlist_url",
            )
            export_playlist = st.checkbox("Export transcripts as TXT", key="export_playlist")
            keep_video_playlist = st.checkbox("Download & keep full video", key="keep_video_playlist")

            if st.button("Load Playlist", use_container_width=True, disabled=not pl_url, key="btn_playlist"):
                with st.spinner("⏳ Loading playlist..."):
                    try:
                        resp = requests.post(
                            "http://localhost:8001/ingest-playlist",
                            json={"url": pl_url, "export_txt": export_playlist, "keep_video": keep_video_playlist},
                            timeout=3600,
                        )
                        resp.raise_for_status()
                        data = resp.json()
                        st.success(
                            f"✅ Ingested: {data['ingested_count']} videos"
                            + (f" | ❌ Errors: {data['error_count']}" if data["error_count"] else "")
                        )
                        for err in data.get("errors", []):
                            st.error(f"❌ {err.get('title', err.get('video_id', '?'))}: {err['error']}")
                        if export_playlist and data["items"]:
                            st.session_state["playlist_transcripts"] = [
                                (item["title"], item.get("transcript", ""))
                                for item in data["items"]
                            ]
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

            for i, (title, transcript) in enumerate(st.session_state.get("playlist_transcripts", [])):
                st.download_button(
                    label=f"⬇️ {title}.txt",
                    data=transcript,
                    file_name=f"{title}.txt",
                    mime="text/plain",
                    key=f"dl_playlist_{i}",
                )

        # --- Tab 3: Local Files ---
        with tab_upload:
            uploaded_files = st.file_uploader(
                "Upload video files",
                type=["mp4", "mkv", "avi", "mov", "webm"],
                accept_multiple_files=True,
                key="uploader",
            )
            export_upload = st.checkbox("Export transcripts as TXT", key="export_upload")

            if st.button(
                "Transcribe & Ingest",
                use_container_width=True,
                disabled=not uploaded_files,
                key="btn_upload",
            ):
                with st.spinner("⏳ Uploading and transcribing..."):
                    try:
                        files_payload = [
                            ("files", (f.name, f.getvalue(), "application/octet-stream"))
                            for f in uploaded_files
                        ]
                        resp = requests.post(
                            f"http://localhost:8001/ingest-upload?export_txt={'true' if export_upload else 'false'}",
                            files=files_payload,
                            timeout=3600,
                        )
                        resp.raise_for_status()
                        data = resp.json()
                        st.success(
                            f"✅ Ingested: {data['ingested_count']} files"
                            + (f" | ❌ Errors: {data['error_count']}" if data["error_count"] else "")
                        )
                        for err in data.get("errors", []):
                            st.error(f"❌ {err['filename']}: {err['error']}")
                        if export_upload and data["items"]:
                            st.session_state["upload_transcripts"] = [
                                (item["title"], item.get("transcript", ""))
                                for item in data["items"]
                            ]
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

            for i, (title, transcript) in enumerate(st.session_state.get("upload_transcripts", [])):
                st.download_button(
                    label=f"⬇️ {title}.txt",
                    data=transcript,
                    file_name=f"{title}.txt",
                    mime="text/plain",
                    key=f"dl_upload_{i}",
                )

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
                        "http://localhost:8001/forward",
                        json={
                            "question": prompt,
                            "top_k": top_k,
                            "similarity_threshold": similarity_threshold,
                        },
                        timeout=60,
                    )
                    resp.raise_for_status()  # Raise error for bad status
                    result = resp.json()
                    # Display answer
                    st.markdown(result["answer"])

                    # Display sources
                    if "context" in result:
                        display_context(result["context"])

                except Exception as e:
                    error_msg = f"❌ Error processing query: {str(e)}"
                    st.error(error_msg)


if __name__ == "__main__":
    main()
