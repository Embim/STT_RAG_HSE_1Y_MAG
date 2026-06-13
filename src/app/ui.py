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
import time
import requests

_NO_PROXY = {"http": None, "https": None}

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from system.rag.pipeline import run


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


def display_context(context: str):
    if not context:
        return
    with st.expander("📄 Retrieved context", expanded=False):
        st.text(context)


def load_source_files() -> list[str]:
    resp = requests.get(
        "http://localhost:8001/source-files",
        timeout=30,
        proxies=_NO_PROXY,
    )
    resp.raise_for_status()
    return resp.json().get("files", [])


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
            value=0.5,
            step=0.05,
            help="Minimum chunk relevance (0-1)"
        )
        use_rewrite = st.checkbox(
            "Rewrite question before retrieval",
            value=True,
            help="Reduces noisy matches by reformulating short/ambiguous queries",
        )

        st.markdown("### 📁 Search Scope")
        if "source_files" not in st.session_state:
            st.session_state["source_files"] = []
        if st.button("Refresh file list", use_container_width=True, key="refresh_source_files"):
            try:
                st.session_state["source_files"] = load_source_files()
            except Exception as e:
                st.warning(f"Failed to refresh file list: {e}")
        if not st.session_state["source_files"]:
            try:
                st.session_state["source_files"] = load_source_files()
            except Exception:
                pass
        selected_source_file = st.selectbox(
            "Search in video",
            options=["All files"] + st.session_state["source_files"],
            help="Limit retrieval to a single video title",
        )

        # Ingest section with tabs
        st.markdown("---")
        st.markdown("### 📥 Add Lecture")

        tab_youtube, tab_upload = st.tabs(["YouTube", "Upload"])

        # --- Tab 1: YouTube (video or playlist) ---
        with tab_youtube:
            yt_url = st.text_input(
                "YouTube URL",
                placeholder="https://www.youtube.com/watch?v=... or playlist?list=...",
                label_visibility="collapsed",
                key="yt_url",
            )
            export_yt = st.checkbox("Export transcript as TXT", key="export_yt")
            export_yt_json = st.checkbox("Export transcript as JSON (с сегментами)", key="export_yt_json")
            keep_audio_yt = st.checkbox("Save extracted audio", key="keep_audio_yt")
            use_ocr_yt = st.checkbox("Extract text from video (OCR)", key="use_ocr_yt")

            if st.button("Load", use_container_width=True, disabled=not yt_url, key="btn_yt"):
                with st.spinner("⏳ Downloading and transcribing..."):
                    try:
                        resp = requests.post(
                            "http://localhost:8001/ingest",
                            json={
                                "url": yt_url,
                                "export_txt": export_yt,
                                "export_json": export_yt_json,
                                "keep_audio": keep_audio_yt,
                                "use_ocr": use_ocr_yt,
                            },
                            timeout=3600,
                            proxies=_NO_PROXY,
                        )
                        resp.raise_for_status()
                        data = resp.json()
                        st.success(
                            f"✅ Ingested: {data['ingested_count']} video(s)"
                            + (f" | ❌ Errors: {data['error_count']}" if data["error_count"] else "")
                        )
                        for err in data.get("errors", []):
                            st.error(f"❌ {err['url']}: {err['error']}")
                        if export_yt and data["items"]:
                            st.session_state["yt_transcripts"] = [
                                (item["title"], item.get("transcript", ""))
                                for item in data["items"]
                            ]
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

            for i, (title, transcript) in enumerate(st.session_state.get("yt_transcripts", [])):
                st.download_button(
                    label=f"⬇️ {title}.txt",
                    data=transcript,
                    file_name=f"{title}.txt",
                    mime="text/plain",
                    key=f"dl_yt_{i}",
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
            export_upload_json = st.checkbox("Export transcripts as JSON (с сегментами)", key="export_upload_json")
            keep_audio_upload = st.checkbox("Save extracted audio", key="keep_audio_upload")
            use_ocr_upload = st.checkbox("Extract text from video (OCR)", key="use_ocr_upload")

            if st.button(
                "Transcribe & Ingest",
                use_container_width=True,
                disabled=not uploaded_files,
                key="btn_upload",
            ):
                total_files = len(uploaded_files)
                progress_bar = st.progress(0.0, text=f"⏳ Starting upload: 0/{total_files}")
                status_box = st.empty()
                started_at = time.time()

                all_items = []
                all_errors = []
                ingested_total = 0

                for index, upload_file in enumerate(uploaded_files, start=1):
                    status_box.info(f"Processing {index}/{total_files}: {upload_file.name}")
                    try:
                        files_payload = [
                            ("files", (upload_file.name, upload_file.getvalue(), "application/octet-stream"))
                        ]
                        params = {
                            "export_txt": "true" if export_upload else "false",
                            "export_json": "true" if export_upload_json else "false",
                            "keep_audio": "true" if keep_audio_upload else "false",
                            "use_ocr": "true" if use_ocr_upload else "false",
                        }
                        resp = requests.post(
                            "http://localhost:8001/ingest-upload",
                            params=params,
                            files=files_payload,
                            timeout=3600,
                            proxies=_NO_PROXY,
                        )
                        resp.raise_for_status()
                        data = resp.json()
                        ingested_total += data.get("ingested_count", 0)
                        all_items.extend(data.get("items", []))
                        all_errors.extend(data.get("errors", []))
                    except Exception as e:
                        all_errors.append({"filename": upload_file.name, "error": str(e)})

                    elapsed = time.time() - started_at
                    avg_per_file = elapsed / index
                    remaining_sec = int(avg_per_file * (total_files - index))
                    progress_bar.progress(
                        index / total_files,
                        text=(
                            f"⏳ Processed {index}/{total_files} file(s)"
                            f" | ~{remaining_sec}s remaining"
                        ),
                    )

                status_box.empty()

                error_count = len(all_errors)
                st.success(
                    f"✅ Ingested: {ingested_total} file(s)"
                    + (f" | ❌ Errors: {error_count}" if error_count else "")
                )
                for err in all_errors:
                    st.error(f"❌ {err['filename']}: {err['error']}")
                if export_upload and all_items:
                    st.session_state["upload_transcripts"] = [
                        (item["title"], item.get("transcript", ""))
                        for item in all_items
                    ]

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
                            "use_rewrite": use_rewrite,
                            "source_title": (
                                None if selected_source_file == "All files" else selected_source_file
                            ),
                        },
                        timeout=60,
                        proxies=_NO_PROXY,
                    )
                    resp.raise_for_status()  # Raise error for bad status
                    result = resp.json()
                    if result.get("rewrite_applied"):
                        st.caption(f"🔁 Retrieval query: {result.get('retrieval_query', prompt)}")
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
