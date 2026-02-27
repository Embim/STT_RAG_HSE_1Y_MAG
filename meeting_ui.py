#!/usr/bin/env python3
"""
Web UI для обработки файлов встреч и генерации резюме.

Запуск:
    streamlit run meeting_ui.py

Требуется установленная переменная окружения:
    OPENROUTER_API_KEY - API ключ OpenRouter
"""

import sys
from pathlib import Path
import streamlit as st
import tempfile
import json
import os
from dotenv import load_dotenv

# Загружаем переменные из .env файла
load_dotenv()

# Добавляем корень проекта в sys.path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.downloader.utils.meeting_processor import process_meeting_file


def display_summary(summary: dict):
    """Отображает протокол встречи в Streamlit."""

    if "error" in summary:
        st.error(f"❌ Ошибка: {summary.get('error')}")
        st.error(f"Детали: {summary.get('details', summary.get('raw_response', summary.get('protocol', 'N/A')))}")
        return

    # Заголовок
    if "title" in summary:
        st.header(f"📋 {summary['title']}")

    # Основной текст протокола
    if "protocol" in summary:
        st.markdown("---")
        # Разбиваем на абзацы и отображаем
        paragraphs = summary["protocol"].split("\n\n")
        for paragraph in paragraphs:
            if paragraph.strip():
                st.markdown(paragraph.strip())
                st.markdown("")  # Пустая строка между абзацами
    else:
        # Fallback на старый формат если протокол не в новом формате
        if "summary" in summary:
            st.markdown(summary["summary"])

        if "key_points" in summary and summary["key_points"]:
            st.subheader("🔑 Ключевые пункты")
            for point in summary["key_points"]:
                st.markdown(f"- {point}")

        if "decisions" in summary and summary["decisions"]:
            st.subheader("✅ Решения")
            for decision in summary["decisions"]:
                st.markdown(f"- {decision}")


def main():
    """Главная функция UI."""
    st.set_page_config(
        page_title="Обработка встреч",
        page_icon="🎙️",
        layout="wide"
    )

    st.title("🎙️ Обработка файлов встреч")
    st.markdown("Загрузите аудио/видео/текстовый файл встречи для автоматической генерации резюме")

    # Sidebar с настройками
    with st.sidebar:
        st.header("⚙️ Настройки")

        # API ключ
        api_key = st.text_input(
            "OpenRouter API Key",
            value=os.getenv("OPENROUTER_API_KEY", ""),
            type="password",
            help="API ключ OpenRouter. Можно также установить через переменную окружения OPENROUTER_API_KEY"
        )

        st.markdown("---")
        st.markdown("### 📊 Поддерживаемые форматы")
        st.markdown("""
        **Аудио/Видео:**
        - MP3, WAV, M4A, FLAC, OGG
        - MP4, AVI, MKV

        **Текст:**
        - TXT (готовая транскрипция)
        """)

        st.markdown("---")
        st.markdown("### 🤖 Используемые модели")
        st.markdown("""
        - **Транскрипция:** Whisper Large V3 Turbo
        - **Резюме:** Nvidia Llama 3.1 Nemotron 70B
        """)

    # Проверка API ключа
    if not api_key:
        st.warning("⚠️ Укажите OpenRouter API ключ в боковой панели")
        st.stop()

    # Загрузка файла
    uploaded_file = st.file_uploader(
        "Выберите файл встречи",
        type=["mp3", "wav", "m4a", "flac", "ogg", "mp4", "avi", "mkv", "txt"],
        help="Загрузите аудио, видео или текстовый файл"
    )

    if uploaded_file is not None:
        # Показываем информацию о файле
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Имя файла", uploaded_file.name)
        with col2:
            st.metric("Размер", f"{uploaded_file.size / 1024 / 1024:.2f} MB")
        with col3:
            st.metric("Тип", uploaded_file.type)

        # Кнопка обработки
        if st.button("🚀 Обработать файл", type="primary", use_container_width=True):
            # Сохраняем во временный файл
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = Path(tmp_file.name)

            try:
                with st.spinner("⏳ Обработка файла... Это может занять несколько минут..."):
                    # Обрабатываем файл
                    result = process_meeting_file(
                        file_path=tmp_path,
                        openrouter_api_key=api_key,
                        save_files=False  # Не сохраняем файлы на диск в UI режиме
                    )

                st.success("✅ Обработка завершена успешно!")

                # Отображаем резюме
                display_summary(result["summary"])

                # Кнопки для скачивания результатов
                col1, col2 = st.columns(2)

                with col1:
                    if "transcript" in result:
                        st.download_button(
                            label="📄 Скачать транскрипцию",
                            data=result["transcript"],
                            file_name=f"{Path(uploaded_file.name).stem}_transcript.txt",
                            mime="text/plain"
                        )

                with col2:
                    if "summary" in result:
                        summary_json = json.dumps(result["summary"], ensure_ascii=False, indent=2)
                        st.download_button(
                            label="📊 Скачать резюме (JSON)",
                            data=summary_json,
                            file_name=f"{Path(uploaded_file.name).stem}_summary.json",
                            mime="application/json"
                        )

            except Exception as e:
                st.error(f"❌ Ошибка при обработке: {str(e)}")
                st.exception(e)

            finally:
                # Удаляем временный файл
                if tmp_path.exists():
                    tmp_path.unlink()

    else:
        # Показываем инструкции
        st.info("👆 Загрузите файл для начала обработки")

        with st.expander("ℹ️ Как это работает?"):
            st.markdown("""
            1. **Загрузите файл** - аудио, видео или текстовую транскрипцию встречи
            2. **Нажмите "Обработать"** - система автоматически:
               - Транскрибирует аудио/видео (если это не текст)
               - Отправит транскрипцию в Nvidia Nemotron 70B
               - Получит структурированное резюме встречи
            3. **Получите результат** - резюме в формате JSON с:
               - Ключевыми пунктами обсуждения
               - Принятыми решениями
               - Списком задач и поручений
               - Следующими шагами

            Все пункты представлены в виде развернутых абзацев с полным контекстом.
            """)


if __name__ == "__main__":
    main()
