"""
Pipeline orchestrator.

Координирует весь процесс обработки: источники -> процессоры -> эмбеддинги -> хранилище.
"""

from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import logging
import time

from tqdm import tqdm

from .models import ContentItem, ProcessingResult, PipelineStats, ProcessingStage, TextChunk
from .interfaces import BaseSource, BaseProcessor, BaseChunker, BaseEmbedder, BaseStore
from .config import Config
from .registry import PluginRegistry

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Главный оркестратор обработки контента.

    Координирует работу всех компонентов:
    - Источники получают контент
    - Процессоры извлекают текст
    - Чанкеры разбивают на части
    - Эмбеддеры генерируют векторы
    - Хранилища сохраняют результаты

    Поддерживает прогресс-бары, статистику и обработку ошибок.
    """

    def __init__(
        self,
        config: Config,
        source: BaseSource,
        processors: List[BaseProcessor],
        chunker: BaseChunker,
        embedder: BaseEmbedder,
        store: Optional[BaseStore] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        result_callback: Optional[Callable[[ProcessingResult], None]] = None,
    ):
        """
        Инициализация pipeline.

        Args:
            config: Конфигурация
            source: Источник данных
            processors: Список процессоров
            chunker: Чанкер
            embedder: Эмбеддер
            store: Хранилище векторов (опционально)
            progress_callback: Callback для обновления прогресса
            result_callback: Callback вызываемый после обработки каждого элемента
        """
        self.config = config
        self.source = source
        self.processors = processors
        self.chunker = chunker
        self.embedder = embedder
        self.store = store
        self.progress_callback = progress_callback
        self.result_callback = result_callback

        self.stats = PipelineStats()
        self._setup_complete = False

    def setup(self) -> None:
        """Инициализирует все компоненты."""
        if self._setup_complete:
            return

        logger.info("Setting up pipeline components...")

        # Настраиваем процессоры
        for processor in self.processors:
            processor.setup({})

        # Настраиваем эмбеддер
        self.embedder.setup({})

        # Подключаемся к хранилищу
        if self.store:
            self.store.connect()

        self._setup_complete = True
        logger.info("Pipeline setup complete")

    def teardown(self) -> None:
        """Освобождает ресурсы."""
        logger.info("Tearing down pipeline...")

        for processor in self.processors:
            processor.teardown()

        self.embedder.teardown()

        if self.store:
            self.store.close()

        self._setup_complete = False
        logger.info("Pipeline teardown complete")

    def _find_processor(self, item: ContentItem) -> Optional[BaseProcessor]:
        """Находит подходящий процессор для item."""
        for processor in self.processors:
            if processor.can_process(item):
                return processor
        return None

    def _save_transcript(self, item: ContentItem) -> None:
        """Сохраняет транскрипт в текстовый и JSON файлы."""
        try:
            from pathlib import Path
            import json
            transcripts_dir = Path(self.config.transcripts_dir)
            transcripts_dir.mkdir(parents=True, exist_ok=True)

            # Имя файла из названия видео (безопасное для файловой системы)
            # Используем title если есть, иначе source_id
            filename_base = item.title if item.title else item.source_id
            safe_name = "".join(c for c in filename_base if c.isalnum() or c in (' ', '-', '_')).strip()

            # Ограничиваем длину имени файла (Windows лимит 255 символов)
            if len(safe_name) > 200:
                safe_name = safe_name[:200].strip()

            # Если после очистки имя пустое, используем source_id
            if not safe_name:
                safe_name = item.source_id

            # 1. Сохраняем TXT формат
            transcript_file = transcripts_dir / f"{safe_name}.txt"
            lines = [
                f"Title: {item.title}",
                f"Author: {item.author}",
                f"Source ID: {item.source_id}",
                f"Duration: {item.duration:.2f}s" if item.duration else "Duration: N/A",
                f"Language: {item.language}",
                "",
                "=" * 80,
                "",
            ]

            # Добавляем сегменты с timestamps
            for seg in item.segments:
                timestamp = f"[{seg.start:.2f}s - {seg.end:.2f}s]"
                lines.append(f"{timestamp} {seg.text}")

            transcript_file.write_text("\n".join(lines), encoding="utf-8")
            logger.debug(f"Saved TXT transcript: {transcript_file}")

            # 2. Сохраняем JSON формат
            json_file = transcripts_dir / f"{safe_name}.json"
            json_data = {
                "source_id": item.source_id,
                "title": item.title,
                "author": item.author,
                "url": item.url,
                "duration": item.duration,
                "language": item.language,
                "segments": [
                    {
                        "id": seg.id,
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text,
                        "confidence": seg.confidence,
                    }
                    for seg in item.segments
                ],
                "chunks_count": len(item.chunks),
                "processed_at": datetime.now().isoformat(),
            }

            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            logger.debug(f"Saved JSON transcript: {json_file}")

        except Exception as e:
            logger.warning(f"Failed to save transcript for {item.source_id}: {e}")

    def _process_item(self, item: ContentItem) -> ProcessingResult:
        """
        Обрабатывает один ContentItem через весь pipeline.

        Args:
            item: ContentItem для обработки

        Returns:
            ProcessingResult с результатами
        """
        start_time = time.time()

        try:
            # 1. Находим процессор
            processor = self._find_processor(item)
            if not processor:
                raise ValueError(f"No processor found for content type: {item.content_type}")

            # 2. Извлекаем текст/сегменты
            item = processor.process(item)

            if not item.segments and not item.raw_text:
                raise ValueError("Processing produced no text content")

            # 3. Создаём чанки
            item.chunks = self.chunker.chunk(
                item.segments,
                min_length=self.config.chunking_min_length,
                max_length=self.config.chunking_max_length,
            )

            if not item.chunks:
                raise ValueError("Chunking produced no chunks")

            item.update_stage(ProcessingStage.CHUNKED)

            # 4. Генерируем эмбеддинги (документы, не запросы)
            texts = [chunk.text for chunk in item.chunks]
            embeddings = self.embedder.embed(texts, is_query=False)

            for chunk, embedding in zip(item.chunks, embeddings):
                chunk.embedding = embedding

            item.update_stage(ProcessingStage.EMBEDDED)

            # 5. Сохраняем в хранилище
            stored_count = 0
            if self.store and self.config.upload_to_store:
                import asyncio
                metadata = {
                    "source_id": item.source_id,
                    "source_type": item.source_type.value,
                    "title": item.title,
                    "author": item.author,
                    "url": item.url,
                    "language": item.language,
                }
                # VectorStoreAdapter использует async метод
                asyncio.run(self.store.add_chunks(item.chunks, metadata))
                stored_count = len(item.chunks)
                item.update_stage(ProcessingStage.STORED)

            # 6. Сохраняем транскрипт в файл
            if self.config.transcripts_dir and item.segments:
                self._save_transcript(item)

            # 7. Удаляем временные файлы (аудио) после успешной обработки
            # ОТКЛЮЧЕНО: Сохраняем скачанные видео
            # if item.source_path and item.source_path.exists():
            #     try:
            #         item.source_path.unlink()
            #         logger.debug(f"Deleted temporary file: {item.source_path}")
            #     except Exception as e:
            #         logger.warning(f"Failed to delete {item.source_path}: {e}")

            processing_time = time.time() - start_time

            return ProcessingResult(
                item=item,
                success=True,
                chunks_created=len(item.chunks),
                embeddings_generated=len(embeddings),
                stored_count=stored_count,
                processing_time=processing_time,
            )

        except Exception as e:
            import traceback
            logger.error(f"Error processing item {item.source_id}: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            item.add_error(str(e))
            item.update_stage(ProcessingStage.FAILED)

            return ProcessingResult(
                item=item,
                success=False,
                processing_time=time.time() - start_time,
                error_message=str(e),
            )

    def run(
        self,
        show_progress: bool = True,
        **source_kwargs,
    ) -> List[ProcessingResult]:
        """
        Запускает pipeline.

        Args:
            show_progress: Показывать прогресс-бар
            **source_kwargs: Аргументы для source.fetch()

        Returns:
            Список ProcessingResult
        """
        # Инициализация
        self.setup()

        self.stats = PipelineStats()
        self.stats.start_time = datetime.now()

        results = []

        try:
            # Получаем генератор элементов из источника (без материализации в список!)
            items_generator = self.source.fetch(**source_kwargs)

            # Начинаем итерацию - это позволит источнику установить total_count
            pbar = None
            idx = 0

            for item in items_generator:
                idx += 1
                self.stats.total_items = idx

                # Создаем прогресс-бар на первой итерации (когда источник установил total_count)
                if idx == 1 and show_progress:
                    total = getattr(self.source, 'total_count', None)
                    if total and total > 0:
                        pbar = tqdm(
                            total=total,
                            desc=f"Processing",
                            unit="video",
                            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}',
                        )
                    else:
                        pbar = tqdm(
                            desc=f"Processing",
                            unit="item",
                            bar_format='{desc}: {n_fmt} items [{elapsed}] {postfix}',
                        )

                # Обновляем прогресс-бар
                if pbar is not None:
                    title = item.title[:40] if item.title else str(item.source_id)[:40]
                    pbar.set_postfix_str(title)
                    pbar.update(1)

                # Обрабатываем элемент
                result = self._process_item(item)
                results.append(result)

                # Вызываем callback если задан (для обновления tracker)
                if self.result_callback:
                    self.result_callback(result)

                # Обновляем статистику
                self.stats.processed += 1
                if result.success:
                    self.stats.succeeded += 1
                    self.stats.chunks_created += result.chunks_created
                    self.stats.embeddings_generated += result.embeddings_generated
                    self.stats.stored += result.stored_count
                else:
                    self.stats.failed += 1
                    if not self.config.continue_on_error:
                        break

                # Очистка GPU кеша
                if idx % self.config.gpu_cache_clear_interval == 0:
                    self.embedder.clear_cache()

                # Callback прогресса
                if self.progress_callback:
                    if total:
                        self.progress_callback(
                            f"Processed {idx}/{total}",
                            idx / total,
                        )
                    else:
                        self.progress_callback(
                            f"Processed {idx}",
                            0.0,
                        )

            if pbar is not None:
                pbar.close()

        finally:
            self.stats.end_time = datetime.now()
            self._log_stats()

        return results

    def _log_stats(self) -> None:
        """Выводит статистику."""
        stats = self.stats.to_dict()

        logger.info("=" * 60)
        logger.info("PIPELINE STATISTICS")
        logger.info("=" * 60)
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 60)


class PipelineBuilder:
    """
    Builder для создания pipeline из конфигурации.

    Упрощает создание pipeline с правильными компонентами.
    """

    def __init__(self, config: Config):
        """
        Инициализация builder.

        Args:
            config: Конфигурация
        """
        self.config = config
        self.registry = PluginRegistry()

        # Автообнаружение плагинов
        self.registry.discover_all()

    def build(
        self,
        source_name: str,
        source_params: Dict[str, Any] = None,
    ) -> Pipeline:
        """
        Создаёт pipeline для указанного источника.

        Args:
            source_name: Имя источника (youtube, local_files и т.д.)
            source_params: Дополнительные параметры источника

        Returns:
            Готовый Pipeline
        """
        # Создаём источник
        source_class = self.registry.get_source(source_name)
        if not source_class:
            raise ValueError(f"Unknown source: {source_name}")

        # Разделяем параметры на конфигурационные (для __init__) и запросные (для fetch)
        # Параметры запроса не передаются в __init__, они будут переданы в run() -> fetch()
        fetch_params_keys = {'urls', 'paths', 'url', 'path', 'extensions', 'recursive', 'skip_ids'}

        # Только конфигурационные параметры для __init__
        init_params = {}
        source_config = self.config.get_source_config(source_name)
        if source_config:
            init_params.update({k: v for k, v in source_config.params.items() if k not in fetch_params_keys})

        # source_params тоже фильтруем
        if source_params:
            init_params.update({k: v for k, v in source_params.items() if k not in fetch_params_keys})

        source = source_class(**init_params)

        # Создаём процессоры
        processors = []
        for proc_name in self.config.get_enabled_processors():
            proc_class = self.registry.get_processor(proc_name)
            if proc_class:
                proc_config = self.config.get_processor_config(proc_name)
                proc_params = proc_config.params if proc_config else {}
                processors.append(proc_class(**proc_params))

        # Если нет процессоров - добавляем стандартные
        if not processors:
            whisper = self.registry.get_processor("whisper")
            if whisper:
                processors.append(whisper(
                    model_name="openai/whisper-large-v3-turbo",
                    device="cuda",
                    language="ru",
                ))

        # Создаём чанкер
        chunker_class = self.registry.get_chunker("text")
        if not chunker_class:
            from ..processors.text.chunker import TextChunker
            chunker_class = TextChunker

        chunker = chunker_class(
            min_length=self.config.chunking_min_length,
            max_length=self.config.chunking_max_length,
        )

        # Создаём эмбеддер (используем Infinity адаптер)
        from src.downloader.adapters.infinity_embedder import InfinityEmbedder

        embedder = InfinityEmbedder(
            url="http://localhost:7997",  # Infinity API
            batch_size=self.config.embedder_batch_size,
        )

        # Создаём хранилище (используем VectorStore адаптер)
        store = None
        if self.config.upload_to_store:
            from src.downloader.adapters.vector_store_adapter import VectorStoreAdapter

            store = VectorStoreAdapter(
                collection_name=self.config.store_collection,
            )

        return Pipeline(
            config=self.config,
            source=source,
            processors=processors,
            chunker=chunker,
            embedder=embedder,
            store=store,
        )
