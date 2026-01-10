#!/usr/bin/env python3
"""
Video Processing Pipeline - CLI Interface.

Команды:
    python main.py process --source youtube --urls "https://..."
    python main.py process --source local_files --paths "E:/documents"
    python main.py search "поисковый запрос"
    python main.py serve
    python main.py status
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

# Подавляем warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# Подавляем warnings от audioread
warnings.filterwarnings("ignore", module="audioread")


def setup_logging(level: str = "INFO", log_file: str = None):
    """Настраивает логирование."""
    handlers = [logging.StreamHandler()]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )

    # Подавляем логи сторонних библиотек
    for lib in ["httpx", "weaviate-client", "sentence_transformers", "huggingface_hub", "transformers", "torch"]:
        logging.getLogger(lib).setLevel(logging.WARNING)


def cmd_process(args):
    """Обрабатывает контент из источника."""
    from src.core.config import ConfigLoader
    from src.core.pipeline import PipelineBuilder
    from src.utils.tracking import ProcessingTracker
    import logging as log

    # Загружаем конфигурацию
    config = ConfigLoader.load(args.config)
    config.ensure_directories()

    # Добавляем FileHandler если не задан явно
    if not args.log_file and hasattr(config, 'base_dir'):
        log_file = config.base_dir / "logs" / "pipeline.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Добавляем file handler к root logger
        file_handler = logging.FileHandler(str(log_file), encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        logging.getLogger().addHandler(file_handler)
        log.info(f"Logging to file: {log_file}")

    # Разделяем параметры на конфигурационные (для build) и запросные (для run)
    config_params = {}  # Параметры для __init__ источника
    fetch_params = {}   # Параметры для fetch() источника

    if args.source == "youtube":
        # Конфигурационные параметры
        config_params["output_dir"] = str(config.audio_dir)
        # Параметры запроса
        urls = args.urls.split(",") if args.urls else []
        fetch_params["urls"] = urls
    elif args.source == "local_files":
        # Параметры запроса
        paths = args.paths.split(",") if args.paths else []
        fetch_params["paths"] = paths
        fetch_params["recursive"] = args.recursive
        if args.extensions:
            fetch_params["extensions"] = args.extensions.split(",")

    # Создаём tracker для пропуска обработанных
    tracker = ProcessingTracker(
        str(config.processed_journal),
        str(config.error_journal),
    )
    skip_ids = tracker.get_processed_ids()

    # Callback для обновления tracker после каждого обработанного элемента
    def on_item_processed(result):
        """Сохраняет результат в tracker сразу после обработки."""
        if result.success:
            tracker.mark_processed(
                source_id=result.item.source_id,
                title=result.item.title,
                author=result.item.author,
                chunks_count=result.chunks_created,
                duration=result.item.duration,
            )
        else:
            tracker.log_error(
                source_id=result.item.source_id,
                stage="processing",
                error_type="ProcessingError",
                error_message=result.error_message or "Unknown error",
            )

    # Создаём pipeline (только конфигурационные параметры)
    builder = PipelineBuilder(config)
    pipeline = builder.build(args.source, config_params)
    pipeline.result_callback = on_item_processed

    try:
        # Запускаем обработку (передаём параметры запроса)
        fetch_params["skip_ids"] = list(skip_ids) if args.source == "youtube" else None
        results = pipeline.run(
            show_progress=not args.quiet,
            **fetch_params,
        )

        # Выводим статистику
        stats = tracker.get_statistics()
        print(f"\nProcessing complete!")
        print(f"  Total processed: {stats['total_processed']}")
        print(f"  Completed: {stats['completed']}")
        print(f"  Failed: {stats['failed']}")
        print(f"  Total chunks: {stats['total_chunks']}")

    finally:
        pipeline.teardown()


def cmd_search(args):
    """Ищет по базе."""
    from src.core.config import ConfigLoader
    from src.embedders.sentence_transformer import SentenceTransformerEmbedder
    from src.stores.weaviate import WeaviateStore

    config = ConfigLoader.load(args.config)

    # Инициализируем компоненты
    embedder = SentenceTransformerEmbedder(
        model_name=config.embedder_model,
        device=config.embedder_device,
    )
    embedder.setup({})

    store = WeaviateStore(
        url=config.store_url,
        collection_name=config.store_collection,
    )
    store.connect()

    try:
        # Генерируем эмбеддинг запроса
        query_embedding = embedder.embed_single(args.query)

        # Ищем
        results = store.search(query_embedding, limit=args.limit)

        # Выводим результаты
        print(f"\nSearch results for: {args.query}\n")
        print("=" * 80)

        for i, r in enumerate(results, 1):
            score = r.get("score", 0)
            title = r.get("title", "Unknown")
            text = r.get("text", "")[:200]
            source_id = r.get("source_id", "")
            start = r.get("start_position", 0)

            print(f"{i}. [{score:.3f}] {title}")
            print(f"   Source: {source_id} @ {start:.1f}s")
            print(f"   {text}...")
            print()

        print("=" * 80)
        print(f"Total: {len(results)} results")

    finally:
        store.close()
        embedder.teardown()


def cmd_serve(args):
    """Запускает API сервер."""
    import uvicorn
    from src.api.main import app

    print(f"Starting API server at http://{args.host}:{args.port}")
    print("Documentation: http://{args.host}:{args.port}/docs")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info" if not args.quiet else "warning",
    )


def cmd_status(args):
    """Показывает статус системы."""
    import warnings
    from src.core.config import ConfigLoader
    from src.core.registry import PluginRegistry
    from src.stores.weaviate import WeaviateStore
    from src.utils.tracking import ProcessingTracker

    # Подавляем ResourceWarning и логи от Weaviate
    warnings.filterwarnings("ignore", category=ResourceWarning)
    logging.getLogger("src.stores.weaviate").setLevel(logging.WARNING)
    logging.getLogger("weaviate").setLevel(logging.WARNING)

    config = ConfigLoader.load(args.config)

    # Статус реестра
    registry = PluginRegistry()
    registry.discover_all()

    print("\n" + "=" * 60)
    print("VIDEO PROCESSING PIPELINE - STATUS")
    print("=" * 60)

    print("\nRegistered Plugins:")
    print(f"  Sources: {', '.join(registry.list_sources()) or 'none'}")
    print(f"  Processors: {', '.join(registry.list_processors()) or 'none'}")
    print(f"  Chunkers: {', '.join(registry.list_chunkers()) or 'none'}")
    print(f"  Embedders: {', '.join(registry.list_embedders()) or 'none'}")
    print(f"  Stores: {', '.join(registry.list_stores()) or 'none'}")

    # Статус хранилища
    print("\nVector Store:")
    store = None
    try:
        store = WeaviateStore(
            url=config.store_url,
            collection_name=config.store_collection,
        )
        store.connect()
        count = store.count()
        print(f"  URL: {config.store_url}")
        print(f"  Collection: {config.store_collection}")
        print(f"  Total records: {count}")
    except Exception as e:
        print(f"  Status: NOT CONNECTED ({e})")
    finally:
        if store:
            store.close()

    # Статус tracker
    print("\nProcessing Tracker:")
    tracker = ProcessingTracker(
        str(config.processed_journal),
        str(config.error_journal),
    )
    stats = tracker.get_statistics()
    print(f"  Total processed: {stats['total_processed']}")
    print(f"  Completed: {stats['completed']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Total errors: {stats['total_errors']}")

    print("\n" + "=" * 60)


def main():
    """Главная функция CLI."""
    parser = argparse.ArgumentParser(
        description="Video Processing Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--log-file",
        help="Log file path",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet mode (less output)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # process
    process_parser = subparsers.add_parser("process", help="Process content from source")
    process_parser.add_argument(
        "--source",
        required=True,
        choices=["youtube", "local_files"],
        help="Source type",
    )
    process_parser.add_argument(
        "--urls",
        help="Comma-separated YouTube URLs (for youtube source)",
    )
    process_parser.add_argument(
        "--paths",
        help="Comma-separated file/directory paths (for local_files source)",
    )
    process_parser.add_argument(
        "--extensions",
        help="Comma-separated file extensions filter",
    )
    process_parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Recursive directory search",
    )

    # search
    search_parser = subparsers.add_parser("search", help="Search in vector store")
    search_parser.add_argument(
        "query",
        help="Search query",
    )
    search_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Max results",
    )

    # serve
    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind",
    )

    # status
    subparsers.add_parser("status", help="Show system status")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Настраиваем логирование
    setup_logging(args.log_level, args.log_file)

    # Выполняем команду
    if args.command == "process":
        cmd_process(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "status":
        cmd_status(args)


if __name__ == "__main__":
    main()
