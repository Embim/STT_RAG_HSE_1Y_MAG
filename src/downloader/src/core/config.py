"""
Configuration management.

Загрузка и управление конфигурацией из YAML файлов.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path
import yaml
import os
import logging

logger = logging.getLogger(__name__)


@dataclass
class SourceConfig:
    """Конфигурация источника данных."""
    name: str
    type: str
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessorConfig:
    """Конфигурация процессора."""
    name: str
    type: str
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    """Полная конфигурация pipeline."""

    # Пути
    base_dir: Path = field(default_factory=lambda: Path("E:/video_pipeline"))
    temp_dir: Optional[Path] = None
    audio_dir: Optional[Path] = None
    transcripts_dir: Optional[Path] = None
    archives_dir: Optional[Path] = None
    logs_dir: Optional[Path] = None

    # Источники
    sources: Dict[str, SourceConfig] = field(default_factory=dict)

    # Процессоры
    processors: Dict[str, ProcessorConfig] = field(default_factory=dict)

    # Чанкинг
    chunking_min_length: int = 50
    chunking_max_length: int = 2000
    chunking_merge_short: bool = True

    # Эмбеддер
    embedder_type: str = "sentence_transformer"
    embedder_model: str = "ai-forever/sbert_large_nlu_ru"
    embedder_device: str = "cuda"
    embedder_batch_size: int = 64
    embedder_max_length: int = 512
    embedder_normalize: bool = True
    embedder_use_fp16: bool = True

    # Хранилище
    store_type: str = "weaviate"
    store_url: str = "http://localhost:8080"
    store_collection: str = "ContentChunks"
    store_vector_dimension: int = 1024
    store_batch_size: int = 100

    # Обработка
    continue_on_error: bool = True
    save_transcripts: bool = True
    upload_to_store: bool = True
    cleanup_temp_files: bool = True
    gpu_cache_clear_interval: int = 5

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Журналы
    processed_journal: Optional[Path] = None
    error_journal: Optional[Path] = None

    def __post_init__(self):
        """Инициализирует пути по умолчанию."""
        if self.temp_dir is None:
            self.temp_dir = self.base_dir / "temp"
        if self.audio_dir is None:
            self.audio_dir = self.base_dir / "audio"
        if self.transcripts_dir is None:
            self.transcripts_dir = self.base_dir / "transcripts"
        if self.archives_dir is None:
            self.archives_dir = self.base_dir / "archives"
        if self.logs_dir is None:
            self.logs_dir = self.base_dir / "logs"
        if self.processed_journal is None:
            self.processed_journal = self.base_dir / "processed.json"
        if self.error_journal is None:
            self.error_journal = self.base_dir / "errors.json"

    def ensure_directories(self) -> None:
        """Создаёт все необходимые директории."""
        dirs = [
            self.base_dir,
            self.temp_dir,
            self.audio_dir,
            self.transcripts_dir,
            self.archives_dir,
            self.logs_dir,
        ]
        for d in dirs:
            if d:
                d.mkdir(parents=True, exist_ok=True)

    def get_source_config(self, name: str) -> Optional[SourceConfig]:
        """Возвращает конфигурацию источника по имени."""
        return self.sources.get(name)

    def get_processor_config(self, name: str) -> Optional[ProcessorConfig]:
        """Возвращает конфигурацию процессора по имени."""
        return self.processors.get(name)

    def get_enabled_sources(self) -> List[str]:
        """Возвращает список включённых источников."""
        return [name for name, cfg in self.sources.items() if cfg.enabled]

    def get_enabled_processors(self) -> List[str]:
        """Возвращает список включённых процессоров."""
        return [name for name, cfg in self.processors.items() if cfg.enabled]


class ConfigLoader:
    """Загрузчик конфигурации из YAML файлов."""

    @staticmethod
    def load(config_path: str = "config/config.yaml") -> Config:
        """
        Загружает конфигурацию из YAML файла.

        Args:
            config_path: Путь к файлу конфигурации

        Returns:
            Объект Config
        """
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return Config()

        with open(path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f) or {}

        return ConfigLoader._parse_config(raw_config)

    @staticmethod
    def _parse_config(raw: Dict[str, Any]) -> Config:
        """Парсит сырой YAML в объект Config."""
        config = Config()

        # Пути
        if "paths" in raw:
            paths = raw["paths"]
            if "base_dir" in paths:
                config.base_dir = Path(paths["base_dir"])
            if "temp_dir" in paths:
                config.temp_dir = Path(paths["temp_dir"])
            if "audio_dir" in paths:
                config.audio_dir = Path(paths["audio_dir"])
            if "transcripts_dir" in paths:
                config.transcripts_dir = Path(paths["transcripts_dir"])
            if "archives_dir" in paths:
                config.archives_dir = Path(paths["archives_dir"])
            if "logs_dir" in paths:
                config.logs_dir = Path(paths["logs_dir"])
            if "processed_journal" in paths:
                config.processed_journal = Path(paths["processed_journal"])
            if "error_journal" in paths:
                config.error_journal = Path(paths["error_journal"])

        # Переинициализируем пути после установки base_dir
        config.__post_init__()

        # Источники
        if "sources" in raw:
            for name, source_cfg in raw["sources"].items():
                config.sources[name] = SourceConfig(
                    name=name,
                    type=source_cfg.get("type", name),
                    enabled=source_cfg.get("enabled", True),
                    params=source_cfg.get("params", {}),
                )

        # Процессоры
        if "processors" in raw:
            for name, proc_cfg in raw["processors"].items():
                config.processors[name] = ProcessorConfig(
                    name=name,
                    type=proc_cfg.get("type", name),
                    enabled=proc_cfg.get("enabled", True),
                    params=proc_cfg.get("params", {}),
                )

        # Чанкинг
        if "chunking" in raw:
            chunking = raw["chunking"]
            config.chunking_min_length = chunking.get("min_length", 50)
            config.chunking_max_length = chunking.get("max_length", 2000)
            config.chunking_merge_short = chunking.get("merge_short", True)

        # Эмбеддер
        if "embedder" in raw:
            emb = raw["embedder"]
            config.embedder_type = emb.get("type", "sentence_transformer")
            config.embedder_model = emb.get("model", "ai-forever/sbert_large_nlu_ru")
            config.embedder_device = emb.get("device", "cuda")
            config.embedder_batch_size = emb.get("batch_size", 64)
            config.embedder_max_length = emb.get("max_length", 512)
            config.embedder_normalize = emb.get("normalize", True)
            config.embedder_use_fp16 = emb.get("use_fp16", True)

        # Хранилище
        if "store" in raw:
            store = raw["store"]
            config.store_type = store.get("type", "weaviate")
            config.store_url = store.get("url", "http://localhost:8080")
            config.store_collection = store.get("collection", "ContentChunks")
            config.store_vector_dimension = store.get("vector_dimension", 1024)
            config.store_batch_size = store.get("batch_size", 100)

        # Обработка
        if "processing" in raw:
            proc = raw["processing"]
            config.continue_on_error = proc.get("continue_on_error", True)
            config.save_transcripts = proc.get("save_transcripts", True)
            config.upload_to_store = proc.get("upload_to_store", True)
            config.cleanup_temp_files = proc.get("cleanup_temp_files", True)
            config.gpu_cache_clear_interval = proc.get("gpu_cache_clear_interval", 5)

        # API
        if "api" in raw:
            api = raw["api"]
            config.api_host = api.get("host", "0.0.0.0")
            config.api_port = api.get("port", 8000)

        return config

    @staticmethod
    def save(config: Config, config_path: str = "config/config.yaml") -> None:
        """
        Сохраняет конфигурацию в YAML файл.

        Args:
            config: Объект Config
            config_path: Путь к файлу
        """
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "paths": {
                "base_dir": str(config.base_dir),
                "temp_dir": str(config.temp_dir),
                "audio_dir": str(config.audio_dir),
                "transcripts_dir": str(config.transcripts_dir),
                "archives_dir": str(config.archives_dir),
                "logs_dir": str(config.logs_dir),
                "processed_journal": str(config.processed_journal),
                "error_journal": str(config.error_journal),
            },
            "sources": {
                name: {
                    "type": src.type,
                    "enabled": src.enabled,
                    "params": src.params,
                }
                for name, src in config.sources.items()
            },
            "processors": {
                name: {
                    "type": proc.type,
                    "enabled": proc.enabled,
                    "params": proc.params,
                }
                for name, proc in config.processors.items()
            },
            "chunking": {
                "min_length": config.chunking_min_length,
                "max_length": config.chunking_max_length,
                "merge_short": config.chunking_merge_short,
            },
            "embedder": {
                "type": config.embedder_type,
                "model": config.embedder_model,
                "device": config.embedder_device,
                "batch_size": config.embedder_batch_size,
                "max_length": config.embedder_max_length,
                "normalize": config.embedder_normalize,
                "use_fp16": config.embedder_use_fp16,
            },
            "store": {
                "type": config.store_type,
                "url": config.store_url,
                "collection": config.store_collection,
                "vector_dimension": config.store_vector_dimension,
                "batch_size": config.store_batch_size,
            },
            "processing": {
                "continue_on_error": config.continue_on_error,
                "save_transcripts": config.save_transcripts,
                "upload_to_store": config.upload_to_store,
                "cleanup_temp_files": config.cleanup_temp_files,
                "gpu_cache_clear_interval": config.gpu_cache_clear_interval,
            },
            "api": {
                "host": config.api_host,
                "port": config.api_port,
            },
        }

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        logger.info(f"Config saved to {config_path}")


# === Глобальный экземпляр ===

_global_config: Optional[Config] = None


def get_config(config_path: str = "config/config.yaml") -> Config:
    """
    Возвращает глобальный экземпляр конфигурации.

    Args:
        config_path: Путь к файлу конфигурации (используется при первой загрузке)

    Returns:
        Объект Config
    """
    global _global_config
    if _global_config is None:
        _global_config = ConfigLoader.load(config_path)
    return _global_config


def reload_config(config_path: str = "config/config.yaml") -> Config:
    """
    Перезагружает конфигурацию.

    Args:
        config_path: Путь к файлу конфигурации

    Returns:
        Новый объект Config
    """
    global _global_config
    _global_config = ConfigLoader.load(config_path)
    return _global_config
