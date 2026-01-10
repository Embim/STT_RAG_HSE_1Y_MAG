"""
Plugin registry for dynamic component discovery.

Центральный реестр для всех плагинов (источников, процессоров, эмбеддеров, хранилищ).
Поддерживает автоматическое обнаружение и регистрацию через декораторы.
"""

from typing import Dict, Type, List, Optional, Callable
from pathlib import Path
import importlib
import pkgutil
import logging

from .interfaces import BaseSource, BaseProcessor, BaseChunker, BaseEmbedder, BaseStore

logger = logging.getLogger(__name__)


class PluginRegistry:
    """
    Центральный реестр для всех плагинов.

    Реализует паттерн Singleton для глобального доступа.
    Поддерживает auto-discovery для автоматической регистрации плагинов.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self._sources: Dict[str, Type[BaseSource]] = {}
        self._processors: Dict[str, Type[BaseProcessor]] = {}
        self._chunkers: Dict[str, Type[BaseChunker]] = {}
        self._embedders: Dict[str, Type[BaseEmbedder]] = {}
        self._stores: Dict[str, Type[BaseStore]] = {}

        logger.debug("PluginRegistry initialized")

    # === Методы регистрации ===

    def register_source(self, source_class: Type[BaseSource]) -> None:
        """Регистрирует источник данных."""
        name = source_class.source_name
        self._sources[name] = source_class
        logger.debug(f"Registered source: {name}")

    def register_processor(self, processor_class: Type[BaseProcessor]) -> None:
        """Регистрирует процессор контента."""
        name = processor_class.processor_name
        self._processors[name] = processor_class
        logger.debug(f"Registered processor: {name}")

    def register_chunker(self, chunker_class: Type[BaseChunker]) -> None:
        """Регистрирует чанкер."""
        name = chunker_class.chunker_name
        self._chunkers[name] = chunker_class
        logger.debug(f"Registered chunker: {name}")

    def register_embedder(self, embedder_class: Type[BaseEmbedder]) -> None:
        """Регистрирует эмбеддер."""
        name = embedder_class.embedder_name
        self._embedders[name] = embedder_class
        logger.debug(f"Registered embedder: {name}")

    def register_store(self, store_class: Type[BaseStore]) -> None:
        """Регистрирует хранилище векторов."""
        name = store_class.store_name
        self._stores[name] = store_class
        logger.debug(f"Registered store: {name}")

    # === Методы получения ===

    def get_source(self, name: str) -> Optional[Type[BaseSource]]:
        """Возвращает класс источника по имени."""
        return self._sources.get(name)

    def get_processor(self, name: str) -> Optional[Type[BaseProcessor]]:
        """Возвращает класс процессора по имени."""
        return self._processors.get(name)

    def get_chunker(self, name: str) -> Optional[Type[BaseChunker]]:
        """Возвращает класс чанкера по имени."""
        return self._chunkers.get(name)

    def get_embedder(self, name: str) -> Optional[Type[BaseEmbedder]]:
        """Возвращает класс эмбеддера по имени."""
        return self._embedders.get(name)

    def get_store(self, name: str) -> Optional[Type[BaseStore]]:
        """Возвращает класс хранилища по имени."""
        return self._stores.get(name)

    # === Методы перечисления ===

    def list_sources(self) -> List[str]:
        """Возвращает список имён всех зарегистрированных источников."""
        return list(self._sources.keys())

    def list_processors(self) -> List[str]:
        """Возвращает список имён всех зарегистрированных процессоров."""
        return list(self._processors.keys())

    def list_chunkers(self) -> List[str]:
        """Возвращает список имён всех зарегистрированных чанкеров."""
        return list(self._chunkers.keys())

    def list_embedders(self) -> List[str]:
        """Возвращает список имён всех зарегистрированных эмбеддеров."""
        return list(self._embedders.keys())

    def list_stores(self) -> List[str]:
        """Возвращает список имён всех зарегистрированных хранилищ."""
        return list(self._stores.keys())

    # === Создание экземпляров ===

    def create_source(self, name: str, **kwargs) -> Optional[BaseSource]:
        """Создаёт экземпляр источника."""
        source_class = self.get_source(name)
        if source_class:
            return source_class(**kwargs)
        return None

    def create_processor(self, name: str, **kwargs) -> Optional[BaseProcessor]:
        """Создаёт экземпляр процессора."""
        processor_class = self.get_processor(name)
        if processor_class:
            return processor_class(**kwargs)
        return None

    def create_chunker(self, name: str, **kwargs) -> Optional[BaseChunker]:
        """Создаёт экземпляр чанкера."""
        chunker_class = self.get_chunker(name)
        if chunker_class:
            return chunker_class(**kwargs)
        return None

    def create_embedder(self, name: str, **kwargs) -> Optional[BaseEmbedder]:
        """Создаёт экземпляр эмбеддера."""
        embedder_class = self.get_embedder(name)
        if embedder_class:
            return embedder_class(**kwargs)
        return None

    def create_store(self, name: str, **kwargs) -> Optional[BaseStore]:
        """Создаёт экземпляр хранилища."""
        store_class = self.get_store(name)
        if store_class:
            return store_class(**kwargs)
        return None

    # === Auto-discovery ===

    def discover_plugins(self, package_path: str) -> None:
        """
        Автоматически обнаруживает и регистрирует все плагины в пакете.

        Args:
            package_path: Путь импорта пакета (например 'src.sources')
        """
        try:
            package = importlib.import_module(package_path)
            package_dir = Path(package.__file__).parent

            for _, module_name, _ in pkgutil.iter_modules([str(package_dir)]):
                if module_name.startswith("_"):
                    continue

                try:
                    module = importlib.import_module(f"{package_path}.{module_name}")

                    # Находим и регистрируем классы плагинов
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if isinstance(attr, type):
                            self._try_register(attr)

                except Exception as e:
                    logger.warning(f"Failed to import module {module_name}: {e}")

        except Exception as e:
            logger.error(f"Failed to discover plugins in {package_path}: {e}")

    def _try_register(self, cls: type) -> None:
        """Пытается зарегистрировать класс если он наследуется от базового."""
        if issubclass(cls, BaseSource) and cls != BaseSource:
            if hasattr(cls, "source_name") and cls.source_name != "base":
                self.register_source(cls)
        elif issubclass(cls, BaseProcessor) and cls != BaseProcessor:
            if hasattr(cls, "processor_name") and cls.processor_name != "base":
                self.register_processor(cls)
        elif issubclass(cls, BaseChunker) and cls != BaseChunker:
            if hasattr(cls, "chunker_name") and cls.chunker_name != "base":
                self.register_chunker(cls)
        elif issubclass(cls, BaseEmbedder) and cls != BaseEmbedder:
            if hasattr(cls, "embedder_name") and cls.embedder_name != "base":
                self.register_embedder(cls)
        elif issubclass(cls, BaseStore) and cls != BaseStore:
            if hasattr(cls, "store_name") and cls.store_name != "base":
                self.register_store(cls)

    def discover_all(self) -> None:
        """Обнаруживает все плагины во всех стандартных пакетах."""
        packages = [
            "src.sources",
            "src.processors.audio",
            "src.processors.text",
            "src.processors.vision",
            "src.embedders",
            "src.stores",
        ]
        for package in packages:
            try:
                self.discover_plugins(package)
            except Exception as e:
                logger.debug(f"Could not discover plugins in {package}: {e}")

    # === Информация о реестре ===

    def get_info(self) -> Dict[str, List[str]]:
        """Возвращает информацию о всех зарегистрированных плагинах."""
        return {
            "sources": self.list_sources(),
            "processors": self.list_processors(),
            "chunkers": self.list_chunkers(),
            "embedders": self.list_embedders(),
            "stores": self.list_stores(),
        }


# === Декораторы для удобной регистрации ===


def register_source(cls: Type[BaseSource]) -> Type[BaseSource]:
    """
    Декоратор для регистрации источника.

    Пример:
        @register_source
        class YouTubeSource(BaseSource):
            source_name = "youtube"
            ...
    """
    PluginRegistry().register_source(cls)
    return cls


def register_processor(cls: Type[BaseProcessor]) -> Type[BaseProcessor]:
    """Декоратор для регистрации процессора."""
    PluginRegistry().register_processor(cls)
    return cls


def register_chunker(cls: Type[BaseChunker]) -> Type[BaseChunker]:
    """Декоратор для регистрации чанкера."""
    PluginRegistry().register_chunker(cls)
    return cls


def register_embedder(cls: Type[BaseEmbedder]) -> Type[BaseEmbedder]:
    """Декоратор для регистрации эмбеддера."""
    PluginRegistry().register_embedder(cls)
    return cls


def register_store(cls: Type[BaseStore]) -> Type[BaseStore]:
    """Декоратор для регистрации хранилища."""
    PluginRegistry().register_store(cls)
    return cls


# === Глобальная функция доступа ===


def get_registry() -> PluginRegistry:
    """Возвращает глобальный экземпляр реестра."""
    return PluginRegistry()
