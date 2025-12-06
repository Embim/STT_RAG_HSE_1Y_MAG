"""
RAG Pipeline для обработки транскрибированных лекций
Этапы:
1. Загрузка текстов из JSON файлов
2. Разбиение на чанки
3. Векторизация с помощью Qwen3-Embedding-0.6B
4. Загрузка в Weaviate
5. Retrieval для поиска релевантных фрагментов
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple
import weaviate
import spacy
from weaviate.classes.config import Property, DataType, Configure, VectorDistances
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class LectureProcessor:
    """Класс для обработки лекций из JSON файлов"""

    def __init__(self, transcripts_folder: str):
        self.transcripts_folder = Path(transcripts_folder)

    def load_lectures(self) -> List[Dict]:
        """Загружает все лекции из папки transcripted_text"""
        lectures = []
        json_files = sorted(self.transcripts_folder.glob("lecture_*.json"))

        print(f"Найдено {len(json_files)} файлов лекций")

        for json_file in tqdm(json_files, desc="Загрузка лекций"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    lectures.append({
                        'hash': data.get('hash', ''),
                        'text': data.get('text', ''),
                        'segments': data.get('segments', []),
                        'lecture_id': json_file.stem,
                        'filename': json_file.name
                    })
            except Exception as e:
                print(f"Ошибка при загрузке {json_file}: {e}")

        return lectures


class TextChunker:
    """Класс для разбиения текста на чанки"""

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        """
        Args:
            chunk_size: Размер чанка в словах
            overlap: Перекрытие между чанками в словах
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.nlp_cleaner = spacy.load("ru_core_news_sm")

    def clean_text(self, text: str):
        doc = self.nlp_cleaner(text)
        cleaned_tokens = [token.text for token in doc if token.pos_ not in ["INTJ", "PART", "SYM"]]
        cleaned_text = " ".join(cleaned_tokens)
        return cleaned_text

    def chunk_text(self, text: str, metadata: Dict) -> List[Dict]:
        """
        Разбивает текст на чанки с перекрытием

        Args:
            text: Текст для разбиения
            metadata: Метаданные (lecture_id, hash и т.д.)

        Returns:
            Список словарей с чанками и метаданными
        """
        words = self.clean_text(text).split()
        chunks = []

        start = 0
        chunk_id = 0

        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_text = ' '.join(words[start:end])

            chunks.append({
                'text': chunk_text,
                'chunk_id': chunk_id,
                'lecture_id': metadata.get('lecture_id', ''),
                'lecture_hash': metadata.get('hash', ''),
                'filename': metadata.get('filename', ''),
                'word_start': start,
                'word_end': end,
                'total_words': len(words)
            })

            chunk_id += 1
            start += self.chunk_size - self.overlap

        return chunks

    def process_lectures(self, lectures: List[Dict]) -> List[Dict]:
        """Обрабатывает все лекции и разбивает их на чанки"""
        all_chunks = []

        for lecture in tqdm(lectures, desc="Разбиение на чанки"):
            chunks = self.chunk_text(lecture['text'], lecture)
            all_chunks.extend(chunks)

        print(f"Создано {len(all_chunks)} чанков из {len(lectures)} лекций")
        return all_chunks


class EmbeddingModel:
    """Класс для векторизации текста"""

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B", device: str = "cuda"):
        """
        Инициализация модели эмбеддингов

        Args:
            model_name: Название модели из HuggingFace
            device: Устройство для выполнения ('cuda' для GPU, 'cpu' для CPU)
        """
        print(f"Загрузка модели {model_name} на {device}...")
        self.model = SentenceTransformer(model_name, device=device)
        print(f"Модель загружена успешно на {device}!")

    def encode_chunks(self, chunks: List[Dict], batch_size: int = 32) -> List[Dict]:
        """
        Векторизует чанки текста

        Args:
            chunks: Список чанков с текстом
            batch_size: Размер батча для обработки

        Returns:
            Список чанков с добавленными эмбеддингами
        """
        texts = [chunk['text'] for chunk in chunks]

        print(f"Векторизация {len(texts)} чанков...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Добавляем эмбеддинги к чанкам
        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding.tolist()

        return chunks


class WeaviateIndexer:
    """Класс для работы с Weaviate векторной БД"""

    def __init__(self, url: str = "http://localhost:8080", collection_name: str = "LectureChunks"):
        """
        Инициализация подключения к Weaviate

        Args:
            url: URL для подключения к Weaviate
            collection_name: Название коллекции
        """
        self.collection_name = collection_name
        self.client = None
        self.url = url

    def connect(self):
        """Подключение к Weaviate"""
        try:
            print(f"Подключение к Weaviate по адресу {self.url}...")
            self.client = weaviate.connect_to_local(host=self.url.replace("http://", "").split(":")[0])
            print("Подключение установлено!")
        except Exception as e:
            print(f"Ошибка подключения к Weaviate: {e}")
            print("Убедитесь, что Weaviate запущен. Для запуска используйте Docker:")
            print("docker run -d -p 8080:8080 -p 50051:50051 semitechnologies/weaviate:latest")
            raise

    def create_schema(self, vector_dim: int = 768):
        """
        Создает схему коллекции в Weaviate

        Args:
            vector_dim: Размерность векторов эмбеддингов
        """
        if self.client is None:
            raise ValueError("Необходимо сначала подключиться к Weaviate")

        try:
            # Удаляем коллекцию если существует
            if self.client.collections.exists(self.collection_name):
                print(f"Удаление существующей коллекции {self.collection_name}...")
                self.client.collections.delete(self.collection_name)

            # Создаем новую коллекцию
            print(f"Создание коллекции {self.collection_name}...")
            
            self.client.collections.create(
                name=self.collection_name,
                properties=[
                    Property(name="text", data_type=DataType.TEXT),
                    Property(name="lecture_id", data_type=DataType.TEXT),
                    Property(name="lecture_hash", data_type=DataType.TEXT),
                    Property(name="filename", data_type=DataType.TEXT),
                    Property(name="chunk_id", data_type=DataType.INT),
                    Property(name="word_start", data_type=DataType.INT),
                    Property(name="word_end", data_type=DataType.INT),
                    Property(name="total_words", data_type=DataType.INT),
                ],
                vector_config=[
                    Configure.Vectors.self_provided(
                        name="default",
                        vector_index_config=Configure.VectorIndex.hnsw(
                            distance_metric=VectorDistances.COSINE,
                            vector_cache_max_objects=100000,
                            ef_construction=200
                        )
                    )
                ]
            )
            print("Схема создана успешно!")

        except Exception as e:
            print(f"Ошибка при создании схемы: {e}")
            raise

    def index_chunks(self, chunks: List[Dict], batch_size: int = 100):
        """
        Индексирует чанки в Weaviate

        Args:
            chunks: Список чанков с эмбеддингами
            batch_size: Размер батча для загрузки
        """
        if self.client is None:
            raise ValueError("Необходимо сначала подключиться к Weaviate")

        collection = self.client.collections.get(self.collection_name)

        print(f"Загрузка {len(chunks)} чанков в Weaviate...")

        for i in tqdm(range(0, len(chunks), batch_size), desc="Индексация"):
            batch = chunks[i:i + batch_size]

            with collection.batch.dynamic() as batch_obj:
                for chunk in batch:
                    properties = {
                        "text": chunk['text'],
                        "lecture_id": chunk['lecture_id'],
                        "lecture_hash": chunk['lecture_hash'],
                        "filename": chunk['filename'],
                        "chunk_id": chunk['chunk_id'],
                        "word_start": chunk['word_start'],
                        "word_end": chunk['word_end'],
                        "total_words": chunk['total_words']
                    }

                    batch_obj.add_object(
                        properties=properties,
                        vector=chunk['embedding']
                    )

        print("Индексация завершена!")

    def close(self):
        """Закрывает подключение к Weaviate"""
        if self.client:
            self.client.close()


class RAGRetriever:
    """Класс для поиска релевантных чанков"""
    HYBRID_SEARCH_ALPHA = 0.7

    def __init__(self, weaviate_indexer: WeaviateIndexer, embedding_model: EmbeddingModel):
        """
        Инициализация retriever

        Args:
            weaviate_indexer: Индексер Weaviate
            embedding_model: Модель для векторизации запросов
        """
        self.indexer = weaviate_indexer
        self.embedding_model = embedding_model

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Находит top_k наиболее релевантных чанков для запроса

        Args:
            query: Текст запроса
            top_k: Количество результатов для возврата

        Returns:
            Список найденных чанков с метаданными и скорами
        """
        if self.indexer.client is None:
            raise ValueError("Необходимо сначала подключиться к Weaviate")

        # Векторизуем запрос
        query_embedding = self.embedding_model.model.encode(
            query,
            prompt_name="query",  # Используем специальный промпт для запросов
            convert_to_numpy=True
        )

        # Поиск в Weaviate
        collection = self.indexer.client.collections.get(self.indexer.collection_name)
        
        # Берем гибридные варианты
        results = collection.query.hybrid(
            query=query,
            vector=query_embedding.tolist(),
            alpha=self.HYBRID_SEARCH_ALPHA,
            limit=top_k,
            return_properties=["text", "lecture_id", "filename", "chunk_id", "lecture_hash"],
            return_metadata=["score"]
        )

        # Форматируем результаты
        retrieved_chunks = []
        for item in results.objects:
            retrieved_chunks.append({
                "text": item.properties.get("text"),
                "lecture_id": item.properties.get("lecture_id"),
                "filename": item.properties.get("filename"),
                "chunk_id": item.properties.get("chunk_id"),
                "lecture_hash": item.properties.get("lecture_hash"),
                "score": item.metadata.score
            })

        return retrieved_chunks

    def display_results(self, results: List[Dict]):
        """Красиво отображает результаты поиска"""
        print("\n" + "="*80)
        print(f"Найдено {len(results)} релевантных фрагментов:")
        print("="*80 + "\n")

        for i, result in enumerate(results, 1):
            print(f"#{i} | Скор: {result['score']:.4f} | {result['filename']} (chunk {result['chunk_id']})")
            print("-" * 80)
            print(result['text'][:300] + "..." if len(result['text']) > 300 else result['text'])
            print("\n")


def main():
    """Основная функция пайплайна"""

    # Настройки
    TRANSCRIPTS_FOLDER = r"/tmp/json_lectures"
    CHUNK_SIZE = 500  # слов
    OVERLAP = 50  # слов
    WEAVIATE_URL = "http://80.90.190.85:8080"

    print("="*80)
    print("RAG PIPELINE ДЛЯ ЛЕКЦИЙ")
    print("="*80 + "\n")

    # 1. Загрузка лекций
    print("\n[Шаг 1/6] Загрузка лекций из JSON...")
    processor = LectureProcessor(TRANSCRIPTS_FOLDER)
    lectures = processor.load_lectures()

    # 2. Разбиение на чанки
    print("\n[Шаг 2/6] Разбиение текстов на чанки...")
    chunker = TextChunker(chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    chunks = chunker.process_lectures(lectures)

    # 3. Векторизация
    print("\n[Шаг 3/6] Векторизация чанков...")
    embedding_model = EmbeddingModel(device='cpu')
    chunks_with_embeddings = embedding_model.encode_chunks(chunks)

    # 4. Подключение к Weaviate и создание схемы
    print("\n[Шаг 4/6] Подключение к Weaviate...")
    indexer = WeaviateIndexer(url=WEAVIATE_URL)
    indexer.connect()

    print("\n[Шаг 5/6] Создание схемы...")
    indexer.create_schema(vector_dim=len(chunks_with_embeddings[0]['embedding']))

    # 5. Индексация
    print("\n[Шаг 6/6] Индексация в Weaviate...")
    indexer.index_chunks(chunks_with_embeddings)

    # 6. Инициализация Retriever
    print("\n" + "="*80)
    print("ПАЙПЛАЙН УСПЕШНО ЗАВЕРШЕН!")
    print("="*80 + "\n")

    retriever = RAGRetriever(indexer, embedding_model)

    # Примеры поиска
    print("\nПримеры поиска:\n")

    test_queries = [
        "Что такое машинное обучение?",
        "Какие бывают типы признаков?",
        "Расскажи про регрессию"
    ]

    for query in test_queries:
        print(f"\nЗапрос: '{query}'")
        results = retriever.retrieve(query, top_k=3)
        retriever.display_results(results)

    # Закрываем подключение
    indexer.close()
    print("\nРабота завершена!")


if __name__ == "__main__":
    main()
