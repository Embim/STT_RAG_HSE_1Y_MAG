"""
Интерактивный поиск по проиндексированным лекциям
Использует уже созданную базу Weaviate
"""

from rag_pipeline import EmbeddingModel, WeaviateIndexer, RAGRetriever


def main():
    """Интерактивный поиск по лекциям"""

    print("=" * 80)
    print("ИНТЕРАКТИВНЫЙ ПОИСК ПО ЛЕКЦИЯМ")
    print("=" * 80)
    print("\nПодключение к системе...")

    # Инициализация
    try:
        embedding_model = EmbeddingModel()
        indexer = WeaviateIndexer(url="http://localhost:8080")
        indexer.connect()
        retriever = RAGRetriever(indexer, embedding_model)
        print("✓ Система готова к работе!\n")
    except Exception as e:
        print(f"\n✗ Ошибка при инициализации: {e}")
        print("\nУбедитесь что:")
        print("1. Weaviate запущен (docker run ...)")
        print("2. Пайплайн был запущен хотя бы один раз (python rag_pipeline.py)")
        return

    print("Введите ваш запрос или 'exit' для выхода")
    print("Примеры запросов:")
    print("  - Что такое машинное обучение?")
    print("  - Какие бывают типы признаков?")
    print("  - Расскажи про нейронные сети")
    print("-" * 80)

    while True:
        try:
            # Получаем запрос от пользователя
            query = input("\n🔍 Ваш запрос: ").strip()

            if not query:
                continue

            if query.lower() in ['exit', 'quit', 'выход']:
                print("\nДо свидания!")
                break

            # Поиск
            print("\nПоиск...")
            results = retriever.retrieve(query, top_k=5)

            # Вывод результатов
            retriever.display_results(results)

        except KeyboardInterrupt:
            print("\n\nПрервано пользователем. До свидания!")
            break
        except Exception as e:
            print(f"\n✗ Ошибка при поиске: {e}")

    # Закрываем соединение
    indexer.close()


if __name__ == "__main__":
    main()
