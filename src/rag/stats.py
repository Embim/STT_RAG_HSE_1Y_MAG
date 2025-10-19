"""
Статистика по проиндексированным лекциям
"""

import weaviate
from collections import Counter


def get_stats():
    """Получает и выводит статистику по базе"""

    print("=" * 80)
    print("СТАТИСТИКА ПО БАЗЕ ЛЕКЦИЙ")
    print("=" * 80)

    try:
        # Подключение
        print("\nПодключение к Weaviate...")
        client = weaviate.connect_to_local()
        print("✓ Подключено\n")

        # Получаем коллекцию
        collection = client.collections.get("LectureChunks")

        # Общее количество чанков
        print("📊 ОБЩАЯ СТАТИСТИКА")
        print("-" * 80)

        # Получаем все объекты
        all_objects = collection.query.fetch_objects(limit=10000)

        total_chunks = len(all_objects.objects)
        print(f"Всего чанков в базе: {total_chunks}")

        if total_chunks == 0:
            print("\n⚠ База пуста. Запустите сначала: python rag_pipeline.py")
            client.close()
            return

        # Статистика по лекциям
        lecture_ids = []
        filenames = []
        total_words = 0

        for obj in all_objects.objects:
            props = obj.properties
            lecture_ids.append(props.get('lecture_id', 'unknown'))
            filenames.append(props.get('filename', 'unknown'))
            total_words += props.get('total_words', 0)

        # Подсчет
        lecture_counter = Counter(lecture_ids)
        unique_lectures = len(lecture_counter)

        print(f"Уникальных лекций: {unique_lectures}")
        print(f"Среднее количество чанков на лекцию: {total_chunks / unique_lectures:.1f}")

        # Топ лекций по количеству чанков
        print("\nТОП-10 ЛЕКЦИЙ ПО КОЛИЧЕСТВУ ЧАНКОВ")
        print("-" * 80)

        for lecture_id, count in lecture_counter.most_common(10):
            print(f"{lecture_id}: {count} чанков")

        # Распределение размеров
        print("\n СТАТИСТИКА ПО РАЗМЕРАМ")
        print("-" * 80)

        word_counts = []
        for obj in all_objects.objects:
            props = obj.properties
            words_in_chunk = props.get('word_end', 0) - props.get('word_start', 0)
            word_counts.append(words_in_chunk)

        if word_counts:
            avg_words = sum(word_counts) / len(word_counts)
            min_words = min(word_counts)
            max_words = max(word_counts)

            print(f"Средний размер чанка: {avg_words:.1f} слов")
            print(f"Минимальный размер: {min_words} слов")
            print(f"Максимальный размер: {max_words} слов")

        # Список всех лекций
        print("\n СПИСОК ВСЕХ ЛЕКЦИЙ")
        print("-" * 80)

        unique_filenames = sorted(set(filenames))
        for i, filename in enumerate(unique_filenames, 1):
            chunks_count = lecture_counter.get(filename.replace('.json', ''), 0)
            print(f"{i:2d}. {filename} ({chunks_count} чанков)")

        print("\n" + "=" * 80)

        # Закрываем соединение
        client.close()

    except Exception as e:
        print(f"\n✗ Ошибка: {e}")
        print("\nУбедитесь что:")
        print("1. Weaviate запущен")
        print("2. База проиндексирована (python rag_pipeline.py)")


if __name__ == "__main__":
    get_stats()
