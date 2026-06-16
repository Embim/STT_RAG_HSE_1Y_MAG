import weaviate
from weaviate.classes.config import Configure, Property, DataType

client = weaviate.connect_to_local(host="localhost", port=8080)

try:
    print("✓ Подключение к Weaviate успешно")
    
    if client.collections.exists("TestDoc"):
        client.collections.delete("TestDoc")
        print("✓ Старая коллекция удалена")
    
    collection = client.collections.create(
        name="TestDoc",
        vectorizer_config=Configure.Vectorizer.text2vec_openai(
            model="ai-forever/FRIDA",
            base_url="http://localhost:7997",  # Weaviate добавит /v1/embeddings
            vectorize_collection_name=False,
        ),
        properties=[
            Property(name="title", data_type=DataType.TEXT),
            Property(name="content", data_type=DataType.TEXT),
        ]
    )
    print("✓ Коллекция создана")
    
    test_docs = [
        {"title": "Искусственный интеллект", "content": "ИИ меняет мир технологий"},
        {"title": "Машинное обучение", "content": "ML - это подраздел искусственного интеллекта"},
        {"title": "Нейронные сети", "content": "Глубокое обучение использует нейронные сети"},
    ]
    
    collection = client.collections.get("TestDoc")
    
    print("\nДобавление документов...")
    for i, doc in enumerate(test_docs, 1):
        collection.data.insert(doc)
        print(f"  ✓ Документ {i} добавлен")
    
    print("\n--- Тест поиска ---")
    response = collection.query.near_text(query="что такое AI", limit=2)
    
    print(f"\nНайдено {len(response.objects)} результатов:\n")
    for i, obj in enumerate(response.objects, 1):
        print(f"{i}. {obj.properties['title']}")
        print(f"   {obj.properties['content']}")
        if obj.metadata.distance:
            print(f"   Расстояние: {obj.metadata.distance:.4f}\n")
    
    print("✅ Все тесты пройдены успешно!")
    
except Exception as e:
    print(f"❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()
    
finally:
    client.close()