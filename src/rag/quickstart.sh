#!/bin/bash

echo "========================================"
echo "RAG PIPELINE - БЫСТРЫЙ СТАРТ"
echo "========================================"
echo ""

echo "[1/3] Проверка Weaviate..."
if ! curl -s http://localhost:8080/v1/meta > /dev/null 2>&1; then
    echo "✗ Weaviate не запущен"
    echo ""
    echo "Запустите Weaviate одной из команд:"
    echo "  docker run -d -p 8080:8080 -p 50051:50051 semitechnologies/weaviate:latest"
    echo "  docker-compose up -d"
    echo ""
    exit 1
fi
echo "✓ Weaviate работает"

echo ""
echo "[2/3] Проверка зависимостей..."
if ! python -c "import sentence_transformers, weaviate, tqdm" 2>/dev/null; then
    echo "✗ Не все зависимости установлены"
    echo "Установка зависимостей..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "✗ Ошибка установки"
        exit 1
    fi
fi
echo "✓ Зависимости установлены"

echo ""
echo "[3/3] Запуск пайплайна..."
python rag_pipeline.py

echo ""
echo "========================================"
echo "Готово! Теперь можете использовать:"
echo "  python interactive_search.py - интерактивный поиск"
echo "  python stats.py - статистика по базе"
echo "========================================"
