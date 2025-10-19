@echo off
chcp 65001 > nul
echo ========================================
echo RAG PIPELINE - БЫСТРЫЙ СТАРТ
echo ========================================
echo.

echo [1/3] Проверка Weaviate...
curl -s http://localhost:8080/v1/meta > nul 2>&1
if %errorlevel% neq 0 (
    echo ✗ Weaviate не запущен
    echo.
    echo Запустите Weaviate одной из команд:
    echo   docker run -d -p 8080:8080 -p 50051:50051 semitechnologies/weaviate:latest
    echo   docker-compose up -d
    echo.
    pause
    exit /b 1
)
echo ✓ Weaviate работает

echo.
echo [2/3] Проверка зависимостей...
python -c "import sentence_transformers, weaviate, tqdm" 2>nul
if %errorlevel% neq 0 (
    echo ✗ Не все зависимости установлены
    echo Установка зависимостей...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo ✗ Ошибка установки
        pause
        exit /b 1
    )
)
echo ✓ Зависимости установлены

echo.
echo [3/3] Запуск пайплайна...
python rag_pipeline.py

echo.
echo ========================================
echo Готово! Теперь можете использовать:
echo   python interactive_search.py - интерактивный поиск
echo   python stats.py - статистика по базе
echo ========================================
pause
