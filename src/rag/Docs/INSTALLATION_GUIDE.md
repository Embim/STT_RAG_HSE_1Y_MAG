# Полное руководство по установке RAG Pipeline

## Предварительные требования

### 1. Проверка Python

```bash
python --version
```

Требуется Python 3.8 или выше. Если не установлен:
- Windows: https://www.python.org/downloads/
- Linux: `sudo apt install python3 python3-pip`
- Mac: `brew install python3`

### 2. Проверка Docker

```bash
docker --version
```

Если Docker не установлен:
- Windows/Mac: https://www.docker.com/products/docker-desktop
- Linux: `sudo apt install docker.io && sudo systemctl start docker`

### 3. Проверка Git (опционально)

```bash
git --version
```

Для клонирования репозитория. Если не установлен:
- Windows: https://git-scm.com/download/win
- Linux: `sudo apt install git`
- Mac: `brew install git`

## Шаг 1: Получение проекта

### Вариант A: Клонирование репозитория
```bash
git clone <repository-url>
cd <project-folder>
```

### Вариант B: Скачивание архива
1. Скачайте ZIP архив проекта
2. Распакуйте в удобную папку
3. Откройте терминал в этой папке

## Шаг 2: Проверка структуры проекта

Убедитесь что присутствуют следующие файлы:

```bash
ls -la  # Linux/Mac
dir     # Windows
```

Должны быть файлы:
- `rag_pipeline.py` - основной скрипт
- `requirements.txt` - зависимости
- `docker-compose.yml` - конфигурация Docker
- `transcripted_text/` - папка с лекциями

## Шаг 3: Создание виртуального окружения (рекомендуется)

### Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### Linux/Mac:
```bash
python3 -m venv venv
source venv/bin/activate
```

После активации в начале строки должно появиться `(venv)`.

## Шаг 4: Установка Python зависимостей

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Это может занять 5-10 минут. Будут установлены:
- sentence-transformers (модель эмбеддингов)
- weaviate-client (клиент БД)
- transformers (трансформерные модели)
- tqdm (прогресс-бары)
- numpy, scikit-learn (утилиты)

### Возможные проблемы:

**Ошибка: "No module named 'pip'"**
```bash
python -m ensurepip --upgrade
```

**Ошибка: "Could not build wheels for..."**
```bash
# Установите build tools
# Windows: Установите Visual Studio Build Tools
# Linux: sudo apt install python3-dev build-essential
# Mac: xcode-select --install
```

**Timeout при загрузке**
```bash
pip install -r requirements.txt --default-timeout=100
```

## Шаг 5: Запуск Weaviate

### Вариант A: Docker Compose (рекомендуется)

```bash
docker-compose up -d
```

Проверка:
```bash
docker-compose ps
```

Должен быть статус "Up".

### Вариант B: Прямая команда Docker

```bash
docker run -d \
  --name weaviate \
  -p 8080:8080 \
  -p 50051:50051 \
  semitechnologies/weaviate:latest
```

Проверка:
```bash
docker ps
```

### Вариант C: Weaviate Cloud

Если не можете запустить локально:
1. Зарегистрируйтесь на https://console.weaviate.cloud
2. Создайте кластер
3. Получите URL и API ключ
4. Измените в коде:
   ```python
   indexer = WeaviateIndexer(url="https://your-cluster.weaviate.network")
   ```

### Проверка работы Weaviate

Откройте в браузере:
```
http://localhost:8080/v1/meta
```

Или через curl:
```bash
curl http://localhost:8080/v1/meta
```

Должен вернуться JSON с информацией о Weaviate.

### Возможные проблемы:

**Порт 8080 занят**
```bash
# Найти процесс на порту
# Windows: netstat -ano | findstr :8080
# Linux: lsof -i :8080

# Изменить порт в docker-compose.yml:
ports:
  - "8081:8080"  # Используем 8081 вместо 8080
```

**Docker не запущен**
```bash
# Windows: Запустите Docker Desktop
# Linux: sudo systemctl start docker
# Mac: Откройте Docker Desktop
```

**Permission denied (Linux)**
```bash
sudo docker-compose up -d
# или добавьте пользователя в группу docker:
sudo usermod -aG docker $USER
# Перелогиньтесь
```

## Шаг 6: Первый запуск - индексация

### Автоматический запуск

**Windows:**
```bash
quickstart.bat
```

**Linux/Mac:**
```bash
chmod +x quickstart.sh
./quickstart.sh
```

### Ручной запуск

```bash
python rag_pipeline.py
```

### Что происходит:

1. **Загрузка лекций** (~5 сек)
   - Читаются все JSON файлы
   - Парсится структура

2. **Разбиение на чанки** (~10 сек)
   - Текст делится на фрагменты
   - Добавляются метаданные

3. **Загрузка модели эмбеддингов** (~30 сек первый раз)
   - Скачивается модель Qwen3-Embedding (~2.5 GB)
   - Кешируется в `~/.cache/huggingface/`

4. **Векторизация** (~5-10 мин)
   - Каждый чанк превращается в вектор
   - Показывается прогресс-бар

5. **Индексация в Weaviate** (~30 сек)
   - Векторы загружаются в БД
   - Создается индекс

### Возможные проблемы:

**Out of Memory**
```python
# Уменьшите batch_size в rag_pipeline.py:
chunks_with_embeddings = embedding_model.encode_chunks(chunks, batch_size=16)
```

**Slow download**
```python
# Модель скачивается с HuggingFace
# Если медленно - используйте VPN или зеркало
```

**Connection refused to Weaviate**
```bash
# Проверьте что Weaviate запущен:
docker-compose ps
# Перезапустите:
docker-compose restart
```

## Шаг 7: Проверка работы

### 1. Проверка статистики

```bash
python stats.py
```

Должна показаться статистика:
- Количество чанков
- Количество лекций
- Размеры чанков

### 2. Интерактивный поиск

```bash
python interactive_search.py
```

Введите запрос, например:
```
🔍 Ваш запрос: Что такое машинное обучение?
```

Должны появиться результаты с найденными фрагментами.

### 3. Примеры использования

```bash
python examples.py
```

Выберите один из 6 примеров для тестирования.

## Шаг 8: Настройка (опционально)

### Изменение параметров chunking

Отредактируйте `rag_pipeline.py`:

```python
# Строка ~325
CHUNK_SIZE = 500    # Измените на нужное значение
OVERLAP = 50        # Измените перекрытие
```

Перезапустите индексацию:
```bash
python rag_pipeline.py
```

### Использование GPU

Если у вас есть NVIDIA GPU:

1. Установите CUDA toolkit
2. Установите PyTorch с GPU:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```
3. Векторизация автоматически использует GPU

Проверка:
```python
import torch
print(torch.cuda.is_available())  # Должно быть True
```

### Ускорение с Flash Attention

```bash
pip install flash-attn --no-build-isolation
```

Раскомментируйте в `rag_pipeline.py`:
```python
model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-0.6B",
    model_kwargs={"attn_implementation": "flash_attention_2"},
)
```

## Шаг 9: Ежедневное использование

После первой установки для работы нужно:

### Запуск системы:
```bash
# 1. Активировать окружение (если используется)
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 2. Убедиться что Weaviate запущен
docker-compose ps

# 3. Использовать систему
python interactive_search.py
```

### Остановка системы:
```bash
# Остановить Weaviate (данные сохраняются)
docker-compose stop

# Деактивировать окружение
deactivate
```

### Полная очистка:
```bash
# Удалить все данные Weaviate
docker-compose down -v

# Удалить виртуальное окружение
rm -rf venv  # Linux/Mac
rmdir /s venv  # Windows
```

## Troubleshooting

### Общие проблемы

**Импорт не работает**
```bash
# Убедитесь что находитесь в правильной папке
pwd  # Linux/Mac
cd   # Windows

# Убедитесь что окружение активировано
which python  # Должен быть путь к venv
```

**Медленная работа**
- Используйте GPU если доступен
- Уменьшите batch_size
- Закройте другие программы

**Ошибки кодировки (Windows)**
```python
# Добавьте в начало скриптов:
# -*- coding: utf-8 -*-
```

### Логи и отладка

**Просмотр логов Weaviate:**
```bash
docker-compose logs -f
```

**Verbose режим Python:**
```bash
python -v rag_pipeline.py
```

**Проверка импортов:**
```python
python -c "import sentence_transformers, weaviate, tqdm; print('OK')"
```

## Следующие шаги

После успешной установки:

1. Прочитайте [README.md](README.md) для подробной документации
2. Изучите [examples.py](examples.py) для примеров использования
3. Ознакомьтесь с [ARCHITECTURE.md](ARCHITECTURE.md) для понимания устройства
4. Используйте [QUICKREF.md](QUICKREF.md) как шпаргалку

## Получение помощи

Если возникли проблемы:

1. Проверьте раздел Troubleshooting выше
2. Изучите документацию проекта
3. Проверьте логи: `docker-compose logs`
4. Убедитесь что все зависимости установлены: `pip list`

## Контрольный список установки

- [ ] Python 3.8+ установлен
- [ ] Docker установлен и запущен
- [ ] Проект скачан
- [ ] Виртуальное окружение создано
- [ ] Зависимости установлены (`pip install -r requirements.txt`)
- [ ] Weaviate запущен (`docker-compose up -d`)
- [ ] Weaviate отвечает (http://localhost:8080/v1/meta)
- [ ] Индексация выполнена (`python rag_pipeline.py`)
- [ ] Поиск работает (`python interactive_search.py`)

Если все пункты выполнены - установка завершена успешно!
