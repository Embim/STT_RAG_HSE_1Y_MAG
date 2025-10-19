# RAG Pipeline - Навигация по проекту

## Быстрый старт

Хотите быстро начать? Читайте в таком порядке:

1. **[INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md)** - пошаговая установка (15 мин)
2. **[QUICKREF.md](QUICKREF.md)** - шпаргалка по командам (2 мин)
3. Запустите `python interactive_search.py` - пробуйте!

## Документация

### Для начинающих

| Файл | Описание | Время чтения |
|------|----------|--------------|
| [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md) | Полное руководство по установке | 15 мин |
| [README.md](README.md) | Основная документация проекта | 20 мин |
| [QUICKREF.md](QUICKREF.md) | Быстрая справка | 5 мин |

### Для разработчиков

| Файл | Описание | Время чтения |
|------|----------|--------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Архитектура системы | 20 мин |
| [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) | Сводка по проекту | 10 мин |
| [examples.py](examples.py) | Примеры использования API | Практика |

## Исходный код

### Основные модули

| Файл | Описание | Строки кода |
|------|----------|-------------|
| [rag_pipeline.py](rag_pipeline.py) | Основной пайплайн, все классы | ~450 |
| [interactive_search.py](interactive_search.py) | Интерактивный CLI поиск | ~70 |
| [stats.py](stats.py) | Статистика по базе | ~100 |
| [examples.py](examples.py) | 6 примеров использования | ~300 |

### Вспомогательные файлы

| Файл | Описание |
|------|----------|
| [RAG.py](RAG.py) | Демо работы с эмбеддингами |
| [requirements.txt](requirements.txt) | Python зависимости |
| [docker-compose.yml](docker-compose.yml) | Конфигурация Weaviate |
| [.gitignore](.gitignore) | Git исключения |

### Скрипты

| Файл | Описание | Платформа |
|------|----------|-----------|
| [quickstart.bat](quickstart.bat) | Автоматический запуск | Windows |
| [quickstart.sh](quickstart.sh) | Автоматический запуск | Linux/Mac |

## Структура проекта

```
RAG Pipeline/
│
├── 📁 transcripted_text/         Входные данные (40 JSON файлов)
│
├── 🐍 Основной код
│   ├── rag_pipeline.py           Главный модуль
│   ├── interactive_search.py     CLI поиск
│   ├── stats.py                  Статистика
│   └── examples.py               Примеры
│
├── 📚 Документация
│   ├── INDEX.md                  Этот файл (навигация)
│   ├── README.md                 Основная документация
│   ├── INSTALLATION_GUIDE.md     Руководство по установке
│   ├── QUICKREF.md               Быстрая справка
│   ├── ARCHITECTURE.md           Архитектура
│   └── PROJECT_SUMMARY.md        Сводка проекта
│
├── ⚙️ Конфигурация
│   ├── requirements.txt          Python зависимости
│   ├── docker-compose.yml        Docker конфигурация
│   └── .gitignore                Git исключения
│
└── 🚀 Утилиты
    ├── quickstart.bat            Быстрый старт (Windows)
    └── quickstart.sh             Быстрый старт (Linux/Mac)
```

## Сценарии использования

### Сценарий 1: Первая установка

```
1. Читаю: INSTALLATION_GUIDE.md
2. Устанавливаю зависимости
3. Запускаю: quickstart.bat (или .sh)
4. Пробую: python interactive_search.py
```

### Сценарий 2: Понимание системы

```
1. Читаю: README.md (обзор)
2. Читаю: ARCHITECTURE.md (детали)
3. Изучаю: rag_pipeline.py (код)
4. Экспериментирую: examples.py
```

### Сценарий 3: Интеграция в свой проект

```
1. Читаю: examples.py (примеры API)
2. Импортирую классы из rag_pipeline.py
3. Адаптирую под свои данные
4. Использую RAGRetriever для поиска
```

### Сценарий 4: Ежедневное использование

```
1. Запускаю: docker-compose up -d
2. Использую: python interactive_search.py
3. Или: импортирую RAGRetriever в свой код
4. По окончании: docker-compose stop
```

## FAQ - Какой файл читать?

### "Как установить систему?"
→ [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md)

### "Как быстро начать работу?"
→ [QUICKREF.md](QUICKREF.md)

### "Как использовать в своем коде?"
→ [examples.py](examples.py)

### "Как работает система внутри?"
→ [ARCHITECTURE.md](ARCHITECTURE.md)

### "Какие есть возможности?"
→ [README.md](README.md)

### "Где настройки и параметры?"
→ [rag_pipeline.py](rag_pipeline.py) строки 320-330

### "Как добавить свои данные?"
→ [README.md](README.md) → "Пример 2: Обработка новых лекций"

### "У меня проблема при запуске"
→ [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md) → Troubleshooting

### "Какую модель эмбеддингов использовать?"
→ [README.md](README.md) → "Модель эмбеддингов"

### "Как улучшить производительность?"
→ [README.md](README.md) → "Troubleshooting" → "Медленная работа"

## Обучающий трек

### Уровень 1: Пользователь (1-2 часа)
1. [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md) - установка
2. [QUICKREF.md](QUICKREF.md) - основные команды
3. `python interactive_search.py` - практика

**Результат:** Умеете искать информацию в лекциях

### Уровень 2: Интегратор (3-4 часа)
1. [README.md](README.md) - полная документация
2. [examples.py](examples.py) - изучение API
3. Создание своего скрипта с RAGRetriever

**Результат:** Умеете интегрировать в свой проект

### Уровень 3: Разработчик (8-10 часов)
1. [ARCHITECTURE.md](ARCHITECTURE.md) - архитектура
2. [rag_pipeline.py](rag_pipeline.py) - изучение кода
3. Модификация chunking стратегии
4. Добавление новых фич

**Результат:** Понимаете систему изнутри и можете расширять

## Шпаргалка по файлам

### Хочу быстро начать
```bash
QUICKREF.md → quickstart.bat → interactive_search.py
```

### Хочу разобраться
```bash
README.md → ARCHITECTURE.md → rag_pipeline.py
```

### Хочу интегрировать
```bash
examples.py → rag_pipeline.py (API) → свой код
```

### Хочу настроить
```bash
rag_pipeline.py (параметры) → requirements.txt (зависимости)
```

## Размер файлов

| Категория | Файлы | Общий размер |
|-----------|-------|--------------|
| Документация | 6 файлов | ~100 KB |
| Код | 5 файлов | ~50 KB |
| Конфигурация | 4 файла | ~5 KB |
| **Всего** | **15 файлов** | **~155 KB** |

*Не считая данных в transcripted_text/ (~20 MB)*

## Полезные ссылки

### Внутренние
- [README.md](README.md) - главная документация
- [QUICKREF.md](QUICKREF.md) - шпаргалка
- [ARCHITECTURE.md](ARCHITECTURE.md) - архитектура

### Внешние
- [Weaviate Docs](https://weaviate.io/developers/weaviate)
- [Sentence Transformers](https://www.sbert.net/)
- [Qwen3 Model Card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)

## Что дальше?

После освоения базового функционала:

1. **Экспериментируйте** с параметрами chunking
2. **Добавьте** свои данные
3. **Интегрируйте** с LLM для генерации ответов
4. **Создайте** web-интерфейс (Streamlit/Gradio)
5. **Оптимизируйте** под свой use case

## Обратная связь

Нашли ошибку в документации?
- Проверьте другие файлы документации
- Изучите [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md) → Troubleshooting
- Посмотрите примеры в [examples.py](examples.py)

---

**Совет:** Добавьте этот файл в закладки как точку входа в проект!
