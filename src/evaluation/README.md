# Evaluation harness

Считает качество ASR-моделей через WER/CER на публичных бенчмарках (Common Voice ru, FLEURS) и качество RAG-ответов через RAGAS на курированном Q&A наборе. Результаты пишутся параллельно в Langfuse и в CSV под `data/eval/results/`.

## Установка

```bash
uv sync --extra eval
docker compose --profile full --profile langfuse up -d
```

В `.env` к существующим RAG-настройкам добавляются `WHISPER_URL` (OpenAI-совместимый эндпоинт транскрипции), `ASR_NAME` (метка модели для отчётов), `ASR_MODEL_ID`, `ASR_LANGUAGE`, и опционально `RAGAS_JUDGE_MODEL` — отдельная LLM-судья, иначе берётся `LLM_MODEL`.

## ASR

Логика сравнения моделей последовательная: выставил `WHISPER_URL` и `ASR_NAME` в `.env`, прогнал, поменял переменные, прогнал снова, сам сравнил два CSV.

```bash
python -m evaluation.asr.runner --max-samples 50 --langfuse-dataset asr-cv-ru
```

Ранер прогоняет текущий эндпоинт против `ASR_BENCHMARK` (по умолчанию Common Voice ru), считает WER/CER/MER/WIL и складывает результат в CSV и Langfuse. Флаг `--transcribe-dir <path>` переключает в режим без метрик: транскрибирует все аудио из папки в JSON-файлы, готовые для `src/downloader/ingest.py`.

### Какие датасеты можно использовать

Бенчмарки берутся через HuggingFace `datasets` в стриминговом режиме и кэшируются в `data/eval/benchmarks/<dataset>/<lang>/*.wav`. По умолчанию — `mozilla-foundation/common_voice_17_0` с конфигом `ru`, сплитом `test`. Common Voice — gated dataset: чтобы он скачивался, нужно один раз принять условия на странице датасета (https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0) и залогиниться через `huggingface-cli login` (или экспортировать `HF_TOKEN` в окружение). Если не хочется возиться с gated-датасетами, есть FLEURS — он открытый, ставится одной командой:

```bash
python -m evaluation.asr.runner --benchmark google/fleurs --lang ru_ru --max-samples 50
```

Ещё рабочие варианты для русского ASR с HF: `bond005/sberdevices_golos_10h_crowd` (Sber Golos, открытый), `bond005/sova_rudevices` (SOVA), `mozilla-foundation/common_voice_13_0` (CV постарше, тоже gated). Любой датасет, у которого в записи есть поля `audio` (с `array` и `sampling_rate`) и текстовое поле `sentence`/`text`/`transcription`/`raw_transcription`, подключается без правок кода — нужно только указать `--benchmark <hf_id> --lang <config> --split <split>`. Поменять умолчания насовсем — через `ASR_BENCHMARK`, `ASR_BENCHMARK_LANG`, `ASR_BENCHMARK_SPLIT` в `.env`. Кэш разово занимает несколько сот мегабайт; повторные прогоны идут с диска, без сети.

### Свой бенчмарк из видео/аудио лекций

Когда хочется померить качество ASR именно на своём домене (например, на лекциях курса), сборка бенчмарка вынесена в одноразовый helper-скрипт [scripts/build_asr_benchmark.py](../../scripts/build_asr_benchmark.py) — он не часть runtime-кода, дёргается руками когда нужно подготовить данные. Два режима. Если уже есть JSON-транскрипт с сегментами и таймкодами (формат faster-whisper-server: `{segments: [{text, start, end}, ...]}`), `ffmpeg` сам нарежет аудио по сегментам:

```bash
python scripts/build_asr_benchmark.py from-json \
    --json E:/video_pipeline/transcripts/lecture_01.json \
    --audio E:/video_pipeline/audio/lecture_01.mp3 \
    --output data/eval/benchmarks/local/lecture_01 \
    --max-segments 30
```

Получится директория с `manifest.json` и подпапкой `audio/`. Reference-текст берётся из самого JSON; если он сделан старой ASR-моделью и в нём есть галлюцинации — открой `manifest.json`, поправь те `reference`, что заметишь, и поставь `verified: true` на годных. Дальше прогон через runner ровно как для HF-бенчмарка, только источник меняется:

```bash
python -m evaluation.asr.runner --local-benchmark \
    data/eval/benchmarks/local/lecture_01/manifest.json \
    --max-samples 30 --only-verified
```

Альтернатива — порезать аудио руками (например, в Audacity на естественных границах фраз) и подложить CSV `filename,reference`: `python scripts/build_asr_benchmark.py from-folder --audio-dir <cut> --references-csv <refs.csv> --output <dir>`. В этом случае все айтемы автоматически получают `verified: true` (раз ты их сам подписал — значит, проверял). Локальный бенчмарк так же привязывается к Langfuse через `--langfuse-dataset`, как и HF.

### Self-evaluation leak: важный момент

Если JSON-транскрипт, по которому собран бенчмарк, сделан той же ASR-моделью, которую ты потом тестируешь — WER будет ≈0 по построению (модель сравнивает свой вывод со своим же). Чтобы этого не пропустить, манифест хранит поле `reference_source` (имя ASR-движка, который произвёл reference-тексты), а runner перед прогоном проверяет: если `backend.name == reference_source`, печатает большой `WARNING` про утечку и подсказывает поменять backend или включить `--only-verified`. Прогон при этом не блокируется — твоё право.

### Honest human-recorded бенчмарк

Чтобы получить совсем честную оценку (без утечки и без курации часов аудио), можно дать друзьям короткий текст для начитки — reference написан до записи и не зависит ни от какой ASR. Helper'ы для этого:

```bash
# 1. Сгенерировать скрипт для начитки (50 разнообразных фраз из существующих манифестов).
#    Можно скармливать несколько источников через повторяющиеся --from-manifest.
uv run python scripts/make_recording_script.py \
    --from-manifest data/eval/benchmarks/local/recsys_01_lecture/manifest.json \
    --from-manifest data/eval/benchmarks/local/recsys_02_lecture/manifest.json \
    --count 50 \
    --output data/eval/recordings/shared
# → script.txt (для друзей) и script.csv (для нашего pipeline'а)

# 2. Отправить data/eval/recordings/shared/script.txt друзьям. Каждый записывает
#    голосовые в Telegram по одной фразе на сообщение, в порядке нумерации.

# 3. Скачать голосовые из Telegram (см. ниже) и положить в
#    data/eval/recordings/<имя_друга>/audio/ под именами 001.ogg, 002.ogg, …

# 4. Собрать манифест:
uv run python scripts/build_asr_benchmark.py from-folder \
    --audio-dir       data/eval/recordings/aleks/audio \
    --references-csv  data/eval/recordings/shared/script.csv \
    --output          data/eval/benchmarks/local/human_aleks \
    --reference-source "human:aleks"

# 5. Прогон — теперь без leak'а:
uv run python -m evaluation.asr.runner \
    --local-benchmark data/eval/benchmarks/local/human_aleks/manifest.json \
    --langfuse-dataset asr-human-recorded
```

`script.csv` содержит filename без расширения (`001`, `002`, ...), а `from-folder` сам ищет в папке `001.ogg` / `001.m4a` / `001.mp3` / любой другой поддерживаемый аудиоформат. Так что друзья могут присылать в любом виде — Telegram даёт `.ogg/opus`, диктофон iPhone — `.m4a`, всё равно подхватится.

### Как скачивать голосовые из Telegram

Самый простой способ — Telegram Desktop: правой кнопкой на голосовом сообщении → **Save Voice Message As...** → файл сохраняется как `.ogg`. Переименуй в `001.ogg`, `002.ogg`, … соответственно нумерации в `script.txt`. На большой объём это нудно, но для 50 голосовых на одного друга — терпимо за 5-10 минут.

Если друзей много и хочется автоматизации — можно через Telethon (MTProto-клиент) скачать всю историю чата с другом одним скриптом. Нужны только API_ID и API_HASH с https://my.telegram.org. В нашем стеке такого скрипта нет, но добавить тривиально — попроси.

Telegram воиз идут как Opus в OGG-контейнере, 48kHz mono. faster-whisper-server и ему подобные принимают это формат напрямую, ничего конвертировать не нужно.

## RAG

Курированный тестсет уже лежит в [data/eval/testsets/recsys_v1.json](../data/eval/testsets/recsys_v1.json) — 117 вопросов по лекциям 1, 2 и 4 курса по RecSys, все `verified=true`. Сами транскрипты этих лекций лежат в [data/transcripts/recsys/](../data/transcripts/recsys/). Перед первым прогоном их нужно один раз залить в Weaviate, затем запускается ранер:

```bash
python -m downloader.ingest --from-dir data/transcripts/recsys
python -m evaluation.rag.runner --testset data/eval/testsets/recsys_v1.json \
    --run-name baseline --langfuse-dataset stt-rag-recsys-v1
```

Ранер для каждого вопроса дёргает основной `pipeline.run()`, отдаёт ответ и retrieved-документы в RAGAS, считает faithfulness/answer relevancy/context precision/recall/answer correctness через LLM-судью и пушит всё в CSV и Langfuse Scores. В Langfuse трейсится весь конвейер: прогоны pipeline (rewrite → retrieve → answer), внутри `ragas-evaluate` для каждого вопроса есть свой sub-span `item-<item_id>` с вложенными вызовами судьи (через `langfuse.langchain.CallbackHandler`), плюс дата-сет run items и скоры. Скоры привязываются к trace_id того pipeline-прогона, по которому они посчитаны — открыв любой trace `rag-pipeline` в UI, ты увидишь его метрики прямо там; корпусные агрегаты остаются стенд-алоном. Флаг `--collection` нужен, когда сравниваешь ASR-модели косвенно: транскрипты от каждой модели лежат в своей Weaviate-коллекции, а тестсет один. На 117 items × 7 метрик уходит ~820 вызовов судьи, на gpt-4o-mini это $1–2; для smoke-теста есть `--max-items 3`.

Если захочется расширить тестсет — `python -m evaluation.rag.testset --transcripts <dir> --size N` сгенерирует новые Q&A через RAGAS, дальше через `evaluation.rag.curate --to-csv` правится в Excel и `--to-json` возвращается обратно (только `verified=TRUE` строки). Автогенерация обычно даёт ~30% мусора (тривиальные вопросы, галлюцинированные `reference_answer`), курация обязательна.

## Где что

CSV-результаты лежат в `data/eval/results/asr_*.csv` и `rag_*.csv` с единой схемой `ts, run_name, kind, model, item_id, metric, value, extra` (строка с `item_id="__corpus__"` — агрегат). Тестсеты — `data/eval/testsets/*.json` (canonical) и `*.csv` (для Excel). Бенчмарки HuggingFace кэшируются в `data/eval/benchmarks/` автоматически.

## Подвох с Qwen3-ASR

Готового OpenAI-совместимого Docker-образа для Qwen3-ASR в публичных реестрах пока нет (модель открыта 29 января 2026). Любой контейнер, реализующий контракт `POST /v1/audio/transcriptions` с ответом `{"text", "segments": [...]}`, подключается через смену `WHISPER_URL` в `.env` без правок кода — но саму обёртку (например, FastAPI поверх `Qwen3ASRModel`) пока надо писать руками или ждать поддержки в vLLM/SGLang.
