# ASR backends — выбор моделей и инфраструктура

Документ фиксирует, как и почему отобраны ASR-модели для проекта, что в
итоге попало в `docker-compose.yml` (профили `asr-*`), что **не** попало, и
по каким критериям. Цель — чтобы при возврате к задаче через полгода не
надо было пере-проводить тот же ресёрч.

Дата актуальности: **2026-05-13**. На этой дате выпущены Qwen3-ASR
(январь 2026), Voxtral Transcribe 2 (февраль 2026), ElevenLabs Scribe v2
(март 2026), и Cohere Transcribe (март 2026).

---

## 1. Постановка задачи

Проект транскрибирует **учебные лекции по RecSys / ML / DL на русском с
плотным code-switching ru-en**: «эмбеддинг», «attention», «collaborative
filtering», «LightGBM», «BPR», «factorization machines» — в одном
предложении могут встречаться 2-3 английских термина среди русской речи.

Что мы хотим от ASR:

1. **Сохранять английские термины в латинице**, не транслитерировать в
   кириллицу. «collaborative filtering» должно остаться так, а не стать
   «коллаборэйшен филтеринг».
2. **Не ломаться на доменной лексике**. Должно быть либо корректно из
   коробки, либо через подаваемый glossary / hotwords / `initial_prompt`.
3. **Работать на одной RTX 5080 (16 GB VRAM)** в эксклюзивном режиме —
   один ASR-бэкенд за раз, остальное (Weaviate, infinity, и т.д.)
   остановлены при бенчмарке.
4. **Запускаться через docker compose** одной командой, OpenAI-compatible
   `/v1/audio/transcriptions` для единого runtime-кода (`ASRBackend`
   protocol в `src/evaluation/asr/backends/`).

Это сильно сужает выбор: моноязычные русские модели (GigaAM, podlodka)
отпадают как класс, потому что **их токенизатор физически не содержит
латиницы** и они аппаратно не могут писать «BPR» — будет «би-пи-эр»
или `[UNK]`.

---

## 2. Архитектурные классы и проблема language lock-in

Способность ASR работать с code-switching определяется **архитектурой**,
а не качеством модели на отдельном языке.

| Класс | Примеры | Поведение на code-switch |
|---|---|---|
| **Encoder-decoder с language-токеном** | Whisper, podlodka | Декодер «лочится» на язык, заданный токеном `<\|ru\|>`. Кириллица и латиница оба в словаре, но статистически декодер сильно предпочитает не переключаться. Снимается через `initial_prompt` с biased-токенами, но это костыль. |
| **Монолингвальный токенизатор** | GigaAM-v3, podlodka-turbo (русская версия) | В словаре физически нет латиницы → жёсткий lock-in. Никакие prompt'ы не помогут — модель не может выдать английский токен. |
| **LLM-decoder** | Qwen3-ASR, Phi-4-MM, Canary-Qwen, VibeVoice, Voxtral, Qwen3-Omni, GPT-4o-audio | Декодер — обычная LLM. Никакого language-токена нет. Декодер генерирует любые токены, любого языка, в любой позиции. CS работает из коробки, плюс можно дать system-prompt с инструкцией «сохраняй английские термины в латинице». |

**Вывод:** под нашу задачу нужны модели третьего класса. Первый — только
как baseline с `initial_prompt`-фокусами. Второй — выбывает.

---

## 3. Финальный список бэкендов в `docker-compose.yml`

8 профилей, все на одном порту 8000 с сетевым алиасом `asr`. Один
профиль = один backend на GPU. Runtime-код звонит на
`http://asr:8000/v1/audio/transcriptions` — кто за этим стоит, зависит
только от поднятого профиля.

| Профиль | Модель | VRAM | Лицензия | CS ru-en |
|---|---|:---:|---|---|
| `asr-whisper` | `deepdml/faster-whisper-large-v3-turbo-ct2` | ~2 GB | MIT | через `initial_prompt` |
| `asr-qwen3` | `Qwen/Qwen3-ASR-1.7B` | ~4 GB | Apache-2.0 | **native** ✅ |
| `asr-parakeet` | `nvidia/parakeet-tdt-0.6b-v3` | ~2 GB | CC-BY-4.0 | ❌ только ru baseline |
| `asr-phi4-nvfp4` | `nvidia/Phi-4-multimodal-instruct-NVFP4` | ~4 GB | NVIDIA Open + MIT | + промпт |
| `asr-vibevoice` ⭐ | `microsoft/VibeVoice-ASR-HF` (через transformers + bnb 4-bit) | ~5-6 GB | MIT | **native + hotwords + diarization** |

### Healthchecks

У всех профилей одинаковая логика проверки готовности:

```yaml
healthcheck:
  test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/v1/models', timeout=3)"]
```

`start_period` варьируется от 60 сек (whisper) до 15 мин (phi4-nvfp4) —
учитывает реальное время cold start для каждой архитектуры.

### Кастомные FastAPI-обёртки

Для бэкендов, которые не имеют готового OpenAI-compatible HTTP, в
`asr_servers/` лежат тонкие FastAPI-обёртки:

- `asr_servers/parakeet_server.py` — для NeMo Parakeet-TDT-0.6B-v3
- `asr_servers/vibevoice_server.py` — для VibeVoice-ASR-HF через
  `transformers` + bitsandbytes 4-bit, потому что vLLM-путь сломан
  (см. секцию «Что НЕ попало»). Hotwords мапятся в OpenAI `prompt` поле.

Обе подмонтированы в свои контейнеры и запускаются через `python server.py`.

---

## 4. Ranking по релевантности задаче

От самого подходящего к baseline:

1. **`asr-vibevoice`** — единственный с **родными hotwords и diarization**,
   плюс CS и 60-мин audio одним проходом. Через community 4-bit квант
   (`scerz/VibeVoice-ASR-4bit`) умещается в 5-6 GB на 5080. Запускается
   через собственный vLLM плагин (см. setup-блок ниже).
2. **`asr-qwen3`** — native CS на 30 языках через vLLM, LLM-decoder без
   language-lock. 4 GB на 5080. Главный простой кандидат на боевой backend.
3. **`asr-phi4-nvfp4`** — multimodal LLM через TensorRT-LLM (FP4). Можно
   подавать system-prompt с глоссарием терминов лекций.
4. **`asr-whisper`** — baseline encoder-decoder через `initial_prompt`-
   биас. Для понимания, насколько вообще current Whisper хуже LLM-decoder
   моделей на CS.
5. **`asr-parakeet`** — ru-only baseline без CS. Контроль «насколько
   моноязычный подход хуже» на чисто русских кусках лекций.

### Setup для `asr-vibevoice` (один раз)

VibeVoice грузится через `transformers ≥ 5.3.0` (поддерживает её нативно
из коробки), а не через vLLM. Quantization on-the-fly через bitsandbytes
4-bit (nf4 + double quant + bf16 compute) — 9B BF16 → ~5-6 GB в VRAM.

Никаких внешних склонирований не нужно — server полностью в
`asr_servers/vibevoice_server.py`, контейнер ставит deps при старте.

---

## 5. Что НЕ попало в стек и почему

| Модель | Причина |
|---|---|
| **GigaAM-v3** (Sber) | Russian-only тоkenizer → не может физически писать латиницей. На лекциях с англицизмами — `[UNK]` или транслитерация |
| **bond005/whisper-podlodka-turbo** | Сам автор пишет в карточке: «performance on code-switching speech has not been specifically evaluated». Тренировался на отдельных Ru/En датасетах, не смешанных |
| **mistralai/Voxtral-Mini-3B-2507**, **Voxtral-Small-24B-2507** | Поддерживают только 8 европейских языков (en/es/fr/pt/hi/de/nl/it). **Русского нет**, бесполезны для нашей задачи |
| **bartowski/Voxtral-*-GGUF** | Помимо отсутствия ru — llama.cpp **не умеет audio для Voxtral** (явно сказано в карточке). GGUF годится для текстовой части, аудио на вход подать нельзя |
| **argmaxinc/whisperkit-coreml** | Apple Silicon only (CoreML). На NVIDIA не запускается |
| **microsoft/Phi-4-multimodal-instruct** (Microsoft repo) | Заменён на **NVIDIA-пере-выложенную FP4 версию** `nvidia/Phi-4-multimodal-instruct-NVFP4` — те же веса, но без gating и в FP4 (~4 GB вместо 12 GB fp16) |
| **ibm-granite/granite-speech-4.1-2b**, **2b-plus** | Только en/fr/de/es/pt/ja — **русского нет** |
| **XiaomiMiMo/MiMo-V2.5-ASR** (8B) | Mandarin + en + китайские диалекты, **русского нет** |
| **CohereLabs/cohere-transcribe-03-2026** | 14 enterprise языков, **русского нет**. Плюс в карточке: «inconsistent performance on code-switched audio» |
| **facebook/seamless-m4t-v2-large** | 2023 года, мультиязык, но качество хуже Whisper, и CS на нём не работает |
| **microsoft/VibeVoice-1.5B** | Это TTS-вариант, не ASR — путаница на HF |
| **microsoft/VibeVoice-ASR (base BF16) через vLLM** | 9B BF16 = 18 GB, на 5080 не лезет. Microsoft `start_server.py` хардкодит `--dtype bfloat16` |
| **scerz/VibeVoice-ASR-4bit + vLLM плагин** | 4-bit квант от `-HF`-варианта, vLLM плагин не знает архитектуру `VibeVoiceASRForConditionalGeneration` (только `VibeVoice`, `VibeVoiceForASRTraining`) |
| **microsoft/VibeVoice-ASR-HF через vLLM** | vLLM v1 engine падает: `AttributeError: VibeVoiceAsrProcessor has no attribute '_get_num_multimodal_tokens'`. **Используем эту модель, но через transformers, а не vLLM** (см. `asr_servers/vibevoice_server.py`) |
| **nvidia/canary-qwen-2.5b** | NeMo баг с кэшем: контейнер удаляет существующий кэш и качает заново каждый restart → preload бесполезен. Плюс HF Fastly CDN из РФ медленно отдаёт. На canary тратили 15+ минут на каждый старт. Дропнули — русского в canary всё равно мало |
| **Qwen3-Omni-30B-A3B-Instruct** | Даже int4 ~20 GB, не влезает на 16 GB |
| **Voxtral-Mini-4B-Realtime-2602** | Единственный Voxtral, который поддерживает русский (13 языков). Но API — WebSocket streaming (`/v1/realtime`), не batch `/v1/audio/transcriptions`, под наш runtime не лезет без отдельного WS→REST моста. Плюс известный [bug на 16 GB](https://github.com/vllm-project/vllm/issues/38233) с переполнением encoder_cache |
| **ElevenLabs Scribe v2** | Closed API ($0.22/час), без локального deployment. Можно использовать как 3-й голос в ensemble разметке, но не как Docker-backend |

---

## 6. Hardware budget анализ (RTX 5080 16 GB)

Базовые цифры — vLLM по умолчанию резервирует 90% VRAM под KV-cache, поэтому
fp16 веса модели должны помещаться примерно в 12-14 GB.

| Что фит без квантизации (fp16) | Что фит только с квантизацией | Что не фит |
|---|---|---|
| Qwen3-ASR-1.7B (~4 GB) | VibeVoice-ASR 9B (4-bit ~5-6 GB) | VibeVoice 9B BF16 (18 GB) |
| Whisper-large-v3-turbo (~2 GB) | Phi-4-MM (NVFP4 ~4 GB) | Qwen3-Omni-30B-A3B |
| Parakeet-TDT-0.6B (~2 GB) | Qwen2.5-Omni-7B (int4 ~5 GB) | Voxtral-Small-24B (любой формат — нет ru) |
| Parakeet-TDT-0.6B (~2 GB) | | |
| Whisper-large-v3-turbo (~2 GB) | | |

---

## 7. Ссылки на model cards

### Используемые (в стеке)

- [deepdml/faster-whisper-large-v3-turbo-ct2](https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2)
- [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
- [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- [nvidia/Phi-4-multimodal-instruct-NVFP4](https://huggingface.co/nvidia/Phi-4-multimodal-instruct-NVFP4)
- [microsoft/VibeVoice-ASR-HF](https://huggingface.co/microsoft/VibeVoice-ASR-HF)
- [VibeVoice ASR Transformers integration docs](https://huggingface.co/docs/transformers/en/model_doc/vibevoice_asr)

### Проверены и отклонены

- [ibm-granite/granite-speech-4.1-2b](https://huggingface.co/ibm-granite/granite-speech-4.1-2b)
- [XiaomiMiMo/MiMo-V2.5-ASR](https://huggingface.co/XiaomiMiMo/MiMo-V2.5-ASR)
- [CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
- [bartowski/mistralai_Voxtral-Small-24B-2507-GGUF](https://huggingface.co/bartowski/mistralai_Voxtral-Small-24B-2507-GGUF)
- [argmaxinc/whisperkit-coreml](https://huggingface.co/argmaxinc/whisperkit-coreml)
- [bond005/whisper-podlodka-turbo](https://huggingface.co/bond005/whisper-podlodka-turbo)

### Контекст и бенчмарки

- [SwitchLingua dataset (HF audio)](https://huggingface.co/datasets/Shelton1013/SwitchLingua_audio)
- [SwitchLingua paper (arXiv 2506.00087)](https://arxiv.org/html/2506.00087v1)
- [Open ASR Leaderboard](https://huggingface.co/blog/open-asr-leaderboard)
- [Gladia — Best open-source STT 2026](https://www.gladia.io/blog/best-open-source-speech-to-text-models)
- [Code-switching in Speech Recognition (Gladia 2026 guide)](https://www.gladia.io/blog/what-is-code-switching-in-speech-recognition)
- [Multilingual and code-switched ASR with NVIDIA NeMo](https://developer.nvidia.com/blog/multilingual-and-code-switched-automatic-speech-recognition-with-nvidia-nemo/)

