# Хостинг демо на локальной машине

Как поднять DS Navigator на своём компьютере и показать **RAG-поиск** по
интернету (удалённой комиссии) через один HTTPS-URL.

Топология: бэкенд-сервисы в Docker, FastAPI на хосте отдаёт и API, и кастомный
веб-фронт (один origin → CORS не нужен), наружу — туннель `cloudflared`.

```
браузер комиссии
      │  https://<...>.trycloudflare.com/?key=<токен>
      ▼
 cloudflared  ──►  FastAPI :8001  ──►  rewrite/answer  ──►  OpenRouter (облако)
   (туннель)        ├─ / (SPA, src/web/index.html)
                    └─ /forward, /source-files
                              │  эмбеддинг запроса
                              ├──► Infinity/FRIDA :7997 (GPU)
                              └──► Weaviate :8080 (поиск)
```

---

## 0. Предусловия (один раз)

1. **`.env`** в корне проекта (см. `.env_example`). Минимум — ключи OpenRouter:
   ```env
   LLM_MODEL=openai/gpt-4o-mini
   LLM_BASE_URL=https://openrouter.ai/api/v1
   LLM_API_KEY_1=...
   LLM_API_KEY_2=...
   LLM_API_KEY_3=sk-or-...   # этим ходит RAG
   ```
2. **Docker** с поддержкой NVIDIA GPU (`nvidia-container-toolkit`) — FRIDA в
   контейнере `infinity` требует `runtime: nvidia`.
3. **uv** (зависимости Python): `uv sync`.
4. **cloudflared** (для выдачи наружу): `winget install --id Cloudflare.cloudflared`.
   Аккаунт Cloudflare для разового quick-туннеля **не нужен**.
5. **Данные в Weaviate.** Поиск работает только если коллекция уже наполнена
   транскриптами, эмбеднутыми FRIDA. Проверка после старта:
   `http://localhost:8001/check-vdb` → `documents_in_vdb > 0`.
   Если пусто — сначала разово прогнать ингест:
   ```powershell
   # из каталога src, при поднятых vdb+emb (и asr-whisper, если из аудио)
   uv run python -m downloader.ingest --from-dir data/transcripts/recsys
   ```

---

## 1. Быстрый старт (2 терминала)

**Терминал 1 — сервисы + API:**
```powershell
.\scripts\demo_up.ps1 -Token "hse2026"
```
Скрипт: поднимает Docker (`vdb` + `emb`), ждёт готовности Weaviate, запускает
FastAPI на `:8001`. Выведет локальную ссылку с ключом.
Без `-Token` сгенерирует случайный и покажет его.

**Терминал 2 — туннель наружу:**
```powershell
.\scripts\demo_tunnel.ps1
```
Выведет публичный адрес `https://<случайное>.trycloudflare.com`.

**Ссылка для комиссии:**
```
https://<случайное>.trycloudflare.com/?key=hse2026
```
Параметр `?key=` фронт подхватывает один раз, кладёт в localStorage и убирает из
адресной строки. Дальше все запросы уходят с заголовком `X-Demo-Token`.

После демо просто `Ctrl+C` в обоих терминалах — туннель и API закрыты.

> Для живого ингеста (загрузка YouTube/файлов прямо на демо) подними ещё и
> Whisper: `.\scripts\demo_up.ps1 -Token "hse2026" -WithAsr`.

---

## 2. Безопасность (важно для интернета)

Туннель публичен — любой с URL может слать запросы, а `/forward` тратит
OpenRouter-кредиты. Поэтому:

- **Гейт доступа включён**, когда задан `DEMO_ACCESS_TOKEN`. Защищены `/forward`,
  `/ingest`, `/ingest-upload` — без верного `X-Demo-Token` отдают `401`.
  Раздача сайта, `/source-files`, `/health` открыты (безвредны).
- **Не публикуй ключ** вместе со скриншотами. Меняй `-Token` между показами.
- **Поднимай туннель только на время демо.** quick-URL живёт, пока работает
  `cloudflared`; закрыл терминал — адрес мёртв.
- Реализация гейта: middleware `access_gate` в `src/api/main.py`. Если
  `DEMO_ACCESS_TOKEN` пуст — гейт выключен (локальная отладка как раньше).

---

## 3. Стабильный URL (опционально)

quick-туннель даёт новый адрес при каждом запуске. Если ссылку нужно разослать
заранее и чтобы она не менялась — именованный туннель Cloudflare (нужен бесплатный
аккаунт + домен в Cloudflare):

```powershell
cloudflared tunnel login
cloudflared tunnel create ds-navigator
cloudflared tunnel route dns ds-navigator demo.example.com
# config.yml: ingress -> service: http://localhost:8001
cloudflared tunnel run ds-navigator
```
Тогда демо живёт на `https://demo.example.com` стабильно.

---

## 4. Что слушает какие порты

| Сервис            | Порт  | Нужен для демо-поиска | Профиль          |
|-------------------|-------|------------------------|------------------|
| FastAPI + SPA     | 8001  | да (хост, uvicorn)     | —                |
| Weaviate          | 8080  | да                     | `vdb`            |
| Infinity / FRIDA  | 7997  | да (GPU)               | `emb`            |
| Whisper (ASR)     | 8000  | только для ингеста     | `asr-whisper`    |

Остальные профили (`langfuse`, `mlflow`, `airflow`, `monitoring`) для демо-поиска
не нужны.

---

## 5. Траблшутинг

- **API падает на старте с ошибкой Weaviate** — Weaviate ещё не поднялся.
  `demo_up.ps1` ждёт его сам; если запускаешь uvicorn вручную — сначала `docker
  compose --profile vdb --profile emb up -d` и дождись `:8080/v1/.well-known/ready`.
- **Первый запрос висит ~30–60 c** — Infinity догружает модель FRIDA в VRAM.
  Дальше быстро.
- **`/check-vdb` → documents_in_vdb: 0** — нет данных, см. предусловие 5 (ингест).
- **На фронте «сервер недоступен»** — не запущен FastAPI или закрыт порт 8001.
- **`401 unauthorized`** — в URL нет `?key=` или токен не совпадает с тем, что
  передан в `-Token`.
- **`cloudflared` не находится** — `winget install --id Cloudflare.cloudflared`,
  затем перезапусти терминал.
