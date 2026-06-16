# DS Navigator (Audio2RAG) — Архитектура self-hosted публикации

> Дата: 2026-06-14. Статус: **approved** — пользователь подтвердил дизайн и принял решения (ниже). Реализация — поэтапно.
> Синтез исследований по подсистемам + adversarial-вердиктов. При конфликте вердикт побеждает «оптимистичное» исследование.

> **Подтверждённые решения (2026-06-14):**
> - **Аутентификация — весь сайт за логином.** Публичного доступа НЕТ: поиск и ингест видны только залогиненным. Аккаунты **выдаёт только админ** (открытой регистрации нет). Это ОТМЕНЯЕТ ранее принятый «публичный доступ» из §1 и снимает часть §11.
>   - БД пользователей — **SQLite** в docker-volume (без отдельного Postgres). Пароли — `passlib`/bcrypt. Сессии — **JWT** (`Authorization: Bearer`). Роли `user`/`admin`. Первый admin — из env (`ADMIN_USERNAME`/`ADMIN_PASSWORD`) при старте.
>   - Эндпоинты: `POST /auth/login`, `GET /auth/me`; админ: `POST /admin/users` (создать), `GET /admin/users` (список), деактивация. Все data-роуты (`/forward`, `/ingest*`, `/source-files`) — `Depends(current_user)`; админ-роуты — `Depends(admin_user)`.
>   - **`DEMO_ACCESS_TOKEN` ретайрится** — заменяется per-user аутентификацией.
>   - **Cloudflare Turnstile УБИРАЕТСЯ** — раз сайт целиком за логином и аккаунты выдаёт админ, анонимных абьюзеров нет. SSRF-allowlist для yt-dlp оставляем как good practice.
> - **Реализация поэтапная, с ревью.** **Фаза 1:** контейнеризация API (lazy-init) + nginx + TLS + Angular-скелет + **поиск** (локально, пока без логина). **Фаза 2 (Auth):** SQLite users + JWT + защита роутов + login-страница + route-guard + JWT-интерсептор + админ-страница выдачи аккаунтов. **Фаза 3:** **ingest-UI + async-jobs** (за логином). **Фаза 4:** WireGuard + security-hardening + сетевой рунбук.
> - **Дашборды (Grafana/MLflow/Langfuse/Airflow/Weaviate) — только через VPN.** Снимаем публикацию их портов (§11, риск #2).
> - **Ингест — асинхронный** (`202 {job_id}` + `GET /ingest-status/{id}` + поллинг Angular, §10). Решает 504 и даёт прогресс-UX.
>
> Открытые вопросы §12 #1 (CGNAT), #2 (значение DOMAIN), #3 (bridge R2), #4 (HTTP-01 vs DNS-01), #8 (ip6_tables на WSL2), #10 (конкретные лимиты) — остаются на этап деплоя/настройки; код/конфиг от них не блокируется (конфигурируемые переменные + рунбук).

## 1. Обзор и цель

Цель — опубликовать RAG-сервис над транскриптами лекций (поиск + публичный ingest видео) в интернет на **белом/публичном IP** с автоматическим HTTPS, при этом:

- Фронтенд переписывается на **Angular** (полная замена single-file SPA `src/web/index.html`).
- **nginx** (Docker-сервис) терминирует TLS, раздаёт статику Angular и реверс-проксирует FastAPI.
- **FastAPI контейнеризируется** (uv-образ) и стоит **за** nginx в docker-сети `rag-net` — наружу не публикуется.
- Секция «добавить видео / транскрипция» (**ingest**) и поиск — **публичны** для всех посетителей (не admin-only).
- Админ-поверхности (Grafana, MLflow, Langfuse, Airflow, Weaviate-консоль, wg-easy UI) доступны владельцу **только через WireGuard-туннель**, а НЕ публикуются на белый IP.
- **WireGuard (wg-easy)** — личный канал владельца к машине (НЕ гейт для зрителей).
- Сетевая топология — **двойной NAT**: белый IP терминируется на ВХОДНОМ роутере → второй роутер → ПК. Проброс портов настраивается ВРУЧНУЮ на ОБОИХ роутерах (документируем точные правила, не автоматизируем).
- Домен покупается пользователем → Let's Encrypt с конфигурируемой переменной `DOMAIN`.

Ключевой архитектурный риск, проходящий через весь дизайн: **импорт `api.main` устанавливает соединение с Weaviate во время импорта модуля** (через `CHAT_VECTORE_STORE_MANAGER`), поэтому контейнеризация FastAPI требует одновременно (а) healthcheck + `depends_on: service_healthy` и (б) ленивой инициализации клиента — см. §10/§4.

---

## 2. ASCII-схема топологии

```
                              ┌─────────────────────────┐
                              │   OpenRouter (cloud LLM) │
                              └────────────▲────────────┘
                                           │ HTTPS (исходящий, из api)
                              ИНТЕРНЕТ      │
   ┌──────────────┐                        │
   │ Браузеры      │   HTTPS :443           │           ┌─── Владелец (ноут/телефон)
   │ посетителей   │   HTTP  :80 (ACME)     │           │    WireGuard-клиент
   └──────┬───────┘                         │           └──────────┬───────────
          │  DNS A-record DOMAIN → белый IP             UDP :51820  │
          ▼                                                         ▼
   ╔══════════════════════════════════════════════════════════════════════╗
   ║ ВХОДНОЙ РОУТЕР R1   (WAN = белый IP, напр. 203.0.113.50)                ║
   ║   forward 80/tcp,443/tcp,51820/udp → R2.WAN (напр. 192.168.0.2)         ║
   ╚════════════════════════════════╤═══════════════════════════════════════╝
                                    │  (двойной NAT, ручной проброс)
   ╔════════════════════════════════╧═══════════════════════════════════════╗
   ║ ВТОРОЙ РОУТЕР R2    (WAN = 192.168.0.2, LAN = 192.168.1.0/24)            ║
   ║   forward 80/tcp,443/tcp,51820/udp → PC.LAN (напр. 192.168.1.10)         ║
   ╚════════════════════════════════╤═══════════════════════════════════════╝
                                    │
   ╔════════════════════════════════╧═══════════════════════════════════════╗
   ║ ПК (статический LAN IP 192.168.1.10, Windows + Docker Desktop/WSL2)      ║
   ║   Windows Firewall: allow inbound 80/tcp, 443/tcp, 51820/udp            ║
   ║                                                                          ║
   ║   ┌──────────── docker network "rag-net" ──────────────────────────┐    ║
   ║   │                                                                  │    ║
   ║   │  :80/:443 ┌─────────┐   /        ┌──────────────────────┐        │    ║
   ║   │ ─────────▶│  nginx  │───────────▶│ Angular static (html)│        │    ║
   ║   │  (host    │ (TLS,   │            └──────────────────────┘        │    ║
   ║   │   ports)  │ certbot)│   /forward /ingest* /source-files ...      │    ║
   ║   │           │         │───────────▶┌──────────┐                    │    ║
   ║   │           └─────────┘  proxy     │  api     │  (FastAPI :8001,    │    ║
   ║   │                                  │ (uvicorn)│   expose-only)      │    ║
   ║   │                                  └────┬─────┘                     │    ║
   ║   │              ┌──────────────┬─────────┼───────────┐               │    ║
   ║   │              ▼              ▼         ▼            ▼               │    ║
   ║   │        ┌──────────┐  ┌───────────┐ ┌──────┐  (OpenRouter →        │    ║
   ║   │        │ weaviate │  │ infinity  │ │ asr  │   исходящий, вне       │    ║
   ║   │        │ :8080    │  │ :7997 GPU │ │:8000 │   docker)             │    ║
   ║   │        │ :50051   │  │ (FRIDA)   │ │ GPU  │                       │    ║
   ║   │        └──────────┘  └───────────┘ └──────┘                       │    ║
   ║   │                                                                  │    ║
   ║   │   ┌──────────┐   только через туннель (НЕ публикуется наружу):   │    ║
   ║   │   │ wg-easy   │◀── UDP :51820 (туннель) ; UI :51821 (loopback)    │    ║
   ║   │   │           │    Grafana/MLflow/Langfuse/Airflow/Weaviate-UI    │    ║
   ║   │   └──────────┘                                                    │    ║
   ║   └──────────────────────────────────────────────────────────────────┘  ║
   ╚══════════════════════════════════════════════════════════════════════════╝
```

Через двойной NAT наружу пробрасываются **только** 80/tcp, 443/tcp, 51820/udp. Всё остальное — внутри `rag-net` / через туннель.

---

## 3. Состав docker-compose (сервисы)

Новые/затрагиваемые сервисы. Профили подобраны так, чтобы веб-стек поднимался профилем `web`/`full`, а туннель — отдельным `vpn`.

| Сервис | Образ | Порт (host) | Сеть | Профиль | Назначение |
|---|---|---|---|---|---|
| **nginx** | `jonasal/nginx-certbot:6.2.0` (pinned) | `80:80`, `443:443` | rag-net | web, full | TLS-терминация, раздача Angular-статики, reverse-proxy на api, ACME HTTP-01 + авто-renew |
| **api** | `stt-rag-api` (build `Dockerfile.api`) | — (`expose: 8001`) | rag-net | api, full | FastAPI/uvicorn; RAG, ingest. НЕ публикуется наружу |
| **frontend-build** | `ds-navigator-frontend` (build `frontend/Dockerfile`, multi-stage) | — | — | web, full | `ng build` → bundle запекается в образ nginx (НЕ shared volume, см. §9) |
| **weaviate** | `cr.weaviate.io/semitechnologies/weaviate:1.28.0` | — (internal) | rag-net | vdb, full | Вектор-БД (REST :8080, gRPC :50051). Добавляем healthcheck (§10) |
| **infinity** | (текущий, FRIDA) | — (internal) | rag-net | emb, full | Эмбеддинги FRIDA, NVIDIA GPU :7997. Добавляем healthcheck |
| **asr-whisper** | `fedirz/faster-whisper-server:latest-cuda` | — (internal) | rag-net (alias `asr`) | asr-whisper, full | ASR, NVIDIA GPU :8000. Только один ASR-профиль активен |
| **wg-easy** | `ghcr.io/wg-easy/wg-easy:15` (pin major) | `51820:51820/udp`, `127.0.0.1:51821:51821/tcp` | rag-net | vpn, full | VPN владельца. UI только на loopback (см. §8) |
| certbot/acme | (встроен в `jonasal/nginx-certbot`) | — | rag-net | web, full | Выпуск/renew сертификатов внутри nginx-образа. Fallback DNS-01 — см. §6 |
| langfuse *(opt)* | (текущий) | — (internal/loopback) | rag-net | langfuse | Трейсинг. Доступ ТОЛЬКО через туннель |
| mlflow *(opt)* | (текущий) | — (internal/loopback) | rag-net | mlflow | Реестр моделей/прогресс ingest. ТОЛЬКО через туннель |

> Решение по TLS-сервису (разрешение конфликта исследований): вместо отдельного `certbot`-sidecar используем **единый образ `jonasal/nginx-certbot:6.2.0`** (nginx mainline + certbot + авто-renew по таймеру). Он сам кладёт self-signed dummy-cert, чтобы nginx стартовал до выпуска реального серта, и решает «курицу-яйцо». Это default; DNS-01 fallback — в §6.

---

## 4. План файлов

Все пути абсолютные, под `C:\Users\Petr.Chernov\Documents\Хлам\UNIC\STT_RAG_HSE_1Y_MAG\`.

### Создать (NEW)

- `frontend\` — корень Angular-проекта (Angular 21.2.x, standalone + zoneless, Node 22 LTS):
  - `frontend\Dockerfile` — multi-stage (node:22-alpine build → nginx статика / или артефакт для nginx-certbot).
  - `frontend\nginx\` — НЕ нужен, если TLS-nginx (jonasal) раздаёт статику сам; иначе шаблон location-блока API.
  - `frontend\src\app\core\api.service.ts`, `api.types.ts`, `auth.interceptor.ts`, `auth.store.ts`, `timecode.util.ts`.
  - `frontend\src\app\features\search\`, `frontend\src\app\features\ingest\`.
  - `frontend\src\environments\environment.ts` (prod: `apiBase=''`), `environment.development.ts`.
  - `frontend\proxy.conf.json` — для `ng serve` (same-origin без CORS в dev).
- `nginx\user_conf.d\dsnavigator.conf` — server-блоки 80→443, TLS, location'ы proxy/SPA (§5). Каталог монтируется в jonasal-образ как `/etc/nginx/user_conf.d`.
- `nginx\proxy_common.conf` — общий набор `proxy_set_header` + таймауты.
- `Dockerfile.api` (корень репо) — uv multi-stage, ffmpeg в runtime, non-root, `WORKDIR /app/src`, `--proxy-headers` (§10).
- `.dockerignore` (корень репо) — исключить `data/`, `hf_cache/`, `infinity_data/`, `.git`, `src/web/`, `.env`, логи.
- `docs\runbook-network.md` — ручной runbook двойного NAT (содержимое §7).

### Изменить (EXISTING)

- `docker-compose.yml` — добавить сервисы `nginx`, `api`, `frontend-build`, `wg-easy`; volume `letsencrypt`, `api-data`; healthcheck в `weaviate` и `infinity`; снять публикацию портов с админ-дашбордов (Grafana/MLflow/Langfuse/Airflow/MinIO) → `expose` или `127.0.0.1:` (§11).
- `src\api\main.py` — (1) убрать/загейтить `app.mount("/", StaticFiles(...))` (строки ~171–181) под env-флаг, т.к. статику теперь раздаёт nginx; (2) `/check-vdb` (строка 104) и `/source-files` (строка 164) перевести на `get_chat_vectore_store_manager()`; (3) глобальный ingest-семафор / 503 при насыщении (§11).
- `src\system\llm\llm_services.py` — заменить eager `CHAT_VECTORE_STORE_MANAGER = init_vectore_store_manager()` (строка 62) на ленивый `get_chat_vectore_store_manager()` (§10).
- **`src\system\rag\pipeline.py` — ОБЯЗАТЕЛЬНО** (упущено первичным исследованием, найдено вердиктом и подтверждено): строка 25 `vector_store_manager: VectorStoreManager = CHAT_VECTORE_STORE_MANAGER` — это **второй import-time путь** к eager-синглтону (default-arg вычисляется при импорте). Заменить default на `None` и резолвить внутри `run()` через `get_chat_vectore_store_manager()`. Без этого ленивая инициализация в `llm_services.py` НЕ устранит crash-loop.
- `src\api\schemas.py` — добавить SSRF-валидатор `url` в `IngestRequest` (allowlist хостов + scheme), `max_length` (§11).
- `src\downloader\sources\youtube.py` — `default_search="error"`, `playlist_items` cap, запрет generic-экстрактора (§11).
- `src\downloader\processor.py` — глобальный `INGEST_SEM` (module scope); upload — стриминг на диск чанками вместо `await upload_file.read()` (строка 118), cap файлов/байт, `Path(filename).name` против path-traversal (§11).

---

## 5. nginx: обязанности и ключевые директивы

Обязанности: TLS-терминация; раздача Angular SPA с fallback на `index.html`; reverse-proxy API под тем же origin (→ без CORS); 80→443 редирект + проксирование ACME; большие загрузки; длинные таймауты для ingest; rate-limit на дорогих ручках; security-headers.

Ключевые директивы (`nginx\user_conf.d\dsnavigator.conf`, + зоны в `http{}`):

```nginx
# --- http{}: rate-limit зоны (ключ по IP) ---
limit_req_zone  $binary_remote_addr zone=ingest_rl:10m  rate=6r/m;   # ~1 раз / 10с
limit_req_zone  $binary_remote_addr zone=forward_rl:10m rate=30r/m;  # LLM-стоимость
limit_req_zone  $binary_remote_addr zone=general_rl:10m rate=30r/s;
limit_conn_zone $binary_remote_addr zone=conn_per_ip:10m;
limit_req_status 429; limit_conn_status 429;
server_tokens off;
client_header_timeout 15s;   # анти-Slowloris

upstream rag_api { server api:8001; keepalive 16; }

server {                       # :80 — ACME + редирект
    listen 80; server_name ${DOMAIN};
    location /.well-known/acme-challenge/ { root /var/www/letsencrypt; }
    location / { return 301 https://$host$request_uri; }
}

server {                       # :443 — основной сайт
    listen 443 ssl; http2 on; server_name ${DOMAIN};
    ssl_certificate     /etc/letsencrypt/live/${DOMAIN}/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/${DOMAIN}/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;

    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "DENY" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    # CSP — затянуть под Angular (и Turnstile-origin, если включат).

    client_max_body_size 25m; limit_conn conn_per_ip 20;
    root /usr/share/nginx/html; index index.html;

    location = /ingest-upload {           # большие видео + долгая транскрипция
        proxy_pass http://rag_api; include /etc/nginx/proxy_common.conf;
        limit_req zone=ingest_rl burst=1 nodelay; limit_conn conn_per_ip 1;
        client_max_body_size 200m;        # жёсткий cap на краю (вердикт: НЕ 2g)
        proxy_request_buffering on;       # буферизуем тело — защищаем single-worker api
        proxy_read_timeout 1800s; proxy_send_timeout 300s; client_body_timeout 600s;
    }
    location = /ingest {                  # YouTube ingest (download+ASR)
        proxy_pass http://rag_api; include /etc/nginx/proxy_common.conf;
        limit_req zone=ingest_rl burst=2 nodelay; limit_conn conn_per_ip 2;
        client_max_body_size 1m; proxy_read_timeout 900s;
    }
    location = /forward {                 # LLM-запрос (стоимость OpenRouter)
        proxy_pass http://rag_api; include /etc/nginx/proxy_common.conf;
        limit_req zone=forward_rl burst=10 nodelay; limit_conn conn_per_ip 4;
        proxy_read_timeout 120s; proxy_buffering off;
    }
    location ~ ^/(source-files|check-vdb|api|auth-check)(/|$) {
        proxy_pass http://rag_api; include /etc/nginx/proxy_common.conf;
        limit_req zone=general_rl burst=30 nodelay;
    }
    location ~ ^/(docs|redoc|openapi\.json|health)$ {   # внутреннее — только через VPN
        allow 10.8.0.0/24; deny all;
        proxy_pass http://rag_api; include /etc/nginx/proxy_common.conf;
    }
    location ~* \.(?:js|css|woff2?|png|jpg|svg|ico|webp|map)$ {
        expires 1y; add_header Cache-Control "public, immutable"; try_files $uri =404;
    }
    location = /index.html { add_header Cache-Control "no-cache"; }
    location / { limit_req zone=general_rl burst=60 nodelay; try_files $uri $uri/ /index.html; }
}
```

`proxy_common.conf`: `proxy_http_version 1.1; Host $host; X-Real-IP $remote_addr; X-Forwarded-For $proxy_add_x_forwarded_for; X-Forwarded-Proto $scheme;`.

Разрешения конфликтов по вердиктам: upload-cap = **200m на краю** (а не 2g) с `proxy_request_buffering on` для защиты single-worker uvicorn (security-вердикт сильнее nginx-оптимизма про streaming). Глобальный `client_max_body_size` мал (25m), на upload — точечно. uvicorn запускать с `--proxy-headers --forwarded-allow-ips='*'` (безопасно: единственный вход — nginx).

---

## 6. TLS / Let's Encrypt

**Выбор (default): HTTP-01 webroot, обслуживаемый публичным nginx** (внутри `jonasal/nginx-certbot:6.2.0`). Нулевые DNS-креды, совпадает с топологией «nginx на белом IP», автo-renew встроен. ACME валидация бьёт `http://${DOMAIN}/.well-known/acme-challenge/...` с порта **80 only** → оба роутера обязаны постоянно пробрасывать 80 (renew, не только выпуск).

**Fallback (становится primary, если выполнено любое):** ISP блокирует/перехватывает входящий 80; за CGNAT; нужен wildcard; или 80 нельзя надёжно пробросить через оба роутера. Тогда — **DNS-01** (нужны API-креды DNS-провайдера, входящие порты не требуются вообще).

Поправки по вердикту Let's Encrypt (фактические ошибки исходного исследования — исправлены):
- Утверждение «у RU-регистраторов нет acme.sh-хуков» **неверно**. На 2025 существуют `dns_regru`, `dns_beget`, `dns_yandex360`, `dns_yc`. Cloudflare предпочтителен по надёжности/скорости пропагации, **а НЕ** из-за «отсутствия поддержки».
- Multi-perspective validation — **кворум**, не «все vantage points обязаны достучаться».
- Wildcard `*.example.com` **не** покрывает apex → выпускать `-d example.com -d *.example.com`.
- Пропагация: Cloudflare ~60s, **Yandex 360 ~600s** (+ есть баг refresh OAuth-токена) — тюнить per-provider.

**Renewal:** jonasal проверяет каждые `RENEWAL_INTERVAL=8d`, certbot реально обновляет при <30 днях до истечения; per-cert метод (webroot/dns) запоминается в `renewal/<domain>.conf`, поэтому одна команда renew обслуживает оба. nginx перечитывает серты автоматически в этом образе. Перед боем — `STAGING=1` чтобы не выжечь лимит (5 дублей/домен/неделю).

**Cert volume:** именованный том `letsencrypt:/etc/letsencrypt` (rw). Монтировать **всю** директорию (там симлинки `live/`→`archive/`); монтирование отдельных `.pem` ломает renew. HSTS включать только после подтверждённо валидного серта.

---

## 7. Сеть (двойной NAT) — РУЧНЫЕ шаги пользователя

> Всё ниже — ручная настройка в веб-админках роутеров и на ПК. Значения IP — плейсхолдеры.

**STEP 0 — проверить, что белый IP настоящий (не CGNAT):** на R1 WAN-странице IP должен совпадать с `curl https://api.ipify.org` с ПК. Если различаются или WAN ∈ `100.64.0.0/10` → это CGNAT, проброс **не сработает** (нужен статический IP у провайдера). Это hard-prerequisite и для сайта, и для WireGuard.

**STEP 1 — статические LAN-адреса (до проброса):**
- R1: DHCP-reservation WAN-MAC R2 → фиксированный `192.168.0.2`.
- R2: DHCP-reservation MAC ПК → `192.168.1.10`. R1 и R2 — **разные подсети** (`192.168.0.0/24` vs `192.168.1.0/24`).

**STEP 2 — таблица проброса (две пары правил, протокол сохраняется на обоих хопах):**

| Сервис | Внешний порт | Proto | R1 (вход) → | R2 → |
|---|---|---|---|---|
| HTTP / ACME | 80 | TCP | `192.168.0.2:80` | `192.168.1.10:80` |
| HTTPS / сайт | 443 | TCP | `192.168.0.2:443` | `192.168.1.10:443` |
| WireGuard | 51820 | **UDP** | `192.168.0.2:51820` | `192.168.1.10:51820` |

Частая ошибка №1: в R1 указать IP ПК — нельзя; R1 знает только WAN R2. НЕ использовать DMZ-shortcut на R1.

**STEP 3 — DNS A-record:** `@` и `www` → белый IP (R1 WAN). TTL 300 на время тестов. Если IP динамический → DDNS на R1/у регистратора. Серверное имя для серта = `${DOMAIN}`.

**STEP 4 — Windows Defender Firewall (PowerShell от админа):**
```powershell
New-NetFirewallRule -DisplayName "nginx 80"  -Direction Inbound -Action Allow -Protocol TCP -LocalPort 80  -Profile Any
New-NetFirewallRule -DisplayName "nginx 443" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 443 -Profile Any
New-NetFirewallRule -DisplayName "wg 51820"  -Direction Inbound -Action Allow -Protocol UDP -LocalPort 51820 -Profile Any
```
Проверить, что Docker реально публикует порт на `0.0.0.0` (не `127.0.0.1`): `Get-NetTCPConnection -State Listen -LocalPort 80,443`.

**STEP 5 — тест ИЗВНЕ LAN (обязательно с мобильного интернета / VPS), не из дома:** `nmap -Pn -p 80,443 <белый IP>`, `curl -Ik https://${DOMAIN}`. Для UDP 51820 эталон — живой WireGuard-клиент с успешным handshake. Бисекция: с ноута в LAN R1 проверить `nmap -p 80,443 192.168.0.2` (изолирует R1 vs R2).

**NAT hairpin:** изнутри LAN `https://${DOMAIN}` часто не работает (роутер не делает U-turn). Фикс: включить NAT Loopback на R1 и R2, ИЛИ split-horizon DNS / hosts-запись `192.168.1.10 ${DOMAIN}` на ПК. Тест извне (мобильный) — единственная правда.

---

## 8. WireGuard (wg-easy)

Назначение: личная зашифрованная дверь владельца к машине — НЕ гейт для зрителей. Через туннель владелец получает: SSH/RDP к ПК; доступ к внутренним консолям (Weaviate :8080, Langfuse :3000, MLflow :5001, Grafana :3001, Airflow :8081) и к wg-easy UI :51821 — **ничего из этого не публикуется на белый IP**.

**Образ:** `ghcr.io/wg-easy/wg-easy:15` (pin major). v15 — переписан: SQLite, HTTPS-UI на :51821, env-схема `INIT_*` (старые `WG_HOST`/`PASSWORD_HASH` из v14 **игнорируются**). Маппинг: `WG_HOST→INIT_HOST`, пароль→`INIT_PASSWORD` (plaintext только при первом старте, потом удалить), `WG_PORT→INIT_PORT`, `WG_DEFAULT_ADDRESS→INIT_IPV4_CIDR`, `WG_ALLOWED_IPS→INIT_ALLOWED_IPS`.

**Требуемые пробросы:** только **UDP 51820** на обоих роутерах + inbound UDP 51820 в Windows Firewall + статическая reservation ПК. **51821 НЕ пробрасывать.** `PersistentKeepalive=25` держит NAT-маппинг (но НЕ заменяет проброс — vanilla WireGuard не делает hole-punching).

`INIT_HOST` = домен (или белый IP) — только строка `Endpoint` в клиентском конфиге; к TLS/LE отношения не имеет.

Поправки по вердикту WireGuard (критичные):
- **FATAL на WSL2:** v15 жёстко ставит ip6tables-правила при старте и **падает**, если в ядре нет `ip6_tables` и задан `INIT_IPV6_CIDR` (`WG_IPV6_DISABLED` не помогает, issue #2097). На Windows/WSL2 это самая вероятная причина «контейнер не стартует» → перед первым запуском убедиться, что WSL2-ядро содержит `ip6_tables`, либо не задавать IPv6-группу.
- **UI: loopback ИЛИ туннель — взаимоисключающе.** Бинд `127.0.0.1:51821` **не** доступен через `10.8.0.1` (пакет на `wg0` не попадёт в loopback-сокет). Выбрать одно: loopback (через `ssh -L`) ИЛИ бинд на интерфейс/контейнер-IP для доступа через туннель.
- **Bootstrap первого клиента:** создать первый клиент через `INIT_*`/CLI ДО того, как полагаться на UI через туннель (иначе «нужен туннель, чтобы открыть UI, который создаёт клиент туннеля»).
- `INIT_ENABLED` — отдельная группа; `INIT_*` применяются ТОЛЬКО при первом старте (позже менять в UI или стереть том).
- Доступ к физическим LAN-хостам через WSL2-NAT — проверять, не гарантирован; до самого ПК и docker-сервисов — ок.

---

## 9. Angular

**Стек:** Angular 21.2.x, standalone + **zoneless** (signals), Node 22 LTS для сборки, билдер `@angular/build:application` (esbuild) → `dist/<app>/browser/`. SSR выключен (статический SPA).

**Views:**
- **SearchView** (`/search`): вопрос, контролы (`top_k` 1–10, `similarity_threshold` 0–1, toggle rewrite, `<select>` источника из `/source-files`); рендер markdown-ответа (`marked`+`DOMPurify`), карточки `retrieved_documents` с таймкодами и deep-link на YouTube (`?t=<n>s`). Фильтр биндить на **`source_title`** (НЕ `source_file_name`).
- **IngestView** (`/ingest`): панель YouTube (`url` + 4 toggle → `POST /ingest`) и панель загрузки файлов (`multiple`, поле `files`; booleans `export_txt/export_json/keep_audio` — **query-параметры**, не form-fields → `HttpParams`).

**ApiService** — типизированная обёртка `HttpClient`: `forward()`, `sourceFiles()`, `ingestYoutube()`, `ingestUpload()` (`reportProgress + observe:'events'`), `authCheck()`. Контракт типов — точное соответствие `schemas.py`/`/forward` (см. `api.types.ts`). `auth.interceptor.ts` инжектит `X-Demo-Token` и на 401 открывает gate; `?key=`-bootstrap отрабатывает ДО первого запроса.

**Same-origin API:** prod `apiBase=''` — nginx и SPA, и API под одним хостом → CORS не нужен. Dev — `ng serve --proxy-config proxy.conf.json`.

**Build & serve (разрешение конфликта — выбран baked image):** Angular-бандл **запекается в образ** (multi-stage), а НЕ раздаётся из shared volume. Причина: атомарность/воспроизводимость деплоя, корректный rollback по тегу, отсутствие гонки «build пишет → nginx читает пустой/устаревший том» и рассинхрона `index.html`↔хеши чанков. Хешированные ассеты — `immutable` на год, `index.html` — `no-cache`.

---

## 10. Ingest / транскрипция

**Реальный поток (имена функций):** `POST /ingest` → `process_youtube(url,...)` (детект `/playlist` → `get_playlist_urls`; per-video под `asyncio.Semaphore(INGEST_CONCURRENCY=3)` → `prepare_audio/download_audio` → `transcribe_and_ingest` → `transcribe_chunked(chunk_minutes=5)` → per-chunk `HttpASRBackend.transcribe()` → `compute_quality_signals` → `os.remove(audio)` в finally → `ingest_json_to_vector_store` → `create_documents_from_json` → `add_texts` в Weaviate, эмбеддинги через Weaviate-vectorizer на Infinity/FRIDA). `POST /ingest-upload` → `process_uploaded_files` (temp-dir → `extract_audio_from_video` → тот же путь).

**Сейчас: ПОЛНОСТЬЮ синхронно.** HTTP-запрос держится открытым всё время download→transcribe→embed→write, нет job-id, нет прогресса для браузера (`run.log_progress` пишет только в MLflow серверно). Время = минуты на лекцию, десятки минут на плейлист; per-chunk ASR read-timeout = 1 час. За nginx с дефолтным `proxy_read_timeout 60s` это **гарантированный 504** → отсюда таймауты §5.

**Рекомендуемое минимальное изменение (для прогресс-UX):** перевести ingest на **fire-and-forget job + polling**:
1. На `POST /ingest`/`/ingest-upload` генерировать `job_id`, запускать `process_*` как background task, сразу возвращать `202 {job_id, status:"queued"}`.
2. Добавить `GET /ingest-status/{job_id}` → `{status, progress_pct, current_item, total_items, items, errors}`. Angular поллит каждые ~2–3с.
3. Прогресс брать из уже существующего колбэка `_on_chunk(idx,total,extra)` и цикла плейлиста — писать в job-store вместо/вдобавок к MLflow.
4. Это снимает проблему proxy-timeout (все HTTP-вызовы короткие).

Оговорка: in-process dict-store умирает при рестарте api и не переживает `--workers >1`. Для single-replica демо — ок; для масштабирования нужен Redis/БД. ASR (`asr-whisper`) и Infinity — **GPU-обязательны** и должны быть подняты; сам api — CPU-only, но требует **ffmpeg/ffprobe** в образе. Partial success плейлиста = HTTP 200 с `errors[]` — UI должен читать `error_count`, не только статус. yt-dlp с серверного IP часто блокируется YouTube — закладывать сообщения об ошибках.

**Crash-loop fix (контейнеризация, по вердикту fastapi-import):** импорт `api.main` коннектится к Weaviate **двумя** путями — `llm_services.py:62` И **`pipeline.py:25` (default-arg)**. Поэтому: (A) healthcheck в `weaviate` (`/dev/tcp` или `wget --spider /v1/.well-known/ready`, т.к. в образе может не быть curl) + `depends_on: service_healthy`; **и** (B) ленивый `get_chat_vectore_store_manager()`, затрагивающий `llm_services.py`, `pipeline.py:25`, `main.py:104/164`. **A и B ставить ВМЕСТЕ** (не «B потом»): пока импорт eager, даже `/health` недоступен при недоступной Weaviate. Профили `api`/`full` должны активировать также `vdb`+`emb`, иначе `depends_on` на сервисы из других профилей сломает `docker compose --profile api up`.

---

## 11. Безопасность публичного ингеста

Defense-in-depth: край (nginx rate-limit/conn-limit/body-cap/headers, §5) + приложение (то, что край не может). `DEMO_ACCESS_TOKEN` остаётся **kill switch**: пусто → гейт выкл; задать секрет → мгновенно закрыть `/forward`+`/ingest*`. Усилить: header-only (убрать `?key=` из логов/истории), `secrets.compare_digest`.

**Ранжированный список рисков:**

| # | Severity | Риск | Митигация |
|---|---|---|---|
| 1 | **Critical** | **SSRF** через `IngestRequest.url` → yt-dlp тянет произвольные URL (cloud-metadata 169.254.169.254, внутренние сервисы rag-net, loopback, local-file) | allowlist хостов+scheme в `schemas.py`; `default_search="error"`; **egress-firewall контейнера** блокирует RFC-1918/link-local/loopback |
| 2 | **Critical** | **Внутренние дашборды с дефолтными кредами** (Grafana/MLflow/Langfuse/Airflow `admin`; Weaviate anonymous) — одна ошибка проброса = компрометация | снять `ports:` (→ `expose`/`127.0.0.1:`), пробрасывать только 80/443/51820, доступ через WireGuard, сменить все дефолтные пароли |
| 3 | **Critical** | **GPU/CPU DoS** — `INGEST_CONCURRENCY`-семафор сейчас per-request (не лимитит параллельные запросы) | **module-level `INGEST_SEM`** + fail-fast 503; `limit_conn`/`limit_req` на `/ingest*` |
| 4 | **High** | **Upload memory DoS** — `await upload_file.read()` грузит весь файл в RAM, нет cap | `client_max_body_size 200m`+`proxy_request_buffering on`; стриминг на диск чанками + cap байт/файлов + `mem_limit` |
| 5 | **High** | **Cost amplification** — `/forward` жжёт OpenRouter-кредиты, `/ingest` — GPU | `forward_rl`/`ingest_rl`; опц. Turnstile + soft per-IP дневная квота; `DEMO_ACCESS_TOKEN` |
| 6 | **High** | **Playlist fan-out bomb** — плейлист на 1000+ видео из одного запроса | `playlist_items` cap + отказ при `len(urls) > N` |
| 7 | **Medium** | **Path traversal** через `upload_file.filename` | `Path(filename).name` + scrub |
| 8 | **Medium** | **yt-dlp CVE/supply-chain** | пинить и регулярно обновлять yt-dlp; не читать user-config |
| 9 | **Medium** | **Утечка токена** `?key=` в логи/историю; non-constant-time | header-only, `compare_digest`, scrub логов |
| 10 | **Medium** | **Slowloris / истощение коннектов** single-worker | `client_header/body_timeout`, `limit_conn 20`, `proxy_request_buffering on` |
| 11 | **Low/Med** | TLS-downgrade / нет HSTS / clickjacking / MIME-sniff | TLS1.2+1.3, HSTS, CSP, `X-Frame-Options: DENY`, `nosniff` |
| 12 | **Low** | Info-leak через `/docs`,`/openapi.json` | deny/VPN-restrict на nginx |
| 13 | **Low** | потеря real-IP при добавлении CDN | `set_real_ip_from` + ключ лимита по `CF-Connecting-IP` |

Порядок внедрения: #1 SSRF+egress, #2 снять дашборды, #3 глобальный семафор, #4 upload-cap — затем rate-limit/Turnstile; `DEMO_ACCESS_TOKEN` — мгновенный выключатель до тюнинга.

---

## 12. Открытые вопросы / решения, которые нужно подтвердить у пользователя

1. **(Критично) Реальный белый IP vs CGNAT.** Подтвердить тестом R1-WAN == `api.ipify.org`. Если CGNAT — вся входящая модель (сайт + WireGuard) не работает без статического IP у провайдера. Блокирует всё.
2. **Значение `DOMAIN`.** Какой домен куплен/будет куплен; кто регистратор (для выбора HTTP-01 vs DNS-01 и провайдера DNS-плагина).
3. **Мостить ли второй роутер (R2) в bridge/AP-mode.** Рекомендация: bridge (single-NAT) сильно упрощает эксплуатацию, чинит hairpin, вдвое сокращает правила. Подтвердить, или сознательно оставить двойной NAT.
4. **HTTP-01 или DNS-01.** Если провайдер/ISP блокирует входящий 80, или нужен wildcard, или 80 не пробросить надёжно — переключаемся на DNS-01 (нужны API-креды DNS; Cloudflare предпочтителен, RU-хуки есть но флакают).
5. **Принятие риска публичного ингеста.** Подтвердить, что #1–#6 из §11 митигируются ДО публикации (особенно egress-firewall против SSRF и снятие дашбордов). Это решение пользователя — публиковать ingest всем.
6. **Включать ли Cloudflare Turnstile** на `/ingest*` (анти-бот) — добавляет зависимость/CSP-origin, но реально режет abuse ботнетами.
7. **Какие доп-дашборды отдавать только через VPN** (Langfuse/MLflow/Grafana/Airflow) и сменить ли дефолтные креды сейчас.
8. **IPv6 на WSL2 для wg-easy.** Подтвердить наличие `ip6_tables` в WSL2-ядре или согласовать запуск без IPv6-группы (иначе контейнер не стартует).
9. **Async-ingest сейчас или потом.** Внедрять ли `202 + /ingest-status` (прогресс-UX, снимает 504) в этой итерации, или временно жить с синхронным ingest и большими таймаутами nginx.
10. **Лимиты ingest:** конкретные значения — `INGEST_CONCURRENCY`, max upload (200m?), max файлов/запрос, max видео в плейлисте, дневная per-IP квота.
