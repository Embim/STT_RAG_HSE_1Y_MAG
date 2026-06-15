# Продакшн-деплой DS Navigator

Чеклист ввода в эксплуатацию на GPU-машине с публичным HTTPS.

---

## Предусловия

Перед запуском убедитесь в следующем:

1. **GPU-машина готова**
   - Docker Engine + NVIDIA Container Toolkit установлены и работают:
     ```powershell
     docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi
     ```
   - WSL2 (если Windows) обновлён: `wsl --update`.

2. **Домен куплен и направлен на белый IP**
   - DNS A-записи `@` и `www` → ваш белый IP уже распространились
     (`Resolve-DnsName your-domain.ru` возвращает нужный IP).
   - CGNAT проверен: WAN IP на R1 совпадает с `(Invoke-WebRequest https://api.ipify.org).Content`.

3. **Порты пробрасованы через роутеры и открыты в файерволе**
   - Выполнены шаги из [docs/runbook-network.md](runbook-network.md): STEP 1–4.
   - Тест с мобильного (STEP 5) пройден — порты 80, 443 и 51820 доступны снаружи.

---

## Конфигурация `.env` для прода

Создайте `.env` в корне репозитория. Ниже — шаблон с реальными значениями
(замените каждый `<...>`):

```env
# ── LLM ─────────────────────────────────────────────────────────────────────
LLM_MODEL=openai/gpt-4o-mini
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY_1=sk-or-...
LLM_API_KEY_2=sk-or-...
LLM_API_KEY_3=sk-or-...

# ── Auth ─────────────────────────────────────────────────────────────────────
JWT_SECRET=<32+ случайных байта, например: openssl rand -hex 32>
ADMIN_USERNAME=admin
ADMIN_PASSWORD=<надёжный пароль>
AUTH_DB_PASSWORD=<надёжный пароль БД>

# ── Web / TLS — БОЕВОЙ РЕЖИМ (не self-signed) ────────────────────────────────
DOMAIN=your-domain.ru
CERTBOT_EMAIL=you@example.com
STAGING=0
USE_LOCAL_CA=

# ── WireGuard ────────────────────────────────────────────────────────────────
WG_HOST=your-domain.ru
WG_PASSWORD=<пароль wg-easy UI>
```

> `USE_LOCAL_CA=` (пустое без дефолта) — именно так отключается самоподписанный
> CA. Значение `-` в синтаксисе `${VAR-default}` позволяет передать настоящую
> пустую строку через `docker compose`.

Сгенерировать случайные секреты (WSL2 или Linux):
```bash
openssl rand -hex 32   # для JWT_SECRET
openssl rand -hex 24   # для паролей
```

---

## Выпуск сертификата: сначала STAGING

При первом запуске **обязательно** используйте тестовый CA Let's Encrypt, чтобы
не выжечь лимит (5 дублей сертификата на домен в неделю):

```env
STAGING=1
```

Поднимите стек (только `web` профиль, без GPU-сервисов):
```powershell
docker compose --profile vdb --profile web up -d --build
```

Дождитесь выпуска сертификата (смотрите логи nginx):
```powershell
docker logs -f ds-navigator-nginx
```

Ищите строки вида `Congratulations! Your certificate and chain have been saved`.
При `STAGING=1` браузер будет показывать предупреждение (staging CA не
доверяется) — это нормально. Убедитесь, что файл сертификата появился:
```powershell
docker exec ds-navigator-nginx ls /etc/letsencrypt/live/your-domain.ru/
```

Когда staging-сертификат выпущен, переключитесь на боевой CA:
```env
STAGING=0
```

Пересоздайте только nginx-контейнер (без остановки API):
```powershell
docker compose --profile web up -d --build nginx
```

---

## Поднять полный стек (GPU-машина)

```powershell
docker compose `
  --profile vdb `
  --profile emb `
  --profile asr-whisper `
  --profile web `
  --profile vpn `
  up -d --build
```

Профили:
- `vdb` — Weaviate (векторная БД)
- `emb` — Infinity/FRIDA (эмбеддинги, GPU)
- `asr-whisper` — Whisper (транскрипция, GPU)
- `web` — API (FastAPI) + nginx (TLS + фронт) + auth-postgres
- `vpn` — wg-easy (WireGuard VPN)

> **Опционально: выбор ASR-модели прямо из UI с авто-свапом.** Если хотите, чтобы
> пользователь мог выбирать модель распознавания при загрузке (а API сам
> переключал GPU-контейнер), включите overlay `docker-compose.autoswap.yml` —
> см. [docs/asr_model_selection.md](asr_model_selection.md). Без него выбор тоже
> работает, но транскрипция идёт на единственный поднятый бэкенд (`asr-whisper`).

> **Опционально: 3D-карта тем (UMAP).** Раздел «Карта тем» работает из коробки на
> PCA. Для лучшего разделения кластеров соберите образ API с UMAP — добавьте
> `--extra viz` в `uv sync` в `Dockerfile.api` (тянет numba/llvmlite). Подробнее:
> [docs/embedding_map.md](embedding_map.md).

Проверить статус:
```powershell
docker compose ps
```

Все сервисы должны иметь статус `running` (или `healthy` для тех, у кого есть
healthcheck).

---

## Bootstrap: первый вход и выдача аккаунтов

1. **Администратор создаётся автоматически** при старте API из переменных
   `ADMIN_USERNAME` / `ADMIN_PASSWORD`.

2. Откройте `https://your-domain.ru` — должна появиться страница входа.
   Войдите с логином/паролем из `.env`.

3. Перейдите в раздел **«Доступы»** (Admin → Users). Здесь можно:
   - создавать новые аккаунты для зрителей (кнопка «Создать пользователя»),
   - деактивировать существующие.

   Открытой саморегистрации нет — аккаунты выдаёт только администратор.

---

## Загрузка данных

### Через веб-интерфейс

Перейдите в раздел **«Добавить»** (Ingest). Доступно:
- Загрузка по YouTube URL (требует запущенного ASR-бэкенда).
- Загрузка аудио/видео-файла напрямую.
- **Выбор модели распознавания (ASR)** — выпадающий список с подсказкой «когда
  какую использовать». По умолчанию предвыбрана уже поднятая модель. Если
  включён авто-свап, выбор другой модели сам переключит GPU-контейнер (см.
  [docs/asr_model_selection.md](asr_model_selection.md)).
- **OCR** — распознавание текста со слайдов (галочка «текст с экрана»).

Прогресс отображается на странице в реальном времени (поллинг статуса задания).

> Задания хранятся в памяти процесса API. При перезапуске контейнера задания
> сбрасываются — это ожидаемое поведение для single-replica демо.

### Через CLI

```powershell
# из корня репозитория, при поднятом стеке
uv run python -m downloader.ingest --from-dir data/transcripts/recsys
```

---

## Доступ к внутренним инструментам через WireGuard

После подключения к WireGuard-туннелю (клиентский конфиг — в wg-easy UI по
`http://127.0.0.1:51821`, доступен с ПК или через SSH port-forward) будут
доступны:

| Инструмент | Адрес (через туннель) |
|-----------|----------------------|
| wg-easy UI | `http://127.0.0.1:51821` (loopback ПК, не через туннель) |
| Weaviate console | `http://<pc-lan-ip>:8080` |
| Langfuse | `http://<pc-lan-ip>:3000` |
| MLflow | `http://<pc-lan-ip>:5001` |
| Grafana | `http://<pc-lan-ip>:3001` |
| Airflow | `http://<pc-lan-ip>:8081` |

> `<pc-lan-ip>` — LAN-адрес ПК (напр. `192.168.1.10`). Через туннель VPN-клиент
> попадает в `rag-net` и может достучаться до этих портов напрямую.
>
> **Ни один из этих адресов не публикуется в интернет.** Снаружи доступны только
> порты 80, 443 (сайт) и 51820/UDP (WireGuard).

**Первый клиент WireGuard: bootstrap-проблема**

wg-easy UI доступен по loopback `http://127.0.0.1:51821` только с самого ПК.
Создайте первый клиентский профиль одним из способов:
- Откройте браузер на ПК и перейдите на `http://127.0.0.1:51821`.
- С удалённой машины: `ssh -L 51821:localhost:51821 user@<pc-ip>`, затем
  откройте `http://127.0.0.1:51821` в локальном браузере.

После создания клиента и успешного рукопожатия — доступ через туннель к LAN-ресурсам.

---

## Чеклист безопасности перед публикацией

Выполните **до** того как поделиться URL:

- [ ] **CGNAT**: белый IP R1 совпадает с `api.ipify.org` — проброс работает.
- [ ] **Только 3 порта наружу**: 80/tcp, 443/tcp, 51820/udp. Ничего больше.
- [ ] **Дашборды не опубликованы**: порты Langfuse (3000), MLflow (5001),
      Grafana (3001), Airflow (8081), Weaviate (8080) не пробрасываются через
      роутеры и не публикуются в `docker-compose.yml` без привязки к loopback.
      Профили `langfuse`, `mlflow`, `airflow`, `monitoring` поднимаются только
      для локальной работы — доступ через WireGuard.
- [ ] **Пароли изменены**: `WG_PASSWORD`, `ADMIN_PASSWORD`, `AUTH_DB_PASSWORD`
      — не дефолтные значения.
- [ ] **JWT_SECRET установлен**: не пустая строка, не `dev-insecure-change-me`.
- [ ] **TLS боевой**: `STAGING=0`, `USE_LOCAL_CA=` (пусто), сертификат выпущен
      от реального CA — браузер не показывает предупреждение.
- [ ] **SSRF-guard активен**: `IngestRequest.url` проходит allowlist-валидацию
      (реализовано в коде); YouTube-экстрактор ограничен allowlist хостов.
- [ ] **DEMO_ACCESS_TOKEN не задан** (или задан как экстренный выключатель):
      аутентификация работает через per-user JWT, `DEMO_ACCESS_TOKEN` устарел.
- [ ] **Одиночная реплика API**: ingest-задания хранятся в памяти процесса —
      запускайте не более одного контейнера `api`. При перезапуске активные
      задания теряются — предупредите пользователей.
- [ ] **Загрузка файлов ограничена**: лимит на nginx — 200 МБ (`client_max_body_size`);
      плейлисты — ограничены по количеству видео в коде API.

---

## Обслуживание

### Продление сертификата

Происходит автоматически внутри контейнера `ds-navigator-nginx` (таймер certbot
каждые 8 дней, обновление при сроке <30 дней). Убедитесь, что порт 80 остаётся
проброшенным постоянно — он нужен для ACME HTTP-01 challenge при каждом
продлении.

### Обновление стека

```powershell
git pull
docker compose --profile vdb --profile emb --profile asr-whisper --profile web --profile vpn up -d --build
```

### Просмотр логов

```powershell
docker compose logs -f api          # API FastAPI
docker compose logs -f ds-navigator-nginx  # nginx + certbot
docker compose logs -f wg-easy      # WireGuard
```

### Остановка

```powershell
docker compose --profile vdb --profile emb --profile asr-whisper --profile web --profile vpn down
```

Данные сохраняются в именованных томах (`weaviate-data`, `letsencrypt`,
`auth-postgres-data`, `wg-easy-data`).
