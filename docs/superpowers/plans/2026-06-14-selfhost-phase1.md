# Self-hosted publication — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Containerize the FastAPI backend behind a Dockerized nginx (TLS-ready), make the Weaviate client lazy so the container can't crash-loop, and replace the single-file demo SPA with an Angular search application — all locally buildable and verifiable without a real domain.

**Architecture:** nginx (`jonasal/nginx-certbot`) terminates TLS, serves the Angular bundle baked into its image, and reverse-proxies the FastAPI `api` container over the docker `rag-net` (single origin → no CORS). The API connects to Weaviate lazily (on first use, not at import), so it starts cleanly even before Weaviate is ready. LLM stays in the cloud (OpenRouter).

**Tech Stack:** FastAPI + uv (Python 3.12), Docker Compose, nginx (jonasal/nginx-certbot), Angular 21 (standalone, esbuild), Node 22, pytest.

**Scope note:** This is **Phase 1 of 4** from [the design](../specs/2026-06-14-selfhost-topology-design.md). Phase 2 (Auth), Phase 3 (ingest UI + async jobs), Phase 4 (WireGuard + hardening + runbook) get their own plans. Phase 1 produces a working, locally-verifiable site: search only. The existing demo at `src/web/index.html` stays as a fallback (gated by `SERVE_SPA`) until Angular replaces it.

> **⚠️ Поправка (новое требование — аутентификация, весь сайт за логином; аккаунты выдаёт только админ).** Auth — это отдельная **Фаза 2**. Влияние на Фазу 1:
> - **Task 7 урезан:** НЕ создавать `auth.store.ts` / `auth.interceptor.ts` и `?key=`-bootstrap (старая токен-модель устарела). Task 7 = только `environments` + `app.routes` + корневой компонент + dev-proxy. В Step 4 `app.config.ts` использовать `provideHttpClient()` **без** `withInterceptors(...)` и **без** `bootstrapToken()`. JWT-интерсептор, route-guard и login-страница — в Фазе 2.
> - **`DEMO_ACCESS_TOKEN` ретайрится, Turnstile убирается** (сайт не публичный) — это уже поздние фазы; в Фазе 1 ничего про токен/Turnstile делать не нужно.
> - Задачи 1–6 и 8 — без изменений.

**Prerequisites for execution:**
- Docker Desktop running; the existing `vdb`+`emb` stack can come up (`docker compose --profile vdb --profile emb up -d`).
- `.env` present in repo root (OpenRouter keys) — needed for the API container to import settings.
- Node 22 LTS + npm available for the Angular tasks.
- Work on a feature branch (e.g. `feat/selfhost-phase1`), not directly on `dev`.

---

## File Structure

**Created:**
- `Dockerfile.api` — uv multi-stage image for the FastAPI service (ffmpeg, non-root).
- `.dockerignore` — keep build context small (exclude data/, caches, .venv, .git).
- `tests/test_lazy_vectorstore.py` — proves import does not connect to Weaviate.
- `frontend/` — Angular project root (scaffolded by `ng new`).
  - `frontend/src/app/core/api.types.ts` — typed contracts for the API.
  - `frontend/src/app/core/api.service.ts` — typed HttpClient wrapper.
  - `frontend/src/app/core/timecode.util.ts` — seconds→`H:MM:SS` + deep-link builder.
  - `frontend/src/app/core/timecode.util.spec.ts` — unit tests for the util.
  - `frontend/src/app/core/auth.interceptor.ts` — injects `X-Demo-Token`, handles 401.
  - `frontend/src/app/core/auth.store.ts` — token storage + `?key=` bootstrap.
  - `frontend/src/app/features/search/search.component.ts|html|css` — the search view.
  - `frontend/src/environments/environment.ts` / `environment.development.ts`.
  - `frontend/proxy.conf.json` — dev same-origin proxy for `ng serve`.
  - `frontend/Dockerfile` — multi-stage: build Angular → bake into nginx-certbot.
- `nginx/user_conf.d/dsnavigator.conf` — server blocks (80→443, TLS, proxy, SPA fallback).
- `nginx/proxy_common.conf` — shared proxy headers.

**Modified:**
- `src/system/llm/llm_services.py` — eager singleton → lazy `get_chat_vectore_store_manager()`.
- `src/system/rag/pipeline.py` — default-arg `= CHAT_VECTORE_STORE_MANAGER` → `= None`, resolve lazily.
- `src/api/main.py` — `/check-vdb` + `/source-files` use the getter; gate `StaticFiles` mount behind `SERVE_SPA`.
- `src/downloader/ingest.py` — use the getter instead of the module-level singleton.
- `pyproject.toml` — add pytest config (`pythonpath = ["src"]`) + dev deps.
- `docker-compose.yml` — add `api` + `nginx` services, `weaviate` healthcheck, `web`/`api` profiles, volumes.

---

## Task 1: Lazy-init the Weaviate manager

Eliminates the import-time Weaviate connection (the container crash-loop root cause). Touches all 4 reference sites.

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/test_lazy_vectorstore.py`
- Modify: `src/system/llm/llm_services.py:12-13,62`
- Modify: `src/system/rag/pipeline.py:4,25,62-69`
- Modify: `src/api/main.py:14,104,164`
- Modify: `src/downloader/ingest.py:8,133`

- [ ] **Step 1: Add pytest config + dev deps to `pyproject.toml`**

Add (near the end of the file, create the tables if absent):

```toml
[dependency-groups]
dev = ["pytest>=8", "pytest-asyncio>=0.23"]

[tool.pytest.ini_options]
pythonpath = ["src"]
asyncio_mode = "auto"
```

Then sync: `uv sync --group dev`
Expected: pytest installed in `.venv`.

- [ ] **Step 2: Write the failing test**

Create `tests/test_lazy_vectorstore.py`:

```python
import os
# settings.py reads these at import; provide dummies so import never touches a real .env-less env.
os.environ.setdefault("LLM_MODEL", "test/model")
os.environ.setdefault("LLM_API_KEY_1", "x")
os.environ.setdefault("LLM_API_KEY_2", "x")
os.environ.setdefault("LLM_API_KEY_3", "x")

import importlib
from unittest.mock import patch


def test_import_does_not_connect_and_getter_is_cached():
    # Patch the Weaviate connect call BEFORE importing the module under test.
    with patch("system.rag.vectore_store.weaviate.connect_to_local") as connect:
        import system.llm.llm_services as svc
        importlib.reload(svc)  # ensure a clean module state for the assertion

        # Importing the module must NOT open a Weaviate connection.
        assert connect.call_count == 0

        mgr1 = svc.get_chat_vectore_store_manager()
        assert connect.call_count == 1          # connects on first use

        mgr2 = svc.get_chat_vectore_store_manager()
        assert connect.call_count == 1          # cached — no second connection
        assert mgr1 is mgr2
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest tests/test_lazy_vectorstore.py -v`
Expected: FAIL — `AttributeError: module 'system.llm.llm_services' has no attribute 'get_chat_vectore_store_manager'`.

- [ ] **Step 4: Implement the lazy getter in `src/system/llm/llm_services.py`**

Replace lines 12-13:

```python
def init_vectore_store_manager() -> VectorStoreManager:
    return VectorStoreManager()
```

with:

```python
_CHAT_VECTORE_STORE_MANAGER: VectorStoreManager | None = None


def get_chat_vectore_store_manager() -> VectorStoreManager:
    """Lazily create and cache the Weaviate-backed manager.

    Connecting at import time (the old module-level singleton) crash-looped the
    containerized API whenever Weaviate wasn't ready yet. Connect on first use.
    """
    global _CHAT_VECTORE_STORE_MANAGER
    if _CHAT_VECTORE_STORE_MANAGER is None:
        _CHAT_VECTORE_STORE_MANAGER = VectorStoreManager()
    return _CHAT_VECTORE_STORE_MANAGER
```

And delete line 62: `CHAT_VECTORE_STORE_MANAGER = init_vectore_store_manager()`.

- [ ] **Step 5: Update `src/system/rag/pipeline.py`**

Line 4 — change the import:

```python
from system.llm.llm_services import get_chat_vectore_store_manager
```

Line 25 — change the default arg from the eager singleton to `None`:

```python
        vector_store_manager: VectorStoreManager | None = None,
```

Inside `run()`, immediately after the `try:` (before first use, around line 27), resolve it:

```python
        if vector_store_manager is None:
            vector_store_manager = get_chat_vectore_store_manager()
```

- [ ] **Step 6: Update `src/api/main.py`**

Line 14 — change the import:

```python
from system.llm.llm_services import get_chat_vectore_store_manager
```

Line 104 (`/check-vdb`): `collection = CHAT_VECTORE_STORE_MANAGER.collection` →

```python
        collection = get_chat_vectore_store_manager().collection
```

Line 164 (`/source-files`): `files = CHAT_VECTORE_STORE_MANAGER.list_source_titles()` →

```python
        files = get_chat_vectore_store_manager().list_source_titles()
```

- [ ] **Step 7: Update `src/downloader/ingest.py`**

Line 8 — change the import:

```python
from system.llm.llm_services import get_chat_vectore_store_manager
```

Line 133 — `await CHAT_VECTORE_STORE_MANAGER.add_texts(...)` →

```python
    await get_chat_vectore_store_manager().add_texts(texts=texts, metadatas=metas, ids=ids)
```

- [ ] **Step 8: Run the test to verify it passes**

Run: `uv run pytest tests/test_lazy_vectorstore.py -v`
Expected: PASS.

- [ ] **Step 9: Sanity-check the API still imports**

Run: `cd src; uv run python -c "import os; os.environ.setdefault('LLM_MODEL','x'); os.environ.setdefault('LLM_API_KEY_1','x'); os.environ.setdefault('LLM_API_KEY_2','x'); os.environ.setdefault('LLM_API_KEY_3','x'); from unittest.mock import patch; p=patch('system.rag.vectore_store.weaviate.connect_to_local'); p.start(); import api.main; print('import OK, no eager connect')"`
Expected: `import OK, no eager connect` (no Weaviate connection attempt during import).

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml tests/test_lazy_vectorstore.py src/system/llm/llm_services.py src/system/rag/pipeline.py src/api/main.py src/downloader/ingest.py
git commit -m "refactor: lazy Weaviate manager to prevent container crash-loop"
```

---

## Task 2: Gate the StaticFiles SPA mount behind `SERVE_SPA`

nginx will serve the Angular bundle, so the container API must NOT also mount the old SPA. Keep it on by default for local dev (the Phase-0 cloudflared demo).

**Files:**
- Modify: `src/api/main.py` (the `WEB_DIR` mount block near the end)

- [ ] **Step 1: Make the mount conditional**

Replace the static-mount block:

```python
WEB_DIR = Path(__file__).resolve().parent.parent / "web"
if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
    logger.info("Serving SPA from %s at /", WEB_DIR)
else:
    logger.warning("Web dir %s not found — SPA not served (API-only mode)", WEB_DIR)
```

with:

```python
# nginx serves the Angular bundle in production → SERVE_SPA=false in the api container.
# Defaults to "true" so local dev (uvicorn without nginx) still serves src/web/index.html.
WEB_DIR = Path(__file__).resolve().parent.parent / "web"
_serve_spa = os.getenv("SERVE_SPA", "true").lower() not in ("false", "0", "no")
if _serve_spa and WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
    logger.info("Serving SPA from %s at /", WEB_DIR)
else:
    logger.info("SPA serving disabled (SERVE_SPA=%s) — API-only mode", os.getenv("SERVE_SPA"))
```

- [ ] **Step 2: Verify it compiles**

Run: `cd src; uv run python -m py_compile api/main.py`
Expected: no output (success).

- [ ] **Step 3: Commit**

```bash
git add src/api/main.py
git commit -m "feat: gate SPA static mount behind SERVE_SPA env"
```

---

## Task 3: `Dockerfile.api` + `.dockerignore`

**Files:**
- Create: `Dockerfile.api` (repo root)
- Create: `.dockerignore` (repo root)

- [ ] **Step 1: Create `.dockerignore`**

```
.git
.venv
**/__pycache__
data/
hf_cache/
infinity_data/
weaviate-data/
*.log
logs/
src/web/
frontend/node_modules/
frontend/dist/
.broker-workspace/
docs/
*.ipynb
```

- [ ] **Step 2: Create `Dockerfile.api`**

```dockerfile
# syntax=docker/dockerfile:1
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev
COPY src ./src

FROM python:3.12-slim AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 appuser
COPY --from=builder --chown=appuser:appuser /app /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    SERVE_SPA=false
WORKDIR /app/src
USER appuser
EXPOSE 8001
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001", \
     "--proxy-headers", "--forwarded-allow-ips=*"]
```

- [ ] **Step 3: Build the image**

Run: `docker build -f Dockerfile.api -t stt-rag-api:latest .`
Expected: build succeeds; final image tagged `stt-rag-api:latest`.

- [ ] **Step 4: Smoke-boot against the running Weaviate**

First ensure Weaviate is up: `docker compose --profile vdb --profile emb up -d`
Then run the API container on the same network, lazy-init means it boots even before queries:

```bash
docker run --rm --network rag-net --env-file .env -e WEAVIATE_HOST=weaviate -e SERVE_SPA=false -p 8001:8001 --name stt-rag-api-smoke stt-rag-api:latest
```

In another terminal: `curl -s http://localhost:8001/health`
Expected: `{"status":"ok"}` (the container started without Weaviate-at-import crash). Then `Ctrl+C`.

- [ ] **Step 5: Commit**

```bash
git add Dockerfile.api .dockerignore
git commit -m "feat: containerize FastAPI with uv multi-stage image"
```

---

## Task 4: docker-compose — `api` service + `weaviate` healthcheck

**Files:**
- Modify: `docker-compose.yml`

- [ ] **Step 1: Add a healthcheck to the `weaviate` service**

Inside the `weaviate:` service block, add:

```yaml
    healthcheck:
      test: ["CMD", "wget", "--no-verbose", "--tries=1", "--spider", "http://localhost:8080/v1/.well-known/ready"]
      interval: 5s
      timeout: 5s
      retries: 12
      start_period: 30s
```

- [ ] **Step 2: Add the `api` service**

Add a new service (after `weaviate`):

```yaml
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    image: stt-rag-api:latest
    container_name: stt-rag-api
    restart: unless-stopped
    networks:
      - rag-net
    expose:
      - "8001"
    profiles:
      - web
      - full
    env_file: .env
    environment:
      WEAVIATE_HOST: weaviate
      WEAVIATE_PORT: "8080"
      EMBEDDING_URL: http://infinity:7997/v1/embeddings
      WEAVIATE_VECTORIZER_BASE_URL: http://infinity:7997
      WHISPER_URL: http://asr:8000
      SERVE_SPA: "false"
    depends_on:
      weaviate:
        condition: service_healthy
      infinity:
        condition: service_started
    <<: *shared-logs
```

- [ ] **Step 3: Validate compose config**

Run: `docker compose config > $null` (PowerShell) — or `docker compose config | head -5`
Expected: no error; merged config prints.

- [ ] **Step 4: Bring up the data + api stack**

Run: `docker compose --profile vdb --profile emb --profile web up -d weaviate infinity api`
Then: `docker compose ps`
Expected: `weaviate` healthy, `api` running. (`infinity` may still be loading FRIDA — fine, not needed for /health.)

- [ ] **Step 5: Verify api reachable on the network (via a one-shot curl container)**

Run: `docker run --rm --network rag-net curlimages/curl:latest -s http://api:8001/health`
Expected: `{"status":"ok"}`.

- [ ] **Step 6: Commit**

```bash
git add docker-compose.yml
git commit -m "feat: add api compose service + weaviate healthcheck"
```

---

## Task 5: Angular scaffold + core (types, ApiService, timecode util)

**Files:**
- Create: `frontend/` via `ng new`
- Create: `frontend/src/app/core/api.types.ts`, `api.service.ts`, `timecode.util.ts`, `timecode.util.spec.ts`

- [ ] **Step 1: Scaffold the Angular app**

From repo root:
```bash
npx -y @angular/cli@latest new frontend --style=css --ssr=false --routing --skip-git --package-manager=npm
```
Accept defaults. Expected: `frontend/` created; `cd frontend; npm run build` works out of the box. (Angular 21 scaffolds standalone + zoneless + the esbuild `application` builder.)

- [ ] **Step 2: Create the API types — `frontend/src/app/core/api.types.ts`**

```typescript
export interface ForwardRequest {
  question: string;
  top_k?: number;
  similarity_threshold?: number;
  use_rewrite: boolean;
  source_title?: string | null;
}

export interface RetrievedDocMeta {
  start_sec?: number | string | null;
  end_sec?: number | string | null;
  source_url?: string | null;
  title?: string | null;
  source_file_name?: string | null;
  [k: string]: unknown;
}

export interface RetrievedDoc {
  text: string;
  metadata: RetrievedDocMeta;
}

export interface ForwardResponse {
  answer: string;
  context?: string;
  retrieved_documents: RetrievedDoc[];
  retrieval_query?: string;
  rewrite_applied?: boolean;
  source_title?: string | null;
  trace_id?: string | null;
}

export interface SourceFilesResponse {
  files: string[];
}

export interface AuthCheckResponse {
  gate: boolean;
  ok: boolean;
}
```

- [ ] **Step 3: Create the timecode util — `frontend/src/app/core/timecode.util.ts`**

```typescript
/** Seconds → "H:MM:SS" (or "M:SS" when under an hour). Returns null for invalid input. */
export function formatTimecode(sec: number | string | null | undefined): string | null {
  if (sec === null || sec === undefined || sec === '') return null;
  const t = Math.floor(Number(sec));
  if (!Number.isFinite(t) || t < 0) return null;
  const h = Math.floor(t / 3600);
  const m = Math.floor((t % 3600) / 60);
  const s = t % 60;
  const pad = (n: number) => String(n).padStart(2, '0');
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`;
}

/** Build a deep link to the source at the given timecode (YouTube gets ?t=<n>s). */
export function deepLink(rawUrl: string | null | undefined, sec: number | string | null | undefined): string | null {
  if (!rawUrl) return null;
  const t = sec === null || sec === undefined || sec === '' ? null : Math.floor(Number(sec));
  try {
    const u = new URL(rawUrl);
    if (t && Number.isFinite(t)) {
      const isYt = /youtube\.com|youtu\.be/.test(u.hostname);
      u.searchParams.set('t', isYt ? `${t}s` : String(t));
    }
    return u.toString();
  } catch {
    return rawUrl;
  }
}
```

- [ ] **Step 4: Write the unit test — `frontend/src/app/core/timecode.util.spec.ts`**

```typescript
import { formatTimecode, deepLink } from './timecode.util';

describe('formatTimecode', () => {
  it('formats under an hour as M:SS', () => expect(formatTimecode(754)).toBe('12:34'));
  it('formats over an hour as H:MM:SS', () => expect(formatTimecode(3725)).toBe('1:02:05'));
  it('returns null for invalid', () => {
    expect(formatTimecode(null)).toBeNull();
    expect(formatTimecode('')).toBeNull();
    expect(formatTimecode(-5)).toBeNull();
  });
});

describe('deepLink', () => {
  it('appends ?t=<n>s for youtube', () =>
    expect(deepLink('https://www.youtube.com/watch?v=abc', 754)).toContain('t=754s'));
  it('returns the url unchanged when no timecode', () =>
    expect(deepLink('https://example.com/v', null)).toBe('https://example.com/v'));
  it('returns null when no url', () => expect(deepLink(null, 10)).toBeNull());
});
```

- [ ] **Step 5: Create the ApiService — `frontend/src/app/core/api.service.ts`**

```typescript
import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../environments/environment';
import {
  AuthCheckResponse, ForwardRequest, ForwardResponse, SourceFilesResponse,
} from './api.types';

@Injectable({ providedIn: 'root' })
export class ApiService {
  private http = inject(HttpClient);
  private base = environment.apiBase; // '' in prod (same origin behind nginx)

  authCheck(): Observable<AuthCheckResponse> {
    return this.http.get<AuthCheckResponse>(`${this.base}/auth-check`);
  }

  sourceFiles(): Observable<SourceFilesResponse> {
    return this.http.get<SourceFilesResponse>(`${this.base}/source-files`);
  }

  forward(req: ForwardRequest): Observable<ForwardResponse> {
    return this.http.post<ForwardResponse>(`${this.base}/forward`, req);
  }
}
```

- [ ] **Step 6: Run the unit test**

Run: `cd frontend; npm test -- --watch=false --browsers=ChromeHeadless`
Expected: the `formatTimecode` / `deepLink` specs PASS. (If Chrome isn't available in the exec env, note it and run `npm run build` instead to confirm compilation — see Step 7.)

- [ ] **Step 7: Confirm it builds**

Run: `cd frontend; npm run build`
Expected: build succeeds; output in `frontend/dist/`.

- [ ] **Step 8: Commit**

```bash
git add frontend
git commit -m "feat: scaffold Angular app + core api service, types, timecode util"
```

---

## Task 6: Search feature component

**Files:**
- Create: `frontend/src/app/features/search/search.component.ts|html|css`
- Add deps: `marked`, `dompurify`

- [ ] **Step 1: Add markdown deps**

Run: `cd frontend; npm install marked dompurify; npm install -D @types/dompurify`
Expected: deps added to `frontend/package.json`.

- [ ] **Step 2: Create `frontend/src/app/features/search/search.component.ts`**

```typescript
import { Component, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { marked } from 'marked';
import DOMPurify from 'dompurify';
import { ApiService } from '../../core/api.service';
import { ForwardResponse, RetrievedDoc } from '../../core/api.types';
import { formatTimecode, deepLink } from '../../core/timecode.util';

@Component({
  selector: 'app-search',
  standalone: true,
  imports: [FormsModule],
  templateUrl: './search.component.html',
  styleUrl: './search.component.css',
})
export class SearchComponent {
  private api = inject(ApiService);

  question = '';
  topK = 5;
  threshold = 0.5;
  useRewrite = true;
  sourceTitle = '';

  sources = signal<string[]>([]);
  loading = signal(false);
  error = signal<string | null>(null);
  result = signal<ForwardResponse | null>(null);

  ngOnInit() {
    this.api.sourceFiles().subscribe({
      next: (r) => this.sources.set(r.files ?? []),
      error: () => {},
    });
  }

  search() {
    const q = this.question.trim();
    if (!q || this.loading()) return;
    this.loading.set(true);
    this.error.set(null);
    this.result.set(null);
    this.api.forward({
      question: q,
      top_k: this.topK,
      similarity_threshold: this.threshold,
      use_rewrite: this.useRewrite,
      source_title: this.sourceTitle || null,
    }).subscribe({
      next: (r) => {
        if (!r || !r.answer) {
          this.error.set('Модель не вернула ответ — попробуйте переформулировать вопрос или снизить порог.');
        } else {
          this.result.set(r);
        }
        this.loading.set(false);
      },
      error: (e) => {
        this.error.set(e?.error?.detail ?? e?.message ?? 'Ошибка запроса');
        this.loading.set(false);
      },
    });
  }

  answerHtml(md: string): string {
    return DOMPurify.sanitize(marked.parse(md, { async: false }) as string);
  }

  timecode = formatTimecode;
  link = deepLink;

  docTitle(d: RetrievedDoc, i: number): string {
    return d.metadata?.title || d.metadata?.source_file_name || `Фрагмент ${i + 1}`;
  }
}
```

- [ ] **Step 3: Create `frontend/src/app/features/search/search.component.html`**

```html
<section class="wrap">
  <h1>DS Navigator</h1>
  <p class="lede">Семантический поиск по транскриптам DS-лекций с ссылками на таймкоды.</p>

  <div class="search-bar">
    <input [(ngModel)]="question" (keydown.enter)="search()"
           placeholder="Например: что такое механизм внимания в трансформерах?" />
    <button (click)="search()" [disabled]="loading()">Найти</button>
  </div>

  <details class="controls">
    <summary>Параметры</summary>
    <label>источников: {{ topK }}
      <input type="range" min="1" max="10" [(ngModel)]="topK" /></label>
    <label>порог: {{ threshold.toFixed(2) }}
      <input type="range" min="0" max="1" step="0.05" [(ngModel)]="threshold" /></label>
    <label><input type="checkbox" [(ngModel)]="useRewrite" /> переформулировать запрос</label>
    <label>область:
      <select [(ngModel)]="sourceTitle">
        <option value="">все материалы</option>
        @for (s of sources(); track s) { <option [value]="s">{{ s }}</option> }
      </select>
    </label>
  </details>

  @if (loading()) { <p class="status">🔍 Поиск по базе…</p> }
  @if (error()) { <p class="err">{{ error() }}</p> }

  @if (result(); as r) {
    @if (r.rewrite_applied && r.retrieval_query) {
      <p class="rewrite">🔁 запрос переформулирован → <b>{{ r.retrieval_query }}</b></p>
    }
    <article class="answer" [innerHTML]="answerHtml(r.answer)"></article>

    @if (r.retrieved_documents?.length) {
      <h2 class="src-h">Источники [{{ r.retrieved_documents.length }}]</h2>
      @for (d of r.retrieved_documents; track $index; let i = $index) {
        <div class="src">
          <div class="idx">{{ (i + 1).toString().padStart(2, '0') }}</div>
          <div>
            <div class="src-meta">
              <span class="title">{{ docTitle(d, i) }}</span>
              @if (timecode(d.metadata.start_sec); as tc) { <span class="chip">◷ {{ tc }}</span> }
            </div>
            <p class="snip">{{ d.text }}</p>
            @if (link(d.metadata.source_url, d.metadata.start_sec); as href) {
              <a class="goto" [href]="href" target="_blank" rel="noopener">→ перейти к источнику</a>
            }
          </div>
        </div>
      }
    }
  }
</section>
```

- [ ] **Step 4: Create `frontend/src/app/features/search/search.component.css`**

```css
.wrap { max-width: 860px; margin: 0 auto; padding: 32px 20px 80px; }
h1 { font-size: 28px; margin: 0 0 4px; }
.lede { color: #666; margin: 0 0 24px; }
.search-bar { display: flex; gap: 10px; }
.search-bar input { flex: 1; padding: 12px 14px; font-size: 16px; border: 1px solid #ccc; border-radius: 10px; }
.search-bar button { padding: 12px 22px; font-weight: 600; border: 0; border-radius: 10px; background: #e3b341; cursor: pointer; }
.search-bar button:disabled { opacity: .5; cursor: not-allowed; }
.controls { margin: 14px 0; display: grid; gap: 8px; color: #555; }
.controls summary { cursor: pointer; }
.status { color: #555; }
.err { color: #c0392b; border: 1px solid #e0b4ab; padding: 12px; border-radius: 8px; }
.rewrite { font-family: monospace; font-size: 13px; color: #2c7a7b; border-left: 2px solid #2c7a7b; padding-left: 12px; }
.answer { background: #faf8f3; border: 1px solid #eee; border-radius: 12px; padding: 22px 24px; margin: 18px 0; line-height: 1.6; }
.src-h { font-size: 13px; letter-spacing: .14em; text-transform: uppercase; color: #999; margin: 28px 0 12px; }
.src { display: grid; grid-template-columns: auto 1fr; gap: 14px; border: 1px solid #eee; border-radius: 12px; padding: 16px 18px; margin-bottom: 10px; }
.idx { font-family: monospace; color: #b8860b; border: 1px solid #e3b341; border-radius: 8px; width: 32px; height: 32px; display: grid; place-items: center; }
.src-meta { display: flex; gap: 10px; align-items: center; margin-bottom: 6px; }
.title { font-weight: 600; }
.chip { font-family: monospace; font-size: 12px; color: #2c7a7b; border: 1px solid #bee3e3; border-radius: 999px; padding: 2px 9px; }
.snip { color: #555; font-size: 14.5px; margin: 0 0 8px; }
.goto { color: #b8860b; font-family: monospace; font-size: 13px; text-decoration: none; }
.goto:hover { text-decoration: underline; }
```

- [ ] **Step 5: Verify it builds**

Run: `cd frontend; npm run build`
Expected: build succeeds (the search component compiles).

- [ ] **Step 6: Commit**

```bash
git add frontend
git commit -m "feat: Angular search view with markdown answer + source cards"
```

---

## Task 7: Auth interceptor, token store, environments, routing

**Files:**
- Create: `frontend/src/app/core/auth.store.ts`, `auth.interceptor.ts`
- Modify: `frontend/src/environments/environment.ts`, create `environment.development.ts`
- Modify: `frontend/src/app/app.config.ts`, `frontend/src/app/app.routes.ts`, `frontend/src/app/app.ts` (root component)
- Create: `frontend/proxy.conf.json`; Modify `frontend/angular.json` (dev proxy)

- [ ] **Step 1: Token store — `frontend/src/app/core/auth.store.ts`**

```typescript
const LS_KEY = 'ds_navigator_token';

/** Reads ?key=… from the URL once (then strips it), else falls back to localStorage. */
export function bootstrapToken(): string {
  const url = new URL(location.href);
  const fromUrl = url.searchParams.get('key');
  if (fromUrl) {
    localStorage.setItem(LS_KEY, fromUrl);
    url.searchParams.delete('key');
    history.replaceState(null, '', url.pathname + url.search + url.hash);
  }
  return localStorage.getItem(LS_KEY) ?? '';
}

export function getToken(): string {
  return localStorage.getItem(LS_KEY) ?? '';
}

export function setToken(t: string): void {
  localStorage.setItem(LS_KEY, t);
}
```

- [ ] **Step 2: Interceptor — `frontend/src/app/core/auth.interceptor.ts`**

```typescript
import { HttpInterceptorFn } from '@angular/common/http';
import { getToken } from './auth.store';

export const authInterceptor: HttpInterceptorFn = (req, next) => {
  const token = getToken();
  if (token) {
    req = req.clone({ setHeaders: { 'X-Demo-Token': token } });
  }
  return next(req);
};
```

- [ ] **Step 3: Environments**

`frontend/src/environments/environment.ts` (prod — same origin behind nginx):

```typescript
export const environment = { apiBase: '' };
```

`frontend/src/environments/environment.development.ts`:

```typescript
export const environment = { apiBase: '' }; // dev uses ng serve proxy → same paths
```

- [ ] **Step 4: Wire providers — `frontend/src/app/app.config.ts`**

Ensure the config provides HttpClient with the interceptor and the router. Replace the file body with (keep whatever zoneless/error providers `ng new` generated, add HttpClient + interceptor):

```typescript
import { ApplicationConfig, provideZonelessChangeDetection } from '@angular/core';
import { provideRouter } from '@angular/router';
import { provideHttpClient, withInterceptors } from '@angular/common/http';
import { routes } from './app.routes';
import { authInterceptor } from './core/auth.interceptor';
import { bootstrapToken } from './core/auth.store';

bootstrapToken(); // capture ?key= before the first HTTP call

export const appConfig: ApplicationConfig = {
  providers: [
    provideZonelessChangeDetection(),
    provideRouter(routes),
    provideHttpClient(withInterceptors([authInterceptor])),
  ],
};
```

> If `ng new` did NOT enable zoneless, omit `provideZonelessChangeDetection()` and keep its generated change-detection provider instead.

- [ ] **Step 5: Routes — `frontend/src/app/app.routes.ts`**

```typescript
import { Routes } from '@angular/router';
import { SearchComponent } from './features/search/search.component';

export const routes: Routes = [
  { path: '', component: SearchComponent },
  { path: '**', redirectTo: '' },
];
```

- [ ] **Step 6: Root component renders the router outlet**

Edit `frontend/src/app/app.ts` (or `app.component.ts`) template to be exactly:

```typescript
import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet],
  template: '<router-outlet />',
})
export class App {}
```

> Keep the class name `ng new` generated (`App` or `AppComponent`) and update `main.ts`/bootstrap reference accordingly if you rename.

- [ ] **Step 7: Dev proxy — `frontend/proxy.conf.json`**

```json
{
  "/forward": { "target": "http://localhost:8001", "secure": false },
  "/source-files": { "target": "http://localhost:8001", "secure": false },
  "/auth-check": { "target": "http://localhost:8001", "secure": false },
  "/ingest": { "target": "http://localhost:8001", "secure": false }
}
```

In `frontend/angular.json`, under `projects.frontend.architect.serve.options`, add: `"proxyConfig": "proxy.conf.json"`.

- [ ] **Step 8: Build**

Run: `cd frontend; npm run build`
Expected: build succeeds.

- [ ] **Step 9: Commit**

```bash
git add frontend
git commit -m "feat: Angular auth interceptor, token bootstrap, routing, dev proxy"
```

---

## Task 8: frontend Dockerfile + nginx service (TLS-ready)

Bake the Angular bundle into a `jonasal/nginx-certbot` image and add the nginx compose service. Local verification uses the image's self-signed dummy cert (real Let's Encrypt is a deploy-time step requiring the real domain + open ports).

**Files:**
- Create: `frontend/Dockerfile`
- Create: `nginx/user_conf.d/dsnavigator.conf`, `nginx/proxy_common.conf`
- Modify: `docker-compose.yml` (add `nginx` service + `letsencrypt` volume)

- [ ] **Step 1: `frontend/Dockerfile` (multi-stage: build → bake into nginx-certbot)**

Build context is the **repo root** (`context: .` in compose), so all `COPY` paths are repo-root-relative:

```dockerfile
# syntax=docker/dockerfile:1
FROM node:22-alpine AS build
WORKDIR /app
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build

FROM jonasal/nginx-certbot:6.2.0
# Angular static bundle (browser output of the `frontend` project) served by nginx
COPY --from=build /app/dist/frontend/browser /usr/share/nginx/html
# nginx server config + shared proxy headers (from the repo-root context)
COPY nginx/user_conf.d /etc/nginx/user_conf.d
COPY nginx/proxy_common.conf /etc/nginx/proxy_common.conf
```

> The Angular output path is `dist/frontend/browser` for a project named `frontend` (the esbuild `application` builder default). If `ng new` chose a different `outputPath` in `angular.json`, match it here.

- [ ] **Step 2: `nginx/proxy_common.conf`**

```nginx
proxy_http_version 1.1;
proxy_set_header Host $host;
proxy_set_header X-Real-IP $remote_addr;
proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
proxy_set_header X-Forwarded-Proto $scheme;
```

- [ ] **Step 3: `nginx/user_conf.d/dsnavigator.conf`**

```nginx
upstream rag_api { server api:8001; keepalive 16; }

server {
    listen 80;
    server_name ${DOMAIN};
    location /.well-known/acme-challenge/ { root /var/www/letsencrypt; }
    location / { return 301 https://$host$request_uri; }
}

server {
    listen 443 ssl;
    http2 on;
    server_name ${DOMAIN};
    ssl_certificate     /etc/letsencrypt/live/${DOMAIN}/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/${DOMAIN}/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;

    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "DENY" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    client_max_body_size 25m;
    root /usr/share/nginx/html;
    index index.html;

    location = /forward {
        proxy_pass http://rag_api;
        include /etc/nginx/proxy_common.conf;
        proxy_read_timeout 120s;
        proxy_buffering off;
    }
    location ~ ^/(source-files|check-vdb|api|auth-check|health)(/|$) {
        proxy_pass http://rag_api;
        include /etc/nginx/proxy_common.conf;
    }
    location ~* \.(?:js|css|woff2?|png|jpg|svg|ico|webp|map)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
        try_files $uri =404;
    }
    location = /index.html { add_header Cache-Control "no-cache"; }
    location / { try_files $uri $uri/ /index.html; }
}
```

> Phase 1 proxies only search-related routes. `/ingest*` rate-limits, `/docs` VPN-gating, HSTS, and Turnstile come in Phase 2/3. `${DOMAIN}` is substituted by the jonasal image from the `DOMAIN` env var.

- [ ] **Step 4: Add the `nginx` service to `docker-compose.yml`**

```yaml
  nginx:
    build:
      context: .
      dockerfile: frontend/Dockerfile
    image: ds-navigator-web:latest
    container_name: ds-navigator-nginx
    restart: unless-stopped
    networks:
      - rag-net
    ports:
      - "80:80"
      - "443:443"
    profiles:
      - web
      - full
    environment:
      DOMAIN: ${DOMAIN:-localhost}
      CERTBOT_EMAIL: ${CERTBOT_EMAIL:-admin@example.com}
      STAGING: "1"        # use LE staging until the real domain is verified; flip to 0 for prod
    volumes:
      - letsencrypt:/etc/letsencrypt
    depends_on:
      - api
    <<: *shared-logs
```

Add to the top-level `volumes:` block: `  letsencrypt:`

- [ ] **Step 5: Build the web image**

Run: `docker compose build nginx`
Expected: Angular builds in-image; final `ds-navigator-web:latest` produced.

- [ ] **Step 6: Bring up the full web stack and verify through nginx**

Run: `docker compose --profile vdb --profile emb --profile web up -d`
Then (the jonasal image serves a self-signed dummy cert until real LE issuance, so use `-k`):
- `curl -ksS https://localhost/health` → expected `{"status":"ok"}` (proxied to api).
- `curl -ksS https://localhost/ | Select-String "<app-root"` → expected the Angular `index.html` shell.
- `curl -ksS http://localhost/health -I` → expected `301` redirect to https.

Expected: nginx serves the Angular shell on 443 and proxies API routes to the `api` container. (Real cert issuance fails for `localhost` — that's expected locally; it succeeds at deploy time with the real `DOMAIN` and open ports per the Phase-3 runbook.)

- [ ] **Step 7: Commit**

```bash
git add frontend/Dockerfile nginx docker-compose.yml
git commit -m "feat: nginx (TLS-ready) serves Angular bundle + proxies API"
```

---

## Phase 1 Self-Review checklist (run before handoff)

- [ ] `tests/test_lazy_vectorstore.py` passes; importing `api.main` opens no Weaviate connection.
- [ ] `docker build -f Dockerfile.api .` succeeds; container `/health` returns ok with Weaviate up and before any query.
- [ ] `docker compose config` is valid; `weaviate` reports healthy; `api` reachable at `http://api:8001` on `rag-net`.
- [ ] `cd frontend; npm run build` succeeds; timecode unit specs pass.
- [ ] Through nginx: `https://localhost/health` proxies to api; `https://localhost/` serves the Angular shell; `http://localhost/` 301s to https.
- [ ] `SERVE_SPA=false` in the api container (Angular is the only frontend in prod); local `uvicorn` still serves `src/web/index.html` by default.

## Deploy-time steps (NOT part of Phase 1 code — owner does these later)
- Set real `DOMAIN` + `CERTBOT_EMAIL`, flip `STAGING=0`, ensure ports 80/443 forwarded end-to-end (Phase-3 runbook) → real Let's Encrypt cert.
- Verify the public IP is not CGNAT before any of this (design §12 #1).

---

## Next phases (separate plans)
- **Phase 2:** Ingest view (YouTube + upload w/ progress), async jobs (`202` + `GET /ingest-status/{id}`), Cloudflare Turnstile on `/ingest*`, nginx ingest rate-limits + large-upload tuning, SSRF allowlist.
- **Phase 3:** wg-easy WireGuard service (+ WSL2 `ip6_tables` caveat), pull admin dashboards off public ports (VPN-only), HSTS/CSP, `/docs` VPN-gating, `DEMO_ACCESS_TOKEN` hardening, the double-NAT network runbook (`docs/runbook-network.md`).
