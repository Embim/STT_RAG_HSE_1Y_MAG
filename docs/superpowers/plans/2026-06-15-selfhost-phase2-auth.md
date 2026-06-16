# Self-hosted publication — Phase 2 (Auth) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`). **Commit hygiene (student project): every commit message must be plain — NO `Co-Authored-By`, NO mention of Claude/AI/Anthropic/"generated with".**

**Goal:** Put the whole site behind a login. Admin-issued accounts only (no self-registration). SQLite users DB, bcrypt password hashing, JWT bearer sessions, roles `user`/`admin`. First admin bootstrapped from env. Angular gets a login page, route guards, a JWT interceptor, and an admin page to create/list/deactivate users.

**Architecture:** FastAPI gains an `auth` package (SQLModel `User` in SQLite, bcrypt hashing, PyJWT tokens) + `get_current_user`/`require_admin` dependencies that protect every data route. The old `DEMO_ACCESS_TOKEN` middleware and `/auth-check` are removed (replaced by real auth). Angular stores the JWT, sends it as `Authorization: Bearer`, guards routes, and on 401 redirects to `/login`. nginx proxies the new `/auth/*` and `/admin/*` routes.

**Tech Stack:** FastAPI, SQLModel (SQLite), bcrypt, PyJWT, Angular 19 (standalone, signals), nginx.

**Scope note:** Phase 2 of 4 (after Phase 1 infra). Phase 3 = ingest UI + async jobs (behind auth). Phase 4 = WireGuard + network runbook + prod hardening. **YAGNI:** no refresh tokens, no server-side session store, no password reset, no email — JWT is stateless; logout = drop the token client-side; expired token = re-login.

**Prerequisites:** Phase 1 merged/complete on branch `feat/selfhost-phase1` (or a Phase-2 branch off it). Docker running, Node 22, uv.

---

## File Structure

**Created (backend):**
- `src/system/auth/__init__.py`
- `src/system/auth/models.py` — SQLModel `User` table.
- `src/system/auth/db.py` — engine, `init_db()`, `get_session()`.
- `src/system/auth/security.py` — `hash_password`, `verify_password`, `create_access_token`, `decode_token`.
- `src/system/auth/service.py` — `get_user`, `create_user`, `authenticate_user`, `list_users`, `set_active`, `ensure_admin`.
- `src/system/auth/deps.py` — `get_current_user`, `require_admin` FastAPI dependencies.
- `src/api/auth_routes.py` — `/auth/login`, `/auth/me`, `/admin/users` (POST/GET), `/admin/users/{username}/active`.
- `tests/test_auth.py` — unit + integration tests.

**Created (frontend):**
- `frontend/src/app/core/auth.types.ts`
- `frontend/src/app/core/auth.service.ts`
- `frontend/src/app/core/auth.interceptor.ts`
- `frontend/src/app/core/auth.guard.ts` (functional `authGuard`, `adminGuard`)
- `frontend/src/app/features/login/login.component.{ts,html,css}`
- `frontend/src/app/features/admin/admin.component.{ts,html,css}`

**Modified:**
- `src/settings.py` — JWT/admin/db settings.
- `src/api/main.py` — include auth router; `ensure_admin()` at startup; remove `DEMO_ACCESS_TOKEN` middleware + `/auth-check`; protect data routes with `Depends(get_current_user)`.
- `pyproject.toml` + `uv.lock` — add `sqlmodel`, `bcrypt`, `pyjwt`.
- `docker-compose.yml` — api env (`JWT_SECRET`, `ADMIN_USERNAME`, `ADMIN_PASSWORD`, `AUTH_DB_PATH`); `auth-data` volume; nginx proxy `/auth` + `/admin`.
- `Dockerfile.api` — create `/app/auth` owned by appuser (writable SQLite dir).
- `nginx/user_conf.d/dsnavigator.conf` — proxy `/auth` + `/admin`; drop `auth-check`.
- `frontend/src/app/app.config.ts` — `withInterceptors([authInterceptor])` + app initializer that calls `loadMe()` when a token exists.
- `frontend/src/app/app.routes.ts` — `/login`, `authGuard` on `''`, `adminGuard` on `/admin`.
- `frontend/src/app/app.component.html` — header (current user + logout + admin link).
- `.gitignore` — add `auth/` (local SQLite db).

---

## Task A1: Backend auth core (settings, deps, SQLModel store, hashing, JWT) + unit tests

**Files:** modify `pyproject.toml`, `src/settings.py`; create `src/system/auth/{__init__,models,db,security,service}.py`, `tests/test_auth.py`.

- [ ] **Step 1: add deps**
`uv add sqlmodel bcrypt pyjwt` (updates `pyproject.toml` + `uv.lock`). Verify: `uv run python -c "import sqlmodel, bcrypt, jwt; print('ok')"`.

- [ ] **Step 2: settings** — append to the `Settings` class in `src/settings.py`:
```python
    # ── Auth (Phase 2) ──────────────────────────────────────────────
    JWT_SECRET: str = "dev-insecure-change-me"   # MUST override in .env for prod
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 720                # 12h
    ADMIN_USERNAME: str = "admin"
    ADMIN_PASSWORD: str = ""                      # set in .env → admin auto-created on startup
    AUTH_DB_PATH: str = ""                        # empty → <repo>/auth/users.db
```

- [ ] **Step 3: `src/system/auth/__init__.py`** — empty file.

- [ ] **Step 4: `src/system/auth/models.py`**
```python
from datetime import datetime, timezone
from sqlmodel import SQLModel, Field


class User(SQLModel, table=True):
    username: str = Field(primary_key=True)
    password_hash: str
    role: str = Field(default="user")  # "user" | "admin"
    is_active: bool = Field(default=True)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
```

- [ ] **Step 5: `src/system/auth/db.py`**
```python
from pathlib import Path
from sqlmodel import SQLModel, Session, create_engine
from settings import settings

_engine = None


def _db_path() -> str:
    if settings.AUTH_DB_PATH:
        return settings.AUTH_DB_PATH
    # src/system/auth/db.py -> parents[3] == repo root
    return str(Path(__file__).resolve().parents[3] / "auth" / "users.db")


def get_engine():
    global _engine
    if _engine is None:
        path = _db_path()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})
    return _engine


def init_db() -> None:
    import system.auth.models  # noqa: F401 — register the table
    SQLModel.metadata.create_all(get_engine())


def get_session() -> Session:
    return Session(get_engine())
```

- [ ] **Step 6: `src/system/auth/security.py`**
```python
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

from settings import settings


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def create_access_token(username: str, role: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": username,
        "role": role,
        "iat": now,
        "exp": now + timedelta(minutes=settings.JWT_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    return jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
```

- [ ] **Step 7: `src/system/auth/service.py`**
```python
import logging
from typing import List, Optional

from sqlmodel import select

from settings import settings
from system.auth.db import get_session, init_db
from system.auth.models import User
from system.auth.security import hash_password, verify_password

logger = logging.getLogger(__name__)


def get_user(username: str) -> Optional[User]:
    with get_session() as s:
        return s.get(User, username)


def create_user(username: str, password: str, role: str = "user") -> User:
    with get_session() as s:
        if s.get(User, username):
            raise ValueError("user already exists")
        user = User(username=username, password_hash=hash_password(password), role=role)
        s.add(user)
        s.commit()
        s.refresh(user)
        return user


def authenticate_user(username: str, password: str) -> Optional[User]:
    user = get_user(username)
    if not user or not user.is_active:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user


def list_users() -> List[User]:
    with get_session() as s:
        return list(s.exec(select(User)).all())


def set_active(username: str, active: bool) -> Optional[User]:
    with get_session() as s:
        user = s.get(User, username)
        if not user:
            return None
        user.is_active = active
        s.add(user)
        s.commit()
        s.refresh(user)
        return user


def ensure_admin() -> None:
    """Create the env-configured admin on startup, if set and absent."""
    init_db()
    if not settings.ADMIN_PASSWORD:
        logger.info("ADMIN_PASSWORD not set — skipping admin bootstrap")
        return
    if get_user(settings.ADMIN_USERNAME):
        return
    create_user(settings.ADMIN_USERNAME, settings.ADMIN_PASSWORD, role="admin")
    logger.info("Bootstrapped admin user %r", settings.ADMIN_USERNAME)
```

- [ ] **Step 8: write `tests/test_auth.py` (TDD — write before deps wired if you like; here deps exist so write + run)**
```python
import os
os.environ.setdefault("LLM_MODEL", "test/model")
os.environ.setdefault("LLM_API_KEY_1", "x")
os.environ.setdefault("LLM_API_KEY_2", "x")
os.environ.setdefault("LLM_API_KEY_3", "x")

import pytest


@pytest.fixture
def auth(tmp_path, monkeypatch):
    # Point the auth DB at a temp file and reset the cached engine.
    monkeypatch.setenv("AUTH_DB_PATH", str(tmp_path / "users.db"))
    import importlib
    from settings import settings as s
    s.AUTH_DB_PATH = str(tmp_path / "users.db")
    import system.auth.db as db
    db._engine = None
    import system.auth.service as service
    importlib.reload(service)
    db.init_db()
    return service


def test_hash_roundtrip():
    from system.auth.security import hash_password, verify_password
    h = hash_password("secret123")
    assert h != "secret123"
    assert verify_password("secret123", h)
    assert not verify_password("wrong", h)


def test_token_roundtrip():
    from system.auth.security import create_access_token, decode_token
    tok = create_access_token("alice", "admin")
    payload = decode_token(tok)
    assert payload["sub"] == "alice"
    assert payload["role"] == "admin"


def test_create_and_authenticate(auth):
    auth.create_user("bob", "pw", role="user")
    assert auth.authenticate_user("bob", "pw") is not None
    assert auth.authenticate_user("bob", "bad") is None
    assert auth.authenticate_user("ghost", "pw") is None


def test_duplicate_user_rejected(auth):
    auth.create_user("dup", "pw")
    with pytest.raises(ValueError):
        auth.create_user("dup", "pw2")


def test_inactive_user_cannot_authenticate(auth):
    auth.create_user("carol", "pw")
    auth.set_active("carol", False)
    assert auth.authenticate_user("carol", "pw") is None
```

- [ ] **Step 9: run tests** — `uv run pytest tests/test_auth.py -v` → all pass.

- [ ] **Step 10: commit** — stage `pyproject.toml uv.lock src/settings.py src/system/auth/ tests/test_auth.py`. Message: `feat: auth core (sqlite users, bcrypt hashing, jwt tokens)`

---

## Task A2: FastAPI auth wiring (deps, routes, protect endpoints, startup bootstrap)

**Files:** create `src/system/auth/deps.py`, `src/api/auth_routes.py`; modify `src/api/main.py`; extend `tests/test_auth.py`.

- [ ] **Step 1: `src/system/auth/deps.py`**
```python
import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from system.auth.models import User
from system.auth.security import decode_token
from system.auth.service import get_user

_bearer = HTTPBearer(auto_error=False)


def get_current_user(creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> User:
    if creds is None:
        raise HTTPException(status_code=401, detail="not authenticated")
    try:
        payload = decode_token(creds.credentials)
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="invalid or expired token")
    username = payload.get("sub")
    user = get_user(username) if username else None
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="user not found or inactive")
    return user


def require_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="admin only")
    return user
```

- [ ] **Step 2: `src/api/auth_routes.py`**
```python
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from system.auth import service
from system.auth.deps import get_current_user, require_admin
from system.auth.models import User
from system.auth.security import create_access_token

router = APIRouter()


class LoginRequest(BaseModel):
    username: str
    password: str


class CreateUserRequest(BaseModel):
    username: str
    password: str
    role: str = "user"


class UserOut(BaseModel):
    username: str
    role: str
    is_active: bool


@router.post("/auth/login", tags=["Auth"])
def login(req: LoginRequest):
    user = service.authenticate_user(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="invalid credentials")
    token = create_access_token(user.username, user.role)
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {"username": user.username, "role": user.role},
    }


@router.get("/auth/me", tags=["Auth"])
def me(user: User = Depends(get_current_user)):
    return {"username": user.username, "role": user.role}


@router.post("/admin/users", response_model=UserOut, tags=["Admin"])
def admin_create_user(req: CreateUserRequest, _: User = Depends(require_admin)):
    if req.role not in ("user", "admin"):
        raise HTTPException(status_code=400, detail="invalid role")
    try:
        u = service.create_user(req.username, req.password, req.role)
    except ValueError:
        raise HTTPException(status_code=409, detail="user already exists")
    return UserOut(username=u.username, role=u.role, is_active=u.is_active)


@router.get("/admin/users", response_model=list[UserOut], tags=["Admin"])
def admin_list_users(_: User = Depends(require_admin)):
    return [UserOut(username=u.username, role=u.role, is_active=u.is_active) for u in service.list_users()]


@router.post("/admin/users/{username}/active", response_model=UserOut, tags=["Admin"])
def admin_set_active(username: str, active: bool, _: User = Depends(require_admin)):
    u = service.set_active(username, active)
    if not u:
        raise HTTPException(status_code=404, detail="not found")
    return UserOut(username=u.username, role=u.role, is_active=u.is_active)
```

- [ ] **Step 3: modify `src/api/main.py`**
  1. Remove the `DEMO_ACCESS_TOKEN` block (the `DEMO_ACCESS_TOKEN`/`_PROTECTED_PATHS` vars and the `@app.middleware("http") access_gate` function) AND the `@app.get("/auth-check")` route — they're replaced by real auth.
  2. Add imports: `from system.auth.deps import get_current_user` and `from system.auth.models import User` and `from system.auth.service import ensure_admin` and `from api.auth_routes import router as auth_router`.
  3. Bootstrap admin + init DB at startup in the lifespan. Change `lifespan` to:
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_admin()
    yield
    langfuse_flush()
```
  4. Register the auth router right after `app = FastAPI(...)`: `app.include_router(auth_router)`.
  5. Protect the data endpoints by adding a dependency parameter to each (keep `/health` and `/api` open):
     - `forward`: `async def forward(req: ForwardRequest, user: User = Depends(get_current_user)):`
     - `source_files`: `async def source_files(user: User = Depends(get_current_user)):`
     - `check_vdb`: `async def check_vdb(user: User = Depends(get_current_user)):`
     - `ingest`: `async def ingest(req: IngestRequest, user: User = Depends(get_current_user)):`
     - `ingest_upload`: add `user: User = Depends(get_current_user)` parameter.
     (Add `from fastapi import Depends` to the existing fastapi import line.)

- [ ] **Step 4: extend `tests/test_auth.py` with API-level tests** (append):
```python
def _client(auth, monkeypatch):
    # Build a TestClient whose app shares the temp auth DB from the `auth` fixture.
    from fastapi.testclient import TestClient
    from unittest.mock import patch
    with patch("system.rag.vectore_store.weaviate.connect_to_local"):
        import importlib
        import api.main as main
        importlib.reload(main)
        return TestClient(main.app)


def test_login_and_protected_route(auth, monkeypatch):
    auth.create_user("alice", "pw", role="admin")
    c = _client(auth, monkeypatch)
    # no token -> 401
    assert c.post("/forward", json={"question": "x", "use_rewrite": False}).status_code == 401
    # bad creds -> 401
    assert c.post("/auth/login", json={"username": "alice", "password": "bad"}).status_code == 401
    # good creds -> token
    r = c.post("/auth/login", json={"username": "alice", "password": "pw"})
    assert r.status_code == 200
    token = r.json()["access_token"]
    me = c.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.json()["role"] == "admin"


def test_admin_only(auth, monkeypatch):
    auth.create_user("admin1", "pw", role="admin")
    auth.create_user("user1", "pw", role="user")
    c = _client(auth, monkeypatch)
    admin_tok = c.post("/auth/login", json={"username": "admin1", "password": "pw"}).json()["access_token"]
    user_tok = c.post("/auth/login", json={"username": "user1", "password": "pw"}).json()["access_token"]
    # user cannot create users
    assert c.post("/admin/users", json={"username": "z", "password": "p"},
                  headers={"Authorization": f"Bearer {user_tok}"}).status_code == 403
    # admin can
    assert c.post("/admin/users", json={"username": "z", "password": "p"},
                  headers={"Authorization": f"Bearer {admin_tok}"}).status_code == 200
```
> NOTE: `_client` reloads `api.main` so it picks up the temp auth DB. If reload ordering causes the protected routes to import a stale `get_current_user`, instead construct the client once per test after the `auth` fixture. If `ensure_admin()` in lifespan needs `ADMIN_PASSWORD` unset for these tests, it already no-ops when empty.

- [ ] **Step 5: run** — `uv run pytest tests/test_auth.py -v` → all pass. Also `cd src; uv run python -m py_compile api/main.py api/auth_routes.py system/auth/deps.py`.

- [ ] **Step 6: commit** — stage `src/system/auth/deps.py src/api/auth_routes.py src/api/main.py tests/test_auth.py`. Message: `feat: jwt-protected API routes + admin user endpoints`

---

## Task A3: Container + nginx + .gitignore wiring

**Files:** modify `docker-compose.yml`, `Dockerfile.api`, `nginx/user_conf.d/dsnavigator.conf`, `.gitignore`.

- [ ] **Step 1: `.gitignore`** — append `auth/` (the local SQLite users db).

- [ ] **Step 2: `Dockerfile.api`** — ensure the auth DB dir is writable by the non-root user. After the `useradd` line and BEFORE `USER appuser`, add:
```dockerfile
RUN mkdir -p /app/auth && chown -R appuser:appuser /app/auth
ENV AUTH_DB_PATH=/app/auth/users.db
```

- [ ] **Step 3: `docker-compose.yml`** — in the `api` service `environment:` add:
```yaml
      JWT_SECRET: ${JWT_SECRET:-dev-insecure-change-me}
      ADMIN_USERNAME: ${ADMIN_USERNAME:-admin}
      ADMIN_PASSWORD: ${ADMIN_PASSWORD:-}
      AUTH_DB_PATH: /app/auth/users.db
```
Add a volume mount to the `api` service:
```yaml
    volumes:
      - auth-data:/app/auth
```
And declare `auth-data:` under the top-level `volumes:` block. (Persists users across container restarts.)

- [ ] **Step 4: `nginx/user_conf.d/dsnavigator.conf`** — update the API proxy regex location: replace the line
```nginx
    location ~ ^/(source-files|check-vdb|api|auth-check|health)(/|$) {
```
with
```nginx
    location ~ ^/(source-files|check-vdb|api|auth|admin|health)(/|$) {
```
(`/auth/login`, `/auth/me`, `/admin/users` now proxy to the api; the removed `auth-check` is gone.)

- [ ] **Step 5: verify** — `docker compose --profile vdb --profile web config >/dev/null && echo OK`. (Full live verify is Task A7.)

- [ ] **Step 6: commit** — stage the 4 files. Message: `feat: wire auth env + volume into compose; proxy /auth and /admin`

---

## Task A4: Angular auth service, interceptor, guards

**Files:** create `frontend/src/app/core/auth.types.ts`, `auth.service.ts`, `auth.interceptor.ts`, `auth.guard.ts`.

- [ ] **Step 1: `auth.types.ts`**
```typescript
export interface AuthUser { username: string; role: string; }
export interface LoginResponse { access_token: string; token_type: string; user: AuthUser; }
```

- [ ] **Step 2: `auth.service.ts`**
```typescript
import { Injectable, inject, signal, computed } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, tap } from 'rxjs';
import { environment } from '../../environments/environment';
import { AuthUser, LoginResponse } from './auth.types';

const TOKEN_KEY = 'ds_token';

@Injectable({ providedIn: 'root' })
export class AuthService {
  private http = inject(HttpClient);
  private base = environment.apiBase;
  private _user = signal<AuthUser | null>(null);

  readonly user = this._user.asReadonly();
  readonly isAuthenticated = computed(() => this._user() !== null);
  readonly isAdmin = computed(() => this._user()?.role === 'admin');

  token(): string | null { return localStorage.getItem(TOKEN_KEY); }
  hasToken(): boolean { return !!localStorage.getItem(TOKEN_KEY); }

  login(username: string, password: string): Observable<LoginResponse> {
    return this.http.post<LoginResponse>(`${this.base}/auth/login`, { username, password }).pipe(
      tap((r) => { localStorage.setItem(TOKEN_KEY, r.access_token); this._user.set(r.user); }),
    );
  }

  loadMe(): Observable<AuthUser> {
    return this.http.get<AuthUser>(`${this.base}/auth/me`).pipe(tap((u) => this._user.set(u)));
  }

  logout(): void { localStorage.removeItem(TOKEN_KEY); this._user.set(null); }
}
```

- [ ] **Step 3: `auth.interceptor.ts`**
```typescript
import { HttpInterceptorFn } from '@angular/common/http';
import { inject } from '@angular/core';
import { Router } from '@angular/router';
import { catchError, throwError } from 'rxjs';
import { AuthService } from './auth.service';

export const authInterceptor: HttpInterceptorFn = (req, next) => {
  const auth = inject(AuthService);
  const router = inject(Router);
  const token = auth.token();
  const authedReq = token ? req.clone({ setHeaders: { Authorization: `Bearer ${token}` } }) : req;
  return next(authedReq).pipe(
    catchError((err) => {
      if (err?.status === 401) {
        auth.logout();
        router.navigate(['/login']);
      }
      return throwError(() => err);
    }),
  );
};
```

- [ ] **Step 4: `auth.guard.ts`**
```typescript
import { inject } from '@angular/core';
import { CanActivateFn, Router } from '@angular/router';
import { AuthService } from './auth.service';

export const authGuard: CanActivateFn = () => {
  const router = inject(Router);
  if (inject(AuthService).hasToken()) return true;
  router.navigate(['/login']);
  return false;
};

export const adminGuard: CanActivateFn = () => {
  const auth = inject(AuthService);
  const router = inject(Router);
  if (auth.isAdmin()) return true;
  router.navigate(['/']);
  return false;
};
```

- [ ] **Step 5: build** — `cd frontend; npm run build` → succeeds (these are wired in Task A5/A6 routing; building now just type-checks them — they may tree-shake out, that's fine).

- [ ] **Step 6: commit** — stage the 4 core files. Message: `feat: Angular auth service, bearer interceptor, route guards`

---

## Task A5: Login page + routing wiring + app initializer

**Files:** create `frontend/src/app/features/login/login.component.{ts,html,css}`; modify `frontend/src/app/app.routes.ts`, `frontend/src/app/app.config.ts`.

- [ ] **Step 1: `login.component.ts`**
```typescript
import { Component, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { AuthService } from '../../core/auth.service';

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [FormsModule],
  templateUrl: './login.component.html',
  styleUrl: './login.component.css',
})
export class LoginComponent {
  private auth = inject(AuthService);
  private router = inject(Router);

  username = '';
  password = '';
  loading = signal(false);
  error = signal<string | null>(null);

  submit(): void {
    if (!this.username || !this.password || this.loading()) return;
    this.loading.set(true);
    this.error.set(null);
    this.auth.login(this.username.trim(), this.password).subscribe({
      next: () => { this.loading.set(false); this.router.navigate(['/']); },
      error: (e) => {
        this.loading.set(false);
        this.error.set(e?.status === 401 ? 'Неверный логин или пароль' : (e?.error?.detail ?? 'Ошибка входа'));
      },
    });
  }
}
```

- [ ] **Step 2: `login.component.html`**
```html
<section class="login">
  <h1>DS Navigator</h1>
  <p class="sub">Вход в систему</p>
  <input [(ngModel)]="username" (keydown.enter)="submit()" placeholder="Логин" autocomplete="username" />
  <input [(ngModel)]="password" (keydown.enter)="submit()" type="password" placeholder="Пароль" autocomplete="current-password" />
  <button (click)="submit()" [disabled]="loading()">{{ loading() ? 'Вход…' : 'Войти' }}</button>
  @if (error()) { <p class="err">{{ error() }}</p> }
</section>
```

- [ ] **Step 3: `login.component.css`**
```css
.login { max-width: 340px; margin: 12vh auto; display: grid; gap: 12px; padding: 0 20px; }
h1 { font-size: 24px; margin: 0; }
.sub { color: #777; margin: 0 0 8px; }
.login input { padding: 12px 14px; font-size: 15px; border: 1px solid #ccc; border-radius: 10px; }
.login button { padding: 12px; font-weight: 600; border: 0; border-radius: 10px; background: #e3b341; cursor: pointer; }
.login button:disabled { opacity: .5; cursor: not-allowed; }
.err { color: #c0392b; margin: 0; }
```

- [ ] **Step 4: `app.routes.ts`** — replace with:
```typescript
import { Routes } from '@angular/router';
import { authGuard, adminGuard } from './core/auth.guard';

export const routes: Routes = [
  { path: 'login', loadComponent: () => import('./features/login/login.component').then(m => m.LoginComponent) },
  { path: '', canActivate: [authGuard], loadComponent: () => import('./features/search/search.component').then(m => m.SearchComponent) },
  { path: 'admin', canActivate: [adminGuard], loadComponent: () => import('./features/admin/admin.component').then(m => m.AdminComponent) },
  { path: '**', redirectTo: '' },
];
```
> NOTE: the `admin` route imports a component created in Task A6. If you implement A5 before A6, temporarily point `admin` at the search component or create an empty `AdminComponent` stub so the build passes; A6 fills it in. Prefer doing A6's component file first if running out of order.

- [ ] **Step 5: `app.config.ts`** — wire the interceptor and an initializer that restores the session when a token exists:
```typescript
import { ApplicationConfig, provideZonelessChangeDetection, provideAppInitializer, inject } from '@angular/core';
import { provideRouter } from '@angular/router';
import { provideHttpClient, withInterceptors } from '@angular/common/http';
import { firstValueFrom } from 'rxjs';
import { routes } from './app.routes';
import { authInterceptor } from './core/auth.interceptor';
import { AuthService } from './core/auth.service';

export const appConfig: ApplicationConfig = {
  providers: [
    provideZonelessChangeDetection(),
    provideRouter(routes),
    provideHttpClient(withInterceptors([authInterceptor])),
    provideAppInitializer(() => {
      const auth = inject(AuthService);
      if (!auth.hasToken()) return;
      return firstValueFrom(auth.loadMe()).catch(() => auth.logout());
    }),
  ],
};
```
> Keep whatever change-detection provider `ng new` generated (zoneless or not). If the project is NOT zoneless, drop `provideZonelessChangeDetection()` and keep the existing provider. `provideAppInitializer` is Angular 19+. If unavailable, use the classic `{ provide: APP_INITIALIZER, useFactory, multi: true }` form.

- [ ] **Step 6: build** — `cd frontend; npm run build` → succeeds (create the `AdminComponent` stub first if needed — see A6).

- [ ] **Step 7: commit** — stage login component, `app.routes.ts`, `app.config.ts`. Message: `feat: Angular login page, route guards wiring, session restore`

---

## Task A6: Admin page (create/list users) + header with logout

**Files:** create `frontend/src/app/features/admin/admin.component.{ts,html,css}`; modify `frontend/src/app/app.component.html` (+ `.ts` imports).

- [ ] **Step 1: `admin.component.ts`**
```typescript
import { Component, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { environment } from '../../environments/environment';

interface UserOut { username: string; role: string; is_active: boolean; }

@Component({
  selector: 'app-admin',
  standalone: true,
  imports: [FormsModule],
  templateUrl: './admin.component.html',
  styleUrl: './admin.component.css',
})
export class AdminComponent {
  private http = inject(HttpClient);
  private base = environment.apiBase;

  users = signal<UserOut[]>([]);
  newUsername = '';
  newPassword = '';
  newRole = 'user';
  error = signal<string | null>(null);
  ok = signal<string | null>(null);

  ngOnInit(): void { this.refresh(); }

  refresh(): void {
    this.http.get<UserOut[]>(`${this.base}/admin/users`).subscribe({
      next: (u) => this.users.set(u),
      error: (e) => this.error.set(e?.error?.detail ?? 'Не удалось загрузить пользователей'),
    });
  }

  create(): void {
    this.error.set(null); this.ok.set(null);
    if (!this.newUsername || !this.newPassword) return;
    this.http.post(`${this.base}/admin/users`, {
      username: this.newUsername.trim(), password: this.newPassword, role: this.newRole,
    }).subscribe({
      next: () => {
        this.ok.set(`Создан пользователь ${this.newUsername}`);
        this.newUsername = ''; this.newPassword = ''; this.newRole = 'user';
        this.refresh();
      },
      error: (e) => this.error.set(e?.status === 409 ? 'Пользователь уже существует' : (e?.error?.detail ?? 'Ошибка создания')),
    });
  }

  setActive(u: UserOut, active: boolean): void {
    this.http.post(`${this.base}/admin/users/${encodeURIComponent(u.username)}/active?active=${active}`, {})
      .subscribe({ next: () => this.refresh(), error: () => this.refresh() });
  }
}
```

- [ ] **Step 2: `admin.component.html`**
```html
<section class="admin">
  <h1>Управление доступом</h1>

  <div class="card">
    <h2>Выдать доступ</h2>
    <div class="row">
      <input [(ngModel)]="newUsername" placeholder="Логин" />
      <input [(ngModel)]="newPassword" type="password" placeholder="Пароль" />
      <select [(ngModel)]="newRole">
        <option value="user">user</option>
        <option value="admin">admin</option>
      </select>
      <button (click)="create()">Создать</button>
    </div>
    @if (ok()) { <p class="ok">{{ ok() }}</p> }
    @if (error()) { <p class="err">{{ error() }}</p> }
  </div>

  <h2>Пользователи</h2>
  <table>
    <tr><th>Логин</th><th>Роль</th><th>Активен</th><th></th></tr>
    @for (u of users(); track u.username) {
      <tr>
        <td>{{ u.username }}</td>
        <td>{{ u.role }}</td>
        <td>{{ u.is_active ? 'да' : 'нет' }}</td>
        <td>
          @if (u.is_active) { <button (click)="setActive(u, false)">отключить</button> }
          @else { <button (click)="setActive(u, true)">включить</button> }
        </td>
      </tr>
    }
  </table>
</section>
```

- [ ] **Step 3: `admin.component.css`**
```css
.admin { max-width: 760px; margin: 0 auto; padding: 32px 20px; }
h1 { font-size: 24px; } h2 { font-size: 16px; margin-top: 24px; }
.card { border: 1px solid #eee; border-radius: 12px; padding: 16px 18px; background: #faf8f3; }
.row { display: flex; gap: 10px; flex-wrap: wrap; }
.row input, .row select { padding: 10px 12px; border: 1px solid #ccc; border-radius: 8px; }
.row button { padding: 10px 18px; border: 0; border-radius: 8px; background: #e3b341; font-weight: 600; cursor: pointer; }
table { width: 100%; border-collapse: collapse; margin-top: 10px; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #eee; font-size: 14px; }
.ok { color: #2c7a7b; } .err { color: #c0392b; }
```

- [ ] **Step 4: header in `app.component.html`** — render a small top bar above the outlet (only when logged in), with current user + admin link + logout:
```html
@if (auth.isAuthenticated()) {
  <header class="topbar">
    <a routerLink="/" class="brand">DS Navigator</a>
    <span class="spacer"></span>
    @if (auth.isAdmin()) { <a routerLink="/admin">Доступы</a> }
    <span class="who">{{ auth.user()?.username }}</span>
    <button (click)="logout()">Выйти</button>
  </header>
}
<router-outlet />
```
Update the root component class (`app.ts`/`app.component.ts`) to import `RouterOutlet`, `RouterLink`, inject `AuthService` as public `auth`, inject `Router`, and add `logout()`:
```typescript
import { Component, inject } from '@angular/core';
import { RouterOutlet, RouterLink, Router } from '@angular/router';
import { AuthService } from './core/auth.service';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, RouterLink],
  templateUrl: './app.component.html',  // or keep inline; match what ng new generated
})
export class App {
  auth = inject(AuthService);
  private router = inject(Router);
  logout(): void { this.auth.logout(); this.router.navigate(['/login']); }
}
```
Add minimal `.topbar` CSS to the root component's stylesheet:
```css
.topbar { display: flex; align-items: center; gap: 14px; padding: 10px 20px; border-bottom: 1px solid #eee; }
.topbar .brand { font-weight: 700; text-decoration: none; color: #222; }
.topbar .spacer { flex: 1; }
.topbar a { color: #b8860b; text-decoration: none; }
.topbar .who { color: #777; font-size: 14px; }
.topbar button { border: 0; background: #f0e6c8; border-radius: 8px; padding: 6px 12px; cursor: pointer; }
```
> Match the exact root component filename/class `ng new` generated (`App` in `app.ts`, or `AppComponent` in `app.component.ts`). If it uses an inline template, move the header markup inline or switch to `templateUrl` consistently.

- [ ] **Step 5: build** — `cd frontend; npm run build` → succeeds, zero errors.

- [ ] **Step 6: commit** — stage admin component + root component files. Message: `feat: admin user-management page and app header with logout`

---

## Task A7: End-to-end verification + final review

**No new files.** Verify the whole auth flow on the Docker stack.

- [ ] **Step 1: env for the run** — create a temporary `.env` (gitignored) with the LLM dummies + an admin bootstrap:
```
LLM_MODEL=dummy/model
LLM_API_KEY_1=x
LLM_API_KEY_2=x
LLM_API_KEY_3=x
JWT_SECRET=local-dev-secret
ADMIN_USERNAME=admin
ADMIN_PASSWORD=admin12345
```

- [ ] **Step 2: bring up + rebuild** — `docker compose --profile vdb --profile emb --profile web up -d --build`. Wait for `ds-navigator-nginx` running.

- [ ] **Step 3: verify the API auth flow over nginx (self-signed → -k):**
  - Login: `curl -ksS https://localhost/auth/login -H "Content-Type: application/json" -d '{"username":"admin","password":"admin12345"}'` → returns `access_token` + `user.role=admin`.
  - Protected without token: `curl -ksS -o NUL -w "%{http_code}" https://localhost/source-files` → `401`.
  - Protected with token: `curl -ksS https://localhost/source-files -H "Authorization: Bearer <token>"` → `200` (`{"files":[...]}`).
  - Admin create user: `curl -ksS https://localhost/admin/users -H "Authorization: Bearer <token>" -H "Content-Type: application/json" -d '{"username":"guest","password":"guestpw","role":"user"}'` → `200`.
  - The new user can log in: `curl -ksS https://localhost/auth/login ... guest/guestpw` → token; and that token gets `403` on `POST /admin/users`.

- [ ] **Step 4: UI smoke (browser or note for the user)** — open `https://localhost`: unauthenticated → redirected to `/login`; log in as admin → search page loads with a header showing the username + "Доступы" link + "Выйти"; the admin page lists users and can create one; logout returns to `/login`.

- [ ] **Step 5: tear down + remove temp `.env`** — `docker compose --profile vdb --profile emb --profile web down`; delete `.env`; confirm `git status` clean.

- [ ] **Step 6: final review** — run the full `uv run pytest tests/ -v` (lazy-store + auth tests pass) and `cd frontend; npm run build` (clean). Then a holistic read of the auth surface: tokens signed with `JWT_SECRET`, no secrets logged, every data route carries `Depends(get_current_user)`, admin routes carry `Depends(require_admin)`.

- [ ] **Step 7: commit any verification-driven fixes** with clean messages.

---

## Phase 2 Self-Review checklist
- [ ] `tests/test_auth.py` passes (hashing, tokens, create/auth, login, protected 401, admin 403/200).
- [ ] `/forward`, `/source-files`, `/check-vdb`, `/ingest`, `/ingest-upload` all require a valid token (401 without).
- [ ] `/admin/*` require role `admin` (403 for plain users).
- [ ] Old `DEMO_ACCESS_TOKEN` middleware + `/auth-check` removed.
- [ ] Admin bootstrapped from `ADMIN_PASSWORD`; users persist in the `auth-data` volume.
- [ ] Angular: unauthenticated → `/login`; 401 → interceptor logs out + redirects; admin link only for admins; session restored on refresh via app initializer.
- [ ] nginx proxies `/auth/*` and `/admin/*`.
- [ ] All commits clean (no AI attribution).

## Deploy note
For production set in `.env`: a strong random `JWT_SECRET`, a real `ADMIN_PASSWORD` (change the default), plus the Phase-1 deploy vars (`DOMAIN`, `CERTBOT_EMAIL`, `STAGING=0`, `USE_LOCAL_CA=`). Users are issued by the admin via the `/admin` page.
