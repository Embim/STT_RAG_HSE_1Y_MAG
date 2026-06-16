# Self-hosted publication — Phase 3 (Ingest UI + async jobs) Plan

> Execute subagent-driven. Commit hygiene: plain messages, NO AI attribution.

**Goal:** A logged-in user can add lectures (YouTube URL or uploaded video files) and watch transcription progress, without HTTP timeouts. Ingest becomes asynchronous (fire-and-forget job + status polling) and gets a dedicated Angular view.

**Why async:** `process_youtube`/`process_uploaded_files` block for the full download→transcribe→embed (minutes), which 504s behind nginx. Fix: endpoint returns `202 {job_id}` immediately; a background task does the work and updates an in-process job store; the client polls `GET /ingest-status/{job_id}`.

**Scope / YAGNI:** in-process job store (dict) — fine for the single-replica demo (lost on restart, not multi-worker — documented). No queue/Redis. ASR (`asr-whisper`) must be up for real transcription (GPU). All ingest routes stay behind `get_current_user` (Phase 2).

---

## API contract
- `POST /ingest` `{url, export_txt?, export_json?, keep_video?, keep_audio?}` → `202 {job_id, status:"queued"}`
- `POST /ingest-upload` (multipart `files` + query `export_txt/export_json/keep_audio`) → `202 {job_id, status:"queued"}` (the endpoint SAVES the uploaded bytes to a temp dir synchronously — UploadFile is dead after the response — then backgrounds the transcription)
- `GET /ingest-status/{job_id}` → `{job_id, status: queued|running|done|error, progress: 0..100, total_items, done_items, current_item, items:[...], errors:[...], error_count, detail?}`
- All require a valid JWT (already, via `Depends(get_current_user)`).

## File plan
- **Create** `src/system/ingest_jobs.py` — in-process job store: `create_job()`, `update_job(id, **fields)`, `get_job(id)`, capped size with prune.
- **Modify** `src/downloader/processor.py` — add optional `progress_cb` to `process_youtube` and `transcribe_and_ingest`; extract `process_saved_uploads(saved, ...)` (path-based) out of `process_uploaded_files` so the background task can run from saved temp files; smooth %-progress via per-item chunk fractions.
- **Modify** `src/api/main.py` — `/ingest` + `/ingest-upload` return `202` and spawn background tasks; add `GET /ingest-status/{job_id}`.
- **Modify** `src/api/schemas.py` — SSRF guard on `IngestRequest.url` (http/https + host allowlist: youtube.com/youtu.be/vimeo.com).
- **Create** `tests/test_ingest_jobs.py` — job lifecycle + status endpoint with `process_*` mocked (no real ASR); SSRF validator tests.
- **Create** `frontend/src/app/features/ingest/ingest.component.{ts,html,css}` — YouTube + upload (HttpClient `reportProgress`), submit → job_id → poll status → progress bar + results/errors. Cartography-themed.
- **Modify** `frontend/src/app/core/api.service.ts` — `ingestYoutube()`, `ingestUpload()` (upload events), `ingestStatus(id)`.
- **Modify** `app.routes.ts` (route `/ingest`, `authGuard`) + the header (nav link "Добавить" for any logged-in user).

## Tasks
- **C1 (backend):** ingest_jobs store + async endpoints + status + progress_cb plumbing + SSRF guard + tests (mocked ASR). Acceptance: `POST /ingest` returns 202+job_id; `GET /ingest-status` transitions queued→running→done with progress; protected (401 w/o token); SSRF rejects non-allowlisted/internal URLs; `uv run pytest` green.
- **C2 (frontend):** Ingest view + ApiService methods + routing + header link, cartography-styled, upload progress + status polling. Acceptance: `ng build` green; view renders; polls and shows progress/results.
- **C3 (verify):** pytest + ng build green; UI smoke (mock backend) of the ingest view (submit → progress → done); confirm route is auth-guarded. (Real GPU transcription e2e is optional/manual — needs `asr-whisper` + media.)

## Deploy note
Real ingest needs `asr-whisper` running: `docker compose --profile vdb --profile emb --profile asr-whisper --profile web up -d` (whisper = GPU). The in-process job store means: single api replica only; jobs reset on restart.
