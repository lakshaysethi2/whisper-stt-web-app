# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- Add durable project-specific notes here as they are discovered through real work.


## Key Architecture

- **Job IDs**: Full uuid4 hex (32 chars, 128 bits entropy) — generated via `uuid.uuid4().hex`. Used in `/api/transcribe`, `/api/upload/finish`, and as URL path `/j/{job_id}`.
- **Job lifecycle**: Jobs are created with status "processing", then transition to "completed" or "failed". Status stored in-memory `_jobs` dict AND persisted to disk as `status.json` under `WORK_DIR/<job_id>/`.
- **Split retention**: `JOB_RETENTION_SECONDS` (default 7200 = 2h, ≥1h required) governs transcripts (`status.json`). `AUDIO_RETENTION_SECONDS` (default 1800 = 30min) governs recordings (`input*` files), which are also deleted immediately on completion/failure via `cleanup_job_audio` (keeps status.json). Running jobs are never cleaned up.
- **Disk persistence**: `_persist_job()` writes `status.json` on completion/failure. `cleanup_job_audio` (in the transcription task's `finally`) keeps status.json — do NOT revert to `cleanup_job` there (it would delete the transcript). `_load_persisted_jobs()` reads them on startup for resume after restart.
- **Startup cleanup**: `cleanup_all_jobs()` is retention-aware — it does NOT wipe WORK_DIR. It removes only job dirs older than `JOB_RETENTION_SECONDS` and input audio older than `AUDIO_RETENTION_SECONDS`; "chunks" is preserved. Job age comes from `status.json` `created_at` (see `job_created_at`).
- **Expiry API**: `GET /api/transcribe/status/{job_id}` returns additive `retention_seconds`, `expires_at` (epoch), `expires_in_seconds`. `GET /api/upload/config` returns `job_retention_seconds`/`audio_retention_seconds`; the frontend uses them for honest expiry messages (save-link note + resume-expired view).
- **tmpfs caveat**: whisper-test runs `WORK_DIR=/tmp/whisper-stt` on a 2GB tmpfs — transcripts still vanish on container restart there (documented in README/docs; moving WORK_DIR to real disk is a deploy decision).
- **Progress tracking**: `app/transcriber.py` exposes `_progress: dict[str, float]` and `get_progress(job_id)`. Progress = `seg.end / info.duration` (audio position processed). Entire segment iteration runs in `to_thread` to avoid blocking the event loop.
- **SPA route**: `GET /j/{job_id}` serves `static/index.html` — the frontend reads `job_id` from `window.location.pathname` and resumes polling/display.
- **Save link UI**: After job creation, browser URL updates to `/j/{job_id}` and a "Save this link" card with copyable absolute URL is shown.
- **Model choice**: Users must explicitly choose a model (base or large-v3-turbo) before transcribing. No silent default. Rejected with 400 if missing/invalid.

## Tests

- Run via `python -m pytest tests/ -v` (uses uv venv at `.venv/`)
- Key test files: `tests/test_chunked_upload.py`, `tests/test_retention.py` (split retention + startup cleanup + expiry API)
- `tests/conftest.py` redirects `WORK_DIR` to a temp dir so tests never touch a real deployment's `/tmp/whisper-stt`
- Tests mock `app.main.transcribe_audio` using `_make_async_mock` helper for fast/slow results
- SPA route tests: `test_spa_job_route_serves_html`, `test_spa_job_route_any_job_id`
- UUID entropy test: `test_job_id_has_full_uuid_entropy` (asserts 32-char hex)
- Model required tests: `test_transcribe_missing_model_rejected`, `test_transcribe_empty_model_rejected`, `test_transcribe_invalid_model_rejected`, `test_chunked_upload_missing_model_rejected`

## Cypress E2E tests

- Run Cypress headless: `npx cypress run` (requires running server)
- Run Cypress UI: `npx cypress open`
- Cypress config: `cypress.config.js` — `baseUrl` from `CYPRESS_BASE_URL` env (default `http://localhost:8000`)
- Test specs in `cypress/e2e/` (plain JS):
  1. `home-button.cy.js` — verifies `#header-new-btn` on homepage
  2. `resume-button.cy.js` — verifies header button + resume home button on `/j/{id}`
  3. `model-guard.cy.js` — verifies record button disabled until model selected
  4. `file-keep.cy.js` — verifies file name kept on model validation error
  5. `result-button.cy.js` — verifies header button visible when result is shown
- Artifacts (screenshots, videos, downloads) are gitignored

## UI structure

- Cards are in a `<main>` column laid out top-to-bottom.
- Record card comes first, then Model card (required), then Language, then Upload with Transcribe button.
- Model card has inline error via `<p class="model-error">` and error styling via `.input-error` on the `<select>`.
- Header has a "New transcription" button (`#header-new-btn`) always visible, navigating to `/`.
- Resume states all show a "Start new transcription" button (`#resume-home-btn`).
- Completed resume results show a "Start new transcription" button in the result toolbar (`#result-new-btn`).
- Model errors are shown inline (red border + error text) instead of only toast.
## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
