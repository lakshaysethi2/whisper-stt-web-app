# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- Add durable project-specific notes here as they are discovered through real work.


## Key Architecture

- **Job IDs**: Full uuid4 hex (32 chars, 128 bits entropy) — generated via `uuid.uuid4().hex`. Used in `/api/transcribe`, `/api/upload/finish`, and as URL path `/j/{job_id}`.
- **Job lifecycle**: Jobs are created with status "processing", then transition to "completed" or "failed". Status stored in-memory `_jobs` dict AND persisted to disk as `status.json` under `WORK_DIR/<job_id>/`.
- **Retention**: Controlled by `JOB_RETENTION_SECONDS` env var (default 7200 = 2h). Running jobs are never cleaned up.
- **Disk persistence**: `_persist_job()` writes `status.json` on completion/failure. `_load_persisted_jobs()` reads them on startup for resume after restart (best-effort).
- **Progress tracking**: `app/transcriber.py` exposes `_progress: dict[str, float]` and `get_progress(job_id)`. Progress = `seg.end / info.duration` (audio position processed). Entire segment iteration runs in `to_thread` to avoid blocking the event loop.
- **SPA route**: `GET /j/{job_id}` serves `static/index.html` — the frontend reads `job_id` from `window.location.pathname` and resumes polling/display.
- **Save link UI**: After job creation, browser URL updates to `/j/{job_id}` and a "Save this link" card with copyable absolute URL is shown.
- **Model choice**: Users must explicitly choose a model (base, large-v3-turbo, or crisperwhisper-large/turbo/medium/small) before transcribing. No silent default. Rejected with 400 if missing/invalid.

## CrisperWhisper 2.0 (verbatim ASR)

- **App-facing names** map to upstream HF ids in `CRISPER_MODEL_IDS` (app/config.py): `crisperwhisper-large` → `nyralabs/CrisperWhisper2.0_large` etc. `is_crisper_model()` routes transcriber paths.
- **Mode param**: `/api/transcribe` takes `mode` as Form, `/api/upload/finish` as Query (`verbatim|intended`); `validate_mode()` defaults to `CRISPER_MODE` env (default `verbatim`). Faster-whisper models ignore the mode — behavior unchanged.
- **Backend**: requirements pins `crisperwhisper[transformers]==2.0.1`. Do NOT install `[ct2]` here: it conflicts with faster-whisper's upstream ctranslate2 (overwrites the fork) and has no ARM64 wheels (audio.lak.nz is ARM64). `_resolve_crisper_backend_choice()` in app/transcriber.py never passes `backend="auto"` straight through (the library's auto picks ct2 whenever any ctranslate2 is importable) — it checks for the `ctranslate2-crisperwhisper` distribution.
- **Single dedicated thread**: CrisperWhisper models load AND run on `_crisper_executor` (`ThreadPoolExecutor(max_workers=1)`) — upstream requires model creation and inference on one thread (ct2 recovery primitives are thread-affine). Never call `CrisperWhisperModel.transcribe` off that executor.
- **Result shape**: same as faster-whisper (`text`, `segments[].text/t0/t1` ms) plus `mode`, `backend`, top-level `words` and per-segment `words` (`{word, t0, t1}` ms). Always calls `word_timestamps=True`. Progress stays `None` → UI shows "working, progress unknown".
- **Dockerfile.cpu** installs CPU-only `torch` from `https://download.pytorch.org/whl/cpu` before `-r requirements.txt` so the nvidia-* CUDA wheels aren't pulled into CPU images. The GPU Dockerfile omits that step.

## Tests

- Run via `python -m pytest tests/ -v` (uses uv venv at `.venv/`)
- Key test files: `tests/test_chunked_upload.py`
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
