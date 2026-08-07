# User Requirements — whisper-stt-web-app

This file tracks user requirements and design decisions for the project.

## Core purpose

A self-hosted speech-to-text web application using faster-whisper (CTranslate2).
Upload audio files or record from microphone; transcription runs on user's own server.

## Key requirements

### Deployment

1. **Docker-only deployment** — all operations (build, run, test) use Docker.
2. **CPU-first** — default model is `base`, GPU support is optional override.
3. **ARM-compatible** — Oracle A1 (ARM64) is a primary deployment target.
4. **Single command** to deploy: `docker compose up -d`.
5. **Port 8561** is the default host port (→ container 8000).

### Storage & disk

1. **tmpfs work dir** — `/tmp/whisper-stt` is tmpfs, size=2G, no host disk used.
2. **MAX_FILE_SIZE** default: 512 MB (safe on 17G free disk).
3. **MIN_FREE_DISK_BYTES** default: 2 GB — uploads rejected below threshold.
4. **Auto cleanup** — stale jobs/chunks older than 30 min removed every 10 min.
5. **Startup cleanup** — all stale data wiped on container start.

### Model

1. Default model: `base` (CPU-friendly, ~150 MB disk cache).
2. Model overridable via `WHISPER_MODEL` env var.
3. GPU auto-detection with fallback to CPU int8.
4. Supported: tiny, base, small, medium, large-v3, large-v3-turbo.
5. **Mandatory model choice** — User must explicitly choose a model in the UI before transcribing.
6. **No silent default** — No pre-selected default; server rejects with 400 if model is missing/invalid.
7. **Model hot-switching** — Changing model sends a request-time param; server loads/caches on demand. WHISPER_MODEL env determines initial preload only.

### Testing

1. Test audio fixtures in `test_audio/` — LibriSpeech speaker 3081, chapter 166546.
2. Fixtures reproducible via `scripts/build_test_audio.sh`.
3. Accuracy benchmark via `scripts/run_accuracy_bench.sh`.
4. Accuracy gate: >= 95% word accuracy on clean read speech (1min, 5min).

### API

1. `GET /health` — health check with model/device info.
2. `POST /api/transcribe` — direct file upload + transcription.
3. `POST /api/upload/start|chunk|finish` — chunked upload for large files.
4. `GET /api/transcribe/status/{job_id}` — poll async transcription result.
5. `GET /api/models` — list available models and current device.
6. `GET /api/upload/config` — client configuration for chunked upload.
7. `GET /j/{job_id}` — SPA route for bookmarkable job pages.

### GPU (optional)

1. GPU support via `-f docker-compose.gpu.yml` override.
2. Default GPU model: `large-v3-turbo` (better quality on GPU).
3. CUDA compute capability 5.0+ supported.
4. Auto-selects float16/float32 based on GPU architecture.

## Bookmarkable Job Resume Links

### Problem
Transcription of large files takes a long time. Users may close the browser while waiting. They need a unique link they can save and reopen later to see the same job progress/result.

### Requirements
1. After a transcription job is created (upload finishes, server returns `job_id`):
   - Browser URL updates to `/j/{job_id}`
   - A "Save this link" control is displayed (copyable URL button)
   - Works on mobile and desktop

2. Reopening `/j/{job_id}` (new tab, after close, on phone):
   - App detects job_id from URL
   - Fetches status from `/api/transcribe/status/{job_id}`
   - Shows appropriate UI: progress (processing), result (completed), error (failed), or expired message (not found)
   - Resumes polling for in-progress jobs

3. Job IDs use full uuid4 hex (128 bits entropy) — not trivially guessable

4. Completed/failed jobs retained for configurable period (`JOB_RETENTION_SECONDS`, default 2 hours)
   - Running jobs are never cleaned up

5. Job status+result persisted to disk as `status.json` under the job directory
   - Survives container restart for completed/failed jobs (best-effort)
   - Running jobs after restart show as expired

6. Both direct upload and chunked upload paths produce bookmarkable job links

## Real Progress Tracking (Bug Fix)

### Problem
- Progress was fake: `min(0.9, elapsed/60.0)` always shows 90% at 60 seconds regardless of actual transcription progress
- Segment generator from faster-whisper was iterated on the asyncio event loop, blocking health/status endpoints for multi-hour files

### Fix
1. Moved entire segment iteration into `to_thread` so event loop is never blocked
2. Real progress: `seg.end / info.duration` (audio position processed / total audio duration)
3. When no real progress data yet (model loading / VAD analyzing), show honest "working, progress unknown"
4. UI shows "working, % unknown" when `progress_note` is "working", else shows real percentage

## Mandatory Model Choice

### Problem
Previously the UI silently used the server's WHISPER_MODEL default. Users had no idea what model was running or how to switch.

### Requirements
1. **Mandatory explicit choice** — User must select a model (base or large-v3-turbo) before transcription can start.
2. **No pre-selected default** — Empty/null until user picks. The select element starts with a placeholder "Choose a model".
3. **Plain-language tradeoffs** — Each option shows:
   - **base**: faster, lighter on CPU/RAM, OK for drafts/short audio; lower transcript quality.
   - **large-v3-turbo**: much better quality; slower and heavier on CPU/RAM; long files take longer.
4. **Device hint** — Shows runtime device (cpu vs cuda) next to model options.
5. **Server-side validation** — Transcribe/finish endpoints reject missing/invalid model with 400.
6. **WHISPER_MODEL env** — May list available models or cold-start preload policy, but must not force a hidden choice that bypasses the UI.
7. **Model hot-switching** — Preferred: support request-time model param with lazy load/cache. If single-model process, document restart requirement.

## CrisperWhisper 2.0 (Verbatim ASR)

### Problem
The live site should also support verbatim speech-to-text — transcribing exactly what was said (fillers, repeats, stutters) with word-level timestamps — not just the clean Whisper-style transcripts faster-whisper produces.

### Requirements
1. **New model family** alongside faster-whisper (do not remove faster-whisper): `crisperwhisper-large/turbo/medium/small` in `UI_MODEL_CHOICES` and `/api/models`, validated exactly like existing models.
2. **Exact upstream model names** — map to `nyralabs/CrisperWhisper2.0_{large,turbo,medium,small}` (the IDs the upstream `crisperwhisper` package supports).
3. **Verbatim default, intended optional** — per-job `mode=verbatim|intended` query/form param; defaults to `verbatim` (or `CRISPER_MODE` env). Faster-whisper models ignore the mode and keep their existing behaviour.
4. **Same result shape** — text/segments/t0/t1 stay identical for the UI; CrisperWhisper results additionally include `mode`, `backend`, a top-level `words` list and per-segment `words` (`{word, t0, t1}` in ms).
5. **Word-level timestamps always on** — `word_timestamps=True` (no measurable overhead per upstream docs).
6. **Backend** — install `crisperwhisper[transformers]` (pure PyTorch): the `[ct2]` extra conflicts with faster-whisper's upstream `ctranslate2` and has no ARM64 wheels (the reference host is Oracle A1-class ARM64). `CRISPER_BACKEND=auto` resolves to ct2 only when the `ctranslate2-crisperwhisper` fork is actually installed.
7. **Single-thread serving** — CrisperWhisper models are loaded AND run on one dedicated `ThreadPoolExecutor(max_workers=1)` thread (upstream requirement for deterministic ct2 recovery; serializes inference).
8. **Progress** — CrisperWhisper exposes no incremental progress; jobs show honest "working, progress unknown" instead of a stuck 0%.
9. **UI** — a "Transcription style" selector (verbatim/intended) appears in the Model card only when a `crisperwhisper-*` model is chosen; record/upload flows send `mode`.

### Decisions
- **transformers over ct2**: faster-whisper is a hard dependency and its upstream `ctranslate2` overwrites the `ctranslate2-crisperwhisper` fork's site-packages files; additionally the fork has no ARM64 wheels. The transformers backend is slower (~4-5x on GPU) but feature-complete (modes, word timings, longform, hallucination repair).
- **Dockerfile.cpu** installs CPU-only `torch` from the PyTorch CPU index first so the CUDA `nvidia-*` wheel bundle isn't pulled into a CPU-only image.

## Browser STT Links

### Requirements
1. Add a UI section "Transcribe in your browser (other sites)" with links to:
   - https://huggingface.co/spaces/Xenova/whisper-web
   - https://huggingface.co/spaces/Xenova/whisper-webgpu
2. Short note: runs on their device; not this server; large files may struggle on phones.
3. Do not embed Transformers.js / in-app client Whisper.

## Always-visible Start New Transcription

### Problem
After viewing a resume link (`/j/{id}`), users could not find a way to start a new transcription.

The model selector was buried below the upload card, making it easy to miss.

### Requirements
1. **Header button** — A "New transcription" button always visible in the sticky header on all pages.
2. **Resume view** — "Start new transcription" button always visible in the resume view for all states: loading, progress (polling), error, expired.
3. **Completed result** — When viewing a completed result from a resume link, show a prominent "Start new transcription" button in the result toolbar.
4. **Clean slate** — Clicking "Start new transcription" navigates to `/` via `window.location.href = "/"` to give a fresh main UI.
5. **No cancel** — During in-progress resume polling, the button lets the user start a new transcription without cancelling the server job.

## Model Choice Unmissable

### Problem
The model selector was placed after the upload card, so users could proceed to upload/transcribe without noticing it.

### Requirements
1. **Reorder** — Model card must appear BEFORE the upload card in the DOM so the user sees model choice first.
2. **Inline error** — If user hits Transcribe without a model, show an inline error message directly on the model control (red border + error text).
3. **No toast-only** — Previously used only a toast for model validation error; now also uses inline error on the model select.
4. **Error clears** — Inline error disappears when the user changes the model selection.
5. **Required indicator** — Show a `*` next to the Model card label to indicate it is required.
