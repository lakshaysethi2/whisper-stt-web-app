import asyncio
import json
import logging
import math
import re
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import time

from app.config import (
    WHISPER_MODEL, WHISPER_LANGUAGE, MAX_FILE_SIZE,
    MIN_FREE_DISK_BYTES, check_disk_space,
    ALLOWED_EXTENSIONS, SUPPORTED_MODELS, UI_MODEL_CHOICES,
    validate_model,
    get_job_dir, cleanup_job, cleanup_job_audio,
    cleanup_all_jobs, WORK_DIR, JOB_RETENTION_SECONDS,
    AUDIO_RETENTION_SECONDS,
    job_created_at, cleanup_stale_input_audio,
)
from app.transcriber import load_model, transcribe_audio, get_progress, get_loaded_models, _device_info

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

VALID_LANG_RE = re.compile(r"^[a-z]{2}(-[a-zA-Z]{2,})?$")
_JOB_ID_RE = re.compile(r"^[a-f0-9]{32}$")

# Directory for in-progress chunked uploads.
CHUNK_DIR = WORK_DIR / "chunks"
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

# Per-request chunk ceiling. Cloudflare Free/Pro = 100 MB per request.
# Keep chunks well below that to leave room for multipart overhead and headers.
CHUNK_MAX_SIZE = 50 * 1024 * 1024  # 50 MB

# Recommended chunk size for clients.
CHUNK_SIZE = 5 * 1024 * 1024  # 5 MB

# Maximum number of chunks, derived from max file size and recommended chunk size.
MAX_CHUNKS = math.ceil(MAX_FILE_SIZE / CHUNK_SIZE)

# Files under this size keep using the original single-request path.
DIRECT_UPLOAD_THRESHOLD = 50 * 1024 * 1024  # 50 MB

# In-memory locks to prevent concurrent finish calls for the same upload.
_finish_locks: set[str] = set()

# In-memory store of running/completed transcription jobs.
# Format: job_id -> {status, progress, result?, error?, created_at}
_jobs: dict[str, dict] = {}
_jobs_lock: asyncio.Lock = asyncio.Lock()
# Track running transcription tasks so they aren't garbage-collected
# and can be awaited/cancelled on shutdown.
_tasks: set[asyncio.Task] = set()


def _status_file(job_dir: Path) -> Path:
    return job_dir / "status.json"


def _persist_job(job_id: str, job_data: dict) -> None:
    """Best-effort write job status+result to disk.

    Called at job CREATION (status "processing") and again on completion/failure,
    so a restart can restore the job instead of reporting it as expired.
    """
    try:
        job_dir = get_job_dir(job_id)
        sf = _status_file(job_dir)
        # Strip non-serializable fields
        persist = {k: v for k, v in job_data.items() if k != "task"}
        sf.write_text(json.dumps(persist, default=str))
    except Exception as e:
        logger.warning("Failed to persist job %s to disk: %s", job_id, e)


def _is_job_dir(p: Path) -> bool:
    """True if p looks like a transcription job directory (32-hex uuid name)."""
    return p.is_dir() and p.name != "chunks" and bool(_JOB_ID_RE.match(p.name))


INTERRUPTED_ERROR = (
    "Transcription was interrupted before it finished "
    "(server restart or crash). The recording is retained on the server; "
    "please re-upload to transcribe."
)


def _read_persisted_job(job_dir: Path) -> dict | None:
    """Read one job dir's persisted state.

    Returns a job dict, or None when the dir has no usable status.json
    (e.g. a crash between dir creation and persist, or legacy dirs).
    Persisted "processing"/"pending" jobs are restored as "interrupted" — the
    process that was transcribing them is gone after a restart.
    """
    sf = job_dir / "status.json"
    if not sf.exists():
        return None
    try:
        data = json.loads(sf.read_text())
    except Exception:
        return None
    status = data.get("status")
    if status in ("completed", "failed"):
        return data
    if status in ("processing", "pending"):
        data = dict(data)
        data["status"] = "interrupted"
        data["error"] = INTERRUPTED_ERROR
        return data
    return None


def _job_has_recording(job_id: str) -> bool:
    """True if the job dir still holds its uploaded recording (input* files)."""
    try:
        d = WORK_DIR / job_id
        if not d.is_dir():
            return False
        return any(p.is_file() and p.name.startswith("input") for p in d.iterdir())
    except OSError:
        return False


def _load_persisted_jobs() -> dict[str, dict]:
    """Load job statuses from disk into the in-memory store.

    - completed/failed jobs are restored as-is;
    - dirs with a "processing"/"pending" status.json (persisted at creation),
      or with no valid status at all, are restored as "interrupted" — the job
      was in flight when the process died. NEVER treated as unknown/expired.
    """
    loaded: dict[str, dict] = {}
    if not WORK_DIR.exists():
        return loaded
    for p in list(WORK_DIR.iterdir()):
        if not _is_job_dir(p):
            continue
        try:
            data = _read_persisted_job(p)
            if data is None:
                # Job dir exists but no valid status (crash before persist, or
                # legacy dir from before status.json was written at creation):
                # report as interrupted, never expired.
                data = {
                    "status": "interrupted",
                    "error": INTERRUPTED_ERROR,
                    "created_at": p.stat().st_mtime,
                }
            data["created_at"] = data.get("created_at", p.stat().st_mtime)
            loaded[p.name] = data
            logger.info("Loaded persisted job %s (status=%s)", p.name, data["status"])
        except Exception as e:
            logger.warning("Failed to load persisted job %s: %s", p.name, e)
    return loaded


async def periodic_cleanup(interval_seconds: int = 600):
    max_age = JOB_RETENTION_SECONDS
    logger.info("Starting periodic cleanup task (interval=%ds, max_age=%ds)", interval_seconds, max_age)
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            now = time.time()
            if not WORK_DIR.exists():
                continue

            # Track which job_ids are still running (in-memory)
            running_ids: set[str] = set()
            async with _jobs_lock:
                for jid, jstate in _jobs.items():
                    if jstate.get("status") in ("processing", "pending"):
                        running_ids.add(jid)

            # Clean chunk sessions (their own lifecycle) inline; job dirs are
            # handled by the retention-aware sweep below.
            for p in list(WORK_DIR.iterdir()):
                if not p.is_dir() or p.name != "chunks":
                    continue
                try:
                    for cp in list(p.iterdir()):
                        try:
                            if not cp.is_dir():
                                continue
                            meta_path = cp / "meta.json"
                            created = cp.stat().st_mtime
                            if meta_path.exists():
                                try:
                                    meta = json.loads(meta_path.read_text())
                                    created = meta.get("created", created)
                                except Exception:
                                    pass
                            if now - created > max_age:
                                logger.info("Removing expired chunk session: %s (age: %.1fs)", cp.name, now - created)
                                shutil.rmtree(cp, ignore_errors=True)
                        except Exception as cp_err:
                            logger.error("Error cleaning up chunk session %s: %s", cp.name, cp_err)
                except Exception as p_err:
                    logger.error("Error cleaning chunk directory %s: %s", p.name, p_err)

            # Retention-aware sweep: transcripts expire after JOB_RETENTION_SECONDS,
            # recordings are deleted after AUDIO_RETENTION_SECONDS (sooner).
            _cleanup_expired_jobs(now, running_ids)

            # Sweep stale job entries from the in-memory job store.
            try:
                now_ts = now
                stale = []
                async with _jobs_lock:
                    for jid, jstate in list(_jobs.items()):
                        if now_ts - jstate.get("created_at", 0) > max_age:
                            stale.append(jid)
                    for jid in stale:
                        _jobs.pop(jid, None)
                        # Also remove disk status file for expired jobs
                        job_dir = WORK_DIR / jid
                        sf = _status_file(job_dir)
                        if sf.exists():
                            try:
                                sf.unlink()
                            except Exception:
                                pass
                if stale:
                    logger.info("Removed %d expired transcription jobs from store", len(stale))
            except Exception as jc_err:
                logger.error("Error cleaning up job store: %s", jc_err)

        except asyncio.CancelledError:
            logger.info("Periodic cleanup task cancelled")
            break
        except Exception as e:
            logger.error("Unexpected error in periodic cleanup loop: %s", e)


def _cleanup_expired_jobs(now: float, running_ids: set[str]) -> list[str]:
    """Retention-aware sweep of job directories (shared by periodic_cleanup).

    - Removes job dirs older than JOB_RETENTION_SECONDS entirely.
    - Deletes input recordings older than AUDIO_RETENTION_SECONDS in kept dirs.
    - Never touches still-running jobs or the "chunks" session directory.
    Returns the ids of removed job dirs.
    """
    removed: list[str] = []
    if not WORK_DIR.exists():
        return removed
    for p in list(WORK_DIR.iterdir()):
        try:
            if not p.is_dir() or p.name == "chunks":
                continue
            # Skip still-running jobs
            if p.name in running_ids:
                continue
            created = job_created_at(p)
            if now - created > JOB_RETENTION_SECONDS:
                logger.info("Removing expired job directory: %s (age: %.1fs)", p.name, now - created)
                shutil.rmtree(p, ignore_errors=True)
                removed.append(p.name)
            elif cleanup_stale_input_audio(p, now):
                logger.info("Removed stale input audio for job %s (kept transcript)", p.name)
        except Exception as p_err:
            logger.error("Error checking path %s during cleanup: %s", p.name, p_err)
    return removed


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Perform startup cleanup of any stale files
    try:
        cleanup_all_jobs()
        logger.info("Startup cleanup completed successfully.")
    except Exception as e:
        logger.error("Error during startup cleanup: %s", e)

    # Check disk space at startup
    free_ok = check_disk_space()
    if not free_ok:
        logger.warning(
            "Low disk space at startup: free below MIN_FREE_DISK_BYTES (%d bytes). "
            "Uploads may be rejected.",
            MIN_FREE_DISK_BYTES,
        )
    else:
        usage = __import__("shutil").disk_usage(str(WORK_DIR))
        logger.info(
            "Disk space OK: %d MB free (threshold %d MB)",
            usage.free // (1024 * 1024),
            MIN_FREE_DISK_BYTES // (1024 * 1024),
        )

    # Ensure the chunk directory exists after startup cleanup
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)

    # Load persisted completed/failed jobs from disk (best-effort resume)
    try:
        persisted = _load_persisted_jobs()
        if persisted:
            async with _jobs_lock:
                for jid, jdata in persisted.items():
                    if jid not in _jobs:
                        _jobs[jid] = jdata
            logger.info("Loaded %d persisted jobs from disk", len(persisted))
    except Exception as e:
        logger.warning("Failed to load persisted jobs: %s", e)

    # Start the background periodic cleanup task
    cleanup_task = asyncio.create_task(periodic_cleanup())

    await asyncio.to_thread(load_model)
    yield

    # Cancel the periodic cleanup task on shutdown
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

    # Give any in-flight transcription tasks a chance to finish cleanly.
    if _tasks:
        logger.info("Awaiting %d in-flight transcription task(s) before shutdown", len(_tasks))
        try:
            await asyncio.wait_for(asyncio.gather(*_tasks, return_exceptions=True), timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning("Transcription tasks did not finish within 30s; cancelling")
            for t in _tasks:
                t.cancel()


app = FastAPI(title="Whisper STT", version="3.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/j/{job_id}")
async def job_page(job_id: str):
    """SPA route for bookmarkable job pages.
    Serves the same index.html; the frontend reads job_id from path."""
    return FileResponse("static/index.html")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": WHISPER_MODEL,
        "device": _device_info.get("device", "unknown"),
        "compute_type": _device_info.get("compute_type", "unknown"),
        "compute_capability": _device_info.get("compute_capability", 0),
    }


@app.get("/api/models")
async def list_models():
    return {
        "current": WHISPER_MODEL,
        "device": _device_info.get("device", "unknown"),
        "compute_type": _device_info.get("compute_type", "unknown"),
        "available": SUPPORTED_MODELS,
        "ui_choices": UI_MODEL_CHOICES,
        "loaded": get_loaded_models(),
    }


@app.get("/api/upload/config")
async def upload_config():
    return {
        "chunk_size": CHUNK_SIZE,
        "max_chunk_size": CHUNK_MAX_SIZE,
        "max_file_size": MAX_FILE_SIZE,
        "max_chunks": MAX_CHUNKS,
        "direct_upload_threshold": DIRECT_UPLOAD_THRESHOLD,
        "allowed_extensions": sorted(ALLOWED_EXTENSIONS),
        # Retention settings (additive): transcripts vs recordings have different TTLs.
        "job_retention_seconds": JOB_RETENTION_SECONDS,
        "audio_retention_seconds": AUDIO_RETENTION_SECONDS,
    }


def _cleanup_stale_chunks(max_age_seconds: int = 1800):
    now = time.time()
    if not CHUNK_DIR.exists():
        return
    for cp in list(CHUNK_DIR.iterdir()):
        try:
            if not cp.is_dir():
                continue
            meta_path = cp / "meta.json"
            created = cp.stat().st_mtime
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    created = meta.get("created", created)
                except Exception:
                    pass
            if now - created > max_age_seconds:
                logger.info("Cleaning stale chunk session on start: %s (age: %.1fs)", cp.name, now - created)
                shutil.rmtree(cp, ignore_errors=True)
        except Exception:
            pass


@app.post("/api/upload/start")
async def upload_start(
    filename: str = Form(...),
    size: int = Form(...),
    total_chunks: int = Form(...),
):
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400,
            f"Unsupported format: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    if size > MAX_FILE_SIZE:
        raise HTTPException(
            413,
            f"File too large. Max: {MAX_FILE_SIZE // (1024 * 1024)} MB",
        )

    if size < 1:
        raise HTTPException(400, "File size must be at least 1 byte")

    if total_chunks < 1:
        raise HTTPException(400, "total_chunks must be at least 1")

    expected_chunks = math.ceil(size / CHUNK_SIZE)
    if total_chunks != expected_chunks:
        raise HTTPException(
            400,
            f"Invalid total_chunks: expected {expected_chunks} for {size} bytes "
            f"with chunk_size={CHUNK_SIZE}",
        )

    if total_chunks > MAX_CHUNKS:
        raise HTTPException(
            400,
            f"Too many chunks: {total_chunks}. Max: {MAX_CHUNKS}",
        )

    _cleanup_stale_chunks()

    stat = shutil.disk_usage(str(CHUNK_DIR))
    if stat.free < size + 100 * 1024 * 1024:
        _cleanup_stale_chunks(max_age_seconds=0)
        stat = shutil.disk_usage(str(CHUNK_DIR))
        if stat.free < size + 100 * 1024 * 1024:
            raise HTTPException(
                507,
                f"Not enough disk space. Need {size // (1024*1024)} MB, "
                f"have {stat.free // (1024*1024)} MB free.",
            )

    upload_id = str(uuid.uuid4())[:8]
    upload_dir = CHUNK_DIR / upload_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "filename": filename,
        "size": size,
        "total_chunks": total_chunks,
        "created": time.time(),
    }
    (upload_dir / "meta.json").write_text(json.dumps(meta))

    logger.info(
        "Started chunked upload id=%s file=%s size=%d chunks=%d",
        upload_id, filename, size, total_chunks,
    )
    return {"upload_id": upload_id}


@app.post("/api/upload/chunk/{upload_id}")
async def upload_chunk(
    upload_id: str,
    chunk_index: int = Form(...),
    file: UploadFile = File(...),
):
    upload_dir = CHUNK_DIR / upload_id
    if not upload_dir.exists():
        raise HTTPException(404, "Upload session not found")

    meta_path = upload_dir / "meta.json"
    if not meta_path.exists():
        raise HTTPException(404, "Upload session metadata missing")

    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        raise HTTPException(500, "Failed to read upload metadata")

    if chunk_index < 0 or chunk_index >= meta["total_chunks"]:
        raise HTTPException(400, "Invalid chunk index")

    expected_start = chunk_index * CHUNK_SIZE
    expected_size = min(CHUNK_SIZE, meta["size"] - expected_start)

    chunk_path = upload_dir / f"{chunk_index}.part"
    tmp_path = upload_dir / f"{chunk_index}.{uuid.uuid4().hex}.tmp"
    total_bytes = 0

    try:
        with tmp_path.open("wb") as buffer:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > CHUNK_MAX_SIZE:
                    tmp_path.unlink(missing_ok=True)
                    raise HTTPException(
                        413,
                        f"Chunk too large. Max: {CHUNK_MAX_SIZE // (1024 * 1024)} MB",
                    )
                buffer.write(chunk)

        if total_bytes != expected_size:
            tmp_path.unlink(missing_ok=True)
            raise HTTPException(
                400,
                f"Invalid chunk size for index {chunk_index}: "
                f"expected {expected_size} bytes, got {total_bytes}",
            )

        tmp_path.replace(chunk_path)

        received_count = sum(
            1 for i in range(meta["total_chunks"])
            if (upload_dir / f"{i}.part").exists()
        )

        logger.info(
            "Received chunk upload=%s index=%d size=%d",
            upload_id, chunk_index, total_bytes,
        )
        return {
            "ok": True,
            "received": received_count,
            "total": meta["total_chunks"],
        }

    except HTTPException:
        tmp_path.unlink(missing_ok=True)
        raise
    except OSError as e:
        tmp_path.unlink(missing_ok=True)
        if e.errno == 28:
            _cleanup_stale_chunks(max_age_seconds=0)
            logger.error("No space left, cleaned stale chunks. upload=%s index=%d", upload_id, chunk_index)
            raise HTTPException(507, "Disk full. Stale uploads have been cleaned — please retry.")
        logger.error("Chunk upload failed upload=%s index=%d error=%s", upload_id, chunk_index, str(e))
        raise HTTPException(500, f"Chunk upload failed: {str(e)}")
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        logger.error("Chunk upload failed upload=%s index=%d error=%s", upload_id, chunk_index, str(e))
        raise HTTPException(500, f"Chunk upload failed: {str(e)}")


@app.post("/api/upload/finish/{upload_id}")
async def upload_finish(
    upload_id: str,
    language: str = Query(default=""),
    model: str = Query(default=""),
):
    if upload_id in _finish_locks:
        raise HTTPException(409, "Upload is already being finalized")

    _finish_locks.add(upload_id)
    upload_dir = CHUNK_DIR / upload_id
    job_id: str | None = None
    success = False

    try:
        if not upload_dir.exists():
            raise HTTPException(404, "Upload session not found")

        meta_path = upload_dir / "meta.json"
        if not meta_path.exists():
            raise HTTPException(404, "Upload session metadata missing")

        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            raise HTTPException(500, "Failed to read upload metadata")

        missing = [
            i for i in range(meta["total_chunks"])
            if not (upload_dir / f"{i}.part").exists()
        ]
        if missing:
            raise HTTPException(
                400,
                detail={
                    "message": f"Missing chunks: {missing}",
                    "upload_id": upload_id,
                    "total_chunks": meta["total_chunks"],
                    "received": meta["total_chunks"] - len(missing),
                    "missing": missing,
                },
            )

        lang = language or WHISPER_LANGUAGE or "en"
        if not VALID_LANG_RE.match(lang):
            raise HTTPException(
                400,
                f"Invalid language code: {lang}. Expected format: xx or xx-XX (e.g. en, en-US)",
            )

        # Validate model choice (required, no silent default)
        chosen_model = validate_model(model)

        ext = Path(meta["filename"]).suffix.lower()
        job_id = uuid.uuid4().hex
        job_dir = get_job_dir(job_id)
        file_path = job_dir / f"input{ext}"

        def assemble_chunks():
            assembled_size = 0
            with file_path.open("wb") as out:
                for i in range(meta["total_chunks"]):
                    chunk_path = upload_dir / f"{i}.part"
                    assembled_size += chunk_path.stat().st_size
                    if assembled_size > MAX_FILE_SIZE:
                        raise HTTPException(
                            413,
                            f"Assembled file too large: {assembled_size} bytes "
                            f"(max {MAX_FILE_SIZE})",
                        )
                    with chunk_path.open("rb") as inp:
                        shutil.copyfileobj(inp, out)
            return assembled_size

        assembled_size = await asyncio.to_thread(assemble_chunks)

        if assembled_size != meta["size"]:
            raise HTTPException(
                400,
                f"Upload size mismatch: expected {meta['size']} bytes, "
                f"got {assembled_size}",
            )

        # Chunks are valid and the file is assembled — release the upload session
        # so the user can't accidentally trigger another transcription of the
        # same chunks, and so disk is freed.
        shutil.rmtree(upload_dir, ignore_errors=True)
        _finish_locks.discard(upload_id)

        # Register the job BEFORE spawning the background task so a fast-polling
        # client cannot see "not found" between spawn and registration. Persist
        # at creation so a restart can restore this job (as "interrupted").
        async with _jobs_lock:
            _jobs[job_id] = {
                "status": "processing",
                "progress": 0.0,
                "created_at": time.time(),
            }
        _persist_job(job_id, _jobs[job_id])

        logger.info(
            "Spawned transcription job=%s file=%s (%d bytes) lang=%s",
            job_id, meta["filename"], meta["size"], lang,
        )

        async def _run_transcription():
            try:
                result = await transcribe_audio(str(file_path), lang, job_id, model_name=chosen_model)
                logger.info(
                    "Transcription complete job=%s segments=%d duration=%.1fs process=%.2fs",
                    job_id, len(result.get("segments", [])),
                    result.get("duration", 0), result.get("process_time", 0),
                )
                created_at: float = 0.0
                async with _jobs_lock:
                    created_at = _jobs[job_id]["created_at"]
                    _jobs[job_id] = {
                        "status": "completed",
                        "progress": 1.0,
                        "result": result,
                        "model": chosen_model,
                        "created_at": created_at,
                    }
                _persist_job(job_id, _jobs[job_id])
            except Exception as e:
                logger.error("Transcription failed job=%s error=%s", job_id, str(e))
                async with _jobs_lock:
                    _jobs[job_id] = {
                        "status": "failed",
                        "error": str(e),
                        "model": chosen_model,
                        "created_at": _jobs[job_id]["created_at"],
                    }
                _persist_job(job_id, _jobs[job_id])
            finally:
                # Keep status.json (the transcript) for resume; drop the recording
                # audio as soon as it is no longer needed.
                cleanup_job_audio(job_id)

        task = asyncio.create_task(_run_transcription())
        _tasks.add(task)
        task.add_done_callback(_tasks.discard)

        success = True
        return JSONResponse({
            "job_id": job_id,
            "status": "processing",
            "progress": 0.0,
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Finish failed upload=%s error=%s", upload_id, str(e))
        raise HTTPException(500, f"Finish failed: {str(e)}")
    finally:
        _finish_locks.discard(upload_id)
        if not success and job_id is None:
            pass


@app.get("/api/transcribe/status/{job_id}")
async def transcribe_status(job_id: str):
    """Poll for the status / result of a transcription job spawned by /finish."""
    async with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        # Not in memory: the job may have been interrupted by a restart/crash
        # while its dir (and possibly the recording) is still on disk. Report
        # "interrupted" — NEVER expired. Only a missing dir is genuinely expired.
        job_dir = WORK_DIR / job_id
        if _is_job_dir(job_dir):
            persisted = _read_persisted_job(job_dir)
            if persisted is None:
                persisted = {
                    "status": "interrupted",
                    "error": INTERRUPTED_ERROR,
                    "created_at": job_dir.stat().st_mtime,
                }
            persisted["created_at"] = persisted.get("created_at", job_dir.stat().st_mtime)
            job = persisted
        else:
            raise HTTPException(404, "Job not found (or expired)")

    now_ts = time.time()
    elapsed = now_ts - job["created_at"]
    expires_at = job["created_at"] + JOB_RETENTION_SECONDS
    response = {
        "job_id": job_id,
        "status": job["status"],
        "elapsed_seconds": round(elapsed, 1),
        # Honest expiry: transcripts are kept for JOB_RETENTION_SECONDS.
        "retention_seconds": JOB_RETENTION_SECONDS,
        "expires_at": round(expires_at, 1),
        "expires_in_seconds": max(0.0, round(expires_at - now_ts, 1)),
    }
    if job["status"] == "processing":
        # Real progress from transcriber (audio position / total duration)
        real_progress = get_progress(job_id)
        if real_progress is not None:
            response["progress"] = min(0.99, real_progress)  # cap below 1.0 until fully complete
        else:
            # No real progress data yet (model still loading / VAD analyzing)
            response["progress"] = 0.0
            response["progress_note"] = "working"  # honest signal: we are working, % unknown
    elif job["status"] == "completed":
        response["progress"] = 1.0
        response["result"] = job["result"]
    elif job["status"] == "failed":
        response["error"] = job.get("error", "Unknown error")
    elif job["status"] == "interrupted":
        response["progress"] = 0.0
        retained = _job_has_recording(job_id)
        response["recording_retained"] = retained
        if retained:
            response["error"] = job.get("error", INTERRUPTED_ERROR)
        else:
            response["error"] = (
                "Transcription was interrupted before it finished "
                "(server restart or crash), and the recording is no longer on the "
                "server. Please re-upload to transcribe."
            )
    return JSONResponse(response)


@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form(default=""),
    model: str = Form(default=""),
):
    """
    Upload a file for transcription (synchronous/small files).
    Saves file, registers a job, spawns background transcription, and returns
    {job_id, status, progress} immediately. Poll GET /api/transcribe/status/{job_id}
    for the result.
    """
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported format: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")

    job_id = uuid.uuid4().hex
    job_dir = get_job_dir(job_id)
    file_path = job_dir / f"input{ext}"

    # Reject early if disk space is critically low
    if not check_disk_space():
        # Try cleaning stale jobs first, then recheck
        cleanup_all_jobs()
        if not check_disk_space():
            raise HTTPException(
                507,
                f"Server disk space too low. Minimum required: "
                f"{MIN_FREE_DISK_BYTES // (1024*1024)} MB",
            )

    # Stream uploaded file to disk in chunks to avoid high RAM usage
    chunk_size = 1024 * 1024  # 1 MB chunks
    total_bytes = 0
    try:
        with file_path.open("wb") as buffer:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_FILE_SIZE:
                    raise HTTPException(413, f"File too large. Max: {MAX_FILE_SIZE // (1024*1024)} MB")
                buffer.write(chunk)
    except Exception:
        cleanup_job(job_id)
        raise

    lang = language or WHISPER_LANGUAGE or "en"
    if not VALID_LANG_RE.match(lang):
        cleanup_job(job_id)
        raise HTTPException(400, f"Invalid language code: {lang}. Expected format: xx or xx-XX (e.g. en, en-US)")

    # Validate model choice (required, no silent default)
    chosen_model = validate_model(model)

    # Register the job BEFORE spawning the background task. Persist at creation
    # so a restart can restore this job (as "interrupted") instead of expiring it.
    async with _jobs_lock:
        _jobs[job_id] = {
            "status": "processing",
            "progress": 0.0,
            "model": chosen_model,
            "created_at": time.time(),
        }
    _persist_job(job_id, _jobs[job_id])

    logger.info(
        "Transcribing %s (%d bytes) lang=%s job=%s",
        file.filename, total_bytes, lang, job_id,
    )

    async def _run_transcribe():
        try:
            result = await transcribe_audio(str(file_path), lang, job_id, model_name=chosen_model)
            logger.info(
                "Transcription complete job=%s segments=%d duration=%.1fs process=%.2fs",
                job_id, len(result.get("segments", [])),
                result.get("duration", 0), result.get("process_time", 0),
            )
            async with _jobs_lock:
                _jobs[job_id] = {
                    "status": "completed",
                    "progress": 1.0,
                    "result": result,
                    "model": chosen_model,
                    "created_at": _jobs[job_id]["created_at"],
                }
            _persist_job(job_id, _jobs[job_id])
        except Exception as e:
            logger.error("Transcription failed job=%s error=%s", job_id, str(e))
            async with _jobs_lock:
                _jobs[job_id] = {
                    "status": "failed",
                    "error": str(e),
                    "model": chosen_model,
                    "created_at": _jobs[job_id]["created_at"],
                }
            _persist_job(job_id, _jobs[job_id])
        finally:
            # Keep status.json (the transcript) for resume; drop the recording
            # audio as soon as it is no longer needed.
            cleanup_job_audio(job_id)

    task = asyncio.create_task(_run_transcribe())
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)

    return JSONResponse({
        "job_id": job_id,
        "status": "processing",
        "progress": 0.0,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)