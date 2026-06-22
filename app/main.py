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
import threading
import time

from app.config import (
    WHISPER_MODEL, WHISPER_LANGUAGE, MAX_FILE_SIZE,
    ALLOWED_EXTENSIONS, SUPPORTED_MODELS, get_job_dir, cleanup_job,
    cleanup_all_jobs, WORK_DIR,
)
from app.transcriber import load_model, transcribe_audio, _device_info

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

VALID_LANG_RE = re.compile(r"^[a-z]{2}(-[a-zA-Z]{2,})?$")

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
_jobs_lock = threading.Lock()
JOB_MAX_AGE_SECONDS = 30 * 60  # 30 minutes


async def periodic_cleanup(interval_seconds: int = 600, max_age_seconds: int = 1800):
    logger.info("Starting periodic cleanup task (interval=%ds, max_age=%ds)", interval_seconds, max_age_seconds)
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            now = time.time()
            if not WORK_DIR.exists():
                continue

            for p in list(WORK_DIR.iterdir()):
                try:
                    if not p.is_dir():
                        continue

                    if p.name == "chunks":
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
                                if now - created > max_age_seconds:
                                    logger.info("Removing expired chunk session: %s (age: %.1fs)", cp.name, now - created)
                                    shutil.rmtree(cp, ignore_errors=True)
                            except Exception as cp_err:
                                logger.error("Error cleaning up chunk session %s: %s", cp.name, cp_err)
                        continue

                    mtime = p.stat().st_mtime
                    if now - mtime > max_age_seconds:
                        logger.info("Removing expired job directory: %s (age: %.1fs)", p.name, now - mtime)
                        shutil.rmtree(p, ignore_errors=True)
                except Exception as p_err:
                    logger.error("Error checking path %s during cleanup: %s", p.name, p_err)

            # Sweep stale job entries from the in-memory job store.
            try:
                now_ts = now
                stale = []
                with _jobs_lock:
                    for jid, jstate in list(_jobs.items()):
                        if now_ts - jstate.get("created_at", 0) > JOB_MAX_AGE_SECONDS:
                            stale.append(jid)
                    for jid in stale:
                        _jobs.pop(jid, None)
                if stale:
                    logger.info("Removed %d expired transcription jobs from store", len(stale))
            except Exception as jc_err:
                logger.error("Error cleaning up job store: %s", jc_err)

        except asyncio.CancelledError:
            logger.info("Periodic cleanup task cancelled")
            break
        except Exception as e:
            logger.error("Unexpected error in periodic cleanup loop: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Perform startup cleanup of any stale files
    try:
        cleanup_all_jobs()
        logger.info("Startup cleanup completed successfully.")
    except Exception as e:
        logger.error("Error during startup cleanup: %s", e)

    # Ensure the chunk directory exists after startup cleanup
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)

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


app = FastAPI(title="Whisper STT", version="2.2.0", lifespan=lifespan)

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
    }


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
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        logger.error("Chunk upload failed upload=%s index=%d error=%s", upload_id, chunk_index, str(e))
        raise HTTPException(500, f"Chunk upload failed: {str(e)}")


@app.post("/api/upload/finish/{upload_id}")
async def upload_finish(
    upload_id: str,
    language: str = Query(default=""),
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

        ext = Path(meta["filename"]).suffix.lower()
        job_id = str(uuid.uuid4())[:8]
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
        # client cannot see "not found" between spawn and registration.
        with _jobs_lock:
            _jobs[job_id] = {
                "status": "processing",
                "progress": 0.0,
                "created_at": time.time(),
            }

        logger.info(
            "Spawned transcription job=%s file=%s (%d bytes) lang=%s",
            job_id, meta["filename"], meta["size"], lang,
        )

        async def _run_transcription():
            try:
                result = await transcribe_audio(str(file_path), lang, job_id)
                logger.info(
                    "Transcription complete job=%s segments=%d duration=%.1fs process=%.2fs",
                    job_id, len(result.get("segments", [])),
                    result.get("duration", 0), result.get("process_time", 0),
                )
                with _jobs_lock:
                    _jobs[job_id] = {
                        "status": "completed",
                        "progress": 1.0,
                        "result": result,
                        "created_at": _jobs[job_id]["created_at"],
                    }
            except Exception as e:
                logger.error("Transcription failed job=%s error=%s", job_id, str(e))
                with _jobs_lock:
                    _jobs[job_id] = {
                        "status": "failed",
                        "error": str(e),
                        "created_at": _jobs[job_id]["created_at"],
                    }
            finally:
                cleanup_job(job_id)

        asyncio.create_task(_run_transcription())

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
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found (or expired)")

    elapsed = time.time() - job["created_at"]
    response = {
        "job_id": job_id,
        "status": job["status"],
        "elapsed_seconds": round(elapsed, 1),
    }
    if job["status"] == "processing":
        # crude progress hint: cap at 0.9 so the UI doesn't look "done" until completed
        response["progress"] = min(0.9, elapsed / max(1.0, job.get("expected_seconds", 60.0)))
    elif job["status"] == "completed":
        response["progress"] = 1.0
        response["result"] = job["result"]
    elif job["status"] == "failed":
        response["error"] = job.get("error", "Unknown error")
    return JSONResponse(response)


@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form(default=""),
):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported format: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")

    job_id = str(uuid.uuid4())[:8]
    job_dir = get_job_dir(job_id)
    file_path = job_dir / f"input{ext}"

    try:
        # Stream uploaded file to disk in chunks to avoid high RAM usage
        chunk_size = 1024 * 1024  # 1 MB chunks
        total_bytes = 0
        with file_path.open("wb") as buffer:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_FILE_SIZE:
                    raise HTTPException(413, f"File too large. Max: {MAX_FILE_SIZE // (1024*1024)} MB")
                buffer.write(chunk)

        lang = language or WHISPER_LANGUAGE or "en"
        if not VALID_LANG_RE.match(lang):
            raise HTTPException(400, f"Invalid language code: {lang}. Expected format: xx or xx-XX (e.g. en, en-US)")

        logger.info("Transcribing %s (%d bytes) lang=%s job=%s", file.filename, total_bytes, lang, job_id)

        result = await transcribe_audio(str(file_path), lang, job_id)
        logger.info("Transcription complete job=%s segments=%d duration=%.1fs process=%.2fs",
                     job_id, len(result.get("segments", [])),
                     result.get("duration", 0), result.get("process_time", 0))
        return JSONResponse(result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Transcription failed job=%s error=%s", job_id, str(e))
        raise HTTPException(500, f"Transcription failed: {str(e)}")
    finally:
        cleanup_job(job_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)