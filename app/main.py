import asyncio
import json
import logging
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid
from pathlib import Path
import shutil
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

# Files under this size keep using the original single-request path.
DIRECT_UPLOAD_THRESHOLD = 50 * 1024 * 1024  # 50 MB


async def periodic_cleanup(interval_seconds: int = 600, max_age_seconds: int = 1800):
    logger.info("Starting periodic cleanup task (interval=%ds, max_age=%ds)", interval_seconds, max_age_seconds)
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            now = time.time()
            if not WORK_DIR.exists():
                continue

            for p in WORK_DIR.iterdir():
                if not p.is_dir():
                    continue

                # Chunked upload sessions are kept under WORK_DIR/chunks.
                # Clean individual upload sessions, not the whole container.
                if p.name == "chunks":
                    for cp in p.iterdir():
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
                    continue

                # Normal transcription job directories
                mtime = p.stat().st_mtime
                if now - mtime > max_age_seconds:
                    logger.info("Removing expired job directory: %s (age: %.1fs)", p.name, now - mtime)
                    shutil.rmtree(p, ignore_errors=True)

        except asyncio.CancelledError:
            logger.info("Periodic cleanup task cancelled")
            break
        except Exception as e:
            logger.error("Error in periodic cleanup task: %s", e)


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

    if total_chunks < 1:
        raise HTTPException(400, "total_chunks must be at least 1")

    upload_id = str(uuid.uuid4())[:8]
    upload_dir = CHUNK_DIR / upload_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "filename": filename,
        "size": size,
        "total_chunks": total_chunks,
        "received": [],
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

    chunk_path = upload_dir / f"{chunk_index}.part"
    total_bytes = 0

    try:
        with chunk_path.open("wb") as buffer:
            while True:
                chunk = await file.read(1024 * 1024)  # 1 MB at a time
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > CHUNK_MAX_SIZE:
                    chunk_path.unlink(missing_ok=True)
                    raise HTTPException(
                        413,
                        f"Chunk too large. Max: {CHUNK_MAX_SIZE // (1024 * 1024)} MB",
                    )
                buffer.write(chunk)

        if chunk_index not in meta["received"]:
            meta["received"].append(chunk_index)
            meta_path.write_text(json.dumps(meta))

        logger.info(
            "Received chunk upload=%s index=%d size=%d",
            upload_id, chunk_index, total_bytes,
        )
        return {
            "ok": True,
            "received": len(meta["received"]),
            "total": meta["total_chunks"],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Chunk upload failed upload=%s index=%d error=%s", upload_id, chunk_index, str(e))
        raise HTTPException(500, f"Chunk upload failed: {str(e)}")


@app.post("/api/upload/finish/{upload_id}")
async def upload_finish(
    upload_id: str,
    language: str = Form(default=""),
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

    missing = [i for i in range(meta["total_chunks"]) if i not in meta["received"]]
    if missing:
        raise HTTPException(400, f"Missing chunks: {missing}")

    ext = Path(meta["filename"]).suffix.lower()
    job_id = str(uuid.uuid4())[:8]
    job_dir = get_job_dir(job_id)
    file_path = job_dir / f"input{ext}"

    try:
        # Reassemble chunks in order
        with file_path.open("wb") as out:
            for i in range(meta["total_chunks"]):
                chunk_path = upload_dir / f"{i}.part"
                with chunk_path.open("rb") as inp:
                    shutil.copyfileobj(inp, out)

        lang = language or WHISPER_LANGUAGE or "en"
        if not VALID_LANG_RE.match(lang):
            raise HTTPException(
                400,
                f"Invalid language code: {lang}. Expected format: xx or xx-XX (e.g. en, en-US)",
            )

        logger.info(
            "Transcribing chunked upload file=%s (%d bytes) lang=%s job=%s",
            meta["filename"], meta["size"], lang, job_id,
        )

        result = await transcribe_audio(str(file_path), lang, job_id)
        logger.info(
            "Transcription complete job=%s segments=%d duration=%.1fs process=%.2fs",
            job_id, len(result.get("segments", [])),
            result.get("duration", 0), result.get("process_time", 0),
        )
        return JSONResponse(result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Transcription failed job=%s error=%s", job_id, str(e))
        raise HTTPException(500, f"Transcription failed: {str(e)}")

    finally:
        cleanup_job(job_id)
        shutil.rmtree(upload_dir, ignore_errors=True)


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