import asyncio
import logging
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid
from pathlib import Path

from app.config import (
    WHISPER_MODEL, WHISPER_LANGUAGE, MAX_FILE_SIZE,
    ALLOWED_EXTENSIONS, SUPPORTED_MODELS, get_job_dir, cleanup_job,
)
from app.transcriber import load_model, transcribe_audio, _device_info

logger = logging.getLogger(__name__)

VALID_LANG_RE = re.compile(r"^[a-z]{2}(-[a-zA-Z]{2,})?$")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(load_model)
    yield


app = FastAPI(title="Whisper STT", version="2.1.0", lifespan=lifespan)

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
