import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid
from pathlib import Path

from app.config import (
    WHISPER_MODEL, WHISPER_LANGUAGE, MAX_FILE_SIZE,
    ALLOWED_EXTENSIONS, get_job_dir, cleanup_job,
)
from app.transcriber import load_model, transcribe_audio


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(load_model)
    yield


app = FastAPI(title="Whisper STT", version="2.0.0", lifespan=lifespan)

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
    return {"status": "ok", "model": WHISPER_MODEL}


@app.get("/api/models")
async def list_models():
    models = [
        {"name": "tiny", "params": "39M", "speed": "~32x"},
        {"name": "base", "params": "74M", "speed": "~16x"},
        {"name": "base.en", "params": "74M", "speed": "~16x"},
        {"name": "small", "params": "244M", "speed": "~6x"},
        {"name": "small.en", "params": "244M", "speed": "~6x"},
        {"name": "medium", "params": "769M", "speed": "~2x"},
        {"name": "large-v3", "params": "1550M", "speed": "1x"},
    ]
    return {"current": WHISPER_MODEL, "available": models}


@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form(default=""),
):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported format: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(413, f"File too large. Max: {MAX_FILE_SIZE // (1024*1024)} MB")

    job_id = str(uuid.uuid4())[:8]
    job_dir = get_job_dir(job_id)
    file_path = job_dir / f"input{ext}"
    file_path.write_bytes(content)

    lang = language or WHISPER_LANGUAGE or "en"

    try:
        result = await transcribe_audio(str(file_path), lang, job_id)
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(500, f"Transcription failed: {str(e)}")
    finally:
        cleanup_job(job_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
