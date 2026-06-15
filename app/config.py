import os
import shutil
from pathlib import Path

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3-turbo")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en")
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "262144000"))

WORK_DIR = Path(os.getenv("WORK_DIR", "/tmp/whisper-stt"))
WORK_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {
    ".mp3", ".wav", ".flac", ".ogg", ".m4a",
    ".aac", ".wma", ".opus", ".webm", ".mp4",
}


def get_job_dir(job_id: str) -> Path:
    d = WORK_DIR / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def cleanup_job(job_id: str) -> None:
    d = WORK_DIR / job_id
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
