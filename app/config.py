import os
import shutil
from pathlib import Path

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en")
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "262144000"))

WORK_DIR = Path(os.getenv("WORK_DIR", "/tmp/whisper-stt"))
WORK_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {
    ".mp3", ".wav", ".flac", ".ogg", ".m4a",
    ".aac", ".wma", ".opus", ".webm", ".mp4",
}

SUPPORTED_MODELS = [
    {"name": "tiny", "params": "39M", "vram_fp32": "200", "vram_fp16": "128"},
    {"name": "base", "params": "74M", "vram_fp32": "400", "vram_fp16": "256"},
    {"name": "small", "params": "244M", "vram_fp32": "1200", "vram_fp16": "600"},
    {"name": "medium", "params": "769M", "vram_fp32": "3200", "vram_fp16": "1600"},
    {"name": "large-v3", "params": "1550M", "vram_fp32": "6200", "vram_fp16": "3100"},
    {"name": "large-v3-turbo", "params": "809M", "vram_fp32": "3400", "vram_fp16": "1700"},
]


def get_job_dir(job_id: str) -> Path:
    d = WORK_DIR / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def cleanup_job(job_id: str) -> None:
    d = WORK_DIR / job_id
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
