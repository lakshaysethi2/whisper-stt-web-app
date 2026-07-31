import os
import shutil
from pathlib import Path

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en")
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "536870912"))   # 512 MB (default; safer for tight-disk VPS)
MIN_FREE_DISK_BYTES = int(os.getenv("MIN_FREE_DISK_BYTES", "2147483648"))  # 2 GB minimum free space
JOB_RETENTION_SECONDS = int(os.getenv("JOB_RETENTION_SECONDS", "7200"))  # 2 hours default

WORK_DIR = Path(os.getenv("WORK_DIR", "/tmp/whisper-stt"))
WORK_DIR.mkdir(parents=True, exist_ok=True)


ALLOWED_EXTENSIONS = {
    # Audio formats
    ".mp3", ".wav", ".flac", ".ogg", ".m4a",
    ".aac", ".wma", ".opus",
    # Video formats
    ".mp4", ".webm", ".avi", ".mov", ".mkv",
    ".flv", ".wmv", ".mpeg", ".mpg", ".3gp",
    ".m4v", ".asf",
}

UI_MODEL_CHOICES = ["base", "large-v3-turbo"]

SUPPORTED_MODELS = [
    {"name": "tiny", "params": "39M", "vram_fp32": "200", "vram_fp16": "128"},
    {"name": "base", "params": "74M", "vram_fp32": "400", "vram_fp16": "256"},
    {"name": "small", "params": "244M", "vram_fp32": "1200", "vram_fp16": "600"},
    {"name": "medium", "params": "769M", "vram_fp32": "3200", "vram_fp16": "1600"},
    {"name": "large-v3", "params": "1550M", "vram_fp32": "6200", "vram_fp16": "3100"},
    {"name": "large-v3-turbo", "params": "809M", "vram_fp32": "3400", "vram_fp16": "1700"},
]


def validate_model(model_name: str | None) -> str:
    """Validate a model choice. Returns the validated model name or raises HTTPException(400)."""
    if not model_name or not isinstance(model_name, str) or not model_name.strip():
        from fastapi import HTTPException
        raise HTTPException(400, "Model choice is required. Choose 'base' or 'large-v3-turbo'.")
    model_name = model_name.strip()
    valid_names = UI_MODEL_CHOICES
    if model_name not in valid_names:
        from fastapi import HTTPException
        raise HTTPException(
            400,
            f"Invalid model: '{model_name}'. Valid choices: {', '.join(valid_names)}",
        )
    return model_name


def get_job_dir(job_id: str) -> Path:
    d = WORK_DIR / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def cleanup_job(job_id: str) -> None:
    d = WORK_DIR / job_id
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)


def check_disk_space(path: str | Path = None) -> bool:
    """Return True if free disk at path is above MIN_FREE_DISK_BYTES."""
    target = Path(path or WORK_DIR)
    try:
        usage = shutil.disk_usage(str(target))
        return usage.free >= MIN_FREE_DISK_BYTES
    except OSError:
        # If we cannot check, assume safe (transient mount issue, etc.)
        return True


def cleanup_all_jobs() -> None:
    if WORK_DIR.exists():
        for p in WORK_DIR.iterdir():
            # Preserve the chunked-upload sessions directory; it is cleaned
            # by its own logic in app.main.
            if p.is_dir() and p.name != "chunks":
                shutil.rmtree(p, ignore_errors=True)
            elif p.is_file():
                p.unlink(missing_ok=True)

