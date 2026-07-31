import asyncio
import logging
import time

from faster_whisper import WhisperModel, BatchedInferencePipeline

from app.config import WHISPER_MODEL

logger = logging.getLogger(__name__)

# Multi-model cache: model_name -> (WhisperModel, BatchedInferencePipeline)
_models: dict[str, tuple[WhisperModel, BatchedInferencePipeline]] = {}
_device_info = {}

# Shared progress dict for real-time transcription progress tracking.
# Keyed by job_id, value is progress float (0.0 to 1.0) or None when unknown.
_progress: dict[str, float | None] = {}


def _get_gpu_info() -> dict:
    """Query nvidia-smi for GPU name and memory."""
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            line = result.stdout.strip().split("\n")[0]
            parts = line.rsplit(",", 1)
            return {"name": parts[0].strip(), "vram_mb": int(parts[1].strip())}
    except (subprocess.TimeoutExpired, FileNotFoundError, IndexError, ValueError):
        pass
    return {}


def _gpu_name_to_cc(name: str) -> int:
    """Map GPU name to approximate compute capability."""
    name_lower = name.lower()
    if any(x in name_lower for x in ["940mx", "950m", "960m", "940m", "gtx 9", "gtx9"]):
        return 50
    if any(x in name_lower for x in ["p100", "p40", "p4", "gtx 10", "gtx10", "tesla p"]):
        return 61
    if any(x in name_lower for x in ["v100", "tesla v"]):
        return 70
    if any(x in name_lower for x in ["rtx 20", "rtx20", "t4", "quadro rtx"]):
        return 75
    if any(x in name_lower for x in ["a100", "a30", "a10", "rtx 30", "rtx30", "l4"]):
        return 80
    if any(x in name_lower for x in ["rtx 40", "rtx40", "l40", "l40s"]):
        return 89
    if any(x in name_lower for x in ["b100", "b200", "b40"]):
        return 90
    return 70


def _detect_device() -> tuple[str, str, int]:
    """Detect the best device and compute type for the available GPU.

    Returns (device, compute_type, compute_capability_x10).
    Compute capability is reported as integer (e.g., 50 for CC 5.0).
    """
    try:
        import ctypes
        ctypes.CDLL("libcuda.so.1")
    except OSError:
        try:
            import ctypes
            ctypes.CDLL("libcuda.so")
        except OSError:
            logger.info("No CUDA library found, using CPU int8")
            return "cpu", "int8", 0

    gpu_info = _get_gpu_info()
    if not gpu_info:
        logger.warning("CUDA found but nvidia-smi failed, falling back to CPU int8")
        return "cpu", "int8", 0

    device_name = gpu_info["name"]
    vram_mb = gpu_info["vram_mb"]
    cc_int = _gpu_name_to_cc(device_name)
    cc_major = cc_int // 10
    cc_minor = cc_int % 10

    logger.info("GPU: %s (CC %d.%d, %d MB VRAM)", device_name, cc_major, cc_minor, vram_mb)

    if cc_major >= 7:
        compute_type = "float16"
    elif cc_major >= 5:
        compute_type = "float32"
    else:
        logger.warning("GPU CC %d.%d too old for CUDA kernels, falling back to CPU", cc_major, cc_minor)
        return "cpu", "int8", cc_int

    return "cuda", compute_type, cc_int


def get_loaded_models() -> list[str]:
    """Return list of currently loaded model names."""
    return list(_models.keys())


def load_model(model_name: str | None = None) -> None:
    """Load a specific model into the cache. If model_name is None, load WHISPER_MODEL."""
    global _device_info

    target = model_name or WHISPER_MODEL

    if target in _models:
        logger.info("Model %s already loaded", target)
        return

    if not _device_info:
        device, compute_type, cc = _detect_device()
        _device_info.update({"device": device, "compute_type": compute_type, "compute_capability": cc})

    device = _device_info.get("device", "cpu")
    compute_type = _device_info.get("compute_type", "int8")

    logger.info("Loading model %s on %s (%s)...", target, device, compute_type)

    try:
        model = WhisperModel(
            target,
            device=device,
            compute_type=compute_type,
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and device == "cuda":
            logger.warning(
                "GPU OOM with %s on %s, falling back to CPU int8", target, compute_type
            )
            device, compute_type, cc = "cpu", "int8", 0
            _device_info = {"device": device, "compute_type": compute_type, "compute_capability": cc}
            model = WhisperModel(
                target,
                device=device,
                compute_type=compute_type,
            )
        else:
            raise

    batch_size = 16 if device == "cuda" else 1
    batched = BatchedInferencePipeline(model=model)
    _models[target] = (model, batched)
    logger.info("Model %s loaded on %s/%s (batch_size=%d).", target, device, compute_type, batch_size)


def get_progress(job_id: str) -> float | None:
    """Thread-safe read of real transcription progress for a job.
    Returns a float 0.0-1.0, or None if no progress data is available."""
    return _progress.get(job_id)


def _transcribe_sync(
    audio_path: str,
    language: str,
    job_id: str,
    model_name: str | None = None,
) -> dict:
    """Synchronous transcription function that runs entirely in a thread.
    Consumes the segment generator inside the thread so the asyncio event loop
    is never blocked. Updates real progress in _progress dict during iteration."""
    target = model_name or WHISPER_MODEL
    device = _device_info.get("device", "cpu")

    # Get or load the model
    if target not in _models:
        load_model(target)
    _model, batched = _models[target]

    start = time.monotonic()
    segments, info = batched.transcribe(
        audio_path,
        language=language or "en",
        batch_size=16 if device == "cuda" else 1,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=500,
            speech_pad_ms=200,
        ),
        word_timestamps=False,
        condition_on_previous_text=True,
    )
    elapsed = time.monotonic() - start

    logger.info("Transcription done in %.2fs device=%s model=%s", elapsed, device, target)

    result = {"id": job_id, "segments": []}
    result["language"] = info.language
    result["device"] = _device_info.get("device", "unknown")
    result["compute_type"] = _device_info.get("compute_type", "unknown")
    result["model"] = target

    audio_duration = info.duration  # total audio duration in seconds
    full_text_parts = []
    total_duration = 0.0
    max_end = 0.0

    for seg in segments:
        text = seg.text.strip()
        if not text:
            max_end = max(max_end, seg.end)
            if audio_duration and audio_duration > 0:
                _progress[job_id] = min(1.0, max_end / audio_duration)
            continue
        full_text_parts.append(text)
        duration = seg.end - seg.start
        total_duration += duration
        max_end = max(max_end, seg.end)

        result["segments"].append({
            "text": text,
            "t0": int(seg.start * 1000),
            "t1": int(seg.end * 1000),
        })

        # Update real progress based on audio position
        if audio_duration and audio_duration > 0:
            _progress[job_id] = min(1.0, max_end / audio_duration)

    result["text"] = " ".join(full_text_parts)
    result["duration"] = total_duration
    result["process_time"] = round(elapsed, 2)

    if total_duration > 0:
        result["realtime_factor"] = round(total_duration / elapsed, 1)

    return result


async def transcribe_audio(
    audio_path: str,
    language: str = None,
    job_id: str = "unknown",
    model_name: str | None = None,
) -> dict:
    """Transcribe audio in a thread. Never blocks the asyncio event loop.
    Optionally specify model_name to use a specific Whisper model.
    Returns the result dict once both inference and segment iteration complete."""
    _progress[job_id] = 0.0
    try:
        result = await asyncio.to_thread(
            _transcribe_sync,
            audio_path,
            language or "en",
            job_id,
            model_name,
        )
        _progress[job_id] = 1.0
        return result
    finally:
        _progress.pop(job_id, None)
