import asyncio
import logging
import time

from faster_whisper import WhisperModel

from app.config import WHISPER_MODEL, get_job_dir

logger = logging.getLogger(__name__)

_model = None
_device_info = {}
_use_batched = False


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
    """Detect GPU and compute type. GPU-only — raises if no CUDA available."""
    try:
        import ctypes
        ctypes.CDLL("libcuda.so.1")
    except OSError:
        try:
            ctypes.CDLL("libcuda.so")
        except OSError:
            raise RuntimeError("No CUDA library found — GPU is required")

    gpu_info = _get_gpu_info()
    if not gpu_info:
        raise RuntimeError("nvidia-smi failed — cannot detect GPU")

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
        raise RuntimeError(f"GPU CC {cc_major}.{cc_minor} too old for CUDA kernels")

    return "cuda", compute_type, cc_int


def load_model():
    global _model, _device_info, _use_batched
    if _model is not None:
        return

    device, compute_type, cc = _detect_device()
    _device_info.update({"device": device, "compute_type": compute_type, "compute_capability": cc})
    _use_batched = cc >= 70

    logger.info("Loading model %s on %s (%s, batched=%s)...", WHISPER_MODEL, device, compute_type, _use_batched)

    _model = WhisperModel(
        WHISPER_MODEL,
        device=device,
        compute_type=compute_type,
    )

    logger.info("Model %s loaded on %s/%s.", WHISPER_MODEL, device, compute_type)


async def transcribe_audio(
    audio_path: str,
    language: str = None,
    job_id: str = "unknown",
) -> dict:
    start = time.monotonic()

    if _use_batched:
        from faster_whisper import BatchedInferencePipeline
        batched = BatchedInferencePipeline(model=_model)
        segments, info = await asyncio.to_thread(
            batched.transcribe,
            audio_path,
            language=language or "en",
            batch_size=16,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200),
            word_timestamps=False,
            condition_on_previous_text=True,
        )
    else:
        segments, info = await asyncio.to_thread(
            _model.transcribe,
            audio_path,
            language=language or "en",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200),
            word_timestamps=False,
            condition_on_previous_text=True,
        )
    elapsed = time.monotonic() - start

    logger.info("Transcription done in %.2fs device=%s", elapsed, _device_info.get("device"))

    result = {"id": job_id, "segments": []}
    result["language"] = info.language
    result["device"] = _device_info.get("device", "unknown")
    result["compute_type"] = _device_info.get("compute_type", "unknown")

    full_text_parts = []
    total_duration = 0.0

    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        full_text_parts.append(text)
        duration = seg.end - seg.start
        total_duration += duration

        result["segments"].append({
            "text": text,
            "t0": int(seg.start * 1000),
            "t1": int(seg.end * 1000),
        })

    result["text"] = " ".join(full_text_parts)
    result["duration"] = total_duration
    result["process_time"] = round(elapsed, 2)

    if total_duration > 0:
        result["realtime_factor"] = round(total_duration / elapsed, 1)

    return result
