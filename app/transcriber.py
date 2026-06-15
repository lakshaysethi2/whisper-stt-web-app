import asyncio
import logging

from faster_whisper import WhisperModel, BatchedInferencePipeline

from app.config import WHISPER_MODEL, get_job_dir

logger = logging.getLogger(__name__)

_model = None
_batched = None


def load_model():
    global _model, _batched
    if _model is not None:
        return
    logger.info("Loading model %s...", WHISPER_MODEL)
    try:
        _model = WhisperModel(
            WHISPER_MODEL,
            device="cuda",
            compute_type="float32",
        )
    except Exception:
        _model = WhisperModel(
            WHISPER_MODEL,
            device="cpu",
            compute_type="int8",
        )
    _batched = BatchedInferencePipeline(model=_model)
    logger.info("Model %s loaded.", WHISPER_MODEL)


async def transcribe_audio(
    audio_path: str,
    language: str = None,
    job_id: str = "unknown",
) -> dict:
    segments, info = await asyncio.to_thread(
        _batched.transcribe,
        audio_path,
        language=language or "en",
        batch_size=16,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=500,
            speech_pad_ms=200,
        ),
        word_timestamps=False,
        condition_on_previous_text=True,
    )

    result = {"id": job_id, "segments": []}
    result["language"] = info.language

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

    return result
