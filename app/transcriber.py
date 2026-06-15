import asyncio
import shutil
from pathlib import Path

from faster_whisper import WhisperModel, BatchedInferencePipeline

from app.config import WHISPER_MODEL, get_job_dir

_model = None
_batched = None


def _get_model():
    global _model, _batched
    if _model is None:
        _model = WhisperModel(
            WHISPER_MODEL,
            device="cuda",
            compute_type="int8_float16",
        )
        _batched = BatchedInferencePipeline(model=_model)
    return _batched


async def transcribe_audio(
    audio_path: str,
    language: str = None,
    job_id: str = "unknown",
) -> dict:
    pipeline = _get_model()

    segments, info = await asyncio.to_thread(
        pipeline.transcribe,
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
