"""Tests for CrisperWhisper 2.0 model support.

Covers: model validation, mode (verbatim/intended) handling, the /api/models
contract, end-to-end routing with a mocked transcriber, and the CrisperWhisper
result -> app result shape conversion.
"""

import asyncio
import math
import shutil
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.config import (
    CRISPER_MODEL_IDS, CRISPER_MODES, UI_MODEL_CHOICES,
    is_crisper_model, validate_model, validate_mode,
)
from app.main import app, CHUNK_DIR, CHUNK_SIZE, MAX_CHUNKS

client = TestClient(app)


def _cleanup_upload(upload_id: str):
    d = CHUNK_DIR / upload_id
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)


def _make_async_mock(result_dict):
    async def _mock(*args, **kwargs):
        return result_dict
    return _mock


def _make_capture_mock(captured):
    """Async mock that records (args, kwargs) into `captured` and returns a result."""
    async def _mock(*args, **kwargs):
        captured.append((args, kwargs))
        return {
            "text": "captured", "language": "en", "duration": 1.0,
            "process_time": 0.1, "segments": [], "id": "x",
            "device": "cpu", "compute_type": "float32",
        }
    return _mock


# ---------------------------------------------------------------------------
# Model validation
# ---------------------------------------------------------------------------


def test_crisper_models_in_ui_choices():
    for name in CRISPER_MODEL_IDS:
        assert name in UI_MODEL_CHOICES


def test_crisper_model_ids_match_upstream():
    """Exact HF repo names the upstream CrisperWhisper supports."""
    assert CRISPER_MODEL_IDS == {
        "crisperwhisper-large": "nyralabs/CrisperWhisper2.0_large",
        "crisperwhisper-turbo": "nyralabs/CrisperWhisper2.0_turbo",
        "crisperwhisper-medium": "nyralabs/CrisperWhisper2.0_medium",
        "crisperwhisper-small": "nyralabs/CrisperWhisper2.0_small",
    }


def test_validate_model_accepts_crisper_models():
    for name in CRISPER_MODEL_IDS:
        assert validate_model(name) == name
        assert is_crisper_model(name)


def test_validate_model_rejects_unknown_crisper_model():
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        validate_model("crisperwhisper-huge")
    assert exc.value.status_code == 400
    assert "Invalid model" in str(exc.value.detail)


def test_validate_model_error_lists_crisper_choices():
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        validate_model("nonexistent")
    msg = str(exc.value.detail)
    assert "crisperwhisper-large" in msg
    assert "base" in msg


def test_validate_model_still_accepts_faster_whisper_models():
    assert validate_model("base") == "base"
    assert validate_model("large-v3-turbo") == "large-v3-turbo"


# ---------------------------------------------------------------------------
# Mode validation
# ---------------------------------------------------------------------------


def test_validate_mode_defaults_to_verbatim():
    assert validate_mode("") == "verbatim"
    assert validate_mode(None) == "verbatim"


def test_validate_mode_accepts_both_modes():
    assert validate_mode("verbatim") == "verbatim"
    assert validate_mode("intended") == "intended"
    assert validate_mode("INTENDED") == "intended"


def test_validate_mode_rejects_invalid():
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        validate_mode("clean")
    assert exc.value.status_code == 400
    for m in CRISPER_MODES:
        assert m in str(exc.value.detail)


# ---------------------------------------------------------------------------
# /api/models contract
# ---------------------------------------------------------------------------


def test_api_models_includes_crisper_choices():
    resp = client.get("/api/models")
    assert resp.status_code == 200
    data = resp.json()
    for name in CRISPER_MODEL_IDS:
        assert name in data["ui_choices"]
    assert "crisper_backend" in data
    assert data["crisper_backend"]["configured"] in ("auto", "ct2", "transformers")


def test_api_models_available_lists_crisper_family():
    resp = client.get("/api/models")
    data = resp.json()
    names = [m["name"] for m in data["available"]]
    for name in CRISPER_MODEL_IDS:
        assert name in names


# ---------------------------------------------------------------------------
# End-to-end API flow (transcriber mocked)
# ---------------------------------------------------------------------------


def test_transcribe_with_crisper_model_and_mode_accepted():
    """POST /api/transcribe with a crisperwhisper model + mode succeeds and
    the mode reaches the transcriber."""
    from app.main import _jobs
    captured = []
    fast_result = {
        "text": "verbatim result", "language": "en",
        "duration": 0.5, "process_time": 0.05,
        "segments": [], "id": "x",
        "device": "cpu", "compute_type": "float32",
    }
    with patch("app.main.transcribe_audio", side_effect=_make_capture_mock(captured)):
        resp = client.post(
            "/api/transcribe",
            files={"file": ("test.wav", b"X" * 1024, "audio/wav")},
            data={"language": "en", "model": "crisperwhisper-small", "mode": "intended"},
        )
        assert resp.status_code == 200, resp.text
        job_id = resp.json()["job_id"]
        assert len(captured) == 1
        _, kwargs = captured[0]
        assert kwargs["model_name"] == "crisperwhisper-small"
        assert kwargs["mode"] == "intended"
        _jobs.pop(job_id, None)


def test_transcribe_crisper_default_mode_is_verbatim():
    """No mode param -> verbatim is passed to the transcriber for crisper models."""
    from app.main import _jobs
    captured = []
    with patch("app.main.transcribe_audio", side_effect=_make_capture_mock(captured)):
        resp = client.post(
            "/api/transcribe",
            files={"file": ("test.wav", b"X" * 1024, "audio/wav")},
            data={"language": "en", "model": "crisperwhisper-medium"},
        )
        assert resp.status_code == 200
        _, kwargs = captured[0]
        assert kwargs["mode"] == "verbatim"
        _jobs.pop(resp.json()["job_id"], None)


def test_transcribe_invalid_mode_rejected():
    resp = client.post(
        "/api/transcribe",
        files={"file": ("test.wav", b"X" * 1024, "audio/wav")},
        data={"language": "en", "model": "crisperwhisper-small", "mode": "clean"},
    )
    assert resp.status_code == 400
    assert "Invalid mode" in resp.json()["detail"]


def test_faster_whisper_model_ignores_mode():
    """faster-whisper models still work when a mode is supplied (ignored)."""
    from app.main import _jobs
    captured = []
    with patch("app.main.transcribe_audio", side_effect=_make_capture_mock(captured)):
        resp = client.post(
            "/api/transcribe",
            files={"file": ("test.wav", b"X" * 1024, "audio/wav")},
            data={"language": "en", "model": "base", "mode": "verbatim"},
        )
        assert resp.status_code == 200
        _, kwargs = captured[0]
        assert kwargs["model_name"] == "base"
        _jobs.pop(resp.json()["job_id"], None)


def test_chunked_upload_crisper_model_with_mode():
    """Chunked upload finish accepts a crisper model and passes mode through."""
    from app.main import _jobs
    captured = []
    upload_id = None
    try:
        resp = client.post(
            "/api/upload/start",
            data={"filename": "test.wav", "size": 10, "total_chunks": 1},
        )
        assert resp.status_code == 200
        upload_id = resp.json()["upload_id"]
        client.post(
            f"/api/upload/chunk/{upload_id}",
            data={"chunk_index": 0},
            files={"file": ("test.wav.part0", b"X" * 10, "audio/wav")},
        )
        with patch("app.main.transcribe_audio", side_effect=_make_capture_mock(captured)):
            resp = client.post(
                f"/api/upload/finish/{upload_id}",
                params={"model": "crisperwhisper-large", "mode": "verbatim"},
                data={},
                timeout=2.0,
            )
            assert resp.status_code == 200, resp.text
            job_id = resp.json()["job_id"]
            assert len(captured) == 1
            _, kwargs = captured[0]
            assert kwargs["model_name"] == "crisperwhisper-large"
            assert kwargs["mode"] == "verbatim"
            _jobs.pop(job_id, None)
    finally:
        if upload_id:
            _cleanup_upload(upload_id)


# ---------------------------------------------------------------------------
# Transcriber: CrisperWhisper result -> app result shape
# ---------------------------------------------------------------------------


def _fake_crisper_result(text="hello [um] world", words=None):
    return SimpleNamespace(
        text=text,
        language="en",
        mode="verbatim",
        duration=4.0,
        processing_time=1.5,
        chunks=None,
        words=words or [
            SimpleNamespace(word="hello", start=0.0, end=0.5),
            SimpleNamespace(word="[um]", start=0.6, end=0.8),
            SimpleNamespace(word="world", start=0.9, end=1.4),
        ],
    )


def test_crisper_result_shape_conversion():
    """_transcribe_crisper_sync converts a CrisperWhisper result into the app
    result shape (segments with ms timings, per-word timestamps, mode)."""
    from app import transcriber

    fake_model = SimpleNamespace(transcribe=lambda *a, **k: _fake_crisper_result())
    with patch.dict(transcriber._crisper_models, {"crisperwhisper-small": fake_model}), \
         patch.object(transcriber, "_resolved_crisper_backend", "transformers"), \
         patch.object(transcriber, "_device_info", {"device": "cpu"}):
        result = transcriber._transcribe_crisper_sync(
            "/tmp/audio.wav", "en", "job123", "crisperwhisper-small", "verbatim"
        )

    assert result["text"] == "hello [um] world"
    assert result["language"] == "en"
    assert result["mode"] == "verbatim"
    assert result["model"] == "crisperwhisper-small"
    assert result["backend"] == "transformers"
    assert result["device"] == "cpu"
    assert result["compute_type"] == "float32"
    assert result["duration"] == 4.0
    assert result["process_time"] > 0

    # word-level timestamps, ms ints
    assert result["words"] == [
        {"word": "hello", "t0": 0, "t1": 500},
        {"word": "[um]", "t0": 600, "t1": 800},
        {"word": "world", "t0": 900, "t1": 1400},
    ]
    # segments group words and keep per-word data
    assert len(result["segments"]) == 1
    seg = result["segments"][0]
    assert seg["text"] == "hello [um] world"
    assert seg["t0"] == 0
    assert seg["t1"] == 1400
    assert len(seg["words"]) == 3


def test_crisper_words_to_segments_grouping():
    """Words are grouped into bounded segments; sentence ends break segments."""
    from app.transcriber import _crisper_words_to_segments

    words = [
        {"word": f"w{i}", "t0": i * 100, "t1": i * 100 + 50}
        for i in range(25)
    ]
    # force a sentence break at index 5 (before the 10-word cap)
    words[5] = {"word": "stop.", "t0": 500, "t1": 550}
    segments = _crisper_words_to_segments(words)
    # break at sentence end: words 0..5 in the first segment
    assert segments[0]["text"].endswith("stop.")
    assert len(segments[0]["words"]) == 6
    # remaining 19 words -> max 10 per segment, so two more segments
    assert len(segments) == 3
    assert all(len(s["words"]) <= 10 for s in segments)
    # contiguous timings
    for s in segments:
        assert s["t0"] <= s["t1"]


def test_crisper_unplaceable_words_are_skipped():
    """WordTimestamp entries with None timings are dropped."""
    from app import transcriber

    words = [
        SimpleNamespace(word="keep", start=0.0, end=0.4),
        SimpleNamespace(word="drop", start=None, end=None),
        SimpleNamespace(word="tail", start=0.8, end=1.2),
    ]
    fake_model = SimpleNamespace(transcribe=lambda *a, **k: _fake_crisper_result(words=words))
    with patch.dict(transcriber._crisper_models, {"crisperwhisper-small": fake_model}), \
         patch.object(transcriber, "_resolved_crisper_backend", "transformers"), \
         patch.object(transcriber, "_device_info", {"device": "cpu"}):
        result = transcriber._transcribe_crisper_sync(
            "/tmp/audio.wav", "en", "job123", "crisperwhisper-small", "intended"
        )
    assert [w["word"] for w in result["words"]] == ["keep", "tail"]
    assert result["mode"] == "intended"


def test_transcribe_audio_routes_crisper_to_dedicated_executor():
    """transcribe_audio routes a crisper model through _transcribe_sync and
    the dedicated crisper executor, returning the converted result."""
    from app import transcriber

    fake_model = SimpleNamespace(transcribe=lambda *a, **k: _fake_crisper_result())
    with patch.dict(transcriber._crisper_models, {"crisperwhisper-small": fake_model}), \
         patch.object(transcriber, "_resolved_crisper_backend", "transformers"), \
         patch.object(transcriber, "_device_info", {"device": "cpu"}):
        result = asyncio.run(
            transcriber.transcribe_audio(
                "/tmp/audio.wav", "en", "job123", "crisperwhisper-small", "verbatim"
            )
        )
    assert result["text"] == "hello [um] world"
    assert result["backend"] == "transformers"
    assert result["model"] == "crisperwhisper-small"


def test_load_crisper_model_missing_extra_raises():
    """When crisperwhisper is not importable, a clear RuntimeError is raised."""
    from app import transcriber

    with patch.object(transcriber, "CrisperWhisperModel", None):
        with pytest.raises(RuntimeError) as exc:
            transcriber._load_crisper_model_sync("crisperwhisper-small")
        assert "crisperwhisper" in str(exc.value)


def test_load_model_routes_crisper_models():
    """load_model() routes crisperwhisper-* choices to the crisper loader."""
    from app import transcriber

    loaded = []
    fake_model = object()

    def _fake_load_sync(model_name):
        loaded.append(model_name)
        return fake_model

    with patch.object(transcriber, "_crisper_executor") as mock_exec, \
         patch.object(transcriber, "_device_info", {"device": "cpu"}):
        mock_exec.submit.return_value.result.return_value = None
        transcriber.load_model("crisperwhisper-turbo")
        assert mock_exec.submit.called
        # crisper load does not touch the faster-whisper cache
        assert transcriber._crisper_models == {}
