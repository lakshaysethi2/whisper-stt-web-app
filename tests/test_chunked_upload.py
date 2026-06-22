import json
import shutil
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app, CHUNK_DIR

client = TestClient(app)


def _cleanup_upload(upload_id: str):
    d = CHUNK_DIR / upload_id
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)


def test_chunked_upload_flow():
    """Start → chunks → finish (mocked transcription)."""
    upload_id = None
    try:
        # 1. Start
        resp = client.post(
            "/api/upload/start",
            data={"filename": "test.wav", "size": 12, "total_chunks": 2},
        )
        assert resp.status_code == 200
        upload_id = resp.json()["upload_id"]

        # 2. Upload two small chunks
        for i, payload in enumerate([b"AA", b"BB"]):
            resp = client.post(
                f"/api/upload/chunk/{upload_id}",
                data={"chunk_index": i},
                files={"file": (f"test.wav.part{i}", payload, "audio/wav")},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["ok"] is True
            assert body["received"] == i + 1
            assert body["total"] == 2

        # 3. Finish – mock transcribe_audio to avoid model load
        with patch("app.main.transcribe_audio") as mock_transcribe:
            mock_transcribe.return_value = {
                "text": "hello world",
                "language": "en",
                "duration": 2.0,
                "process_time": 0.1,
                "segments": [],
            }
            resp = client.post(
                f"/api/upload/finish/{upload_id}",
                data={"language": "en"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["text"] == "hello world"

    finally:
        if upload_id:
            _cleanup_upload(upload_id)


def test_missing_chunks_returns_detail():
    """finish with missing chunks returns upload_id and missing list."""
    upload_id = None
    try:
        resp = client.post(
            "/api/upload/start",
            data={"filename": "test.wav", "size": 20, "total_chunks": 3},
        )
        upload_id = resp.json()["upload_id"]

        # Only upload chunk 0 and 2, skip 1
        for i in [0, 2]:
            client.post(
                f"/api/upload/chunk/{upload_id}",
                data={"chunk_index": i},
                files={"file": (f"test.wav.part{i}", b"XX", "audio/wav")},
            )

        resp = client.post(f"/api/upload/finish/{upload_id}", data={})
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert "upload_id" in detail
        assert 1 in detail["missing"]
        assert detail["total_chunks"] == 3
    finally:
        if upload_id:
            _cleanup_upload(upload_id)


def test_chunk_too_large():
    """Reject chunks exceeding CHUNK_MAX_SIZE."""
    from app.main import CHUNK_MAX_SIZE

    upload_id = None
    try:
        resp = client.post(
            "/api/upload/start",
            data={"filename": "big.wav", "size": 100, "total_chunks": 1},
        )
        upload_id = resp.json()["upload_id"]

        oversized = b"\x00" * (CHUNK_MAX_SIZE + 1)
        resp = client.post(
            f"/api/upload/chunk/{upload_id}",
            data={"chunk_index": 0},
            files={"file": ("big.wav.part0", oversized, "audio/wav")},
        )
        assert resp.status_code == 413
    finally:
        if upload_id:
            _cleanup_upload(upload_id)


def test_concurrent_finish_guard():
    """Second finish call while first is in-progress returns 409."""
    from app.main import _finish_locks

    upload_id = None
    try:
        resp = client.post(
            "/api/upload/start",
            data={"filename": "test.wav", "size": 5, "total_chunks": 1},
        )
        upload_id = resp.json()["upload_id"]

        client.post(
            f"/api/upload/chunk/{upload_id}",
            data={"chunk_index": 0},
            files={"file": ("test.wav.part0", b"hello", "audio/wav")},
        )

        # Simulate an in-progress finish by adding to lock set
        _finish_locks.add(upload_id)
        try:
            resp = client.post(f"/api/upload/finish/{upload_id}", data={})
            assert resp.status_code == 409
        finally:
            _finish_locks.discard(upload_id)
    finally:
        if upload_id:
            _cleanup_upload(upload_id)


def test_upload_config():
    """GET /api/upload/config returns expected keys."""
    resp = client.get("/api/upload/config")
    assert resp.status_code == 200
    data = resp.json()
    assert "chunk_size" in data
    assert "max_file_size" in data
    assert "direct_upload_threshold" in data
    assert "allowed_extensions" in data
    assert isinstance(data["allowed_extensions"], list)
