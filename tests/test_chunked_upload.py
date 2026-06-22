import json
import math
import shutil
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app, CHUNK_DIR, CHUNK_SIZE, MAX_CHUNKS

client = TestClient(app)


def _cleanup_upload(upload_id: str):
    d = CHUNK_DIR / upload_id
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)


def test_chunked_upload_flow():
    """Start -> chunks -> finish (mocked transcription)."""
    upload_id = None
    try:
        payload = b"AA" * 512
        size = len(payload)
        resp = client.post(
            "/api/upload/start",
            data={"filename": "test.wav", "size": size, "total_chunks": math.ceil(size / CHUNK_SIZE)},
        )
        assert resp.status_code == 200
        upload_id = resp.json()["upload_id"]
        total_chunks = math.ceil(size / CHUNK_SIZE)

        resp = client.post(
            f"/api/upload/chunk/{upload_id}",
            data={"chunk_index": 0},
            files={"file": ("test.wav.part0", payload, "audio/wav")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["received"] == 1
        assert body["total"] == total_chunks

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


def test_missing_chunks_returns_detail_and_preserves_session():
    """finish with missing chunks returns detail and does NOT delete the session."""
    with patch("app.main.CHUNK_SIZE", 10), patch("app.main.MAX_CHUNKS", 100):
        upload_id = None
        try:
            resp = client.post(
                "/api/upload/start",
                data={"filename": "test.wav", "size": 20, "total_chunks": math.ceil(20 / 10)},
            )
            upload_id = resp.json()["upload_id"]
            total_chunks = math.ceil(20 / 10)

            # Upload only chunks 0 and 2, skip chunk 1
            for i in [0, 2]:
                client.post(
                    f"/api/upload/chunk/{upload_id}",
                    data={"chunk_index": i},
                    files={"file": (f"test.wav.part{i}", b"X" * 10, "audio/wav")},
                )

            resp = client.post(f"/api/upload/finish/{upload_id}", data={})
            assert resp.status_code == 400
            detail = resp.json()["detail"]
            assert "upload_id" in detail
            assert detail["total_chunks"] == total_chunks

            # Session must still exist on disk so client can upload missing chunks
            assert (CHUNK_DIR / upload_id).exists()
        finally:
            if upload_id:
                _cleanup_upload(upload_id)


def test_invalid_language_preserves_session():
    """Invalid language code rejects finish without deleting the session."""
    with patch("app.main.CHUNK_SIZE", 10), patch("app.main.MAX_CHUNKS", 100):
        upload_id = None
        try:
            resp = client.post(
                "/api/upload/start",
                data={"filename": "test.wav", "size": 20, "total_chunks": math.ceil(20 / 10)},
            )
            upload_id = resp.json()["upload_id"]
            total_chunks = math.ceil(20 / 10)

            for i in range(total_chunks):
                client.post(
                    f"/api/upload/chunk/{upload_id}",
                    data={"chunk_index": i},
                    files={"file": (f"test.wav.part{i}", b"X" * 10, "audio/wav")},
                )

            resp = client.post(f"/api/upload/finish/{upload_id}", data={"language": "!!!"})
            assert resp.status_code == 400
            assert "Invalid language" in resp.json()["detail"]

            # Session must still exist
            assert (CHUNK_DIR / upload_id).exists()
        finally:
            if upload_id:
                _cleanup_upload(upload_id)


def test_size_mismatch_rejected():
    """Assembled size must match declared size."""
    with patch("app.main.CHUNK_SIZE", 10), patch("app.main.MAX_CHUNKS", 100):
        upload_id = None
        try:
            resp = client.post(
                "/api/upload/start",
                data={"filename": "test.wav", "size": 20, "total_chunks": math.ceil(20 / 10)},
            )
            upload_id = resp.json()["upload_id"]
            total_chunks = math.ceil(20 / 10)

            # Upload wrong amount of data
            for i in range(total_chunks):
                client.post(
                    f"/api/upload/chunk/{upload_id}",
                    data={"chunk_index": i},
                    files={"file": (f"test.wav.part{i}", b"X" * 5, "audio/wav")},
                )

            with patch("app.main.transcribe_audio"):
                resp = client.post(f"/api/upload/finish/{upload_id}", data={"language": "en"})
                assert resp.status_code == 400
                assert "size mismatch" in resp.json()["detail"].lower()
        finally:
            if upload_id:
                _cleanup_upload(upload_id)


def test_invalid_total_chunks_rejected():
    """total_chunks must match expected count for declared size."""
    with patch("app.main.CHUNK_SIZE", 10), patch("app.main.MAX_CHUNKS", 100):
        resp = client.post(
            "/api/upload/start",
            data={"filename": "test.wav", "size": 100, "total_chunks": 999},
        )
        assert resp.status_code == 400
        assert "Invalid total_chunks" in resp.json()["detail"]


def test_chunk_too_large():
    """Reject chunks exceeding CHUNK_MAX_SIZE."""
    with patch("app.main.CHUNK_SIZE", 10), patch("app.main.MAX_CHUNKS", 100), \
         patch("app.main.CHUNK_MAX_SIZE", 1024):
        upload_id = None
        try:
            resp = client.post(
                "/api/upload/start",
                data={"filename": "big.wav", "size": 200, "total_chunks": math.ceil(200 / 10)},
            )
            assert resp.status_code == 200
            upload_id = resp.json()["upload_id"]

            oversized = b"\x00" * 1025
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

    with patch("app.main.CHUNK_SIZE", 10), patch("app.main.MAX_CHUNKS", 100):
        upload_id = None
        try:
            resp = client.post(
                "/api/upload/start",
                data={"filename": "test.wav", "size": 10, "total_chunks": 1},
            )
            upload_id = resp.json()["upload_id"]

            client.post(
                f"/api/upload/chunk/{upload_id}",
                data={"chunk_index": 0},
                files={"file": ("test.wav.part0", b"X" * 10, "audio/wav")},
            )

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
    assert data["chunk_size"] == CHUNK_SIZE
    assert "max_chunk_size" in data
    assert "max_file_size" in data
    assert "max_chunks" in data
    assert "direct_upload_threshold" in data
    assert "allowed_extensions" in data
    assert isinstance(data["allowed_extensions"], list)
