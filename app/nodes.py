"""Transcription node configuration.

GPU workers are listed in config/nodes.yaml when present (see
config/nodes.example.yaml for the shape); otherwise the legacy
WHISPER_WORKER_URLS / WHISPER_WORKER_NAMES env vars are used so plain
single-host deployments keep working unchanged. Each node is
{"name": <public label>, "url": <reachable base URL>}.

Public labels must stay neutral (e.g. "GPU Node A") — no machine names.
"""
import logging
import os
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# The built-in CPU-only node. Selecting it never dispatches to a GPU worker.
CPU_NODE = "CPU"

_NODES_FILE = Path(__file__).resolve().parent.parent / "config" / "nodes.yaml"


def _from_yaml() -> list[dict]:
    try:
        data = yaml.safe_load(_NODES_FILE.read_text()) or {}
    except FileNotFoundError:
        return []
    except Exception as e:
        logger.warning("Failed to read %s: %s", _NODES_FILE, e)
        return []
    out = []
    for n in data.get("nodes") or []:
        name = str(n.get("name", "")).strip()
        url = str(n.get("url", "")).strip().rstrip("/")
        if name and url:
            out.append({"name": name, "url": url})
    if out:
        logger.info("Loaded %d node(s) from %s", len(out), _NODES_FILE)
    return out


def _from_env() -> list[dict]:
    urls = [u.strip().rstrip("/") for u in os.getenv("WHISPER_WORKER_URLS", "").split(",") if u.strip()]
    names = [n.strip() for n in os.getenv("WHISPER_WORKER_NAMES", "").split(",") if n.strip()]
    out = []
    for i, url in enumerate(urls):
        name = names[i] if i < len(names) else f"GPU Node {'ABCDEFGH'[i % 8]}"
        out.append({"name": name, "url": url})
    if out:
        logger.info("Loaded %d node(s) from WHISPER_WORKER_URLS env", len(out))
    return out


NODES: list[dict] = _from_yaml() or _from_env()
