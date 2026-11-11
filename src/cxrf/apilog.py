"""Append-only JSONL log for every API call (study plan: reproducibility rules).

Each record stores the provider, the exact model snapshot string returned by the API,
the endpoint, the UTC timestamp, the prompt, the parameters, the raw response and token
usage. Image bytes are never inlined: images are referenced by path + sha256.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

LOG_DIR = Path(__file__).resolve().parents[2] / "logs" / "api"


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonable(obj: Any) -> Any:
    """Best-effort conversion of SDK response objects to plain JSON."""
    if obj is None or isinstance(obj, (dict, list, str, int, float, bool)):
        return obj
    for attr in ("model_dump", "to_dict", "dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass
    if isinstance(obj, (bytes, bytearray)):
        return {"bytes_sha256": hashlib.sha256(obj).hexdigest(), "len": len(obj)}
    return repr(obj)


def log_call(
    *,
    experiment: str,
    provider: str,
    endpoint: str,
    requested_model: str,
    returned_model: str | None,
    prompt: str,
    params: dict,
    response: Any,
    input_images: list[str] | None = None,
    output_images: list[str] | None = None,
    usage: dict | None = None,
    error: str | None = None,
    log_name: str | None = None,
) -> str:
    """Append one record; returns its call_id."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    call_id = uuid.uuid4().hex
    rec = {
        "call_id": call_id,
        "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "experiment": experiment,
        "provider": provider,
        "endpoint": endpoint,
        "requested_model": requested_model,
        "returned_model": returned_model,
        "prompt": prompt,
        "params": params,
        "input_images": [{"path": str(p), "sha256": sha256_file(p)} for p in (input_images or [])],
        "output_images": [{"path": str(p), "sha256": sha256_file(p)} for p in (output_images or [])],
        "usage": usage,
        "error": error,
        "response": _jsonable(response),
    }
    path = LOG_DIR / f"{log_name or experiment}.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=repr, ensure_ascii=False) + "\n")
    return call_id
