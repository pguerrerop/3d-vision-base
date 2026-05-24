from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULTS_FILENAME = "pipeline_defaults_25d.json"


def defaults_path(state_dir: Path) -> Path:
    return state_dir / DEFAULTS_FILENAME


def read_25d_defaults(state_dir: Path) -> dict[str, Any]:
    path = defaults_path(state_dir)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def write_25d_defaults(state_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = defaults_path(state_dir)
    sanitized = payload if isinstance(payload, dict) else {}
    path.write_text(json.dumps(sanitized, indent=2), encoding="utf-8")
    return sanitized


def clear_25d_defaults(state_dir: Path) -> None:
    path = defaults_path(state_dir)
    if path.exists():
        path.unlink()


def merge_with_25d_defaults(state_dir: Path, stage_params: dict[str, Any] | None) -> dict[str, Any]:
    saved = read_25d_defaults(state_dir)
    incoming = stage_params or {}
    if not saved:
        return dict(incoming)
    merged: dict[str, Any] = {}
    keys = set(saved.keys()) | set(incoming.keys())
    for key in keys:
        saved_val = saved.get(key)
        incoming_val = incoming.get(key)
        if isinstance(saved_val, dict) and isinstance(incoming_val, dict):
            sub: dict[str, Any] = {**saved_val, **{k: v for k, v in incoming_val.items() if v is not None}}
            merged[key] = sub
        elif incoming_val is not None:
            merged[key] = incoming_val
        else:
            merged[key] = saved_val
    return merged
