from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, AsyncIterator

from vision_3d_acquisition.api.filesystem import latest_take, read_json
from vision_3d_acquisition.api.settings import ApiSettings


def format_sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def read_last_event(events_file: Path, offset: int) -> tuple[int, list[dict[str, Any]]]:
    if not events_file.is_file():
        return offset, []
    size = events_file.stat().st_size
    if size < offset:
        offset = 0
    events: list[dict[str, Any]] = []
    with events_file.open("r", encoding="utf-8") as handle:
        handle.seek(offset)
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                events.append(value)
        offset = handle.tell()
    return offset, events


async def stream_events(settings: ApiSettings) -> AsyncIterator[str]:
    events_file = settings.events_dir / "events.jsonl"
    offset = events_file.stat().st_size if events_file.is_file() else 0
    last_latest_payload: dict[str, Any] | None = None
    last_runtime_payload: dict[str, Any] | None = None
    yield format_sse("connected", {"status": "ok"})

    while True:
        latest_payload = read_json(settings.state_dir / "latest.json")
        if latest_payload is None:
            latest = latest_take(settings)
            latest_payload = latest.model_dump() if latest else {"take_id": None}
        if latest_payload != last_latest_payload:
            yield format_sse("latest", latest_payload)
            last_latest_payload = latest_payload
        runtime_payload = read_json(settings.state_dir / "runtime.json") or {}
        if runtime_payload != last_runtime_payload:
            yield format_sse("runtime", runtime_payload)
            last_runtime_payload = runtime_payload

        offset, events = read_last_event(events_file, offset)
        for event in events:
            yield format_sse(str(event.get("event", "update")), event)

        await asyncio.sleep(1.0)
