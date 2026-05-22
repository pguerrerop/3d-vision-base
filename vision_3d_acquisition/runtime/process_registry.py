from __future__ import annotations

import atexit
import json
import socket
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


class RuntimeProcessRegistry:
    def __init__(self, data_dir: Path, *, stale_after_seconds: float = 15.0) -> None:
        self.data_dir = data_dir
        self.registry_dir = data_dir / "runtime" / "processes"
        self.stale_after_seconds = max(1.0, float(stale_after_seconds))
        self._lock = threading.RLock()

    def process_path(self, process_name: str) -> Path:
        return self.registry_dir / f"{process_name}.json"

    def register(
        self,
        *,
        process_name: str,
        runtime_role: str,
        pid: int,
        version: str = "poc",
        status: str = "running",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = _now_iso()
        payload = {
            "process_name": process_name,
            "runtime_role": runtime_role,
            "pid": int(pid),
            "status": status,
            "started_at": now,
            "last_heartbeat": now,
            "host": socket.gethostname(),
            "version": version,
            "metadata": dict(metadata or {}),
            "updated_at": now,
        }
        with self._lock:
            _write_json_atomic(self.process_path(process_name), payload)
        return payload

    def heartbeat(
        self,
        process_name: str,
        *,
        status: str = "running",
        counters: dict[str, Any] | None = None,
        metadata_patch: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            path = self.process_path(process_name)
            payload = _read_json(path) or {"process_name": process_name, "started_at": _now_iso()}
            payload["last_heartbeat"] = _now_iso()
            payload["status"] = status
            payload["updated_at"] = payload["last_heartbeat"]
            if counters:
                existing = payload.get("counters") if isinstance(payload.get("counters"), dict) else {}
                merged = dict(existing)
                merged.update(counters)
                payload["counters"] = merged
            if metadata_patch:
                existing_meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
                merged_meta = dict(existing_meta)
                merged_meta.update(metadata_patch)
                payload["metadata"] = merged_meta
            _write_json_atomic(path, payload)
            return payload

    def set_status(self, process_name: str, status: str, *, message: str | None = None) -> dict[str, Any]:
        patch: dict[str, Any] = {}
        if message:
            patch["message"] = message
        return self.heartbeat(process_name, status=status, metadata_patch=patch)

    def stop(self, process_name: str, *, remove_file: bool = False) -> None:
        with self._lock:
            path = self.process_path(process_name)
            if remove_file:
                path.unlink(missing_ok=True)
                return
            payload = _read_json(path) or {"process_name": process_name, "started_at": _now_iso()}
            now = _now_iso()
            payload["status"] = "stopped"
            payload["updated_at"] = now
            payload["last_heartbeat"] = now
            payload["stopped_at"] = now
            _write_json_atomic(path, payload)

    def list_processes(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if not self.registry_dir.is_dir():
            return rows
        now = datetime.now(UTC)
        for path in sorted(self.registry_dir.glob("*.json"), key=lambda item: item.name):
            payload = _read_json(path)
            if not isinstance(payload, dict):
                continue
            last = str(payload.get("last_heartbeat") or "")
            is_stale = False
            if last:
                try:
                    ts = datetime.fromisoformat(last.replace("Z", "+00:00"))
                    is_stale = (now - ts).total_seconds() > self.stale_after_seconds
                except Exception:
                    is_stale = True
            status = str(payload.get("status") or "running")
            rendered = "stale" if is_stale and status == "running" else status
            rows.append(
                {
                    "process_name": str(payload.get("process_name") or path.stem),
                    "runtime_role": str(payload.get("runtime_role") or "service"),
                    "status": rendered,
                    "pid": payload.get("pid"),
                    "last_heartbeat": payload.get("last_heartbeat"),
                    "started_at": payload.get("started_at"),
                    "host": payload.get("host"),
                    "version": payload.get("version"),
                    "is_stale": is_stale,
                    "metadata": payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
                    "counters": payload.get("counters") if isinstance(payload.get("counters"), dict) else {},
                }
            )
        return rows


class ProcessHeartbeat:
    def __init__(
        self,
        registry: RuntimeProcessRegistry,
        *,
        process_name: str,
        runtime_role: str,
        pid: int,
        version: str = "poc",
        interval_seconds: float = 3.0,
        base_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.registry = registry
        self.process_name = process_name
        self.runtime_role = runtime_role
        self.pid = int(pid)
        self.version = version
        self.interval_seconds = max(1.0, float(interval_seconds))
        self.base_metadata = dict(base_metadata or {})
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_emit = 0.0
        self._lock = threading.RLock()

    def start(self) -> None:
        self.registry.register(
            process_name=self.process_name,
            runtime_role=self.runtime_role,
            pid=self.pid,
            version=self.version,
            status="running",
            metadata=self.base_metadata,
        )
        self._thread = threading.Thread(target=self._loop, name=f"hb_{self.process_name}", daemon=True)
        self._thread.start()
        atexit.register(self._atexit_stop)

    def pulse(self, *, counters: dict[str, Any] | None = None, metadata_patch: dict[str, Any] | None = None) -> None:
        with self._lock:
            self.registry.heartbeat(
                self.process_name,
                status="running",
                counters=counters,
                metadata_patch=metadata_patch,
            )
            self._last_emit = time.monotonic()

    def set_failed(self, message: str) -> None:
        self.registry.set_status(self.process_name, "failed", message=message)

    def stop(self, *, status: str = "stopped") -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.registry.stop(self.process_name, remove_file=False if status == "stopped" else True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            now = time.monotonic()
            if now - self._last_emit >= self.interval_seconds:
                self.pulse()
            self._stop.wait(timeout=0.5)

    def _atexit_stop(self) -> None:
        try:
            self.stop(status="stopped")
        except Exception:
            pass

