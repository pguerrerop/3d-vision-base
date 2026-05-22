from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import RLock
from typing import Any


def _now() -> datetime:
    return datetime.now(UTC)


def _now_iso() -> str:
    return _now().isoformat()


class TakeProcessingQueue:
    def __init__(self, data_dir: Path) -> None:
        self.root = data_dir / "runtime" / "queues"
        self.pending_path = self.root / "pending_processing.jsonl"
        self.claims_path = self.root / "processing_claims.json"
        self.completed_path = self.root / "completed_processing.jsonl"
        self.failed_path = self.root / "failed_processing.jsonl"
        self._lock = RLock()
        self.root.mkdir(parents=True, exist_ok=True)

    def enqueue(self, entry: dict[str, Any]) -> dict[str, Any] | None:
        take_id = str(entry.get("take_id") or "")
        if not take_id:
            return None
        with self._lock:
            if self._is_terminal_take(take_id) or self._is_claimed(take_id):
                return None
            for row in reversed(self._read_jsonl(self.pending_path)):
                if str(row.get("take_id") or "") == take_id and str(row.get("status") or "pending") in {"pending", "retry_wait"}:
                    return None
            payload = {
                "take_id": take_id,
                "source_id": str(entry.get("source_id") or ""),
                "modality": str(entry.get("modality") or ""),
                "modality_family": str(entry.get("modality_family") or ""),
                "purpose": str(entry.get("purpose") or "acquisition_inspection"),
                "acquisition_group_id": entry.get("acquisition_group_id"),
                "frameset_id": entry.get("frameset_id"),
                "capture_timestamp": entry.get("capture_timestamp"),
                "fusion_readiness": {
                    "acquisition_group_id": entry.get("acquisition_group_id"),
                    "frameset_id": entry.get("frameset_id"),
                    "capture_timestamp": entry.get("capture_timestamp"),
                },
                "status": "pending",
                "retry_count": int(entry.get("retry_count") or 0),
                "max_retries": int(entry.get("max_retries") or 3),
                "next_retry_at": entry.get("next_retry_at"),
                "last_error": entry.get("last_error"),
                "enqueued_at": _now_iso(),
                "metadata": dict(entry.get("metadata") or {}),
            }
            self._append_jsonl(self.pending_path, payload)
            return payload

    def claim_next(
        self,
        *,
        worker_id: str,
        modality: str,
        purpose: str = "acquisition_inspection",
        claim_timeout_sec: float = 90.0,
    ) -> dict[str, Any] | None:
        with self._lock:
            self.recover_stale_claims(claim_timeout_sec=claim_timeout_sec)
            claims = self._read_json(self.claims_path)
            for row in self._read_jsonl(self.pending_path):
                if str(row.get("status") or "") not in {"pending", "retry_wait"}:
                    continue
                if str(row.get("modality") or "") != modality:
                    continue
                if str(row.get("purpose") or "") != purpose:
                    continue
                take_id = str(row.get("take_id") or "")
                if not take_id or take_id in claims or self._is_terminal_take(take_id):
                    continue
                next_retry_at = str(row.get("next_retry_at") or "")
                if next_retry_at and _now() < _parse_iso(next_retry_at):
                    continue
                claim = {
                    "take_id": take_id,
                    "worker_id": worker_id,
                    "claimed_at": _now_iso(),
                    "claim_timeout_sec": float(claim_timeout_sec),
                    "modality": modality,
                    "purpose": purpose,
                }
                claims[take_id] = claim
                self._write_json(self.claims_path, claims)
                return row
            return None

    def release_claim(self, take_id: str) -> None:
        with self._lock:
            claims = self._read_json(self.claims_path)
            if take_id in claims:
                claims.pop(take_id, None)
                self._write_json(self.claims_path, claims)

    def mark_completed(self, take_id: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self.release_claim(take_id)
            row = {
                "take_id": take_id,
                "status": "completed",
                "completed_at": _now_iso(),
                **dict(payload),
            }
            self._append_jsonl(self.completed_path, row)

    def mark_failed(self, take_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self.release_claim(take_id)
            pending = self._latest_pending(take_id) or {}
            retry_count = int(pending.get("retry_count") or 0) + 1
            max_retries = int(pending.get("max_retries") or 3)
            error_message = str(payload.get("error") or payload.get("warning") or "processing_failed")
            failed = {
                "take_id": take_id,
                "status": "failed",
                "failed_at": _now_iso(),
                "retry_count": retry_count,
                "max_retries": max_retries,
                "error": error_message,
                **dict(payload),
            }
            self._append_jsonl(self.failed_path, failed)
            if retry_count <= max_retries:
                delay = min(60.0, 2.0 ** max(0, retry_count - 1))
                retry_entry = dict(pending) if pending else {"take_id": take_id}
                retry_entry.update(
                    {
                        "status": "retry_wait",
                        "retry_count": retry_count,
                        "max_retries": max_retries,
                        "last_error": error_message,
                        "next_retry_at": (_now() + timedelta(seconds=delay)).isoformat(),
                        "enqueued_at": _now_iso(),
                    }
                )
                self._append_jsonl(self.pending_path, retry_entry)
                return {"retried": True, "retry_count": retry_count, "next_retry_at": retry_entry["next_retry_at"]}
            return {"retried": False, "retry_count": retry_count}

    def recover_stale_claims(self, *, claim_timeout_sec: float = 90.0) -> list[str]:
        with self._lock:
            claims = self._read_json(self.claims_path)
            if not claims:
                return []
            now = _now()
            stale: list[str] = []
            for take_id, claim in list(claims.items()):
                claimed_at = _parse_iso(str((claim or {}).get("claimed_at") or ""))
                timeout = float((claim or {}).get("claim_timeout_sec") or claim_timeout_sec)
                if now - claimed_at > timedelta(seconds=timeout):
                    stale.append(take_id)
                    claims.pop(take_id, None)
            if stale:
                self._write_json(self.claims_path, claims)
            return stale

    def queue_status(self) -> dict[str, Any]:
        with self._lock:
            pending_all = self._read_jsonl(self.pending_path)
            pending = [row for row in pending_all if str(row.get("status") or "") in {"pending", "retry_wait"}]
            claims = self._read_json(self.claims_path)
            return {
                "pending_count": len(pending),
                "active_claims_count": len(claims),
                "completed_count": len(self._read_jsonl(self.completed_path)),
                "failed_count": len(self._read_jsonl(self.failed_path)),
                "active_claims": list(claims.values()),
            }

    def queue_depth_for(self, *, modality: str, purpose: str = "acquisition_inspection") -> int:
        with self._lock:
            depth = 0
            for row in self._read_jsonl(self.pending_path):
                if str(row.get("status") or "") not in {"pending", "retry_wait"}:
                    continue
                if str(row.get("modality") or "") == modality and str(row.get("purpose") or "") == purpose:
                    next_retry_at = str(row.get("next_retry_at") or "")
                    if not next_retry_at or _now() >= _parse_iso(next_retry_at):
                        depth += 1
            return depth

    def _is_claimed(self, take_id: str) -> bool:
        return take_id in self._read_json(self.claims_path)

    def _is_terminal_take(self, take_id: str) -> bool:
        for row in reversed(self._read_jsonl(self.completed_path)):
            if str(row.get("take_id") or "") == take_id:
                return True
        pending = self._latest_pending(take_id)
        if pending and int(pending.get("retry_count") or 0) > int(pending.get("max_retries") or 3):
            return True
        return False

    def _latest_pending(self, take_id: str) -> dict[str, Any] | None:
        for row in reversed(self._read_jsonl(self.pending_path)):
            if str(row.get("take_id") or "") == take_id:
                return row
        return None

    @staticmethod
    def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
            except Exception:
                continue
            if isinstance(value, dict):
                rows.append(value)
        return rows

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _parse_iso(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return datetime.fromtimestamp(0, tz=UTC)
