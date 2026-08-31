"""Where dataset documents live.

DatasetService used to build a path and call ``_read_json`` / ``_write_json`` in
28 different methods. Stage 2 moves the source of truth into SQLite, and doing
that per method would have meant 28 divergent rewrites plus 28 fallbacks. So the
path handling is pulled behind one seam instead: every method asks for a document
by what it is — a dataset, a session, a take's sidecar — and the backend decides
whether that means a file or a row.

Two backends, chosen by SENSOR_STUDIO_INDEX:

``SqlDocuments``          the source of truth. Stores each document verbatim in a
                          payload_json column, with the table's other columns as
                          a queryable projection, so a read returns exactly what
                          was written and no field can be lost to a schema that
                          has not caught up.
``FilesystemDocuments``   the previous layout, kept reachable for one release.

Membership resolution is the clearest gain. On files it walks every dataset and
every session looking for a directory, which is what made the take listing scale
with datasets x sessions and what let a take end up with two sidecars. On rows it
is a primary key lookup that cannot return two winners.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Protocol


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


class Documents(Protocol):
    """The document operations DatasetService needs."""

    def list_datasets(self) -> list[dict[str, Any]]: ...
    def read_dataset(self, dataset_id: str) -> dict[str, Any] | None: ...
    def write_dataset(self, dataset_id: str, payload: dict[str, Any]) -> None: ...

    def list_sessions(self, dataset_id: str | None = None) -> list[dict[str, Any]]: ...
    def read_session(self, dataset_id: str, session_id: str) -> dict[str, Any] | None: ...
    def write_session(self, dataset_id: str, session_id: str, payload: dict[str, Any]) -> None: ...

    def read_take(self, dataset_id: str, session_id: str, take_id: str) -> dict[str, Any] | None: ...
    def write_take(self, dataset_id: str, session_id: str, take_id: str, payload: dict[str, Any]) -> None: ...
    def delete_take(self, dataset_id: str, session_id: str, take_id: str) -> bool: ...
    def take_memberships(self, take_id: str) -> list[tuple[str, str]]: ...
    def iter_dataset_takes(
        self, dataset_id: str, session_id: str | None = None
    ) -> list[tuple[str, str, dict[str, Any]]]: ...

    def list_ml_sets(self, dataset_id: str | None = None) -> list[dict[str, Any]]: ...
    def read_ml_set(self, dataset_id: str, ml_set_id: str) -> dict[str, Any] | None: ...
    def write_ml_set(self, dataset_id: str, ml_set_id: str, payload: dict[str, Any]) -> None: ...
    def read_memberships(self, dataset_id: str, ml_set_id: str) -> list[dict[str, Any]]: ...
    def write_memberships(
        self, dataset_id: str, ml_set_id: str, rows: list[dict[str, Any]]
    ) -> None: ...

    def list_physical_objects(self, dataset_id: str) -> list[dict[str, Any]]: ...
    def read_physical_object(self, dataset_id: str, object_id: str) -> dict[str, Any] | None: ...
    def write_physical_object(
        self, dataset_id: str, object_id: str, payload: dict[str, Any]
    ) -> None: ...


class FilesystemDocuments:
    """The layout under data/datasets/. Unchanged behaviour, extracted."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.datasets_dir = self.data_dir / "datasets"
        self.datasets_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ paths

    def _dataset_dir(self, dataset_id: str) -> Path:
        return self.datasets_dir / f"dataset_{dataset_id}"

    def _session_dir(self, dataset_id: str, session_id: str) -> Path:
        return self._dataset_dir(dataset_id) / "sessions" / f"session_{session_id}"

    def _ml_set_dir(self, dataset_id: str, ml_set_id: str) -> Path:
        return self._dataset_dir(dataset_id) / "ml_sets" / f"ml_set_{ml_set_id}"

    def _take_dir(self, dataset_id: str, session_id: str, take_id: str) -> Path:
        return self._session_dir(dataset_id, session_id) / "takes" / take_id

    # --------------------------------------------------------------- datasets

    def list_datasets(self) -> list[dict[str, Any]]:
        items = [
            payload
            for folder in self.datasets_dir.glob("dataset_*")
            if folder.is_dir() and (payload := _read_json(folder / "dataset.json"))
        ]
        return sorted(items, key=lambda item: str(item.get("created_at") or ""), reverse=True)

    def read_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        return _read_json(self._dataset_dir(dataset_id) / "dataset.json")

    def write_dataset(self, dataset_id: str, payload: dict[str, Any]) -> None:
        _write_json(self._dataset_dir(dataset_id) / "dataset.json", payload)

    # --------------------------------------------------------------- sessions

    def list_sessions(self, dataset_id: str | None = None) -> list[dict[str, Any]]:
        dataset_ids = [dataset_id] if dataset_id else [str(item.get("id") or "") for item in self.list_datasets()]
        sessions: list[dict[str, Any]] = []
        for did in dataset_ids:
            if not did:
                continue
            sessions_dir = self._dataset_dir(did) / "sessions"
            if not sessions_dir.is_dir():
                continue
            for folder in sessions_dir.glob("session_*"):
                payload = _read_json(folder / "session.json")
                if payload:
                    sessions.append(payload)
        return sorted(sessions, key=lambda item: str(item.get("created_at") or ""), reverse=True)

    def read_session(self, dataset_id: str, session_id: str) -> dict[str, Any] | None:
        return _read_json(self._session_dir(dataset_id, session_id) / "session.json")

    def write_session(self, dataset_id: str, session_id: str, payload: dict[str, Any]) -> None:
        _write_json(self._session_dir(dataset_id, session_id) / "session.json", payload)

    # ------------------------------------------------------------------ takes

    def read_take(self, dataset_id: str, session_id: str, take_id: str) -> dict[str, Any] | None:
        return _read_json(self._take_dir(dataset_id, session_id, take_id) / "metadata.json")

    def write_take(self, dataset_id: str, session_id: str, take_id: str, payload: dict[str, Any]) -> None:
        _write_json(self._take_dir(dataset_id, session_id, take_id) / "metadata.json", payload)

    def delete_take(self, dataset_id: str, session_id: str, take_id: str) -> bool:
        target = self._take_dir(dataset_id, session_id, take_id)
        if not target.exists():
            return False
        shutil.rmtree(target, ignore_errors=True)
        return True

    def take_memberships(self, take_id: str) -> list[tuple[str, str]]:
        rows: list[tuple[str, str]] = []
        for dataset in self.list_datasets():
            did = str(dataset.get("id") or "")
            if not did:
                continue
            sessions_dir = self._dataset_dir(did) / "sessions"
            if not sessions_dir.is_dir():
                continue
            for folder in sessions_dir.glob("session_*"):
                sid = folder.name.replace("session_", "", 1)
                if (folder / "takes" / take_id).is_dir():
                    rows.append((did, sid))
        return rows

    def iter_dataset_takes(
        self, dataset_id: str, session_id: str | None = None
    ) -> list[tuple[str, str, dict[str, Any]]]:
        rows: list[tuple[str, str, dict[str, Any]]] = []
        sessions_dir = self._dataset_dir(dataset_id) / "sessions"
        if not sessions_dir.is_dir():
            return rows
        for folder in sorted(sessions_dir.glob("session_*")):
            sid = folder.name.replace("session_", "", 1)
            if session_id and sid != session_id:
                continue
            takes_dir = folder / "takes"
            if not takes_dir.is_dir():
                continue
            for take_dir in sorted(takes_dir.iterdir()):
                payload = _read_json(take_dir / "metadata.json")
                if payload:
                    rows.append((sid, take_dir.name, payload))
        return rows

    # ---------------------------------------------------------------- ml sets

    def list_ml_sets(self, dataset_id: str | None = None) -> list[dict[str, Any]]:
        dataset_ids = [dataset_id] if dataset_id else [str(item.get("id") or "") for item in self.list_datasets()]
        items: list[dict[str, Any]] = []
        for did in dataset_ids:
            if not did:
                continue
            ml_sets_dir = self._dataset_dir(did) / "ml_sets"
            if not ml_sets_dir.is_dir():
                continue
            for folder in ml_sets_dir.glob("ml_set_*"):
                payload = _read_json(folder / "ml_set.json")
                if payload:
                    items.append(payload)
        return sorted(items, key=lambda item: str(item.get("created_at") or ""), reverse=True)

    def read_ml_set(self, dataset_id: str, ml_set_id: str) -> dict[str, Any] | None:
        return _read_json(self._ml_set_dir(dataset_id, ml_set_id) / "ml_set.json")

    def write_ml_set(self, dataset_id: str, ml_set_id: str, payload: dict[str, Any]) -> None:
        _write_json(self._ml_set_dir(dataset_id, ml_set_id) / "ml_set.json", payload)

    def read_memberships(self, dataset_id: str, ml_set_id: str) -> list[dict[str, Any]]:
        payload = _read_json(self._ml_set_dir(dataset_id, ml_set_id) / "memberships.json") or {}
        rows = payload.get("memberships")
        return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []

    def write_memberships(self, dataset_id: str, ml_set_id: str, rows: list[dict[str, Any]]) -> None:
        _write_json(
            self._ml_set_dir(dataset_id, ml_set_id) / "memberships.json", {"memberships": rows}
        )

    # ------------------------------------------------------- physical objects

    def list_physical_objects(self, dataset_id: str) -> list[dict[str, Any]]:
        objects_dir = self._dataset_dir(dataset_id) / "physical_objects"
        if not objects_dir.is_dir():
            return []
        items = [
            payload
            for folder in sorted(objects_dir.glob("physical_object_*"))
            if (payload := _read_json(folder / "physical_object.json"))
        ]
        return items

    def read_physical_object(self, dataset_id: str, object_id: str) -> dict[str, Any] | None:
        return _read_json(
            self._dataset_dir(dataset_id)
            / "physical_objects"
            / f"physical_object_{object_id}"
            / "physical_object.json"
        )

    def write_physical_object(self, dataset_id: str, object_id: str, payload: dict[str, Any]) -> None:
        _write_json(
            self._dataset_dir(dataset_id)
            / "physical_objects"
            / f"physical_object_{object_id}"
            / "physical_object.json",
            payload,
        )
