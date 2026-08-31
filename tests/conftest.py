"""Shared fixtures for tests that need a data directory and a catalog database.

Built in stage 0 on purpose: stage 2 moves ~320 call sites onto SQLite, and that
is only tractable if the tests already have one place to get a throwaway data
directory and a migrated catalog from.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Callable

import pytest

from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.storage import db


@pytest.fixture(autouse=True)
def _reset_catalog_connections():
    """Drop the per-process catalog connections between tests.

    catalog_sync caches one connection per data directory, and each test builds a
    new tmp_path. Without this the cache and its file handles accumulate across
    the suite, and a test can see a connection opened against another test's
    directory.
    """
    db.close_process_connections()
    yield
    db.close_process_connections()


@pytest.fixture
def catalog_data_dir(tmp_path: Path) -> Path:
    return tmp_path / "data"


@pytest.fixture
def catalog_settings(catalog_data_dir: Path) -> ApiSettings:
    settings = ApiSettings(
        data_dir=catalog_data_dir,
        incoming_dir=catalog_data_dir / "incoming",
        processed_dir=catalog_data_dir / "processed",
        state_dir=catalog_data_dir / "state",
        events_dir=catalog_data_dir / "events",
        sessions_dir=catalog_data_dir / "sessions",
        datasets_dir=catalog_data_dir / "datasets",
    )
    settings.ensure_directories()
    return settings


@pytest.fixture
def catalog_conn(catalog_settings: ApiSettings) -> sqlite3.Connection:
    """The process connection, not a second one.

    Opening a private connection here deadlocks against the one the code under
    test uses: the second waits for a write lock the first is holding, and no
    busy timeout can break it. One connection per process is the rule the code
    follows, so the tests have to follow it too.
    """
    return db.catalog_for_process(catalog_settings.data_dir)


@pytest.fixture
def write_take(catalog_settings: ApiSettings) -> Callable[..., Path]:
    """Create an incoming take, optionally with a processed result."""

    def _write(
        take_id: str,
        *,
        created_at: str = "2026-05-19T10:00:00Z",
        session_id: str = "session_live",
        metadata: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
        done: bool = False,
    ) -> Path:
        take_dir = catalog_settings.incoming_dir / take_id
        take_dir.mkdir(parents=True, exist_ok=True)
        (take_dir / "rgb.png").write_bytes(b"fake")
        payload = {
            "take_id": take_id,
            "created_at": created_at,
            "session_id": session_id,
            "files": {"rgb": "rgb.png"},
        }
        payload.update(metadata or {})
        (take_dir / "metadata.json").write_text(json.dumps(payload), encoding="utf-8")
        (take_dir / "READY").touch()

        if result is not None or done:
            processed_dir = catalog_settings.processed_dir / take_id
            processed_dir.mkdir(parents=True, exist_ok=True)
            (processed_dir / "result.json").write_text(
                json.dumps(result or {"status": "success", "summary": {"decision": "accept"}}),
                encoding="utf-8",
            )
            if done:
                (processed_dir / "DONE").touch()
        return take_dir

    return _write
