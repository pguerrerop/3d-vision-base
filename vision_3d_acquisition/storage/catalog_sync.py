"""Keep the catalog current from the write path.

Verifying staleness on every read was measured at ~290 ms for 753 takes, because
the sidecar reimport that the freshness hash depends on has to run first. So the
index is pushed to instead: whoever changes a take tells the catalog, and the
read path trusts it.

Every entry point here is best effort by design. The catalog is a derived index
in stage 1 — the filesystem is still the source of truth — so a failed refresh
must never break the write that triggered it. Failures degrade into drift, which
`sensor-studio index status` reports and `index rebuild` repairs.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from vision_3d_acquisition.storage import db

log = logging.getLogger(__name__)

ENV_FLAG = "SENSOR_STUDIO_INDEX"
MIGRATION_KEY = "documents_migrated_at"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


@contextmanager
def _maybe_transaction(conn):
    """Open a transaction, unless the caller already holds one."""
    if conn.in_transaction:
        yield conn
        return
    with db.transaction(conn):
        yield conn


def index_enabled() -> bool:
    """The catalog is on unless explicitly turned off.

    Stage 1 keeps the filesystem path reachable for one release: setting
    SENSOR_STUDIO_INDEX=off reverts reads and writes to it without a rollback.
    """
    return str(os.environ.get(ENV_FLAG, "on")).strip().lower() not in {"off", "0", "false", "no"}


def catalog_available(data_dir: Path) -> bool:
    return index_enabled() and db.database_path(data_dir).is_file()


def catalog_ready_for_reads(data_dir: Path, take_ids_on_disk: int):
    """The connection to serve a listing from, or None to walk the filesystem.

    Existing is not the same as being complete. A push creates the database the
    first time a take's metadata is edited, and answering a listing from that
    would silently return the handful of takes that happen to have been pushed.
    So a listing is served only from an index that has finished a full scan, and
    only while its take count still matches the disk — a count mismatch means
    takes appeared or vanished without a push, and it repairs itself before
    answering rather than serving a stale page.
    """
    if not catalog_available(data_dir):
        return None
    try:
        conn = db.catalog_for_process(Path(data_dir))
        if db.read_meta(conn, "last_full_scan_at") is None:
            return None
        indexed = int(conn.execute("SELECT count(*) FROM take_index").fetchone()[0])
        if indexed != take_ids_on_disk:
            log.info("catalog drift: %s indexed vs %s on disk, refreshing", indexed, take_ids_on_disk)
            _indexer(data_dir).rebuild()
        return conn
    except Exception:  # noqa: BLE001 - a broken index must not take the API down
        log.warning("catalog unavailable for reads, falling back to the filesystem", exc_info=True)
        return None


def ensure_documents_migrated(data_dir: Path) -> dict[str, int]:
    """Import data/datasets/ into the catalog once, then never again.

    Stage 2 makes the tables the source of truth, so this runs exactly one time
    per data directory and is recorded in index_meta. Emptiness would be the
    wrong flag here — unlike process runs, having no datasets is a legitimate
    state, and re-importing over live rows would resurrect deleted documents.
    """
    root = Path(data_dir)
    conn = db.catalog_for_process(root)
    if db.read_meta(conn, MIGRATION_KEY):
        return {}

    from vision_3d_acquisition.datasets.documents import FilesystemDocuments
    from vision_3d_acquisition.storage import dataset_store

    files = FilesystemDocuments(root)
    counts = {"datasets": 0, "sessions": 0, "takes": 0, "ml_sets": 0, "memberships": 0, "objects": 0}
    # The indexer calls this from inside its rebuild transaction; opening a
    # second one on the same connection raises rather than waiting.
    with _maybe_transaction(conn):
        for dataset in files.list_datasets():
            dataset_id = str(dataset.get("id") or "")
            if not dataset_id:
                continue
            dataset_store.project_dataset(conn, dataset_id, dataset)
            counts["datasets"] += 1

            for session in files.list_sessions(dataset_id):
                session_id = str(session.get("id") or "")
                if not session_id:
                    continue
                dataset_store.project_session(conn, dataset_id, session_id, session)
                counts["sessions"] += 1

            # First membership wins, matching what the filesystem resolution
            # surfaced. take_id is a primary key, so the rest cannot be stored.
            seen: set[str] = set()
            for session_id, take_id, payload in files.iter_dataset_takes(dataset_id):
                if take_id in seen:
                    conn.execute(
                        "INSERT OR IGNORE INTO take_metadata_conflict(take_id, dataset_id,"
                        " session_id, path) VALUES (?,?,?,?)",
                        (take_id, dataset_id, session_id, f"{dataset_id}/{session_id}/{take_id}"),
                    )
                    continue
                seen.add(take_id)
                dataset_store.project_take(conn, take_id, dataset_id, session_id, payload)
                counts["takes"] += 1

            for ml_set in files.list_ml_sets(dataset_id):
                ml_set_id = str(ml_set.get("id") or "")
                if not ml_set_id:
                    continue
                dataset_store.project_ml_set(conn, dataset_id, ml_set_id, ml_set)
                counts["ml_sets"] += 1
                for row in files.read_memberships(dataset_id, ml_set_id):
                    dataset_store.project_membership(conn, dataset_id, ml_set_id, row)
                    counts["memberships"] += 1

            for obj in files.list_physical_objects(dataset_id):
                object_id = str(obj.get("physical_object_id") or obj.get("id") or "")
                if not object_id:
                    continue
                dataset_store.project_physical_object(conn, dataset_id, object_id, obj)
                counts["objects"] += 1

        db.write_meta(conn, MIGRATION_KEY, _now_iso())
    log.info("catalog documents migrated: %s", counts)
    return counts


def refresh_take(data_dir: Path, take_id: str) -> None:
    """Reproject one take after its metadata, result or run history changed."""
    if not catalog_available(data_dir) or not take_id:
        return
    try:
        _indexer(data_dir).refresh_take(take_id)
    except Exception:  # noqa: BLE001 - a derived index must not break a write
        log.warning("catalog refresh failed for take %s", take_id, exc_info=True)


def refresh_takes(data_dir: Path, take_ids: list[str]) -> None:
    if not catalog_available(data_dir):
        return
    for take_id in take_ids:
        refresh_take(data_dir, take_id)


def _indexer(data_dir: Path):
    from vision_3d_acquisition.api.settings import ApiSettings
    from vision_3d_acquisition.storage.catalog_index import CatalogIndexer

    root = Path(data_dir)
    settings = ApiSettings(
        data_dir=root,
        incoming_dir=root / "incoming",
        processed_dir=root / "processed",
        state_dir=root / "state",
        events_dir=root / "events",
        sessions_dir=root / "sessions",
        datasets_dir=root / "datasets",
    )
    return CatalogIndexer(settings, db.catalog_for_process(root))
