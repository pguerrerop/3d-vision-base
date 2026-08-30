"""Build the SQLite catalog from the filesystem.

The indexer is the only writer in stage 0, and it is deliberately dumb: it reads
the same JSON files the API reads today and projects them into tables. The take
projection reuses :func:`get_take_summary` rather than reimplementing it, so the
indexed row is the API's own answer, computed once at index time instead of on
every request.

Two modes:

``full``   Reproject every take. Roughly the cost of one ``/api/takes`` call.
``stale``  Reproject only takes whose inputs changed. The authoritative tables
           are always reimported in full — that is ~1200 small JSON files and
           costs little — and only ``take_index`` takes the shortcut.

A take is stale when any of its inputs moved: the ``metadata.json`` or
``result.json`` stat, the acquisition-processing status file, the mtime of the
containing directories (which covers markers and assets appearing), the hash of
everything the summary reads outside those files, or the projection version.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from vision_3d_acquisition.acquisition.processing_status import status_path as acquisition_status_path
from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.storage import dataset_store, db

# Bump when the shape of summary_json or of any take_index column changes.
# The indexer treats every row with a lower value as stale.
#   3: summary_json no longer embeds acquisition_processing_status, which was
#      159 KB of the 161 KB a row occupied. Callers that want it ask for it.
PROJECTION_VERSION = 3

LABEL_KINDS = {
    "tags": "tag",
    "semantic_labels": "semantic",
    "superclass_labels": "superclass",
    "categories": "category",
    "labels": "label",
}


@dataclass
class ReindexReport:
    mode: str = "stale"
    datasets: int = 0
    sessions: int = 0
    take_metadata: int = 0
    labels: int = 0
    object_annotations: int = 0
    ml_sets: int = 0
    ml_set_memberships: int = 0
    physical_objects: int = 0
    take_metadata_conflicts: int = 0
    process_runs: int = 0
    takes_on_disk: int = 0
    takes_projected: int = 0
    takes_unchanged: int = 0
    takes_removed: int = 0
    take_failures: list[str] = field(default_factory=list)
    duration_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class CatalogIndexer:
    def __init__(self, settings: ApiSettings, conn: sqlite3.Connection | None = None) -> None:
        self.settings = settings
        self.data_dir = Path(settings.data_dir)
        # The shared per-process connection, not a fresh one. Two connections in
        # one process deadlock against each other: the second waits on a write
        # lock the first is holding, and no timeout can resolve it.
        self.conn = conn if conn is not None else db.catalog_for_process(self.data_dir)

    # ------------------------------------------------------------------ public

    def rebuild(self, *, full: bool = False) -> ReindexReport:
        started = time.perf_counter()
        report = ReindexReport(mode="full" if full else "stale")
        now = _now_iso()

        from vision_3d_acquisition.storage import catalog_sync

        catalog_sync.ensure_documents_migrated(self.data_dir)

        with db.transaction(self.conn):
            dependencies = self._import_authoritative(report)
            self._import_process_runs(report, dependencies)
            self._project_takes(report, dependencies, full=full, indexed_at=now)

            db.write_meta(self.conn, "projection_version", str(PROJECTION_VERSION))
            db.write_meta(self.conn, "last_stale_scan_at", now)
            if full:
                db.write_meta(self.conn, "last_full_scan_at", now)

        if full:
            self._export_runs_mirror()
            # A full rebuild rewrites every row, and SQLite keeps the freed pages
            # rather than returning them. Dropping the embedded status field left
            # 106 MB of them behind; reclaiming took 65 ms.
            self.conn.execute("VACUUM")

        report.duration_ms = round((time.perf_counter() - started) * 1000, 1)
        return report

    def _export_runs_mirror(self) -> None:
        """Refresh the read-only runs.json. Nothing reads it; it is a fallback."""
        from vision_3d_acquisition.processing.status_index import index_path
        from vision_3d_acquisition.storage import run_store

        try:
            run_store.export_json_mirror(self.data_dir, index_path(self.data_dir))
        except OSError:
            pass

    def refresh_take(self, take_id: str) -> bool:
        """Reproject one take and its sidecar. Returns False if it is gone.

        The push path. Verifying staleness on every read was measured at ~290 ms
        for 753 takes — the authoritative reimport dominates — so writers tell the
        index what changed instead of the index going looking.

        Everything is gathered before the transaction opens. Building the summary
        reads a good deal of the filesystem, and holding the single write lock
        across that starves the other writers: with four processes appending runs
        it produced "database is locked" outright.
        """
        from vision_3d_acquisition.api.filesystem import get_take_summary

        if take_id not in self._take_ids_on_disk():
            with db.transaction(self.conn):
                self.conn.execute("DELETE FROM take_index WHERE take_id = ?", (take_id,))
            return False

        dependency = self._take_dependencies(take_id)
        summary = get_take_summary(
            self.settings, take_id, include_acquisition_processing_status=False
        )
        acquisition = _read_json(self.settings.incoming_dir / take_id / "metadata.json") or {}
        fingerprint = self._take_fingerprint(take_id, dependency)

        with db.transaction(self.conn):
            self._write_take_row(summary, fingerprint, acquisition, indexed_at=_now_iso())
        return True

    def _take_dependencies(self, take_id: str) -> dict[str, Any]:
        """The summary inputs that live outside the take's own directory.

        Read only. take_metadata and its labels are authoritative as of stage 2
        and belong to DatasetService — a refresh of the derived projection must
        not delete and rewrite them, which is what made a metadata edit vanish
        the moment it pushed.
        """
        from vision_3d_acquisition.datasets import DatasetService

        service = DatasetService(self.data_dir)
        dependency: dict[str, Any] = {"take": None, "dataset": None, "session": None, "runs": []}

        memberships = service.resolve_all_take_memberships(take_id)
        if memberships:
            dataset_id, session_id = memberships[0]
            dependency = {
                "take": service.docs.read_take(dataset_id, session_id, take_id),
                "dataset": service.get_dataset(dataset_id),
                "session": service.get_session(dataset_id, session_id),
                "runs": [],
            }

        dependency["runs"] = self._reimport_runs_for_take(take_id)
        return dependency

    def status(self) -> dict[str, Any]:
        """Counts plus a cheap drift check against the filesystem."""
        counts = {
            table: int(self.conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
            for table in (
                "dataset",
                "experiment_session",
                "take_metadata",
                "take_label",
                "ml_set",
                "ml_set_membership",
                "physical_object",
                "take_index",
                "process_run",
                "take_metadata_conflict",
            )
        }
        on_disk = len(self._take_ids_on_disk())
        return {
            "database": str(db.database_path(self.data_dir)),
            "schema_version": db.schema_version(self.conn),
            "latest_schema_version": db.latest_schema_version(),
            "projection_version": PROJECTION_VERSION,
            "indexed_projection_version": db.read_meta(self.conn, "projection_version"),
            "last_full_scan_at": db.read_meta(self.conn, "last_full_scan_at"),
            "last_stale_scan_at": db.read_meta(self.conn, "last_stale_scan_at"),
            "counts": counts,
            "takes_on_disk": on_disk,
            "drift": on_disk - counts["take_index"],
        }

    def verify(self, *, limit: int | None = None) -> dict[str, Any]:
        """Recompute every indexed summary from disk and diff it against the row.

        This is the safety net the rest of the migration leans on: as long as it
        reports no mismatch, an endpoint reading ``summary_json`` returns exactly
        what it returns today.
        """
        from vision_3d_acquisition.api.filesystem import get_take_summary

        on_disk = self._take_ids_on_disk()
        indexed = [
            row["take_id"]
            for row in self.conn.execute("SELECT take_id FROM take_index ORDER BY take_id")
        ]
        checked = 0
        mismatched: list[dict[str, Any]] = []
        failures: list[str] = []

        for take_id in indexed[: limit or len(indexed)]:
            row = self.conn.execute(
                "SELECT summary_json FROM take_index WHERE take_id = ?", (take_id,)
            ).fetchone()
            try:
                expected = get_take_summary(
                    self.settings, take_id, include_acquisition_processing_status=False
                ).model_dump(mode="json")
            except Exception as error:  # noqa: BLE001
                failures.append(f"{take_id}: {type(error).__name__}: {error}")
                continue
            checked += 1
            stored = json.loads(row["summary_json"])
            if stored != expected:
                differing = sorted(
                    key
                    for key in set(stored) | set(expected)
                    if stored.get(key) != expected.get(key)
                )
                mismatched.append({"take_id": take_id, "fields": differing})

        return {
            "checked": checked,
            "mismatched": mismatched,
            "mismatched_count": len(mismatched),
            "missing_from_index": sorted(on_disk - set(indexed)),
            "stale_in_index": sorted(set(indexed) - on_disk),
            "failures": failures,
            "ok": not mismatched and not failures and on_disk == set(indexed),
        }

    # ------------------------------------------------------------ authoritative

    def _import_authoritative(self, report: ReindexReport) -> dict[str, dict[str, Any]]:
        """Collect the hash inputs a take projection needs.

        Stage 2 made these tables the source of truth, so a rebuild no longer
        clears and reloads them from data/datasets/ — that would delete operator
        work that exists nowhere else. It migrates the files in once, then reads
        the rows.
        """
        from vision_3d_acquisition.storage import catalog_sync

        catalog_sync.ensure_documents_migrated(self.data_dir)

        datasets = {
            row["id"]: _loads(row["payload_json"])
            for row in self.conn.execute("SELECT id, payload_json FROM dataset")
        }
        sessions = {
            (row["dataset_id"], row["id"]): _loads(row["payload_json"])
            for row in self.conn.execute("SELECT dataset_id, id, payload_json FROM experiment_session")
        }
        report.datasets = len(datasets)
        report.sessions = len(sessions)

        dependencies: dict[str, dict[str, Any]] = {}
        for row in self.conn.execute(
            "SELECT take_id, dataset_id, session_id, payload_json FROM take_metadata"
        ):
            take_id = str(row["take_id"])
            dependencies[take_id] = {
                "take": _loads(row["payload_json"]),
                "dataset": datasets.get(row["dataset_id"]),
                "session": sessions.get((row["dataset_id"], row["session_id"])),
            }
            report.take_metadata += 1

        report.labels = int(self.conn.execute("SELECT count(*) FROM take_label").fetchone()[0])
        report.object_annotations = int(
            self.conn.execute("SELECT count(*) FROM object_annotation").fetchone()[0]
        )
        report.ml_sets = int(self.conn.execute("SELECT count(*) FROM ml_set").fetchone()[0])
        report.ml_set_memberships = int(
            self.conn.execute("SELECT count(*) FROM ml_set_membership").fetchone()[0]
        )
        report.physical_objects = int(
            self.conn.execute("SELECT count(*) FROM physical_object").fetchone()[0]
        )
        report.take_metadata_conflicts = int(
            self.conn.execute("SELECT count(*) FROM take_metadata_conflict").fetchone()[0]
        )
        return dependencies

    # ------------------------------------------------------------------- runs

    def _import_process_runs(self, report: ReindexReport, dependencies: dict[str, dict[str, Any]]) -> None:
        """Collect the run history a rebuild needs.

        process_run is authoritative as of stage 2, so a rebuild must not clear
        and reload it the way it does the derived tables — that would delete run
        history that exists nowhere else. It only migrates runs.json in on the
        first run, then reads what is already stored.
        """
        from vision_3d_acquisition.processing.status_index import (
            ensure_runs_migrated,
            normalize_run_entry,
        )
        from vision_3d_acquisition.storage import run_store

        ensure_runs_migrated(self.data_dir, self.conn)
        entries = [
            normalize_run_entry(run_store.row_to_entry(row))
            for row in self.conn.execute("SELECT * FROM process_run")
        ]
        for entry in entries:
            take_id = str(entry.get("take_id") or "")
            if take_id:
                dependencies.setdefault(take_id, {}).setdefault("runs", []).append(entry)
        report.process_runs = len(entries)

    def _reimport_runs_for_take(self, take_id: str) -> list[dict[str, Any]]:
        """Read one take's stored runs. The rows are the source; nothing is reloaded."""
        from vision_3d_acquisition.processing.status_index import normalize_run_entry
        from vision_3d_acquisition.storage import run_store

        return [
            normalize_run_entry(run_store.row_to_entry(row))
            for row in self.conn.execute(
                "SELECT * FROM process_run WHERE take_id = ? ORDER BY created_at, run_id", (take_id,)
            )
        ]

    # ------------------------------------------------------------------ takes

    def _project_takes(
        self,
        report: ReindexReport,
        dependencies: dict[str, dict[str, Any]],
        *,
        full: bool,
        indexed_at: str,
    ) -> None:
        from vision_3d_acquisition.api.filesystem import get_take_summary

        take_ids = self._take_ids_on_disk()
        report.takes_on_disk = len(take_ids)

        stored = {
            row["take_id"]: row
            for row in self.conn.execute(
                "SELECT take_id, metadata_mtime, metadata_size, result_mtime, result_size,"
                " status_mtime, status_size, management_hash, projection_version FROM take_index"
            )
        }

        for take_id in sorted(take_ids):
            fingerprint = self._take_fingerprint(take_id, dependencies.get(take_id))
            previous = stored.get(take_id)
            if not full and previous is not None and _fingerprint_matches(previous, fingerprint):
                report.takes_unchanged += 1
                continue
            try:
                summary = get_take_summary(
                self.settings, take_id, include_acquisition_processing_status=False
            )
            except Exception as error:  # noqa: BLE001 - one bad take must not stop the scan
                report.take_failures.append(f"{take_id}: {type(error).__name__}: {error}")
                continue
            acquisition = _read_json(self.settings.incoming_dir / take_id / "metadata.json") or {}
            self._write_take_row(summary, fingerprint, acquisition, indexed_at=indexed_at)
            report.takes_projected += 1

        removed = set(stored) - take_ids
        if removed:
            self.conn.executemany(
                "DELETE FROM take_index WHERE take_id = ?", [(take_id,) for take_id in sorted(removed)]
            )
            report.takes_removed = len(removed)

    def _write_take_row(
        self,
        summary: Any,
        fingerprint: dict[str, Any],
        acquisition: dict[str, Any],
        *,
        indexed_at: str,
    ) -> None:
        payload = summary.model_dump(mode="json")
        take_id = str(payload["take_id"])
        self.conn.execute(
            "INSERT OR REPLACE INTO take_index(take_id, status, has_ready, has_done, created_at,"
            " processed_at, decision, engine, object_count, ball_count, plane_found,"
            " processing_time_ms, warning_count, calibration_id, frame_count, frameset_id,"
            " acquisition_session_id, processed_class_label, processed_superclass,"
            " latest_run_status, thumbnail_path, summary_json, metadata_mtime, metadata_size,"
            " result_mtime, result_size, status_mtime, status_size, management_hash,"
            " projection_version, indexed_at, acquisition_created_at, acquisition_calibration_id,"
            " metadata_session_id)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                take_id,
                payload["status"],
                int(bool(payload["has_ready"])),
                int(bool(payload["has_done"])),
                payload.get("created_at"),
                None,
                payload.get("decision"),
                payload.get("engine"),
                _as_int(payload.get("object_count")),
                _as_int(payload.get("ball_count")),
                _as_optional_bool(payload.get("plane_found")),
                _as_float(payload.get("processing_time_ms")),
                int(payload.get("warning_count") or 0),
                payload.get("calibration_id"),
                _as_int(payload.get("frame_count")),
                payload.get("frameset_id"),
                payload.get("session_id"),
                payload.get("processed_class_label"),
                payload.get("processed_superclass"),
                payload.get("latest_run_status"),
                payload.get("thumbnail_path"),
                json.dumps(payload, sort_keys=True),
                fingerprint["metadata_mtime"],
                fingerprint["metadata_size"],
                fingerprint["result_mtime"],
                fingerprint["result_size"],
                fingerprint["status_mtime"],
                fingerprint["status_size"],
                fingerprint["management_hash"],
                PROJECTION_VERSION,
                indexed_at,
                acquisition.get("created_at"),
                acquisition.get("calibration_id"),
                acquisition.get("session_id"),
            ),
        )
        self.conn.execute("DELETE FROM take_modality WHERE take_id = ?", (take_id,))
        self.conn.executemany(
            "INSERT OR IGNORE INTO take_modality(take_id, modality) VALUES (?,?)",
            sorted({(take_id, str(item)) for item in (payload.get("modalities") or []) if str(item)}),
        )

    # ---------------------------------------------------------------- internals

    def _datasets_in_resolution_order(self) -> list[dict[str, Any]]:
        """Datasets in the order DatasetService walks them.

        Take membership is resolved by taking the first hit while iterating
        datasets then sessions, so the index has to walk them the same way to
        agree with what the API returns for a take with more than one copy.
        """
        from vision_3d_acquisition.datasets import DatasetService

        return DatasetService(self.data_dir).list_datasets()

    def _take_ids_on_disk(self) -> set[str]:
        take_ids: set[str] = set()
        for root in (self.settings.incoming_dir, self.settings.processed_dir):
            if not root.exists():
                continue
            for child in root.iterdir():
                if child.is_dir() and not child.name.startswith("."):
                    take_ids.add(child.name)
        return take_ids

    def _take_fingerprint(self, take_id: str, dependency: dict[str, Any] | None) -> dict[str, Any]:
        incoming_dir = self.settings.incoming_dir / take_id
        processed_dir = self.settings.processed_dir / take_id
        metadata_mtime, metadata_size = _stat_pair(incoming_dir / "metadata.json", incoming_dir)
        result_mtime, result_size = _stat_pair(processed_dir / "result.json", processed_dir)
        status_mtime, status_size = _stat_pair(acquisition_status_path(self.data_dir, take_id))
        return {
            "metadata_mtime": metadata_mtime,
            "metadata_size": metadata_size,
            "result_mtime": result_mtime,
            "result_size": result_size,
            "status_mtime": status_mtime,
            "status_size": status_size,
            "management_hash": _dependency_hash(dependency),
        }


def _fingerprint_matches(row: sqlite3.Row, fingerprint: dict[str, Any]) -> bool:
    if int(row["projection_version"] or 0) != PROJECTION_VERSION:
        return False
    return all(row[key] == value for key, value in fingerprint.items())


# The run fields the take summary actually reads, via summarize_take_processing
# and latest_run_status. Projecting to exactly these keys lets the full rebuild
# (which has index entries) and refresh_take (which has process_run rows) hash
# the same value, so a pushed refresh does not make the take look stale.
_RUN_HASH_KEYS = ("run_id", "pipeline_family", "status", "created_at", "path", "pipeline_instance_id")


def _canonical_runs(items: Iterable[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return sorted(
        ({key: item.get(key) for key in _RUN_HASH_KEYS} for item in (items or [])),
        key=lambda item: str(item.get("run_id") or ""),
    )


def _dependency_hash(dependency: dict[str, Any] | None) -> str:
    """Hash every summary input that lives outside the take's own directory."""
    payload = {
        "take": (dependency or {}).get("take"),
        "dataset": (dependency or {}).get("dataset"),
        "session": (dependency or {}).get("session"),
        "runs": _canonical_runs((dependency or {}).get("runs")),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stat_pair(path: Path, *directory: Path) -> tuple[float | None, int | None]:
    """Return (mtime, size) for a file, folding in directory mtimes.

    A directory's mtime moves when a file is added or removed inside it, which is
    how a new READY marker, a DONE marker or a fresh debug image invalidates the
    row without the indexer having to stat each one.
    """
    mtime: float | None = None
    size: int | None = None
    try:
        stat = path.stat()
    except OSError:
        pass
    else:
        mtime, size = stat.st_mtime, stat.st_size
    for extra in directory:
        try:
            extra_mtime = extra.stat().st_mtime
        except OSError:
            continue
        mtime = extra_mtime if mtime is None else max(mtime, extra_mtime)
    return (mtime, size)


def _loads(value: str | None) -> dict[str, Any]:
    try:
        payload = json.loads(value or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_optional_bool(value: Any) -> int | None:
    return None if value is None else int(bool(value))


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def reindex(settings: ApiSettings, *, full: bool = False) -> ReindexReport:
    return CatalogIndexer(settings).rebuild(full=full)


def iter_indexed_summaries(conn: sqlite3.Connection) -> Iterable[dict[str, Any]]:
    for row in conn.execute("SELECT summary_json FROM take_index ORDER BY take_id"):
        yield json.loads(row["summary_json"])
