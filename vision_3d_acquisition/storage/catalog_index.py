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
from vision_3d_acquisition.storage import db

# Bump when the shape of summary_json or of any take_index column changes.
# The indexer treats every row with a lower value as stale.
PROJECTION_VERSION = 1

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
        self.conn = conn if conn is not None else db.open_catalog(self.data_dir)

    # ------------------------------------------------------------------ public

    def rebuild(self, *, full: bool = False) -> ReindexReport:
        started = time.perf_counter()
        report = ReindexReport(mode="full" if full else "stale")
        now = _now_iso()

        with db.transaction(self.conn):
            dependencies = self._import_authoritative(report)
            self._import_process_runs(report, dependencies)
            self._project_takes(report, dependencies, full=full, indexed_at=now)

            db.write_meta(self.conn, "projection_version", str(PROJECTION_VERSION))
            db.write_meta(self.conn, "last_stale_scan_at", now)
            if full:
                db.write_meta(self.conn, "last_full_scan_at", now)

        report.duration_ms = round((time.perf_counter() - started) * 1000, 1)
        return report

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
                expected = get_take_summary(self.settings, take_id).model_dump(mode="json")
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
        """Reimport datasets, sessions, take metadata, ML sets and physical objects.

        Returns, per take id, the inputs to the summary that do not live in the
        take's own directory. The indexer hashes them to decide staleness.
        """
        conn = self.conn
        datasets_dir = self.data_dir / "datasets"

        for table in (
            "object_annotation",
            "take_label",
            "take_metadata_conflict",
            "take_metadata",
            "ml_set_membership",
            "ml_set",
            "physical_object_observation",
            "physical_object",
            "experiment_session",
            "dataset",
        ):
            conn.execute(f"DELETE FROM {table}")

        dataset_rows: dict[str, dict[str, Any]] = {}
        session_rows: dict[tuple[str, str], dict[str, Any]] = {}
        dependencies: dict[str, dict[str, Any]] = {}
        resolved_takes: set[str] = set()

        for payload in self._datasets_in_resolution_order():
            dataset_id = str(payload.get("id") or "")
            if not dataset_id:
                continue
            dataset_dir = datasets_dir / f"dataset_{dataset_id}"
            dataset_rows[dataset_id] = payload
            conn.execute(
                "INSERT OR REPLACE INTO dataset(id, name, description, notes, tags_json,"
                " created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
                (
                    dataset_id,
                    str(payload.get("name") or dataset_id),
                    payload.get("description"),
                    payload.get("notes"),
                    json.dumps(payload.get("tags") or []),
                    payload.get("created_at"),
                    payload.get("updated_at"),
                ),
            )
            report.datasets += 1

            for session_dir in (dataset_dir / "sessions").glob("session_*"):
                session = _read_json(session_dir / "session.json")
                if not session:
                    continue
                session_id = str(session.get("id") or session_dir.name.replace("session_", "", 1))
                session_rows[(dataset_id, session_id)] = session
                conn.execute(
                    "INSERT OR REPLACE INTO experiment_session(dataset_id, id, name, session_type,"
                    " description, notes, tags_json, metadata_json, created_at, updated_at)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        dataset_id,
                        session_id,
                        session.get("name"),
                        str(session.get("session_type") or "engineering"),
                        session.get("description"),
                        session.get("notes"),
                        json.dumps(session.get("tags") or []),
                        json.dumps(session.get("metadata") or {}),
                        session.get("created_at"),
                        session.get("updated_at"),
                    ),
                )
                report.sessions += 1

                for take_dir in sorted((session_dir / "takes").glob("*")):
                    take_payload = _read_json(take_dir / "metadata.json")
                    if not take_payload:
                        continue
                    take_id = str(take_payload.get("take_id") or take_dir.name)
                    if take_id in resolved_takes:
                        # Same take carrying metadata under a second session. The
                        # API shows the first one it finds, so the index does too
                        # and the loser is recorded as drift.
                        conn.execute(
                            "INSERT OR IGNORE INTO take_metadata_conflict(take_id, dataset_id,"
                            " session_id, path) VALUES (?,?,?,?)",
                            (take_id, dataset_id, session_id, str(take_dir / "metadata.json")),
                        )
                        report.take_metadata_conflicts += 1
                        continue
                    resolved_takes.add(take_id)
                    self._insert_take_metadata(take_id, dataset_id, session_id, take_payload)
                    report.take_metadata += 1
                    report.labels += self._insert_take_labels(take_id, take_payload)
                    report.object_annotations += self._insert_object_annotations(take_id, take_payload)
                    dependencies[take_id] = {
                        "take": take_payload,
                        "dataset": dataset_rows.get(dataset_id),
                        "session": session_rows.get((dataset_id, session_id)),
                    }

            for ml_set_dir in sorted((dataset_dir / "ml_sets").glob("ml_set_*")):
                ml_set = _read_json(ml_set_dir / "ml_set.json")
                if not ml_set:
                    continue
                ml_set_id = str(ml_set.get("id") or ml_set_dir.name.replace("ml_set_", "", 1))
                conn.execute(
                    "INSERT OR REPLACE INTO ml_set(dataset_id, id, name, description, task_type,"
                    " notes, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
                    (
                        dataset_id,
                        ml_set_id,
                        ml_set.get("name"),
                        ml_set.get("description"),
                        ml_set.get("task_type"),
                        ml_set.get("notes"),
                        ml_set.get("created_at"),
                        ml_set.get("updated_at"),
                    ),
                )
                report.ml_sets += 1
                report.ml_set_memberships += self._insert_memberships(
                    dataset_id, ml_set_id, _read_json(ml_set_dir / "memberships.json")
                )

            for object_dir in sorted((dataset_dir / "physical_objects").glob("physical_object_*")):
                report.physical_objects += self._insert_physical_object(
                    dataset_id, _read_json(object_dir / "physical_object.json")
                )

        return dependencies

    def _insert_take_metadata(
        self, take_id: str, dataset_id: str, session_id: str, payload: dict[str, Any]
    ) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO take_metadata(take_id, dataset_id, session_id, friendly_name,"
            " notes, operator_notes, session_notes, validation_status, normalized_class,"
            " normalization_version, expected_class, expected_diameter_mm, expected_count,"
            " physical_object_id, acquisition_group_id, reference_type, is_reference,"
            " is_golden_sample, archived, archived_at, archived_reason, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                take_id,
                dataset_id,
                session_id,
                payload.get("friendly_name") or take_id,
                payload.get("notes"),
                payload.get("operator_notes"),
                payload.get("session_notes"),
                str(payload.get("validation_status") or "unreviewed"),
                payload.get("normalized_class"),
                payload.get("normalization_version"),
                payload.get("expected_class"),
                _as_float(payload.get("expected_diameter_mm")),
                _as_int(payload.get("expected_count")),
                payload.get("physical_object_id"),
                payload.get("acquisition_group_id"),
                payload.get("reference_type"),
                int(bool(payload.get("is_reference"))),
                int(bool(payload.get("is_golden_sample"))),
                int(bool(payload.get("archived"))),
                payload.get("archived_at"),
                payload.get("archived_reason"),
                payload.get("created_at"),
                payload.get("updated_at"),
            ),
        )

    def _insert_take_labels(self, take_id: str, payload: dict[str, Any]) -> int:
        rows = {
            (take_id, kind, str(value).strip())
            for source_key, kind in LABEL_KINDS.items()
            for value in (payload.get(source_key) or [])
            if str(value).strip()
        }
        self.conn.executemany(
            "INSERT OR IGNORE INTO take_label(take_id, kind, value) VALUES (?,?,?)", sorted(rows)
        )
        return len(rows)

    def _insert_object_annotations(self, take_id: str, payload: dict[str, Any]) -> int:
        annotations = [item for item in (payload.get("object_annotations") or []) if isinstance(item, dict)]
        for annotation in annotations:
            self.conn.execute(
                "INSERT OR REPLACE INTO object_annotation(take_id, id, source_stage,"
                " source_artifact_id, candidate_id, bbox_json, centroid_json, contour_ref,"
                " labels_json, expected_class, expected_diameter_mm, notes, validation_status,"
                " created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    take_id,
                    str(annotation.get("id") or ""),
                    annotation.get("source_stage"),
                    annotation.get("source_artifact_id"),
                    annotation.get("candidate_id"),
                    json.dumps(annotation.get("bbox")) if annotation.get("bbox") is not None else None,
                    json.dumps(annotation.get("centroid")) if annotation.get("centroid") is not None else None,
                    annotation.get("contour_ref"),
                    json.dumps(annotation.get("labels") or []),
                    annotation.get("expected_class"),
                    _as_float(annotation.get("expected_diameter_mm")),
                    annotation.get("notes"),
                    str(annotation.get("validation_status") or "unreviewed"),
                    annotation.get("created_at"),
                    annotation.get("updated_at"),
                ),
            )
        return len(annotations)

    def _insert_memberships(self, dataset_id: str, ml_set_id: str, payload: dict[str, Any] | None) -> int:
        rows = [item for item in ((payload or {}).get("memberships") or []) if isinstance(item, dict)]
        for row in rows:
            self.conn.execute(
                'INSERT OR REPLACE INTO ml_set_membership(dataset_id, ml_set_id, take_id, split,'
                ' "include", default_trainable, trainable, physical_object_id, expected_label,'
                " expected_class, expected_subclass, raw_label, label_policy, review_required,"
                " normalization_version, source_row, notes, measurements_mm_json,"
                " extra_fields_json, created_at, updated_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    dataset_id,
                    ml_set_id,
                    str(row.get("take_id") or ""),
                    str(row.get("split") or "unassigned"),
                    int(bool(row.get("include", True))),
                    _as_optional_bool(row.get("default_trainable")),
                    _as_optional_bool(row.get("trainable")),
                    row.get("physical_object_id"),
                    row.get("expected_label"),
                    row.get("expected_class"),
                    row.get("expected_subclass"),
                    row.get("raw_label"),
                    row.get("label_policy"),
                    _as_optional_bool(row.get("review_required")),
                    row.get("normalization_version"),
                    row.get("source_row"),
                    row.get("notes"),
                    json.dumps(row.get("measurements_mm") or {}),
                    json.dumps(row.get("extra_fields") or {}),
                    row.get("created_at"),
                    row.get("updated_at"),
                ),
            )
        return len(rows)

    def _insert_physical_object(self, dataset_id: str, payload: dict[str, Any] | None) -> int:
        if not payload:
            return 0
        object_id = str(payload.get("physical_object_id") or payload.get("id") or "")
        if not object_id:
            return 0
        self.conn.execute(
            "INSERT OR REPLACE INTO physical_object(dataset_id, id, raw_operator_label,"
            " normalized_class, superclass, d1_mm, d2_mm, d3_mm, diameter_mean_mm,"
            " diameter_range_mm, annotation_confidence, needs_review, source_type,"
            " source_row_index, notes, tags_json, source_session_ids_json, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                dataset_id,
                object_id,
                payload.get("raw_operator_label"),
                payload.get("normalized_class"),
                payload.get("superclass"),
                _as_float(payload.get("d1_mm")),
                _as_float(payload.get("d2_mm")),
                _as_float(payload.get("d3_mm")),
                _as_float(payload.get("diameter_mean_mm")),
                _as_float(payload.get("diameter_range_mm")),
                payload.get("annotation_confidence"),
                int(bool(payload.get("needs_review"))),
                payload.get("source_type"),
                _as_int(payload.get("source_row_index")),
                payload.get("notes"),
                json.dumps(payload.get("tags") or []),
                json.dumps(payload.get("source_session_ids") or []),
                payload.get("created_at"),
            ),
        )
        observations = {
            (dataset_id, object_id, str(take_id))
            for take_id in (payload.get("observation_take_ids") or [])
            if str(take_id)
        }
        self.conn.executemany(
            "INSERT OR IGNORE INTO physical_object_observation(dataset_id, object_id, take_id)"
            " VALUES (?,?,?)",
            sorted(observations),
        )
        return 1

    # ------------------------------------------------------------------- runs

    def _import_process_runs(self, report: ReindexReport, dependencies: dict[str, dict[str, Any]]) -> None:
        from vision_3d_acquisition.processing.status_index import load_process_entries, normalize_run_entry

        self.conn.execute("DELETE FROM process_run")
        entries = [normalize_run_entry(entry) for entry in load_process_entries(self.data_dir)]
        for entry in entries:
            run_id = str(entry.get("run_id") or "")
            if not run_id:
                continue
            self.conn.execute(
                "INSERT OR REPLACE INTO process_run(run_id, take_id, pipeline_instance_id,"
                " pipeline_id, pipeline_family, status, created_at, path, recipe_id,"
                " recipe_version_id, config_snapshot_hash, source_id, acquisition_group_id,"
                " calibration_profile_id, execution_mode, parent_run_id, parent_take_id,"
                " partial_rerun_plan_id, boundary_stage_id, boundary_unit_id, selected_unit_id,"
                " comparison_id, summary_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    entry.get("take_id"),
                    entry.get("pipeline_instance_id"),
                    entry.get("pipeline_id"),
                    str(entry.get("pipeline_family") or "generic"),
                    str(entry.get("status") or "completed"),
                    str(entry.get("created_at") or ""),
                    entry.get("path"),
                    entry.get("recipe_id"),
                    entry.get("recipe_version_id"),
                    entry.get("config_snapshot_hash"),
                    entry.get("source_id"),
                    entry.get("acquisition_group_id"),
                    entry.get("calibration_profile_id"),
                    str(entry.get("execution_mode") or "full_run"),
                    entry.get("parent_run_id"),
                    entry.get("parent_take_id"),
                    entry.get("partial_rerun_plan_id"),
                    entry.get("boundary_stage_id"),
                    entry.get("boundary_unit_id"),
                    entry.get("selected_unit_id"),
                    entry.get("comparison_id"),
                    json.dumps(entry.get("summary") or {}, sort_keys=True),
                ),
            )
            take_id = str(entry.get("take_id") or "")
            if take_id:
                dependencies.setdefault(take_id, {}).setdefault("runs", []).append(entry)
        report.process_runs = len(entries)

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
                summary = get_take_summary(self.settings, take_id)
            except Exception as error:  # noqa: BLE001 - one bad take must not stop the scan
                report.take_failures.append(f"{take_id}: {type(error).__name__}: {error}")
                continue
            self._write_take_row(summary, fingerprint, indexed_at=indexed_at)
            report.takes_projected += 1

        removed = set(stored) - take_ids
        if removed:
            self.conn.executemany(
                "DELETE FROM take_index WHERE take_id = ?", [(take_id,) for take_id in sorted(removed)]
            )
            report.takes_removed = len(removed)

    def _write_take_row(self, summary: Any, fingerprint: dict[str, Any], *, indexed_at: str) -> None:
        payload = summary.model_dump(mode="json")
        take_id = str(payload["take_id"])
        self.conn.execute(
            "INSERT OR REPLACE INTO take_index(take_id, status, has_ready, has_done, created_at,"
            " processed_at, decision, engine, object_count, ball_count, plane_found,"
            " processing_time_ms, warning_count, calibration_id, frame_count, frameset_id,"
            " acquisition_session_id, processed_class_label, processed_superclass,"
            " latest_run_status, thumbnail_path, summary_json, metadata_mtime, metadata_size,"
            " result_mtime, result_size, status_mtime, status_size, management_hash,"
            " projection_version, indexed_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
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


def _dependency_hash(dependency: dict[str, Any] | None) -> str:
    """Hash every summary input that lives outside the take's own directory."""
    payload = {
        "take": (dependency or {}).get("take"),
        "dataset": (dependency or {}).get("dataset"),
        "session": (dependency or {}).get("session"),
        "runs": sorted(
            (dependency or {}).get("runs") or [],
            key=lambda item: str(item.get("run_id") or ""),
        ),
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
