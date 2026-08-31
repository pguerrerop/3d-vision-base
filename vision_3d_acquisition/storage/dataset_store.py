"""Dataset documents stored as rows.

The SQL backend for :mod:`vision_3d_acquisition.datasets.documents`. Every write
stores the document verbatim in ``payload_json`` and projects the fields the
listing filters on into columns; every read parses the document back. That keeps
the flip to SQLite lossless — a caller gets back exactly the dict it wrote, keys
the schema has never heard of included — while the columns stay available to
filter and join on.

Take membership is the one place the semantics genuinely improve rather than
being reproduced. On files a take could carry a sidecar under two sessions at
once and the winner depended on directory iteration order, which is how 421 takes
ended up showing empty labels. ``take_metadata.take_id`` is a primary key, so the
state simply cannot be represented any more.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vision_3d_acquisition.storage import db

LABEL_KINDS = {
    "tags": "tag",
    "semantic_labels": "semantic",
    "superclass_labels": "superclass",
    "categories": "category",
    "labels": "label",
}


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


def _doc(row, column: str = "payload_json") -> dict[str, Any]:
    try:
        payload = json.loads(row[column] or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _upsert(conn, table: str, keys: tuple[str, ...], values: dict[str, Any]) -> None:
    """Insert or update in place, without deleting the row first.

    Not INSERT OR REPLACE: that is a DELETE followed by an INSERT, so on a table
    with ON DELETE CASCADE children it silently takes them with it. Writing an
    ML set's document used to wipe every one of its memberships that way.
    """
    columns = list(values)
    assignments = ", ".join(f"{column} = excluded.{column}" for column in columns if column not in keys)
    conn.execute(
        f"INSERT INTO {table}({', '.join(columns)})"
        f" VALUES ({', '.join('?' * len(columns))})"
        f" ON CONFLICT({', '.join(keys)}) DO UPDATE SET {assignments}",
        [values[column] for column in columns],
    )


# --------------------------------------------------------------- projections
# Shared with the indexer's one-shot import so a row written by either path has
# the same columns filled in.


def project_dataset(conn, dataset_id: str, payload: dict[str, Any]) -> None:
    _upsert(
        conn,
        "dataset",
        ("id",),
        {
            "id": dataset_id,
            "name": str(payload.get("name") or dataset_id),
            "description": payload.get("description"),
            "notes": payload.get("notes"),
            "tags_json": json.dumps(payload.get("tags") or []),
            "created_at": payload.get("created_at"),
            "updated_at": payload.get("updated_at"),
            "payload_json": json.dumps(payload, sort_keys=True),
        },
    )


def project_session(conn, dataset_id: str, session_id: str, payload: dict[str, Any]) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO experiment_session(dataset_id, id, name, session_type,"
        " session_type_raw, description, notes, tags_json, metadata_json, created_at, updated_at,"
        " payload_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            dataset_id,
            session_id,
            payload.get("name"),
            str(payload.get("session_type") or "engineering"),
            str(payload.get("session_type") or ""),
            payload.get("description"),
            payload.get("notes"),
            json.dumps(payload.get("tags") or []),
            json.dumps(payload.get("metadata") or {}),
            payload.get("created_at"),
            payload.get("updated_at"),
            json.dumps(payload, sort_keys=True),
        ),
    )


def project_take(conn, take_id: str, dataset_id: str, session_id: str, payload: dict[str, Any]) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO take_metadata(take_id, dataset_id, session_id, friendly_name,"
        " notes, operator_notes, session_notes, validation_status, normalized_class,"
        " normalization_version, expected_class, expected_diameter_mm, expected_count,"
        " physical_object_id, acquisition_group_id, reference_type, is_reference,"
        " is_golden_sample, archived, archived_at, archived_reason, created_at, updated_at,"
        " split, sidecar_calibration_id, sidecar_latest_run_status, sidecar_object_count,"
        " payload_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
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
            payload.get("split"),
            payload.get("calibration_id"),
            payload.get("latest_run_status"),
            _as_int(payload.get("object_count")),
            json.dumps(payload, sort_keys=True),
        ),
    )
    project_take_labels(conn, take_id, payload)
    project_object_annotations(conn, take_id, payload)


def project_take_labels(conn, take_id: str, payload: dict[str, Any]) -> int:
    conn.execute("DELETE FROM take_label WHERE take_id = ?", (take_id,))
    rows = {
        (take_id, kind, str(value).strip())
        for source_key, kind in LABEL_KINDS.items()
        for value in (payload.get(source_key) or [])
        if str(value).strip()
    }
    conn.executemany(
        "INSERT OR IGNORE INTO take_label(take_id, kind, value) VALUES (?,?,?)", sorted(rows)
    )
    return len(rows)


def project_object_annotations(conn, take_id: str, payload: dict[str, Any]) -> int:
    conn.execute("DELETE FROM object_annotation WHERE take_id = ?", (take_id,))
    annotations = [item for item in (payload.get("object_annotations") or []) if isinstance(item, dict)]
    for annotation in annotations:
        conn.execute(
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


def project_ml_set(conn, dataset_id: str, ml_set_id: str, payload: dict[str, Any]) -> None:
    _upsert(
        conn,
        "ml_set",
        ("dataset_id", "id"),
        {
            "dataset_id": dataset_id,
            "id": ml_set_id,
            "name": payload.get("name"),
            "description": payload.get("description"),
            "task_type": payload.get("task_type"),
            "notes": payload.get("notes"),
            "created_at": payload.get("created_at"),
            "updated_at": payload.get("updated_at"),
            "payload_json": json.dumps(payload, sort_keys=True),
        },
    )


def project_membership(conn, dataset_id: str, ml_set_id: str, row: dict[str, Any]) -> None:
    conn.execute(
        'INSERT OR REPLACE INTO ml_set_membership(dataset_id, ml_set_id, take_id, split,'
        ' "include", default_trainable, trainable, physical_object_id, expected_label,'
        " expected_class, expected_subclass, raw_label, label_policy, review_required,"
        " normalization_version, source_row, notes, measurements_mm_json, extra_fields_json,"
        " created_at, updated_at, payload_json)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            dataset_id,
            ml_set_id,
            str(row.get("take_id") or ""),
            str(row.get("split") or "unassigned"),
            int(bool(row.get("include", True))),
            None if row.get("default_trainable") is None else int(bool(row.get("default_trainable"))),
            None if row.get("trainable") is None else int(bool(row.get("trainable"))),
            row.get("physical_object_id"),
            row.get("expected_label"),
            row.get("expected_class"),
            row.get("expected_subclass"),
            row.get("raw_label"),
            row.get("label_policy"),
            None if row.get("review_required") is None else int(bool(row.get("review_required"))),
            row.get("normalization_version"),
            row.get("source_row"),
            row.get("notes"),
            json.dumps(row.get("measurements_mm") or {}),
            json.dumps(row.get("extra_fields") or {}),
            row.get("created_at"),
            row.get("updated_at"),
            json.dumps(row, sort_keys=True),
        ),
    )


def project_physical_object(conn, dataset_id: str, object_id: str, payload: dict[str, Any]) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO physical_object(dataset_id, id, raw_operator_label,"
        " normalized_class, superclass, d1_mm, d2_mm, d3_mm, diameter_mean_mm,"
        " diameter_range_mm, annotation_confidence, needs_review, source_type,"
        " source_row_index, notes, tags_json, source_session_ids_json, created_at, payload_json)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
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
            json.dumps(payload, sort_keys=True),
        ),
    )
    conn.execute(
        "DELETE FROM physical_object_observation WHERE dataset_id = ? AND object_id = ?",
        (dataset_id, object_id),
    )
    observations = {
        (dataset_id, object_id, str(take_id))
        for take_id in (payload.get("observation_take_ids") or [])
        if str(take_id)
    }
    conn.executemany(
        "INSERT OR IGNORE INTO physical_object_observation(dataset_id, object_id, take_id)"
        " VALUES (?,?,?)",
        sorted(observations),
    )


# ------------------------------------------------------------------- backend


class SqlDocuments:
    """Dataset documents backed by the catalog. The source of truth in stage 2."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)

    @property
    def conn(self):
        return db.catalog_for_process(self.data_dir)

    def _write(self, projector, *args) -> None:
        conn = self.conn
        if conn.in_transaction:
            projector(conn, *args)
            return
        with db.transaction(conn):
            projector(conn, *args)

    # --------------------------------------------------------------- datasets

    def list_datasets(self) -> list[dict[str, Any]]:
        return [
            _doc(row)
            for row in self.conn.execute(
                "SELECT payload_json FROM dataset ORDER BY coalesce(created_at, '') DESC, id"
            )
        ]

    def read_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT payload_json FROM dataset WHERE id = ?", (dataset_id,)
        ).fetchone()
        return _doc(row) if row else None

    def write_dataset(self, dataset_id: str, payload: dict[str, Any]) -> None:
        self._write(project_dataset, dataset_id, payload)

    # --------------------------------------------------------------- sessions

    def list_sessions(self, dataset_id: str | None = None) -> list[dict[str, Any]]:
        if dataset_id:
            rows = self.conn.execute(
                "SELECT payload_json FROM experiment_session WHERE dataset_id = ?"
                " ORDER BY coalesce(created_at, '') DESC, id",
                (dataset_id,),
            )
        else:
            rows = self.conn.execute(
                "SELECT payload_json FROM experiment_session"
                " ORDER BY coalesce(created_at, '') DESC, dataset_id, id"
            )
        return [_doc(row) for row in rows]

    def read_session(self, dataset_id: str, session_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT payload_json FROM experiment_session WHERE dataset_id = ? AND id = ?",
            (dataset_id, session_id),
        ).fetchone()
        return _doc(row) if row else None

    def write_session(self, dataset_id: str, session_id: str, payload: dict[str, Any]) -> None:
        self._write(project_session, dataset_id, session_id, payload)

    # ------------------------------------------------------------------ takes

    def read_take(self, dataset_id: str, session_id: str, take_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT payload_json FROM take_metadata WHERE take_id = ? AND dataset_id = ?"
            " AND session_id = ?",
            (take_id, dataset_id, session_id),
        ).fetchone()
        return _doc(row) if row else None

    def write_take(self, dataset_id: str, session_id: str, take_id: str, payload: dict[str, Any]) -> None:
        self._write(project_take, take_id, dataset_id, session_id, payload)

    def delete_take(self, dataset_id: str, session_id: str, take_id: str) -> bool:
        conn = self.conn
        with db.transaction(conn):
            cursor = conn.execute(
                "DELETE FROM take_metadata WHERE take_id = ? AND dataset_id = ? AND session_id = ?",
                (take_id, dataset_id, session_id),
            )
            removed = cursor.rowcount > 0
            if removed:
                conn.execute("DELETE FROM take_label WHERE take_id = ?", (take_id,))
                conn.execute("DELETE FROM object_annotation WHERE take_id = ?", (take_id,))
        return removed

    def take_memberships(self, take_id: str) -> list[tuple[str, str]]:
        """At most one. The primary key makes a second membership unrepresentable."""
        row = self.conn.execute(
            "SELECT dataset_id, session_id FROM take_metadata WHERE take_id = ?", (take_id,)
        ).fetchone()
        if not row or not row["dataset_id"] or not row["session_id"]:
            return []
        return [(str(row["dataset_id"]), str(row["session_id"]))]

    def iter_dataset_takes(
        self, dataset_id: str, session_id: str | None = None
    ) -> list[tuple[str, str, dict[str, Any]]]:
        query = (
            "SELECT session_id, take_id, payload_json FROM take_metadata WHERE dataset_id = ?"
        )
        params: list[Any] = [dataset_id]
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        query += " ORDER BY session_id, take_id"
        return [
            (str(row["session_id"]), str(row["take_id"]), _doc(row))
            for row in self.conn.execute(query, params)
        ]

    # ---------------------------------------------------------------- ml sets

    def list_ml_sets(self, dataset_id: str | None = None) -> list[dict[str, Any]]:
        if dataset_id:
            rows = self.conn.execute(
                "SELECT payload_json FROM ml_set WHERE dataset_id = ?"
                " ORDER BY coalesce(created_at, '') DESC, id",
                (dataset_id,),
            )
        else:
            rows = self.conn.execute(
                "SELECT payload_json FROM ml_set ORDER BY coalesce(created_at, '') DESC, dataset_id, id"
            )
        return [_doc(row) for row in rows]

    def read_ml_set(self, dataset_id: str, ml_set_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT payload_json FROM ml_set WHERE dataset_id = ? AND id = ?",
            (dataset_id, ml_set_id),
        ).fetchone()
        return _doc(row) if row else None

    def write_ml_set(self, dataset_id: str, ml_set_id: str, payload: dict[str, Any]) -> None:
        self._write(project_ml_set, dataset_id, ml_set_id, payload)

    def read_memberships(self, dataset_id: str, ml_set_id: str) -> list[dict[str, Any]]:
        return [
            _doc(row)
            for row in self.conn.execute(
                "SELECT payload_json FROM ml_set_membership WHERE dataset_id = ? AND ml_set_id = ?"
                " ORDER BY take_id",
                (dataset_id, ml_set_id),
            )
        ]

    def write_memberships(self, dataset_id: str, ml_set_id: str, rows: list[dict[str, Any]]) -> None:
        conn = self.conn

        def apply(target) -> None:
            target.execute(
                "DELETE FROM ml_set_membership WHERE dataset_id = ? AND ml_set_id = ?",
                (dataset_id, ml_set_id),
            )
            for row in rows:
                project_membership(target, dataset_id, ml_set_id, row)

        if conn.in_transaction:
            apply(conn)
            return
        with db.transaction(conn):
            apply(conn)

    # ------------------------------------------------------- physical objects

    def list_physical_objects(self, dataset_id: str) -> list[dict[str, Any]]:
        return [
            _doc(row)
            for row in self.conn.execute(
                "SELECT payload_json FROM physical_object WHERE dataset_id = ? ORDER BY id",
                (dataset_id,),
            )
        ]

    def read_physical_object(self, dataset_id: str, object_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT payload_json FROM physical_object WHERE dataset_id = ? AND id = ?",
            (dataset_id, object_id),
        ).fetchone()
        return _doc(row) if row else None

    def write_physical_object(self, dataset_id: str, object_id: str, payload: dict[str, Any]) -> None:
        self._write(project_physical_object, dataset_id, object_id, payload)
