"""Read the take listing from the catalog instead of walking the filesystem.

This is the stage 1 read path. It has one job and one hard constraint: return
exactly what ``api.filesystem.list_takes_paged`` returns, byte for byte, while
doing the filtering in SQL so the cost stops scaling with the number of takes.

Two decisions keep the equivalence honest rather than hopeful:

* the rows come back as ``summary_json``, the TakeSummary the indexer already
  stored and that ``index verify`` checks against a fresh filesystem read, so the
  page payload is not recomposed here;
* the bucket counting stays in Python, running over the filtered rows with the
  same branches as the original, rather than being rewritten as SQL aggregates.

Only the row selection moved to SQL, which is the part that was O(takes).

Faithfulness notes, where the original is subtle:

* ``expected_class`` is a substring match, not equality.
* ``created_at`` for filtering and sorting is the acquisition metadata's, falling
  back to the sidecar's — not the summary's, which falls back to the result's.
* ``calibration_linkage_only`` looks at the sidecar's calibration_id, falling
  back to the acquisition metadata's, never the result's.
* takes with no sidecar behave as ``default_take_metadata``: unreviewed, not
  archived, friendly_name equal to the take id, session from the acquisition
  metadata.
* ties on the sort key were resolved by set iteration order, i.e. arbitrary.
  Here they break by take_id descending, which is the one deliberate change.
"""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Any

from vision_3d_acquisition.api.schemas import TakeSummary

# take_metadata is LEFT JOINed: a take with no sidecar has to read as the
# defaults default_take_metadata would have produced.
VALIDATION = "lower(coalesce(m.validation_status, 'unreviewed'))"
ARCHIVED = "coalesce(m.archived, 0)"
FRIENDLY = "coalesce(m.friendly_name, t.take_id)"
SESSION = "coalesce(nullif(m.session_id, ''), t.metadata_session_id, '')"
# default_take_metadata fills a missing created_at with datetime.now(), so a take
# with neither an acquisition date nor a sidecar is treated as brand new. The
# index substitutes indexed_at, which keeps that take sorting as recent without
# the filesystem path's property of answering differently on every call.
CREATED = "coalesce(nullif(t.acquisition_created_at, ''), nullif(m.created_at, ''), t.indexed_at, '')"
SORT_KEY = f"coalesce(nullif({CREATED}, ''), t.take_id)"
CALIBRATION = (
    "coalesce(nullif(m.sidecar_calibration_id, ''), nullif(t.acquisition_calibration_id, ''), '')"
)
SIDECAR_RUN_STATUS = "coalesce(m.sidecar_latest_run_status, '')"
SIDECAR_OBJECT_COUNT = "m.sidecar_object_count"

VALIDATION_BUCKETS = {
    "valid": "validated",
    "validated": "validated",
    "invalid": "rejected",
    "rejected": "rejected",
    "needs_review": "needs_review",
    "golden_sample": "golden_sample",
    "benchmark_approved": "benchmark_approved",
    "unreviewed": "unreviewed",
}


def _label_clause(kind: str, param: str) -> str:
    return (
        "EXISTS (SELECT 1 FROM take_label l WHERE l.take_id = t.take_id"
        f" AND l.kind = '{kind}' AND l.value_norm = lower(trim(:{param})))"
    )


def _build_filters(params: dict[str, Any]) -> list[str]:
    """Clauses for everything except the archived and dataset pair.

    Those two are applied separately because total_count is taken after them and
    before the rest, which is what the original counts.
    """
    clauses: list[str] = []
    if params.get("session_id"):
        clauses.append(f"{SESSION} = :session_id")
    if params.get("validation_status"):
        clauses.append(f"{VALIDATION} = lower(:validation_status)")
    for kind, key in (
        ("tag", "tag"),
        ("semantic", "semantic_label"),
        ("superclass", "superclass_label"),
        ("category", "category"),
    ):
        if params.get(key):
            clauses.append(_label_clause(kind, key))
    if params.get("physical_object_id"):
        clauses.append(
            "lower(trim(coalesce(m.physical_object_id, ''))) = lower(trim(:physical_object_id))"
        )
    if params.get("session_type"):
        # session_type_raw, not session_type: the listing compares against what
        # the session file literally holds, so a session written before the field
        # existed matches nothing rather than matching 'engineering'.
        clauses.append("lower(coalesce(s.session_type_raw, '')) = lower(trim(:session_type))")
    if params.get("reference_type"):
        clauses.append("lower(trim(coalesce(m.reference_type, ''))) = lower(trim(:reference_type))")
    if params.get("is_reference") is not None:
        clauses.append("coalesce(m.is_reference, 0) = :is_reference")
    if params.get("is_golden_sample") is not None:
        clauses.append("coalesce(m.is_golden_sample, 0) = :is_golden_sample")
    if params.get("search"):
        clauses.append(
            f"(instr(lower(t.take_id), lower(:search)) > 0 OR instr(lower({FRIENDLY}), lower(:search)) > 0)"
        )
    if params.get("expected_class"):
        # Substring, matching the original's `q not in value`.
        clauses.append(
            "instr(lower(coalesce(m.expected_class, '')), lower(trim(:expected_class))) > 0"
        )
    if params.get("split"):
        clauses.append("lower(trim(coalesce(m.split, ''))) = lower(trim(:split))")
    if params.get("calibration_linkage_only"):
        clauses.append(f"{CALIBRATION} != ''")
    if params.get("created_from"):
        clauses.append(f"({CREATED} != '' AND substr({CREATED}, 1, 10) >= :created_from)")
    if params.get("created_to"):
        clauses.append(f"({CREATED} != '' AND substr({CREATED}, 1, 10) <= :created_to)")
    if params.get("created_from") or params.get("created_to"):
        clauses.append(f"{CREATED} != ''")
    return clauses


def _base_clauses(params: dict[str, Any]) -> list[str]:
    clauses: list[str] = []
    if not params.get("include_archived"):
        clauses.append(f"{ARCHIVED} = 0")
    if params.get("dataset_id"):
        clauses.append("coalesce(m.dataset_id, '') = :dataset_id")
    return clauses


def _where(clauses: list[str]) -> str:
    return f"WHERE {' AND '.join(clauses)}" if clauses else ""


FROM_CLAUSE = (
    " FROM take_index t"
    " LEFT JOIN take_metadata m ON m.take_id = t.take_id"
    " LEFT JOIN experiment_session s ON s.dataset_id = m.dataset_id AND s.id = m.session_id"
)


# Not every param is a bind value: `settings` is an object the hydration needs,
# and the include flag never appears in a WHERE clause.
_NON_BINDING = {"settings", "profile", "include_acquisition_processing_status"}


def _bindings(params: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (int(value) if isinstance(value, bool) else value)
        for key, value in params.items()
        if key not in _NON_BINDING and value is not None and not isinstance(value, (list, dict))
    }


def _summary_counts(rows: list[sqlite3.Row]) -> dict[str, Any]:
    counts: dict[str, Any] = {
        "validation": {
            "validated": 0,
            "unreviewed": 0,
            "rejected": 0,
            "needs_review": 0,
            "golden_sample": 0,
            "benchmark_approved": 0,
        },
        "missing_labels": 0,
        "missing_split": 0,
        "missing_calibration": 0,
        "processing_failed": 0,
        "processing_incomplete": 0,
        "no_objects_detected": 0,
    }
    for row in rows:
        bucket = VALIDATION_BUCKETS.get(str(row["validation_status"] or "").strip(), "unreviewed")
        counts["validation"][bucket] += 1
        if not str(row["expected_class"] or "").strip():
            counts["missing_labels"] += 1
        if not str(row["split"] or "").strip():
            counts["missing_split"] += 1
        if not str(row["calibration"] or "").strip():
            counts["missing_calibration"] += 1
        latest_status = str(row["latest_run_status"] or "").lower().strip()
        if latest_status == "failed":
            counts["processing_failed"] += 1
        if latest_status not in {"success", "completed", "done"}:
            counts["processing_incomplete"] += 1
        object_count = row["object_count"]
        if isinstance(object_count, (int, float)) and int(object_count) <= 0:
            counts["no_objects_detected"] += 1
    return counts


def list_takes_paged(conn: sqlite3.Connection, params: dict[str, Any]) -> dict[str, Any]:
    started_at = time.perf_counter()
    resolved_limit = max(1, min(int(params.get("limit") or 50), 200))
    resolved_offset = max(0, int(params.get("offset") or 0))

    bindings = _bindings(params)

    base = _base_clauses(params)
    total_started_at = time.perf_counter()
    total_count = int(
        conn.execute(f"SELECT count(*){FROM_CLAUSE} {_where(base)}", bindings).fetchone()[0]
    )
    total_ms = (time.perf_counter() - total_started_at) * 1000.0

    select_started_at = time.perf_counter()
    rows = conn.execute(
        "SELECT t.take_id AS take_id,"
        f" {VALIDATION} AS validation_status,"
        " coalesce(m.expected_class, '') AS expected_class,"
        " coalesce(m.split, '') AS split,"
        f" {CALIBRATION} AS calibration,"
        f" {SIDECAR_RUN_STATUS} AS latest_run_status,"
        f" {SIDECAR_OBJECT_COUNT} AS object_count"
        f"{FROM_CLAUSE} {_where(base + _build_filters(params))}"
        f" ORDER BY {SORT_KEY} DESC, t.take_id DESC",
        bindings,
    ).fetchall()
    select_ms = (time.perf_counter() - select_started_at) * 1000.0

    page_take_ids = [row["take_id"] for row in rows[resolved_offset : resolved_offset + resolved_limit]]

    hydrate_started_at = time.perf_counter()
    page_items = _load_summaries(
        conn,
        page_take_ids,
        settings=params.get("settings"),
        include_acquisition_processing_status=bool(
            params.get("include_acquisition_processing_status")
        ),
    )
    hydrate_ms = (time.perf_counter() - hydrate_started_at) * 1000.0

    next_offset = resolved_offset + len(page_items)
    has_more = next_offset < len(rows)
    payload: dict[str, Any] = {
        "items": page_items,
        "limit": resolved_limit,
        "offset": resolved_offset,
        "has_more": has_more,
        "next_offset": next_offset if has_more else None,
        "filtered_count": len(rows),
        "total_count": total_count,
        "summary_counts": _summary_counts(rows),
    }
    if params.get("profile"):
        payload["profile"] = {
            "enabled": True,
            "source": "catalog_index",
            "limit": resolved_limit,
            "offset": resolved_offset,
            "counters": {
                "scanned_take_ids": total_count,
                "candidate_count": len(rows),
                "page_item_count": len(page_items),
                "total_count": total_count,
            },
            "phase_ms": {
                "count_total": round(total_ms, 2),
                "select_candidates": round(select_ms, 2),
                "hydrate_page_items": round(hydrate_ms, 2),
                "total": round((time.perf_counter() - started_at) * 1000.0, 2),
            },
        }
    return payload


def list_takes(conn: sqlite3.Connection, params: dict[str, Any]) -> list[TakeSummary]:
    """The unpaged listing.

    /api/takes and /api/takes/paged take the same query parameters and do not
    mean the same thing by two of them. The unpaged one filters on the fields of
    the TakeSummary itself: ``session_id`` is the acquisition session, not the
    dataset session the take was filed under, and ``session_type`` is the
    effective one, defaulting a missing value to 'engineering'. The paged one
    reads both from the sidecar and the session file instead.

    That difference is reproduced here rather than reconciled: stage 1 changes
    where the answer comes from, not what it is.
    """
    filters = [
        clause
        for clause in _build_filters(params)
        if not clause.startswith((SESSION, "lower(coalesce(s.session_type_raw"))
    ]
    if params.get("session_id"):
        filters.append("coalesce(t.acquisition_session_id, '') = :session_id")
    if params.get("session_type"):
        filters.append("lower(coalesce(s.session_type, '')) = lower(trim(:session_type))")

    rows = conn.execute(
        f"SELECT t.take_id AS take_id{FROM_CLAUSE}"
        f" {_where(_base_clauses(params) + filters)}"
        f" ORDER BY {SORT_KEY} DESC, t.take_id DESC",
        _bindings(params),
    ).fetchall()
    return _load_summaries(
        conn,
        [row["take_id"] for row in rows],
        settings=params.get("settings"),
        include_acquisition_processing_status=bool(
            params.get("include_acquisition_processing_status")
        ),
    )


def _load_summaries(
    conn: sqlite3.Connection,
    take_ids: list[str],
    *,
    settings: Any = None,
    include_acquisition_processing_status: bool = False,
) -> list[TakeSummary]:
    if not take_ids:
        return []
    placeholders = ",".join("?" * len(take_ids))
    stored = {
        row["take_id"]: row["summary_json"]
        for row in conn.execute(
            f"SELECT take_id, summary_json FROM take_index WHERE take_id IN ({placeholders})",
            take_ids,
        )
    }
    # Validated, not constructed: skipping validation was measured at 0.1 ms per
    # page against a hydration cost that is dominated by parsing the documents
    # themselves. There is no reason to give up the check.
    summaries = [
        TakeSummary.model_validate(json.loads(stored[take_id]))
        for take_id in take_ids
        if take_id in stored
    ]

    if include_acquisition_processing_status and settings is not None:
        # Read per page, not per row: the stored summary deliberately leaves this
        # out, so the cost is only paid by callers that asked for it.
        from vision_3d_acquisition.acquisition.processing_status import read_status

        for summary in summaries:
            summary.acquisition_processing_status = read_status(settings.data_dir, summary.take_id)

    return summaries
