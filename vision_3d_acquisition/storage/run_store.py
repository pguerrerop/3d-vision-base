"""Process runs, stored in SQLite instead of a shared JSON document.

This is the first table stage 2 promotes from derived to authoritative, and it is
first on purpose: the rows can be rebuilt from ``processes/runs/`` if anything
goes wrong, so the concurrent write path gets proven here before anything
irrecoverable moves.

What it replaces: ``append_process_run_index`` used to read
``data/processes/index/runs.json`` whole, append one entry, and rewrite the file
with ``write_text`` — no lock, no atomic replace — from four processes at once
(API, 25D worker, RGB worker, fusion publisher). Two runs finishing together lost
an entry; a crash mid-write truncated the index. Here an append is one INSERT
inside ``BEGIN IMMEDIATE``: concurrent writers serialize instead of clobbering.

``runs.json`` is still written, as a read-only mirror refreshed by
``sensor-studio index rebuild``. Nothing in the codebase reads it any more — it
is kept for one release as a human-readable fallback.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from vision_3d_acquisition.storage import db

# Column order is the entry contract: these are the keys callers pass and the
# keys readers get back, so the JSON mirror and the table cannot drift apart.
RUN_COLUMNS = (
    "run_id",
    "take_id",
    "pipeline_instance_id",
    "pipeline_id",
    "pipeline_family",
    "status",
    "created_at",
    "path",
    "recipe_id",
    "recipe_version_id",
    "config_snapshot_hash",
    "source_id",
    "acquisition_group_id",
    "calibration_profile_id",
    "execution_mode",
    "parent_run_id",
    "parent_take_id",
    "partial_rerun_plan_id",
    "boundary_stage_id",
    "boundary_unit_id",
    "selected_unit_id",
    "comparison_id",
)

_NOT_NULL_DEFAULTS = {
    "pipeline_family": "generic",
    "status": "completed",
    "created_at": "",
    "execution_mode": "full_run",
}


def write_run(conn, entry: dict[str, Any]) -> None:
    """Insert or replace one run row. Caller owns the transaction."""
    run_id = str(entry.get("run_id") or "")
    if not run_id:
        return
    values = [
        _NOT_NULL_DEFAULTS[column]
        if column in _NOT_NULL_DEFAULTS and not entry.get(column)
        else entry.get(column)
        for column in RUN_COLUMNS
    ]
    values.append(json.dumps(entry.get("summary") or {}, sort_keys=True))
    placeholders = ",".join("?" * (len(RUN_COLUMNS) + 1))
    conn.execute(
        f"INSERT OR REPLACE INTO process_run({', '.join(RUN_COLUMNS)}, summary_json)"
        f" VALUES ({placeholders})",
        values,
    )


def row_to_entry(row) -> dict[str, Any]:
    entry = {column: row[column] for column in RUN_COLUMNS}
    try:
        entry["summary"] = json.loads(row["summary_json"] or "{}")
    except (TypeError, json.JSONDecodeError):
        entry["summary"] = {}
    return entry


def append_run(data_dir: Path, entry: dict[str, Any]) -> None:
    """Record one run. One INSERT, one short transaction, no shared document."""
    conn = _connect(data_dir)
    with db.transaction(conn):
        write_run(conn, entry)


def update_run(data_dir: Path, run_id: str, patch: dict[str, Any]) -> list[str]:
    """Patch one run in place. Returns the take ids whose history changed."""
    conn = _connect(data_dir)
    columns = [key for key in patch if key in RUN_COLUMNS]
    if not columns:
        return []
    assignments = ", ".join(f"{column} = ?" for column in columns)
    with db.transaction(conn):
        take_ids = [
            str(row["take_id"])
            for row in conn.execute(
                "SELECT take_id FROM process_run WHERE run_id = ? AND take_id IS NOT NULL",
                (run_id,),
            )
        ]
        conn.execute(
            f"UPDATE process_run SET {assignments} WHERE run_id = ?",
            [patch[column] for column in columns] + [run_id],
        )
    return take_ids


def load_runs(data_dir: Path) -> list[dict[str, Any]]:
    conn = _connect(data_dir)
    return [row_to_entry(row) for row in conn.execute("SELECT * FROM process_run ORDER BY created_at, run_id")]


def runs_for_take(data_dir: Path, take_id: str) -> list[dict[str, Any]]:
    conn = _connect(data_dir)
    return [
        row_to_entry(row)
        for row in conn.execute(
            "SELECT * FROM process_run WHERE take_id = ? ORDER BY created_at, run_id", (take_id,)
        )
    ]


def import_entries(data_dir: Path, entries: Iterable[dict[str, Any]], conn=None) -> int:
    """One-shot import, used to migrate runs.json into the table.

    Pass ``conn`` when already inside a transaction — the indexer does, and
    opening a second one from the same process deadlocks against the first.
    """
    if conn is not None:
        count = 0
        for entry in entries:
            write_run(conn, entry)
            count += 1
        return count

    owned = _connect(data_dir)
    count = 0
    with db.transaction(owned):
        for entry in entries:
            write_run(owned, entry)
            count += 1
    return count


def is_empty(data_dir: Path, conn=None) -> bool:
    target = conn if conn is not None else _connect(data_dir)
    return int(target.execute("SELECT count(*) FROM process_run").fetchone()[0]) == 0


def export_json_mirror(data_dir: Path, path: Path) -> int:
    """Refresh the read-only runs.json mirror. Not a source of truth any more."""
    entries = load_runs(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "entries": entries,
                "source": "sqlite:process_run",
                "readonly": True,
                "note": "Mirror refreshed by `sensor-studio index rebuild`. Writes go to data/index.db.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return len(entries)


def _connect(data_dir: Path):
    return db.catalog_for_process(Path(data_dir))
