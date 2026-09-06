"""Controlled vocabulary for physical-object normalized_class/superclass pairs.

Single source of truth, replacing the two contradictory dictionaries in
vision_3d_acquisition/ml/label_normalization.py and the stale JSON config
under config/label_normalization/ that previously disagreed on real labels
(e.g. "cadena" mapped to BALL_SCRAP instead of SCRAP_METAL).
"""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import Any

# (normalized_class, superclass, is_uncertain, notes)
# is_uncertain marks classes whose own meaning is a hedge ("looks like...",
# "doubt"), not instances flagged individually via the per-object
# needs_review field.
SEED_ENTRIES: list[tuple[str, str, bool, str | None]] = [
    ("buena", "BALL_GOOD", False, None),
    ("buena_menos", "BALL_GOOD", True, "Borderline-good ball; operator hedge in the raw label."),
    ("ahuevada", "BALL_SCRAP", False, "Deformed ball."),
    ("mitad", "BALL_SCRAP", False, "Half ball."),
    ("chica", "BALL_SCRAP", False, "Undersized ball; exact mm cutoff pending plant confirmation."),
    ("bola_con_chip", "BALL_SCRAP", False, "Chipped ball."),
    ("cadena", "SCRAP_METAL", False, "Foreign object (chain link), not a ball defect."),
    ("duda", "REVIEW", True, "No real class assigned by the operator; needs human review, not a default class."),
    ("chatarra", "SCRAP_METAL", False, "Generic scrap."),
    ("pedazo", "SCRAP_METAL", False, "Scrap fragment."),
    ("parece_planchuela", "SCRAP_METAL", True, "Operator hedge ('looks like a flat bar')."),
    ("planchuela", "SCRAP_METAL", False, "Flat bar."),
    ("planchuela_doblada", "SCRAP_METAL", True, "Bent flat bar; often recorded with a trailing '?'."),
    ("perno", "SCRAP_METAL", False, "Bolt."),
    ("tuerca", "SCRAP_METAL", False, "Nut."),
    ("cubo", "SCRAP_METAL", False, "Calibration cube; excluded from training by ml_set policy, not a labeling call."),
]


def _now() -> str:
    return datetime.now(UTC).isoformat()


def ensure_seeded(conn: sqlite3.Connection) -> None:
    now = _now()
    conn.executemany(
        "INSERT OR IGNORE INTO label_taxonomy(normalized_class, superclass, is_uncertain, notes, updated_at, updated_by) "
        "VALUES (?, ?, ?, ?, ?, 'seed')",
        [(cls, superclass, int(uncertain), notes, now) for cls, superclass, uncertain, notes in SEED_ENTRIES],
    )
    conn.commit()


def _row_to_entry(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "normalized_class": row["normalized_class"],
        "superclass": row["superclass"],
        "is_uncertain": bool(row["is_uncertain"]),
        "notes": row["notes"],
        "updated_at": row["updated_at"],
        "updated_by": row["updated_by"],
    }


def list_entries(conn: sqlite3.Connection, query: str | None = None) -> list[dict[str, Any]]:
    ensure_seeded(conn)
    conn.row_factory = sqlite3.Row
    if query and query.strip():
        like = f"%{query.strip().lower()}%"
        cursor = conn.execute(
            "SELECT * FROM label_taxonomy WHERE lower(normalized_class) LIKE ? OR lower(superclass) LIKE ? "
            "ORDER BY normalized_class",
            (like, like),
        )
    else:
        cursor = conn.execute("SELECT * FROM label_taxonomy ORDER BY normalized_class")
    return [_row_to_entry(row) for row in cursor.fetchall()]


def get_entry(conn: sqlite3.Connection, normalized_class: str) -> dict[str, Any] | None:
    ensure_seeded(conn)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM label_taxonomy WHERE normalized_class = ?",
        (normalized_class.strip(),),
    ).fetchone()
    return _row_to_entry(row) if row else None


def upsert_entry(conn: sqlite3.Connection, *, normalized_class: str, superclass: str,
                  is_uncertain: bool = False, notes: str | None = None, updated_by: str | None = None) -> dict[str, Any]:
    normalized_class = normalized_class.strip()
    superclass = superclass.strip()
    if not normalized_class:
        raise ValueError("normalized_class is required")
    if not superclass:
        raise ValueError("superclass is required")
    ensure_seeded(conn)
    now = _now()
    with conn:
        conn.execute(
            "INSERT INTO label_taxonomy(normalized_class, superclass, is_uncertain, notes, updated_at, updated_by) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(normalized_class) DO UPDATE SET "
            "superclass=excluded.superclass, is_uncertain=excluded.is_uncertain, "
            "notes=excluded.notes, updated_at=excluded.updated_at, updated_by=excluded.updated_by",
            (normalized_class, superclass, int(is_uncertain), notes, now, updated_by or "studio"),
        )
    entry = get_entry(conn, normalized_class)
    assert entry is not None
    return entry
