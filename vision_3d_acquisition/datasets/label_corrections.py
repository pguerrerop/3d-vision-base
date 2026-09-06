"""Transactional, object-scoped label corrections with a SQLite audit trail."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4
from typing import Any

from vision_3d_acquisition.datasets import label_taxonomy
from vision_3d_acquisition.datasets.service import DatasetService
from vision_3d_acquisition.storage import db


def _now() -> str:
    return datetime.now(UTC).isoformat()


def correct_physical_object_label(*, data_dir: Path, dataset_id: str, physical_object_id: str,
                                  normalized_class: str, superclass: str | None, raw_label: str | None,
                                  actor: str, reason: str | None) -> dict[str, Any]:
    service = DatasetService(data_dir)
    obj = service.docs.read_physical_object(dataset_id, physical_object_id)
    if obj is None:
        raise ValueError(f"physical object not found: {dataset_id}/{physical_object_id}")
    if not normalized_class.strip():
        raise ValueError("normalized_class is required")

    conn = db.catalog_for_process(data_dir)

    normalized_class = normalized_class.strip()
    superclass = superclass.strip() if superclass else None
    taxonomy_entry = label_taxonomy.get_entry(conn, normalized_class)
    if taxonomy_entry is not None:
        if superclass and superclass != taxonomy_entry["superclass"]:
            raise ValueError(
                f"normalized_class '{normalized_class}' is registered as {taxonomy_entry['superclass']} in "
                f"label_taxonomy, not {superclass}. Fix the class name, or update the taxonomy entry explicitly "
                "if the pairing itself is wrong."
            )
        superclass = taxonomy_entry["superclass"]
    else:
        if not superclass:
            raise ValueError(
                f"normalized_class '{normalized_class}' is not in label_taxonomy yet — provide a superclass "
                "to register it as a new class."
            )
        label_taxonomy.upsert_entry(
            conn, normalized_class=normalized_class, superclass=superclass,
            updated_by=actor or "studio", notes=f"Auto-registered while correcting {physical_object_id}",
        )

    now = _now()
    affected_takes: list[str] = []
    affected_sets: list[str] = []
    before = {"physical_object": obj, "takes": {}, "memberships": {}}
    after_obj = {**obj, "normalized_class": normalized_class.strip(), "superclass": superclass,
                 "raw_operator_label": raw_label if raw_label is not None else obj.get("raw_operator_label"),
                 "needs_review": False, "updated_at": now}
    with db.transaction(conn):
        service.docs.write_physical_object(dataset_id, physical_object_id, after_obj)
        for session_id, take_id, take in service.docs.iter_dataset_takes(dataset_id):
            if str(take.get("physical_object_id") or "") != physical_object_id:
                continue
            before["takes"][take_id] = take
            updated = {**take, "expected_class": normalized_class.strip(), "semantic_labels": [normalized_class.strip()],
                       "superclass_labels": [superclass] if superclass else [], "validation_status": "valid",
                       "label_correction_id": None, "updated_at": now}
            service.docs.write_take(dataset_id, session_id, take_id, updated)
            affected_takes.append(take_id)
        for ml_set in service.docs.list_ml_sets(dataset_id):
            ml_set_id = str(ml_set.get("id") or "")
            rows = service.docs.read_memberships(dataset_id, ml_set_id)
            changed = False
            next_rows: list[dict[str, Any]] = []
            for row in rows:
                if str(row.get("physical_object_id") or "") == physical_object_id:
                    before["memberships"].setdefault(ml_set_id, []).append(row)
                    row = {**row, "expected_class": normalized_class.strip(), "expected_subclass": superclass,
                           "raw_label": raw_label if raw_label is not None else row.get("raw_label"),
                           "review_required": False, "updated_at": now}
                    changed = True
                next_rows.append(row)
            if changed:
                service.docs.write_memberships(dataset_id, ml_set_id, next_rows)
                affected_sets.append(ml_set_id)
        correction_id = f"label_correction_{uuid4().hex}"
        conn.execute("INSERT INTO label_correction(id,dataset_id,physical_object_id,actor,reason,before_json,after_json,affected_take_ids_json,affected_ml_set_ids_json,created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (correction_id, dataset_id, physical_object_id, actor or "studio", reason,
                      json.dumps(before, ensure_ascii=False), json.dumps(after_obj, ensure_ascii=False),
                      json.dumps(affected_takes), json.dumps(affected_sets), now))
        # Make the audit id observable from every corrected document.
        for session_id, take_id, take in service.docs.iter_dataset_takes(dataset_id):
            if take_id in affected_takes:
                service.docs.write_take(dataset_id, session_id, take_id, {**take, "label_correction_id": correction_id})
    return {"id": correction_id, "physical_object": after_obj, "affected_take_ids": affected_takes,
            "affected_ml_set_ids": affected_sets, "created_at": now}
