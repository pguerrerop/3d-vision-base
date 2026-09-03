from __future__ import annotations

import json

from vision_3d_acquisition.datasets import DatasetService
from vision_3d_acquisition.datasets.label_corrections import correct_physical_object_label


def test_object_label_correction_updates_documents_and_writes_audit(catalog_settings, catalog_conn):
    service = DatasetService(catalog_settings.data_dir)
    service.create_dataset(dataset_id="balls", name="Balls")
    service.create_session(dataset_id="balls", session_id="may", name="May")
    service.docs.write_physical_object("balls", "obj_9", {
        "physical_object_id": "obj_9", "dataset_id": "balls", "normalized_class": "BALL_SCRAP",
        "superclass": "BALL", "raw_operator_label": "cadena", "source_session_ids": ["may"],
    })
    service.docs.write_take("balls", "may", "take_9", {
        "take_id": "take_9", "dataset_id": "balls", "session_id": "may", "physical_object_id": "obj_9",
        "expected_class": "BALL_SCRAP", "semantic_labels": ["BALL_SCRAP"], "superclass_labels": ["BALL"],
    })
    service.create_ml_set(dataset_id="balls", ml_set_id="set_1", name="set", task_type="classification")
    service.docs.write_memberships("balls", "set_1", [{
        "take_id": "take_9", "physical_object_id": "obj_9", "expected_class": "BALL_SCRAP",
        "expected_subclass": "BALL", "raw_label": "cadena", "split": "test", "include": True,
    }])

    result = correct_physical_object_label(
        data_dir=catalog_settings.data_dir, dataset_id="balls", physical_object_id="obj_9",
        normalized_class="SCRAP_METAL", superclass="SCRAP", raw_label="cadena", actor="qa", reason="taxonomy fix",
    )

    assert result["affected_take_ids"] == ["take_9"]
    assert result["affected_ml_set_ids"] == ["set_1"]
    assert service.docs.read_take("balls", "may", "take_9")["expected_class"] == "SCRAP_METAL"
    assert service.docs.read_memberships("balls", "set_1")[0]["expected_class"] == "SCRAP_METAL"
    audit = catalog_conn.execute("SELECT actor, before_json, after_json FROM label_correction").fetchone()
    assert audit["actor"] == "qa"
    assert json.loads(audit["before_json"])["physical_object"]["normalized_class"] == "BALL_SCRAP"
    assert json.loads(audit["after_json"])["normalized_class"] == "SCRAP_METAL"
