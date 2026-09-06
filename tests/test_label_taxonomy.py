from __future__ import annotations

import pytest

from vision_3d_acquisition.datasets import DatasetService, label_taxonomy
from vision_3d_acquisition.datasets.label_corrections import correct_physical_object_label


def test_seed_fixes_cadena_to_scrap_metal(catalog_conn):
    entry = label_taxonomy.get_entry(catalog_conn, "cadena")
    assert entry is not None
    assert entry["superclass"] == "SCRAP_METAL"


def test_list_entries_filters_by_typed_query(catalog_conn):
    results = label_taxonomy.list_entries(catalog_conn, "chat")
    assert [entry["normalized_class"] for entry in results] == ["chatarra"]

    by_superclass = label_taxonomy.list_entries(catalog_conn, "ball_good")
    classes = {entry["normalized_class"] for entry in by_superclass}
    assert classes == {"buena", "buena_menos"}


def _make_object(service: DatasetService, dataset_id: str, physical_object_id: str, take_id: str) -> None:
    service.create_dataset(dataset_id=dataset_id, name=dataset_id)
    service.create_session(dataset_id=dataset_id, session_id="s1", name="s1")
    service.docs.write_physical_object(dataset_id, physical_object_id, {
        "physical_object_id": physical_object_id, "dataset_id": dataset_id,
        "normalized_class": "cadena", "superclass": "BALL_SCRAP", "raw_operator_label": "Cadena",
        "source_session_ids": ["s1"],
    })
    service.docs.write_take(dataset_id, "s1", take_id, {
        "take_id": take_id, "dataset_id": dataset_id, "session_id": "s1", "physical_object_id": physical_object_id,
        "expected_class": "cadena", "semantic_labels": ["cadena"], "superclass_labels": ["BALL_SCRAP"],
    })


def test_correction_rejects_superclass_mismatch_for_known_class(catalog_settings):
    service = DatasetService(catalog_settings.data_dir)
    _make_object(service, "balls", "obj_1", "take_1")

    with pytest.raises(ValueError, match="registered as SCRAP_METAL"):
        correct_physical_object_label(
            data_dir=catalog_settings.data_dir, dataset_id="balls", physical_object_id="obj_1",
            normalized_class="cadena", superclass="BALL_SCRAP", raw_label="Cadena", actor="qa", reason="typo test",
        )


def test_correction_fixes_known_class_to_taxonomy_superclass(catalog_settings):
    service = DatasetService(catalog_settings.data_dir)
    _make_object(service, "balls", "obj_1", "take_1")

    result = correct_physical_object_label(
        data_dir=catalog_settings.data_dir, dataset_id="balls", physical_object_id="obj_1",
        normalized_class="cadena", superclass=None, raw_label="Cadena", actor="qa", reason="apply taxonomy",
    )

    assert result["physical_object"]["superclass"] == "SCRAP_METAL"
    assert service.docs.read_take("balls", "s1", "take_1")["superclass_labels"] == ["SCRAP_METAL"]


def test_correction_registers_a_new_class_when_superclass_given(catalog_settings, catalog_conn):
    service = DatasetService(catalog_settings.data_dir)
    _make_object(service, "balls", "obj_1", "take_1")

    correct_physical_object_label(
        data_dir=catalog_settings.data_dir, dataset_id="balls", physical_object_id="obj_1",
        normalized_class="oxidada", superclass="BALL_SCRAP", raw_label="Oxidada", actor="qa", reason="new class",
    )

    entry = label_taxonomy.get_entry(catalog_conn, "oxidada")
    assert entry is not None
    assert entry["superclass"] == "BALL_SCRAP"
    assert entry["updated_by"] == "qa"


def test_correction_requires_superclass_for_unknown_class(catalog_settings):
    service = DatasetService(catalog_settings.data_dir)
    _make_object(service, "balls", "obj_1", "take_1")

    with pytest.raises(ValueError, match="not in label_taxonomy yet"):
        correct_physical_object_label(
            data_dir=catalog_settings.data_dir, dataset_id="balls", physical_object_id="obj_1",
            normalized_class="oxidada", superclass=None, raw_label="Oxidada", actor="qa", reason=None,
        )
