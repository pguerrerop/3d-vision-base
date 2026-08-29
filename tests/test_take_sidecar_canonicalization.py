"""One sidecar per take, enforced where the sidecar is written.

A take used to be able to hold a metadata.json under two sessions at once, and
which one the API showed depended on Path.glob iteration order. A bulk
move_to_session created 421 of those in production: the destination sidecar was
seeded from the defaults, so the labels stayed behind in the origin and became
invisible.
"""

from __future__ import annotations

import json
from pathlib import Path

from vision_3d_acquisition.api.main import TakeBulkMetadataRequest, bulk_update_take_metadata
from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.datasets import DatasetService


def _sidecars(settings: ApiSettings, take_id: str) -> list[Path]:
    return sorted(
        settings.datasets_dir.glob(f"dataset_*/sessions/session_*/takes/{take_id}/metadata.json")
    )


def _labeled_take(settings: ApiSettings, take_id: str) -> DatasetService:
    service = DatasetService(settings.data_dir)
    service.create_dataset(dataset_id="demo", name="Demo")
    service.create_session(dataset_id="demo", session_id="origen", name="Origen")
    service.create_session(dataset_id="demo", session_id="destino", name="Destino")
    service.upsert_take_metadata(
        take_id=take_id,
        dataset_id="demo",
        session_id="origen",
        updates={
            "semantic_labels": ["chatarra"],
            "superclass_labels": ["SCRAP_METAL"],
            "validation_status": "valid",
            "physical_object_id": "obj_0042",
            "expected_class": "chatarra",
        },
    )
    return service


def test_filing_a_take_under_a_new_session_carries_its_content(
    catalog_settings: ApiSettings, write_take
) -> None:
    write_take("take_a")
    service = _labeled_take(catalog_settings, "take_a")

    moved = service.upsert_take_metadata(
        take_id="take_a", dataset_id="demo", session_id="destino", updates={"session_id": "destino"}
    )

    assert moved["semantic_labels"] == ["chatarra"]
    assert moved["validation_status"] == "valid"
    assert moved["physical_object_id"] == "obj_0042"
    assert moved["session_id"] == "destino"


def test_a_take_never_keeps_two_sidecars(catalog_settings: ApiSettings, write_take) -> None:
    write_take("take_a")
    service = _labeled_take(catalog_settings, "take_a")
    assert len(_sidecars(catalog_settings, "take_a")) == 1

    service.upsert_take_metadata(
        take_id="take_a", dataset_id="demo", session_id="destino", updates={"session_id": "destino"}
    )

    remaining = _sidecars(catalog_settings, "take_a")
    assert len(remaining) == 1
    assert "session_destino" in remaining[0].parts
    assert service.resolve_all_take_memberships("take_a") == [("demo", "destino")]


def test_bulk_move_to_session_preserves_labels(
    catalog_settings: ApiSettings, write_take, monkeypatch
) -> None:
    """The exact regression: 421 takes lost their labels through this endpoint."""
    write_take("take_a")
    _labeled_take(catalog_settings, "take_a")
    monkeypatch.setattr(
        "vision_3d_acquisition.api.main.get_settings", lambda: catalog_settings, raising=False
    )

    response = bulk_update_take_metadata(
        TakeBulkMetadataRequest(
            mode="ids", action="move_to_session", take_ids=["take_a"], session_id="destino"
        ),
        settings=catalog_settings,
    )

    assert response["ok"] is True
    assert response["affected_count"] == 1

    sidecars = _sidecars(catalog_settings, "take_a")
    assert len(sidecars) == 1
    payload = json.loads(sidecars[0].read_text(encoding="utf-8"))
    assert payload["session_id"] == "destino"
    assert payload["semantic_labels"] == ["chatarra"]
    assert payload["superclass_labels"] == ["SCRAP_METAL"]
    assert payload["validation_status"] == "valid"
    assert payload["physical_object_id"] == "obj_0042"


def test_update_take_session_id_still_moves_and_prunes(
    catalog_settings: ApiSettings, write_take
) -> None:
    write_take("take_a")
    service = _labeled_take(catalog_settings, "take_a")

    updated = service.update_take_session_id(
        take_id="take_a", dataset_id="demo", new_session_id="destino"
    )

    assert updated["session_id"] == "destino"
    assert updated["semantic_labels"] == ["chatarra"]
    assert len(_sidecars(catalog_settings, "take_a")) == 1
