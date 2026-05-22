from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from vision_3d_acquisition.acquisition.process_integration import (
    normalize_processing_modality,
    process_acquired_take_if_bound,
)
from vision_3d_acquisition.acquisition.processing_contracts import AcquiredTakeReady
from vision_3d_acquisition.acquisition.processing_status import read_status
from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.processes.bindings import ProcessBindingService


def make_settings(data_dir: Path) -> ApiSettings:
    settings = ApiSettings(
        data_dir=data_dir,
        incoming_dir=data_dir / "incoming",
        processed_dir=data_dir / "processed",
        state_dir=data_dir / "state",
        events_dir=data_dir / "events",
        sessions_dir=data_dir / "sessions",
        datasets_dir=data_dir / "datasets",
    )
    settings.ensure_directories()
    return settings


def _write_recipe(settings: ApiSettings, recipe_id: str, pipeline_id: str) -> None:
    recipes_dir = settings.data_dir / "processes" / "recipes"
    recipes_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": recipe_id,
        "pipeline_instance_id": "instance_test",
        "source_run_id": None,
        "type": "production",
        "notes": "test recipe",
        "created_at": datetime.now(UTC).isoformat(),
        "template_id": "synthetic_template",
        "pipeline_snapshot": {"pipeline_id": pipeline_id},
    }
    (recipes_dir / f"{recipe_id}.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_take(settings: ApiSettings, take_id: str, modality: str) -> None:
    take_dir = settings.incoming_dir / take_id
    take_dir.mkdir(parents=True, exist_ok=True)
    (take_dir / "metadata.json").write_text(
        json.dumps(
            {
                "take_id": take_id,
                "source": "source_1",
                "created_at": "2026-05-21T10:00:00Z",
                "modalities": [modality],
                "files": {"heightmap": "heightmap.npz"},
            }
        ),
        encoding="utf-8",
    )
    (take_dir / "heightmap.npz").write_bytes(b"fake")
    (take_dir / "READY").touch()


def test_modality_normalization_rules() -> None:
    assert normalize_processing_modality("rgb") == ("rgb", None)
    assert normalize_processing_modality("heightmap") == ("heightmap", None)
    assert normalize_processing_modality("heightmap_2_5d") == ("heightmap", "heightmap_2_5d")


def test_no_binding_returns_warning_without_failure(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    _write_take(settings, "take_no_binding", "heightmap")
    acquired = AcquiredTakeReady(
        take_id="take_no_binding",
        source_id="source_1",
        modality="heightmap",
        modality_family="25d",
        asset_paths={"heightmap": str(settings.incoming_dir / "take_no_binding" / "heightmap.npz")},
    )
    decision = process_acquired_take_if_bound(settings, acquired, auto_process=True)
    assert decision.ok is True
    assert decision.status == "processing_binding_missing"
    assert "No active processing binding found" in (decision.warning or "")
    assert acquired.asset_paths["heightmap"].endswith("heightmap.npz")


def test_bound_25d_pipeline_dispatch_preserves_recipe_and_metadata(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path / "data")
    take_id = "take_bound_25d"
    _write_take(settings, take_id, "heightmap")
    _write_recipe(settings, "recipe_25d", "mining_steel_ball_classification_25d")
    binding = ProcessBindingService(settings).create_binding(
        name="bound-25d",
        source_id="source_1",
        modality="heightmap",
        purpose="acquisition_inspection",
        pipeline_id="mining_steel_ball_classification_25d",
        active_recipe_version_id="recipe_25d",
        calibration_profile_id="cal_1",
        enabled=True,
    )

    captured: dict[str, object] = {}

    def _fake_dispatch(**kwargs):
        captured.update(kwargs)
        return {"ok": True, "status": "success", "result": {"config_snapshot_hash": "abc123"}}

    monkeypatch.setattr("vision_3d_acquisition.acquisition.process_integration.dispatch_take_processing", _fake_dispatch)

    acquired = AcquiredTakeReady(
        take_id=take_id,
        source_id="source_1",
        modality="heightmap_2_5d",
        modality_family="25d",
        acquisition_group_id="group_99",
        acquisition_process_id="proc_7",
        acquisition_run_id="run_8",
        asset_paths={"heightmap": str(settings.incoming_dir / take_id / "heightmap.npz")},
    )
    decision = process_acquired_take_if_bound(settings, acquired, auto_process=True)

    assert decision.ok is True
    assert decision.status == "processing_completed"
    assert decision.pipeline_id == "mining_steel_ball_classification_25d"
    assert decision.pipeline_id == binding["pipeline_id"]
    assert captured["pipeline_id"] == "mining_steel_ball_classification_25d"
    assert captured["recipe_version_id"] == "recipe_25d"
    assert captured["acquisition_group_id"] == "group_99"
    assert captured["reprocess"] is True

    status = read_status(settings.data_dir, take_id) or {}
    assert status.get("status") == "processing_completed"
    run_metadata = status.get("run_metadata") if isinstance(status.get("run_metadata"), dict) else {}
    assert run_metadata.get("acquisition_group_id") == "group_99"
    assert run_metadata.get("acquisition_process_id") == "proc_7"
    assert run_metadata.get("acquisition_run_id") == "run_8"
    assert run_metadata.get("modality") == "heightmap"
    metadata = decision.metadata
    assert metadata.get("original_modality") == "heightmap_2_5d"
    assert acquired.asset_paths["heightmap"].endswith("heightmap.npz")
