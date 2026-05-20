from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from vision_3d_acquisition.apps.ball_inspection import run_ball_inspection_flow
from vision_3d_acquisition.contracts.metadata import AcquisitionMetadata
from vision_3d_acquisition.contracts.result import ProcessingResult
from vision_3d_acquisition.contracts.modalities import infer_modalities
from vision_3d_acquisition.processing.pipeline import ProcessingConfig, process_take
from vision_3d_acquisition.vision_core.pipelines import PipelineContext, PipelineRunner


@dataclass
class _Stage:
    name: str
    category: str = "classification"
    marker: str = ""

    def run(self, context: PipelineContext) -> None:
        order = list(context.get_artifact("order", []))
        order.append(self.marker or self.name)
        context.set_artifact("order", order)


def _write_synthetic_ply(path: Path) -> None:
    points: list[tuple[float, float, float]] = []
    for x in range(-4, 5):
        for y in range(-4, 5):
            points.append((float(x), float(y), 0.0))
    for base_x in (-12.0, 12.0):
        for dx in (0.0, 0.6, 1.2):
            for dy in (0.0, 0.6, 1.2):
                for dz in (0.0, 0.5):
                    points.append((base_x + dx, dy, 3.0 + dz))
    path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                f"element vertex {len(points)}",
                "property float x",
                "property float y",
                "property float z",
                "end_header",
                *[f"{x} {y} {z}" for x, y, z in points],
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_take(data_dir: Path, take_id: str = "ball_take_001") -> Path:
    take_dir = data_dir / "incoming" / take_id
    take_dir.mkdir(parents=True, exist_ok=True)
    _write_synthetic_ply(take_dir / "point_cloud.ply")
    (take_dir / "metadata.json").write_text(
        json.dumps(
            {
                "take_id": take_id,
                "source": "test",
                "mode": "offline",
                "created_at": "2026-05-16T00:00:00Z",
                "frame_count": 1,
                "files": {"point_cloud": "point_cloud.ply"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (take_dir / "READY").touch()
    return take_dir


def _write_image_take(data_dir: Path, take_id: str = "image_take_001") -> Path:
    take_dir = data_dir / "incoming" / take_id
    take_dir.mkdir(parents=True, exist_ok=True)
    (take_dir / "rgb.png").write_bytes(b"not-a-real-image")
    (take_dir / "metadata.json").write_text(
        json.dumps(
            {
                "take_id": take_id,
                "source": "test",
                "mode": "offline",
                "created_at": "2026-05-16T00:00:00Z",
                "frame_count": 1,
                "files": {"rgb": "rgb.png"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (take_dir / "READY").touch()
    return take_dir


def test_pipeline_runner_executes_stages_in_order() -> None:
    context = PipelineContext()
    runner = PipelineRunner(
        [
            _Stage(name="first", marker="a"),
            _Stage(name="second", marker="b"),
            _Stage(name="third", marker="c"),
        ]
    )
    result = runner.run(context)
    assert result.artifacts["order"] == ["a", "b", "c"]


def test_pipeline_context_stores_and_retrieves_artifacts() -> None:
    context = PipelineContext()
    context.set_artifact("value", {"key": 42})
    assert context.get_artifact("value") == {"key": 42}
    assert context.require_artifact("value")["key"] == 42


def test_metadata_modality_detection_and_old_file_compatibility() -> None:
    old_payload = {
        "take_id": "old_take",
        "source": "test",
        "mode": "offline",
        "created_at": "2026-05-16T00:00:00Z",
        "frame_count": 1,
        "files": {"point_cloud": "point_cloud.ply"},
    }
    validated = AcquisitionMetadata.model_validate(old_payload)
    assert validated.modalities == ["point_cloud"]
    assert validated.frameset is not None
    assert validated.frameset.frame_count == 1
    assert infer_modalities({"files": {"rgb": "rgb.png", "reflectance": "reflectance.png"}}) == ["reflectance", "rgb"]


def test_vision_core_layers_do_not_import_ball_app() -> None:
    core_root = Path(__file__).resolve().parents[1] / "vision_3d_acquisition" / "vision_core"
    for path in core_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "apps.ball_inspection" not in text
        assert "ball_inspection" not in text


def test_ball_pipeline_runs_and_preserves_result_contract(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_take(data_dir)
    config_path = tmp_path / "processing.yaml"
    config_path.write_text(
        "\n".join(
            [
                "voxel_size: 0.0",
                "enable_outlier_removal: false",
                "plane_distance_threshold: 0.2",
                "plane_ransac_n: 3",
                "plane_num_iterations: 200",
                "dbscan_eps: 1.6",
                "dbscan_min_points: 4",
                "min_cluster_points: 6",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_ball_inspection_flow(data_dir, config_path=config_path, skip_debug_images=True)
    result_json = result.output_dir / "result.json"
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    validated = ProcessingResult.model_validate(payload)

    assert validated.take_id == "ball_take_001"
    assert validated.processing_mode == "real"
    assert validated.algorithm_stage == "classification"
    assert validated.summary.object_count == len(validated.objects)
    assert validated.files.point_cloud == "point_cloud.ply"
    assert validated.input_modalities == ["point_cloud"]
    assert validated.processing_pipeline is not None
    assert validated.processing_pipeline.required_modalities == ["point_cloud"]
    assert validated.calibration is not None
    assert validated.calibration.calibration_type == "plane_3d"
    assert validated.timing_ms.total >= 0
    assert validated.profiling is not None
    assert any(stage.name == "PlaneFilterStage" for stage in validated.profiling.stages)
    assert not any(stage.name == "RunLegacySegmentationStage" for stage in validated.profiling.stages)
    assert (result.output_dir / "DONE").is_file()


def test_ball_pipeline_rejects_image_only_take_clearly(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_image_take(data_dir)

    with pytest.raises(ValueError, match="requires point_cloud but take has rgb"):
        run_ball_inspection_flow(data_dir, skip_debug_images=True)


def test_stage_modality_error_is_recorded_in_profile() -> None:
    context = PipelineContext()
    context.set_artifact("modalities", ["rgb"])
    runner = PipelineRunner([_Stage(name="needs_cloud")])
    runner.stages[0].required_modalities = ("point_cloud",)  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="requires point_cloud"):
        runner.run(context)

    profile = context.profiler.as_dict()
    assert profile["stages"][0]["metadata"]["modality_error"] == "Stage needs_cloud requires point_cloud but take has rgb"


def test_ball_pipeline_matches_legacy_core_fields_on_fixture(tmp_path: Path) -> None:
    legacy_data = tmp_path / "legacy_data"
    new_data = tmp_path / "new_data"
    legacy_take = _write_take(legacy_data, take_id="fixture_take_001")
    _write_take(new_data, take_id="fixture_take_001")

    legacy_output = legacy_data / "processed" / "fixture_take_001"
    legacy_result = process_take(
        legacy_take,
        legacy_output,
        config=ProcessingConfig(
            voxel_size=0.0,
            enable_outlier_removal=False,
            plane_distance_threshold=0.2,
            plane_ransac_n=3,
            plane_num_iterations=200,
            dbscan_eps=1.6,
            dbscan_min_points=4,
            min_cluster_points=6,
        ),
        skip_debug_images=True,
    )

    config_path = tmp_path / "new_processing.yaml"
    config_path.write_text(
        "\n".join(
            [
                "voxel_size: 0.0",
                "enable_outlier_removal: false",
                "plane_distance_threshold: 0.2",
                "plane_ransac_n: 3",
                "plane_num_iterations: 200",
                "dbscan_eps: 1.6",
                "dbscan_min_points: 4",
                "min_cluster_points: 6",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    new_result = run_ball_inspection_flow(new_data, config_path=config_path, skip_debug_images=True)
    new_payload = json.loads((new_result.output_dir / "result.json").read_text(encoding="utf-8"))
    new_validated = ProcessingResult.model_validate(new_payload)

    assert new_validated.take_id == legacy_result.take_id
    assert new_validated.processing_mode == legacy_result.processing_mode == "real"
    assert tuple(new_validated.plane_model or ()) == tuple(legacy_result.plane_model or ())
    assert new_validated.input_stats is not None
    assert legacy_result.input_stats is not None
    assert new_validated.input_stats.point_count == legacy_result.input_stats.point_count
    assert len(new_validated.objects) == len(legacy_result.objects)
    assert new_validated.files.point_cloud == legacy_result.files.point_cloud
    assert new_validated.profiling is not None
    assert any(stage.name == "SerializeProcessingResultStage" for stage in new_validated.profiling.stages)
