from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi import HTTPException
from starlette.requests import Request
from PIL import Image

from vision_3d_acquisition.api.filesystem import safe_take_file
from vision_3d_acquisition.api.feature_catalog import (
    FEATURE_REGISTRY,
    discover_numeric_object_feature_keys,
    feature_definition_for_key,
    stable_schema_feature_keys,
)
from vision_3d_acquisition.api.histogram import load_or_compute_histogram, resolve_source_image_path
from vision_3d_acquisition.api.main import (
    feature_analytics_distributions,
    feature_analytics_features,
    feature_analytics_objects,
    health,
    latest,
    list_operations_cards,
    pipeline_processing_units,
    pipelines,
    runtime_mjpeg_stream,
    runtime_preview_metadata,
    runtime_process_events,
    runtime_process_ftp_status,
    dataset_label_summary,
    runtime_process_logs,
    runtime_process_restart,
    runtime_process_start,
    runtime_process_status,
    runtime_process_stop,
    runtime_processes,
    session_summary_endpoint,
    sessions,
    source_controls,
    sources,
    take_object_thumbnail,
    take_object_thumbnail_info,
    take_detail,
    takes,
    update_source_controls,
    parse_feature_analytics_query,
    pipeline_runs,
    pipeline_run_lineage,
    pipeline_run_generate_comparison,
)
from vision_3d_acquisition.api.settings import ApiSettings, get_cors_origins, get_settings
from vision_3d_acquisition.processing.status_index import append_process_run_index
from vision_3d_acquisition.api.main import SourceControlsUpdateRequest
from vision_3d_acquisition.datasets import DatasetService


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


def make_test_settings(data_dir: Path) -> ApiSettings:
    settings = make_settings(data_dir)
    get_settings.cache_clear()
    return settings


def write_take(data_dir: Path, take_id: str, created_at: str, session_id: str | None = None) -> None:
    take_dir = data_dir / "incoming" / take_id
    take_dir.mkdir(parents=True)
    (take_dir / "point_cloud.ply").write_text("ply\nformat ascii 1.0\nend_header\n", encoding="utf-8")
    (take_dir / "metadata.json").write_text(
        json.dumps(
            {
                "take_id": take_id,
                "source": "test",
                "mode": "offline",
                "created_at": created_at,
                "frame_count": 1,
                "session_id": session_id,
                "files": {"point_cloud": "point_cloud.ply"},
            }
        ),
        encoding="utf-8",
    )
    (take_dir / "READY").touch()


def write_result(data_dir: Path, take_id: str) -> None:
    result_dir = data_dir / "processed" / take_id
    result_dir.mkdir(parents=True)
    (result_dir / "result.json").write_text(
        json.dumps(
            {
                "take_id": take_id,
                "processed_at": "2026-05-16T18:00:00Z",
                "status": "ok",
                "summary": {
                    "object_count": 1,
                    "ball_count": 1,
                    "non_ball_count": 0,
                    "decision": "accept",
                    "confidence": 0.91,
                },
                "objects": [],
                "files": {"overlay": "overlay.png", "point_cloud": "point_cloud.ply"},
                "timing_ms": {"load": 1, "segmentation": 2, "classification": 3, "total": 6},
                "object_candidates": [],
                "error": None,
            }
        ),
        encoding="utf-8",
    )
    (result_dir / "overlay.png").write_bytes(b"fake")
    (result_dir / "DONE").touch()


def test_health_endpoint(tmp_path: Path) -> None:
    settings = make_test_settings(tmp_path / "data")

    response = health(settings)

    payload = response.model_dump()
    assert payload["status"] == "ok"
    assert payload["incoming_count"] == 0
    assert payload["processed_count"] == 0


def test_pipeline_processing_units_endpoint_returns_full_25d_contract() -> None:
    payload = pipeline_processing_units("mining_steel_ball_classification_25d")

    assert payload["pipeline_id"] == "mining_steel_ball_classification_25d"
    units = payload["processing_units"]
    assert isinstance(units, list)
    assert any(unit.get("id") == "input" for unit in units)
    assert any(unit.get("id") == "detect_belt_plane.candidate_support_refinement" for unit in units)
    assert any(unit.get("id") == "normalize_heights_to_plane.normalization_diagnostics" for unit in units)
    assert any(unit.get("id") == "remove_belt_segment_objects.morphology_cleanup" for unit in units)
    assert any(unit.get("id") == "geometry.ellipse_fitting" for unit in units)
    assert any(unit.get("id") == "measurement_diagnostics.known_object_validation" for unit in units)
    assert any(unit.get("id") == "classification.explanation_generation" for unit in units)
    assert any(unit.get("id") == "overlay.classification_overlay" for unit in units)
    roi_unit = next(unit for unit in units if unit.get("id") == "detect_belt_plane.roi")
    params = {item["id"]: item for item in roi_unit.get("parameters") or []}
    assert params["reference_surface_region"]["type"] == "roi"


def test_runtime_preview_metadata_endpoint(tmp_path: Path) -> None:
    settings = make_test_settings(tmp_path / "data")
    preview_dir = settings.data_dir / "runtime" / "previews"
    preview_dir.mkdir(parents=True)
    (preview_dir / "usb_camera_0.jpg").write_bytes(b"fake")
    (preview_dir / "usb_camera_0.json").write_text(
        json.dumps(
            {
                "source": "usb_camera",
                "camera_index": 0,
                "timestamp": "2999-01-01T00:00:00+00:00",
                "resolution": [640, 480],
                "stale": False,
                "fps_estimate": 4.8,
            }
        ),
        encoding="utf-8",
    )

    payload = runtime_preview_metadata(settings)

    assert payload["source"] == "usb_camera"
    assert payload["resolution"] == [640, 480]
    assert payload["stale"] is False
    assert payload["fps_estimate"] == 4.8


def test_operations_cards_endpoint_returns_superclass_for_automatic_output(tmp_path: Path) -> None:
    from vision_3d_acquisition.operations.summary import reindex_recent_operations_cards

    settings = make_test_settings(tmp_path / "data")
    take_id = "take_auto_25d"
    incoming = settings.data_dir / "incoming" / take_id
    processed = settings.data_dir / "processed" / take_id
    incoming.mkdir(parents=True)
    processed.mkdir(parents=True)
    (incoming / "metadata.json").write_text(
        json.dumps({"take_id": take_id, "created_at": "2026-05-22T11:00:00Z", "modalities": ["heightmap"], "source": "trispector_ftp"}),
        encoding="utf-8",
    )
    (incoming / "runtime_state.json").write_text(
        json.dumps({"take_id": take_id, "state": "completed", "source": "trispector_ftp"}),
        encoding="utf-8",
    )
    (processed / "result.json").write_text(
        json.dumps(
            {
                "take_id": take_id,
                "status": "ok",
                "processed_at": "2026-05-22T11:05:00Z",
                "summary": {"label": "non_ball", "superclass": "SCRAP_METAL", "confidence": 0.77},
                "objects": [{"object_id": 1, "label": "non_ball", "superclass": "SCRAP_METAL", "confidence": 0.77}],
                "files": {},
            }
        ),
        encoding="utf-8",
    )

    reindex_recent_operations_cards(settings.data_dir, limit=10)
    payload = list_operations_cards(limit=20, settings=settings)
    card = next(item for item in payload["cards"] if item["take_id"] == take_id)
    assert card["superclass"] == "SCRAP_METAL"
    assert card["label"] == "non_ball"


def test_sources_endpoint_reports_freshness(tmp_path: Path) -> None:
    settings = make_test_settings(tmp_path / "data")
    preview_dir = settings.data_dir / "runtime" / "previews"
    preview_dir.mkdir(parents=True)
    (preview_dir / "usb_camera_0.jpg").write_bytes(b"fake")
    (preview_dir / "usb_camera_0.json").write_text(
        json.dumps(
            {
                "source": "usb_camera",
                "camera_index": 0,
                "timestamp": "2999-01-01T00:00:00+00:00",
                "resolution": [1280, 720],
                "stale": False,
                "fps_estimate": 30.0,
            }
        ),
        encoding="utf-8",
    )

    payload = sources(settings=settings, max_index=0)

    listed = next(item for item in payload if item["id"] == "usb_camera_0")
    assert listed["modality"] == "rgb"
    assert listed["status"] in {"live", "stale"}
    assert listed["resolution"] == [1280, 720]


def test_runtime_mjpeg_endpoint_uses_multipart_stream(tmp_path: Path, monkeypatch) -> None:
    settings = make_test_settings(tmp_path / "data")

    def fake_stream_frames(*_args, **_kwargs):
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\nfake\r\n"

    monkeypatch.setattr("vision_3d_acquisition.api.main.mjpeg_stream_frames", fake_stream_frames)
    response = runtime_mjpeg_stream(source_id="usb_camera_0", fps=10.0, settings=settings)
    assert response.media_type == "multipart/x-mixed-replace; boundary=frame"


def test_runtime_process_ftp_status_endpoint(tmp_path: Path, monkeypatch) -> None:
    _ = make_test_settings(tmp_path / "data")

    class FakeSupervisor:
        def ftp_status(self, process_id: str) -> dict[str, object]:
            return {"process_id": process_id, "listening": True, "port": 2121, "auth_mode": "anonymous"}

    monkeypatch.setattr("vision_3d_acquisition.api.main.get_runtime_supervisor", lambda: FakeSupervisor())
    payload = runtime_process_ftp_status("trispector_ftp")
    assert payload["process_id"] == "trispector_ftp"
    assert payload["auth_mode"] == "anonymous"


def test_source_controls_endpoint_shape_for_unsupported_source() -> None:
    payload = source_controls("simulated_0")
    exposure = payload["controls"]["exposure"]
    assert exposure["supported"] is False
    assert "readable" in exposure
    assert "writable" in exposure
    assert "backend_property" in exposure


def test_source_id_maps_to_camera_index_zero_for_controls(monkeypatch) -> None:
    seen = {"index": None}
    def fake_snapshot(index: int):
        seen["index"] = index
        return {"backend": "FAKE", "camera_index": index, "controls": {}, "diagnostics": {}}
    monkeypatch.setattr("vision_3d_acquisition.api.main.camera_controls_snapshot", fake_snapshot)
    source_controls("usb_camera_0")
    assert seen["index"] == 0


def test_update_controls_ignores_unsupported(monkeypatch) -> None:
    def fake_apply(index: int, controls):
        return {"backend": "FAKE", "camera_index": index, "controls": {"exposure": {"supported": True}}, "applied": {"foo": {"warnings": ["unsupported_property"]}}}
    monkeypatch.setattr("vision_3d_acquisition.api.main.apply_camera_controls", fake_apply)
    payload = update_source_controls("usb_camera_0", SourceControlsUpdateRequest(controls={"foo": {"value": 1}}))
    assert "applied" in payload


def test_pipelines_endpoint_exposes_registry() -> None:
    payload = pipelines()

    current = next(item for item in payload if item["id"] == "3d_ball_inspection")
    fusion = next(item for item in payload if item["id"] == "2d_3d_fusion")
    assert current["required_modalities"] == ["point_cloud"]
    assert current["stages"][0]["id"] == "segmentation"
    assert fusion["implemented"] is False


def test_dataset_label_summary_endpoint(tmp_path: Path) -> None:
    settings = make_test_settings(tmp_path / "data")
    from vision_3d_acquisition.datasets import DatasetService

    svc = DatasetService(settings.data_dir)
    svc.create_dataset(dataset_id="d1", name="Dataset 1")
    svc.create_session(dataset_id="d1", session_id="s1", name="Session 1")
    svc.upsert_take_metadata(
        take_id="take_001",
        dataset_id="d1",
        session_id="s1",
        updates={"tags": ["ok"], "semantic_labels": ["BALL_GOOD_STANDARD"], "superclass_labels": ["BALL_GOOD"], "normalization_version": "v1"},
        source_metadata={},
    )

    payload = dataset_label_summary("d1", settings)
    assert payload["dataset_id"] == "d1"
    assert payload["raw_tag_counts"]["ok"] == 1


def test_settings_read_ports_from_env_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("API_PORT", raising=False)
    monkeypatch.delenv("FRONTEND_PORT", raising=False)
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    (tmp_path / ".env").write_text("API_PORT=8012\nFRONTEND_PORT=5178\n", encoding="utf-8")
    get_settings.cache_clear()

    try:
        settings = get_settings()

        assert settings.api_port == 8012
        assert settings.frontend_port == 5178
        assert get_cors_origins() == ("http://localhost:5178", "http://127.0.0.1:5178")
    finally:
        get_settings.cache_clear()
        os.environ.pop("API_PORT", None)
        os.environ.pop("FRONTEND_PORT", None)


def test_takes_listing(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    write_take(data_dir, "take_a", "2026-05-16T17:00:00Z")
    write_take(data_dir, "take_b", "2026-05-16T18:00:00Z")
    write_result(data_dir, "take_b")
    settings = make_test_settings(data_dir)

    response = takes(None, settings)

    payload = [item.model_dump() for item in response]
    assert [item["take_id"] for item in payload] == ["take_b", "take_a"]
    assert payload[0]["status"] == "processed"
    assert payload[0]["decision"] == "accept"


def test_take_detail_includes_object_candidates(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    write_take(data_dir, "take_objcand", "2026-05-16T17:00:00Z")
    write_result(data_dir, "take_objcand")
    settings = make_test_settings(data_dir)

    detail = take_detail("take_objcand", settings)
    assert detail.result is not None
    assert isinstance(detail.result.get("object_candidates"), list)


def test_session_listing_and_take_filtering(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    write_take(data_dir, "take_s1", "2026-05-16T17:00:00Z", session_id="session_a")
    write_take(data_dir, "take_s2", "2026-05-16T18:00:00Z", session_id="session_b")
    session_dir = data_dir / "sessions" / "session_a"
    (session_dir / "takes").mkdir(parents=True, exist_ok=True)
    (session_dir / "metadata.json").write_text(
        json.dumps(
            {
                "session_id": "session_a",
                "operator": "tester",
                "acquisition_mode": "live",
                "created_at": "2026-05-16T16:00:00Z",
                "take_ids": ["take_s1"],
            }
        ),
        encoding="utf-8",
    )
    settings = make_test_settings(data_dir)

    filtered = takes("session_a", settings)
    listed_sessions = sessions(settings)
    summary = session_summary_endpoint("session_a", settings)

    assert len(filtered) == 1
    assert filtered[0].take_id == "take_s1"
    assert listed_sessions[0]["session_id"] == "session_a"
    assert summary["take_count"] == 1


def test_latest_take_prefers_processed_done(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    write_take(data_dir, "incoming_newer", "2026-05-16T19:00:00Z")
    write_take(data_dir, "processed_older", "2026-05-16T18:00:00Z")
    write_result(data_dir, "processed_older")
    settings = make_test_settings(data_dir)

    response = latest(settings)

    assert response is not None
    assert response.take_id == "processed_older"


def test_file_serving_rejects_path_traversal(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    write_take(data_dir, "take_a", "2026-05-16T17:00:00Z")
    settings = make_test_settings(data_dir)

    response = safe_take_file(settings, "take_a", "../metadata.json")

    assert response is None


def test_file_serving_resolves_reused_artifact_path_for_child_run(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    write_take(data_dir, "take_b", "2026-05-16T17:00:00Z")
    settings = make_test_settings(data_dir)

    run_dir = data_dir / "processes" / "runs" / "child_run_1"
    reused_dir = run_dir / "reused" / "detect_belt_plane"
    reused_dir.mkdir(parents=True, exist_ok=True)
    (reused_dir / "belt_plane.json").write_text("{}", encoding="utf-8")
    append_process_run_index(
        data_dir,
        take_id="take_b",
        pipeline_instance_id="partial_rerun_25d",
        run_id="child_run_1",
        pipeline_family="25d",
        status="completed",
        run_dir=run_dir,
        created_at="2026-05-16T18:00:00Z",
        pipeline_id="mining_steel_ball_classification_25d",
    )

    resolved = safe_take_file(settings, "take_b", "reused/detect_belt_plane/belt_plane.json")
    assert resolved is not None
    assert resolved.is_file()

    traversal_attempt = safe_take_file(settings, "take_b", "reused/../../../../etc/passwd")
    assert traversal_attempt is None


def test_pipeline_runs_endpoint_lists_full_and_partial_runs(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    settings = make_test_settings(data_dir)
    pipeline_id = "mining_steel_ball_classification_25d"

    for run_id, kwargs in (
        ("run_full", {}),
        ("run_child", {"execution_mode": "rerun_from_public_stage_boundary", "parent_run_id": "run_full"}),
    ):
        run_dir = data_dir / "processes" / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        append_process_run_index(
            data_dir,
            take_id="take_runs",
            pipeline_instance_id="instance_25d",
            run_id=run_id,
            pipeline_family="25d",
            status="completed",
            run_dir=run_dir,
            created_at="2026-05-16T18:00:00Z" if run_id == "run_full" else "2026-05-16T19:00:00Z",
            pipeline_id=pipeline_id,
            **kwargs,
        )

    response = pipeline_runs(pipeline_id, take_id="take_runs", settings=settings)

    assert response["pipeline_id"] == pipeline_id
    run_ids = [entry["run_id"] for entry in response["runs"]]
    assert run_ids == ["run_child", "run_full"]
    types_by_id = {entry["run_id"]: entry["run_type"] for entry in response["runs"]}
    assert types_by_id["run_full"] == "full_run"
    assert types_by_id["run_child"] == "partial_rerun"


def test_pipeline_run_lineage_endpoint_returns_parent_and_children(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    settings = make_test_settings(data_dir)
    pipeline_id = "mining_steel_ball_classification_25d"

    for run_id, kwargs in (
        ("run_full", {}),
        ("run_child", {"execution_mode": "rerun_from_public_stage_boundary", "parent_run_id": "run_full"}),
    ):
        run_dir = data_dir / "processes" / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        append_process_run_index(
            data_dir,
            take_id="take_lineage",
            pipeline_instance_id="instance_25d",
            run_id=run_id,
            pipeline_family="25d",
            status="completed",
            run_dir=run_dir,
            created_at="2026-05-16T18:00:00Z",
            pipeline_id=pipeline_id,
            **kwargs,
        )

    child_lineage = pipeline_run_lineage(pipeline_id, "run_child", settings=settings)
    parent_lineage = pipeline_run_lineage(pipeline_id, "run_full", settings=settings)

    assert child_lineage["parent"]["run_id"] == "run_full"
    assert child_lineage["children"] == []
    assert parent_lineage["parent"] is None
    assert [entry["run_id"] for entry in parent_lineage["children"]] == ["run_child"]


def test_pipeline_run_generate_comparison_endpoint_error_handling(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    settings = make_test_settings(data_dir)
    pipeline_id = "mining_steel_ball_classification_25d"

    run_dir = data_dir / "processes" / "runs" / "run_no_parent"
    run_dir.mkdir(parents=True, exist_ok=True)
    append_process_run_index(
        data_dir,
        take_id="take_lineage",
        pipeline_instance_id="instance_25d",
        run_id="run_no_parent",
        pipeline_family="25d",
        status="completed",
        run_dir=run_dir,
        created_at="2026-05-16T18:00:00Z",
        pipeline_id=pipeline_id,
    )

    with pytest.raises(HTTPException) as missing_run_exc:
        pipeline_run_generate_comparison(pipeline_id, "does_not_exist", settings=settings)
    assert missing_run_exc.value.status_code == 404

    with pytest.raises(HTTPException) as no_parent_exc:
        pipeline_run_generate_comparison(pipeline_id, "run_no_parent", settings=settings)
    assert no_parent_exc.value.status_code == 400


def test_take_detail_endpoint_accepts_run_id_query_param(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    write_take(data_dir, "take_runid", "2026-05-16T17:00:00Z")
    write_result(data_dir, "take_runid")
    settings = make_test_settings(data_dir)

    detail_default = take_detail("take_runid", settings)
    detail_latest = take_detail("take_runid", settings, run_id="latest")

    assert detail_default is not None
    assert detail_latest is not None
    assert detail_default.result is not None
    assert detail_latest.result is not None
    assert detail_default.result.get("status") == detail_latest.result.get("status")


def test_take_detail_endpoint_legacy_behavior_unchanged(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    write_take(data_dir, "take_legacy", "2026-05-16T17:00:00Z")
    write_result(data_dir, "take_legacy")
    settings = make_test_settings(data_dir)

    detail = take_detail("take_legacy", settings)

    assert detail is not None
    assert detail.take_id == "take_legacy"
    assert detail.result is not None
    assert isinstance(detail.result.get("artifacts"), list)


def test_take_detail_normalizes_artifacts(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    write_take(data_dir, "take_norm", "2026-05-16T17:00:00Z")
    write_result(data_dir, "take_norm")
    settings = make_test_settings(data_dir)

    detail = take_detail("take_norm", settings)

    assert detail is not None
    artifacts = (detail.result or {}).get("artifacts") or []
    assert any(item.get("artifact_id") == "result_payload" for item in artifacts)
    assert any(item.get("stage_id") == "classification" for item in artifacts)


def test_take_status_is_contextual_by_pipeline_family(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    write_take(data_dir, "take_ctx", "2026-05-16T17:00:00Z")
    write_result(data_dir, "take_ctx")
    index_dir = data_dir / "processes" / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "runs.json").write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "take_id": "take_ctx",
                        "pipeline_instance_id": "pipeline_1",
                        "run_id": "run_1",
                        "pipeline_family": "2d",
                        "status": "success",
                        "created_at": "2026-05-16T18:10:00Z",
                        "path": "data/processes/runs/pipeline_1/run_1",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    settings = make_test_settings(data_dir)

    summary = takes(None, settings)[0].model_dump()

    by_family = {item["family"]: item for item in summary["processing_by_family"]}
    assert by_family["3d"]["hasCompletedOutput"] is True
    assert by_family["2d"]["hasCompletedOutput"] is True


def test_source_histogram_resolves_rgb_and_computes_cached_payload(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    take_id = "take_rgb_hist"
    take_dir = data_dir / "incoming" / take_id
    take_dir.mkdir(parents=True, exist_ok=True)
    image_path = take_dir / "rgb.png"
    Image.new("RGB", (32, 24), (120, 140, 160)).save(image_path)
    (take_dir / "metadata.json").write_text(
        json.dumps(
            {
                "take_id": take_id,
                "created_at": "2026-05-18T10:00:00Z",
                "files": {"rgb": "rgb.png"},
            }
        ),
        encoding="utf-8",
    )
    settings = make_test_settings(data_dir)
    detail = take_detail(take_id, settings)
    assert detail is not None

    resolved_path, filename = resolve_source_image_path(settings, detail)
    assert resolved_path is not None
    assert filename == "rgb.png"

    payload = load_or_compute_histogram(settings, take_id, filename, resolved_path, max_dim=32, bins=16)
    assert payload["source"] == "rgb.png"
    assert payload["sampled_pixels"] > 0
    assert len(payload["bins"]) == 16
    assert payload["mean"] > 0

    cache_dir = settings.incoming_dir / take_id / ".stage_cache"
    assert cache_dir.is_dir()
    assert any(path.name.startswith("hist_") for path in cache_dir.iterdir())


def test_runtime_process_endpoints_delegate_to_supervisor(monkeypatch, tmp_path: Path) -> None:
    class FakeSupervisor:
        def list_processes(self):
            return [{"process_id": "trispector_ftp", "status": "stopped"}]

        def status(self, process_id: str):
            return {"process_id": process_id, "status": "stopped"}

        def start(self, process_id: str):
            return {"process_id": process_id, "status": "running"}

        def stop(self, process_id: str):
            return {"process_id": process_id, "status": "stopped"}

        def restart(self, process_id: str):
            return {"process_id": process_id, "status": "running", "restart_count": 1}

        def tail_logs(self, process_id: str, limit: int = 200):
            return {"process_id": process_id, "lines": [f"limit={limit}"]}

        def events(self, process_id: str, limit: int = 200):
            return {"process_id": process_id, "events": [{"event_type": "PROCESS_STARTED", "limit": limit}]}

    monkeypatch.setattr("vision_3d_acquisition.api.main.get_runtime_supervisor", lambda: FakeSupervisor())
    data_dir = tmp_path / "data"
    settings = make_settings(data_dir)
    process_dir = data_dir / "runtime" / "processes"
    process_dir.mkdir(parents=True, exist_ok=True)
    (process_dir / "run_25d_worker.json").write_text(
        json.dumps(
            {
                "process_name": "run_25d_worker",
                "runtime_role": "worker",
                "pid": 1234,
                "status": "running",
                "started_at": "2026-05-22T10:00:00+00:00",
                "last_heartbeat": "2026-05-22T10:00:01+00:00",
                "host": "test-host",
                "version": "poc",
            }
        ),
        encoding="utf-8",
    )
    rows = runtime_processes(settings=settings)
    assert rows[0]["process_name"] == "run_25d_worker"
    assert runtime_process_status("trispector_ftp")["status"] == "stopped"
    assert runtime_process_start("trispector_ftp")["status"] == "running"
    assert runtime_process_stop("trispector_ftp")["status"] == "stopped"
    assert runtime_process_restart("trispector_ftp")["restart_count"] == 1
    assert runtime_process_logs("trispector_ftp", limit=3)["lines"] == ["limit=3"]
    assert runtime_process_events("trispector_ftp", limit=4)["events"][0]["event_type"] == "PROCESS_STARTED"


def test_feature_analytics_endpoints(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    take_id = "take_fa_1"
    write_take(data_dir, take_id, "2026-05-28T12:00:00Z")
    result_dir = data_dir / "processed" / take_id
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "result.json").write_text(
        json.dumps(
            {
                "take_id": take_id,
                "processed_at": "2026-05-28T12:01:00Z",
                "status": "ok",
                "processing_pipeline": {"id": "mining_steel_ball_classification_25d"},
                "summary": {"decision": "accept"},
                "objects": [
                    {
                        "object_id": 1,
                        "class_name": "BALL_GOOD",
                        "superclass": "BALL_GOOD",
                        "feature_eccentricity": 0.22,
                        "feature_sphericity_3d": 0.94,
                        "height_above_belt_mm": {"max": 13.1, "mean": 8.5},
                        "confidence": 0.88,
                    },
                    {
                        "object_id": 2,
                        "class_name": "SCRAP_METAL",
                        "superclass": "SCRAP",
                        "feature_eccentricity": 0.78,
                        "feature_sphericity_3d": 0.46,
                        "height_above_belt_mm": {"max": 6.1, "mean": 3.2},
                        "confidence": 0.55,
                    },
                ],
                "rejected_objects": [],
                "files": {},
                "timing_ms": {"load": 1, "segmentation": 2, "classification": 3, "total": 6},
                "error": None,
            }
        ),
        encoding="utf-8",
    )
    (result_dir / "DONE").touch()
    settings = make_test_settings(data_dir)

    features_payload = feature_analytics_features(settings=settings)
    assert features_payload["record_count"] == 2
    assert any(item["feature_key"] == "feature_eccentricity" for item in features_payload["feature_definitions"])
    assert "debug" in features_payload
    assert features_payload["debug"]["takes_scanned"] >= 1

    distribution_payload = feature_analytics_distributions(feature_key="feature_eccentricity", bins=8, settings=settings)
    assert distribution_payload["feature_key"] == "feature_eccentricity"
    assert len(distribution_payload["groups"]) >= 1
    assert "debug" in distribution_payload

    objects_payload = feature_analytics_objects(feature_key="feature_eccentricity", min_value=0.7, max_value=0.9, settings=settings)
    assert len(objects_payload["objects"]) == 1
    assert objects_payload["objects"][0]["object_id"] == "2"
    assert "debug" in objects_payload


def test_feature_analytics_features_empty_returns_immediately_with_debug(tmp_path: Path) -> None:
    settings = make_test_settings(tmp_path / "data")
    payload = feature_analytics_features(settings=settings, max_takes=20, time_budget_ms=100)
    assert payload["record_count"] == 0
    assert any(item["feature_key"] == "surface_sphere_fit_rmse_mm" for item in payload["feature_definitions"])
    assert payload["debug"]["duration_ms"] >= 0


def test_feature_catalog_registry_marks_stable_and_diagnostic_features() -> None:
    stable_keys = {
        "feature_eccentricity",
        "feature_sphericity_3d",
        "feature_flatness",
        "feature_edge_roughness",
        "feature_volume_proxy_mm3",
        "height_max_mm",
        "height_p95_mm",
        "height_mean_mm",
        "footprint_radial_cv",
        "surface_sphere_fit_rmse_mm",
        "surface_sphere_fit_rmse_norm",
        "surface_sphere_fit_residual_p95_norm",
        "surface_sphere_fit_residual_mad_norm",
        "surface_sphere_radius_error_norm",
        "surface_visible_cap_fraction",
        "surface_volume_fill_ratio",
        "surface_sphere_fit_confidence",
        "damage_flat_region_ratio",
        "damage_surface_discontinuity_score",
    }
    for key in stable_keys:
        entry = FEATURE_REGISTRY[key]
        assert entry.stable_schema is True
        if key != "surface_sphere_fit_confidence":
            assert entry.diagnostic_only is False
        assert entry.formula
        assert entry.algorithm_summary

    confidence_entry = feature_definition_for_key("confidence")
    assert confidence_entry.diagnostic_only is True
    assert confidence_entry.stable_schema is False

    sph3d = feature_definition_for_key("feature_sphericity_3d")
    assert sph3d.display_name == "3D axis balance"
    assert any("not a true sphere-fit metric" in caveat.lower() or "not a true sphere fit metric" in caveat.lower() for caveat in sph3d.caveats)

    sphere_rmse = feature_definition_for_key("surface_sphere_fit_rmse_mm")
    assert sphere_rmse.display_name == "Raw sphere-fit RMSE"
    assert sphere_rmse.family == "sphere_consistency"
    assert sphere_rmse.higher_is_worse is True
    assert "sphere" in sphere_rmse.formula.lower()
    assert "rmse" in sphere_rmse.formula.lower()

    deformation = feature_definition_for_key("surface_deformation_score")
    curvature = feature_definition_for_key("feature_local_curvature_proxy")
    assert deformation.display_name == "Surface deformation score"
    assert curvature.display_name == "Local curvature proxy"
    assert deformation.feature_key != curvature.feature_key
    assert curvature.diagnostic_only is True
    assert curvature.stable_schema is False
    assert stable_schema_feature_keys() == stable_keys | {"surface_deformation_score"}


def test_feature_analytics_feature_payload_includes_catalog_metadata(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    take_id = "take_catalog_1"
    write_take(data_dir, take_id, "2026-05-28T12:00:00Z")
    result_dir = data_dir / "processed" / take_id
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "result.json").write_text(
        json.dumps(
            {
                "take_id": take_id,
                "processed_at": "2026-05-28T12:01:00Z",
                "status": "ok",
                "processing_pipeline": {"id": "mining_steel_ball_classification_25d"},
                "summary": {"decision": "accept"},
                "objects": [
                    {
                        "object_id": 1,
                        "class_name": "BALL_GOOD",
                        "superclass": "BALL_GOOD",
                        "feature_eccentricity": 0.22,
                        "feature_sphericity_3d": 0.94,
                        "feature_flatness": 0.61,
                        "feature_edge_roughness": 0.17,
                        "feature_volume_proxy_mm3": 330.0,
                        "feature_local_curvature_proxy": 0.19,
                        "feature_height_asymmetry": 0.07,
                        "feature_footprint_roundness": 0.88,
                        "height_above_belt_mm": {"max_height_mm": 13.1, "mean_height_mm": 8.5, "p95_height_mm": 12.8},
                        "footprint_geometry": {"radial_cv": 0.08},
                        "surface_geometry": {"sphere_fit_rmse_mm": 1.7, "deformation_score": 0.11},
                        "damage_metrics": {"flat_region_ratio": 0.14, "surface_discontinuity_score": 0.03},
                        "confidence": 0.88,
                    },
                ],
                "rejected_objects": [],
                "files": {},
                "timing_ms": {"load": 1, "segmentation": 2, "classification": 3, "total": 6},
                "error": None,
            }
        ),
        encoding="utf-8",
    )
    (result_dir / "DONE").touch()
    settings = make_test_settings(data_dir)

    payload = feature_analytics_features(settings=settings)
    definitions = {item["feature_key"]: item for item in payload["feature_definitions"]}

    sph3d = definitions["feature_sphericity_3d"]
    assert sph3d["display_name"] == "3D axis balance"
    assert sph3d["family"] == "surface_shape"
    assert sph3d["source_stage"] == "measurement"
    assert sph3d["stable_schema"] is True
    assert sph3d["diagnostic_only"] is False
    assert "min(dim_x_mm, dim_y_mm, dim_z_mm)" in sph3d["formula"]
    assert any("not a true sphere" in caveat.lower() for caveat in sph3d["caveats"])

    confidence = definitions["confidence"]
    assert confidence["diagnostic_only"] is True
    assert confidence["stable_schema"] is False
    assert confidence["family"] == "classification_diagnostics"

    sphere_rmse = definitions["surface_sphere_fit_rmse_mm"]
    # This key is the raw residual. The visible-surface wording is retained only
    # as a legacy alias; changing it would conflate two different semantics.
    assert sphere_rmse["display_name"] == "Raw sphere-fit RMSE"
    assert sphere_rmse["source_stage"] == "measurement"
    assert sphere_rmse["unit"] == "mm"
    assert sphere_rmse["higher_is_worse"] is True

    deformation = definitions["surface_deformation_score"]
    curvature = definitions["feature_local_curvature_proxy"]
    assert deformation["display_name"] == "Surface deformation score"
    assert curvature["display_name"] == "Local curvature proxy"

    for item in payload["feature_definitions"]:
        assert item["feature_key"]
        assert item["display_name"]
        assert item["family"]
        assert item["source_stage"]
        assert item["formula"] or item["algorithm_summary"]


def test_feature_catalog_covers_numeric_object_features() -> None:
    sample_object = {
        "object_id": 1,
        "confidence": 0.81,
        "feature_eccentricity": 0.22,
        "feature_sphericity_3d": 0.94,
        "feature_flatness": 0.61,
        "feature_edge_roughness": 0.17,
        "feature_volume_proxy_mm3": 330.0,
        "feature_local_curvature_proxy": 0.19,
        "feature_height_asymmetry": 0.07,
        "feature_footprint_roundness": 0.88,
        "height_above_belt_mm": {"max_height_mm": 13.1, "mean_height_mm": 8.5, "p95_height_mm": 12.8},
        "footprint_geometry": {"radial_cv": 0.08},
        "surface_geometry": {"sphere_fit_rmse_mm": 1.7, "deformation_score": 0.11},
        "damage_metrics": {"flat_region_ratio": 0.14, "surface_discontinuity_score": 0.03},
        "valid_pixel_ratio": 0.98,
        "plane_residual_std": 0.42,
        "equivalent_diameter_mm": 81.2,
    }
    discovered = discover_numeric_object_feature_keys(sample_object)
    expected = {
        "confidence",
        "feature_eccentricity",
        "feature_sphericity_3d",
        "feature_flatness",
        "feature_edge_roughness",
        "feature_volume_proxy_mm3",
        "feature_local_curvature_proxy",
        "feature_height_asymmetry",
        "feature_footprint_roundness",
        "height_max_mm",
        "height_mean_mm",
        "height_p95_mm",
        "footprint_radial_cv",
        "surface_sphere_fit_rmse_mm",
        "surface_deformation_score",
        "damage_flat_region_ratio",
        "damage_surface_discontinuity_score",
        "valid_pixel_ratio",
        "plane_residual_std",
        "equivalent_diameter_mm",
    }
    assert expected.issubset(discovered)
    assert "feature_local_curvature_proxy" in FEATURE_REGISTRY
    assert "deformation" not in FEATURE_REGISTRY["feature_local_curvature_proxy"].display_name.lower()


def test_feature_analytics_ml_set_and_physical_object_filters_agree(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    dataset_id = "bolas-2-5-1"
    session_id = "bolas_labeled_table_2026_05_25"
    take_id = "2026-05-25T145918_021"
    write_take(data_dir, take_id, "2026-05-25T14:59:18Z")
    service = DatasetService(data_dir)
    service.create_dataset(name="bolas_2.5_1", dataset_id=dataset_id)
    service.create_session(dataset_id=dataset_id, session_id=session_id, name=session_id)
    service.upsert_take_metadata(
        take_id=take_id,
        dataset_id=dataset_id,
        session_id=session_id,
        updates={
            "physical_object_id": "obj_0001",
            "validation_status": "needs_review",
            "normalized_class": "cubo",
            "labels": ["cubo"],
            "superclass_labels": ["SCRAP_METAL"],
        },
        source_metadata={"session_id": session_id, "created_at": "2026-05-25T14:59:18Z"},
    )
    service.create_ml_set(dataset_id=dataset_id, ml_set_id="balls_scrap_2026_05_25_29_table_v1", name="balls_scrap_2026_05_25_29_table_v1", task_type="classification")
    service.add_take_to_ml_set(
        dataset_id=dataset_id,
        ml_set_id="balls_scrap_2026_05_25_29_table_v1",
        take_id=take_id,
        physical_object_id="obj_0001",
        split="unassigned",
        include=True,
        default_trainable=False,
        expected_class="cubo",
    )
    result_dir = data_dir / "processed" / take_id
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "result.json").write_text(
        json.dumps(
            {
                "take_id": take_id,
                "processed_at": "2026-05-29T20:39:43Z",
                "processing_pipeline": {"id": "mining_steel_ball_classification_25d"},
                "objects": [
                    {
                        "object_id": 1,
                        "class_name": "non_ball",
                        "superclass": "SCRAP_METAL",
                        "feature_eccentricity": 0.22,
                        "feature_sphericity_3d": 0.37,
                        "feature_flatness": -0.67,
                        "height_above_belt_mm": {"max": 142.8, "mean": 12.8},
                        "confidence": 0.016,
                    }
                ],
                "rejected_objects": [],
            }
        ),
        encoding="utf-8",
    )
    (result_dir / "DONE").touch()
    settings = make_test_settings(data_dir)
    query = {
        "dataset_id": dataset_id,
        "ml_set_id": "balls_scrap_2026_05_25_29_table_v1",
        "physical_object_ids": ["obj_0001"],
    }

    features_payload = feature_analytics_features(query=query, settings=settings, max_takes=600, time_budget_ms=5000)
    distributions_payload = feature_analytics_distributions(feature_key="feature_eccentricity", query=query, settings=settings, max_takes=600, time_budget_ms=5000)
    objects_payload = feature_analytics_objects(feature_key="feature_eccentricity", query=query, settings=settings, max_takes=600, time_budget_ms=5000)

    assert features_payload["scope_summary"]["object_count"] == 1
    assert features_payload["scope_summary"]["take_count"] == 1
    assert any(item["feature_key"] == "feature_eccentricity" for item in features_payload["feature_definitions"])
    assert distributions_payload["stats"]["count"] == 1
    assert len(distributions_payload["groups"]) == 1
    assert len(objects_payload["objects"]) == 1
    assert objects_payload["objects"][0]["physical_object_id"] == "obj_0001"

    missing_distribution = feature_analytics_distributions(
        feature_key="surface_sphere_fit_rmse_mm",
        query=query,
        settings=settings,
        max_takes=600,
        time_budget_ms=5000,
    )
    assert missing_distribution["feature_key"] == "surface_sphere_fit_rmse_mm"
    assert missing_distribution["groups"] == []
    assert missing_distribution["stats"]["count"] == 0
    assert missing_distribution["stats"]["missing"] == 1


def test_feature_analytics_labeled_superclass_resolution_and_endpoint_consistency(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    dataset_id = "bolas-2-5-1"
    session_id = "bolas_labeled_table_2026_05_25"
    service = DatasetService(data_dir)
    service.create_dataset(name="bolas_2.5_1", dataset_id=dataset_id)
    service.create_session(dataset_id=dataset_id, session_id=session_id, name=session_id)
    service.create_ml_set(dataset_id=dataset_id, ml_set_id="balls_scrap_2026_05_25_29_table_v1", name="balls_scrap_2026_05_25_29_table_v1", task_type="classification")

    rows = [
        {"take_id": "take_good", "physical_object_id": "obj_good", "normalized_class": "buena", "labels": ["buena"], "expected_subclass": None, "processed_superclass": "BALL_GOOD"},
        {"take_id": "take_scrap", "physical_object_id": "obj_scrap", "normalized_class": "mitad", "labels": ["mitad"], "expected_subclass": None, "processed_superclass": "BALL_SCRAP"},
        {"take_id": "take_metal", "physical_object_id": "obj_metal", "normalized_class": "tuerca", "labels": ["tuerca"], "expected_subclass": None, "processed_superclass": "SCRAP_METAL"},
        {"take_id": "take_override", "physical_object_id": "obj_override", "normalized_class": "tuerca", "labels": ["tuerca"], "expected_subclass": "BALL_GOOD", "processed_superclass": "BALL_GOOD"},
        {"take_id": "take_unknown", "physical_object_id": "obj_unknown", "normalized_class": "", "labels": [], "expected_subclass": None, "processed_superclass": "SCRAP_METAL"},
    ]

    for index, row in enumerate(rows, start=1):
        take_id = str(row["take_id"])
        write_take(data_dir, take_id, f"2026-05-25T14:59:{10 + index:02d}Z")
        service.upsert_take_metadata(
            take_id=take_id,
            dataset_id=dataset_id,
            session_id=session_id,
            updates={
                "physical_object_id": row["physical_object_id"],
                "validation_status": "approved",
                "normalized_class": row["normalized_class"] or None,
                "labels": row["labels"],
                "superclass_labels": [],
            },
            source_metadata={"session_id": session_id, "created_at": f"2026-05-25T14:59:{10 + index:02d}Z"},
        )
        service.add_take_to_ml_set(
            dataset_id=dataset_id,
            ml_set_id="balls_scrap_2026_05_25_29_table_v1",
            take_id=take_id,
            physical_object_id=str(row["physical_object_id"]),
            split="train",
            include=True,
            default_trainable=True,
            expected_class=str(row["normalized_class"] or "") or None,
            expected_subclass=str(row["expected_subclass"] or "") or None,
        )
        result_dir = data_dir / "processed" / take_id
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / "result.json").write_text(
            json.dumps(
                {
                    "take_id": take_id,
                    "processed_at": f"2026-05-29T20:39:{10 + index:02d}Z",
                    "processing_pipeline": {"id": "mining_steel_ball_classification_25d"},
                    "objects": [
                        {
                            "object_id": 1,
                            "class_name": "ball" if row["processed_superclass"] == "BALL_GOOD" else "non_ball",
                            "superclass": row["processed_superclass"],
                            "feature_eccentricity": 0.1 * index,
                            "feature_sphericity_3d": 0.15 * index,
                            "confidence": 0.75,
                        }
                    ],
                    "rejected_objects": [],
                }
            ),
            encoding="utf-8",
        )
        (result_dir / "DONE").touch()

    settings = make_test_settings(data_dir)
    base_query = {"dataset_id": dataset_id, "ml_set_id": "balls_scrap_2026_05_25_29_table_v1"}

    distributions_payload = feature_analytics_distributions(
        feature_key="feature_eccentricity",
        group_by="labeled_superclass",
        query=base_query,
        settings=settings,
        max_takes=600,
        time_budget_ms=5000,
    )
    objects_payload = feature_analytics_objects(
        feature_key="feature_eccentricity",
        query=base_query,
        settings=settings,
        max_takes=600,
        time_budget_ms=5000,
    )
    good_only_objects = feature_analytics_objects(
        feature_key="feature_eccentricity",
        query={**base_query, "labeled_superclasses": ["BALL_GOOD"]},
        settings=settings,
        max_takes=600,
        time_budget_ms=5000,
    )

    objects_by_take = {item["take_id"]: item for item in objects_payload["objects"]}
    assert objects_by_take["take_good"]["labeled_superclass"] == "BALL_GOOD"
    assert objects_by_take["take_scrap"]["labeled_superclass"] == "BALL_SCRAP"
    assert objects_by_take["take_metal"]["labeled_superclass"] == "SCRAP_METAL"
    assert objects_by_take["take_override"]["labeled_superclass"] == "BALL_GOOD"
    assert objects_by_take["take_unknown"]["labeled_superclass"] == "UNKNOWN"

    groups = {item["group"]: item["count"] for item in distributions_payload["groups"]}
    assert groups["BALL_GOOD"] == 2
    assert groups["BALL_SCRAP"] == 1
    assert groups["SCRAP_METAL"] == 1
    assert groups["UNKNOWN"] == 1
    assert objects_payload["scope_summary"]["object_count"] == 5
    assert good_only_objects["scope_summary"]["object_count"] == 2
    assert {item["labeled_superclass"] for item in good_only_objects["objects"]} == {"BALL_GOOD"}


def test_parse_feature_analytics_query_accepts_physical_object_variants() -> None:
    request_plural = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/feature-analytics/features",
            "headers": [],
            "query_string": b"physical_object_ids=obj_0001&physical_object_ids=obj_0002",
        }
    )
    request_csv = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/feature-analytics/features",
            "headers": [],
            "query_string": b"physical_object_ids=obj_0001,obj_0002",
        }
    )
    request_singular = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/feature-analytics/features",
            "headers": [],
            "query_string": b"physical_object_id=obj_0001",
        }
    )

    assert parse_feature_analytics_query(request_plural)["physical_object_ids"] == ["obj_0001", "obj_0002"]
    assert parse_feature_analytics_query(request_csv)["physical_object_ids"] == ["obj_0001", "obj_0002"]
    assert parse_feature_analytics_query(request_singular)["physical_object_ids"] == ["obj_0001"]


def test_feature_analytics_reads_surface_sphere_fit_rmse(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    take_id = "take_sphere_rmse_1"
    write_take(data_dir, take_id, "2026-05-28T12:00:00Z")
    result_dir = data_dir / "processed" / take_id
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "result.json").write_text(
        json.dumps(
            {
                "take_id": take_id,
                "processed_at": "2026-05-28T12:01:00Z",
                "status": "ok",
                "run_id": "run_sphere_1",
                "processing_pipeline": {"id": "mining_steel_ball_classification_25d"},
                "summary": {"decision": "accept"},
                "objects": [
                    {
                        "object_id": 1,
                        "class_name": "BALL_GOOD",
                        "superclass": "BALL_GOOD",
                        "feature_eccentricity": 0.22,
                        "surface_geometry": {"sphere_fit_rmse_mm": 1.42},
                        "confidence": 0.88,
                    },
                    {
                        "object_id": 2,
                        "class_name": "SCRAP_METAL",
                        "superclass": "SCRAP",
                        "feature_eccentricity": 0.78,
                        "surface_geometry": {"sphere_fit_rmse_mm": 3.17},
                        "confidence": 0.55,
                    },
                ],
                "rejected_objects": [],
                "files": {},
                "timing_ms": {"load": 1, "segmentation": 2, "classification": 3, "total": 6},
                "error": None,
            }
        ),
        encoding="utf-8",
    )
    (result_dir / "DONE").touch()
    settings = make_test_settings(data_dir)

    distribution_payload = feature_analytics_distributions(
        feature_key="surface_sphere_fit_rmse_mm",
        bins=8,
        settings=settings,
    )
    assert distribution_payload["stats"]["count"] == 2
    assert distribution_payload["stats"]["missing_pct"] < 100.0
    assert len(distribution_payload["groups"]) >= 1

    objects_payload = feature_analytics_objects(
        feature_key="surface_sphere_fit_rmse_mm",
        settings=settings,
    )
    assert len(objects_payload["objects"]) == 2
    assert all(item["feature_value"] is not None for item in objects_payload["objects"])
    assert all(item.get("feature_source") == "pipeline_run" for item in objects_payload["objects"])


def test_feature_analytics_reads_backfilled_surface_sphere_fit_rmse(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    take_id = "take_sphere_backfill_1"
    write_take(data_dir, take_id, "2026-05-28T12:00:00Z")
    result_dir = data_dir / "processed" / take_id
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "result.json").write_text(
        json.dumps(
            {
                "take_id": take_id,
                "processed_at": "2026-05-28T12:01:00Z",
                "status": "ok",
                "run_id": "run_backfill_1",
                "processing_pipeline": {"id": "mining_steel_ball_classification_25d"},
                "summary": {"decision": "accept"},
                "objects": [
                    {
                        "object_id": 1,
                        "class_name": "BALL_GOOD",
                        "superclass": "BALL_GOOD",
                        "feature_eccentricity": 0.22,
                        "confidence": 0.88,
                    },
                ],
                "rejected_objects": [],
                "files": {},
                "timing_ms": {"load": 1, "segmentation": 2, "classification": 3, "total": 6},
                "error": None,
            }
        ),
        encoding="utf-8",
    )
    (result_dir / "feature_backfill_surface_sphere_fit_rmse_mm.json").write_text(
        json.dumps(
            {
                "feature_key": "surface_sphere_fit_rmse_mm",
                "feature_algorithm_version": "surface_sphere_fit_v1",
                "source": "feature_backfill",
                "entries": [
                    {
                        "feature_key": "surface_sphere_fit_rmse_mm",
                        "feature_algorithm_version": "surface_sphere_fit_v1",
                        "source": "feature_backfill",
                        "pipeline_id": "mining_steel_ball_classification_25d",
                        "run_id": "run_backfill_1",
                        "take_id": take_id,
                        "object_id": "1",
                        "value": 2.11,
                        "generated_at": "2026-05-28T12:02:00Z",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (result_dir / "DONE").touch()
    settings = make_test_settings(data_dir)

    distribution_payload = feature_analytics_distributions(
        feature_key="surface_sphere_fit_rmse_mm",
        bins=8,
        settings=settings,
    )
    assert distribution_payload["stats"]["count"] == 1
    assert distribution_payload["stats"]["missing_pct"] == 0.0

    objects_payload = feature_analytics_objects(
        feature_key="surface_sphere_fit_rmse_mm",
        settings=settings,
    )
    assert len(objects_payload["objects"]) == 1
    assert objects_payload["objects"][0]["feature_value"] == 2.11
    assert objects_payload["objects"][0]["feature_source"] == "feature_backfill"


def test_feature_analytics_registered_feature_all_missing_reports_full_missing_pct(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    take_id = "take_missing_sphere_rmse"
    write_take(data_dir, take_id, "2026-05-28T12:00:00Z")
    result_dir = data_dir / "processed" / take_id
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "result.json").write_text(
        json.dumps(
            {
                "take_id": take_id,
                "processed_at": "2026-05-28T12:01:00Z",
                "status": "ok",
                "processing_pipeline": {"id": "mining_steel_ball_classification_25d"},
                "summary": {"decision": "accept"},
                "objects": [
                    {
                        "object_id": 1,
                        "class_name": "BALL_GOOD",
                        "superclass": "BALL_GOOD",
                        "feature_eccentricity": 0.22,
                        "confidence": 0.88,
                    },
                ],
                "rejected_objects": [],
                "files": {},
                "timing_ms": {"load": 1, "segmentation": 2, "classification": 3, "total": 6},
                "error": None,
            }
        ),
        encoding="utf-8",
    )
    (result_dir / "DONE").touch()
    settings = make_test_settings(data_dir)

    features_payload = feature_analytics_features(settings=settings)
    assert any(item["feature_key"] == "surface_sphere_fit_rmse_mm" for item in features_payload["feature_definitions"])
    assert features_payload["scope_summary"]["object_count"] == 1

    distribution_payload = feature_analytics_distributions(
        feature_key="surface_sphere_fit_rmse_mm",
        bins=8,
        settings=settings,
    )
    assert distribution_payload["stats"]["count"] == 0
    assert distribution_payload["stats"]["missing"] == 1
    assert distribution_payload["stats"]["missing_pct"] == 100.0
    assert distribution_payload["groups"] == []


def test_take_object_thumbnail_endpoint_returns_cached_thumb(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    take_id = "take_thumb_1"
    write_take(data_dir, take_id, "2026-05-31T10:00:00Z")
    take_dir = data_dir / "incoming" / take_id
    Image.new("RGB", (180, 120), (120, 150, 180)).save(take_dir / "rgb.png")
    (take_dir / "metadata.json").write_text(
        json.dumps(
            {
                "take_id": take_id,
                "source": "test",
                "mode": "offline",
                "created_at": "2026-05-31T10:00:00Z",
                "frame_count": 1,
                "files": {"rgb": "rgb.png"},
            }
        ),
        encoding="utf-8",
    )
    result_dir = data_dir / "processed" / take_id
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "result.json").write_text(
        json.dumps(
            {
                "take_id": take_id,
                "processed_at": "2026-05-31T10:01:00Z",
                "status": "ok",
                "summary": {"decision": "accept"},
                "objects": [{"object_id": 7, "class_name": "BALL_GOOD", "bbox_px": [40, 20, 30, 24]}],
                "rejected_objects": [],
                "files": {"point_cloud": None, "input_preview": "rgb.png"},
                "timing_ms": {"load": 1, "segmentation": 2, "classification": 3, "total": 6},
                "error": None,
            }
        ),
        encoding="utf-8",
    )
    (result_dir / "DONE").touch()
    settings = make_test_settings(data_dir)

    info = take_object_thumbnail_info(take_id, "7", mode="reflectance", settings=settings)
    assert info["requested_mode"] == "reflectance"
    assert info["resolved_source"] in {"reflectance", "source_crop"}
    assert info["fallback_used"] is False

    response = take_object_thumbnail(take_id, "7", mode="reflectance", settings=settings)
    assert response.path is not None
    thumb_path = Path(str(response.path))
    assert thumb_path.is_file()

    overlay_info = take_object_thumbnail_info(take_id, "7", mode="classification_overlay", settings=settings)
    assert overlay_info["requested_mode"] == "classification_overlay"
    assert overlay_info["fallback_used"] is True
    assert overlay_info["resolved_source"] in {"reflectance", "source_crop"}

    overlay_response = take_object_thumbnail(take_id, "7", mode="classification_overlay", settings=settings)
    assert overlay_response.path is not None
    overlay_thumb_path = Path(str(overlay_response.path))
    assert overlay_thumb_path.is_file()
    assert overlay_thumb_path.name != thumb_path.name
