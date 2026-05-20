from __future__ import annotations

import json
import os
from pathlib import Path

from PIL import Image

from vision_3d_acquisition.api.filesystem import safe_take_file
from vision_3d_acquisition.api.histogram import load_or_compute_histogram, resolve_source_image_path
from vision_3d_acquisition.api.main import health, latest, pipelines, runtime_preview_metadata, session_summary_endpoint, sessions, sources, take_detail, takes
from vision_3d_acquisition.api.settings import ApiSettings, get_cors_origins, get_settings


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


def test_pipelines_endpoint_exposes_registry() -> None:
    payload = pipelines()

    current = next(item for item in payload if item["id"] == "3d_ball_inspection")
    fusion = next(item for item in payload if item["id"] == "2d_3d_fusion")
    assert current["required_modalities"] == ["point_cloud"]
    assert current["stages"][0]["id"] == "segmentation"
    assert fusion["implemented"] is False


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
