from __future__ import annotations

import json
import signal
import subprocess
import sys
import time
from pathlib import Path

from vision_3d_acquisition.calibration.calibration_models import ObjectFilterConfig, PlaneCalibration, SystemCalibration
from vision_3d_acquisition.calibration.compatibility import evaluate_calibration_compatibility
from vision_3d_acquisition.contracts.metadata import AcquisitionMetadata, FileReferences
from vision_3d_acquisition.state.runtime_status import is_stale, read_runtime_status, update_runtime_status
from vision_3d_acquisition.storage.publisher import AcquisitionPublisher


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_synthetic_take(data_dir: Path, take_id: str) -> None:
    take_dir = data_dir / "incoming" / take_id
    take_dir.mkdir(parents=True, exist_ok=True)
    points = [(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (-1.0, 1.0, 0.0), (1.0, 1.0, 0.0), (10.0, 0.0, 4.0), (10.5, 0.2, 4.2)]
    ply = "\n".join(
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
    (take_dir / "point_cloud.ply").write_text(ply + "\n", encoding="utf-8")
    (take_dir / "metadata.json").write_text(
        json.dumps(
            {
                "take_id": take_id,
                "source": "live_sensor",
                "mode": "live",
                "created_at": "2026-05-17T12:00:00Z",
                "session_id": "session_test_live",
                "frame_count": 1,
                "frameset": {
                    "frameset_id": f"{take_id}_fs0",
                    "timestamp": "2026-05-17T12:00:00Z",
                    "assets": {"point_cloud": "point_cloud.ply"},
                    "synchronization": {"mode": "timestamp", "confidence": 0.9},
                },
                "files": {"point_cloud": "point_cloud.ply"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (take_dir / "READY").touch()


def test_live_publish_auto_creates_session(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    source = tmp_path / "sample.ply"
    source.write_text("ply\nformat ascii 1.0\nend_header\n", encoding="utf-8")
    publisher = AcquisitionPublisher(data_dir)
    metadata = AcquisitionMetadata(
        take_id="live_take_001",
        source="live_sensor",
        mode="live",
        created_at="2026-05-17T12:00:00Z",
        frame_count=1,
        files=FileReferences(point_cloud="point_cloud.ply"),
    )
    publisher.publish_take("live_take_001", metadata, {"point_cloud": source})
    written = json.loads((data_dir / "incoming" / "live_take_001" / "metadata.json").read_text(encoding="utf-8"))
    assert written["session_id"].startswith("session_")
    assert (data_dir / "sessions" / written["session_id"] / "metadata.json").is_file()


def test_frameset_contract_backwards_compatible() -> None:
    metadata = AcquisitionMetadata(
        take_id="take_001",
        source="offline",
        mode="offline",
        created_at="2026-05-17T12:00:00Z",
        frame_count=1,
        files=FileReferences(point_cloud="point_cloud.ply"),
    )
    payload = metadata.model_dump(mode="json")
    assert payload["frameset"]["frameset_id"] == "take_001_fs0"
    assert payload["frameset"]["synchronization"]["mode"] == "none"


def test_runtime_status_updates_and_stale_detection(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    update_runtime_status(
        state_dir,
        status="running",
        latest_frame_timestamp="2026-01-01T00:00:00Z",
        queue_size=2,
    )
    runtime = read_runtime_status(state_dir)
    assert runtime["status"] == "running"
    assert runtime["queue_size"] == 2
    assert is_stale(runtime)


def test_calibration_compatibility_validation() -> None:
    calibration = SystemCalibration(
        calibration_id="cal_test",
        reference_take_id="take_ref",
        created_at="2026-05-16T00:00:00Z",
        source_modalities=["point_cloud", "rgb"],
        planes=[
            PlaneCalibration(
                plane_id="belt",
                label="belt",
                plane_model=(0.0, 0.0, 1.0, 0.0),
                point_count=100,
                bbox_min_mm=(-1.0, -1.0, -1.0),
                bbox_max_mm=(1.0, 1.0, 1.0),
                extent_mm=(2.0, 2.0, 2.0),
                avg_z_mm=0.0,
                roi_polygon_xy_mm=[(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0)],
            )
        ],
        object_filter=ObjectFilterConfig(),
    )
    diagnostics = evaluate_calibration_compatibility(calibration, take_modalities=["point_cloud"], metadata={"frame_count": 3})
    assert diagnostics["status"] == "incompatible"
    assert "rgb" in diagnostics["missing_modalities"]
    assert diagnostics["confidence"] <= 0.2


def test_live_pipeline_loop_processes_and_shutdown(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_synthetic_take(data_dir, "live_loop_take_001")
    process = subprocess.Popen(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_live_pipeline.py"),
            "--data-dir",
            str(data_dir),
            "--engine",
            "legacy",
            "--poll-interval",
            "0.1",
            "--skip-debug-images",
            "--session",
            "session_test_live",
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.time() + 25
        done_file = data_dir / "processed" / "live_loop_take_001" / "DONE"
        while time.time() < deadline and not done_file.is_file():
            time.sleep(0.2)
        assert done_file.is_file(), "live loop timeout"
    finally:
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=10)
    result_payload = json.loads((data_dir / "processed" / "live_loop_take_001" / "result.json").read_text(encoding="utf-8"))
    runtime_payload = json.loads((data_dir / "state" / "runtime.json").read_text(encoding="utf-8"))
    assert result_payload.get("session_id") == "session_test_live"
    assert "throughput" in result_payload
    assert runtime_payload["status"] == "stopped"
