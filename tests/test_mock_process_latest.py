from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from vision_3d_acquisition.acquisition.offline_ply import OfflinePlyAcquisition
from vision_3d_acquisition.storage.publisher import AcquisitionPublisher


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_mock_processing_creates_result_and_done(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    sample_ply = tmp_path / "sample.ply"
    sample_ply.write_text("ply\nformat ascii 1.0\nend_header\n", encoding="utf-8")
    publisher = AcquisitionPublisher(data_dir)
    take_id, _ = OfflinePlyAcquisition(publisher).acquire(sample_ply)

    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "mock_process_latest.py"), "--data-dir", str(data_dir)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    processed_dir = data_dir / "processed" / take_id
    assert (processed_dir / "result.json").is_file()
    assert (processed_dir / "input_point_cloud_preview.png").is_file()
    assert (processed_dir / "overlay.png").is_file()
    assert (processed_dir / "debug_height.png").is_file()
    assert (processed_dir / "debug_segmentation.png").is_file()
    assert (processed_dir / "DONE").is_file()
    result = json.loads((processed_dir / "result.json").read_text(encoding="utf-8"))
    assert result["take_id"] == take_id
    assert result["processing_mode"] == "mock"
    assert result["files"]["input_preview"] == "input_point_cloud_preview.png"
    assert result["input_stats"]["point_count"] == 0
    assert result["input_stats"]["file_size_bytes"] > 0
    assert result["summary"]["decision"] in {"accept", "reject", "review"}
    assert (data_dir / "state" / "latest.json").is_file()
    assert (data_dir / "events" / "events.jsonl").is_file()
