from __future__ import annotations

import json
from pathlib import Path

from scripts import sensor_studio_cli as cli
from vision_3d_acquisition.datasets import DatasetService


def _write_take(data_dir: Path, take_id: str, *, session_id: str = "session_live") -> None:
    take_dir = data_dir / "incoming" / take_id
    take_dir.mkdir(parents=True, exist_ok=True)
    (take_dir / "rgb.png").write_bytes(b"fake")
    (take_dir / "metadata.json").write_text(
        json.dumps({"take_id": take_id, "created_at": "2026-05-25T10:00:00Z", "session_id": session_id, "files": {"rgb": "rgb.png"}}),
        encoding="utf-8",
    )
    (take_dir / "READY").touch()


def test_list_takes_can_filter_by_dataset_id(monkeypatch, tmp_path: Path, capsys) -> None:
    data_dir = tmp_path / "data"
    _write_take(data_dir, "t1")
    _write_take(data_dir, "t2")
    _write_take(data_dir, "t3")

    service = DatasetService(data_dir)
    service.create_dataset(dataset_id="d1", name="Dataset 1")
    service.create_dataset(dataset_id="d2", name="Dataset 2")
    service.create_session(dataset_id="d1", session_id="s1", name="Session 1")
    service.create_session(dataset_id="d2", session_id="s2", name="Session 2")
    service.upsert_take_metadata(take_id="t1", dataset_id="d1", session_id="s1", updates={}, source_metadata={"session_id": "session_live"})
    service.upsert_take_metadata(take_id="t2", dataset_id="d2", session_id="s2", updates={}, source_metadata={"session_id": "session_live"})
    service.upsert_take_metadata(take_id="t3", dataset_id="d1", session_id="s1", updates={}, source_metadata={"session_id": "session_live"})

    monkeypatch.setattr(
        "sys.argv",
        [
            "sensor_studio_cli.py",
            "acquisition",
            "list-takes",
            "--data-dir",
            str(data_dir),
            "--dataset-id",
            "d1",
            "--show-db",
            "--show-headers",
            "--no-limit",
        ],
    )
    code = cli.main()
    out_lines = capsys.readouterr().out.strip().splitlines()

    assert code == 0
    assert out_lines[0] == "take_id\tdataset_id"
    assert set(out_lines[1:]) == {"t1\td1", "t3\td1"}


def test_list_takes_dataset_filter_applies_before_limit(monkeypatch, tmp_path: Path, capsys) -> None:
    data_dir = tmp_path / "data"
    _write_take(data_dir, "t1")
    _write_take(data_dir, "t2")
    _write_take(data_dir, "t3")

    service = DatasetService(data_dir)
    service.create_dataset(dataset_id="d1", name="Dataset 1")
    service.create_dataset(dataset_id="d2", name="Dataset 2")
    service.create_session(dataset_id="d1", session_id="s1", name="Session 1")
    service.create_session(dataset_id="d2", session_id="s2", name="Session 2")
    service.upsert_take_metadata(take_id="t1", dataset_id="d1", session_id="s1", updates={}, source_metadata={"session_id": "session_live"})
    service.upsert_take_metadata(take_id="t2", dataset_id="d2", session_id="s2", updates={}, source_metadata={"session_id": "session_live"})
    service.upsert_take_metadata(take_id="t3", dataset_id="d1", session_id="s1", updates={}, source_metadata={"session_id": "session_live"})

    monkeypatch.setattr(
        "sys.argv",
        [
            "sensor_studio_cli.py",
            "acquisition",
            "list-takes",
            "--data-dir",
            str(data_dir),
            "--dataset-id",
            "d1",
            "--limit",
            "1",
        ],
    )
    code = cli.main()
    out_lines = capsys.readouterr().out.strip().splitlines()

    assert code == 0
    assert len(out_lines) == 1
    assert out_lines[0] in {"t1", "t3"}
