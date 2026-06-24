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


def test_update_take_session_dry_run_does_not_write(monkeypatch, tmp_path: Path, capsys) -> None:
    data_dir = tmp_path / "data"
    _write_take(data_dir, "t1")
    _write_take(data_dir, "t2")
    service = DatasetService(data_dir)
    service.create_dataset(dataset_id="d1", name="Dataset 1")
    service.create_session(dataset_id="d1", session_id="s_old", name="Old Session")
    service.upsert_take_metadata(take_id="t1", dataset_id="d1", session_id="s_old", updates={}, source_metadata={"session_id": "session_live"})
    service.upsert_take_metadata(take_id="t2", dataset_id="d1", session_id="s_old", updates={}, source_metadata={"session_id": "session_live"})

    monkeypatch.setattr(
        "sys.argv",
        [
            "sensor_studio_cli.py",
            "acquisition",
            "update-take-session",
            "--data-dir",
            str(data_dir),
            "--session-id",
            "s_new",
            "--create-session",
            "--take-id",
            "t1",
            "--take-id",
            "t2",
        ],
    )
    code = cli.main()
    out = capsys.readouterr().out

    assert code == 0
    assert "[DRY RUN]" in out
    assert service.get_session("d1", "s_new") is None
    assert service.load_take_metadata(take_id="t1", source_metadata={"session_id": "session_live"}).get("session_id") == "s_old"
    assert not list((data_dir / "runtime" / "logs").glob("update_take_session_*.json"))


def test_update_take_session_apply_updates_and_logs(monkeypatch, tmp_path: Path, capsys) -> None:
    data_dir = tmp_path / "data"
    _write_take(data_dir, "t1")
    _write_take(data_dir, "t2")
    service = DatasetService(data_dir)
    service.create_dataset(dataset_id="d1", name="Dataset 1")
    service.create_session(dataset_id="d1", session_id="s_old", name="Old Session")
    service.upsert_take_metadata(take_id="t1", dataset_id="d1", session_id="s_old", updates={}, source_metadata={"session_id": "session_live"})
    service.upsert_take_metadata(take_id="t2", dataset_id="d1", session_id="s_old", updates={}, source_metadata={"session_id": "session_live"})
    take_file = tmp_path / "takes.txt"
    take_file.write_text("t1\nt2\nt2\n", encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "sensor_studio_cli.py",
            "acquisition",
            "update-take-session",
            "--data-dir",
            str(data_dir),
            "--session-id",
            "s_new",
            "--session-name",
            "Session New",
            "--create-session",
            "--take-file",
            str(take_file),
            "--apply",
        ],
    )
    code = cli.main()
    out = capsys.readouterr().out

    assert code == 0
    assert "[APPLY]" in out
    assert service.get_session("d1", "s_new") is not None
    assert service.load_take_metadata(take_id="t1", source_metadata={"session_id": "session_live"}).get("session_id") == "s_new"
    assert service.load_take_metadata(take_id="t2", source_metadata={"session_id": "session_live"}).get("session_id") == "s_new"
    logs = list((data_dir / "runtime" / "logs").glob("update_take_session_*.json"))
    assert logs
    payload = json.loads(logs[0].read_text(encoding="utf-8"))
    assert payload["dataset_id"] == "d1"
    assert payload["new_session_id"] == "s_new"

    # Idempotent rerun should remain successful and keep canonical membership.
    monkeypatch.setattr(
        "sys.argv",
        [
            "sensor_studio_cli.py",
            "acquisition",
            "update-take-session",
            "--data-dir",
            str(data_dir),
            "--session-id",
            "s_new",
            "--take-id",
            "t1",
            "--take-id",
            "t2",
            "--apply",
        ],
    )
    second_code = cli.main()
    assert second_code == 0
    assert service.resolve_all_take_memberships("t1") == [("d1", "s_new")]
