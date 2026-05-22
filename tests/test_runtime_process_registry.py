from __future__ import annotations

from pathlib import Path

from vision_3d_acquisition.runtime.process_registry import RuntimeProcessRegistry


def test_registry_register_heartbeat_and_stale_detection(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    registry = RuntimeProcessRegistry(data_dir, stale_after_seconds=0.01)

    registry.register(process_name="run_25d_worker", runtime_role="worker", pid=123)
    rows = registry.list_processes()
    assert len(rows) == 1
    assert rows[0]["status"] == "running"
    assert rows[0]["is_stale"] is False

    # Force stale by backdating heartbeat file directly.
    path = data_dir / "runtime" / "processes" / "run_25d_worker.json"
    payload = path.read_text(encoding="utf-8")
    payload = payload.replace(rows[0]["last_heartbeat"], "2000-01-01T00:00:00+00:00")
    path.write_text(payload, encoding="utf-8")

    stale_rows = registry.list_processes()
    assert stale_rows[0]["status"] == "stale"
    assert stale_rows[0]["is_stale"] is True


def test_registry_stop_marks_stopped(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    registry = RuntimeProcessRegistry(data_dir)
    registry.register(process_name="trispector_take_watcher", runtime_role="watcher", pid=456)
    registry.stop("trispector_take_watcher")

    rows = registry.list_processes()
    assert rows[0]["status"] == "stopped"
