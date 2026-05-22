from __future__ import annotations

from pathlib import Path

from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.runtime.registry import build_runtime_registry


def _settings(data_dir: Path) -> ApiSettings:
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


def test_registry_uses_resolved_python_and_normalized_command(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    registry = build_runtime_registry(settings)
    definition = registry["trispector_ftp"]
    assert definition.command[0]
    assert definition.command[1] == "-m"
    assert "normalized_command" in definition.config
