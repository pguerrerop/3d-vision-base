from __future__ import annotations

from pathlib import Path

import pytest

from vision_3d_acquisition.runtime_config import (
    DEPRECATED_PROCESSING_LOCAL_CALIBRATION_MSG,
    RuntimeConfig,
    clear_default_calibration,
    read_runtime_config,
    resolve_calibration,
    set_default_calibration,
    warn_if_deprecated_processing_local_calibration,
    write_runtime_config,
)


def test_runtime_config_defaults_when_missing(tmp_path: Path) -> None:
    status = read_runtime_config(tmp_path / "config" / "runtime.json")

    assert status.config.default_calibration_file is None
    assert status.config.require_calibration_for_demo is True
    assert status.config.allow_auto_plane_without_calibration is True
    assert status.default_calibration_valid is False


def test_runtime_config_set_clear_and_resolve(tmp_path: Path) -> None:
    path = tmp_path / "config" / "runtime.json"
    calibration = tmp_path / "config" / "calibrations" / "belt.json"
    calibration.parent.mkdir(parents=True)
    calibration.write_text("{}", encoding="utf-8")

    set_status = set_default_calibration(calibration, path)
    resolved = resolve_calibration(None, config_path=path)
    explicit = resolve_calibration(tmp_path / "other.json", config_path=path)
    cleared = clear_default_calibration(path)

    assert set_status.default_calibration_valid is True
    assert resolved.source == "runtime_default"
    assert resolved.calibration == calibration
    assert explicit.source == "explicit"
    assert explicit.calibration == tmp_path / "other.json"
    assert cleared.config.default_calibration_file is None


def test_warns_when_processing_local_has_default_calibration(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import vision_3d_acquisition.runtime_config as runtime_config_module

    processing_local = tmp_path / "config" / "processing.local.json"
    processing_local.parent.mkdir(parents=True)
    processing_local.write_text(
        '{"default_calibration": "config/calibrations/old.json"}\n',
        encoding="utf-8",
    )
    runtime_config_module._deprecation_warned = False

    warn_if_deprecated_processing_local_calibration(processing_local)

    captured = capsys.readouterr()
    assert DEPRECATED_PROCESSING_LOCAL_CALIBRATION_MSG in captured.err
    assert runtime_config_module._deprecation_warned is True

    warn_if_deprecated_processing_local_calibration(processing_local)
    assert capsys.readouterr().err == ""


def test_missing_runtime_default_reports_warning(tmp_path: Path) -> None:
    path = tmp_path / "config" / "runtime.json"
    write_runtime_config(RuntimeConfig(default_calibration_file="config/calibrations/missing.json"), path)

    status = read_runtime_config(path)
    resolved = resolve_calibration(None, config_path=path)

    assert status.default_calibration_valid is False
    assert status.warning is not None
    assert resolved.source == "none"
    assert resolved.warning is not None
