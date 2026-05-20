from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel


RUNTIME_CONFIG_PATH = Path("config/runtime.json")
PROCESSING_LOCAL_CONFIG_PATH = Path("config/processing.local.json")
DEPRECATED_PROCESSING_LOCAL_CALIBRATION_MSG = (
    "default_calibration in processing.local.json is ignored. Use config/runtime.json."
)
CalibrationResolutionSource = Literal["explicit", "runtime_default", "none"]

_deprecation_warned = False


class RuntimeConfig(BaseModel):
    default_calibration_file: str | None = None
    require_calibration_for_demo: bool = True
    allow_auto_plane_without_calibration: bool = True


@dataclass(frozen=True)
class RuntimeConfigStatus:
    config: RuntimeConfig
    path: Path
    default_calibration_file: str | None
    default_calibration_valid: bool
    warning: str | None = None


@dataclass(frozen=True)
class CalibrationResolution:
    calibration: Path | None
    source: CalibrationResolutionSource
    active: bool
    warning: str | None = None


def warn_if_deprecated_processing_local_calibration(
    processing_local_path: Path = PROCESSING_LOCAL_CONFIG_PATH,
) -> None:
    """Warn once per process if legacy default_calibration is still configured."""
    global _deprecation_warned
    if _deprecation_warned:
        return
    if not processing_local_path.is_file():
        return
    try:
        payload = json.loads(processing_local_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if "default_calibration" not in payload:
        return
    _deprecation_warned = True
    print(DEPRECATED_PROCESSING_LOCAL_CALIBRATION_MSG, file=sys.stderr)


def read_runtime_config(path: Path = RUNTIME_CONFIG_PATH) -> RuntimeConfigStatus:
    warn_if_deprecated_processing_local_calibration()
    if not path.is_file():
        config = RuntimeConfig()
        return RuntimeConfigStatus(config=config, path=path, default_calibration_file=None, default_calibration_valid=False)
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = RuntimeConfig.model_validate(payload)
    default = config.default_calibration_file
    if not default:
        return RuntimeConfigStatus(config=config, path=path, default_calibration_file=None, default_calibration_valid=False)
    resolved = Path(default)
    if resolved.is_file():
        return RuntimeConfigStatus(config=config, path=path, default_calibration_file=str(resolved), default_calibration_valid=True)
    return RuntimeConfigStatus(
        config=config,
        path=path,
        default_calibration_file=str(resolved),
        default_calibration_valid=False,
        warning=f"Runtime default calibration file is missing: {resolved}",
    )


def write_runtime_config(config: RuntimeConfig, path: Path = RUNTIME_CONFIG_PATH) -> RuntimeConfigStatus:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
    return read_runtime_config(path)


def set_default_calibration(calibration_file: Path | str, path: Path = RUNTIME_CONFIG_PATH) -> RuntimeConfigStatus:
    calibration_path = Path(calibration_file)
    if not calibration_path.is_file():
        raise FileNotFoundError(f"Calibration file not found: {calibration_path}")
    status = read_runtime_config(path)
    config = status.config.model_copy(update={"default_calibration_file": str(calibration_path)})
    return write_runtime_config(config, path)


def clear_default_calibration(path: Path = RUNTIME_CONFIG_PATH) -> RuntimeConfigStatus:
    status = read_runtime_config(path)
    config = status.config.model_copy(update={"default_calibration_file": None})
    return write_runtime_config(config, path)


def resolve_calibration(
    explicit_calibration: Path | str | None,
    *,
    config_path: Path = RUNTIME_CONFIG_PATH,
) -> CalibrationResolution:
    warn_if_deprecated_processing_local_calibration()
    if explicit_calibration is not None:
        return CalibrationResolution(calibration=Path(explicit_calibration), source="explicit", active=True)
    status = read_runtime_config(config_path)
    if status.default_calibration_file and status.default_calibration_valid:
        return CalibrationResolution(
            calibration=Path(status.default_calibration_file),
            source="runtime_default",
            active=True,
        )
    if status.warning:
        return CalibrationResolution(calibration=None, source="none", active=False, warning=status.warning)
    return CalibrationResolution(calibration=None, source="none", active=False)
