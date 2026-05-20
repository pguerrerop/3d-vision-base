from __future__ import annotations

import json
from pathlib import Path

from vision_3d_acquisition.calibration.calibration_models import SystemCalibration


DEFAULT_CALIBRATION_DIR = Path("config/calibrations")


def save_calibration(
    calibration: SystemCalibration,
    calibration_dir: Path = DEFAULT_CALIBRATION_DIR,
) -> Path:
    calibration_dir.mkdir(parents=True, exist_ok=True)
    filename = calibration.calibration_id
    if not filename.endswith(".json"):
        filename = f"{filename}.json"
    path = calibration_dir / filename
    path.write_text(
        json.dumps(calibration.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def load_calibration(
    calibration_id_or_path: str | Path,
    calibration_dir: Path = DEFAULT_CALIBRATION_DIR,
) -> SystemCalibration:
    raw_path = Path(calibration_id_or_path)
    if raw_path.is_file():
        path = raw_path
    else:
        name = str(calibration_id_or_path)
        if not name.endswith(".json"):
            name = f"{name}.json"
        path = calibration_dir / name
    return SystemCalibration.model_validate_json(path.read_text(encoding="utf-8"))


def list_calibrations(calibration_dir: Path = DEFAULT_CALIBRATION_DIR) -> list[SystemCalibration]:
    if not calibration_dir.exists():
        return []
    calibrations: list[SystemCalibration] = []
    for path in sorted(calibration_dir.glob("*.json"), reverse=True):
        try:
            calibrations.append(load_calibration(path, calibration_dir))
        except Exception:
            continue
    return sorted(calibrations, key=lambda item: item.created_at, reverse=True)

