from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from vision_3d_acquisition.calibration.calibration_models import SystemCalibration


def evaluate_calibration_compatibility(
    calibration: SystemCalibration,
    *,
    take_modalities: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    take_modalities = take_modalities or []
    metadata = metadata or {}
    warnings: list[str] = []
    errors: list[str] = []

    missing_modalities = [mod for mod in calibration.source_modalities if mod not in take_modalities]
    if missing_modalities:
        errors.append(f"missing modalities: {', '.join(missing_modalities)}")

    sensor = metadata.get("sensor") if isinstance(metadata.get("sensor"), dict) else {}
    sensor_serial = sensor.get("serial")
    if sensor_serial and calibration.reference_take_id and sensor_serial not in calibration.reference_take_id:
        warnings.append("sensor serial differs from calibration reference")

    frameset = metadata.get("frameset") if isinstance(metadata.get("frameset"), dict) else {}
    roi_hint = frameset.get("roi")
    if roi_hint is not None:
        warnings.append("ROI hint present; verify calibration ROI alignment")

    frame_count = int(metadata.get("frame_count") or 1)
    if frame_count > 1 and calibration.calibration_type == "plane_3d":
        warnings.append("multi-frame acquisition with single-plane calibration")

    confidence = _compatibility_confidence(errors=errors, warnings=warnings)
    if errors:
        status = "incompatible"
    elif warnings:
        status = "warning"
    else:
        status = "compatible"
    return {
        "status": status,
        "compatible": not errors,
        "confidence": confidence,
        "missing_modalities": missing_modalities,
        "warnings": warnings,
        "errors": errors,
        "calibration_age_seconds": _calibration_age_seconds(calibration),
        "calibration_created_at": calibration.created_at.isoformat(),
    }


def pick_recommended_calibration(
    calibrations: list[SystemCalibration],
    *,
    take_modalities: list[str] | None = None,
    sensor_setup_hint: str | None = None,
) -> SystemCalibration | None:
    take_modalities = take_modalities or []
    ranked: list[tuple[float, SystemCalibration]] = []
    for calibration in calibrations:
        score = 0.0
        missing = [mod for mod in calibration.source_modalities if mod not in take_modalities]
        if missing:
            continue
        score += 50.0
        if sensor_setup_hint and sensor_setup_hint in calibration.reference_take_id:
            score += 25.0
        age_seconds = _calibration_age_seconds(calibration)
        score += max(0.0, 25.0 - min(25.0, age_seconds / 3600.0))
        ranked.append((score, calibration))
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[0][1]


def _compatibility_confidence(*, errors: list[str], warnings: list[str]) -> float:
    if errors:
        return 0.1
    if not warnings:
        return 1.0
    return max(0.4, 1.0 - (0.2 * len(warnings)))


def _calibration_age_seconds(calibration: SystemCalibration) -> float:
    created = calibration.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=UTC)
    return max(0.0, (datetime.now(UTC) - created.astimezone(UTC)).total_seconds())
