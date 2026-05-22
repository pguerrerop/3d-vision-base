from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vision_3d_acquisition.vision_core.heightmap import HeightmapFrame, save_heightmap_npz


def create_synthetic_25d_take(
    data_dir: Path,
    *,
    session_id: str = "synthetic_25d_demo",
    include_reflectance: bool = True,
    seed: int = 25,
    take_id: str | None = None,
) -> tuple[str, Path]:
    data_dir = data_dir.resolve()
    incoming_dir = data_dir / "incoming"
    incoming_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(UTC)
    resolved_take_id = take_id or f"{now.strftime('%Y-%m-%dT%H%M%S')}_25d"
    take_dir = incoming_dir / resolved_take_id
    take_dir.mkdir(parents=True, exist_ok=True)

    frame = _build_heightmap(seed=seed)
    heightmap_name = "heightmap.npz"
    save_heightmap_npz(frame, take_dir / heightmap_name)

    reflectance_name: str | None = None
    if include_reflectance:
        reflectance_name = "reflectance.png"
        cv2.imwrite(str(take_dir / reflectance_name), _build_reflectance(frame, seed=seed))

    metadata = _metadata_payload(
        take_id=resolved_take_id,
        session_id=session_id,
        created_at=now.isoformat(),
        heightmap_name=heightmap_name,
        reflectance_name=reflectance_name,
        frame=frame,
    )
    (take_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (take_dir / "READY").touch()
    return resolved_take_id, take_dir


def _build_heightmap(*, seed: int) -> HeightmapFrame:
    rng = np.random.default_rng(seed)
    h, w = 256, 384
    x_mm = np.linspace(0.0, 383.0, w, dtype=np.float32)
    y_mm = np.linspace(0.0, 255.0, h, dtype=np.float32)
    xx, yy = np.meshgrid(x_mm, y_mm)

    # Slightly tilted conveyor plane in metric space.
    belt = 120.0 + 0.018 * xx - 0.011 * yy

    z = belt.copy()

    # Ball-like object (mostly round dome).
    z += _dome(xx, yy, cx=120.0, cy=120.0, rx=28.0, ry=28.0, peak=34.0)
    # Flattened/deformed ball-like object.
    z += _dome(xx, yy, cx=220.0, cy=145.0, rx=33.0, ry=24.0, peak=21.0)
    # Elongated scrap-like object.
    z += _dome(xx, yy, cx=310.0, cy=170.0, rx=16.0, ry=44.0, peak=18.0)

    z += rng.normal(0.0, 0.45, size=z.shape).astype(np.float32)

    valid_mask = np.ones_like(z, dtype=bool)
    invalid = (
        ((xx - 70.0) ** 2 + (yy - 30.0) ** 2 < 12.0**2)
        | ((xx - 340.0) ** 2 + (yy - 230.0) ** 2 < 10.0**2)
        | ((xx > 170.0) & (xx < 190.0) & (yy > 200.0) & (yy < 220.0))
    )
    valid_mask[invalid] = False
    z[~valid_mask] = 0.0

    return HeightmapFrame(
        z_mm=z.astype(np.float32),
        valid_mask=valid_mask,
        reflectance=None,
        x_resolution_mm=1.0,
        y_resolution_mm=1.0,
        origin_x_mm=0.0,
        origin_y_mm=0.0,
        coordinate_system="sensor_xy_z_mm",
    )


def _dome(xx: np.ndarray, yy: np.ndarray, *, cx: float, cy: float, rx: float, ry: float, peak: float) -> np.ndarray:
    nx = (xx - cx) / max(rx, 1e-6)
    ny = (yy - cy) / max(ry, 1e-6)
    rr = nx * nx + ny * ny
    value = np.zeros_like(xx, dtype=np.float32)
    inside = rr <= 1.0
    value[inside] = peak * (1.0 - rr[inside])
    return value


def _build_reflectance(frame: HeightmapFrame, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 99)
    z = frame.z_mm
    valid = frame.valid_mask
    lo = float(np.percentile(z[valid], 2)) if np.any(valid) else 0.0
    hi = float(np.percentile(z[valid], 98)) if np.any(valid) else 1.0
    hi = max(hi, lo + 1.0)
    scaled = np.clip((z - lo) / (hi - lo), 0.0, 1.0)
    refl = np.asarray(np.round((0.35 + 0.65 * scaled) * 255.0), dtype=np.uint8)
    refl = np.clip(refl.astype(np.int16) + rng.integers(-7, 8, size=refl.shape), 0, 255).astype(np.uint8)
    refl[~valid] = 0
    return refl


def _metadata_payload(
    *,
    take_id: str,
    session_id: str,
    created_at: str,
    heightmap_name: str,
    reflectance_name: str | None,
    frame: HeightmapFrame,
) -> dict[str, Any]:
    return {
        "take_id": take_id,
        "source": "synthetic_25d_generator",
        "mode": "offline",
        "created_at": created_at,
        "frame_count": 1,
        "session_id": session_id,
        "modalities": ["heightmap", "reflectance"] if reflectance_name else ["heightmap"],
        "files": {
            "heightmap": heightmap_name,
            "reflectance": reflectance_name,
        },
        "units": {"x": "mm", "y": "mm", "z": "mm"},
        "calibration": {
            "profile_distance_mm": frame.y_resolution_mm,
            "x_resolution_mm": frame.x_resolution_mm,
            "z_scale": 1.0,
            "z_offset": 0.0,
        },
        "frameset": {
            "frameset_id": f"{take_id}_fs0",
            "timestamp": created_at,
            "assets": {"reflectance": reflectance_name},
            "synchronization": {"mode": "none", "confidence": 1.0},
            "frame_count": 1,
            "synchronized": False,
            "timestamp_source": "acquisition_metadata",
        },
        "scan_direction": "forward",
        "belt_speed_mm_s": 380.0,
        "encoder_ticks_per_mm": 4.0,
    }
