from __future__ import annotations

from dataclasses import dataclass
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vision_3d_acquisition.vision_core.heightmap import HeightmapFrame, save_heightmap_npz


OBJECT_TYPES = {
    "good_ball",
    "worn_ellipsoid",
    "truncated_sphere",
    "chipped_sphere",
    "flattened_ball",
    "elongated_scrap",
}


@dataclass(frozen=True)
class SyntheticNoiseConfig:
    gaussian_height_std_mm: float = 0.35
    missing_region_fraction: float = 0.0
    edge_aliasing: bool = True
    belt_tilt_x_mm_per_px: float = 0.012
    belt_tilt_y_mm_per_px: float = -0.008


def create_advanced_synthetic_25d_take(
    data_dir: Path,
    *,
    object_type: str,
    session_id: str = "synthetic_25d_geometry_suite",
    include_reflectance: bool = True,
    seed: int = 101,
    take_id: str | None = None,
    noise: SyntheticNoiseConfig | None = None,
) -> tuple[str, Path]:
    if object_type not in OBJECT_TYPES:
        raise ValueError(f"Unsupported object_type={object_type}")
    cfg = noise or SyntheticNoiseConfig()
    data_dir = data_dir.resolve()
    incoming_dir = data_dir / "incoming"
    incoming_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    resolved_take_id = take_id or f"{now.strftime('%Y-%m-%dT%H%M%S')}_{object_type}_25d"
    take_dir = incoming_dir / resolved_take_id
    take_dir.mkdir(parents=True, exist_ok=True)

    frame, generation = build_advanced_synthetic_frame(object_type=object_type, seed=seed, noise=cfg)
    save_heightmap_npz(frame, take_dir / "heightmap.npz")
    reflectance_name: str | None = None
    if include_reflectance:
        reflectance_name = "reflectance.png"
        cv2.imwrite(str(take_dir / reflectance_name), _build_reflectance(frame, seed=seed))
    cv2.imwrite(str(take_dir / "synthetic_gt_mask.png"), (generation["ground_truth_mask"].astype(np.uint8) * 255))
    cv2.imwrite(str(take_dir / "synthetic_flat_regions_gt.png"), (generation["flat_regions_gt"].astype(np.uint8) * 255))

    metadata = _metadata_payload(
        take_id=resolved_take_id,
        session_id=session_id,
        created_at=now.isoformat(),
        frame=frame,
        reflectance_name=reflectance_name,
        object_type=object_type,
        generation=generation,
        noise=cfg,
    )
    (take_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (take_dir / "READY").touch()
    return resolved_take_id, take_dir


def build_advanced_synthetic_frame(
    *,
    object_type: str,
    seed: int,
    noise: SyntheticNoiseConfig,
) -> tuple[HeightmapFrame, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    h, w = 256, 384
    x_mm = np.linspace(0.0, float(w - 1), w, dtype=np.float32)
    y_mm = np.linspace(0.0, float(h - 1), h, dtype=np.float32)
    xx, yy = np.meshgrid(x_mm, y_mm)
    belt = 120.0 + noise.belt_tilt_x_mm_per_px * xx + noise.belt_tilt_y_mm_per_px * yy
    z = belt.copy()
    cx, cy = 192.0, 130.0
    gt_mask = np.zeros_like(z, dtype=bool)
    flat_gt = np.zeros_like(z, dtype=bool)
    params: dict[str, Any] = {}

    if object_type == "good_ball":
        cap = _spherical_cap(xx, yy, cx, cy, radius_xy=34.0, height_peak=32.0)
        gt_mask = cap > 0.0
        z += cap
        params = {"radius_xy_mm": 34.0, "height_peak_mm": 32.0}
    elif object_type == "worn_ellipsoid":
        cap = _ellipsoidal_cap(xx, yy, cx, cy, rx=39.0, ry=29.0, height_peak=30.0)
        gt_mask = cap > 0.0
        z += cap
        params = {"rx_mm": 39.0, "ry_mm": 29.0, "height_peak_mm": 30.0}
    elif object_type == "truncated_sphere":
        cap = _spherical_cap(xx, yy, cx, cy, radius_xy=36.0, height_peak=32.0)
        trunc = yy > (cy + 9.0)
        cap = np.where(trunc, cap * 0.35, cap)
        gt_mask = cap > 0.0
        z += cap
        params = {"radius_xy_mm": 36.0, "height_peak_mm": 32.0, "truncation_side": "lower", "truncation_keep_ratio": 0.35}
    elif object_type == "chipped_sphere":
        cap = _spherical_cap(xx, yy, cx, cy, radius_xy=35.0, height_peak=30.0)
        chip1 = ((xx - (cx + 16.0)) ** 2 + (yy - (cy - 9.0)) ** 2) < 9.0**2
        chip2 = ((xx - (cx - 18.0)) ** 2 + (yy - (cy + 12.0)) ** 2) < 7.0**2
        cap = np.where(chip1 | chip2, cap * 0.15, cap)
        gt_mask = cap > 0.0
        z += cap
        params = {"radius_xy_mm": 35.0, "height_peak_mm": 30.0, "chips": 2}
    elif object_type == "flattened_ball":
        cap = _spherical_cap(xx, yy, cx, cy, radius_xy=34.0, height_peak=31.0)
        flat = (xx > (cx - 10.0)) & (xx < (cx + 12.0)) & (yy > (cy - 2.0)) & (yy < (cy + 11.0))
        cap = np.where(flat, np.minimum(cap, 13.0), cap)
        gt_mask = cap > 0.0
        flat_gt = flat & gt_mask
        z += cap
        params = {"radius_xy_mm": 34.0, "height_peak_mm": 31.0, "flat_patch": [cx - 10.0, cy - 2.0, cx + 12.0, cy + 11.0]}
    else:  # elongated_scrap
        cap = _ellipsoidal_cap(xx, yy, cx, cy, rx=20.0, ry=58.0, height_peak=20.0)
        ridge = np.exp(-(((xx - (cx - 3.0)) / 8.0) ** 2 + ((yy - (cy + 1.0)) / 18.0) ** 2)).astype(np.float32) * 5.0
        cap = cap + ridge
        gt_mask = cap > 0.0
        z += cap
        params = {"rx_mm": 20.0, "ry_mm": 58.0, "height_peak_mm": 20.0}

    if noise.edge_aliasing:
        edge = cv2.Canny((gt_mask.astype(np.uint8) * 255), 40, 120) > 0
        z[edge] += rng.normal(0.0, max(0.2, noise.gaussian_height_std_mm), size=int(np.count_nonzero(edge))).astype(np.float32)

    z += rng.normal(0.0, noise.gaussian_height_std_mm, size=z.shape).astype(np.float32)
    valid_mask = np.ones_like(z, dtype=bool)
    if noise.missing_region_fraction > 0.0:
        missing_count = int(noise.missing_region_fraction * z.size)
        if missing_count > 0:
            idx = rng.choice(z.size, size=missing_count, replace=False)
            valid_mask.reshape(-1)[idx] = False
    z[~valid_mask] = 0.0

    frame = HeightmapFrame(
        z_mm=z.astype(np.float32),
        valid_mask=valid_mask,
        reflectance=None,
        x_resolution_mm=1.0,
        y_resolution_mm=1.0,
        origin_x_mm=0.0,
        origin_y_mm=0.0,
        coordinate_system="sensor_xy_z_mm",
    )
    expected = _expected_family_reactions(object_type)
    generation = {
        "object_type": object_type,
        "generation_parameters": params,
        "expected_failure_modes": expected["failure_modes"],
        "expected_metric_family_reactions": expected["families"],
        "ground_truth_mask": gt_mask,
        "flat_regions_gt": flat_gt,
    }
    return frame, generation


def _spherical_cap(xx: np.ndarray, yy: np.ndarray, cx: float, cy: float, radius_xy: float, height_peak: float) -> np.ndarray:
    nx = (xx - cx) / max(radius_xy, 1e-6)
    ny = (yy - cy) / max(radius_xy, 1e-6)
    rr = nx * nx + ny * ny
    out = np.zeros_like(xx, dtype=np.float32)
    inside = rr <= 1.0
    out[inside] = height_peak * np.sqrt(np.clip(1.0 - rr[inside], 0.0, 1.0))
    return out


def _ellipsoidal_cap(xx: np.ndarray, yy: np.ndarray, cx: float, cy: float, rx: float, ry: float, height_peak: float) -> np.ndarray:
    nx = (xx - cx) / max(rx, 1e-6)
    ny = (yy - cy) / max(ry, 1e-6)
    rr = nx * nx + ny * ny
    out = np.zeros_like(xx, dtype=np.float32)
    inside = rr <= 1.0
    out[inside] = height_peak * np.sqrt(np.clip(1.0 - rr[inside], 0.0, 1.0))
    return out


def _build_reflectance(frame: HeightmapFrame, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 41)
    z = frame.z_mm
    valid = frame.valid_mask
    lo = float(np.percentile(z[valid], 2)) if np.any(valid) else 0.0
    hi = float(np.percentile(z[valid], 98)) if np.any(valid) else 1.0
    hi = max(hi, lo + 1.0)
    scaled = np.clip((z - lo) / (hi - lo), 0.0, 1.0)
    refl = np.asarray(np.round((0.3 + 0.7 * scaled) * 255.0), dtype=np.uint8)
    refl = np.clip(refl.astype(np.int16) + rng.integers(-9, 10, size=refl.shape), 0, 255).astype(np.uint8)
    refl[~valid] = 0
    return refl


def _expected_family_reactions(object_type: str) -> dict[str, Any]:
    base = {
        "footprint_geometry": "PASS",
        "surface_geometry": "PASS",
        "sphere_consistency": "PASS",
        "damage_metrics": "LOW",
    }
    failures: list[str] = []
    if object_type == "worn_ellipsoid":
        base.update({"footprint_geometry": "MEDIUM", "surface_geometry": "MEDIUM", "sphere_consistency": "MEDIUM", "damage_metrics": "LOW"})
        failures = ["ellipsoidal_deformation"]
    elif object_type == "truncated_sphere":
        base.update({"footprint_geometry": "MEDIUM", "surface_geometry": "GOOD", "sphere_consistency": "FAIL", "damage_metrics": "MEDIUM"})
        failures = ["volume_deficit", "truncation"]
    elif object_type == "chipped_sphere":
        base.update({"footprint_geometry": "FAIL", "surface_geometry": "MEDIUM", "sphere_consistency": "FAIL", "damage_metrics": "FAIL"})
        failures = ["boundary_irregularity", "surface_damage"]
    elif object_type == "flattened_ball":
        base.update({"footprint_geometry": "MEDIUM", "surface_geometry": "MEDIUM", "sphere_consistency": "FAIL", "damage_metrics": "FAIL"})
        failures = ["flat_regions", "radial_height_inconsistency"]
    elif object_type == "elongated_scrap":
        base.update({"footprint_geometry": "FAIL", "surface_geometry": "FAIL", "sphere_consistency": "FAIL", "damage_metrics": "MEDIUM"})
        failures = ["elongation", "non_spherical_shape"]
    return {"families": base, "failure_modes": failures}


def _metadata_payload(
    *,
    take_id: str,
    session_id: str,
    created_at: str,
    frame: HeightmapFrame,
    reflectance_name: str | None,
    object_type: str,
    generation: dict[str, Any],
    noise: SyntheticNoiseConfig,
) -> dict[str, Any]:
    return {
        "take_id": take_id,
        "source": "synthetic_25d_generator_advanced",
        "mode": "offline",
        "created_at": created_at,
        "frame_count": 1,
        "session_id": session_id,
        "modalities": ["heightmap", "reflectance"] if reflectance_name else ["heightmap"],
        "files": {
            "heightmap": "heightmap.npz",
            "reflectance": reflectance_name,
            "synthetic_gt_mask": "synthetic_gt_mask.png",
            "synthetic_flat_regions_gt": "synthetic_flat_regions_gt.png",
        },
        "units": {"x": "mm", "y": "mm", "z": "mm"},
        "calibration": {
            "profile_distance_mm": frame.y_resolution_mm,
            "x_resolution_mm": frame.x_resolution_mm,
            "z_scale": 1.0,
            "z_offset": 0.0,
        },
        "scan_direction": "forward",
        "belt_speed_mm_s": 380.0,
        "encoder_ticks_per_mm": 4.0,
        "synthetic_object_type": object_type,
        "generation_parameters": generation.get("generation_parameters") or {},
        "expected_failure_modes": generation.get("expected_failure_modes") or [],
        "expected_metric_family_reactions": generation.get("expected_metric_family_reactions") or {},
        "acquisition_effects": {
            "gaussian_height_std_mm": noise.gaussian_height_std_mm,
            "missing_region_fraction": noise.missing_region_fraction,
            "edge_aliasing": noise.edge_aliasing,
            "belt_tilt_x_mm_per_px": noise.belt_tilt_x_mm_per_px,
            "belt_tilt_y_mm_per_px": noise.belt_tilt_y_mm_per_px,
        },
    }
