from __future__ import annotations

import math

import numpy as np

from vision_3d_acquisition.vision_core.geometry.surface_sphere_fit import (
    fit_sphere_least_squares,
    surface_sphere_fit_rmse_mm,
)


def _sphere_cap_points(
    *,
    radius: float = 20.0,
    center: tuple[float, float, float] = (0.0, 0.0, 10.0),
    count: int = 400,
    z_min: float = 5.0,
    noise_std: float = 0.0,
    seed: int = 7,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    points: list[list[float]] = []
    while len(points) < count:
        vec = rng.normal(size=3)
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-9:
            continue
        direction = vec / norm
        candidate = np.asarray(center, dtype=np.float64) + direction * radius
        if candidate[2] >= z_min:
            points.append(candidate.tolist())
    arr = np.asarray(points, dtype=np.float64)
    if noise_std > 0.0:
        arr += rng.normal(scale=noise_std, size=arr.shape)
    return arr


def test_sphere_fit_clean_sphere_cap_low_rmse() -> None:
    points = _sphere_cap_points(noise_std=0.0)
    rmse, fit = surface_sphere_fit_rmse_mm(points)
    assert fit.get("valid") is True
    assert rmse is not None
    assert math.isfinite(rmse)
    assert rmse < 0.05


def test_sphere_fit_noisy_sphere_cap_moderate_rmse() -> None:
    clean_rmse, _ = surface_sphere_fit_rmse_mm(_sphere_cap_points(noise_std=0.0))
    noisy_rmse, fit = surface_sphere_fit_rmse_mm(_sphere_cap_points(noise_std=0.35, seed=11))
    assert fit.get("valid") is True
    assert noisy_rmse is not None
    assert math.isfinite(noisy_rmse)
    assert noisy_rmse > (clean_rmse or 0.0)
    assert noisy_rmse < 1.5


def test_sphere_fit_flattened_cap_higher_rmse_than_sphere() -> None:
    sphere_points = _sphere_cap_points(noise_std=0.05, seed=3)
    sphere_rmse, _ = surface_sphere_fit_rmse_mm(sphere_points)

    flattened = sphere_points.copy()
    flattened[:, 2] *= 0.25
    flat_rmse, fit = surface_sphere_fit_rmse_mm(flattened)
    assert fit.get("valid") is True
    assert flat_rmse is not None
    assert flat_rmse > (sphere_rmse or 0.0)


def test_sphere_fit_too_few_points_returns_null_with_reason() -> None:
    points = _sphere_cap_points(count=12)[:5]
    fit = fit_sphere_least_squares(points)
    rmse, _ = surface_sphere_fit_rmse_mm(points)
    assert fit.get("valid") is False
    assert fit.get("reason") == "insufficient_points"
    assert rmse is None
