from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vision_3d_acquisition.vision_core.projections.rasterize import ProjectionRaster, rasterize_projection


@dataclass
class ProjectionArtifact:
    artifact_id: str
    projection_type: str
    filename: str
    raster: ProjectionRaster


def generate_canonical_projections(
    *,
    points: np.ndarray,
    output_dir: Path,
    prefix: str = "projection",
    pixel_per_mm: float = 2.0,
) -> list[ProjectionArtifact]:
    if points.ndim != 2 or points.shape[1] < 3 or points.shape[0] == 0:
        return []
    projections = [
        ("xy_topdown", points[:, 0], points[:, 1], points[:, 2]),
        ("xz_side", points[:, 0], points[:, 2], points[:, 1]),
        ("yz_side", points[:, 1], points[:, 2], points[:, 0]),
    ]
    generated: list[ProjectionArtifact] = []
    for proj_type, primary, secondary, depth in projections:
        raster = rasterize_projection(
            primary=primary,
            secondary=secondary,
            depth=depth,
            projection_type=proj_type,
            pixel_per_mm=pixel_per_mm,
        )
        filename = f"{prefix}_{proj_type}.png"
        raster.image.save(output_dir / filename)
        generated.append(
            ProjectionArtifact(
                artifact_id=proj_type,
                projection_type=proj_type,
                filename=filename,
                raster=raster,
            )
        )
    return generated


def projection_metadata_payload(item: ProjectionArtifact) -> dict[str, Any]:
    t = item.raster.transform
    return {
        "projection_type": item.projection_type,
        "coordinate_system": {
            "origin": [t.bounds["primary_min"], t.bounds["secondary_min"], 0.0],
            "x_axis": [1.0, 0.0, 0.0],
            "y_axis": [0.0, 1.0, 0.0],
            "z_axis": [0.0, 0.0, 1.0],
            "pixel_per_mm": t.pixel_per_mm,
            "world_bounds_mm": {
                "primary_min": t.bounds["primary_min"],
                "primary_max": t.bounds["primary_max"],
                "secondary_min": t.bounds["secondary_min"],
                "secondary_max": t.bounds["secondary_max"],
            },
            "image_width": t.width,
            "image_height": t.height,
            "affine_transform": [
                [t.pixel_per_mm, 0.0, -t.bounds["primary_min"] * t.pixel_per_mm],
                [0.0, t.pixel_per_mm, -t.bounds["secondary_min"] * t.pixel_per_mm],
                [0.0, 0.0, 1.0],
            ],
        },
        "depth_range": [item.raster.depth_range[0], item.raster.depth_range[1]],
        "transform_id": t.transform_id,
        "background_style": "black",
        "colormap": "gray",
    }

