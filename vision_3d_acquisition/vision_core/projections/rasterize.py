from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from vision_3d_acquisition.vision_core.projections.transforms import ProjectionTransform


@dataclass
class ProjectionRaster:
    image: Image.Image
    transform: ProjectionTransform
    depth_range: tuple[float, float]


def rasterize_projection(
    *,
    primary: np.ndarray,
    secondary: np.ndarray,
    depth: np.ndarray,
    projection_type: str,
    pixel_per_mm: float = 2.0,
    margin_mm: float = 5.0,
) -> ProjectionRaster:
    p_min = float(np.min(primary) - margin_mm)
    p_max = float(np.max(primary) + margin_mm)
    s_min = float(np.min(secondary) - margin_mm)
    s_max = float(np.max(secondary) + margin_mm)
    width = max(64, int(np.ceil((p_max - p_min) * pixel_per_mm)))
    height = max(64, int(np.ceil((s_max - s_min) * pixel_per_mm)))
    transform = ProjectionTransform(
        transform_id=f"{projection_type}_transform",
        projection_type=projection_type,
        bounds={
            "primary_min": p_min,
            "primary_max": p_max,
            "secondary_min": s_min,
            "secondary_max": s_max,
        },
        width=width,
        height=height,
        pixel_per_mm=pixel_per_mm,
    )
    px, py = transform.to_pixel(primary, secondary)
    canvas = np.zeros((height, width), dtype=np.float32)
    near = float(np.min(depth))
    far = float(np.max(depth))
    denom = max(1e-6, far - near)
    normalized = (depth - near) / denom
    for i in range(px.size):
        x = int(px[i])
        y = int(py[i])
        value = float(1.0 - normalized[i])
        canvas[y, x] = max(canvas[y, x], value)
    image = Image.fromarray((canvas * 255.0).astype(np.uint8), mode="L").convert("RGB")
    return ProjectionRaster(image=image, transform=transform, depth_range=(near, far))

