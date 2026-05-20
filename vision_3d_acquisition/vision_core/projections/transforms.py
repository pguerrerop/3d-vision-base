from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ProjectionTransform:
    transform_id: str
    projection_type: str
    bounds: dict[str, float]
    width: int
    height: int
    pixel_per_mm: float

    def to_pixel(self, primary: np.ndarray, secondary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_min = self.bounds["primary_min"]
        y_min = self.bounds["secondary_min"]
        px = np.clip(((primary - x_min) * self.pixel_per_mm).astype(np.int32), 0, self.width - 1)
        py = np.clip(((secondary - y_min) * self.pixel_per_mm).astype(np.int32), 0, self.height - 1)
        return px, (self.height - 1) - py

