from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


OUT_DIR = Path("data/demo/rgb_steel_balls")


def _base() -> np.ndarray:
    img = np.zeros((320, 480, 3), dtype=np.uint8)
    img[:] = (28, 28, 28)
    return img


def _save(name: str, image: np.ndarray) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_DIR / name), image)


def generate() -> dict[str, dict[str, object]]:
    rng = np.random.default_rng(42)
    metadata: dict[str, dict[str, object]] = {}

    one = _base()
    cv2.circle(one, (240, 160), 58, (215, 215, 215), -1)
    _save("one_good_ball.png", one)
    metadata["one_good_ball.png"] = {"expected_object_count": 1, "expected_classes": ["ball"], "notes": "single centered spherical ball"}

    multi = _base()
    cv2.circle(multi, (120, 170), 42, (205, 205, 205), -1)
    cv2.circle(multi, (250, 145), 54, (220, 220, 220), -1)
    cv2.circle(multi, (360, 190), 36, (195, 195, 195), -1)
    _save("multiple_good_balls.png", multi)
    metadata["multiple_good_balls.png"] = {"expected_object_count": 3, "expected_classes": ["ball", "ball", "ball"], "notes": "multiple near-spherical objects"}

    oval = _base()
    cv2.ellipse(oval, (240, 160), (96, 52), 18, 0, 360, (210, 210, 210), -1)
    _save("oval_non_spherical.png", oval)
    metadata["oval_non_spherical.png"] = {"expected_object_count": 1, "expected_classes": ["non-ball"], "notes": "elongated shape for non-spherical classification"}

    mixed = _base()
    cv2.circle(mixed, (110, 135), 44, (210, 210, 210), -1)
    cv2.rectangle(mixed, (230, 95), (330, 200), (185, 185, 185), -1)
    tri = np.array([[390, 210], [450, 120], [452, 260]], dtype=np.int32)
    cv2.fillPoly(mixed, [tri], (165, 165, 165))
    _save("mixed_objects.png", mixed)
    metadata["mixed_objects.png"] = {"expected_object_count": 3, "expected_classes": ["ball", "non-ball", "non-ball"], "notes": "mixed spherical and non-spherical objects"}

    low = _base()
    gradient = np.tile(np.linspace(20, 52, low.shape[1], dtype=np.uint8), (low.shape[0], 1))
    low[:, :, 0] = gradient
    low[:, :, 1] = gradient
    low[:, :, 2] = gradient
    cv2.circle(low, (240, 160), 56, (90, 90, 90), -1)
    noise = rng.normal(0, 3, low.shape).astype(np.int16)
    low = np.clip(low.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    _save("low_contrast_ball.png", low)
    metadata["low_contrast_ball.png"] = {"expected_object_count": 1, "expected_classes": ["ball"], "notes": "low contrast object for normalization and threshold tuning"}

    return metadata


def main() -> None:
    metadata = generate()
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Generated {len(metadata)} RGB demo samples in {OUT_DIR}")


if __name__ == "__main__":
    main()
