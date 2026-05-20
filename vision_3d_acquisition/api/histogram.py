from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from vision_3d_acquisition.api.schemas import TakeDetail
from vision_3d_acquisition.api.settings import ApiSettings


def resolve_source_image_path(settings: ApiSettings, detail: TakeDetail, requested_filename: str | None = None) -> tuple[Path | None, str | None]:
    if requested_filename:
        candidate = (settings.incoming_dir / detail.take_id / requested_filename).resolve()
        base = (settings.incoming_dir / detail.take_id).resolve()
        try:
            candidate.relative_to(base)
        except ValueError:
            return None, None
        return (candidate if candidate.is_file() else None), requested_filename

    groups = detail.assets or {}
    for group, keys in (("rgb", ["rgb"]), ("reflectance", ["reflectance"]), ("heightmap", ["heightmap", "height"])):
        payload = groups.get(group) or {}
        for key in keys:
            filename = payload.get(key)
            if not filename:
                continue
            candidate = settings.incoming_dir / detail.take_id / filename
            if candidate.is_file():
                return candidate, filename
    return None, None


def _cache_path(settings: ApiSettings, take_id: str, filename: str, max_dim: int, bins: int, image_path: Path) -> Path:
    stat = image_path.stat()
    key = f"{take_id}|{filename}|{stat.st_mtime_ns}|{stat.st_size}|{max_dim}|{bins}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    cache_dir = settings.incoming_dir / take_id / ".stage_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"hist_{digest}.json"


def load_or_compute_histogram(settings: ApiSettings, take_id: str, filename: str, image_path: Path, max_dim: int = 512, bins: int = 32) -> dict[str, Any]:
    cache_file = _cache_path(settings, take_id, filename, max_dim, bins, image_path)
    if cache_file.is_file():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        width, height = image.size
        scale = min(1.0, float(max_dim) / float(max(width, height) if max(width, height) > 0 else 1))
        sample_w = max(1, int(round(width * scale)))
        sample_h = max(1, int(round(height * scale)))
        if (sample_w, sample_h) != (width, height):
            image = image.resize((sample_w, sample_h), Image.Resampling.BILINEAR)
        arr = np.asarray(image, dtype=np.uint8)

    lum = (0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]).astype(np.float32)
    hist, _ = np.histogram(lum, bins=bins, range=(0, 256))
    payload: dict[str, Any] = {
        "source": filename,
        "sampled_width": int(arr.shape[1]),
        "sampled_height": int(arr.shape[0]),
        "sampled_pixels": int(arr.shape[0] * arr.shape[1]),
        "bins": [int(x) for x in hist.tolist()],
        "min": float(lum.min()) if lum.size else 0.0,
        "max": float(lum.max()) if lum.size else 0.0,
        "mean": float(lum.mean()) if lum.size else 0.0,
        "bin_count": bins,
        "max_dimension": max_dim,
    }
    try:
        cache_file.write_text(json.dumps(payload), encoding="utf-8")
    except Exception:
        pass
    return payload
