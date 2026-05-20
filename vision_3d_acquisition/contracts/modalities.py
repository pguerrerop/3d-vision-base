from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


CaptureModality = Literal[
    "point_cloud",
    "heightmap",
    "reflectance",
    "rgb",
    "rgb_video",
    "laser_rgb",
]

MODALITY_ORDER: tuple[CaptureModality, ...] = (
    "point_cloud",
    "heightmap",
    "reflectance",
    "rgb",
    "rgb_video",
    "laser_rgb",
)

POINT_CLOUD_FILE_KEYS = {"point_cloud", "point_cloud_npz"}
MODALITY_FILE_KEYS: dict[CaptureModality, tuple[str, ...]] = {
    "point_cloud": ("point_cloud", "point_cloud_npz"),
    "heightmap": ("heightmap", "height"),
    "reflectance": ("reflectance",),
    "rgb": ("rgb",),
    "rgb_video": ("rgb_video",),
    "laser_rgb": ("laser_rgb", "laser_overlay", "laser_image"),
}


@dataclass(frozen=True)
class CaptureAsset:
    modality: CaptureModality
    key: str
    path: Path


def normalize_modalities(values: list[str] | tuple[str, ...] | None) -> list[CaptureModality]:
    if not values:
        return []
    known = set(MODALITY_ORDER)
    normalized: list[CaptureModality] = []
    for value in values:
        if value in known and value not in normalized:
            normalized.append(value)  # type: ignore[arg-type]
    return sorted(normalized, key=MODALITY_ORDER.index)


def infer_modalities(metadata: dict[str, Any] | None = None, take_dir: Path | None = None) -> list[CaptureModality]:
    metadata = metadata or {}
    explicit = normalize_modalities(metadata.get("modalities"))
    inferred: set[CaptureModality] = set(explicit)
    files = metadata.get("files") if isinstance(metadata.get("files"), dict) else {}
    for modality, keys in MODALITY_FILE_KEYS.items():
        if any(files.get(key) for key in keys):
            inferred.add(modality)
    if take_dir is not None and take_dir.is_dir():
        for child in take_dir.iterdir():
            if child.is_file():
                modality = infer_modality_from_filename(child.name)
                if modality is not None:
                    inferred.add(modality)
    return sorted(inferred, key=MODALITY_ORDER.index)


def infer_modality_from_filename(filename: str) -> CaptureModality | None:
    name = filename.lower()
    suffix = Path(name).suffix
    if suffix in {".ply", ".pcd"}:
        return "point_cloud"
    if suffix == ".npz" and "point" in name:
        return "point_cloud"
    if suffix in {".tif", ".tiff"} and ("height" in name or "depth" in name):
        return "heightmap"
    if "height" in name and suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy", ".npz"}:
        return "heightmap"
    if "reflectance" in name and suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        return "reflectance"
    if ("laser" in name or "overlay" in name) and suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        return "laser_rgb"
    if name.startswith(("rgb_video", "video", "camera_video")) and suffix in {".mp4", ".avi", ".mov", ".mkv"}:
        return "rgb_video"
    if name.startswith(("rgb", "image", "camera", "preview")) and suffix in {".png", ".jpg", ".jpeg", ".bmp"}:
        return "rgb"
    return None


def assets_by_modality(metadata: dict[str, Any] | None, take_dir: Path | None = None) -> dict[str, dict[str, str]]:
    metadata = metadata or {}
    files = metadata.get("files") if isinstance(metadata.get("files"), dict) else {}
    grouped: dict[str, dict[str, str]] = {}
    for modality, keys in MODALITY_FILE_KEYS.items():
        for key in keys:
            value = files.get(key)
            if value:
                grouped.setdefault(modality, {})[key] = str(value)
    if take_dir is not None and take_dir.is_dir():
        known_files = {str(value) for value in files.values() if value}
        for child in take_dir.iterdir():
            if not child.is_file() or child.name in known_files or child.name in {"READY", "metadata.json"}:
                continue
            modality = infer_modality_from_filename(child.name)
            if modality is not None:
                grouped.setdefault(modality, {}).setdefault(modality, child.name)
    return grouped


def format_modalities(modalities: list[str] | tuple[str, ...] | set[str] | None) -> str:
    values = list(modalities or [])
    return ", ".join(values) if values else "none"
