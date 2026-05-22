from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vision_3d_acquisition.contracts.result import ProcessingArtifact


def normalize_processing_artifacts(result: dict[str, Any], *, output_dir: Path | None = None) -> list[dict[str, Any]]:
    explicit = _validate_artifacts(result.get("artifacts"), generated="explicit")
    derived = _derive_artifacts(result, output_dir=output_dir)
    by_id: dict[str, ProcessingArtifact] = {item.artifact_id: item for item in explicit}
    for item in derived:
        by_id.setdefault(item.artifact_id, item)
    return [item.model_dump(mode="json") for item in by_id.values()]


def _validate_artifacts(value: Any, *, generated: str) -> list[ProcessingArtifact]:
    if not isinstance(value, list):
        return []
    validated: list[ProcessingArtifact] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        payload = dict(item)
        payload.setdefault("generated", generated)
        payload.setdefault("metadata", {})
        payload.setdefault("preview_available", bool(payload.get("path")))
        payload.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        if payload.get("kind") == "overlay":
            payload.setdefault("coordinate_space", "projection_pixel")
            payload.setdefault("approximate", False)
            payload.setdefault("overlay_warnings", [])
            if not payload.get("target_artifact_id"):
                payload["approximate"] = True
                warnings = list(payload.get("overlay_warnings") or [])
                marker = "Compatibility mode: missing target_artifact_id; overlay rendered approximately."
                if marker not in warnings:
                    warnings.append(marker)
                payload["overlay_warnings"] = warnings
            if payload.get("coordinate_space") == "world_mm":
                payload["approximate"] = True
                warnings = list(payload.get("overlay_warnings") or [])
                marker = "Overlay is approximate: world coordinates projected onto static debug image."
                if marker not in warnings:
                    warnings.append(marker)
                payload["overlay_warnings"] = warnings
        try:
            validated.append(ProcessingArtifact.model_validate(payload))
        except Exception:
            continue
    return validated


def _derive_artifacts(result: dict[str, Any], *, output_dir: Path | None) -> list[ProcessingArtifact]:
    files = result.get("files") if isinstance(result.get("files"), dict) else {}
    processed_at = result.get("processed_at") or datetime.now(timezone.utc).isoformat()
    artifacts: list[ProcessingArtifact] = []
    if isinstance(files.get("input_preview"), str) and files.get("input_preview"):
        artifacts.append(
            _artifact(
                artifact_id="input_preview",
                stage_id="source_decode",
                kind="image",
                title="Input preview",
                path=str(files.get("input_preview")),
                mime_type="image/png",
                processed_at=processed_at,
                metadata={"source": "result_files", "derived_from": ["input_preview"]},
            )
        )
    if isinstance(files.get("projection_xy_topdown"), str) and files.get("projection_xy_topdown"):
        artifacts.append(
            _artifact(
                artifact_id="xy_topdown",
                stage_id="segmentation",
                kind="image",
                title="XY top-down projection",
                path=str(files.get("projection_xy_topdown")),
                mime_type="image/png",
                processed_at=processed_at,
                projection_type="xy_topdown",
                projection_coordinate_system={"pixel_per_mm": 2.0},
                projection_metadata={"background_style": "black", "colormap": "gray"},
                projection_transform_id="xy_topdown_transform",
                metadata={"source": "result_files", "derived_from": ["projection_xy_topdown"]},
            )
        )
    if isinstance(files.get("projection_xz_side"), str) and files.get("projection_xz_side"):
        artifacts.append(
            _artifact(
                artifact_id="xz_side",
                stage_id="segmentation",
                kind="image",
                title="XZ side projection",
                path=str(files.get("projection_xz_side")),
                mime_type="image/png",
                processed_at=processed_at,
                projection_type="xz_side",
                projection_coordinate_system={"pixel_per_mm": 2.0},
                projection_metadata={"background_style": "black", "colormap": "gray"},
                projection_transform_id="xz_side_transform",
                metadata={"source": "result_files", "derived_from": ["projection_xz_side"]},
            )
        )
    if isinstance(files.get("projection_yz_side"), str) and files.get("projection_yz_side"):
        artifacts.append(
            _artifact(
                artifact_id="yz_side",
                stage_id="segmentation",
                kind="image",
                title="YZ side projection",
                path=str(files.get("projection_yz_side")),
                mime_type="image/png",
                processed_at=processed_at,
                projection_type="yz_side",
                projection_coordinate_system={"pixel_per_mm": 2.0},
                projection_metadata={"background_style": "black", "colormap": "gray"},
                projection_transform_id="yz_side_transform",
                metadata={"source": "result_files", "derived_from": ["projection_yz_side"]},
            )
        )
    if isinstance(files.get("overlay"), str) and files.get("overlay"):
        artifacts.append(
            _artifact(
                artifact_id="classification_overlay_image",
                stage_id="classification",
                kind="image",
                title="Classification overlay",
                path=str(files.get("overlay")),
                mime_type="image/png",
                processed_at=processed_at,
                metadata={"source": "result_files", "derived_from": ["overlay"]},
            )
        )
    if isinstance(files.get("point_cloud"), str) and files.get("point_cloud"):
        artifacts.append(
            _artifact(
                artifact_id="point_cloud_input",
                stage_id="source_decode",
                kind="point_cloud",
                title="Input point cloud",
                path=str(files.get("point_cloud")),
                mime_type="application/octet-stream",
                processed_at=processed_at,
                preview_available=False,
                metadata={
                    "source": "result_files",
                    "coordinate_frame": "camera",
                    "units": "mm",
                    "point_count": ((result.get("input_stats") or {}) if isinstance(result.get("input_stats"), dict) else {}).get("point_count"),
                    "projection_references": ["input_preview", "foreground_clusters"],
                },
            )
        )
    debug_map = {
        "debug_plane_segmentation": ("plane_segmentation", "Plane segmentation"),
        "debug_foreground": ("foreground_extraction", "Foreground extraction"),
        "debug_clusters": ("foreground_clusters", "Foreground clusters"),
        "debug_height": ("height", "Height debug"),
        "debug_segmentation": ("segmentation", "Segmentation debug"),
        "debug_calibrated_planes": ("calibrated_planes", "Calibrated planes"),
    }
    for key, (artifact_id, title) in debug_map.items():
        path = files.get(key)
        if not isinstance(path, str) or not path:
            continue
        artifacts.append(
            _artifact(
                artifact_id=artifact_id,
                stage_id="segmentation",
                kind="image",
                title=title,
                path=path,
                mime_type="image/png",
                processed_at=processed_at,
                metadata={"source": "result_files", "derived_from": [key]},
            )
        )
    if not any(item.artifact_id == "xy_topdown" for item in artifacts):
        legacy_projection_path = files.get("debug_clusters") or files.get("debug_foreground") or files.get("input_preview")
        if isinstance(legacy_projection_path, str) and legacy_projection_path:
            artifacts.append(
                _artifact(
                    artifact_id="xy_topdown",
                    stage_id="segmentation",
                    kind="image",
                    title="XY top-down projection (compatibility)",
                    path=str(legacy_projection_path),
                    mime_type="image/png",
                    processed_at=processed_at,
                    projection_type="xy_topdown",
                    projection_coordinate_system={"pixel_per_mm": 1.0},
                    projection_metadata={"compatibility_mode": True, "source": "legacy_debug_screenshot"},
                    projection_transform_id="compat_xy_topdown_transform",
                    metadata={"source": "result_files", "derived_from": ["debug_clusters"], "compatibility_mode": True},
                )
            )
    objects = list(result.get("objects") or []) + list(result.get("rejected_objects") or [])
    if objects:
        artifacts.append(
            _artifact(
                artifact_id="classification_table",
                stage_id="classification",
                kind="table",
                title="Classification results",
                processed_at=processed_at,
                metadata={"source": "result_objects", "producer": "BallClassificationStage", "modality": "point_cloud"},
            )
        )
        artifacts.append(
            _artifact(
                artifact_id="measurement_table",
                stage_id="measurement",
                kind="table",
                title="Object measurements",
                processed_at=processed_at,
                metadata={"source": "result_objects", "producer": "MeasureObjectsStage", "modality": "point_cloud"},
            )
        )
        for obj in objects:
            object_id = obj.get("object_id")
            if not isinstance(object_id, int):
                continue
            artifacts.append(
                _artifact(
                    artifact_id=f"classification_object_{object_id}",
                    stage_id="classification",
                    object_id=object_id,
                    kind="json",
                    title=f"Object #{object_id} classification",
                    processed_at=processed_at,
                    metadata={"source": "result_objects", "producer": "BallClassificationStage"},
                )
            )
            artifacts.append(
                _artifact(
                    artifact_id=f"measurement_object_{object_id}",
                    stage_id="measurement",
                    object_id=object_id,
                    kind="metric",
                    title=f"Object #{object_id} measurement",
                    processed_at=processed_at,
                    metadata={"source": "result_objects", "producer": "MeasureObjectsStage"},
                )
            )
        artifacts.extend(_derive_overlay_artifacts(objects=objects, artifacts=artifacts, processed_at=processed_at, output_dir=output_dir))
        artifacts.append(
            _artifact(
                artifact_id="measurement_summary",
                stage_id="measurement",
                kind="metric",
                title="Diameter/statistics summary",
                processed_at=processed_at,
                metadata={"source": "result_summary", "producer": "StatisticsStage"},
            )
        )
    artifacts.append(
        _artifact(
            artifact_id="result_payload",
            stage_id="result",
            kind="json",
            title="Full result payload",
            path="result.json",
            mime_type="application/json",
            preview_available=True,
            processed_at=processed_at,
            metadata={"source": "result_json"},
        )
    )
    return artifacts


def _derive_overlay_artifacts(
    *,
    objects: list[dict[str, Any]],
    artifacts: list[ProcessingArtifact],
    processed_at: str,
    output_dir: Path | None,
) -> list[ProcessingArtifact]:
    target_info = _preferred_overlay_target(artifacts)
    if target_info is None:
        return []
    target, coordinate_space = target_info
    positioned = _project_objects_for_overlay(objects)
    overlays: list[ProcessingArtifact] = []
    for item in positioned:
        object_id = item["object_id"]
        label = item["label"]
        bbox = item["bbox"]
        centroid = item["centroid"]
        object_target = target
        crop_artifact = _derive_object_crop_projection(
            object_id=object_id,
            bbox=bbox,
            base_target_id=target,
            artifacts=artifacts,
            processed_at=processed_at,
            output_dir=output_dir,
        )
        if crop_artifact is not None:
            overlays.append(crop_artifact)
            object_target = crop_artifact.artifact_id
        crop_coordinate = (crop_artifact.projection_coordinate_system if crop_artifact is not None else {}) or {}
        crop_width = float(crop_coordinate.get("image_width") or bbox["width"])
        crop_height = float(crop_coordinate.get("image_height") or bbox["height"])
        crop_cx = crop_width / 2.0
        crop_cy = crop_height / 2.0
        overlays.append(
            _artifact(
                artifact_id=f"segmentation_overlay_bbox_{object_id}",
                stage_id="segmentation",
                object_id=object_id,
                kind="overlay",
                title=f"Object #{object_id} bounding box",
                processed_at=processed_at,
                metadata={
                    "source": "result_objects",
                    "derived_from": [f"measurement_object_{object_id}"],
                    "plot_width": 960,
                    "plot_height": 540,
                },
                overlay_type="bbox",
                coordinate_space=coordinate_space,
                target_artifact_id=target,
                geometry=bbox,
                style={"stroke": "#00ff88", "line_width": 2, "fill": "rgba(0,255,136,0.08)"},
                source_artifact_ids=[f"measurement_object_{object_id}"],
                approximate=False,
            )
        )
        overlays.append(
            _artifact(
                artifact_id=f"classification_overlay_ellipse_{object_id}",
                stage_id="classification",
                object_id=object_id,
                kind="overlay",
                title=f"Object #{object_id} ellipse fit",
                processed_at=processed_at,
                metadata={
                    "source": "result_objects",
                    "derived_from": [f"classification_object_{object_id}"],
                    "plot_width": 960,
                    "plot_height": 540,
                },
                overlay_type="ellipse",
                coordinate_space=coordinate_space,
                target_artifact_id=object_target,
                geometry={
                    "cx": crop_cx,
                    "cy": crop_cy,
                    "rx": max(8, crop_width * 0.42),
                    "ry": max(8, crop_height * 0.42),
                    "rotation_deg": 0,
                },
                style={"stroke": "#4f8cff", "line_width": 2, "fill": "rgba(79,140,255,0.12)"},
                source_artifact_ids=[f"classification_object_{object_id}"],
                approximate=False,
            )
        )
        overlays.append(
            _artifact(
                artifact_id=f"measurement_overlay_centroid_{object_id}",
                stage_id="measurement",
                object_id=object_id,
                kind="overlay",
                title=f"Object #{object_id} centroid",
                processed_at=processed_at,
                metadata={
                    "source": "result_objects",
                    "derived_from": [f"measurement_object_{object_id}"],
                    "plot_width": 960,
                    "plot_height": 540,
                },
                overlay_type="centroid",
                coordinate_space=coordinate_space,
                target_artifact_id=object_target,
                geometry={"x": crop_cx, "y": crop_cy},
                style={"stroke": "#ff8a00", "fill": "#ff8a00", "line_width": 2, "radius": 5},
                source_artifact_ids=[f"measurement_object_{object_id}"],
                approximate=False,
            )
        )
        overlays.append(
            _artifact(
                artifact_id=f"classification_overlay_label_{object_id}",
                stage_id="classification",
                object_id=object_id,
                kind="overlay",
                title=f"Object #{object_id} label",
                processed_at=processed_at,
                metadata={
                    "source": "result_objects",
                    "derived_from": [f"classification_object_{object_id}"],
                    "plot_width": 960,
                    "plot_height": 540,
                },
                overlay_type="text",
                coordinate_space=coordinate_space,
                target_artifact_id=object_target,
                geometry={
                    "x": 8,
                    "y": max(16, crop_height - 8),
                    "anchor_x": crop_cx,
                    "anchor_y": crop_cy,
                },
                style={"label": label, "stroke": "#ffffff", "fill": "#101820"},
                source_artifact_ids=[f"classification_object_{object_id}"],
                approximate=False,
            )
        )
    return overlays


def _derive_object_crop_projection(
    *,
    object_id: int,
    bbox: dict[str, float],
    base_target_id: str,
    artifacts: list[ProcessingArtifact],
    processed_at: str,
    output_dir: Path | None,
) -> ProcessingArtifact | None:
    if output_dir is None:
        return None
    base = next((item for item in artifacts if item.artifact_id == base_target_id and item.path), None)
    if base is None:
        return None
    source = output_dir / str(base.path)
    if not source.is_file():
        return None
    try:
        from PIL import Image
    except Exception:
        return None
    image = Image.open(source).convert("RGB")
    x = max(0, int(float(bbox.get("x", 0.0))))
    y = max(0, int(float(bbox.get("y", 0.0))))
    w = max(16, int(float(bbox.get("width", 32.0))))
    h = max(16, int(float(bbox.get("height", 32.0))))
    right = min(image.width, x + w)
    bottom = min(image.height, y + h)
    if right <= x or bottom <= y:
        return None
    crop = image.crop((x, y, right, bottom))
    filename = f"projection_object_{object_id}.png"
    crop.save(output_dir / filename)
    return _artifact(
        artifact_id=f"object_crop_{object_id}",
        stage_id="classification",
        object_id=object_id,
        kind="image",
        title=f"Object #{object_id} local projection",
        path=filename,
        mime_type="image/png",
        preview_available=True,
        processed_at=processed_at,
        projection_type="object_crop",
        projection_coordinate_system={"image_width": crop.width, "image_height": crop.height},
        projection_metadata={"source_point_cloud_id": "point_cloud_input", "source_object_id": object_id},
        projection_transform_id=f"object_crop_{object_id}_transform",
        metadata={"source": "derived_projection_crop", "derived_from": [base_target_id]},
    )


def _preferred_overlay_target(artifacts: list[ProcessingArtifact]) -> tuple[str, str] | None:
    by_id = {item.artifact_id for item in artifacts if item.kind == "image"}
    by_artifact = {item.artifact_id: item for item in artifacts if item.kind == "image"}

    # Legacy compatibility: when debug_* artifacts are the primary rendered surface,
    # keep overlay targets and spaces aligned with historical plot-based rendering.
    if "foreground_clusters" in by_id:
        foreground = by_artifact["foreground_clusters"]
        if not foreground.projection_type:
            return "foreground_clusters", "plot_pixel"

    for candidate in ("xy_topdown", "classification_overlay_image", "foreground_extraction", "plane_segmentation", "input_preview"):
        if candidate in by_id:
            return candidate, "projection_pixel"

    if "foreground_clusters" in by_id:
        return "foreground_clusters", "projection_pixel"
    return None


def _project_objects_for_overlay(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    coordinates: list[tuple[float, float]] = []
    for index, obj in enumerate(objects):
        object_id = obj.get("object_id")
        if not isinstance(object_id, int):
            continue
        center = _xy_from_object(obj, fallback_index=index)
        coordinates.append(center)
        projected.append(
            {
                "object_id": object_id,
                "raw_x": center[0],
                "raw_y": center[1],
                "dimensions": obj.get("dimensions_mm"),
                "class_name": obj.get("class_name") or "unknown",
                "confidence": obj.get("confidence"),
            }
        )
    if not projected:
        return []
    min_x = min(x for x, _ in coordinates)
    max_x = max(x for x, _ in coordinates)
    min_y = min(y for _, y in coordinates)
    max_y = max(y for _, y in coordinates)
    spread_x = max(1.0, max_x - min_x)
    spread_y = max(1.0, max_y - min_y)
    width = 960.0
    height = 540.0
    margin = 48.0
    for item in projected:
        nx = margin + ((item["raw_x"] - min_x) / spread_x) * (width - margin * 2)
        ny = margin + ((item["raw_y"] - min_y) / spread_y) * (height - margin * 2)
        dims = item["dimensions"]
        bbox_w = 78.0
        bbox_h = 62.0
        if isinstance(dims, (list, tuple)) and len(dims) >= 2:
            try:
                bbox_w = max(40.0, min(220.0, float(dims[0]) * 2.0))
                bbox_h = max(34.0, min(200.0, float(dims[1]) * 2.0))
            except (TypeError, ValueError):
                pass
        item["bbox"] = {
            "x": round(nx - bbox_w / 2, 1),
            "y": round(ny - bbox_h / 2, 1),
            "width": round(bbox_w, 1),
            "height": round(bbox_h, 1),
        }
        item["centroid"] = {"x": round(nx, 1), "y": round(ny, 1)}
        confidence = item["confidence"]
        suffix = ""
        if isinstance(confidence, (int, float)):
            suffix = f" {int(round(float(confidence) * 100))}%"
        item["label"] = f"{item['class_name']}{suffix}"
    return projected


def _xy_from_object(obj: dict[str, Any], *, fallback_index: int) -> tuple[float, float]:
    center = obj.get("center_mm")
    if isinstance(center, (list, tuple)) and len(center) >= 2:
        try:
            return float(center[0]), float(center[1])
        except (TypeError, ValueError):
            pass
    bbox_min = obj.get("bbox_min_mm")
    bbox_max = obj.get("bbox_max_mm")
    if isinstance(bbox_min, (list, tuple)) and isinstance(bbox_max, (list, tuple)) and len(bbox_min) >= 2 and len(bbox_max) >= 2:
        try:
            return (float(bbox_min[0]) + float(bbox_max[0])) / 2, (float(bbox_min[1]) + float(bbox_max[1])) / 2
        except (TypeError, ValueError):
            pass
    return float(fallback_index), float(fallback_index)


def _artifact(
    *,
    artifact_id: str,
    stage_id: str,
    kind: str,
    title: str,
    processed_at: str,
    object_id: int | None = None,
    description: str | None = None,
    path: str | None = None,
    mime_type: str | None = None,
    preview_available: bool | None = None,
    metadata: dict[str, Any] | None = None,
    overlay_type: str | None = None,
    coordinate_space: str | None = None,
    target_artifact_id: str | None = None,
    geometry: dict[str, Any] | None = None,
    style: dict[str, Any] | None = None,
    source_artifact_ids: list[str] | None = None,
    approximate: bool = False,
    overlay_warnings: list[str] | None = None,
    projection_type: str | None = None,
    projection_coordinate_system: dict[str, Any] | None = None,
    projection_metadata: dict[str, Any] | None = None,
    projection_transform_id: str | None = None,
) -> ProcessingArtifact:
    return ProcessingArtifact.model_validate(
        {
            "artifact_id": artifact_id,
            "stage_id": stage_id,
            "object_id": object_id,
            "kind": kind,
            "title": title,
            "description": description,
            "path": path,
            "mime_type": mime_type,
            "preview_available": bool(path) if preview_available is None else preview_available,
            "created_at": processed_at,
            "metadata": metadata or {},
            "generated": "derived",
            "overlay_type": overlay_type,
            "coordinate_space": coordinate_space,
            "target_artifact_id": target_artifact_id,
            "geometry": geometry or {},
            "style": style or {},
            "source_artifact_ids": source_artifact_ids or [],
            "approximate": approximate,
            "overlay_warnings": overlay_warnings or [],
            "projection_type": projection_type,
            "projection_coordinate_system": projection_coordinate_system or {},
            "projection_metadata": projection_metadata or {},
            "projection_transform_id": projection_transform_id,
        }
    )
