from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from vision_3d_acquisition.calibration.camera_2d import axis_length_mm
from vision_3d_acquisition.processes.models import PipelineStepConfig, StepRunReport


@dataclass
class StepExecutionResult:
    artifacts: list[dict[str, Any]]
    overlays: list[dict[str, Any]]
    measurements: list[dict[str, Any]]
    warnings: list[str]
    metrics: dict[str, float]
    debug: dict[str, Any]
    state: dict[str, Any]


class PipelineExecutor2D:
    def __init__(self) -> None:
        self._working: dict[str, Any] = {}

    def run_step(self, step: PipelineStepConfig, *, image: np.ndarray, existing: dict[str, Any]) -> tuple[StepRunReport, StepExecutionResult]:
        started = time.perf_counter()
        self._working = dict(existing)
        warnings: list[str] = []
        errors: list[str] = []
        artifacts: list[dict[str, Any]] = []
        overlays: list[dict[str, Any]] = []
        measurements: list[dict[str, Any]] = []
        metrics: dict[str, float] = {}
        debug: dict[str, Any] = {}

        try:
            if not step.enabled:
                report = StepRunReport(step_id=step.step_id, algorithm_key=step.algorithm_key, status="skipped")
                return report, StepExecutionResult([], [], [], [], {}, {}, self._working)

            if step.step_id == "input":
                self._working["source_rgb"] = image
                self._working["current_rgb"] = image.copy()
                artifacts.append(_artifact("source_rgb_image", "input", "image", "Source RGB image"))
            elif step.step_id == "crop_roi":
                rgb = _require_rgb(self._working)
                h, w = rgb.shape[:2]
                roi = {
                    "enabled": bool(step.params.get("enabled", False)),
                    "x": int(step.params.get("x", 0)),
                    "y": int(step.params.get("y", 0)),
                    "width": int(step.params.get("width", w)),
                    "height": int(step.params.get("height", h)),
                }
                if roi["enabled"]:
                    x = max(0, min(roi["x"], w - 1))
                    y = max(0, min(roi["y"], h - 1))
                    rw = max(1, min(roi["width"], w - x))
                    rh = max(1, min(roi["height"], h - y))
                    cropped = rgb[y : y + rh, x : x + rw].copy()
                    self._working["current_rgb"] = cropped
                    self._working["roi_offset"] = (x, y)
                    artifacts.append(_artifact("roi_image", "crop_roi", "image", "ROI image"))
                else:
                    self._working["roi_offset"] = (0, 0)
                    artifacts.append(_artifact("roi_passthrough", "crop_roi", "json", "ROI pass-through metadata"))
                self._working["roi"] = roi
                artifacts.append(_artifact("roi_metadata", "crop_roi", "json", "ROI metadata"))
                overlays.append(_overlay("roi_overlay", "crop_roi", "bbox", {"x": roi["x"], "y": roi["y"], "width": roi["width"], "height": roi["height"]}, style={"stroke": "#ffb703"}))
            elif step.step_id == "rgb_to_gray":
                rgb = _require_rgb(self._working)
                method = str(step.params.get("method", "luminance"))
                gray = _to_gray(rgb, method)
                self._working["gray"] = gray
                artifacts.append(_artifact("grayscale_image", "rgb_to_gray", "image", "Grayscale image"))
            elif step.step_id == "normalize_lighting":
                gray = _require_gray(self._working)
                method = str(step.params.get("method", "none"))
                clip_limit = float(step.params.get("clip_limit", 2.0))
                tile = int(step.params.get("tile_grid_size", 8))
                blur_kernel = int(step.params.get("background_blur_kernel", 31))
                out = gray.copy()
                if method == "clahe":
                    clahe = cv2.createCLAHE(clipLimit=max(0.1, clip_limit), tileGridSize=(max(2, tile), max(2, tile)))
                    out = clahe.apply(gray)
                elif method == "background_subtract":
                    if blur_kernel % 2 == 0:
                        blur_kernel += 1
                    blurred = cv2.GaussianBlur(gray, (max(3, blur_kernel), max(3, blur_kernel)), 0)
                    out = cv2.normalize(cv2.subtract(gray, blurred), None, 0, 255, cv2.NORM_MINMAX)
                self._working["normalized_gray"] = out
                artifacts.append(_artifact("normalized_grayscale_image", "normalize_lighting", "image", "Normalized grayscale image"))
            elif step.step_id == "threshold":
                gray = self._working.get("normalized_gray")
                if not isinstance(gray, np.ndarray):
                    gray = _require_gray(self._working)
                source_h, source_w = gray.shape[:2]
                blur_kernel = max(1, int(step.params.get("blur_kernel", 5)))
                if blur_kernel % 2 == 0:
                    blur_kernel += 1
                if blur_kernel > 1:
                    gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
                mode = str(step.params.get("mode", step.params.get("threshold_mode", "fixed"))).lower()
                value = int(step.params.get("value", step.params.get("threshold_value", 125)))
                adaptive_block = int(step.params.get("adaptive_block_size", 31))
                adaptive_c = float(step.params.get("adaptive_c", 2))
                invert = bool(step.params.get("invert", False))
                roi_enabled = bool(step.params.get("roi_enabled", False))
                roi_type = str(step.params.get("roi_type", "rectangle")).lower()
                roi_x = int(step.params.get("roi_x", 0))
                roi_y = int(step.params.get("roi_y", 0))
                roi_w = int(step.params.get("roi_width", source_w))
                roi_h = int(step.params.get("roi_height", source_h))
                roi_polygon_raw = step.params.get("roi_polygon_points") or []
                if adaptive_block % 2 == 0:
                    adaptive_block += 1
                roi_meta = {
                    "enabled": roi_enabled,
                    "x": 0,
                    "y": 0,
                    "width": source_w,
                    "height": source_h,
                }
                roi_mask = None
                if roi_enabled and roi_type == "polygon":
                    points: list[list[int]] = []
                    if isinstance(roi_polygon_raw, list):
                        for item in roi_polygon_raw:
                            if not isinstance(item, (list, tuple)) or len(item) < 2:
                                continue
                            px = int(max(0, min(source_w - 1, round(float(item[0])))))
                            py = int(max(0, min(source_h - 1, round(float(item[1])))))
                            points.append([px, py])
                    if len(points) >= 3:
                        roi_mask = np.zeros_like(gray, dtype=np.uint8)
                        cv2.fillPoly(roi_mask, [np.asarray(points, dtype=np.int32)], 255)
                        xs = [point[0] for point in points]
                        ys = [point[1] for point in points]
                        roi_meta = {
                            "enabled": True,
                            "type": "polygon",
                            "polygon_points": points,
                            "bounds": {
                                "x": int(min(xs)),
                                "y": int(min(ys)),
                                "width": int(max(xs) - min(xs) + 1),
                                "height": int(max(ys) - min(ys) + 1),
                            },
                        }
                        threshold_input = gray
                    else:
                        warnings.append("Polygon ROI has fewer than 3 points; falling back to rectangle ROI.")
                        roi_type = "rectangle"
                if roi_enabled and roi_type == "rectangle":
                    rx = max(0, min(roi_x, source_w - 1))
                    ry = max(0, min(roi_y, source_h - 1))
                    rw = max(1, min(roi_w, source_w - rx))
                    rh = max(1, min(roi_h, source_h - ry))
                    roi_meta = {"enabled": True, "type": "rectangle", "x": rx, "y": ry, "width": rw, "height": rh}
                    threshold_input = gray[ry:ry + rh, rx:rx + rw]
                    roi_mask = np.zeros_like(gray, dtype=np.uint8)
                    roi_mask[ry:ry + rh, rx:rx + rw] = 255
                else:
                    threshold_input = gray
                threshold_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
                if mode in {"adaptive", "adaptive_gaussian"}:
                    mask_roi = cv2.adaptiveThreshold(
                        threshold_input,
                        255,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        threshold_type,
                        max(3, adaptive_block),
                        adaptive_c,
                    )
                elif mode == "otsu":
                    _, mask_roi = cv2.threshold(threshold_input, 0, 255, threshold_type | cv2.THRESH_OTSU)
                elif mode in {"inverse_binary"}:
                    _, mask_roi = cv2.threshold(threshold_input, value, 255, cv2.THRESH_BINARY_INV)
                else:
                    _, mask_roi = cv2.threshold(threshold_input, value, 255, threshold_type)
                if roi_enabled and roi_type == "rectangle":
                    mask = np.zeros_like(gray, dtype=np.uint8)
                    rx = int(roi_meta["x"])
                    ry = int(roi_meta["y"])
                    rw = int(roi_meta["width"])
                    rh = int(roi_meta["height"])
                    mask[ry:ry + rh, rx:rx + rw] = mask_roi
                else:
                    mask = mask_roi
                if isinstance(roi_mask, np.ndarray):
                    mask = cv2.bitwise_and(mask, roi_mask)
                self._working["mask"] = mask
                self._working["threshold_mask"] = mask.copy()
                self._working["segmentation_roi"] = roi_meta
                self._working["segmentation_roi_mask"] = roi_mask
                artifacts.append(_artifact("threshold_mask", "threshold", "image", "Threshold mask"))
                threshold_foreground_pixels = int(np.count_nonzero(mask))
                threshold_coverage = float(threshold_foreground_pixels) / float(mask.size) if mask.size else 0.0
                artifacts[-1]["metadata"] = {
                    "threshold_mode": mode,
                    "threshold_value": value,
                    "invert": invert,
                    "adaptive_block_size": adaptive_block,
                    "adaptive_c": adaptive_c,
                    "blur_kernel": blur_kernel,
                    "foreground_pixels": threshold_foreground_pixels,
                    "foreground_coverage": threshold_coverage,
                    "roi": roi_meta,
                    "source_artifact_id": "normalized_grayscale_image",
                }
                metrics["foreground_ratio"] = threshold_coverage
            elif step.step_id == "morphology":
                mask = _require_mask(self._working)
                op = str(step.params.get("operation", "open_close")).lower()
                open_kernel = max(1, int(step.params.get("open_kernel", step.params.get("kernel_size", 3))))
                close_kernel = max(1, int(step.params.get("close_kernel", step.params.get("kernel_size", 5))))
                iterations = max(1, int(step.params.get("iterations", 1)))
                fill_holes = bool(step.params.get("fill_holes", True))
                cleanup_min_area = max(0, int(step.params.get("cleanup_min_area", step.params.get("min_component_area", 80))))
                cleanup_max_area_raw = step.params.get("cleanup_max_area", step.params.get("max_component_area"))
                cleanup_max_area = int(cleanup_max_area_raw) if cleanup_max_area_raw not in {None, ""} else None
                if cleanup_max_area is not None and cleanup_max_area <= 0:
                    cleanup_max_area = None
                cleanup_min_width = max(0, int(step.params.get("cleanup_min_width", 0)))
                cleanup_min_height = max(0, int(step.params.get("cleanup_min_height", 0)))
                cleanup_max_aspect_ratio = float(step.params.get("cleanup_max_aspect_ratio", 0.0))
                cleanup_border_reject = bool(step.params.get("cleanup_border_reject", False))
                cleanup_keep_largest_n = max(0, int(step.params.get("cleanup_keep_largest_n", 0)))
                open_kernel_mat = np.ones((open_kernel, open_kernel), np.uint8)
                close_kernel_mat = np.ones((close_kernel, close_kernel), np.uint8)
                op_map = {
                    "erode": lambda value: cv2.morphologyEx(value, cv2.MORPH_ERODE, open_kernel_mat, iterations=iterations),
                    "dilate": lambda value: cv2.morphologyEx(value, cv2.MORPH_DILATE, open_kernel_mat, iterations=iterations),
                    "open_only": lambda value: cv2.morphologyEx(value, cv2.MORPH_OPEN, open_kernel_mat, iterations=iterations),
                    "close_only": lambda value: cv2.morphologyEx(value, cv2.MORPH_CLOSE, close_kernel_mat, iterations=iterations),
                    "open_close": lambda value: cv2.morphologyEx(cv2.morphologyEx(value, cv2.MORPH_OPEN, open_kernel_mat, iterations=iterations), cv2.MORPH_CLOSE, close_kernel_mat, iterations=iterations),
                    "close_open": lambda value: cv2.morphologyEx(cv2.morphologyEx(value, cv2.MORPH_CLOSE, close_kernel_mat, iterations=iterations), cv2.MORPH_OPEN, open_kernel_mat, iterations=iterations),
                    "none": lambda value: value.copy(),
                }
                applied_op = op if op in op_map else "open_close"
                if op not in op_map:
                    warnings.append(f"Unknown morphology operation '{op}', using open_close.")
                out = op_map[applied_op](mask)
                if fill_holes:
                    flood = out.copy()
                    h, w = flood.shape[:2]
                    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
                    cv2.floodFill(flood, flood_mask, (0, 0), 255)
                    out = out | cv2.bitwise_not(flood)
                roi_meta = self._working.get("segmentation_roi") if isinstance(self._working.get("segmentation_roi"), dict) else {"enabled": False}
                roi_mask = self._working.get("segmentation_roi_mask")
                if isinstance(roi_mask, np.ndarray):
                    out = cv2.bitwise_and(out, roi_mask)
                num_labels_before, labels_before, stats_before, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
                kept_labels: list[int] = []
                rejected_labels: dict[int, str] = {}
                candidates: list[tuple[int, int]] = []
                for label in range(1, num_labels_before):
                    area = int(stats_before[label, cv2.CC_STAT_AREA])
                    x = int(stats_before[label, cv2.CC_STAT_LEFT])
                    y = int(stats_before[label, cv2.CC_STAT_TOP])
                    w = int(stats_before[label, cv2.CC_STAT_WIDTH])
                    h = int(stats_before[label, cv2.CC_STAT_HEIGHT])
                    aspect = float(max(w, h) / max(1, min(w, h)))
                    touches_border = (x <= 0 or y <= 0 or (x + w) >= out.shape[1] or (y + h) >= out.shape[0])
                    reason: str | None = None
                    if area < cleanup_min_area:
                        reason = "small"
                    elif cleanup_max_area is not None and area > cleanup_max_area:
                        reason = "large"
                    elif cleanup_min_width > 0 and w < cleanup_min_width:
                        reason = "small_width"
                    elif cleanup_min_height > 0 and h < cleanup_min_height:
                        reason = "small_height"
                    elif cleanup_max_aspect_ratio > 0 and aspect > cleanup_max_aspect_ratio:
                        reason = "aspect"
                    elif cleanup_border_reject and touches_border:
                        reason = "border"
                    if reason is not None:
                        rejected_labels[label] = reason
                    else:
                        candidates.append((label, area))
                if cleanup_keep_largest_n > 0 and len(candidates) > cleanup_keep_largest_n:
                    candidates = sorted(candidates, key=lambda item: item[1], reverse=True)
                    keep_set = {item[0] for item in candidates[:cleanup_keep_largest_n]}
                    for label, _area in candidates[cleanup_keep_largest_n:]:
                        rejected_labels[label] = "beyond_top_n"
                    kept_labels = list(keep_set)
                else:
                    kept_labels = [label for label, _ in candidates]
                cleaned = np.zeros_like(out)
                for label in kept_labels:
                    cleaned[labels_before == label] = 255
                out = cleaned
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
                self._working["mask"] = out
                self._working["morphology_mask"] = out.copy()
                self._working["cleaned_mask"] = out.copy()
                artifacts.append(_artifact("cleaned_mask", "morphology", "image", "Cleaned mask"))
                artifacts[-1]["metadata"] = {
                    "morphology_operation": applied_op,
                    "open_kernel": open_kernel,
                    "close_kernel": close_kernel,
                    "iterations": iterations,
                    "fill_holes": fill_holes,
                    "cleanup_min_area": cleanup_min_area,
                    "cleanup_max_area": cleanup_max_area,
                    "roi": roi_meta,
                    "source_artifact_id": "threshold_mask",
                }
                source_rgb = self._working.get("source_rgb")
                if isinstance(source_rgb, np.ndarray):
                    overlay = source_rgb.copy()
                    if overlay.ndim == 2:
                        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
                    overlay_mask = cv2.resize(out, (overlay.shape[1], overlay.shape[0]), interpolation=cv2.INTER_NEAREST)
                    color_layer = np.zeros_like(overlay)
                    color_layer[:, :] = (
                        int(step.params.get("overlay_color_b", 0)),
                        int(step.params.get("overlay_color_g", 255)),
                        int(step.params.get("overlay_color_r", 0)),
                    )
                    alpha = float(step.params.get("overlay_alpha", 0.35))
                    active = overlay_mask > 0
                    overlay[active] = cv2.addWeighted(overlay[active], 1.0 - alpha, color_layer[active], alpha, 0)
                    self._working["morphology_overlay_image"] = overlay
                    artifacts.append(_artifact("overlay_image", "morphology", "image", "Segmentation overlay image"))
                    artifacts[-1]["metadata"] = {
                        "overlay_type": "segmentation_mask",
                        "source_artifact_id": "source_rgb_image",
                        "mask_artifact_id": "cleaned_mask",
                        "overlay_alpha": alpha,
                    }
                    rejected_overlay = source_rgb.copy()
                    if rejected_overlay.ndim == 2:
                        rejected_overlay = cv2.cvtColor(rejected_overlay, cv2.COLOR_GRAY2BGR)
                    rejected_mask = np.zeros(out.shape, dtype=np.uint8)
                    for label, _reason in rejected_labels.items():
                        rejected_mask[labels_before == label] = 255
                    rejected_active = rejected_mask > 0
                    rejected_color = np.zeros_like(rejected_overlay)
                    rejected_color[:, :] = (24, 24, 220)
                    rejected_overlay[rejected_active] = cv2.addWeighted(rejected_overlay[rejected_active], 0.4, rejected_color[rejected_active], 0.6, 0)
                    self._working["rejected_components_overlay_image"] = rejected_overlay
                    artifacts.append(_artifact("rejected_components_overlay", "morphology", "image", "Rejected components overlay"))
                    artifacts[-1]["metadata"] = {
                        "overlay_type": "rejected_components",
                        "source_artifact_id": "source_rgb_image",
                        "mask_artifact_id": "threshold_mask",
                    }
                area_values: list[int] = []
                for label in range(1, num_labels):
                    area_values.append(int(stats[label, cv2.CC_STAT_AREA]))
                foreground = float(np.count_nonzero(out)) / float(out.size) if out.size else 0.0
                threshold_pixels = int(np.count_nonzero(mask))
                cleaned_pixels = int(np.count_nonzero(out))
                components_before = int(max(0, num_labels_before - 1))
                components_after = int(max(0, num_labels - 1))
                kept_component_areas = sorted(area_values)
                rejected_small = sum(1 for item in rejected_labels.values() if item in {"small", "small_width", "small_height"})
                rejected_large = sum(1 for item in rejected_labels.values() if item == "large")
                rejected_aspect = sum(1 for item in rejected_labels.values() if item == "aspect")
                rejected_border = sum(1 for item in rejected_labels.values() if item == "border")
                morph_metrics = {
                    "connected_components": int(max(0, num_labels - 1)),
                    "foreground_coverage_percent": round(foreground * 100.0, 4),
                    "largest_component_area": int(max(area_values) if area_values else 0),
                    "mean_component_area": float(sum(area_values) / len(area_values)) if area_values else 0.0,
                    "median_component_area": float(np.median(area_values)) if area_values else 0.0,
                    "average_component_area": float(sum(area_values) / len(area_values)) if area_values else 0.0,
                    "cleaned_foreground_percent": round(foreground * 100.0, 4),
                    "masks_generated": 2,
                    "components_before": components_before,
                    "components_after": components_after,
                    "removed_components": max(0, components_before - components_after),
                    "components_before_cleanup": components_before,
                    "components_after_cleanup": components_after,
                    "rejected_small": rejected_small,
                    "rejected_large": rejected_large,
                    "rejected_aspect": rejected_aspect,
                    "rejected_border": rejected_border,
                    "kept_component_areas": kept_component_areas,
                    "threshold_foreground_pixels": threshold_pixels,
                    "cleaned_foreground_pixels": cleaned_pixels,
                    "threshold_foreground_coverage": float(threshold_pixels) / float(mask.size) if mask.size else 0.0,
                    "cleaned_foreground_coverage": float(cleaned_pixels) / float(out.size) if out.size else 0.0,
                    "roi_enabled": bool(roi_meta.get("enabled", False)),
                }
                self._working["morphology_metrics"] = morph_metrics
                artifacts.append(_artifact("morphology_metrics", "morphology", "metric", "Morphology metrics"))
                artifacts[-1]["metadata"] = morph_metrics
                debug_payload = {
                    "stage": "segmentation",
                    "threshold": {
                        "mode": str(step.params.get("mode", step.params.get("threshold_mode", "fixed"))).lower(),
                        "value": int(step.params.get("value", step.params.get("threshold_value", 125))),
                        "invert": bool(step.params.get("invert", False)),
                        "blur_kernel": blur_kernel if "blur_kernel" in locals() else int(step.params.get("blur_kernel", 5)),
                        "foreground_pixels": threshold_pixels,
                        "foreground_coverage": float(threshold_pixels) / float(mask.size) if mask.size else 0.0,
                    },
                    "morphology": {
                        "operation": applied_op,
                        "open_kernel": open_kernel,
                        "close_kernel": close_kernel,
                        "iterations": iterations,
                        "fill_holes": fill_holes,
                        "min_component_area": cleanup_min_area,
                        "max_component_area": cleanup_max_area,
                        "min_width": cleanup_min_width,
                        "min_height": cleanup_min_height,
                        "max_aspect_ratio": cleanup_max_aspect_ratio,
                        "border_reject": cleanup_border_reject,
                        "keep_largest_n": cleanup_keep_largest_n,
                        "components_before": components_before,
                        "components_after": components_after,
                        "removed_components": max(0, components_before - components_after),
                        "rejected_small": rejected_small,
                        "rejected_large": rejected_large,
                        "rejected_aspect": rejected_aspect,
                        "rejected_border": rejected_border,
                        "kept_component_areas": kept_component_areas,
                        "cleaned_foreground_pixels": cleaned_pixels,
                        "cleaned_foreground_coverage": float(cleaned_pixels) / float(out.size) if out.size else 0.0,
                    },
                    "roi": roi_meta,
                    "artifacts": {
                        "threshold_mask": "threshold_mask.png",
                        "cleaned_mask": "cleaned_mask.png",
                        "overlay_image": "overlay_image.png",
                        "rejected_components_overlay": "rejected_components_overlay.png",
                    },
                }
                self._working["morphology_debug_json"] = debug_payload
                artifacts.append(_artifact("morphology_debug_json", "morphology", "json", "Morphology debug JSON"))
                artifacts[-1]["metadata"] = debug_payload
                metrics["connected_components"] = float(morph_metrics["connected_components"])
                metrics["foreground_coverage_percent"] = float(morph_metrics["foreground_coverage_percent"])
            elif step.step_id == "blob_detection":
                mask = self._working.get("cleaned_mask")
                if not isinstance(mask, np.ndarray):
                    mask = _require_mask(self._working)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                min_area = float(step.params.get("min_area", 80.0))
                max_area = float(step.params.get("max_area", 250000.0))
                min_width = int(step.params.get("min_width", 0))
                min_height = int(step.params.get("min_height", 0))
                max_aspect_ratio = float(step.params.get("max_aspect_ratio", 0.0))
                min_circularity = float(step.params.get("min_circularity", 0.4))
                reject_border_touching = bool(step.params.get("reject_border_touching", False))
                keep_largest_n = int(step.params.get("keep_largest_n", 0))
                max_objects = int(step.params.get("max_objects", 100))
                blobs: list[dict[str, Any]] = []
                rejected: list[dict[str, Any]] = []
                offset_x, offset_y = self._working.get("roi_offset", (0, 0))
                roi_meta = self._working.get("segmentation_roi") if isinstance(self._working.get("segmentation_roi"), dict) else {"enabled": False}
                for contour in contours:
                    area = float(cv2.contourArea(contour))
                    if area < min_area:
                        rejected.append({"reason": "small_area", "area": area})
                        continue
                    if area > max_area:
                        rejected.append({"reason": "large_area", "area": area})
                        continue
                    perimeter = float(cv2.arcLength(contour, True))
                    circularity = float((4.0 * np.pi * area) / ((perimeter * perimeter) + 1e-6)) if perimeter > 0 else 0.0
                    if circularity < min_circularity:
                        rejected.append({"reason": "low_circularity", "area": area, "circularity": circularity})
                        continue
                    m = cv2.moments(contour)
                    cx = float(m["m10"] / m["m00"]) if m["m00"] else 0.0
                    cy = float(m["m01"] / m["m00"]) if m["m00"] else 0.0
                    x, y, w, h = cv2.boundingRect(contour)
                    if min_width > 0 and w < min_width:
                        rejected.append({"reason": "small_width", "area": area, "bbox": [x, y, w, h]})
                        continue
                    if min_height > 0 and h < min_height:
                        rejected.append({"reason": "small_height", "area": area, "bbox": [x, y, w, h]})
                        continue
                    object_id = len(blobs) + 1
                    object_key = _object_key(object_id)
                    world_contour = contour.squeeze(1).astype(float)
                    world_contour[:, 0] += offset_x
                    world_contour[:, 1] += offset_y
                    hull = cv2.convexHull(contour)
                    hull_area = float(cv2.contourArea(hull)) if len(hull) >= 3 else 0.0
                    solidity = float(area / hull_area) if hull_area > 0 else 0.0
                    eq_d = float(np.sqrt((4.0 * area) / np.pi)) if area > 0 else 0.0
                    aspect = float(max(w, h) / max(1, min(w, h)))
                    touches_border = (x <= 0 or y <= 0 or (x + w) >= mask.shape[1] or (y + h) >= mask.shape[0])
                    if max_aspect_ratio > 0 and aspect > max_aspect_ratio:
                        rejected.append({"reason": "aspect", "area": area, "aspect_ratio": aspect, "bbox": [x, y, w, h]})
                        continue
                    if reject_border_touching and touches_border:
                        rejected.append({"reason": "border_touching", "area": area, "bbox": [x, y, w, h]})
                        continue
                    blobs.append(
                        {
                            "object_id": object_id,
                            "object_key": object_key,
                            "contour": contour,
                            "contour_world": world_contour.tolist(),
                            "area": area,
                            "perimeter": perimeter,
                            "circularity": circularity,
                            "centroid": (cx + offset_x, cy + offset_y),
                            "centroid_local": (cx, cy),
                            "bbox": (x + offset_x, y + offset_y, w, h),
                            "equivalent_diameter": eq_d,
                            "aspect_ratio": aspect,
                            "solidity": solidity,
                            "touches_border": touches_border,
                            "parent_roi": roi_meta,
                        }
                    )
                    overlays.append(_overlay(f"contour_{object_key}", "blob_detection", "polyline", {"points": world_contour.tolist()}, object_id=object_id, style={"stroke": "#3a86ff", "label": object_key}))
                    overlays.append(_overlay(f"bbox_{object_key}", "blob_detection", "bbox", {"x": x + offset_x, "y": y + offset_y, "width": w, "height": h}, object_id=object_id, style={"stroke": "#8338ec", "label": object_key}))
                    overlays.append(_overlay(f"centroid_{object_key}", "blob_detection", "centroid", {"x": cx + offset_x, "y": cy + offset_y}, object_id=object_id, style={"fill": "#fb5607", "label": object_key}))
                    if len(blobs) >= max_objects:
                        break
                if keep_largest_n > 0 and len(blobs) > keep_largest_n:
                    ranked = sorted(blobs, key=lambda item: float(item.get("area", 0.0)), reverse=True)
                    keep = ranked[:keep_largest_n]
                    drop = ranked[keep_largest_n:]
                    for item in drop:
                        rejected.append({"reason": "beyond_top_n", "blob_id": item["object_id"], "area": item.get("area")})
                    blobs = keep
                self._working["blobs"] = blobs
                self._working["blob_contours_json"] = [
                    {
                        "blob_id": item["object_id"],
                        "contour_points": item["contour_world"],
                        "bbox": item["bbox"],
                        "centroid": item["centroid"],
                        "parent_roi": item.get("parent_roi"),
                    }
                    for item in blobs
                ]
                self._working["blob_metrics_json"] = [
                    {
                        "blob_id": item["object_id"],
                        "area_px": item["area"],
                        "centroid": item["centroid"],
                        "bbox": item["bbox"],
                        "contour_perimeter": item["perimeter"],
                        "equivalent_diameter": item["equivalent_diameter"],
                        "circularity": item["circularity"],
                        "aspect_ratio": item["aspect_ratio"],
                        "solidity": item["solidity"],
                        "touches_border": item["touches_border"],
                        "parent_roi": item["parent_roi"],
                    }
                    for item in blobs
                ]
                self._working["blob_rejected_json"] = rejected
                reason_counts: dict[str, int] = {}
                for item in rejected:
                    reason = str(item.get("reason") or "unknown")
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
                areas = [float(item.get("area", 0.0)) for item in blobs if float(item.get("area", 0.0)) > 0]
                diameters = [float(item.get("equivalent_diameter", 0.0)) for item in blobs if float(item.get("equivalent_diameter", 0.0)) > 0]
                circularities = [float(item.get("circularity", 0.0)) for item in blobs]
                aspects = [float(item.get("aspect_ratio", 0.0)) for item in blobs]
                self._working["blob_metrics_summary_json"] = {
                    "params": {
                        "min_area": min_area,
                        "max_area": max_area,
                        "min_width": min_width,
                        "min_height": min_height,
                        "max_aspect_ratio": max_aspect_ratio,
                        "min_circularity": min_circularity,
                        "reject_border_touching": reject_border_touching,
                        "keep_largest_n": keep_largest_n,
                        "max_objects": max_objects,
                    },
                    "candidate_count": len(blobs),
                    "rejected_count": len(rejected),
                    "rejected_reason_counts": reason_counts,
                    "stats": {
                        "area": _min_med_max(areas),
                        "diameter": _min_med_max(diameters),
                        "circularity": _min_med_max(circularities),
                        "aspect_ratio": _min_med_max(aspects),
                    },
                    "candidates": self._working["blob_metrics_json"],
                }
                source_rgb = self._working.get("source_rgb")
                if isinstance(source_rgb, np.ndarray):
                    debug_overlay = source_rgb.copy()
                    label_image = np.zeros_like(debug_overlay)
                    for item in blobs:
                        contour_local = np.asarray(item["contour"], dtype=np.int32)
                        cv2.drawContours(debug_overlay, [contour_local], -1, (58, 134, 255), 2)
                        x, y, w, h = item["bbox"]
                        cv2.rectangle(debug_overlay, (int(x), int(y)), (int(x + w), int(y + h)), (131, 56, 236), 1)
                        cx, cy = item["centroid"]
                        cv2.putText(debug_overlay, str(item["object_id"]), (int(cx) + 4, int(cy) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                        cv2.drawContours(label_image, [contour_local], -1, (0, 255, 0), thickness=-1)
                        cv2.putText(label_image, str(item["object_id"]), (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                    self._working["blob_debug_overlay_image"] = debug_overlay
                    self._working["blob_labels_image"] = label_image
                    artifacts.append(_artifact("blob_debug_overlay", "blob_detection", "image", "Blob debug overlay"))
                    artifacts.append(_artifact("blob_labels", "blob_detection", "image", "Blob labels image"))
                artifacts.append(_artifact("blob_contours", "blob_detection", "json", "Blob contours JSON"))
                artifacts.append(_artifact("blob_metrics", "blob_detection", "json", "Blob metrics JSON"))
                artifacts.append(_artifact("blob_rejected", "blob_detection", "json", "Rejected blob candidates JSON"))
                artifacts.append(_artifact("contour_blob_artifact", "blob_detection", "json", "Contour/blob artifact"))
                artifacts[-4]["metadata"] = {"semantic_kind": "blob_contours", "entries": self._working["blob_contours_json"]}
                artifacts[-3]["metadata"] = {"semantic_kind": "blob_metrics", "entries": self._working["blob_metrics_json"], **self._working["blob_metrics_summary_json"]}
                artifacts[-2]["metadata"] = {"semantic_kind": "blob_rejected", "entries": self._working["blob_rejected_json"]}
                metrics["blob_count"] = float(len(blobs))
                metrics["blob_rejected_count"] = float(len(rejected))
            elif step.step_id == "ellipse_fitting":
                blobs = self._working.get("blobs") or []
                if not blobs:
                    blobs = _blobs_from_blob_artifacts(self._working)
                enabled = bool(step.params.get("enabled", True))
                min_points = int(step.params.get("min_points", step.params.get("min_fit_points", 5)))
                fit_error_threshold = float(step.params.get("fit_error_threshold", 0.25))
                max_fit_rmse = float(step.params.get("max_fit_rmse", 9999.0))
                max_eccentricity = float(step.params.get("max_eccentricity", 1.0))
                min_fill_ratio = float(step.params.get("min_fill_ratio", 0.0))
                reject_border_touching = bool(step.params.get("reject_border_touching", False))
                offset_x, offset_y = self._working.get("roi_offset", (0, 0))
                ellipse_rows: list[dict[str, Any]] = []
                rejected_fit_count = 0
                fit_warnings: list[str] = []
                overlay_canvas = self._working.get("source_rgb")
                if isinstance(overlay_canvas, np.ndarray):
                    if overlay_canvas.ndim == 2:
                        overlay_canvas = cv2.cvtColor(overlay_canvas, cv2.COLOR_GRAY2BGR)
                    else:
                        overlay_canvas = overlay_canvas.copy()
                debug_canvas = overlay_canvas.copy() if isinstance(overlay_canvas, np.ndarray) else None
                for blob in blobs:
                    contour = blob["contour"]
                    if not enabled or len(contour) < min_points:
                        blob["fit_confidence"] = 0.0
                        blob["fit_error"] = 1.0
                        blob["valid_fit"] = False
                        blob["fit_rejection_reason"] = "insufficient_points" if len(contour) < min_points else "disabled"
                        rejected_fit_count += 1
                        continue
                    try:
                        (cx, cy), (ma, mi), angle = cv2.fitEllipse(contour)
                    except Exception:
                        blob["fit_confidence"] = 0.0
                        blob["fit_error"] = 1.0
                        blob["valid_fit"] = False
                        blob["fit_rejection_reason"] = "fit_exception"
                        rejected_fit_count += 1
                        continue
                    major = float(max(ma, mi))
                    minor = float(min(ma, mi))
                    if major <= 0 or minor <= 0:
                        blob["fit_confidence"] = 0.0
                        blob["fit_error"] = 1.0
                        blob["valid_fit"] = False
                        blob["fit_rejection_reason"] = "degenerate_axes"
                        rejected_fit_count += 1
                        continue
                    ecc = float(np.sqrt(max(0.0, 1.0 - ((minor * minor) / (major * major + 1e-6)))))
                    circularity = float(blob.get("circularity", 0.0))
                    diameter = (major + minor) / 2.0
                    fit_error = abs(major - minor) / max(1e-6, major)
                    rmse, max_error = _ellipse_fit_errors(contour, cx, cy, major / 2.0, minor / 2.0, angle)
                    ellipse_area = float(np.pi * (major / 2.0) * (minor / 2.0))
                    contour_area = float(blob.get("area", 0.0))
                    fill_ratio = float(contour_area / ellipse_area) if ellipse_area > 1e-6 else 0.0
                    valid_fit = True
                    rejection_reason: str | None = None
                    if reject_border_touching and bool(blob.get("touches_border")):
                        valid_fit = False
                        rejection_reason = "border_touching"
                    elif rmse > max_fit_rmse:
                        valid_fit = False
                        rejection_reason = "rmse"
                    elif ecc > max_eccentricity:
                        valid_fit = False
                        rejection_reason = "eccentricity"
                    elif fill_ratio < min_fill_ratio:
                        valid_fit = False
                        rejection_reason = "fill_ratio"
                    fit_confidence = max(0.0, min(1.0, 1.0 - (fit_error / max(1e-6, fit_error_threshold))))
                    if not valid_fit:
                        rejected_fit_count += 1
                    blob.update(
                        {
                            "ellipse": {
                                "cx": cx + offset_x,
                                "cy": cy + offset_y,
                                "rx": major / 2.0,
                                "ry": minor / 2.0,
                                "rotation": angle,
                            },
                            "eccentricity": ecc,
                            "diameter_estimate": diameter,
                            "fit_error": fit_error,
                            "fit_confidence": fit_confidence,
                            "fit_rmse": rmse,
                            "fit_max_error": max_error,
                            "ellipse_fill_ratio": fill_ratio,
                            "ellipse_area": ellipse_area,
                            "major_axis": major,
                            "minor_axis": minor,
                            "valid_fit": valid_fit,
                            "fit_rejection_reason": rejection_reason,
                        }
                    )
                    overlays.append(_overlay(f"ellipse_{blob['object_key']}", "ellipse_fitting", "ellipse", blob["ellipse"], object_id=blob["object_id"], style={"stroke": "#00b4d8", "label": blob["object_key"]}))
                    ex = float(blob["ellipse"]["cx"])
                    ey = float(blob["ellipse"]["cy"])
                    rx = float(blob["ellipse"]["rx"])
                    ry = float(blob["ellipse"]["ry"])
                    theta = float(np.deg2rad(blob["ellipse"]["rotation"]))
                    major_dx = rx * np.cos(theta)
                    major_dy = rx * np.sin(theta)
                    minor_dx = -ry * np.sin(theta)
                    minor_dy = ry * np.cos(theta)
                    overlays.append(
                        _overlay(
                            f"ellipse_major_axis_{blob['object_key']}",
                            "ellipse_fitting",
                            "polyline",
                            {"points": [[ex - major_dx, ey - major_dy], [ex + major_dx, ey + major_dy]]},
                            object_id=blob["object_id"],
                            style={"stroke": "#00e5ff"},
                        )
                    )
                    overlays.append(
                        _overlay(
                            f"ellipse_minor_axis_{blob['object_key']}",
                            "ellipse_fitting",
                            "polyline",
                            {"points": [[ex - minor_dx, ey - minor_dy], [ex + minor_dx, ey + minor_dy]]},
                            object_id=blob["object_id"],
                            style={"stroke": "#b5179e"},
                        )
                    )
                    overlays.append(
                        _overlay(
                            f"ellipse_label_{blob['object_key']}",
                            "ellipse_fitting",
                            "text",
                            {"x": ex + 4.0, "y": ey - 6.0, "text": blob["object_key"]},
                            object_id=blob["object_id"],
                            style={"label": blob["object_key"], "fill": "#22c55e" if valid_fit else "#ef4444"},
                        )
                    )
                    ellipse_rows.append(
                        {
                            "candidate_id": blob["object_id"],
                            "object_key": blob["object_key"],
                            "center_x": ex,
                            "center_y": ey,
                            "major_axis": major,
                            "minor_axis": minor,
                            "angle_deg": float(blob["ellipse"]["rotation"]),
                            "aspect_ratio": float(major / max(1e-6, minor)),
                            "equivalent_diameter": diameter,
                            "ellipse_area": ellipse_area,
                            "contour_area": contour_area,
                            "ellipse_fill_ratio": fill_ratio,
                            "eccentricity": ecc,
                            "fit_rmse": rmse,
                            "fit_max_error": max_error,
                            "circularity": circularity,
                            "solidity": float(blob.get("solidity", 0.0)),
                            "touches_border": bool(blob.get("touches_border", False)),
                            "valid_fit": valid_fit,
                            "fit_rejection_reason": rejection_reason,
                        }
                    )
                    calibration_2d = self._working.get("active_2d_calibration")
                    if isinstance(calibration_2d, dict):
                        belt_plane = calibration_2d.get("belt_plane") if isinstance(calibration_2d.get("belt_plane"), dict) else {}
                        homography = belt_plane.get("homography")
                        mm_per_px_x = float(belt_plane.get("mm_per_px_x", 0.0) or 0.0)
                        mm_per_px_y = float(belt_plane.get("mm_per_px_y", 0.0) or 0.0)
                        if isinstance(homography, list) and len(homography) == 3:
                            major_mm = axis_length_mm((ex, ey), rx, theta, homography)
                            minor_mm = axis_length_mm((ex, ey), ry, theta + (np.pi / 2.0), homography)
                            eq_mm = float((major_mm + minor_mm) / 2.0)
                        elif mm_per_px_x > 0 and mm_per_px_y > 0:
                            scale = (mm_per_px_x + mm_per_px_y) / 2.0
                            major_mm = float(major * scale)
                            minor_mm = float(minor * scale)
                            eq_mm = float(diameter * scale)
                        else:
                            major_mm = None
                            minor_mm = None
                            eq_mm = None
                        ellipse_rows[-1]["major_axis_mm"] = major_mm
                        ellipse_rows[-1]["minor_axis_mm"] = minor_mm
                        ellipse_rows[-1]["equivalent_diameter_mm"] = eq_mm
                    else:
                        ellipse_rows[-1]["major_axis_mm"] = None
                        ellipse_rows[-1]["minor_axis_mm"] = None
                        ellipse_rows[-1]["equivalent_diameter_mm"] = None
                    if isinstance(overlay_canvas, np.ndarray):
                        color = (60, 220, 120) if valid_fit else (40, 40, 220)
                        cv2.ellipse(
                            overlay_canvas,
                            (int(round(ex)), int(round(ey))),
                            (int(round(rx)), int(round(ry))),
                            float(blob["ellipse"]["rotation"]),
                            0,
                            360,
                            color,
                            3 if valid_fit else 2,
                        )
                        cv2.putText(
                            overlay_canvas,
                            str(blob["object_id"]),
                            (int(round(ex)) + 4, int(round(ey)) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )
                    if isinstance(debug_canvas, np.ndarray):
                        contour_local = np.asarray(contour, dtype=np.int32)
                        cv2.drawContours(debug_canvas, [contour_local], -1, (255, 172, 0), 1)
                        if valid_fit:
                            cv2.ellipse(
                                debug_canvas,
                                (int(round(ex)), int(round(ey))),
                                (int(round(rx)), int(round(ry))),
                                float(blob["ellipse"]["rotation"]),
                                0,
                                360,
                                (79, 140, 255),
                                2,
                            )
                        else:
                            cv2.ellipse(
                                debug_canvas,
                                (int(round(ex)), int(round(ey))),
                                (int(round(rx)), int(round(ry))),
                                float(blob["ellipse"]["rotation"]),
                                0,
                                360,
                                (16, 16, 220),
                                2,
                            )
                if ellipse_rows:
                    major_values = [float(item["major_axis"]) for item in ellipse_rows]
                    minor_values = [float(item["minor_axis"]) for item in ellipse_rows]
                    diam_values = [float(item["equivalent_diameter"]) for item in ellipse_rows]
                    ecc_values = [float(item["eccentricity"]) for item in ellipse_rows]
                    rmse_values = [float(item["fit_rmse"]) for item in ellipse_rows]
                else:
                    major_values = []
                    minor_values = []
                    diam_values = []
                    ecc_values = []
                    rmse_values = []
                summary = {
                    "candidate_count": len(blobs),
                    "fitted_count": len(ellipse_rows),
                    "rejected_fit_count": rejected_fit_count,
                    "avg_major_axis": float(np.mean(major_values)) if major_values else 0.0,
                    "avg_minor_axis": float(np.mean(minor_values)) if minor_values else 0.0,
                    "avg_diameter": float(np.mean(diam_values)) if diam_values else 0.0,
                    "avg_eccentricity": float(np.mean(ecc_values)) if ecc_values else 0.0,
                    "avg_fit_rmse": float(np.mean(rmse_values)) if rmse_values else 0.0,
                    "params": {
                        "enabled": enabled,
                        "min_fit_points": min_points,
                        "max_fit_rmse": max_fit_rmse,
                        "max_eccentricity": max_eccentricity,
                        "min_fill_ratio": min_fill_ratio,
                        "reject_border_touching": reject_border_touching,
                        "fit_error_threshold": fit_error_threshold,
                    },
                    "warnings": fit_warnings,
                }
                if not blobs:
                    source = self._working.get("blob_contours_json")
                    if source is None:
                        summary["warnings"].append("blob_contours artifact missing")
                    elif isinstance(source, (list, dict)) and len(source) == 0:
                        summary["warnings"].append("blob_contours parsed but empty")
                    else:
                        summary["warnings"].append("candidates present but invalid or missing contour points")
                self._working["ellipse_metrics_json"] = ellipse_rows
                self._working["ellipse_summary_json"] = summary
                self._working["ellipse_debug_json"] = {
                    "stage": "ellipse_fitting",
                    "candidate_count": len(blobs),
                    "fitted_count": len(ellipse_rows),
                    "rejected_fit_count": rejected_fit_count,
                    "warnings": summary.get("warnings", []),
                    "params": summary.get("params", {}),
                }
                if isinstance(overlay_canvas, np.ndarray):
                    self._working["ellipse_overlay_image"] = overlay_canvas
                if isinstance(debug_canvas, np.ndarray):
                    self._working["ellipse_debug_overlay_image"] = debug_canvas
                artifacts.append(_artifact("ellipse_overlay", "ellipse_fitting", "image", "Ellipse overlay image"))
                artifacts[-1]["metadata"] = {"semantic_kind": "ellipse_overlay", "source_artifact_id": "source_rgb_image"}
                artifacts.append(_artifact("ellipse_metrics", "ellipse_fitting", "json", "Ellipse metrics JSON"))
                artifacts[-1]["metadata"] = {"semantic_kind": "ellipse_metrics", "entries": ellipse_rows}
                artifacts.append(_artifact("ellipse_summary", "ellipse_fitting", "json", "Ellipse summary JSON"))
                artifacts[-1]["metadata"] = {"semantic_kind": "ellipse_summary", **summary}
                artifacts.append(_artifact("ellipse_debug_json", "ellipse_fitting", "json", "Ellipse debug JSON"))
                artifacts[-1]["metadata"] = {"semantic_kind": "ellipse_debug_json", **dict(self._working["ellipse_debug_json"])}
                artifacts.append(_artifact("ellipse_debug_overlay", "ellipse_fitting", "image", "Ellipse debug overlay image"))
                artifacts[-1]["metadata"] = {"semantic_kind": "ellipse_debug_overlay", "source_artifact_id": "source_rgb_image"}
                artifacts.append(_artifact("ellipse_fit_artifact", "ellipse_fitting", "table", "Ellipse fit artifact"))
                metrics["ellipse_fitted_count"] = float(len(ellipse_rows))
                metrics["ellipse_rejected_count"] = float(rejected_fit_count)
            elif step.step_id == "metrics":
                blobs = self._working.get("blobs") or []
                rows = []
                for blob in blobs:
                    row = {
                        "object_id": blob["object_id"],
                        "object_key": blob["object_key"],
                        "area": blob.get("area"),
                        "centroid": blob.get("centroid"),
                        "diameter": blob.get("diameter_estimate"),
                        "circularity": blob.get("circularity"),
                        "eccentricity": blob.get("eccentricity"),
                    }
                    rows.append(row)
                measurements.extend(rows)
                self._working["measurements"] = rows
                artifacts.append(_artifact("measurement_table_artifact", "metrics", "table", "Measurement table artifact"))
            elif step.step_id == "classification":
                blobs = self._working.get("blobs") or []
                params = {
                    "min_area": float(step.params.get("min_area", 80.0)),
                    "max_area": float(step.params.get("max_area", 250000.0)),
                    "min_circularity_for_ball": float(step.params.get("min_circularity_for_ball", 0.78)),
                    "min_confidence": float(step.params.get("min_confidence", 0.45)),
                    "spherical_enough_threshold": float(step.params.get("spherical_enough_threshold", 0.72)),
                    "diameter_min": float(step.params.get("diameter_min", 10.0)),
                    "diameter_max": float(step.params.get("diameter_max", 300.0)),
                }
                classes: list[dict[str, Any]] = []
                for blob in blobs:
                    area = float(blob.get("area", 0.0))
                    circularity = float(blob.get("circularity", 0.0))
                    fit_conf = float(blob.get("fit_confidence", 0.0))
                    confidence = max(fit_conf, min(1.0, circularity * 0.9))
                    diameter = float(blob.get("diameter_estimate", 0.0)) if blob.get("diameter_estimate") is not None else 0.0
                    in_area = params["min_area"] <= area <= params["max_area"]
                    in_diameter = params["diameter_min"] <= diameter <= params["diameter_max"]
                    if not in_area or not in_diameter or confidence < params["min_confidence"]:
                        cls = "unknown"
                    elif circularity >= params["min_circularity_for_ball"]:
                        cls = "ball"
                    else:
                        cls = "non-ball"
                    spherical = "spherical-enough" if circularity >= params["spherical_enough_threshold"] else "non-spherical-enough"
                    if cls == "unknown":
                        spherical = "unknown / rejected"
                    record = {
                        "object_id": blob["object_id"],
                        "object_key": blob["object_key"],
                        "classification": cls,
                        "spherical_class": spherical,
                        "confidence": confidence,
                    }
                    classes.append(record)
                    blob.update(record)
                    label = f"{cls} {confidence:.2f}"
                    overlays.append(
                        _overlay(
                            f"class_{blob['object_key']}",
                            "classification",
                            "text",
                            {"x": blob["centroid"][0], "y": blob["centroid"][1], "text": label},
                            object_id=blob["object_id"],
                            style={
                                "label": f"{blob['object_key']} {label}",
                                "fill": "#06d6a0" if cls == "ball" else "#ef476f" if cls == "non-ball" else "#adb5bd",
                            },
                        )
                    )
                self._working["classifications"] = classes
                artifacts.append(_artifact("classification_result_artifact", "classification", "table", "Classification result artifact"))
            elif step.step_id == "overlay":
                artifacts.append(_artifact("overlay_summary_artifact", "overlay", "json", "Overlay summary artifact"))
            elif step.step_id == "summary":
                classes = self._working.get("classifications") or []
                summary = {
                    "total_objects": len(classes),
                    "balls": sum(1 for item in classes if item.get("classification") == "ball"),
                    "non_balls": sum(1 for item in classes if item.get("classification") == "non-ball"),
                    "unknown_or_rejected": sum(1 for item in classes if item.get("classification") == "unknown"),
                    "spherical_enough": sum(1 for item in classes if item.get("spherical_class") == "spherical-enough"),
                    "non_spherical_enough": sum(1 for item in classes if item.get("spherical_class") == "non-spherical-enough"),
                }
                debug["summary"] = summary
                artifacts.append(_artifact("result_summary_artifact", "summary", "json", "Result summary artifact"))
            else:
                warnings.append(f"Step {step.step_id} is not implemented.")

            status = "warning" if warnings else "success"
        except Exception as exc:
            status = "failed"
            errors.append(str(exc))

        elapsed = (time.perf_counter() - started) * 1000.0
        report = StepRunReport(
            step_id=step.step_id,
            algorithm_key=step.algorithm_key,
            status=status,
            duration_ms=elapsed,
            warnings=warnings,
            errors=errors,
            artifact_ids=[item["artifact_id"] for item in artifacts],
            overlay_ids=[item["artifact_id"] for item in overlays],
            measurements=measurements,
            metrics=metrics,
        )
        return report, StepExecutionResult(artifacts, overlays, measurements, warnings, metrics, debug, self._working)


def _artifact(artifact_id: str, stage_id: str, kind: str, title: str) -> dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "stage_id": stage_id,
        "kind": kind,
        "title": title,
        "preview_available": kind == "image",
        "generated": "explicit",
        "metadata": {},
    }


def _overlay(artifact_id: str, stage_id: str, overlay_type: str, geometry: dict[str, Any], *, object_id: int | None = None, style: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "stage_id": stage_id,
        "kind": "overlay",
        "title": artifact_id,
        "object_id": object_id,
        "overlay_type": overlay_type,
        "coordinate_space": "image_pixel",
        "target_artifact_id": "source_rgb_image",
        "geometry": geometry,
        "style": style or {},
        "source_artifact_ids": [],
        "preview_available": False,
        "metadata": {},
    }


def _to_gray(image: np.ndarray, method: str) -> np.ndarray:
    if image.ndim == 2:
        return image
    if method == "red":
        return image[:, :, 2]
    if method == "green":
        return image[:, :, 1]
    if method == "blue":
        return image[:, :, 0]
    if method == "max_channel":
        return image.max(axis=2).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _require_rgb(state: dict[str, Any]) -> np.ndarray:
    image = state.get("current_rgb")
    if not isinstance(image, np.ndarray):
        raise ValueError("RGB image is missing; input step must run before this step.")
    return image


def _require_gray(state: dict[str, Any]) -> np.ndarray:
    gray = state.get("gray")
    if not isinstance(gray, np.ndarray):
        raise ValueError("Grayscale image is missing; rgb_to_gray step must run before this step.")
    return gray


def _require_mask(state: dict[str, Any]) -> np.ndarray:
    mask = state.get("mask")
    if not isinstance(mask, np.ndarray):
        raise ValueError("Mask is missing; threshold step must run before this step.")
    return mask


def _min_med_max(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
    }


def _object_key(object_id: int) -> str:
    return f"object_{object_id:03d}"


def _ellipse_fit_errors(contour: np.ndarray, cx: float, cy: float, rx: float, ry: float, angle_deg: float) -> tuple[float, float]:
    if rx <= 1e-6 or ry <= 1e-6:
        return 0.0, 0.0
    pts = contour.reshape(-1, 2).astype(np.float64)
    theta = np.deg2rad(float(angle_deg))
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    errs: list[float] = []
    scale = float((rx + ry) * 0.5)
    for px, py in pts:
        dx = float(px - cx)
        dy = float(py - cy)
        xr = dx * cos_t + dy * sin_t
        yr = -dx * sin_t + dy * cos_t
        rnorm = np.sqrt((xr / max(rx, 1e-6)) ** 2 + (yr / max(ry, 1e-6)) ** 2)
        err = abs(float(rnorm) - 1.0) * scale
        errs.append(err)
    if not errs:
        return 0.0, 0.0
    arr = np.asarray(errs, dtype=np.float64)
    return float(np.sqrt(np.mean(arr * arr))), float(np.max(arr))


def _blobs_from_blob_artifacts(state: dict[str, Any]) -> list[dict[str, Any]]:
    raw = state.get("blob_contours_json")
    rows: list[dict[str, Any]] = []
    if isinstance(raw, list):
        rows = [item for item in raw if isinstance(item, dict)]
    elif isinstance(raw, dict):
        for key in ("candidates", "blobs", "contours"):
            candidate = raw.get(key)
            if isinstance(candidate, list):
                rows = [item for item in candidate if isinstance(item, dict)]
                break
    metrics_rows = state.get("blob_metrics_json")
    metric_by_id: dict[int, dict[str, Any]] = {}
    if isinstance(metrics_rows, list):
        for item in metrics_rows:
            if not isinstance(item, dict):
                continue
            blob_id = int(item.get("blob_id", item.get("id", 0)) or 0)
            if blob_id > 0:
                metric_by_id[blob_id] = item
    blobs: list[dict[str, Any]] = []
    for idx, item in enumerate(rows, start=1):
        blob_id = int(item.get("blob_id", item.get("id", idx)) or idx)
        contour_points = item.get("contour_points", item.get("contour"))
        if not isinstance(contour_points, list) or len(contour_points) < 5:
            continue
        contour_arr = np.asarray(contour_points, dtype=np.float32).reshape((-1, 1, 2))
        metric = metric_by_id.get(blob_id, {})
        centroid = item.get("centroid", metric.get("centroid"))
        bbox = item.get("bbox", metric.get("bbox"))
        area = float(metric.get("area_px", metric.get("area", 0.0)) or 0.0)
        perimeter = float(metric.get("contour_perimeter", metric.get("perimeter", 0.0)) or 0.0)
        circularity = float(metric.get("circularity", 0.0) or 0.0)
        solidity = float(metric.get("solidity", 0.0) or 0.0)
        aspect_ratio = float(metric.get("aspect_ratio", metric.get("aspectRatio", 0.0)) or 0.0)
        touches_border = bool(metric.get("touches_border", metric.get("touchesBorder", False)))
        if not (isinstance(centroid, (list, tuple)) and len(centroid) >= 2):
            centroid = [0.0, 0.0]
        if not (isinstance(bbox, (list, tuple)) and len(bbox) >= 4):
            x, y, w, h = cv2.boundingRect(contour_arr.astype(np.int32))
            bbox = [x, y, w, h]
        blobs.append(
            {
                "object_id": blob_id,
                "object_key": _object_key(blob_id),
                "contour": contour_arr,
                "contour_world": np.asarray(contour_points, dtype=float).reshape((-1, 2)).tolist(),
                "area": area,
                "perimeter": perimeter,
                "circularity": circularity,
                "centroid": (float(centroid[0]), float(centroid[1])),
                "bbox": tuple(int(v) for v in bbox[:4]),
                "equivalent_diameter": float(metric.get("equivalent_diameter", metric.get("diameter", 0.0)) or 0.0),
                "aspect_ratio": aspect_ratio,
                "solidity": solidity,
                "touches_border": touches_border,
                "parent_roi": item.get("parent_roi", metric.get("parent_roi")),
            }
        )
    return blobs
