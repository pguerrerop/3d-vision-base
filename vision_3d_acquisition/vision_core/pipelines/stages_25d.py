from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vision_3d_acquisition.contracts.modalities import CaptureModality
from vision_3d_acquisition.operations.classification_superclass import (
    apply_sphericity_3d_fallback_to_object,
    map_label_to_superclass,
    select_dominant_classification,
)
from vision_3d_acquisition.vision_core.heightmap import (
    HeightmapFrame,
    load_heightmap_npz,
    save_heightmap_npz,
    write_heightmap_metadata_json,
    write_heightmap_preview_png,
)
from vision_3d_acquisition.vision_core.pipelines.core import PipelineContext


_PREVIEW_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_SUPPORTED_HEIGHT_TRANSFORM_TYPES = {
    "raw_decode",
    "plane_fit",
    "signed_distance",
    "invert_signed_distance",
    "normalization",
    "clipping",
    "scaling",
    "projection",
    "smoothing",
    "masking",
}


def _display_only_meta(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    meta: dict[str, Any] = {"display_only": True, "numeric_source": False}
    if extra:
        meta.update(extra)
    return meta


def _numeric_source_meta(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    meta: dict[str, Any] = {"display_only": False, "numeric_source": True}
    if extra:
        meta.update(extra)
    return meta


def _height_semantic_meta(
    *,
    semantic_field: str,
    semantic_label: str,
    units: str,
    value_min: float,
    value_max: float,
    color_scale_min: float,
    color_scale_max: float,
    color_map: str,
    positive_direction: str,
    is_measurement_authoritative: bool,
    source_artifact_id: str,
    stage_id: str,
) -> dict[str, Any]:
    # `color_mapping` is the canonical contract shared by:
    #   - image renderer (this writes the PNG using these bounds)
    #   - HeightLegend (renders gradient/labels from the same LUT)
    #   - hover / debug reconstruction (scalar->RGB diff uses these bounds)
    # Studio considers metadata.color_mapping authoritative.
    color_mapping = build_height_color_mapping_block(
        semantic_field=semantic_field,
        units=units,
        value_min=value_min,
        value_max=value_max,
        color_scale_min=color_scale_min,
        color_scale_max=color_scale_max,
        color_map=color_map,
        direction=_direction_for_semantic(semantic_field, positive_direction),
        source="artifact_metadata",
    )
    return {
        "semantic_field": semantic_field,
        "semantic_label": semantic_label,
        "units": units,
        "value_min": float(value_min),
        "value_max": float(value_max),
        "color_scale_min": float(color_scale_min),
        "color_scale_max": float(color_scale_max),
        "color_map": color_map,
        "positive_direction": positive_direction,
        "is_measurement_authoritative": bool(is_measurement_authoritative),
        "source_artifact_id": source_artifact_id,
        "stage_id": stage_id,
        "color_mapping": color_mapping,
    }


def build_height_color_mapping_block(
    *,
    semantic_field: str,
    units: str,
    value_min: float,
    value_max: float,
    color_scale_min: float,
    color_scale_max: float,
    color_map: str,
    direction: str = "higher_is_hotter",
    clamp: bool = True,
    source: str = "artifact_metadata",
) -> dict[str, Any]:
    """Build a canonical color_mapping block.

    Direction is data-native: the LUT is always anchored at value_min (cool)
    to value_max (hot). The `direction` field only affects how labels/ticks
    are rendered by HeightLegend; it never inverts the LUT.
    """
    return {
        "semantic_field": semantic_field,
        "units": units,
        "value_min": float(value_min),
        "value_max": float(value_max),
        "color_scale_min": float(color_scale_min),
        "color_scale_max": float(color_scale_max),
        "color_map": str(color_map or "turbo"),
        "direction": direction if direction in ("higher_is_hotter", "lower_is_hotter") else "higher_is_hotter",
        "clamp": bool(clamp),
        "source": source if source in ("artifact_metadata", "render_context", "legacy_fallback") else "artifact_metadata",
    }


def _direction_for_semantic(semantic_field: str, positive_direction: str | None) -> str:
    key = str(positive_direction or "").lower()
    if "lower" in key:
        return "lower_is_hotter"
    if semantic_field in ("height_above_belt", "raw_sensor_z", "plane_signed_distance", "preview_normalized"):
        return "higher_is_hotter"
    return "higher_is_hotter"


def _assert_not_preview_semantic(context: PipelineContext, *, stage_name: str) -> None:
    if str(context.get_artifact("height_processing_semantic_field", "")).strip() == "preview_normalized":
        raise ValueError(f"{stage_name} cannot consume preview_normalized")


def build_height_transform_metadata(
    *,
    semantic_field: str,
    derived_from: str,
    transform_type: str,
    plane_id: str | None = None,
    units: str = "mm",
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = str(transform_type or "").strip()
    if normalized not in _SUPPORTED_HEIGHT_TRANSFORM_TYPES:
        raise ValueError(f"Unsupported height transform type: {transform_type}")
    transform: dict[str, Any] = {
        "type": normalized,
        "units": units,
    }
    if plane_id:
        transform["plane_id"] = plane_id
    if extras:
        transform.update(extras)
    return {
        "semantic_field": semantic_field,
        "derived_from": derived_from,
        "transform": transform,
    }


def _semantic_raster_meta(
    *,
    semantic_field: str,
    source_stage: str,
    coordinate_space: str,
    units: str = "mm",
    dtype: str = "float32",
    derived_from: str,
    transform_type: str,
    plane_id: str | None = None,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        **_numeric_source_meta(),
        "semantic_field": semantic_field,
        "coordinate_space": coordinate_space,
        "units": units,
        "dtype": dtype,
        "source_stage": source_stage,
    }
    meta.update(
        build_height_transform_metadata(
            semantic_field=semantic_field,
            derived_from=derived_from,
            transform_type=transform_type,
            plane_id=plane_id,
            units=units,
        )
    )
    if extras:
        meta.update(extras)
    return meta


def resolve_height_artifact(
    context: PipelineContext,
    *,
    semantic_field: str,
    representation: str,
) -> dict[str, Any] | None:
    def _matches(item: dict[str, Any]) -> bool:
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        if str(metadata.get("semantic_field") or "") != semantic_field:
            return False
        kind = str(item.get("kind") or "")
        path = str(item.get("path") or "")
        if representation == "numeric_raster":
            return kind == "file" or path.endswith(".values.f32")
        if representation == "preview_image":
            return kind == "image"
        if representation == "debug_overlay":
            return kind == "overlay"
        if representation == "histogram":
            return "histogram" in str(item.get("artifact_id") or "") or "histogram" in path
        if representation == "mask":
            return "mask" in str(item.get("artifact_id") or "") or "mask" in path
        if representation == "residual_map":
            return "residual" in str(item.get("artifact_id") or "") or "residual" in path
        return False

    for artifact in context.processing_artifacts:
        if isinstance(artifact, dict) and _matches(artifact):
            return artifact
    return None


def _valid_min_max_mm(values: np.ndarray, valid_mask: np.ndarray) -> tuple[float, float]:
    vals = np.asarray(values, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=bool)
    valid = vals[mask]
    if valid.size <= 0:
        return 0.0, 0.0
    return float(np.min(valid)), float(np.max(valid))


@dataclass
class LoadHeightmapCaptureStage:
    name: str = "LoadHeightmapCapture"
    category: str = "input"
    required_modalities: tuple[CaptureModality, ...] = ("heightmap",)
    produced_modalities: tuple[CaptureModality, ...] = ("heightmap",)
    produced_artifacts: tuple[str, ...] = ("heightmap_frame",)
    stage_category: str | None = "input"

    def run(self, context: PipelineContext) -> None:
        metadata = context.require_artifact("metadata")
        take_dir: Path = context.require_artifact("take_dir")
        output_dir: Path = context.require_artifact("output_dir")
        files = metadata.get("files") if isinstance(metadata.get("files"), dict) else {}
        height_file = str(files.get("heightmap") or files.get("height") or "")
        if not height_file:
            raise ValueError("Missing heightmap file in metadata.files.heightmap/height")
        height_path = take_dir / height_file
        reflectance_path = None
        reflectance_file = files.get("reflectance")
        if isinstance(reflectance_file, str) and reflectance_file:
            candidate = take_dir / reflectance_file
            if candidate.is_file():
                reflectance_path = candidate

        frame = _load_heightmap_frame(height_path, reflectance_path, metadata)
        context.set_artifact("heightmap_frame", frame)
        context.set_artifact("files.heightmap", height_file)
        context.set_artifact("encoder_metadata", _encoder_metadata(metadata))

        npz_name = "heightmap_frame.npz"
        preview_name = "heightmap_input_preview.png"
        meta_name = "heightmap_frame.metadata.json"
        raw_semantic_raster = "raw_sensor_z.values.f32"
        save_heightmap_npz(frame, output_dir / npz_name)
        write_heightmap_preview_png(frame.z_mm, frame.valid_mask, output_dir / preview_name)
        write_heightmap_metadata_json(frame, output_dir / meta_name, extra={"source_file": height_file})
        frame.z_mm.astype(np.float32).tofile(output_dir / raw_semantic_raster)
        # Backward-compatible alias while studio migrates to semantic-first paths.
        frame.z_mm.astype(np.float32).tofile(output_dir / "raw_heightmap.values.f32")
        input_min_mm, input_max_mm = _valid_min_max_mm(frame.z_mm, frame.valid_mask)

        files_payload: dict[str, str | None] = {
            "heightmap": height_file,
            "heightmap_npz": npz_name,
            "input_preview": preview_name,
            "heightmap_metadata": meta_name,
        }
        if isinstance(reflectance_file, str):
            files_payload["reflectance"] = reflectance_file
        context.set_artifact("files", files_payload)

        context.add_processing_artifact(
            artifact_id="heightmap",
            stage_id="input",
            kind="image",
            title="Heightmap input preview",
            path=preview_name,
            mime_type="image/png",
            preview_available=True,
            projection_type="heightmap",
            projection_coordinate_system={"pixel_per_mm": 1.0 / max(frame.x_resolution_mm, 1e-6)},
            metadata={
                **_display_only_meta(),
                **_height_semantic_meta(
                    semantic_field="raw_sensor_z",
                    semantic_label="Raw sensor Z",
                    units="mm",
                    value_min=input_min_mm,
                    value_max=input_max_mm,
                    color_scale_min=input_min_mm,
                    color_scale_max=input_max_mm,
                    color_map="turbo",
                    positive_direction="higher_value",
                    is_measurement_authoritative=False,
                    source_artifact_id="heightmap_frame.npz:z_mm",
                    stage_id="input",
                ),
                "source": "native_pipeline",
                "producer": self.name,
                "modality": "heightmap",
                "min_mm": input_min_mm,
                "max_mm": input_max_mm,
                "units": "mm",
                "display_semantic": "raw_sensor_z_mm",
                "display_units": "mm",
                "display_range": {"min": input_min_mm, "max": input_max_mm},
                "raw_semantic": "raw_sensor_z_mm",
                "semantic_source_id": "heightmap_frame.npz:z_mm",
                **build_height_transform_metadata(
                    semantic_field="raw_sensor_z",
                    derived_from="sensor_heightmap_decode",
                    transform_type="raw_decode",
                    units="mm",
                ),
            },
        )
        context.add_processing_artifact(
            artifact_id="raw_sensor_z_raster",
            stage_id="input",
            kind="file",
            title="Raw sensor Z semantic raster",
            path=raw_semantic_raster,
            mime_type="application/octet-stream",
            preview_available=False,
            metadata=_semantic_raster_meta(
                semantic_field="raw_sensor_z",
                source_stage="load_heightmap_capture",
                coordinate_space="sensor_xy_pixels",
                derived_from="sensor_heightmap_decode",
                transform_type="raw_decode",
            ),
        )


@dataclass
class ApplyCalibration25DStage:
    name: str = "ApplyCalibration"
    category: str = "calibration"
    required_modalities: tuple[CaptureModality, ...] = ()
    produced_modalities: tuple[CaptureModality, ...] = ()
    produced_artifacts: tuple[str, ...] = ("heightmap_calibration",)
    stage_category: str | None = "calibration"

    def run(self, context: PipelineContext) -> None:
        metadata = context.get_artifact("metadata", {})
        calibration = {
            "x_resolution_mm": float(((metadata.get("calibration") or {}).get("x_resolution_mm") or 1.0)),
            "y_resolution_mm": float(((metadata.get("calibration") or {}).get("profile_distance_mm") or 1.0)),
            "z_scale": float(((metadata.get("calibration") or {}).get("z_scale") or 1.0)),
            "z_offset": float(((metadata.get("calibration") or {}).get("z_offset") or 0.0)),
            "coordinate_system": "belt_xy_z_mm",
            "tilt_correction": None,
        }
        context.set_artifact("heightmap_calibration", calibration)


@dataclass
class DetectBeltPlaneStage:
    background_detection_strategy: str = "low_gradient_surface"
    plane_fit_roi: dict[str, Any] | None = None
    plane_fit_downsample: int = 30000
    background_selection_mode: str = "nearest_percentile"
    background_percentile: float = 20.0
    background_candidate_morphology: bool = True
    background_candidate_open_kernel: int = 3
    background_candidate_min_component_area: int = 250
    background_must_touch_roi_border: bool = True
    plane_fit_min_valid_pixels: int = 64
    plane_fit_min_inlier_ratio: float = 0.35
    plane_fit_residual_threshold_mm: float = 1.25
    # Multiplier on the candidate surface's z-MAD used to expand the RANSAC
    # inlier tolerance for noisy sensors / raw-unit heightmaps. The effective
    # threshold is max(plane_fit_residual_threshold_mm, multiplier * z_mad).
    plane_fit_residual_threshold_adaptive_multiplier: float = 3.0
    plane_fit_max_iterations: int = 250
    plane_background_residual_tolerance_mm: float = 2.5
    plane_background_residual_tolerance_mode: str = "adaptive"
    plane_background_residual_adaptive_multiplier: float = 2.5
    plane_background_min_coverage_ratio: float = 0.25
    plane_background_fill_holes: bool = True
    plane_background_close_kernel: int = 5
    plane_refit_after_expansion: bool = True
    plane_refit_max_iterations: int = 2
    gradient_smoothing_kernel: int = 3
    gradient_method: str = "sobel"
    gradient_threshold_mode: str = "percentile"
    gradient_threshold_value: float = 2.0
    gradient_threshold_percentile: float = 70.0
    invalid_neighbor_policy: str = "mark_high_gradient"
    low_gradient_morphology_enabled: bool = True
    low_gradient_open_kernel: int = 3
    # Close kernel larger than the smallest object-to-belt spacing can bridge
    # the conveyor mask into adjacent low-gradient surfaces (cube/ball plateaus),
    # inflating the selected component and corrupting the plane fit.
    low_gradient_close_kernel: int = 5
    low_gradient_min_component_area: int = 1500
    low_gradient_fill_holes: bool = True
    reference_surface_selection_mode: str = "largest_constant_z"
    reference_surface_min_area_ratio: float = 0.08
    reference_surface_max_z_std_mm: float = 8.0
    reference_surface_border_bonus: float = 0.2
    reference_surface_depth_preference_weight: float = 0.2
    reference_surface_constancy_weight: float = 0.5
    reference_surface_area_weight: float = 0.3
    reference_surface_model: str = "auto"
    reference_surface_max_plane_residual_p95_mm: float = 3.0
    reference_surface_height_gate_enabled: bool = True
    reference_surface_height_gate_margin_mm: float = 8.0
    reference_surface_height_gate_min_coverage_ratio: float = 0.10
    reference_surface_height_gate_max_coverage_ratio: float = 0.95
    random_seed: int = 7
    name: str = "DetectBeltPlane"
    category: str = "calibration"
    required_modalities: tuple[CaptureModality, ...] = ("heightmap",)
    produced_modalities: tuple[CaptureModality, ...] = ()
    produced_artifacts: tuple[str, ...] = ("belt_plane",)
    stage_category: str | None = "calibration"

    def run(self, context: PipelineContext) -> None:
        frame: HeightmapFrame = context.require_artifact("heightmap_frame")
        output_dir: Path = context.require_artifact("output_dir")
        write_heightmap_preview_png(frame.z_mm, frame.valid_mask, output_dir / "raw_heightmap_preview.png")
        raw_min_mm, raw_max_mm = _valid_min_max_mm(frame.z_mm, frame.valid_mask)
        valid_mask_u8 = (frame.valid_mask.astype(np.uint8) * 255)
        cv2.imwrite(str(output_dir / "valid_mask.png"), valid_mask_u8)

        roi_cfg = self.plane_fit_roi or _roi_from_metadata(context.get_artifact("metadata", {}))
        roi_mask = _roi_mask(frame.valid_mask.shape, roi_cfg) if roi_cfg is not None else np.ones_like(frame.valid_mask, dtype=bool)
        cv2.imwrite(str(output_dir / "plane_fit_roi_mask.png"), (roi_mask.astype(np.uint8) * 255))
        valid_for_fit = frame.valid_mask & roi_mask
        valid_pixel_count = int(np.count_nonzero(valid_for_fit))
        roi_enabled = roi_cfg is not None
        roi_type = str(roi_cfg.get("type") or roi_cfg.get("roi_type") or "rectangle").lower() if isinstance(roi_cfg, dict) else "rectangle"
        roi_x = int(roi_cfg.get("x", 0)) if isinstance(roi_cfg, dict) else 0
        roi_w = int(roi_cfg.get("width", frame.valid_mask.shape[1])) if isinstance(roi_cfg, dict) else int(frame.valid_mask.shape[1])
        if roi_type == "vertical_band":
            roi_y = 0
            roi_h = int(frame.valid_mask.shape[0])
        else:
            roi_y = int(roi_cfg.get("y", 0)) if isinstance(roi_cfg, dict) else 0
            roi_h = int(roi_cfg.get("height", frame.valid_mask.shape[0])) if isinstance(roi_cfg, dict) else int(frame.valid_mask.shape[0])
        roi_area_ratio = float(np.count_nonzero(roi_mask) / max(1, roi_mask.size))

        status = "success"
        failure_reason: str | None = None
        sampled_pixel_count = 0
        inlier_count = 0
        inlier_ratio = 0.0
        coeffs = None
        warnings: list[str] = []
        border_connected_count = 0
        component_stats: list[dict[str, Any]] = []
        expanded_plane_coverage = 0.0
        seed_mask = np.zeros_like(valid_for_fit, dtype=bool)
        expanded_plane_mask = np.zeros_like(valid_for_fit, dtype=bool)
        final_plane_inlier_mask = np.zeros_like(valid_for_fit, dtype=bool)
        candidate_mask = np.zeros_like(valid_for_fit, dtype=bool)
        inferred_depth_convention = "unknown"
        candidate_coverage = 0.0
        z_thresholds: dict[str, float] = {}
        height_gate_mask = valid_for_fit.copy()
        active_height_gate_mask = valid_for_fit.copy()
        height_gate_debug: dict[str, Any] = {"enabled": bool(self.reference_surface_height_gate_enabled), "status": "not_evaluated"}
        gradient_debug: dict[str, Any] = {}
        cand_z_mad = 0.0
        effective_residual_threshold_mm = float(self.plane_fit_residual_threshold_mm)

        if valid_pixel_count < self.plane_fit_min_valid_pixels:
            status = "failed"
            failure_reason = "not_enough_valid_pixels"
        else:
            z_values = frame.z_mm[valid_for_fit].astype(np.float64)
            p = float(np.clip(self.background_percentile, 1.0, 99.0))
            near_q = float(np.percentile(z_values, p))
            far_q = float(np.percentile(z_values, 100.0 - p))
            z_thresholds = {"near_q": near_q, "far_q": far_q, "percentile": p}
            low_mask = valid_for_fit & (frame.z_mm <= near_q)
            high_mask = valid_for_fit & (frame.z_mm >= far_q)
            height_gate_mask, height_gate_debug = _reference_height_gate_from_distribution(
                frame.z_mm,
                valid_for_fit,
                enabled=bool(self.reference_surface_height_gate_enabled),
                margin_mm=float(self.reference_surface_height_gate_margin_mm),
                min_coverage_ratio=float(self.reference_surface_height_gate_min_coverage_ratio),
                max_coverage_ratio=float(self.reference_surface_height_gate_max_coverage_ratio),
            )
            active_height_gate_mask = height_gate_mask if self.background_detection_strategy == "low_gradient_surface" else valid_for_fit

            if self.background_detection_strategy == "low_gradient_surface":
                grad = _compute_depth_gradient(
                    frame.z_mm,
                    frame.valid_mask,
                    smoothing_kernel=int(self.gradient_smoothing_kernel),
                    method=str(self.gradient_method),
                    invalid_neighbor_policy=str(self.invalid_neighbor_policy),
                )
                gvals = grad[valid_for_fit]
                if self.gradient_threshold_mode == "fixed":
                    gthr = float(self.gradient_threshold_value)
                elif self.gradient_threshold_mode == "otsu":
                    gthr = _otsu_threshold(gvals)
                else:
                    gp = float(np.clip(self.gradient_threshold_percentile, 1.0, 99.0))
                    gthr = float(np.percentile(gvals, gp)) if gvals.size else float(self.gradient_threshold_value)
                low_grad_mask = valid_for_fit & (grad <= gthr)
                if self.low_gradient_morphology_enabled:
                    ok = max(1, int(self.low_gradient_open_kernel))
                    ck = max(1, int(self.low_gradient_close_kernel))
                    u8 = (low_grad_mask.astype(np.uint8) * 255)
                    u8 = cv2.morphologyEx(u8, cv2.MORPH_OPEN, np.ones((ok, ok), np.uint8))
                    u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, np.ones((ck, ck), np.uint8))
                    low_grad_mask = u8 > 0
                lg_result, selected_label = _select_low_gradient_component(
                    low_grad_mask=low_grad_mask,
                    gradient=grad,
                    z=frame.z_mm,
                    valid_mask=valid_for_fit,
                    roi_mask=roi_mask,
                    selection_mode=self.reference_surface_selection_mode,
                    min_area_ratio=float(self.reference_surface_min_area_ratio),
                    border_bonus=float(self.reference_surface_border_bonus),
                    area_weight=float(self.reference_surface_area_weight),
                    constancy_weight=float(self.reference_surface_constancy_weight),
                    depth_pref_weight=float(self.reference_surface_depth_preference_weight),
                )
                selected_mask = np.zeros_like(valid_for_fit, dtype=bool)
                if selected_label > 0:
                    selected_mask = (lg_result["labels"] == selected_label) & valid_for_fit
                candidate_mask = selected_mask & height_gate_mask
                inferred_depth_convention = "low_gradient_surface"
                selected_row = next((row for row in (lg_result.get("candidates") or []) if int(row.get("label") or 0) == int(selected_label)), {})
                gradient_debug = {
                    "strategy": "low_gradient_surface",
                    "threshold_mode": self.gradient_threshold_mode,
                    "threshold_value": float(gthr),
                    "selected_component_id": int(selected_label),
                    "selected_component_area_ratio": float(np.count_nonzero(candidate_mask) / max(1, np.count_nonzero(valid_for_fit))),
                    "selected_component_z_std_mm": float(selected_row.get("z_std") or 0.0),
                    "selected_component_z_mad_mm": float(selected_row.get("z_mad") or 0.0),
                    "selected_component_gradient_p95": float(selected_row.get("gradient_p95") or 0.0),
                    "selected_component_score": float(selected_row.get("score") or 0.0),
                    "component_count": int(len(lg_result.get("candidates") or [])),
                    "height_gate": height_gate_debug,
                }
                write_heightmap_preview_png(grad, valid_for_fit, output_dir / "depth_gradient_magnitude.png")
                cv2.imwrite(str(output_dir / "low_gradient_mask.png"), (low_grad_mask.astype(np.uint8) * 255))
                cv2.imwrite(str(output_dir / "reference_surface_selected_mask.png"), (candidate_mask.astype(np.uint8) * 255))
                labels = np.asarray(lg_result.get("labels"), dtype=np.int32)
                cmap = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
                if labels.size:
                    lut = cv2.applyColorMap((labels % 251).astype(np.uint8), cv2.COLORMAP_TURBO)
                    cmap = lut
                    cmap[labels == 0] = (0, 0, 0)
                cv2.imwrite(str(output_dir / "low_gradient_components_overlay.png"), cmap)
                (output_dir / "low_gradient_components.json").write_text(json.dumps(lg_result.get("candidates", []), indent=2), encoding="utf-8")
                (output_dir / "reference_surface_candidates.json").write_text(json.dumps(lg_result.get("candidates", []), indent=2), encoding="utf-8")
                (output_dir / "gradient_debug.json").write_text(json.dumps(gradient_debug, indent=2), encoding="utf-8")
            else:
                if self.background_selection_mode == "nearest_percentile":
                    inferred_depth_convention = "smaller_z_is_farther"
                    candidate_mask = low_mask
                elif self.background_selection_mode == "farthest_percentile":
                    inferred_depth_convention = "larger_z_is_farther"
                    candidate_mask = high_mask
                else:
                    low_score = _background_candidate_score(low_mask, roi_mask)
                    high_score = _background_candidate_score(high_mask, roi_mask)
                    if high_score >= low_score:
                        inferred_depth_convention = "larger_z_is_farther"
                        candidate_mask = high_mask
                    else:
                        inferred_depth_convention = "smaller_z_is_farther"
                        candidate_mask = low_mask

            hist_counts, hist_edges = np.histogram(z_values, bins=64)
            hist_payload = {
                "bin_counts": [int(x) for x in hist_counts.tolist()],
                "bin_edges": [float(x) for x in hist_edges.tolist()],
                "sample_count": int(z_values.size),
                "percentile": p,
                "near_q": near_q,
                "far_q": far_q,
            }
            (output_dir / "background_depth_histogram.json").write_text(json.dumps(hist_payload, indent=2), encoding="utf-8")
            _write_depth_plot_png(z_values, near_q=near_q, far_q=far_q, out_path=output_dir / "background_depth_plot.png")

            if self.background_candidate_morphology:
                k = max(1, int(self.background_candidate_open_kernel))
                kernel = np.ones((k, k), dtype=np.uint8)
                opened = cv2.morphologyEx((candidate_mask.astype(np.uint8) * 255), cv2.MORPH_OPEN, kernel)
                candidate_mask = opened > 0

            def _filter_components(mask: np.ndarray) -> tuple[np.ndarray, int, list[dict[str, Any]]]:
                if not np.count_nonzero(mask):
                    return mask, 0, []
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
                filtered_local = np.zeros_like(mask, dtype=bool)
                border_mask = _roi_border_mask(roi_mask)
                component_stats_local: list[dict[str, Any]] = []
                border_count_local = 0
                for label in range(1, num_labels):
                    area = int(stats[label, cv2.CC_STAT_AREA])
                    touch_border = bool(np.any(border_mask & (labels == label)))
                    component_stats_local.append({"label": int(label), "area_px": area, "touches_roi_border": touch_border})
                    if area < int(self.background_candidate_min_component_area):
                        continue
                    if self.background_must_touch_roi_border and not touch_border:
                        continue
                    filtered_local[labels == label] = True
                    if touch_border:
                        border_count_local += 1
                return filtered_local, border_count_local, component_stats_local

            if np.count_nonzero(candidate_mask):
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_mask.astype(np.uint8), connectivity=8)
                filtered = np.zeros_like(candidate_mask, dtype=bool)
                border_mask = _roi_border_mask(roi_mask)
                for label in range(1, num_labels):
                    area = int(stats[label, cv2.CC_STAT_AREA])
                    touch_border = bool(np.any(border_mask & (labels == label)))
                    component_stats.append({"label": int(label), "area_px": area, "touches_roi_border": touch_border})
                    if area < int(self.background_candidate_min_component_area):
                        continue
                    if self.background_must_touch_roi_border and not touch_border:
                        continue
                    filtered[labels == label] = True
                    if touch_border:
                        border_connected_count += 1
                if np.count_nonzero(filtered):
                    candidate_mask = filtered
                elif np.count_nonzero(candidate_mask) and self.background_must_touch_roi_border:
                    warnings.append("border_connected_filtering_removed_all_candidates")
                    # automatic mode fallback: try opposite depth convention if current side fails border prior
                    if self.background_selection_mode == "automatic":
                        opposite_mask = low_mask if inferred_depth_convention == "larger_z_is_farther" else high_mask
                        if self.background_candidate_morphology:
                            k = max(1, int(self.background_candidate_open_kernel))
                            kernel = np.ones((k, k), dtype=np.uint8)
                            opened = cv2.morphologyEx((opposite_mask.astype(np.uint8) * 255), cv2.MORPH_OPEN, kernel)
                            opposite_mask = opened > 0
                        opposite_filtered, opposite_border_count, opposite_stats = _filter_components(opposite_mask)
                        if np.count_nonzero(opposite_filtered) and opposite_border_count > 0:
                            candidate_mask = opposite_filtered
                            border_connected_count = opposite_border_count
                            component_stats = opposite_stats
                            inferred_depth_convention = (
                                "smaller_z_is_farther"
                                if inferred_depth_convention == "larger_z_is_farther"
                                else "larger_z_is_farther"
                            )
                            warnings.append("automatic_convention_flipped_due_to_border_prior")

            candidate_coverage = float(np.count_nonzero(candidate_mask) / max(1, np.count_nonzero(valid_for_fit)))
            seed_mask = candidate_mask.copy()
            if self.background_detection_strategy == "low_gradient_surface":
                if candidate_coverage < float(self.reference_surface_min_area_ratio):
                    warnings.append("selected_reference_surface_too_small")
                if candidate_coverage < 0.02:
                    warnings.append("low_gradient_mask_coverage_too_low")
            if (
                inferred_depth_convention == "smaller_z_is_farther"
                and abs(near_q) < 1e-6
                and candidate_coverage > 0.30
            ):
                warnings.append("background_selection_bias_toward_zero_depth")
            if np.count_nonzero(candidate_mask) < self.plane_fit_min_valid_pixels:
                status = "failed"
                failure_reason = "not_enough_background_candidates"
                warnings.append("background_candidate_selection_too_sparse")

            candidates = _heightmap_points(frame, valid_mask=candidate_mask)
            if candidates.shape[0] > self.plane_fit_downsample:
                rng = np.random.default_rng(self.random_seed)
                idx = rng.choice(candidates.shape[0], size=self.plane_fit_downsample, replace=False)
                candidates = candidates[idx]
            sampled_pixel_count = int(candidates.shape[0])
            # Scale the RANSAC inlier tolerance to the actual noise of the
            # selected candidate surface. Without this, sensors / calibrations
            # whose belt noise std is larger than `plane_fit_residual_threshold_mm`
            # (e.g. raw TriSpector counts) cannot reach the inlier-ratio gate
            # and degenerate into a spurious tilted plane.
            if candidates.shape[0] >= 3:
                cand_z = candidates[:, 2]
                cand_med = float(np.median(cand_z))
                cand_z_mad = float(np.median(np.abs(cand_z - cand_med)))
            else:
                cand_z_mad = 0.0
            effective_residual_threshold_mm = max(
                float(self.plane_fit_residual_threshold_mm),
                float(self.plane_fit_residual_threshold_adaptive_multiplier) * cand_z_mad,
            )
            if sampled_pixel_count < max(3, self.plane_fit_min_valid_pixels // 2):
                status = "failed"
                failure_reason = "not_enough_sampled_points_after_filtering"
            else:
                coeffs, inliers = _fit_plane_ransac(
                    candidates,
                    iterations=self.plane_fit_max_iterations,
                    threshold_mm=effective_residual_threshold_mm,
                    seed=self.random_seed,
                )
                inlier_count = int(inliers.shape[0])
                inlier_ratio = float(inlier_count / max(1, sampled_pixel_count))
                if inlier_ratio < self.plane_fit_min_inlier_ratio:
                    status = "failed"
                    failure_reason = "inlier_ratio_below_threshold"

            # Expand belt/background over all valid ROI pixels via residual tolerance.
            if coeffs is not None:
                residual_seed = _residual_map(frame, coeffs)
                seed_residual_values = np.abs(residual_seed[candidate_mask])
                adaptive_tol = float(self.plane_background_residual_tolerance_mm)
                if self.plane_background_residual_tolerance_mode == "adaptive" and seed_residual_values.size:
                    adaptive_tol = max(
                        effective_residual_threshold_mm,
                        float(np.std(seed_residual_values) * self.plane_background_residual_adaptive_multiplier),
                    )
                tol = adaptive_tol if self.plane_background_residual_tolerance_mode == "adaptive" else float(self.plane_background_residual_tolerance_mm)
                expanded_plane_mask = (np.abs(residual_seed) <= tol) & valid_for_fit & active_height_gate_mask
                if self.plane_background_fill_holes:
                    close_k = max(1, int(self.plane_background_close_kernel))
                    kernel = np.ones((close_k, close_k), np.uint8)
                    expanded_u8 = (expanded_plane_mask.astype(np.uint8) * 255)
                    expanded_u8 = cv2.morphologyEx(expanded_u8, cv2.MORPH_CLOSE, kernel)
                    if self.plane_background_fill_holes:
                        flood = expanded_u8.copy()
                        h, w = flood.shape
                        flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
                        cv2.floodFill(flood, flood_mask, (0, 0), 255)
                        expanded_u8 = expanded_u8 | cv2.bitwise_not(flood)
                    expanded_plane_mask = (expanded_u8 > 0) & active_height_gate_mask
                expanded_plane_coverage = float(np.count_nonzero(expanded_plane_mask) / max(1, np.count_nonzero(valid_for_fit)))
                if expanded_plane_coverage < float(self.plane_background_min_coverage_ratio):
                    warnings.append("expanded_plane_coverage_too_low")

                final_plane_inlier_mask = expanded_plane_mask.copy()
                if self.plane_refit_after_expansion and np.count_nonzero(final_plane_inlier_mask) >= max(3, self.plane_fit_min_valid_pixels):
                    for _ in range(max(1, int(self.plane_refit_max_iterations))):
                        refit_points = _heightmap_points(frame, valid_mask=final_plane_inlier_mask)
                        if refit_points.shape[0] < 3:
                            break
                        coeffs = _least_squares_plane(refit_points)
                        residual_refit = _residual_map(frame, coeffs)
                        final_plane_inlier_mask = (np.abs(residual_refit) <= tol) & valid_for_fit & active_height_gate_mask
                else:
                    final_plane_inlier_mask = expanded_plane_mask.copy()

        # Fallback plane for deterministic downstream execution
        if coeffs is None:
            # z = median(valid), plane normal aligned with z-axis
            valid_values = frame.z_mm[valid_for_fit]
            z0 = float(np.median(valid_values)) if valid_values.size else 0.0
            coeffs = np.asarray([0.0, 0.0, 1.0, -z0], dtype=np.float64)

        all_valid_points = _heightmap_points(frame, valid_mask=valid_for_fit)
        residuals = _plane_residuals(all_valid_points, coeffs) if all_valid_points.size else np.asarray([], dtype=np.float64)
        residual_abs = np.abs(residuals) if residuals.size else np.asarray([0.0], dtype=np.float64)
        residual_map = _residual_map(frame, coeffs)
        if np.count_nonzero(final_plane_inlier_mask):
            belt_mask = final_plane_inlier_mask
        else:
            belt_mask = (np.abs(residual_map) <= effective_residual_threshold_mm) & valid_for_fit
        background_vals = residual_map[belt_mask]
        bg_mean = float(np.mean(background_vals)) if background_vals.size else 0.0
        bg_std = float(np.std(background_vals)) if background_vals.size else 0.0
        bg_p95_abs = float(np.percentile(np.abs(background_vals), 95)) if background_vals.size else 0.0
        foreground_vals = residual_map[(~belt_mask) & valid_for_fit]
        fg_mean = float(np.mean(np.abs(foreground_vals))) if foreground_vals.size else 0.0
        # Noise-scaled gates: a sensor whose belt noise floor is 6mm shouldn't
        # trip the "near zero" warning when residual p95 is naturally ~12mm.
        bg_near_zero_limit_mm = max(4.0, 2.0 * effective_residual_threshold_mm)
        max_plane_residual_p95_limit_mm = max(
            float(self.reference_surface_max_plane_residual_p95_mm),
            2.0 * effective_residual_threshold_mm,
        )
        if bg_p95_abs > bg_near_zero_limit_mm:
            warnings.append("background_not_near_zero_after_plane_fit")
        if fg_mean < 3.0:
            warnings.append("foreground_background_separation_too_small")
        if inlier_ratio < max(0.15, self.plane_fit_min_inlier_ratio * 0.5):
            warnings.append("inlier_ratio_very_low")
        if sampled_pixel_count and sampled_pixel_count <= max(3, self.plane_fit_min_valid_pixels // 2):
            warnings.append("plane_fit_used_very_few_samples")
        if bg_p95_abs > 100.0:
            warnings.append("extreme_plane_fit_residuals_detected")
        if float(np.percentile(residual_abs, 95)) > max_plane_residual_p95_limit_mm:
            warnings.append("model_residual_p95_too_high")
        if np.count_nonzero(candidate_mask) and np.count_nonzero(belt_mask):
            overlap = float(np.count_nonzero(candidate_mask & belt_mask) / max(1, np.count_nonzero(candidate_mask)))
            if overlap < 0.3:
                warnings.append("background_candidates_poorly_overlap_plane_inliers")

        stats = {
            "count": int(residuals.shape[0]),
            "mean_abs_mm": float(np.mean(residual_abs)),
            "p95_abs_mm": float(np.percentile(residual_abs, 95)),
            "max_abs_mm": float(np.max(residual_abs)),
            "inlier_ratio": inlier_ratio,
        }
        reference_model_type = "plane"
        model_selection_reason = "plane_fit_ok"
        reference_constant_z_mm = float(np.median(frame.z_mm[candidate_mask])) if np.count_nonzero(candidate_mask) else float(np.median(frame.z_mm[valid_for_fit]))
        use_constant_z = False
        if self.reference_surface_model == "constant_z":
            use_constant_z = True
            model_selection_reason = "forced_constant_z"
        elif self.reference_surface_model == "plane":
            use_constant_z = False
            model_selection_reason = "forced_plane"
        elif self.reference_surface_model == "auto":
            auto_fail = (
                status != "success"
                or inlier_ratio < max(0.15, self.plane_fit_min_inlier_ratio * 0.6)
                or bg_p95_abs > max_plane_residual_p95_limit_mm
            )
            if auto_fail:
                use_constant_z = True
                model_selection_reason = "auto_constant_z_fallback"
            else:
                model_selection_reason = "auto_plane_selected"
        if use_constant_z:
            reference_model_type = "constant_z"
            coeffs = np.asarray([0.0, 0.0, 1.0, -reference_constant_z_mm], dtype=np.float64)
            residual_map = frame.z_mm.astype(np.float32) - np.float32(reference_constant_z_mm)
            residual_map[~frame.valid_mask] = 0.0
            warnings.append("constant_z_fallback_used")

        context.set_artifact("belt_plane", tuple(float(x) for x in coeffs.tolist()))
        context.set_artifact("plane_model", tuple(float(x) for x in coeffs.tolist()))
        context.set_artifact(
            "reference_surface_model",
            {
                "type": reference_model_type,
                "plane_coefficients": [float(x) for x in coeffs.tolist()] if reference_model_type == "plane" else None,
                "constant_z_mm": reference_constant_z_mm if reference_model_type == "constant_z" else None,
                "model_selection_reason": model_selection_reason,
            },
        )
        context.set_artifact("belt_plane_stats", stats)
        context.set_artifact("plane_inlier_mask", belt_mask)
        context.set_artifact("background_seed_mask", seed_mask)
        context.set_artifact("expanded_plane_mask", expanded_plane_mask)
        context.set_artifact("final_plane_inlier_mask", belt_mask)
        context.set_artifact("reference_surface_selected_mask", candidate_mask)
        context.set_artifact("plane_fit_status", status)
        context.set_artifact("plane_fit_failure_reason", failure_reason)

        plane_json = {
            "reference_surface_model_type": reference_model_type,
            "constant_z_mm": reference_constant_z_mm if reference_model_type == "constant_z" else None,
            "model_selection_reason": model_selection_reason,
            "equation": {"a": float(coeffs[0]), "b": float(coeffs[1]), "c": float(coeffs[2]), "d": float(coeffs[3])},
            "residual_stats_mm": stats,
            "status": status,
            "failure_reason": failure_reason,
        }
        (output_dir / "belt_plane.json").write_text(json.dumps(plane_json, indent=2), encoding="utf-8")

        write_heightmap_preview_png(
            np.abs(residual_map),
            frame.valid_mask,
            output_dir / "belt_plane_residuals.png",
            colorbar_label="absolute residual (mm)",
        )
        residual_raster_name = "plane_signed_distance.values.f32"
        residual_map.astype(np.float32).tofile(output_dir / residual_raster_name)
        # Backward-compatible alias while semantic raster contracts propagate.
        residual_map.astype(np.float32).tofile(output_dir / "plane_residuals.values.f32")
        cv2.imwrite(str(output_dir / "plane_inlier_mask.png"), (belt_mask.astype(np.uint8) * 255))
        cv2.imwrite(str(output_dir / "background_candidate_mask.png"), (candidate_mask.astype(np.uint8) * 255))
        cv2.imwrite(str(output_dir / "background_seed_mask.png"), (seed_mask.astype(np.uint8) * 255))
        cv2.imwrite(str(output_dir / "expanded_plane_mask.png"), (expanded_plane_mask.astype(np.uint8) * 255))
        cv2.imwrite(str(output_dir / "final_plane_inlier_mask.png"), (belt_mask.astype(np.uint8) * 255))
        overlay = _render_belt_plane_overlay(frame, belt_mask)
        cv2.imwrite(str(output_dir / "belt_plane_overlay.png"), overlay)

        roi_valid_ratio = float(np.count_nonzero(valid_for_fit) / max(1, np.count_nonzero(roi_mask)))
        roi_candidate_coverage = float(np.count_nonzero(candidate_mask) / max(1, np.count_nonzero(roi_mask)))
        roi_selected_surface_coverage = float(np.count_nonzero(belt_mask) / max(1, np.count_nonzero(roi_mask)))
        if roi_valid_ratio < 0.2:
            warnings.append("roi_valid_pixels_too_low")
        if roi_candidate_coverage < 0.05:
            warnings.append("roi_candidate_coverage_too_low")
        if roi_selected_surface_coverage < 0.05:
            warnings.append("roi_selected_surface_coverage_too_low")
        if roi_area_ratio < 0.03:
            warnings.append("roi_coverage_too_small")

        background_selection_debug = {
            "background_detection_strategy": self.background_detection_strategy,
            "inferred_depth_convention": inferred_depth_convention,
            "background_selection_mode": self.background_selection_mode,
            "percentile_thresholds": z_thresholds,
            "selected_candidate_pixel_count": int(np.count_nonzero(candidate_mask)),
            "candidate_coverage_percent": candidate_coverage * 100.0,
            "seed_coverage_percent": candidate_coverage * 100.0,
            "expanded_plane_coverage_percent": expanded_plane_coverage * 100.0,
            "final_inlier_coverage_percent": float(np.count_nonzero(belt_mask) / max(1, np.count_nonzero(valid_for_fit)) * 100.0),
            "morphology": {
                "enabled": bool(self.background_candidate_morphology),
                "open_kernel": int(self.background_candidate_open_kernel),
                "min_component_area": int(self.background_candidate_min_component_area),
            },
            "border_connected_filtering": {
                "enabled": bool(self.background_must_touch_roi_border),
                "border_connected_component_count": int(border_connected_count),
            },
            "height_gate": height_gate_debug,
            "component_stats": component_stats[:40],
            "warnings": warnings,
            "roi_inspector": {
                "enabled": roi_enabled,
                "coverage_ratio": roi_area_ratio,
                "coverage_percent": roi_area_ratio * 100.0,
                "valid_pixel_ratio": roi_valid_ratio,
                "candidate_coverage_ratio": roi_candidate_coverage,
                "selected_surface_coverage_ratio": roi_selected_surface_coverage,
            },
        }
        gradient_debug_path = output_dir / "gradient_debug.json"
        if gradient_debug_path.is_file():
            try:
                background_selection_debug["gradient_debug"] = json.loads(gradient_debug_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        (output_dir / "background_selection_debug.json").write_text(json.dumps(background_selection_debug, indent=2), encoding="utf-8")
        debug_payload = {
            "status": status,
            "failure_reason": failure_reason,
            "valid_pixel_count": valid_pixel_count,
            "sampled_pixel_count": sampled_pixel_count,
            "inlier_count": inlier_count,
            "inlier_ratio": inlier_ratio,
            "effective_plane_fit_residual_threshold_mm": float(effective_residual_threshold_mm),
            "candidate_z_mad_mm": float(cand_z_mad),
            "plane_coefficients": [float(x) for x in coeffs.tolist()],
            "residual_mean_mm": float(np.mean(residuals)) if residuals.size else 0.0,
            "residual_median_mm": float(np.median(residuals)) if residuals.size else 0.0,
            "residual_p95_mm": float(np.percentile(np.abs(residuals), 95)) if residuals.size else 0.0,
            "residual_max_abs_mm": float(np.max(np.abs(residuals))) if residuals.size else 0.0,
            "background_height_mean_after_normalization_mm": bg_mean,
            "background_height_std_after_normalization_mm": bg_std,
            "background_height_p95_abs_after_normalization_mm": bg_p95_abs,
            "foreground_mean_height_mm": fg_mean,
            "reference_surface_model_type": reference_model_type,
            "constant_z_mm": reference_constant_z_mm if reference_model_type == "constant_z" else None,
            "model_selection_reason": model_selection_reason,
            "model_residual_mean_mm": float(np.mean(residual_abs)) if residual_abs.size else 0.0,
            "model_residual_p95_mm": float(np.percentile(residual_abs, 95)) if residual_abs.size else 0.0,
            "roi_enabled": roi_enabled,
            "roi_type": roi_type,
            "roi_x": roi_x,
            "roi_y": roi_y,
            "roi_width": roi_w,
            "roi_height": roi_h,
            "roi_area_ratio": roi_area_ratio,
            "warnings": warnings,
            "roi": {
                "enabled": roi_enabled,
                "type": roi_type,
                "x": roi_x,
                "y": roi_y,
                "width": roi_w,
                "height": roi_h,
                "area_ratio": roi_area_ratio,
                "valid_pixel_ratio": roi_valid_ratio,
                "candidate_coverage_ratio": roi_candidate_coverage,
                "selected_surface_coverage_ratio": roi_selected_surface_coverage,
            },
            "background_selection": background_selection_debug,
            "config": {
                "plane_fit_roi": roi_cfg,
                "background_detection_strategy": self.background_detection_strategy,
                "plane_fit_downsample": self.plane_fit_downsample,
                "background_selection_mode": self.background_selection_mode,
                "background_percentile": self.background_percentile,
                "background_candidate_morphology": self.background_candidate_morphology,
                "background_candidate_open_kernel": self.background_candidate_open_kernel,
                "background_candidate_min_component_area": self.background_candidate_min_component_area,
                "background_must_touch_roi_border": self.background_must_touch_roi_border,
                "plane_fit_min_valid_pixels": self.plane_fit_min_valid_pixels,
                "plane_fit_min_inlier_ratio": self.plane_fit_min_inlier_ratio,
                "plane_fit_residual_threshold_mm": self.plane_fit_residual_threshold_mm,
                "plane_fit_residual_threshold_adaptive_multiplier": self.plane_fit_residual_threshold_adaptive_multiplier,
                "plane_fit_max_iterations": self.plane_fit_max_iterations,
                "gradient_smoothing_kernel": self.gradient_smoothing_kernel,
                "gradient_method": self.gradient_method,
                "gradient_threshold_mode": self.gradient_threshold_mode,
                "gradient_threshold_value": self.gradient_threshold_value,
                "gradient_threshold_percentile": self.gradient_threshold_percentile,
                "invalid_neighbor_policy": self.invalid_neighbor_policy,
                "reference_surface_selection_mode": self.reference_surface_selection_mode,
                "reference_surface_model": self.reference_surface_model,
                "reference_surface_height_gate_enabled": self.reference_surface_height_gate_enabled,
                "reference_surface_height_gate_margin_mm": self.reference_surface_height_gate_margin_mm,
                "reference_surface_height_gate_min_coverage_ratio": self.reference_surface_height_gate_min_coverage_ratio,
                "reference_surface_height_gate_max_coverage_ratio": self.reference_surface_height_gate_max_coverage_ratio,
            },
        }
        residual_hist_counts, residual_hist_edges = np.histogram(residuals.astype(np.float64) if residuals.size else np.asarray([0.0]), bins=64)
        residual_hist_payload = {
            "bin_counts": [int(x) for x in residual_hist_counts.tolist()],
            "bin_edges": [float(x) for x in residual_hist_edges.tolist()],
            "sample_count": int(residuals.size),
            "signed": True,
            "units": "mm",
        }
        (output_dir / "plane_residual_histogram.json").write_text(json.dumps(residual_hist_payload, indent=2), encoding="utf-8")
        (output_dir / "plane_fit_debug.json").write_text(json.dumps(debug_payload, indent=2), encoding="utf-8")
        selected_surface_debug = {
            "reference_surface_model_type": reference_model_type,
            "plane_coefficients": [float(x) for x in coeffs.tolist()] if reference_model_type == "plane" else None,
            "constant_z_mm": reference_constant_z_mm if reference_model_type == "constant_z" else None,
            "selected_component_id": (gradient_debug.get("selected_component_id") if isinstance(gradient_debug, dict) else None),
            "selected_component_area_ratio": (gradient_debug.get("selected_component_area_ratio") if isinstance(gradient_debug, dict) else None),
            "model_residual_mean_mm": float(np.mean(residual_abs)) if residual_abs.size else 0.0,
            "model_residual_p95_mm": float(np.percentile(residual_abs, 95)) if residual_abs.size else 0.0,
            "model_selection_reason": model_selection_reason,
        }
        (output_dir / "selected_surface_debug.json").write_text(json.dumps(selected_surface_debug, indent=2), encoding="utf-8")
        files = dict(context.get_artifact("files", {}))
        files.update(
            {
                "belt_plane": "belt_plane.json",
                "raw_heightmap_preview": "raw_heightmap_preview.png",
                "valid_mask": "valid_mask.png",
                "plane_fit_roi_mask": "plane_fit_roi_mask.png",
                "background_candidate_mask": "background_candidate_mask.png",
                "background_depth_histogram": "background_depth_histogram.json",
                "background_depth_plot": "background_depth_plot.png",
                "depth_gradient_magnitude": "depth_gradient_magnitude.png",
                "low_gradient_mask": "low_gradient_mask.png",
                "low_gradient_components_overlay": "low_gradient_components_overlay.png",
                "low_gradient_components": "low_gradient_components.json",
                "reference_surface_selected_mask": "reference_surface_selected_mask.png",
                "reference_surface_candidates": "reference_surface_candidates.json",
                "gradient_debug": "gradient_debug.json",
                "selected_surface_debug": "selected_surface_debug.json",
                "background_selection_debug": "background_selection_debug.json",
                "background_seed_mask": "background_seed_mask.png",
                "expanded_plane_mask": "expanded_plane_mask.png",
                "final_plane_inlier_mask": "final_plane_inlier_mask.png",
                "plane_inlier_mask": "plane_inlier_mask.png",
                "plane_fit_debug": "plane_fit_debug.json",
                "belt_plane_overlay": "belt_plane_overlay.png",
                "belt_plane_residuals": "belt_plane_residuals.png",
                "plane_signed_distance_values": "plane_signed_distance.values.f32",
                "plane_residual_histogram": "plane_residual_histogram.json",
            }
        )
        context.set_artifact("files", files)

        context.add_processing_artifact(
            artifact_id="plane_residual_histogram",
            stage_id="detect_belt_plane",
            kind="json",
            title="Plane residual histogram",
            path="plane_residual_histogram.json",
            mime_type="application/json",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name, **residual_hist_payload},
        )
        context.add_processing_artifact(
            artifact_id="belt_plane",
            stage_id="detect_belt_plane",
            kind="json",
            title="Belt plane model",
            path="belt_plane.json",
            mime_type="application/json",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name, "plane_coefficients": [float(x) for x in coeffs.tolist()], "residual_stats_mm": stats},
        )
        context.add_processing_artifact(
            artifact_id="belt_plane_residuals",
            stage_id="detect_belt_plane",
            kind="image",
            title="Belt plane residuals",
            path="belt_plane_residuals.png",
            mime_type="image/png",
            preview_available=True,
            metadata={
                **_display_only_meta(),
                **_height_semantic_meta(
                    semantic_field="plane_signed_distance",
                    semantic_label="Plane signed distance",
                    units="mm",
                    value_min=float(-stats.get("max_abs_mm", 0.0) if isinstance(stats, dict) else 0.0),
                    value_max=float(stats.get("max_abs_mm", 0.0) if isinstance(stats, dict) else 0.0),
                    color_scale_min=0.0,
                    color_scale_max=float(stats.get("max_abs_mm", 0.0) if isinstance(stats, dict) else 0.0),
                    color_map="turbo",
                    positive_direction="higher_value",
                    is_measurement_authoritative=False,
                    source_artifact_id="heightmap_frame.npz:z_mm",
                    stage_id="detect_belt_plane",
                ),
                "source": "native_pipeline",
                "producer": self.name,
                "residual_stats_mm": stats,
                "min_mm": 0.0,
                "max_mm": float(stats.get("max_abs_mm", 0.0) if isinstance(stats, dict) else 0.0),
                "units": "mm",
                "display_semantic": "plane_residual_mm",
                "display_units": "mm",
                "display_range": {
                    "min": 0.0,
                    "max": float(stats.get("max_abs_mm", 0.0) if isinstance(stats, dict) else 0.0),
                },
                "raw_semantic": "raw_sensor_z_mm",
                "semantic_source_id": "heightmap_frame.npz:z_mm",
                **build_height_transform_metadata(
                    semantic_field="plane_signed_distance",
                    derived_from="raw_sensor_z",
                    transform_type="signed_distance",
                    plane_id="plane_001",
                    units="mm",
                ),
            },
        )
        context.add_processing_artifact(
            artifact_id="plane_signed_distance_raster",
            stage_id="detect_belt_plane",
            kind="file",
            title="Plane signed distance semantic raster",
            path=residual_raster_name,
            mime_type="application/octet-stream",
            preview_available=False,
            metadata=_semantic_raster_meta(
                semantic_field="plane_signed_distance",
                source_stage="detect_belt_plane",
                coordinate_space="belt_xy_pixels",
                derived_from="raw_sensor_z",
                transform_type="signed_distance",
                plane_id="plane_001",
            ),
        )
        context.add_processing_artifact(
            artifact_id="belt_plane_overlay",
            stage_id="detect_belt_plane",
            kind="image",
            title="Belt plane overlay",
            path="belt_plane_overlay.png",
            mime_type="image/png",
            preview_available=True,
            metadata={
                "source": "native_pipeline",
                "producer": self.name,
                "roi_enabled": roi_enabled,
                "roi_x": roi_x,
                "roi_y": roi_y,
                "roi_width": roi_w,
                "roi_height": roi_h,
                "roi_area_ratio": roi_area_ratio,
            },
        )
        context.add_processing_artifact(
            artifact_id="raw_heightmap_preview",
            stage_id="detect_belt_plane",
            kind="image",
            title="Raw heightmap preview",
            path="raw_heightmap_preview.png",
            mime_type="image/png",
            preview_available=True,
            metadata={
                **_display_only_meta(),
                **_height_semantic_meta(
                    semantic_field="raw_sensor_z",
                    semantic_label="Raw sensor Z",
                    units="mm",
                    value_min=raw_min_mm,
                    value_max=raw_max_mm,
                    color_scale_min=raw_min_mm,
                    color_scale_max=raw_max_mm,
                    color_map="turbo",
                    positive_direction="higher_value",
                    is_measurement_authoritative=False,
                    source_artifact_id="heightmap_frame.npz:z_mm",
                    stage_id="detect_belt_plane",
                ),
                "source": "native_pipeline",
                "producer": self.name,
                "min_mm": raw_min_mm,
                "max_mm": raw_max_mm,
                "units": "mm",
                "display_semantic": "raw_sensor_z_mm",
                "display_units": "mm",
                "display_range": {"min": raw_min_mm, "max": raw_max_mm},
                "raw_semantic": "raw_sensor_z_mm",
                "semantic_source_id": "heightmap_frame.npz:z_mm",
                **build_height_transform_metadata(
                    semantic_field="raw_sensor_z",
                    derived_from="sensor_heightmap_decode",
                    transform_type="raw_decode",
                    units="mm",
                ),
            },
        )
        context.add_processing_artifact(
            artifact_id="valid_mask",
            stage_id="detect_belt_plane",
            kind="image",
            title="Valid mask",
            path="valid_mask.png",
            mime_type="image/png",
            preview_available=True,
            metadata=_display_only_meta({"source": "native_pipeline", "producer": self.name, "numeric_source_artifact": "normalized_heightmap.npz"}),
        )
        context.add_processing_artifact(
            artifact_id="plane_inlier_mask",
            stage_id="detect_belt_plane",
            kind="image",
            title="Plane inlier mask",
            path="plane_inlier_mask.png",
            mime_type="image/png",
            preview_available=True,
            metadata=_display_only_meta({"source": "native_pipeline", "producer": self.name, "numeric_source_artifact": "normalized_heightmap.npz"}),
        )
        context.add_processing_artifact(
            artifact_id="background_candidate_mask",
            stage_id="detect_belt_plane",
            kind="image",
            title="Background candidate mask",
            path="background_candidate_mask.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name, **background_selection_debug},
        )
        context.add_processing_artifact(
            artifact_id="background_seed_mask",
            stage_id="detect_belt_plane",
            kind="image",
            title="Background seed mask",
            path="background_seed_mask.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name, **background_selection_debug},
        )
        context.add_processing_artifact(
            artifact_id="expanded_plane_mask",
            stage_id="detect_belt_plane",
            kind="image",
            title="Expanded plane mask",
            path="expanded_plane_mask.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name, **background_selection_debug},
        )
        context.add_processing_artifact(
            artifact_id="final_plane_inlier_mask",
            stage_id="detect_belt_plane",
            kind="image",
            title="Final plane inlier mask",
            path="final_plane_inlier_mask.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name, **background_selection_debug},
        )
        context.add_processing_artifact(
            artifact_id="depth_gradient_magnitude",
            stage_id="detect_belt_plane",
            kind="image",
            title="Depth gradient magnitude",
            path="depth_gradient_magnitude.png",
            mime_type="image/png",
            preview_available=True,
            metadata=_display_only_meta({"source": "native_pipeline", "producer": self.name, "numeric_source_artifact": "normalized_heightmap.npz"}),
        )
        context.add_processing_artifact(
            artifact_id="low_gradient_mask",
            stage_id="detect_belt_plane",
            kind="image",
            title="Low-gradient mask",
            path="low_gradient_mask.png",
            mime_type="image/png",
            preview_available=True,
            metadata=_display_only_meta({"source": "native_pipeline", "producer": self.name, "numeric_source_artifact": "normalized_heightmap.npz"}),
        )
        context.add_processing_artifact(
            artifact_id="low_gradient_components_overlay",
            stage_id="detect_belt_plane",
            kind="image",
            title="Low-gradient components overlay",
            path="low_gradient_components_overlay.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
        )
        context.add_processing_artifact(
            artifact_id="low_gradient_components",
            stage_id="detect_belt_plane",
            kind="json",
            title="Low-gradient components",
            path="low_gradient_components.json",
            mime_type="application/json",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
        )
        context.add_processing_artifact(
            artifact_id="reference_surface_selected_mask",
            stage_id="detect_belt_plane",
            kind="image",
            title="Selected reference-surface mask",
            path="reference_surface_selected_mask.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
        )
        context.add_processing_artifact(
            artifact_id="reference_surface_candidates",
            stage_id="detect_belt_plane",
            kind="json",
            title="Reference-surface candidates",
            path="reference_surface_candidates.json",
            mime_type="application/json",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
        )
        context.add_processing_artifact(
            artifact_id="gradient_debug",
            stage_id="detect_belt_plane",
            kind="json",
            title="Gradient debug",
            path="gradient_debug.json",
            mime_type="application/json",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
        )
        context.add_processing_artifact(
            artifact_id="background_depth_histogram",
            stage_id="detect_belt_plane",
            kind="json",
            title="Background depth histogram",
            path="background_depth_histogram.json",
            mime_type="application/json",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name, **z_thresholds},
        )
        context.add_processing_artifact(
            artifact_id="background_depth_plot",
            stage_id="detect_belt_plane",
            kind="image",
            title="Background depth plot",
            path="background_depth_plot.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name, **z_thresholds},
        )
        context.add_processing_artifact(
            artifact_id="selected_surface_debug",
            stage_id="detect_belt_plane",
            kind="json",
            title="Selected surface debug",
            path="selected_surface_debug.json",
            mime_type="application/json",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name, **selected_surface_debug},
        )
        context.add_processing_artifact(
            artifact_id="background_selection_debug",
            stage_id="detect_belt_plane",
            kind="json",
            title="Background selection debug",
            path="background_selection_debug.json",
            mime_type="application/json",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name, **background_selection_debug},
        )
        context.add_processing_artifact(
            artifact_id="plane_fit_roi_mask",
            stage_id="detect_belt_plane",
            kind="image",
            title="Plane fit ROI mask",
            path="plane_fit_roi_mask.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
        )
        context.add_processing_artifact(
            artifact_id="plane_fit_debug",
            stage_id="detect_belt_plane",
            kind="json",
            title="Plane fit debug",
            path="plane_fit_debug.json",
            mime_type="application/json",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name, **debug_payload},
        )


@dataclass
class NormalizeHeightsToPlaneStage:
    normalized_clip_negative: bool = False
    normalized_negative_tolerance_mm: float = 0.0
    normalized_height_sign: str = "auto"
    normalization_background_p95_warning_mm: float = 3.5
    normalization_fg_bg_separation_warning_mm: float = 3.0
    normalization_min_inlier_ratio_warning: float = 0.2
    below_reference_tolerance_mm: float = 0.0
    name: str = "NormalizeHeightsToPlane"
    category: str = "calibration"
    required_modalities: tuple[CaptureModality, ...] = ("heightmap",)
    produced_modalities: tuple[CaptureModality, ...] = ("heightmap",)
    produced_artifacts: tuple[str, ...] = ("normalized_heightmap",)
    stage_category: str | None = "calibration"

    def run(self, context: PipelineContext) -> None:
        frame: HeightmapFrame = context.require_artifact("heightmap_frame")
        output_dir: Path = context.require_artifact("output_dir")
        ref_model = context.get_artifact("reference_surface_model", {}) or {}
        coeffs = np.asarray(context.require_artifact("belt_plane"), dtype=np.float64)
        model_type = str(ref_model.get("type") or "plane")
        if model_type == "constant_z":
            z0 = float(ref_model.get("constant_z_mm") if ref_model.get("constant_z_mm") is not None else -coeffs[3])
            normalized = frame.z_mm.astype(np.float32) - np.float32(z0)
            convention = "height_above_reference_mm = raw_z_mm - constant_z_mm"
        else:
            normalized = _residual_map(frame, coeffs)
            convention = "height_above_reference_mm = raw_z_mm - plane_z_at_xy"
        plane_mask = np.asarray(context.get_artifact("plane_inlier_mask", np.zeros_like(frame.valid_mask, dtype=bool)), dtype=bool)
        bg_probe = normalized[plane_mask & frame.valid_mask]
        fg_probe = normalized[(~plane_mask) & frame.valid_mask]
        if self.normalized_height_sign in {"auto", "positive_objects"} and fg_probe.size and bg_probe.size and float(np.mean(fg_probe)) < float(np.mean(bg_probe)):
            normalized = -normalized
            convention = f"{convention}; sign_inverted"

        normalized[~frame.valid_mask] = 0.0
        if self.normalized_clip_negative:
            normalized = np.where(normalized < -abs(self.normalized_negative_tolerance_mm), 0.0, normalized)
        context.set_artifact("normalized_heightmap_mm", normalized)
        context.set_artifact("height_processing_semantic_field", "height_above_belt")
        object_min_height_mm = float((context.get_artifact("stage_params.remove_belt_segment_objects", {}) or {}).get("min_height_mm", 8.0))
        below_reference_mask = frame.valid_mask & (normalized <= float(self.below_reference_tolerance_mm))
        above_threshold_mask = frame.valid_mask & (normalized > object_min_height_mm)
        cv2.imwrite(str(output_dir / "below_reference_mask.png"), (below_reference_mask.astype(np.uint8) * 255))
        cv2.imwrite(str(output_dir / "above_threshold_mask.png"), (above_threshold_mask.astype(np.uint8) * 255))

        norm_frame = HeightmapFrame(
            z_mm=normalized.astype(np.float32),
            valid_mask=frame.valid_mask.copy(),
            reflectance=frame.reflectance,
            x_resolution_mm=frame.x_resolution_mm,
            y_resolution_mm=frame.y_resolution_mm,
            origin_x_mm=frame.origin_x_mm,
            origin_y_mm=frame.origin_y_mm,
            coordinate_system="belt_plane_normalized_mm",
        )
        save_heightmap_npz(norm_frame, output_dir / "normalized_heightmap.npz")
        valid_values = normalized[frame.valid_mask]
        p1 = float(np.percentile(valid_values, 1)) if valid_values.size else 0.0
        p2 = float(np.percentile(valid_values, 2)) if valid_values.size else 0.0
        p5 = float(np.percentile(valid_values, 5)) if valid_values.size else 0.0
        p50 = float(np.percentile(valid_values, 50)) if valid_values.size else 0.0
        p95 = float(np.percentile(valid_values, 95)) if valid_values.size else 0.0
        p98 = float(np.percentile(valid_values, 98)) if valid_values.size else 0.0
        p99 = float(np.percentile(valid_values, 99)) if valid_values.size else 0.0
        display_vmin = p2
        display_vmax = p98 if p98 > p2 else (p2 + 1.0)
        write_heightmap_preview_png(
            normalized,
            frame.valid_mask,
            output_dir / "normalized_heightmap.png",
            vmin=display_vmin,
            vmax=display_vmax,
            clipping_percentiles=(2.0, 98.0),
        )
        write_heightmap_metadata_json(norm_frame, output_dir / "normalized_heightmap.metadata.json")
        # Canonical numeric hover/debug sources (never infer values from colorized PNG).
        normalized.astype(np.float32).tofile(output_dir / "height_above_belt.values.f32")
        frame.z_mm.astype(np.float32).tofile(output_dir / "raw_sensor_z.values.f32")
        # Backward-compatible aliases.
        normalized.astype(np.float32).tofile(output_dir / "normalized_heightmap.values.f32")
        frame.z_mm.astype(np.float32).tofile(output_dir / "raw_heightmap.values.f32")
        frame.valid_mask.astype(np.uint8).tofile(output_dir / "valid_mask.u8")
        valid_u8 = frame.valid_mask.astype(np.uint8)
        invalid_holes = int(np.count_nonzero((valid_u8 == 0) & (normalized != 0.0)))
        print(
            "[25d-hover-debug] wrote normalized hover rasters "
            f"take_dir={output_dir.name} "
            f"shape={tuple(int(v) for v in normalized.shape)} "
            f"normalized_count={int(normalized.size)} "
            f"raw_count={int(frame.z_mm.size)} "
            f"valid_mask_count={int(valid_u8.size)} "
            f"valid_count={int(np.count_nonzero(frame.valid_mask))} "
            f"invalid_count={int(frame.valid_mask.size - np.count_nonzero(frame.valid_mask))} "
            f"invalid_nonzero_normalized_count={invalid_holes} "
            f"normalized_minmax=({float(np.nanmin(normalized)):.6g},{float(np.nanmax(normalized)):.6g}) "
            f"raw_minmax=({float(np.nanmin(frame.z_mm)):.6g},{float(np.nanmax(frame.z_mm)):.6g}) "
            f"files=(height_above_belt.values.f32,raw_sensor_z.values.f32,valid_mask.u8)",
            flush=True,
        )
        bg_values = normalized[plane_mask & frame.valid_mask]
        fg_values = normalized[(~plane_mask) & frame.valid_mask]
        norm_min_mm = float(np.min(valid_values)) if valid_values.size else 0.0
        norm_max_mm = float(np.max(valid_values)) if valid_values.size else 0.0
        hist_payload = {
            "bin_edges_mm": [],
            "bin_counts": [],
            "sample_count": int(valid_values.size),
            "min_mm": norm_min_mm,
            "max_mm": norm_max_mm,
            "mean_mm": float(np.mean(valid_values)) if valid_values.size else 0.0,
            "median_mm": float(np.median(valid_values)) if valid_values.size else 0.0,
            "normalization_convention": convention,
            "clip_negative": bool(self.normalized_clip_negative),
            "negative_tolerance_mm": float(self.normalized_negative_tolerance_mm),
        }
        if valid_values.size:
            counts, edges = np.histogram(valid_values, bins=32)
            hist_payload["bin_counts"] = counts.astype(int).tolist()
            hist_payload["bin_edges_mm"] = [float(x) for x in edges.tolist()]
        render_context_color_mapping = build_height_color_mapping_block(
            semantic_field="height_above_belt",
            units="mm",
            value_min=norm_min_mm,
            value_max=norm_max_mm,
            color_scale_min=float(display_vmin),
            color_scale_max=float(display_vmax),
            color_map="turbo",
            direction="higher_is_hotter",
            source="render_context",
        )
        render_context = {
            "render_vmin": float(display_vmin),
            "render_vmax": float(display_vmax),
            "value_min": float(norm_min_mm),
            "value_max": float(norm_max_mm),
            "semantic_field": "height_above_belt",
            "units": "mm",
            "clipping_mode": "percentile_clipped",
            "clipping_percentiles": [2.0, 98.0],
            "auto_range_mode": "percentile",
            "semantic_range_mode": "height_above_belt",
            "colormap_id": "turbo",
            "direction": "higher_is_hotter",
            "clamp": True,
            "interpolation_mode": "nearest",
            "roi_purpose": "plane_fit_only",
            "render_coordinate_space": "full_frame",
            "source_shape": [int(normalized.shape[0]), int(normalized.shape[1])],
            "rendered_png_shape": [int(normalized.shape[0]), int(normalized.shape[1])],
            "numeric_array_shape": [int(normalized.shape[0]), int(normalized.shape[1])],
            "valid_mask_shape": [int(frame.valid_mask.shape[0]), int(frame.valid_mask.shape[1])],
            "numeric_values_path": "height_above_belt.values.f32",
            "raw_values_path": "raw_sensor_z.values.f32",
            "valid_mask_path": "valid_mask.u8",
            "source_artifact_id": "normalized_heightmap.npz:z_mm",
            "color_mapping": render_context_color_mapping,
        }
        (output_dir / "normalized_heightmap.render_context.json").write_text(json.dumps(render_context, indent=2), encoding="utf-8")
        display_meta = {
            "numeric_source_artifact": "normalized_heightmap.npz",
            "display_source_field": "height_above_belt",
            "units": "mm",
            "display_semantic": "height_above_belt",
            "display_semantic_label": "Height above belt",
            "raw_semantic": "raw_sensor_z",
            "raw_units": "mm",
            "shape": [int(normalized.shape[0]), int(normalized.shape[1])],
            "valid_count": int(np.count_nonzero(frame.valid_mask)),
            "invalid_count": int(frame.valid_mask.size - np.count_nonzero(frame.valid_mask)),
            "raw_min": norm_min_mm,
            "raw_max": norm_max_mm,
            "p1": p1,
            "p2": p2,
            "p5": p5,
            "p50": p50,
            "p95": p95,
            "p98": p98,
            "p99": p99,
            "display_vmin": float(display_vmin),
            "display_vmax": float(display_vmax),
            "display_scaling": "percentile_clipped",
            "display_percentiles": [2, 98],
            "display_range": {"min": float(display_vmin), "max": float(display_vmax)},
            "colormap": "turbo",
            "interpolation_mode": "nearest",
            "roi_purpose": "plane_fit_only",
            "render_coordinate_space": "full_frame",
            "source_shape": [int(normalized.shape[0]), int(normalized.shape[1])],
            "rendered_png_shape": [int(normalized.shape[0]), int(normalized.shape[1])],
            "numeric_array_shape": [int(normalized.shape[0]), int(normalized.shape[1])],
            "valid_mask_shape": [int(frame.valid_mask.shape[0]), int(frame.valid_mask.shape[1])],
            "invalid_color": "black",
            "zero_reference_mm": 0.0,
            "numeric_values_path": "height_above_belt.values.f32",
            "raw_values_path": "raw_sensor_z.values.f32",
            "valid_mask_path": "valid_mask.u8",
            "semantic_source_id": "normalized_heightmap.npz:z_mm",
            "render_context_path": "normalized_heightmap.render_context.json",
            "render_context": render_context,
            "color_mapping": build_height_color_mapping_block(
                semantic_field="height_above_belt",
                units="mm",
                value_min=norm_min_mm,
                value_max=norm_max_mm,
                color_scale_min=float(display_vmin),
                color_scale_max=float(display_vmax),
                color_map="turbo",
                direction="higher_is_hotter",
                source="artifact_metadata",
            ),
        }
        (output_dir / "normalized_heightmap_display.json").write_text(json.dumps(display_meta, indent=2), encoding="utf-8")
        (output_dir / "normalized_height_histogram.json").write_text(json.dumps(hist_payload, indent=2), encoding="utf-8")
        norm_debug = {
            "semantic_source_id": "normalized_heightmap.npz:z_mm",
            "display_semantic": "height_above_plane_mm",
            "raw_semantic": "raw_sensor_z_mm",
            "normalization_convention": convention,
            "normalization_formula": convention,
            "model_type": model_type,
            "positive_object_direction": "positive",
            "clip_negative": bool(self.normalized_clip_negative),
            "negative_tolerance_mm": float(self.normalized_negative_tolerance_mm),
            "background_height_mean_after_normalization_mm": float(np.mean(bg_values)) if bg_values.size else 0.0,
            "background_height_std_after_normalization_mm": float(np.std(bg_values)) if bg_values.size else 0.0,
            "background_height_p95_abs_after_normalization_mm": float(np.percentile(np.abs(bg_values), 95)) if bg_values.size else 0.0,
            "foreground_mean_height_mm": float(np.mean(fg_values)) if fg_values.size else 0.0,
            "below_reference_pixel_count": int(np.count_nonzero(below_reference_mask)),
            "above_threshold_pixel_count": int(np.count_nonzero(above_threshold_mask)),
            "valid_ratio": float(np.count_nonzero(frame.valid_mask) / max(1, frame.valid_mask.size)),
            "display_vmin": float(display_meta["display_vmin"]),
            "display_vmax": float(display_meta["display_vmax"]),
            "display_percentiles": display_meta["display_percentiles"],
            "display_scaling": display_meta["display_scaling"],
        }
        norm_warnings: list[str] = []
        if norm_debug["background_height_p95_abs_after_normalization_mm"] > float(self.normalization_background_p95_warning_mm):
            norm_warnings.append("background_not_near_zero_after_normalization")
        if (norm_debug["foreground_mean_height_mm"] - abs(norm_debug["background_height_mean_after_normalization_mm"])) < float(self.normalization_fg_bg_separation_warning_mm):
            norm_warnings.append("foreground_background_separation_too_small_after_normalization")
        plane_stats = context.get_artifact("belt_plane_stats", {}) or {}
        try:
            inlier_ratio = float((plane_stats.get("inlier_ratio") if isinstance(plane_stats, dict) else 0.0) or 0.0)
        except Exception:
            inlier_ratio = 0.0
        if inlier_ratio < float(self.normalization_min_inlier_ratio_warning):
            norm_warnings.append("plane_inlier_ratio_low")
        norm_debug["warnings"] = norm_warnings
        plane_fit_status = str(context.get_artifact("plane_fit_status", "unknown"))
        belt_plane_stats = context.get_artifact("belt_plane_stats", {}) or {}
        normalization_quality = {
            "status": "ok",
            "reason": None,
            "plane_fit_status": plane_fit_status,
            "plane_fit_inlier_ratio": float((belt_plane_stats.get("inlier_ratio") or 0.0) if isinstance(belt_plane_stats, dict) else 0.0),
            "plane_fit_residual_p95_mm": float((belt_plane_stats.get("p95_abs_mm") or 0.0) if isinstance(belt_plane_stats, dict) else 0.0),
            "warning": None,
        }
        if model_type == "constant_z" and plane_fit_status != "success":
            normalization_quality = {
                "status": "degraded",
                "reason": "plane_fit_failed_constant_z_fallback",
                "plane_fit_status": plane_fit_status,
                "plane_fit_inlier_ratio": float((belt_plane_stats.get("inlier_ratio") or 0.0) if isinstance(belt_plane_stats, dict) else 0.0),
                "plane_fit_residual_p95_mm": float((belt_plane_stats.get("p95_abs_mm") or 0.0) if isinstance(belt_plane_stats, dict) else 0.0),
                "warning": "Normalized height is fallback-based; belt may not be near 0 mm.",
            }
            norm_warnings.append("normalized_height_degraded_constant_z_fallback")
            norm_debug["warnings"] = norm_warnings
        norm_debug["normalization_quality"] = normalization_quality
        (output_dir / "normalization_debug.json").write_text(json.dumps(norm_debug, indent=2), encoding="utf-8")

        files = dict(context.get_artifact("files", {}))
        files.update(
            {
                "normalized_heightmap": "normalized_heightmap.npz",
                "debug_height": "normalized_heightmap.png",
                "height_above_belt_values": "height_above_belt.values.f32",
                "raw_sensor_z_values": "raw_sensor_z.values.f32",
                "normalized_height_histogram": "normalized_height_histogram.json",
                "normalization_debug": "normalization_debug.json",
                "normalized_heightmap_display": "normalized_heightmap_display.json",
                "normalized_heightmap_render_context": "normalized_heightmap.render_context.json",
                "below_reference_mask": "below_reference_mask.png",
                "above_threshold_mask": "above_threshold_mask.png",
            }
        )
        context.set_artifact("files", files)

        context.add_processing_artifact(
            artifact_id="normalized_heightmap",
            stage_id="normalize_heights_to_plane",
            kind="image",
            title="Normalized heightmap",
            path="normalized_heightmap.png",
            mime_type="image/png",
            preview_available=True,
            projection_type="heightmap",
            metadata={
                **_display_only_meta(),
                **_height_semantic_meta(
                    semantic_field="height_above_belt",
                    semantic_label="Height above belt",
                    units="mm",
                    value_min=norm_min_mm,
                    value_max=norm_max_mm,
                    color_scale_min=float(display_meta["display_vmin"]),
                    color_scale_max=float(display_meta["display_vmax"]),
                    color_map="turbo",
                    positive_direction="higher_above_belt",
                    is_measurement_authoritative=True,
                    source_artifact_id="normalized_heightmap.npz:z_mm",
                    stage_id="normalize_heights_to_plane",
                ),
                "source": "native_pipeline",
                "producer": self.name,
                "semantic": "height_above_belt_mm",
                "display_semantic": "height_above_belt",
                "raw_semantic": "raw_sensor_z",
                "normalization_convention": convention,
                "reference_surface_model_type": model_type,
                "min_mm": norm_min_mm,
                "max_mm": norm_max_mm,
                "units": "mm",
                "display_range": {
                    "min": float(display_meta["display_vmin"]),
                    "max": float(display_meta["display_vmax"]),
                },
                "semantic_source_id": "normalized_heightmap.npz:z_mm",
                "display_metadata_path": "normalized_heightmap_display.json",
                "render_context_path": "normalized_heightmap.render_context.json",
                "render_vmin": float(render_context["render_vmin"]),
                "render_vmax": float(render_context["render_vmax"]),
                "render_colormap_id": str(render_context["colormap_id"]),
                "render_interpolation_mode": str(render_context["interpolation_mode"]),
                "normalization_quality": norm_debug.get("normalization_quality"),
                **build_height_transform_metadata(
                    semantic_field="height_above_belt",
                    derived_from="plane_signed_distance",
                    transform_type="invert_signed_distance",
                    plane_id="plane_001",
                    units="mm",
                ),
            },
        )
        context.add_processing_artifact(
            artifact_id="height_above_belt_raster",
            stage_id="normalize_heights_to_plane",
            kind="file",
            title="Height above belt semantic raster",
            path="height_above_belt.values.f32",
            mime_type="application/octet-stream",
            preview_available=False,
            metadata=_semantic_raster_meta(
                semantic_field="height_above_belt",
                source_stage="normalize_heights_to_plane",
                coordinate_space="belt_xy_pixels",
                derived_from="plane_signed_distance",
                transform_type="invert_signed_distance",
                plane_id="plane_001",
                extras={
                    "color_mapping": build_height_color_mapping_block(
                        semantic_field="height_above_belt",
                        units="mm",
                        value_min=norm_min_mm,
                        value_max=norm_max_mm,
                        color_scale_min=float(display_vmin),
                        color_scale_max=float(display_vmax),
                        color_map="turbo",
                        direction="higher_is_hotter",
                        source="artifact_metadata",
                    ),
                },
            ),
        )
        context.add_processing_artifact(
            artifact_id="normalized_heightmap_display",
            stage_id="normalize_heights_to_plane",
            kind="json",
            title="Normalized height display transform",
            path="normalized_heightmap_display.json",
            mime_type="application/json",
            preview_available=True,
            metadata=_numeric_source_meta(
                {
                    "source": "native_pipeline",
                    "producer": self.name,
                    **_height_semantic_meta(
                        semantic_field="preview_normalized",
                        semantic_label="Preview normalized",
                        units="mm",
                        value_min=norm_min_mm,
                        value_max=norm_max_mm,
                        color_scale_min=float(display_meta["display_vmin"]),
                        color_scale_max=float(display_meta["display_vmax"]),
                        color_map="turbo",
                        positive_direction="higher_value",
                        is_measurement_authoritative=False,
                        source_artifact_id="normalized_heightmap.npz:z_mm",
                        stage_id="normalize_heights_to_plane",
                    ),
                    **build_height_transform_metadata(
                        semantic_field="preview_normalized",
                        derived_from="height_above_belt",
                        transform_type="normalization",
                        units="mm",
                    ),
                    **display_meta,
                }
            ),
        )
        context.add_processing_artifact(
            artifact_id="normalized_heightmap_render_context",
            stage_id="normalize_heights_to_plane",
            kind="json",
            title="Normalized height render context",
            path="normalized_heightmap.render_context.json",
            mime_type="application/json",
            preview_available=True,
            metadata=_numeric_source_meta(
                {
                    "source": "native_pipeline",
                    "producer": self.name,
                    "render_context_for": "normalized_heightmap.png",
                    **render_context,
                    "display_metadata_path": "normalized_heightmap_display.json",
                }
            ),
        )
        context.add_processing_artifact(
            artifact_id="below_reference_mask",
            stage_id="normalize_heights_to_plane",
            kind="image",
            title="Pixels below or on reference",
            path="below_reference_mask.png",
            mime_type="image/png",
            preview_available=True,
            metadata=_display_only_meta({"source": "native_pipeline", "producer": self.name}),
        )
        context.add_processing_artifact(
            artifact_id="above_threshold_mask",
            stage_id="normalize_heights_to_plane",
            kind="image",
            title="Pixels above object threshold",
            path="above_threshold_mask.png",
            mime_type="image/png",
            preview_available=True,
            metadata=_display_only_meta({"source": "native_pipeline", "producer": self.name, "object_min_height_mm": object_min_height_mm}),
        )
        context.add_processing_artifact(
            artifact_id="normalized_height_histogram",
            stage_id="normalize_heights_to_plane",
            kind="json",
            title="Normalized height histogram",
            path="normalized_height_histogram.json",
            mime_type="application/json",
            preview_available=True,
            metadata=_numeric_source_meta({"source": "native_pipeline", "producer": self.name, **hist_payload}),
        )
        context.add_processing_artifact(
            artifact_id="normalization_debug",
            stage_id="normalize_heights_to_plane",
            kind="json",
            title="Normalized height debug",
            path="normalization_debug.json",
            mime_type="application/json",
            preview_available=True,
            metadata=_numeric_source_meta({"source": "native_pipeline", "producer": self.name, **norm_debug}),
        )


@dataclass
class RemoveBeltAndSegmentObjectsStage:
    min_height_mm: float = 8.0
    reference_tolerance_mm: float = 0.0
    max_height_mm: float | None = None
    suppress_plane_mask_in_segmentation: bool = True
    ignore_small_residual_background: bool = True
    morphology_kernel: int = 5
    min_component_area: int = 120
    fill_holes: bool = True
    smoothing_kernel: int = 3
    name: str = "RemoveBeltAndSegmentObjects"
    category: str = "segmentation"
    required_modalities: tuple[CaptureModality, ...] = ("heightmap",)
    produced_modalities: tuple[CaptureModality, ...] = ()
    produced_artifacts: tuple[str, ...] = ("height_segmentation",)
    stage_category: str | None = "segmentation"

    def run(self, context: PipelineContext) -> None:
        _assert_not_preview_semantic(context, stage_name=self.name)
        frame: HeightmapFrame = context.require_artifact("heightmap_frame")
        normalized = np.asarray(context.require_artifact("normalized_heightmap_mm"), dtype=np.float32)
        output_dir: Path = context.require_artifact("output_dir")

        below_or_on_reference = frame.valid_mask & (normalized <= float(self.reference_tolerance_mm))
        foreground = normalized > self.min_height_mm
        if self.max_height_mm is not None:
            foreground &= normalized <= self.max_height_mm
        foreground &= frame.valid_mask
        foreground_before_suppression = foreground.copy()
        cv2.imwrite(str(output_dir / "foreground_before_plane_suppression.png"), (foreground_before_suppression.astype(np.uint8) * 255))
        cv2.imwrite(str(output_dir / "below_reference_mask.png"), (below_or_on_reference.astype(np.uint8) * 255))
        cv2.imwrite(str(output_dir / "above_threshold_mask.png"), (foreground_before_suppression.astype(np.uint8) * 255))

        plane_mask = np.asarray(
            context.get_artifact(
                "reference_surface_selected_mask",
                context.get_artifact("expanded_plane_mask", context.get_artifact("plane_inlier_mask", np.zeros_like(frame.valid_mask, dtype=bool))),
            ),
            dtype=bool,
        )
        suppressed_plane_pixels = foreground & plane_mask
        if self.suppress_plane_mask_in_segmentation:
            foreground = foreground & (~plane_mask)
        cv2.imwrite(str(output_dir / "plane_suppressed_mask.png"), (foreground.astype(np.uint8) * 255))
        cv2.imwrite(str(output_dir / "rejected_background_residuals.png"), (suppressed_plane_pixels.astype(np.uint8) * 255))

        threshold_mask = (foreground.astype(np.uint8) * 255)
        cv2.imwrite(str(output_dir / "normalized_height_threshold_mask.png"), threshold_mask)

        k = max(1, int(self.morphology_kernel))
        kernel = np.ones((k, k), np.uint8)
        cleaned = cv2.morphologyEx(threshold_mask, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        if self.fill_holes:
            flood = cleaned.copy()
            h, w = flood.shape
            flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
            cv2.floodFill(flood, flood_mask, (0, 0), 255)
            cleaned = cleaned | cv2.bitwise_not(flood)
        if self.smoothing_kernel > 1:
            ks = int(self.smoothing_kernel)
            if ks % 2 == 0:
                ks += 1
            cleaned = cv2.GaussianBlur(cleaned, (ks, ks), 0)
            cleaned = np.where(cleaned > 127, 255, 0).astype(np.uint8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        filtered = np.zeros_like(cleaned)
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= self.min_component_area:
                filtered[labels == label] = 255

        cv2.imwrite(str(output_dir / "cleaned_object_mask.png"), filtered)
        cv2.imwrite(str(output_dir / "final_object_mask.png"), filtered)
        seg_overlay = _overlay_mask_on_heightmap(normalized, frame.valid_mask, filtered > 0)
        cv2.imwrite(str(output_dir / "segmentation_overlay.png"), seg_overlay)
        cv2.imwrite(str(output_dir / "connected_components_overlay.png"), seg_overlay)


        num_labels_before, _, _, _ = cv2.connectedComponentsWithStats((foreground_before_suppression.astype(np.uint8) * 255), connectivity=8)
        num_labels_after_suppress, _, _, _ = cv2.connectedComponentsWithStats((foreground.astype(np.uint8) * 255), connectivity=8)
        seg_debug = {
            "foreground_before_plane_suppression_percent": float(np.count_nonzero(foreground_before_suppression) / max(1, filtered.size) * 100.0),
            "foreground_after_plane_suppression_percent": float(np.count_nonzero(foreground) / max(1, filtered.size) * 100.0),
            "foreground_coverage_percent": float(np.count_nonzero(filtered) / max(1, filtered.size) * 100.0),
            "threshold_config": {
                "min_height_mm": self.min_height_mm,
                "max_height_mm": self.max_height_mm,
                "suppress_plane_mask_in_segmentation": self.suppress_plane_mask_in_segmentation,
                "ignore_small_residual_background": self.ignore_small_residual_background,
                "reference_tolerance_mm": self.reference_tolerance_mm,
                "morphology_kernel": self.morphology_kernel,
                "min_component_area": self.min_component_area,
                "fill_holes": self.fill_holes,
                "smoothing_kernel": self.smoothing_kernel,
            },
            "plane_suppressed_pixel_count": int(np.count_nonzero(suppressed_plane_pixels)),
            "below_or_on_reference_pixel_count": int(np.count_nonzero(below_or_on_reference)),
            "connected_components_before_suppression": int(max(num_labels_before - 1, 0)),
            "connected_components_after_suppression": int(max(num_labels_after_suppress - 1, 0)),
            "connected_component_count": int(max(num_labels - 1, 0)),
        }
        seg_warnings: list[str] = []
        fg_after = float(seg_debug["foreground_after_plane_suppression_percent"])
        if fg_after <= 0.0:
            seg_warnings.append("foreground_coverage_zero_after_reference_suppression")
        if fg_after > 60.0:
            seg_warnings.append("foreground_coverage_too_high_possible_background_leak")
        seg_debug["warnings"] = seg_warnings
        (output_dir / "segmentation_debug.json").write_text(json.dumps(seg_debug, indent=2), encoding="utf-8")
        context.set_artifact("segmentation_mask", filtered > 0)
        files = dict(context.get_artifact("files", {}))
        files.update(
            {
                "debug_segmentation": "segmentation_overlay.png",
                "height_segmentation_overlay": "connected_components_overlay.png",
                "normalized_height_threshold_mask": "normalized_height_threshold_mask.png",
                "cleaned_object_mask": "cleaned_object_mask.png",
                "foreground_before_plane_suppression": "foreground_before_plane_suppression.png",
                "below_reference_mask": "below_reference_mask.png",
                "above_threshold_mask": "above_threshold_mask.png",
                "plane_suppressed_mask": "plane_suppressed_mask.png",
                "final_object_mask": "final_object_mask.png",
                "rejected_background_residuals": "rejected_background_residuals.png",
            }
        )
        context.set_artifact("files", files)

        context.add_processing_artifact(
            artifact_id="normalized_height_threshold_mask",
            stage_id="remove_belt_segment_objects",
            kind="image",
            title="Normalized-height threshold mask",
            path="normalized_height_threshold_mask.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
        )
        context.add_processing_artifact(
            artifact_id="cleaned_object_mask",
            stage_id="remove_belt_segment_objects",
            kind="image",
            title="Cleaned object mask",
            path="cleaned_object_mask.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
        )
        context.add_processing_artifact(
            artifact_id="foreground_before_plane_suppression",
            stage_id="remove_belt_segment_objects",
            kind="image",
            title="Foreground before plane suppression",
            path="foreground_before_plane_suppression.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
        )
        context.add_processing_artifact(
            artifact_id="below_reference_mask",
            stage_id="remove_belt_segment_objects",
            kind="image",
            title="Below/equal reference mask",
            path="below_reference_mask.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
        )
        context.add_processing_artifact(
            artifact_id="above_threshold_mask",
            stage_id="remove_belt_segment_objects",
            kind="image",
            title="Above object threshold mask",
            path="above_threshold_mask.png",
            mime_type="image/png",
            preview_available=True,
            metadata=_display_only_meta({"source": "native_pipeline", "producer": self.name, "object_min_height_mm": self.min_height_mm, "numeric_source_artifact": "normalized_heightmap.npz"}),
        )
        context.add_processing_artifact(
            artifact_id="plane_suppressed_mask",
            stage_id="remove_belt_segment_objects",
            kind="image",
            title="Plane suppressed mask",
            path="plane_suppressed_mask.png",
            mime_type="image/png",
            preview_available=True,
            metadata=_display_only_meta({"source": "native_pipeline", "producer": self.name, "numeric_source_artifact": "normalized_heightmap.npz"}),
        )
        context.add_processing_artifact(
            artifact_id="final_object_mask",
            stage_id="remove_belt_segment_objects",
            kind="image",
            title="Final object mask",
            path="final_object_mask.png",
            mime_type="image/png",
            preview_available=True,
            metadata=_display_only_meta({"source": "native_pipeline", "producer": self.name, "numeric_source_artifact": "normalized_heightmap.npz"}),
        )
        context.add_processing_artifact(
            artifact_id="rejected_background_residuals",
            stage_id="remove_belt_segment_objects",
            kind="image",
            title="Rejected background residuals",
            path="rejected_background_residuals.png",
            mime_type="image/png",
            preview_available=True,
            metadata=_display_only_meta({"source": "native_pipeline", "producer": self.name, "numeric_source_artifact": "normalized_heightmap.npz"}),
        )

        context.add_processing_artifact(
            artifact_id="segmentation_debug",
            stage_id="remove_belt_segment_objects",
            kind="json",
            title="Segmentation debug",
            path="segmentation_debug.json",
            mime_type="application/json",
            preview_available=True,
            metadata=_numeric_source_meta({"source": "native_pipeline", "producer": self.name, "numeric_source_artifact": "normalized_heightmap.npz", **seg_debug}),
        )
        context.add_processing_artifact(
            artifact_id="height_segmentation",
            stage_id="remove_belt_segment_objects",
            kind="image",
            title="Height segmentation",
            path="segmentation_overlay.png",
            mime_type="image/png",
            preview_available=True,
            metadata={
                **_display_only_meta(),
                "source": "native_pipeline",
                "producer": self.name,
                "numeric_source_artifact": "normalized_heightmap.npz",
                "height_threshold_config": {
                    "min_height_mm": self.min_height_mm,
                    "max_height_mm": self.max_height_mm,
                    "morphology_kernel": self.morphology_kernel,
                    "min_component_area": self.min_component_area,
                    "fill_holes": self.fill_holes,
                    "smoothing_kernel": self.smoothing_kernel,
                },
            },
        )
        context.add_processing_artifact(
            artifact_id="connected_components_overlay",
            stage_id="remove_belt_segment_objects",
            kind="overlay",
            title="Connected components overlay",
            path="connected_components_overlay.png",
            mime_type="image/png",
            preview_available=True,
            overlay_type="segmentation",
            coordinate_space="image_pixel",
            target_artifact_id="height_segmentation",
            source_artifact_ids=["height_segmentation", "cleaned_object_mask"],
            metadata=_display_only_meta({"source": "native_pipeline", "producer": self.name, "semantic": "primary_human_view", "numeric_source_artifact": "normalized_heightmap.npz"}),
        )


@dataclass
class ExtractConnectedComponentsStage:
    name: str = "ExtractConnectedComponents"
    category: str = "segmentation"
    required_modalities: tuple[CaptureModality, ...] = ("heightmap",)
    produced_modalities: tuple[CaptureModality, ...] = ()
    produced_artifacts: tuple[str, ...] = ("components",)
    stage_category: str | None = "segmentation"

    def run(self, context: PipelineContext) -> None:
        _assert_not_preview_semantic(context, stage_name=self.name)
        mask = np.asarray(context.require_artifact("segmentation_mask"), dtype=bool)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        components: list[dict[str, Any]] = []
        for label in range(1, num_labels):
            comp_mask = labels == label
            ys, xs = np.where(comp_mask)
            if xs.size == 0:
                continue
            contour_data = cv2.findContours(comp_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contour_data[0] if len(contour_data) == 2 else contour_data[1]
            contour = max(contours, key=cv2.contourArea) if contours else None
            components.append(
                {
                    "object_id": len(components) + 1,
                    "label": int(label),
                    "mask": comp_mask,
                    "bbox": (
                        int(stats[label, cv2.CC_STAT_LEFT]),
                        int(stats[label, cv2.CC_STAT_TOP]),
                        int(stats[label, cv2.CC_STAT_WIDTH]),
                        int(stats[label, cv2.CC_STAT_HEIGHT]),
                    ),
                    "area_px": int(stats[label, cv2.CC_STAT_AREA]),
                    "contour": contour,
                }
            )
        context.set_artifact("components", components)
        context.metrics["object_count"] = len(components)
        context.add_processing_artifact(
            artifact_id="connected_components",
            stage_id="remove_belt_segment_objects",
            kind="json",
            title="Connected components",
            preview_available=True,
            metadata=_numeric_source_meta(
                {
                    "source": "native_pipeline",
                    "producer": self.name,
                    "component_count": len(components),
                    "numeric_source_artifact": "normalized_heightmap.npz",
                }
            ),
        )


@dataclass
class FitObjectGeometryStage:
    name: str = "FitObjectGeometry"
    category: str = "geometry"
    required_modalities: tuple[CaptureModality, ...] = ("heightmap",)
    produced_modalities: tuple[CaptureModality, ...] = ()
    produced_artifacts: tuple[str, ...] = ("geometry",)
    stage_category: str | None = "geometry"

    def run(self, context: PipelineContext) -> None:
        _assert_not_preview_semantic(context, stage_name=self.name)
        components: list[dict[str, Any]] = context.require_artifact("components")
        frame: HeightmapFrame = context.require_artifact("heightmap_frame")
        pixel_area = frame.pixel_area_mm2()
        for comp in components:
            contour = comp.get("contour")
            if contour is None or len(contour) < 5:
                comp["ellipse"] = None
                comp["major_axis_mm"] = None
                comp["minor_axis_mm"] = None
                comp["eccentricity"] = None
                comp["roundness"] = None
            else:
                ellipse = cv2.fitEllipse(contour)
                (cx, cy), (ax1, ax2), angle = ellipse
                major_px = max(float(ax1), float(ax2))
                minor_px = min(float(ax1), float(ax2))
                comp["ellipse"] = {
                    "cx": float(cx),
                    "cy": float(cy),
                    "major_px": major_px,
                    "minor_px": minor_px,
                    "rotation_deg": float(angle),
                }
                comp["major_axis_mm"] = major_px * frame.x_resolution_mm
                comp["minor_axis_mm"] = minor_px * frame.y_resolution_mm
                comp["eccentricity"] = float(np.sqrt(max(0.0, 1.0 - (minor_px * minor_px) / max(major_px * major_px, 1e-9))))
                comp["roundness"] = float(minor_px / max(major_px, 1e-9))
            hull = cv2.convexHull(contour) if contour is not None and len(contour) >= 3 else None
            comp["convex_hull"] = hull
            comp["footprint_area_mm2"] = float(comp["area_px"] * pixel_area)
        context.set_artifact("components", components)


@dataclass
class ComputeHeightMetricsStage:
    name: str = "ComputeHeightMetrics"
    category: str = "measurement"
    required_modalities: tuple[CaptureModality, ...] = ("heightmap",)
    produced_modalities: tuple[CaptureModality, ...] = ()
    produced_artifacts: tuple[str, ...] = ("objects",)
    stage_category: str | None = "measurement"

    def run(self, context: PipelineContext) -> None:
        _assert_not_preview_semantic(context, stage_name=self.name)
        # Canonical geometry invariant:
        # measurements must be computed from height_above_belt (not raw_sensor_z or preview artifacts).
        semantic_raster = resolve_height_artifact(
            context,
            semantic_field="height_above_belt",
            representation="numeric_raster",
        )
        numeric_source_artifact_id = str((semantic_raster or {}).get("artifact_id") or "height_above_belt_raster")
        components: list[dict[str, Any]] = context.require_artifact("components")
        normalized = np.asarray(context.require_artifact("normalized_heightmap_mm"), dtype=np.float32)
        frame: HeightmapFrame = context.require_artifact("heightmap_frame")
        pixel_area = frame.pixel_area_mm2()
        objects: list[dict[str, Any]] = []
        for comp in components:
            mask = comp["mask"] & frame.valid_mask
            values = normalized[mask]
            if values.size == 0:
                continue
            pos_values = np.maximum(values, 0.0)
            gy, gx = np.gradient(np.where(mask, normalized, 0.0))
            inner_mask = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), dtype=np.uint8), iterations=1).astype(bool)
            if int(np.count_nonzero(inner_mask)) < max(25, int(0.15 * np.count_nonzero(mask))):
                inner_mask = mask
            smoothed_surface = cv2.GaussianBlur(np.where(mask, normalized, 0.0).astype(np.float32), (7, 7), 0)
            surface_residuals = normalized[inner_mask] - smoothed_surface[inner_mask]
            asym = float(abs(np.mean(pos_values[::2]) - np.mean(pos_values[1::2]))) if pos_values.size > 4 else 0.0
            flatness = float(1.0 - np.std(pos_values) / max(np.mean(pos_values), 1e-6))
            # Curvature is the mean absolute slope of the height surface. Restrict to
            # inner_mask so the artificial 0->height step at the segmentation boundary
            # does not dominate the average.
            curvature_gx_inner = float(np.mean(np.abs(gx[inner_mask]))) if int(np.count_nonzero(inner_mask)) > 0 else 0.0
            curvature_gy_inner = float(np.mean(np.abs(gy[inner_mask]))) if int(np.count_nonzero(inner_mask)) > 0 else 0.0
            curvature = float(curvature_gx_inner + curvature_gy_inner)
            roughness = float(1.4826 * np.median(np.abs(surface_residuals - np.median(surface_residuals)))) if surface_residuals.size else 0.0
            volume_proxy = float(np.sum(pos_values) * pixel_area)
            centroid = _mask_centroid(mask)
            max_height = float(np.max(values))
            footprint_x = float(comp.get("major_axis_mm") or comp["bbox"][2] * frame.x_resolution_mm)
            footprint_y = float(comp.get("minor_axis_mm") or comp["bbox"][3] * frame.y_resolution_mm)
            sphere_radius_mm = float(max(footprint_x, footprint_y, max_height) / 2.0)
            sphere_diameter_mm = float(sphere_radius_mm * 2.0)
            truncation_ratio = float(max(0.0, 1.0 - (max_height / max(1e-6, sphere_diameter_mm))))
            object_row = {
                "object_id": int(comp["object_id"]),
                "class_name": "unknown",
                "confidence": None,
                "point_count": int(values.size),
                "center_mm": (
                    round((frame.origin_x_mm + centroid[0] * frame.x_resolution_mm), 4),
                    round((frame.origin_y_mm + centroid[1] * frame.y_resolution_mm), 4),
                    round(float(np.mean(values)), 4),
                ),
                "dimensions_mm": (
                    round(float(comp["bbox"][2] * frame.x_resolution_mm), 4),
                    round(float(comp["bbox"][3] * frame.y_resolution_mm), 4),
                    round(max_height, 4),
                ),
                "diameter_estimate_mm": round(float(max(comp["bbox"][2] * frame.x_resolution_mm, comp["bbox"][3] * frame.y_resolution_mm)), 4),
                "bbox_min_mm": (
                    round(frame.origin_x_mm + comp["bbox"][0] * frame.x_resolution_mm, 4),
                    round(frame.origin_y_mm + comp["bbox"][1] * frame.y_resolution_mm, 4),
                    round(float(np.min(values)), 4),
                ),
                "bbox_max_mm": (
                    round(frame.origin_x_mm + (comp["bbox"][0] + comp["bbox"][2]) * frame.x_resolution_mm, 4),
                    round(frame.origin_y_mm + (comp["bbox"][1] + comp["bbox"][3]) * frame.y_resolution_mm, 4),
                    round(float(np.max(values)), 4),
                ),
                "diameter_mm": round(float(max(comp["major_axis_mm"] or 0.0, comp["minor_axis_mm"] or 0.0)), 4) if comp.get("major_axis_mm") else None,
                "major_axis_mm": round(float(comp.get("major_axis_mm") or 0.0), 4) if comp.get("major_axis_mm") else None,
                "minor_axis_mm": round(float(comp.get("minor_axis_mm") or 0.0), 4) if comp.get("minor_axis_mm") else None,
                "sphericity_score": round(float(comp.get("roundness") or 0.0), 4),
                "fit_rmse_mm": None,
                "height_above_belt_mm": {
                    "max_height_mm": round(max_height, 4),
                    "mean_height_mm": round(float(np.mean(values)), 4),
                    "median_height_mm": round(float(np.median(values)), 4),
                    "p95_height_mm": round(float(np.percentile(values, 95)), 4),
                    "height_std_mm": round(float(np.std(values)), 4),
                },
                "height_metrics_semantic_field": "height_above_belt",
                "fraction_points_inside_belt": 1.0,
                "filter_status": "kept",
                "filter_reason": None,
                "feature_volume_proxy_mm3": round(volume_proxy, 4),
                "feature_flatness": round(flatness, 4),
                "feature_eccentricity": round(float(comp.get("eccentricity") or 0.0), 4),
                # 3D axis balance is defined over the canonical XYZ extents
                # (bbox in raw mm here; later replaced by corrected dim_x/y/z so it
                # stays a true min/max ratio over physical dimensions).
                "feature_sphericity_3d": round(
                    _axis_balance_3d(
                        float(comp["bbox"][2] * frame.x_resolution_mm),
                        float(comp["bbox"][3] * frame.y_resolution_mm),
                        float(max_height),
                    ),
                    4,
                ),
                "feature_edge_roughness": round(roughness, 4),
                "feature_local_curvature_proxy": round(curvature, 4),
                # Per-axis gradient components stored so the curvature proxy can be
                # rescaled into corrected metric space later (scale_z/scale_x for x,
                # scale_z/scale_y for y). Captured in raw heightmap mm-per-pixel units.
                "feature_curvature_components_raw": {
                    "mean_abs_gx_mm_per_mm": round(curvature_gx_inner, 6),
                    "mean_abs_gy_mm_per_mm": round(curvature_gy_inner, 6),
                    "x_resolution_mm": float(frame.x_resolution_mm),
                    "y_resolution_mm": float(frame.y_resolution_mm),
                    "computed_over": "inner_mask" if int(np.count_nonzero(inner_mask)) > 0 else "mask",
                },
                "feature_height_asymmetry": round(asym, 4),
                "feature_footprint_roundness": round(float(comp.get("roundness") or 0.0), 4),
                "footprint_area_mm2": round(float(comp.get("footprint_area_mm2") or 0.0), 4),
                "bbox_px": [
                    int(comp["bbox"][0]),
                    int(comp["bbox"][1]),
                    int(comp["bbox"][2]),
                    int(comp["bbox"][3]),
                ],
                "centroid_px": [round(float(centroid[0]), 3), round(float(centroid[1]), 3)],
                "contour_px": (
                    comp["contour"].reshape(-1, 2).astype(float).tolist()
                    if comp.get("contour") is not None and len(comp.get("contour")) > 0
                    else None
                ),
                "measurement_provenance": {
                    "raw_bbox_px": [int(comp["bbox"][0]), int(comp["bbox"][1]), int(comp["bbox"][2]), int(comp["bbox"][3])],
                    "segmented_bbox_px": [int(comp["bbox"][0]), int(comp["bbox"][1]), int(comp["bbox"][2]), int(comp["bbox"][3])],
                    "segmentation_coverage": float(np.count_nonzero(mask) / max(1, comp["bbox"][2] * comp["bbox"][3])),
                    "removed_points": 0,
                    "morphology_removals": 0,
                    "contour_point_count": int(len(comp.get("contour"))) if comp.get("contour") is not None else 0,
                    "height_point_count": int(values.size),
                },
                "geometry_coordinate_space": "raw_metric_mm",
                "geometry_metric_context": {
                    "x_resolution_mm": float(frame.x_resolution_mm),
                    "y_resolution_mm": float(frame.y_resolution_mm),
                },
                "sphere_fit_diagnostics": {
                    "estimated_center_mm": [round((frame.origin_x_mm + centroid[0] * frame.x_resolution_mm), 4), round((frame.origin_y_mm + centroid[1] * frame.y_resolution_mm), 4), round(float(np.mean(values)), 4)],
                    "estimated_radius_mm": round(sphere_radius_mm, 4),
                    "estimated_diameter_mm": round(sphere_diameter_mm, 4),
                    "fit_rmse_mm": None,
                    "residual_percentiles_mm": {"p50": round(float(np.percentile(values, 50)), 4), "p95": round(float(np.percentile(values, 95)), 4)},
                    "estimated_truncation_ratio": round(truncation_ratio, 4),
                    "fit_confidence": round(float(max(0.0, min(1.0, 1.0 - truncation_ratio))), 4),
                },
            }
            objects.append(object_row)
            oid = object_row["object_id"]
            context.add_processing_artifact(
                artifact_id=f"measurement_object_{oid}",
                stage_id="measurement",
                object_id=oid,
                kind="json",
                title=f"Object #{oid} measurements",
                metadata=_numeric_source_meta(
                    {
                        "source": "native_pipeline",
                        "producer": self.name,
                        "modality": "heightmap",
                        "numeric_source_artifact": numeric_source_artifact_id,
                        "semantic_field": "height_above_belt",
                    }
                ),
            )
        context.set_artifact("objects", objects)


@dataclass
class ValidateKnownObjectScale25DStage:
    name: str = "ValidateKnownObjectScale25D"
    category: str = "measurement"
    required_modalities: tuple[CaptureModality, ...] = ("heightmap",)
    produced_modalities: tuple[CaptureModality, ...] = ()
    produced_artifacts: tuple[str, ...] = ("known_object_scale_validation",)
    stage_category: str | None = "measurement"

    def run(self, context: PipelineContext) -> None:
        metadata = context.get_artifact("metadata", {})
        runtime_cfg = context.get_artifact("stage_params.known_object_25d", {})
        cfg = runtime_cfg if isinstance(runtime_cfg, dict) and runtime_cfg else (metadata.get("known_object_25d") if isinstance(metadata, dict) else None)
        if not isinstance(cfg, dict):
            self._annotate_measurement_artifacts(context, {"correction_source": "none", "correction_applied": False})
            self._record_result(context, {"enabled": False, "status": "skipped"})
            return

        objects = context.get_artifact("objects", [])
        if not isinstance(objects, list) or not objects:
            self._annotate_measurement_artifacts(context, {"correction_source": "none", "correction_applied": False})
            self._record_result(context, {"enabled": bool(cfg.get("enabled", False)), "status": "failed", "reason": "no_objects"})
            return

        saved_scale_x = cfg.get("persisted_scale_correction_x")
        saved_scale_y = cfg.get("persisted_scale_correction_y")
        saved_scale_z = cfg.get("persisted_scale_correction_z")
        has_saved_scale = saved_scale_x is not None and saved_scale_y is not None and saved_scale_z is not None
        apply_saved_correction = bool(cfg.get("apply_persisted_correction", has_saved_scale))
        correction_context_id = f"known_cube:{float(saved_scale_x) if saved_scale_x is not None else 'na'}:{float(saved_scale_y) if saved_scale_y is not None else 'na'}:{float(saved_scale_z) if saved_scale_z is not None else 'na'}"
        calibration_enabled = bool(cfg.get("enabled", False))
        if has_saved_scale and apply_saved_correction and not calibration_enabled:
            _apply_known_object_scale_correction(
                objects,
                scale_x=float(saved_scale_x),
                scale_y=float(saved_scale_y),
                scale_z=float(saved_scale_z),
                correction_source="persisted_known_cube",
                correction_context_id=correction_context_id,
            )
            context.set_artifact("objects", objects)
            context.set_artifact(
                "measurement_correction_context",
                {
                    "correction_source": "persisted_known_cube",
                    "scale_x": float(saved_scale_x),
                    "scale_y": float(saved_scale_y),
                    "scale_z": float(saved_scale_z),
                    "correction_applied": True,
                    "correction_context_id": correction_context_id,
                },
            )
            # Persisted-only path also exposes the raw vs corrected geometry so
            # the Geometry-debug workspace is populated identically regardless
            # of whether the cube was just measured or restored from disk.
            self._emit_geometry_debug_artifacts(context, objects)

        if not calibration_enabled:
            self._annotate_measurement_artifacts(
                context,
                {
                    "correction_source": "persisted_known_cube" if has_saved_scale and apply_saved_correction else "none",
                    "scale_x": float(saved_scale_x) if saved_scale_x is not None else None,
                    "scale_y": float(saved_scale_y) if saved_scale_y is not None else None,
                    "scale_z": float(saved_scale_z) if saved_scale_z is not None else None,
                    "correction_applied": bool(has_saved_scale and apply_saved_correction),
                    "correction_context_id": correction_context_id if has_saved_scale and apply_saved_correction else None,
                },
            )
            self._record_result(
                context,
                {
                    "enabled": False,
                    "status": "saved_correction_applied" if has_saved_scale and apply_saved_correction else "skipped",
                    "applied_correction": has_saved_scale and apply_saved_correction,
                    "applied_persisted_correction": has_saved_scale and apply_saved_correction,
                    "persisted_scale_correction_x": float(saved_scale_x) if saved_scale_x is not None else None,
                    "persisted_scale_correction_y": float(saved_scale_y) if saved_scale_y is not None else None,
                    "persisted_scale_correction_z": float(saved_scale_z) if saved_scale_z is not None else None,
                },
            )
            return

        selection_mode = str(cfg.get("target_selection") or "largest_component")
        target = None
        if selection_mode == "manual_component_id":
            requested = int(cfg.get("manual_component_id") or 0)
            target = next((item for item in objects if int(item.get("object_id") or -1) == requested), None)
        elif selection_mode == "label_match":
            label = str(cfg.get("object_label") or "").lower()
            target = next((item for item in objects if label and label in str(item.get("label") or "").lower()), None)
        if target is None:
            target = max(objects, key=lambda item: float(item.get("footprint_area_mm2") or 0.0))

        known_w = float(cfg.get("known_width_mm") or 0.0)
        known_d = float(cfg.get("known_depth_mm") or 0.0)
        known_h = float(cfg.get("known_height_mm") or 0.0)
        tol_pct = float(cfg.get("tolerance_percent") or 5.0)
        dims = target.get("dimensions_mm") if isinstance(target.get("dimensions_mm"), (list, tuple)) else [0, 0, 0]
        measured_w = float(dims[0] or target.get("major_axis_mm") or 0.0)
        measured_d = float(dims[1] or target.get("minor_axis_mm") or 0.0)
        measured_h = float((target.get("height_above_belt_mm") or {}).get("max_height_mm") or 0.0)

        def _err(measured: float, known: float) -> float:
            return ((measured - known) / known * 100.0) if known > 1e-9 else 0.0

        err_x = _err(measured_w, known_w)
        err_y = _err(measured_d, known_d)
        err_z = _err(measured_h, known_h)
        scale_x = (known_w / measured_w) if measured_w > 1e-9 else 1.0
        scale_y = (known_d / measured_d) if measured_d > 1e-9 else 1.0
        scale_z = (known_h / measured_h) if measured_h > 1e-9 else 1.0
        apply_correction = bool(cfg.get("apply_correction", True))
        if apply_correction:
            _apply_known_object_scale_correction(
                objects,
                scale_x=scale_x,
                scale_y=scale_y,
                scale_z=scale_z,
                correction_source="known_cube_measured",
                correction_context_id=f"known_cube:{scale_x:.8f}:{scale_y:.8f}:{scale_z:.8f}",
            )
            context.set_artifact("objects", objects)
            corrected_target = next((item for item in objects if int(item.get("object_id") or -1) == int(target.get("object_id") or -2)), target)
            corrected_dims = corrected_target.get("dimensions_mm") if isinstance(corrected_target.get("dimensions_mm"), (list, tuple)) else [None, None, None]
            corrected_height = (corrected_target.get("height_above_belt_mm") or {}).get("max_height_mm") if isinstance(corrected_target.get("height_above_belt_mm"), dict) else None
        else:
            corrected_dims = [None, None, None]
            corrected_height = None
        result = {
            "enabled": True,
            "status": "ok",
            "target_object_id": int(target.get("object_id") or 0),
            "object_label": str(cfg.get("object_label") or "cubo"),
            "known_width_mm": known_w,
            "known_depth_mm": known_d,
            "known_height_mm": known_h,
            "measured_width_mm": measured_w,
            "measured_depth_mm": measured_d,
            "measured_height_mm": measured_h,
            "corrected_width_mm": float(corrected_dims[0]) if corrected_dims[0] is not None else None,
            "corrected_depth_mm": float(corrected_dims[1]) if corrected_dims[1] is not None else None,
            "corrected_height_mm": float(corrected_height) if corrected_height is not None else None,
            "scale_error_x_percent": err_x,
            "scale_error_y_percent": err_y,
            "scale_error_z_percent": err_z,
            "pass_x": abs(err_x) <= tol_pct,
            "pass_y": abs(err_y) <= tol_pct,
            "pass_z": abs(err_z) <= tol_pct,
            "recommended_scale_correction_x": scale_x,
            "recommended_scale_correction_y": scale_y,
            "recommended_scale_correction_z": scale_z,
            "applied_correction": apply_correction,
            "applied_persisted_correction": False,
            "persisted_scale_correction_x": float(saved_scale_x) if saved_scale_x is not None else None,
            "persisted_scale_correction_y": float(saved_scale_y) if saved_scale_y is not None else None,
            "persisted_scale_correction_z": float(saved_scale_z) if saved_scale_z is not None else None,
            "tolerance_percent": tol_pct,
            "correction_source": "known_cube_measured" if apply_correction else "none",
            "scale_x": scale_x,
            "scale_y": scale_y,
            "scale_z": scale_z,
            "correction_applied": bool(apply_correction),
            "correction_context_id": f"known_cube:{scale_x:.8f}:{scale_y:.8f}:{scale_z:.8f}",
        }
        context.set_artifact(
            "measurement_correction_context",
            {
                "correction_source": result["correction_source"],
                "scale_x": result["scale_x"],
                "scale_y": result["scale_y"],
                "scale_z": result["scale_z"],
                "correction_applied": bool(result["correction_applied"]),
                "correction_context_id": result["correction_context_id"],
            },
        )
        self._annotate_measurement_artifacts(context, context.get_artifact("measurement_correction_context", {}))
        self._emit_geometry_debug_artifacts(context, objects)
        self._record_result(context, result)

    def _annotate_measurement_artifacts(self, context: PipelineContext, correction: dict[str, Any]) -> None:
        for artifact in context.processing_artifacts:
            if not isinstance(artifact, dict):
                continue
            if str(artifact.get("stage_id") or "") != "measurement":
                continue
            meta = artifact.get("metadata") if isinstance(artifact.get("metadata"), dict) else {}
            meta = dict(meta)
            meta.update(
                {
                    "correction_source": correction.get("correction_source"),
                    "scale_x": correction.get("scale_x"),
                    "scale_y": correction.get("scale_y"),
                    "scale_z": correction.get("scale_z"),
                    "correction_applied": bool(correction.get("correction_applied", False)),
                    "correction_context_id": correction.get("correction_context_id"),
                }
            )
            artifact["metadata"] = meta

    def _record_result(self, context: PipelineContext, result: dict[str, Any]) -> None:
        output_dir: Path = context.require_artifact("output_dir")
        (output_dir / "known_object_scale_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        context.set_artifact("known_object_scale_validation", result)
        files = dict(context.get_artifact("files", {}))
        files["known_object_scale_validation"] = "known_object_scale_validation.json"
        context.set_artifact("files", files)
        context.add_processing_artifact(
            artifact_id="known_object_scale_validation",
            stage_id="measurement",
            kind="json",
            title="Known object scale validation",
            path="known_object_scale_validation.json",
            mime_type="application/json",
            preview_available=True,
            metadata=_numeric_source_meta({"source": "native_pipeline", "producer": self.name, **result}),
        )

    def _emit_geometry_debug_artifacts(self, context: PipelineContext, objects: list[dict[str, Any]]) -> None:
        output_dir: Path = context.require_artifact("output_dir")
        rows: list[dict[str, Any]] = []
        for row in objects:
            oid = int(row.get("object_id") or 0)
            payload = {
                "object_id": oid,
                "geometry_coordinate_space": row.get("geometry_coordinate_space"),
                "metrics_coordinate_space": row.get("metrics_coordinate_space"),
                "correction_source": row.get("correction_source"),
                "scale_correction_applied": row.get("scale_correction_applied"),
                "geometry_debug": row.get("geometry_debug"),
                "geometry_invariant_warnings": row.get("geometry_invariant_warnings") or [],
                "raw_eccentricity": ((row.get("raw_metrics") or {}).get("feature_eccentricity") if isinstance(row.get("raw_metrics"), dict) else None),
                "corrected_eccentricity": row.get("feature_eccentricity"),
                "raw_roundness": ((row.get("raw_metrics") or {}).get("sphericity_score") if isinstance(row.get("raw_metrics"), dict) else None),
                "corrected_roundness": row.get("sphericity_score"),
                "raw_diameter_mm": ((row.get("raw_metrics") or {}).get("diameter_mm") if isinstance(row.get("raw_metrics"), dict) else None),
                "corrected_diameter_mm": row.get("diameter_mm"),
            }
            rows.append(payload)
            file_name = f"geometry_debug_object_{oid}.json"
            (output_dir / file_name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            context.add_processing_artifact(
                artifact_id=f"geometry_debug_object_{oid}",
                stage_id="measurement",
                object_id=oid,
                kind="json",
                title=f"Object #{oid} geometry debug",
                path=file_name,
                mime_type="application/json",
                preview_available=True,
                metadata=_numeric_source_meta(
                    {
                        "source": "native_pipeline",
                        "producer": self.name,
                        "artifact_type": "geometry_debug",
                        "semantic_type": "geometry_debug",
                    }
                ),
            )
        summary_path = "geometry_debug_summary.json"
        (output_dir / summary_path).write_text(json.dumps({"objects": rows}, indent=2), encoding="utf-8")
        context.add_processing_artifact(
            artifact_id="geometry_debug_summary",
            stage_id="measurement",
            kind="json",
            title="Geometry debug summary",
            path=summary_path,
            mime_type="application/json",
            preview_available=True,
            metadata=_numeric_source_meta(
                {
                    "source": "native_pipeline",
                    "producer": self.name,
                    "artifact_type": "geometry_debug",
                    "semantic_type": "geometry_debug",
                }
            ),
        )


def _apply_known_object_scale_correction(
    objects: list[dict[str, Any]],
    *,
    scale_x: float,
    scale_y: float,
    scale_z: float,
    correction_source: str = "known_object_scale_validation",
    correction_context_id: str | None = None,
) -> bool:
    changed = False

    def _scale_tuple(value: Any, scales: tuple[float, ...]) -> tuple[float, ...] | Any:
        if not isinstance(value, (list, tuple)):
            return value
        out: list[float] = []
        for idx, item in enumerate(value):
            factor = scales[min(idx, len(scales) - 1)]
            out.append(round(float(item) * factor, 4))
        return tuple(out)

    def _scale_number(row: dict[str, Any], key: str, factor: float) -> None:
        if row.get(key) is not None:
            row[key] = round(float(row[key]) * factor, 4)

    for row in objects:
        existing = row.get("scale_correction_applied")
        if isinstance(existing, dict):
            ex = _safe_float(existing.get("x"))
            ey = _safe_float(existing.get("y"))
            ez = _safe_float(existing.get("z"))
            if ex is not None and ey is not None and ez is not None:
                if abs(ex - scale_x) < 1e-9 and abs(ey - scale_y) < 1e-9 and abs(ez - scale_z) < 1e-9:
                    row["metrics_coordinate_space"] = "corrected_mm"
                    row["correction_source"] = correction_source
                    row["correction_context_id"] = correction_context_id or row.get("correction_context_id")
                    continue
        raw_snapshot = row.get("raw_metrics")
        if not isinstance(raw_snapshot, dict):
            row["raw_metrics"] = {
                "dimensions_mm": list(row.get("dimensions_mm") or []) if isinstance(row.get("dimensions_mm"), (list, tuple)) else None,
                "major_axis_mm": row.get("major_axis_mm"),
                "minor_axis_mm": row.get("minor_axis_mm"),
                "diameter_mm": row.get("diameter_mm"),
                "sphericity_score": row.get("sphericity_score"),
                "feature_sphericity_3d": row.get("feature_sphericity_3d"),
                "feature_eccentricity": row.get("feature_eccentricity"),
                "feature_volume_proxy_mm3": row.get("feature_volume_proxy_mm3"),
                "feature_edge_roughness": row.get("feature_edge_roughness"),
                "feature_local_curvature_proxy": row.get("feature_local_curvature_proxy"),
                "feature_curvature_components_raw": dict(row.get("feature_curvature_components_raw") or {}) if isinstance(row.get("feature_curvature_components_raw"), dict) else None,
                "footprint_area_mm2": row.get("footprint_area_mm2"),
                "height_above_belt_mm": dict(row.get("height_above_belt_mm") or {}) if isinstance(row.get("height_above_belt_mm"), dict) else None,
            }
        # NOTE: Keep positional fields (center_mm, bbox_*_mm) in heightmap coordinates so that
        # overlay rendering (which converts mm -> pixel using the original x/y resolution) still
        # draws markers on the actual object. Only physical size fields get scaled.
        changed = True
        row["dimensions_mm"] = _scale_tuple(row.get("dimensions_mm"), (scale_x, scale_y, scale_z))
        _scale_number(row, "major_axis_mm", scale_x)
        _scale_number(row, "minor_axis_mm", scale_y)
        if row.get("major_axis_mm") is not None and row.get("minor_axis_mm") is not None:
            major = max(float(row["major_axis_mm"]), float(row["minor_axis_mm"]))
            minor = min(float(row["major_axis_mm"]), float(row["minor_axis_mm"]))
            row["major_axis_mm"] = round(major, 4)
            row["minor_axis_mm"] = round(minor, 4)
            row["feature_eccentricity"] = round(float(np.sqrt(max(0.0, 1.0 - (minor * minor) / max(major * major, 1e-9)))), 4)
            row["sphericity_score"] = round(float(minor / max(major, 1e-9)), 4)
        dims = row.get("dimensions_mm")
        if isinstance(dims, (list, tuple)) and len(dims) >= 2:
            row["diameter_estimate_mm"] = round(float(max(float(dims[0]), float(dims[1]))), 4)
        if row.get("major_axis_mm") is not None or row.get("minor_axis_mm") is not None:
            row["diameter_mm"] = round(max(float(row.get("major_axis_mm") or 0.0), float(row.get("minor_axis_mm") or 0.0)), 4)
        heights = row.get("height_above_belt_mm")
        if isinstance(heights, dict):
            for key in ("max_height_mm", "mean_height_mm", "median_height_mm", "p95_height_mm", "height_std_mm"):
                if heights.get(key) is not None:
                    heights[key] = round(float(heights[key]) * scale_z, 4)
        if row.get("major_axis_mm") is not None and row.get("minor_axis_mm") is not None and isinstance(heights, dict):
            max_h = float(heights.get("max_height_mm") or 0.0)
            row["feature_sphericity_3d"] = round(_axis_balance_3d(float(row["major_axis_mm"]), float(row["minor_axis_mm"]), max_h), 4)
        _scale_number(row, "feature_volume_proxy_mm3", scale_x * scale_y * scale_z)
        _scale_number(row, "feature_edge_roughness", scale_z)
        _scale_number(row, "footprint_area_mm2", scale_x * scale_y)
        # Recompute the curvature proxy in corrected metric space. Curvature is
        # mean(|dH/dx|) + mean(|dH/dy|). After correction, dH gets * scale_z while
        # dx gets * scale_x and dy gets * scale_y, so the per-axis correction factors
        # are scale_z/scale_x and scale_z/scale_y, not just scale_z. Without this the
        # proxy stays inflated whenever scale_x and scale_y differ from each other.
        curvature_components = row.get("feature_curvature_components_raw")
        if (
            isinstance(curvature_components, dict)
            and curvature_components.get("mean_abs_gx_mm_per_mm") is not None
            and curvature_components.get("mean_abs_gy_mm_per_mm") is not None
            and scale_x > 1e-9
            and scale_y > 1e-9
        ):
            gx_raw = float(curvature_components.get("mean_abs_gx_mm_per_mm") or 0.0)
            gy_raw = float(curvature_components.get("mean_abs_gy_mm_per_mm") or 0.0)
            corrected_curvature = gx_raw * (scale_z / scale_x) + gy_raw * (scale_z / scale_y)
            row["feature_local_curvature_proxy"] = round(corrected_curvature, 4)
            row["feature_curvature_components_corrected"] = {
                "mean_abs_gx_mm_per_mm": round(gx_raw * (scale_z / scale_x), 6),
                "mean_abs_gy_mm_per_mm": round(gy_raw * (scale_z / scale_y), 6),
                "scale_z_over_scale_x": round(scale_z / scale_x, 6),
                "scale_z_over_scale_y": round(scale_z / scale_y, 6),
            }
        else:
            # Fallback for objects without per-axis curvature components (e.g.
            # measurement stage from an older pipeline run). Use the isotropic
            # min(scale_z/scale_x, scale_z/scale_y) as a conservative bound so
            # the proxy doesn't stay inflated at raw-units magnitudes.
            if scale_x > 1e-9 and scale_y > 1e-9:
                effective_factor = min(scale_z / scale_x, scale_z / scale_y, scale_z)
                _scale_number(row, "feature_local_curvature_proxy", effective_factor)
        metric_ctx = row.get("geometry_metric_context") if isinstance(row.get("geometry_metric_context"), dict) else {}
        x_res = _safe_float(metric_ctx.get("x_resolution_mm")) or 1.0
        y_res = _safe_float(metric_ctx.get("y_resolution_mm")) or 1.0
        raw_contour = row.get("contour_px")
        raw_geom = _contour_metrics_in_metric_space(
            raw_contour if isinstance(raw_contour, list) else None,
            x_resolution_mm=x_res,
            y_resolution_mm=y_res,
            scale_x=1.0,
            scale_y=1.0,
        )
        corrected_geom = _contour_metrics_in_metric_space(
            raw_contour if isinstance(raw_contour, list) else None,
            x_resolution_mm=x_res,
            y_resolution_mm=y_res,
            scale_x=scale_x,
            scale_y=scale_y,
        )
        contour_correction_applied = bool(corrected_geom.get("valid"))
        if contour_correction_applied:
            row["major_axis_mm"] = corrected_geom.get("major_axis_mm")
            row["minor_axis_mm"] = corrected_geom.get("minor_axis_mm")
            row["feature_eccentricity"] = corrected_geom.get("eccentricity")
            row["sphericity_score"] = corrected_geom.get("roundness")
            row["feature_footprint_roundness"] = corrected_geom.get("roundness")
            row["feature_circularity"] = corrected_geom.get("circularity")
            row["footprint_area_mm2"] = corrected_geom.get("area_mm2")
            row["perimeter_mm"] = corrected_geom.get("perimeter_mm")
            row["diameter_mm"] = corrected_geom.get("major_axis_mm")
            row["geometry_debug"] = {
                "raw": raw_geom,
                "corrected": corrected_geom,
                "point_cloud_mm": row.get("center_mm"),
                "geometry_coordinate_space": "corrected_metric_mm",
                "scale_x": scale_x,
                "scale_y": scale_y,
                "scale_z": scale_z,
            }
        # Recompute feature_sphericity_3d in corrected metric space using the
        # canonical XYZ extents (dimensions_mm). Doing it here, AFTER any
        # contour-based override, prevents stretched / misaligned ellipse axes
        # from collapsing the axis-balance ratio to a near-zero value when the
        # raw heightmap is anisotropic (scale_x != scale_y != scale_z).
        dims_for_sph3d = row.get("dimensions_mm") if isinstance(row.get("dimensions_mm"), (list, tuple)) else None
        if dims_for_sph3d is not None and len(dims_for_sph3d) >= 3:
            row["feature_sphericity_3d"] = round(
                _axis_balance_3d(
                    float(dims_for_sph3d[0]),
                    float(dims_for_sph3d[1]),
                    float(dims_for_sph3d[2]),
                ),
                4,
            )
        invariants: list[str] = []
        dx = _safe_float((row.get("dimensions_mm") or [None, None, None])[0] if isinstance(row.get("dimensions_mm"), (list, tuple)) else None)
        dy = _safe_float((row.get("dimensions_mm") or [None, None, None])[1] if isinstance(row.get("dimensions_mm"), (list, tuple)) else None)
        ecc = _safe_float(row.get("feature_eccentricity"))
        if dx is not None and dy is not None and ecc is not None and abs(dx - dy) <= max(2.5, 0.05 * max(dx, dy)) and ecc > 0.85:
            invariants.append("invariant_violation:near_equal_dims_with_extreme_eccentricity")
        sel_d = _safe_float(row.get("diameter_mm"))
        if dx is not None and dy is not None and sel_d is not None and sel_d > (2.2 * max(dx, dy)):
            invariants.append("invariant_violation:diameter_far_exceeds_dims")
        roundness = _safe_float(row.get("sphericity_score"))
        if dx is not None and dy is not None and roundness is not None and max(dx, dy) > 20 and roundness < 0.05:
            invariants.append("invariant_violation:compact_blob_with_near_zero_roundness")
        # Only flag anisotropic scales as a metric-stability concern when the
        # contour-based corrected geometry could not be derived. With a valid
        # corrected contour the planar metrics ARE in true mm space, so the
        # raw->corrected anisotropy is information, not a defect.
        if (
            not contour_correction_applied
            and (max(scale_x, scale_y, scale_z) / max(min(scale_x, scale_y, scale_z), 1e-9)) > 4.0
        ):
            invariants.append("warning:anisotropic_scale_factors_may_destabilize_metrics")
        if invariants:
            row["geometry_invariant_warnings"] = invariants
        else:
            row.pop("geometry_invariant_warnings", None)
        row["geometry_coordinate_space"] = "corrected_metric_mm"
        row["scale_correction_applied"] = {"x": scale_x, "y": scale_y, "z": scale_z}
        row["metrics_coordinate_space"] = "corrected_mm"
        row["correction_source"] = correction_source
        row["correction_context_id"] = correction_context_id
    return changed


def _axis_balance_3d(x_mm: float, y_mm: float, z_mm: float) -> float:
    axes = [abs(float(x_mm)), abs(float(y_mm)), abs(float(z_mm))]
    largest = max(axes)
    if largest <= 1e-9:
        return 0.0
    return min(axes) / largest


@dataclass
class ClassifyMiningBall25DStage:
    name: str = "ClassifyMiningBall25D"
    category: str = "classification"
    required_modalities: tuple[CaptureModality, ...] = ("heightmap",)
    produced_modalities: tuple[CaptureModality, ...] = ()
    produced_artifacts: tuple[str, ...] = ("classification",)
    stage_category: str | None = "classification"

    def run(self, context: PipelineContext) -> None:
        _assert_not_preview_semantic(context, stage_name=self.name)
        # Canonical geometry invariant:
        # classification consumes height_above_belt semantics only.
        semantic_raster = resolve_height_artifact(
            context,
            semantic_field="height_above_belt",
            representation="numeric_raster",
        )
        numeric_source_artifact_id = str((semantic_raster or {}).get("artifact_id") or "height_above_belt_raster")
        if str(context.get_artifact("height_processing_semantic_field", "")) != "height_above_belt":
            raise ValueError("classification cannot use raw_sensor_z unless explicitly debug")
        objects = context.get_artifact("objects", [])
        ball_count = 0
        non_ball = 0
        explanations: list[dict[str, Any]] = []
        for item in objects:
            if "metrics_coordinate_space" not in item:
                item["metrics_coordinate_space"] = "raw_mm"
            if "correction_source" not in item:
                item["correction_source"] = "none"
            label, display_label, class_name, confidence = _classify_25d(item)
            superclass = map_label_to_superclass(label)
            label, display_label, class_name, confidence, superclass = apply_sphericity_3d_fallback_to_object(
                item,
                label=label,
                display_label=display_label,
                class_name=class_name,
                confidence=confidence,
                superclass=superclass,
            )
            item["class_name"] = class_name
            item["confidence"] = confidence
            item["label"] = label
            item["superclass"] = superclass
            item["display_label"] = display_label
            item["class_group"] = _class_group(superclass)
            item["classification"] = {
                "label": label,
                "display_label": display_label,
                "superclass": superclass,
                "confidence": confidence,
            }
            diameter_metrics = _compose_diameter_metrics(item)
            item["diameter_ellipse_mm"] = diameter_metrics.get("diameter_ellipse_mm")
            item["diameter_equivalent_area_mm"] = diameter_metrics.get("diameter_equivalent_area_mm")
            item["diameter_circumference_mm"] = diameter_metrics.get("diameter_circumference_mm")
            item["diameter_selected_mm"] = diameter_metrics.get("diameter_selected_mm")
            item["diameter_selected_source"] = diameter_metrics.get("diameter_selected_source")
            item["diameter_sanity_status"] = diameter_metrics.get("diameter_sanity_status")
            item["diameter_sanity_message"] = diameter_metrics.get("diameter_sanity_message")
            reasons: list[str] = []
            if float(item.get("sphericity_score") or 0.0) < 0.75:
                reasons.append("low_sphericity")
            trunc = float(((item.get("sphere_fit_diagnostics") or {}).get("estimated_truncation_ratio") or 0.0) if isinstance(item.get("sphere_fit_diagnostics"), dict) else 0.0)
            if trunc > 0.2:
                reasons.append("missing_lower_hemisphere")
            if float(item.get("feature_eccentricity") or 0.0) > 0.6:
                reasons.append("excessive_eccentricity")
            if float(((item.get("height_above_belt_mm") or {}).get("height_std_mm") or 0.0) if isinstance(item.get("height_above_belt_mm"), dict) else 0.0) < 0.5:
                reasons.append("truncated_height_distribution")
            item["classification_explanation"] = {
                "contributing_metrics": {
                    "sphericity_score": item.get("sphericity_score"),
                    "feature_eccentricity": item.get("feature_eccentricity"),
                    "feature_sphericity_3d": item.get("feature_sphericity_3d"),
                    "height_max_mm": ((item.get("height_above_belt_mm") or {}).get("max_height_mm") if isinstance(item.get("height_above_belt_mm"), dict) else None),
                    "truncation_ratio": trunc,
                },
                "rejected_ball_reasons": reasons,
            }
            explanation = _classification_explanation_for_object(item)
            item["classification_explanation"] = explanation
            item["classification_explanation_ref"] = {
                "artifact_id": "classification_explanation",
                "object_id": int(item.get("object_id", 0) or 0),
            }
            explanations.append(explanation)
            if class_name == "ball":
                ball_count += 1
            elif class_name == "non_ball":
                non_ball += 1
            oid = int(item.get("object_id", 0) or 0)
            context.add_processing_artifact(
                artifact_id=f"classification_object_{oid}",
                stage_id="classification",
                object_id=oid,
                kind="json",
                title=f"Object #{oid} classification",
                metadata=_numeric_source_meta(
                    {
                        "source": "native_pipeline",
                        "producer": self.name,
                        "label": label,
                        "superclass": superclass,
                        "numeric_source_artifact": numeric_source_artifact_id,
                        "semantic_field": "height_above_belt",
                        "classification_explanation_artifact_id": "classification_explanation",
                    }
                ),
            )
        object_count = len(objects)
        decision = "accept" if object_count > 0 and non_ball == 0 else "review"
        canonical = select_dominant_classification({"objects": objects})
        context.set_artifact(
            "summary",
            {
                "object_count": object_count,
                "ball_count": ball_count,
                "non_ball_count": non_ball,
                "decision": decision,
                "confidence": canonical.get("confidence"),
                "label": canonical.get("label"),
                "superclass": canonical.get("superclass"),
            },
        )
        context.set_artifact(
            "classification_result_25d",
            {
                "label": canonical.get("label"),
                "superclass": canonical.get("superclass"),
                "confidence": canonical.get("confidence"),
                "object_count": object_count,
                "objects": [
                    {
                        "object_id": item.get("object_id"),
                        "label": item.get("label"),
                        "display_label": item.get("display_label"),
                        "superclass": item.get("superclass"),
                        "confidence": item.get("confidence"),
                    }
                    for item in objects
                ],
            },
        )
        context.add_processing_artifact(
            artifact_id="classification_result_25d",
            stage_id="classification",
            kind="json",
            title="25D classification result",
            metadata=_numeric_source_meta(
                {
                    "source": "native_pipeline",
                    "producer": self.name,
                    "superclass": canonical.get("superclass"),
                    "numeric_source_artifact": numeric_source_artifact_id,
                    "semantic_field": "height_above_belt",
                }
            ),
        )
        output_dir: Path = context.require_artifact("output_dir")
        explanation_payload = {
            "stage": "classification",
            "artifact_id": "classification_explanation",
            "scope": "global",
            "objects": explanations,
        }
        metric_payload = {
            "stage": "classification",
            "artifact_id": "metric_explanation",
            "scope": "global",
            "objects": [
                {
                    "object_id": row.get("object_id"),
                    "metric_trace": row.get("metric_trace") if isinstance(row, dict) else [],
                }
                for row in explanations
            ],
        }
        (output_dir / "classification_explanation.json").write_text(json.dumps(explanation_payload, indent=2), encoding="utf-8")
        (output_dir / "metric_explanation.json").write_text(json.dumps(metric_payload, indent=2), encoding="utf-8")
        files = dict(context.get_artifact("files", {}))
        files["classification_explanation"] = "classification_explanation.json"
        files["metric_explanation"] = "metric_explanation.json"
        context.set_artifact("files", files)
        context.add_processing_artifact(
            artifact_id="classification_explanation",
            stage_id="classification",
            kind="json",
            title="Classification rule explanation",
            path="classification_explanation.json",
            mime_type="application/json",
            preview_available=True,
            metadata=_numeric_source_meta(
                {
                    "source": "native_pipeline",
                    "producer": self.name,
                    "scope": "global",
                    "semantic_type": "classification_explanation",
                    "artifact_type": "classification_explanation",
                    "object_count": len(explanations),
                    "objects": explanations,
                }
            ),
        )
        context.add_processing_artifact(
            artifact_id="metric_explanation",
            stage_id="classification",
            kind="json",
            title="Metric explanation",
            path="metric_explanation.json",
            mime_type="application/json",
            preview_available=True,
            metadata=_numeric_source_meta(
                {
                    "source": "native_pipeline",
                    "producer": self.name,
                    "scope": "global",
                    "semantic_type": "metric_explanation",
                    "artifact_type": "metric_explanation",
                    "object_count": len(explanations),
                    "objects": metric_payload["objects"],
                }
            ),
        )
        context.set_artifact("algorithm_stage", "classification")
        context.set_artifact("objects", objects)


@dataclass
class Generate25DOverlaysStage:
    name: str = "Generate25DOverlays"
    category: str = "overlay"
    required_modalities: tuple[CaptureModality, ...] = ("heightmap",)
    produced_modalities: tuple[CaptureModality, ...] = ()
    produced_artifacts: tuple[str, ...] = ("overlays",)
    stage_category: str | None = "overlay"

    def run(self, context: PipelineContext) -> None:
        output_dir: Path = context.require_artifact("output_dir")
        frame: HeightmapFrame = context.require_artifact("heightmap_frame")
        normalized = np.asarray(context.require_artifact("normalized_heightmap_mm"), dtype=np.float32)
        mask = np.asarray(context.require_artifact("segmentation_mask"), dtype=bool)
        objects = context.get_artifact("objects", [])

        write_heightmap_preview_png(normalized, frame.valid_mask, output_dir / "height_overlay.png")
        comp_overlay = _overlay_mask_on_heightmap(normalized, frame.valid_mask, mask)
        cv2.imwrite(str(output_dir / "segmentation_overlay.png"), comp_overlay)

        meas = comp_overlay.copy()
        for item in objects:
            bbox = item.get("bbox_min_mm"), item.get("bbox_max_mm")
            center = item.get("center_mm")
            if isinstance(center, (list, tuple)) and len(center) >= 2:
                cx = int(round((float(center[0]) - frame.origin_x_mm) / frame.x_resolution_mm))
                cy = int(round((float(center[1]) - frame.origin_y_mm) / frame.y_resolution_mm))
                cv2.circle(meas, (cx, cy), 4, (255, 255, 255), -1)
                label = f"#{item.get('object_id')} {item.get('label', item.get('class_name', 'unknown'))}"
                cv2.putText(meas, label, (cx + 8, max(14, cy - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                h = ((item.get("height_above_belt_mm") or {}).get("max_height_mm"))
                if h is not None:
                    cv2.putText(meas, f"h={float(h):.1f}mm", (cx + 8, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1, cv2.LINE_AA)
            if isinstance(bbox[0], (list, tuple)) and isinstance(bbox[1], (list, tuple)):
                x0 = int(round((float(bbox[0][0]) - frame.origin_x_mm) / frame.x_resolution_mm))
                y0 = int(round((float(bbox[0][1]) - frame.origin_y_mm) / frame.y_resolution_mm))
                x1 = int(round((float(bbox[1][0]) - frame.origin_x_mm) / frame.x_resolution_mm))
                y1 = int(round((float(bbox[1][1]) - frame.origin_y_mm) / frame.y_resolution_mm))
                cv2.rectangle(meas, (x0, y0), (x1, y1), (255, 255, 255), 1)

        cv2.imwrite(str(output_dir / "measurement_overlay.png"), meas)

        classification = meas.copy()
        for item in objects:
            center = item.get("center_mm")
            if not isinstance(center, (list, tuple)) or len(center) < 2:
                continue
            cx = int(round((float(center[0]) - frame.origin_x_mm) / frame.x_resolution_mm))
            cy = int(round((float(center[1]) - frame.origin_y_mm) / frame.y_resolution_mm))
            metrics = item.get("height_above_belt_mm") or {}
            max_h = metrics.get("max_height_mm") if isinstance(metrics, dict) else None
            lines = [
                f"max_h={float(max_h):.1f}mm" if max_h is not None else "max_h=-",
                f"ecc={float(item.get('feature_eccentricity') or 0.0):.3f}",
                f"sph3d={float(item.get('feature_sphericity_3d') or 0.0):.3f}",
                f"flat={float(item.get('feature_flatness') or 0.0):.3f}",
                f"rough={float(item.get('feature_edge_roughness') or 0.0):.1f}",
            ]
            x = min(max(cx + 8, 4), max(4, classification.shape[1] - 150))
            y = min(max(cy + 26, 14), max(14, classification.shape[0] - 72))
            for idx, line in enumerate(lines):
                cv2.putText(
                    classification,
                    line,
                    (x, y + idx * 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (245, 245, 245),
                    1,
                    cv2.LINE_AA,
                )
        cv2.imwrite(str(output_dir / "classification_overlay.png"), classification)
        classification_result = context.get_artifact("classification_result_25d", {})

        files = dict(context.get_artifact("files", {}))
        files.update(
            {
                "overlay": "classification_overlay.png",
                "debug_plane_segmentation": "belt_plane_overlay.png",
                "debug_foreground": "segmentation_overlay.png",
                "debug_clusters": "measurement_overlay.png",
            }
        )
        context.set_artifact("files", files)

        _artifact_defs = [
            ("height_overlay", "overlay", "Height colormap", "height_overlay.png"),
            ("measurement_overlay", "overlay", "Measurement overlay", "measurement_overlay.png"),
            ("classification_overlay", "overlay", "Classification overlay", "classification_overlay.png"),
        ]
        for artifact_id, stage_id, title, path in _artifact_defs:
            context.add_processing_artifact(
                artifact_id=artifact_id,
                stage_id=stage_id,
                kind="image",
                title=title,
                path=path,
                mime_type="image/png",
                preview_available=True,
                metadata=_display_only_meta({"source": "native_pipeline", "producer": self.name, "modality": "heightmap", "numeric_source_artifact": "normalized_heightmap.npz"}),
            )
        context.add_processing_artifact(
            artifact_id="classification_overlay_metadata",
            stage_id="classification",
            kind="json",
            title="Classification overlay metadata",
            metadata={
                "source": "native_pipeline",
                "producer": self.name,
                "classification": classification_result if isinstance(classification_result, dict) else {},
            },
        )


def _load_heightmap_frame(height_path: Path, reflectance_path: Path | None, metadata: dict[str, Any]) -> HeightmapFrame:
    if not height_path.is_file():
        raise FileNotFoundError(f"Heightmap file not found: {height_path}")
    if height_path.suffix.lower() in _PREVIEW_IMAGE_SUFFIXES:
        raise ValueError(
            f"Preview image cannot be used as numeric height source: {height_path.name}. "
            "Use canonical numeric height artifacts (height16.tif or heightmap_frame.npz)."
        )
    if height_path.suffix.lower() == ".npz":
        frame = load_heightmap_npz(height_path)
    else:
        z = cv2.imread(str(height_path), cv2.IMREAD_UNCHANGED)
        if z is None:
            raise ValueError(f"Failed to decode heightmap: {height_path}")
        z = np.asarray(z)
        if z.ndim == 3:
            z = cv2.cvtColor(z, cv2.COLOR_BGR2GRAY)
        z = z.astype(np.float32)
        calib = metadata.get("calibration") if isinstance(metadata.get("calibration"), dict) else {}
        z = z * float(calib.get("z_scale") or 1.0) + float(calib.get("z_offset") or 0.0)
        valid = np.isfinite(z)
        # TriSpector-style 16-bit height images commonly encode no-return as 0.
        # If zeros dominate a non-NPZ source, treat them as invalid/no-return.
        finite_count = int(np.count_nonzero(valid))
        zero_mask = valid & (z == 0.0)
        zero_fraction = float(np.count_nonzero(zero_mask) / max(1, finite_count))
        if zero_fraction >= 0.02:
            valid = valid & (z > 0.0)
        frame = HeightmapFrame(
            z_mm=z,
            valid_mask=valid,
            reflectance=None,
            x_resolution_mm=float(calib.get("x_resolution_mm") or 1.0),
            y_resolution_mm=float(calib.get("profile_distance_mm") or 1.0),
            origin_x_mm=0.0,
            origin_y_mm=0.0,
            coordinate_system="sensor_xy_z_mm",
        )
    if reflectance_path is not None:
        refl = cv2.imread(str(reflectance_path), cv2.IMREAD_UNCHANGED)
        if refl is not None:
            frame.reflectance = refl
    return frame


def _heightmap_points(frame: HeightmapFrame, valid_mask: np.ndarray | None = None) -> np.ndarray:
    mask = frame.valid_mask if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    ys, xs = np.where(mask)
    z = frame.z_mm[ys, xs]
    x_mm = frame.origin_x_mm + xs.astype(np.float64) * frame.x_resolution_mm
    y_mm = frame.origin_y_mm + ys.astype(np.float64) * frame.y_resolution_mm
    return np.column_stack((x_mm, y_mm, z.astype(np.float64)))


def _reference_height_gate_from_distribution(
    z_mm: np.ndarray,
    valid_mask: np.ndarray,
    *,
    enabled: bool,
    margin_mm: float,
    min_coverage_ratio: float,
    max_coverage_ratio: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    valid = np.asarray(valid_mask, dtype=bool)
    if not enabled:
        return valid.copy(), {"enabled": False, "status": "disabled"}

    vals = np.asarray(z_mm[valid], dtype=np.float64)
    if vals.size < 8:
        return valid.copy(), {"enabled": True, "status": "too_few_values", "sample_count": int(vals.size)}

    min_cov = float(np.clip(min_coverage_ratio, 0.01, 0.95))
    max_cov = float(np.clip(max_coverage_ratio, min_cov + 0.01, 0.99))
    sample_count = int(min(2048, vals.size))
    qs = np.linspace(0.0, 1.0, sample_count, dtype=np.float64)
    curve = np.quantile(vals, qs)
    z_min = float(curve[0])
    z_max = float(curve[-1])
    z_range = z_max - z_min
    if z_range <= 1e-6:
        cutoff = z_max + float(max(0.0, margin_mm))
        gate = valid & (z_mm <= cutoff)
        return (
            gate,
            {
                "enabled": True,
                "status": "flat_distribution",
                "cutoff_z_mm": cutoff,
                "coverage_percent": float(np.count_nonzero(gate) / max(1, np.count_nonzero(valid)) * 100.0),
                "sample_count": int(vals.size),
            },
        )

    sorted_vals = np.sort(vals)
    gap_start = max(0, min(int(np.floor(min_cov * (sorted_vals.size - 1))), sorted_vals.size - 2))
    gap_end = max(gap_start + 1, min(int(np.ceil(max_cov * (sorted_vals.size - 1))), sorted_vals.size - 1))
    gap_values = np.diff(sorted_vals)
    search_gaps = gap_values[gap_start:gap_end]
    gap_cutoff: float | None = None
    gap_ratio = 0.0
    if search_gaps.size:
        local_gap_idx = int(np.argmax(search_gaps))
        gap_idx = gap_start + local_gap_idx
        robust_gap_scale = float(np.median(gap_values[gap_values > 0])) if np.any(gap_values > 0) else 0.0
        largest_gap = float(gap_values[gap_idx])
        gap_ratio = largest_gap / max(robust_gap_scale, 1e-6)
        if largest_gap >= max(1.0, robust_gap_scale * 8.0):
            gap_cutoff = float(sorted_vals[gap_idx] + max(0.0, margin_mm))

    smooth_window = max(5, min(51, (sample_count // 40) | 1))
    kernel = np.ones(smooth_window, dtype=np.float64) / float(smooth_window)
    smooth = np.convolve(np.pad(curve, (smooth_window // 2, smooth_window // 2), mode="edge"), kernel, mode="valid")
    normalized = (smooth - z_min) / z_range
    curvature = np.gradient(np.gradient(normalized, qs), qs)
    search = np.where((qs >= min_cov) & (qs <= max_cov))[0]
    if search.size:
        knee_idx = int(search[np.argmax(curvature[search])])
        reason = "max_positive_second_derivative"
    else:
        knee_idx = int(np.clip(round(min_cov * (sample_count - 1)), 0, sample_count - 1))
        reason = "fallback_min_coverage"

    if gap_cutoff is not None:
        cutoff = gap_cutoff
        reason = "largest_height_gap"
    else:
        cutoff = float(smooth[knee_idx] + max(0.0, margin_mm))
    coverage = float(np.count_nonzero(vals <= cutoff) / max(1, vals.size))
    if coverage < min_cov:
        cutoff = float(np.percentile(vals, min_cov * 100.0) + max(0.0, margin_mm))
        reason = "clamped_to_min_coverage"
    elif coverage > max_cov:
        cutoff = float(np.percentile(vals, max_cov * 100.0))
        reason = "clamped_to_max_coverage"

    gate = valid & (z_mm <= cutoff)
    coverage_percent = float(np.count_nonzero(gate) / max(1, np.count_nonzero(valid)) * 100.0)
    return (
        gate,
        {
            "enabled": True,
            "status": "ok",
            "reason": reason,
            "cutoff_z_mm": cutoff,
            "knee_quantile": float(qs[knee_idx]),
            "knee_z_mm": float(smooth[knee_idx]),
            "margin_mm": float(max(0.0, margin_mm)),
            "coverage_percent": coverage_percent,
            "sample_count": int(vals.size),
            "largest_gap_ratio": float(gap_ratio),
        },
    )


def _fit_plane_ransac(points: np.ndarray, *, iterations: int, threshold_mm: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    best_inliers = np.array([], dtype=np.int64)
    best_coeffs = None
    n = points.shape[0]
    if n < 3:
        raise ValueError("Not enough points for plane fitting")
    for _ in range(max(1, iterations)):
        idx = rng.choice(n, size=3, replace=False)
        coeffs = _plane_from_three(points[idx[0]], points[idx[1]], points[idx[2]])
        if coeffs is None:
            continue
        residuals = np.abs(_plane_residuals(points, coeffs))
        inliers = np.where(residuals <= threshold_mm)[0]
        if inliers.shape[0] > best_inliers.shape[0]:
            best_inliers = inliers
            best_coeffs = coeffs
    if best_coeffs is None:
        best_coeffs = _least_squares_plane(points)
        best_inliers = np.arange(points.shape[0])
    refined = _least_squares_plane(points[best_inliers]) if best_inliers.size >= 3 else best_coeffs
    return refined, best_inliers


def _plane_from_three(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray | None:
    xyz = np.asarray([p1, p2, p3], dtype=np.float64)
    design = np.column_stack((xyz[:, 0], xyz[:, 1], np.ones(3, dtype=np.float64)))
    if abs(float(np.linalg.det(design))) < 1e-9:
        return None
    sx, sy, intercept = np.linalg.solve(design, xyz[:, 2])
    return _heightmap_plane_coeffs(float(sx), float(sy), float(intercept))


def _least_squares_plane(points: np.ndarray) -> np.ndarray:
    xyz = np.asarray(points, dtype=np.float64)
    design = np.column_stack((xyz[:, 0], xyz[:, 1], np.ones(xyz.shape[0], dtype=np.float64)))
    sx, sy, intercept = np.linalg.lstsq(design, xyz[:, 2], rcond=None)[0]
    return _heightmap_plane_coeffs(float(sx), float(sy), float(intercept))


def _heightmap_plane_coeffs(sx: float, sy: float, intercept: float) -> np.ndarray:
    # Model reference surfaces as z = sx*x + sy*y + intercept; near-vertical planes are invalid for heightmaps.
    coeffs = np.array([-sx, -sy, 1.0, -intercept], dtype=np.float64)
    return coeffs / max(float(np.linalg.norm(coeffs[:3])), 1e-12)


def _plane_residuals(points: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    a, b, c, d = [float(x) for x in coeffs]
    denom = max(np.sqrt(a * a + b * b + c * c), 1e-12)
    return (points[:, 0] * a + points[:, 1] * b + points[:, 2] * c + d) / denom


def _residual_map(frame: HeightmapFrame, coeffs: np.ndarray) -> np.ndarray:
    h, w = frame.z_mm.shape
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    x_mm = frame.origin_x_mm + xx * frame.x_resolution_mm
    y_mm = frame.origin_y_mm + yy * frame.y_resolution_mm
    pts = np.column_stack((x_mm.ravel(), y_mm.ravel(), frame.z_mm.astype(np.float64).ravel()))
    residual = _plane_residuals(pts, coeffs).reshape((h, w)).astype(np.float32)
    residual[~frame.valid_mask] = 0.0
    return residual


def _render_belt_plane_overlay(frame: HeightmapFrame, inlier_mask: np.ndarray) -> np.ndarray:
    img = _height_color_image(frame.z_mm, frame.valid_mask)
    if img is None:
        img = np.zeros((frame.z_mm.shape[0], frame.z_mm.shape[1], 3), dtype=np.uint8)
    overlay = img.copy()
    overlay[inlier_mask] = (30, 190, 30)
    return cv2.addWeighted(img, 0.65, overlay, 0.35, 0.0)


def _overlay_mask_on_heightmap(values: np.ndarray, valid_mask: np.ndarray, mask: np.ndarray) -> np.ndarray:
    base = _height_color_image(values, valid_mask)
    if base is None:
        base = np.zeros((values.shape[0], values.shape[1], 3), dtype=np.uint8)
    out = base.copy()
    out[mask] = (0, 255, 255)
    return cv2.addWeighted(base, 0.7, out, 0.3, 0.0)


def _roi_border_mask(roi_mask: np.ndarray) -> np.ndarray:
    roi = np.asarray(roi_mask, dtype=bool)
    border = np.zeros_like(roi, dtype=bool)
    if not np.count_nonzero(roi):
        return border
    border[0, :] = roi[0, :]
    border[-1, :] = roi[-1, :]
    border[:, 0] = roi[:, 0]
    border[:, -1] = roi[:, -1]
    eroded = cv2.erode((roi.astype(np.uint8) * 255), np.ones((3, 3), np.uint8), iterations=1) > 0
    border |= roi & (~eroded)
    return border


def _background_candidate_score(mask: np.ndarray, roi_mask: np.ndarray) -> float:
    candidate = np.asarray(mask, dtype=bool)
    if not np.count_nonzero(candidate):
        return -1.0
    roi = np.asarray(roi_mask, dtype=bool)
    coverage = float(np.count_nonzero(candidate) / max(1, np.count_nonzero(roi)))
    border = _roi_border_mask(roi)
    border_touch_ratio = float(np.count_nonzero(candidate & border) / max(1, np.count_nonzero(candidate)))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), connectivity=8)
    largest = 0
    for label in range(1, num_labels):
        largest = max(largest, int(stats[label, cv2.CC_STAT_AREA]))
    largest_ratio = float(largest / max(1, np.count_nonzero(candidate)))
    # Prefer masks that touch border strongly and are spatially coherent.
    return (2.0 * border_touch_ratio) + largest_ratio + min(coverage, 0.35)


def _compute_depth_gradient(
    z_mm: np.ndarray,
    valid_mask: np.ndarray,
    *,
    smoothing_kernel: int,
    method: str,
    invalid_neighbor_policy: str,
) -> np.ndarray:
    z = np.asarray(z_mm, dtype=np.float32).copy()
    z[~valid_mask] = 0.0
    k = max(1, int(smoothing_kernel))
    if k > 1:
        if k % 2 == 0:
            k += 1
        z = cv2.GaussianBlur(z, (k, k), 0)
    if str(method).lower() == "finite_difference":
        gx = np.zeros_like(z, dtype=np.float32)
        gy = np.zeros_like(z, dtype=np.float32)
        gx[:, 1:-1] = 0.5 * (z[:, 2:] - z[:, :-2])
        gy[1:-1, :] = 0.5 * (z[2:, :] - z[:-2, :])
    else:
        gx = cv2.Sobel(z, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(z, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt((gx * gx) + (gy * gy))
    if invalid_neighbor_policy == "mark_high_gradient":
        vm = valid_mask.astype(np.uint8) * 255
        eroded = cv2.erode(vm, np.ones((3, 3), np.uint8), iterations=1) > 0
        border_invalid = valid_mask & (~eroded)
        if np.any(border_invalid):
            high = float(np.percentile(grad[valid_mask], 99)) if np.count_nonzero(valid_mask) else 1.0
            grad[border_invalid] = high
    grad[~valid_mask] = 0.0
    return grad


def _otsu_threshold(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float32)
    vals = vals[np.isfinite(vals)]
    if vals.size < 8:
        return float(np.median(vals)) if vals.size else 0.0
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if vmax <= vmin:
        return vmin
    scaled = np.clip(((vals - vmin) / (vmax - vmin) * 255.0), 0, 255).astype(np.uint8).reshape(-1, 1)
    thr, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(vmin + (float(thr) / 255.0) * (vmax - vmin))


def _select_low_gradient_component(
    *,
    low_grad_mask: np.ndarray,
    gradient: np.ndarray,
    z: np.ndarray,
    valid_mask: np.ndarray,
    roi_mask: np.ndarray,
    selection_mode: str,
    min_area_ratio: float,
    border_bonus: float,
    area_weight: float,
    constancy_weight: float,
    depth_pref_weight: float,
) -> tuple[dict[str, Any], int]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(low_grad_mask.astype(np.uint8), connectivity=8)
    total = max(1, int(np.count_nonzero(valid_mask)))
    border = _roi_border_mask(roi_mask)
    candidates: list[dict[str, Any]] = []
    best_label = 0
    best_score = -1e9
    for label in range(1, num_labels):
        mask = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        area_ratio = float(area / total)
        if area_ratio < float(min_area_ratio):
            continue
        # Restrict statistics to valid pixels: morphology (e.g. CLOSE) can dilate
        # the component into invalid pixels whose z (often a 0 sentinel) would
        # otherwise dominate std/mean/percentile readings.
        stat_mask = mask & valid_mask
        zvals = z[stat_mask]
        gvals = gradient[stat_mask]
        if zvals.size == 0:
            continue
        z_std = float(np.std(zvals))
        z_mad = float(np.median(np.abs(zvals - np.median(zvals))))
        gp95 = float(np.percentile(gvals, 95)) if gvals.size else 0.0
        touch = bool(np.any(mask & border))
        depth_pref = float(np.median(zvals))
        constancy = 1.0 / (1.0 + z_std + z_mad + gp95)
        score = (area_weight * area_ratio) + (constancy_weight * constancy) + (depth_pref_weight * depth_pref * 1e-3) + (border_bonus if touch else 0.0)
        if selection_mode.endswith("nearest_constant_z"):
            score -= depth_pref_weight * depth_pref * 1e-3
        row = {
            "label": int(label),
            "area_px": area,
            "area_ratio": area_ratio,
            "z_mean": float(np.mean(zvals)),
            "z_median": float(np.median(zvals)),
            "z_std": z_std,
            "z_mad": z_mad,
            "gradient_p95": gp95,
            "touches_border": touch,
            "score": float(score),
        }
        candidates.append(row)
        if score > best_score:
            best_score = float(score)
            best_label = int(label)
    return {"labels": labels, "candidates": candidates}, best_label


def _write_depth_plot_png(z_values: np.ndarray, *, near_q: float, far_q: float, out_path: Path) -> None:
    vals = np.asarray(z_values, dtype=np.float64)
    if vals.size == 0:
        canvas = np.zeros((420, 1200, 3), dtype=np.uint8)
        cv2.imwrite(str(out_path), canvas)
        return
    width, height = 1200, 420
    margin_l, margin_r, margin_t, margin_b = 60, 20, 20, 40
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    canvas = np.full((height, width, 3), 22, dtype=np.uint8)

    # deterministic downsample to keep rendering stable and fast
    if vals.size > 60000:
        step = int(np.ceil(vals.size / 60000))
        vals = vals[::step]
    vals_sorted = np.sort(vals)
    vmin = float(vals_sorted[0])
    vmax = float(vals_sorted[-1])
    if vmax <= vmin:
        vmax = vmin + 1.0

    xs = np.linspace(0, vals_sorted.size - 1, num=vals_sorted.size, dtype=np.float64)
    xpix = margin_l + (xs / max(1.0, float(vals_sorted.size - 1))) * plot_w
    ypix = margin_t + (1.0 - ((vals_sorted - vmin) / (vmax - vmin))) * plot_h
    pts = np.column_stack((xpix.astype(np.int32), ypix.astype(np.int32)))
    for p in pts:
        cv2.circle(canvas, (int(p[0]), int(p[1])), 1, (180, 180, 180), -1)

    def y_for(v: float) -> int:
        return int(round(margin_t + (1.0 - ((v - vmin) / (vmax - vmin))) * plot_h))

    y_near = y_for(near_q)
    y_far = y_for(far_q)
    cv2.line(canvas, (margin_l, y_near), (margin_l + plot_w, y_near), (255, 180, 0), 1)
    cv2.line(canvas, (margin_l, y_far), (margin_l + plot_w, y_far), (0, 220, 255), 1)
    cv2.rectangle(canvas, (margin_l, margin_t), (margin_l + plot_w, margin_t + plot_h), (90, 90, 90), 1)
    cv2.putText(canvas, f"Depth distribution (valid z), n={vals_sorted.size}", (margin_l, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"near_q={near_q:.1f}", (margin_l + 6, max(14, y_near - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 180, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"far_q={far_q:.1f}", (margin_l + 6, max(14, y_far - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"z_min={vmin:.1f}  z_max={vmax:.1f}", (margin_l, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)


def _height_color_image(values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    img = np.zeros(values.shape, dtype=np.uint8)
    mask = np.asarray(valid_mask, dtype=bool)
    if np.count_nonzero(mask) > 0:
        valid = values[mask]
        lo = float(np.nanpercentile(valid, 2.0))
        hi = float(np.nanpercentile(valid, 98.0))
        if hi <= lo:
            hi = lo + 1.0
        scaled = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
        img = np.asarray(np.round(scaled * 255.0), dtype=np.uint8)
    color = cv2.applyColorMap(img, cv2.COLORMAP_TURBO)
    color[~mask] = (0, 0, 0)
    return color


def _mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return (0.0, 0.0)
    return (float(np.mean(xs)), float(np.mean(ys)))


def _contour_metrics_in_metric_space(
    contour_px: list[list[float]] | None,
    *,
    x_resolution_mm: float,
    y_resolution_mm: float,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> dict[str, Any]:
    if not isinstance(contour_px, list) or len(contour_px) < 5:
        return {
            "valid": False,
            "perimeter_mm": None,
            "area_mm2": None,
            "major_axis_mm": None,
            "minor_axis_mm": None,
            "eccentricity": None,
            "roundness": None,
            "circularity": None,
            "ellipse_fit_mm": None,
            "covariance_axes_mm": None,
            "contour_mm": None,
        }
    pts = np.asarray(contour_px, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 2:
        return {"valid": False}
    mm = np.column_stack(
        (
            pts[:, 0] * float(x_resolution_mm) * float(scale_x),
            pts[:, 1] * float(y_resolution_mm) * float(scale_y),
        )
    ).astype(np.float32)
    mm_cv = mm.reshape((-1, 1, 2))
    perimeter_mm = float(cv2.arcLength(mm_cv, True))
    area_mm2 = float(abs(cv2.contourArea(mm_cv)))
    ellipse = cv2.fitEllipse(mm_cv)
    (_cx, _cy), (ax1, ax2), _angle = ellipse
    major = max(float(ax1), float(ax2))
    minor = min(float(ax1), float(ax2))
    eccentricity = float(np.sqrt(max(0.0, 1.0 - (minor * minor) / max(major * major, 1e-9))))
    roundness = float(minor / max(major, 1e-9))
    circularity = float((4.0 * np.pi * area_mm2) / max(perimeter_mm * perimeter_mm, 1e-9))
    cov = np.cov(mm.T)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    cov_major = float(2.0 * np.sqrt(max(evals[0], 0.0)))
    cov_minor = float(2.0 * np.sqrt(max(evals[1], 0.0)))
    return {
        "valid": True,
        "perimeter_mm": round(perimeter_mm, 4),
        "area_mm2": round(area_mm2, 4),
        "major_axis_mm": round(major, 4),
        "minor_axis_mm": round(minor, 4),
        "eccentricity": round(eccentricity, 4),
        "roundness": round(roundness, 4),
        "circularity": round(circularity, 4),
        "ellipse_fit_mm": {
            "major_axis_mm": round(major, 4),
            "minor_axis_mm": round(minor, 4),
            "eccentricity": round(eccentricity, 4),
            "roundness": round(roundness, 4),
            "circularity": round(circularity, 4),
        },
        "covariance_axes_mm": {
            "major_axis_mm": round(cov_major, 4),
            "minor_axis_mm": round(cov_minor, 4),
            "eigenvectors": evecs.astype(float).tolist(),
        },
        "contour_mm": mm.astype(float).tolist(),
    }


def _roi_from_metadata(metadata: dict[str, Any]) -> dict[str, Any] | None:
    roi = metadata.get("roi_25d")
    return roi if isinstance(roi, dict) else None


def _roi_mask(shape: tuple[int, int], roi: dict[str, Any]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    roi_type = str(roi.get("type") or roi.get("roi_type") or "rectangle").lower()
    if roi_type == "polygon":
        points = roi.get("points")
        if isinstance(points, list) and len(points) >= 3:
            poly = []
            for p in points:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    poly.append([int(round(float(p[0]))), int(round(float(p[1])))])
            if len(poly) >= 3:
                tmp = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(tmp, [np.asarray(poly, dtype=np.int32)], 255)
                return tmp > 0
    x = max(0, int(roi.get("x") or 0))
    rw = max(1, int(roi.get("width") or w))
    if roi_type == "vertical_band":
        y = 0
        rh = h
    else:
        y = max(0, int(roi.get("y") or 0))
        rh = max(1, int(roi.get("height") or h))
    x2 = min(w, x + rw)
    y2 = min(h, y + rh)
    mask[y:y2, x:x2] = True
    return mask


def _encoder_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    cal = metadata.get("calibration") if isinstance(metadata.get("calibration"), dict) else {}
    return {
        "encoder_ticks_per_mm": metadata.get("encoder_ticks_per_mm") or metadata.get("encoder_ticks") or cal.get("encoder_ticks_per_mm"),
        "scan_direction": metadata.get("scan_direction") or "forward",
        "profile_distance_mm": cal.get("profile_distance_mm"),
        "belt_speed_mm_s": metadata.get("belt_speed_mm_s") or metadata.get("conveyor_speed"),
    }


def _classify_25d(item: dict[str, Any]) -> tuple[str, str, str, float]:
    metrics = item.get("height_above_belt_mm") or {}
    max_h = float(metrics.get("max_height_mm") or 0.0)
    p95_h = float(metrics.get("p95_height_mm") or 0.0)
    ecc = float(item.get("feature_eccentricity") or 0.0)
    sph3d = float(item.get("feature_sphericity_3d") or 0.0)
    flat = float(item.get("feature_flatness") or 0.0)
    rough = float(item.get("feature_edge_roughness") or 0.0)
    vol = float(item.get("feature_volume_proxy_mm3") or 0.0)
    if 8.0 <= max_h <= 95.0 and sph3d >= 0.8 and ecc < 0.45 and flat > 0.15 and rough < 8.0:
        return "bola_buena", "Bola buena", "ball", 0.82
    if ecc >= 0.75 or flat < -0.2 or rough > 12.0:
        return "bola_deformada", "Scrap de Bola - Bola deformada", "non_ball", 0.78
    if max_h < 6.0 or p95_h < 4.0 or vol < 4000.0:
        return "planchuela", "Chatarra - Planchuelas", "non_ball", 0.72
    return "bola_con_chip", "Scrap de Bola - Bola con chip", "non_ball", 0.66


def _metric(item: dict[str, Any], key: str) -> float | None:
    if key == "max_height_mm":
        return _safe_float(((item.get("height_above_belt_mm") or {}).get("max_height_mm")) if isinstance(item.get("height_above_belt_mm"), dict) else None)
    if key == "p95_height_mm":
        return _safe_float(((item.get("height_above_belt_mm") or {}).get("p95_height_mm")) if isinstance(item.get("height_above_belt_mm"), dict) else None)
    if key == "mean_height_mm":
        return _safe_float(((item.get("height_above_belt_mm") or {}).get("mean_height_mm")) if isinstance(item.get("height_above_belt_mm"), dict) else None)
    return _safe_float(item.get(key))


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _rule(
    *,
    rule_id: str,
    label: str,
    description: str,
    metric_key: str,
    value: float | None,
    comparator: str,
    expected: dict[str, float] | None,
    passed: bool,
    severity: str,
    contribution: str,
    message: str,
    raw_value: float | None = None,
    corrected_value: float | None = None,
    correction_scale: float | None = None,
    metric_source: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "rule_id": rule_id,
        "label": label,
        "description": description,
        "metric_key": metric_key,
        "value": value,
        "comparator": comparator,
        "passed": bool(passed),
        "severity": severity,
        "contribution": contribution,
        "message": message,
    }
    if expected:
        payload["expected_range"] = expected
        payload["threshold"] = expected
    payload["raw_value"] = raw_value
    payload["corrected_value"] = corrected_value if corrected_value is not None else value
    payload["correction_scale"] = correction_scale
    payload["metric_source"] = metric_source or "object_metrics"
    return payload


def _raw_metric(item: dict[str, Any], key: str) -> float | None:
    raw = item.get("raw_metrics")
    if not isinstance(raw, dict):
        return None
    if key in ("max_height_mm", "mean_height_mm", "p95_height_mm"):
        heights = raw.get("height_above_belt_mm")
        if isinstance(heights, dict):
            return _safe_float(heights.get(key))
        return None
    return _safe_float(raw.get(key))


def _compose_diameter_metrics(item: dict[str, Any]) -> dict[str, Any]:
    footprint_area = _safe_float(item.get("footprint_area_mm2"))
    circumference = _safe_float(item.get("perimeter_mm"))
    major = _safe_float(item.get("major_axis_mm"))
    minor = _safe_float(item.get("minor_axis_mm"))
    dims = item.get("dimensions_mm") if isinstance(item.get("dimensions_mm"), (list, tuple)) else None
    dim_x = _safe_float(dims[0] if isinstance(dims, (list, tuple)) and len(dims) >= 1 else major)
    dim_y = _safe_float(dims[1] if isinstance(dims, (list, tuple)) and len(dims) >= 2 else minor)
    dim_z = _safe_float(dims[2] if isinstance(dims, (list, tuple)) and len(dims) >= 3 else ((item.get("height_above_belt_mm") or {}).get("max_height_mm") if isinstance(item.get("height_above_belt_mm"), dict) else None))
    diameter_ellipse = max(v for v in (major, minor) if v is not None) if (major is not None or minor is not None) else None
    diameter_equivalent_area = (2.0 * float(np.sqrt(max(0.0, footprint_area) / np.pi))) if footprint_area is not None and footprint_area > 0 else None
    diameter_circumference = (circumference / np.pi) if circumference is not None and circumference > 0 else None
    diameter_selected = _safe_float(item.get("diameter_mm"))
    selected_source = "legacy_diameter_mm"
    if diameter_selected is not None:
        if diameter_ellipse is not None and abs(diameter_selected - diameter_ellipse) < 1e-3:
            selected_source = "ellipse"
        elif diameter_equivalent_area is not None and abs(diameter_selected - diameter_equivalent_area) < 1e-3:
            selected_source = "equivalent_area"
        elif diameter_circumference is not None and abs(diameter_selected - diameter_circumference) < 1e-3:
            selected_source = "circumference"
    max_dim_xy = max(v for v in (dim_x, dim_y) if v is not None) if (dim_x is not None or dim_y is not None) else None
    ratio = (diameter_selected / max_dim_xy) if diameter_selected is not None and max_dim_xy is not None and max_dim_xy > 1e-9 else None
    sanity_status = "ok"
    sanity_message = "Diameter is consistent with XY dimensions."
    if ratio is not None and ratio > 1.8:
        sanity_status = "suspicious"
        sanity_message = f"Selected diameter ({diameter_selected:.2f}mm) is inconsistent with max(dim_x,dim_y) ({max_dim_xy:.2f}mm); ratio={ratio:.2f}."
    if ratio is not None and ratio > 2.5:
        sanity_status = "invalid"
        sanity_message = f"Selected diameter appears impossible for object footprint; ratio={ratio:.2f}."
    if sanity_status in ("suspicious", "invalid"):
        if diameter_ellipse is not None:
            diameter_selected = diameter_ellipse
            selected_source = "ellipse_fallback_from_sanity"
        elif max_dim_xy is not None:
            diameter_selected = max_dim_xy
            selected_source = "dim_xy_fallback_from_sanity"
    return {
        "dim_x_mm": dim_x,
        "dim_y_mm": dim_y,
        "dim_z_mm": dim_z,
        "diameter_ellipse_mm": diameter_ellipse,
        "diameter_equivalent_area_mm": diameter_equivalent_area,
        "diameter_circumference_mm": diameter_circumference,
        "diameter_selected_mm": diameter_selected,
        "diameter_selected_source": selected_source,
        "diameter_sanity_status": sanity_status,
        "diameter_sanity_message": sanity_message,
    }


def _classification_explanation_for_object(item: dict[str, Any]) -> dict[str, Any]:
    final_class_name = str(item.get("class_name") or "unknown")
    final_class_label = str(item.get("label") or item.get("display_label") or "unknown")
    superclass = str(item.get("superclass") or "UNKNOWN")
    subclass = item.get("subclass_label")
    confidence = _safe_float(item.get("confidence"))
    diameter = _compose_diameter_metrics(item)
    dim_x = diameter["dim_x_mm"]
    dim_y = diameter["dim_y_mm"]
    dim_z = diameter["dim_z_mm"]
    max_dim_xy = max(v for v in (dim_x, dim_y) if v is not None) if (dim_x is not None or dim_y is not None) else None
    selected_d = diameter["diameter_selected_mm"]
    ratio = (selected_d / max_dim_xy) if selected_d is not None and max_dim_xy is not None and max_dim_xy > 1e-9 else None
    # Preserve the original (pre-fallback) diameter so the explanation rule can
    # still surface a "Suspicious diameter mismatch" even after _compose_diameter_metrics
    # has swapped in a sane fallback for downstream classification.
    raw_diameter_for_sanity = _safe_float(item.get("diameter_mm"))
    raw_ratio = (
        raw_diameter_for_sanity / max_dim_xy
        if raw_diameter_for_sanity is not None and max_dim_xy is not None and max_dim_xy > 1e-9
        else None
    )
    diameter_sanity_status = str(diameter.get("diameter_sanity_status") or "ok")
    diameter_sanity_message = str(diameter.get("diameter_sanity_message") or "")

    rules: list[dict[str, Any]] = []
    max_h = _metric(item, "max_height_mm")
    p95_h = _metric(item, "p95_height_mm")
    ecc = _metric(item, "feature_eccentricity")
    sph3d = _metric(item, "feature_sphericity_3d")
    flat = _metric(item, "feature_flatness")
    rough = _metric(item, "feature_edge_roughness")
    vol = _metric(item, "feature_volume_proxy_mm3")
    roundness = _metric(item, "sphericity_score")
    deformation = _metric(item, "feature_local_curvature_proxy")
    invariant_warnings = item.get("geometry_invariant_warnings") if isinstance(item.get("geometry_invariant_warnings"), list) else []
    footprint_area = _metric(item, "footprint_area_mm2")
    trunc = _safe_float(((item.get("sphere_fit_diagnostics") or {}).get("estimated_truncation_ratio")) if isinstance(item.get("sphere_fit_diagnostics"), dict) else None)
    fit_rmse = _metric(item, "fit_rmse_mm")
    scale = (item.get("scale_correction_applied") or {}) if isinstance(item.get("scale_correction_applied"), dict) else {}
    scale_x = _safe_float(scale.get("x"))
    scale_y = _safe_float(scale.get("y"))
    scale_z = _safe_float(scale.get("z"))

    rules.extend(
        [
            _rule(
                rule_id="shape.sphericity_3d",
                label="3D sphericity",
                description="3D axis balance should be high for spherical balls.",
                metric_key="feature_sphericity_3d",
                value=sph3d,
                comparator=">=",
                expected={"min": 0.8},
                passed=(sph3d is not None and sph3d >= 0.8),
                severity="critical",
                contribution="positive" if (sph3d is not None and sph3d >= 0.8) else "negative",
                message="Sphericity supports ball classification." if (sph3d is not None and sph3d >= 0.8) else "Low 3D sphericity pushes toward non-ball.",
            ),
            _rule(
                rule_id="shape.eccentricity",
                label="Eccentricity",
                description="Projected eccentricity should stay low for near-round objects.",
                metric_key="feature_eccentricity",
                value=ecc,
                comparator="<",
                expected={"max": 0.45},
                passed=(ecc is not None and ecc < 0.45),
                severity="critical",
                contribution="positive" if (ecc is not None and ecc < 0.45) else "negative",
                message="Low eccentricity supports round object." if (ecc is not None and ecc < 0.45) else "High eccentricity indicates deformation/non-ball.",
            ),
            _rule(
                rule_id="shape.roundness",
                label="Roundness score",
                description="Roundness from footprint/axes should not collapse for true balls.",
                metric_key="sphericity_score",
                value=roundness,
                comparator=">=",
                expected={"min": 0.75},
                passed=(roundness is not None and roundness >= 0.75),
                severity="critical",
                contribution="positive" if (roundness is not None and roundness >= 0.75) else "negative",
                message="Roundness supports ball." if (roundness is not None and roundness >= 0.75) else "Very low roundness is decisive non-ball evidence.",
            ),
            _rule(
                rule_id="height.max",
                label="Max height band",
                description="Ball-like objects should appear within expected height range.",
                metric_key="max_height_mm",
                value=max_h,
                comparator="between",
                expected={"min": 8.0, "max": 95.0},
                passed=(max_h is not None and 8.0 <= max_h <= 95.0),
                severity="info",
                contribution="positive" if (max_h is not None and 8.0 <= max_h <= 95.0) else "neutral",
                message="Height in valid band." if (max_h is not None and 8.0 <= max_h <= 95.0) else "Height outside preferred ball range.",
            ),
            _rule(
                rule_id="shape.flatness",
                label="Flatness",
                description="Flatness underflow indicates truncated or non-spherical shape.",
                metric_key="feature_flatness",
                value=flat,
                comparator=">",
                expected={"min": 0.15},
                passed=(flat is not None and flat > 0.15),
                severity="warning",
                contribution="positive" if (flat is not None and flat > 0.15) else "negative",
                message="Flatness supports full body shape." if (flat is not None and flat > 0.15) else "Low flatness suggests chip/truncation.",
            ),
            _rule(
                rule_id="shape.roughness",
                label="Edge roughness",
                description="Excessive roughness is a defect/scrap indicator.",
                metric_key="feature_edge_roughness",
                value=rough,
                comparator="<",
                expected={"max": 8.0},
                passed=(rough is not None and rough < 8.0),
                severity="warning",
                contribution="positive" if (rough is not None and rough < 8.0) else "negative",
                message="Roughness within expected band." if (rough is not None and rough < 8.0) else "High roughness supports non-ball/deformed class.",
            ),
            _rule(
                rule_id="volume.minimum",
                label="Volume floor",
                description="Very low volume is consistent with flat scraps.",
                metric_key="feature_volume_proxy_mm3",
                value=vol,
                comparator=">=",
                expected={"min": 4000.0},
                passed=(vol is not None and vol >= 4000.0),
                severity="warning",
                contribution="positive" if (vol is not None and vol >= 4000.0) else "negative",
                message="Volume supports ball-like mass." if (vol is not None and vol >= 4000.0) else "Low volume is non-ball evidence.",
            ),
            _rule(
                rule_id="height.p95_floor",
                label="Height P95 floor",
                description="Low p95 height indicates flattened profile.",
                metric_key="p95_height_mm",
                value=p95_h,
                comparator=">=",
                expected={"min": 4.0},
                passed=(p95_h is not None and p95_h >= 4.0),
                severity="warning",
                contribution="positive" if (p95_h is not None and p95_h >= 4.0) else "negative",
                message="Height distribution supports object body." if (p95_h is not None and p95_h >= 4.0) else "Low p95 height indicates planchuela-like profile.",
            ),
        ]
    )

    rules.extend(
        [
            _rule(
                rule_id="sanity.footprint_area_present",
                label="Footprint area present",
                description="Footprint area must exist for diameter cross-checks.",
                metric_key="footprint_area_mm2",
                value=footprint_area,
                comparator=">",
                expected={"min": 0.0},
                passed=(footprint_area is not None and footprint_area > 0.0),
                severity="warning",
                contribution="neutral" if (footprint_area is not None and footprint_area > 0.0) else "negative",
                message="Footprint area available." if (footprint_area is not None and footprint_area > 0.0) else "Missing footprint area disables reliable equivalent-diameter checks.",
            ),
            _rule(
                rule_id="sanity.ellipse_fit_valid",
                label="Ellipse fit validity",
                description="Invalid ellipse fit reduces trust in ellipse-derived diameter.",
                metric_key="fit_rmse_mm",
                value=fit_rmse,
                comparator="<=",
                expected={"max": 5.0},
                passed=(fit_rmse is None or fit_rmse <= 5.0),
                severity="warning",
                contribution="neutral" if (fit_rmse is None or fit_rmse <= 5.0) else "negative",
                message="Ellipse fit quality acceptable." if (fit_rmse is None or fit_rmse <= 5.0) else "High fit RMSE indicates unstable ellipse metrics.",
            ),
            _rule(
                rule_id="sanity.scale_factors",
                label="Scale factors",
                description="Reports the calibration scale envelope; aggressive scales are informational once a calibrated correction is applied.",
                metric_key="scale_correction_applied",
                value=max(v for v in (scale_x, scale_y, scale_z) if v is not None) if (scale_x is not None or scale_y is not None or scale_z is not None) else None,
                comparator="between",
                expected={"min": 0.5, "max": 2.0},
                # Once a contour-based corrected metric geometry has been produced
                # (geometry_coordinate_space == corrected_metric_mm), the post-
                # correction metrics live in true mm space regardless of how
                # aggressive the per-axis raw->corrected scales are. Don't fail
                # this rule purely because the calibration chain is anisotropic.
                passed=(
                    (scale_x is None and scale_y is None and scale_z is None)
                    or str(item.get("geometry_coordinate_space") or "") == "corrected_metric_mm"
                    or all(0.5 <= v <= 2.0 for v in (scale_x, scale_y, scale_z) if v is not None)
                ),
                severity="info",
                contribution="neutral",
                message=(
                    "Scale factors within expected envelope."
                    if (
                        (scale_x is None and scale_y is None and scale_z is None)
                        or all(0.5 <= v <= 2.0 for v in (scale_x, scale_y, scale_z) if v is not None)
                    )
                    else "Calibrated scale factors are aggressive; metrics are computed in corrected mm space."
                    if str(item.get("geometry_coordinate_space") or "") == "corrected_metric_mm"
                    else "Suspicious scale correction factors detected; inspect calibration chain."
                ),
            ),
            _rule(
                rule_id="sanity.diameter_vs_dims",
                label="Diameter consistency",
                description="Selected diameter should be consistent with measured XY footprint dimensions.",
                metric_key="diameter_selected_mm",
                value=selected_d,
                comparator="ratio_to_max_dim_xy",
                expected={"max": 1.8},
                # Surface the suspicion even when _compose_diameter_metrics has
                # already replaced the bad selected diameter with a sane
                # ellipse_fallback_from_sanity: the raw ratio still tells us
                # the calibration / measurement chain produced an inconsistent
                # diameter and that information must reach the explanation UI.
                passed=(
                    diameter_sanity_status == "ok"
                    and (ratio is None or ratio <= 1.8)
                    and (raw_ratio is None or raw_ratio <= 1.8)
                ),
                severity=(
                    "critical"
                    if (
                        (ratio is not None and ratio > 2.5)
                        or (raw_ratio is not None and raw_ratio > 2.5)
                        or diameter_sanity_status == "invalid"
                    )
                    else "warning"
                ),
                contribution=(
                    "negative"
                    if (
                        diameter_sanity_status != "ok"
                        or (ratio is not None and ratio > 1.8)
                        or (raw_ratio is not None and raw_ratio > 1.8)
                    )
                    else "neutral"
                ),
                message=(
                    f"Suspicious diameter mismatch: raw diameter={raw_diameter_for_sanity:.2f}mm vs max(dim_x,dim_y)={max_dim_xy:.2f}mm (ratio={raw_ratio:.2f}). {diameter_sanity_message}".strip()
                    if (
                        raw_ratio is not None and raw_ratio > 1.8
                        and raw_diameter_for_sanity is not None
                        and max_dim_xy is not None
                    )
                    else (
                        f"Suspicious diameter mismatch: selected={selected_d:.2f}mm vs max(dim_x,dim_y)={max_dim_xy:.2f}mm (ratio={ratio:.2f})."
                        if ratio is not None and ratio > 1.8 and selected_d is not None and max_dim_xy is not None
                        else "Diameter and footprint dimensions are broadly consistent."
                    )
                ),
            ),
            _rule(
                rule_id="sanity.extremely_low_shape_metrics",
                label="Extremely low shape metrics",
                description="Near-zero roundness/sphericity strongly indicates non-ball or broken metrics.",
                metric_key="shape_low_guard",
                value=min(v for v in (sph3d, roundness) if v is not None) if (sph3d is not None or roundness is not None) else None,
                comparator=">=",
                expected={"min": 0.05},
                passed=(
                    (sph3d is None and roundness is None)
                    or ((sph3d if sph3d is not None else 1.0) >= 0.05 and (roundness if roundness is not None else 1.0) >= 0.05)
                ),
                severity="critical",
                contribution="negative"
                if ((sph3d is not None and sph3d < 0.05) or (roundness is not None and roundness < 0.05))
                else "neutral",
                message="Shape metrics above extreme-failure floor."
                if not ((sph3d is not None and sph3d < 0.05) or (roundness is not None and roundness < 0.05))
                else "Roundness/sphericity are extremely low and are decisive non-ball evidence.",
            ),
        ]
    )

    if trunc is not None:
        rules.append(
            _rule(
                rule_id="shape.truncation_ratio",
                label="Truncation ratio",
                description="Estimated missing lower hemisphere from sphere-fit diagnostics.",
                metric_key="estimated_truncation_ratio",
                value=trunc,
                comparator="<=",
                expected={"max": 0.2},
                passed=trunc <= 0.2,
                severity="warning",
                contribution="negative" if trunc > 0.2 else "neutral",
                message="Truncation within expected range." if trunc <= 0.2 else "High truncation suggests chip/cut object.",
            )
        )
    if deformation is not None:
        # Curvature proxy is sensitive to anisotropic raw spacing and to mask
        # boundary artifacts. When the primary shape evidence already strongly
        # supports a ball (high 3D sphericity, high roundness, low eccentricity),
        # demote deformation to informational so it cannot single-handedly flip
        # the classification of an obviously round object.
        ball_shape_consensus = (
            sph3d is not None and sph3d >= 0.8
            and roundness is not None and roundness >= 0.75
            and ecc is not None and ecc < 0.45
        )
        rules.append(
            _rule(
                rule_id="shape.deformation_score",
                label="Deformation score",
                description="Curvature proxy used as deformation evidence (in corrected metric space).",
                metric_key="feature_local_curvature_proxy",
                value=deformation,
                comparator="<=",
                expected={"max": 12.0},
                passed=(deformation <= 12.0) or ball_shape_consensus,
                severity="info" if ball_shape_consensus else "warning",
                contribution=(
                    "neutral" if (deformation <= 12.0 or ball_shape_consensus) else "negative"
                ),
                message=(
                    "Deformation score is within expected limits."
                    if deformation <= 12.0
                    else "Deformation score is elevated but overall shape metrics are ball-consistent; treated as informational."
                    if ball_shape_consensus
                    else "High deformation score supports non-ball/deformed class."
                ),
            )
        )
    for idx, warning in enumerate(invariant_warnings):
        warning_text = str(warning)
        # Entries prefixed with "warning:" are advisory (e.g. anisotropic raw
        # scaling). Entries prefixed with "invariant_violation:" represent
        # internal inconsistency (e.g. near-equal dims with extreme eccentricity).
        is_violation = warning_text.startswith("invariant_violation:")
        rules.append(
            _rule(
                rule_id=f"sanity.geometry_invariant_{idx+1}",
                label="Geometry invariant warning" if is_violation else "Geometry invariant info",
                description="Invariant check over corrected metric geometry.",
                metric_key="geometry_invariant_warnings",
                value=None,
                comparator="warning",
                expected=None,
                passed=not is_violation,
                severity="warning" if is_violation else "info",
                contribution="negative" if is_violation else "neutral",
                message=warning_text,
            )
        )

    scale_ctx = (item.get("scale_correction_applied") or {}) if isinstance(item.get("scale_correction_applied"), dict) else {}
    sx = _safe_float(scale_ctx.get("x"))
    sy = _safe_float(scale_ctx.get("y"))
    sz = _safe_float(scale_ctx.get("z"))
    for rule in rules:
        key = str(rule.get("metric_key") or "")
        raw_value: float | None = None
        corrected_value: float | None = _safe_float(rule.get("value"))
        correction_scale: float | None = None
        if key in ("feature_sphericity_3d", "sphericity_score", "feature_eccentricity", "feature_flatness"):
            raw_value = _raw_metric(item, key)
            correction_scale = 1.0
        elif key in ("max_height_mm", "mean_height_mm", "p95_height_mm", "feature_edge_roughness"):
            raw_value = _raw_metric(item, key)
            correction_scale = sz
        elif key in ("feature_volume_proxy_mm3",):
            raw_value = _raw_metric(item, key)
            correction_scale = (sx * sy * sz) if sx is not None and sy is not None and sz is not None else None
        elif key in ("footprint_area_mm2",):
            raw_value = _raw_metric(item, key)
            correction_scale = (sx * sy) if sx is not None and sy is not None else None
        elif key in ("diameter_selected_mm",):
            raw_value = _raw_metric(item, "diameter_mm")
            correction_scale = sx
        elif key in ("feature_local_curvature_proxy", "fit_rmse_mm", "estimated_truncation_ratio"):
            raw_value = _raw_metric(item, key)
            correction_scale = 1.0
        rule["raw_value"] = raw_value
        rule["corrected_value"] = corrected_value
        rule["correction_scale"] = correction_scale
        rule["metric_source"] = "corrected_metrics" if str(item.get("metrics_coordinate_space") or "") == "corrected_mm" else "raw_metrics"

    decisive = [
        rule for rule in rules
        if (
            (rule["contribution"] == "negative" and not rule["passed"])
            or (rule["contribution"] == "positive" and rule["passed"])
        )
    ]
    if not decisive:
        fallback = next((r for r in rules if r["rule_id"] == "shape.sphericity_3d"), rules[0])
        fallback["contribution"] = "negative" if final_class_name != "ball" else "positive"
        decisive = [fallback]

    decision_summary = {
        "decisive_rule_ids": [str(rule["rule_id"]) for rule in decisive[:3]],
        "critical_failures": [str(rule["rule_id"]) for rule in rules if rule["severity"] == "critical" and not bool(rule["passed"])],
        "suspicious_metrics": [str(rule["rule_id"]) for rule in rules if rule["rule_id"].startswith("sanity.") and not bool(rule["passed"])],
        "final_decision_path": f"{' / '.join([str(rule['rule_id']) for rule in decisive[:3]])} => {final_class_name}",
    }

    metrics_used = {
        "max_height_mm": max_h,
        "mean_height_mm": _metric(item, "mean_height_mm"),
        "p95_height_mm": p95_h,
        "feature_eccentricity": ecc,
        "feature_sphericity_3d": sph3d,
        "sphericity_score": roundness,
        "feature_flatness": flat,
        "feature_edge_roughness": rough,
        "feature_volume_proxy_mm3": vol,
        "feature_local_curvature_proxy": deformation,
        "geometry_invariant_warnings": invariant_warnings,
        "fit_rmse_mm": fit_rmse,
        "estimated_truncation_ratio": trunc,
        **diameter,
    }

    metric_trace = _build_metric_trace_for_object(item, rules)
    trace_by_key = {str(row.get("metric_key") or ""): row for row in metric_trace}
    for rule in rules:
        key = str(rule.get("metric_key") or "")
        trace = trace_by_key.get(key)
        if trace is not None:
            rule["metric_trace_ref"] = {
                "metric_key": key,
                "trace_id": trace.get("trace_id"),
            }
    return {
        "object_id": int(item.get("object_id") or 0),
        "final_class_name": final_class_name,
        "final_class_label": final_class_label,
        "superclass": superclass,
        "subclass": subclass,
        "confidence": confidence,
        "metrics_coordinate_space": str(item.get("metrics_coordinate_space") or "raw_mm"),
        "correction_source": item.get("correction_source"),
        "correction_context_id": item.get("correction_context_id"),
        "correction_scales": item.get("scale_correction_applied"),
        "decision_summary": decision_summary,
        "metrics_used": metrics_used,
        "metric_trace": metric_trace,
        "rules": rules,
    }


def _build_metric_trace_for_object(item: dict[str, Any], rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scale = item.get("scale_correction_applied") if isinstance(item.get("scale_correction_applied"), dict) else {}
    sx = _safe_float(scale.get("x"))
    sy = _safe_float(scale.get("y"))
    sz = _safe_float(scale.get("z"))
    corrected_space = str(item.get("metrics_coordinate_space") or "raw_mm")
    raw = item.get("raw_metrics") if isinstance(item.get("raw_metrics"), dict) else {}
    geom = item.get("geometry_debug") if isinstance(item.get("geometry_debug"), dict) else {}
    used_keys = {str(rule.get("metric_key") or "") for rule in rules}
    dims = item.get("dimensions_mm") if isinstance(item.get("dimensions_mm"), (list, tuple)) else None
    dim_x = float(dims[0]) if dims is not None and len(dims) >= 1 and dims[0] is not None else None
    dim_y = float(dims[1]) if dims is not None and len(dims) >= 2 and dims[1] is not None else None
    dim_z = float(dims[2]) if dims is not None and len(dims) >= 3 and dims[2] is not None else None
    invariants = item.get("geometry_invariant_warnings") if isinstance(item.get("geometry_invariant_warnings"), list) else []
    raw_heights = (raw.get("height_above_belt_mm") if isinstance(raw, dict) else None) or {}
    corrected_heights = item.get("height_above_belt_mm") if isinstance(item.get("height_above_belt_mm"), dict) else {}
    curv_raw = item.get("feature_curvature_components_raw") if isinstance(item.get("feature_curvature_components_raw"), dict) else {}
    curv_corrected = item.get("feature_curvature_components_corrected") if isinstance(item.get("feature_curvature_components_corrected"), dict) else {}
    # If a known-cube correction was applied AND a contour-based corrected
    # geometry was produced, planar metrics are sourced from
    # `contour_corrected`. Otherwise we fall back to bbox-scaled extents or
    # raw measurement output.
    if geom and bool(((geom.get("corrected") or {}) if isinstance(geom.get("corrected"), dict) else {}).get("valid")):
        geometry_metric_source = "contour_corrected"
    elif scale:
        geometry_metric_source = "bbox_corrected"
    else:
        geometry_metric_source = "raw_measurement"

    def _input(name: str, value: Any) -> dict[str, Any]:
        return {"name": name, "value": value}

    # Per-row specs:
    #   (metric_key, final_value, formula_name, formula_human_readable,
    #    formula_inputs, intermediate_values, warnings,
    #    correction_factor_used, source_artifact_id)
    metric_specs: list[tuple[str, Any, str, str, list[dict[str, Any]], dict[str, Any], list[Any], float | None, str]] = [
        (
            "feature_eccentricity",
            item.get("feature_eccentricity"),
            "ellipse_eccentricity_from_corrected_contour",
            "sqrt(1 - (minor_axis_mm^2 / major_axis_mm^2))  // axes in corrected metric mm",
            [
                _input("major_axis_mm", item.get("major_axis_mm")),
                _input("minor_axis_mm", item.get("minor_axis_mm")),
            ],
            {
                "major_axis_mm": item.get("major_axis_mm"),
                "minor_axis_mm": item.get("minor_axis_mm"),
                "circularity": item.get("feature_circularity"),
                "perimeter_mm": item.get("perimeter_mm"),
                "area_mm2": item.get("footprint_area_mm2"),
                "geometry_metric_source": geometry_metric_source,
            },
            list(invariants),
            None,
            "geometry_debug_summary" if geom else "measurement_object",
        ),
        (
            "sphericity_score",
            item.get("sphericity_score"),
            "roundness_ratio_from_corrected_axes",
            "minor_axis_mm / major_axis_mm  // axes in corrected metric mm",
            [
                _input("major_axis_mm", item.get("major_axis_mm")),
                _input("minor_axis_mm", item.get("minor_axis_mm")),
            ],
            {
                "major_axis_mm": item.get("major_axis_mm"),
                "minor_axis_mm": item.get("minor_axis_mm"),
                "feature_footprint_roundness": item.get("feature_footprint_roundness"),
                "geometry_metric_source": geometry_metric_source,
            },
            [],
            None,
            "geometry_debug_summary" if geom else "measurement_object",
        ),
        (
            "feature_sphericity_3d",
            item.get("feature_sphericity_3d"),
            "axis_balance_3d_from_corrected_dimensions",
            "min(dim_x_mm, dim_y_mm, dim_z_mm) / max(dim_x_mm, dim_y_mm, dim_z_mm)  // corrected XYZ extents",
            [
                _input("dim_x_mm", dim_x),
                _input("dim_y_mm", dim_y),
                _input("dim_z_mm", dim_z),
            ],
            {
                "min_axis_mm": min(v for v in (dim_x, dim_y, dim_z) if v is not None) if any(v is not None for v in (dim_x, dim_y, dim_z)) else None,
                "max_axis_mm": max(v for v in (dim_x, dim_y, dim_z) if v is not None) if any(v is not None for v in (dim_x, dim_y, dim_z)) else None,
                "axis_ratio": item.get("feature_sphericity_3d"),
                "geometry_metric_source": geometry_metric_source,
            },
            [],
            None,
            "measurement_object",
        ),
        (
            "diameter_selected_mm",
            item.get("diameter_selected_mm") or item.get("diameter_mm"),
            "diameter_selection_with_sanity",
            "select(best of ellipse/equivalent_area/circumference) with sanity fallback when diameter is incoherent with dim_x/dim_y",
            [
                _input("diameter_ellipse_mm", item.get("diameter_ellipse_mm")),
                _input("diameter_equivalent_area_mm", item.get("diameter_equivalent_area_mm")),
                _input("diameter_circumference_mm", item.get("diameter_circumference_mm")),
                _input("dim_x_mm", dim_x),
                _input("dim_y_mm", dim_y),
            ],
            {
                "diameter_selected_source": item.get("diameter_selected_source"),
                "diameter_sanity_status": item.get("diameter_sanity_status"),
                "diameter_selected_mm": item.get("diameter_selected_mm"),
            },
            [item.get("diameter_sanity_message")] if item.get("diameter_sanity_message") else [],
            sx,
            "measurement_object",
        ),
        (
            "feature_local_curvature_proxy",
            item.get("feature_local_curvature_proxy"),
            "anisotropic_curvature_proxy_in_corrected_mm" if curv_raw else "isotropic_curvature_proxy_scale_z",
            (
                "mean_abs_gx_raw * (scale_z / scale_x) + mean_abs_gy_raw * (scale_z / scale_y)  // per-axis correction; gradients sampled on inner-eroded mask"
                if curv_raw
                else "feature_local_curvature_proxy_raw * scale_z  // fallback when per-axis gradient components were not stored"
            ),
            [
                _input("mean_abs_gx_mm_per_mm_raw", curv_raw.get("mean_abs_gx_mm_per_mm") if isinstance(curv_raw, dict) else None),
                _input("mean_abs_gy_mm_per_mm_raw", curv_raw.get("mean_abs_gy_mm_per_mm") if isinstance(curv_raw, dict) else None),
                _input("scale_x", sx),
                _input("scale_y", sy),
                _input("scale_z", sz),
            ],
            {
                "mean_abs_gx_mm_per_mm_corrected": curv_corrected.get("mean_abs_gx_mm_per_mm") if isinstance(curv_corrected, dict) else None,
                "mean_abs_gy_mm_per_mm_corrected": curv_corrected.get("mean_abs_gy_mm_per_mm") if isinstance(curv_corrected, dict) else None,
                "scale_z_over_scale_x": curv_corrected.get("scale_z_over_scale_x") if isinstance(curv_corrected, dict) else None,
                "scale_z_over_scale_y": curv_corrected.get("scale_z_over_scale_y") if isinstance(curv_corrected, dict) else None,
                "computed_over": curv_raw.get("computed_over") if isinstance(curv_raw, dict) else None,
            },
            [],
            None,
            "measurement_object",
        ),
        (
            "max_height_mm",
            corrected_heights.get("max_height_mm") if isinstance(corrected_heights, dict) else None,
            "max_height_mm_corrected",
            "max(height_above_belt_mm) * scale_z  // applied during known-cube correction",
            [
                _input("max_height_mm_raw", raw_heights.get("max_height_mm")),
                _input("scale_z", sz),
            ],
            {
                "p95_height_mm": corrected_heights.get("p95_height_mm") if isinstance(corrected_heights, dict) else None,
                "mean_height_mm": corrected_heights.get("mean_height_mm") if isinstance(corrected_heights, dict) else None,
            },
            [],
            sz,
            "measurement_object",
        ),
        (
            "footprint_area_mm2",
            item.get("footprint_area_mm2"),
            "footprint_area_mm2_corrected",
            "footprint_area_raw * scale_x * scale_y  // or recomputed from corrected contour when available",
            [
                _input("footprint_area_raw", raw.get("footprint_area_mm2") if isinstance(raw, dict) else None),
                _input("scale_x", sx),
                _input("scale_y", sy),
            ],
            {
                "scale_xy_product": (sx * sy) if sx is not None and sy is not None else None,
                "geometry_metric_source": geometry_metric_source,
            },
            [],
            (sx * sy) if sx is not None and sy is not None else None,
            "geometry_debug_summary" if geom else "measurement_object",
        ),
        (
            "feature_volume_proxy_mm3",
            item.get("feature_volume_proxy_mm3"),
            "volume_proxy_mm3_corrected",
            "volume_proxy_raw * scale_x * scale_y * scale_z",
            [
                _input("volume_proxy_raw", raw.get("feature_volume_proxy_mm3") if isinstance(raw, dict) else None),
                _input("scale_x", sx),
                _input("scale_y", sy),
                _input("scale_z", sz),
            ],
            {
                "scale_xyz_product": (sx * sy * sz) if sx is not None and sy is not None and sz is not None else None,
            },
            [],
            (sx * sy * sz) if sx is not None and sy is not None and sz is not None else None,
            "measurement_object",
        ),
    ]

    trace_rows: list[dict[str, Any]] = []
    for idx, (
        metric_key,
        final_value,
        formula_name,
        formula_hr,
        formula_inputs,
        intermediate,
        warnings,
        correction_factor_used,
        source_artifact_id,
    ) in enumerate(metric_specs, start=1):
        rv = raw.get(metric_key) if isinstance(raw, dict) else None
        if rv is None and metric_key == "diameter_selected_mm" and isinstance(raw, dict):
            rv = raw.get("diameter_mm")
        if rv is None and metric_key == "max_height_mm" and isinstance(raw_heights, dict):
            rv = raw_heights.get("max_height_mm")
        cspace_before = "raw_metric_mm" if isinstance(raw, dict) and raw else corrected_space
        cspace_after = corrected_space
        validity = "valid"
        warning_texts = [str(w) for w in warnings if w]
        if metric_key == "diameter_selected_mm":
            status = str(item.get("diameter_sanity_status") or "ok")
            if status == "suspicious":
                validity = "suspicious"
            elif status == "invalid":
                validity = "invalid"
        if metric_key in ("feature_eccentricity", "sphericity_score") and warning_texts:
            # Only invariant_violation:* entries downgrade validity; warning:*
            # advisories are informational and the corrected metric is still trusted.
            if any(str(text).startswith("invariant_violation:") for text in warning_texts):
                validity = "suspicious"
        trace_rows.append(
            {
                "trace_id": f"obj_{int(item.get('object_id') or 0)}_{idx}",
                "metric_key": metric_key,
                "final_value": final_value,
                "raw_value": rv,
                "corrected_value": final_value,
                "correction_applied": bool(scale),
                "correction_scales": {"x": sx, "y": sy, "z": sz},
                "correction_factor_used": correction_factor_used,
                "coordinate_space_before": cspace_before,
                "coordinate_space_after": cspace_after,
                "source_artifact_id": source_artifact_id,
                "source_stage": "measurement",
                "formula_name": formula_name,
                "formula_human_readable": formula_hr,
                "formula_inputs": formula_inputs,
                "intermediate_values": intermediate,
                "validity_status": validity,
                "warnings": warning_texts,
                "used_by_classifier": metric_key in used_keys,
                "geometry_metric_source": geometry_metric_source,
            }
        )

    # Synthetic calibration / invariants context entries. These do not feed
    # into the classifier directly but are surfaced in the Metric details tab
    # so engineers can read the correction chain without leaving the table.
    trace_rows.append(
        {
            "trace_id": f"obj_{int(item.get('object_id') or 0)}_ctx_calibration",
            "metric_key": "scale_correction_applied",
            "final_value": None,
            "raw_value": None,
            "corrected_value": None,
            "correction_applied": bool(scale),
            "correction_scales": {"x": sx, "y": sy, "z": sz},
            "correction_factor_used": None,
            "coordinate_space_before": "raw_metric_mm",
            "coordinate_space_after": corrected_space,
            "source_artifact_id": "known_object_scale_validation",
            "source_stage": "measurement",
            "formula_name": "known_cube_scale_correction",
            "formula_human_readable": (
                "known_width / measured_width, known_depth / measured_depth, known_height / measured_height"
            ),
            "formula_inputs": [
                _input("correction_source", item.get("correction_source")),
                _input("correction_context_id", item.get("correction_context_id")),
            ],
            "intermediate_values": {
                "scale_x": sx,
                "scale_y": sy,
                "scale_z": sz,
                "geometry_metric_source": geometry_metric_source,
                "geometry_coordinate_space": item.get("geometry_coordinate_space"),
            },
            "validity_status": "valid" if bool(scale) else "not_applied",
            "warnings": [],
            "used_by_classifier": False,
            "geometry_metric_source": geometry_metric_source,
        }
    )
    if invariants:
        trace_rows.append(
            {
                "trace_id": f"obj_{int(item.get('object_id') or 0)}_ctx_invariants",
                "metric_key": "geometry_invariant_warnings",
                "final_value": None,
                "raw_value": None,
                "corrected_value": None,
                "correction_applied": bool(scale),
                "correction_scales": {"x": sx, "y": sy, "z": sz},
                "correction_factor_used": None,
                "coordinate_space_before": "raw_metric_mm",
                "coordinate_space_after": corrected_space,
                "source_artifact_id": "geometry_debug_summary" if geom else "measurement_object",
                "source_stage": "measurement",
                "formula_name": "geometry_invariant_checks",
                "formula_human_readable": (
                    "near_equal_dims_with_extreme_eccentricity | diameter_far_exceeds_dims | "
                    "compact_blob_with_near_zero_roundness | anisotropic_scale_factors_may_destabilize_metrics"
                ),
                "formula_inputs": [
                    _input("dim_x_mm", dim_x),
                    _input("dim_y_mm", dim_y),
                    _input("dim_z_mm", dim_z),
                    _input("feature_eccentricity", item.get("feature_eccentricity")),
                    _input("sphericity_score", item.get("sphericity_score")),
                    _input("diameter_mm", item.get("diameter_mm")),
                    _input("scale_x", sx),
                    _input("scale_y", sy),
                    _input("scale_z", sz),
                ],
                "intermediate_values": {
                    "violations": [str(w) for w in invariants if str(w).startswith("invariant_violation:")],
                    "advisories": [str(w) for w in invariants if str(w).startswith("warning:")],
                },
                "validity_status": "suspicious" if any(str(w).startswith("invariant_violation:") for w in invariants) else "valid",
                "warnings": [str(w) for w in invariants],
                "used_by_classifier": False,
                "geometry_metric_source": geometry_metric_source,
            }
        )

    return trace_rows


def _class_group(superclass: str) -> str:
    if superclass == "BALL_GOOD":
        return "Bola buena"
    if superclass == "BALL_SCRAP":
        return "Scrap de Bola"
    return "Chatarra"


# Backward-compatible class aliases for legacy imports
EstimateBeltPlaneStage = DetectBeltPlaneStage
NormalizeHeightRelativeToPlaneStage = NormalizeHeightsToPlaneStage
HeightSegmentationStage = RemoveBeltAndSegmentObjectsStage
