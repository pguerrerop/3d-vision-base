from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vision_3d_acquisition.contracts.modalities import CaptureModality
from vision_3d_acquisition.vision_core.heightmap import (
    HeightmapFrame,
    load_heightmap_npz,
    save_heightmap_npz,
    write_heightmap_metadata_json,
    write_heightmap_preview_png,
)
from vision_3d_acquisition.vision_core.pipelines.core import PipelineContext


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
        save_heightmap_npz(frame, output_dir / npz_name)
        write_heightmap_preview_png(frame.z_mm, frame.valid_mask, output_dir / preview_name)
        write_heightmap_metadata_json(frame, output_dir / meta_name, extra={"source_file": height_file})

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
            metadata={"source": "native_pipeline", "producer": self.name, "modality": "heightmap"},
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
    gradient_threshold_percentile: float = 35.0
    invalid_neighbor_policy: str = "mark_high_gradient"
    low_gradient_morphology_enabled: bool = True
    low_gradient_open_kernel: int = 3
    low_gradient_close_kernel: int = 5
    low_gradient_min_component_area: int = 800
    low_gradient_fill_holes: bool = True
    reference_surface_selection_mode: str = "largest_constant_z"
    reference_surface_min_area_ratio: float = 0.08
    reference_surface_max_z_std_mm: float = 8.0
    reference_surface_border_bonus: float = 0.2
    reference_surface_depth_preference_weight: float = 0.2
    reference_surface_constancy_weight: float = 0.5
    reference_surface_area_weight: float = 0.3
    reference_surface_model: str = "auto"
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
        valid_mask_u8 = (frame.valid_mask.astype(np.uint8) * 255)
        cv2.imwrite(str(output_dir / "valid_mask.png"), valid_mask_u8)

        roi_cfg = self.plane_fit_roi or _roi_from_metadata(context.get_artifact("metadata", {}))
        roi_mask = _roi_mask(frame.valid_mask.shape, roi_cfg) if roi_cfg is not None else np.ones_like(frame.valid_mask, dtype=bool)
        cv2.imwrite(str(output_dir / "plane_fit_roi_mask.png"), (roi_mask.astype(np.uint8) * 255))
        valid_for_fit = frame.valid_mask & roi_mask
        valid_pixel_count = int(np.count_nonzero(valid_for_fit))

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

            gradient_debug: dict[str, Any] = {}
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
                candidate_mask = selected_mask
                inferred_depth_convention = "low_gradient_surface"
                gradient_debug = {
                    "strategy": "low_gradient_surface",
                    "threshold_mode": self.gradient_threshold_mode,
                    "threshold_value": float(gthr),
                    "selected_component_id": int(selected_label),
                    "selected_component_area_ratio": float(np.count_nonzero(candidate_mask) / max(1, np.count_nonzero(valid_for_fit))),
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
            if sampled_pixel_count < max(3, self.plane_fit_min_valid_pixels // 2):
                status = "failed"
                failure_reason = "not_enough_sampled_points_after_filtering"
            else:
                coeffs, inliers = _fit_plane_ransac(
                    candidates,
                    iterations=self.plane_fit_max_iterations,
                    threshold_mm=self.plane_fit_residual_threshold_mm,
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
                        float(self.plane_fit_residual_threshold_mm),
                        float(np.std(seed_residual_values) * self.plane_background_residual_adaptive_multiplier),
                    )
                tol = adaptive_tol if self.plane_background_residual_tolerance_mode == "adaptive" else float(self.plane_background_residual_tolerance_mm)
                expanded_plane_mask = (np.abs(residual_seed) <= tol) & valid_for_fit
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
                    expanded_plane_mask = expanded_u8 > 0
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
                        final_plane_inlier_mask = (np.abs(residual_refit) <= tol) & valid_for_fit
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
            belt_mask = (np.abs(residual_map) <= self.plane_fit_residual_threshold_mm) & valid_for_fit
        background_vals = residual_map[belt_mask]
        bg_mean = float(np.mean(background_vals)) if background_vals.size else 0.0
        bg_std = float(np.std(background_vals)) if background_vals.size else 0.0
        bg_p95_abs = float(np.percentile(np.abs(background_vals), 95)) if background_vals.size else 0.0
        foreground_vals = residual_map[(~belt_mask) & valid_for_fit]
        fg_mean = float(np.mean(np.abs(foreground_vals))) if foreground_vals.size else 0.0
        if bg_p95_abs > 4.0:
            warnings.append("background_not_near_zero_after_plane_fit")
        if fg_mean < 3.0:
            warnings.append("foreground_background_separation_too_small")
        if inlier_ratio < max(0.15, self.plane_fit_min_inlier_ratio * 0.5):
            warnings.append("inlier_ratio_very_low")
        if sampled_pixel_count and sampled_pixel_count <= max(3, self.plane_fit_min_valid_pixels // 2):
            warnings.append("plane_fit_used_very_few_samples")
        if bg_p95_abs > 100.0:
            warnings.append("extreme_plane_fit_residuals_detected")
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
        reference_constant_z_mm = float(np.median(frame.z_mm[candidate_mask])) if np.count_nonzero(candidate_mask) else float(np.median(frame.z_mm[valid_for_fit]))
        use_constant_z = False
        if self.reference_surface_model == "constant_z":
            use_constant_z = True
        elif self.reference_surface_model == "auto":
            if status != "success" or inlier_ratio < max(0.15, self.plane_fit_min_inlier_ratio * 0.6):
                use_constant_z = True
        if use_constant_z:
            reference_model_type = "constant_z"
            coeffs = np.asarray([0.0, 0.0, 1.0, -reference_constant_z_mm], dtype=np.float64)
            residual_map = frame.z_mm.astype(np.float32) - np.float32(reference_constant_z_mm)
            residual_map[~frame.valid_mask] = 0.0

        context.set_artifact("belt_plane", tuple(float(x) for x in coeffs.tolist()))
        context.set_artifact("plane_model", tuple(float(x) for x in coeffs.tolist()))
        context.set_artifact(
            "reference_surface_model",
            {
                "type": reference_model_type,
                "plane_coefficients": [float(x) for x in coeffs.tolist()] if reference_model_type == "plane" else None,
                "constant_z_mm": reference_constant_z_mm if reference_model_type == "constant_z" else None,
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
            "equation": {"a": float(coeffs[0]), "b": float(coeffs[1]), "c": float(coeffs[2]), "d": float(coeffs[3])},
            "residual_stats_mm": stats,
            "status": status,
            "failure_reason": failure_reason,
        }
        (output_dir / "belt_plane.json").write_text(json.dumps(plane_json, indent=2), encoding="utf-8")

        write_heightmap_preview_png(np.abs(residual_map), frame.valid_mask, output_dir / "belt_plane_residuals.png")
        cv2.imwrite(str(output_dir / "plane_inlier_mask.png"), (belt_mask.astype(np.uint8) * 255))
        cv2.imwrite(str(output_dir / "background_candidate_mask.png"), (candidate_mask.astype(np.uint8) * 255))
        cv2.imwrite(str(output_dir / "background_seed_mask.png"), (seed_mask.astype(np.uint8) * 255))
        cv2.imwrite(str(output_dir / "expanded_plane_mask.png"), (expanded_plane_mask.astype(np.uint8) * 255))
        cv2.imwrite(str(output_dir / "final_plane_inlier_mask.png"), (belt_mask.astype(np.uint8) * 255))
        overlay = _render_belt_plane_overlay(frame, belt_mask)
        cv2.imwrite(str(output_dir / "belt_plane_overlay.png"), overlay)

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
            "component_stats": component_stats[:40],
            "warnings": warnings,
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
            "plane_coefficients": [float(x) for x in coeffs.tolist()],
            "residual_mean_mm": float(np.mean(residuals)) if residuals.size else 0.0,
            "residual_median_mm": float(np.median(residuals)) if residuals.size else 0.0,
            "residual_p95_mm": float(np.percentile(np.abs(residuals), 95)) if residuals.size else 0.0,
            "residual_max_abs_mm": float(np.max(np.abs(residuals))) if residuals.size else 0.0,
            "background_height_mean_after_normalization_mm": bg_mean,
            "background_height_std_after_normalization_mm": bg_std,
            "background_height_p95_abs_after_normalization_mm": bg_p95_abs,
            "foreground_mean_height_mm": fg_mean,
            "warnings": warnings,
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
                "plane_fit_max_iterations": self.plane_fit_max_iterations,
                "gradient_smoothing_kernel": self.gradient_smoothing_kernel,
                "gradient_method": self.gradient_method,
                "gradient_threshold_mode": self.gradient_threshold_mode,
                "gradient_threshold_value": self.gradient_threshold_value,
                "gradient_threshold_percentile": self.gradient_threshold_percentile,
                "invalid_neighbor_policy": self.invalid_neighbor_policy,
                "reference_surface_selection_mode": self.reference_surface_selection_mode,
                "reference_surface_model": self.reference_surface_model,
            },
        }
        (output_dir / "plane_fit_debug.json").write_text(json.dumps(debug_payload, indent=2), encoding="utf-8")
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
                "background_selection_debug": "background_selection_debug.json",
                "background_seed_mask": "background_seed_mask.png",
                "expanded_plane_mask": "expanded_plane_mask.png",
                "final_plane_inlier_mask": "final_plane_inlier_mask.png",
                "plane_inlier_mask": "plane_inlier_mask.png",
                "plane_fit_debug": "plane_fit_debug.json",
                "belt_plane_overlay": "belt_plane_overlay.png",
                "belt_plane_residuals": "belt_plane_residuals.png",
            }
        )
        context.set_artifact("files", files)

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
            metadata={"source": "native_pipeline", "producer": self.name, "residual_stats_mm": stats},
        )
        context.add_processing_artifact(
            artifact_id="belt_plane_overlay",
            stage_id="detect_belt_plane",
            kind="image",
            title="Belt plane overlay",
            path="belt_plane_overlay.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
        )
        context.add_processing_artifact(
            artifact_id="raw_heightmap_preview",
            stage_id="detect_belt_plane",
            kind="image",
            title="Raw heightmap preview",
            path="raw_heightmap_preview.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
        )
        context.add_processing_artifact(
            artifact_id="valid_mask",
            stage_id="detect_belt_plane",
            kind="image",
            title="Valid mask",
            path="valid_mask.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
        )
        context.add_processing_artifact(
            artifact_id="plane_inlier_mask",
            stage_id="detect_belt_plane",
            kind="image",
            title="Plane inlier mask",
            path="plane_inlier_mask.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
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
            metadata={"source": "native_pipeline", "producer": self.name},
        )
        context.add_processing_artifact(
            artifact_id="low_gradient_mask",
            stage_id="detect_belt_plane",
            kind="image",
            title="Low-gradient mask",
            path="low_gradient_mask.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
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
        write_heightmap_preview_png(normalized, frame.valid_mask, output_dir / "normalized_heightmap.png")
        write_heightmap_metadata_json(norm_frame, output_dir / "normalized_heightmap.metadata.json")
        bg_values = normalized[plane_mask & frame.valid_mask]
        fg_values = normalized[(~plane_mask) & frame.valid_mask]
        valid_values = normalized[frame.valid_mask]
        hist_payload = {
            "bin_edges_mm": [],
            "bin_counts": [],
            "sample_count": int(valid_values.size),
            "min_mm": float(np.min(valid_values)) if valid_values.size else 0.0,
            "max_mm": float(np.max(valid_values)) if valid_values.size else 0.0,
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
        (output_dir / "normalized_height_histogram.json").write_text(json.dumps(hist_payload, indent=2), encoding="utf-8")
        norm_debug = {
            "normalization_convention": convention,
            "clip_negative": bool(self.normalized_clip_negative),
            "negative_tolerance_mm": float(self.normalized_negative_tolerance_mm),
            "background_height_mean_after_normalization_mm": float(np.mean(bg_values)) if bg_values.size else 0.0,
            "background_height_std_after_normalization_mm": float(np.std(bg_values)) if bg_values.size else 0.0,
            "background_height_p95_abs_after_normalization_mm": float(np.percentile(np.abs(bg_values), 95)) if bg_values.size else 0.0,
            "foreground_mean_height_mm": float(np.mean(fg_values)) if fg_values.size else 0.0,
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
        (output_dir / "normalization_debug.json").write_text(json.dumps(norm_debug, indent=2), encoding="utf-8")

        files = dict(context.get_artifact("files", {}))
        files.update(
            {
                "normalized_heightmap": "normalized_heightmap.npz",
                "debug_height": "normalized_heightmap.png",
                "normalized_height_histogram": "normalized_height_histogram.json",
                "normalization_debug": "normalization_debug.json",
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
                "source": "native_pipeline",
                "producer": self.name,
                "semantic": "height_above_belt_mm",
                "normalization_convention": convention,
                "reference_surface_model_type": model_type,
            },
        )
        context.add_processing_artifact(
            artifact_id="normalized_height_histogram",
            stage_id="normalize_heights_to_plane",
            kind="json",
            title="Normalized height histogram",
            path="normalized_height_histogram.json",
            mime_type="application/json",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name, **hist_payload},
        )
        context.add_processing_artifact(
            artifact_id="normalization_debug",
            stage_id="normalize_heights_to_plane",
            kind="json",
            title="Normalized height debug",
            path="normalization_debug.json",
            mime_type="application/json",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name, **norm_debug},
        )


@dataclass
class RemoveBeltAndSegmentObjectsStage:
    min_height_mm: float = 8.0
    max_height_mm: float | None = None
    suppress_plane_mask_in_segmentation: bool = True
    ignore_small_residual_background: bool = True
    morphology_kernel: int = 5
    min_component_area: int = 120
    fill_holes: bool = True
    smoothing_kernel: int = 3
    roi: dict[str, Any] | None = None
    name: str = "RemoveBeltAndSegmentObjects"
    category: str = "segmentation"
    required_modalities: tuple[CaptureModality, ...] = ("heightmap",)
    produced_modalities: tuple[CaptureModality, ...] = ()
    produced_artifacts: tuple[str, ...] = ("height_segmentation",)
    stage_category: str | None = "segmentation"

    def run(self, context: PipelineContext) -> None:
        frame: HeightmapFrame = context.require_artifact("heightmap_frame")
        normalized = np.asarray(context.require_artifact("normalized_heightmap_mm"), dtype=np.float32)
        output_dir: Path = context.require_artifact("output_dir")

        foreground = normalized > self.min_height_mm
        if self.max_height_mm is not None:
            foreground &= normalized <= self.max_height_mm
        foreground &= frame.valid_mask
        foreground_before_suppression = foreground.copy()
        cv2.imwrite(str(output_dir / "foreground_before_plane_suppression.png"), (foreground_before_suppression.astype(np.uint8) * 255))

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

        roi_cfg = self.roi or _roi_from_metadata(context.get_artifact("metadata", {}))
        if roi_cfg is not None:
            foreground &= _roi_mask(foreground.shape, roi_cfg)
            context.set_artifact("roi_25d", roi_cfg)

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
                "morphology_kernel": self.morphology_kernel,
                "min_component_area": self.min_component_area,
                "fill_holes": self.fill_holes,
                "smoothing_kernel": self.smoothing_kernel,
            },
            "plane_suppressed_pixel_count": int(np.count_nonzero(suppressed_plane_pixels)),
            "connected_components_before_suppression": int(max(num_labels_before - 1, 0)),
            "connected_components_after_suppression": int(max(num_labels_after_suppress - 1, 0)),
            "connected_component_count": int(max(num_labels - 1, 0)),
        }
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
            artifact_id="plane_suppressed_mask",
            stage_id="remove_belt_segment_objects",
            kind="image",
            title="Plane suppressed mask",
            path="plane_suppressed_mask.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
        )
        context.add_processing_artifact(
            artifact_id="final_object_mask",
            stage_id="remove_belt_segment_objects",
            kind="image",
            title="Final object mask",
            path="final_object_mask.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
        )
        context.add_processing_artifact(
            artifact_id="rejected_background_residuals",
            stage_id="remove_belt_segment_objects",
            kind="image",
            title="Rejected background residuals",
            path="rejected_background_residuals.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
        )

        context.add_processing_artifact(
            artifact_id="segmentation_debug",
            stage_id="remove_belt_segment_objects",
            kind="json",
            title="Segmentation debug",
            path="segmentation_debug.json",
            mime_type="application/json",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name, **seg_debug},
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
                "source": "native_pipeline",
                "producer": self.name,
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
            target_artifact_id="heightmap_preview",
            source_artifact_ids=["heightmap", "cleaned_object_mask"],
            metadata={"source": "native_pipeline", "producer": self.name, "semantic": "primary_human_view"},
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
            metadata={"source": "native_pipeline", "producer": self.name, "component_count": len(components)},
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
            edge = cv2.Canny((mask.astype(np.uint8) * 255), 50, 100)
            edge_heights = normalized[edge > 0]
            asym = float(abs(np.mean(pos_values[::2]) - np.mean(pos_values[1::2]))) if pos_values.size > 4 else 0.0
            flatness = float(1.0 - np.std(pos_values) / max(np.mean(pos_values), 1e-6))
            curvature = float(np.mean(np.abs(gx[mask])) + np.mean(np.abs(gy[mask])))
            roughness = float(np.std(edge_heights)) if edge_heights.size else 0.0
            volume_proxy = float(np.sum(pos_values) * pixel_area)
            centroid = _mask_centroid(mask)
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
                    round(float(np.max(values) - np.min(values)), 4),
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
                    "max_height_mm": round(float(np.max(values)), 4),
                    "mean_height_mm": round(float(np.mean(values)), 4),
                    "median_height_mm": round(float(np.median(values)), 4),
                    "p95_height_mm": round(float(np.percentile(values, 95)), 4),
                    "height_std_mm": round(float(np.std(values)), 4),
                },
                "fraction_points_inside_belt": 1.0,
                "filter_status": "kept",
                "filter_reason": None,
                "feature_volume_proxy_mm3": round(volume_proxy, 4),
                "feature_flatness": round(flatness, 4),
                "feature_eccentricity": round(float(comp.get("eccentricity") or 0.0), 4),
                "feature_edge_roughness": round(roughness, 4),
                "feature_local_curvature_proxy": round(curvature, 4),
                "feature_height_asymmetry": round(asym, 4),
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
            }
            objects.append(object_row)
            oid = object_row["object_id"]
            context.add_processing_artifact(
                artifact_id=f"measurement_object_{oid}",
                stage_id="measurement",
                object_id=oid,
                kind="json",
                title=f"Object #{oid} measurements",
                metadata={"source": "native_pipeline", "producer": self.name, "modality": "heightmap"},
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
        cfg = metadata.get("known_object_25d") if isinstance(metadata, dict) else None
        if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
            context.set_artifact("known_object_scale_validation", {"enabled": False, "status": "skipped"})
            return

        objects = context.get_artifact("objects", [])
        if not isinstance(objects, list) or not objects:
            context.set_artifact("known_object_scale_validation", {"enabled": True, "status": "failed", "reason": "no_objects"})
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
        measured_w = float(target.get("major_axis_mm") or target.get("dimensions_mm", [0, 0, 0])[0] or 0.0)
        measured_d = float(target.get("minor_axis_mm") or target.get("dimensions_mm", [0, 0, 0])[1] or 0.0)
        measured_h = float((target.get("height_above_belt_mm") or {}).get("max_height_mm") or 0.0)

        def _err(measured: float, known: float) -> float:
            return ((measured - known) / known * 100.0) if known > 1e-9 else 0.0

        err_x = _err(measured_w, known_w)
        err_y = _err(measured_d, known_d)
        err_z = _err(measured_h, known_h)
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
            "scale_error_x_percent": err_x,
            "scale_error_y_percent": err_y,
            "scale_error_z_percent": err_z,
            "pass_x": abs(err_x) <= tol_pct,
            "pass_y": abs(err_y) <= tol_pct,
            "pass_z": abs(err_z) <= tol_pct,
            "recommended_scale_correction_x": (known_w / measured_w) if measured_w > 1e-9 else 1.0,
            "recommended_scale_correction_y": (known_d / measured_d) if measured_d > 1e-9 else 1.0,
            "recommended_scale_correction_z": (known_h / measured_h) if measured_h > 1e-9 else 1.0,
            "tolerance_percent": tol_pct,
        }
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
            metadata={"source": "native_pipeline", "producer": self.name},
        )


@dataclass
class ClassifyMiningBall25DStage:
    name: str = "ClassifyMiningBall25D"
    category: str = "classification"
    required_modalities: tuple[CaptureModality, ...] = ("heightmap",)
    produced_modalities: tuple[CaptureModality, ...] = ()
    produced_artifacts: tuple[str, ...] = ("classification",)
    stage_category: str | None = "classification"

    def run(self, context: PipelineContext) -> None:
        objects = context.get_artifact("objects", [])
        ball_count = 0
        non_ball = 0
        for item in objects:
            label, class_name, confidence = _classify_25d(item)
            item["class_name"] = class_name
            item["confidence"] = confidence
            item["label"] = label
            item["class_group"] = _class_group(label)
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
                metadata={"source": "native_pipeline", "producer": self.name, "label": label},
            )
        object_count = len(objects)
        decision = "accept" if object_count > 0 and non_ball == 0 else "review"
        context.set_artifact(
            "summary",
            {
                "object_count": object_count,
                "ball_count": ball_count,
                "non_ball_count": non_ball,
                "decision": decision,
                "confidence": round(ball_count / max(1, object_count), 3) if object_count else None,
            },
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
        cv2.imwrite(str(output_dir / "classification_overlay.png"), meas)

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
                metadata={"source": "native_pipeline", "producer": self.name, "modality": "heightmap"},
            )


def _load_heightmap_frame(height_path: Path, reflectance_path: Path | None, metadata: dict[str, Any]) -> HeightmapFrame:
    if not height_path.is_file():
        raise FileNotFoundError(f"Heightmap file not found: {height_path}")
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
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm < 1e-9:
        return None
    normal = normal / norm
    d = -float(np.dot(normal, p1))
    return np.array([normal[0], normal[1], normal[2], d], dtype=np.float64)


def _least_squares_plane(points: np.ndarray) -> np.ndarray:
    xyz = np.asarray(points, dtype=np.float64)
    centroid = np.mean(xyz, axis=0)
    centered = xyz - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal = normal / max(np.linalg.norm(normal), 1e-12)
    d = -float(np.dot(normal, centroid))
    return np.array([normal[0], normal[1], normal[2], d], dtype=np.float64)


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
        zvals = z[mask]
        gvals = gradient[mask]
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


def _roi_from_metadata(metadata: dict[str, Any]) -> dict[str, Any] | None:
    roi = metadata.get("roi_25d")
    return roi if isinstance(roi, dict) else None


def _roi_mask(shape: tuple[int, int], roi: dict[str, Any]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    roi_type = str(roi.get("type") or "rectangle").lower()
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
    y = max(0, int(roi.get("y") or 0))
    rw = max(1, int(roi.get("width") or w))
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


def _classify_25d(item: dict[str, Any]) -> tuple[str, str, float]:
    metrics = item.get("height_above_belt_mm") or {}
    max_h = float(metrics.get("max_height_mm") or 0.0)
    p95_h = float(metrics.get("p95_height_mm") or 0.0)
    ecc = float(item.get("feature_eccentricity") or 0.0)
    flat = float(item.get("feature_flatness") or 0.0)
    rough = float(item.get("feature_edge_roughness") or 0.0)
    vol = float(item.get("feature_volume_proxy_mm3") or 0.0)
    if 8.0 <= max_h <= 95.0 and ecc < 0.45 and flat > 0.15 and rough < 8.0:
        return "Bola buena", "ball", 0.82
    if ecc >= 0.75 or flat < -0.2 or rough > 12.0:
        return "Scrap de Bola - Bola deformada", "non_ball", 0.78
    if max_h < 6.0 or p95_h < 4.0 or vol < 4000.0:
        return "Chatarra - Planchuelas", "non_ball", 0.72
    return "Scrap de Bola - Bola con chip", "non_ball", 0.66


def _class_group(label: str) -> str:
    if label.startswith("Bola buena"):
        return "Bola buena"
    if label.startswith("Scrap de Bola"):
        return "Scrap de Bola"
    return "Chatarra"


# Backward-compatible class aliases for legacy imports
EstimateBeltPlaneStage = DetectBeltPlaneStage
NormalizeHeightRelativeToPlaneStage = NormalizeHeightsToPlaneStage
HeightSegmentationStage = RemoveBeltAndSegmentObjectsStage
