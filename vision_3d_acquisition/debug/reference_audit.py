from __future__ import annotations

import csv
import dataclasses
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vision_3d_acquisition.processing.status_index import process_entries_for_take
from vision_3d_acquisition.vision_core.pipelines.stages_25d import (
    DetectBeltPlaneStage,
    NormalizeHeightsToPlaneStage,
    RemoveBeltAndSegmentObjectsStage,
    ValidateKnownObjectScale25DStage,
)

PIPELINE_ID_25D = "mining_steel_ball_classification_25d"

STEP_FOLDERS = (
    "00_input",
    "01_roi",
    "02_gradient",
    "03_height_gate",
    "04_blob_clusters",
    "04_plateaus",
    "05_stripes",
    "06_candidate_support",
    "07_plane_fit",
    "08_normalization",
    "09_segmentation",
    "10_summary",
)

STEP_ARTIFACTS: dict[str, list[tuple[str, str]]] = {
    "00_input": [
        ("raw_heightmap_preview", "raw_heightmap_preview.png"),
        ("valid_mask", "valid_mask.png"),
        ("heightmap_preview", "heightmap_preview.png"),
        ("reflectance_preview", "reflectance_preview.png"),
        ("acquisition_metadata_excerpt", "acquisition_metadata_excerpt.json"),
    ],
    "01_roi": [
        ("plane_fit_roi_mask", "plane_fit_roi_mask.png"),
        ("valid_for_fit_proxy", "valid_for_fit_proxy.png"),
        ("rejected_by_roi", "rejected_by_roi.png"),
    ],
    "02_gradient": [
        ("depth_gradient_magnitude", "depth_gradient_magnitude.png"),
        ("low_gradient_mask", "low_gradient_mask.png"),
        ("low_gradient_components_overlay", "low_gradient_components_overlay.png"),
        ("rejected_by_gradient", "rejected_by_gradient.png"),
        ("rejected_by_gradient_overlay", "rejected_by_gradient_overlay.png"),
    ],
    "03_height_gate": [
        ("flat_candidate_mask", "flat_candidate_mask.png"),
        ("background_selected_plateau_mask", "background_selected_plateau_mask.png"),
        ("gradient_debug_excerpt", "gradient_debug_excerpt.json"),
        ("rejected_by_height_gate", "rejected_by_height_gate.png"),
        ("rejected_by_height_gate_overlay", "rejected_by_height_gate_overlay.png"),
    ],
    "04_blob_clusters": [
        ("low_gradient_blob_components_overlay", "low_gradient_blob_components_overlay.png"),
        ("low_gradient_blob_id_mask", "low_gradient_blob_id_mask.png"),
        ("low_gradient_blob_summary", "low_gradient_blob_summary.json"),
        ("height_aware_blob_components_overlay", "height_aware_blob_components_overlay.png"),
        ("height_aware_blob_id_mask", "height_aware_blob_id_mask.png"),
        ("height_aware_connectivity_rejected_edges", "height_aware_connectivity_rejected_edges.png"),
        ("height_aware_connectivity_debug", "height_aware_connectivity_debug.json"),
        ("height_border_strength", "height_border_strength.png"),
        ("height_border_cut_mask", "height_border_cut_mask.png"),
        ("height_border_fragments_overlay", "height_border_fragments_overlay.png"),
        ("height_border_fragments_mask", "height_border_fragments_mask.png"),
        ("height_border_fragments_id_mask", "height_border_fragments_id_mask.png"),
        ("height_border_split_debug", "height_border_split_debug.json"),
        ("fragment_merge_debug", "fragment_merge_debug.json"),
        ("height_split_blob_fragments_overlay", "height_split_blob_fragments_overlay.png"),
        ("height_split_blob_fragments_mask", "height_split_blob_fragments_mask.png"),
        ("height_split_blob_id_mask", "height_split_blob_id_mask.png"),
        ("height_split_debug", "height_split_debug.json"),
        ("height_consistent_blob_summary", "height_consistent_blob_summary.json"),
        ("blob_height_clusters", "blob_height_clusters.json"),
        ("blob_cluster_score_table", "blob_cluster_score_table.csv"),
        ("selected_blob_cluster_mask", "selected_blob_cluster_mask.png"),
        ("rejected_blob_clusters_mask", "rejected_blob_clusters_mask.png"),
        ("selected_blob_cluster_overlay", "selected_blob_cluster_overlay.png"),
        ("rejected_blob_clusters_overlay", "rejected_blob_clusters_overlay.png"),
        ("blob_cluster_selection_debug", "blob_cluster_selection_debug.json"),
        ("rejected_by_blob_cluster_selection", "rejected_by_blob_cluster_selection.png"),
        ("rejected_by_blob_cluster_selection_overlay", "rejected_by_blob_cluster_selection_overlay.png"),
        ("rejected_by_blob_mad_refinement", "rejected_by_blob_mad_refinement.png"),
        ("rejected_by_blob_mad_refinement_overlay", "rejected_by_blob_mad_refinement_overlay.png"),
        ("rejected_by_height_aware_connectivity", "rejected_by_height_aware_connectivity.png"),
    ],
    "04_plateaus": [
        ("flat_candidate_mask", "flat_candidate_mask.png"),
        ("background_selected_plateau_mask", "background_selected_plateau_mask.png"),
        ("rejected_raised_plateau_mask", "rejected_raised_plateau_mask.png"),
        ("rejected_raised_structure_mask", "rejected_raised_structure_mask.png"),
        ("background_plateau_plot", "background_plateau_plot.png"),
        ("flat_candidate_depth_plot", "flat_candidate_depth_plot.png"),
        ("reference_surface_plateaus", "reference_surface_plateaus.json"),
        ("flat_candidate_histogram", "flat_candidate_histogram.json"),
        ("rejected_by_plateau_selection", "rejected_by_plateau_selection.png"),
        ("rejected_by_plateau_selection_overlay", "rejected_by_plateau_selection_overlay.png"),
    ],
    "05_stripes": [
        ("belt_stripes_mask", "belt_stripes_mask.png"),
        ("belt_base_mask", "belt_base_mask.png"),
        ("belt_stripes_tophat_mask", "belt_stripes_tophat_mask.png"),
        ("belt_stripes_shape_mask", "belt_stripes_shape_mask.png"),
        ("belt_above_belt_mask", "belt_above_belt_mask.png"),
        ("belt_wide_object_mask", "belt_wide_object_mask.png"),
        ("belt_altitude_histogram_image", "belt_altitude_histogram.png"),
        ("belt_altitude_local_min", "belt_altitude_local_min.png"),
        ("removed_by_stripe_filter", "removed_by_stripe_filter.png"),
        ("removed_by_stripe_filter_overlay", "removed_by_stripe_filter_overlay.png"),
        ("belt_stripe_filter_debug", "belt_stripe_filter_debug.json"),
    ],
    "06_candidate_support": [
        ("reference_surface_selected_mask", "reference_surface_selected_mask.png"),
        ("background_candidate_mask", "background_candidate_mask.png"),
        ("expanded_plane_mask", "expanded_plane_mask.png"),
        ("final_plane_inlier_mask", "final_plane_inlier_mask.png"),
        ("plane_inlier_mask", "plane_inlier_mask.png"),
        ("belt_plane_overlay", "belt_plane_overlay.png"),
    ],
    "07_plane_fit": [
        ("belt_plane_residuals", "belt_plane_residuals.png"),
        ("plane_residual_histogram", "plane_residual_histogram.json"),
        ("final_plane_inlier_mask", "final_plane_inlier_mask.png"),
        ("rejected_by_plane_residual", "rejected_by_plane_residual.png"),
        ("rejected_by_plane_residual_overlay", "rejected_by_plane_residual_overlay.png"),
        ("plane_fit_debug", "plane_fit_debug.json"),
    ],
    "08_normalization": [
        ("normalized_heightmap", "normalized_heightmap.png"),
        ("below_reference_mask", "below_reference_mask.png"),
        ("above_threshold_mask", "above_threshold_mask.png"),
        ("normalized_height_histogram", "normalized_height_histogram.json"),
        ("normalization_debug", "normalization_debug.json"),
    ],
    "09_segmentation": [
        ("foreground_before_plane_suppression", "foreground_before_plane_suppression.png"),
        ("plane_suppressed_mask", "plane_suppressed_mask.png"),
        ("rejected_background_residuals", "rejected_background_residuals.png"),
        ("rejected_belt_stripes", "rejected_belt_stripes.png"),
        ("normalized_height_threshold_mask", "normalized_height_threshold_mask.png"),
        ("cleaned_object_mask", "cleaned_object_mask.png"),
        ("final_object_mask", "final_object_mask.png"),
        ("connected_components_overlay", "connected_components_overlay.png"),
        ("segmentation_overlay", "segmentation_overlay.png"),
        ("rejected_by_segmentation_plane_suppression", "rejected_by_segmentation_plane_suppression.png"),
        ("rejected_by_segmentation_stripe_suppression", "rejected_by_segmentation_stripe_suppression.png"),
        ("segmentation_debug", "segmentation_debug.json"),
    ],
    "10_summary": [
        ("support_loss", "support_loss.json"),
        ("support_loss_csv", "support_loss.csv"),
        ("summary_contact_sheet", "summary_contact_sheet.png"),
    ],
}

OVERLAY_ARTIFACTS = (
    "belt_plane_overlay",
    "low_gradient_components_overlay",
    "connected_components_overlay",
    "segmentation_overlay",
    "height_segmentation_overlay",
)

KNOWN_ARTIFACT_FILENAMES: dict[str, str] = {
    artifact_id: filename
    for step_items in STEP_ARTIFACTS.values()
    for artifact_id, filename in step_items
    if not artifact_id.endswith("_overlay") and not artifact_id.endswith("_excerpt") and artifact_id not in {"support_loss", "support_loss_csv", "summary_contact_sheet", "valid_for_fit_proxy", "acquisition_metadata_excerpt"}
}


@dataclasses.dataclass
class ResolvedRun:
    take_id: str
    pipeline_id: str
    run_id: str
    run_descriptor: str
    processed_dir: Path
    result_path: Path
    result_payload: dict[str, Any]


@dataclasses.dataclass
class AuditBundleResult:
    output_dir: Path
    manifest: dict[str, Any]
    artifacts_copied: int
    derived_masks_generated: int
    missing_artifacts: list[str]


def resolve_processed_run(
    data_dir: Path,
    *,
    take_id: str,
    pipeline_id: str = PIPELINE_ID_25D,
    run_id: str = "latest",
) -> ResolvedRun | None:
    processed_root = data_dir / "processed" / take_id
    default_result = processed_root / "result.json"
    entries = [
        item
        for item in process_entries_for_take(data_dir, take_id)
        if str(item.get("pipeline_family") or "") in {"25d", "2.5d", "generic", ""}
    ]
    entries = sorted(entries, key=lambda item: str(item.get("created_at") or ""), reverse=True)

    candidates: list[tuple[str, Path, dict[str, Any] | None]] = []
    if default_result.is_file():
        payload = _read_json(default_result)
        if _pipeline_matches(payload, pipeline_id):
            descriptor = str(payload.get("processed_at") or "processed")
            candidates.append(("processed", default_result, payload))

    for entry in entries:
        entry_run_id = str(entry.get("run_id") or "")
        entry_pipeline = str(entry.get("pipeline_id") or pipeline_id)
        if pipeline_id and entry_pipeline not in {pipeline_id, ""}:
            continue
        run_path = entry.get("path")
        if not run_path:
            continue
        run_dir = Path(str(run_path))
        result_path = run_dir / "result.json"
        if not result_path.is_file():
            continue
        payload = _read_json(result_path)
        if not _pipeline_matches(payload, pipeline_id):
            continue
        candidates.append((entry_run_id or "process_run", result_path, payload))

    if not candidates:
        return None

    if run_id == "latest":
        selected = candidates[0]
    else:
        selected = next((item for item in candidates if item[0] == run_id), None)
        if selected is None:
            selected = next((item for item in candidates if item[1].parent.name == run_id), None)
        if selected is None and default_result.is_file():
            payload = _read_json(default_result)
            if _pipeline_matches(payload, pipeline_id):
                selected = ("processed", default_result, payload)

    if selected is None:
        return None

    resolved_run_id, result_path, payload = selected
    if payload is None:
        payload = _read_json(result_path)
    pipeline = payload.get("processing_pipeline") or {}
    return ResolvedRun(
        take_id=take_id,
        pipeline_id=str(pipeline.get("id") or pipeline_id),
        run_id=resolved_run_id,
        run_descriptor=resolved_run_id if resolved_run_id != "processed" else str(payload.get("processed_at") or "latest"),
        processed_dir=result_path.parent,
        result_path=result_path,
        result_payload=payload,
    )


def build_reference_audit_bundle(
    data_dir: Path,
    *,
    take_id: str,
    pipeline_id: str = PIPELINE_ID_25D,
    run_id: str = "latest",
    output_root: Path | None = None,
    include_overlays: bool = True,
    include_diffs: bool = True,
) -> AuditBundleResult:
    resolved = resolve_processed_run(data_dir, take_id=take_id, pipeline_id=pipeline_id, run_id=run_id)
    if resolved is None:
        raise FileNotFoundError(
            f"No processed 25D result found for take_id={take_id!r} pipeline_id={pipeline_id!r} run_id={run_id!r}. "
            "Re-run with --rerun or process the take first."
        )

    run_folder = _safe_folder_name(resolved.run_descriptor)
    bundle_dir = (output_root or (data_dir / "debug" / "25d_reference_audit")) / take_id / run_folder
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    for step in STEP_FOLDERS:
        (bundle_dir / step).mkdir(parents=True, exist_ok=True)

    artifact_index = _build_artifact_index(resolved)
    loaded_masks: dict[str, np.ndarray] = {}
    copied = 0
    derived = 0
    found: list[str] = []
    missing: list[str] = []
    derived_masks_meta: list[dict[str, Any]] = []
    derived_skip_reasons: list[dict[str, str]] = []
    warnings: list[str] = []

    _write_acquisition_metadata_excerpt(data_dir, take_id, bundle_dir / "00_input" / "acquisition_metadata_excerpt.json")
    _write_stage_params_resolved(resolved, bundle_dir / "stage_params_resolved.json")

    for artifact_id, filename in _iter_copy_targets(include_overlays=include_overlays):
        step = _step_for_artifact(artifact_id, filename)
        dest = bundle_dir / step / filename
        source = _resolve_artifact_path(resolved, artifact_index, artifact_id)
        if source is None or not source.is_file():
            missing.append(artifact_id)
            continue
        if artifact_id.endswith("_excerpt") or artifact_id == "acquisition_metadata_excerpt":
            continue
        if source.suffix.lower() == ".json":
            payload = _read_json(source)
            dest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        else:
            shutil.copy2(source, dest)
        copied += 1
        found.append(artifact_id)
        if source.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"} and "mask" in artifact_id:
            mask = load_binary_mask(source)
            if mask is not None:
                loaded_masks[artifact_id] = mask

    gradient_debug_path = _resolve_artifact_path(resolved, artifact_index, "gradient_debug")
    if gradient_debug_path and gradient_debug_path.is_file():
        payload = _as_dict(_read_json(gradient_debug_path))
        excerpt = {
            "height_gate": payload.get("height_gate"),
            "gradient_threshold": payload.get("gradient_threshold"),
            "selected_component_id": payload.get("selected_component_id"),
            "selected_component_area_ratio": payload.get("selected_component_area_ratio"),
        }
        dest = bundle_dir / "03_height_gate" / "gradient_debug_excerpt.json"
        dest.write_text(json.dumps(excerpt, indent=2, sort_keys=True), encoding="utf-8")
        copied += 1
        found.append("gradient_debug_excerpt")
    else:
        missing.append("gradient_debug_excerpt")

    if include_diffs:
        diff_results = _generate_derived_masks(bundle_dir, loaded_masks, resolved, artifact_index)
        derived += diff_results["generated"]
        derived_masks_meta.extend(diff_results["entries"])
        derived_skip_reasons.extend(diff_results["skipped"])
        missing.extend(item for item in diff_results["missing"] if item not in missing)

    support_losses = _compute_support_losses(loaded_masks, derived_masks_meta)
    support_loss_path = bundle_dir / "10_summary" / "support_loss.json"
    support_loss_path.write_text(json.dumps(support_losses, indent=2), encoding="utf-8")
    _write_support_loss_csv(bundle_dir / "10_summary" / "support_loss.csv", support_losses)
    copied += 2
    found.extend(["support_loss", "support_loss_csv"])

    contact_sheet = _maybe_contact_sheet(bundle_dir)
    if contact_sheet is not None:
        copied += 1
        found.append("summary_contact_sheet")

    stage_params_info = _stage_params_info(resolved.result_payload)
    derived_thresholds, threshold_warnings = _extract_derived_thresholds(resolved.processed_dir, artifact_index, resolved)
    warnings.extend(threshold_warnings)
    manifest = {
        "take_id": take_id,
        "pipeline_id": resolved.pipeline_id,
        "run_id": resolved.run_id,
        "run_descriptor": resolved.run_descriptor,
        "processed_dir": str(resolved.processed_dir),
        "created_at": datetime.now(UTC).isoformat(),
        "source_result_json": str(resolved.result_path),
        "stage_params": stage_params_info,
        "derived_thresholds": derived_thresholds,
        "artifacts_found": sorted(set(found)),
        "artifacts_missing": sorted(set(missing)),
        "derived_masks": derived_masks_meta,
        "derived_masks_skipped": derived_skip_reasons,
        "warnings": warnings,
    }
    manifest_path = bundle_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    _write_report_md(bundle_dir, manifest, support_losses)
    _write_index_html(bundle_dir, manifest)

    return AuditBundleResult(
        output_dir=bundle_dir,
        manifest=manifest,
        artifacts_copied=copied,
        derived_masks_generated=derived,
        missing_artifacts=sorted(set(missing)),
    )


def load_binary_mask(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    return image > 127


def save_binary_mask(path: Path, mask: np.ndarray) -> None:
    cv2.imwrite(str(path), (np.asarray(mask, dtype=bool).astype(np.uint8) * 255))


def mask_a_minus_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=bool) & ~np.asarray(b, dtype=bool)


def mask_b_minus_a(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(b, dtype=bool) & ~np.asarray(a, dtype=bool)


def mask_intersection(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=bool) & np.asarray(b, dtype=bool)


def mask_union(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=bool) | np.asarray(b, dtype=bool)


def overlay_diff(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    before_bool = np.asarray(before, dtype=bool)
    after_bool = np.asarray(after, dtype=bool)
    h, w = before_bool.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    kept = before_bool & after_bool
    removed = before_bool & ~after_bool
    added = ~before_bool & after_bool
    out[kept] = (40, 180, 90)
    out[removed] = (220, 70, 70)
    out[added] = (70, 110, 220)
    return out


def compute_support_loss(
    before: np.ndarray,
    after: np.ndarray,
    *,
    step: str,
    notes: str = "",
) -> dict[str, Any]:
    before_bool = np.asarray(before, dtype=bool)
    after_bool = np.asarray(after, dtype=bool)
    before_pixels = int(np.count_nonzero(before_bool))
    after_pixels = int(np.count_nonzero(after_bool))
    kept_pixels = int(np.count_nonzero(before_bool & after_bool))
    removed_pixels = int(np.count_nonzero(before_bool & ~after_bool))
    added_pixels = int(np.count_nonzero(~before_bool & after_bool))
    union_pixels = int(np.count_nonzero(before_bool | after_bool))
    iou = float(kept_pixels / union_pixels) if union_pixels else 1.0
    removed_fraction = float(removed_pixels / before_pixels) if before_pixels else 0.0
    added_fraction = float(added_pixels / after_pixels) if after_pixels else 0.0
    return {
        "step": step,
        "before_pixels": before_pixels,
        "after_pixels": after_pixels,
        "removed_pixels": removed_pixels,
        "added_pixels": added_pixels,
        "kept_pixels": kept_pixels,
        "removed_fraction": removed_fraction,
        "added_fraction": added_fraction,
        "iou": iou,
        "notes": notes,
    }


def _generate_derived_masks(
    bundle_dir: Path,
    loaded_masks: dict[str, np.ndarray],
    resolved: ResolvedRun,
    artifact_index: dict[str, Path],
) -> dict[str, Any]:
    generated = 0
    entries: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    missing: list[str] = []

    def ensure_mask(name: str) -> np.ndarray | None:
        if name in loaded_masks:
            return loaded_masks[name]
        source = _resolve_artifact_path(resolved, artifact_index, name)
        if source is None or not source.is_file():
            return None
        mask = load_binary_mask(source)
        if mask is not None:
            loaded_masks[name] = mask
        return mask

    def write_pair(step: str, artifact_id: str, before: np.ndarray | None, after: np.ndarray | None, notes: str) -> None:
        nonlocal generated
        if before is None or after is None:
            skipped.append({"artifact_id": artifact_id, "reason": notes or "missing before/after masks"})
            missing.append(artifact_id)
            return
        if before.shape != after.shape:
            skipped.append({"artifact_id": artifact_id, "reason": f"shape mismatch {before.shape} vs {after.shape}"})
            missing.append(artifact_id)
            return
        diff = mask_a_minus_b(before, after)
        save_binary_mask(bundle_dir / step / f"{artifact_id}.png", diff)
        cv2.imwrite(str(bundle_dir / step / f"{artifact_id}_overlay.png"), overlay_diff(before, after))
        loaded_masks[artifact_id] = diff
        entries.append({"artifact_id": artifact_id, "step": step, "notes": notes})
        generated += 1

    valid = ensure_mask("valid_mask")
    roi = ensure_mask("plane_fit_roi_mask")
    if valid is not None and roi is not None:
        valid_for_fit = mask_intersection(valid, roi)
        save_binary_mask(bundle_dir / "01_roi" / "valid_for_fit_proxy.png", valid_for_fit)
        loaded_masks["valid_for_fit_proxy"] = valid_for_fit
        write_pair("01_roi", "rejected_by_roi", valid, valid_for_fit, "valid minus valid∩roi")
    else:
        skipped.append({"artifact_id": "rejected_by_roi", "reason": "missing valid_mask or plane_fit_roi_mask"})
        missing.append("rejected_by_roi")

    low_grad = ensure_mask("low_gradient_mask")
    gradient_before = valid
    if valid is not None and roi is not None:
        gradient_before = mask_intersection(valid, roi)
    write_pair("02_gradient", "rejected_by_gradient", gradient_before, low_grad, "support to low-gradient")

    flat_candidate = ensure_mask("flat_candidate_mask")
    write_pair("03_height_gate", "rejected_by_height_gate", low_grad, flat_candidate, "low-gradient to flat candidates")

    selected_plateau = ensure_mask("background_selected_plateau_mask")
    write_pair("04_plateaus", "rejected_by_plateau_selection", flat_candidate, selected_plateau, "flat candidates to selected plateau")

    selected_blob_pre = ensure_mask("selected_blob_cluster_pre_refine_mask")
    selected_blob_refined = ensure_mask("selected_blob_cluster_refined_mask")
    selected_blob_final = ensure_mask("selected_blob_cluster_mask")
    rejected_blob_clusters = ensure_mask("rejected_blob_clusters_mask")
    height_border_fragments = ensure_mask("height_border_fragments_mask")
    height_split_fragments = ensure_mask("height_split_blob_fragments_mask")
    if low_grad is not None and selected_blob_pre is not None:
        write_pair("04_blob_clusters", "rejected_by_blob_cluster_selection", low_grad, selected_blob_pre, "low-gradient blobs to selected blob cluster")
    height_aware_rejected_edges = ensure_mask("height_aware_connectivity_rejected_edges")
    if height_aware_rejected_edges is not None:
        save_binary_mask(bundle_dir / "04_blob_clusters" / "rejected_by_height_aware_connectivity.png", height_aware_rejected_edges)
        loaded_masks["rejected_by_height_aware_connectivity"] = height_aware_rejected_edges
        entries.append(
            {
                "artifact_id": "rejected_by_height_aware_connectivity",
                "step": "04_blob_clusters",
                "notes": "pixels participating in rejected height-aware neighbor edges",
            }
        )
        generated += 1
    if low_grad is not None and height_border_fragments is not None:
        write_pair("04_blob_clusters", "rejected_by_height_border_cut", low_grad, height_border_fragments, "low-gradient blobs to height-border fragments")
    if height_border_fragments is not None and height_split_fragments is not None:
        write_pair("04_blob_clusters", "rejected_by_height_border_fragment_filter", height_border_fragments, height_split_fragments, "height-border fragments to final split fragments")
    if height_split_fragments is not None and height_border_fragments is not None:
        write_pair("04_blob_clusters", "restored_by_fragment_merge", height_split_fragments, height_border_fragments, "final split fragments to height-border fragments")
    if selected_blob_pre is not None and selected_blob_refined is not None:
        write_pair("04_blob_clusters", "rejected_by_blob_mad_refinement", selected_blob_pre, selected_blob_refined, "selected blob cluster pre/post MAD refinement")
    elif selected_blob_pre is not None and selected_blob_final is not None:
        write_pair("04_blob_clusters", "rejected_by_blob_mad_refinement", selected_blob_pre, selected_blob_final, "selected blob cluster pre/final refinement")

    belt_base = ensure_mask("belt_base_mask")
    before_stripes = selected_plateau
    if before_stripes is None:
        before_stripes = ensure_mask("reference_surface_selected_mask")
    write_pair("05_stripes", "removed_by_stripe_filter", before_stripes, belt_base, "pre-stripe plateau support to belt base")

    final_inliers = ensure_mask("final_plane_inlier_mask")
    candidate_support = ensure_mask("reference_surface_selected_mask")
    if candidate_support is None:
        candidate_support = ensure_mask("background_candidate_mask")
    write_pair("07_plane_fit", "rejected_by_plane_residual", candidate_support, final_inliers, "candidate support to final plane inliers")

    rejected_plane = ensure_mask("rejected_background_residuals")
    if rejected_plane is not None:
        save_binary_mask(bundle_dir / "09_segmentation" / "rejected_by_segmentation_plane_suppression.png", rejected_plane)
        loaded_masks["rejected_by_segmentation_plane_suppression"] = rejected_plane
        entries.append({"artifact_id": "rejected_by_segmentation_plane_suppression", "step": "09_segmentation", "notes": "copied persisted rejected_background_residuals"})
        generated += 1
    else:
        foreground = ensure_mask("foreground_before_plane_suppression")
        suppressed = ensure_mask("plane_suppressed_mask")
        write_pair("09_segmentation", "rejected_by_segmentation_plane_suppression", foreground, suppressed, "foreground before/after plane suppression")

    rejected_stripes = ensure_mask("rejected_belt_stripes")
    if rejected_stripes is not None:
        save_binary_mask(bundle_dir / "09_segmentation" / "rejected_by_segmentation_stripe_suppression.png", rejected_stripes)
        loaded_masks["rejected_by_segmentation_stripe_suppression"] = rejected_stripes
        entries.append({"artifact_id": "rejected_by_segmentation_stripe_suppression", "step": "09_segmentation", "notes": "copied persisted rejected_belt_stripes"})
        generated += 1
    else:
        skipped.append({"artifact_id": "rejected_by_segmentation_stripe_suppression", "reason": "missing rejected_belt_stripes and no reconstructable before/after pair"})
        missing.append("rejected_by_segmentation_stripe_suppression")

    return {"generated": generated, "entries": entries, "skipped": skipped, "missing": missing}


def _compute_support_losses(loaded_masks: dict[str, np.ndarray], derived_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs = [
        ("rejected_by_roi", "valid_mask", "valid_for_fit_proxy", "ROI gate"),
        ("rejected_by_gradient", "valid_for_fit_proxy", "low_gradient_mask", "Gradient threshold"),
        ("rejected_by_height_gate", "low_gradient_mask", "flat_candidate_mask", "Height gate"),
        ("rejected_by_height_border_cut", "low_gradient_mask", "height_border_fragments_mask", "Height-border cut"),
        ("rejected_by_height_border_fragment_filter", "height_border_fragments_mask", "height_split_blob_fragments_mask", "Height-border fragment filter"),
        ("rejected_by_blob_cluster_selection", "low_gradient_mask", "selected_blob_cluster_pre_refine_mask", "Blob cluster selection"),
        ("rejected_by_blob_mad_refinement", "selected_blob_cluster_pre_refine_mask", "selected_blob_cluster_refined_mask", "Blob MAD refinement"),
        ("rejected_by_plateau_selection", "flat_candidate_mask", "background_selected_plateau_mask", "Plateau selection"),
        ("removed_by_stripe_filter", "background_selected_plateau_mask", "belt_base_mask", "Stripe filter"),
        ("rejected_by_plane_residual", "reference_surface_selected_mask", "final_plane_inlier_mask", "Plane residual expansion"),
        ("rejected_by_segmentation_plane_suppression", "foreground_before_plane_suppression", "plane_suppressed_mask", "Segmentation plane suppression"),
    ]
    rows: list[dict[str, Any]] = []
    for step, before_id, after_id, notes in specs:
        before = loaded_masks.get(before_id)
        after = loaded_masks.get(after_id)
        if before is None or after is None:
            continue
        if before.shape != after.shape:
            continue
        rows.append(compute_support_loss(before, after, step=step, notes=notes))

    return rows


def _write_support_loss_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "step",
        "before_pixels",
        "after_pixels",
        "removed_pixels",
        "added_pixels",
        "kept_pixels",
        "removed_fraction",
        "added_fraction",
        "iou",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _write_stage_params_resolved(resolved: ResolvedRun, dest: Path) -> None:
    payload = _build_resolved_stage_params(resolved.result_payload)
    dest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _build_resolved_stage_params(result_payload: dict[str, Any]) -> dict[str, Any]:
    overrides = result_payload.get("stage_params") if isinstance(result_payload.get("stage_params"), dict) else {}
    defaults = {
        "detect_belt_plane": dataclasses.asdict(DetectBeltPlaneStage()),
        "remove_belt_segment_objects": _dataclass_defaults(RemoveBeltAndSegmentObjectsStage),
        "known_object_25d": _dataclass_defaults(ValidateKnownObjectScale25DStage),
        "normalize_heights_to_plane": _dataclass_defaults(NormalizeHeightsToPlaneStage),
    }
    resolved: dict[str, Any] = {"stage_params": {}, "defaults_available": True, "overrides_available": bool(overrides)}
    for stage_id, stage_defaults in defaults.items():
        stage_overrides = dict(overrides.get(stage_id) or {})
        merged = dict(stage_defaults)
        merged.update(stage_overrides)
        resolved["stage_params"][stage_id] = merged
        if stage_overrides:
            resolved.setdefault("overrides", {})[stage_id] = stage_overrides
        resolved.setdefault("defaults", {})[stage_id] = stage_defaults
    if not overrides:
        resolved["warnings"] = ["result.json did not include stage_params overrides; defaults merged from stage classes only"]
    return resolved


def _stage_params_info(result_payload: dict[str, Any]) -> dict[str, Any]:
    overrides = result_payload.get("stage_params") if isinstance(result_payload.get("stage_params"), dict) else {}
    resolved = _build_resolved_stage_params(result_payload)
    return {
        "defaults": resolved.get("defaults"),
        "overrides": overrides or None,
        "resolved": resolved.get("stage_params"),
        "defaults_missing": not resolved.get("defaults_available", True),
        "overrides_missing": not bool(overrides),
    }


def _extract_derived_thresholds(
    processed_dir: Path,
    artifact_index: dict[str, Path],
    resolved: ResolvedRun,
) -> tuple[dict[str, Any], list[str]]:
    thresholds: dict[str, Any] = {}
    warnings: list[str] = []

    gradient_debug_path = artifact_index.get("gradient_debug") or processed_dir / "gradient_debug.json"
    gradient_payload = _load_debug_dict(
        gradient_debug_path,
        filename="gradient_debug.json",
        skipped_fields="gradient/height_gate thresholds",
        warnings=warnings,
    )
    if gradient_payload:
        thresholds["gradient_threshold"] = gradient_payload.get("gradient_threshold")
        height_gate = gradient_payload.get("height_gate")
        if isinstance(height_gate, dict):
            thresholds["height_gate"] = height_gate

    plateaus_path = artifact_index.get("reference_surface_plateaus") or processed_dir / "reference_surface_plateaus.json"
    plateaus_payload = _load_debug_dict(
        plateaus_path,
        filename="reference_surface_plateaus.json",
        skipped_fields="selected_plateau",
        warnings=warnings,
    )
    if plateaus_payload:
        selected = plateaus_payload.get("selected_plateau") if isinstance(plateaus_payload.get("selected_plateau"), dict) else {}
        thresholds["selected_plateau_z_band"] = selected.get("z_band_mm") or selected.get("z_band")

    stripe_debug_path = artifact_index.get("belt_stripe_filter_debug") or processed_dir / "belt_stripe_filter_debug.json"
    stripe_payload = _load_debug_dict(
        stripe_debug_path,
        filename="belt_stripe_filter_debug.json",
        skipped_fields="stripe threshold and belt_z",
        warnings=warnings,
    )
    if stripe_payload:
        thresholds["stripe_threshold"] = stripe_payload.get("stripe_threshold") or stripe_payload.get("altitude_threshold")
        thresholds["belt_z_estimate"] = stripe_payload.get("belt_z_estimate_mm") or stripe_payload.get("reference_z_mm")

    plane_fit_path = artifact_index.get("plane_fit_debug") or processed_dir / "plane_fit_debug.json"
    plane_fit_payload = _load_debug_dict(
        plane_fit_path,
        filename="plane_fit_debug.json",
        skipped_fields="RANSAC/expansion thresholds",
        warnings=warnings,
    )
    if plane_fit_payload:
        thresholds["ransac_threshold_mm"] = plane_fit_payload.get("ransac_threshold_mm") or plane_fit_payload.get(
            "effective_ransac_threshold_mm"
        )
        thresholds["expansion_tolerance_mm"] = plane_fit_payload.get("expansion_tolerance_mm") or plane_fit_payload.get(
            "residual_tolerance_mm"
        )

    resolved_params = _build_resolved_stage_params(resolved.result_payload).get("stage_params", {})
    detect = resolved_params.get("detect_belt_plane") if isinstance(resolved_params.get("detect_belt_plane"), dict) else {}
    if thresholds.get("gradient_threshold") is None and detect.get("gradient_threshold_value") is not None:
        thresholds["gradient_threshold"] = detect.get("gradient_threshold_value")
    if thresholds.get("ransac_threshold_mm") is None and detect.get("plane_fit_residual_threshold_mm") is not None:
        thresholds["ransac_threshold_mm"] = detect.get("plane_fit_residual_threshold_mm")
    if thresholds.get("expansion_tolerance_mm") is None and detect.get("plane_background_residual_tolerance_mm") is not None:
        thresholds["expansion_tolerance_mm"] = detect.get("plane_background_residual_tolerance_mm")

    height_aware_debug_path = artifact_index.get("height_aware_connectivity_debug") or processed_dir / "height_aware_connectivity_debug.json"
    height_aware_payload = _load_debug_dict(
        height_aware_debug_path,
        filename="height_aware_connectivity_debug.json",
        skipped_fields="height-aware connectivity summary",
        warnings=warnings,
    )
    if height_aware_payload:
        thresholds["blob_component_mode"] = height_aware_payload.get("parameters", {}).get("blob_component_mode") or detect.get("blob_component_mode")
        thresholds["blob_connectivity"] = height_aware_payload.get("parameters", {}).get("blob_connectivity") or detect.get("blob_connectivity")
        thresholds["blob_neighbor_z_tolerance_mm"] = height_aware_payload.get("parameters", {}).get("blob_neighbor_z_tolerance_mm") or detect.get("blob_neighbor_z_tolerance_mm")
        thresholds["height_aware_component_count"] = height_aware_payload.get("component_count")
        thresholds["height_aware_rejected_neighbor_edges"] = height_aware_payload.get("rejected_neighbor_edges")
    elif detect.get("blob_component_mode"):
        thresholds["blob_component_mode"] = detect.get("blob_component_mode")
    return thresholds, warnings


def _write_report_md(bundle_dir: Path, manifest: dict[str, Any], support_losses: list[dict[str, Any]]) -> None:
    missing = manifest.get("artifacts_missing") or []
    derived = manifest.get("derived_thresholds") or {}
    stage_params = manifest.get("stage_params") or {}
    overrides = stage_params.get("overrides") or {}
    lines = [
        "# 25D Reference Surface Audit",
        "",
        f"- **Take:** `{manifest.get('take_id')}`",
        f"- **Run:** `{manifest.get('run_id')}` ({manifest.get('run_descriptor')})",
        f"- **Pipeline:** `{manifest.get('pipeline_id')}`",
        "",
        "This bundle collects persisted reference-surface, stripe-filter, plane-fit, normalization, and segmentation artifacts for one 25D run. Derived diff masks highlight support losses between major steps.",
        "",
        "## Parameter override summary",
    ]
    if overrides:
        lines.append(json.dumps(overrides, indent=2))
    else:
        lines.append("- No stage param overrides recorded in result.json.")
    lines.extend(["", "## Derived threshold summary", json.dumps(derived, indent=2)])
    manifest_warnings = manifest.get("warnings") or []
    lines.extend(["", "## Warnings"])
    if manifest_warnings:
        for item in manifest_warnings:
            lines.append(f"- {item}")
    else:
        lines.append("- None")
    lines.extend(["", "## Missing artifacts"])
    if missing:
        for item in missing:
            lines.append(f"- `{item}`")
    else:
        lines.append("- None")
    lines.extend(["", "## Step-by-step audit", "", "| Step | Before px | After px | Removed px | Removed % | IoU | Notes |", "|---|---:|---:|---:|---:|---:|---|"])
    for row in support_losses:
        lines.append(
            f"| {row.get('step')} | {row.get('before_pixels')} | {row.get('after_pixels')} | {row.get('removed_pixels')} | "
            f"{float(row.get('removed_fraction', 0.0)) * 100:.2f}% | {float(row.get('iou', 0.0)):.3f} | {row.get('notes') or ''} |"
        )
    lines.extend(["", "## Largest support losses"])
    by_fraction = sorted(support_losses, key=lambda item: float(item.get("removed_fraction") or 0.0), reverse=True)
    by_abs = sorted(support_losses, key=lambda item: int(item.get("removed_pixels") or 0), reverse=True)
    if by_fraction:
        top = by_fraction[0]
        lines.append(f"- Highest removed fraction: `{top.get('step')}` ({float(top.get('removed_fraction', 0.0)) * 100:.2f}%)")
    if by_abs:
        top = by_abs[0]
        lines.append(f"- Highest absolute removal: `{top.get('step')}` ({top.get('removed_pixels')} px)")
    lines.extend(["", "## Key images"])
    for rel in (
        "02_gradient/low_gradient_mask.png",
        "04_blob_clusters/selected_blob_cluster_mask.png",
        "04_plateaus/background_selected_plateau_mask.png",
        "05_stripes/belt_stripes_mask.png",
        "06_candidate_support/reference_surface_selected_mask.png",
        "07_plane_fit/final_plane_inlier_mask.png",
        "08_normalization/normalized_heightmap.png",
        "09_segmentation/final_object_mask.png",
    ):
        if (bundle_dir / rel).is_file():
            lines.append(f"- [{rel}]({rel})")
    lines.extend(
        [
            "",
            "## Interpretation notes",
            "- Compare `rejected_by_*` masks with the corresponding overlay images to see where support shrinks.",
            "- If a persisted intermediate is missing, the manifest records it instead of re-running internal stage logic.",
            "- Segmentation plane/stripe suppression diffs prefer persisted `rejected_background_residuals` and `rejected_belt_stripes` when available.",
        ]
    )
    (bundle_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_index_html(bundle_dir: Path, manifest: dict[str, Any]) -> None:
    sections: list[tuple[str, str]] = [
        ("Input", "00_input"),
        ("ROI", "01_roi"),
        ("Gradient / low-gradient", "02_gradient"),
        ("Height gate", "03_height_gate"),
        ("Blob clusters", "04_blob_clusters"),
        ("Plateaus", "04_plateaus"),
        ("Stripes", "05_stripes"),
        ("Candidate support", "06_candidate_support"),
        ("Plane fit", "07_plane_fit"),
        ("Normalization", "08_normalization"),
        ("Segmentation", "09_segmentation"),
        ("Summary", "10_summary"),
    ]
    parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<title>25D Reference Audit</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;margin:24px;color:#111}",
        "h1,h2{margin-bottom:8px}",
        ".meta{color:#444;margin-bottom:24px}",
        ".grid{display:flex;flex-wrap:wrap;gap:12px}",
        ".card{border:1px solid #ddd;border-radius:8px;padding:8px;width:220px}",
        ".card img{max-width:100%;height:auto;display:block;background:#222}",
        ".card a{display:block;margin-top:6px;font-size:12px;word-break:break-all}",
        "code{background:#f5f5f5;padding:2px 4px;border-radius:4px}",
        "</style></head><body>",
        "<h1>25D Reference Surface Audit</h1>",
        f"<div class='meta'>take <code>{manifest.get('take_id')}</code> · run <code>{manifest.get('run_id')}</code> · pipeline <code>{manifest.get('pipeline_id')}</code></div>",
        "<p><a href='report.md'>report.md</a> · <a href='manifest.json'>manifest.json</a> · <a href='stage_params_resolved.json'>stage_params_resolved.json</a></p>",
    ]
    for title, folder in sections:
        step_dir = bundle_dir / folder
        if not step_dir.is_dir():
            continue
        parts.append(f"<h2>{title}</h2><div class='grid'>")
        for path in sorted(step_dir.iterdir()):
            if not path.is_file():
                continue
            rel = f"{folder}/{path.name}"
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                parts.append(f"<div class='card'><img src='{rel}' alt='{path.name}' loading='lazy'><a href='{rel}'>{path.name}</a></div>")
            elif path.suffix.lower() == ".json":
                parts.append(f"<div class='card'><strong>{path.name}</strong><a href='{rel}'>{rel}</a></div>")
            elif path.suffix.lower() == ".csv":
                parts.append(f"<div class='card'><strong>{path.name}</strong><a href='{rel}'>{rel}</a></div>")
        parts.append("</div>")
    parts.append("</body></html>")
    (bundle_dir / "index.html").write_text("\n".join(parts), encoding="utf-8")


def _maybe_contact_sheet(bundle_dir: Path) -> Path | None:
    candidates = [
        bundle_dir / "06_candidate_support" / "reference_surface_selected_mask.png",
        bundle_dir / "05_stripes" / "belt_stripes_mask.png",
        bundle_dir / "07_plane_fit" / "final_plane_inlier_mask.png",
        bundle_dir / "08_normalization" / "normalized_heightmap.png",
        bundle_dir / "09_segmentation" / "final_object_mask.png",
    ]
    images: list[np.ndarray] = []
    for path in candidates:
        if not path.is_file():
            continue
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            continue
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        thumb = cv2.resize(image, (240, 180), interpolation=cv2.INTER_AREA)
        images.append(thumb)
    if not images:
        return None
    cols = min(3, len(images))
    rows = (len(images) + cols - 1) // cols
    canvas = np.zeros((rows * 180, cols * 240, 3), dtype=np.uint8)
    for idx, image in enumerate(images):
        r, c = divmod(idx, cols)
        canvas[r * 180 : (r + 1) * 180, c * 240 : (c + 1) * 240] = image
    out = bundle_dir / "10_summary" / "summary_contact_sheet.png"
    cv2.imwrite(str(out), canvas)
    return out


def _write_acquisition_metadata_excerpt(data_dir: Path, take_id: str, dest: Path) -> None:
    metadata_path = data_dir / "incoming" / take_id / "metadata.json"
    if not metadata_path.is_file():
        dest.write_text(json.dumps({"status": "missing", "path": str(metadata_path)}, indent=2), encoding="utf-8")
        return
    payload = _read_json(metadata_path)
    excerpt = {
        "take_id": payload.get("take_id") or take_id,
        "session_id": payload.get("session_id"),
        "created_at": payload.get("created_at"),
        "files": payload.get("files"),
        "calibration": payload.get("calibration"),
        "source_path": str(metadata_path),
    }
    dest.write_text(json.dumps(excerpt, indent=2, sort_keys=True), encoding="utf-8")


def _build_artifact_index(resolved: ResolvedRun) -> dict[str, Path]:
    index: dict[str, Path] = {}
    payload = resolved.result_payload
    for artifact in payload.get("artifacts") or []:
        if not isinstance(artifact, dict):
            continue
        artifact_id = str(artifact.get("artifact_id") or "")
        rel_path = artifact.get("path")
        if not artifact_id or not rel_path:
            continue
        candidate = (resolved.processed_dir / str(rel_path)).resolve()
        if candidate.is_file():
            index[artifact_id] = candidate
    files = payload.get("files") if isinstance(payload.get("files"), dict) else {}
    for artifact_id, rel_path in files.items():
        if artifact_id in index or not rel_path:
            continue
        candidate = (resolved.processed_dir / str(rel_path)).resolve()
        if candidate.is_file():
            index[artifact_id] = candidate
    for artifact_id, filename in KNOWN_ARTIFACT_FILENAMES.items():
        if artifact_id in index:
            continue
        candidate = resolved.processed_dir / filename
        if candidate.is_file():
            index[artifact_id] = candidate
    debug_names = {
        "gradient_debug": "gradient_debug.json",
        "plane_fit_debug": "plane_fit_debug.json",
        "normalization_debug": "normalization_debug.json",
        "segmentation_debug": "segmentation_debug.json",
    }
    for artifact_id, filename in debug_names.items():
        if artifact_id in index:
            continue
        candidate = resolved.processed_dir / filename
        if candidate.is_file():
            index[artifact_id] = candidate
    return index


def _resolve_artifact_path(resolved: ResolvedRun, artifact_index: dict[str, Path], artifact_id: str) -> Path | None:
    if artifact_id in artifact_index:
        return artifact_index[artifact_id]
    filename = KNOWN_ARTIFACT_FILENAMES.get(artifact_id)
    if filename:
        candidate = resolved.processed_dir / filename
        if candidate.is_file():
            return candidate
    if artifact_id == "heightmap_preview":
        metadata_path = resolved.processed_dir.parent.parent / "incoming" / resolved.take_id / "metadata.json"
        if metadata_path.is_file():
            metadata = _read_json(metadata_path)
            files = metadata.get("files") if isinstance(metadata.get("files"), dict) else {}
            rel = files.get("heightmap_preview") or files.get("reflectance") or files.get("rgb")
            if rel:
                candidate = metadata_path.parent / str(rel)
                if candidate.is_file():
                    return candidate
    return None


def _iter_copy_targets(*, include_overlays: bool) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    seen: set[str] = set()
    for step_items in STEP_ARTIFACTS.values():
        for artifact_id, filename in step_items:
            if artifact_id in seen:
                continue
            if artifact_id.endswith("_overlay") and not include_overlays:
                continue
            if artifact_id in {"support_loss", "support_loss_csv", "summary_contact_sheet", "valid_for_fit_proxy", "rejected_by_roi", "rejected_by_gradient", "rejected_by_height_gate", "rejected_by_plateau_selection", "removed_by_stripe_filter", "rejected_by_plane_residual", "rejected_by_segmentation_plane_suppression", "rejected_by_segmentation_stripe_suppression", "gradient_debug_excerpt", "acquisition_metadata_excerpt"}:
                continue
            seen.add(artifact_id)
            items.append((artifact_id, filename))
    if include_overlays:
        for artifact_id in OVERLAY_ARTIFACTS:
            filename = KNOWN_ARTIFACT_FILENAMES.get(artifact_id) or f"{artifact_id}.png"
            if artifact_id not in seen:
                items.append((artifact_id, filename))
                seen.add(artifact_id)
    return items


def _step_for_artifact(artifact_id: str, filename: str) -> str:
    for step, items in STEP_ARTIFACTS.items():
        for candidate_id, candidate_filename in items:
            if candidate_id == artifact_id or candidate_filename == filename:
                return step
    if artifact_id in OVERLAY_ARTIFACTS:
        if "segmentation" in artifact_id:
            return "09_segmentation"
        if "gradient" in artifact_id:
            return "02_gradient"
        if "belt" in artifact_id:
            return "06_candidate_support"
    return "10_summary"


def _safe_folder_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", ".", "T", ":"} else "_" for ch in value)
    return cleaned or "latest"


def _pipeline_matches(payload: dict[str, Any], pipeline_id: str) -> bool:
    pipeline = payload.get("processing_pipeline") or {}
    actual = str(pipeline.get("id") or "")
    if not pipeline_id:
        return True
    return actual == pipeline_id or not actual


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _payload_shape(value: object, *, max_keys: int = 12) -> dict[str, Any]:
    if isinstance(value, dict):
        return {"type": "dict", "keys": sorted(str(key) for key in value.keys())[:max_keys]}
    if isinstance(value, list):
        return {"type": "list", "count": len(value)}
    if value is None:
        return {"type": "null"}
    return {"type": type(value).__name__}


def _load_debug_dict(
    path: Path,
    *,
    filename: str,
    skipped_fields: str,
    warnings: list[str],
) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        raw = _read_json(path)
    except json.JSONDecodeError as exc:
        warnings.append(f"{filename} could not be parsed: {exc}")
        return {}
    except OSError as exc:
        warnings.append(f"{filename} could not be read: {exc}")
        return {}
    if isinstance(raw, dict):
        return raw
    shape = _payload_shape(raw)
    if shape["type"] == "list":
        warnings.append(f"{filename} payload was list; {skipped_fields} extraction skipped")
    else:
        warnings.append(f"{filename} payload was {shape['type']}; {skipped_fields} extraction skipped")
    return {}


def _dataclass_defaults(cls: type[Any]) -> dict[str, Any]:
    if dataclasses.is_dataclass(cls):
        return dataclasses.asdict(cls())  # type: ignore[call-arg]
    return {}
