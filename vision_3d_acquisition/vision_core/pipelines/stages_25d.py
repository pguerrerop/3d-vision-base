from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import UTC, datetime
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
from vision_3d_acquisition.classifiers.mining_ball_rules import (
    DEFAULT_RULE_PARAMS,
    predict_superclass_from_rules,
    resolve_classifier_rule_set,
)
from vision_3d_acquisition.ml import MLService
from vision_3d_acquisition.ml.features.extractor import features_from_object
from vision_3d_acquisition.ml.features.schemas import feature_ordering_hash
from vision_3d_acquisition.ml.features.ux_contracts import (
    aggregate_operations_summary,
    build_object_feature_ux_summary,
    filter_for_classifier_studio,
    filter_for_operations,
    filter_for_studio,
)
from vision_3d_acquisition.ml.storage import MLStorage
from vision_3d_acquisition.vision_core.geometry.sphere_consistency_features import compute_sphere_consistency_features
from vision_3d_acquisition.vision_core.geometry.surface_sphere_fit import fit_sphere_least_squares
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


_DETECT_REFERENCE_SUBSTAGE_SPECS: dict[str, dict[str, Any]] = {
    "raw_heightmap_preview": {"substage_id": "input", "display_label": "Raw heightmap", "order_index": 0, "semantic_type": "raw_heightmap", "role": "input"},
    "valid_mask": {"substage_id": "input", "display_label": "Valid ROI", "order_index": 0, "semantic_type": "valid_mask", "role": "input"},
    "plane_fit_roi_mask": {"substage_id": "input", "display_label": "Plane-fit ROI", "order_index": 0, "semantic_type": "roi_mask", "role": "diagnostic"},
    "depth_gradient_magnitude": {"substage_id": "gradient", "display_label": "Depth gradient", "order_index": 1, "semantic_type": "gradient_heatmap", "role": "diagnostic"},
    "low_gradient_mask": {"substage_id": "gradient", "display_label": "Low-gradient mask", "order_index": 1, "semantic_type": "low_gradient_mask", "role": "intermediate"},
    "low_gradient_components_overlay": {"substage_id": "gradient", "display_label": "Low-gradient components", "order_index": 1, "semantic_type": "low_gradient_components", "role": "diagnostic"},
    "low_gradient_components": {"substage_id": "gradient", "display_label": "Low-gradient component summary", "order_index": 1, "semantic_type": "low_gradient_components_summary", "role": "diagnostic"},
    "gradient_debug": {"substage_id": "gradient", "display_label": "Gradient debug", "order_index": 1, "semantic_type": "gradient_debug", "role": "diagnostic"},
    "belt_bg_mask": {"substage_id": "height_gate", "display_label": "Belt background mask", "order_index": 2, "semantic_type": "belt_background_mask", "role": "fit_support"},
    "belt_bg_components": {"substage_id": "height_gate", "display_label": "Belt background components", "order_index": 2, "semantic_type": "belt_background_components", "role": "diagnostic"},
    "height_gate_mask": {"substage_id": "height_gate", "display_label": "Height gate mask", "order_index": 2, "semantic_type": "height_gate_mask", "role": "intermediate"},
    "flat_candidate_mask": {"substage_id": "height_gate", "display_label": "Height-gated candidates", "order_index": 2, "semantic_type": "flat_candidate_mask", "role": "intermediate"},
    "reference_surface_candidates": {"substage_id": "height_gate", "display_label": "Reference-surface candidates", "order_index": 2, "semantic_type": "candidate_summary", "role": "diagnostic"},
    "reference_surface_plateaus": {"substage_id": "height_gate", "display_label": "Reference-surface plateaus", "order_index": 2, "semantic_type": "plateau_summary", "role": "diagnostic"},
    "background_selection_debug": {"substage_id": "height_gate", "display_label": "Background selection debug", "order_index": 2, "semantic_type": "background_selection_debug", "role": "diagnostic"},
    "low_gradient_blob_components_overlay": {"substage_id": "component_formation", "display_label": "XY-only components", "order_index": 3, "semantic_type": "blob_components_overlay", "role": "connectivity_support"},
    "low_gradient_blob_id_mask": {"substage_id": "component_formation", "display_label": "XY-only component ids", "order_index": 3, "semantic_type": "blob_components_id_mask", "role": "connectivity_support"},
    "height_aware_blob_components_overlay": {"substage_id": "component_formation", "display_label": "Height-aware components", "order_index": 3, "semantic_type": "height_aware_components_overlay", "role": "connectivity_support"},
    "height_aware_blob_id_mask": {"substage_id": "component_formation", "display_label": "Height-aware component ids", "order_index": 3, "semantic_type": "height_aware_components_id_mask", "role": "connectivity_support"},
    "low_gradient_blob_summary": {"substage_id": "component_formation", "display_label": "Component formation summary", "order_index": 3, "semantic_type": "component_summary", "role": "diagnostic"},
    "component_formation_debug": {"substage_id": "component_formation", "display_label": "Component formation debug", "order_index": 3, "semantic_type": "component_formation_debug", "role": "diagnostic"},
    "height_aware_connectivity_rejected_edges": {"substage_id": "height_aware_connectivity", "display_label": "Rejected Z-connections", "order_index": 4, "semantic_type": "rejected_connectivity_edges", "role": "diagnostic"},
    "height_aware_connectivity_debug": {"substage_id": "height_aware_connectivity", "display_label": "Height-aware connectivity debug", "order_index": 4, "semantic_type": "height_aware_connectivity_debug", "role": "diagnostic"},
    "height_border_strength": {"substage_id": "height_border_detection", "display_label": "Height-border strength", "order_index": 5, "semantic_type": "height_border_strength", "role": "diagnostic"},
    "height_border_cut_mask": {"substage_id": "height_border_detection", "display_label": "Height-border accepted cuts", "order_index": 5, "semantic_type": "height_border_accepted_cuts", "role": "diagnostic"},
    "height_border_detection_debug": {"substage_id": "height_border_detection", "display_label": "Height-border detection debug", "order_index": 5, "semantic_type": "height_border_detection_debug", "role": "diagnostic"},
    "height_border_split_debug": {"substage_id": "blob_splitting", "display_label": "Height-border split debug", "order_index": 6, "semantic_type": "height_border_split_debug", "role": "diagnostic"},
    "height_border_fragments_overlay": {"substage_id": "blob_splitting", "display_label": "Height-border fragments", "order_index": 6, "semantic_type": "height_border_fragments_overlay", "role": "logical_support"},
    "height_border_fragments_mask": {"substage_id": "blob_splitting", "display_label": "Height-border fragments mask", "order_index": 6, "semantic_type": "height_border_fragments_mask", "role": "logical_support"},
    "height_split_blob_fragments_overlay": {"substage_id": "blob_splitting", "display_label": "Height-split blobs", "order_index": 6, "semantic_type": "height_split_fragments_overlay", "role": "logical_support"},
    "height_split_blob_fragments_mask": {"substage_id": "blob_splitting", "display_label": "Height-split blobs mask", "order_index": 6, "semantic_type": "height_split_fragments_mask", "role": "logical_support"},
    "height_split_debug": {"substage_id": "blob_splitting", "display_label": "Blob-splitting debug", "order_index": 6, "semantic_type": "height_split_debug", "role": "diagnostic"},
    "fragment_merge_debug": {"substage_id": "fragment_merge", "display_label": "Fragment-merge debug", "order_index": 7, "semantic_type": "fragment_merge_debug", "role": "diagnostic"},
    "blob_height_clusters": {"substage_id": "height_clustering", "display_label": "Height clusters", "order_index": 8, "semantic_type": "height_clusters", "role": "diagnostic"},
    "blob_cluster_score_table": {"substage_id": "height_clustering", "display_label": "Cluster score table", "order_index": 8, "semantic_type": "cluster_score_table", "role": "diagnostic"},
    "blob_cluster_selection_debug": {"substage_id": "height_clustering", "display_label": "Cluster-scoring debug", "order_index": 8, "semantic_type": "cluster_scoring_debug", "role": "diagnostic"},
    "selected_blob_cluster_mask": {"substage_id": "candidate_support_refinement", "display_label": "Selected logical support", "order_index": 9, "semantic_type": "selected_cluster_mask", "role": "logical_support"},
    "selected_blob_cluster_pre_refine_mask": {"substage_id": "candidate_support_refinement", "display_label": "Selected cluster pre-refine", "order_index": 9, "semantic_type": "selected_cluster_pre_refine", "role": "logical_support"},
    "selected_blob_cluster_refined_mask": {"substage_id": "candidate_support_refinement", "display_label": "Selected cluster refined", "order_index": 9, "semantic_type": "selected_cluster_refined", "role": "fit_support"},
    "support_removed_by_candidate_refinement": {"substage_id": "candidate_support_refinement", "display_label": "Removed by candidate refinement", "order_index": 9, "semantic_type": "support_removed_by_candidate_refinement", "role": "diagnostic"},
    "candidate_support_refinement_debug": {"substage_id": "candidate_support_refinement", "display_label": "Candidate-refinement debug", "order_index": 9, "semantic_type": "candidate_support_refinement_debug", "role": "diagnostic"},
    "belt_base_mask": {"substage_id": "stripes", "display_label": "Belt base mask", "order_index": 10, "semantic_type": "belt_base_mask", "role": "logical_support"},
    "belt_stripes_mask": {"substage_id": "stripes", "display_label": "Belt stripes mask", "order_index": 10, "semantic_type": "belt_stripes_mask", "role": "diagnostic"},
    "belt_stripe_candidates_overlay": {"substage_id": "stripes", "display_label": "Stripe candidate components", "order_index": 10, "semantic_type": "belt_stripe_candidates_overlay", "role": "diagnostic"},
    "belt_stripe_components": {"substage_id": "stripes", "display_label": "Stripe component summary", "order_index": 10, "semantic_type": "belt_stripe_components", "role": "diagnostic"},
    "unknown_low_gradient_mask": {"substage_id": "stripes", "display_label": "Unknown low-gradient mask", "order_index": 10, "semantic_type": "unknown_low_gradient_mask", "role": "diagnostic"},
    "surface_suppression_mask": {"substage_id": "stripes", "display_label": "Surface suppression mask", "order_index": 10, "semantic_type": "surface_suppression_mask", "role": "suppression_mask"},
    "object_search_domain_mask": {"substage_id": "stripes", "display_label": "Object search domain", "order_index": 10, "semantic_type": "object_search_domain_mask", "role": "diagnostic"},
    "belt_above_belt_mask": {"substage_id": "stripes", "display_label": "Above-belt mask", "order_index": 10, "semantic_type": "belt_above_belt_mask", "role": "diagnostic"},
    "belt_wide_object_mask": {"substage_id": "stripes", "display_label": "Wide-object mask", "order_index": 10, "semantic_type": "belt_wide_object_mask", "role": "diagnostic"},
    "support_removed_by_stripe_filter": {"substage_id": "stripes", "display_label": "Removed by stripe suppression", "order_index": 10, "semantic_type": "support_removed_by_stripe_filter", "role": "diagnostic"},
    "belt_stripe_filter_debug": {"substage_id": "stripes", "display_label": "Stripe-filter debug", "order_index": 10, "semantic_type": "stripe_filter_debug", "role": "diagnostic"},
    "selected_reference_support_mask": {"substage_id": "final_support", "display_label": "Final selected support", "order_index": 11, "semantic_type": "final_selected_support_mask", "role": "logical_support"},
    "final_selected_support_mask": {"substage_id": "final_support", "display_label": "Final selected support", "order_index": 11, "semantic_type": "final_selected_support_mask", "role": "logical_support"},
    "reference_model_support_mask": {"substage_id": "final_support", "display_label": "Reference-model fit support", "order_index": 11, "semantic_type": "reference_model_support_mask", "role": "fit_support"},
    "reference_surface_selected_mask": {"substage_id": "final_support", "display_label": "Selected support alias", "order_index": 11, "semantic_type": "reference_surface_selected_mask", "role": "logical_support"},
    "reference_suppression_mask": {"substage_id": "final_support", "display_label": "Reference suppression mask", "order_index": 11, "semantic_type": "reference_suppression_mask", "role": "suppression_mask"},
    "final_support_debug": {"substage_id": "final_support", "display_label": "Final support debug", "order_index": 11, "semantic_type": "final_support_debug", "role": "diagnostic"},
    "support_loss_waterfall": {"substage_id": "final_support", "display_label": "Support-loss waterfall", "order_index": 11, "semantic_type": "support_loss_waterfall", "role": "diagnostic"},
    "belt_plane": {"substage_id": "plane_fit", "display_label": "Reference model", "order_index": 12, "semantic_type": "reference_model_json", "role": "final"},
    "reference_model": {"substage_id": "plane_fit", "display_label": "Reference model", "order_index": 12, "semantic_type": "reference_model_json", "role": "final"},
    "plane_inlier_mask": {"substage_id": "plane_fit", "display_label": "Plane inliers", "order_index": 12, "semantic_type": "plane_inlier_mask", "role": "fit_support"},
    "final_plane_inlier_mask": {"substage_id": "plane_fit", "display_label": "Final plane inliers", "order_index": 12, "semantic_type": "final_plane_inlier_mask", "role": "fit_support"},
    "belt_plane_residuals": {"substage_id": "plane_fit", "display_label": "Plane residual heatmap", "order_index": 12, "semantic_type": "plane_residual_heatmap", "role": "diagnostic"},
    "plane_residual_heatmap": {"substage_id": "plane_fit", "display_label": "Plane residual heatmap", "order_index": 12, "semantic_type": "plane_residual_heatmap", "role": "diagnostic"},
    "plane_residual_histogram": {"substage_id": "plane_fit", "display_label": "Plane residual histogram", "order_index": 12, "semantic_type": "plane_residual_histogram", "role": "diagnostic"},
    "plane_fit_debug": {"substage_id": "plane_fit", "display_label": "Plane-fit debug", "order_index": 12, "semantic_type": "plane_fit_debug", "role": "diagnostic"},
    "selected_surface_debug": {"substage_id": "final_support", "display_label": "Selected support debug", "order_index": 11, "semantic_type": "selected_surface_debug", "role": "diagnostic"},
}


def _detect_reference_artifact_contract(artifact_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
    contract = dict(_DETECT_REFERENCE_SUBSTAGE_SPECS.get(artifact_id, {}))
    if not contract:
        if artifact_id.endswith("_mask"):
            contract["role"] = "diagnostic"
            contract["semantic_type"] = "mask"
        elif artifact_id.endswith("_debug"):
            contract["role"] = "diagnostic"
            contract["semantic_type"] = "debug_json"
        elif artifact_id.endswith("_overlay"):
            contract["role"] = "diagnostic"
            contract["semantic_type"] = "overlay"
        contract.setdefault("substage_id", "normalization")
        contract.setdefault("display_label", str(metadata.get("display_label") or artifact_id.replace("_", " ")))
        contract.setdefault("order_index", 13)
    contract.setdefault("coordinate_space", str(metadata.get("coordinate_space") or "heightmap_pixel"))
    contract.setdefault("kind", str(metadata.get("kind") or "unknown"))
    contract.setdefault("warnings", list(metadata.get("warnings") or []))
    contract.setdefault("fallback_reason", metadata.get("fallback_reason"))
    role = str(contract.get("role") or "diagnostic")
    contract["is_authoritative_for_fit"] = role == "fit_support"
    contract["is_authoritative_for_suppression"] = role == "suppression_mask"
    contract["is_measurement_authoritative"] = False
    return contract


def _make_debug_compact_summary(*, title: str, input_count: int | None = None, output_count: int | None = None, selected_id: str | int | None = None, selected_reason: str | None = None, rejected_reasons: dict[str, int] | None = None, fallback_used: bool | None = None, fallback_reason: str | None = None, warnings: list[str] | None = None, key_parameters: dict[str, Any] | None = None, affected_downstream_artifacts: list[str] | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "title": title,
        "input_count": input_count,
        "output_count": output_count,
        "selected_id": selected_id,
        "selected_reason": selected_reason,
        "rejected_reasons": rejected_reasons or {},
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "warnings": warnings or [],
        "key_parameters": key_parameters or {},
        "affected_downstream_artifacts": affected_downstream_artifacts or [],
    }
    if extra:
        payload.update(extra)
    return payload


def _support_loss_row(
    *,
    row_id: str,
    label: str,
    substage_id: str,
    mask_artifact_id: str | None,
    mask: np.ndarray | None,
    previous_mask: np.ndarray | None,
    valid_roi_mask: np.ndarray,
    parameters_used_summary: dict[str, Any] | None = None,
    largest_loss_reason: str | None = None,
    status: str = "ok",
    warning: str | None = None,
    fallback_reason: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    valid_total = max(1, int(np.count_nonzero(valid_roi_mask)))
    area_px = None if mask is None else int(np.count_nonzero(np.asarray(mask, dtype=bool) & valid_roi_mask))
    prev_area_px = None if previous_mask is None else int(np.count_nonzero(np.asarray(previous_mask, dtype=bool) & valid_roi_mask))
    percent_valid_roi = None if area_px is None else float(area_px / valid_total)
    percent_previous = None if area_px is None or prev_area_px in (None, 0) else float(area_px / max(1, prev_area_px))
    delta_px = None if area_px is None or prev_area_px is None else int(area_px - prev_area_px)
    delta_valid = None if percent_valid_roi is None or previous_mask is None or prev_area_px is None else float(percent_valid_roi - float(prev_area_px / valid_total))
    return {
        "row_id": row_id,
        "label": label,
        "substage_id": substage_id,
        "mask_artifact_id": mask_artifact_id,
        "area_px": area_px,
        "percent_valid_roi": percent_valid_roi,
        "percent_previous": percent_previous,
        "delta_px_from_previous": delta_px,
        "delta_percent_valid_roi_from_previous": delta_valid,
        "largest_loss_reason": largest_loss_reason,
        "status": status,
        "warning": warning,
        "fallback_reason": fallback_reason,
        "parameters_used_summary": parameters_used_summary or {},
        "notes": notes,
    }


def _runtime_trace_mask_unit(
    context: PipelineContext,
    *,
    unit_id: str,
    stage_id: str,
    parent_id: str | None = None,
    status: str = "completed",
    parameters: dict[str, Any] | None = None,
    input_artifacts: list[str] | None = None,
    output_artifacts: list[str] | None = None,
    metrics: dict[str, Any] | None = None,
    diagnostics: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
) -> None:
    with context.trace_unit(unit_id, stage_id=stage_id, parent_id=parent_id) as trace:
        trace.add_parameters(dict(parameters or {}))
        for artifact_id in input_artifacts or []:
            trace.add_input_artifact(artifact_id)
        for artifact_id in output_artifacts or []:
            trace.add_output_artifact(artifact_id)
        for key, value in dict(metrics or {}).items():
            trace.add_metric(key, value)
        for key, value in dict(diagnostics or {}).items():
            trace.add_diagnostic(key, value)
        for message in warnings or []:
            trace.warn(message)
        normalized_status = str(status or "completed").strip().lower()
        if normalized_status in {"skipped", "skip"}:
            trace.skip((diagnostics or {}).get("skip_reason") if isinstance(diagnostics, dict) else None)
        elif normalized_status in {"warning", "warn"}:
            trace.set_status("warning")
        elif normalized_status in {"failed", "error"}:
            trace.set_status("failed")
        else:
            trace.set_status("completed")


def _numeric_summary(values: list[float] | np.ndarray | None) -> dict[str, float]:
    if values is None:
        return {}
    try:
        arr = np.asarray(values, dtype=np.float64)
    except Exception:
        return {}
    if arr.size == 0:
        return {}
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def _enrich_detect_reference_processing_artifacts(context: PipelineContext) -> None:
    for artifact in context.processing_artifacts:
        if str(artifact.get("stage_id") or "") != "detect_belt_plane":
            continue
        metadata = dict(artifact.get("metadata") or {})
        contract = _detect_reference_artifact_contract(str(artifact.get("artifact_id") or ""), metadata)
        metadata.update({
            "artifact_id": artifact.get("artifact_id"),
            "stage_id": artifact.get("stage_id"),
            "kind": artifact.get("kind"),
            **contract,
        })
        metadata.setdefault("source_artifact_ids", list(artifact.get("source_artifact_ids") or []))
        metadata.setdefault("display_label", artifact.get("title"))
        artifact["metadata"] = metadata


def _resolve_background_detection_strategy(
    background_detection_strategy: str,
    support_selection_method: str | None,
) -> str:
    value = str(support_selection_method or "").strip().lower()
    if value == "low_gradient_depth_plateaus":
        return "low_gradient_depth_plateaus"
    if value == "low_gradient_bg_and_stripes":
        return "low_gradient_bg_and_stripes"
    if value == "low_gradient_surface":
        return "low_gradient_surface"
    if value == "blob_height_clusters":
        return "low_gradient_blob_height_clusters"
    if value == "percentile_or_legacy":
        return "nearest_percentile"
    return str(background_detection_strategy or "low_gradient_surface")


def _support_selection_method_from_strategy(background_detection_strategy: str) -> str:
    strategy = str(background_detection_strategy or "").strip().lower()
    if strategy == "low_gradient_depth_plateaus":
        return "low_gradient_depth_plateaus"
    if strategy == "low_gradient_bg_and_stripes":
        return "low_gradient_bg_and_stripes"
    if strategy == "low_gradient_surface":
        return "low_gradient_surface"
    if strategy == "low_gradient_blob_height_clusters":
        return "blob_height_clusters"
    return "percentile_or_legacy"


def _normalize_reference_suppression_mask_policy(value: str | None) -> str:
    policy = str(value or "auto").strip().lower()
    if policy in {"auto", "selected_support", "expanded_support", "final_model_inliers"}:
        return policy
    return "auto"


def _resolve_reference_suppression_mask(
    policy: str,
    *,
    selected_support_mask: np.ndarray,
    expanded_support_mask: np.ndarray,
    final_model_inlier_mask: np.ndarray,
) -> np.ndarray:
    if policy == "selected_support":
        return selected_support_mask.copy()
    if policy == "expanded_support":
        if np.count_nonzero(expanded_support_mask):
            return expanded_support_mask.copy()
        if np.count_nonzero(selected_support_mask):
            return selected_support_mask.copy()
        return final_model_inlier_mask.copy()
    if policy == "final_model_inliers":
        if np.count_nonzero(final_model_inlier_mask):
            return final_model_inlier_mask.copy()
        if np.count_nonzero(expanded_support_mask):
            return expanded_support_mask.copy()
        return selected_support_mask.copy()
    if np.count_nonzero(selected_support_mask):
        return selected_support_mask.copy()
    if np.count_nonzero(expanded_support_mask):
        return expanded_support_mask.copy()
    return final_model_inlier_mask.copy()


def _reference_method_preset_id(
    *,
    background_detection_strategy: str,
    blob_component_mode: str,
    blob_split_method: str,
    reference_surface_model: str,
    reference_suppression_mask_policy: str,
    blob_component_use_smoothed_z: bool,
    blob_neighbor_z_tolerance_mm: float,
) -> str | None:
    if (
        background_detection_strategy == "low_gradient_depth_plateaus"
        and reference_surface_model == "auto"
        and reference_suppression_mask_policy == "auto"
    ):
        return "current_default_plateau_plane"
    if (
        background_detection_strategy == "low_gradient_blob_height_clusters"
        and blob_component_mode == "xy_only"
        and blob_split_method == "height_borders"
        and reference_surface_model == "auto"
        and reference_suppression_mask_policy == "auto"
    ):
        return "blob_xy_height_borders"
    if (
        background_detection_strategy == "low_gradient_blob_height_clusters"
        and blob_component_mode == "height_aware"
        and blob_split_method == "disabled"
        and not blob_component_use_smoothed_z
        and abs(float(blob_neighbor_z_tolerance_mm) - 2.0) < 1e-6
        and reference_surface_model == "auto"
        and reference_suppression_mask_policy == "auto"
    ):
        return "blob_height_aware_auto"
    if (
        background_detection_strategy == "low_gradient_blob_height_clusters"
        and blob_component_mode == "height_aware"
        and blob_split_method == "disabled"
        and not blob_component_use_smoothed_z
        and abs(float(blob_neighbor_z_tolerance_mm) - 2.0) < 1e-6
        and reference_surface_model == "constant_z"
        and reference_suppression_mask_policy == "selected_support"
    ):
        return "blob_height_aware_constant_z"
    if (
        background_detection_strategy == "low_gradient_blob_height_clusters"
        and blob_component_mode == "height_aware"
        and blob_split_method == "disabled"
        and reference_surface_model == "plane"
        and reference_suppression_mask_policy == "final_model_inliers"
    ):
        return "blob_height_aware_strict_plane_qa"
    return None


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
        source_json_name = "source_json.json"
        raw_semantic_raster = "raw_sensor_z.values.f32"
        save_heightmap_npz(frame, output_dir / npz_name)
        write_heightmap_preview_png(frame.z_mm, frame.valid_mask, output_dir / preview_name)
        write_heightmap_metadata_json(frame, output_dir / meta_name, extra={"source_file": height_file})
        (output_dir / source_json_name).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
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
        context.set_artifact("input_source_modalities", list(context.get_artifact("modalities", []) or []))

        valid_pixel_count = int(np.count_nonzero(frame.valid_mask))
        total_pixels = int(frame.valid_mask.size)
        valid_ratio = float(valid_pixel_count / max(1, total_pixels))

        _runtime_trace_mask_unit(
            context,
            unit_id="input.acquisition_metadata",
            stage_id="input",
            parent_id="input",
            output_artifacts=["source_json"],
            metrics={
                "frame_count": int(context.get_artifact("frame_count", 1) or 1),
                "source_modality_count": int(len(list(context.get_artifact("modalities", []) or []))),
                "asset_count": int(len(list(context.get_artifact("assets", []) or []))),
            },
            diagnostics={
                "session_id": metadata.get("session_id"),
                "take_id": metadata.get("take_id"),
                "modalities": list(context.get_artifact("modalities", []) or []),
            },
        )
        _runtime_trace_mask_unit(
            context,
            unit_id="input.raw_heightmap",
            stage_id="input",
            parent_id="input",
            input_artifacts=["source_json"],
            output_artifacts=["raw_heightmap_preview"],
            metrics={
                "width_px": int(frame.z_mm.shape[1]),
                "height_px": int(frame.z_mm.shape[0]),
                "valid_pixel_count": valid_pixel_count,
                "invalid_pixel_count": int(total_pixels - valid_pixel_count),
                "valid_pixel_ratio": valid_ratio,
                "raw_z_min_mm": float(input_min_mm),
                "raw_z_max_mm": float(input_max_mm),
            },
        )
        _runtime_trace_mask_unit(
            context,
            unit_id="input.valid_mask",
            stage_id="input",
            parent_id="input",
            input_artifacts=["raw_heightmap_preview"],
            output_artifacts=["valid_mask"],
            metrics={
                "width_px": int(frame.valid_mask.shape[1]),
                "height_px": int(frame.valid_mask.shape[0]),
                "valid_pixel_count": valid_pixel_count,
                "invalid_pixel_count": int(total_pixels - valid_pixel_count),
                "valid_pixel_ratio": valid_ratio,
            },
        )

        context.add_processing_artifact(
            artifact_id="source_heightmap_preview",
            stage_id="input",
            kind="image",
            title="Source height preview",
            path=preview_name,
            mime_type="image/png",
            preview_available=True,
            projection_type="heightmap",
            projection_coordinate_system={"pixel_per_mm": 1.0 / max(frame.x_resolution_mm, 1e-6)},
            metadata={"source": "native_pipeline", "producer": self.name, "artifact_alias": "heightmap"},
        )
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
            artifact_id="source_json",
            stage_id="input",
            kind="json",
            title="Source metadata JSON",
            path=source_json_name,
            mime_type="application/json",
            preview_available=True,
            metadata=_numeric_source_meta({"source": "native_pipeline", "producer": self.name, **(metadata if isinstance(metadata, dict) else {})}),
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
        output_dir: Path = context.require_artifact("output_dir")
        calibration = {
            "x_resolution_mm": float(((metadata.get("calibration") or {}).get("x_resolution_mm") or 1.0)),
            "y_resolution_mm": float(((metadata.get("calibration") or {}).get("profile_distance_mm") or 1.0)),
            "z_scale": float(((metadata.get("calibration") or {}).get("z_scale") or 1.0)),
            "z_offset": float(((metadata.get("calibration") or {}).get("z_offset") or 0.0)),
            "coordinate_system": "belt_xy_z_mm",
            "tilt_correction": None,
        }
        context.set_artifact("heightmap_calibration", calibration)
        source_metadata_name = "source_metadata.json"
        (output_dir / source_metadata_name).write_text(json.dumps(calibration, indent=2), encoding="utf-8")
        context.add_processing_artifact(
            artifact_id="source_metadata",
            stage_id="input",
            kind="json",
            title="Source calibration context",
            path=source_metadata_name,
            mime_type="application/json",
            preview_available=True,
            metadata=_numeric_source_meta({"source": "native_pipeline", "producer": self.name, **calibration}),
        )
        _runtime_trace_mask_unit(
            context,
            unit_id="input.calibration_context",
            stage_id="input",
            parent_id="input",
            input_artifacts=["source_json"],
            output_artifacts=["source_metadata"],
            metrics={
                "x_resolution_mm": float(calibration["x_resolution_mm"]),
                "y_resolution_mm": float(calibration["y_resolution_mm"]),
                "z_scale": float(calibration["z_scale"]),
                "z_offset": float(calibration["z_offset"]),
            },
            diagnostics={
                "coordinate_system": calibration.get("coordinate_system"),
                "calibration_resolution_source": context.get_artifact("calibration_resolution_source", "metadata"),
            },
        )
        frame: HeightmapFrame = context.require_artifact("heightmap_frame")
        valid_pixel_count = int(np.count_nonzero(frame.valid_mask))
        _runtime_trace_mask_unit(
            context,
            unit_id="input",
            stage_id="input",
            input_artifacts=["source_json"],
            output_artifacts=["source_heightmap_preview", "source_json", "source_metadata", "valid_mask"],
            metrics={
                "width_px": int(frame.z_mm.shape[1]),
                "height_px": int(frame.z_mm.shape[0]),
                "valid_pixel_count": valid_pixel_count,
                "valid_pixel_ratio": float(valid_pixel_count / max(1, frame.valid_mask.size)),
                "source_modality_count": int(len(list(context.get_artifact("modalities", []) or []))),
            },
            diagnostics={"coordinate_system": calibration.get("coordinate_system")},
        )


@dataclass
class DetectBeltPlaneStage:
    background_detection_strategy: str = "low_gradient_surface"
    support_selection_method: str | None = None
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
    low_gradient_plateau_use_hessian_filter: bool = True
    low_gradient_plateau_hessian_percentile: float = 70.0
    low_gradient_plateau_hist_bins: int = 96
    low_gradient_plateau_min_fraction: float = 0.05
    low_gradient_plateau_min_pixels: int = 500
    # Smoothing/peak-detection knobs (used by _detect_depth_plateaus). A wider
    # smoothing kernel and a relaxed peak-drop ratio recover broader plateaus
    # (e.g. a slightly-tilted belt whose z histogram fans out over many bins).
    low_gradient_plateau_smoothing_sigma_bins: float = 2.5
    low_gradient_plateau_peak_drop_ratio: float = 0.40
    # Selection: prefer a *dominant* plateau (≥ select_min_area_fraction of the
    # flat candidates) before falling back to lowest-z. This stops the chevron
    # ribs from winning just because they form a sharper, smaller peak than
    # the noisier belt distribution.
    low_gradient_plateau_select_min_area_fraction: float = 0.20
    low_gradient_plateau_selection_mode: str = "lowest_dominant"  # lowest_dominant | largest | lowest
    # Belt-stripe filter: after BG-plateau selection, separate true belt pixels
    # from coplanar/embossed surface texture (chevron ribs, etc.) via a local-min
    # baseline (morphological top-hat) over the BG-plateau pixels only.
    #
    # Defaults are sized for raised chevron ribs ≈ 5–20 mm tall (typical for
    # textured industrial conveyor belts; e.g. the Trispector test scenes have
    # ~14 mm stripes). For sub-mm surface noise rather than discrete ribs,
    # lower `min_altitude_mm` and `fixed_threshold_mm`, and consider raising
    # `min_stripe_fraction` so the filter doesn't fire on belt micro-texture.
    belt_stripe_filter_enabled: bool = True
    belt_stripe_filter_window_mm: float = 30.0       # window > stripe period; 3 cm covers ≥1 valley+crest at typical chevron pitch
    belt_stripe_filter_direction: str = "auto"       # raised | recessed | auto (picks the more-bimodal hat)
    belt_stripe_filter_threshold_mode: str = "otsu"  # otsu | k_mad | fixed
    belt_stripe_filter_min_altitude_mm: float = 10.0  # Otsu floor — well above belt noise & macro-tilt; ≈0.7× typical rib height (14 mm)
    belt_stripe_filter_k_mad: float = 3.0
    belt_stripe_filter_fixed_threshold_mm: float = 10.0  # ~ 0.7 × typical rib height (14 mm) → clean belt/rib separation
    belt_stripe_filter_min_stripe_fraction: float = 0.02  # below this we treat the BG plateau as a clean belt and skip the split
    # Apply the local-min top-hat globally (across valid_for_fit) instead of
    # restricting it to the BG plateau. Required so stripes that fall outside
    # the BG plateau band (i.e. above belt-z) are still recognised and excluded
    # from expanded_plane_mask + foreground candidates. Baseline = erode(z) so
    # altitude = z − local_min(window); narrow raised ribs survive, and the
    # baseline never sits above the pixel or its neighbours in the window.
    belt_stripe_filter_scope: str = "global"  # global | bg_plateau
    # ── Complementary z-floor + shape pass ──────────────────────────────
    # The erosion top-hat alone fails for one common case:
    #   1. A stripe that becomes locally wider than the kernel — erosion
    #      baseline ≈ stripe_z inside the wide rib, so altitude ≈ 0.
    # The shape-based pass sidesteps both: it classifies every above-belt
    # pixel as "raised" (purely a z-threshold over a robust belt-z estimate
    # taken from the BG plateau median), then carves out genuine objects
    # via a binary morphological opening with an object-sized kernel.
    # Pixels left after the carve are stripes/texture (narrow blobs).
    belt_stripe_filter_z_floor_enabled: bool = True
    # Use the BG plateau's UPPER bound (≈ z_hi from the histogram) as the
    # belt reference instead of the median. Required so the threshold
    # genuinely sits above the entire belt distribution — otherwise tilted /
    # curved belt areas in the upper tail of the plateau (e.g. corners) get
    # wrongly classified as "above belt". The median fallback is kept for
    # diagnostic compatibility (belt_z_estimate_mm in JSON).
    belt_stripe_filter_z_floor_use_upper_bound: bool = True
    belt_stripe_filter_z_floor_margin_mm: float = 20.0  # safety margin above belt_z_upper
    # Anything higher than (belt_z_upper + max_stripe_height_mm) is treated
    # as an object / outlier and excluded from the stripe candidate set
    # regardless of width. Set to 0 to disable this cap. Default 500 mm
    # comfortably covers chevron ribs (≈ 14 mm), tall belt-side guides
    # (≈ 200–300 mm), and excludes typical large foreground objects.
    belt_stripe_filter_max_stripe_height_mm: float = 500.0
    # Close-then-open the above-belt mask before the object/width filter.
    # The close step bridges small interior holes in the object mask (sensor
    # noise, occlusion gaps) so the opening doesn't shave off thin tails of
    # an otherwise-wide object and dump them into the stripe set.
    belt_stripe_filter_above_belt_close_mm: float = 30.0
    belt_stripe_filter_object_kernel_mm: float = 100.0  # binary opening kernel that separates objects (≥ this width) from stripes (< this width)
    belt_stripe_filter_object_kernel_shape: str = "ellipse"  # ellipse | rect — ellipse is smoother on round/irregular objects
    # ── Previously-hardcoded knobs surfaced for tuning ──────────────────
    # All defaults match the pre-existing hardcoded constants so behavior
    # is unchanged unless the user overrides them.
    belt_stripe_filter_altitude_hist_bins: int = 64
    belt_stripe_filter_auto_bimodality_margin: float = 1.10        # recessed wins only if bm > raised * margin
    belt_stripe_filter_z_floor_upper_percentile: float = 99.0       # plateau percentile used as the "above belt" reference
    belt_stripe_filter_z_floor_fallback_lower_percentile: float = 10.0  # used when no BG plateau is available
    belt_stripe_filter_z_floor_fallback_upper_percentile: float = 25.0
    belt_stripe_filter_warn_removed_fraction: float = 0.40          # warn if BG-scope filter removes > this fraction of BG plateau
    stripe_min_height_over_base_mm: float = 6.0
    stripe_max_height_over_base_mm: float = 40.0
    stripe_min_width_mm: float = 5.0
    stripe_max_width_mm: float = 80.0
    stripe_min_height_width_ratio: float = 0.03
    stripe_max_height_width_ratio: float = 2.0
    stripe_max_width_cv: float = 0.75
    stripe_min_length_width_ratio: float = 1.5
    stripe_max_area_fraction: float = 0.08
    stripe_component_min_overlap_fraction: float = 0.50
    stripe_rule_overlap_enabled: bool = True
    stripe_rule_height_range_enabled: bool = True
    stripe_rule_width_range_enabled: bool = True
    stripe_rule_height_width_ratio_enabled: bool = True
    stripe_rule_width_cv_enabled: bool = True
    stripe_rule_length_width_ratio_enabled: bool = True
    stripe_rule_area_fraction_enabled: bool = True
    stripe_height_mm: float = 20.0                                   # target belt-stripe height above the belt-bg baseline (bg_and_stripes blob classification)
    stripe_height_tolerance_mm: float = 15.0                         # +/- tolerance band around stripe_height_mm for blob-level stripe classification
    low_gradient_plateau_robust_band_mad_k: float = 3.0             # k for median ± k·MAD fallback band in plateau detection
    low_gradient_plateau_detection_min_count_floor: int = 25        # absolute floor on the laxer per-bin detection threshold
    low_gradient_plateau_detection_min_count_fraction: float = 0.25 # fraction of min_pixels used as the detection floor
    # ── Experimental low-gradient blob + height-cluster strategy ───────────
    blob_cluster_min_component_area: int = 250
    blob_cluster_height_gap_mm: float = 8.0
    blob_cluster_height_gap_mode: str = "adaptive"  # fixed | adaptive
    blob_cluster_min_area_fraction: float = 0.10
    blob_cluster_min_total_pixels: int = 1200
    blob_cluster_max_z_mad_mm: float = 12.0
    blob_cluster_merge_close_clusters: bool = True
    blob_cluster_merge_gap_mm: float = 4.0
    blob_cluster_area_weight: float = 1.0
    blob_cluster_border_weight: float = 0.45
    blob_cluster_constancy_weight: float = 0.75
    blob_cluster_spatial_coverage_weight: float = 0.40
    blob_cluster_depth_preference_weight: float = 0.20
    blob_cluster_fragmentation_penalty: float = 0.25
    blob_cluster_centrality_penalty: float = 0.20
    blob_cluster_object_like_penalty: float = 0.25
    blob_cluster_stripe_overlap_penalty: float = 0.20
    blob_cluster_refine_by_mad: bool = True
    blob_cluster_refine_mad_k: float = 2.5
    blob_cluster_refine_floor_mm: float = 1.0
    blob_cluster_refine_keep_border_support: bool = True
    blob_cluster_fallback_strategy: str = "low_gradient_depth_plateaus"  # low_gradient_depth_plateaus | low_gradient_surface | constant_z | fail
    blob_split_by_height_enabled: bool = True
    blob_split_method: str = "height_borders"  # disabled | histogram_gap | height_borders | height_borders_then_histogram
    blob_split_min_height_range_mm: float = 8.0
    blob_split_min_pixels: int = 300
    blob_split_gap_mm: float = 4.0
    blob_split_mode: str = "histogram_gap"  # histogram_gap
    blob_split_hist_bins: int = 48
    blob_split_min_band_fraction: float = 0.08
    blob_split_min_band_pixels: int = 150
    blob_split_merge_gap_mm: float = 2.0
    blob_split_refine_morphology: bool = True
    blob_split_open_kernel: int = 1
    blob_split_close_kernel: int = 3
    blob_split_height_border_enabled: bool = True
    blob_split_height_border_mode: str = "morphological_gradient"  # sobel_gradient | local_range | morphological_gradient
    blob_split_height_border_smoothing_kernel: int = 5
    blob_split_height_border_threshold_mode: str = "percentile"  # percentile | fixed | otsu
    blob_split_height_border_percentile: float = 85.0
    blob_split_height_border_min_delta_mm: float = 4.0
    blob_split_height_border_dilate_kernel: int = 3
    blob_split_height_border_close_kernel: int = 3
    blob_split_height_border_min_length_px: int = 20
    blob_split_min_fragment_area_px: int = 300
    blob_split_merge_weak_boundaries: bool = True
    blob_split_merge_max_median_z_gap_mm: float = 3.0
    blob_split_merge_max_boundary_strength_mm: float = 6.0
    blob_split_merge_background_like_only: bool = True
    blob_component_mode: str = "xy_only"  # xy_only | height_aware
    blob_connectivity: int = 8
    blob_neighbor_z_tolerance_mm: float = 3.0
    blob_component_z_tolerance_mm: float = 8.0
    blob_component_tolerance_mode: str = "fixed"  # fixed | mad_adaptive
    blob_component_mad_k: float = 3.0
    blob_component_mad_floor_mm: float = 1.0
    blob_component_allow_gradual_slope: bool = True
    blob_component_max_local_slope_mm_per_px: float = 3.0
    blob_component_min_area_px: int = 300
    blob_component_use_smoothed_z: bool = True
    blob_component_z_smoothing_kernel: int = 3
    # ── Low-gradient surface (legacy strategy) tuning ────────────────────
    # These were previously hardcoded inside the strategy branch.
    low_gradient_surface_support_z_mad_multiplier: float = 2.5  # extends candidate's z window by k·MAD
    low_gradient_surface_support_z_floor_mm: float = 1.0        # never extend by less than this many mm
    low_gradient_surface_support_z_mad_floor_mm: float = 0.25   # MAD floor used inside max(floor, k·MAD)
    low_gradient_surface_ridge_percentile: float = 90.0         # gradient percentile used as ridge rejection threshold
    # Height-gate gap detection (used to clip away foreground modes from
    # the candidate flat-pixel set).
    reference_surface_height_gate_gap_floor_mm: float = 1.0
    reference_surface_height_gate_gap_ratio: float = 8.0
    # Depth-plot rendering — adaptive caps for the Studio diagnostics.
    plot_depth_plot_max_render_samples: int = 60000
    plot_y_robust_percentile: float = 98.0   # used to clip y-axis ceilings in histograms
    reference_surface_selection_mode: str = "largest_constant_z"
    reference_surface_region_mode: str = "none"
    reference_surface_min_area_ratio: float = 0.08
    reference_surface_max_z_std_mm: float = 8.0
    reference_surface_border_bonus: float = 0.2
    reference_surface_depth_preference_weight: float = 0.2
    reference_surface_constancy_weight: float = 0.5
    reference_surface_area_weight: float = 0.3
    reference_surface_model: str = "auto"
    reference_suppression_mask_policy: str = "auto"
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
        background_detection_strategy = _resolve_background_detection_strategy(
            self.background_detection_strategy,
            self.support_selection_method,
        )
        support_selection_method = _support_selection_method_from_strategy(background_detection_strategy)
        reference_suppression_mask_policy = _normalize_reference_suppression_mask_policy(
            self.reference_suppression_mask_policy,
        )
        write_heightmap_preview_png(frame.z_mm, frame.valid_mask, output_dir / "raw_heightmap_preview.png")
        raw_min_mm, raw_max_mm = _valid_min_max_mm(frame.z_mm, frame.valid_mask)
        valid_mask_u8 = (frame.valid_mask.astype(np.uint8) * 255)
        cv2.imwrite(str(output_dir / "valid_mask.png"), valid_mask_u8)

        roi_cfg = self.plane_fit_roi or _roi_from_metadata(context.get_artifact("metadata", {}))
        region_mode = str(self.reference_surface_region_mode or "none").lower()
        if (
            region_mode == "none"
            and isinstance(roi_cfg, dict)
            and bool(roi_cfg.get("enabled", True))
        ):
            region_mode = str(roi_cfg.get("type") or roi_cfg.get("roi_type") or "rectangle").lower()
            if region_mode == "vertical_band":
                region_mode = "full_height_x_band"
        if region_mode not in {"none", "full_height_x_band", "rectangle", "polygon"}:
            region_mode = "none"
        if region_mode == "none":
            roi_mask = np.ones_like(frame.valid_mask, dtype=bool)
        else:
            roi_mask = _roi_mask(frame.valid_mask.shape, roi_cfg or {}, region_mode=region_mode)
        cv2.imwrite(str(output_dir / "plane_fit_roi_mask.png"), (roi_mask.astype(np.uint8) * 255))
        valid_for_fit = frame.valid_mask & roi_mask
        valid_pixel_count = int(np.count_nonzero(valid_for_fit))
        roi_enabled = region_mode != "none"
        roi_type = region_mode
        roi_x = int(roi_cfg.get("x", 0)) if isinstance(roi_cfg, dict) else 0
        roi_w = int(roi_cfg.get("width", frame.valid_mask.shape[1])) if isinstance(roi_cfg, dict) else int(frame.valid_mask.shape[1])
        if roi_type == "full_height_x_band":
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
        raised_candidate_rejected_mask = np.zeros_like(valid_for_fit, dtype=bool)
        # Belt-stripe filter outputs are pre-initialised so non-plateau
        # strategies still expose a well-defined (empty) mask to downstream
        # stages (segmentation reads `belt_stripes_mask_array` to suppress
        # surface texture from foreground detection).
        belt_base_mask = candidate_mask.copy()
        low_grad_mask = np.zeros_like(valid_for_fit, dtype=bool)
        stripes_mask = np.zeros_like(valid_for_fit, dtype=bool)
        belt_bg_mask = np.zeros_like(valid_for_fit, dtype=bool)
        unknown_low_gradient_mask = np.zeros_like(valid_for_fit, dtype=bool)
        surface_suppression_mask = np.zeros_like(valid_for_fit, dtype=bool)
        object_search_domain_mask = valid_for_fit.copy()
        inferred_depth_convention = "unknown"
        candidate_coverage = 0.0
        z_thresholds: dict[str, float] = {}
        height_gate_mask = valid_for_fit.copy()
        active_height_gate_mask = valid_for_fit.copy()
        height_gate_debug: dict[str, Any] = {"enabled": bool(self.reference_surface_height_gate_enabled), "status": "not_evaluated"}
        gradient_debug: dict[str, Any] = {}
        low_gradient_component_rows: list[dict[str, Any]] = []
        belt_bg_component_rows: list[dict[str, Any]] = []
        belt_stripe_component_rows: list[dict[str, Any]] = []
        stripe_info: dict[str, Any] = {"enabled": bool(self.belt_stripe_filter_enabled), "applied": False, "skip_reason": "disabled"}
        altitude_hist_payload: dict[str, Any] = {"sample_count": 0, "bin_counts": [], "bin_edges": [], "threshold_value_mm": 0.0, "direction_used": str(self.belt_stripe_filter_direction)}
        baseline_map = np.zeros_like(frame.z_mm, dtype=np.float32)
        altitude_map = np.zeros_like(frame.z_mm, dtype=np.float32)
        filter_scope = str(self.belt_stripe_filter_scope or "global").strip().lower()
        selected_cluster: dict[str, Any] | None = None
        selected_pre_refine_mask = np.zeros_like(valid_for_fit, dtype=bool)
        selected_refined_mask = np.zeros_like(valid_for_fit, dtype=bool)
        blob_rows: list[dict[str, Any]] = []
        fragment_rows: list[dict[str, Any]] = []
        blob_labels = np.zeros_like(valid_for_fit, dtype=np.int32)
        fragment_labels = np.zeros_like(valid_for_fit, dtype=np.int32)
        border_cut_mask = np.zeros_like(valid_for_fit, dtype=bool)
        border_fragment_mask = np.zeros_like(valid_for_fit, dtype=bool)
        height_aware_rejected_edge_mask = np.zeros_like(valid_for_fit, dtype=bool)
        height_aware_connectivity_debug: dict[str, Any] = {}
        blob_component_mode_used = str(self.blob_component_mode)
        fallback_used = False
        fallback_reason: str | None = None
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
                gap_floor_mm=float(self.reference_surface_height_gate_gap_floor_mm),
                gap_ratio=float(self.reference_surface_height_gate_gap_ratio),
            )
            active_height_gate_mask = height_gate_mask if background_detection_strategy in {"low_gradient_surface", "low_gradient_depth_plateaus", "low_gradient_blob_height_clusters", "low_gradient_bg_and_stripes"} else valid_for_fit

            if background_detection_strategy in {"low_gradient_surface", "low_gradient_depth_plateaus", "low_gradient_blob_height_clusters", "low_gradient_bg_and_stripes"}:
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
                    low_grad_mask = (u8 > 0) & valid_for_fit
                raised_candidate_rejected_mask = np.zeros_like(valid_for_fit, dtype=bool)
                write_heightmap_preview_png(grad, valid_for_fit, output_dir / "depth_gradient_magnitude.png")
                cv2.imwrite(str(output_dir / "low_gradient_mask.png"), (low_grad_mask.astype(np.uint8) * 255))
                if background_detection_strategy == "low_gradient_surface":
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
                    if np.count_nonzero(candidate_mask):
                        zvals = frame.z_mm[candidate_mask].astype(np.float64)
                        z_med = float(np.median(zvals))
                        z_mad = float(np.median(np.abs(zvals - z_med)))
                        z_mad_floor = float(self.low_gradient_surface_support_z_mad_floor_mm)
                        z_mad_mult = float(self.low_gradient_surface_support_z_mad_multiplier)
                        z_floor_mm = float(self.low_gradient_surface_support_z_floor_mm)
                        max_support_z = z_med + max(z_floor_mm, z_mad_mult * max(z_mad_floor, z_mad))
                        ridge_pct = float(np.clip(self.low_gradient_surface_ridge_percentile, 1.0, 99.0))
                        ridge_thr = float(np.percentile(grad[candidate_mask], ridge_pct)) if np.count_nonzero(candidate_mask) else 0.0
                        keep_mask = candidate_mask & (frame.z_mm <= max_support_z) & (grad <= ridge_thr)
                        raised_candidate_rejected_mask = candidate_mask & (~keep_mask)
                        candidate_mask = keep_mask
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
                    labels = np.asarray(lg_result.get("labels"), dtype=np.int32)
                    cmap = _label_colormap_overlay(labels)
                    cv2.imwrite(str(output_dir / "low_gradient_components_overlay.png"), cmap)
                    (output_dir / "low_gradient_components.json").write_text(json.dumps(lg_result.get("candidates", []), indent=2), encoding="utf-8")
                    (output_dir / "reference_surface_candidates.json").write_text(json.dumps(lg_result.get("candidates", []), indent=2), encoding="utf-8")
                    (output_dir / "gradient_debug.json").write_text(json.dumps(gradient_debug, indent=2), encoding="utf-8")
                elif background_detection_strategy == "low_gradient_bg_and_stripes":
                    blob_component_mode = _normalize_blob_component_mode(self.blob_component_mode)
                    blob_result = _summarize_low_gradient_blobs(
                        low_grad_mask=low_grad_mask,
                        gradient=grad,
                        z=frame.z_mm,
                        valid_mask=valid_for_fit,
                        roi_mask=roi_mask,
                        height_gate_mask=height_gate_mask,
                        min_component_area=max(1, int(self.blob_cluster_min_component_area)),
                        component_mode=blob_component_mode,
                        blob_connectivity=max(4, int(self.blob_connectivity)),
                        blob_neighbor_z_tolerance_mm=float(self.blob_neighbor_z_tolerance_mm),
                        blob_component_z_tolerance_mm=float(self.blob_component_z_tolerance_mm),
                        blob_component_tolerance_mode=str(self.blob_component_tolerance_mode),
                        blob_component_mad_k=float(self.blob_component_mad_k),
                        blob_component_mad_floor_mm=float(self.blob_component_mad_floor_mm),
                        blob_component_allow_gradual_slope=bool(self.blob_component_allow_gradual_slope),
                        blob_component_max_local_slope_mm_per_px=float(self.blob_component_max_local_slope_mm_per_px),
                        blob_component_min_area_px=max(1, int(self.blob_component_min_area_px)),
                        blob_component_use_smoothed_z=bool(self.blob_component_use_smoothed_z),
                        blob_component_z_smoothing_kernel=max(1, int(self.blob_component_z_smoothing_kernel)),
                    )
                    blob_labels = np.asarray(blob_result.get("labels"), dtype=np.int32)
                    blob_rows = list(blob_result.get("blobs") or [])
                    blob_component_mode_used = str(blob_result.get("component_mode") or blob_component_mode)
                    blob_id_cmap = _label_colormap_overlay(blob_labels)
                    cv2.imwrite(str(output_dir / "low_gradient_blob_components_overlay.png"), blob_id_cmap)
                    cv2.imwrite(str(output_dir / "low_gradient_blob_id_mask.png"), (blob_labels % 255).astype(np.uint8))
                    if blob_component_mode_used == "height_aware":
                        cv2.imwrite(str(output_dir / "height_aware_blob_components_overlay.png"), blob_id_cmap)
                        cv2.imwrite(str(output_dir / "height_aware_blob_id_mask.png"), (blob_labels % 255).astype(np.uint8))
                    low_gradient_component_rows = []
                    flat_candidates_pre_hessian = low_grad_mask & height_gate_mask
                    flat_candidates = flat_candidates_pre_hessian.copy()
                    hessian_dbg = {"enabled": bool(self.low_gradient_plateau_use_hessian_filter)}
                    if self.low_gradient_plateau_use_hessian_filter and np.count_nonzero(flat_candidates):
                        hessian = _compute_hessian_response(
                            frame.z_mm,
                            frame.valid_mask,
                            smoothing_kernel=int(self.gradient_smoothing_kernel),
                        )
                        hvals = hessian[flat_candidates]
                        hp = float(np.clip(self.low_gradient_plateau_hessian_percentile, 1.0, 99.0))
                        hthr = float(np.percentile(hvals, hp)) if hvals.size else 0.0
                        flat_candidates = flat_candidates & (hessian <= hthr)
                        hessian_dbg.update(
                            {
                                "threshold_percentile": hp,
                                "threshold_value": hthr,
                                "input_count": int(np.count_nonzero(flat_candidates_pre_hessian)),
                                "post_filter_count": int(np.count_nonzero(flat_candidates)),
                            }
                        )
                    cv2.imwrite(str(output_dir / "flat_candidate_mask.png"), (flat_candidates.astype(np.uint8) * 255))
                    flat_z_values = frame.z_mm[flat_candidates].astype(np.float64) if np.count_nonzero(flat_candidates) else np.asarray([], dtype=np.float64)
                    plateau_result = _detect_depth_plateaus(
                        z_values=flat_z_values,
                        bins=max(16, int(self.low_gradient_plateau_hist_bins)),
                        min_fraction=float(self.low_gradient_plateau_min_fraction),
                        min_pixels=max(16, int(self.low_gradient_plateau_min_pixels)),
                        smoothing_sigma_bins=float(self.low_gradient_plateau_smoothing_sigma_bins),
                        peak_drop_ratio=float(self.low_gradient_plateau_peak_drop_ratio),
                        robust_band_mad_k=float(self.low_gradient_plateau_robust_band_mad_k),
                        detection_min_count_floor=int(self.low_gradient_plateau_detection_min_count_floor),
                        detection_min_count_fraction=float(self.low_gradient_plateau_detection_min_count_fraction),
                    )
                    plateaus = list(plateau_result.get("plateaus", []))
                    selected_plateau = _select_background_plateau(
                        plateaus,
                        min_area_fraction=float(self.low_gradient_plateau_select_min_area_fraction),
                        mode=str(self.low_gradient_plateau_selection_mode),
                        robust_band=plateau_result.get("robust_band"),
                    )
                    # Tier 1: classify whole blobs as belt_bg by how much of their own height range
                    # overlaps the detected belt band, instead of pixel-filtering everything to a single
                    # z-slice. Any blob (not just the largest) that clears the overlap threshold joins
                    # belt_bg_mask in full, so a coherent belt component isn't chopped down to a sliver.
                    selection_reason = "no_flat_candidates"
                    band_z_lo: float | None = None
                    band_z_hi: float | None = None
                    if np.count_nonzero(flat_candidates) == 0:
                        warnings.append("bg_and_stripes_no_flat_candidates")
                    elif selected_plateau is None:
                        robust_band = plateau_result.get("robust_band") or {}
                        if robust_band.get("z_min") is not None and robust_band.get("z_max") is not None:
                            band_z_lo = float(robust_band["z_min"])
                            band_z_hi = float(robust_band["z_max"])
                            selection_reason = "no_plateau_detected_fallback_to_robust_band"
                        else:
                            band_z_lo = float(np.min(flat_z_values))
                            band_z_hi = float(np.max(flat_z_values))
                            selection_reason = "no_plateau_detected_fallback_to_flat_candidates"
                        warnings.append("bg_and_stripes_no_plateau_detected")
                    else:
                        band_z_lo = float(selected_plateau.get("z_min", float(np.min(flat_z_values))))
                        band_z_hi = float(selected_plateau.get("z_max", float(np.max(flat_z_values))))
                        selection_reason = f"selected_via_{str(self.low_gradient_plateau_selection_mode).lower()}"

                    pixel_size_mm = float(max(frame.x_resolution_mm or 1.0, frame.y_resolution_mm or 1.0))
                    belt_bg_mask = np.zeros_like(valid_for_fit, dtype=bool)
                    bg_blob_ids: set[int] = set()
                    if band_z_lo is not None and blob_labels.size:
                        for row in blob_rows:
                            component_id = int(row.get("blob_id") or row.get("component_id") or 0)
                            component_mask = (blob_labels == component_id) & valid_for_fit
                            area_px = int(np.count_nonzero(component_mask))
                            if area_px <= 0:
                                continue
                            comp_z = frame.z_mm[component_mask].astype(np.float64)
                            band_overlap_fraction = float(np.count_nonzero((comp_z >= band_z_lo) & (comp_z <= band_z_hi)) / area_px)
                            if band_overlap_fraction >= float(self.stripe_component_min_overlap_fraction):
                                belt_bg_mask |= component_mask
                                bg_blob_ids.add(component_id)

                    candidate_mask = belt_bg_mask.copy()
                    belt_base_mask = belt_bg_mask.copy()
                    inferred_depth_convention = "low_gradient_bg_and_stripes"
                    # Baseline model: flat belt_bg mean height for now. A fitted (or curved) belt-surface
                    # model is a deliberate follow-up once stripe detection itself is validated against
                    # this simpler reference.
                    belt_bg_mean_z = (
                        float(np.mean(frame.z_mm[belt_bg_mask]))
                        if np.count_nonzero(belt_bg_mask)
                        else float(np.median(frame.z_mm[valid_for_fit])) if np.count_nonzero(valid_for_fit) else 0.0
                    )
                    plane_z_map = np.full(frame.z_mm.shape, belt_bg_mean_z, dtype=np.float32)
                    raw_altitude = frame.z_mm.astype(np.float32) - plane_z_map
                    raw_negative_altitude_pixel_count = int(np.count_nonzero(valid_for_fit & (raw_altitude < -1e-6)))
                    baseline_map = np.minimum(plane_z_map, frame.z_mm.astype(np.float32))
                    baseline_map[~valid_for_fit] = 0.0
                    baseline_above_source_pixel_count = int(
                        np.count_nonzero(valid_for_fit & ((baseline_map - frame.z_mm.astype(np.float32)) > 1e-6))
                    )
                    altitude_map = np.maximum(frame.z_mm.astype(np.float32) - baseline_map, 0.0)
                    altitude_map[~valid_for_fit] = 0.0
                    negative_altitude_pixel_count = int(np.count_nonzero(valid_for_fit & (altitude_map < -1e-6)))

                    # Tier 2: from the remaining (non-bg) blobs, classify belt_stripe by how much of the
                    # blob's height-over-base falls within a target band (stripe_height_mm +/- tolerance),
                    # instead of a fixed [min, max] gate. Again a whole-blob decision, many blobs can qualify.
                    stripe_band_lo = max(0.0, float(self.stripe_height_mm) - float(self.stripe_height_tolerance_mm))
                    stripe_band_hi = float(self.stripe_height_mm) + float(self.stripe_height_tolerance_mm)
                    stripe_evidence_mask = valid_for_fit & (altitude_map >= stripe_band_lo) & (altitude_map <= stripe_band_hi) & (~belt_bg_mask)
                    stripes_mask = np.zeros_like(valid_for_fit, dtype=bool)
                    unknown_low_gradient_mask = np.zeros_like(valid_for_fit, dtype=bool)
                    belt_bg_component_rows = []
                    belt_stripe_component_rows = []
                    for row in blob_rows:
                        component_id = int(row.get("blob_id") or row.get("component_id") or 0)
                        component_mask = (blob_labels == component_id) & valid_for_fit
                        if not np.count_nonzero(component_mask):
                            continue
                        area_px = int(np.count_nonzero(component_mask))
                        overlap_bg_fraction = float(np.count_nonzero(component_mask & belt_bg_mask) / max(1, area_px))
                        metrics = _component_oriented_metrics(component_mask, pixel_size_mm=pixel_size_mm)
                        comp_alt = altitude_map[component_mask].astype(np.float64)
                        h_median = float(np.median(comp_alt)) if comp_alt.size else 0.0
                        h_mean = float(np.mean(comp_alt)) if comp_alt.size else 0.0
                        h_p95 = float(np.percentile(comp_alt, 95)) if comp_alt.size else 0.0
                        width_mm = float(metrics["width_mm"])
                        length_mm = float(metrics["length_mm"])
                        width_px = float(metrics["width_px"])
                        length_px = float(metrics["length_px"])
                        width_cv = float(metrics["width_cv"])
                        length_width_ratio = float(length_mm / max(width_mm, 1e-6))
                        height_width_ratio = float(h_p95 / max(width_mm, 1e-6))
                        stripe_band_overlap_fraction = float(np.count_nonzero((comp_alt >= stripe_band_lo) & (comp_alt <= stripe_band_hi)) / max(1, area_px))
                        decision = dict(row)
                        decision.update(
                            {
                                "component_id": component_id,
                                "width_px": width_px,
                                "length_px": length_px,
                                "width_mm": width_mm,
                                "length_mm": length_mm,
                                "orientation_deg": float(metrics["orientation_deg"]),
                                "width_cv": width_cv,
                                "height_over_base_median_mm": h_median,
                                "height_over_base_mean_mm": h_mean,
                                "height_over_base_p95_mm": h_p95,
                                "length_width_ratio": length_width_ratio,
                                "height_width_ratio": height_width_ratio,
                                "stripe_band_overlap_fraction": stripe_band_overlap_fraction,
                                "overlap_with_belt_bg": overlap_bg_fraction,
                                "plateau_membership": component_id in bg_blob_ids,
                            }
                        )
                        if component_id in bg_blob_ids:
                            decision["role"] = "belt_bg"
                            decision["reason"] = (
                                f"height range overlaps belt band [{band_z_lo:.1f}, {band_z_hi:.1f}]mm by >= {float(self.stripe_component_min_overlap_fraction):.0%}"
                                if band_z_lo is not None
                                else "selected belt-background plateau support"
                            )
                            belt_bg_component_rows.append(decision)
                        else:
                            passes_rules = (
                                (not self.stripe_rule_overlap_enabled or stripe_band_overlap_fraction >= float(self.stripe_component_min_overlap_fraction))
                                and (not self.stripe_rule_height_range_enabled or stripe_band_lo <= h_p95 <= stripe_band_hi)
                                and (not self.stripe_rule_width_range_enabled or float(self.stripe_min_width_mm) <= width_mm <= float(self.stripe_max_width_mm))
                                and (not self.stripe_rule_height_width_ratio_enabled or float(self.stripe_min_height_width_ratio) <= height_width_ratio <= float(self.stripe_max_height_width_ratio))
                                and (not self.stripe_rule_width_cv_enabled or width_cv <= float(self.stripe_max_width_cv))
                                and (not self.stripe_rule_length_width_ratio_enabled or length_width_ratio >= float(self.stripe_min_length_width_ratio))
                                and (not self.stripe_rule_area_fraction_enabled or float(row.get("area_ratio") or 0.0) <= float(self.stripe_max_area_fraction))
                            )
                            if passes_rules:
                                decision["role"] = "belt_stripe"
                                decision["reason"] = f"height-over-base overlaps stripe band [{stripe_band_lo:.1f}, {stripe_band_hi:.1f}]mm by >= {float(self.stripe_component_min_overlap_fraction):.0%}"
                                stripes_mask |= component_mask
                                belt_stripe_component_rows.append(decision)
                            else:
                                decision["role"] = "unknown"
                                failed_rules: list[str] = []
                                if self.stripe_rule_overlap_enabled and stripe_band_overlap_fraction < float(self.stripe_component_min_overlap_fraction):
                                    failed_rules.append("insufficient_stripe_band_overlap")
                                if self.stripe_rule_height_range_enabled and not (stripe_band_lo <= h_p95 <= stripe_band_hi):
                                    failed_rules.append("height_out_of_range")
                                if self.stripe_rule_width_range_enabled and not (float(self.stripe_min_width_mm) <= width_mm <= float(self.stripe_max_width_mm)):
                                    failed_rules.append("width_out_of_range")
                                if self.stripe_rule_height_width_ratio_enabled and not (float(self.stripe_min_height_width_ratio) <= height_width_ratio <= float(self.stripe_max_height_width_ratio)):
                                    failed_rules.append("height_width_ratio_out_of_range")
                                if self.stripe_rule_width_cv_enabled and width_cv > float(self.stripe_max_width_cv):
                                    failed_rules.append("width_variation_too_high")
                                if self.stripe_rule_length_width_ratio_enabled and length_width_ratio < float(self.stripe_min_length_width_ratio):
                                    failed_rules.append("not_elongated_enough")
                                if self.stripe_rule_area_fraction_enabled and float(row.get("area_ratio") or 0.0) > float(self.stripe_max_area_fraction):
                                    failed_rules.append("area_fraction_too_large")
                                decision["reason"] = ",".join(failed_rules) if failed_rules else "kept as unknown_low_gradient"
                                unknown_low_gradient_mask |= component_mask
                        low_gradient_component_rows.append(decision)
                    belt_bg_mask = belt_bg_mask & valid_for_fit
                    stripes_mask = stripes_mask & valid_for_fit & (~belt_bg_mask)
                    unknown_low_gradient_mask = (low_grad_mask & valid_for_fit & (~belt_bg_mask) & (~stripes_mask)) | unknown_low_gradient_mask
                    unknown_low_gradient_mask = unknown_low_gradient_mask & valid_for_fit & (~belt_bg_mask) & (~stripes_mask)
                    surface_suppression_mask = belt_bg_mask | stripes_mask
                    object_search_domain_mask = valid_for_fit & (~surface_suppression_mask)
                    unknown_low_gradient_suppressed_pixel_count = int(
                        np.count_nonzero(unknown_low_gradient_mask & surface_suppression_mask)
                    )
                    raised_candidate_rejected_mask = stripes_mask.copy()
                    stripe_info = {
                        "background_detection_strategy": "low_gradient_bg_and_stripes",
                        "strategy": "low_gradient_bg_and_stripes",
                        "enabled": True,
                        "applied": bool(np.count_nonzero(stripes_mask) > 0),
                        "belt_bg_count_px": int(np.count_nonzero(belt_bg_mask)),
                        "belt_stripes_count_px": int(np.count_nonzero(stripes_mask)),
                        "unknown_low_gradient_count_px": int(np.count_nonzero(unknown_low_gradient_mask)),
                        "surface_roles": {
                            "belt_bg_pixels": int(np.count_nonzero(belt_bg_mask)),
                            "belt_stripe_pixels": int(np.count_nonzero(stripes_mask)),
                            "unknown_low_gradient_pixels": int(np.count_nonzero(unknown_low_gradient_mask)),
                            "bg_stripe_overlap_pixels": int(np.count_nonzero(belt_bg_mask & stripes_mask)),
                            "suppression_pixels": int(np.count_nonzero(surface_suppression_mask)),
                        },
                        "invariants": {
                            "background_support_equals_belt_bg": bool(np.array_equal(candidate_mask, belt_bg_mask)),
                            "surface_suppression_equals_bg_or_stripes": bool(
                                np.array_equal(surface_suppression_mask, belt_bg_mask | stripes_mask)
                            ),
                            "stripes_excluded_from_background_fit": int(np.count_nonzero(belt_bg_mask & stripes_mask)) == 0,
                            "baseline_above_source_pixel_count": baseline_above_source_pixel_count,
                            "negative_altitude_pixel_count": negative_altitude_pixel_count,
                            "raw_negative_altitude_pixel_count": raw_negative_altitude_pixel_count,
                            "unknown_low_gradient_suppressed_pixel_count": unknown_low_gradient_suppressed_pixel_count,
                        },
                        "stripe_rules": {
                            "stripe_height_mm": self.stripe_height_mm,
                            "stripe_height_tolerance_mm": self.stripe_height_tolerance_mm,
                            "stripe_band_low_mm": stripe_band_lo,
                            "stripe_band_high_mm": stripe_band_hi,
                            "bg_band_low_mm": band_z_lo,
                            "bg_band_high_mm": band_z_hi,
                            "min_width_mm": self.stripe_min_width_mm,
                            "max_width_mm": self.stripe_max_width_mm,
                            "min_height_width_ratio": self.stripe_min_height_width_ratio,
                            "max_height_width_ratio": self.stripe_max_height_width_ratio,
                            "max_width_cv": self.stripe_max_width_cv,
                            "min_length_width_ratio": self.stripe_min_length_width_ratio,
                            "max_area_fraction": self.stripe_max_area_fraction,
                            "min_overlap_fraction": self.stripe_component_min_overlap_fraction,
                        },
                        "stripe_rule_enabled": {
                            "overlap": self.stripe_rule_overlap_enabled,
                            "height_range": self.stripe_rule_height_range_enabled,
                            "width_range": self.stripe_rule_width_range_enabled,
                            "height_width_ratio": self.stripe_rule_height_width_ratio_enabled,
                            "width_cv": self.stripe_rule_width_cv_enabled,
                            "length_width_ratio": self.stripe_rule_length_width_ratio_enabled,
                            "area_fraction": self.stripe_rule_area_fraction_enabled,
                        },
                        "component_decisions": low_gradient_component_rows,
                        "selection_reason": selection_reason,
                        "selected_plateau": selected_plateau,
                        "baseline_model": "belt_bg_mean_z",
                        "belt_bg_mean_z_mm": belt_bg_mean_z,
                    }
                    altitude_hist_counts, altitude_hist_edges = np.histogram(
                        altitude_map[valid_for_fit].astype(np.float64) if np.count_nonzero(valid_for_fit) else np.asarray([0.0], dtype=np.float64),
                        bins=max(8, int(self.belt_stripe_filter_altitude_hist_bins)),
                    )
                    altitude_hist_payload = {
                        "sample_count": int(np.count_nonzero(valid_for_fit)),
                        "bin_counts": [int(x) for x in altitude_hist_counts.tolist()],
                        "bin_edges": [float(x) for x in altitude_hist_edges.tolist()],
                        "threshold_value_mm": float(stripe_band_lo),
                        "min_altitude_mm": float(stripe_band_lo),
                        "direction_used": "raised_from_belt_bg_model",
                    }
                    cv2.imwrite(str(output_dir / "belt_bg_mask.png"), (belt_bg_mask.astype(np.uint8) * 255))
                    cv2.imwrite(str(output_dir / "belt_base_mask.png"), (belt_bg_mask.astype(np.uint8) * 255))
                    cv2.imwrite(str(output_dir / "belt_stripes_mask.png"), (stripes_mask.astype(np.uint8) * 255))
                    cv2.imwrite(str(output_dir / "unknown_low_gradient_mask.png"), (unknown_low_gradient_mask.astype(np.uint8) * 255))
                    cv2.imwrite(str(output_dir / "surface_suppression_mask.png"), (surface_suppression_mask.astype(np.uint8) * 255))
                    cv2.imwrite(str(output_dir / "object_search_domain_mask.png"), (object_search_domain_mask.astype(np.uint8) * 255))
                    cv2.imwrite(str(output_dir / "belt_stripes_tophat_mask.png"), (stripe_evidence_mask.astype(np.uint8) * 255))
                    empty_diag = np.zeros_like(valid_for_fit, dtype=np.uint8)
                    cv2.imwrite(str(output_dir / "belt_stripes_shape_mask.png"), empty_diag)
                    cv2.imwrite(str(output_dir / "belt_above_belt_mask.png"), (stripe_evidence_mask.astype(np.uint8) * 255))
                    cv2.imwrite(str(output_dir / "belt_wide_object_mask.png"), empty_diag)
                    surface_overlay = _surface_role_overlay(
                        frame.z_mm,
                        valid_for_fit,
                        belt_bg_mask=belt_bg_mask,
                        belt_stripes_mask=stripes_mask,
                        unknown_low_gradient_mask=unknown_low_gradient_mask,
                    )
                    cv2.imwrite(str(output_dir / "low_gradient_components_overlay.png"), surface_overlay)
                    cv2.imwrite(str(output_dir / "belt_stripe_candidates_overlay.png"), surface_overlay)
                    write_heightmap_preview_png(baseline_map, valid_for_fit, output_dir / "belt_baseline_local_min.png", colorbar_label="belt-bg baseline (mm)")
                    write_heightmap_preview_png(altitude_map, valid_for_fit, output_dir / "belt_altitude_local_min.png", colorbar_label="height over belt-bg baseline (mm)")
                    _write_altitude_histogram_png(
                        altitude_hist_payload,
                        out_path=output_dir / "belt_altitude_histogram.png",
                        robust_y_percentile=float(self.plot_y_robust_percentile),
                    )
                    (output_dir / "belt_altitude_histogram.json").write_text(json.dumps(altitude_hist_payload, indent=2), encoding="utf-8")
                    (output_dir / "low_gradient_components.json").write_text(json.dumps(low_gradient_component_rows, indent=2), encoding="utf-8")
                    (output_dir / "belt_bg_components.json").write_text(json.dumps(belt_bg_component_rows, indent=2), encoding="utf-8")
                    (output_dir / "belt_stripe_components.json").write_text(json.dumps(belt_stripe_component_rows, indent=2), encoding="utf-8")
                    (output_dir / "reference_surface_candidates.json").write_text(json.dumps(low_gradient_component_rows, indent=2), encoding="utf-8")
                    (output_dir / "reference_surface_plateaus.json").write_text(json.dumps(plateaus, indent=2), encoding="utf-8")
                    (output_dir / "flat_candidate_histogram.json").write_text(json.dumps(plateau_result, indent=2), encoding="utf-8")
                    (output_dir / "belt_stripe_filter_debug.json").write_text(json.dumps(stripe_info, indent=2), encoding="utf-8")
                    gradient_debug = {
                        "strategy": "low_gradient_bg_and_stripes",
                        "selection_reason": selection_reason,
                        "threshold_mode": self.gradient_threshold_mode,
                        "threshold_value": float(gthr),
                        "valid_for_fit_count": int(np.count_nonzero(valid_for_fit)),
                        "low_grad_mask_count": int(np.count_nonzero(low_grad_mask)),
                        "flat_candidate_count": int(np.count_nonzero(flat_candidates)),
                        "selected_component_area_ratio": float(np.count_nonzero(belt_bg_mask) / max(1, np.count_nonzero(valid_for_fit))),
                        "selected_component_count_px": int(np.count_nonzero(belt_bg_mask)),
                        "rejected_raised_count_px": int(np.count_nonzero(stripes_mask)),
                        "plateau_count": int(len(plateaus)),
                        "selected_plateau": selected_plateau,
                        "robust_band": plateau_result.get("robust_band"),
                        "belt_stripe_filter": stripe_info,
                        "hessian_filter": hessian_dbg,
                        "height_gate": height_gate_debug,
                    }
                    (output_dir / "gradient_debug.json").write_text(json.dumps(gradient_debug, indent=2), encoding="utf-8")
                elif background_detection_strategy == "low_gradient_blob_height_clusters":
                    blob_component_mode = _normalize_blob_component_mode(self.blob_component_mode)
                    blob_result = _summarize_low_gradient_blobs(
                        low_grad_mask=low_grad_mask,
                        gradient=grad,
                        z=frame.z_mm,
                        valid_mask=valid_for_fit,
                        roi_mask=roi_mask,
                        height_gate_mask=height_gate_mask,
                        min_component_area=max(1, int(self.blob_cluster_min_component_area)),
                        component_mode=blob_component_mode,
                        blob_connectivity=max(4, int(self.blob_connectivity)),
                        blob_neighbor_z_tolerance_mm=float(self.blob_neighbor_z_tolerance_mm),
                        blob_component_z_tolerance_mm=float(self.blob_component_z_tolerance_mm),
                        blob_component_tolerance_mode=str(self.blob_component_tolerance_mode),
                        blob_component_mad_k=float(self.blob_component_mad_k),
                        blob_component_mad_floor_mm=float(self.blob_component_mad_floor_mm),
                        blob_component_allow_gradual_slope=bool(self.blob_component_allow_gradual_slope),
                        blob_component_max_local_slope_mm_per_px=float(self.blob_component_max_local_slope_mm_per_px),
                        blob_component_min_area_px=max(1, int(self.blob_component_min_area_px)),
                        blob_component_use_smoothed_z=bool(self.blob_component_use_smoothed_z),
                        blob_component_z_smoothing_kernel=max(1, int(self.blob_component_z_smoothing_kernel)),
                    )
                    blob_labels = np.asarray(blob_result.get("labels"), dtype=np.int32)
                    blob_rows = list(blob_result.get("blobs") or [])
                    blob_component_mode_used = str(blob_result.get("component_mode") or blob_component_mode)
                    height_aware_rejected_edge_mask = np.asarray(blob_result.get("rejected_edge_mask"), dtype=bool) if blob_result.get("rejected_edge_mask") is not None else np.zeros_like(valid_for_fit, dtype=bool)
                    height_aware_connectivity_debug = dict(blob_result.get("connectivity_debug") or {})
                    original_component_count = int(len(blob_rows))
                    fragment_result = _split_low_gradient_blob_candidates_by_height(
                        labels=blob_labels,
                        blob_rows=blob_rows,
                        gradient=grad,
                        z=frame.z_mm,
                        valid_mask=valid_for_fit,
                        roi_mask=roi_mask,
                        height_gate_mask=height_gate_mask,
                        min_component_area=max(1, int(self.blob_cluster_min_component_area)),
                        split_enabled=bool(self.blob_split_by_height_enabled),
                        split_method=str(self.blob_split_method),
                        split_min_height_range_mm=float(self.blob_split_min_height_range_mm),
                        split_min_pixels=max(1, int(self.blob_split_min_pixels)),
                        split_gap_mm=float(self.blob_split_gap_mm),
                        split_mode=str(self.blob_split_mode),
                        split_hist_bins=max(8, int(self.blob_split_hist_bins)),
                        split_min_band_fraction=float(self.blob_split_min_band_fraction),
                        split_min_band_pixels=max(1, int(self.blob_split_min_band_pixels)),
                        split_merge_gap_mm=float(self.blob_split_merge_gap_mm),
                        split_refine_morphology=bool(self.blob_split_refine_morphology),
                        split_open_kernel=max(1, int(self.blob_split_open_kernel)),
                        split_close_kernel=max(1, int(self.blob_split_close_kernel)),
                        split_height_border_enabled=bool(self.blob_split_height_border_enabled),
                        split_height_border_mode=str(self.blob_split_height_border_mode),
                        split_height_border_smoothing_kernel=max(1, int(self.blob_split_height_border_smoothing_kernel)),
                        split_height_border_threshold_mode=str(self.blob_split_height_border_threshold_mode),
                        split_height_border_percentile=float(self.blob_split_height_border_percentile),
                        split_height_border_min_delta_mm=float(self.blob_split_height_border_min_delta_mm),
                        split_height_border_dilate_kernel=max(1, int(self.blob_split_height_border_dilate_kernel)),
                        split_height_border_close_kernel=max(1, int(self.blob_split_height_border_close_kernel)),
                        split_height_border_min_length_px=max(1, int(self.blob_split_height_border_min_length_px)),
                        split_min_fragment_area_px=max(1, int(self.blob_split_min_fragment_area_px)),
                        split_merge_weak_boundaries=bool(self.blob_split_merge_weak_boundaries),
                        split_merge_max_median_z_gap_mm=float(self.blob_split_merge_max_median_z_gap_mm),
                        split_merge_max_boundary_strength_mm=float(self.blob_split_merge_max_boundary_strength_mm),
                        split_merge_background_like_only=bool(self.blob_split_merge_background_like_only),
                    )
                    fragment_labels = np.asarray(fragment_result.get("labels"), dtype=np.int32)
                    fragment_rows = list(fragment_result.get("fragments") or [])
                    split_debug_rows = list(fragment_result.get("debug") or [])
                    split_method_used = str(fragment_result.get("method") or "disabled")
                    border_strength = np.asarray(fragment_result.get("height_border_strength"), dtype=np.float32)
                    border_cut_mask = np.asarray(fragment_result.get("height_border_cut_mask"), dtype=bool)
                    border_fragment_labels = np.asarray(fragment_result.get("height_border_fragment_labels"), dtype=np.int32)
                    border_fragment_mask = np.asarray(fragment_result.get("height_border_fragment_mask"), dtype=bool)
                    merge_debug_rows = list(fragment_result.get("merge_debug") or [])
                    clusters = _cluster_low_gradient_blobs_by_height(
                        fragment_rows,
                        height_gap_mm=float(self.blob_cluster_height_gap_mm),
                        height_gap_mode=str(self.blob_cluster_height_gap_mode),
                        merge_close_clusters=bool(self.blob_cluster_merge_close_clusters),
                        merge_gap_mm=float(self.blob_cluster_merge_gap_mm),
                    )
                    scored_clusters = _score_blob_height_clusters(
                        clusters,
                        valid_pixel_count=int(np.count_nonzero(valid_for_fit)),
                        min_area_fraction=float(self.blob_cluster_min_area_fraction),
                        min_total_pixels=int(self.blob_cluster_min_total_pixels),
                        max_z_mad_mm=float(self.blob_cluster_max_z_mad_mm),
                        area_weight=float(self.blob_cluster_area_weight),
                        border_weight=float(self.blob_cluster_border_weight),
                        constancy_weight=float(self.blob_cluster_constancy_weight),
                        spatial_coverage_weight=float(self.blob_cluster_spatial_coverage_weight),
                        depth_preference_weight=float(self.blob_cluster_depth_preference_weight),
                        fragmentation_penalty=float(self.blob_cluster_fragmentation_penalty),
                        centrality_penalty=float(self.blob_cluster_centrality_penalty),
                        object_like_penalty=float(self.blob_cluster_object_like_penalty),
                        stripe_overlap_penalty=float(self.blob_cluster_stripe_overlap_penalty),
                    )
                    selected_cluster = next((row for row in scored_clusters if not bool(row.get("rejected"))), None)
                    fallback_used = False
                    fallback_reason = None
                    fallback_strategy = None
                    candidate_mask = np.zeros_like(valid_for_fit, dtype=bool)
                    selected_pre_refine_mask = np.zeros_like(valid_for_fit, dtype=bool)
                    selected_refined_mask = np.zeros_like(valid_for_fit, dtype=bool)
                    rejected_blob_clusters_mask = np.zeros_like(valid_for_fit, dtype=bool)

                    if selected_cluster is not None:
                        selected_blob_ids = {int(blob_id) for blob_id in (selected_cluster.get("blob_ids") or [])}
                        selected_pre_refine_mask = np.isin(fragment_labels, list(selected_blob_ids)) & valid_for_fit & height_gate_mask
                        candidate_mask = selected_pre_refine_mask.copy()
                        if bool(self.blob_cluster_refine_by_mad) and np.count_nonzero(candidate_mask):
                            selected_rows = [row for row in fragment_rows if int(row.get("blob_id") or 0) in selected_blob_ids]
                            medians = np.asarray([float(row.get("median_z") or 0.0) for row in selected_rows], dtype=np.float64)
                            weights = np.asarray([max(1, int(row.get("area_px") or 0)) for row in selected_rows], dtype=np.float64)
                            z_med = float(np.average(medians, weights=weights)) if medians.size else float(np.median(frame.z_mm[candidate_mask]))
                            z_mad = float(np.average(np.asarray([float(row.get("mad_z") or 0.0) for row in selected_rows]), weights=weights)) if selected_rows else 0.0
                            z_band = max(float(self.blob_cluster_refine_floor_mm), float(self.blob_cluster_refine_mad_k) * max(0.25, z_mad))
                            selected_refined_mask = candidate_mask & (frame.z_mm >= (z_med - z_band)) & (frame.z_mm <= (z_med + z_band))
                            if bool(self.blob_cluster_refine_keep_border_support):
                                border_mask = _roi_border_mask(roi_mask)
                                selected_refined_mask = selected_refined_mask | (candidate_mask & border_mask)
                            candidate_mask = selected_refined_mask.copy()
                        else:
                            selected_refined_mask = candidate_mask.copy()
                        rejected_blob_ids = {
                            int(blob_id)
                            for row in scored_clusters
                            if str(row.get("cluster_id") or "") != str(selected_cluster.get("cluster_id") or "")
                            for blob_id in (row.get("blob_ids") or [])
                        }
                        if rejected_blob_ids:
                            rejected_blob_clusters_mask = np.isin(fragment_labels, list(rejected_blob_ids)) & valid_for_fit
                    else:
                        fallback_strategy = str(self.blob_cluster_fallback_strategy or "low_gradient_depth_plateaus").strip().lower()
                        fallback_used = True
                        fallback_reason = "no_cluster_passed_filters"
                        warnings.append("blob_cluster_strategy_fallback_used")
                        if fallback_strategy == "fail":
                            candidate_mask = np.zeros_like(valid_for_fit, dtype=bool)
                        elif fallback_strategy == "constant_z":
                            candidate_mask = np.zeros_like(valid_for_fit, dtype=bool)
                        elif fallback_strategy == "low_gradient_surface":
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
                            if selected_label > 0:
                                candidate_mask = (lg_result["labels"] == selected_label) & valid_for_fit & height_gate_mask
                        else:
                            flat_candidates = low_grad_mask & height_gate_mask
                            if np.count_nonzero(flat_candidates):
                                flat_z_values = frame.z_mm[flat_candidates].astype(np.float64)
                                plateau_result = _detect_depth_plateaus(
                                    z_values=flat_z_values,
                                    bins=max(16, int(self.low_gradient_plateau_hist_bins)),
                                    min_fraction=float(self.low_gradient_plateau_min_fraction),
                                    min_pixels=max(16, int(self.low_gradient_plateau_min_pixels)),
                                    smoothing_sigma_bins=float(self.low_gradient_plateau_smoothing_sigma_bins),
                                    peak_drop_ratio=float(self.low_gradient_plateau_peak_drop_ratio),
                                    robust_band_mad_k=float(self.low_gradient_plateau_robust_band_mad_k),
                                    detection_min_count_floor=int(self.low_gradient_plateau_detection_min_count_floor),
                                    detection_min_count_fraction=float(self.low_gradient_plateau_detection_min_count_fraction),
                                )
                                selected_plateau = _select_background_plateau(
                                    plateau_result.get("plateaus", []),
                                    min_area_fraction=float(self.low_gradient_plateau_select_min_area_fraction),
                                    mode=str(self.low_gradient_plateau_selection_mode),
                                    robust_band=plateau_result.get("robust_band"),
                                )
                                if selected_plateau is None:
                                    candidate_mask = flat_candidates.copy()
                                else:
                                    z_lo = float(selected_plateau.get("z_min", float(np.min(flat_z_values))))
                                    z_hi = float(selected_plateau.get("z_max", float(np.max(flat_z_values))))
                                    candidate_mask = flat_candidates & (frame.z_mm >= z_lo) & (frame.z_mm <= z_hi)

                    belt_base_mask = candidate_mask.copy()
                    stripes_mask = np.zeros_like(valid_for_fit, dtype=bool)
                    stripe_info: dict[str, Any] = {"enabled": bool(self.belt_stripe_filter_enabled), "applied": False, "skip_reason": "disabled"}
                    altitude_hist_payload: dict[str, Any] = {"sample_count": 0, "bin_counts": [], "bin_edges": [], "threshold_value_mm": 0.0, "direction_used": str(self.belt_stripe_filter_direction)}
                    baseline_map = np.zeros_like(frame.z_mm, dtype=np.float32)
                    altitude_map = np.zeros_like(frame.z_mm, dtype=np.float32)
                    filter_scope = str(self.belt_stripe_filter_scope or "global").strip().lower()
                    if filter_scope not in {"global", "bg_plateau"}:
                        filter_scope = "global"
                    if self.belt_stripe_filter_enabled and np.count_nonzero(candidate_mask) > 0:
                        pixel_size_mm = float(max(frame.x_resolution_mm or 1.0, frame.y_resolution_mm or 1.0))
                        domain_mask = np.asarray(valid_for_fit, dtype=bool) if filter_scope == "global" else np.asarray(candidate_mask, dtype=bool)
                        stripe_result = _compute_belt_stripe_filter(
                            z_mm=frame.z_mm,
                            domain_mask=domain_mask,
                            pixel_size_mm=pixel_size_mm,
                            window_mm=float(self.belt_stripe_filter_window_mm),
                            direction=str(self.belt_stripe_filter_direction),
                            threshold_mode=str(self.belt_stripe_filter_threshold_mode),
                            min_altitude_mm=float(self.belt_stripe_filter_min_altitude_mm),
                            k_mad=float(self.belt_stripe_filter_k_mad),
                            fixed_threshold_mm=float(self.belt_stripe_filter_fixed_threshold_mm),
                            min_stripe_fraction=float(self.belt_stripe_filter_min_stripe_fraction),
                            bg_plateau_mask=np.asarray(candidate_mask, dtype=bool),
                            z_floor_enabled=bool(self.belt_stripe_filter_z_floor_enabled),
                            z_floor_use_upper_bound=bool(self.belt_stripe_filter_z_floor_use_upper_bound),
                            z_floor_margin_mm=float(self.belt_stripe_filter_z_floor_margin_mm),
                            max_stripe_height_mm=float(self.belt_stripe_filter_max_stripe_height_mm),
                            above_belt_close_mm=float(self.belt_stripe_filter_above_belt_close_mm),
                            object_kernel_mm=float(self.belt_stripe_filter_object_kernel_mm),
                            object_kernel_shape=str(self.belt_stripe_filter_object_kernel_shape),
                            altitude_hist_bins=int(self.belt_stripe_filter_altitude_hist_bins),
                            auto_bimodality_margin=float(self.belt_stripe_filter_auto_bimodality_margin),
                            z_floor_upper_percentile=float(self.belt_stripe_filter_z_floor_upper_percentile),
                            z_floor_fallback_lower_percentile=float(self.belt_stripe_filter_z_floor_fallback_lower_percentile),
                            z_floor_fallback_upper_percentile=float(self.belt_stripe_filter_z_floor_fallback_upper_percentile),
                        )
                        stripes_mask = stripe_result["stripes_mask"]
                        baseline_map = stripe_result["baseline_map"]
                        altitude_map = stripe_result["altitude_map"]
                        stripe_info = stripe_result["info"]
                        stripe_info["scope"] = filter_scope
                        altitude_hist_payload = stripe_result["altitude_histogram"]
                        if stripe_info.get("applied"):
                            bg_snapshot = candidate_mask.copy()
                            belt_base_mask = bg_snapshot & (~stripes_mask)
                            candidate_mask = belt_base_mask
                            raised_candidate_rejected_mask = raised_candidate_rejected_mask | stripes_mask
                        else:
                            belt_base_mask = candidate_mask.copy()
                        cv2.imwrite(str(output_dir / "belt_stripes_mask.png"), (stripes_mask.astype(np.uint8) * 255))
                        cv2.imwrite(str(output_dir / "belt_base_mask.png"), (belt_base_mask.astype(np.uint8) * 255))
                        if "tophat_stripes_mask" in stripe_result:
                            cv2.imwrite(str(output_dir / "belt_stripes_tophat_mask.png"), (stripe_result["tophat_stripes_mask"].astype(np.uint8) * 255))
                        if "shape_stripes_mask" in stripe_result:
                            cv2.imwrite(str(output_dir / "belt_stripes_shape_mask.png"), (stripe_result["shape_stripes_mask"].astype(np.uint8) * 255))
                        if "above_belt_mask" in stripe_result:
                            cv2.imwrite(str(output_dir / "belt_above_belt_mask.png"), (stripe_result["above_belt_mask"].astype(np.uint8) * 255))
                        if "wide_object_mask" in stripe_result:
                            cv2.imwrite(str(output_dir / "belt_wide_object_mask.png"), (stripe_result["wide_object_mask"].astype(np.uint8) * 255))
                        diag_domain = np.asarray(valid_for_fit, dtype=bool) if filter_scope == "global" else (candidate_mask | belt_base_mask | stripes_mask)
                        write_heightmap_preview_png(baseline_map, diag_domain, output_dir / "belt_baseline_local_min.png", colorbar_label="local-min z baseline (mm)")
                        write_heightmap_preview_png(altitude_map, diag_domain, output_dir / "belt_altitude_local_min.png", colorbar_label="altitude over local-min (mm)")
                        _write_altitude_histogram_png(altitude_hist_payload, out_path=output_dir / "belt_altitude_histogram.png", robust_y_percentile=float(self.plot_y_robust_percentile))
                        (output_dir / "belt_altitude_histogram.json").write_text(json.dumps(altitude_hist_payload, indent=2), encoding="utf-8")
                        (output_dir / "belt_stripe_filter_debug.json").write_text(json.dumps(stripe_info, indent=2), encoding="utf-8")
                    if not (output_dir / "belt_stripes_mask.png").is_file():
                        empty_mask = np.zeros_like(candidate_mask, dtype=np.uint8)
                        cv2.imwrite(str(output_dir / "belt_stripes_mask.png"), empty_mask)
                        cv2.imwrite(str(output_dir / "belt_base_mask.png"), (candidate_mask.astype(np.uint8) * 255))
                        for _name in (
                            "belt_stripes_tophat_mask.png",
                            "belt_stripes_shape_mask.png",
                            "belt_above_belt_mask.png",
                            "belt_wide_object_mask.png",
                        ):
                            cv2.imwrite(str(output_dir / _name), empty_mask)
                        _write_plateau_plot_not_applicable_png(
                            out_path=output_dir / "belt_baseline_local_min.png",
                            strategy="low_gradient_blob_height_clusters",
                        )
                        _write_plateau_plot_not_applicable_png(
                            out_path=output_dir / "belt_altitude_local_min.png",
                            strategy="low_gradient_blob_height_clusters",
                        )
                        _write_plateau_plot_not_applicable_png(
                            out_path=output_dir / "belt_altitude_histogram.png",
                            strategy="low_gradient_blob_height_clusters",
                        )
                        (output_dir / "belt_altitude_histogram.json").write_text(
                            json.dumps({"strategy": "low_gradient_blob_height_clusters", "applied": False, "sample_count": 0, "bin_counts": [], "bin_edges": [], "threshold_value_mm": 0.0}, indent=2),
                            encoding="utf-8",
                        )
                        (output_dir / "belt_stripe_filter_debug.json").write_text(
                            json.dumps({"strategy": "low_gradient_blob_height_clusters", "applied": False, "reason": "disabled_or_empty_candidate"}, indent=2),
                            encoding="utf-8",
                        )

                    inferred_depth_convention = "low_gradient_blob_height_clusters"
                    gradient_debug = {
                        "strategy": "low_gradient_blob_height_clusters",
                        "threshold_mode": self.gradient_threshold_mode,
                        "threshold_value": float(gthr),
                        "low_grad_mask_count": int(np.count_nonzero(low_grad_mask)),
                        "blob_count": int(len(blob_rows)),
                        "component_mode": blob_component_mode_used,
                        "blob_connectivity": int(self.blob_connectivity),
                        "blob_neighbor_z_tolerance_mm": float(self.blob_neighbor_z_tolerance_mm),
                        "accepted_neighbor_edges": int(height_aware_connectivity_debug.get("accepted_neighbor_edges") or 0),
                        "rejected_neighbor_edges": int(height_aware_connectivity_debug.get("rejected_neighbor_edges") or 0),
                        "height_aware_component_count": int(height_aware_connectivity_debug.get("component_count") or len(blob_rows)),
                        "height_consistent_fragment_count": int(len(fragment_rows)),
                        "kept_blob_count": int(len(blob_result.get("kept_ids") or [])),
                        "rejected_blob_count": int(len(blob_result.get("rejected_ids") or [])),
                        "split_method": split_method_used,
                        "eligible_blob_count": int(fragment_result.get("eligible_blob_count") or 0),
                        "height_border_split_blob_count": int(fragment_result.get("height_border_split_blob_count") or 0),
                        "histogram_split_blob_count": int(fragment_result.get("histogram_split_blob_count") or 0),
                        "merged_fragment_count": int(fragment_result.get("merge_back_count") or 0),
                        "kept_fragment_count": int(len(fragment_result.get("kept_ids") or [])),
                        "rejected_fragment_count": int(len(fragment_result.get("rejected_ids") or [])),
                        "height_splitting_enabled": bool(self.blob_split_by_height_enabled),
                        "height_splitting_applied": bool(fragment_result.get("using_split_fragments")),
                        "cluster_count": int(len(scored_clusters)),
                        "selected_cluster_id": selected_cluster.get("cluster_id") if isinstance(selected_cluster, dict) else None,
                        "selected_blob_ids": selected_cluster.get("blob_ids") if isinstance(selected_cluster, dict) else [],
                        "selected_source_blob_ids": selected_cluster.get("source_blob_ids") if isinstance(selected_cluster, dict) else [],
                        "selected_component_area_ratio": float(np.count_nonzero(candidate_mask) / max(1, np.count_nonzero(valid_for_fit))),
                        "fallback_used": bool(fallback_used),
                        "fallback_reason": fallback_reason,
                        "fallback_strategy": fallback_strategy,
                        "belt_stripe_filter": stripe_info,
                        "height_gate": height_gate_debug,
                    }
                    cmap = _label_colormap_overlay(blob_labels)
                    cv2.imwrite(str(output_dir / "low_gradient_blob_components_overlay.png"), cmap)
                    cv2.imwrite(str(output_dir / "low_gradient_blob_id_mask.png"), (blob_labels % 255).astype(np.uint8))
                    if blob_component_mode_used == "height_aware":
                        cv2.imwrite(str(output_dir / "height_aware_blob_components_overlay.png"), cmap)
                        cv2.imwrite(str(output_dir / "height_aware_blob_id_mask.png"), (blob_labels % 255).astype(np.uint8))
                        rejected_edge_vis = _label_colormap_overlay(blob_labels)
                        rejected_edge_vis[height_aware_rejected_edge_mask & low_grad_mask] = (0, 0, 255)
                        cv2.imwrite(str(output_dir / "height_aware_connectivity_rejected_edges.png"), rejected_edge_vis)
                        (output_dir / "height_aware_connectivity_debug.json").write_text(
                            json.dumps(height_aware_connectivity_debug, indent=2),
                            encoding="utf-8",
                        )
                    write_heightmap_preview_png(border_strength, valid_for_fit, output_dir / "height_border_strength.png", colorbar_label="height-border strength (mm)")
                    cv2.imwrite(str(output_dir / "height_border_cut_mask.png"), (border_cut_mask.astype(np.uint8) * 255))
                    height_border_overlay = _label_colormap_overlay(border_fragment_labels)
                    cv2.imwrite(str(output_dir / "height_border_fragments_overlay.png"), height_border_overlay)
                    cv2.imwrite(str(output_dir / "height_border_fragments_mask.png"), (border_fragment_mask.astype(np.uint8) * 255))
                    cv2.imwrite(str(output_dir / "height_border_fragments_id_mask.png"), (border_fragment_labels % 255).astype(np.uint8))
                    fragment_overlay = _label_colormap_overlay(fragment_labels)
                    cv2.imwrite(str(output_dir / "height_split_blob_fragments_overlay.png"), fragment_overlay)
                    cv2.imwrite(str(output_dir / "height_split_blob_fragments_mask.png"), ((fragment_labels > 0).astype(np.uint8) * 255))
                    cv2.imwrite(str(output_dir / "height_split_blob_id_mask.png"), (fragment_labels % 255).astype(np.uint8))
                    cv2.imwrite(str(output_dir / "selected_blob_cluster_mask.png"), (candidate_mask.astype(np.uint8) * 255))
                    cv2.imwrite(str(output_dir / "selected_blob_cluster_pre_refine_mask.png"), (selected_pre_refine_mask.astype(np.uint8) * 255))
                    cv2.imwrite(str(output_dir / "selected_blob_cluster_refined_mask.png"), (selected_refined_mask.astype(np.uint8) * 255))
                    cv2.imwrite(str(output_dir / "rejected_blob_clusters_mask.png"), (rejected_blob_clusters_mask.astype(np.uint8) * 255))
                    selected_overlay = _height_color_image(frame.z_mm, valid_for_fit)
                    selected_overlay[candidate_mask] = (0, 255, 0)
                    rejected_overlay = _height_color_image(frame.z_mm, valid_for_fit)
                    rejected_overlay[rejected_blob_clusters_mask] = (0, 0, 255)
                    cv2.imwrite(str(output_dir / "selected_blob_cluster_overlay.png"), selected_overlay)
                    cv2.imwrite(str(output_dir / "rejected_blob_clusters_overlay.png"), rejected_overlay)
                    (output_dir / "low_gradient_blob_summary.json").write_text(json.dumps(blob_rows, indent=2), encoding="utf-8")
                    (output_dir / "height_consistent_blob_summary.json").write_text(json.dumps(fragment_rows, indent=2), encoding="utf-8")
                    (output_dir / "height_split_debug.json").write_text(
                        json.dumps(
                            {
                                "strategy": "low_gradient_blob_height_clusters",
                                "enabled": bool(self.blob_split_by_height_enabled),
                                "method": split_method_used,
                                "mode": str(self.blob_split_mode),
                                "original_blob_count": len(blob_rows),
                                "eligible_blob_count": int(fragment_result.get("eligible_blob_count") or 0),
                                "height_border_split_blob_count": int(fragment_result.get("height_border_split_blob_count") or 0),
                                "histogram_split_blob_count": int(fragment_result.get("histogram_split_blob_count") or 0),
                                "final_fragment_count": len(fragment_rows),
                                "merge_back_count": int(fragment_result.get("merge_back_count") or 0),
                                "using_split_fragments": bool(fragment_result.get("using_split_fragments")),
                                "blob_count": len(blob_rows),
                                "fragment_count": len(fragment_rows),
                                "warnings": list(fragment_result.get("warnings") or []),
                                "blobs": split_debug_rows,
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    (output_dir / "height_border_split_debug.json").write_text(
                        json.dumps(
                            {
                                "strategy": "low_gradient_blob_height_clusters",
                                "enabled": bool(self.blob_split_height_border_enabled),
                                "method": split_method_used,
                                "original_blob_count": len(blob_rows),
                                "eligible_blob_count": int(fragment_result.get("eligible_blob_count") or 0),
                                "height_border_split_blob_count": int(fragment_result.get("height_border_split_blob_count") or 0),
                                "merge_back_count": int(fragment_result.get("merge_back_count") or 0),
                                "blobs": split_debug_rows,
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    (output_dir / "fragment_merge_debug.json").write_text(
                        json.dumps(
                            {
                                "strategy": "low_gradient_blob_height_clusters",
                                "method": split_method_used,
                                "merge_back_count": int(fragment_result.get("merge_back_count") or 0),
                                "merges": merge_debug_rows,
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    selected_cluster_id_for_rows = (
                        selected_cluster.get("cluster_id") if isinstance(selected_cluster, dict) else None
                    )

                    def _cluster_decision(row: dict[str, Any]) -> str:
                        if selected_cluster_id_for_rows is not None and row.get("cluster_id") == selected_cluster_id_for_rows:
                            return "selected"
                        return "rejected" if bool(row.get("rejected")) else "candidate"

                    (output_dir / "blob_height_clusters.json").write_text(
                        json.dumps(
                            [
                                {
                                    **row,
                                    "selected": bool(
                                        selected_cluster_id_for_rows is not None
                                        and row.get("cluster_id") == selected_cluster_id_for_rows
                                    ),
                                    "decision": _cluster_decision(row),
                                    "component_ids": list(row.get("source_blob_ids") or []),
                                    "fragment_ids": list(row.get("blob_ids") or []),
                                    "component_mode": blob_component_mode_used,
                                    "original_component_count": original_component_count,
                                    "height_aware_component_count": int(height_aware_connectivity_debug.get("component_count") or original_component_count),
                                    "split_method": split_method_used,
                                }
                                for row in scored_clusters
                            ],
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    with (output_dir / "blob_cluster_score_table.csv").open("w", encoding="utf-8", newline="") as handle:
                        writer = csv.DictWriter(
                            handle,
                            fieldnames=[
                                "selected", "cluster_id", "score", "decision", "rejected_reason",
                                "blob_count", "fragment_count", "original_blob_count", "split_fragment_count",
                                "mixed_blob_source_count", "component_mode", "height_aware_component_count", "split_method",
                                "total_pixels", "area_fraction", "median_z",
                                "z_mad", "z_mad_weighted", "touches_roi_border_fraction", "touches_frame_border_fraction",
                                "border_contact", "spatial_coverage",
                                "mean_centrality", "mean_gradient_p95", "mean_solidity", "mean_compactness",
                                "centrality_penalty", "object_like_penalty", "fragmentation_penalty",
                                "height_gate_overlap", "stripe_overlap", "rejected",
                            ],
                        )
                        writer.writeheader()
                        for row in scored_clusters:
                            writer.writerow(
                                {
                                    **{key: row.get(key) for key in writer.fieldnames},
                                    "selected": bool(
                                        selected_cluster_id_for_rows is not None
                                        and row.get("cluster_id") == selected_cluster_id_for_rows
                                    ),
                                    "decision": _cluster_decision(row),
                                    "component_mode": blob_component_mode_used,
                                    "height_aware_component_count": int(height_aware_connectivity_debug.get("component_count") or original_component_count),
                                    "split_method": split_method_used,
                                }
                            )
                    (output_dir / "blob_cluster_selection_debug.json").write_text(
                        json.dumps(
                            {
                                "strategy": "low_gradient_blob_height_clusters",
                                "selected_cluster_id": selected_cluster.get("cluster_id") if isinstance(selected_cluster, dict) else None,
                                "selected_blob_ids": selected_cluster.get("blob_ids") if isinstance(selected_cluster, dict) else [],
                                "selected_source_blob_ids": selected_cluster.get("source_blob_ids") if isinstance(selected_cluster, dict) else [],
                                "blob_count": len(blob_rows),
                                "fragment_count": len(fragment_rows),
                                "component_mode": blob_component_mode_used,
                                "original_component_count": original_component_count,
                                "height_aware_component_count": int(height_aware_connectivity_debug.get("component_count") or original_component_count),
                                "rejected_neighbor_edges": int(height_aware_connectivity_debug.get("rejected_neighbor_edges") or 0),
                                "split_method": split_method_used,
                                "eligible_blob_count": int(fragment_result.get("eligible_blob_count") or 0),
                                "height_border_split_blob_count": int(fragment_result.get("height_border_split_blob_count") or 0),
                                "histogram_split_blob_count": int(fragment_result.get("histogram_split_blob_count") or 0),
                                "merge_back_count": int(fragment_result.get("merge_back_count") or 0),
                                "cluster_count": len(scored_clusters),
                                "fallback_used": bool(fallback_used),
                                "fallback_reason": fallback_reason,
                                "fallback_strategy": fallback_strategy,
                                "height_splitting_enabled": bool(self.blob_split_by_height_enabled),
                                "height_splitting_applied": bool(fragment_result.get("using_split_fragments")),
                                "top_rejected_clusters": [
                                    {
                                        "cluster_id": row.get("cluster_id"),
                                        "score": row.get("score"),
                                        "rejected_reason": row.get("rejected_reason"),
                                    }
                                    for row in scored_clusters
                                    if bool(row.get("rejected"))
                                ][:5],
                                # Per-cluster decision + component/fragment membership and score
                                # breakdown so the UI can explain "why this cluster" and bridge
                                # components -> clusters -> selected support without re-deriving it.
                                "clusters": [
                                    {
                                        "cluster_id": row.get("cluster_id"),
                                        "selected": bool(
                                            selected_cluster_id_for_rows is not None
                                            and row.get("cluster_id") == selected_cluster_id_for_rows
                                        ),
                                        "decision": _cluster_decision(row),
                                        "score": row.get("score"),
                                        "rejected_reason": row.get("rejected_reason"),
                                        "component_ids": list(row.get("source_blob_ids") or []),
                                        "fragment_ids": list(row.get("blob_ids") or []),
                                        "area_fraction": row.get("area_fraction"),
                                        "median_z": row.get("median_z"),
                                        "z_mad": row.get("z_mad"),
                                        "border_contact": row.get("border_contact"),
                                        "spatial_coverage": row.get("spatial_coverage"),
                                        "centrality_penalty": row.get("centrality_penalty"),
                                        "object_like_penalty": row.get("object_like_penalty"),
                                        "fragmentation_penalty": row.get("fragmentation_penalty"),
                                        "stripe_overlap": row.get("stripe_overlap"),
                                        "score_terms": row.get("score_terms"),
                                    }
                                    for row in scored_clusters
                                ],
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    (output_dir / "reference_surface_candidates.json").write_text(json.dumps(fragment_rows, indent=2), encoding="utf-8")
                    (output_dir / "gradient_debug.json").write_text(json.dumps(gradient_debug, indent=2), encoding="utf-8")
                else:
                    # ─── low_gradient_depth_plateaus strategy ────────────────
                    # Pipeline:
                    #   1. flat candidates  = low_grad_mask ∩ height_gate_mask
                    #   2. (optional) refine with low-Hessian (low curvature)
                    #   3. histogram z over flat candidates → detect plateaus
                    #   4. pick lowest plateau with enough area → belt support
                    #   5. all higher plateaus → rejected raised structures
                    #   6. plane fit consumes selected plateau pixels only
                    raised_candidate_rejected_mask = np.zeros_like(valid_for_fit, dtype=bool)
                    flat_candidates_pre_hessian = low_grad_mask & height_gate_mask
                    flat_candidates = flat_candidates_pre_hessian.copy()
                    hessian_dbg: dict[str, Any] = {"enabled": bool(self.low_gradient_plateau_use_hessian_filter)}
                    if self.low_gradient_plateau_use_hessian_filter and np.count_nonzero(flat_candidates):
                        hessian = _compute_hessian_response(
                            frame.z_mm,
                            frame.valid_mask,
                            smoothing_kernel=int(self.gradient_smoothing_kernel),
                        )
                        hvals = hessian[flat_candidates]
                        hp = float(np.clip(self.low_gradient_plateau_hessian_percentile, 1.0, 99.0))
                        hthr = float(np.percentile(hvals, hp)) if hvals.size else 0.0
                        flat_candidates = flat_candidates & (hessian <= hthr)
                        hessian_dbg.update(
                            {
                                "threshold_percentile": hp,
                                "threshold_value": hthr,
                                "input_count": int(np.count_nonzero(flat_candidates_pre_hessian)),
                                "post_filter_count": int(np.count_nonzero(flat_candidates)),
                            }
                        )
                        cv2.imwrite(
                            str(output_dir / "flat_candidate_mask.png"),
                            (flat_candidates.astype(np.uint8) * 255),
                        )
                    else:
                        cv2.imwrite(
                            str(output_dir / "flat_candidate_mask.png"),
                            (flat_candidates.astype(np.uint8) * 255),
                        )

                    flat_candidate_count = int(np.count_nonzero(flat_candidates))
                    flat_z_values = (
                        frame.z_mm[flat_candidates].astype(np.float64)
                        if flat_candidate_count
                        else np.array([], dtype=np.float64)
                    )
                    plateau_result = _detect_depth_plateaus(
                        z_values=flat_z_values,
                        bins=max(16, int(self.low_gradient_plateau_hist_bins)),
                        min_fraction=float(self.low_gradient_plateau_min_fraction),
                        min_pixels=max(16, int(self.low_gradient_plateau_min_pixels)),
                        smoothing_sigma_bins=float(self.low_gradient_plateau_smoothing_sigma_bins),
                        peak_drop_ratio=float(self.low_gradient_plateau_peak_drop_ratio),
                        robust_band_mad_k=float(self.low_gradient_plateau_robust_band_mad_k),
                        detection_min_count_floor=int(self.low_gradient_plateau_detection_min_count_floor),
                        detection_min_count_fraction=float(self.low_gradient_plateau_detection_min_count_fraction),
                    )
                    plateaus = plateau_result.get("plateaus", [])
                    selected_plateau = _select_background_plateau(
                        plateaus,
                        min_area_fraction=float(self.low_gradient_plateau_select_min_area_fraction),
                        mode=str(self.low_gradient_plateau_selection_mode),
                        robust_band=plateau_result.get("robust_band"),
                    )
                    selection_reason: str
                    plot_notes: str | None = None
                    if flat_candidate_count == 0:
                        candidate_mask = np.zeros_like(valid_for_fit, dtype=bool)
                        selection_reason = "no_flat_candidates"
                        warnings.append("plateau_strategy_no_flat_candidates")
                        plot_notes = "No flat candidates after gradient/Hessian filtering."
                    elif selected_plateau is None:
                        candidate_mask = flat_candidates.copy()
                        selection_reason = "no_plateau_detected_fallback_to_flat_candidates"
                        warnings.append("plateau_strategy_no_plateau_detected")
                        plot_notes = (
                            "No dominant plateau passed min_fraction/min_pixels — "
                            "falling back to all flat candidates."
                        )
                    else:
                        z_lo = float(selected_plateau.get("z_min", float(np.min(flat_z_values))))
                        z_hi = float(selected_plateau.get("z_max", float(np.max(flat_z_values))))
                        candidate_mask = flat_candidates & (frame.z_mm >= z_lo) & (frame.z_mm <= z_hi)
                        raised_candidate_rejected_mask = flat_candidates & (frame.z_mm > z_hi)
                        if str(selected_plateau.get("source") or "") == "robust_median_mad":
                            selection_reason = "fallback_robust_median_mad_band"
                            warnings.append("plateau_strategy_used_robust_band_fallback")
                        elif str(self.low_gradient_plateau_selection_mode).lower() == "robust_band":
                            selection_reason = "selected_robust_median_mad_band"
                        else:
                            selection_reason = f"selected_via_{str(self.low_gradient_plateau_selection_mode).lower()}"
                    inferred_depth_convention = "low_gradient_depth_plateaus"

                    rejected_plateaus: list[dict[str, Any]] = []
                    if selected_plateau is not None:
                        for row in plateaus:
                            if abs(float(row.get("z_center", 0.0)) - float(selected_plateau.get("z_center", 0.0))) < 1e-6:
                                continue
                            rejected_plateaus.append(row)

                    # ── Belt-stripe filter (local-min top-hat) ─────────────────────
                    # Runs globally over valid_for_fit (NOT just the BG plateau) so
                    # stripes that fall outside the BG plateau band (i.e. above
                    # belt-z, where they actually sit in 3-D) are still detected
                    # and excluded from expanded_plane_mask + downstream foreground.
                    # Baseline = erode(z); narrow raised ribs survive as positive
                    # altitude above the local belt minimum.
                    #
                    # Outputs consumed downstream:
                    #   - candidate_mask          ← candidate_mask AND NOT stripes
                    #   - raised_candidate_rejected_mask ← merged with stripes
                    #   - expanded_plane_mask    ← cleaned later, line ~1152
                    #   - belt_stripes_mask_array ← context (segmentation reads this)
                    belt_base_mask = candidate_mask.copy()
                    stripes_mask = np.zeros_like(valid_for_fit, dtype=bool)
                    stripe_info: dict[str, Any] = {"enabled": bool(self.belt_stripe_filter_enabled), "applied": False, "skip_reason": "disabled"}
                    altitude_hist_payload: dict[str, Any] = {"sample_count": 0, "bin_counts": [], "bin_edges": [], "threshold_value_mm": 0.0, "direction_used": str(self.belt_stripe_filter_direction)}
                    baseline_map = np.zeros_like(frame.z_mm, dtype=np.float32)
                    altitude_map = np.zeros_like(frame.z_mm, dtype=np.float32)
                    filter_scope = str(self.belt_stripe_filter_scope or "global").strip().lower()
                    if filter_scope not in {"global", "bg_plateau"}:
                        filter_scope = "global"
                    if (
                        self.belt_stripe_filter_enabled
                        and np.count_nonzero(candidate_mask) > 0
                    ):
                        pixel_size_mm = float(max(frame.x_resolution_mm or 1.0, frame.y_resolution_mm or 1.0))
                        # Domain: full valid frame (global) or just BG plateau.
                        domain_mask = (
                            np.asarray(valid_for_fit, dtype=bool)
                            if filter_scope == "global"
                            else np.asarray(candidate_mask, dtype=bool)
                        )
                        # Snapshot the BG plateau BEFORE we narrow candidate_mask
                        # below; the shape-based pass uses it to estimate belt-z
                        # robustly (median of plateau z).
                        bg_plateau_snapshot_for_filter = np.asarray(candidate_mask, dtype=bool)
                        stripe_result = _compute_belt_stripe_filter(
                            z_mm=frame.z_mm,
                            domain_mask=domain_mask,
                            pixel_size_mm=pixel_size_mm,
                            window_mm=float(self.belt_stripe_filter_window_mm),
                            direction=str(self.belt_stripe_filter_direction),
                            threshold_mode=str(self.belt_stripe_filter_threshold_mode),
                            min_altitude_mm=float(self.belt_stripe_filter_min_altitude_mm),
                            k_mad=float(self.belt_stripe_filter_k_mad),
                            fixed_threshold_mm=float(self.belt_stripe_filter_fixed_threshold_mm),
                            min_stripe_fraction=float(self.belt_stripe_filter_min_stripe_fraction),
                            bg_plateau_mask=bg_plateau_snapshot_for_filter,
                            z_floor_enabled=bool(self.belt_stripe_filter_z_floor_enabled),
                            z_floor_use_upper_bound=bool(self.belt_stripe_filter_z_floor_use_upper_bound),
                            z_floor_margin_mm=float(self.belt_stripe_filter_z_floor_margin_mm),
                            max_stripe_height_mm=float(self.belt_stripe_filter_max_stripe_height_mm),
                            above_belt_close_mm=float(self.belt_stripe_filter_above_belt_close_mm),
                            object_kernel_mm=float(self.belt_stripe_filter_object_kernel_mm),
                            object_kernel_shape=str(self.belt_stripe_filter_object_kernel_shape),
                            altitude_hist_bins=int(self.belt_stripe_filter_altitude_hist_bins),
                            auto_bimodality_margin=float(self.belt_stripe_filter_auto_bimodality_margin),
                            z_floor_upper_percentile=float(self.belt_stripe_filter_z_floor_upper_percentile),
                            z_floor_fallback_lower_percentile=float(self.belt_stripe_filter_z_floor_fallback_lower_percentile),
                            z_floor_fallback_upper_percentile=float(self.belt_stripe_filter_z_floor_fallback_upper_percentile),
                        )
                        stripes_mask = stripe_result["stripes_mask"]
                        baseline_map = stripe_result["baseline_map"]
                        altitude_map = stripe_result["altitude_map"]
                        stripe_info = stripe_result["info"]
                        stripe_info["scope"] = filter_scope
                        altitude_hist_payload = stripe_result["altitude_histogram"]
                        # belt_base_mask returned by the helper is "domain minus
                        # stripes". When we ran globally that's the whole valid
                        # belt — too broad to feed into plane-fit. We always
                        # intersect with candidate_mask (the BG plateau) so the
                        # plane fit sees belt-only pixels.
                        if stripe_info.get("applied"):
                            # Snapshot BG plateau before we narrow candidate_mask
                            bg_plateau_snapshot = candidate_mask.copy()
                            bg_plateau_pixels = int(np.count_nonzero(bg_plateau_snapshot))
                            stripes_in_bg = int(np.count_nonzero(stripes_mask & bg_plateau_snapshot))
                            belt_base_mask = bg_plateau_snapshot & (~stripes_mask)
                            candidate_mask = belt_base_mask
                            raised_candidate_rejected_mask = raised_candidate_rejected_mask | stripes_mask
                            stripe_info["belt_base_count_px"] = int(np.count_nonzero(belt_base_mask))
                            stripe_info["bg_plateau_count_px"] = bg_plateau_pixels
                            stripe_info["stripes_in_bg_count_px"] = stripes_in_bg
                            stripe_info["stripes_fraction_of_bg"] = (
                                float(stripes_in_bg / bg_plateau_pixels)
                                if bg_plateau_pixels > 0
                                else 0.0
                            )
                            if (
                                stripe_info["stripes_fraction_of_bg"]
                                > float(self.belt_stripe_filter_warn_removed_fraction)
                                and filter_scope == "bg_plateau"
                            ):
                                warnings.append("belt_stripe_filter_removed_large_bg_fraction")
                        else:
                            belt_base_mask = candidate_mask.copy()
                    elif not self.belt_stripe_filter_enabled:
                        stripe_info = {"enabled": False, "applied": False, "skip_reason": "disabled_in_params", "scope": filter_scope}

                    n_dominant = sum(
                        1
                        for row in plateaus
                        if float(row.get("fraction", 0.0))
                        >= float(self.low_gradient_plateau_select_min_area_fraction)
                    )
                    gradient_debug = {
                        "strategy": "low_gradient_depth_plateaus",
                        "selection_reason": selection_reason,
                        "threshold_mode": self.gradient_threshold_mode,
                        "threshold_value": float(gthr),
                        "valid_for_fit_count": int(np.count_nonzero(valid_for_fit)),
                        "low_grad_mask_count": int(np.count_nonzero(low_grad_mask)),
                        "flat_candidate_count_pre_hessian": int(np.count_nonzero(flat_candidates_pre_hessian)),
                        "flat_candidate_count": flat_candidate_count,
                        "selected_component_area_ratio": float(
                            np.count_nonzero(candidate_mask) / max(1, np.count_nonzero(valid_for_fit))
                        ),
                        "selected_component_count_px": int(np.count_nonzero(candidate_mask)),
                        "rejected_raised_count_px": int(np.count_nonzero(raised_candidate_rejected_mask)),
                        "plateau_count": int(len(plateaus)),
                        "dominant_plateau_count": int(n_dominant),
                        "selected_plateau": selected_plateau,
                        "rejected_plateaus": rejected_plateaus,
                        "robust_band": plateau_result.get("robust_band"),
                        "belt_stripe_filter": stripe_info,
                        "hessian_filter": hessian_dbg,
                        "plateau_parameters": {
                            "hist_bins": int(self.low_gradient_plateau_hist_bins),
                            "min_fraction": float(self.low_gradient_plateau_min_fraction),
                            "min_pixels": int(self.low_gradient_plateau_min_pixels),
                            "smoothing_sigma_bins": float(self.low_gradient_plateau_smoothing_sigma_bins),
                            "peak_drop_ratio": float(self.low_gradient_plateau_peak_drop_ratio),
                            "select_min_area_fraction": float(self.low_gradient_plateau_select_min_area_fraction),
                            "selection_mode": str(self.low_gradient_plateau_selection_mode),
                            "computed_min_count": int(plateau_result.get("min_count") or 0),
                        },
                        "height_gate": height_gate_debug,
                    }
                    (output_dir / "reference_surface_plateaus.json").write_text(
                        json.dumps(plateaus, indent=2), encoding="utf-8"
                    )
                    (output_dir / "reference_surface_candidates.json").write_text(
                        json.dumps(plateaus, indent=2), encoding="utf-8"
                    )
                    (output_dir / "flat_candidate_histogram.json").write_text(
                        json.dumps(plateau_result, indent=2), encoding="utf-8"
                    )
                    _write_plateau_plot_png(
                        plateau_result,
                        out_path=output_dir / "background_plateau_plot.png",
                        selected_plateau=selected_plateau,
                        sample_count=flat_candidate_count,
                        strategy="low_gradient_depth_plateaus",
                        notes=plot_notes,
                        robust_y_percentile=float(self.plot_y_robust_percentile),
                    )
                    # Sorted-z scanline of filtered points only — same style as
                    # the existing background_depth_plot but restricted to the
                    # low-gradient (± low-Hessian) candidates so the user can
                    # see directly which z plateaus survive the filtering.
                    _write_depth_plot_png(
                        flat_z_values,
                        near_q=float(selected_plateau.get("z_min")) if selected_plateau else float("nan"),
                        far_q=float(selected_plateau.get("z_max")) if selected_plateau else float("nan"),
                        out_path=output_dir / "flat_candidate_depth_plot.png",
                        title_override="Filtered depth distribution (low-gradient candidates)",
                        near_label="plateau_z_min",
                        far_label="plateau_z_max",
                        max_render_samples=int(self.plot_depth_plot_max_render_samples),
                    )
                    cv2.imwrite(
                        str(output_dir / "background_selected_plateau_mask.png"),
                        (candidate_mask.astype(np.uint8) * 255),
                    )
                    cv2.imwrite(
                        str(output_dir / "rejected_raised_plateau_mask.png"),
                        (raised_candidate_rejected_mask.astype(np.uint8) * 255),
                    )
                    # Belt-stripe filter diagnostics. Always emit (even when
                    # the filter was skipped) so the Studio view never points
                    # at a missing file.
                    cv2.imwrite(
                        str(output_dir / "belt_stripes_mask.png"),
                        (stripes_mask.astype(np.uint8) * 255),
                    )
                    cv2.imwrite(
                        str(output_dir / "belt_base_mask.png"),
                        (belt_base_mask.astype(np.uint8) * 255),
                    )
                    # Per-pass stripe diagnostics so the user can see whether
                    # a flagged pixel came from the top-hat or the shape pass.
                    if "tophat_stripes_mask" in stripe_result:
                        cv2.imwrite(
                            str(output_dir / "belt_stripes_tophat_mask.png"),
                            (stripe_result["tophat_stripes_mask"].astype(np.uint8) * 255),
                        )
                    if "shape_stripes_mask" in stripe_result:
                        cv2.imwrite(
                            str(output_dir / "belt_stripes_shape_mask.png"),
                            (stripe_result["shape_stripes_mask"].astype(np.uint8) * 255),
                        )
                    if "above_belt_mask" in stripe_result:
                        cv2.imwrite(
                            str(output_dir / "belt_above_belt_mask.png"),
                            (stripe_result["above_belt_mask"].astype(np.uint8) * 255),
                        )
                    if "wide_object_mask" in stripe_result:
                        cv2.imwrite(
                            str(output_dir / "belt_wide_object_mask.png"),
                            (stripe_result["wide_object_mask"].astype(np.uint8) * 255),
                        )
                    # Show the baseline/altitude across the full domain that was
                    # actually filtered (global scope ⇒ valid_for_fit), so the
                    # user can see whether stripes outside the BG plateau also
                    # got picked up.
                    diag_domain = (
                        valid_for_fit if filter_scope == "global"
                        else (candidate_mask | belt_base_mask | stripes_mask)
                    )
                    write_heightmap_preview_png(
                        baseline_map, diag_domain,
                        output_dir / "belt_baseline_local_min.png",
                        colorbar_label="local-min z baseline (mm)",
                    )
                    write_heightmap_preview_png(
                        altitude_map, diag_domain,
                        output_dir / "belt_altitude_local_min.png",
                        colorbar_label="altitude over local-min (mm)",
                    )
                    _write_altitude_histogram_png(
                        altitude_hist_payload,
                        out_path=output_dir / "belt_altitude_histogram.png",
                        robust_y_percentile=float(self.plot_y_robust_percentile),
                    )
                    (output_dir / "belt_altitude_histogram.json").write_text(
                        json.dumps(altitude_hist_payload, indent=2), encoding="utf-8"
                    )
                    (output_dir / "belt_stripe_filter_debug.json").write_text(
                        json.dumps(stripe_info, indent=2), encoding="utf-8"
                    )
                    (output_dir / "gradient_debug.json").write_text(
                        json.dumps(gradient_debug, indent=2), encoding="utf-8"
                    )
                cv2.imwrite(str(output_dir / "reference_surface_selected_mask.png"), (candidate_mask.astype(np.uint8) * 255))
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
            _write_depth_plot_png(
                z_values,
                near_q=near_q,
                far_q=far_q,
                out_path=output_dir / "background_depth_plot.png",
                max_render_samples=int(self.plot_depth_plot_max_render_samples),
            )
            # The plateau plot only makes semantic sense for the
            # `low_gradient_depth_plateaus` strategy. For every other strategy
            # write an explicit placeholder so Studio shows a clear "not
            # applicable" message instead of misleading content or a 404.
            plateau_plot_path = output_dir / "background_plateau_plot.png"
            filtered_depth_plot_path = output_dir / "flat_candidate_depth_plot.png"
            if background_detection_strategy not in {"low_gradient_depth_plateaus", "low_gradient_blob_height_clusters"}:
                _write_plateau_plot_not_applicable_png(
                    out_path=plateau_plot_path,
                    strategy=str(background_detection_strategy),
                )
                _write_plateau_plot_not_applicable_png(
                    out_path=filtered_depth_plot_path,
                    strategy=str(background_detection_strategy),
                )
                if not (output_dir / "reference_surface_plateaus.json").is_file():
                    (output_dir / "reference_surface_plateaus.json").write_text(
                        json.dumps(
                            {
                                "strategy": str(background_detection_strategy),
                                "plateau_analysis_active": False,
                                "reason": "plateau analysis only runs for strategy=low_gradient_depth_plateaus",
                                "plateaus": [],
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                if not (output_dir / "flat_candidate_histogram.json").is_file():
                    (output_dir / "flat_candidate_histogram.json").write_text(
                        json.dumps(
                            {
                                "strategy": str(background_detection_strategy),
                                "plateau_analysis_active": False,
                                "reason": "plateau analysis only runs for strategy=low_gradient_depth_plateaus",
                                "sample_count": 0,
                                "bin_edges": [],
                                "bin_counts": [],
                                "plateaus": [],
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                # Belt-stripe artifacts also require the plateau strategy to
                # have produced a candidate BG plateau. Write empty masks +
                # explicit "not applicable" diagnostics so the Studio views
                # stay stable across all strategies.
                empty_mask = np.zeros_like(candidate_mask, dtype=np.uint8) if candidate_mask is not None else np.zeros((1, 1), dtype=np.uint8)
                if not (output_dir / "belt_stripes_mask.png").is_file():
                    cv2.imwrite(str(output_dir / "belt_stripes_mask.png"), empty_mask)
                if not (output_dir / "belt_base_mask.png").is_file():
                    cv2.imwrite(str(output_dir / "belt_base_mask.png"), empty_mask)
                for _name in (
                    "belt_stripes_tophat_mask.png",
                    "belt_stripes_shape_mask.png",
                    "belt_above_belt_mask.png",
                    "belt_wide_object_mask.png",
                ):
                    if not (output_dir / _name).is_file():
                        cv2.imwrite(str(output_dir / _name), empty_mask)
                if not (output_dir / "belt_baseline_local_min.png").is_file():
                    _write_plateau_plot_not_applicable_png(
                        out_path=output_dir / "belt_baseline_local_min.png",
                        strategy=str(background_detection_strategy),
                    )
                if not (output_dir / "belt_altitude_local_min.png").is_file():
                    _write_plateau_plot_not_applicable_png(
                        out_path=output_dir / "belt_altitude_local_min.png",
                        strategy=str(background_detection_strategy),
                    )
                if not (output_dir / "belt_altitude_histogram.png").is_file():
                    _write_plateau_plot_not_applicable_png(
                        out_path=output_dir / "belt_altitude_histogram.png",
                        strategy=str(background_detection_strategy),
                    )
                if not (output_dir / "belt_altitude_histogram.json").is_file():
                    (output_dir / "belt_altitude_histogram.json").write_text(
                        json.dumps({"strategy": str(background_detection_strategy), "applied": False, "reason": "belt-stripe filter only runs for strategy=low_gradient_depth_plateaus", "sample_count": 0, "bin_counts": [], "bin_edges": [], "threshold_value_mm": 0.0}, indent=2),
                        encoding="utf-8",
                    )
                if not (output_dir / "belt_stripe_filter_debug.json").is_file():
                    (output_dir / "belt_stripe_filter_debug.json").write_text(
                        json.dumps({"strategy": str(background_detection_strategy), "applied": False, "reason": "belt-stripe filter only runs for strategy=low_gradient_depth_plateaus"}, indent=2),
                        encoding="utf-8",
                    )
                for _name in (
                    "belt_bg_mask.png",
                    "unknown_low_gradient_mask.png",
                    "surface_suppression_mask.png",
                    "object_search_domain_mask.png",
                    "belt_stripe_candidates_overlay.png",
                ):
                    if not (output_dir / _name).is_file():
                        cv2.imwrite(str(output_dir / _name), empty_mask)
                if not (output_dir / "belt_bg_components.json").is_file():
                    (output_dir / "belt_bg_components.json").write_text(json.dumps([], indent=2), encoding="utf-8")
                if not (output_dir / "belt_stripe_components.json").is_file():
                    (output_dir / "belt_stripe_components.json").write_text(json.dumps([], indent=2), encoding="utf-8")
            if background_detection_strategy != "low_gradient_blob_height_clusters":
                empty_mask = np.zeros_like(candidate_mask, dtype=np.uint8) if candidate_mask is not None else np.zeros((1, 1), dtype=np.uint8)
                if not (output_dir / "low_gradient_blob_components_overlay.png").is_file():
                    cv2.imwrite(str(output_dir / "low_gradient_blob_components_overlay.png"), empty_mask)
                if not (output_dir / "low_gradient_blob_id_mask.png").is_file():
                    cv2.imwrite(str(output_dir / "low_gradient_blob_id_mask.png"), empty_mask)
                for _name in (
                    "height_border_strength.png",
                    "height_border_cut_mask.png",
                    "height_border_fragments_overlay.png",
                    "height_border_fragments_mask.png",
                    "height_border_fragments_id_mask.png",
                    "height_aware_blob_components_overlay.png",
                    "height_aware_blob_id_mask.png",
                    "height_aware_connectivity_rejected_edges.png",
                    "height_split_blob_fragments_overlay.png",
                    "height_split_blob_fragments_mask.png",
                    "height_split_blob_id_mask.png",
                    "selected_blob_cluster_mask.png",
                    "selected_blob_cluster_pre_refine_mask.png",
                    "selected_blob_cluster_refined_mask.png",
                    "rejected_blob_clusters_mask.png",
                    "selected_blob_cluster_overlay.png",
                    "rejected_blob_clusters_overlay.png",
                ):
                    if not (output_dir / _name).is_file():
                        cv2.imwrite(str(output_dir / _name), empty_mask)
                if not (output_dir / "low_gradient_blob_summary.json").is_file():
                    (output_dir / "low_gradient_blob_summary.json").write_text(
                        json.dumps({"strategy": str(background_detection_strategy), "blob_analysis_active": False, "blobs": []}, indent=2),
                        encoding="utf-8",
                    )
                if not (output_dir / "height_consistent_blob_summary.json").is_file():
                    (output_dir / "height_consistent_blob_summary.json").write_text(
                        json.dumps({"strategy": str(background_detection_strategy), "height_split_active": False, "fragments": []}, indent=2),
                        encoding="utf-8",
                    )
                if not (output_dir / "height_border_split_debug.json").is_file():
                    (output_dir / "height_border_split_debug.json").write_text(
                        json.dumps({"strategy": str(background_detection_strategy), "height_border_split_active": False, "blobs": []}, indent=2),
                        encoding="utf-8",
                    )
                if not (output_dir / "fragment_merge_debug.json").is_file():
                    (output_dir / "fragment_merge_debug.json").write_text(
                        json.dumps({"strategy": str(background_detection_strategy), "merge_active": False, "merges": []}, indent=2),
                        encoding="utf-8",
                    )
                if not (output_dir / "height_split_debug.json").is_file():
                    (output_dir / "height_split_debug.json").write_text(
                        json.dumps({"strategy": str(background_detection_strategy), "height_split_active": False, "blobs": []}, indent=2),
                        encoding="utf-8",
                    )
                if not (output_dir / "blob_height_clusters.json").is_file():
                    (output_dir / "blob_height_clusters.json").write_text(
                        json.dumps({"strategy": str(background_detection_strategy), "blob_clustering_active": False, "clusters": []}, indent=2),
                        encoding="utf-8",
                    )
                if not (output_dir / "blob_cluster_score_table.csv").is_file():
                    (output_dir / "blob_cluster_score_table.csv").write_text(
                        "cluster_id,score,blob_count,fragment_count,original_blob_count,split_fragment_count,mixed_blob_source_count,total_pixels,area_fraction,median_z,z_mad_weighted,touches_roi_border_fraction,touches_frame_border_fraction,mean_centrality,mean_gradient_p95,mean_solidity,mean_compactness,height_gate_overlap,stripe_overlap,rejected,rejected_reason\n",
                        encoding="utf-8",
                    )
                if not (output_dir / "blob_cluster_selection_debug.json").is_file():
                    (output_dir / "blob_cluster_selection_debug.json").write_text(
                        json.dumps({"strategy": str(background_detection_strategy), "blob_cluster_strategy_active": False}, indent=2),
                        encoding="utf-8",
                    )
                if not (output_dir / "height_aware_connectivity_debug.json").is_file():
                    (output_dir / "height_aware_connectivity_debug.json").write_text(
                        json.dumps({"strategy": str(background_detection_strategy), "component_mode": "xy_only", "active": False}, indent=2),
                        encoding="utf-8",
                    )

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
            if background_detection_strategy in {"low_gradient_surface", "low_gradient_depth_plateaus", "low_gradient_bg_and_stripes"}:
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
                # Subtract belt-stripe pixels (chevron ribs detected by the
                # morphological top-hat) from the expanded plane mask.
                # Without this they leak back in through plane expansion +
                # close/flood-fill, since they happen to be close to the
                # fitted plane after the closing step bridges them with the
                # belt. Subtracting here makes "Selected surface" / "Plane
                # inliers" cleanly reflect belt-base support only.
                if np.count_nonzero(stripes_mask):
                    expanded_plane_mask = expanded_plane_mask & (~stripes_mask)
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
                        if np.count_nonzero(stripes_mask):
                            final_plane_inlier_mask = final_plane_inlier_mask & (~stripes_mask)
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
        if background_detection_strategy == "low_gradient_bg_and_stripes":
            expanded_plane_mask = belt_bg_mask.copy()
            final_plane_inlier_mask = belt_bg_mask.copy()
        if np.count_nonzero(final_plane_inlier_mask):
            belt_mask = final_plane_inlier_mask
        else:
            belt_mask = (np.abs(residual_map) <= effective_residual_threshold_mm) & valid_for_fit
        # Strip out belt-stripe pixels so foreground/background statistics
        # below (bg_p95_abs, fg_mean, etc.) aren't biased by chevron ribs.
        if np.count_nonzero(stripes_mask):
            belt_mask = belt_mask & (~stripes_mask)
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

        reference_model_inlier_mask = belt_mask.copy()
        if background_detection_strategy == "low_gradient_bg_and_stripes":
            reference_model_inlier_mask = belt_bg_mask.copy()
            reference_suppression_mask = (belt_bg_mask | stripes_mask).copy()
            surface_suppression_mask = reference_suppression_mask.copy()
            object_search_domain_mask = valid_for_fit & (~surface_suppression_mask)
            belt_mask = reference_model_inlier_mask.copy()
        else:
            reference_suppression_mask = _resolve_reference_suppression_mask(
                reference_suppression_mask_policy,
                selected_support_mask=candidate_mask,
                expanded_support_mask=expanded_plane_mask,
                final_model_inlier_mask=reference_model_inlier_mask,
            )
        selected_blob_cluster_mask = selected_pre_refine_mask.copy() if background_detection_strategy == "low_gradient_blob_height_clusters" else candidate_mask.copy()
        candidate_refined_support_mask = selected_refined_mask.copy() if background_detection_strategy == "low_gradient_blob_height_clusters" else candidate_mask.copy()
        stripe_filtered_reference_support_mask = candidate_mask.copy()
        selected_reference_support_mask = candidate_mask.copy()
        reference_model_support_mask = reference_model_inlier_mask.copy()
        support_removed_by_candidate_refinement = (
            selected_blob_cluster_mask & (~candidate_refined_support_mask)
            if background_detection_strategy == "low_gradient_blob_height_clusters"
            else np.zeros_like(candidate_mask, dtype=bool)
        )
        support_removed_by_stripe_filter = (
            candidate_refined_support_mask & (~stripe_filtered_reference_support_mask)
            if background_detection_strategy == "low_gradient_blob_height_clusters"
            else np.zeros_like(candidate_mask, dtype=bool)
        )
        support_removed_by_model_residual = selected_reference_support_mask & (~reference_model_support_mask)
        support_added_by_candidate_refinement = candidate_refined_support_mask & (~selected_blob_cluster_mask)
        support_added_by_stripe_filter = stripe_filtered_reference_support_mask & (~candidate_refined_support_mask)
        support_added_by_model_expansion = reference_model_support_mask & (~selected_reference_support_mask)
        support_removed_by_suppression_policy = reference_model_support_mask & (~reference_suppression_mask)
        support_added_by_suppression_policy = reference_suppression_mask & (~reference_model_support_mask)

        def _lineage_overlay(source_mask: np.ndarray, removed_mask: np.ndarray, added_mask: np.ndarray | None = None) -> np.ndarray:
            overlay_img = _height_color_image(frame.z_mm, valid_for_fit)
            source_mask_bool = np.asarray(source_mask, dtype=bool)
            removed_mask_bool = np.asarray(removed_mask, dtype=bool)
            added_mask_bool = np.asarray(added_mask, dtype=bool) if added_mask is not None else np.zeros_like(source_mask_bool, dtype=bool)
            kept_mask = source_mask_bool & (~removed_mask_bool)
            overlay_img[kept_mask] = (0, 255, 0)
            overlay_img[removed_mask_bool] = (0, 0, 255)
            overlay_img[added_mask_bool] = (255, 0, 0)
            return overlay_img

        def _mask_step(
            *,
            step_id: str,
            label: str,
            status: str,
            enabled: bool,
            input_mask: np.ndarray | None,
            output_mask: np.ndarray | None,
            removed_mask: np.ndarray | None,
            added_mask: np.ndarray | None,
            primary_reason: str,
            input_artifacts: list[str],
            output_artifacts: list[str],
            removed_artifacts: list[str],
            added_artifacts: list[str],
            parameters: dict[str, Any],
        ) -> dict[str, Any]:
            input_pixels = None if input_mask is None else int(np.count_nonzero(input_mask))
            output_pixels = None if output_mask is None else int(np.count_nonzero(output_mask))
            input_valid_roi_ratio = None if input_mask is None else float(np.count_nonzero(np.asarray(input_mask, dtype=bool) & valid_for_fit) / max(1, np.count_nonzero(valid_for_fit)))
            output_valid_roi_ratio = None if output_mask is None else float(np.count_nonzero(np.asarray(output_mask, dtype=bool) & valid_for_fit) / max(1, np.count_nonzero(valid_for_fit)))
            if removed_mask is None:
                removed_pixels = None
                removed_fraction = None
            else:
                removed_pixels = int(np.count_nonzero(removed_mask))
                removed_fraction = None if input_pixels in (None, 0) else float(removed_pixels / max(1, input_pixels))
            if added_mask is None:
                added_pixels = None
                added_fraction = None
            else:
                added_pixels = int(np.count_nonzero(added_mask))
                added_fraction = None if input_pixels in (None, 0) else float(added_pixels / max(1, input_pixels))
            net_change_pixels = (
                None
                if input_pixels is None or output_pixels is None
                else int(output_pixels - input_pixels)
            )
            net_change_fraction = None if input_pixels in (None, 0) or net_change_pixels is None else float(net_change_pixels / max(1, input_pixels))
            if input_mask is None or output_mask is None:
                change_type = "unknown"
            elif status == "skipped" and input_pixels == output_pixels:
                change_type = "skipped"
            elif input_pixels == output_pixels and np.array_equal(np.asarray(input_mask, dtype=bool), np.asarray(output_mask, dtype=bool)):
                change_type = "alias"
            elif removed_pixels and added_pixels:
                change_type = "mixed"
            elif output_pixels > input_pixels:
                change_type = "expand"
            elif output_pixels < input_pixels:
                change_type = "shrink"
            elif input_pixels == output_pixels:
                change_type = "same"
            else:
                change_type = "unknown"
            return {
                "id": step_id,
                "label": label,
                "status": status,
                "enabled": bool(enabled),
                "input_pixels": input_pixels,
                "output_pixels": output_pixels,
                "input_valid_roi_ratio": input_valid_roi_ratio,
                "output_valid_roi_ratio": output_valid_roi_ratio,
                "removed_pixels": removed_pixels,
                "removed_fraction": removed_fraction,
                "added_pixels": added_pixels,
                "added_fraction": added_fraction,
                "net_change_pixels": net_change_pixels,
                "net_change_fraction": net_change_fraction,
                "change_type": change_type,
                "primary_reason": primary_reason,
                "input_artifacts": input_artifacts,
                "output_artifacts": output_artifacts,
                "removed_artifacts": removed_artifacts,
                "added_artifacts": added_artifacts,
                "parameters": parameters,
            }

        resolved_suppression_policy = (
            "bg_or_stripes" if background_detection_strategy == "low_gradient_bg_and_stripes"
            else "selected_support" if np.array_equal(reference_suppression_mask, selected_reference_support_mask)
            else "expanded_support" if np.array_equal(reference_suppression_mask, expanded_plane_mask)
            else "final_model_inliers" if np.array_equal(reference_suppression_mask, reference_model_support_mask)
            else "custom"
        )
        selected_surface_alias_resolves_to = (
            "selected_reference_support_mask"
            if np.count_nonzero(selected_reference_support_mask)
            else "reference_surface_selected_mask"
        )
        reference_support_lineage_steps = [
            _mask_step(
                step_id="selected_blob_cluster",
                label="Selected blob cluster",
                status="ok",
                enabled=True,
                input_mask=None,
                output_mask=selected_blob_cluster_mask,
                removed_mask=np.zeros_like(selected_blob_cluster_mask, dtype=bool),
                added_mask=None,
                primary_reason=(
                    f"{selected_cluster.get('cluster_id')} selected by score"
                    if background_detection_strategy == "low_gradient_blob_height_clusters" and isinstance(selected_cluster, dict)
                    else "selected support seed resolved"
                ),
                input_artifacts=[],
                output_artifacts=["selected_blob_cluster_mask"],
                removed_artifacts=[],
                added_artifacts=[],
                parameters={
                    "selected_cluster_id": (
                        selected_cluster.get("cluster_id")
                        if background_detection_strategy == "low_gradient_blob_height_clusters" and isinstance(selected_cluster, dict)
                        else None
                    ),
                },
            ),
            _mask_step(
                step_id="candidate_refinement",
                label="Candidate refinement",
                status="ok" if bool(self.blob_cluster_refine_by_mad) else "skipped",
                enabled=bool(self.blob_cluster_refine_by_mad),
                input_mask=selected_blob_cluster_mask,
                output_mask=candidate_refined_support_mask,
                removed_mask=support_removed_by_candidate_refinement,
                added_mask=support_added_by_candidate_refinement,
                primary_reason=(
                    "MAD band refinement applied"
                    if bool(self.blob_cluster_refine_by_mad)
                    else "refine_by_mad disabled"
                ),
                input_artifacts=["selected_blob_cluster_pre_refine_mask"],
                output_artifacts=["selected_blob_cluster_refined_mask"],
                removed_artifacts=["support_removed_by_candidate_refinement"],
                added_artifacts=["support_added_by_candidate_refinement"],
                parameters={
                    "blob_cluster_refine_by_mad": bool(self.blob_cluster_refine_by_mad),
                    "blob_cluster_refine_mad_k": float(self.blob_cluster_refine_mad_k),
                    "blob_cluster_refine_mad_floor_mm": float(self.blob_cluster_refine_floor_mm),
                    "blob_cluster_refine_keep_border_support": bool(self.blob_cluster_refine_keep_border_support),
                    "blob_cluster_min_total_pixels": int(self.blob_cluster_min_total_pixels),
                    "blob_cluster_fallback_strategy": str(self.blob_cluster_fallback_strategy),
                },
            ),
            _mask_step(
                step_id="stripe_suppression",
                label="Stripe suppression",
                status="ok" if bool(stripe_info.get("applied")) else ("skipped" if not bool(self.belt_stripe_filter_enabled) else "ok"),
                enabled=bool(self.belt_stripe_filter_enabled),
                input_mask=candidate_refined_support_mask,
                output_mask=stripe_filtered_reference_support_mask,
                removed_mask=support_removed_by_stripe_filter,
                added_mask=support_added_by_stripe_filter,
                primary_reason=(
                    "belt_stripes_mask removed from support"
                    if np.count_nonzero(support_removed_by_stripe_filter)
                    else str(stripe_info.get("skip_reason") or "no stripe pixels removed")
                ),
                input_artifacts=["selected_blob_cluster_refined_mask"],
                output_artifacts=["stripe_filtered_reference_support_mask", "selected_reference_support_mask"],
                removed_artifacts=["support_removed_by_stripe_filter"],
                added_artifacts=["support_added_by_stripe_filter"],
                parameters={
                    "belt_stripe_filter_enabled": bool(self.belt_stripe_filter_enabled),
                    "belt_stripe_filter_scope": filter_scope,
                    "belt_stripe_filter_threshold_mode": str(self.belt_stripe_filter_threshold_mode),
                    "belt_stripe_filter_min_altitude_mm": float(self.belt_stripe_filter_min_altitude_mm),
                    "belt_stripe_filter_z_floor_enabled": bool(self.belt_stripe_filter_z_floor_enabled),
                    "belt_stripe_filter_object_kernel_mm": float(self.belt_stripe_filter_object_kernel_mm),
                    "belt_stripe_filter_max_stripe_height_mm": float(self.belt_stripe_filter_max_stripe_height_mm),
                    "belt_stripe_filter_warn_removed_fraction": float(self.belt_stripe_filter_warn_removed_fraction),
                },
            ),
            _mask_step(
                step_id="reference_model_support",
                label="Reference model support",
                status="ok",
                enabled=True,
                input_mask=selected_reference_support_mask,
                output_mask=reference_model_support_mask,
                removed_mask=support_removed_by_model_residual,
                added_mask=support_added_by_model_expansion,
                primary_reason=(
                    "constant_z residual threshold / model inlier filtering"
                    if reference_model_type == "constant_z"
                    else "plane residual threshold / model inlier filtering"
                ),
                input_artifacts=["selected_reference_support_mask"],
                output_artifacts=["reference_model_support_mask", "reference_model_inlier_mask"],
                removed_artifacts=["support_removed_by_model_residual"],
                added_artifacts=["support_added_by_model_expansion"],
                parameters={
                    "reference_surface_model": str(self.reference_surface_model),
                    "resolved_reference_surface_model": reference_model_type,
                    "plane_fit_residual_threshold_mm": float(self.plane_fit_residual_threshold_mm),
                    "plane_background_residual_tolerance_mode": str(self.plane_background_residual_tolerance_mode),
                    "plane_background_residual_tolerance_mm": float(self.plane_background_residual_tolerance_mm),
                    "plane_background_residual_adaptive_multiplier": float(self.plane_background_residual_adaptive_multiplier),
                    "plane_fit_min_inlier_ratio": float(self.plane_fit_min_inlier_ratio),
                    "plane_refit_after_expansion": bool(self.plane_refit_after_expansion),
                },
            ),
            _mask_step(
                step_id="suppression_mask",
                label="Suppression mask",
                status="ok",
                enabled=True,
                input_mask=reference_model_support_mask,
                output_mask=reference_suppression_mask,
                removed_mask=support_removed_by_suppression_policy,
                added_mask=support_added_by_suppression_policy,
                primary_reason=f"policy {reference_suppression_mask_policy} resolved to {resolved_suppression_policy}",
                input_artifacts=["reference_model_support_mask"],
                output_artifacts=["reference_suppression_mask"],
                removed_artifacts=["support_removed_by_suppression_policy"],
                added_artifacts=["support_added_by_suppression_policy"],
                parameters={
                    "reference_suppression_mask_policy": reference_suppression_mask_policy,
                    "resolved_policy": resolved_suppression_policy,
                },
            ),
        ]
        largest_loss_step = None
        largest_loss_fraction = 0.0
        largest_gain_step = None
        largest_gain_fraction = 0.0
        for step in reference_support_lineage_steps:
            frac = step.get("removed_fraction")
            if isinstance(frac, (int, float)) and float(frac) > largest_loss_fraction:
                largest_loss_fraction = float(frac)
                largest_loss_step = str(step.get("id"))
            gain_frac = step.get("added_fraction")
            if isinstance(gain_frac, (int, float)) and float(gain_frac) > largest_gain_fraction:
                largest_gain_fraction = float(gain_frac)
                largest_gain_step = str(step.get("id"))
        reference_support_lineage_summary = {
            "selected_cluster_artifact": "selected_blob_cluster_mask",
            "selected_reference_support_artifact": "selected_reference_support_mask",
            "reference_model_support_artifact": "reference_model_support_mask",
            "reference_suppression_artifact": "reference_suppression_mask",
            "selected_surface_alias_resolves_to": selected_surface_alias_resolves_to,
            "selected_surface_alias_role": "selected_support",
            "largest_support_loss_step": largest_loss_step,
            "largest_support_loss_fraction": largest_loss_fraction,
            "largest_support_gain_step": largest_gain_step,
            "largest_support_gain_fraction": largest_gain_fraction,
            "selected_cluster_pixels": int(np.count_nonzero(selected_blob_cluster_mask)),
            "selected_reference_support_pixels": int(np.count_nonzero(selected_reference_support_mask)),
            "reference_model_support_pixels": int(np.count_nonzero(reference_model_support_mask)),
            "reference_suppression_pixels": int(np.count_nonzero(reference_suppression_mask)),
            "selected_cluster_valid_roi_ratio": float(np.count_nonzero(selected_blob_cluster_mask) / max(1, np.count_nonzero(valid_for_fit))),
            "selected_reference_support_valid_roi_ratio": float(np.count_nonzero(selected_reference_support_mask) / max(1, np.count_nonzero(valid_for_fit))),
            "reference_model_support_valid_roi_ratio": float(np.count_nonzero(reference_model_support_mask) / max(1, np.count_nonzero(valid_for_fit))),
            "reference_suppression_valid_roi_ratio": float(np.count_nonzero(reference_suppression_mask) / max(1, np.count_nonzero(valid_for_fit))),
        }
        selected_support_lineage = {
            "stage": "detect_belt_plane",
            "strategy": background_detection_strategy,
            "selected_cluster_id": (
                selected_cluster.get("cluster_id")
                if background_detection_strategy == "low_gradient_blob_height_clusters" and isinstance(selected_cluster, dict)
                else None
            ),
            "reference_surface_model": str(self.reference_surface_model),
            "resolved_reference_surface_model": reference_model_type,
            "reference_suppression_mask_policy": reference_suppression_mask_policy,
            "resolved_suppression_policy": resolved_suppression_policy,
            "fallback_used": bool(fallback_used) if background_detection_strategy == "low_gradient_blob_height_clusters" else False,
            "fallback_reason": fallback_reason if background_detection_strategy == "low_gradient_blob_height_clusters" else None,
            "selected_surface_alias_resolves_to": selected_surface_alias_resolves_to,
            "selected_surface_alias_role": "selected_support",
            "steps": reference_support_lineage_steps,
            "summary": reference_support_lineage_summary,
            "warnings": warnings,
        }
        rejection_reason_counts: dict[str, int] = {}
        for row in fragment_rows:
            if bool(row.get("rejected")):
                key = str(row.get("rejection_reason") or "rejected")
                rejection_reason_counts[key] = rejection_reason_counts.get(key, 0) + 1
        component_formation_debug = {
            "component_mode": blob_component_mode_used if background_detection_strategy == "low_gradient_blob_height_clusters" else str(self.blob_component_mode),
            "connectivity": int(self.blob_connectivity),
            "neighbor_z_tolerance_mm": float(self.blob_neighbor_z_tolerance_mm),
            "max_local_slope_mm_per_px": float(self.blob_component_max_local_slope_mm_per_px),
            "use_smoothed_z": bool(self.blob_component_use_smoothed_z),
            "min_area_px": int(self.blob_component_min_area_px),
            "components_before_filter": int(height_aware_connectivity_debug.get("component_count") or len(blob_rows)),
            "components_after_filter": int(len(blob_rows)),
            "rejected_edges_count": int(height_aware_connectivity_debug.get("rejected_neighbor_edges") or 0),
            "selected_component_ids": [int(v) for v in (selected_cluster.get("blob_ids") or [])] if isinstance(selected_cluster, dict) else [],
            "compact_summary": _make_debug_compact_summary(
                title="Component formation",
                input_count=int(np.count_nonzero(low_grad_mask)),
                output_count=int(len(blob_rows)),
                rejected_reasons=rejection_reason_counts,
                fallback_used=bool(fallback_used) if background_detection_strategy == "low_gradient_blob_height_clusters" else False,
                fallback_reason=fallback_reason if background_detection_strategy == "low_gradient_blob_height_clusters" else None,
                warnings=warnings,
                key_parameters={
                    "blob_component_mode": blob_component_mode_used if background_detection_strategy == "low_gradient_blob_height_clusters" else str(self.blob_component_mode),
                    "blob_connectivity": int(self.blob_connectivity),
                    "blob_neighbor_z_tolerance_mm": float(self.blob_neighbor_z_tolerance_mm),
                    "blob_component_max_local_slope_mm_per_px": float(self.blob_component_max_local_slope_mm_per_px),
                    "blob_component_use_smoothed_z": bool(self.blob_component_use_smoothed_z),
                },
                affected_downstream_artifacts=["height_aware_connectivity_rejected_edges", "height_border_fragments_mask", "blob_height_clusters"],
            ),
        }
        height_border_detection_debug = {
            "threshold_mode": str(self.blob_split_height_border_threshold_mode),
            "percentile": float(self.blob_split_height_border_percentile),
            "min_delta_mm": float(self.blob_split_height_border_min_delta_mm),
            "smoothing_kernel": int(self.blob_split_height_border_smoothing_kernel),
            "candidate_border_count": int(np.count_nonzero(border_cut_mask)),
            "accepted_cut_count": int(np.count_nonzero(border_cut_mask)),
            "rejected_cut_count": max(0, int(np.count_nonzero(height_aware_rejected_edge_mask)) - int(np.count_nonzero(border_cut_mask))),
            "compact_summary": _make_debug_compact_summary(
                title="Height-border detection",
                input_count=int(np.count_nonzero(fragment_labels > 0)) if np.any(fragment_labels > 0) else int(np.count_nonzero(blob_labels > 0)),
                output_count=int(np.count_nonzero(border_cut_mask)),
                warnings=warnings,
                key_parameters={
                    "blob_split_height_border_threshold_mode": str(self.blob_split_height_border_threshold_mode),
                    "blob_split_height_border_percentile": float(self.blob_split_height_border_percentile),
                    "blob_split_height_border_min_delta_mm": float(self.blob_split_height_border_min_delta_mm),
                    "blob_split_height_border_smoothing_kernel": int(self.blob_split_height_border_smoothing_kernel),
                },
                affected_downstream_artifacts=["height_border_fragments_mask", "height_split_blob_fragments_mask"],
            ),
        }
        candidate_support_refinement_debug = {
            "pre_refine_area_px": int(np.count_nonzero(selected_blob_cluster_mask)),
            "post_refine_area_px": int(np.count_nonzero(candidate_refined_support_mask)),
            "removed_by_candidate_refinement_px": int(np.count_nonzero(support_removed_by_candidate_refinement)),
            "removed_by_mad_px": int(np.count_nonzero(support_removed_by_candidate_refinement)),
            "removed_by_border_px": 0,
            "removed_by_min_area_px": 0,
            "refine_by_mad_enabled": bool(self.blob_cluster_refine_by_mad),
            "keep_border_support_enabled": bool(self.blob_cluster_refine_keep_border_support),
            "compact_summary": _make_debug_compact_summary(
                title="Candidate support refinement",
                input_count=int(np.count_nonzero(selected_blob_cluster_mask)),
                output_count=int(np.count_nonzero(candidate_refined_support_mask)),
                selected_id=str(selected_cluster.get("cluster_id")) if isinstance(selected_cluster, dict) else None,
                warnings=warnings,
                key_parameters={
                    "blob_cluster_refine_by_mad": bool(self.blob_cluster_refine_by_mad),
                    "blob_cluster_refine_mad_k": float(self.blob_cluster_refine_mad_k),
                    "blob_cluster_refine_floor_mm": float(self.blob_cluster_refine_floor_mm),
                    "blob_cluster_refine_keep_border_support": bool(self.blob_cluster_refine_keep_border_support),
                },
                affected_downstream_artifacts=["selected_blob_cluster_refined_mask", "support_removed_by_candidate_refinement", "selected_reference_support_mask"],
            ),
        }
        stripe_overlap_with_fit_support_px = int(np.count_nonzero(reference_model_support_mask & stripes_mask))
        final_support_debug = {
            "logical_support_area_px": int(np.count_nonzero(selected_blob_cluster_mask)),
            "fit_support_area_px": int(np.count_nonzero(reference_model_support_mask)),
            "suppression_mask_area_px": int(np.count_nonzero(reference_suppression_mask)),
            "bridge_only_area_px": max(0, int(np.count_nonzero(selected_blob_cluster_mask)) - int(np.count_nonzero(selected_reference_support_mask))),
            "stripe_overlap_with_fit_support_px": stripe_overlap_with_fit_support_px,
            "fit_support_excludes_stripes": stripe_overlap_with_fit_support_px == 0,
            "fit_support_excludes_bridge_pixels": int(np.count_nonzero(reference_model_support_mask & (~selected_reference_support_mask))) == 0,
            "final_model_type": reference_model_type,
            "fallback_reason": fallback_reason if fallback_used else None,
            "compact_summary": _make_debug_compact_summary(
                title="Final support vs fit support",
                input_count=int(np.count_nonzero(selected_blob_cluster_mask)),
                output_count=int(np.count_nonzero(reference_model_support_mask)),
                selected_id=str(selected_cluster.get("cluster_id")) if isinstance(selected_cluster, dict) else None,
                selected_reason=model_selection_reason,
                fallback_used=bool(fallback_used),
                fallback_reason=fallback_reason,
                warnings=warnings,
                key_parameters={
                    "reference_surface_model": str(self.reference_surface_model),
                    "resolved_reference_surface_model": reference_model_type,
                    "reference_suppression_mask_policy": reference_suppression_mask_policy,
                },
                affected_downstream_artifacts=["reference_model_support_mask", "reference_suppression_mask", "final_plane_inlier_mask"],
                extra={"stripe_overlap_with_fit_support_px": stripe_overlap_with_fit_support_px},
            ),
        }
        support_loss_waterfall = [
            _support_loss_row(row_id="valid_roi", label="Valid ROI", substage_id="input", mask_artifact_id="valid_mask", mask=valid_for_fit, previous_mask=None, valid_roi_mask=valid_for_fit, notes="Stage-valid pixels inside ROI."),
            _support_loss_row(row_id="low_gradient_candidates", label="Low-gradient candidates", substage_id="gradient", mask_artifact_id="low_gradient_mask", mask=low_grad_mask, previous_mask=valid_for_fit, valid_roi_mask=valid_for_fit, parameters_used_summary={"gradient_threshold_percentile": float(self.gradient_threshold_percentile)}),
            _support_loss_row(row_id="height_gate_candidates", label="Height-gated candidates", substage_id="height_gate", mask_artifact_id="flat_candidate_mask", mask=flat_candidate_mask if 'flat_candidate_mask' in locals() else None, previous_mask=low_grad_mask, valid_roi_mask=valid_for_fit),
            _support_loss_row(row_id="component_candidates", label="Component candidates", substage_id="component_formation", mask_artifact_id="low_gradient_blob_id_mask", mask=(blob_labels > 0) if 'blob_labels' in locals() else None, previous_mask=flat_candidate_mask if 'flat_candidate_mask' in locals() else low_grad_mask, valid_roi_mask=valid_for_fit),
            _support_loss_row(row_id="height_aware_components", label="Height-aware components", substage_id="height_aware_connectivity", mask_artifact_id="height_aware_blob_id_mask", mask=(blob_labels > 0) if background_detection_strategy == "low_gradient_blob_height_clusters" else None, previous_mask=(blob_labels > 0) if 'blob_labels' in locals() else None, valid_roi_mask=valid_for_fit, notes="Connectivity-compatible components after rejected Z-connections."),
            _support_loss_row(row_id="height_border_fragments", label="Height-border fragments", substage_id="blob_splitting", mask_artifact_id="height_border_fragments_mask", mask=border_fragment_mask if 'border_fragment_mask' in locals() else None, previous_mask=(blob_labels > 0) if 'blob_labels' in locals() else None, valid_roi_mask=valid_for_fit),
            _support_loss_row(row_id="post_merge_fragments", label="Post-merge fragments", substage_id="fragment_merge", mask_artifact_id="height_split_blob_fragments_mask", mask=(fragment_labels > 0) if 'fragment_labels' in locals() else None, previous_mask=border_fragment_mask if 'border_fragment_mask' in locals() else None, valid_roi_mask=valid_for_fit, largest_loss_reason="weak_boundary_merge_disabled" if not bool(self.blob_split_merge_weak_boundaries) else None),
            _support_loss_row(row_id="height_clusters", label="Height clusters", substage_id="height_clustering", mask_artifact_id="height_split_blob_fragments_mask", mask=np.isin(fragment_labels, [int(row.get("fragment_id") or 0) for row in fragment_rows if not bool(row.get("rejected"))]) if 'fragment_labels' in locals() else None, previous_mask=(fragment_labels > 0) if 'fragment_labels' in locals() else None, valid_roi_mask=valid_for_fit),
            _support_loss_row(row_id="selected_height_cluster", label="Selected height cluster", substage_id="height_clustering", mask_artifact_id="selected_blob_cluster_mask", mask=selected_blob_cluster_mask, previous_mask=np.isin(fragment_labels, [int(row.get("fragment_id") or 0) for row in fragment_rows if not bool(row.get("rejected"))]) if 'fragment_labels' in locals() else None, valid_roi_mask=valid_for_fit, notes=(str(selected_cluster.get("cluster_id")) if isinstance(selected_cluster, dict) else None)),
            _support_loss_row(row_id="selected_cluster_pre_refine", label="Selected cluster pre-refine", substage_id="candidate_support_refinement", mask_artifact_id="selected_blob_cluster_pre_refine_mask", mask=selected_blob_cluster_mask, previous_mask=selected_blob_cluster_mask, valid_roi_mask=valid_for_fit),
            _support_loss_row(row_id="selected_cluster_refined", label="Selected cluster refined", substage_id="candidate_support_refinement", mask_artifact_id="selected_blob_cluster_refined_mask", mask=candidate_refined_support_mask, previous_mask=selected_blob_cluster_mask, valid_roi_mask=valid_for_fit, largest_loss_reason="candidate_refinement"),
            _support_loss_row(row_id="stripe_filtered_support", label="Stripe-filtered support", substage_id="stripes", mask_artifact_id="stripe_filtered_reference_support_mask", mask=stripe_filtered_reference_support_mask, previous_mask=candidate_refined_support_mask, valid_roi_mask=valid_for_fit, largest_loss_reason="stripe_suppression" if np.count_nonzero(support_removed_by_stripe_filter) else None),
            _support_loss_row(row_id="final_selected_support", label="Final selected support", substage_id="final_support", mask_artifact_id="selected_reference_support_mask", mask=selected_reference_support_mask, previous_mask=stripe_filtered_reference_support_mask, valid_roi_mask=valid_for_fit),
            _support_loss_row(row_id="reference_model_fit_support", label="Reference-model fit support", substage_id="final_support", mask_artifact_id="reference_model_support_mask", mask=reference_model_support_mask, previous_mask=selected_reference_support_mask, valid_roi_mask=valid_for_fit, largest_loss_reason="model_residual_filter"),
            _support_loss_row(row_id="reference_suppression_mask", label="Reference suppression mask", substage_id="final_support", mask_artifact_id="reference_suppression_mask", mask=reference_suppression_mask, previous_mask=reference_model_support_mask, valid_roi_mask=valid_for_fit),
            _support_loss_row(row_id="plane_inliers", label="Plane inliers", substage_id="plane_fit", mask_artifact_id="final_plane_inlier_mask", mask=final_plane_inlier_mask, previous_mask=reference_model_support_mask, valid_roi_mask=valid_for_fit),
        ]
        for index, row in enumerate(support_loss_waterfall):
            row["order_index"] = index
        resolved_reference_method = {
            "preset_id": _reference_method_preset_id(
                background_detection_strategy=background_detection_strategy,
                blob_component_mode=str(self.blob_component_mode),
                blob_split_method=str(self.blob_split_method),
                reference_surface_model=str(self.reference_surface_model),
                reference_suppression_mask_policy=reference_suppression_mask_policy,
                blob_component_use_smoothed_z=bool(self.blob_component_use_smoothed_z),
                blob_neighbor_z_tolerance_mm=float(self.blob_neighbor_z_tolerance_mm),
            ),
            "support_selection_method": support_selection_method,
            "background_detection_strategy": background_detection_strategy,
            "blob_component_mode": str(self.blob_component_mode),
            "blob_split_method": str(self.blob_split_method),
            "reference_surface_model": str(self.reference_surface_model),
            "resolved_reference_surface_model": reference_model_type,
            "reference_suppression_mask_policy": reference_suppression_mask_policy,
            "fallback_used": model_selection_reason == "auto_constant_z_fallback",
            "fallback_strategy": "constant_z" if model_selection_reason == "auto_constant_z_fallback" else None,
        }

        _runtime_trace_mask_unit(
            context,
            unit_id="detect_belt_plane.roi",
            stage_id="detect_belt_plane",
            parent_id="detect_belt_plane",
            parameters={"plane_fit_roi": roi_cfg} if "roi_cfg" in locals() else {},
            output_artifacts=["plane_fit_roi_mask"],
            metrics={"roi_pixels": int(np.count_nonzero(roi_mask)), "valid_roi_pixels": int(np.count_nonzero(valid_for_fit))},
        )
        _runtime_trace_mask_unit(
            context,
            unit_id="detect_belt_plane.height_gate",
            stage_id="detect_belt_plane",
            parent_id="detect_belt_plane",
            parameters={
                "background_detection_strategy": background_detection_strategy,
                "reference_surface_min_area_ratio": float(self.reference_surface_min_area_ratio),
            },
            input_artifacts=["plane_fit_roi_mask"],
            output_artifacts=["height_gate_mask", "flat_candidate_mask", "reference_surface_candidates"],
            metrics={
                "height_gate_pixels": int(np.count_nonzero(active_height_gate_mask)),
                "candidate_pixels": int(np.count_nonzero(candidate_mask)),
                "candidate_coverage": float(candidate_coverage),
            },
            warnings=[warning for warning in warnings if "background_candidate" in warning or "reference_surface" in warning],
        )
        _runtime_trace_mask_unit(
            context,
            unit_id="detect_belt_plane.depth_gradient",
            stage_id="detect_belt_plane",
            parent_id="detect_belt_plane",
            parameters={"gradient_threshold_percentile": float(self.gradient_threshold_percentile)},
            input_artifacts=["plane_fit_roi_mask"],
            output_artifacts=["depth_gradient_magnitude", "low_gradient_mask", "gradient_debug"],
            metrics={"low_gradient_pixels": int(np.count_nonzero(low_grad_mask))},
        )
        _runtime_trace_mask_unit(
            context,
            unit_id="detect_belt_plane.depth_plateaus",
            stage_id="detect_belt_plane",
            parent_id="detect_belt_plane",
            status="completed" if background_detection_strategy in {"low_gradient_depth_plateaus", "low_gradient_bg_and_stripes"} else "skipped",
            parameters={"low_gradient_plateau_selection_mode": str(self.low_gradient_plateau_selection_mode)},
            input_artifacts=["low_gradient_mask"],
            output_artifacts=["reference_surface_plateaus", "reference_surface_candidates", "background_selected_plateau_mask"],
            metrics={"selected_plateau_pixels": int(np.count_nonzero(candidate_mask))},
            diagnostics={"strategy": background_detection_strategy},
        )
        _runtime_trace_mask_unit(
            context,
            unit_id="detect_belt_plane.blob_components",
            stage_id="detect_belt_plane",
            parent_id="detect_belt_plane",
            status="completed" if background_detection_strategy in {"low_gradient_blob_height_clusters", "low_gradient_bg_and_stripes"} else "skipped",
            parameters={
                "blob_component_mode": str(self.blob_component_mode),
                "blob_connectivity": int(self.blob_connectivity),
                "blob_component_min_area_px": int(self.blob_component_min_area_px),
            },
            input_artifacts=["low_gradient_mask"],
            output_artifacts=["low_gradient_blob_id_mask", "low_gradient_blob_components_overlay", "height_aware_blob_components_overlay"],
            metrics={"component_count": int(len(blob_rows)), "selected_component_count": int(len((selected_cluster.get("blob_ids") or []) if isinstance(selected_cluster, dict) else []))},
        )
        _runtime_trace_mask_unit(
            context,
            unit_id="detect_belt_plane.blob_splitting",
            stage_id="detect_belt_plane",
            parent_id="detect_belt_plane",
            status="skipped" if background_detection_strategy != "low_gradient_blob_height_clusters" else "completed",
            parameters={"blob_split_method": str(self.blob_split_method)},
            input_artifacts=["low_gradient_blob_id_mask"],
            output_artifacts=["height_border_fragments_mask", "height_split_blob_fragments_mask", "height_split_debug"],
            metrics={"fragment_count": int(len(fragment_rows)), "border_cut_pixels": int(np.count_nonzero(border_cut_mask))},
        )
        _runtime_trace_mask_unit(
            context,
            unit_id="detect_belt_plane.fragment_merge",
            stage_id="detect_belt_plane",
            parent_id="detect_belt_plane",
            status="skipped" if background_detection_strategy != "low_gradient_blob_height_clusters" else "completed",
            parameters={"blob_split_merge_weak_boundaries": bool(self.blob_split_merge_weak_boundaries)},
            input_artifacts=["height_split_blob_fragments_mask"],
            output_artifacts=["fragment_merge_debug"],
            metrics={"merged_fragment_count": int(len(fragment_rows))},
        )
        _runtime_trace_mask_unit(
            context,
            unit_id="detect_belt_plane.candidate_support_refinement",
            stage_id="detect_belt_plane",
            parent_id="detect_belt_plane",
            status="skipped" if not bool(self.blob_cluster_refine_by_mad) else "completed",
            parameters={
                "blob_cluster_refine_by_mad": bool(self.blob_cluster_refine_by_mad),
                "blob_cluster_refine_mad_k": float(self.blob_cluster_refine_mad_k),
                "blob_cluster_refine_floor_mm": float(self.blob_cluster_refine_floor_mm),
                "blob_cluster_refine_keep_border_support": bool(self.blob_cluster_refine_keep_border_support),
            },
            input_artifacts=["selected_blob_cluster_pre_refine_mask"],
            output_artifacts=["selected_blob_cluster_refined_mask", "support_removed_by_candidate_refinement"],
            metrics={
                "selected_cluster_pixels": int(np.count_nonzero(selected_blob_cluster_mask)),
                "refined_support_pixels": int(np.count_nonzero(candidate_refined_support_mask)),
                "removed_pixels": int(np.count_nonzero(support_removed_by_candidate_refinement)),
            },
            diagnostics={"selected_cluster_id": selected_cluster.get("cluster_id") if isinstance(selected_cluster, dict) else None},
        )
        _runtime_trace_mask_unit(
            context,
            unit_id="detect_belt_plane.stripe_filter",
            stage_id="detect_belt_plane",
            parent_id="detect_belt_plane",
            status="skipped" if not bool(self.belt_stripe_filter_enabled) or not bool(stripe_info.get("applied")) else "completed",
            parameters={
                "belt_stripe_filter_enabled": bool(self.belt_stripe_filter_enabled),
                "belt_stripe_filter_scope": filter_scope,
                "belt_stripe_filter_threshold_mode": str(self.belt_stripe_filter_threshold_mode),
            },
            input_artifacts=["selected_blob_cluster_refined_mask"],
            output_artifacts=["belt_stripes_mask", "stripe_filtered_reference_support_mask", "support_removed_by_stripe_filter"],
            metrics={
                "stripe_pixels": int(np.count_nonzero(stripes_mask)),
                "removed_pixels": int(np.count_nonzero(support_removed_by_stripe_filter)),
                "output_pixels": int(np.count_nonzero(stripe_filtered_reference_support_mask)),
                "raw_negative_altitude_pixels": int(
                    ((stripe_info.get("invariants") or {}).get("raw_negative_altitude_pixel_count") or 0)
                ),
                "clamped_negative_altitude_pixels": int(
                    ((stripe_info.get("invariants") or {}).get("negative_altitude_pixel_count") or 0)
                ),
            },
            diagnostics={"skip_reason": stripe_info.get("skip_reason"), "applied": stripe_info.get("applied")},
            warnings=[warning for warning in warnings if "stripe" in warning],
        )
        _runtime_trace_mask_unit(
            context,
            unit_id="detect_belt_plane.reference_model_fit",
            stage_id="detect_belt_plane",
            parent_id="detect_belt_plane",
            status="warning" if warnings else "completed",
            parameters={
                "reference_surface_model": str(self.reference_surface_model),
                "resolved_reference_surface_model": reference_model_type,
                "plane_fit_residual_threshold_mm": float(self.plane_fit_residual_threshold_mm),
                "plane_fit_min_inlier_ratio": float(self.plane_fit_min_inlier_ratio),
            },
            input_artifacts=["selected_reference_support_mask"],
            output_artifacts=["belt_plane", "final_plane_inlier_mask", "reference_model_support_mask", "plane_fit_debug"],
            metrics={
                "sampled_pixel_count": int(sampled_pixel_count),
                "inlier_ratio": float(inlier_ratio),
                "residual_p95_abs_mm": float(stats.get("p95_abs_mm") or 0.0),
                "reference_model_support_pixels": int(np.count_nonzero(reference_model_support_mask)),
            },
            diagnostics={"model_selection_reason": model_selection_reason, "plane_fit_status": status},
            warnings=warnings,
        )
        _runtime_trace_mask_unit(
            context,
            unit_id="detect_belt_plane.final_support",
            stage_id="detect_belt_plane",
            parent_id="detect_belt_plane",
            parameters={
                "reference_suppression_mask_policy": reference_suppression_mask_policy,
                "resolved_reference_surface_model": reference_model_type,
            },
            input_artifacts=["reference_model_support_mask", "stripe_filtered_reference_support_mask"],
            output_artifacts=["selected_reference_support_mask", "reference_suppression_mask", "final_support_debug", "support_loss_waterfall"],
            metrics={
                "selected_support_pixels": int(np.count_nonzero(selected_reference_support_mask)),
                "reference_model_support_pixels": int(np.count_nonzero(reference_model_support_mask)),
                "reference_suppression_pixels": int(np.count_nonzero(reference_suppression_mask)),
            },
            diagnostics={"largest_support_loss_step": largest_loss_step, "selected_surface_alias_resolves_to": selected_surface_alias_resolves_to},
            warnings=warnings,
        )

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
        context.set_artifact("selected_reference_support_mask", selected_reference_support_mask)
        context.set_artifact("stripe_filtered_reference_support_mask", stripe_filtered_reference_support_mask)
        context.set_artifact("support_removed_by_candidate_refinement", support_removed_by_candidate_refinement)
        context.set_artifact("support_added_by_candidate_refinement", support_added_by_candidate_refinement)
        context.set_artifact("support_removed_by_stripe_filter", support_removed_by_stripe_filter)
        context.set_artifact("support_added_by_stripe_filter", support_added_by_stripe_filter)
        context.set_artifact("reference_model_inlier_mask", reference_model_inlier_mask)
        context.set_artifact("reference_model_support_mask", reference_model_support_mask)
        context.set_artifact("belt_bg_mask_array", np.asarray(belt_bg_mask, dtype=bool))
        context.set_artifact("unknown_low_gradient_mask_array", np.asarray(unknown_low_gradient_mask, dtype=bool))
        context.set_artifact("surface_suppression_mask_array", np.asarray(reference_suppression_mask, dtype=bool))
        context.set_artifact("object_search_domain_mask_array", np.asarray(object_search_domain_mask, dtype=bool))
        context.set_artifact("support_removed_by_model_residual", support_removed_by_model_residual)
        context.set_artifact("support_added_by_model_expansion", support_added_by_model_expansion)
        context.set_artifact("reference_suppression_mask", reference_suppression_mask)
        context.set_artifact("support_removed_by_suppression_policy", support_removed_by_suppression_policy)
        context.set_artifact("support_added_by_suppression_policy", support_added_by_suppression_policy)
        context.set_artifact("reference_support_lineage", reference_support_lineage_summary)
        context.set_artifact("selected_support_lineage", selected_support_lineage)
        context.set_artifact("component_formation_debug", component_formation_debug)
        context.set_artifact("height_border_detection_debug", height_border_detection_debug)
        context.set_artifact("candidate_support_refinement_debug", candidate_support_refinement_debug)
        context.set_artifact("final_support_debug", final_support_debug)
        context.set_artifact("support_loss_waterfall", support_loss_waterfall)
        context.set_artifact("reference_method", resolved_reference_method)
        # Expose the belt-stripes mask so downstream segmentation can
        # subtract surface texture (chevron ribs) from foreground candidates.
        # Set even when empty so consumers always find a well-defined mask.
        context.set_artifact("belt_stripes_mask_array", np.asarray(stripes_mask, dtype=bool))
        context.set_artifact("belt_base_mask_array", np.asarray(belt_base_mask, dtype=bool))
        context.set_artifact("plane_fit_status", status)
        context.set_artifact("plane_fit_failure_reason", failure_reason)

        plane_json = {
            "reference_surface_model_type": reference_model_type,
            "constant_z_mm": reference_constant_z_mm if reference_model_type == "constant_z" else None,
            "model_selection_reason": model_selection_reason,
            "reference_method": resolved_reference_method,
            "support_role_metadata": {
                "background_model_support_artifact": "belt_bg_mask" if background_detection_strategy == "low_gradient_bg_and_stripes" else "reference_model_support_mask",
                "surface_suppression_artifact": "surface_suppression_mask" if background_detection_strategy == "low_gradient_bg_and_stripes" else "reference_suppression_mask",
                "stripes_excluded_from_background_fit": background_detection_strategy == "low_gradient_bg_and_stripes",
            },
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
        cv2.imwrite(str(output_dir / "rejected_raised_structure_mask.png"), (raised_candidate_rejected_mask.astype(np.uint8) * 255))
        # The plateau-specific masks were already written by the strategy branch
        # with the *original* flat-candidate selection (before downstream
        # morphology / component filtering). Preserve them so the user can
        # verify the lower-belt vs raised-chevron separation against what the
        # plateau analysis actually saw. Only re-emit empty placeholders here
        # for non-plateau strategies so the artifact contract stays stable.
        if background_detection_strategy != "low_gradient_depth_plateaus":
            empty_mask = np.zeros_like(candidate_mask, dtype=np.uint8)
            if not (output_dir / "background_selected_plateau_mask.png").is_file():
                cv2.imwrite(str(output_dir / "background_selected_plateau_mask.png"), empty_mask)
            if not (output_dir / "rejected_raised_plateau_mask.png").is_file():
                cv2.imwrite(str(output_dir / "rejected_raised_plateau_mask.png"), empty_mask)
        cv2.imwrite(str(output_dir / "background_seed_mask.png"), (seed_mask.astype(np.uint8) * 255))
        cv2.imwrite(str(output_dir / "expanded_plane_mask.png"), (expanded_plane_mask.astype(np.uint8) * 255))
        cv2.imwrite(str(output_dir / "final_plane_inlier_mask.png"), (belt_mask.astype(np.uint8) * 255))
        cv2.imwrite(str(output_dir / "selected_reference_support_mask.png"), (selected_reference_support_mask.astype(np.uint8) * 255))
        cv2.imwrite(str(output_dir / "stripe_filtered_reference_support_mask.png"), (stripe_filtered_reference_support_mask.astype(np.uint8) * 255))
        cv2.imwrite(str(output_dir / "support_removed_by_candidate_refinement.png"), (support_removed_by_candidate_refinement.astype(np.uint8) * 255))
        cv2.imwrite(str(output_dir / "support_added_by_candidate_refinement.png"), (support_added_by_candidate_refinement.astype(np.uint8) * 255))
        cv2.imwrite(str(output_dir / "support_removed_by_stripe_filter.png"), (support_removed_by_stripe_filter.astype(np.uint8) * 255))
        cv2.imwrite(str(output_dir / "support_added_by_stripe_filter.png"), (support_added_by_stripe_filter.astype(np.uint8) * 255))
        cv2.imwrite(str(output_dir / "reference_model_inlier_mask.png"), (reference_model_inlier_mask.astype(np.uint8) * 255))
        cv2.imwrite(str(output_dir / "reference_model_support_mask.png"), (reference_model_support_mask.astype(np.uint8) * 255))
        cv2.imwrite(str(output_dir / "support_removed_by_model_residual.png"), (support_removed_by_model_residual.astype(np.uint8) * 255))
        cv2.imwrite(str(output_dir / "support_added_by_model_expansion.png"), (support_added_by_model_expansion.astype(np.uint8) * 255))
        cv2.imwrite(str(output_dir / "reference_suppression_mask.png"), (reference_suppression_mask.astype(np.uint8) * 255))
        cv2.imwrite(str(output_dir / "belt_bg_mask.png"), (belt_bg_mask.astype(np.uint8) * 255))
        cv2.imwrite(str(output_dir / "unknown_low_gradient_mask.png"), (unknown_low_gradient_mask.astype(np.uint8) * 255))
        cv2.imwrite(str(output_dir / "surface_suppression_mask.png"), (reference_suppression_mask.astype(np.uint8) * 255))
        cv2.imwrite(str(output_dir / "object_search_domain_mask.png"), (object_search_domain_mask.astype(np.uint8) * 255))
        cv2.imwrite(str(output_dir / "support_removed_by_suppression_policy.png"), (support_removed_by_suppression_policy.astype(np.uint8) * 255))
        cv2.imwrite(str(output_dir / "support_added_by_suppression_policy.png"), (support_added_by_suppression_policy.astype(np.uint8) * 255))
        cv2.imwrite(
            str(output_dir / "support_removed_by_candidate_refinement_overlay.png"),
            _lineage_overlay(selected_blob_cluster_mask, support_removed_by_candidate_refinement, support_added_by_candidate_refinement),
        )
        cv2.imwrite(
            str(output_dir / "support_removed_by_stripe_filter_overlay.png"),
            _lineage_overlay(candidate_refined_support_mask, support_removed_by_stripe_filter, support_added_by_stripe_filter),
        )
        cv2.imwrite(
            str(output_dir / "support_removed_by_model_residual_overlay.png"),
            _lineage_overlay(selected_reference_support_mask, support_removed_by_model_residual, support_added_by_model_expansion),
        )
        cv2.imwrite(
            str(output_dir / "support_removed_by_suppression_policy_overlay.png"),
            _lineage_overlay(reference_model_support_mask, support_removed_by_suppression_policy, support_added_by_suppression_policy),
        )
        (output_dir / "selected_support_lineage.json").write_text(json.dumps(selected_support_lineage, indent=2), encoding="utf-8")
        (output_dir / "component_formation_debug.json").write_text(json.dumps(component_formation_debug, indent=2), encoding="utf-8")
        (output_dir / "height_border_detection_debug.json").write_text(json.dumps(height_border_detection_debug, indent=2), encoding="utf-8")
        (output_dir / "candidate_support_refinement_debug.json").write_text(json.dumps(candidate_support_refinement_debug, indent=2), encoding="utf-8")
        (output_dir / "final_support_debug.json").write_text(json.dumps(final_support_debug, indent=2), encoding="utf-8")
        (output_dir / "support_loss_waterfall.json").write_text(json.dumps(support_loss_waterfall, indent=2), encoding="utf-8")
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
            "background_detection_strategy": background_detection_strategy,
            "support_selection_method": support_selection_method,
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
            "surface_roles": {
                "belt_bg_pixels": int(np.count_nonzero(belt_bg_mask)),
                "belt_stripe_pixels": int(np.count_nonzero(stripes_mask)),
                "unknown_low_gradient_pixels": int(np.count_nonzero(unknown_low_gradient_mask)),
                "suppression_pixels": int(np.count_nonzero(reference_suppression_mask)),
            },
            "warnings": warnings,
            "roi_inspector": {
                "enabled": roi_enabled,
                "selected_mode": region_mode,
                "coverage_ratio": roi_area_ratio,
                "coverage_percent": roi_area_ratio * 100.0,
                "valid_pixel_ratio": roi_valid_ratio,
                "candidate_coverage_ratio": roi_candidate_coverage,
                "selected_surface_coverage_ratio": roi_selected_surface_coverage,
            },
            "reference_method": resolved_reference_method,
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
            "reference_method": resolved_reference_method,
            "model_residual_mean_mm": float(np.mean(residual_abs)) if residual_abs.size else 0.0,
            "model_residual_p95_mm": float(np.percentile(residual_abs, 95)) if residual_abs.size else 0.0,
            "roi_enabled": roi_enabled,
            "roi_type": roi_type,
            "reference_surface_region_mode": region_mode,
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
            "reference_method": resolved_reference_method,
            "config": {
                "plane_fit_roi": roi_cfg,
                "background_detection_strategy": background_detection_strategy,
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
                "reference_surface_region_mode": self.reference_surface_region_mode,
                "reference_surface_model": self.reference_surface_model,
                "reference_suppression_mask_policy": reference_suppression_mask_policy,
                "support_selection_method": support_selection_method,
                "reference_surface_height_gate_enabled": self.reference_surface_height_gate_enabled,
                "reference_surface_height_gate_margin_mm": self.reference_surface_height_gate_margin_mm,
                "reference_surface_height_gate_min_coverage_ratio": self.reference_surface_height_gate_min_coverage_ratio,
                "reference_surface_height_gate_max_coverage_ratio": self.reference_surface_height_gate_max_coverage_ratio,
                "low_gradient_plateau_use_hessian_filter": self.low_gradient_plateau_use_hessian_filter,
                "low_gradient_plateau_hessian_percentile": self.low_gradient_plateau_hessian_percentile,
                "low_gradient_plateau_hist_bins": self.low_gradient_plateau_hist_bins,
                "low_gradient_plateau_min_fraction": self.low_gradient_plateau_min_fraction,
                "low_gradient_plateau_min_pixels": self.low_gradient_plateau_min_pixels,
                "low_gradient_plateau_smoothing_sigma_bins": self.low_gradient_plateau_smoothing_sigma_bins,
                "low_gradient_plateau_peak_drop_ratio": self.low_gradient_plateau_peak_drop_ratio,
                "low_gradient_plateau_select_min_area_fraction": self.low_gradient_plateau_select_min_area_fraction,
                "low_gradient_plateau_selection_mode": self.low_gradient_plateau_selection_mode,
                "belt_stripe_filter_enabled": self.belt_stripe_filter_enabled,
                "belt_stripe_filter_window_mm": self.belt_stripe_filter_window_mm,
                "belt_stripe_filter_direction": self.belt_stripe_filter_direction,
                "belt_stripe_filter_threshold_mode": self.belt_stripe_filter_threshold_mode,
                "belt_stripe_filter_min_altitude_mm": self.belt_stripe_filter_min_altitude_mm,
                "belt_stripe_filter_k_mad": self.belt_stripe_filter_k_mad,
                "belt_stripe_filter_fixed_threshold_mm": self.belt_stripe_filter_fixed_threshold_mm,
                "belt_stripe_filter_min_stripe_fraction": self.belt_stripe_filter_min_stripe_fraction,
                "belt_stripe_filter_scope": self.belt_stripe_filter_scope,
                "belt_stripe_filter_z_floor_enabled": self.belt_stripe_filter_z_floor_enabled,
                "belt_stripe_filter_z_floor_use_upper_bound": self.belt_stripe_filter_z_floor_use_upper_bound,
                "belt_stripe_filter_z_floor_margin_mm": self.belt_stripe_filter_z_floor_margin_mm,
                "belt_stripe_filter_max_stripe_height_mm": self.belt_stripe_filter_max_stripe_height_mm,
                "belt_stripe_filter_above_belt_close_mm": self.belt_stripe_filter_above_belt_close_mm,
                "belt_stripe_filter_object_kernel_mm": self.belt_stripe_filter_object_kernel_mm,
                "belt_stripe_filter_object_kernel_shape": self.belt_stripe_filter_object_kernel_shape,
                # Newly-exposed knobs (previously hardcoded)
                "belt_stripe_filter_altitude_hist_bins": self.belt_stripe_filter_altitude_hist_bins,
                "belt_stripe_filter_auto_bimodality_margin": self.belt_stripe_filter_auto_bimodality_margin,
                "belt_stripe_filter_z_floor_upper_percentile": self.belt_stripe_filter_z_floor_upper_percentile,
                "belt_stripe_filter_z_floor_fallback_lower_percentile": self.belt_stripe_filter_z_floor_fallback_lower_percentile,
                "belt_stripe_filter_z_floor_fallback_upper_percentile": self.belt_stripe_filter_z_floor_fallback_upper_percentile,
                "belt_stripe_filter_warn_removed_fraction": self.belt_stripe_filter_warn_removed_fraction,
                "stripe_min_height_over_base_mm": self.stripe_min_height_over_base_mm,
                "stripe_max_height_over_base_mm": self.stripe_max_height_over_base_mm,
                "stripe_min_width_mm": self.stripe_min_width_mm,
                "stripe_max_width_mm": self.stripe_max_width_mm,
                "stripe_min_height_width_ratio": self.stripe_min_height_width_ratio,
                "stripe_max_height_width_ratio": self.stripe_max_height_width_ratio,
                "stripe_max_width_cv": self.stripe_max_width_cv,
                "stripe_min_length_width_ratio": self.stripe_min_length_width_ratio,
                "stripe_max_area_fraction": self.stripe_max_area_fraction,
                "stripe_component_min_overlap_fraction": self.stripe_component_min_overlap_fraction,
                "stripe_rule_overlap_enabled": self.stripe_rule_overlap_enabled,
                "stripe_rule_height_range_enabled": self.stripe_rule_height_range_enabled,
                "stripe_rule_width_range_enabled": self.stripe_rule_width_range_enabled,
                "stripe_rule_height_width_ratio_enabled": self.stripe_rule_height_width_ratio_enabled,
                "stripe_rule_width_cv_enabled": self.stripe_rule_width_cv_enabled,
                "stripe_rule_length_width_ratio_enabled": self.stripe_rule_length_width_ratio_enabled,
                "stripe_rule_area_fraction_enabled": self.stripe_rule_area_fraction_enabled,
                "stripe_height_mm": self.stripe_height_mm,
                "stripe_height_tolerance_mm": self.stripe_height_tolerance_mm,
                "low_gradient_plateau_robust_band_mad_k": self.low_gradient_plateau_robust_band_mad_k,
                "low_gradient_plateau_detection_min_count_floor": self.low_gradient_plateau_detection_min_count_floor,
                "low_gradient_plateau_detection_min_count_fraction": self.low_gradient_plateau_detection_min_count_fraction,
                "blob_cluster_min_component_area": self.blob_cluster_min_component_area,
                "blob_cluster_height_gap_mm": self.blob_cluster_height_gap_mm,
                "blob_cluster_height_gap_mode": self.blob_cluster_height_gap_mode,
                "blob_cluster_min_area_fraction": self.blob_cluster_min_area_fraction,
                "blob_cluster_min_total_pixels": self.blob_cluster_min_total_pixels,
                "blob_cluster_max_z_mad_mm": self.blob_cluster_max_z_mad_mm,
                "blob_cluster_merge_close_clusters": self.blob_cluster_merge_close_clusters,
                "blob_cluster_merge_gap_mm": self.blob_cluster_merge_gap_mm,
                "blob_cluster_area_weight": self.blob_cluster_area_weight,
                "blob_cluster_border_weight": self.blob_cluster_border_weight,
                "blob_cluster_constancy_weight": self.blob_cluster_constancy_weight,
                "blob_cluster_spatial_coverage_weight": self.blob_cluster_spatial_coverage_weight,
                "blob_cluster_depth_preference_weight": self.blob_cluster_depth_preference_weight,
                "blob_cluster_fragmentation_penalty": self.blob_cluster_fragmentation_penalty,
                "blob_cluster_centrality_penalty": self.blob_cluster_centrality_penalty,
                "blob_cluster_object_like_penalty": self.blob_cluster_object_like_penalty,
                "blob_cluster_stripe_overlap_penalty": self.blob_cluster_stripe_overlap_penalty,
                "blob_cluster_refine_by_mad": self.blob_cluster_refine_by_mad,
                "blob_cluster_refine_mad_k": self.blob_cluster_refine_mad_k,
                "blob_cluster_refine_floor_mm": self.blob_cluster_refine_floor_mm,
                "blob_cluster_refine_keep_border_support": self.blob_cluster_refine_keep_border_support,
                "blob_cluster_fallback_strategy": self.blob_cluster_fallback_strategy,
                "low_gradient_surface_support_z_mad_multiplier": self.low_gradient_surface_support_z_mad_multiplier,
                "low_gradient_surface_support_z_floor_mm": self.low_gradient_surface_support_z_floor_mm,
                "low_gradient_surface_support_z_mad_floor_mm": self.low_gradient_surface_support_z_mad_floor_mm,
                "low_gradient_surface_ridge_percentile": self.low_gradient_surface_ridge_percentile,
                "reference_surface_height_gate_gap_floor_mm": self.reference_surface_height_gate_gap_floor_mm,
                "reference_surface_height_gate_gap_ratio": self.reference_surface_height_gate_gap_ratio,
                "plot_depth_plot_max_render_samples": self.plot_depth_plot_max_render_samples,
                "plot_y_robust_percentile": self.plot_y_robust_percentile,
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
            "selected_cluster_id": (gradient_debug.get("selected_cluster_id") if isinstance(gradient_debug, dict) else None),
            "selected_component_area_ratio": (gradient_debug.get("selected_component_area_ratio") if isinstance(gradient_debug, dict) else None),
            "selected_cluster_pixels": int(np.count_nonzero(selected_blob_cluster_mask)),
            "selected_reference_support_pixels": int(np.count_nonzero(selected_reference_support_mask)),
            "selected_surface_pixels": int(np.count_nonzero(selected_reference_support_mask)),
            "reference_model_support_pixels": int(np.count_nonzero(reference_model_support_mask)),
            "reference_suppression_pixels": int(np.count_nonzero(reference_suppression_mask)),
            "selected_surface_valid_roi_ratio": reference_support_lineage_summary.get("selected_reference_support_valid_roi_ratio"),
            "reference_model_support_valid_roi_ratio": reference_support_lineage_summary.get("reference_model_support_valid_roi_ratio"),
            "reference_suppression_valid_roi_ratio": reference_support_lineage_summary.get("reference_suppression_valid_roi_ratio"),
            "model_residual_mean_mm": float(np.mean(residual_abs)) if residual_abs.size else 0.0,
            "model_residual_p95_mm": float(np.percentile(residual_abs, 95)) if residual_abs.size else 0.0,
            "model_selection_reason": model_selection_reason,
            "selected_surface_alias_resolves_to": selected_surface_alias_resolves_to,
            "selected_surface_alias_role": "selected_support",
            "reference_support_lineage": reference_support_lineage_summary,
            "reference_method": resolved_reference_method,
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
                "belt_bg_mask": "belt_bg_mask.png",
                "belt_bg_components": "belt_bg_components.json",
                "low_gradient_blob_components_overlay": "low_gradient_blob_components_overlay.png",
                "low_gradient_blob_id_mask": "low_gradient_blob_id_mask.png",
                "low_gradient_blob_summary": "low_gradient_blob_summary.json",
                "height_aware_blob_components_overlay": "height_aware_blob_components_overlay.png",
                "height_aware_blob_id_mask": "height_aware_blob_id_mask.png",
                "height_aware_connectivity_rejected_edges": "height_aware_connectivity_rejected_edges.png",
                "height_aware_connectivity_debug": "height_aware_connectivity_debug.json",
                "component_formation_debug": "component_formation_debug.json",
                "height_gate_mask": "flat_candidate_mask.png",
                "height_border_strength": "height_border_strength.png",
                "height_border_cut_mask": "height_border_cut_mask.png",
                "height_border_detection_debug": "height_border_detection_debug.json",
                "height_border_fragments_overlay": "height_border_fragments_overlay.png",
                "height_border_fragments_mask": "height_border_fragments_mask.png",
                "height_border_fragments_id_mask": "height_border_fragments_id_mask.png",
                "height_border_split_debug": "height_border_split_debug.json",
                "fragment_merge_debug": "fragment_merge_debug.json",
                "height_split_blob_fragments_overlay": "height_split_blob_fragments_overlay.png",
                "height_split_blob_fragments_mask": "height_split_blob_fragments_mask.png",
                "height_split_blob_id_mask": "height_split_blob_id_mask.png",
                "height_split_debug": "height_split_debug.json",
                "height_consistent_blob_summary": "height_consistent_blob_summary.json",
                "blob_height_clusters": "blob_height_clusters.json",
                "blob_cluster_score_table": "blob_cluster_score_table.csv",
                "selected_blob_cluster_mask": "selected_blob_cluster_mask.png",
                "selected_blob_cluster_pre_refine_mask": "selected_blob_cluster_pre_refine_mask.png",
                "selected_blob_cluster_refined_mask": "selected_blob_cluster_refined_mask.png",
                "selected_blob_cluster_overlay": "selected_blob_cluster_overlay.png",
                "support_removed_by_candidate_refinement": "support_removed_by_candidate_refinement.png",
                "candidate_support_refinement_debug": "candidate_support_refinement_debug.json",
                "support_added_by_candidate_refinement": "support_added_by_candidate_refinement.png",
                "support_removed_by_candidate_refinement_overlay": "support_removed_by_candidate_refinement_overlay.png",
                "rejected_blob_clusters_mask": "rejected_blob_clusters_mask.png",
                "rejected_blob_clusters_overlay": "rejected_blob_clusters_overlay.png",
                "blob_cluster_selection_debug": "blob_cluster_selection_debug.json",
                "reference_surface_selected_mask": "reference_surface_selected_mask.png",
                "selected_reference_support_mask": "selected_reference_support_mask.png",
                "final_selected_support_mask": "selected_reference_support_mask.png",
                "stripe_filtered_reference_support_mask": "stripe_filtered_reference_support_mask.png",
                "support_removed_by_stripe_filter": "support_removed_by_stripe_filter.png",
                "support_added_by_stripe_filter": "support_added_by_stripe_filter.png",
                "support_removed_by_stripe_filter_overlay": "support_removed_by_stripe_filter_overlay.png",
                "flat_candidate_mask": "flat_candidate_mask.png",
                "background_selected_plateau_mask": "background_selected_plateau_mask.png",
                "rejected_raised_structure_mask": "rejected_raised_structure_mask.png",
                "rejected_raised_plateau_mask": "rejected_raised_plateau_mask.png",
                "reference_surface_candidates": "reference_surface_candidates.json",
                "reference_surface_plateaus": "reference_surface_plateaus.json",
                "flat_candidate_histogram": "flat_candidate_histogram.json",
                "background_plateau_plot": "background_plateau_plot.png",
                "flat_candidate_depth_plot": "flat_candidate_depth_plot.png",
                "belt_stripes_mask": "belt_stripes_mask.png",
                "belt_base_mask": "belt_base_mask.png",
                "belt_stripe_candidates_overlay": "belt_stripe_candidates_overlay.png",
                "belt_stripe_components": "belt_stripe_components.json",
                "unknown_low_gradient_mask": "unknown_low_gradient_mask.png",
                "surface_suppression_mask": "surface_suppression_mask.png",
                "object_search_domain_mask": "object_search_domain_mask.png",
                "belt_stripes_tophat_mask": "belt_stripes_tophat_mask.png",
                "belt_stripes_shape_mask": "belt_stripes_shape_mask.png",
                "belt_above_belt_mask": "belt_above_belt_mask.png",
                "belt_wide_object_mask": "belt_wide_object_mask.png",
                "belt_baseline_local_min": "belt_baseline_local_min.png",
                "belt_altitude_local_min": "belt_altitude_local_min.png",
                "belt_altitude_histogram_image": "belt_altitude_histogram.png",
                "belt_altitude_histogram": "belt_altitude_histogram.json",
                "belt_stripe_filter_debug": "belt_stripe_filter_debug.json",
                "gradient_debug": "gradient_debug.json",
                "selected_surface_debug": "selected_surface_debug.json",
                "background_selection_debug": "background_selection_debug.json",
                "background_seed_mask": "background_seed_mask.png",
                "expanded_plane_mask": "expanded_plane_mask.png",
                "final_plane_inlier_mask": "final_plane_inlier_mask.png",
                "plane_inlier_mask": "plane_inlier_mask.png",
                "reference_model_inlier_mask": "reference_model_inlier_mask.png",
                "reference_model_support_mask": "reference_model_support_mask.png",
                "final_support_debug": "final_support_debug.json",
                "support_removed_by_model_residual": "support_removed_by_model_residual.png",
                "support_added_by_model_expansion": "support_added_by_model_expansion.png",
                "support_removed_by_model_residual_overlay": "support_removed_by_model_residual_overlay.png",
                "reference_suppression_mask": "reference_suppression_mask.png",
                "support_removed_by_suppression_policy": "support_removed_by_suppression_policy.png",
                "support_added_by_suppression_policy": "support_added_by_suppression_policy.png",
                "support_removed_by_suppression_policy_overlay": "support_removed_by_suppression_policy_overlay.png",
                "selected_support_lineage": "selected_support_lineage.json",
                "support_loss_waterfall": "support_loss_waterfall.json",
                "plane_fit_debug": "plane_fit_debug.json",
                "reference_model": "belt_plane.json",
                "belt_plane_overlay": "belt_plane_overlay.png",
                "belt_plane_residuals": "belt_plane_residuals.png",
                "plane_residual_heatmap": "belt_plane_residuals.png",
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
            artifact_id="reference_model_inlier_mask",
            stage_id="detect_belt_plane",
            kind="image",
            title="Reference model inlier mask",
            path="reference_model_inlier_mask.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name, **background_selection_debug},
        )
        context.add_processing_artifact(
            artifact_id="reference_model_support_mask",
            stage_id="detect_belt_plane",
            kind="image",
            title="Reference model support mask",
            path="reference_model_support_mask.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name, **selected_surface_debug},
        )
        context.add_processing_artifact(
            artifact_id="support_removed_by_model_residual",
            stage_id="detect_belt_plane",
            kind="image",
            title="Support removed by model residual",
            path="support_removed_by_model_residual.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name, "lineage_step": "reference_model_support", "role": "removed_pixels"},
        )
        context.add_processing_artifact(
            artifact_id="support_added_by_model_expansion",
            stage_id="detect_belt_plane",
            kind="image",
            title="Support added by model expansion",
            path="support_added_by_model_expansion.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name, "lineage_step": "reference_model_support", "role": "added_pixels"},
        )
        context.add_processing_artifact(
            artifact_id="support_removed_by_model_residual_overlay",
            stage_id="detect_belt_plane",
            kind="image",
            title="Removed by model residual overlay",
            path="support_removed_by_model_residual_overlay.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name, "lineage_step": "reference_model_support", "role": "difference_overlay", "legend": "Green kept, red removed, blue added"},
        )
        context.add_processing_artifact(
            artifact_id="reference_suppression_mask",
            stage_id="detect_belt_plane",
            kind="image",
            title="Reference suppression mask",
            path="reference_suppression_mask.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name, **background_selection_debug},
        )
        context.add_processing_artifact(
            artifact_id="support_removed_by_suppression_policy",
            stage_id="detect_belt_plane",
            kind="image",
            title="Support removed by suppression policy",
            path="support_removed_by_suppression_policy.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name, "lineage_step": "suppression_mask", "role": "removed_pixels"},
        )
        context.add_processing_artifact(
            artifact_id="support_added_by_suppression_policy",
            stage_id="detect_belt_plane",
            kind="image",
            title="Support added by suppression policy",
            path="support_added_by_suppression_policy.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name, "lineage_step": "suppression_mask", "role": "added_pixels"},
        )
        context.add_processing_artifact(
            artifact_id="support_removed_by_suppression_policy_overlay",
            stage_id="detect_belt_plane",
            kind="image",
            title="Suppression policy difference overlay",
            path="support_removed_by_suppression_policy_overlay.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name, "lineage_step": "suppression_mask", "role": "difference_overlay", "legend": "Green kept, red removed, blue added"},
        )
        context.add_processing_artifact(
            artifact_id="selected_support_lineage",
            stage_id="detect_belt_plane",
            kind="json",
            title="Selected support lineage",
            path="selected_support_lineage.json",
            mime_type="application/json",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name, **selected_support_lineage},
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
            artifact_id="belt_bg_mask",
            stage_id="detect_belt_plane",
            kind="image",
            title="Belt background mask",
            path="belt_bg_mask.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name, "role": "fit_support"},
        )
        context.add_processing_artifact(
            artifact_id="belt_bg_components",
            stage_id="detect_belt_plane",
            kind="json",
            title="Belt background components",
            path="belt_bg_components.json",
            mime_type="application/json",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
        )
        context.add_processing_artifact(
            artifact_id="low_gradient_blob_components_overlay",
            stage_id="detect_belt_plane",
            kind="image",
            title="Low-gradient blob components overlay",
            path="low_gradient_blob_components_overlay.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
        )
        context.add_processing_artifact(
            artifact_id="low_gradient_blob_id_mask",
            stage_id="detect_belt_plane",
            kind="image",
            title="Low-gradient blob id mask",
            path="low_gradient_blob_id_mask.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
        )
        context.add_processing_artifact(
            artifact_id="low_gradient_blob_summary",
            stage_id="detect_belt_plane",
            kind="json",
            title="Low-gradient blob summary",
            path="low_gradient_blob_summary.json",
            mime_type="application/json",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
        )
        context.add_processing_artifact(
            artifact_id="height_aware_blob_components_overlay",
            stage_id="detect_belt_plane",
            kind="image",
            title="Height-aware blob components overlay",
            path="height_aware_blob_components_overlay.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
        )
        context.add_processing_artifact(
            artifact_id="height_aware_blob_id_mask",
            stage_id="detect_belt_plane",
            kind="image",
            title="Height-aware blob id mask",
            path="height_aware_blob_id_mask.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
        )
        context.add_processing_artifact(
            artifact_id="height_aware_connectivity_rejected_edges",
            stage_id="detect_belt_plane",
            kind="image",
            title="Height-aware connectivity rejected edges",
            path="height_aware_connectivity_rejected_edges.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
        )
        context.add_processing_artifact(
            artifact_id="height_aware_connectivity_debug",
            stage_id="detect_belt_plane",
            kind="json",
            title="Height-aware connectivity debug",
            path="height_aware_connectivity_debug.json",
            mime_type="application/json",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
        )
        context.add_processing_artifact(
            artifact_id="height_border_strength",
            stage_id="detect_belt_plane",
            kind="image",
            title="Height-border strength",
            path="height_border_strength.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
        )
        context.add_processing_artifact(
            artifact_id="height_border_cut_mask",
            stage_id="detect_belt_plane",
            kind="image",
            title="Height-border cut mask",
            path="height_border_cut_mask.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
        )
        context.add_processing_artifact(
            artifact_id="height_border_fragments_overlay",
            stage_id="detect_belt_plane",
            kind="image",
            title="Height-border fragments overlay",
            path="height_border_fragments_overlay.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
        )
        context.add_processing_artifact(
            artifact_id="height_border_fragments_mask",
            stage_id="detect_belt_plane",
            kind="image",
            title="Height-border fragments mask",
            path="height_border_fragments_mask.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
        )
        context.add_processing_artifact(
            artifact_id="height_border_fragments_id_mask",
            stage_id="detect_belt_plane",
            kind="image",
            title="Height-border fragments id mask",
            path="height_border_fragments_id_mask.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
        )
        context.add_processing_artifact(
            artifact_id="height_border_split_debug",
            stage_id="detect_belt_plane",
            kind="json",
            title="Height-border split debug",
            path="height_border_split_debug.json",
            mime_type="application/json",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
        )
        context.add_processing_artifact(
            artifact_id="fragment_merge_debug",
            stage_id="detect_belt_plane",
            kind="json",
            title="Fragment merge debug",
            path="fragment_merge_debug.json",
            mime_type="application/json",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
        )
        context.add_processing_artifact(
            artifact_id="height_split_blob_fragments_overlay",
            stage_id="detect_belt_plane",
            kind="image",
            title="Height-split blob fragments overlay",
            path="height_split_blob_fragments_overlay.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
        )
        context.add_processing_artifact(
            artifact_id="height_split_blob_fragments_mask",
            stage_id="detect_belt_plane",
            kind="image",
            title="Height-split blob fragments mask",
            path="height_split_blob_fragments_mask.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
        )
        context.add_processing_artifact(
            artifact_id="height_split_blob_id_mask",
            stage_id="detect_belt_plane",
            kind="image",
            title="Height-split blob id mask",
            path="height_split_blob_id_mask.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
        )
        context.add_processing_artifact(
            artifact_id="height_split_debug",
            stage_id="detect_belt_plane",
            kind="json",
            title="Height split debug",
            path="height_split_debug.json",
            mime_type="application/json",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
        )
        context.add_processing_artifact(
            artifact_id="height_consistent_blob_summary",
            stage_id="detect_belt_plane",
            kind="json",
            title="Height-consistent blob summary",
            path="height_consistent_blob_summary.json",
            mime_type="application/json",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
        )
        context.add_processing_artifact(
            artifact_id="blob_height_clusters",
            stage_id="detect_belt_plane",
            kind="json",
            title="Blob height clusters",
            path="blob_height_clusters.json",
            mime_type="application/json",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
        )
        context.add_processing_artifact(
            artifact_id="blob_cluster_score_table",
            stage_id="detect_belt_plane",
            kind="table",
            title="Blob cluster score table",
            path="blob_cluster_score_table.csv",
            mime_type="text/csv",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
        )
        context.add_processing_artifact(
            artifact_id="selected_blob_cluster_mask",
            stage_id="detect_belt_plane",
            kind="image",
            title="Selected blob cluster mask",
            path="selected_blob_cluster_mask.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
        )
        context.add_processing_artifact(
            artifact_id="selected_blob_cluster_pre_refine_mask",
            stage_id="detect_belt_plane",
            kind="image",
            title="Selected blob cluster pre-refine mask",
            path="selected_blob_cluster_pre_refine_mask.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
        )
        context.add_processing_artifact(
            artifact_id="selected_blob_cluster_refined_mask",
            stage_id="detect_belt_plane",
            kind="image",
            title="Selected blob cluster refined mask",
            path="selected_blob_cluster_refined_mask.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
        )
        context.add_processing_artifact(
            artifact_id="support_removed_by_candidate_refinement",
            stage_id="detect_belt_plane",
            kind="image",
            title="Support removed by candidate refinement",
            path="support_removed_by_candidate_refinement.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name, "lineage_step": "candidate_refinement", "role": "removed_pixels"},
        )
        context.add_processing_artifact(
            artifact_id="support_added_by_candidate_refinement",
            stage_id="detect_belt_plane",
            kind="image",
            title="Support added by candidate refinement",
            path="support_added_by_candidate_refinement.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name, "lineage_step": "candidate_refinement", "role": "added_pixels"},
        )
        context.add_processing_artifact(
            artifact_id="support_removed_by_candidate_refinement_overlay",
            stage_id="detect_belt_plane",
            kind="image",
            title="Removed by candidate refinement overlay",
            path="support_removed_by_candidate_refinement_overlay.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name, "lineage_step": "candidate_refinement", "role": "difference_overlay", "legend": "Green kept, red removed, blue added"},
        )
        context.add_processing_artifact(
            artifact_id="selected_blob_cluster_overlay",
            stage_id="detect_belt_plane",
            kind="image",
            title="Selected blob cluster overlay",
            path="selected_blob_cluster_overlay.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
        )
        context.add_processing_artifact(
            artifact_id="rejected_blob_clusters_mask",
            stage_id="detect_belt_plane",
            kind="image",
            title="Rejected blob clusters mask",
            path="rejected_blob_clusters_mask.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
        )
        context.add_processing_artifact(
            artifact_id="rejected_blob_clusters_overlay",
            stage_id="detect_belt_plane",
            kind="image",
            title="Rejected blob clusters overlay",
            path="rejected_blob_clusters_overlay.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
        )
        context.add_processing_artifact(
            artifact_id="blob_cluster_selection_debug",
            stage_id="detect_belt_plane",
            kind="json",
            title="Blob cluster selection debug",
            path="blob_cluster_selection_debug.json",
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
            metadata={"source": "native_pipeline", "producer": self.name, **selected_surface_debug},
        )
        context.add_processing_artifact(
            artifact_id="selected_reference_support_mask",
            stage_id="detect_belt_plane",
            kind="image",
            title="Selected reference support mask",
            path="selected_reference_support_mask.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name, **selected_surface_debug},
        )
        context.add_processing_artifact(
            artifact_id="stripe_filtered_reference_support_mask",
            stage_id="detect_belt_plane",
            kind="image",
            title="Stripe-filtered reference support mask",
            path="stripe_filtered_reference_support_mask.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name, **selected_surface_debug},
        )
        context.add_processing_artifact(
            artifact_id="support_removed_by_stripe_filter",
            stage_id="detect_belt_plane",
            kind="image",
            title="Support removed by stripe filter",
            path="support_removed_by_stripe_filter.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name, "lineage_step": "stripe_suppression", "role": "removed_pixels"},
        )
        context.add_processing_artifact(
            artifact_id="support_added_by_stripe_filter",
            stage_id="detect_belt_plane",
            kind="image",
            title="Support added by stripe filter",
            path="support_added_by_stripe_filter.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name, "lineage_step": "stripe_suppression", "role": "added_pixels"},
        )
        context.add_processing_artifact(
            artifact_id="support_removed_by_stripe_filter_overlay",
            stage_id="detect_belt_plane",
            kind="image",
            title="Removed by stripe filter overlay",
            path="support_removed_by_stripe_filter_overlay.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name, "lineage_step": "stripe_suppression", "role": "difference_overlay", "legend": "Green kept, red removed, blue added"},
        )
        context.add_processing_artifact(
            artifact_id="flat_candidate_mask",
            stage_id="detect_belt_plane",
            kind="image",
            title="Flat-candidate mask (low-gradient ± low-Hessian)",
            path="flat_candidate_mask.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
        )
        context.add_processing_artifact(
            artifact_id="background_selected_plateau_mask",
            stage_id="detect_belt_plane",
            kind="image",
            title="Selected background plateau mask",
            path="background_selected_plateau_mask.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
        )
        context.add_processing_artifact(
            artifact_id="rejected_raised_plateau_mask",
            stage_id="detect_belt_plane",
            kind="image",
            title="Rejected raised plateau mask",
            path="rejected_raised_plateau_mask.png",
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
            artifact_id="reference_surface_plateaus",
            stage_id="detect_belt_plane",
            kind="json",
            title="Reference-surface plateaus",
            path="reference_surface_plateaus.json",
            mime_type="application/json",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
        )
        context.add_processing_artifact(
            artifact_id="flat_candidate_histogram",
            stage_id="detect_belt_plane",
            kind="json",
            title="Flat-candidate histogram",
            path="flat_candidate_histogram.json",
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
            artifact_id="background_plateau_plot",
            stage_id="detect_belt_plane",
            kind="image",
            title="Background plateau plot",
            path="background_plateau_plot.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
        )
        context.add_processing_artifact(
            artifact_id="flat_candidate_depth_plot",
            stage_id="detect_belt_plane",
            kind="image",
            title="Filtered depth plot (low-gradient candidates)",
            path="flat_candidate_depth_plot.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
        )
        context.add_processing_artifact(
            artifact_id="belt_stripes_mask",
            stage_id="detect_belt_plane",
            kind="image",
            title="Belt stripes mask (surface texture)",
            path="belt_stripes_mask.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name, "role": "background_ignore_mask"},
        )
        context.add_processing_artifact(
            artifact_id="belt_base_mask",
            stage_id="detect_belt_plane",
            kind="image",
            title="Belt base mask (plane-fit support)",
            path="belt_base_mask.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name, "role": "belt_support"},
        )
        context.add_processing_artifact(
            artifact_id="belt_stripe_candidates_overlay",
            stage_id="detect_belt_plane",
            kind="image",
            title="Belt stripe candidate components",
            path="belt_stripe_candidates_overlay.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name, "role": "diagnostic"},
        )
        context.add_processing_artifact(
            artifact_id="belt_stripe_components",
            stage_id="detect_belt_plane",
            kind="json",
            title="Belt stripe components",
            path="belt_stripe_components.json",
            mime_type="application/json",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
        )
        context.add_processing_artifact(
            artifact_id="unknown_low_gradient_mask",
            stage_id="detect_belt_plane",
            kind="image",
            title="Unknown low-gradient mask",
            path="unknown_low_gradient_mask.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name, "role": "diagnostic"},
        )
        context.add_processing_artifact(
            artifact_id="surface_suppression_mask",
            stage_id="detect_belt_plane",
            kind="image",
            title="Surface suppression mask",
            path="surface_suppression_mask.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name, "role": "suppression_mask"},
        )
        context.add_processing_artifact(
            artifact_id="object_search_domain_mask",
            stage_id="detect_belt_plane",
            kind="image",
            title="Object search domain mask",
            path="object_search_domain_mask.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name, "role": "diagnostic"},
        )
        context.add_processing_artifact(
            artifact_id="belt_stripes_tophat_mask",
            stage_id="detect_belt_plane",
            kind="image",
            title="Belt stripes — top-hat pass only",
            path="belt_stripes_tophat_mask.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name, "role": "diagnostic"},
        )
        context.add_processing_artifact(
            artifact_id="belt_stripes_shape_mask",
            stage_id="detect_belt_plane",
            kind="image",
            title="Belt stripes — shape pass (z-floor) only",
            path="belt_stripes_shape_mask.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name, "role": "diagnostic"},
        )
        context.add_processing_artifact(
            artifact_id="belt_above_belt_mask",
            stage_id="detect_belt_plane",
            kind="image",
            title="Above-belt mask (z > belt_z + margin)",
            path="belt_above_belt_mask.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name, "role": "diagnostic"},
        )
        context.add_processing_artifact(
            artifact_id="belt_wide_object_mask",
            stage_id="detect_belt_plane",
            kind="image",
            title="Wide-object mask (object-sized binary opening)",
            path="belt_wide_object_mask.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name, "role": "diagnostic"},
        )
        context.add_processing_artifact(
            artifact_id="belt_baseline_local_min",
            stage_id="detect_belt_plane",
            kind="image",
            title="Belt local-min baseline (heightmap)",
            path="belt_baseline_local_min.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
        )
        context.add_processing_artifact(
            artifact_id="belt_altitude_local_min",
            stage_id="detect_belt_plane",
            kind="image",
            title="Belt altitude over local-min (top-hat)",
            path="belt_altitude_local_min.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
        )
        context.add_processing_artifact(
            artifact_id="belt_altitude_histogram_image",
            stage_id="detect_belt_plane",
            kind="image",
            title="Belt-stripe altitude histogram",
            path="belt_altitude_histogram.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
        )
        context.add_processing_artifact(
            artifact_id="belt_altitude_histogram",
            stage_id="detect_belt_plane",
            kind="json",
            title="Belt-stripe altitude histogram",
            path="belt_altitude_histogram.json",
            mime_type="application/json",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
        )
        context.add_processing_artifact(
            artifact_id="belt_stripe_filter_debug",
            stage_id="detect_belt_plane",
            kind="json",
            title="Belt-stripe filter debug",
            path="belt_stripe_filter_debug.json",
            mime_type="application/json",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name},
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
        context.add_processing_artifact(
            artifact_id="component_formation_debug",
            stage_id="detect_belt_plane",
            kind="json",
            title="Component formation debug",
            path="component_formation_debug.json",
            mime_type="application/json",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name, **component_formation_debug},
        )
        context.add_processing_artifact(
            artifact_id="height_border_detection_debug",
            stage_id="detect_belt_plane",
            kind="json",
            title="Height-border detection debug",
            path="height_border_detection_debug.json",
            mime_type="application/json",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name, **height_border_detection_debug},
        )
        context.add_processing_artifact(
            artifact_id="candidate_support_refinement_debug",
            stage_id="detect_belt_plane",
            kind="json",
            title="Candidate support refinement debug",
            path="candidate_support_refinement_debug.json",
            mime_type="application/json",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name, **candidate_support_refinement_debug},
        )
        context.add_processing_artifact(
            artifact_id="final_support_debug",
            stage_id="detect_belt_plane",
            kind="json",
            title="Final support debug",
            path="final_support_debug.json",
            mime_type="application/json",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name, **final_support_debug},
        )
        context.add_processing_artifact(
            artifact_id="support_loss_waterfall",
            stage_id="detect_belt_plane",
            kind="table",
            title="Support-loss waterfall",
            path="support_loss_waterfall.json",
            mime_type="application/json",
            preview_available=True,
            metadata={
                "source": "native_pipeline",
                "producer": self.name,
                "rows": support_loss_waterfall,
                "compact_summary": _make_debug_compact_summary(
                    title="Support-loss waterfall",
                    input_count=int(np.count_nonzero(valid_for_fit)),
                    output_count=int(np.count_nonzero(final_plane_inlier_mask)),
                    selected_id=str(selected_cluster.get("cluster_id")) if isinstance(selected_cluster, dict) else None,
                    selected_reason=largest_loss_step,
                    fallback_used=bool(fallback_used),
                    fallback_reason=fallback_reason,
                    warnings=warnings,
                    key_parameters={"reference_surface_model": str(self.reference_surface_model)},
                    affected_downstream_artifacts=["selected_reference_support_mask", "reference_model_support_mask", "reference_suppression_mask", "final_plane_inlier_mask"],
                    extra={"largest_loss_step": largest_loss_step, "largest_loss_fraction": largest_loss_fraction},
                ),
            },
        )
        context.add_processing_artifact(
            artifact_id="final_selected_support_mask",
            stage_id="detect_belt_plane",
            kind="image",
            title="Final selected support mask",
            path="selected_reference_support_mask.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name, **selected_surface_debug},
        )
        context.add_processing_artifact(
            artifact_id="reference_model",
            stage_id="detect_belt_plane",
            kind="json",
            title="Reference model",
            path="belt_plane.json",
            mime_type="application/json",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name, **plane_json},
        )
        context.add_processing_artifact(
            artifact_id="plane_residual_heatmap",
            stage_id="detect_belt_plane",
            kind="image",
            title="Plane residual heatmap",
            path="belt_plane_residuals.png",
            mime_type="image/png",
            preview_available=True,
            metadata={"source": "native_pipeline", "producer": self.name, **stats},
        )
        _enrich_detect_reference_processing_artifacts(context)


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
        (output_dir / "diagnostics_normalized_height_histogram.json").write_text(json.dumps(hist_payload, indent=2), encoding="utf-8")
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

        _runtime_trace_mask_unit(
            context,
            unit_id="normalize_heights_to_plane.reference_model_input",
            stage_id="normalize_heights_to_plane",
            parent_id="normalize_heights_to_plane",
            parameters={"normalized_height_sign": str(self.normalized_height_sign)},
            input_artifacts=["belt_plane", "raw_heightmap_preview"],
            output_artifacts=["belt_plane"],
            metrics={
                "valid_pixels": int(valid_values.size),
                "model_type_constant_z": bool(model_type == "constant_z"),
            },
            diagnostics={"model_type": model_type, "normalization_convention": convention},
        )
        _runtime_trace_mask_unit(
            context,
            unit_id="normalize_heights_to_plane.plane_signed_distance",
            stage_id="normalize_heights_to_plane",
            parent_id="normalize_heights_to_plane",
            parameters={
                "normalized_clip_negative": bool(self.normalized_clip_negative),
                "normalized_negative_tolerance_mm": float(self.normalized_negative_tolerance_mm),
            },
            input_artifacts=["belt_plane", "raw_heightmap_preview"],
            output_artifacts=["belt_plane_residuals"],
            metrics={
                "signed_distance_min_mm": float(norm_min_mm),
                "signed_distance_max_mm": float(norm_max_mm),
                "signed_distance_mean_mm": float(hist_payload["mean_mm"]),
                "invalid_pixel_count": int(frame.valid_mask.size - np.count_nonzero(frame.valid_mask)),
            },
            diagnostics={"normalization_convention": convention, "model_type": model_type},
            warnings=norm_warnings,
        )
        _runtime_trace_mask_unit(
            context,
            unit_id="normalize_heights_to_plane.height_above_belt",
            stage_id="normalize_heights_to_plane",
            parent_id="normalize_heights_to_plane",
            output_artifacts=["normalized_heightmap", "height_above_belt_raster"],
            metrics={
                "valid_pixels": int(valid_values.size),
                "height_min_mm": float(norm_min_mm),
                "height_max_mm": float(norm_max_mm),
                "height_mean_mm": float(hist_payload["mean_mm"]),
                "background_p95_abs_mm": float(norm_debug["background_height_p95_abs_after_normalization_mm"]),
            },
        )
        _runtime_trace_mask_unit(
            context,
            unit_id="normalize_heights_to_plane.normalized_display_preview",
            stage_id="normalize_heights_to_plane",
            parent_id="normalize_heights_to_plane",
            input_artifacts=["normalized_heightmap"],
            output_artifacts=["below_reference_mask", "above_threshold_mask", "normalized_heightmap_display", "normalized_heightmap_render_context"],
            metrics={
                "below_reference_pixels": int(np.count_nonzero(below_reference_mask)),
                "above_threshold_pixels": int(np.count_nonzero(above_threshold_mask)),
                "display_vmin_mm": float(display_vmin),
                "display_vmax_mm": float(display_vmax),
            },
        )
        _runtime_trace_mask_unit(
            context,
            unit_id="normalize_heights_to_plane.normalization_diagnostics",
            stage_id="normalize_heights_to_plane",
            parent_id="normalize_heights_to_plane",
            output_artifacts=["normalized_height_histogram", "normalization_debug"],
            metrics={
                "histogram_bin_count": int(len(hist_payload.get("bin_counts") or [])),
                "sample_count": int(hist_payload["sample_count"]),
                "valid_pixel_ratio": float(norm_debug["valid_ratio"]),
            },
            diagnostics={"normalization_quality": norm_debug.get("normalization_quality"), "warnings": norm_warnings},
            warnings=norm_warnings,
        )
        _runtime_trace_mask_unit(
            context,
            unit_id="normalize_heights_to_plane",
            stage_id="normalize_heights_to_plane",
            parameters={
                "normalized_clip_negative": bool(self.normalized_clip_negative),
                "normalized_negative_tolerance_mm": float(self.normalized_negative_tolerance_mm),
                "below_reference_tolerance_mm": float(self.below_reference_tolerance_mm),
            },
            input_artifacts=["belt_plane", "raw_heightmap_preview"],
            output_artifacts=["normalized_heightmap", "normalized_height_histogram", "normalization_debug"],
            metrics={
                "valid_pixels": int(valid_values.size),
                "invalid_pixels": int(frame.valid_mask.size - np.count_nonzero(frame.valid_mask)),
                "height_min_mm": float(norm_min_mm),
                "height_max_mm": float(norm_max_mm),
                "height_mean_mm": float(hist_payload["mean_mm"]),
            },
            diagnostics={"model_type": model_type, "normalization_convention": convention},
            warnings=norm_warnings,
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

        surface_suppression = np.asarray(
            context.get_artifact(
                "surface_suppression_mask_array",
                context.get_artifact("surface_suppression_mask", np.zeros_like(frame.valid_mask, dtype=bool)),
            ),
            dtype=bool,
        )
        plane_mask = np.asarray(
            context.get_artifact(
                "reference_suppression_mask",
                context.get_artifact(
                    "reference_surface_selected_mask",
                    context.get_artifact("expanded_plane_mask", context.get_artifact("plane_inlier_mask", np.zeros_like(frame.valid_mask, dtype=bool))),
                ),
            ),
            dtype=bool,
        )
        if np.any(surface_suppression):
            plane_mask = surface_suppression
        suppressed_plane_pixels = foreground & plane_mask
        if self.suppress_plane_mask_in_segmentation:
            foreground = foreground & (~plane_mask)
        # Belt stripes (surface texture from chevron ribs etc.) are coplanar
        # with the belt but ride a few mm above the local baseline; without
        # this suppression they show up as bright bands in the normalized
        # heightmap and get classified as object candidates. The mask is
        # produced by the reference-surface stage's belt-stripe filter and
        # tagged role=background_ignore_mask.
        stripes_artifact = context.get_artifact("belt_stripes_mask_array")
        if stripes_artifact is None:
            stripes_artifact = context.get_artifact("belt_stripes_mask")
        if isinstance(stripes_artifact, np.ndarray):
            stripes_bool = np.asarray(stripes_artifact, dtype=bool)
        else:
            stripes_bool = np.zeros_like(frame.valid_mask, dtype=bool)
        suppressed_stripe_pixels = foreground_before_suppression & stripes_bool
        if np.any(stripes_bool):
            foreground = foreground & (~stripes_bool)
        cv2.imwrite(str(output_dir / "plane_suppressed_mask.png"), (foreground.astype(np.uint8) * 255))
        cv2.imwrite(str(output_dir / "rejected_background_residuals.png"), (suppressed_plane_pixels.astype(np.uint8) * 255))
        cv2.imwrite(str(output_dir / "rejected_belt_stripes.png"), (suppressed_stripe_pixels.astype(np.uint8) * 255))

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
            "suppression_source": "surface_suppression_mask" if np.any(surface_suppression) else "reference_suppression_mask",
            "surface_suppression_pixel_count": int(np.count_nonzero(plane_mask)),
            "unknown_low_gradient_suppressed_pixel_count": int(np.count_nonzero(plane_mask & np.asarray(context.get_artifact("unknown_low_gradient_mask_array", np.zeros_like(frame.valid_mask, dtype=bool)), dtype=bool))),
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
        _runtime_trace_mask_unit(
            context,
            unit_id="remove_belt_segment_objects.foreground_thresholding",
            stage_id="remove_belt_segment_objects",
            parent_id="remove_belt_segment_objects",
            parameters={"min_height_mm": float(self.min_height_mm), "max_height_mm": self.max_height_mm},
            input_artifacts=["normalized_heightmap"],
            output_artifacts=["foreground_before_plane_suppression", "above_threshold_mask"],
            metrics={"threshold_active_pixels": int(np.count_nonzero(foreground_before_suppression)), "below_reference_pixels": int(np.count_nonzero(below_or_on_reference))},
        )
        _runtime_trace_mask_unit(
            context,
            unit_id="remove_belt_segment_objects.reference_suppression",
            stage_id="remove_belt_segment_objects",
            parent_id="remove_belt_segment_objects",
            status="skipped" if not bool(self.suppress_plane_mask_in_segmentation) else "completed",
            parameters={
                "suppress_plane_mask_in_segmentation": bool(self.suppress_plane_mask_in_segmentation),
                "reference_tolerance_mm": float(self.reference_tolerance_mm),
            },
            input_artifacts=["above_threshold_mask", "reference_suppression_mask"],
            output_artifacts=["plane_suppressed_mask", "rejected_background_residuals", "rejected_belt_stripes"],
            metrics={
                "suppressed_plane_pixels": int(np.count_nonzero(suppressed_plane_pixels)),
                "suppressed_stripe_pixels": int(np.count_nonzero(suppressed_stripe_pixels)),
                "remaining_foreground_pixels": int(np.count_nonzero(foreground)),
            },
            warnings=seg_warnings,
        )
        _runtime_trace_mask_unit(
            context,
            unit_id="remove_belt_segment_objects.stripe_suppression_consumption",
            stage_id="remove_belt_segment_objects",
            parent_id="remove_belt_segment_objects",
            status="skipped" if int(np.count_nonzero(suppressed_stripe_pixels)) == 0 else "completed",
            input_artifacts=["rejected_belt_stripes", "reference_suppression_mask"],
            output_artifacts=["rejected_belt_stripes", "plane_suppressed_mask"],
            metrics={
                "stripe_rejected_pixels": int(np.count_nonzero(suppressed_stripe_pixels)),
                "remaining_foreground_pixels": int(np.count_nonzero(foreground)),
            },
            diagnostics={"suppression_source": seg_debug.get("suppression_source")},
        )
        _runtime_trace_mask_unit(
            context,
            unit_id="remove_belt_segment_objects.morphology_cleanup",
            stage_id="remove_belt_segment_objects",
            parent_id="remove_belt_segment_objects",
            parameters={"morphology_kernel": int(self.morphology_kernel), "fill_holes": bool(self.fill_holes), "smoothing_kernel": int(self.smoothing_kernel)},
            input_artifacts=["normalized_height_threshold_mask"],
            output_artifacts=["cleaned_object_mask"],
            metrics={"cleaned_mask_pixels": int(np.count_nonzero(cleaned > 0))},
            warnings=seg_warnings,
        )
        _runtime_trace_mask_unit(
            context,
            unit_id="remove_belt_segment_objects.connected_component_preparation",
            stage_id="remove_belt_segment_objects",
            parent_id="remove_belt_segment_objects",
            parameters={"min_component_area": int(self.min_component_area)},
            input_artifacts=["cleaned_object_mask"],
            output_artifacts=["final_object_mask", "connected_components_overlay", "segmentation_debug"],
            metrics={
                "final_object_pixels": int(np.count_nonzero(filtered > 0)),
                "component_count": int(max(num_labels - 1, 0)),
            },
            warnings=seg_warnings,
        )
        _runtime_trace_mask_unit(
            context,
            unit_id="remove_belt_segment_objects",
            stage_id="remove_belt_segment_objects",
            parameters={
                "min_height_mm": float(self.min_height_mm),
                "max_height_mm": self.max_height_mm,
                "suppress_plane_mask_in_segmentation": bool(self.suppress_plane_mask_in_segmentation),
                "morphology_kernel": int(self.morphology_kernel),
                "min_component_area": int(self.min_component_area),
            },
            input_artifacts=["normalized_heightmap", "reference_suppression_mask"],
            output_artifacts=["final_object_mask", "connected_components_overlay", "segmentation_debug"],
            metrics={
                "threshold_active_pixels": int(np.count_nonzero(foreground_before_suppression)),
                "plane_suppressed_pixels": int(np.count_nonzero(suppressed_plane_pixels)),
                "stripe_rejected_pixels": int(np.count_nonzero(suppressed_stripe_pixels)),
                "cleaned_mask_pixels": int(np.count_nonzero(cleaned > 0)),
                "final_object_pixels": int(np.count_nonzero(filtered > 0)),
                "components_before_cleanup": int(max(num_labels_before - 1, 0)),
                "components_after_suppression": int(max(num_labels_after_suppress - 1, 0)),
                "components_after_cleanup": int(max(num_labels - 1, 0)),
            },
            diagnostics={"suppression_source": seg_debug.get("suppression_source")},
            warnings=seg_warnings,
        )
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
                "rejected_belt_stripes": "rejected_belt_stripes.png",
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
            artifact_id="rejected_belt_stripes",
            stage_id="remove_belt_segment_objects",
            kind="image",
            title="Rejected belt stripes",
            path="rejected_belt_stripes.png",
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
        # Backward-compatible alias kept for legacy consumers/tests that still
        # look up the segmentation overlay under the old artifact id.
        context.add_processing_artifact(
            artifact_id="height_segmentation_overlay",
            stage_id="remove_belt_segment_objects",
            kind="overlay",
            title="Height segmentation overlay (compat)",
            path="connected_components_overlay.png",
            mime_type="image/png",
            preview_available=True,
            overlay_type="segmentation",
            coordinate_space="image_pixel",
            target_artifact_id="heightmap_preview",
            source_artifact_ids=["height_segmentation", "cleaned_object_mask", "heightmap_preview"],
            metadata=_display_only_meta({"source": "native_pipeline", "producer": self.name, "semantic": "compat_alias", "numeric_source_artifact": "normalized_heightmap.npz"}),
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
        area_values = [int(item.get("area_px") or 0) for item in components if int(item.get("area_px") or 0) > 0]
        area_summary = _numeric_summary(area_values)
        _runtime_trace_mask_unit(
            context,
            unit_id="geometry.connected_component_extraction",
            stage_id="geometry",
            parent_id="geometry",
            input_artifacts=["final_object_mask"],
            output_artifacts=["connected_components"],
            metrics={
                "component_count": int(len(components)),
                "accepted_object_count": int(len(components)),
                "area_min_px": area_summary.get("min"),
                "area_max_px": area_summary.get("max"),
                "area_mean_px": area_summary.get("mean"),
            },
            diagnostics={"object_ids": [int(item.get("object_id") or 0) for item in components]},
        )
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
        contour_count = sum(1 for comp in components if comp.get("contour") is not None)
        hull_count = sum(1 for comp in components if comp.get("convex_hull") is not None)
        ellipse_success_count = sum(1 for comp in components if comp.get("ellipse") is not None)
        ellipse_failure_count = max(0, len(components) - ellipse_success_count)
        footprint_summary = _numeric_summary(
            [float(comp.get("footprint_area_mm2") or 0.0) for comp in components if float(comp.get("footprint_area_mm2") or 0.0) > 0.0]
        )
        _runtime_trace_mask_unit(
            context,
            unit_id="geometry.contour_extraction",
            stage_id="geometry",
            parent_id="geometry",
            input_artifacts=["connected_components"],
            output_artifacts=["contour"],
            metrics={"component_count": int(len(components)), "contour_count": int(contour_count)},
        )
        _runtime_trace_mask_unit(
            context,
            unit_id="geometry.hull_fitting",
            stage_id="geometry",
            parent_id="geometry",
            input_artifacts=["contour"],
            output_artifacts=["convex_hull"],
            metrics={"hull_count": int(hull_count), "component_count": int(len(components))},
        )
        _runtime_trace_mask_unit(
            context,
            unit_id="geometry.ellipse_fitting",
            stage_id="geometry",
            parent_id="geometry",
            input_artifacts=["contour"],
            output_artifacts=["fitted_ellipse"],
            metrics={
                "ellipse_fit_success_count": int(ellipse_success_count),
                "ellipse_fit_failure_count": int(ellipse_failure_count),
            },
        )
        _runtime_trace_mask_unit(
            context,
            unit_id="geometry",
            stage_id="geometry",
            input_artifacts=["connected_components"],
            output_artifacts=["connected_components", "contour", "convex_hull", "fitted_ellipse"],
            metrics={
                "component_count": int(len(components)),
                "accepted_object_count": int(len(components)),
                "footprint_area_min_mm2": footprint_summary.get("min"),
                "footprint_area_max_mm2": footprint_summary.get("max"),
                "footprint_area_mean_mm2": footprint_summary.get("mean"),
            },
            diagnostics={"object_ids": [int(comp.get("object_id") or 0) for comp in components]},
        )


@dataclass
class ComputeHeightMetricsStage:
    name: str = "ComputeHeightMetrics"
    category: str = "measurement"
    required_modalities: tuple[CaptureModality, ...] = ("heightmap",)
    produced_modalities: tuple[CaptureModality, ...] = ()
    produced_artifacts: tuple[str, ...] = ("objects",)
    stage_category: str | None = "measurement"
    height_percentile: float = 99.0
    inner_kernel_size: int = 5
    smoothing_kernel_size: int = 7
    flat_gradient_threshold: float = 0.08

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
            inner_k = max(1, int(self.inner_kernel_size) | 1)
            smooth_k = max(1, int(self.smoothing_kernel_size) | 1)
            inner_mask = cv2.erode(mask.astype(np.uint8), np.ones((inner_k, inner_k), dtype=np.uint8), iterations=1).astype(bool)
            if int(np.count_nonzero(inner_mask)) < max(25, int(0.15 * np.count_nonzero(mask))):
                inner_mask = mask
            smoothed_surface = cv2.GaussianBlur(np.where(mask, normalized, 0.0).astype(np.float32), (smooth_k, smooth_k), 0)
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
            # dim_z uses P99 instead of the absolute max so a few noise peaks (whether
            # on the known calibration cube or on regular objects) do not inflate the
            # canonical Z extent. The true maximum is preserved separately under
            # `height_above_belt_mm.max_height_mm` for diagnostics.
            p99_height = float(np.percentile(values, min(100.0, max(50.0, float(self.height_percentile))))) if values.size else max_height
            dim_z_height = p99_height
            footprint_x = float(comp.get("major_axis_mm") or comp["bbox"][2] * frame.x_resolution_mm)
            footprint_y = float(comp.get("minor_axis_mm") or comp["bbox"][3] * frame.y_resolution_mm)
            sphere_radius_mm = float(max(footprint_x, footprint_y, dim_z_height) / 2.0)
            sphere_diameter_mm = float(sphere_radius_mm * 2.0)
            truncation_ratio = float(max(0.0, 1.0 - (dim_z_height / max(1e-6, sphere_diameter_mm))))
            contour_mm_metrics = _contour_metrics_in_metric_space(
                comp["contour"].reshape(-1, 2).astype(float).tolist()
                if comp.get("contour") is not None and len(comp.get("contour")) > 0
                else None,
                x_resolution_mm=float(frame.x_resolution_mm),
                y_resolution_mm=float(frame.y_resolution_mm),
                scale_x=1.0,
                scale_y=1.0,
            )
            radial_uniformity = _radial_boundary_uniformity_from_contour_mm(contour_mm_metrics.get("contour_mm"))
            pts_y, pts_x = np.where(mask)
            points_xyz_mm = np.column_stack(
                (
                    frame.origin_x_mm + pts_x.astype(np.float64) * float(frame.x_resolution_mm),
                    frame.origin_y_mm + pts_y.astype(np.float64) * float(frame.y_resolution_mm),
                    np.maximum(values.astype(np.float64), 0.0),
                )
            )
            sphere_fit = _fit_sphere_least_squares(points_xyz_mm)
            ellipsoid_fit = _fit_ellipsoid_pca(points_xyz_mm)
            sphere_rmse = float(sphere_fit.get("rmse_mm") or 0.0) if sphere_fit.get("valid") else None
            ellipsoid_rmse = float(ellipsoid_fit.get("rmse") or 0.0) if ellipsoid_fit.get("valid") else None
            sphere_vs_ellipsoid_gain = (
                float(max(0.0, (sphere_rmse - ellipsoid_rmse) / max(sphere_rmse, 1e-9)))
                if sphere_rmse is not None and ellipsoid_rmse is not None
                else None
            )
            deformation_score = (
                float(max(0.0, min(1.0, (sphere_vs_ellipsoid_gain or 0.0) * 0.7 + max(0.0, ((ellipsoid_fit.get("aspect_ratio") or 1.0) - 1.0)) * 0.3)))
                if sphere_vs_ellipsoid_gain is not None and ellipsoid_fit.get("valid")
                else None
            )
            radial_height_rmse_mm = None
            radial_height_max_error_mm = None
            if sphere_fit.get("valid"):
                center = np.asarray(sphere_fit.get("center_mm"), dtype=np.float64)
                radius = float(sphere_fit.get("radius_mm") or 0.0)
                if radius > 0.0:
                    dx = points_xyz_mm[:, 0] - center[0]
                    dy = points_xyz_mm[:, 1] - center[1]
                    rr = np.sqrt(dx * dx + dy * dy)
                    inside = np.maximum(0.0, radius * radius - rr * rr)
                    expected_z = center[2] + np.sqrt(inside)
                    dz = points_xyz_mm[:, 2] - expected_z
                    radial_height_rmse_mm = float(np.sqrt(np.mean(dz ** 2)))
                    radial_height_max_error_mm = float(np.max(np.abs(dz)))
            nonzero_values = np.maximum(values.astype(np.float64), 0.0)
            filled_ratio = float(np.count_nonzero(nonzero_values > 0.05) / max(1, nonzero_values.size))
            surface_completeness_ratio = filled_ratio
            missing_surface_fraction = float(max(0.0, 1.0 - surface_completeness_ratio))
            expected_sphere_volume_mm3 = float((4.0 / 3.0) * np.pi * (float(sphere_fit.get("radius_mm") or sphere_radius_mm) ** 3))
            observed_volume_ratio = float(volume_proxy / max(expected_sphere_volume_mm3, 1e-9))
            volume_deficit_ratio = float(max(0.0, 1.0 - observed_volume_ratio))
            surface_roughness_mean = float(np.mean(np.abs(surface_residuals))) if surface_residuals.size else 0.0
            surface_roughness_std = float(np.std(surface_residuals)) if surface_residuals.size else 0.0
            surface_roughness_score = float(max(0.0, min(1.0, surface_roughness_mean / 8.0)))
            # Flat region proxy: low local gradient zones inside the object mask.
            grad_mag = np.sqrt(gx * gx + gy * gy)
            flat_mask = inner_mask & (grad_mag < float(self.flat_gradient_threshold))
            flat_region_ratio = float(np.count_nonzero(flat_mask) / max(1, np.count_nonzero(inner_mask)))
            largest_flat_region_mm2 = 0.0
            if np.count_nonzero(flat_mask):
                nlab, _labels, stats, _ = cv2.connectedComponentsWithStats(flat_mask.astype(np.uint8), connectivity=8)
                if nlab > 1:
                    largest_flat_region_mm2 = float(np.max(stats[1:, cv2.CC_STAT_AREA]) * pixel_area)
            lap = cv2.Laplacian(np.where(mask, normalized, 0.0).astype(np.float32), cv2.CV_32F)
            lap_vals = np.abs(lap[inner_mask]) if np.count_nonzero(inner_mask) else np.asarray([], dtype=np.float32)
            surface_discontinuity_score = float(np.mean(lap_vals)) if lap_vals.size else 0.0
            high_curvature_ratio = float(np.count_nonzero(lap_vals > 0.2) / max(1, lap_vals.size)) if lap_vals.size else 0.0
            circularity = _safe_float(contour_mm_metrics.get("circularity"))
            circularity_score = float(max(0.0, min(1.0, circularity if circularity is not None else 0.0)))
            footprint_geometry = {
                "circularity": round(circularity, 6) if circularity is not None else None,
                "circularity_score": round(circularity_score, 6),
                **radial_uniformity,
                "eccentricity": round(float(comp.get("eccentricity") or 0.0), 6),
                "aspect_ratio": round(float((comp.get("major_axis_mm") or 0.0) / max(float(comp.get("minor_axis_mm") or 1e-9), 1e-9)), 6),
                "ellipse_fit_quality": round(float(comp.get("roundness") or 0.0), 6),
            }
            surface_geometry = {
                "sphere_fit_center_mm": sphere_fit.get("center_mm") if sphere_fit.get("valid") else None,
                "sphere_fit_radius_mm": sphere_fit.get("radius_mm") if sphere_fit.get("valid") else None,
                "sphere_fit_rmse_mm": sphere_fit.get("rmse_mm") if sphere_fit.get("valid") else None,
                "sphere_fit_max_error_mm": sphere_fit.get("max_error_mm") if sphere_fit.get("valid") else None,
                "sphere_fit_valid_point_count": sphere_fit.get("point_count") if sphere_fit.get("valid") else int(points_xyz_mm.shape[0]),
                "ellipsoid_axes_mm": ellipsoid_fit.get("axes_mm") if ellipsoid_fit.get("valid") else None,
                "ellipsoid_aspect_ratio": ellipsoid_fit.get("aspect_ratio") if ellipsoid_fit.get("valid") else None,
                "ellipsoid_fit_rmse_mm": ellipsoid_fit.get("rmse") if ellipsoid_fit.get("valid") else None,
                "sphere_vs_ellipsoid_gain": round(sphere_vs_ellipsoid_gain, 6) if sphere_vs_ellipsoid_gain is not None else None,
                "deformation_score": round(deformation_score, 6) if deformation_score is not None else None,
            }
            p95_height_mm = float(np.percentile(values, 95)) if values.size else None
            equivalent_diameter_mm = float(np.sqrt((4.0 * float(comp.get("footprint_area_mm2") or 0.0)) / np.pi)) if float(comp.get("footprint_area_mm2") or 0.0) > 0.0 else None
            dim_x_mm = float(comp["bbox"][2] * frame.x_resolution_mm)
            dim_y_mm = float(comp["bbox"][3] * frame.y_resolution_mm)
            segmentation_coverage = float(np.count_nonzero(mask) / max(1, comp["bbox"][2] * comp["bbox"][3]))
            derived_sphere_consistency = compute_sphere_consistency_features(
                sphere_fit=sphere_fit,
                ellipsoid_fit=ellipsoid_fit,
                diameter_selected_mm=sphere_diameter_mm if sphere_diameter_mm > 0.0 else None,
                equivalent_diameter_mm=equivalent_diameter_mm,
                dim_x_mm=dim_x_mm,
                dim_y_mm=dim_y_mm,
                dim_z_mm=float(dim_z_height),
                height_p95_mm=p95_height_mm,
                volume_proxy_mm3=float(volume_proxy),
                segmentation_coverage=segmentation_coverage,
                border_clipped=None,
            )
            sphere_consistency = {
                "radial_height_rmse_mm": round(radial_height_rmse_mm, 6) if radial_height_rmse_mm is not None else None,
                "radial_height_max_error_mm": round(radial_height_max_error_mm, 6) if radial_height_max_error_mm is not None else None,
                "surface_completeness_ratio": round(surface_completeness_ratio, 6),
                "missing_surface_fraction": round(missing_surface_fraction, 6),
                "expected_sphere_volume_mm3": round(expected_sphere_volume_mm3, 4),
                "observed_volume_ratio": round(observed_volume_ratio, 6),
                "volume_deficit_ratio": round(volume_deficit_ratio, 6),
                **derived_sphere_consistency,
            }
            damage_metrics = {
                "surface_roughness_mean": round(surface_roughness_mean, 6),
                "surface_roughness_std": round(surface_roughness_std, 6),
                "surface_roughness_score": round(surface_roughness_score, 6),
                "flat_region_ratio": round(flat_region_ratio, 6),
                "largest_flat_region_mm2": round(largest_flat_region_mm2, 4),
                "surface_discontinuity_score": round(surface_discontinuity_score, 6),
                "high_curvature_ratio": round(high_curvature_ratio, 6),
            }
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
                    round(dim_z_height, 4),
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
                    "p99_height_mm": round(p99_height, 4),
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
                        float(dim_z_height),
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
                "feature_circularity": round(float(circularity or 0.0), 6),
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
                "footprint_geometry": footprint_geometry,
                "surface_geometry": surface_geometry,
                "sphere_consistency": sphere_consistency,
                "damage_metrics": damage_metrics,
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
        max_height_values = [
            float(((item.get("height_above_belt_mm") or {}).get("max_height_mm") or 0.0))
            for item in objects
            if isinstance(item.get("height_above_belt_mm"), dict) and ((item.get("height_above_belt_mm") or {}).get("max_height_mm")) is not None
        ]
        p95_height_values = [
            float(((item.get("height_above_belt_mm") or {}).get("p95_height_mm") or 0.0))
            for item in objects
            if isinstance(item.get("height_above_belt_mm"), dict) and ((item.get("height_above_belt_mm") or {}).get("p95_height_mm")) is not None
        ]
        p99_height_values = [
            float(((item.get("height_above_belt_mm") or {}).get("p99_height_mm") or 0.0))
            for item in objects
            if isinstance(item.get("height_above_belt_mm"), dict) and ((item.get("height_above_belt_mm") or {}).get("p99_height_mm")) is not None
        ]
        volume_values = [
            float(item.get("feature_volume_proxy_mm3") or 0.0)
            for item in objects
            if item.get("feature_volume_proxy_mm3") is not None
        ]
        dimension_x_values = [float(item.get("dim_x_mm") or 0.0) for item in objects if item.get("dim_x_mm") is not None]
        dimension_y_values = [float(item.get("dim_y_mm") or 0.0) for item in objects if item.get("dim_y_mm") is not None]
        dimension_z_values = [float(item.get("dim_z_mm") or 0.0) for item in objects if item.get("dim_z_mm") is not None]
        deformation_values = [
            float(item.get("feature_local_curvature_proxy") or 0.0)
            for item in objects
            if item.get("feature_local_curvature_proxy") is not None
        ]
        _runtime_trace_mask_unit(
            context,
            unit_id="measurement.height_metrics",
            stage_id="measurement",
            parent_id="measurement",
            input_artifacts=["height_above_belt_raster", "connected_components"],
            metrics={
                "object_count": int(len(objects)),
                "max_height_min_mm": _numeric_summary(max_height_values).get("min"),
                "max_height_max_mm": _numeric_summary(max_height_values).get("max"),
                "max_height_mean_mm": _numeric_summary(max_height_values).get("mean"),
                "p95_height_max_mm": _numeric_summary(p95_height_values).get("max"),
                "p99_height_max_mm": _numeric_summary(p99_height_values).get("max"),
            },
            diagnostics={"object_ids": [int(item.get("object_id") or 0) for item in objects]},
        )
        _runtime_trace_mask_unit(
            context,
            unit_id="measurement.volume_proxy",
            stage_id="measurement",
            parent_id="measurement",
            input_artifacts=["height_above_belt_raster"],
            metrics={
                "object_count": int(len(objects)),
                "volume_proxy_min_mm3": _numeric_summary(volume_values).get("min"),
                "volume_proxy_max_mm3": _numeric_summary(volume_values).get("max"),
                "volume_proxy_mean_mm3": _numeric_summary(volume_values).get("mean"),
            },
        )
        _runtime_trace_mask_unit(
            context,
            unit_id="measurement.dimensions",
            stage_id="measurement",
            parent_id="measurement",
            input_artifacts=["connected_components"],
            metrics={
                "object_count": int(len(objects)),
                "dim_x_max_mm": _numeric_summary(dimension_x_values).get("max"),
                "dim_y_max_mm": _numeric_summary(dimension_y_values).get("max"),
                "dim_z_max_mm": _numeric_summary(dimension_z_values).get("max"),
            },
        )
        _runtime_trace_mask_unit(
            context,
            unit_id="measurement.deformation_sphericity",
            stage_id="measurement",
            parent_id="measurement",
            input_artifacts=["connected_components", "height_above_belt_raster"],
            metrics={
                "object_count": int(len(objects)),
                "deformation_min": _numeric_summary(deformation_values).get("min"),
                "deformation_max": _numeric_summary(deformation_values).get("max"),
                "deformation_mean": _numeric_summary(deformation_values).get("mean"),
            },
        )


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
        # Cube calibration measures dim_z the same way the rest of the pipeline does:
        # P99 of height_above_belt to stay robust to top-end noise peaks on the cube
        # surface. Fall back to dim_z from dimensions_mm (which is also P99 sourced)
        # and finally to the absolute max for legacy artifacts. Snapshot max/p99 as
        # scalars BEFORE any in-place scale correction so the final result still
        # reports the raw measurement.
        target_heights = target.get("height_above_belt_mm") if isinstance(target.get("height_above_belt_mm"), dict) else {}
        raw_p99_height = _safe_float(target_heights.get("p99_height_mm"))
        raw_max_height = _safe_float(target_heights.get("max_height_mm"))
        measured_h = float(
            raw_p99_height
            or (dims[2] if len(dims) >= 3 else None)
            or raw_max_height
            or 0.0
        )

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
            corrected_heights = corrected_target.get("height_above_belt_mm") if isinstance(corrected_target.get("height_above_belt_mm"), dict) else {}
            corrected_height = (
                corrected_heights.get("p99_height_mm")
                or (corrected_dims[2] if isinstance(corrected_dims, (list, tuple)) and len(corrected_dims) >= 3 else None)
                or corrected_heights.get("max_height_mm")
            )
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
            "measured_height_source": "p99_height_mm",
            "measured_height_max_mm": raw_max_height,
            "measured_height_p99_mm": raw_p99_height,
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
        _runtime_trace_mask_unit(
            context,
            unit_id="measurement.correction_calibration_context",
            stage_id="measurement",
            parent_id="measurement",
            parameters={
                "enabled": bool(result.get("enabled", False)),
                "apply_correction": bool(result.get("applied_correction", False)),
                "tolerance_percent": result.get("tolerance_percent"),
            },
            input_artifacts=["geometry_debug_summary"],
            output_artifacts=["known_object_scale_validation"],
            metrics={
                "correction_applied": bool(result.get("correction_applied", False)),
                "scale_x": result.get("scale_x"),
                "scale_y": result.get("scale_y"),
                "scale_z": result.get("scale_z"),
            },
            diagnostics={
                "status": result.get("status"),
                "correction_source": result.get("correction_source"),
                "correction_context_id": result.get("correction_context_id"),
            },
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
        invariant_warning_count = sum(
            len(row.get("geometry_invariant_warnings") or [])
            for row in rows
            if isinstance(row, dict)
        )
        _runtime_trace_mask_unit(
            context,
            unit_id="geometry.geometry_summary",
            stage_id="geometry",
            parent_id="geometry",
            input_artifacts=["connected_components", "contour", "convex_hull", "fitted_ellipse"],
            output_artifacts=["geometry_debug_summary"],
            metrics={
                "object_count": int(len(rows)),
                "invariant_warning_count": int(invariant_warning_count),
            },
            diagnostics={"object_ids": [int(row.get("object_id") or 0) for row in rows if isinstance(row, dict)]},
        )
        _runtime_trace_mask_unit(
            context,
            unit_id="geometry",
            stage_id="geometry",
            input_artifacts=["connected_components", "contour", "convex_hull", "fitted_ellipse"],
            output_artifacts=["geometry_debug_summary"],
            metrics={
                "object_count": int(len(rows)),
                "invariant_warning_count": int(invariant_warning_count),
            },
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
            for key in ("max_height_mm", "mean_height_mm", "median_height_mm", "p95_height_mm", "p99_height_mm", "height_std_mm"):
                if heights.get(key) is not None:
                    heights[key] = round(float(heights[key]) * scale_z, 4)
        if row.get("major_axis_mm") is not None and row.get("minor_axis_mm") is not None and isinstance(heights, dict):
            # dim_z follows the noise-robust P99; fall back to the absolute max only
            # when older artifacts predate the p99_height_mm field.
            dim_z_corrected = float(heights.get("p99_height_mm") or heights.get("max_height_mm") or 0.0)
            row["feature_sphericity_3d"] = round(_axis_balance_3d(float(row["major_axis_mm"]), float(row["minor_axis_mm"]), dim_z_corrected), 4)
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
class ComputeMeasurementDiagnosticsStage:
    name: str = "ComputeMeasurementDiagnostics"
    category: str = "measurement"
    required_modalities: tuple[CaptureModality, ...] = ("heightmap",)
    produced_modalities: tuple[CaptureModality, ...] = ()
    produced_artifacts: tuple[str, ...] = ("measurement_diagnostics", "feature_vector", "feature_provenance", "quality_flags")
    stage_category: str | None = "measurement"
    min_valid_pixel_ratio: float = 0.85
    max_invalid_pixel_ratio: float = 0.2
    max_border_touch_ratio: float = 0.02
    max_plane_residual_std_mm: float = 2.0

    def run(self, context: PipelineContext) -> None:
        frame: HeightmapFrame = context.require_artifact("heightmap_frame")
        output_dir: Path = context.require_artifact("output_dir")
        objects = context.get_artifact("objects", [])
        segmentation_mask = np.asarray(context.get_artifact("segmentation_mask", np.zeros_like(frame.valid_mask, dtype=bool)), dtype=bool)
        normalized = np.asarray(context.get_artifact("normalized_heightmap_mm"), dtype=np.float32)

        valid_mask = np.asarray(frame.valid_mask, dtype=bool)
        invalid_mask = ~valid_mask
        total_px = int(valid_mask.size)
        valid_px = int(np.count_nonzero(valid_mask))
        invalid_px = max(0, total_px - valid_px)

        invalid_u8 = invalid_mask.astype(np.uint8)
        hole_count = 0
        hole_area_ratio = 0.0
        if invalid_px > 0:
            cc_count, labels = cv2.connectedComponents(invalid_u8)
            hole_count = max(0, int(cc_count) - 1)
            if hole_count > 0:
                areas = [int(np.count_nonzero(labels == idx)) for idx in range(1, cc_count)]
                hole_area_ratio = float(sum(areas) / max(1, total_px))

        foreground_px = int(np.count_nonzero(segmentation_mask))
        foreground_ratio = float(foreground_px / max(1, total_px))
        roi_occupancy_ratio = foreground_ratio

        border = np.zeros_like(segmentation_mask, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        border_touch_px = int(np.count_nonzero(segmentation_mask & border))
        border_touch_ratio = float(border_touch_px / max(1, foreground_px))

        bboxes = [item.get("bbox_px") for item in objects if isinstance(item, dict) and isinstance(item.get("bbox_px"), list) and len(item.get("bbox_px")) >= 4]
        bbox_area_px = 0
        bbox_width_px = 0.0
        bbox_height_px = 0.0
        if bboxes:
            x0 = int(min(float(box[0]) for box in bboxes))
            y0 = int(min(float(box[1]) for box in bboxes))
            x1 = int(max(float(box[0]) + float(box[2]) for box in bboxes))
            y1 = int(max(float(box[1]) + float(box[3]) for box in bboxes))
            bbox_width_px = float(max(0, x1 - x0))
            bbox_height_px = float(max(0, y1 - y0))
            bbox_area_px = int(bbox_width_px * bbox_height_px)
        bbox_crop_ratio = float(bbox_area_px / max(1, total_px))

        residual_path = output_dir / "belt_plane_residuals.values.f32"
        plane_residual = None
        if residual_path.is_file():
            try:
                plane_residual = np.fromfile(residual_path, dtype=np.float32).reshape(frame.z_mm.shape)
            except Exception:
                plane_residual = None
        if plane_residual is not None:
            residual_values = plane_residual[np.isfinite(plane_residual)]
        else:
            residual_values = np.array([], dtype=np.float32)
        residual_mean = float(np.mean(residual_values)) if residual_values.size else 0.0
        residual_std = float(np.std(residual_values)) if residual_values.size else 0.0
        residual_p95 = float(np.percentile(residual_values, 95)) if residual_values.size else 0.0

        object_mask = segmentation_mask & valid_mask
        object_heights = normalized[object_mask] if int(np.count_nonzero(object_mask)) > 0 else np.array([], dtype=np.float32)
        height_min = float(np.min(object_heights)) if object_heights.size else 0.0
        height_max = float(np.max(object_heights)) if object_heights.size else 0.0
        height_mean = float(np.mean(object_heights)) if object_heights.size else 0.0
        height_std = float(np.std(object_heights)) if object_heights.size else 0.0
        height_p95 = float(np.percentile(object_heights, 95)) if object_heights.size else 0.0
        height_p99 = float(np.percentile(object_heights, 99)) if object_heights.size else 0.0

        primary = self._pick_primary_object(objects)
        pixel_area = float(frame.pixel_area_mm2())
        footprint_area_px = float(primary.get("point_count") or foreground_px or 0.0) if primary else float(foreground_px)
        footprint_area_mm2 = float(primary.get("footprint_area_mm2") or (footprint_area_px * pixel_area)) if primary else float(footprint_area_px * pixel_area)
        major = self._as_float(primary.get("major_axis_mm") if primary else None)
        minor = self._as_float(primary.get("minor_axis_mm") if primary else None)
        equiv_diam = float(np.sqrt((4.0 * footprint_area_mm2) / np.pi)) if footprint_area_mm2 > 0 else 0.0
        eccentricity = self._as_float(primary.get("feature_eccentricity") if primary else None) or 0.0
        roundness = self._as_float(primary.get("sphericity_score") if primary else None) or 0.0
        flatness = self._as_float(primary.get("feature_flatness") if primary else None) or 0.0
        asymmetry = self._as_float(primary.get("feature_height_asymmetry") if primary else None) or 0.0
        volume_proxy = self._as_float(primary.get("feature_volume_proxy_mm3") if primary else None) or 0.0
        footprint_group = primary.get("footprint_geometry") if isinstance(primary, dict) and isinstance(primary.get("footprint_geometry"), dict) else {}
        surface_group = primary.get("surface_geometry") if isinstance(primary, dict) and isinstance(primary.get("surface_geometry"), dict) else {}
        consistency_group = primary.get("sphere_consistency") if isinstance(primary, dict) and isinstance(primary.get("sphere_consistency"), dict) else {}
        damage_group = primary.get("damage_metrics") if isinstance(primary, dict) and isinstance(primary.get("damage_metrics"), dict) else {}

        perimeter_mm = self._as_float(primary.get("perimeter_mm") if primary else None)
        compactness = 0.0
        if perimeter_mm and perimeter_mm > 0:
            compactness = float((4.0 * np.pi * footprint_area_mm2) / (perimeter_mm * perimeter_mm))
        solidity = self._estimate_solidity(primary, frame=frame)
        convex_volume_proxy = float(footprint_area_mm2 * max(height_p95, 0.0))

        features = {
            "valid_pixel_ratio": float(valid_px / max(1, total_px)),
            "invalid_pixel_ratio": float(invalid_px / max(1, total_px)),
            "invalid_hole_count": int(hole_count),
            "invalid_hole_area_ratio": float(hole_area_ratio),
            "border_touch_ratio": float(border_touch_ratio),
            "segmentation_foreground_ratio": float(foreground_ratio),
            "bbox_crop_ratio": float(bbox_crop_ratio),
            "roi_occupancy_ratio": float(roi_occupancy_ratio),
            "plane_residual_mean": float(residual_mean),
            "plane_residual_std": float(residual_std),
            "plane_residual_p95": float(residual_p95),
            "height_min": float(height_min),
            "height_max": float(height_max),
            "height_mean": float(height_mean),
            "height_std": float(height_std),
            "height_p95": float(height_p95),
            "height_p99": float(height_p99),
            "bbox_width_px": float(bbox_width_px),
            "bbox_height_px": float(bbox_height_px),
            "footprint_area_px": float(footprint_area_px),
            "footprint_area_mm2": float(footprint_area_mm2),
            "equivalent_diameter_mm": float(equiv_diam),
            "ellipse_major_axis_mm": float(major or 0.0),
            "ellipse_minor_axis_mm": float(minor or 0.0),
            "eccentricity": float(eccentricity),
            "roundness": float(roundness),
            "solidity": float(solidity),
            "compactness": float(compactness),
            "flatness": float(flatness),
            "asymmetry": float(asymmetry),
            "integrated_height_volume_proxy": float(volume_proxy),
            "convex_volume_proxy": float(convex_volume_proxy),
            "footprint_radial_cv": float(self._as_float(footprint_group.get("radial_cv")) or 0.0),
            "surface_sphere_fit_rmse_mm": float(self._as_float(surface_group.get("sphere_fit_rmse_mm")) or 0.0),
            "consistency_radial_height_rmse_mm": float(self._as_float(consistency_group.get("radial_height_rmse_mm")) or 0.0),
            "damage_flat_region_ratio": float(self._as_float(damage_group.get("flat_region_ratio")) or 0.0),
        }

        feature_provenance = self._build_feature_provenance(features)
        quality_flags = self._build_quality_flags(features)
        feature_vector_payload = {"features": features}
        diagnostics_payload = {
            "feature_count": len(features),
            "object_count": len(objects) if isinstance(objects, list) else 0,
            "primary_object_id": int(primary.get("object_id")) if isinstance(primary, dict) and primary.get("object_id") is not None else None,
            "quality_flag_count": len(quality_flags),
            "confidence": None,
            "expected_error": None,
            "feature_group_summaries": (primary.get("feature_group_summaries") if isinstance(primary, dict) else []) or [],
            "feature_warnings": (primary.get("feature_warnings") if isinstance(primary, dict) else []) or [],
            "feature_readiness": (primary.get("feature_readiness") if isinstance(primary, dict) else {}) or {},
        }

        contour_payload = self._build_contour_payload(objects)
        hull_payload = self._build_hull_payload(objects)
        ellipse_payload = self._build_ellipse_payload(objects)
        axes_payload = self._build_axes_payload(objects)
        radial_payload = self._build_radial_profile(normalized=normalized, mask=object_mask, frame=frame)
        hist_payload = self._build_height_histogram(normalized=normalized, mask=object_mask)

        (output_dir / "measurement_diagnostics.json").write_text(json.dumps(diagnostics_payload, indent=2), encoding="utf-8")
        (output_dir / "feature_vector.json").write_text(json.dumps(feature_vector_payload, indent=2), encoding="utf-8")
        (output_dir / "feature_provenance.json").write_text(json.dumps(feature_provenance, indent=2), encoding="utf-8")
        (output_dir / "quality_flags.json").write_text(json.dumps({"flags": quality_flags}, indent=2), encoding="utf-8")
        (output_dir / "contour.json").write_text(json.dumps(contour_payload, indent=2), encoding="utf-8")
        (output_dir / "convex_hull.json").write_text(json.dumps(hull_payload, indent=2), encoding="utf-8")
        (output_dir / "fitted_ellipse.json").write_text(json.dumps(ellipse_payload, indent=2), encoding="utf-8")
        (output_dir / "principal_axes.json").write_text(json.dumps(axes_payload, indent=2), encoding="utf-8")
        (output_dir / "radial_profile.json").write_text(json.dumps(radial_payload, indent=2), encoding="utf-8")
        (output_dir / "diagnostics_normalized_height_histogram.json").write_text(json.dumps(hist_payload, indent=2), encoding="utf-8")

        context.set_artifact("measurement_diagnostics", diagnostics_payload)
        context.set_artifact("feature_vector", feature_vector_payload)
        context.set_artifact("feature_provenance", feature_provenance)
        context.set_artifact("quality_flags", {"flags": quality_flags})

        files = dict(context.get_artifact("files", {}))
        files.update(
            {
                "measurement_diagnostics": "measurement_diagnostics.json",
                "feature_vector": "feature_vector.json",
                "feature_provenance": "feature_provenance.json",
                "quality_flags": "quality_flags.json",
            }
        )
        context.set_artifact("files", files)

        self._record_artifact(context, "measurement_diagnostics", "measurement_diagnostics.json", "Measurement diagnostics", "world_mm", ["measurement_object_1"], {"semantic_type": "diagnostics", "artifact_kind": "diagnostics"})
        self._record_artifact(context, "feature_vector", "feature_vector.json", "Feature vector", "world_mm", ["measurement_diagnostics"], {"semantic_type": "feature_vector", "artifact_kind": "diagnostics"})
        self._record_artifact(context, "feature_provenance", "feature_provenance.json", "Feature provenance", "world_mm", ["feature_vector"], {"semantic_type": "feature_provenance", "artifact_kind": "diagnostics"})
        self._record_artifact(context, "quality_flags", "quality_flags.json", "Quality flags", "world_mm", ["feature_vector"], {"semantic_type": "quality_flags", "artifact_kind": "diagnostics"})
        self._record_artifact(context, "contour", "contour.json", "Contour geometry", "projection_pixel", ["height_segmentation"], {"semantic_type": "geometry_intermediate", "artifact_kind": "diagnostics"})
        self._record_artifact(context, "convex_hull", "convex_hull.json", "Convex hull geometry", "projection_pixel", ["height_segmentation"], {"semantic_type": "geometry_intermediate", "artifact_kind": "diagnostics"})
        self._record_artifact(context, "fitted_ellipse", "fitted_ellipse.json", "Fitted ellipse geometry", "projection_pixel", ["height_segmentation"], {"semantic_type": "geometry_intermediate", "artifact_kind": "diagnostics"})
        self._record_artifact(context, "principal_axes", "principal_axes.json", "Principal axes geometry", "projection_pixel", ["height_segmentation"], {"semantic_type": "geometry_intermediate", "artifact_kind": "diagnostics"})
        self._record_artifact(context, "radial_profile", "radial_profile.json", "Radial profile", "world_mm", ["height_above_belt_raster"], {"semantic_type": "geometry_intermediate", "artifact_kind": "diagnostics"})
        feature_warning_count = len(diagnostics_payload.get("feature_warnings") or [])
        invariant_warning_rows = list(primary.get("geometry_invariant_warnings") or []) if isinstance(primary, dict) else []
        invariant_warning_count = len(invariant_warning_rows)
        _runtime_trace_mask_unit(
            context,
            unit_id="measurement_diagnostics.feature_vector_generation",
            stage_id="measurement_diagnostics",
            parent_id="measurement_diagnostics",
            input_artifacts=["geometry_debug_summary"],
            output_artifacts=["feature_vector"],
            metrics={
                "feature_count": int(len(features)),
                "object_count": int(len(objects) if isinstance(objects, list) else 0),
            },
        )
        _runtime_trace_mask_unit(
            context,
            unit_id="measurement_diagnostics.quality_flags",
            stage_id="measurement_diagnostics",
            parent_id="measurement_diagnostics",
            input_artifacts=["feature_vector"],
            output_artifacts=["quality_flags"],
            metrics={
                "quality_flag_count": int(len(quality_flags)),
                "objects_with_warnings": int(1 if feature_warning_count else 0),
            },
            diagnostics={"flags": quality_flags},
        )
        _runtime_trace_mask_unit(
            context,
            unit_id="measurement_diagnostics.invariant_checks",
            stage_id="measurement_diagnostics",
            parent_id="measurement_diagnostics",
            input_artifacts=["feature_vector", "known_object_scale_validation"],
            output_artifacts=["feature_provenance"],
            metrics={
                "invariant_warning_count": int(invariant_warning_count),
                "critical_warning_count": int(sum(1 for flag in quality_flags if str((flag or {}).get("severity") or "") == "critical")),
            },
            diagnostics={"feature_warning_count": int(feature_warning_count)},
        )
        known_object_validation = context.get_artifact("known_object_scale_validation", {})
        _runtime_trace_mask_unit(
            context,
            unit_id="measurement_diagnostics.known_object_validation",
            stage_id="measurement_diagnostics",
            parent_id="measurement_diagnostics",
            parameters=dict(context.get_artifact("stage_params.known_object_25d", {}) or {}),
            input_artifacts=["known_object_scale_validation", "geometry_debug_summary"],
            output_artifacts=["measurement_diagnostics"],
            metrics={
                "known_object_validation_enabled": bool((known_object_validation or {}).get("enabled", False)) if isinstance(known_object_validation, dict) else False,
                "scale_x": (known_object_validation or {}).get("scale_x") if isinstance(known_object_validation, dict) else None,
                "scale_y": (known_object_validation or {}).get("scale_y") if isinstance(known_object_validation, dict) else None,
                "scale_z": (known_object_validation or {}).get("scale_z") if isinstance(known_object_validation, dict) else None,
            },
            diagnostics={"status": (known_object_validation or {}).get("status") if isinstance(known_object_validation, dict) else None},
        )
        _runtime_trace_mask_unit(
            context,
            unit_id="measurement_diagnostics",
            stage_id="measurement_diagnostics",
            input_artifacts=["geometry_debug_summary", "known_object_scale_validation"],
            output_artifacts=["measurement_diagnostics", "feature_vector", "feature_provenance", "quality_flags"],
            metrics={
                "feature_count": int(len(features)),
                "quality_flag_count": int(len(quality_flags)),
                "object_count": int(len(objects) if isinstance(objects, list) else 0),
                "warning_count": int(feature_warning_count + invariant_warning_count),
            },
        )

    def _record_artifact(
        self,
        context: PipelineContext,
        artifact_id: str,
        path: str,
        title: str,
        coordinate_space: str,
        source_artifact_ids: list[str],
        extra_meta: dict[str, Any],
    ) -> None:
        meta = _numeric_source_meta(
            {
                "source": "native_pipeline",
                "producer": self.name,
                "stage_id": "measurement_diagnostics",
                "coordinate_space": coordinate_space,
                "lineage": {"source_artifact_ids": source_artifact_ids},
                **extra_meta,
            }
        )
        context.add_processing_artifact(
            artifact_id=artifact_id,
            stage_id="measurement_diagnostics",
            kind="json",
            title=title,
            path=path,
            mime_type="application/json",
            preview_available=True,
            coordinate_space=coordinate_space,  # type: ignore[arg-type]
            source_artifact_ids=source_artifact_ids,
            metadata=meta,
        )

    def _build_feature_provenance(self, features: dict[str, Any]) -> dict[str, Any]:
        provenance: dict[str, Any] = {}
        for key, value in features.items():
            stage = "measurement_diagnostics"
            source_artifact = "measurement_diagnostics"
            derived_from = ["height_segmentation", "height_above_belt_raster"]
            if key.startswith("plane_residual"):
                stage = "detect_belt_plane"
                source_artifact = "belt_plane_residuals"
                derived_from = ["belt_plane_residuals", "belt_plane"]
            elif key in {"ellipse_major_axis_mm", "ellipse_minor_axis_mm", "eccentricity", "roundness"}:
                stage = "measurement"
                source_artifact = "measurement_object"
                derived_from = ["connected_components", "height_segmentation"]
            elif key in {"footprint_area_px", "footprint_area_mm2", "bbox_width_px", "bbox_height_px"}:
                stage = "geometry"
                source_artifact = "connected_components"
                derived_from = ["height_segmentation"]
            validity = "ok"
            if key == "valid_pixel_ratio" and float(value) < 0.85:
                validity = "warning"
            if key == "invalid_pixel_ratio" and float(value) > 0.25:
                validity = "invalid"
            provenance[key] = {
                "value": value,
                "source_stage": stage,
                "source_artifact_id": source_artifact,
                "validity": validity,
                "confidence": None,
                "expected_error": None,
                "derived_from": derived_from,
            }
        return provenance

    def _build_quality_flags(self, features: dict[str, Any]) -> list[dict[str, Any]]:
        flags: list[dict[str, Any]] = []
        if float(features.get("valid_pixel_ratio", 0.0)) < self.min_valid_pixel_ratio:
            flags.append({"code": "LOW_VALID_PIXEL_RATIO", "severity": "warning", "trigger": {"valid_pixel_ratio": features["valid_pixel_ratio"]}, "related_features": ["valid_pixel_ratio"]})
        if float(features.get("invalid_pixel_ratio", 0.0)) > self.max_invalid_pixel_ratio:
            flags.append({"code": "HIGH_INVALID_REGION_RATIO", "severity": "warning", "trigger": {"invalid_pixel_ratio": features["invalid_pixel_ratio"]}, "related_features": ["invalid_pixel_ratio"]})
        if float(features.get("border_touch_ratio", 0.0)) > self.max_border_touch_ratio:
            flags.append({"code": "BORDER_TOUCH", "severity": "warning", "trigger": {"border_touch_ratio": features["border_touch_ratio"]}, "related_features": ["border_touch_ratio"]})
        if float(features.get("plane_residual_std", 0.0)) > self.max_plane_residual_std_mm:
            flags.append({"code": "HIGH_PLANE_RESIDUAL", "severity": "warning", "trigger": {"plane_residual_std": features["plane_residual_std"]}, "related_features": ["plane_residual_std"]})
        if float(features.get("footprint_area_mm2", 0.0)) < 50.0:
            flags.append({"code": "SMALL_OBJECT", "severity": "warning", "trigger": {"footprint_area_mm2": features["footprint_area_mm2"]}, "related_features": ["footprint_area_mm2"]})
        if int(features.get("invalid_hole_count", 0)) > 10:
            flags.append({"code": "SEGMENTATION_FRAGMENTED", "severity": "warning", "trigger": {"invalid_hole_count": features["invalid_hole_count"]}, "related_features": ["invalid_hole_count"]})
        if float(features.get("height_p99", 0.0)) > (float(features.get("height_mean", 0.0)) * 3.0 + 1e-6):
            flags.append({"code": "EXTREME_HEIGHT_OUTLIER", "severity": "warning", "trigger": {"height_p99": features["height_p99"], "height_mean": features["height_mean"]}, "related_features": ["height_p99", "height_mean"]})
        return flags

    def _pick_primary_object(self, objects: Any) -> dict[str, Any] | None:
        if not isinstance(objects, list) or not objects:
            return None
        return max([item for item in objects if isinstance(item, dict)], key=lambda item: float(item.get("footprint_area_mm2") or 0.0), default=None)

    def _build_contour_payload(self, objects: Any) -> dict[str, Any]:
        rows = []
        for item in objects if isinstance(objects, list) else []:
            if not isinstance(item, dict):
                continue
            rows.append({"object_id": item.get("object_id"), "contour_px": item.get("contour_px")})
        return {"coordinate_space": "projection_pixel", "objects": rows}

    def _build_hull_payload(self, objects: Any) -> dict[str, Any]:
        rows = []
        for item in objects if isinstance(objects, list) else []:
            if not isinstance(item, dict):
                continue
            contour = item.get("contour_px")
            if not isinstance(contour, list) or len(contour) < 3:
                continue
            pts = np.asarray(contour, dtype=np.float32).reshape(-1, 1, 2)
            hull = cv2.convexHull(pts).reshape(-1, 2).tolist()
            rows.append({"object_id": item.get("object_id"), "convex_hull_px": hull})
        return {"coordinate_space": "projection_pixel", "objects": rows}

    def _build_ellipse_payload(self, objects: Any) -> dict[str, Any]:
        rows = []
        for item in objects if isinstance(objects, list) else []:
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "object_id": item.get("object_id"),
                    "major_axis_mm": item.get("major_axis_mm"),
                    "minor_axis_mm": item.get("minor_axis_mm"),
                    "eccentricity": item.get("feature_eccentricity"),
                    "roundness": item.get("sphericity_score"),
                }
            )
        return {"coordinate_space": "world_mm", "objects": rows}

    def _build_axes_payload(self, objects: Any) -> dict[str, Any]:
        rows = []
        for item in objects if isinstance(objects, list) else []:
            if not isinstance(item, dict):
                continue
            bbox = item.get("bbox_px")
            centroid = item.get("centroid_px")
            if not (isinstance(bbox, list) and len(bbox) >= 4 and isinstance(centroid, list) and len(centroid) >= 2):
                continue
            rows.append(
                {
                    "object_id": item.get("object_id"),
                    "centroid_px": centroid,
                    "principal_axes_px": {"major": [bbox[2] / 2.0, 0.0], "minor": [0.0, bbox[3] / 2.0]},
                }
            )
        return {"coordinate_space": "projection_pixel", "objects": rows}

    def _build_radial_profile(self, *, normalized: np.ndarray, mask: np.ndarray, frame: HeightmapFrame) -> dict[str, Any]:
        ys, xs = np.where(mask)
        if xs.size == 0:
            return {"coordinate_space": "world_mm", "bins": [], "values": []}
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        r_px = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        values = normalized[ys, xs]
        bins = np.linspace(0.0, float(np.max(r_px)) + 1e-6, num=16)
        idx = np.digitize(r_px, bins, right=False)
        profile = []
        for ii in range(1, len(bins) + 1):
            vals = values[idx == ii]
            profile.append(float(np.mean(vals)) if vals.size else 0.0)
        return {"coordinate_space": "world_mm", "radius_bins_px": [float(v) for v in bins.tolist()], "height_mean_by_radius": profile, "x_resolution_mm": float(frame.x_resolution_mm), "y_resolution_mm": float(frame.y_resolution_mm)}

    def _build_height_histogram(self, *, normalized: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
        values = normalized[mask].astype(np.float64, copy=False)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return {"coordinate_space": "world_mm", "bin_edges": [], "counts": []}
        lo = float(np.min(values))
        hi = float(np.max(values))
        if hi <= lo:
            hi = lo + 1e-6
        counts, edges = np.histogram(values, bins=20, range=(lo, hi))
        return {"coordinate_space": "world_mm", "bin_edges": [float(v) for v in edges.tolist()], "counts": [int(v) for v in counts.tolist()]}

    def _estimate_solidity(self, primary: dict[str, Any] | None, *, frame: HeightmapFrame) -> float:
        if not isinstance(primary, dict):
            return 0.0
        contour = primary.get("contour_px")
        if not isinstance(contour, list) or len(contour) < 3:
            return 0.0
        pts = np.asarray(contour, dtype=np.float32).reshape(-1, 1, 2)
        hull = cv2.convexHull(pts)
        hull_area_px = float(cv2.contourArea(hull))
        area_mm2 = float(primary.get("footprint_area_mm2") or 0.0)
        footprint_area_px = area_mm2 / max(frame.pixel_area_mm2(), 1e-9)
        return float(footprint_area_px / hull_area_px) if hull_area_px > 0 else 0.0

    def _as_float(self, value: Any) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except Exception:
            return None


@dataclass
class ClassifyMiningBall25DStage:
    name: str = "ClassifyMiningBall25D"
    category: str = "classification"
    required_modalities: tuple[CaptureModality, ...] = ("heightmap",)
    produced_modalities: tuple[CaptureModality, ...] = ()
    produced_artifacts: tuple[str, ...] = ("classification",)
    stage_category: str | None = "classification"
    classifier_rules_path: str | None = None
    classifier_rules_pipeline_path: str | None = None

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
        resolved_rules = resolve_classifier_rule_set(
            runtime_override_path=self.classifier_rules_path,
            pipeline_config_path=self.classifier_rules_pipeline_path,
        )
        rule_set_id = resolved_rules.rule_set_id
        rule_set_path = resolved_rules.rule_set_path
        rule_set_version = resolved_rules.rule_set_version
        rule_set_source = resolved_rules.rule_set_source
        rule_params = dict(resolved_rules.params or DEFAULT_RULE_PARAMS)
        use_external_rules = rule_set_source != "builtin_default"
        ml_service = None
        active_deployment = None
        deployment_model_id = None
        deployment_id = None
        deployment_diagnostic: str | None = None
        runtime_resolution: dict[str, Any] = {}
        fallback_used = False
        fallback_reason: str | None = None
        inference_backend = "heuristic"
        inference_count = 0
        inference_duration_ms = 0.0
        incompatible_count = 0
        inference_failure_count = 0
        deployment_resolution_failures = 0
        calibration_compatibility_failures = 0
        schema_mismatch_failures = 0
        try:
            output_dir: Path = context.require_artifact("output_dir")
            data_dir = output_dir.parents[1] if len(output_dir.parents) >= 2 else output_dir.parent
            ml_service = MLService(data_dir)
            pipeline_id = str(context.get_artifact("pipeline_id", "25d_ball_inspection") or "25d_ball_inspection")
            runtime_resolution = ml_service.resolve_active_deployment(
                pipeline_id=pipeline_id,
                stage_id="classification",
                pipeline_family="25d",
            )
            active_deployment = runtime_resolution.get("deployment") if isinstance(runtime_resolution, dict) else None
            if active_deployment and (runtime_resolution.get("diagnostics") or {}).get("compatible"):
                deployment_model_id = str(active_deployment.get("model_id") or "")
                deployment_id = str(active_deployment.get("id") or "")
                inference_backend = "ml_model"
            else:
                fallback_used = True
                fallback_reason = "incompatible_or_missing_deployment"
                incompatible_count += 1
        except Exception as exc:
            deployment_diagnostic = f"ml_runtime_unavailable: {exc}"
            fallback_used = True
            fallback_reason = "runtime_resolution_failed"
            deployment_resolution_failures += 1
        objects = context.get_artifact("objects", [])
        ball_count = 0
        non_ball = 0
        explanations: list[dict[str, Any]] = []
        for item in objects:
            if "metrics_coordinate_space" not in item:
                item["metrics_coordinate_space"] = "raw_mm"
            if "correction_source" not in item:
                item["correction_source"] = "none"
            ml_predicted = False
            if ml_service is not None and deployment_model_id:
                try:
                    infer_t0 = datetime.now(UTC)
                    feat_row = features_from_object(
                        obj=item,
                        take_id=str(context.get_artifact("take_id", "")),
                        dataset_id=None,
                        session_id=None,
                        labels=[],
                        validation_status=None,
                    )
                    calib_available = context.get_artifact("calibration") is not None
                    runtime_input_check = ml_service.runtime.validate_runtime_inputs(
                        model=(runtime_resolution.get("model") if isinstance(runtime_resolution, dict) else {}) or {},
                        feature_row=feat_row,
                        object_metrics=item,
                        calibration_available=bool(calib_available),
                    )
                    if not runtime_input_check.get("compatible"):
                        errs = [str(v) for v in runtime_input_check.get("errors", [])] if isinstance(runtime_input_check.get("errors"), list) else []
                        if "mm_calibration_required" in errs:
                            calibration_compatibility_failures += 1
                        if "feature_missing" in errs or "feature_dtype_mismatch" in errs:
                            schema_mismatch_failures += 1
                        raise ValueError(f"runtime_input_incompatible: {runtime_input_check}")
                    predicted, probs = ml_service.predict_row(model_id=deployment_model_id, row=feat_row)
                    superclass = predicted
                    label, display_label, class_name, confidence = _labels_for_superclass(superclass, confidence=max(probs.values()) if probs else 0.5)
                    item["rule_path"] = "ml_deployment"
                    item["confidence_proxy"] = confidence
                    ml_predicted = True
                    inference_count += 1
                    inference_duration_ms += (datetime.now(UTC) - infer_t0).total_seconds() * 1000.0
                except Exception as exc:
                    deployment_diagnostic = f"ml_prediction_failed: {exc}"
                    fallback_used = True
                    fallback_reason = "inference_failure"
                    incompatible_count += 1
                    inference_failure_count += 1
            if not ml_predicted and use_external_rules:
                pred = predict_superclass_from_rules(item, rule_params)
                superclass = pred.superclass
                label, display_label, class_name, confidence = _labels_for_superclass(superclass, confidence=pred.confidence_proxy)
                item["rule_path"] = pred.rule_path
                item["confidence_proxy"] = pred.confidence_proxy
            elif not ml_predicted:
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
                item["rule_path"] = "builtin_legacy"
                item["confidence_proxy"] = confidence
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
                "classifier_engine": "mining_steel_ball_classification_25d",
                "model_id": deployment_model_id,
                "model_version": ((runtime_resolution.get("model") or {}).get("version") if isinstance(runtime_resolution.get("model"), dict) else None),
                "feature_schema_version": "1.0.0" if deployment_model_id else None,
                "deployment_id": deployment_id,
                "classifier_backend": inference_backend,
                "split_strategy": ((runtime_resolution.get("model") or {}).get("artifact_paths") or {}).get("split_strategy") if isinstance(runtime_resolution.get("model"), dict) else None,
                "training_experiment_id": (runtime_resolution.get("model") or {}).get("training_experiment_id") if isinstance(runtime_resolution.get("model"), dict) else None,
                "inference_timestamp": datetime.now(UTC).isoformat(),
                "deployment_status": (active_deployment or {}).get("status"),
                "backend_name": ((runtime_resolution.get("model") or {}).get("backend") or {}).get("backend_name") if isinstance(runtime_resolution.get("model"), dict) else None,
                "backend_version": ((runtime_resolution.get("model") or {}).get("backend") or {}).get("backend_version") if isinstance(runtime_resolution.get("model"), dict) else None,
                "feature_schema_hash": feature_ordering_hash(),
                "calibration_compatibility": ((runtime_resolution.get("diagnostics") or {}).get("checks") or {}).get("calibration") if isinstance(runtime_resolution, dict) else None,
                "inference_duration_ms": inference_duration_ms,
                "fallback_used": fallback_used,
                "fallback_reason": fallback_reason,
                "rule_set_id": rule_set_id,
                "rule_set_path": rule_set_path,
                "rule_set_version": rule_set_version,
                "rule_set_source": rule_set_source,
                "rule_path": item.get("rule_path"),
                "confidence_proxy": item.get("confidence_proxy"),
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
            if float(item.get("feature_eccentricity") or 0.0) > 0.85:
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
            ux_summary = build_object_feature_ux_summary(item)
            item["feature_group_summaries"] = ux_summary.get("feature_group_summaries") or []
            item["feature_warnings"] = ux_summary.get("feature_warnings") or []
            item["feature_readiness"] = ux_summary.get("feature_readiness") or {}
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
                        "classifier_engine": "mining_steel_ball_classification_25d",
                        "classifier_backend": inference_backend,
                        "fallback_used": fallback_used,
                        "fallback_reason": fallback_reason,
                        "deployment_id": deployment_id,
                        "model_id": deployment_model_id,
                        "feature_schema_version": "1.0.0" if deployment_model_id else None,
                        "rule_set_id": rule_set_id,
                        "rule_set_path": rule_set_path,
                        "rule_set_version": rule_set_version,
                        "rule_set_source": rule_set_source,
                        "rule_path": item.get("rule_path"),
                    }
                ),
            )
        object_count = len(objects)
        decision = "accept" if object_count > 0 and non_ball == 0 else "review"
        canonical = select_dominant_classification({"objects": objects})
        confidence_values = [
            float(item.get("confidence") or 0.0)
            for item in objects
            if isinstance(item, dict) and item.get("confidence") is not None
        ]
        class_distribution: dict[str, int] = {}
        superclass_distribution: dict[str, int] = {}
        for item in objects:
            if not isinstance(item, dict):
                continue
            class_name = str(item.get("class_name") or item.get("label") or "unknown")
            superclass = str(item.get("superclass") or "unknown")
            class_distribution[class_name] = class_distribution.get(class_name, 0) + 1
            superclass_distribution[superclass] = superclass_distribution.get(superclass, 0) + 1
        sph3d_fallback_count = sum(
            1
            for item in objects
            if isinstance(item, dict) and "sph3d" in str(item.get("classification_reason") or "").lower()
        )
        _runtime_trace_mask_unit(
            context,
            unit_id="classification.primary_heuristic_classifier",
            stage_id="classification",
            parent_id="classification",
            parameters={"rule_set_id": rule_set_id, "rule_set_source": rule_set_source},
            input_artifacts=["measurement_diagnostics", numeric_source_artifact_id],
            output_artifacts=["classification_result_25d"],
            metrics={
                "object_count": int(object_count),
                "inference_count": int(inference_count),
                "inference_duration_ms": float(inference_duration_ms),
                "fallback_used": bool(fallback_used),
                "confidence_min": _numeric_summary(confidence_values).get("min"),
                "confidence_max": _numeric_summary(confidence_values).get("max"),
                "confidence_mean": _numeric_summary(confidence_values).get("mean"),
            },
            diagnostics={"classifier_backend": inference_backend, "fallback_reason": fallback_reason, "class_distribution": class_distribution},
            warnings=[deployment_diagnostic] if deployment_diagnostic else [],
        )
        _runtime_trace_mask_unit(
            context,
            unit_id="classification.sph3d_fallback",
            stage_id="classification",
            parent_id="classification",
            status="skipped" if sph3d_fallback_count == 0 else "completed",
            input_artifacts=["classification_result_25d"],
            output_artifacts=["classification_runtime_diagnostics"],
            metrics={"fallback_application_count": int(sph3d_fallback_count)},
            diagnostics={"classifier_backend": inference_backend},
        )
        _runtime_trace_mask_unit(
            context,
            unit_id="classification.superclass_aggregation",
            stage_id="classification",
            parent_id="classification",
            input_artifacts=["classification_result_25d"],
            output_artifacts=["classification_result_25d"],
            metrics={
                "object_count": int(object_count),
                "ball_count": int(ball_count),
                "non_ball_count": int(non_ball),
                "confidence": canonical.get("confidence"),
            },
            diagnostics={"label": canonical.get("label"), "superclass": canonical.get("superclass"), "superclass_distribution": superclass_distribution},
        )
        object_feature_payloads = [
            {
                "feature_group_summaries": item.get("feature_group_summaries") or [],
                "feature_warnings": item.get("feature_warnings") or [],
                "feature_readiness": item.get("feature_readiness") or {},
            }
            for item in objects
            if isinstance(item, dict)
        ]
        summary_feature_runtime = aggregate_operations_summary(object_feature_payloads)
        summary_feature_readiness = {
            "overall_readiness": "UNKNOWN" if not object_feature_payloads else (
                "EXPERIMENTAL"
                if any(str((payload.get("feature_readiness") or {}).get("overall_readiness") or "") == "EXPERIMENTAL" for payload in object_feature_payloads)
                else "GOOD"
                if any(str((payload.get("feature_readiness") or {}).get("overall_readiness") or "") == "GOOD" for payload in object_feature_payloads)
                else "PRODUCTION_READY"
            ),
        }
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
                "feature_runtime_summary": summary_feature_runtime,
                "feature_readiness": summary_feature_readiness,
            },
        )
        context.set_artifact(
            "classification_result_25d",
            {
                "label": canonical.get("label"),
                "superclass": canonical.get("superclass"),
                "confidence": canonical.get("confidence"),
                "object_count": object_count,
                "classifier_engine": "mining_steel_ball_classification_25d",
                "model_id": deployment_model_id,
                "model_version": ((runtime_resolution.get("model") or {}).get("version") if isinstance(runtime_resolution.get("model"), dict) else None),
                "feature_schema_version": "1.0.0" if deployment_model_id else None,
                "deployment_id": deployment_id,
                "classifier_backend": inference_backend,
                "split_strategy": ((runtime_resolution.get("model") or {}).get("artifact_paths") or {}).get("split_strategy") if isinstance(runtime_resolution.get("model"), dict) else None,
                "training_experiment_id": (runtime_resolution.get("model") or {}).get("training_experiment_id") if isinstance(runtime_resolution.get("model"), dict) else None,
                "inference_timestamp": datetime.now(UTC).isoformat(),
                "deployment_status": (active_deployment or {}).get("status"),
                "backend_name": ((runtime_resolution.get("model") or {}).get("backend") or {}).get("backend_name") if isinstance(runtime_resolution.get("model"), dict) else None,
                "backend_version": ((runtime_resolution.get("model") or {}).get("backend") or {}).get("backend_version") if isinstance(runtime_resolution.get("model"), dict) else None,
                "feature_schema_hash": feature_ordering_hash(),
                "calibration_compatibility": ((runtime_resolution.get("diagnostics") or {}).get("checks") or {}).get("calibration") if isinstance(runtime_resolution, dict) else None,
                "inference_duration_ms": inference_duration_ms,
                "fallback_used": fallback_used,
                "fallback_reason": fallback_reason,
                "rule_set_id": rule_set_id,
                "rule_set_path": rule_set_path,
                "rule_set_version": rule_set_version,
                "rule_set_source": rule_set_source,
                "objects": [
                    {
                        "object_id": item.get("object_id"),
                        "label": item.get("label"),
                        "display_label": item.get("display_label"),
                        "superclass": item.get("superclass"),
                        "confidence": item.get("confidence"),
                        "rule_path": item.get("rule_path"),
                        "confidence_proxy": item.get("confidence_proxy"),
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
                    "classifier_engine": "mining_steel_ball_classification_25d",
                    "classifier_backend": inference_backend,
                    "fallback_used": fallback_used,
                    "fallback_reason": fallback_reason,
                    "model_id": deployment_model_id,
                    "deployment_id": deployment_id,
                    "deployment_diagnostic": deployment_diagnostic,
                    "rule_set_id": rule_set_id,
                    "rule_set_path": rule_set_path,
                    "rule_set_version": rule_set_version,
                    "rule_set_source": rule_set_source,
                }
            ),
        )
        output_dir: Path = context.require_artifact("output_dir")
        explanation_payload = {
            "stage": "classification",
            "artifact_id": "classification_explanation",
            "scope": "global",
            "feature_runtime_summary": summary_feature_runtime,
            "feature_readiness": summary_feature_readiness,
            "objects": [
                {
                    **row,
                    "feature_group_summaries": next((item.get("feature_group_summaries") for item in objects if int(item.get("object_id", 0) or 0) == int(row.get("object_id", 0) or 0)), []),
                    "feature_warnings": next((item.get("feature_warnings") for item in objects if int(item.get("object_id", 0) or 0) == int(row.get("object_id", 0) or 0)), []),
                    "feature_readiness": next((item.get("feature_readiness") for item in objects if int(item.get("object_id", 0) or 0) == int(row.get("object_id", 0) or 0)), {}),
                }
                for row in explanations
            ],
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
        per_object_ux_payload = []
        for item in objects:
            ux_payload = {
                "object_id": item.get("object_id"),
                "feature_group_summaries": item.get("feature_group_summaries") or [],
                "feature_warnings": item.get("feature_warnings") or [],
                "feature_readiness": item.get("feature_readiness") or {},
            }
            per_object_ux_payload.append(
                {
                    "object_id": item.get("object_id"),
                    "operations": filter_for_operations(ux_payload),
                    "studio": filter_for_studio(ux_payload),
                    "classifier_studio": filter_for_classifier_studio(ux_payload),
                }
            )
        runtime_summary_payload = {
            "stage": "classification",
            "artifact_id": "feature_runtime_summary",
            "scope": "per_object",
            "summary": summary_feature_runtime,
            "objects": [{"object_id": row.get("object_id"), "operations": row.get("operations")} for row in per_object_ux_payload],
        }
        studio_summary_payload = {
            "stage": "classification",
            "artifact_id": "feature_studio_summary",
            "scope": "per_object",
            "feature_readiness": summary_feature_readiness,
            "objects": [
                {
                    "object_id": row.get("object_id"),
                    "studio": row.get("studio"),
                    "classifier_studio": row.get("classifier_studio"),
                }
                for row in per_object_ux_payload
            ],
        }
        (output_dir / "classification_explanation.json").write_text(json.dumps(explanation_payload, indent=2), encoding="utf-8")
        (output_dir / "metric_explanation.json").write_text(json.dumps(metric_payload, indent=2), encoding="utf-8")
        (output_dir / "feature_runtime_summary.json").write_text(json.dumps(runtime_summary_payload, indent=2), encoding="utf-8")
        (output_dir / "feature_studio_summary.json").write_text(json.dumps(studio_summary_payload, indent=2), encoding="utf-8")
        runtime_diag_payload = {
            "stage": "classification",
            "artifact_id": "classification_runtime_diagnostics",
            "deployment_resolution": runtime_resolution,
            "compatibility_checks": (runtime_resolution.get("diagnostics") if isinstance(runtime_resolution, dict) else {}),
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "deployment_diagnostic": deployment_diagnostic,
            "feature_schema_used": "1.0.0",
            "feature_schema_hash": feature_ordering_hash(),
            "inference_backend_used": inference_backend,
            "inference_duration_ms": inference_duration_ms,
            "execution_timing": {"status": "captured_by_stage_profiler"},
        }
        _runtime_trace_mask_unit(
            context,
            unit_id="classification.explanation_generation",
            stage_id="classification",
            parent_id="classification",
            input_artifacts=["classification_result_25d"],
            output_artifacts=["classification_explanation", "metric_explanation", "classification_runtime_diagnostics"],
            metrics={"explanation_object_count": int(len(explanations)), "warning_count": int(len([item for item in [deployment_diagnostic] if item]))},
            diagnostics={"feature_schema_hash": feature_ordering_hash(), "rule_set_id": rule_set_id},
            warnings=[deployment_diagnostic] if deployment_diagnostic else [],
        )
        _runtime_trace_mask_unit(
            context,
            unit_id="classification",
            stage_id="classification",
            input_artifacts=["feature_vector"],
            output_artifacts=["classification_result_25d", "classification_explanation", "classification_runtime_diagnostics"],
            metrics={
                "object_count": int(object_count),
                "fallback_application_count": int(sph3d_fallback_count),
                "confidence_min": _numeric_summary(confidence_values).get("min"),
                "confidence_max": _numeric_summary(confidence_values).get("max"),
                "confidence_mean": _numeric_summary(confidence_values).get("mean"),
            },
            diagnostics={
                "dominant_superclass": canonical.get("superclass"),
                "dominant_label": canonical.get("label"),
                "class_distribution": class_distribution,
                "superclass_distribution": superclass_distribution,
            },
            warnings=[deployment_diagnostic] if deployment_diagnostic else [],
        )
        (output_dir / "classification_runtime_diagnostics.json").write_text(json.dumps(runtime_diag_payload, indent=2), encoding="utf-8")
        files = dict(context.get_artifact("files", {}))
        files["classification_explanation"] = "classification_explanation.json"
        files["metric_explanation"] = "metric_explanation.json"
        files["classification_runtime_diagnostics"] = "classification_runtime_diagnostics.json"
        files["feature_runtime_summary"] = "feature_runtime_summary.json"
        files["feature_studio_summary"] = "feature_studio_summary.json"
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
            artifact_id="classification_runtime_diagnostics",
            stage_id="classification",
            kind="json",
            title="Classification runtime diagnostics",
            path="classification_runtime_diagnostics.json",
            mime_type="application/json",
            preview_available=True,
            metadata=_numeric_source_meta(
                {
                    "source": "native_pipeline",
                    "producer": self.name,
                    "semantic_type": "classification_runtime_diagnostics",
                    "fallback_used": fallback_used,
                    "fallback_reason": fallback_reason,
                    "deployment_id": deployment_id,
                }
            ),
        )
        context.add_processing_artifact(
            artifact_id="feature_runtime_summary",
            stage_id="classification",
            kind="json",
            title="Feature runtime summary",
            path="feature_runtime_summary.json",
            mime_type="application/json",
            preview_available=True,
            metadata=_numeric_source_meta(
                {
                    "source": "native_pipeline",
                    "producer": self.name,
                    "semantic_type": "feature_runtime_summary",
                    "audience": "operations",
                    "object_count": len(objects),
                }
            ),
        )
        context.add_processing_artifact(
            artifact_id="feature_studio_summary",
            stage_id="classification",
            kind="json",
            title="Feature studio summary",
            path="feature_studio_summary.json",
            mime_type="application/json",
            preview_available=True,
            metadata=_numeric_source_meta(
                {
                    "source": "native_pipeline",
                    "producer": self.name,
                    "semantic_type": "feature_studio_summary",
                    "audience": "studio_classifier_studio",
                    "object_count": len(objects),
                }
            ),
        )
        try:
            stats_storage = MLStorage(data_dir)
            day_key = datetime.now(UTC).strftime("%Y%m%d")
            stats_rel = f"runtime_stats/{day_key}.json"
            stats_path = stats_storage.root / stats_rel
            if stats_path.is_file():
                stats = json.loads(stats_path.read_text(encoding="utf-8"))
            else:
                stats = {
                    "inference_count": 0,
                    "inference_success_count": 0,
                    "inference_failure_count": 0,
                    "fallback_count": 0,
                    "incompatible_deployment_count": 0,
                    "heuristic_usage_count": 0,
                    "deployment_resolution_failures": 0,
                    "calibration_compatibility_failures": 0,
                    "schema_mismatch_failures": 0,
                    "inference_time_total_ms": 0.0,
                }
            stats["inference_count"] = int(stats.get("inference_count", 0)) + int(inference_count)
            stats["inference_success_count"] = int(stats.get("inference_success_count", 0)) + int(max(0, inference_count - inference_failure_count))
            stats["inference_failure_count"] = int(stats.get("inference_failure_count", 0)) + int(inference_failure_count)
            stats["fallback_count"] = int(stats.get("fallback_count", 0)) + int(1 if fallback_used else 0)
            stats["incompatible_deployment_count"] = int(stats.get("incompatible_deployment_count", 0)) + int(incompatible_count)
            stats["heuristic_usage_count"] = int(stats.get("heuristic_usage_count", 0)) + int(1 if inference_backend == "heuristic" else 0)
            stats["deployment_resolution_failures"] = int(stats.get("deployment_resolution_failures", 0)) + int(deployment_resolution_failures)
            stats["calibration_compatibility_failures"] = int(stats.get("calibration_compatibility_failures", 0)) + int(calibration_compatibility_failures)
            stats["schema_mismatch_failures"] = int(stats.get("schema_mismatch_failures", 0)) + int(schema_mismatch_failures)
            stats["inference_time_total_ms"] = float(stats.get("inference_time_total_ms", 0.0)) + float(inference_duration_ms)
            samples = list(stats.get("inference_latency_samples_ms") or [])
            samples.append(float(inference_duration_ms))
            stats["inference_latency_samples_ms"] = samples[-512:]
            total_inf = max(1, int(stats.get("inference_count", 0)))
            stats["average_inference_time_ms"] = float(stats["inference_time_total_ms"]) / float(total_inf)
            stats_storage.write_artifact(stats_rel, stats)
        except Exception:
            pass
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
        overlay_width = int(normalized.shape[1]) if normalized.ndim >= 2 else 0
        overlay_height = int(normalized.shape[0]) if normalized.ndim >= 2 else 0
        _runtime_trace_mask_unit(
            context,
            unit_id="overlay.height_overlay",
            stage_id="overlay",
            parent_id="overlay",
            input_artifacts=["normalized_heightmap"],
            output_artifacts=["height_overlay"],
            metrics={
                "overlay_width_px": overlay_width,
                "overlay_height_px": overlay_height,
                "object_annotation_count": int(len(objects) if isinstance(objects, list) else 0),
            },
        )
        _runtime_trace_mask_unit(
            context,
            unit_id="overlay.measurement_overlay",
            stage_id="overlay",
            parent_id="overlay",
            input_artifacts=["height_overlay", "final_object_mask"],
            output_artifacts=["measurement_overlay"],
            metrics={
                "overlay_width_px": overlay_width,
                "overlay_height_px": overlay_height,
                "object_annotation_count": int(len(objects) if isinstance(objects, list) else 0),
            },
        )
        _runtime_trace_mask_unit(
            context,
            unit_id="overlay.classification_overlay",
            stage_id="overlay",
            parent_id="overlay",
            input_artifacts=["measurement_overlay", "classification_result_25d"],
            output_artifacts=["classification_overlay"],
            metrics={
                "overlay_width_px": overlay_width,
                "overlay_height_px": overlay_height,
                "classification_overlay_object_count": int(len(objects) if isinstance(objects, list) else 0),
            },
        )
        _runtime_trace_mask_unit(
            context,
            unit_id="overlay.summary_overlay_metadata",
            stage_id="overlay",
            parent_id="overlay",
            input_artifacts=["classification_result_25d"],
            output_artifacts=["classification_overlay_metadata"],
            metrics={
                "overlay_file_count": 4,
                "object_count": int(len(objects) if isinstance(objects, list) else 0),
            },
            diagnostics={"classification_keys": sorted(list(classification_result.keys())) if isinstance(classification_result, dict) else []},
        )
        _runtime_trace_mask_unit(
            context,
            unit_id="overlay",
            stage_id="overlay",
            input_artifacts=["normalized_heightmap", "final_object_mask", "classification_result_25d"],
            output_artifacts=["height_overlay", "measurement_overlay", "classification_overlay", "classification_overlay_metadata"],
            metrics={
                "overlay_width_px": overlay_width,
                "overlay_height_px": overlay_height,
                "overlay_files_emitted": 4,
                "object_annotation_count": int(len(objects) if isinstance(objects, list) else 0),
            },
        )


def _apply_heightmap_valid_mask_rules(
    z_mm: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Drop non-finite and non-positive z values from the valid mask.

    TriSpector height images encode no-return as 0. Acquisition already
    treats those as invalid; mirror that rule whenever a heightmap frame is
    loaded so diagnostics never treat sensor holes as real belt points.
    """
    z = np.asarray(z_mm, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(z) & (z > 0.0)
    return valid


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
        frame = HeightmapFrame(
            z_mm=frame.z_mm,
            valid_mask=_apply_heightmap_valid_mask_rules(frame.z_mm, frame.valid_mask),
            reflectance=frame.reflectance,
            x_resolution_mm=frame.x_resolution_mm,
            y_resolution_mm=frame.y_resolution_mm,
            origin_x_mm=frame.origin_x_mm,
            origin_y_mm=frame.origin_y_mm,
            coordinate_system=frame.coordinate_system,
        )
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
        valid = _apply_heightmap_valid_mask_rules(z, np.isfinite(z))
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
    gap_floor_mm: float = 1.0,
    gap_ratio: float = 8.0,
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
    measured_gap_ratio = 0.0
    if search_gaps.size:
        local_gap_idx = int(np.argmax(search_gaps))
        gap_idx = gap_start + local_gap_idx
        robust_gap_scale = float(np.median(gap_values[gap_values > 0])) if np.any(gap_values > 0) else 0.0
        largest_gap = float(gap_values[gap_idx])
        measured_gap_ratio = largest_gap / max(robust_gap_scale, 1e-6)
        if largest_gap >= max(float(gap_floor_mm), robust_gap_scale * float(gap_ratio)):
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
            "largest_gap_ratio": float(measured_gap_ratio),
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


def _compute_hessian_response(z_mm: np.ndarray, valid_mask: np.ndarray, *, smoothing_kernel: int) -> np.ndarray:
    z = np.asarray(z_mm, dtype=np.float32).copy()
    z[~valid_mask] = 0.0
    k = max(1, int(smoothing_kernel))
    if k > 1:
        if k % 2 == 0:
            k += 1
        z = cv2.GaussianBlur(z, (k, k), 0)
    dxx = cv2.Sobel(z, cv2.CV_32F, 2, 0, ksize=3)
    dyy = cv2.Sobel(z, cv2.CV_32F, 0, 2, ksize=3)
    dxy = cv2.Sobel(z, cv2.CV_32F, 1, 1, ksize=3)
    response = np.sqrt((dxx * dxx) + (dyy * dyy) + (2.0 * dxy * dxy))
    response[~valid_mask] = 0.0
    return response


def _compute_belt_stripe_filter(
    *,
    z_mm: np.ndarray,
    domain_mask: np.ndarray,
    pixel_size_mm: float,
    window_mm: float,
    direction: str,
    threshold_mode: str,
    min_altitude_mm: float,
    k_mad: float,
    fixed_threshold_mm: float,
    min_stripe_fraction: float,
    bg_plateau_mask: np.ndarray | None = None,
    z_floor_enabled: bool = False,
    z_floor_use_upper_bound: bool = True,
    z_floor_margin_mm: float = 20.0,
    max_stripe_height_mm: float = 500.0,
    above_belt_close_mm: float = 30.0,
    object_kernel_mm: float = 100.0,
    object_kernel_shape: str = "ellipse",
    altitude_hist_bins: int = 64,
    auto_bimodality_margin: float = 1.10,
    z_floor_upper_percentile: float = 99.0,
    z_floor_fallback_lower_percentile: float = 10.0,
    z_floor_fallback_upper_percentile: float = 25.0,
) -> dict[str, Any]:
    """Separate belt-base pixels from raised surface texture (e.g. chevron ribs).

    Strategy: local-min top-hat (raised) or local-max bottom-hat (recessed)
    over ``domain_mask``. Baseline is always the erosion/dilation phase only
    (no opening/closing follow-up), so for raised mode ``baseline[i]`` is the
    minimum ``z`` in the window and never exceeds ``z[i]`` or neighbours in
    that window. Narrow raised features such as chevron ribs (width < window)
    survive as positive altitude above the local belt minimum.

    Workflow:
        baseline = erode(z, W)                  # raised
                 = dilate(z, W)                 # recessed
        altitude = z − baseline                 # raised
                 = baseline − z                 # recessed
    Then threshold ``altitude`` (Otsu / k·MAD / fixed, clipped to
    ``min_altitude_mm``) to label stripes vs belt base. With
    ``direction="auto"`` both modes are computed and the one with the more
    bimodal-looking altitude tail (p95 − p50) is chosen.

    Returns a dict with all intermediate arrays (heightmap-shaped) and stats.
    Output arrays are zeroed outside ``domain_mask`` so they're safe to render.
    """
    h, w = z_mm.shape[:2]
    dom_raw = np.asarray(domain_mask, dtype=bool)
    # Defensive guard: exclude non-finite / non-positive z (TriSpector
    # no-return holes) from the morphology domain. Without this, an erode
    # window touching a z=0 pixel would set baseline=0 and altitude≈z,
    # producing a fake "0 .. ~12k mm" altitude range that misleads stripe
    # detection. LoadHeightmapCapture also strips these at load time; this
    # is belt-and-braces in case a stale or upstream mask leaks them in.
    z_in_for_validity = np.asarray(z_mm, dtype=np.float32)
    dom_validity = np.isfinite(z_in_for_validity) & (z_in_for_validity > 0.0)
    dom = dom_raw & dom_validity
    dropped_invalid_z = int(np.count_nonzero(dom_raw & (~dom_validity)))
    n_dom = int(np.count_nonzero(dom))
    base_skeleton: dict[str, Any] = {
        "enabled": True,
        "applied": False,
        "skip_reason": None,
        "pixel_size_mm": float(pixel_size_mm),
        "window_mm": float(window_mm),
        "window_px": 0,
        "direction": str(direction),
        "direction_used": None,
        "threshold_mode": str(threshold_mode),
        "threshold_value_mm": 0.0,
        "min_altitude_mm": float(min_altitude_mm),
        # Aliases kept for backward-compat with existing JSON consumers
        # (the field "bg_plateau_count_px" predates the global-domain rename).
        "domain_count_px": n_dom,
        "domain_input_count_px": int(np.count_nonzero(dom_raw)),
        "domain_dropped_invalid_z_px": dropped_invalid_z,
        "bg_plateau_count_px": n_dom,
        "belt_base_count_px": n_dom,
        "stripes_count_px": 0,
        "stripes_fraction": 0.0,
        "stripes_fraction_of_bg": 0.0,
        "altitude_p50_mm": 0.0,
        "altitude_p95_mm": 0.0,
        "altitude_max_mm": 0.0,
        "raised_bimodality": 0.0,
        "recessed_bimodality": 0.0,
    }
    belt_base_mask = dom.copy()
    stripes_mask = np.zeros_like(dom)
    baseline_map = np.zeros((h, w), dtype=np.float32)
    altitude_map = np.zeros((h, w), dtype=np.float32)

    if n_dom == 0:
        base_skeleton["skip_reason"] = "no_domain_pixels"
        return {
            "info": base_skeleton,
            "belt_base_mask": belt_base_mask,
            "stripes_mask": stripes_mask,
            "baseline_map": baseline_map,
            "altitude_map": altitude_map,
            "altitude_histogram": {"sample_count": 0, "bin_edges": [], "bin_counts": []},
        }

    ps = float(pixel_size_mm) if pixel_size_mm and pixel_size_mm > 0 else 1.0
    window_px = max(3, int(round(float(window_mm) / ps)))
    if window_px % 2 == 0:
        window_px += 1
    base_skeleton["window_px"] = int(window_px)

    z_in = np.asarray(z_mm, dtype=np.float32)

    def _hat(mode: str) -> tuple[np.ndarray, np.ndarray, float]:
        """Compute (baseline, altitude, bimodality_score) for "raised"/"recessed".

        Pixels outside ``dom`` are made never-winning so morph ops are
        constrained to valid neighbours: +inf for erode (min), -inf for
        dilate (max). After each op we restore z-values outside ``dom`` so
        subsequent ops on the same array don't propagate sentinels.
        """
        kernel = np.ones((window_px, window_px), dtype=np.uint8)
        if mode == "raised":
            big = np.full_like(z_in, np.finfo(np.float32).max)
            big[dom] = z_in[dom]
            eroded = cv2.erode(big, kernel, borderType=cv2.BORDER_REPLICATE)
            sentinel_mask = (~np.isfinite(eroded)) | (eroded >= np.finfo(np.float32).max * 0.5)
            base = np.where(sentinel_mask, z_in, eroded)
            alt = np.zeros_like(z_in)
            alt[dom] = z_in[dom] - base[dom]
        else:  # recessed
            small = np.full_like(z_in, np.finfo(np.float32).min)
            small[dom] = z_in[dom]
            dilated = cv2.dilate(small, kernel, borderType=cv2.BORDER_REPLICATE)
            sentinel_mask = (~np.isfinite(dilated)) | (dilated <= np.finfo(np.float32).min * 0.5)
            base = np.where(sentinel_mask, z_in, dilated)
            alt = np.zeros_like(z_in)
            alt[dom] = base[dom] - z_in[dom]
        # Bimodality score: spread between bulk and tail (p95 − p50). Bigger
        # = altitude distribution has a clear "raised" tail → more striped.
        vals = alt[dom]
        if vals.size == 0:
            return base, alt, 0.0
        p50 = float(np.percentile(vals, 50))
        p95 = float(np.percentile(vals, 95))
        bm = float(p95 - p50)
        return base, alt, bm

    direction_norm = str(direction or "auto").strip().lower()
    chosen = direction_norm
    bm_raised = bm_recessed = 0.0
    bimodality_margin = float(max(1.0, auto_bimodality_margin))
    if direction_norm == "auto":
        base_r, alt_r, bm_raised = _hat("raised")
        base_d, alt_d, bm_recessed = _hat("recessed")
        if bm_recessed > bm_raised * bimodality_margin:
            chosen = "recessed"
            baseline_map[:] = base_d
            altitude_map[:] = alt_d
        else:
            chosen = "raised"
            baseline_map[:] = base_r
            altitude_map[:] = alt_r
    elif direction_norm == "recessed":
        baseline_map, altitude_map, bm_recessed = _hat("recessed")
        chosen = "recessed"
    else:
        chosen = "raised"
        baseline_map, altitude_map, bm_raised = _hat("raised")

    base_skeleton["direction_used"] = str(chosen)
    base_skeleton["raised_bimodality"] = float(bm_raised)
    base_skeleton["recessed_bimodality"] = float(bm_recessed)

    alt_vals_dom = altitude_map[dom]
    base_skeleton["altitude_p50_mm"] = float(np.percentile(alt_vals_dom, 50)) if alt_vals_dom.size else 0.0
    base_skeleton["altitude_p95_mm"] = float(np.percentile(alt_vals_dom, 95)) if alt_vals_dom.size else 0.0
    base_skeleton["altitude_max_mm"] = float(np.max(alt_vals_dom)) if alt_vals_dom.size else 0.0

    thr_mode = str(threshold_mode or "otsu").strip().lower()
    if thr_mode == "fixed":
        thr = float(fixed_threshold_mm)
    elif thr_mode == "k_mad":
        med = float(np.median(alt_vals_dom))
        mad = float(np.median(np.abs(alt_vals_dom - med)))
        thr = med + float(k_mad) * (mad if mad > 0 else 0.1)
    else:  # otsu
        thr = _otsu_threshold(alt_vals_dom)
    thr = max(float(thr), float(min_altitude_mm))
    base_skeleton["threshold_value_mm"] = float(thr)

    candidate_stripes_tophat = dom & (altitude_map > thr)
    tophat_count = int(np.count_nonzero(candidate_stripes_tophat))

    # ── Shape-based pass (z-floor + binary opening) ────────────────────
    # Catches a failure mode of the erosion top-hat:
    #   1. A stripe that grows wider than the kernel: erosion baseline ≈
    #      stripe_z inside the wide rib, altitude ≈ 0 → top-hat misses it.
    # Recipe:
    #   belt_z_upper = upper bound of BG plateau z (max, or p99 — picked so
    #                  the entire belt distribution sits BELOW it; using the
    #                  median instead falsely flags upper-tail belt pixels
    #                  in tilted/curved areas).
    #   above_belt   = (z > belt_z_upper + margin) AND
    #                  (z < belt_z_upper + max_stripe_height OR cap disabled)
    #                  ← upper cap excludes tall foreground objects whose
    #                  height greatly exceeds typical stripe heights.
    #   smoothed     = morph_CLOSE(above_belt, close_kernel)  ← bridges
    #                  small interior holes/gaps inside an object so the
    #                  opening doesn't shave thin tails off it.
    #   wide_objects = morph_OPEN(smoothed, object_kernel)    ← things wider
    #                  than the object kernel survive (= true objects).
    #                  Ellipse kernel is smoother on round/irregular shapes
    #                  than rect; configurable.
    #   shape_stripes = above_belt AND NOT wide_objects.
    above_belt_mask = np.zeros_like(dom)
    wide_object_mask = np.zeros_like(dom)
    extra_stripes_shape = np.zeros_like(dom)
    belt_z_estimate: float | None = None    # median, kept for diagnostics
    belt_z_upper: float | None = None       # upper bound used as the actual threshold reference
    shape_pass_applied = False
    shape_skip_reason: str | None = None
    threshold_low_mm: float | None = None
    threshold_high_mm: float | None = None
    bg_input: np.ndarray | None = None
    if bg_plateau_mask is not None:
        bg_input = np.asarray(bg_plateau_mask, dtype=bool) & dom_validity
        # The BG plateau may already have been narrowed by an upstream caller.
        # Use whatever was passed; if empty, fall back to the domain itself
        # to keep belt_z_estimate well-defined for diagnostics.
        if int(np.count_nonzero(bg_input)) == 0:
            bg_input = None
    upper_pct = float(np.clip(z_floor_upper_percentile, 50.0, 100.0))
    fb_lower_pct = float(np.clip(z_floor_fallback_lower_percentile, 0.0, 50.0))
    fb_upper_pct = float(np.clip(z_floor_fallback_upper_percentile, fb_lower_pct, 100.0))
    if bool(z_floor_enabled):
        if bg_input is not None and int(np.count_nonzero(bg_input)) > 0:
            belt_z_estimate = float(np.median(z_in[bg_input]))
            # Use a robust upper percentile (default p99) instead of max so a
            # few sentinel/outlier samples that slipped through gradient
            # filtering don't push the reference too high.
            belt_z_upper = float(np.percentile(z_in[bg_input], upper_pct))
        elif n_dom > 0:
            # No BG plateau available — estimate belt-z from the lowest decile
            # of the domain (a defensive fallback; assumes most of the domain
            # is belt).
            belt_z_estimate = float(np.percentile(z_in[dom], fb_lower_pct))
            belt_z_upper = float(np.percentile(z_in[dom], fb_upper_pct))
            shape_skip_reason = (
                f"no_bg_plateau_used_p{fb_lower_pct:g}_p{fb_upper_pct:g}_of_domain"
            )
        if belt_z_estimate is not None and chosen == "raised":
            # Reference threshold is on the plateau's upper bound by default;
            # users can revert to median behaviour by setting
            # z_floor_use_upper_bound=False.
            reference_z = belt_z_upper if (z_floor_use_upper_bound and belt_z_upper is not None) else belt_z_estimate
            threshold_low_mm = float(reference_z) + float(z_floor_margin_mm)
            above_belt_mask = dom & (z_in > threshold_low_mm)
            if float(max_stripe_height_mm) > 0:
                threshold_high_mm = float(reference_z) + float(max_stripe_height_mm)
                above_belt_mask = above_belt_mask & (z_in < threshold_high_mm)

            above_u8 = (above_belt_mask.astype(np.uint8) * 255)

            # Close small interior gaps before the object-vs-stripe opening,
            # so objects with internal noise/occlusion don't get shaved into
            # narrow tails that look like stripes.
            close_mm = float(above_belt_close_mm)
            if close_mm > 0:
                close_k_px = max(3, int(round(close_mm / ps)))
                if close_k_px % 2 == 0:
                    close_k_px += 1
                shape_const = (
                    cv2.MORPH_ELLIPSE if str(object_kernel_shape or "ellipse").lower() == "ellipse"
                    else cv2.MORPH_RECT
                )
                close_k = cv2.getStructuringElement(shape_const, (close_k_px, close_k_px))
                above_u8 = cv2.morphologyEx(above_u8, cv2.MORPH_CLOSE, close_k)

            # Object-sized binary opening (everything bigger than the kernel
            # survives → treat it as an object, not a stripe).
            obj_kernel_px = max(3, int(round(float(object_kernel_mm) / ps)))
            if obj_kernel_px % 2 == 0:
                obj_kernel_px += 1
            shape_const = (
                cv2.MORPH_ELLIPSE if str(object_kernel_shape or "ellipse").lower() == "ellipse"
                else cv2.MORPH_RECT
            )
            obj_kernel = cv2.getStructuringElement(shape_const, (obj_kernel_px, obj_kernel_px))
            opened_u8 = cv2.morphologyEx(above_u8, cv2.MORPH_OPEN, obj_kernel)
            # Note: wide_object_mask is intersected back with above_belt_mask
            # so we don't leak the morphological "bridge" pixels (from the
            # close step) into the final object diagnostic.
            wide_object_mask = (opened_u8 > 0) & above_belt_mask
            extra_stripes_shape = above_belt_mask & (~wide_object_mask)
            shape_pass_applied = True
        elif chosen != "raised":
            # For recessed mode the symmetric "below belt" classification
            # would require z < belt_z − margin; not implemented yet because
            # all current real-world scenes use raised ribs.
            shape_skip_reason = f"shape_pass_only_supports_raised_(direction_used={chosen})"
        else:
            shape_skip_reason = "no_belt_z_estimate_available"
    else:
        shape_skip_reason = "disabled"

    shape_extra_count = int(np.count_nonzero(extra_stripes_shape))
    candidate_stripes = candidate_stripes_tophat | extra_stripes_shape
    stripes_count = int(np.count_nonzero(candidate_stripes))
    stripes_fraction = float(stripes_count / max(1, n_dom))
    base_skeleton["stripes_count_px"] = stripes_count
    base_skeleton["stripes_count_tophat"] = tophat_count
    base_skeleton["stripes_count_shape_pass"] = shape_extra_count
    base_skeleton["stripes_fraction"] = stripes_fraction
    base_skeleton["stripes_fraction_of_bg"] = stripes_fraction  # legacy alias
    base_skeleton["z_floor_applied"] = bool(shape_pass_applied)
    base_skeleton["z_floor_skip_reason"] = shape_skip_reason
    base_skeleton["z_floor_belt_z_median_mm"] = (
        float(belt_z_estimate) if belt_z_estimate is not None else None
    )
    base_skeleton["z_floor_belt_z_upper_mm"] = (
        float(belt_z_upper) if belt_z_upper is not None else None
    )
    base_skeleton["z_floor_use_upper_bound"] = bool(z_floor_use_upper_bound)
    base_skeleton["z_floor_belt_z_mm"] = (
        # Reference value actually used as the threshold base. Kept under the
        # legacy key for backward-compat with consumers reading this field.
        float(belt_z_upper if (z_floor_use_upper_bound and belt_z_upper is not None) else (belt_z_estimate or 0.0))
        if belt_z_estimate is not None else None
    )
    base_skeleton["z_floor_margin_mm"] = float(z_floor_margin_mm)
    base_skeleton["z_floor_max_stripe_height_mm"] = float(max_stripe_height_mm)
    base_skeleton["z_floor_above_belt_close_mm"] = float(above_belt_close_mm)
    base_skeleton["z_floor_object_kernel_mm"] = float(object_kernel_mm)
    base_skeleton["z_floor_object_kernel_shape"] = str(object_kernel_shape)
    base_skeleton["z_floor_threshold_low_mm"] = (
        float(threshold_low_mm) if threshold_low_mm is not None else None
    )
    base_skeleton["z_floor_threshold_high_mm"] = (
        float(threshold_high_mm) if threshold_high_mm is not None else None
    )
    base_skeleton["z_floor_above_belt_count_px"] = int(np.count_nonzero(above_belt_mask))
    base_skeleton["z_floor_wide_object_count_px"] = int(np.count_nonzero(wide_object_mask))

    if stripes_fraction < float(min_stripe_fraction):
        base_skeleton["applied"] = False
        base_skeleton["skip_reason"] = f"stripes_fraction_below_min ({stripes_fraction:.4f} < {min_stripe_fraction})"
        belt_base_mask = dom.copy()
        stripes_mask = np.zeros_like(dom)
    else:
        base_skeleton["applied"] = True
        stripes_mask = candidate_stripes
        belt_base_mask = dom & (~stripes_mask)
        base_skeleton["belt_base_count_px"] = int(np.count_nonzero(belt_base_mask))

    hist_bins = max(8, int(altitude_hist_bins))
    hist_counts, hist_edges = (
        np.histogram(alt_vals_dom, bins=hist_bins)
        if alt_vals_dom.size
        else (np.zeros(hist_bins, dtype=np.int64), np.linspace(0, 1, hist_bins + 1))
    )
    altitude_hist = {
        "sample_count": int(alt_vals_dom.size),
        "bin_counts": [int(x) for x in hist_counts.tolist()],
        "bin_edges": [float(x) for x in hist_edges.tolist()],
        "threshold_value_mm": float(thr),
        "min_altitude_mm": float(min_altitude_mm),
        "direction_used": str(chosen),
    }

    return {
        "info": base_skeleton,
        "belt_base_mask": belt_base_mask,
        "stripes_mask": stripes_mask,
        "baseline_map": baseline_map,
        "altitude_map": altitude_map,
        "altitude_histogram": altitude_hist,
        "tophat_stripes_mask": candidate_stripes_tophat,
        "shape_stripes_mask": extra_stripes_shape,
        "above_belt_mask": above_belt_mask,
        "wide_object_mask": wide_object_mask,
    }


def _plane_height_map(frame: HeightmapFrame, coeffs: np.ndarray) -> np.ndarray:
    h, w = frame.z_mm.shape
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    x_mm = frame.origin_x_mm + xx * frame.x_resolution_mm
    y_mm = frame.origin_y_mm + yy * frame.y_resolution_mm
    a, b, c, d = [float(x) for x in coeffs]
    denom = c if abs(c) > 1e-9 else 1.0
    z_plane = (-(a * x_mm + b * y_mm + d) / denom).astype(np.float32)
    z_plane[~frame.valid_mask] = 0.0
    return z_plane


def _component_width_cv(component_mask: np.ndarray, orientation_deg: float) -> float:
    ys, xs = np.where(component_mask)
    if xs.size <= 1 or ys.size <= 1:
        return 0.0
    x0 = max(0, int(np.min(xs)) - 2)
    y0 = max(0, int(np.min(ys)) - 2)
    x1 = min(component_mask.shape[1], int(np.max(xs)) + 3)
    y1 = min(component_mask.shape[0], int(np.max(ys)) + 3)
    patch = component_mask[y0:y1, x0:x1].astype(np.uint8) * 255
    center = (patch.shape[1] / 2.0, patch.shape[0] / 2.0)
    matrix = cv2.getRotationMatrix2D(center, float(orientation_deg), 1.0)
    rotated = cv2.warpAffine(patch, matrix, (patch.shape[1], patch.shape[0]), flags=cv2.INTER_NEAREST)
    counts = np.count_nonzero(rotated > 0, axis=0).astype(np.float64)
    counts = counts[counts > 0]
    if counts.size <= 1:
        return 0.0
    mean = float(np.mean(counts))
    if mean <= 1e-6:
        return 0.0
    return float(np.std(counts) / mean)


def _component_oriented_metrics(component_mask: np.ndarray, *, pixel_size_mm: float) -> dict[str, float | list[float]]:
    ys, xs = np.where(component_mask)
    if xs.size == 0 or ys.size == 0:
        return {
            "width_px": 0.0,
            "length_px": 0.0,
            "width_mm": 0.0,
            "length_mm": 0.0,
            "orientation_deg": 0.0,
            "width_cv": 0.0,
            "bbox": [0, 0, 0, 0],
        }
    points = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    rect = cv2.minAreaRect(points)
    (cx, cy), (rw, rh), angle = rect
    width_px = float(min(rw, rh))
    length_px = float(max(rw, rh))
    orientation_deg = float(angle if rw >= rh else angle + 90.0)
    if orientation_deg < 0.0:
        orientation_deg += 180.0
    width_cv = _component_width_cv(component_mask, orientation_deg)
    x = int(np.min(xs))
    y = int(np.min(ys))
    w = int(np.max(xs) - x + 1)
    h = int(np.max(ys) - y + 1)
    return {
        "width_px": width_px,
        "length_px": length_px,
        "width_mm": float(width_px * pixel_size_mm),
        "length_mm": float(length_px * pixel_size_mm),
        "orientation_deg": orientation_deg,
        "width_cv": width_cv,
        "bbox": [x, y, w, h],
        "centroid": [float(cx), float(cy)],
    }


def _label_colormap_overlay(labels: np.ndarray) -> np.ndarray:
    """Colorize a component-id label map so distinct components stay visually distinguishable.

    ``label % 251`` collides badly when there are few, low-valued labels (e.g. blob ids
    1/10/13/16 all land in the same dark corner of COLORMAP_TURBO). This spreads however many
    unique labels are present evenly across the colormap's full range instead.
    """
    labels = np.asarray(labels, dtype=np.int32)
    overlay = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    unique_labels = np.unique(labels[labels > 0])
    if unique_labels.size == 0:
        return overlay
    positions = np.array([160], dtype=np.uint8) if unique_labels.size == 1 else np.linspace(35, 255, unique_labels.size).astype(np.uint8)
    remap = np.zeros(int(unique_labels.max()) + 1, dtype=np.uint8)
    remap[unique_labels] = positions
    palette_index = remap[np.clip(labels, 0, remap.shape[0] - 1)]
    overlay = cv2.applyColorMap(palette_index, cv2.COLORMAP_TURBO)
    overlay[labels <= 0] = (0, 0, 0)
    return overlay


def _surface_role_overlay(
    z_mm: np.ndarray,
    valid_mask: np.ndarray,
    *,
    belt_bg_mask: np.ndarray,
    belt_stripes_mask: np.ndarray,
    unknown_low_gradient_mask: np.ndarray,
) -> np.ndarray:
    overlay = _height_color_image(z_mm, valid_mask)
    overlay[np.asarray(unknown_low_gradient_mask, dtype=bool)] = (220, 170, 40)
    overlay[np.asarray(belt_bg_mask, dtype=bool)] = (30, 200, 30)
    overlay[np.asarray(belt_stripes_mask, dtype=bool)] = (40, 110, 230)
    return overlay


def _adaptive_tick_format(span: float) -> str:
    """Choose a printf-style format for axis labels that fits the data span.

    The plot helpers used to hardcode ``.1f``/``.2f``, which yields useless
    "0.0" labels for raw-sensor z values in the 10⁴ range and ugly "11685.4"
    labels when the relevant span is 10–20 mm. This picks a precision that
    keeps roughly 3 significant digits of resolution across a tick step of
    ``span / 5``.
    """
    span = float(max(abs(span), 1e-12))
    step = span / 5.0
    if step <= 0:
        return "%.3f"
    decimals = int(max(0, min(6, -np.floor(np.log10(step)) + 1)))
    return f"%.{decimals}f"


def _adaptive_x_ticks(z_lo: float, z_hi: float, *, target_count: int = 5) -> list[float]:
    """Return ``target_count``-ish "nice" tick z-values for an X axis.

    Uses the standard 1/2/5 ladder rounded to a power of 10 so the labels
    land on round numbers regardless of whether the values are in mm
    (e.g. -10..50) or raw counts (11000..14500).
    """
    span = float(max(abs(z_hi - z_lo), 1e-12))
    raw_step = span / max(1, int(target_count))
    exp = float(np.floor(np.log10(raw_step)))
    base = raw_step / (10.0 ** exp)
    if base < 1.5:
        nice = 1.0
    elif base < 3.0:
        nice = 2.0
    elif base < 7.0:
        nice = 5.0
    else:
        nice = 10.0
    step = nice * (10.0 ** exp)
    if step <= 0:
        return [z_lo, z_hi]
    first = float(np.ceil(z_lo / step) * step)
    ticks: list[float] = []
    val = first
    safety = 0
    while val <= z_hi + 0.5 * step and safety < 50:
        ticks.append(val)
        val += step
        safety += 1
    if not ticks:
        ticks = [float(z_lo), float(z_hi)]
    return ticks


def _adaptive_count_ceiling(counts: np.ndarray, *, robust_percentile: float = 98.0) -> int:
    """Clip y-axis ceiling to a robust upper count so a handful of giant bins
    don't make every other bar invisible. Always at least 1.
    """
    arr = np.asarray(counts, dtype=np.float64)
    if arr.size == 0:
        return 1
    raw_max = float(np.max(arr))
    if raw_max <= 0:
        return 1
    pct = float(np.clip(robust_percentile, 50.0, 100.0))
    cap_val = float(np.percentile(arr[arr > 0], pct)) if np.any(arr > 0) else raw_max
    cap = max(cap_val, raw_max * 0.25)
    return max(1, int(np.ceil(cap)))


def _write_altitude_histogram_png(payload: dict[str, Any], *, out_path: Path, robust_y_percentile: float = 98.0) -> None:
    """Render the belt-stripe altitude histogram with the chosen threshold marked.

    The y-axis ceiling is taken from ``robust_y_percentile`` so a single tall
    bin doesn't flatten the rest of the distribution; x-axis tick spacing
    and label precision scale to whatever altitude range is present.
    """
    counts = np.asarray(payload.get("bin_counts") or [], dtype=np.int64)
    edges = np.asarray(payload.get("bin_edges") or [], dtype=np.float64)
    width, height = 900, 320
    margin_l, margin_r, margin_t, margin_b = 60, 30, 28, 40
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    if counts.size == 0 or edges.size < 2:
        cv2.putText(canvas, "Belt-stripe altitude histogram (no samples)", (margin_l, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.imwrite(str(out_path), canvas)
        return
    raw_max = max(1, int(np.max(counts)))
    cmax = _adaptive_count_ceiling(counts, robust_percentile=robust_y_percentile)
    z_lo = float(edges[0])
    z_hi = float(edges[-1])
    z_span = max(1e-6, z_hi - z_lo)
    y_axis = height - margin_b
    for i, c in enumerate(counts):
        if c <= 0:
            continue
        x0 = int(round(margin_l + (i / max(1, counts.size)) * plot_w))
        x1 = int(round(margin_l + ((i + 1) / max(1, counts.size)) * plot_w))
        y_top = int(round(y_axis - min(1.0, float(c) / cmax) * plot_h))
        cv2.rectangle(canvas, (x0, y_top), (max(x0 + 1, x1), y_axis), (170, 170, 170), -1)
    cv2.rectangle(canvas, (margin_l, margin_t + 14), (width - margin_r, y_axis), (100, 100, 100), 1)

    thr = float(payload.get("threshold_value_mm") or 0.0)

    def _x_for_z(v: float) -> int:
        return int(round(margin_l + ((v - z_lo) / z_span) * plot_w))

    thr_x = _x_for_z(thr)
    cv2.line(canvas, (thr_x, margin_t + 14), (thr_x, y_axis), (60, 220, 80), 2)
    fmt = _adaptive_tick_format(z_span)
    thr_label = f"thr={fmt % thr}mm"
    cv2.putText(canvas, thr_label, (min(width - margin_r - 100, thr_x + 4), margin_t + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 220, 80), 1, cv2.LINE_AA)
    for tick in _adaptive_x_ticks(z_lo, z_hi, target_count=5):
        x = _x_for_z(tick)
        cv2.line(canvas, (x, y_axis), (x, y_axis + 4), (160, 160, 160), 1)
        cv2.putText(canvas, fmt % tick, (x - 14, y_axis + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, "altitude over local-min baseline (mm)", (width - margin_r - 260, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)
    direction = str(payload.get("direction_used") or "raised")
    cv2.putText(canvas, f"Belt-stripe altitude histogram  |  direction={direction}  |  n={int(payload.get('sample_count') or 0)}  |  cap={cmax}  raw_max={raw_max}", (margin_l, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)


def _detect_depth_plateaus(
    *,
    z_values: np.ndarray,
    bins: int,
    min_fraction: float,
    min_pixels: int,
    smoothing_sigma_bins: float = 2.5,
    peak_drop_ratio: float = 0.4,
    robust_band_mad_k: float = 3.0,
    detection_min_count_floor: int = 25,
    detection_min_count_fraction: float = 0.25,
) -> dict[str, Any]:
    """Detect dominant depth plateaus via smoothed-histogram peak detection.

    Strategy
    --------
    1. Histogram the filtered z values.
    2. Smooth the histogram with a 1D Gaussian (`smoothing_sigma_bins`).
       This is essential: a slightly-tilted belt fans out over many bins
       and its raw per-bin counts can be lower than a sharper but smaller
       structure (e.g. coplanar chevron ribs). Smoothing recovers the
       broader-but-shorter mode.
    3. Find local maxima of the smoothed histogram. Use a lenient
       per-bin threshold (`min_pixels`) for *detection* — the stricter
       `min_fraction` test is applied later by selection, so we don't
       lose broad-but-low-density modes (a tilted belt typically has
       per-bin counts well below `min_fraction * n_flat`).
    4. Around each peak, expand the band outward until the smoothed count
       drops below `peak_drop_ratio * peak_height` — this defines the
       plateau bounds (not a strict per-bin threshold). Overlapping bands
       are merged.
    5. Also compute a robust median ± k·MAD band that bounds the bulk of
       the filtered z mass. This is exposed for selection's fallback
       and as a diagnostic plateau when histogram peak-finding misses
       a broad uniform belt distribution.
    6. Emit per-plateau stats including bandwidth, peak, area fraction.
    """
    vals = np.asarray(z_values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    bins_n = max(8, int(bins))
    if vals.size == 0:
        return {
            "sample_count": 0,
            "bin_edges": [],
            "bin_counts": [],
            "bin_counts_smoothed": [],
            "smoothing_sigma_bins": float(smoothing_sigma_bins),
            "peak_drop_ratio": float(peak_drop_ratio),
            "min_count": 0,
            "detection_min_count": 0,
            "robust_band": None,
            "plateaus": [],
        }
    counts, edges = np.histogram(vals, bins=bins_n)
    counts_f = counts.astype(np.float64)

    sigma = max(0.5, float(smoothing_sigma_bins))
    radius = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
    kernel /= float(kernel.sum())
    smoothed = np.convolve(counts_f, kernel, mode="same")

    # `min_count` is the *selection*-grade threshold (used downstream and
    # reported on the plot). `detection_min_count` is intentionally laxer
    # so a broad uniform belt distribution still produces a detectable peak.
    selection_min_count = max(int(min_pixels), int(np.ceil(float(min_fraction) * float(vals.size))))
    detection_min_count = max(
        int(np.ceil(float(detection_min_count_fraction) * float(min_pixels))),
        int(detection_min_count_floor),
    )
    drop_ratio = float(np.clip(peak_drop_ratio, 0.05, 0.95))

    # Robust band around the median — works even when the histogram has no
    # well-defined peak (tilted belt → near-uniform z distribution).
    z_med = float(np.median(vals))
    z_mad = float(np.median(np.abs(vals - z_med)))
    k = float(max(1.0, robust_band_mad_k))
    robust_lo = z_med - (k * z_mad if z_mad > 0 else 0.5 * (float(np.max(vals)) - float(np.min(vals))))
    robust_hi = z_med + (k * z_mad if z_mad > 0 else 0.5 * (float(np.max(vals)) - float(np.min(vals))))
    robust_in = vals[(vals >= robust_lo) & (vals <= robust_hi)]
    robust_band = {
        "kind": "robust_median_mad",
        "z_median": z_med,
        "z_mad": z_mad,
        "k_mad": k,
        "z_min": float(robust_lo),
        "z_max": float(robust_hi),
        "z_center": float(0.5 * (robust_lo + robust_hi)),
        "count": int(robust_in.size),
        "fraction": float(robust_in.size / max(1, vals.size)),
    }

    min_count = detection_min_count

    peak_indices: list[int] = []
    n = int(smoothed.size)
    for i in range(n):
        c = float(smoothed[i])
        if c < min_count:
            continue
        left_ok = (i == 0) or (smoothed[i - 1] <= c)
        right_ok = (i == n - 1) or (smoothed[i + 1] <= c)
        # Strictly greater than at least one side avoids long flat ridges
        # producing a peak per bin; keep the highest bin within a flat run.
        strictly_greater = (
            (i > 0 and smoothed[i - 1] < c) or (i < n - 1 and smoothed[i + 1] < c)
        )
        if left_ok and right_ok and (strictly_greater or n == 1):
            peak_indices.append(i)

    plateaus_raw: list[dict[str, Any]] = []
    for peak_idx in peak_indices:
        peak_h = float(smoothed[peak_idx])
        thr = drop_ratio * peak_h
        lo = peak_idx
        while lo > 0 and smoothed[lo - 1] >= thr:
            lo -= 1
        hi = peak_idx
        while hi < n - 1 and smoothed[hi + 1] >= thr:
            hi += 1
        plateaus_raw.append({"lo": lo, "hi": hi, "peak_idx": peak_idx, "peak_h": peak_h})

    # Merge overlapping bands (keep the highest peak as the representative).
    plateaus_raw.sort(key=lambda r: r["lo"])
    merged: list[dict[str, Any]] = []
    for row in plateaus_raw:
        if merged and row["lo"] <= merged[-1]["hi"]:
            cur = merged[-1]
            cur["hi"] = max(cur["hi"], row["hi"])
            if row["peak_h"] > cur["peak_h"]:
                cur["peak_h"] = row["peak_h"]
                cur["peak_idx"] = row["peak_idx"]
        else:
            merged.append(dict(row))

    plateaus: list[dict[str, Any]] = []
    raw_peak = max(1.0, float(np.max(counts_f))) if counts_f.size else 1.0
    for row in merged:
        lo = int(row["lo"])
        hi = int(row["hi"])
        z_min = float(edges[lo])
        z_max = float(edges[hi + 1])
        in_band = vals[(vals >= z_min) & (vals <= z_max)]
        count = int(in_band.size)
        plateaus.append(
            {
                "band_start_bin": lo,
                "band_end_bin": hi,
                "peak_bin": int(row["peak_idx"]),
                "z_min": z_min,
                "z_max": z_max,
                "z_center": float(0.5 * (z_min + z_max)),
                "z_peak": float(0.5 * (edges[int(row["peak_idx"])] + edges[int(row["peak_idx"]) + 1])),
                "count": count,
                "fraction": float(count / max(1, vals.size)),
                "peak_count_smoothed": float(row["peak_h"]),
                "peak_ratio": float(row["peak_h"] / raw_peak),
                "bandwidth_bins": int(hi - lo + 1),
                "z_median": float(np.median(in_band)) if in_band.size else float(0.5 * (z_min + z_max)),
            }
        )

    return {
        "sample_count": int(vals.size),
        "bin_edges": [float(x) for x in edges.tolist()],
        "bin_counts": [int(x) for x in counts.tolist()],
        "bin_counts_smoothed": [float(x) for x in smoothed.tolist()],
        "smoothing_sigma_bins": float(sigma),
        "peak_drop_ratio": float(drop_ratio),
        "min_count": int(selection_min_count),
        "detection_min_count": int(detection_min_count),
        "robust_band": robust_band,
        "plateaus": sorted(plateaus, key=lambda row: float(row.get("z_center", 0.0))),
    }


def _select_background_plateau(
    plateaus: list[dict[str, Any]],
    *,
    min_area_fraction: float = 0.20,
    mode: str = "lowest_dominant",
    robust_band: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Pick the plateau most likely to be the belt/background support.

    Modes
    -----
    - "lowest_dominant" (default): among plateaus with fraction ≥ min_area_fraction,
      pick the one with the lowest z_center. If none qualify, fall back to the
      robust median ± k·MAD band (when provided), then to the largest plateau.
      Right default for a conveyor belt where the belt should dominate by area
      but might be the second-most-prominent histogram peak because chevron
      ribs form a tighter (smaller-bandwidth) cluster — and where a tilted belt
      may not form any histogram peak that passes `min_count`.
    - "largest": pick the plateau with the highest fraction. Use when the belt
      is reliably the biggest flat surface.
    - "lowest": pick the lowest-z plateau regardless of area. The original
      behaviour — use only when you trust the camera-to-belt convention and
      that no spurious lower flat surface exists.
    - "robust_band": ignore histogram peaks and use the median ± k·MAD band
      directly. Most robust to broadly-distributed belt geometry (tilt, noise,
      texture) but cannot separate two co-existing planes.
    """
    sel_mode = (mode or "lowest_dominant").strip().lower()

    def _from_robust_band() -> dict[str, Any] | None:
        if not robust_band:
            return None
        return {
            "band_start_bin": -1,
            "band_end_bin": -1,
            "peak_bin": -1,
            "z_min": float(robust_band.get("z_min") or 0.0),
            "z_max": float(robust_band.get("z_max") or 0.0),
            "z_center": float(robust_band.get("z_center") or 0.0),
            "z_peak": float(robust_band.get("z_center") or 0.0),
            "count": int(robust_band.get("count") or 0),
            "fraction": float(robust_band.get("fraction") or 0.0),
            "peak_count_smoothed": 0.0,
            "peak_ratio": 0.0,
            "bandwidth_bins": -1,
            "z_median": float(robust_band.get("z_median") or 0.0),
            "source": "robust_median_mad",
        }

    if sel_mode == "robust_band":
        return _from_robust_band()
    if not plateaus:
        return _from_robust_band()
    if sel_mode == "largest":
        return sorted(plateaus, key=lambda r: -float(r.get("fraction", 0.0)))[0]
    if sel_mode == "lowest":
        return sorted(
            plateaus,
            key=lambda r: (float(r.get("z_center", 0.0)), -float(r.get("fraction", 0.0))),
        )[0]
    # lowest_dominant
    qualifying = [
        row for row in plateaus if float(row.get("fraction", 0.0)) >= float(min_area_fraction)
    ]
    if qualifying:
        return sorted(qualifying, key=lambda r: float(r.get("z_center", 0.0)))[0]
    fallback = _from_robust_band()
    if fallback is not None and float(fallback.get("fraction") or 0.0) >= float(min_area_fraction):
        return fallback
    return sorted(plateaus, key=lambda r: -float(r.get("fraction", 0.0)))[0]


def _write_plateau_plot_png(
    plateau_payload: dict[str, Any],
    *,
    out_path: Path,
    selected_plateau: dict[str, Any] | None = None,
    sample_count: int | None = None,
    strategy: str | None = None,
    notes: str | None = None,
    robust_y_percentile: float = 98.0,
    min_band_marker_width_px: int = 3,
) -> None:
    """Render the flat-candidate depth histogram with detected plateaus highlighted.

    Designed to be the canonical Studio "Plateau plot" diagnostic for the
    `low_gradient_depth_plateaus` strategy. The selected (background) plateau is
    drawn in green, rejected (raised) plateaus in red, raw histogram in grey.

    Adaptive behavior
    -----------------
    - Y axis is clipped to ``robust_y_percentile`` of nonzero bin counts so
      one giant bin doesn't flatten the rest. The raw max is still shown
      in the subtitle.
    - X tick spacing and label precision adapt to the z range (mm vs raw
      counts).
    - Narrow plateau bands are widened to ``min_band_marker_width_px`` so
      the selected plateau is always visible even when its z-span is tiny
      compared to the full histogram range.
    """
    counts = np.asarray(plateau_payload.get("bin_counts") or [], dtype=np.int32)
    edges = np.asarray(plateau_payload.get("bin_edges") or [], dtype=np.float64)
    width, height = 1100, 420
    margin_l, margin_r, margin_t, margin_b = 70, 30, 32, 50
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)

    def _put_header(title: str, subtitle: str | None = None) -> None:
        cv2.putText(canvas, title, (margin_l, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
        if subtitle:
            cv2.putText(canvas, subtitle, (margin_l, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 190, 190), 1, cv2.LINE_AA)

    if counts.size == 0 or edges.size < 2:
        _put_header(
            "Flat-candidate depth plateaus (no data)",
            notes or "No flat candidates produced by gradient/Hessian filtering.",
        )
        cv2.rectangle(canvas, (margin_l, margin_t + 24), (width - margin_r, height - margin_b), (90, 90, 90), 1)
        cv2.imwrite(str(out_path), canvas)
        return

    z_lo = float(edges[0])
    z_hi = float(edges[-1])
    z_span = max(1e-6, z_hi - z_lo)
    raw_cmax = max(1, int(np.max(counts)))
    cmax = _adaptive_count_ceiling(counts, robust_percentile=robust_y_percentile)
    tick_fmt = _adaptive_tick_format(z_span)
    band_min_px = max(1, int(min_band_marker_width_px))

    def _x_for_z(z: float) -> int:
        return int(round(margin_l + ((z - z_lo) / z_span) * plot_w))

    def _x_for_bin(idx: int) -> int:
        return int(round(margin_l + (idx / max(1, counts.size)) * plot_w))

    # Histogram bars (grey).
    y_axis = height - margin_b
    for i, c in enumerate(counts):
        if c <= 0:
            continue
        x0 = _x_for_bin(i)
        x1 = _x_for_bin(i + 1)
        y_top = int(round(y_axis - min(1.0, float(c) / cmax) * plot_h))
        cv2.rectangle(canvas, (x0, y_top), (max(x0 + 1, x1), y_axis), (170, 170, 170), -1)

    # Smoothed-histogram polyline (cyan) — the curve the peak-detection works on.
    smoothed = np.asarray(plateau_payload.get("bin_counts_smoothed") or [], dtype=np.float64)
    if smoothed.size == counts.size and smoothed.size >= 2:
        sm_pts: list[tuple[int, int]] = []
        for i in range(smoothed.size):
            cx = int(round((_x_for_bin(i) + _x_for_bin(i + 1)) / 2))
            cy = int(round(y_axis - min(1.0, float(smoothed[i]) / cmax) * plot_h))
            sm_pts.append((cx, cy))
        for a, b in zip(sm_pts, sm_pts[1:]):
            cv2.line(canvas, a, b, (220, 220, 90), 1, cv2.LINE_AA)
        # Min-count threshold line (dashed grey).
        min_count_val = float(plateau_payload.get("min_count") or 0.0)
        if min_count_val > 0 and min_count_val <= cmax:
            y_thr = int(round(y_axis - (min_count_val / cmax) * plot_h))
            for x_seg in range(margin_l, width - margin_r, 12):
                cv2.line(canvas, (x_seg, y_thr), (min(width - margin_r, x_seg + 6), y_thr), (130, 130, 130), 1)
            cv2.putText(canvas, f"min_count={int(min_count_val)}", (width - margin_r - 130, max(margin_t + 32, y_thr - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1, cv2.LINE_AA)

    # Robust median ± k·MAD band drawn as a dashed magenta reference. Useful
    # when the histogram has no clear peak (broad / tilted belt distribution).
    robust_band = plateau_payload.get("robust_band") or {}
    if isinstance(robust_band, dict) and robust_band.get("z_min") is not None and robust_band.get("z_max") is not None:
        rb_x0 = _x_for_z(float(robust_band.get("z_min") or 0.0))
        rb_x1 = _x_for_z(float(robust_band.get("z_max") or 0.0))
        if rb_x1 - rb_x0 < band_min_px:
            mid = (rb_x0 + rb_x1) // 2
            rb_x0 = mid - band_min_px // 2
            rb_x1 = rb_x0 + band_min_px
        for x_seg in range(min(rb_x0, rb_x1), max(rb_x0, rb_x1), 10):
            cv2.line(canvas, (x_seg, margin_t + 24), (min(rb_x1, x_seg + 5), margin_t + 24), (200, 90, 220), 1)
            cv2.line(canvas, (x_seg, y_axis), (min(rb_x1, x_seg + 5), y_axis), (200, 90, 220), 1)
        cv2.line(canvas, (rb_x0, margin_t + 24), (rb_x0, y_axis), (200, 90, 220), 1)
        cv2.line(canvas, (rb_x1, margin_t + 24), (rb_x1, y_axis), (200, 90, 220), 1)
        cv2.putText(
            canvas,
            "median +/- k*MAD",
            (min(rb_x0, rb_x1) + 4, y_axis - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (200, 90, 220),
            1,
            cv2.LINE_AA,
        )

    plateaus = list(plateau_payload.get("plateaus") or [])
    selected_z_center = (
        float(selected_plateau.get("z_center")) if isinstance(selected_plateau, dict) and selected_plateau.get("z_center") is not None else None
    )

    def _is_selected(row: dict[str, Any]) -> bool:
        if selected_z_center is None:
            return False
        try:
            return abs(float(row.get("z_center", 0.0)) - selected_z_center) < 1e-6
        except (TypeError, ValueError):
            return False

    for row in plateaus:
        b0 = int(row.get("band_start_bin") or 0)
        b1 = int(row.get("band_end_bin") or b0)
        x0 = _x_for_bin(b0)
        x1 = _x_for_bin(b1 + 1)
        if x1 - x0 < band_min_px:
            mid = (x0 + x1) // 2
            x0 = mid - band_min_px // 2
            x1 = x0 + band_min_px
        selected = _is_selected(row)
        outline = (60, 220, 80) if selected else (60, 90, 230)
        fill_alpha = 0.18 if selected else 0.10
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x0, margin_t + 24), (max(x0 + 1, x1), y_axis), outline, -1)
        cv2.addWeighted(overlay, fill_alpha, canvas, 1.0 - fill_alpha, 0, canvas)
        cv2.rectangle(canvas, (x0, margin_t + 24), (max(x0 + 1, x1), y_axis), outline, 2)
        label_z = f"z={tick_fmt % float(row.get('z_center', 0.0))}"
        label_frac = f"{float(row.get('fraction', 0.0)) * 100.0:.1f}%"
        cv2.putText(canvas, label_z, (x0 + 4, margin_t + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.38, outline, 1, cv2.LINE_AA)
        cv2.putText(canvas, label_frac, (x0 + 4, margin_t + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.38, outline, 1, cv2.LINE_AA)
        if selected:
            cv2.putText(canvas, "BG", (x0 + 4, margin_t + 76), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 220, 80), 1, cv2.LINE_AA)

    # X-axis ticks (data units, adaptive).
    cv2.rectangle(canvas, (margin_l, margin_t + 24), (width - margin_r, y_axis), (100, 100, 100), 1)
    for tick in _adaptive_x_ticks(z_lo, z_hi, target_count=6):
        x = _x_for_z(tick)
        cv2.line(canvas, (x, y_axis), (x, y_axis + 4), (160, 160, 160), 1)
        cv2.putText(canvas, tick_fmt % tick, (x - 14, y_axis + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, "z (mm)", (width - margin_r - 60, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)
    y_axis_legend = f"count (cap={cmax}, raw_max={raw_cmax})" if cmax != raw_cmax else f"count (max={raw_cmax})"
    cv2.putText(canvas, y_axis_legend, (8, margin_t + 24 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1, cv2.LINE_AA)

    title = "Flat-candidate depth plateaus"
    subtitle_parts: list[str] = []
    if strategy:
        subtitle_parts.append(f"strategy={strategy}")
    if sample_count is not None:
        subtitle_parts.append(f"n_flat={int(sample_count)}")
    subtitle_parts.append(f"plateaus={len(plateaus)}")
    sigma = plateau_payload.get("smoothing_sigma_bins")
    drop = plateau_payload.get("peak_drop_ratio")
    if sigma is not None and drop is not None:
        subtitle_parts.append(f"smooth_sigma={float(sigma):.1f}bins  drop={float(drop):.2f}")
    if selected_plateau:
        sel_lo = float(selected_plateau.get("z_min") or 0.0)
        sel_hi = float(selected_plateau.get("z_max") or 0.0)
        sel_frac = float(selected_plateau.get("fraction") or 0.0)
        subtitle_parts.append(
            f"selected[BG]=[{tick_fmt % sel_lo},{tick_fmt % sel_hi}] (frac={sel_frac:.1%})"
        )
    if notes:
        subtitle_parts.append(notes)
    _put_header(title, "  |  ".join(subtitle_parts) if subtitle_parts else None)
    cv2.imwrite(str(out_path), canvas)


def _write_plateau_plot_not_applicable_png(*, out_path: Path, strategy: str) -> None:
    """Explicit placeholder so Studio doesn't fall back to the depth plot.

    The Plateau plot only makes sense for the `low_gradient_depth_plateaus`
    strategy; for everything else we render a clearly-labelled card explaining
    why this view is intentionally empty for the active strategy.
    """
    width, height = 1100, 420
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    cv2.rectangle(canvas, (24, 24), (width - 24, height - 24), (80, 80, 80), 1)
    cv2.putText(canvas, "Plateau plot not applicable", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (235, 235, 235), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"Active strategy: {strategy}",
        (40, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    msg_lines = [
        "Flat-candidate depth-plateau analysis is only computed when the",
        "reference-surface strategy is `low_gradient_depth_plateaus`.",
        "",
        "Switch the strategy to low_gradient_depth_plateaus and rerun to see",
        "the histogram of low-gradient (and optionally low-Hessian) candidates,",
        "the detected plateaus, and the selected background-belt plateau.",
    ]
    for idx, line in enumerate(msg_lines):
        cv2.putText(canvas, line, (40, 160 + idx * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (190, 190, 190), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)


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


def _component_solidity(mask: np.ndarray) -> float:
    mask_u8 = (np.asarray(mask, dtype=np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    if hull_area <= 1e-9:
        return 0.0
    return float(area / hull_area)


def _summarize_low_gradient_component_row(
    *,
    component_mask: np.ndarray,
    component_id: int,
    gradient: np.ndarray,
    z: np.ndarray,
    valid_mask: np.ndarray,
    roi_mask: np.ndarray,
    height_gate_mask: np.ndarray | None,
    total_valid: int,
    roi_border: np.ndarray,
    frame_shape: tuple[int, int],
    centroid_hint: tuple[float, float] | None = None,
    min_component_area: int,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    stat_mask = component_mask & valid_mask
    area = int(np.count_nonzero(stat_mask))
    if area <= 0:
        return None
    ys, xs = np.where(stat_mask)
    if xs.size == 0 or ys.size == 0:
        return None
    x = int(np.min(xs))
    y = int(np.min(ys))
    w = int(np.max(xs) - x + 1)
    h = int(np.max(ys) - y + 1)
    frame_h, frame_w = frame_shape
    cx_frame = 0.5 * max(0, frame_w - 1)
    cy_frame = 0.5 * max(0, frame_h - 1)
    max_center_dist = max(1.0, float(np.hypot(cx_frame, cy_frame)))
    centroid_x = float(centroid_hint[0]) if centroid_hint is not None else float(np.mean(xs))
    centroid_y = float(centroid_hint[1]) if centroid_hint is not None else float(np.mean(ys))
    zvals = z[stat_mask].astype(np.float64)
    gvals = gradient[stat_mask].astype(np.float64)
    z_med = float(np.median(zvals))
    dist_border = float(min(x, y, max(0, frame_w - (x + w)), max(0, frame_h - (y + h))))
    centrality = float(np.hypot(centroid_x - cx_frame, centroid_y - cy_frame) / max_center_dist)
    touches_frame_border = bool(x <= 0 or y <= 0 or (x + w) >= frame_w or (y + h) >= frame_h)
    touches_roi_border = bool(np.any(component_mask & roi_border))
    gate_overlap = (
        float(np.count_nonzero(component_mask & np.asarray(height_gate_mask, dtype=bool)) / max(1, area))
        if height_gate_mask is not None
        else 0.0
    )
    row = {
        "blob_id": int(component_id),
        "area_px": area,
        "area_ratio": float(area / max(1, total_valid)),
        "median_z": z_med,
        "mean_z": float(np.mean(zvals)),
        "std_z": float(np.std(zvals)),
        "mad_z": float(np.median(np.abs(zvals - z_med))),
        "z_p05": float(np.percentile(zvals, 5)),
        "z_p50": float(np.percentile(zvals, 50)),
        "z_p95": float(np.percentile(zvals, 95)),
        "gradient_mean": float(np.mean(gvals)) if gvals.size else 0.0,
        "gradient_p95": float(np.percentile(gvals, 95)) if gvals.size else 0.0,
        "bbox": [x, y, w, h],
        "centroid": [centroid_x, centroid_y],
        "touches_frame_border": touches_frame_border,
        "touches_roi_border": touches_roi_border,
        "distance_to_frame_border": dist_border,
        "aspect_ratio": float(max(w, h) / max(1.0, float(min(w, h)))),
        "elongation": float((max(w, h) - min(w, h)) / max(1.0, float(max(w, h)))),
        "solidity": _component_solidity(component_mask),
        "compactness": float(area / max(1.0, float(w * h))),
        "overlap_with_roi": float(np.count_nonzero(component_mask & roi_mask) / max(1, area)),
        "overlap_with_height_gate": gate_overlap,
        "overlap_with_stripe_mask": 0.0,
        "centrality": centrality,
        "z_range_robust": float(np.percentile(zvals, 95) - np.percentile(zvals, 5)) if zvals.size else 0.0,
        "component_id": int(component_id),
        "source": "low_gradient_mask",
        "rejected": area < int(min_component_area),
        "rejected_reason": "tiny_component" if area < int(min_component_area) else None,
    }
    if extra_fields:
        row.update(extra_fields)
    return row


def _normalize_blob_component_mode(mode: str | None) -> str:
    value = str(mode or "xy_only").strip().lower()
    if value in {"xy_only", "height_aware"}:
        return value
    return "xy_only"


def _effective_blob_neighbor_z_tolerance_mm(
    *,
    z: np.ndarray,
    candidate_mask: np.ndarray,
    tolerance_mode: str,
    fixed_tolerance_mm: float,
    mad_k: float,
    mad_floor_mm: float,
) -> tuple[float, dict[str, Any]]:
    mode = str(tolerance_mode or "fixed").strip().lower()
    if mode == "mad_adaptive":
        zvals = z[candidate_mask].astype(np.float64)
        if zvals.size:
            med = float(np.median(zvals))
            mad = float(np.median(np.abs(zvals - med)))
            tol = max(float(mad_floor_mm), float(mad_k) * max(0.25, mad))
        else:
            mad = 0.0
            tol = float(fixed_tolerance_mm)
        return tol, {"mode": "mad_adaptive", "mad_mm": mad, "effective_tolerance_mm": tol}
    return float(fixed_tolerance_mm), {"mode": "fixed", "effective_tolerance_mm": float(fixed_tolerance_mm)}


def _prepare_blob_component_z(
    z: np.ndarray,
    valid_mask: np.ndarray,
    *,
    use_smoothed_z: bool,
    smoothing_kernel: int,
) -> np.ndarray:
    z_work = z.astype(np.float32, copy=True)
    if not use_smoothed_z:
        return z_work
    kernel = max(1, int(smoothing_kernel))
    if kernel % 2 == 0:
        kernel += 1
    fill = float(np.median(z_work[valid_mask])) if np.any(valid_mask) else 0.0
    filled = np.where(valid_mask, z_work, fill).astype(np.float32)
    return cv2.GaussianBlur(filled, (kernel, kernel), 0)


def _height_aware_blob_neighbor_offsets(connectivity: int) -> list[tuple[int, int]]:
    offsets = [(0, 1), (1, 0)]
    if int(connectivity) >= 8:
        offsets.extend([(1, 1), (1, -1)])
    return offsets


def _pixels_z_compatible_for_blob_edge(
    za: float,
    zb: float,
    *,
    neighbor_tol_mm: float,
    allow_gradual_slope: bool,
    max_local_slope_mm_per_px: float,
) -> bool:
    dz = abs(float(za) - float(zb))
    step_tol = float(neighbor_tol_mm)
    if allow_gradual_slope:
        step_tol = max(step_tol, float(max_local_slope_mm_per_px))
    return dz <= step_tol


def _build_height_aware_low_gradient_components(
    *,
    low_grad_mask: np.ndarray,
    gradient: np.ndarray,
    z: np.ndarray,
    valid_mask: np.ndarray,
    roi_mask: np.ndarray,
    height_gate_mask: np.ndarray | None,
    min_component_area: int,
    min_area_px: int,
    connectivity: int,
    neighbor_z_tolerance_mm: float,
    component_z_tolerance_mm: float,
    tolerance_mode: str,
    mad_k: float,
    mad_floor_mm: float,
    allow_gradual_slope: bool,
    max_local_slope_mm_per_px: float,
    use_smoothed_z: bool,
    z_smoothing_kernel: int,
) -> dict[str, Any]:
    candidate_mask = np.asarray(low_grad_mask, dtype=bool) & np.asarray(valid_mask, dtype=bool)
    h, w = candidate_mask.shape
    z_work = _prepare_blob_component_z(z, valid_mask, use_smoothed_z=use_smoothed_z, smoothing_kernel=z_smoothing_kernel)
    neighbor_tol_mm, tolerance_debug = _effective_blob_neighbor_z_tolerance_mm(
        z=z_work,
        candidate_mask=candidate_mask,
        tolerance_mode=tolerance_mode,
        fixed_tolerance_mm=neighbor_z_tolerance_mm,
        mad_k=mad_k,
        mad_floor_mm=mad_floor_mm,
    )
    flat_size = h * w
    parent = np.arange(flat_size, dtype=np.int32)

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(a_idx: int, b_idx: int) -> None:
        pa = find(a_idx)
        pb = find(b_idx)
        if pa != pb:
            parent[pb] = pa

    accepted_edges = 0
    rejected_edges = 0
    rejected_edge_mask = np.zeros((h, w), dtype=bool)
    offsets = _height_aware_blob_neighbor_offsets(connectivity)
    candidate_flat = candidate_mask.reshape(-1)
    z_flat = z_work.reshape(-1)
    for y in range(h):
        for x in range(w):
            if not candidate_mask[y, x]:
                continue
            p_idx = y * w + x
            zp = float(z_flat[p_idx])
            for dy, dx in offsets:
                ny, nx = y + dy, x + dx
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                if not candidate_mask[ny, nx]:
                    continue
                q_idx = ny * w + nx
                if q_idx <= p_idx:
                    continue
                zq = float(z_flat[q_idx])
                if _pixels_z_compatible_for_blob_edge(
                    zp,
                    zq,
                    neighbor_tol_mm=neighbor_tol_mm,
                    allow_gradual_slope=allow_gradual_slope,
                    max_local_slope_mm_per_px=max_local_slope_mm_per_px,
                ):
                    union(p_idx, q_idx)
                    accepted_edges += 1
                else:
                    rejected_edges += 1
                    rejected_edge_mask[y, x] = True
                    rejected_edge_mask[ny, nx] = True

    labels = np.zeros((h, w), dtype=np.int32)
    root_to_label: dict[int, int] = {}
    next_label = 1
    for flat_idx in np.flatnonzero(candidate_flat):
        root = find(int(flat_idx))
        if root not in root_to_label:
            root_to_label[root] = next_label
            next_label += 1
        y, x = divmod(int(flat_idx), w)
        labels[y, x] = root_to_label[root]

    small_component_rejected_count = 0
    for label in sorted(set(root_to_label.values())):
        component_mask = labels == label
        area_px = int(np.count_nonzero(component_mask))
        if area_px < max(1, int(min_area_px)):
            labels[component_mask] = 0
            small_component_rejected_count += 1

    total_valid = max(1, int(np.count_nonzero(valid_mask)))
    roi_border = _roi_border_mask(roi_mask)
    rows: list[dict[str, Any]] = []
    kept_ids: list[int] = []
    rejected_ids: list[int] = []
    height_aware_fields = {
        "component_mode": "height_aware",
        "neighbor_z_tolerance_mm": float(neighbor_tol_mm),
        "component_z_tolerance_mm": float(component_z_tolerance_mm),
        "connectivity": int(connectivity),
        "use_smoothed_z": bool(use_smoothed_z),
        "accepted_neighbor_edge_count": int(accepted_edges),
        "rejected_neighbor_edge_count": int(rejected_edges),
        "component_reference_mode": "local_edge_only",
    }
    for label in sorted(int(v) for v in np.unique(labels) if int(v) > 0):
        mask = labels == label
        ys, xs = np.where(mask)
        centroid_hint = (float(np.mean(xs)), float(np.mean(ys))) if xs.size else None
        row = _summarize_low_gradient_component_row(
            component_mask=mask,
            component_id=int(label),
            gradient=gradient,
            z=z,
            valid_mask=valid_mask,
            roi_mask=roi_mask,
            height_gate_mask=height_gate_mask,
            total_valid=total_valid,
            roi_border=roi_border,
            frame_shape=low_grad_mask.shape,
            centroid_hint=centroid_hint,
            min_component_area=min_component_area,
            extra_fields={
                **height_aware_fields,
                "source_blob_id": int(label),
                "fragment_id": int(label),
                "split_index": 0,
                "was_split_from_blob": False,
            },
        )
        if row is None:
            continue
        rows.append(row)
        if row["rejected"]:
            rejected_ids.append(int(label))
        else:
            kept_ids.append(int(label))

    connectivity_debug = {
        "candidate_pixels": int(np.count_nonzero(candidate_mask)),
        "accepted_neighbor_edges": int(accepted_edges),
        "rejected_neighbor_edges": int(rejected_edges),
        "component_count": int(len(rows)),
        "small_component_rejected_count": int(small_component_rejected_count),
        "parameters": {
            "blob_component_mode": "height_aware",
            "blob_connectivity": int(connectivity),
            "blob_neighbor_z_tolerance_mm": float(neighbor_z_tolerance_mm),
            "blob_component_z_tolerance_mm": float(component_z_tolerance_mm),
            "blob_component_tolerance_mode": str(tolerance_mode),
            "blob_component_allow_gradual_slope": bool(allow_gradual_slope),
            "blob_component_max_local_slope_mm_per_px": float(max_local_slope_mm_per_px),
            "blob_component_min_area_px": int(min_area_px),
            "blob_component_use_smoothed_z": bool(use_smoothed_z),
            "blob_component_z_smoothing_kernel": int(z_smoothing_kernel),
        },
        "tolerance": tolerance_debug,
    }
    return {
        "labels": labels,
        "blobs": rows,
        "kept_ids": kept_ids,
        "rejected_ids": rejected_ids,
        "component_mode": "height_aware",
        "rejected_edge_mask": rejected_edge_mask,
        "connectivity_debug": connectivity_debug,
    }


def _summarize_low_gradient_blobs(
    *,
    low_grad_mask: np.ndarray,
    gradient: np.ndarray,
    z: np.ndarray,
    valid_mask: np.ndarray,
    roi_mask: np.ndarray,
    height_gate_mask: np.ndarray | None,
    min_component_area: int,
    component_mode: str = "xy_only",
    blob_connectivity: int = 8,
    blob_neighbor_z_tolerance_mm: float = 3.0,
    blob_component_z_tolerance_mm: float = 8.0,
    blob_component_tolerance_mode: str = "fixed",
    blob_component_mad_k: float = 3.0,
    blob_component_mad_floor_mm: float = 1.0,
    blob_component_allow_gradual_slope: bool = True,
    blob_component_max_local_slope_mm_per_px: float = 3.0,
    blob_component_min_area_px: int = 300,
    blob_component_use_smoothed_z: bool = True,
    blob_component_z_smoothing_kernel: int = 3,
) -> dict[str, Any]:
    mode = _normalize_blob_component_mode(component_mode)
    if mode == "height_aware":
        return _build_height_aware_low_gradient_components(
            low_grad_mask=low_grad_mask,
            gradient=gradient,
            z=z,
            valid_mask=valid_mask,
            roi_mask=roi_mask,
            height_gate_mask=height_gate_mask,
            min_component_area=min_component_area,
            min_area_px=blob_component_min_area_px,
            connectivity=blob_connectivity,
            neighbor_z_tolerance_mm=blob_neighbor_z_tolerance_mm,
            component_z_tolerance_mm=blob_component_z_tolerance_mm,
            tolerance_mode=blob_component_tolerance_mode,
            mad_k=blob_component_mad_k,
            mad_floor_mm=blob_component_mad_floor_mm,
            allow_gradual_slope=blob_component_allow_gradual_slope,
            max_local_slope_mm_per_px=blob_component_max_local_slope_mm_per_px,
            use_smoothed_z=blob_component_use_smoothed_z,
            z_smoothing_kernel=blob_component_z_smoothing_kernel,
        )
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(low_grad_mask.astype(np.uint8), connectivity=8)
    total_valid = max(1, int(np.count_nonzero(valid_mask)))
    roi_border = _roi_border_mask(roi_mask)
    rows: list[dict[str, Any]] = []
    kept_ids: list[int] = []
    rejected_ids: list[int] = []
    for label in range(1, num_labels):
        mask = labels == label
        row = _summarize_low_gradient_component_row(
            component_mask=mask,
            component_id=int(label),
            gradient=gradient,
            z=z,
            valid_mask=valid_mask,
            roi_mask=roi_mask,
            height_gate_mask=height_gate_mask,
            total_valid=total_valid,
            roi_border=roi_border,
            frame_shape=low_grad_mask.shape,
            centroid_hint=(float(centroids[label][0]), float(centroids[label][1])),
            min_component_area=min_component_area,
            extra_fields={
                "component_mode": "xy_only",
                "source_blob_id": int(label),
                "fragment_id": int(label),
                "split_index": 0,
                "was_split_from_blob": False,
            },
        )
        if row is None:
            continue
        rows.append(row)
        if row["rejected"]:
            rejected_ids.append(int(label))
        else:
            kept_ids.append(int(label))
    return {
        "labels": labels,
        "blobs": rows,
        "kept_ids": kept_ids,
        "rejected_ids": rejected_ids,
        "component_mode": "xy_only",
    }


def _normalize_blob_split_method(split_enabled: bool, split_method: str | None) -> str:
    value = str(split_method or "").strip().lower()
    if not value:
        return "height_borders" if split_enabled else "disabled"
    if not split_enabled and value != "disabled":
        return "disabled"
    if value in {"disabled", "histogram_gap", "height_borders", "height_borders_then_histogram"}:
        return value
    return "height_borders" if split_enabled else "disabled"


def _background_like_fragment(row: dict[str, Any]) -> bool:
    touches_border = bool(row.get("touches_frame_border")) or bool(row.get("touches_roi_border"))
    centrality = float(row.get("centrality") or 0.0)
    stripe_overlap = float(row.get("overlap_with_stripe_mask") or 0.0)
    return bool((touches_border or centrality <= 0.55) and stripe_overlap <= 0.20)


def _component_masks_from_binary(mask: np.ndarray) -> list[np.ndarray]:
    if not np.any(mask):
        return []
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out: list[np.ndarray] = []
    for label in range(1, num_labels):
        if int(stats[label, cv2.CC_STAT_AREA]) > 0:
            out.append(labels == label)
    return out


def _merge_fragment_masks_by_weak_boundaries(
    fragment_masks: list[np.ndarray],
    *,
    z: np.ndarray,
    gradient: np.ndarray,
    valid_mask: np.ndarray,
    roi_mask: np.ndarray,
    height_gate_mask: np.ndarray | None,
    total_valid: int,
    roi_border: np.ndarray,
    frame_shape: tuple[int, int],
    min_component_area: int,
    border_strength: np.ndarray | None,
    max_median_z_gap_mm: float,
    max_boundary_strength_mm: float,
    background_like_only: bool,
) -> tuple[list[np.ndarray], list[dict[str, Any]], int]:
    if len(fragment_masks) <= 1:
        return fragment_masks, [], 0
    rows = [
        _summarize_low_gradient_component_row(
            component_mask=(mask & valid_mask),
            component_id=index + 1,
            gradient=gradient,
            z=z,
            valid_mask=valid_mask,
            roi_mask=roi_mask,
            height_gate_mask=height_gate_mask,
            total_valid=total_valid,
            roi_border=roi_border,
            frame_shape=frame_shape,
            min_component_area=min_component_area,
        )
        for index, mask in enumerate(fragment_masks)
    ]
    parent = list(range(len(fragment_masks)))

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(a_idx: int, b_idx: int) -> None:
        pa = find(a_idx)
        pb = find(b_idx)
        if pa != pb:
            parent[pb] = pa

    merge_debug: list[dict[str, Any]] = []
    kernel = np.ones((3, 3), dtype=np.uint8)
    for i in range(len(fragment_masks)):
        row_i = rows[i]
        if row_i is None:
            continue
        dilated_i = cv2.dilate(fragment_masks[i].astype(np.uint8), kernel, iterations=1) > 0
        for j in range(i + 1, len(fragment_masks)):
            row_j = rows[j]
            if row_j is None:
                continue
            adjacency = dilated_i & cv2.dilate(fragment_masks[j].astype(np.uint8), kernel, iterations=1).astype(bool)
            if not np.any(adjacency):
                continue
            median_gap = abs(float(row_i.get("median_z") or 0.0) - float(row_j.get("median_z") or 0.0))
            boundary_strength = float(np.mean(border_strength[adjacency])) if border_strength is not None and np.any(adjacency & np.isfinite(border_strength)) else 0.0
            background_like = _background_like_fragment(row_i) and _background_like_fragment(row_j)
            should_merge = median_gap <= float(max_median_z_gap_mm) and boundary_strength <= float(max_boundary_strength_mm)
            if background_like_only and not background_like:
                should_merge = False
            merge_debug.append(
                {
                    "fragment_a": i + 1,
                    "fragment_b": j + 1,
                    "adjacent": True,
                    "median_z_gap_mm": median_gap,
                    "boundary_strength_mm": boundary_strength,
                    "background_like_pair": background_like,
                    "merged": bool(should_merge),
                }
            )
            if should_merge:
                union(i, j)
    groups: dict[int, list[np.ndarray]] = {}
    for idx, mask in enumerate(fragment_masks):
        groups.setdefault(find(idx), []).append(mask)
    merged_masks = [np.logical_or.reduce(group_masks) for group_masks in groups.values()]
    merge_count = max(0, len(fragment_masks) - len(merged_masks))
    return merged_masks, merge_debug, merge_count


def _histogram_split_fragment_masks(
    component_mask: np.ndarray,
    *,
    z: np.ndarray,
    valid_mask: np.ndarray,
    min_band_fraction: float,
    min_band_pixels: int,
    hist_bins: int,
    split_gap_mm: float,
    split_merge_gap_mm: float,
    refine_morphology: bool,
    open_kernel: int,
    close_kernel: int,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    mask = component_mask & valid_mask
    area_px = int(np.count_nonzero(mask))
    if area_px <= 0:
        return [], {"split": False, "reason": "empty_fragment"}
    zvals = z[mask].astype(np.float64)
    if zvals.size < 2:
        return [mask], {"split": False, "reason": "insufficient_samples"}
    z_lo = float(np.percentile(zvals, 2))
    z_hi = float(np.percentile(zvals, 98))
    if not np.isfinite(z_lo) or not np.isfinite(z_hi) or z_hi <= z_lo:
        return [mask], {"split": False, "reason": "degenerate_histogram_range"}
    clipped = zvals[(zvals >= z_lo) & (zvals <= z_hi)]
    if clipped.size < 2:
        clipped = zvals
    counts, edges = np.histogram(clipped, bins=max(8, int(hist_bins)), range=(z_lo, z_hi))
    smooth_counts = counts.astype(np.float64).copy()
    if smooth_counts.size >= 3:
        smooth_counts[1:-1] = (counts[:-2] + counts[1:-1] + counts[2:]) / 3.0
    occupied_floor = max(1.0, 0.10 * max(int(min_band_pixels), int(min_band_fraction * area_px)))
    occupied = smooth_counts >= occupied_floor
    runs: list[list[int]] = []
    start: int | None = None
    for idx, is_occupied in enumerate(occupied.tolist()):
        if is_occupied and start is None:
            start = idx
        elif (not is_occupied) and start is not None:
            runs.append([start, idx - 1])
            start = None
    if start is not None:
        runs.append([start, len(occupied) - 1])
    if len(runs) <= 1:
        return [mask], {"split": False, "reason": "no_significant_height_gap", "counts": counts.tolist(), "edges": edges.tolist()}
    merged_runs: list[list[int]] = [runs[0]]
    for s_idx, e_idx in runs[1:]:
        prev = merged_runs[-1]
        gap_mm = float(edges[s_idx] - edges[prev[1] + 1])
        if gap_mm < float(split_gap_mm) or gap_mm <= float(split_merge_gap_mm):
            prev[1] = e_idx
        else:
            merged_runs.append([s_idx, e_idx])
    out_masks: list[np.ndarray] = []
    band_debug: list[dict[str, Any]] = []
    for band_index, (s_idx, e_idx) in enumerate(merged_runs):
        band_lo = float(edges[s_idx])
        band_hi = float(edges[e_idx + 1])
        band_mask = mask & (z >= band_lo)
        band_mask = band_mask & (z < band_hi if band_index < len(merged_runs) - 1 else z <= band_hi)
        if refine_morphology and np.count_nonzero(band_mask):
            u8 = (band_mask.astype(np.uint8) * 255)
            if int(open_kernel) > 1:
                u8 = cv2.morphologyEx(u8, cv2.MORPH_OPEN, np.ones((max(1, int(open_kernel)), max(1, int(open_kernel))), np.uint8))
            if int(close_kernel) > 1:
                u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, np.ones((max(1, int(close_kernel)), max(1, int(close_kernel))), np.uint8))
            band_mask = (u8 > 0) & mask
        for component in _component_masks_from_binary(band_mask):
            pixels = int(np.count_nonzero(component))
            fraction = float(pixels / max(1, area_px))
            band_info = {"band_index": band_index, "z_min": band_lo, "z_max": band_hi, "pixels": pixels, "fraction": fraction, "kept": False}
            if pixels >= int(min_band_pixels) and fraction >= float(min_band_fraction):
                out_masks.append(component & mask)
                band_info["kept"] = True
            band_debug.append(band_info)
    if len(out_masks) <= 1:
        return [mask], {"split": False, "reason": "no_valid_bands_after_filtering", "bands": band_debug, "counts": counts.tolist(), "edges": edges.tolist()}
    return out_masks, {"split": True, "reason": "histogram_gap_split", "bands": band_debug, "counts": counts.tolist(), "edges": edges.tolist()}


def _height_border_split_fragment_masks(
    blob_mask: np.ndarray,
    *,
    z: np.ndarray,
    valid_mask: np.ndarray,
    border_mode: str,
    smoothing_kernel: int,
    threshold_mode: str,
    percentile: float,
    min_delta_mm: float,
    dilate_kernel: int,
    close_kernel: int,
    min_length_px: int,
    min_fragment_area_px: int,
    merge_weak_boundaries: bool,
    merge_max_median_z_gap_mm: float,
    merge_max_boundary_strength_mm: float,
    merge_background_like_only: bool,
    gradient: np.ndarray,
    roi_mask: np.ndarray,
    height_gate_mask: np.ndarray | None,
    total_valid: int,
    roi_border: np.ndarray,
    frame_shape: tuple[int, int],
    min_component_area: int,
) -> dict[str, Any]:
    mask = blob_mask & valid_mask
    if not np.any(mask):
        return {"split": False, "reason": "empty_blob_mask", "fragment_masks": [mask], "strength_map": np.zeros_like(z, dtype=np.float32), "cut_mask": np.zeros_like(mask, dtype=bool), "fragment_mask": mask.copy(), "fragment_labels": np.zeros_like(blob_mask, dtype=np.int32), "merge_debug": []}
    ys, xs = np.where(mask)
    pad = max(2, int(max(smoothing_kernel, dilate_kernel, close_kernel)))
    y0 = max(0, int(np.min(ys)) - pad)
    y1 = min(mask.shape[0], int(np.max(ys)) + pad + 1)
    x0 = max(0, int(np.min(xs)) - pad)
    x1 = min(mask.shape[1], int(np.max(xs)) + pad + 1)
    local_mask = mask[y0:y1, x0:x1]
    local_valid = valid_mask[y0:y1, x0:x1]
    local_z = z[y0:y1, x0:x1].astype(np.float32)
    local_fill = float(np.median(local_z[local_mask])) if np.any(local_mask) else 0.0
    local_z_filled = np.where(local_valid, local_z, local_fill).astype(np.float32)
    if int(smoothing_kernel) > 1:
        k = max(1, int(smoothing_kernel))
        if k % 2 == 0:
            k += 1
        local_z_filled = cv2.GaussianBlur(local_z_filled, (k, k), 0)
    mode = str(border_mode or "morphological_gradient").strip().lower()
    if mode == "sobel_gradient":
        gx = cv2.Sobel(local_z_filled, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(local_z_filled, cv2.CV_32F, 0, 1, ksize=3)
        local_strength = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    elif mode == "local_range":
        kernel = np.ones((3, 3), np.uint8)
        local_strength = (cv2.dilate(local_z_filled, kernel) - cv2.erode(local_z_filled, kernel)).astype(np.float32)
    else:
        kernel_size = max(3, int(smoothing_kernel))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        local_strength = (cv2.dilate(local_z_filled, kernel) - cv2.erode(local_z_filled, kernel)).astype(np.float32)
    local_strength = np.where(local_valid, local_strength, 0.0)
    strength_vals = local_strength[local_mask]
    if strength_vals.size == 0:
        return {"split": False, "reason": "no_strength_inside_blob", "fragment_masks": [mask], "strength_map": np.zeros_like(z, dtype=np.float32), "cut_mask": np.zeros_like(mask, dtype=bool), "fragment_mask": mask.copy(), "fragment_labels": np.zeros_like(blob_mask, dtype=np.int32), "merge_debug": []}
    threshold_mode_normalized = str(threshold_mode or "percentile").strip().lower()
    if threshold_mode_normalized == "fixed":
        threshold_mm = float(min_delta_mm)
    elif threshold_mode_normalized == "otsu":
        threshold_mm = max(float(_otsu_threshold(strength_vals)), float(min_delta_mm))
    else:
        threshold_mm = max(float(np.percentile(strength_vals, float(np.clip(percentile, 1.0, 99.0)))), float(min_delta_mm))
    local_cut = (local_strength >= threshold_mm) & local_mask
    if int(close_kernel) > 1:
        local_cut = cv2.morphologyEx((local_cut.astype(np.uint8) * 255), cv2.MORPH_CLOSE, np.ones((int(close_kernel), int(close_kernel)), np.uint8)) > 0
    if int(dilate_kernel) > 1:
        local_cut = cv2.dilate(local_cut.astype(np.uint8), np.ones((int(dilate_kernel), int(dilate_kernel)), np.uint8), iterations=1) > 0
    filtered_cut = np.zeros_like(local_cut, dtype=bool)
    edge_candidate_pixels = int(np.count_nonzero(local_cut))
    for component in _component_masks_from_binary(local_cut):
        ys_c, xs_c = np.where(component)
        length_px = int(max((np.max(xs_c) - np.min(xs_c) + 1) if xs_c.size else 0, (np.max(ys_c) - np.min(ys_c) + 1) if ys_c.size else 0))
        if length_px >= int(min_length_px):
            filtered_cut |= component
    fragment_seed = local_mask & (~filtered_cut)
    fragment_masks_local = [component for component in _component_masks_from_binary(fragment_seed) if int(np.count_nonzero(component)) >= int(min_fragment_area_px)]
    if len(fragment_masks_local) <= 1:
        strength_map = np.zeros_like(z, dtype=np.float32)
        strength_map[y0:y1, x0:x1] = local_strength
        cut_mask = np.zeros_like(mask, dtype=bool)
        cut_mask[y0:y1, x0:x1] = filtered_cut & local_mask
        return {
            "split": False,
            "reason": "no_valid_height_border_split",
            "fragment_masks": [mask],
            "strength_map": strength_map,
            "cut_mask": cut_mask,
            "fragment_mask": mask.copy(),
            "fragment_labels": np.zeros_like(blob_mask, dtype=np.int32),
            "threshold_mm": threshold_mm,
            "edge_candidate_pixels": edge_candidate_pixels,
            "cut_pixels": int(np.count_nonzero(filtered_cut & local_mask)),
            "initial_fragments": len(fragment_masks_local),
            "kept_fragments": 1,
            "merged_fragments": 0,
            "height_border_mean_strength": float(np.mean(strength_vals)),
            "height_border_max_strength": float(np.max(strength_vals)) if strength_vals.size else 0.0,
            "merge_debug": [],
        }
    fragment_masks_global = []
    for component in fragment_masks_local:
        global_mask = np.zeros_like(mask, dtype=bool)
        global_mask[y0:y1, x0:x1] = component
        fragment_masks_global.append(global_mask & mask)
    merge_debug: list[dict[str, Any]] = []
    merge_count = 0
    if merge_weak_boundaries:
        local_strength_global = np.zeros_like(z, dtype=np.float32)
        local_strength_global[y0:y1, x0:x1] = local_strength
        fragment_masks_global, merge_debug, merge_count = _merge_fragment_masks_by_weak_boundaries(
            fragment_masks_global,
            z=z,
            gradient=gradient,
            valid_mask=valid_mask,
            roi_mask=roi_mask,
            height_gate_mask=height_gate_mask,
            total_valid=total_valid,
            roi_border=roi_border,
            frame_shape=frame_shape,
            min_component_area=min_component_area,
            border_strength=local_strength_global,
            max_median_z_gap_mm=merge_max_median_z_gap_mm,
            max_boundary_strength_mm=merge_max_boundary_strength_mm,
            background_like_only=merge_background_like_only,
        )
    fragment_labels = np.zeros_like(blob_mask, dtype=np.int32)
    fragment_mask = np.zeros_like(blob_mask, dtype=bool)
    for index, component in enumerate(fragment_masks_global, start=1):
        fragment_labels[component] = index
        fragment_mask |= component
    strength_map = np.zeros_like(z, dtype=np.float32)
    strength_map[y0:y1, x0:x1] = local_strength
    cut_mask = np.zeros_like(mask, dtype=bool)
    cut_mask[y0:y1, x0:x1] = filtered_cut & local_mask
    return {
        "split": len(fragment_masks_global) > 1,
        "reason": "height_border_split" if len(fragment_masks_global) > 1 else "no_valid_height_border_split",
        "fragment_masks": fragment_masks_global,
        "strength_map": strength_map,
        "cut_mask": cut_mask,
        "fragment_mask": fragment_mask,
        "fragment_labels": fragment_labels,
        "threshold_mm": threshold_mm,
        "edge_candidate_pixels": edge_candidate_pixels,
        "cut_pixels": int(np.count_nonzero(filtered_cut & local_mask)),
        "initial_fragments": len(fragment_masks_local),
        "kept_fragments": len(fragment_masks_global),
        "merged_fragments": merge_count,
        "height_border_mean_strength": float(np.mean(strength_vals)),
        "height_border_max_strength": float(np.max(strength_vals)) if strength_vals.size else 0.0,
        "merge_debug": merge_debug,
    }


def _split_low_gradient_blob_candidates_by_height(
    *,
    labels: np.ndarray,
    blob_rows: list[dict[str, Any]],
    gradient: np.ndarray,
    z: np.ndarray,
    valid_mask: np.ndarray,
    roi_mask: np.ndarray,
    height_gate_mask: np.ndarray | None,
    min_component_area: int,
    split_enabled: bool,
    split_method: str,
    split_min_height_range_mm: float,
    split_min_pixels: int,
    split_gap_mm: float,
    split_mode: str,
    split_hist_bins: int,
    split_min_band_fraction: float,
    split_min_band_pixels: int,
    split_merge_gap_mm: float,
    split_refine_morphology: bool,
    split_open_kernel: int,
    split_close_kernel: int,
    split_height_border_enabled: bool,
    split_height_border_mode: str,
    split_height_border_smoothing_kernel: int,
    split_height_border_threshold_mode: str,
    split_height_border_percentile: float,
    split_height_border_min_delta_mm: float,
    split_height_border_dilate_kernel: int,
    split_height_border_close_kernel: int,
    split_height_border_min_length_px: int,
    split_min_fragment_area_px: int,
    split_merge_weak_boundaries: bool,
    split_merge_max_median_z_gap_mm: float,
    split_merge_max_boundary_strength_mm: float,
    split_merge_background_like_only: bool,
) -> dict[str, Any]:
    total_valid = max(1, int(np.count_nonzero(valid_mask)))
    roi_border = _roi_border_mask(roi_mask)
    fragment_labels = np.zeros_like(labels, dtype=np.int32)
    border_fragment_labels = np.zeros_like(labels, dtype=np.int32)
    fragment_rows: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []
    merge_debug_rows: list[dict[str, Any]] = []
    kept_ids: list[int] = []
    rejected_ids: list[int] = []
    next_fragment_id = 1
    next_border_fragment_id = 1
    strength_map = np.zeros_like(z, dtype=np.float32)
    cut_mask = np.zeros_like(labels, dtype=bool)
    border_fragment_mask = np.zeros_like(labels, dtype=bool)
    method = _normalize_blob_split_method(split_enabled, split_method)
    eligible_blob_count = 0
    height_border_split_blob_count = 0
    histogram_split_blob_count = 0
    merge_back_count = 0
    warnings: list[str] = []

    def append_fragment(
        mask: np.ndarray,
        *,
        source_blob_id: int,
        split_index: int,
        was_split: bool,
        split_method_value: str,
        split_stage: str,
        height_border_split: bool,
        histogram_split: bool,
        merge_group_id: int | None,
        border_mean_strength: float = 0.0,
        border_max_strength: float = 0.0,
        boundary_reason: str | None = None,
    ) -> int | None:
        nonlocal next_fragment_id
        if not np.any(mask & valid_mask):
            return None
        fragment_id = int(next_fragment_id)
        next_fragment_id += 1
        fragment_labels[mask & valid_mask] = fragment_id
        row = _summarize_low_gradient_component_row(
            component_mask=(mask & valid_mask),
            component_id=fragment_id,
            gradient=gradient,
            z=z,
            valid_mask=valid_mask,
            roi_mask=roi_mask,
            height_gate_mask=height_gate_mask,
            total_valid=total_valid,
            roi_border=roi_border,
            frame_shape=labels.shape,
            min_component_area=min_component_area,
            extra_fields={
                "fragment_id": fragment_id,
                "source_blob_id": int(source_blob_id),
                "split_index": int(split_index),
                "was_split_from_blob": bool(was_split),
                "split_method": split_method_value,
                "split_stage": split_stage,
                "height_border_split": bool(height_border_split),
                "histogram_split": bool(histogram_split),
                "height_border_mean_strength": float(border_mean_strength),
                "height_border_max_strength": float(border_max_strength),
                "height_border_boundary_reason": boundary_reason,
                "merge_group_id": None if merge_group_id is None else int(merge_group_id),
            },
        )
        if row is None:
            fragment_labels[mask & valid_mask] = 0
            next_fragment_id -= 1
            return None
        fragment_rows.append(row)
        if row["rejected"]:
            rejected_ids.append(fragment_id)
        else:
            kept_ids.append(fragment_id)
        return fragment_id

    for blob_row in blob_rows:
        source_blob_id = int(blob_row.get("blob_id") or 0)
        blob_mask = (labels == source_blob_id) & valid_mask
        area_px = int(np.count_nonzero(blob_mask))
        z_p05 = float(blob_row.get("z_p05") or 0.0)
        z_p95 = float(blob_row.get("z_p95") or 0.0)
        z_range_robust = float(z_p95 - z_p05)
        debug_entry: dict[str, Any] = {
            "source_blob_id": source_blob_id,
            "area_px": area_px,
            "z_p05": z_p05,
            "z_p50": float(blob_row.get("z_p50") or blob_row.get("median_z") or 0.0),
            "z_p95": z_p95,
            "z_range_robust": z_range_robust,
            "z_mad": float(blob_row.get("mad_z") or 0.0),
            "eligible": False,
            "split": False,
            "method": method,
            "reason": "disabled",
            "fragments": [],
        }
        if not np.any(blob_mask):
            debug_entry["reason"] = "empty_blob_mask"
            debug_rows.append(debug_entry)
            continue
        eligible = area_px >= int(split_min_pixels) and z_range_robust >= float(split_min_height_range_mm)
        debug_entry["eligible"] = bool(eligible)
        if eligible:
            eligible_blob_count += 1
        if method == "disabled" or not eligible:
            debug_entry["reason"] = "disabled" if method == "disabled" else ("below_min_pixels" if area_px < int(split_min_pixels) else "below_min_height_range")
            append_fragment(
                blob_mask,
                source_blob_id=source_blob_id,
                split_index=0,
                was_split=False,
                split_method_value="disabled",
                split_stage="none",
                height_border_split=False,
                histogram_split=False,
                merge_group_id=None,
                boundary_reason=debug_entry["reason"],
            )
            debug_rows.append(debug_entry)
            continue

        current_masks = [blob_mask]
        current_stage = "original_blob"
        border_result: dict[str, Any] | None = None
        if method in {"height_borders", "height_borders_then_histogram"} and bool(split_height_border_enabled):
            border_result = _height_border_split_fragment_masks(
                blob_mask,
                z=z,
                valid_mask=valid_mask,
                border_mode=split_height_border_mode,
                smoothing_kernel=split_height_border_smoothing_kernel,
                threshold_mode=split_height_border_threshold_mode,
                percentile=split_height_border_percentile,
                min_delta_mm=split_height_border_min_delta_mm,
                dilate_kernel=split_height_border_dilate_kernel,
                close_kernel=split_height_border_close_kernel,
                min_length_px=split_height_border_min_length_px,
                min_fragment_area_px=split_min_fragment_area_px,
                merge_weak_boundaries=split_merge_weak_boundaries,
                merge_max_median_z_gap_mm=split_merge_max_median_z_gap_mm,
                merge_max_boundary_strength_mm=split_merge_max_boundary_strength_mm,
                merge_background_like_only=split_merge_background_like_only,
                gradient=gradient,
                roi_mask=roi_mask,
                height_gate_mask=height_gate_mask,
                total_valid=total_valid,
                roi_border=roi_border,
                frame_shape=labels.shape,
                min_component_area=min_component_area,
            )
            strength_map = np.maximum(strength_map, np.asarray(border_result.get("strength_map"), dtype=np.float32))
            cut_mask |= np.asarray(border_result.get("cut_mask"), dtype=bool)
            if border_result.get("split"):
                height_border_split_blob_count += 1
                current_masks = [np.asarray(mask, dtype=bool) for mask in (border_result.get("fragment_masks") or []) if np.any(mask)]
                current_stage = "height_borders"
                local_border_labels = np.asarray(border_result.get("fragment_labels"), dtype=np.int32)
                for local_id in sorted(int(v) for v in np.unique(local_border_labels) if int(v) > 0):
                    border_fragment_labels[local_border_labels == local_id] = next_border_fragment_id
                    next_border_fragment_id += 1
                border_fragment_mask |= np.asarray(border_result.get("fragment_mask"), dtype=bool)
            merge_debug_rows.extend(
                [
                    {"source_blob_id": source_blob_id, **entry}
                    for entry in list(border_result.get("merge_debug") or [])
                ]
            )
            merge_back_count += int(border_result.get("merged_fragments") or 0)
            debug_entry.update(
                {
                    "threshold_mm": border_result.get("threshold_mm"),
                    "edge_candidate_pixels": border_result.get("edge_candidate_pixels"),
                    "cut_pixels": border_result.get("cut_pixels"),
                    "initial_fragments": border_result.get("initial_fragments"),
                    "kept_fragments": border_result.get("kept_fragments"),
                    "merged_fragments": border_result.get("merged_fragments"),
                    "height_border_mean_strength": border_result.get("height_border_mean_strength"),
                    "height_border_max_strength": border_result.get("height_border_max_strength"),
                }
            )
            if not border_result.get("split") and method == "height_borders":
                debug_entry["reason"] = border_result.get("reason") or "no_valid_height_border_split"
            elif border_result.get("split"):
                debug_entry["reason"] = "height_border_split"

        final_masks: list[np.ndarray] = []
        split_index = 0
        for base_mask in current_masks:
            if method in {"histogram_gap", "height_borders_then_histogram"}:
                hist_masks, hist_debug = _histogram_split_fragment_masks(
                    base_mask,
                    z=z,
                    valid_mask=valid_mask,
                    min_band_fraction=split_min_band_fraction,
                    min_band_pixels=split_min_band_pixels,
                    hist_bins=split_hist_bins,
                    split_gap_mm=split_gap_mm,
                    split_merge_gap_mm=split_merge_gap_mm,
                    refine_morphology=split_refine_morphology,
                    open_kernel=split_open_kernel,
                    close_kernel=split_close_kernel,
                )
                if hist_debug.get("split"):
                    histogram_split_blob_count += 1
                    final_masks.extend(hist_masks)
                else:
                    final_masks.append(base_mask)
                if method == "histogram_gap":
                    debug_entry["histogram"] = hist_debug
            else:
                final_masks.append(base_mask)
        if not final_masks:
            final_masks = [blob_mask]
        if not np.any(border_fragment_mask & blob_mask):
            border_fragment_mask |= blob_mask
            border_fragment_labels[blob_mask] = next_border_fragment_id
            next_border_fragment_id += 1

        fragment_ids_for_blob: list[int] = []
        any_border_split = bool(border_result and border_result.get("split"))
        any_histogram_split = method in {"histogram_gap", "height_borders_then_histogram"} and len(final_masks) > len(current_masks)
        for component_idx, component_mask in enumerate(final_masks):
            component_mask = np.asarray(component_mask, dtype=bool) & blob_mask & valid_mask
            if int(np.count_nonzero(component_mask)) < max(1, int(split_min_fragment_area_px)):
                continue
            fragment_id = append_fragment(
                component_mask,
                source_blob_id=source_blob_id,
                split_index=split_index,
                was_split=bool(any_border_split or any_histogram_split),
                split_method_value=method,
                split_stage=current_stage if current_stage != "original_blob" else ("histogram_gap" if any_histogram_split else "none"),
                height_border_split=bool(any_border_split),
                histogram_split=bool(any_histogram_split),
                merge_group_id=component_idx + 1 if any_border_split else None,
                border_mean_strength=float(border_result.get("height_border_mean_strength") or 0.0) if border_result else 0.0,
                border_max_strength=float(border_result.get("height_border_max_strength") or 0.0) if border_result else 0.0,
                boundary_reason=str(debug_entry.get("reason") or ""),
            )
            split_index += 1
            if fragment_id is not None:
                fragment_ids_for_blob.append(fragment_id)
        if len(fragment_ids_for_blob) <= 1 and (any_border_split or any_histogram_split):
            fragment_labels[np.isin(fragment_labels, fragment_ids_for_blob)] = 0
            fragment_rows[:] = [row for row in fragment_rows if int(row.get("fragment_id") or 0) not in set(fragment_ids_for_blob)]
            kept_ids[:] = [frag_id for frag_id in kept_ids if frag_id not in set(fragment_ids_for_blob)]
            rejected_ids[:] = [frag_id for frag_id in rejected_ids if frag_id not in set(fragment_ids_for_blob)]
            next_fragment_id -= len(fragment_ids_for_blob)
            fragment_ids_for_blob = []
            append_fragment(
                blob_mask,
                source_blob_id=source_blob_id,
                split_index=0,
                was_split=False,
                split_method_value="disabled",
                split_stage="none",
                height_border_split=False,
                histogram_split=False,
                merge_group_id=None,
                boundary_reason="fallback_to_original_blob",
            )
            debug_entry["reason"] = "no_valid_split"
        debug_entry["split"] = len(fragment_ids_for_blob) > 1
        debug_entry["fragments"] = fragment_ids_for_blob
        if not debug_entry["split"] and not str(debug_entry.get("reason") or "").startswith("below_"):
            debug_entry["reason"] = str(debug_entry.get("reason") or "no_valid_split")
        debug_rows.append(debug_entry)

    using_split_fragments = bool(any(bool(row.get("was_split_from_blob")) for row in fragment_rows))
    if method != "disabled" and not using_split_fragments:
        warnings.append("height splitting produced no valid fragments; using original blob components")
    return {
        "labels": fragment_labels,
        "fragments": fragment_rows,
        "kept_ids": kept_ids,
        "rejected_ids": rejected_ids,
        "debug": debug_rows,
        "using_split_fragments": using_split_fragments,
        "method": method,
        "eligible_blob_count": eligible_blob_count,
        "height_border_split_blob_count": height_border_split_blob_count,
        "histogram_split_blob_count": histogram_split_blob_count,
        "merge_back_count": merge_back_count,
        "height_border_strength": strength_map,
        "height_border_cut_mask": cut_mask,
        "height_border_fragment_labels": border_fragment_labels,
        "height_border_fragment_mask": border_fragment_mask,
        "merge_debug": merge_debug_rows,
        "warnings": warnings,
    }


def _cluster_low_gradient_blobs_by_height(
    blobs: list[dict[str, Any]],
    *,
    height_gap_mm: float,
    height_gap_mode: str,
    merge_close_clusters: bool,
    merge_gap_mm: float,
) -> list[dict[str, Any]]:
    accepted = [row for row in blobs if not bool(row.get("rejected"))]
    if not accepted:
        return []
    ordered = sorted(accepted, key=lambda row: (float(row.get("median_z") or 0.0), -int(row.get("area_px") or 0)))
    adaptive_gap = float(height_gap_mm)
    if str(height_gap_mode).lower() == "adaptive":
        mads = [float(row.get("mad_z") or 0.0) for row in ordered]
        adaptive_gap = max(float(height_gap_mm), float(np.median(mads) * 2.0) if mads else float(height_gap_mm))
    clusters: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    for row in ordered:
        if not current:
            current = [row]
            continue
        gap = abs(float(row.get("median_z") or 0.0) - float(current[-1].get("median_z") or 0.0))
        if gap <= adaptive_gap:
            current.append(row)
        else:
            clusters.append({"blob_rows": current[:]})
            current = [row]
    if current:
        clusters.append({"blob_rows": current[:]})

    if merge_close_clusters and len(clusters) > 1:
        merged: list[dict[str, Any]] = [clusters[0]]
        for cluster in clusters[1:]:
            prev = merged[-1]
            prev_z = float(np.average([float(row.get("median_z") or 0.0) for row in prev["blob_rows"]], weights=[max(1, int(row.get("area_px") or 0)) for row in prev["blob_rows"]]))
            cur_z = float(np.average([float(row.get("median_z") or 0.0) for row in cluster["blob_rows"]], weights=[max(1, int(row.get("area_px") or 0)) for row in cluster["blob_rows"]]))
            if abs(cur_z - prev_z) <= float(merge_gap_mm):
                prev["blob_rows"].extend(cluster["blob_rows"])
            else:
                merged.append(cluster)
        clusters = merged

    out: list[dict[str, Any]] = []
    for index, cluster in enumerate(clusters, start=1):
        rows = list(cluster["blob_rows"])
        total_pixels = int(sum(int(row.get("area_px") or 0) for row in rows))
        weights = np.asarray([max(1, int(row.get("area_px") or 0)) for row in rows], dtype=np.float64)
        medians = np.asarray([float(row.get("median_z") or 0.0) for row in rows], dtype=np.float64)
        mad_vals = np.asarray([float(row.get("mad_z") or 0.0) for row in rows], dtype=np.float64)
        p05_vals = [float(row.get("z_p05") or 0.0) for row in rows]
        p95_vals = [float(row.get("z_p95") or 0.0) for row in rows]
        out.append(
            {
                "cluster_id": f"cluster_{index:03d}",
                "blob_ids": [int(row.get("blob_id") or 0) for row in rows],
                "source_blob_ids": sorted({int(row.get("source_blob_id") or row.get("blob_id") or 0) for row in rows}),
                "blob_count": int(len(rows)),
                "fragment_count": int(len(rows)),
                "original_blob_count": int(len({int(row.get("source_blob_id") or row.get("blob_id") or 0) for row in rows})),
                "split_fragment_count": int(sum(1 for row in rows if bool(row.get("was_split_from_blob")))),
                "mixed_blob_source_count": int(
                    sum(
                        1
                        for source_blob_id in {int(row.get("source_blob_id") or row.get("blob_id") or 0) for row in rows}
                        if sum(1 for row in rows if int(row.get("source_blob_id") or row.get("blob_id") or 0) == source_blob_id) > 1
                    )
                ),
                "total_pixels": total_pixels,
                "area_fraction": 0.0,
                "median_z": float(np.average(medians, weights=weights)),
                "mean_z": float(np.average(np.asarray([float(row.get("mean_z") or 0.0) for row in rows]), weights=weights)),
                "z_mad_weighted": float(np.average(mad_vals, weights=weights)),
                "z_min": float(min(p05_vals)) if p05_vals else 0.0,
                "z_max": float(max(p95_vals)) if p95_vals else 0.0,
                "touches_roi_border_fraction": float(sum(1 for row in rows if row.get("touches_roi_border")) / max(1, len(rows))),
                "touches_frame_border_fraction": float(sum(1 for row in rows if row.get("touches_frame_border")) / max(1, len(rows))),
                "mean_centrality": float(np.average(np.asarray([float(row.get("centrality") or 0.0) for row in rows]), weights=weights)),
                "mean_gradient_p95": float(np.average(np.asarray([float(row.get("gradient_p95") or 0.0) for row in rows]), weights=weights)),
                "mean_solidity": float(np.average(np.asarray([float(row.get("solidity") or 0.0) for row in rows]), weights=weights)),
                "mean_compactness": float(np.average(np.asarray([float(row.get("compactness") or 0.0) for row in rows]), weights=weights)),
                "height_gate_overlap": float(np.average(np.asarray([float(row.get("overlap_with_height_gate") or 0.0) for row in rows]), weights=weights)),
                "stripe_overlap": float(np.average(np.asarray([float(row.get("overlap_with_stripe_mask") or 0.0) for row in rows]), weights=weights)),
                "blob_rows": rows,
            }
        )
    return out


def _score_blob_height_clusters(
    clusters: list[dict[str, Any]],
    *,
    valid_pixel_count: int,
    min_area_fraction: float,
    min_total_pixels: int,
    max_z_mad_mm: float,
    area_weight: float,
    border_weight: float,
    constancy_weight: float,
    spatial_coverage_weight: float,
    depth_preference_weight: float,
    fragmentation_penalty: float,
    centrality_penalty: float,
    object_like_penalty: float,
    stripe_overlap_penalty: float,
    inferred_depth_preference: str = "lower_z",
) -> list[dict[str, Any]]:
    total_valid = max(1, int(valid_pixel_count))
    if not clusters:
        return []
    z_values = np.asarray([float(row.get("median_z") or 0.0) for row in clusters], dtype=np.float64)
    z_min = float(np.min(z_values))
    z_max = float(np.max(z_values))
    z_span = max(1e-6, z_max - z_min)
    scored: list[dict[str, Any]] = []
    for row in clusters:
        total_pixels = int(row.get("total_pixels") or 0)
        area_fraction = float(total_pixels / total_valid)
        row["area_fraction"] = area_fraction
        constancy = 1.0 / (1.0 + max(0.0, float(row.get("z_mad_weighted") or 0.0)) + max(0.0, float(row.get("mean_gradient_p95") or 0.0)))
        border_term = 0.5 * float(row.get("touches_roi_border_fraction") or 0.0) + 0.5 * float(row.get("touches_frame_border_fraction") or 0.0)
        coverage_term = min(1.0, area_fraction / max(1e-6, float(min_area_fraction)))
        med_z = float(row.get("median_z") or 0.0)
        depth_term = (z_max - med_z) / z_span if inferred_depth_preference == "lower_z" else (med_z - z_min) / z_span
        fragment_count = max(1.0, float(row.get("fragment_count") or row.get("blob_count") or 1))
        fragmentation_term = max(0.0, fragment_count - 1.0) / fragment_count
        centrality_term = float(row.get("mean_centrality") or 0.0)
        object_like_term = max(0.0, float(row.get("mean_solidity") or 0.0) - 0.90) + max(0.0, float(row.get("mean_compactness") or 0.0) - 0.85)
        stripe_term = float(row.get("stripe_overlap") or 0.0)
        score = (
            float(area_weight) * area_fraction
            + float(border_weight) * border_term
            + float(constancy_weight) * constancy
            + float(spatial_coverage_weight) * coverage_term
            + float(depth_preference_weight) * depth_term
            - float(fragmentation_penalty) * fragmentation_term
            - float(centrality_penalty) * centrality_term
            - float(object_like_penalty) * object_like_term
            - float(stripe_overlap_penalty) * stripe_term
        )
        rejected_reason = None
        if total_pixels < int(min_total_pixels):
            rejected_reason = "cluster_below_min_total_pixels"
        elif area_fraction < float(min_area_fraction):
            rejected_reason = "cluster_below_min_area_fraction"
        elif float(row.get("z_mad_weighted") or 0.0) > float(max_z_mad_mm):
            rejected_reason = "cluster_above_max_z_mad"
        scored_row = {
            **row,
            "score": float(score),
            "score_terms": {
                "area": float(area_weight) * area_fraction,
                "border": float(border_weight) * border_term,
                "constancy": float(constancy_weight) * constancy,
                "spatial_coverage": float(spatial_coverage_weight) * coverage_term,
                "depth_preference": float(depth_preference_weight) * depth_term,
                "fragmentation_penalty": -float(fragmentation_penalty) * fragmentation_term,
                "centrality_penalty": -float(centrality_penalty) * centrality_term,
                "object_like_penalty": -float(object_like_penalty) * object_like_term,
                "stripe_overlap_penalty": -float(stripe_overlap_penalty) * stripe_term,
            },
            # Additive, UI-facing explanation fields. These are diagnostics only
            # and do not affect `score` or the selection decision below.
            "border_contact": bool(border_term > 0.0),
            "spatial_coverage": float(coverage_term),
            "centrality_penalty": float(centrality_term),
            "object_like_penalty": float(object_like_term),
            "fragmentation_penalty": float(fragmentation_term),
            "stripe_overlap": float(stripe_term),
            "z_mad": float(row.get("z_mad_weighted") or 0.0),
            "rejected": rejected_reason is not None,
            "rejected_reason": rejected_reason,
        }
        scored.append(scored_row)
    return sorted(scored, key=lambda item: (-float(item.get("score") or 0.0), str(item.get("cluster_id") or "")))


def _write_depth_plot_png(
    z_values: np.ndarray,
    *,
    near_q: float,
    far_q: float,
    out_path: Path,
    title_override: str | None = None,
    near_label: str = "near_q",
    far_label: str = "far_q",
    max_render_samples: int = 60000,
    y_robust_percentile: float | None = None,
) -> None:
    """Plot the sorted-z scanline of valid pixels.

    Parameters
    ----------
    max_render_samples:
        Cap on the number of dots rendered. A deterministic stride is used
        so the plot stays visually stable across reprocessings.
    y_robust_percentile:
        Optional percentile used to clip the y-axis (and a matching one
        from the bottom). Defaults to the data min/max. Setting this in
        the (95, 100) range produces a plot focused on the bulk of the
        distribution when a handful of outliers stretch the range.
    """
    vals = np.asarray(z_values, dtype=np.float64)
    if vals.size == 0:
        canvas = np.zeros((420, 1200, 3), dtype=np.uint8)
        cv2.putText(
            canvas,
            title_override or "Depth distribution - no samples",
            (40, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(out_path), canvas)
        return
    width, height = 1200, 420
    margin_l, margin_r, margin_t, margin_b = 60, 20, 20, 40
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    canvas = np.full((height, width, 3), 22, dtype=np.uint8)

    full_count = int(vals.size)
    cap = max(100, int(max_render_samples))
    if vals.size > cap:
        step = int(np.ceil(vals.size / cap))
        vals = vals[::step]
    vals_sorted = np.sort(vals)
    raw_min = float(vals_sorted[0])
    raw_max = float(vals_sorted[-1])
    if y_robust_percentile is not None and 50.0 < float(y_robust_percentile) < 100.0:
        hi_p = float(y_robust_percentile)
        lo_p = 100.0 - hi_p
        vmin = float(np.percentile(vals_sorted, lo_p))
        vmax = float(np.percentile(vals_sorted, hi_p))
    else:
        vmin = raw_min
        vmax = raw_max
    if vmax <= vmin:
        vmax = vmin + 1.0
    y_span = vmax - vmin

    xs = np.linspace(0, vals_sorted.size - 1, num=vals_sorted.size, dtype=np.float64)
    xpix = margin_l + (xs / max(1.0, float(vals_sorted.size - 1))) * plot_w
    ypix = margin_t + (1.0 - np.clip(((vals_sorted - vmin) / y_span), 0.0, 1.0)) * plot_h
    pts = np.column_stack((xpix.astype(np.int32), ypix.astype(np.int32)))
    for p in pts:
        cv2.circle(canvas, (int(p[0]), int(p[1])), 1, (180, 180, 180), -1)

    def y_for(v: float) -> int:
        clipped = float(np.clip(v, vmin, vmax))
        return int(round(margin_t + (1.0 - ((clipped - vmin) / y_span)) * plot_h))

    fmt = _adaptive_tick_format(y_span)
    if np.isfinite(near_q):
        y_near = y_for(near_q)
        cv2.line(canvas, (margin_l, y_near), (margin_l + plot_w, y_near), (255, 180, 0), 1)
        cv2.putText(canvas, f"{near_label}={fmt % float(near_q)}", (margin_l + 6, max(14, y_near - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 180, 0), 1, cv2.LINE_AA)
    if np.isfinite(far_q):
        y_far = y_for(far_q)
        cv2.line(canvas, (margin_l, y_far), (margin_l + plot_w, y_far), (0, 220, 255), 1)
        cv2.putText(canvas, f"{far_label}={fmt % float(far_q)}", (margin_l + 6, max(14, y_far - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (margin_l, margin_t), (margin_l + plot_w, margin_t + plot_h), (90, 90, 90), 1)
    title = title_override or "Depth distribution (valid z)"
    rendered = int(vals_sorted.size)
    title_extra = f"n_render={rendered}/{full_count}" if rendered != full_count else f"n={full_count}"
    cv2.putText(canvas, f"{title}, {title_extra}", (margin_l, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    y_label = (
        f"y_min={fmt % vmin}  y_max={fmt % vmax}  (raw_min={fmt % raw_min}, raw_max={fmt % raw_max})"
        if (vmin != raw_min or vmax != raw_max)
        else f"z_min={fmt % vmin}  z_max={fmt % vmax}"
    )
    cv2.putText(canvas, y_label, (margin_l, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
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


def _radial_boundary_uniformity_from_contour_mm(contour_mm: list[list[float]] | None) -> dict[str, Any]:
    if not isinstance(contour_mm, list) or len(contour_mm) < 8:
        return {
            "radial_mean": None,
            "radial_std": None,
            "radial_cv": None,
            "radial_max_deviation": None,
        }
    pts = np.asarray(contour_mm, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 2:
        return {
            "radial_mean": None,
            "radial_std": None,
            "radial_cv": None,
            "radial_max_deviation": None,
        }
    center = np.mean(pts[:, :2], axis=0)
    radii = np.sqrt(np.sum((pts[:, :2] - center) ** 2, axis=1))
    if radii.size <= 0:
        return {
            "radial_mean": None,
            "radial_std": None,
            "radial_cv": None,
            "radial_max_deviation": None,
        }
    radial_mean = float(np.mean(radii))
    radial_std = float(np.std(radii))
    radial_cv = float(radial_std / max(radial_mean, 1e-9))
    radial_max_deviation = float(np.max(np.abs(radii - radial_mean)))
    return {
        "radial_mean": round(radial_mean, 4),
        "radial_std": round(radial_std, 4),
        "radial_cv": round(radial_cv, 6),
        "radial_max_deviation": round(radial_max_deviation, 4),
    }


def _fit_sphere_least_squares(points_xyz_mm: np.ndarray) -> dict[str, Any]:
    return fit_sphere_least_squares(points_xyz_mm)


def _fit_ellipsoid_pca(points_xyz_mm: np.ndarray) -> dict[str, Any]:
    if points_xyz_mm.ndim != 2 or points_xyz_mm.shape[1] != 3 or points_xyz_mm.shape[0] < 8:
        return {"valid": False}
    xyz = np.asarray(points_xyz_mm, dtype=np.float64)
    center = np.mean(xyz, axis=0)
    centered = xyz - center
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eigh(cov)
    evals = np.maximum(evals, 1e-12)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    # Gaussian-to-surface proxy scaling factor; used as a stable geometric proxy.
    axes = np.sqrt(evals) * 2.0
    u = centered @ evecs
    n = (u[:, 0] / max(float(axes[0]), 1e-9)) ** 2 + (u[:, 1] / max(float(axes[1]), 1e-9)) ** 2 + (u[:, 2] / max(float(axes[2]), 1e-9)) ** 2
    residual = np.abs(n - 1.0)
    ar = float(max(axes) / max(min(axes), 1e-9))
    return {
        "valid": True,
        "center_mm": [round(float(center[0]), 4), round(float(center[1]), 4), round(float(center[2]), 4)],
        "axes_mm": [round(float(axes[0]), 4), round(float(axes[1]), 4), round(float(axes[2]), 4)],
        "aspect_ratio": round(ar, 6),
        "rmse": round(float(np.sqrt(np.mean(residual ** 2))), 6),
        "point_count": int(xyz.shape[0]),
    }


def _roi_from_metadata(metadata: dict[str, Any]) -> dict[str, Any] | None:
    roi = metadata.get("roi_25d")
    return roi if isinstance(roi, dict) else None


def _roi_mask(shape: tuple[int, int], roi: dict[str, Any], *, region_mode: str | None = None) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    roi_type = str(region_mode or roi.get("type") or roi.get("roi_type") or "rectangle").lower()
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
    if roi_type in {"vertical_band", "full_height_x_band"}:
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
    footprint = item.get("footprint_geometry") if isinstance(item.get("footprint_geometry"), dict) else {}
    surface = item.get("surface_geometry") if isinstance(item.get("surface_geometry"), dict) else {}
    consistency = item.get("sphere_consistency") if isinstance(item.get("sphere_consistency"), dict) else {}
    damage = item.get("damage_metrics") if isinstance(item.get("damage_metrics"), dict) else {}
    radial_cv = float(footprint.get("radial_cv") or 0.0)
    sphere_rmse = float(surface.get("sphere_fit_rmse_mm") or 0.0)
    radial_h_rmse = float(consistency.get("radial_height_rmse_mm") or 0.0)
    completeness = float(consistency.get("surface_completeness_ratio") or 0.0)
    flat_ratio = float(damage.get("flat_region_ratio") or 0.0)
    discontinuity = float(damage.get("surface_discontinuity_score") or 0.0)
    # Eccentricity gates relaxed: cv2.fitEllipse on noisy anisotropic contours
    # routinely produces 0.4-0.5 even for visibly round balls. The bbox/3D
    # sphericity signals are the trustworthy ones for the ball verdict; we keep
    # ecc only as a soft guard against severe elongation.
    if (
        8.0 <= max_h <= 95.0
        and sph3d >= 0.8
        and ecc < 0.65
        and flat > 0.15
        and rough < 8.0
        and radial_cv < 0.14
        and sphere_rmse < 8.0
        and radial_h_rmse < 8.0
        and completeness > 0.55
    ):
        return "bola_buena", "Bola buena", "ball", 0.82
    if ecc >= 0.85 or flat < -0.2 or rough > 12.0 or flat_ratio > 0.65 or discontinuity > 1.25:
        return "bola_deformada", "Scrap de Bola - Bola deformada", "non_ball", 0.78
    if max_h < 6.0 or p95_h < 4.0 or vol < 4000.0:
        return "planchuela", "Chatarra - Planchuelas", "non_ball", 0.72
    return "bola_con_chip", "Scrap de Bola - Bola con chip", "non_ball", 0.66


def _labels_for_superclass(superclass: str, *, confidence: float | None = None) -> tuple[str, str, str, float]:
    conf = float(confidence if confidence is not None else 0.66)
    if superclass == "BALL_GOOD":
        return "bola_buena", "Bola buena", "ball", conf
    if superclass == "SCRAP_METAL":
        return "chatarra", "Chatarra - Chatarra", "non_ball", conf
    return "bola_con_chip", "Scrap de Bola - Bola con chip", "non_ball", conf


def _metric(item: dict[str, Any], key: str) -> float | None:
    if key == "max_height_mm":
        return _safe_float(((item.get("height_above_belt_mm") or {}).get("max_height_mm")) if isinstance(item.get("height_above_belt_mm"), dict) else None)
    if key == "p99_height_mm":
        return _safe_float(((item.get("height_above_belt_mm") or {}).get("p99_height_mm")) if isinstance(item.get("height_above_belt_mm"), dict) else None)
    if key == "p95_height_mm":
        return _safe_float(((item.get("height_above_belt_mm") or {}).get("p95_height_mm")) if isinstance(item.get("height_above_belt_mm"), dict) else None)
    if key == "mean_height_mm":
        return _safe_float(((item.get("height_above_belt_mm") or {}).get("mean_height_mm")) if isinstance(item.get("height_above_belt_mm"), dict) else None)
    if key in {"radial_cv", "circularity", "circularity_score"}:
        group = item.get("footprint_geometry") if isinstance(item.get("footprint_geometry"), dict) else {}
        return _safe_float(group.get(key))
    if key in {"sphere_fit_rmse_mm", "sphere_fit_max_error_mm", "sphere_vs_ellipsoid_gain", "deformation_score"}:
        group = item.get("surface_geometry") if isinstance(item.get("surface_geometry"), dict) else {}
        return _safe_float(group.get(key))
    if key in {"radial_height_rmse_mm", "radial_height_max_error_mm", "surface_completeness_ratio", "observed_volume_ratio", "volume_deficit_ratio"}:
        group = item.get("sphere_consistency") if isinstance(item.get("sphere_consistency"), dict) else {}
        return _safe_float(group.get(key))
    if key in {
        "surface_sphere_fit_rmse_norm",
        "surface_sphere_fit_residual_p95_norm",
        "surface_sphere_fit_residual_mad_norm",
        "surface_sphere_radius_error_norm",
        "surface_sphere_center_depth_ratio",
        "surface_visible_cap_fraction",
        "surface_volume_fill_ratio",
        "surface_sphere_vs_ellipsoid_gain",
        "surface_sphere_fit_confidence",
    }:
        group = item.get("sphere_consistency") if isinstance(item.get("sphere_consistency"), dict) else {}
        return _safe_float(group.get(key))
    if key in {"surface_roughness_mean", "surface_roughness_std", "surface_roughness_score", "flat_region_ratio", "surface_discontinuity_score", "high_curvature_ratio"}:
        group = item.get("damage_metrics") if isinstance(item.get("damage_metrics"), dict) else {}
        return _safe_float(group.get(key))
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
    if key in ("max_height_mm", "mean_height_mm", "p95_height_mm", "p99_height_mm"):
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
    # dim_z is sourced from dimensions_mm[2] (already P99-derived). When that
    # field is missing we prefer p99_height_mm over max_height_mm so the
    # fallback also stays robust to noise peaks.
    if isinstance(dims, (list, tuple)) and len(dims) >= 3:
        dim_z = _safe_float(dims[2])
    else:
        heights = item.get("height_above_belt_mm") if isinstance(item.get("height_above_belt_mm"), dict) else None
        dim_z = _safe_float((heights or {}).get("p99_height_mm")) if heights else None
        if dim_z is None and heights is not None:
            dim_z = _safe_float(heights.get("max_height_mm"))
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
    footprint_group = item.get("footprint_geometry") if isinstance(item.get("footprint_geometry"), dict) else {}
    surface_group = item.get("surface_geometry") if isinstance(item.get("surface_geometry"), dict) else {}
    consistency_group = item.get("sphere_consistency") if isinstance(item.get("sphere_consistency"), dict) else {}
    damage_group = item.get("damage_metrics") if isinstance(item.get("damage_metrics"), dict) else {}
    radial_cv = _safe_float(footprint_group.get("radial_cv"))
    sphere_fit_rmse = _safe_float(surface_group.get("sphere_fit_rmse_mm"))
    sphere_rmse_norm = _safe_float(consistency_group.get("surface_sphere_fit_rmse_norm"))
    sphere_p95_norm = _safe_float(consistency_group.get("surface_sphere_fit_residual_p95_norm"))
    sphere_mad_norm = _safe_float(consistency_group.get("surface_sphere_fit_residual_mad_norm"))
    sphere_radius_error_norm = _safe_float(consistency_group.get("surface_sphere_radius_error_norm"))
    visible_cap_fraction = _safe_float(consistency_group.get("surface_visible_cap_fraction"))
    volume_fill_ratio = _safe_float(consistency_group.get("surface_volume_fill_ratio"))
    ellipsoid_gain = _safe_float(consistency_group.get("surface_sphere_vs_ellipsoid_gain") or surface_group.get("sphere_vs_ellipsoid_gain"))
    sphere_fit_confidence = _safe_float(consistency_group.get("surface_sphere_fit_confidence"))
    radial_height_rmse = _safe_float(consistency_group.get("radial_height_rmse_mm"))
    volume_deficit = _safe_float(consistency_group.get("volume_deficit_ratio"))
    flat_region_ratio = _safe_float(damage_group.get("flat_region_ratio"))
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
                description="Projected eccentricity should stay low for near-round objects (tolerant band to absorb cv2.fitEllipse noise on anisotropic contours).",
                metric_key="feature_eccentricity",
                value=ecc,
                comparator="<",
                expected={"max": 0.65},
                passed=(ecc is not None and ecc < 0.65),
                severity="warning",
                contribution="positive" if (ecc is not None and ecc < 0.65) else "negative",
                message="Eccentricity within tolerant ball band." if (ecc is not None and ecc < 0.65) else "Eccentricity high enough to suggest deformation/non-ball.",
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
            and ecc is not None and ecc < 0.65
        )
    if radial_cv is not None:
        rules.append(
            _rule(
                rule_id="footprint.radial_uniformity",
                label="Radial boundary uniformity",
                description="Contour radial CV should remain low for coherent spheres.",
                metric_key="radial_cv",
                value=radial_cv,
                comparator="<=",
                expected={"max": 0.14},
                passed=radial_cv <= 0.14,
                severity="warning",
                contribution="positive" if radial_cv <= 0.14 else "negative",
                message="Boundary radial uniformity is coherent." if radial_cv <= 0.14 else "Boundary irregularity suggests chips/fracture.",
            )
        )
    if sphere_fit_rmse is not None:
        rules.append(
            _rule(
                rule_id="surface.sphere_fit_rmse",
                label="Raw sphere-fit RMSE",
                description="Visible-surface sphere-fit RMSE in millimeters. Scale-dependent and visible-surface-only; prefer normalized sphere-consistency features for classification.",
                metric_key="sphere_fit_rmse_mm",
                value=sphere_fit_rmse,
                comparator="<=",
                expected={"max": 8.0},
                passed=sphere_fit_rmse <= 8.0,
                severity="info",
                contribution="neutral",
                message="Raw sphere-fit RMSE recorded; use normalized residuals and plausibility checks for ballness.",
            )
        )
    if sphere_rmse_norm is not None:
        rules.append(
            _rule(
                rule_id="consistency.sphere_rmse_norm",
                label="Normalized sphere-fit RMSE",
                description="Scale-normalized visible-surface sphere residual.",
                metric_key="surface_sphere_fit_rmse_norm",
                value=sphere_rmse_norm,
                comparator="<=",
                expected={"max": 0.15},
                passed=sphere_rmse_norm <= 0.15,
                severity="warning",
                contribution="positive" if sphere_rmse_norm <= 0.15 else "negative",
                message="Low normalized sphere RMSE supports ball-like surface." if sphere_rmse_norm <= 0.15 else "Elevated normalized sphere RMSE argues against a clean spherical cap.",
            )
        )
    if sphere_radius_error_norm is not None:
        rules.append(
            _rule(
                rule_id="consistency.sphere_radius_error",
                label="Sphere radius plausibility",
                description="Absolute mismatch between fitted sphere diameter and selected/reference diameter, normalized by diameter.",
                metric_key="surface_sphere_radius_error_norm",
                value=sphere_radius_error_norm,
                comparator="<=",
                expected={"max": 0.25},
                passed=sphere_radius_error_norm <= 0.25,
                severity="warning",
                contribution="positive" if sphere_radius_error_norm <= 0.25 else "negative",
                message="Fitted sphere radius is physically plausible." if sphere_radius_error_norm <= 0.25 else "High radius error argues against a complete ball geometry.",
            )
        )
    if volume_fill_ratio is not None:
        rules.append(
            _rule(
                rule_id="consistency.volume_fill_ratio",
                label="Volume fill ratio",
                description="Measured volume proxy divided by expected sphere or spherical-cap volume.",
                metric_key="surface_volume_fill_ratio",
                value=volume_fill_ratio,
                comparator=">=",
                expected={"min": 0.45},
                passed=volume_fill_ratio >= 0.45,
                severity="warning",
                contribution="positive" if volume_fill_ratio >= 0.45 else "negative",
                message="Volume fill supports a ball-like mass." if volume_fill_ratio >= 0.45 else "Low volume fill ratio suggests fragment, truncation, or scrap.",
            )
        )
    if ellipsoid_gain is not None:
        rules.append(
            _rule(
                rule_id="consistency.sphere_vs_ellipsoid_gain",
                label="Sphere vs ellipsoid gain",
                description="Relative RMSE improvement when fitting an ellipsoid instead of a sphere.",
                metric_key="surface_sphere_vs_ellipsoid_gain",
                value=ellipsoid_gain,
                comparator="<=",
                expected={"max": 0.35},
                passed=ellipsoid_gain <= 0.35,
                severity="warning",
                contribution="positive" if ellipsoid_gain <= 0.35 else "negative",
                message="Ellipsoid does not materially improve the fit." if ellipsoid_gain <= 0.35 else "High ellipsoid gain suggests deformation or non-spherical shape.",
            )
        )
    if sphere_fit_confidence is not None:
        rules.append(
            _rule(
                rule_id="consistency.sphere_fit_confidence",
                label="Sphere fit confidence",
                description="Heuristic confidence score combining fit coverage, normalized residuals, radius plausibility, and segmentation quality.",
                metric_key="surface_sphere_fit_confidence",
                value=sphere_fit_confidence,
                comparator=">=",
                expected={"min": 0.55},
                passed=sphere_fit_confidence >= 0.55,
                severity="info",
                contribution="positive" if sphere_fit_confidence >= 0.55 else "negative",
                message="Sphere-fit diagnostics are trustworthy enough for sphere-based reasoning." if sphere_fit_confidence >= 0.55 else "Low sphere-fit confidence should reduce trust in sphere-based decisions.",
            )
        )
    if sphere_p95_norm is not None:
        rules.append(
            _rule(
                rule_id="consistency.sphere_residual_p95_norm",
                label="Sphere residual P95 / diameter",
                description="Robust high-error sphere residual normalized by reference diameter.",
                metric_key="surface_sphere_fit_residual_p95_norm",
                value=sphere_p95_norm,
                comparator="<=",
                expected={"max": 0.20},
                passed=sphere_p95_norm <= 0.20,
                severity="info",
                contribution="positive" if sphere_p95_norm <= 0.20 else "negative",
                message="Tail sphere residuals remain modest." if sphere_p95_norm <= 0.20 else "Large tail residuals indicate local non-spherical structure.",
            )
        )
    if sphere_mad_norm is not None:
        rules.append(
            _rule(
                rule_id="consistency.sphere_residual_mad_norm",
                label="Sphere residual MAD / diameter",
                description="Robust residual dispersion normalized by reference diameter.",
                metric_key="surface_sphere_fit_residual_mad_norm",
                value=sphere_mad_norm,
                comparator="<=",
                expected={"max": 0.12},
                passed=sphere_mad_norm <= 0.12,
                severity="info",
                contribution="positive" if sphere_mad_norm <= 0.12 else "negative",
                message="Robust residual dispersion supports a spherical cap." if sphere_mad_norm <= 0.12 else "Dispersed sphere residuals suggest irregular surface geometry.",
            )
        )
    if visible_cap_fraction is not None:
        rules.append(
            _rule(
                rule_id="consistency.visible_cap_fraction",
                label="Visible spherical cap fraction",
                description="P95 height relative to fitted sphere diameter; distinguishes shallow caps from fuller balls.",
                metric_key="surface_visible_cap_fraction",
                value=visible_cap_fraction,
                comparator=">=",
                expected={"min": 0.35},
                passed=visible_cap_fraction >= 0.35,
                severity="info",
                contribution="positive" if visible_cap_fraction >= 0.35 else "negative",
                message="Visible cap coverage supports ball-like geometry." if visible_cap_fraction >= 0.35 else "Shallow visible cap suggests fragment or truncated object.",
            )
        )
    if radial_height_rmse is not None:
        rules.append(
            _rule(
                rule_id="consistency.radial_height_rmse",
                label="Radial height coherence",
                description="Observed cap profile should match spherical-cap model.",
                metric_key="radial_height_rmse_mm",
                value=radial_height_rmse,
                comparator="<=",
                expected={"max": 8.0},
                passed=radial_height_rmse <= 8.0,
                severity="warning",
                contribution="positive" if radial_height_rmse <= 8.0 else "negative",
                message="Height profile is sphere-consistent." if radial_height_rmse <= 8.0 else "Height profile mismatch suggests truncation/flattening.",
            )
        )
    if flat_region_ratio is not None:
        rules.append(
            _rule(
                rule_id="damage.flat_region",
                label="Flat region ratio",
                description="Large planar patches indicate impact flats or fractured truncation.",
                metric_key="flat_region_ratio",
                value=flat_region_ratio,
                comparator="<=",
                expected={"max": 0.65},
                passed=flat_region_ratio <= 0.65,
                severity="warning",
                contribution="neutral" if flat_region_ratio <= 0.65 else "negative",
                message="Flat-region evidence is limited." if flat_region_ratio <= 0.65 else "Flat-region coverage is high and supports damage.",
            )
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
        "radial_cv": radial_cv,
        "sphere_fit_rmse_mm": sphere_fit_rmse,
        "radial_height_rmse_mm": radial_height_rmse,
        "volume_deficit_ratio": volume_deficit,
        "flat_region_ratio": flat_region_ratio,
        "footprint_geometry": footprint_group,
        "surface_geometry": surface_group,
        "sphere_consistency": consistency_group,
        "damage_metrics": damage_group,
        **diameter,
    }
    group_summaries = {
        "footprint_geometry": {
            "pass": bool(radial_cv is None or radial_cv <= 0.14),
            "primary_metric": "radial_cv",
            "value": radial_cv,
        },
        "surface_geometry": {
            "pass": bool(sphere_fit_rmse is None or sphere_fit_rmse <= 8.0),
            "primary_metric": "sphere_fit_rmse_mm",
            "value": sphere_fit_rmse,
        },
        "sphere_consistency": {
            "pass": bool(radial_height_rmse is None or radial_height_rmse <= 8.0),
            "primary_metric": "radial_height_rmse_mm",
            "value": radial_height_rmse,
        },
        "damage_metrics": {
            "pass": bool(flat_region_ratio is None or flat_region_ratio <= 0.65),
            "primary_metric": "flat_region_ratio",
            "value": flat_region_ratio,
        },
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
        "semantic_group_summaries": group_summaries,
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
                "p99_height_mm": corrected_heights.get("p99_height_mm") if isinstance(corrected_heights, dict) else None,
                "p95_height_mm": corrected_heights.get("p95_height_mm") if isinstance(corrected_heights, dict) else None,
                "mean_height_mm": corrected_heights.get("mean_height_mm") if isinstance(corrected_heights, dict) else None,
            },
            [],
            sz,
            "measurement_object",
        ),
        (
            "p99_height_mm",
            corrected_heights.get("p99_height_mm") if isinstance(corrected_heights, dict) else None,
            "p99_height_mm_corrected",
            "percentile(height_above_belt_mm, 99) * scale_z  // canonical dim_z source; robust to noise peaks",
            [
                _input("p99_height_mm_raw", raw_heights.get("p99_height_mm")),
                _input("scale_z", sz),
            ],
            {
                "max_height_mm": corrected_heights.get("max_height_mm") if isinstance(corrected_heights, dict) else None,
                "p95_height_mm": corrected_heights.get("p95_height_mm") if isinstance(corrected_heights, dict) else None,
                "dim_z_mm": dim_z,
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
        if rv is None and metric_key == "p99_height_mm" and isinstance(raw_heights, dict):
            rv = raw_heights.get("p99_height_mm")
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
