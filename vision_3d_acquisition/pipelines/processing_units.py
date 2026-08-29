from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Mapping


PROCESSING_UNIT_REGISTRY_VERSION = "2026.07.full-25d.v1"
SUPPORTED_PROCESSING_UNIT_PARAMETER_TYPES = {"string", "number", "integer", "boolean", "roi"}
SUPPORTED_ARTIFACT_RENDERERS = {"image", "mask", "json", "table", "overlay", "histogram"}
SUPPORTED_VIEW_RENDERER_TYPES = {"image", "mask", "json", "table", "overlay", "histogram"}
SUPPORTED_ARTIFACT_DIFF_TYPES = {"binary_mask", "image_metadata", "json_numeric"}
DIFFABLE_BINARY_MASK_ARTIFACT_IDS = {
    "valid_mask",
    "plane_fit_roi_mask",
    "background_candidate_mask",
    "background_seed_mask",
    "expanded_plane_mask",
    "final_plane_inlier_mask",
    "plane_inlier_mask",
    "low_gradient_mask",
    "flat_candidate_mask",
    "background_selected_plateau_mask",
    "rejected_raised_plateau_mask",
    "selected_blob_cluster_pre_refine_mask",
    "selected_blob_cluster_refined_mask",
    "support_removed_by_candidate_refinement",
    "belt_stripes_mask",
    "support_removed_by_stripe_filter",
    "reference_model_support_mask",
    "reference_suppression_mask",
    "foreground_before_plane_suppression",
    "above_threshold_mask",
    "plane_suppressed_mask",
    "cleaned_object_mask",
    "final_object_mask",
    "normalized_height_threshold_mask",
    "selected_reference_support_mask",
    "reference_surface_selected_mask",
    "below_reference_mask",
    "belt_base_mask",
}
MANUAL_PARAMETER_METADATA: dict[str, dict[str, Any]] = {
    "belt_stripe_filter_threshold_mode": {
        "label": "Stripe threshold mode",
        "type": "string",
        "default": "otsu",
        "enum": ["otsu", "k_mad", "fixed"],
        "group": "Belt stripe suppression",
        "help": "Controls whether stripe altitude thresholding uses Otsu, MAD-derived, or fixed cutoffs.",
        "advanced": True,
    },
}


@dataclass(frozen=True)
class ProcessingUnitInput:
    id: str
    label: str
    artifact_id: str | None = None
    kind: str | None = None
    coordinate_space: str | None = None
    units: str | None = None
    required: bool = True
    description: str | None = None


@dataclass(frozen=True)
class ProcessingUnitOutput:
    id: str
    label: str
    artifact_id: str | None = None
    kind: str | None = None
    coordinate_space: str | None = None
    units: str | None = None
    description: str | None = None


@dataclass(frozen=True)
class ProcessingUnitParameter:
    id: str
    label: str
    type: str
    default: Any = None
    min: float | None = None
    max: float | None = None
    options: list[str] | None = None
    advanced: bool = False
    affects: list[str] = field(default_factory=list)
    description: str | None = None
    tuning_hint: str | None = None
    active_when: dict[str, Any] | None = None
    group: str | None = None
    unit: str | None = None
    step: float | None = None
    nullable: bool = False


@dataclass(frozen=True)
class ProcessingUnitArtifact:
    id: str
    label: str
    artifact_id: str
    kind: str
    role: str
    renderer: str
    description: str | None = None
    produced_by: str | None = None
    source_artifact_id: str | None = None
    coordinate_space: str | None = None
    aliases: list[str] = field(default_factory=list)
    diffable: bool = False
    diff_type: str | None = None
    # For role="diagnostic" artifacts specifically: names of other artifacts whose computation
    # actually reuses this one's underlying data (not just "this file gets read again" -- most
    # diagnostics are rendered *from* already-final values and reused nowhere). Empty means this
    # diagnostic is a genuine dead end, verified by reading its write site, not assumed from role.
    feeds_into: list[str] = field(default_factory=list)
    # Additive validation contract.  Existing consumers may ignore it safely.
    validationSpec: dict[str, Any] | None = None


@dataclass(frozen=True)
class ProcessingUnitView:
    id: str
    label: str
    renderer_type: str
    artifact_ids: list[str]
    description: str | None = None
    empty_state: str | None = None


@dataclass(frozen=True)
class ProcessingUnitParameterGroup:
    id: str
    label: str
    param_keys: list[str] = field(default_factory=list)
    description: str | None = None
    affects: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ProcessingUnitTuningHint:
    condition: str
    actions: list[str]


@dataclass(frozen=True)
class ProcessingUnitDefinition:
    id: str
    label: str
    kind: str
    parent_id: str | None
    stage_id: str
    category: str
    order: int
    description: str
    inputs: list[ProcessingUnitInput]
    outputs: list[ProcessingUnitOutput]
    artifacts: list[ProcessingUnitArtifact]
    parameters: list[ProcessingUnitParameter]
    diagnostics: list[str]
    views: list[ProcessingUnitView]
    default_view: str | None
    help_markdown: str | None
    enabled_by_default: bool = True
    strategy_keys: list[str] = field(default_factory=list)
    supports_preview: bool = False
    supports_partial_rerun: bool = False
    controlled_by: list[str] = field(default_factory=list)
    parameter_groups: list[ProcessingUnitParameterGroup] = field(default_factory=list)
    downstream_effects: list[str] = field(default_factory=list)
    tuning_hints: list[ProcessingUnitTuningHint] = field(default_factory=list)


def _schema_field(fields: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = fields.get(key)
    if not isinstance(value, Mapping):
        fallback = MANUAL_PARAMETER_METADATA.get(key)
        if isinstance(fallback, Mapping):
            return fallback
        if key.startswith("belt_stripe_filter_"):
            if key.endswith("_enabled"):
                return {"label": key.replace("_", " "), "type": "boolean", "default": True, "group": "Belt stripe suppression", "advanced": True}
            if key.endswith("_mode") or key.endswith("_scope"):
                return {"label": key.replace("_", " "), "type": "string", "default": "", "group": "Belt stripe suppression", "advanced": True}
            return {"label": key.replace("_", " "), "type": "number", "default": 0.0, "group": "Belt stripe suppression", "advanced": True}
        raise KeyError(f"Unknown detect_belt_plane parameter: {key}")
    return value


def _artifact_diff_type(artifact_id: str, kind: str) -> str | None:
    if artifact_id in DIFFABLE_BINARY_MASK_ARTIFACT_IDS:
        return "binary_mask"
    if kind == "mask":
        return "binary_mask"
    return None


def _parameter(
    fields: Mapping[str, Any],
    key: str,
    *,
    affects: list[str],
    description: str | None = None,
    tuning_hint: str | None = None,
) -> ProcessingUnitParameter:
    meta = _schema_field(fields, key)
    raw_options = meta.get("enum")
    options = [str(item) for item in raw_options] if isinstance(raw_options, list) else None
    minimum = meta.get("minimum")
    maximum = meta.get("maximum")
    step = meta.get("step")
    return ProcessingUnitParameter(
        id=key,
        label=str(meta.get("label") or key.replace("_", " ")),
        type=str(meta.get("type") or "string"),
        default=meta.get("default"),
        min=float(minimum) if isinstance(minimum, (int, float)) else None,
        max=float(maximum) if isinstance(maximum, (int, float)) else None,
        options=options,
        advanced=bool(meta.get("advanced") is True),
        affects=affects,
        description=description or (str(meta.get("help")) if meta.get("help") else None),
        tuning_hint=tuning_hint,
        active_when=dict(meta.get("visible_when")) if isinstance(meta.get("visible_when"), Mapping) else None,
        group=str(meta.get("group")) if meta.get("group") else None,
        unit=str(meta.get("unit")) if meta.get("unit") else None,
        step=float(step) if isinstance(step, (int, float)) else None,
        nullable=bool(meta.get("nullable") is True),
    )


def _artifact(
    unit_id: str,
    artifact_id: str,
    label: str,
    *,
    kind: str = "image",
    role: str = "diagnostic",
    renderer: str = "image",
    description: str | None = None,
    source_artifact_id: str | None = None,
    aliases: list[str] | None = None,
    diffable: bool | None = None,
    diff_type: str | None = None,
    feeds_into: list[str] | None = None,
    validation_spec: dict[str, Any] | None = None,
) -> ProcessingUnitArtifact:
    resolved_diff_type = diff_type if diff_type is not None else _artifact_diff_type(artifact_id, kind)
    return ProcessingUnitArtifact(
        id=artifact_id,
        label=label,
        artifact_id=artifact_id,
        kind=kind,
        role=role,
        renderer=renderer,
        description=description,
        produced_by=unit_id,
        source_artifact_id=source_artifact_id,
        coordinate_space="heightmap_pixel",
        aliases=list(aliases or []),
        diffable=bool(diffable if diffable is not None else resolved_diff_type is not None),
        diff_type=resolved_diff_type,
        feeds_into=list(feeds_into or []),
        validationSpec=validation_spec,
    )


def _view(
    view_id: str,
    label: str,
    artifact_ids: list[str],
    *,
    renderer_type: str = "image",
    description: str | None = None,
    empty_state: str | None = None,
) -> ProcessingUnitView:
    return ProcessingUnitView(
        id=view_id,
        label=label,
        renderer_type=renderer_type,
        artifact_ids=artifact_ids,
        description=description,
        empty_state=empty_state,
    )


def detect_reference_processing_units(stage_parameter_schema: Mapping[str, Any]) -> list[ProcessingUnitDefinition]:
    fields = stage_parameter_schema.get("fields") if isinstance(stage_parameter_schema.get("fields"), Mapping) else {}

    root_id = "detect_belt_plane"
    units = [
        ProcessingUnitDefinition(
            id=root_id,
            label="Detect reference surface",
            kind="stage",
            parent_id=None,
            stage_id="detect_belt_plane",
            category="reference_detection",
            order=0,
            description="Full detect-reference strategy path for the 25D mining steel ball pipeline.",
            inputs=[
                ProcessingUnitInput(id="heightmap", label="Heightmap frame", kind="heightmap", description="Raw calibrated heightmap input."),
                ProcessingUnitInput(id="valid_mask", label="Valid sensor mask", artifact_id="valid_mask", kind="mask", description="Sensor-valid pixels available for downstream support selection."),
            ],
            outputs=[
                ProcessingUnitOutput(id="selected_support", label="Selected reference support", artifact_id="selected_reference_support_mask", kind="mask"),
                ProcessingUnitOutput(id="reference_model", label="Reference model", artifact_id="belt_plane", kind="json"),
            ],
            artifacts=[
                _artifact(root_id, "selected_reference_support_mask", "Final selected support", role="final", description="Selected support after refinement and stripe suppression."),
                _artifact(root_id, "belt_plane", "Reference model", kind="json", role="final", renderer="json", description="Resolved belt/reference surface model."),
                _artifact(root_id, "plane_fit_debug", "Plane-fit debug", kind="json", role="diagnostic", renderer="json"),
            ],
            parameters=[
                _parameter(fields, "background_detection_strategy", affects=["low_gradient_mask", "reference_surface_plateaus", "blob_height_clusters", "selected_reference_support_mask"], tuning_hint="Choose the strategy that most clearly separates belt support from raised objects."),
                _parameter(fields, "reference_surface_model", affects=["belt_plane", "plane_inlier_mask", "final_plane_inlier_mask"], tuning_hint="Use constant-Z only when the plane fit is unstable or residuals show systematic tilt."),
                _parameter(fields, "reference_suppression_mask_policy", affects=["reference_suppression_mask"], tuning_hint="Prefer the smallest mask that still removes the belt cleanly downstream."),
            ],
            diagnostics=["background_selection_debug", "selected_surface_debug", "plane_fit_debug", "support_loss_waterfall"],
            views=[
                _view("raw_heightmap", "Raw heightmap", ["raw_heightmap_preview"]),
                _view("valid_mask", "Valid mask", ["valid_mask"]),
                _view("low_gradient_mask", "Low-gradient mask", ["low_gradient_mask"]),
                _view("surface_candidates", "Surface candidates", ["reference_surface_candidates", "reference_surface_plateaus", "background_selected_plateau_mask"], renderer_type="json"),
                _view("selected_support_lineage", "Selected support lineage", ["selected_blob_cluster_pre_refine_mask", "selected_blob_cluster_refined_mask", "support_removed_by_candidate_refinement", "support_removed_by_stripe_filter"], renderer_type="table"),
                _view("selected_surface", "Selected surface", ["selected_reference_support_mask", "reference_surface_selected_mask"]),
                _view("plane_inliers", "Plane inliers", ["final_plane_inlier_mask", "plane_inlier_mask"]),
                _view("residual_heatmap", "Residual heatmap", ["plane_residual_heatmap", "belt_plane_residuals"]),
                _view("json", "JSON", ["belt_plane", "plane_fit_debug", "selected_surface_debug"], renderer_type="json"),
            ],
            default_view="selected_surface",
            help_markdown="Canonical stage contract for Detect reference surface. The root stage keeps the full strategy path visible while substages scope tuning and artifact inspection.",
            strategy_keys=["low_gradient_surface", "low_gradient_depth_plateaus", "low_gradient_bg_and_stripes", "low_gradient_blob_height_clusters", "nearest_percentile", "farthest_percentile", "automatic"],
            supports_preview=False,
            supports_partial_rerun=False,
        ),
        ProcessingUnitDefinition(
            id="detect_belt_plane.input",
            label="Input / raw reference data",
            kind="substage",
            parent_id=root_id,
            stage_id="detect_belt_plane",
            category="input",
            order=10,
            description="Raw heightmap and validity context entering reference detection.",
            inputs=[ProcessingUnitInput(id="heightmap", label="Heightmap frame", kind="heightmap")],
            outputs=[ProcessingUnitOutput(id="raw_preview", label="Raw heightmap preview", artifact_id="raw_heightmap_preview", kind="image")],
            artifacts=[
                _artifact("detect_belt_plane.input", "raw_heightmap_preview", "Raw heightmap preview", role="input"),
                _artifact("detect_belt_plane.input", "valid_mask", "Valid ROI mask", role="input"),
            ],
            parameters=[
                _parameter(fields, "random_seed", affects=["belt_plane"]),
                _parameter(fields, "plot_depth_plot_max_render_samples", affects=["flat_candidate_depth_plot", "background_plateau_plot"]),
                _parameter(fields, "plot_y_robust_percentile", affects=["belt_altitude_histogram", "flat_candidate_histogram"]),
            ],
            diagnostics=[],
            views=[_view("raw_heightmap", "Raw heightmap", ["raw_heightmap_preview"]), _view("valid_mask", "Valid mask", ["valid_mask"])],
            default_view="raw_heightmap",
            help_markdown="This unit is informational. It exposes the raw heightmap preview and valid-pixel footprint used by all downstream support-selection steps.",
            controlled_by=[],
        ),
        ProcessingUnitDefinition(
            id="detect_belt_plane.roi",
            label="ROI / reference search domain",
            kind="substage",
            parent_id=root_id,
            stage_id="detect_belt_plane",
            category="roi",
            order=20,
            description="Fit-region restriction that decides where reference support may be found.",
            inputs=[ProcessingUnitInput(id="valid_mask", label="Valid mask", artifact_id="valid_mask", kind="mask")],
            outputs=[ProcessingUnitOutput(id="fit_roi_mask", label="Plane-fit ROI mask", artifact_id="plane_fit_roi_mask", kind="mask")],
            artifacts=[_artifact("detect_belt_plane.roi", "plane_fit_roi_mask", "Plane-fit ROI mask", role="intermediate")],
            parameters=[
                ProcessingUnitParameter(
                    id="reference_surface_region",
                    label="Reference ROI",
                    type="roi",
                    affects=["plane_fit_roi_mask", "flat_candidate_mask", "selected_reference_support_mask"],
                    description="Choose the reference-search domain used for support selection and plane fitting.",
                    tuning_hint="Tighten the ROI when the reference search is being distracted by tall objects or irrelevant background structure.",
                ),
                _parameter(fields, "support_selection_method", affects=["plane_fit_roi_mask"]),
                _parameter(fields, "reference_surface_selection_mode", affects=["plane_fit_roi_mask", "reference_surface_candidates"]),
                _parameter(fields, "reference_surface_region_mode", affects=["plane_fit_roi_mask"]),
                _parameter(fields, "reference_surface_min_area_ratio", affects=["reference_surface_candidates"]),
                _parameter(fields, "reference_surface_max_z_std_mm", affects=["reference_surface_candidates"]),
                _parameter(fields, "reference_surface_border_bonus", affects=["reference_surface_candidates"]),
                _parameter(fields, "reference_surface_depth_preference_weight", affects=["reference_surface_candidates"]),
                _parameter(fields, "reference_surface_constancy_weight", affects=["reference_surface_candidates"]),
                _parameter(fields, "reference_surface_area_weight", affects=["reference_surface_candidates"]),
                _parameter(fields, "reference_surface_max_plane_residual_p95_mm", affects=["reference_surface_candidates"]),
                _parameter(fields, "background_selection_mode", affects=["plane_fit_roi_mask"], tuning_hint="Only applies to the nearest_percentile/farthest_percentile/automatic strategies."),
                _parameter(fields, "background_percentile", affects=["plane_fit_roi_mask"]),
                _parameter(fields, "background_candidate_morphology", affects=["plane_fit_roi_mask"]),
                _parameter(fields, "background_candidate_open_kernel", affects=["plane_fit_roi_mask"]),
                _parameter(fields, "background_candidate_min_component_area", affects=["plane_fit_roi_mask"]),
                _parameter(fields, "background_must_touch_roi_border", affects=["plane_fit_roi_mask"]),
                _parameter(fields, "low_gradient_surface_support_z_mad_multiplier", affects=["selected_reference_support_mask"], tuning_hint="Legacy low_gradient_surface strategy only."),
                _parameter(fields, "low_gradient_surface_support_z_floor_mm", affects=["selected_reference_support_mask"]),
                _parameter(fields, "low_gradient_surface_support_z_mad_floor_mm", affects=["selected_reference_support_mask"]),
                _parameter(fields, "low_gradient_surface_ridge_percentile", affects=["selected_reference_support_mask"]),
            ],
            diagnostics=[],
            views=[_view("valid_mask", "ROI", ["plane_fit_roi_mask", "valid_mask"])],
            default_view="valid_mask",
            help_markdown="ROI selection is currently inherited from upstream region settings and metadata. This contract makes the search domain explicit even before ROI controls are fully contract-driven.",
            controlled_by=["detect_belt_plane.input"],
            parameter_groups=[
                ProcessingUnitParameterGroup(
                    id="roi",
                    label="Reference ROI",
                    param_keys=["reference_surface_region"],
                    affects=["Valid mask", "Low-gradient mask", "Selected surface"],
                ),
                ProcessingUnitParameterGroup(
                    id="surface_scoring",
                    label="Reference surface scoring",
                    description="Weights used to rank candidate flat surfaces against each other before one is promoted to reference support.",
                    param_keys=[
                        "reference_surface_selection_mode",
                        "reference_surface_region_mode",
                        "reference_surface_min_area_ratio",
                        "reference_surface_max_z_std_mm",
                        "reference_surface_border_bonus",
                        "reference_surface_depth_preference_weight",
                        "reference_surface_constancy_weight",
                        "reference_surface_area_weight",
                        "reference_surface_max_plane_residual_p95_mm",
                    ],
                    affects=["ROI"],
                ),
                ProcessingUnitParameterGroup(
                    id="percentile_strategy",
                    label="Percentile candidate selection",
                    description="Only used by the nearest_percentile / farthest_percentile / automatic strategies.",
                    param_keys=[
                        "support_selection_method",
                        "background_selection_mode",
                        "background_percentile",
                        "background_candidate_morphology",
                        "background_candidate_open_kernel",
                        "background_candidate_min_component_area",
                        "background_must_touch_roi_border",
                    ],
                    affects=["ROI"],
                ),
                ProcessingUnitParameterGroup(
                    id="legacy_surface_strategy",
                    label="Legacy surface strategy (low_gradient_surface)",
                    param_keys=[
                        "low_gradient_surface_support_z_mad_multiplier",
                        "low_gradient_surface_support_z_floor_mm",
                        "low_gradient_surface_support_z_mad_floor_mm",
                        "low_gradient_surface_ridge_percentile",
                    ],
                    affects=["ROI"],
                ),
            ],
            downstream_effects=["Reference support eligibility", "Plane fit support", "Downstream normalization"],
        ),
        ProcessingUnitDefinition(
            id="detect_belt_plane.height_gate",
            label="Height gate",
            kind="substage",
            parent_id=root_id,
            stage_id="detect_belt_plane",
            category="candidate_pruning",
            order=30,
            description="Early height-based candidate pruning before plateau or blob reasoning.",
            inputs=[ProcessingUnitInput(id="roi_mask", label="Plane-fit ROI mask", artifact_id="plane_fit_roi_mask", kind="mask")],
            outputs=[ProcessingUnitOutput(id="flat_candidates", label="Flat candidates", artifact_id="flat_candidate_mask", kind="mask")],
            artifacts=[
                _artifact("detect_belt_plane.height_gate", "height_gate_mask", "Height gate mask", role="intermediate"),
                _artifact("detect_belt_plane.height_gate", "flat_candidate_mask", "Height-gated candidates", role="intermediate"),
                _artifact("detect_belt_plane.height_gate", "reference_surface_candidates", "Reference-surface candidates", kind="json", role="diagnostic", renderer="json"),
            ],
            parameters=[
                _parameter(fields, "reference_surface_height_gate_enabled", affects=["height_gate_mask", "flat_candidate_mask"], tuning_hint="Disable to skip height-based pruning entirely and pass the full ROI through to gradient/plateau analysis."),
                _parameter(fields, "reference_surface_height_gate_margin_mm", affects=["height_gate_mask", "flat_candidate_mask"]),
                _parameter(fields, "reference_surface_height_gate_gap_floor_mm", affects=["height_gate_mask"]),
                _parameter(fields, "reference_surface_height_gate_gap_ratio", affects=["height_gate_mask"]),
                _parameter(fields, "reference_surface_height_gate_min_coverage_ratio", affects=["flat_candidate_mask"]),
                _parameter(fields, "reference_surface_height_gate_max_coverage_ratio", affects=["flat_candidate_mask"]),
            ],
            diagnostics=["reference_surface_candidates"],
            views=[_view("surface_candidates", "Height-gated candidates", ["height_gate_mask", "flat_candidate_mask", "reference_surface_candidates"])],
            default_view="surface_candidates",
            help_markdown="Height gating trims obviously raised regions before lower-gradient plateau or blob analysis takes over.",
            controlled_by=["detect_belt_plane.roi"],
            parameter_groups=[
                ProcessingUnitParameterGroup(
                    id="height_gate",
                    label="Height gate",
                    param_keys=[
                        "reference_surface_height_gate_enabled",
                        "reference_surface_height_gate_margin_mm",
                        "reference_surface_height_gate_gap_floor_mm",
                        "reference_surface_height_gate_gap_ratio",
                        "reference_surface_height_gate_min_coverage_ratio",
                        "reference_surface_height_gate_max_coverage_ratio",
                    ],
                    affects=["Height-gated candidates"],
                ),
            ],
            downstream_effects=["Candidate support pruning", "Height-aware connectivity", "Object-vs-belt separation"],
            tuning_hints=[
                ProcessingUnitTuningHint(condition="Real belt/reference support is missing from later strategy candidates", actions=["Raise Height gate margin", "Lower Height gate min coverage ratio", "Disable Enable height gate to confirm whether gating is the cause"]),
            ],
        ),
        ProcessingUnitDefinition(
            id="detect_belt_plane.depth_gradient",
            label="Depth gradient / low-gradient mask",
            kind="substage",
            parent_id=root_id,
            stage_id="detect_belt_plane",
            category="gradient_filter",
            order=40,
            description="Gradient magnitude and low-gradient support filtering.",
            inputs=[ProcessingUnitInput(id="flat_candidates", label="Flat candidates", artifact_id="flat_candidate_mask", kind="mask", required=False)],
            outputs=[ProcessingUnitOutput(id="low_gradient_mask", label="Low-gradient mask", artifact_id="low_gradient_mask", kind="mask")],
            artifacts=[
                _artifact("detect_belt_plane.depth_gradient", "depth_gradient_magnitude", "Depth gradient magnitude", role="diagnostic", feeds_into=["low_gradient_mask"]),
                _artifact("detect_belt_plane.depth_gradient", "low_gradient_mask", "Low-gradient mask", role="intermediate"),
                _artifact("detect_belt_plane.depth_gradient", "gradient_debug", "Gradient debug", kind="json", role="diagnostic", renderer="json", aliases=["gradient_debug.json"]),
            ],
            parameters=[
                _parameter(fields, "gradient_threshold_percentile", affects=["low_gradient_mask", "depth_gradient_magnitude"], tuning_hint="Lower this if too much belt disappears; raise it if raised objects leak into the support mask."),
                _parameter(fields, "low_gradient_open_kernel", affects=["low_gradient_mask"], tuning_hint="Use opening to drop speckle before component formation."),
                _parameter(fields, "low_gradient_close_kernel", affects=["low_gradient_mask"], tuning_hint="Use closing to reconnect small low-gradient gaps without merging clear objects."),
                _parameter(fields, "low_gradient_min_component_area", affects=["low_gradient_mask", "reference_surface_plateaus"], tuning_hint="Raise this when tiny low-gradient islands distract plateau or blob selection."),
                _parameter(fields, "gradient_smoothing_kernel", affects=["depth_gradient_magnitude", "low_gradient_mask"]),
                _parameter(fields, "gradient_method", affects=["depth_gradient_magnitude"]),
                _parameter(fields, "gradient_threshold_mode", affects=["low_gradient_mask"]),
                _parameter(fields, "gradient_threshold_value", affects=["low_gradient_mask"], tuning_hint="Used only when Gradient threshold mode is fixed instead of percentile."),
                _parameter(fields, "invalid_neighbor_policy", affects=["depth_gradient_magnitude", "low_gradient_mask"]),
                _parameter(fields, "low_gradient_morphology_enabled", affects=["low_gradient_mask"], tuning_hint="Disable to skip the open/close cleanup pass entirely and see the raw thresholded gradient mask."),
                _parameter(fields, "low_gradient_fill_holes", affects=["low_gradient_mask"]),
            ],
            diagnostics=["gradient_debug"],
            views=[
                _view("depth_gradient", "Depth gradient", ["depth_gradient_magnitude"]),
                _view("low_gradient_mask", "Low-gradient mask", ["low_gradient_mask"]),
            ],
            default_view="low_gradient_mask",
            help_markdown="This unit seeds every later strategy branch. If the low-gradient mask is wrong, all later support-selection decisions drift.",
            controlled_by=["detect_belt_plane.height_gate"],
            parameter_groups=[
                ProcessingUnitParameterGroup(
                    id="gradient",
                    label="Gradient filter",
                    param_keys=["gradient_threshold_percentile", "gradient_threshold_mode", "gradient_threshold_value", "low_gradient_open_kernel", "low_gradient_close_kernel"],
                    affects=["Low-gradient mask", "Blob components", "Selected surface"],
                ),
                ProcessingUnitParameterGroup(
                    id="gradient_computation",
                    label="Gradient computation",
                    description="Controls how the raw gradient magnitude is computed before thresholding.",
                    param_keys=["gradient_smoothing_kernel", "gradient_method", "invalid_neighbor_policy"],
                    affects=["Depth gradient"],
                ),
                ProcessingUnitParameterGroup(
                    id="low_gradient_cleanup",
                    label="Low-gradient mask cleanup",
                    param_keys=["low_gradient_morphology_enabled", "low_gradient_min_component_area", "low_gradient_fill_holes"],
                    affects=["Low-gradient mask"],
                ),
            ],
            downstream_effects=["Low-gradient candidates", "Surface-candidate quality", "Support selection stability"],
        ),
        ProcessingUnitDefinition(
            id="detect_belt_plane.depth_plateaus",
            label="Depth plateaus / plateau selection",
            kind="substage",
            parent_id=root_id,
            stage_id="detect_belt_plane",
            category="plateau_selection",
            order=50,
            description="Histogram and plateau ranking over the low-gradient support candidates.",
            inputs=[ProcessingUnitInput(id="low_gradient_mask", label="Low-gradient mask", artifact_id="low_gradient_mask", kind="mask")],
            outputs=[ProcessingUnitOutput(id="selected_plateau", label="Selected plateau mask", artifact_id="background_selected_plateau_mask", kind="mask")],
            artifacts=[
                _artifact("detect_belt_plane.depth_plateaus", "reference_surface_plateaus", "Reference-surface plateaus", kind="json", role="diagnostic", renderer="json", feeds_into=["background_selected_plateau_mask"]),
                _artifact("detect_belt_plane.depth_plateaus", "reference_surface_candidates", "Reference-surface candidates", kind="json", role="diagnostic", renderer="json"),
                _artifact("detect_belt_plane.depth_plateaus", "flat_candidate_histogram", "Flat-candidate histogram", kind="json", role="diagnostic", renderer="json", feeds_into=["background_selected_plateau_mask"]),
                _artifact("detect_belt_plane.depth_plateaus", "background_plateau_plot", "Background plateau plot", role="diagnostic", feeds_into=["background_selected_plateau_mask"]),
                _artifact("detect_belt_plane.depth_plateaus", "flat_candidate_depth_plot", "Flat-candidate depth plot", role="diagnostic", feeds_into=["background_selected_plateau_mask"]),
                _artifact("detect_belt_plane.depth_plateaus", "background_selected_plateau_mask", "Selected plateau mask", role="intermediate"),
                _artifact("detect_belt_plane.depth_plateaus", "rejected_raised_plateau_mask", "Rejected raised plateau mask", role="diagnostic"),
                _artifact("detect_belt_plane.depth_plateaus", "background_selection_debug", "Background selection debug", kind="json", role="diagnostic", renderer="json"),
            ],
            parameters=[
                _parameter(fields, "low_gradient_plateau_selection_mode", affects=["reference_surface_plateaus", "background_selected_plateau_mask"], tuning_hint="Switch modes when the dominant belt plateau is not the lowest or largest cluster in the depth distribution."),
                _parameter(fields, "low_gradient_plateau_use_hessian_filter", affects=["reference_surface_plateaus", "flat_candidate_histogram"], tuning_hint="Disable if legitimate flat surface is being suppressed as a false ridge."),
                _parameter(fields, "low_gradient_plateau_hessian_percentile", affects=["flat_candidate_histogram"]),
                _parameter(fields, "low_gradient_plateau_hist_bins", affects=["flat_candidate_histogram", "background_plateau_plot"]),
                _parameter(fields, "low_gradient_plateau_min_fraction", affects=["reference_surface_plateaus"], tuning_hint="Raise this to ignore minor histogram peaks as noise; lower it to detect smaller/thinner plateaus."),
                _parameter(fields, "low_gradient_plateau_min_pixels", affects=["reference_surface_plateaus"]),
                _parameter(fields, "low_gradient_plateau_smoothing_sigma_bins", affects=["flat_candidate_histogram", "background_plateau_plot"], tuning_hint="Widen smoothing to recover a broader plateau on a slightly tilted belt; narrow it if nearby peaks are getting merged."),
                _parameter(fields, "low_gradient_plateau_peak_drop_ratio", affects=["reference_surface_plateaus"]),
                _parameter(fields, "low_gradient_plateau_select_min_area_fraction", affects=["background_selected_plateau_mask", "selection_reason"], tuning_hint="Raise this to require the dominant plateau to be a larger share of flat candidates before it's preferred over the lowest-z fallback."),
                _parameter(fields, "low_gradient_plateau_robust_band_mad_k", affects=["background_selected_plateau_mask"]),
                _parameter(fields, "low_gradient_plateau_detection_min_count_floor", affects=["reference_surface_plateaus"]),
                _parameter(fields, "low_gradient_plateau_detection_min_count_fraction", affects=["reference_surface_plateaus"]),
            ],
            diagnostics=["flat_candidate_histogram", "background_selection_debug"],
            views=[
                _view("surface_candidates", "Plateau candidates", ["reference_surface_plateaus", "reference_surface_candidates", "background_selected_plateau_mask"], renderer_type="json"),
                _view("plateau_plot", "Plateau plot", ["background_plateau_plot"]),
                _view("filtered_depth_plot", "Filtered depth plot", ["flat_candidate_depth_plot"]),
            ],
            default_view="surface_candidates",
            help_markdown="Used mainly by the depth-plateau strategy branch to explain which low-gradient depth band was promoted to support. Also used, with the same parameters, by the low_gradient_bg_and_stripes hybrid strategy to find the belt-background plateau before component classification runs.",
            strategy_keys=["low_gradient_depth_plateaus", "low_gradient_bg_and_stripes"],
            controlled_by=["detect_belt_plane.depth_gradient"],
            parameter_groups=[
                ProcessingUnitParameterGroup(
                    id="plateaus",
                    label="Plateau selection",
                    param_keys=["low_gradient_plateau_selection_mode", "low_gradient_plateau_select_min_area_fraction", "low_gradient_plateau_min_fraction", "low_gradient_plateau_min_pixels"],
                    affects=["Surface candidates", "Selected surface"],
                ),
                ProcessingUnitParameterGroup(
                    id="plateau_histogram",
                    label="Histogram / peak detection",
                    description="Controls the z-histogram used to find candidate plateaus before selection ranks them.",
                    param_keys=[
                        "low_gradient_plateau_use_hessian_filter",
                        "low_gradient_plateau_hessian_percentile",
                        "low_gradient_plateau_hist_bins",
                        "low_gradient_plateau_smoothing_sigma_bins",
                        "low_gradient_plateau_peak_drop_ratio",
                        "low_gradient_plateau_robust_band_mad_k",
                        "low_gradient_plateau_detection_min_count_floor",
                        "low_gradient_plateau_detection_min_count_fraction",
                    ],
                    affects=["Plateau plot", "Filtered depth plot", "Surface candidates"],
                ),
            ],
            downstream_effects=["Surface-candidate ranking", "Fallback behavior", "Support coverage", "Belt-background plateau used by the bg_and_stripes component classifier"],
            tuning_hints=[
                ProcessingUnitTuningHint(condition="Wrong plateau selected (chevron ribs win over the real belt)", actions=["Switch Plateau selection mode", "Raise Dominant plateau min area fraction", "Confirm Use Hessian ridge filter is enabled"]),
                ProcessingUnitTuningHint(condition="No plateau detected / falls back to flat candidates", actions=["Lower Plateau min fraction and Plateau min pixels", "Widen Plateau histogram smoothing"]),
            ],
        ),
        ProcessingUnitDefinition(
            id="detect_belt_plane.blob_components",
            label="Blob components",
            kind="substage",
            parent_id=root_id,
            stage_id="detect_belt_plane",
            category="component_formation",
            order=60,
            description="Connected low-gradient support components before splitting and clustering.",
            inputs=[ProcessingUnitInput(id="low_gradient_mask", label="Low-gradient mask", artifact_id="low_gradient_mask", kind="mask")],
            outputs=[ProcessingUnitOutput(id="components", label="Blob components", artifact_id="low_gradient_blob_id_mask", kind="mask")],
            artifacts=[
                _artifact("detect_belt_plane.blob_components", "low_gradient_components_overlay", "Low-gradient components overlay", role="diagnostic"),
                _artifact("detect_belt_plane.blob_components", "low_gradient_components", "Low-gradient component summary", kind="json", role="diagnostic", renderer="json"),
                _artifact("detect_belt_plane.blob_components", "low_gradient_blob_components_overlay", "Blob components overlay", role="intermediate"),
                _artifact("detect_belt_plane.blob_components", "low_gradient_blob_id_mask", "Blob component id mask", role="intermediate"),
                _artifact("detect_belt_plane.blob_components", "height_aware_blob_components_overlay", "Height-aware components overlay", role="diagnostic", feeds_into=["low_gradient_blob_id_mask"]),
                _artifact("detect_belt_plane.blob_components", "height_aware_blob_id_mask", "Height-aware component id mask", role="diagnostic", feeds_into=["low_gradient_blob_id_mask"]),
            ],
            parameters=[
                _parameter(fields, "blob_component_mode", affects=["low_gradient_blob_components_overlay", "height_aware_blob_components_overlay"], tuning_hint="Height-aware mode is safer when low-gradient objects bridge into belt support."),
                _parameter(fields, "blob_connectivity", affects=["low_gradient_blob_id_mask"], tuning_hint="Lower connectivity only if diagonal links are creating unrealistic support bridges."),
                _parameter(fields, "blob_neighbor_z_tolerance_mm", affects=["height_aware_blob_id_mask"], tuning_hint="Lower this when belt and object pixels wrongly join into the same support component."),
                _parameter(fields, "blob_component_z_tolerance_mm", affects=["height_aware_blob_id_mask"], tuning_hint="Use a tighter Z tolerance when support should stay nearly coplanar."),
                _parameter(fields, "blob_component_tolerance_mode", affects=["height_aware_blob_id_mask"], tuning_hint="Adaptive tolerance is safer on uneven belts; fixed tolerance is easier to reason about."),
                _parameter(fields, "blob_component_mad_k", affects=["height_aware_blob_id_mask"], tuning_hint="Lower this to split noisy elevated regions away from the belt sooner."),
                _parameter(fields, "blob_component_mad_floor_mm", affects=["height_aware_blob_id_mask"], tuning_hint="Raise the floor only when MAD collapses and fragments valid support."),
                _parameter(fields, "blob_component_allow_gradual_slope", affects=["height_aware_blob_components_overlay"], tuning_hint="Disable this if gradual ramps are incorrectly pulling objects into the support set."),
                _parameter(fields, "blob_component_max_local_slope_mm_per_px", affects=["height_aware_blob_components_overlay"], tuning_hint="Lower it to reject steep local ramps from support connectivity."),
                _parameter(fields, "blob_component_min_area_px", affects=["low_gradient_blob_id_mask"], tuning_hint="Raise it when tiny support islands dominate downstream cluster scoring."),
                _parameter(fields, "blob_component_use_smoothed_z", affects=["height_aware_blob_components_overlay"], tuning_hint="Disable smoothing only when it is hiding important support boundaries."),
            ],
            diagnostics=["low_gradient_components", "height_aware_blob_components_overlay"],
            views=[
                _view("blob_components", "Blob components", ["low_gradient_blob_components_overlay", "low_gradient_blob_id_mask"]),
                _view("height_aware_blob_components", "Height-aware components", ["height_aware_blob_components_overlay", "height_aware_blob_id_mask"]),
            ],
            default_view="blob_components",
            help_markdown="This unit explains how the low-gradient mask turns into connected candidate supports for the blob-cluster branch.",
            strategy_keys=["low_gradient_blob_height_clusters", "low_gradient_bg_and_stripes"],
            controlled_by=["detect_belt_plane.depth_gradient"],
            parameter_groups=[
                ProcessingUnitParameterGroup(
                    id="component_formation",
                    label="Component formation",
                    param_keys=[
                        "blob_component_mode",
                        "blob_connectivity",
                        "blob_neighbor_z_tolerance_mm",
                        "blob_component_use_smoothed_z",
                        "blob_component_max_local_slope_mm_per_px",
                        "blob_component_min_area_px",
                    ],
                    affects=["Blob components", "Blob clusters", "Selected support"],
                ),
            ],
            downstream_effects=["Component/fragment formation", "Height clustering inputs", "Final support mask"],
            tuning_hints=[
                ProcessingUnitTuningHint(condition="Object and belt are merged", actions=["Lower Neighbor Z tolerance", "Disable Use smoothed Z", "Lower Max local slope"]),
                ProcessingUnitTuningHint(condition="Belt is fragmented", actions=["Raise Neighbor Z tolerance", "Raise Max local slope", "Lower Component min area"]),
            ],
        ),
        ProcessingUnitDefinition(
            id="detect_belt_plane.blob_splitting",
            label="Blob splitting",
            kind="substage",
            parent_id=root_id,
            stage_id="detect_belt_plane",
            category="component_splitting",
            order=70,
            description="Split mixed-height support components into height-consistent fragments.",
            inputs=[ProcessingUnitInput(id="components", label="Blob components", artifact_id="low_gradient_blob_id_mask", kind="mask")],
            outputs=[ProcessingUnitOutput(id="fragments", label="Split fragments", artifact_id="height_split_blob_fragments_mask", kind="mask")],
            artifacts=[
                _artifact("detect_belt_plane.blob_splitting", "height_border_strength", "Height-border strength", role="diagnostic", feeds_into=["height_border_fragments_mask"]),
                _artifact("detect_belt_plane.blob_splitting", "height_border_cut_mask", "Height-border cut mask", role="diagnostic", feeds_into=["height_border_fragments_mask"]),
                _artifact("detect_belt_plane.blob_splitting", "height_border_split_debug", "Height-border split debug", kind="json", role="diagnostic", renderer="json"),
                _artifact("detect_belt_plane.blob_splitting", "height_border_fragments_overlay", "Height-border fragments overlay", role="intermediate"),
                _artifact("detect_belt_plane.blob_splitting", "height_border_fragments_mask", "Height-border fragments mask", role="intermediate"),
                _artifact("detect_belt_plane.blob_splitting", "height_split_blob_fragments_overlay", "Height-split blob overlay", role="intermediate"),
                _artifact("detect_belt_plane.blob_splitting", "height_split_blob_fragments_mask", "Height-split blob mask", role="intermediate"),
                _artifact("detect_belt_plane.blob_splitting", "height_split_debug", "Height-split debug", kind="json", role="diagnostic", renderer="json"),
            ],
            parameters=[
                _parameter(fields, "blob_split_by_height_enabled", affects=["height_border_fragments_mask", "height_split_blob_fragments_mask"], tuning_hint="Disable only when splitting destroys a valid continuous belt support."),
                _parameter(fields, "blob_split_method", affects=["height_split_blob_fragments_mask"], tuning_hint="Use height borders first when object-vs-belt boundaries are spatially visible."),
                _parameter(fields, "blob_split_min_height_range_mm", affects=["height_split_blob_fragments_mask"], tuning_hint="Lower this if subtle but real mixed-height support never gets split."),
                _parameter(fields, "blob_split_min_pixels", affects=["height_split_blob_fragments_mask"], tuning_hint="Lower this if valid narrow fragments are ignored before splitting."),
                _parameter(fields, "blob_split_gap_mm", affects=["height_split_blob_fragments_mask"], tuning_hint="Raise the gap only when histogram splitting under-separates clear height modes."),
                _parameter(fields, "blob_split_mode", affects=["height_split_blob_fragments_mask"]),
                _parameter(fields, "blob_split_hist_bins", affects=["height_split_debug"]),
                _parameter(fields, "blob_split_min_band_fraction", affects=["height_split_debug"]),
                _parameter(fields, "blob_split_min_band_pixels", affects=["height_split_debug"]),
                _parameter(fields, "blob_split_merge_gap_mm", affects=["height_split_blob_fragments_mask"]),
                _parameter(fields, "blob_split_refine_morphology", affects=["height_split_blob_fragments_mask"]),
                _parameter(fields, "blob_split_open_kernel", affects=["height_split_blob_fragments_mask"]),
                _parameter(fields, "blob_split_close_kernel", affects=["height_split_blob_fragments_mask"]),
                _parameter(fields, "blob_split_height_border_enabled", affects=["height_border_strength", "height_border_cut_mask"]),
                _parameter(fields, "blob_split_height_border_mode", affects=["height_border_strength"]),
                _parameter(fields, "blob_split_height_border_smoothing_kernel", affects=["height_border_strength"]),
                _parameter(fields, "blob_split_height_border_threshold_mode", affects=["height_border_cut_mask"]),
                _parameter(fields, "blob_split_height_border_percentile", affects=["height_border_cut_mask"]),
                _parameter(fields, "blob_split_height_border_min_delta_mm", affects=["height_border_cut_mask"]),
                _parameter(fields, "blob_split_height_border_dilate_kernel", affects=["height_border_cut_mask"]),
                _parameter(fields, "blob_split_height_border_close_kernel", affects=["height_border_cut_mask"]),
                _parameter(fields, "blob_split_height_border_min_length_px", affects=["height_border_cut_mask"]),
                _parameter(fields, "blob_split_min_fragment_area_px", affects=["height_split_blob_fragments_mask"], tuning_hint="Raise this only after checking that valid belt slivers are not being discarded."),
            ],
            diagnostics=["height_border_split_debug", "height_split_debug"],
            views=[
                _view("height_borders", "Height borders", ["height_border_strength", "height_border_cut_mask"]),
                _view("height_border_fragments", "Height-border fragments", ["height_border_fragments_overlay", "height_border_fragments_mask"]),
                _view("height_split_blobs", "Height-split blobs", ["height_split_blob_fragments_overlay", "height_split_blob_fragments_mask"]),
            ],
            default_view="height_split_blobs",
            help_markdown="This unit explains where the blob strategy cuts mixed-height support into fragments that can later be merged or clustered.",
            strategy_keys=["low_gradient_blob_height_clusters"],
            controlled_by=["detect_belt_plane.blob_components"],
            parameter_groups=[
                ProcessingUnitParameterGroup(
                    id="blob_splitting",
                    label="Split strategy",
                    param_keys=["blob_split_by_height_enabled", "blob_split_method", "blob_split_min_height_range_mm", "blob_split_min_pixels"],
                    affects=["Eligible blobs", "Height-border fragments", "Height-split blobs"],
                ),
                ProcessingUnitParameterGroup(
                    id="height_border_priority",
                    label="Height-border splitting",
                    description="Primary controls for the default height-border split path.",
                    param_keys=[
                        "blob_split_height_border_enabled",
                        "blob_split_height_border_threshold_mode",
                        "blob_split_height_border_percentile",
                        "blob_split_height_border_min_delta_mm",
                        "blob_split_height_border_smoothing_kernel",
                        "blob_split_height_border_min_length_px",
                        "blob_split_min_fragment_area_px",
                    ],
                    affects=["Height-border strength", "Height-border fragments", "Final split fragments"],
                ),
                ProcessingUnitParameterGroup(
                    id="histogram_splitting",
                    label="Histogram splitting",
                    param_keys=[
                        "blob_split_gap_mm",
                        "blob_split_mode",
                        "blob_split_hist_bins",
                        "blob_split_min_band_fraction",
                        "blob_split_min_band_pixels",
                        "blob_split_merge_gap_mm",
                        "blob_split_refine_morphology",
                        "blob_split_open_kernel",
                        "blob_split_close_kernel",
                    ],
                    affects=["Histogram-derived fragments", "Height-split blobs", "Cluster inputs"],
                ),
            ],
            downstream_effects=["Fragment IDs", "Clustering inputs", "Support-selection stability"],
            tuning_hints=[
                ProcessingUnitTuningHint(condition="Thin belt fragments disappear", actions=["Lower Split min fragment area", "Lower Split min pixels", "Lower Split min height range if valid belt blobs are skipped"]),
                ProcessingUnitTuningHint(condition="Over-splitting creates too many fragments", actions=["Raise Split min fragment area", "Raise Height-border min edge length", "Raise Split merge gap only for histogram-based splitting"]),
                ProcessingUnitTuningHint(condition="Belt background is fragmented", actions=["Raise Height-border min delta", "Raise Height-border percentile", "Raise Height-border smoothing kernel", "Raise Height-border min edge length"]),
                ProcessingUnitTuningHint(condition="Object and belt are still merged", actions=["Lower Height-border min delta", "Lower Height-border percentile", "Reduce Height-border smoothing if boundaries are washed out"]),
            ],
        ),
        ProcessingUnitDefinition(
            id="detect_belt_plane.fragment_merge",
            label="Fragment merge",
            kind="substage",
            parent_id=root_id,
            stage_id="detect_belt_plane",
            category="fragment_merge",
            order=80,
            description="Reconnect weakly split fragments that still look like one belt support region.",
            inputs=[ProcessingUnitInput(id="split_fragments", label="Split fragments", artifact_id="height_split_blob_fragments_mask", kind="mask")],
            outputs=[ProcessingUnitOutput(id="merged_fragments", label="Merged fragments", artifact_id="height_split_blob_fragments_mask", kind="mask")],
            artifacts=[_artifact("detect_belt_plane.fragment_merge", "fragment_merge_debug", "Fragment-merge debug", kind="json", role="diagnostic", renderer="json")],
            parameters=[
                _parameter(fields, "blob_split_merge_weak_boundaries", affects=["height_split_blob_fragments_mask"], tuning_hint="Disable only when fragment rescue repeatedly merges obvious objects into belt support."),
                _parameter(fields, "blob_split_merge_max_median_z_gap_mm", affects=["height_split_blob_fragments_mask"]),
                _parameter(fields, "blob_split_merge_max_boundary_strength_mm", affects=["height_split_blob_fragments_mask"]),
                _parameter(fields, "blob_split_merge_background_like_only", affects=["height_split_blob_fragments_mask"]),
            ],
            diagnostics=["fragment_merge_debug"],
            views=[_view("height_split_blobs", "Merged fragments", ["height_split_blob_fragments_overlay", "height_split_blob_fragments_mask"])],
            default_view="height_split_blobs",
            help_markdown="Weak-boundary merge is the rescue step after over-aggressive splitting.",
            strategy_keys=["low_gradient_blob_height_clusters"],
            controlled_by=["detect_belt_plane.blob_splitting"],
            parameter_groups=[
                ProcessingUnitParameterGroup(
                    id="fragment_merge",
                    label="Weak-boundary rescue",
                    param_keys=["blob_split_merge_weak_boundaries", "blob_split_merge_max_median_z_gap_mm", "blob_split_merge_max_boundary_strength_mm", "blob_split_merge_background_like_only"],
                    affects=["Merged fragments", "Rejected merge candidates", "Cluster inputs"],
                ),
            ],
            downstream_effects=["Fragment rescue", "Cluster formation", "Selected support connectivity"],
            tuning_hints=[
                ProcessingUnitTuningHint(condition="Belt fragments should reconnect", actions=["Enable Merge weak boundaries", "Raise Merge max median Z gap slightly", "Raise Merge max boundary strength slightly"]),
                ProcessingUnitTuningHint(condition="Object and belt merge incorrectly", actions=["Lower Merge max median Z gap", "Lower Merge max boundary strength", "Require Merge background-like only"]),
            ],
        ),
        ProcessingUnitDefinition(
            id="detect_belt_plane.candidate_support_refinement",
            label="Candidate support refinement",
            kind="substage",
            parent_id=root_id,
            stage_id="detect_belt_plane",
            category="support_refinement",
            order=90,
            description="Trim the selected logical support cluster into the fit-support mask used downstream.",
            inputs=[ProcessingUnitInput(id="selected_cluster", label="Selected blob cluster", artifact_id="selected_blob_cluster_pre_refine_mask", kind="mask")],
            outputs=[ProcessingUnitOutput(id="refined_support", label="Refined support", artifact_id="selected_blob_cluster_refined_mask", kind="mask")],
            artifacts=[
                _artifact("detect_belt_plane.candidate_support_refinement", "selected_blob_cluster_mask", "Selected blob cluster", role="intermediate"),
                _artifact("detect_belt_plane.candidate_support_refinement", "selected_blob_cluster_pre_refine_mask", "Selected cluster pre-refine", role="intermediate"),
                _artifact("detect_belt_plane.candidate_support_refinement", "selected_blob_cluster_refined_mask", "Selected cluster refined", role="final"),
                _artifact("detect_belt_plane.candidate_support_refinement", "support_removed_by_candidate_refinement", "Removed by candidate refinement", role="diagnostic"),
                _artifact("detect_belt_plane.candidate_support_refinement", "candidate_support_refinement_debug", "Candidate refinement debug", kind="json", role="diagnostic", renderer="json"),
            ],
            parameters=[
                _parameter(fields, "blob_cluster_refine_by_mad", affects=["selected_blob_cluster_refined_mask", "support_removed_by_candidate_refinement"], tuning_hint="Disable only if MAD trimming removes clearly valid belt support."),
                _parameter(fields, "blob_cluster_refine_mad_k", affects=["selected_blob_cluster_refined_mask"], tuning_hint="Raise this when the selected support shrinks too aggressively."),
                _parameter(fields, "blob_cluster_refine_floor_mm", affects=["selected_blob_cluster_refined_mask"]),
                _parameter(fields, "blob_cluster_refine_keep_border_support", affects=["selected_blob_cluster_refined_mask"], tuning_hint="Keep border support when the belt mostly lives on the outer frame and trimming erodes it away."),
            ],
            diagnostics=["candidate_support_refinement_debug"],
            views=[
                _view("selected_blob_cluster", "Selected blob cluster", ["selected_blob_cluster_mask"]),
                _view("selected_cluster_pre_refine", "Selected cluster pre-refine", ["selected_blob_cluster_pre_refine_mask"]),
                _view("selected_cluster_refined", "Selected cluster refined", ["selected_blob_cluster_refined_mask"]),
                _view("removed_by_refinement", "Removed by refinement", ["support_removed_by_candidate_refinement"]),
            ],
            default_view="selected_cluster_refined",
            help_markdown="This is the highest-pain explanatory step for why support changed: it preserves the before/after/removed lineage explicitly.",
            strategy_keys=["low_gradient_blob_height_clusters"],
            controlled_by=["detect_belt_plane.blob_components", "detect_belt_plane.blob_splitting", "detect_belt_plane.fragment_merge"],
            parameter_groups=[
                ProcessingUnitParameterGroup(
                    id="candidate_support_refinement",
                    label="Candidate refinement",
                    param_keys=["blob_cluster_refine_by_mad", "blob_cluster_refine_mad_k", "blob_cluster_refine_floor_mm", "blob_cluster_refine_keep_border_support"],
                    affects=["Selected cluster pre-refine", "Selected cluster refined", "Removed by refinement"],
                ),
            ],
            downstream_effects=["Fit support size", "Stripe suppression inputs", "Reference model support"],
            tuning_hints=[
                ProcessingUnitTuningHint(condition="Selected support shrinks too much", actions=["Loosen MAD refinement", "Raise MAD floor", "Enable Keep border support"]),
                ProcessingUnitTuningHint(condition="Object-like pixels remain inside support", actions=["Tighten MAD refinement", "Lower MAD floor", "Disable Keep border support if edges are leaking in"]),
            ],
        ),
        ProcessingUnitDefinition(
            id="detect_belt_plane.stripe_filter",
            label="Stripe filter",
            kind="substage",
            parent_id=root_id,
            stage_id="detect_belt_plane",
            category="stripe_suppression",
            order=100,
            description="Remove stripe-like belt texture from the refined support mask and keep the belt-base pixels that should survive into model fitting.",
            inputs=[ProcessingUnitInput(id="refined_support", label="Refined support", artifact_id="selected_blob_cluster_refined_mask", kind="mask", required=False)],
            outputs=[
                ProcessingUnitOutput(id="stripe_filtered_support", label="Stripe-filtered support", artifact_id="stripe_filtered_reference_support_mask", kind="mask"),
                ProcessingUnitOutput(id="selected_support", label="Selected support", artifact_id="selected_reference_support_mask", kind="mask"),
            ],
            artifacts=[
                _artifact("detect_belt_plane.stripe_filter", "selected_blob_cluster_refined_mask", "Pre-stripe support", role="input"),
                _artifact("detect_belt_plane.stripe_filter", "belt_bg_mask", "Belt background mask", role="intermediate"),
                _artifact("detect_belt_plane.stripe_filter", "belt_base_mask", "Belt base mask", role="intermediate"),
                _artifact("detect_belt_plane.stripe_filter", "stripe_filtered_reference_support_mask", "Stripe-filtered support", role="final"),
                _artifact("detect_belt_plane.stripe_filter", "selected_reference_support_mask", "Selected support after stripe suppression", role="final"),
                _artifact("detect_belt_plane.stripe_filter", "belt_stripes_mask", "Belt stripes mask", role="diagnostic", feeds_into=["surface_suppression_mask", "final_object_mask"]),
                _artifact("detect_belt_plane.stripe_filter", "unknown_low_gradient_mask", "Unknown low-gradient mask", role="diagnostic"),
                _artifact("detect_belt_plane.stripe_filter", "surface_suppression_mask", "Surface suppression mask", role="diagnostic", feeds_into=["object_search_domain_mask", "final_object_mask"]),
                _artifact("detect_belt_plane.stripe_filter", "object_search_domain_mask", "Object search domain mask", role="diagnostic"),
                _artifact("detect_belt_plane.stripe_filter", "belt_stripe_candidates_overlay", "Stripe candidate components", role="diagnostic"),
                _artifact("detect_belt_plane.stripe_filter", "belt_stripe_components", "Stripe component summary", kind="json", role="diagnostic", renderer="json"),
                _artifact("detect_belt_plane.stripe_filter", "belt_stripes_tophat_mask", "Stripe top-hat mask", role="diagnostic", feeds_into=["belt_stripes_mask"]),
                _artifact("detect_belt_plane.stripe_filter", "belt_stripes_shape_mask", "Stripe shape mask", role="diagnostic"),
                _artifact("detect_belt_plane.stripe_filter", "belt_above_belt_mask", "Above-belt mask", role="diagnostic", feeds_into=["belt_stripes_mask"]),
                _artifact("detect_belt_plane.stripe_filter", "belt_wide_object_mask", "Wide-object mask", role="diagnostic"),
                _artifact("detect_belt_plane.stripe_filter", "belt_baseline_local_min", "Belt baseline local minimum", role="diagnostic", feeds_into=["belt_altitude_local_min"]),
                _artifact("detect_belt_plane.stripe_filter", "belt_altitude_local_min", "Belt altitude local minimum", role="diagnostic", feeds_into=["belt_stripes_mask"]),
                _artifact("detect_belt_plane.stripe_filter", "belt_altitude_histogram", "Belt altitude histogram", role="diagnostic"),
                _artifact("detect_belt_plane.stripe_filter", "belt_stripe_filter_debug", "Stripe-filter debug", kind="json", role="diagnostic", renderer="json"),
                _artifact("detect_belt_plane.stripe_filter", "support_removed_by_stripe_filter", "Removed by stripe filter", role="diagnostic", aliases=["removed_by_stripe_filter"]),
            ],
            parameters=[
                _parameter(fields, "belt_stripe_filter_enabled", affects=["belt_stripes_mask", "support_removed_by_stripe_filter", "stripe_filtered_reference_support_mask"], tuning_hint="Disable only when stripe suppression is removing real flat support instead of texture."),
                _parameter(fields, "belt_stripe_filter_scope", affects=["belt_stripes_mask", "selected_blob_cluster_refined_mask"]),
                _parameter(fields, "belt_stripe_filter_window_mm", affects=["belt_stripes_tophat_mask"]),
                _parameter(fields, "belt_stripe_filter_direction", affects=["belt_stripes_tophat_mask", "belt_altitude_histogram"], tuning_hint="Use direction when stripes are consistently raised or recessed and auto-selection picks the wrong polarity."),
                _parameter(fields, "belt_stripe_filter_threshold_mode", affects=["belt_stripes_mask", "belt_altitude_histogram"], tuning_hint="Use threshold mode to control whether stripe altitude is fixed, MAD-derived, or chosen from bimodality."),
                _parameter(fields, "belt_stripe_filter_min_altitude_mm", affects=["belt_stripes_mask", "belt_altitude_histogram"]),
                _parameter(fields, "belt_stripe_filter_k_mad", affects=["belt_altitude_histogram"]),
                _parameter(fields, "belt_stripe_filter_fixed_threshold_mm", affects=["belt_altitude_histogram"]),
                _parameter(fields, "belt_stripe_filter_min_stripe_fraction", affects=["belt_stripes_mask", "stripe_filtered_reference_support_mask"], tuning_hint="Raise this when tiny stripe detections should be ignored as noise; lower it when faint stripes are being skipped."),
                _parameter(fields, "belt_stripe_filter_z_floor_enabled", affects=["belt_stripes_shape_mask", "belt_above_belt_mask"], tuning_hint="Enable the shape pass when narrow top-hat stripes miss wider raised ribs or object-adjacent texture."),
                _parameter(fields, "belt_stripe_filter_z_floor_use_upper_bound", affects=["belt_above_belt_mask"], tuning_hint="Use the upper bound when the belt plateau top is the safest reference for above-belt gating; disable it to anchor closer to central belt height."),
                _parameter(fields, "belt_stripe_filter_z_floor_margin_mm", affects=["belt_above_belt_mask"]),
                _parameter(fields, "belt_stripe_filter_max_stripe_height_mm", affects=["belt_stripes_shape_mask"]),
                _parameter(fields, "belt_stripe_filter_above_belt_close_mm", affects=["belt_above_belt_mask"]),
                _parameter(fields, "belt_stripe_filter_object_kernel_mm", affects=["belt_wide_object_mask"]),
                _parameter(fields, "belt_stripe_filter_object_kernel_shape", affects=["belt_wide_object_mask"]),
                _parameter(fields, "belt_stripe_filter_altitude_hist_bins", affects=["belt_altitude_histogram"]),
                _parameter(fields, "belt_stripe_filter_auto_bimodality_margin", affects=["belt_altitude_histogram"], tuning_hint="Raise this when auto direction flips too easily between raised and recessed interpretations."),
                _parameter(fields, "belt_stripe_filter_z_floor_upper_percentile", affects=["belt_above_belt_mask"]),
                _parameter(fields, "belt_stripe_filter_z_floor_fallback_lower_percentile", affects=["belt_above_belt_mask"]),
                _parameter(fields, "belt_stripe_filter_z_floor_fallback_upper_percentile", affects=["belt_above_belt_mask"]),
                _parameter(fields, "belt_stripe_filter_warn_removed_fraction", affects=["support_removed_by_stripe_filter", "belt_base_mask"], tuning_hint="Use this warning threshold to flag runs where stripe suppression is eroding too much valid support."),
                _parameter(fields, "stripe_rule_height_range_enabled", affects=["belt_stripes_mask", "belt_stripe_filter_debug"], tuning_hint="Disable to test the classifier without the height-range gate."),
                _parameter(fields, "stripe_height_mm", affects=["belt_stripes_mask", "belt_stripe_filter_debug"], tuning_hint="Raise or lower this to match the belt's actual stripe/rib height when real stripes are being classified as unknown instead of belt_stripe."),
                _parameter(fields, "stripe_height_tolerance_mm", affects=["belt_stripes_mask", "belt_stripe_filter_debug"], tuning_hint="Widen this when stripe height varies a lot across the belt; narrow it to be more conservative about what counts as a stripe."),
                _parameter(fields, "stripe_rule_width_range_enabled", affects=["belt_stripes_mask", "belt_stripe_filter_debug"], tuning_hint="Disable to test the classifier without the width-range gate."),
                _parameter(fields, "stripe_min_width_mm", affects=["belt_stripes_mask", "belt_stripe_filter_debug"]),
                _parameter(fields, "stripe_max_width_mm", affects=["belt_stripes_mask", "belt_stripe_filter_debug"]),
                _parameter(fields, "stripe_rule_height_width_ratio_enabled", affects=["belt_stripes_mask", "belt_stripe_filter_debug"], tuning_hint="Disable to test the classifier without the height/width-ratio gate."),
                _parameter(fields, "stripe_min_height_width_ratio", affects=["belt_stripes_mask", "belt_stripe_filter_debug"]),
                _parameter(fields, "stripe_max_height_width_ratio", affects=["belt_stripes_mask", "belt_stripe_filter_debug"]),
                _parameter(fields, "stripe_rule_width_cv_enabled", affects=["belt_stripes_mask", "belt_stripe_filter_debug"], tuning_hint="Disable to test the classifier without the width-variation gate."),
                _parameter(fields, "stripe_max_width_cv", affects=["belt_stripes_mask", "belt_stripe_filter_debug"]),
                _parameter(fields, "stripe_rule_length_width_ratio_enabled", affects=["belt_stripes_mask", "belt_stripe_filter_debug"], tuning_hint="Disable to test the classifier without the elongation gate."),
                _parameter(fields, "stripe_min_length_width_ratio", affects=["belt_stripes_mask", "belt_stripe_filter_debug"]),
                _parameter(fields, "stripe_rule_area_fraction_enabled", affects=["belt_stripes_mask", "belt_stripe_filter_debug"], tuning_hint="Disable to test the classifier without the area-fraction gate."),
                _parameter(fields, "stripe_max_area_fraction", affects=["belt_stripes_mask", "belt_stripe_filter_debug"]),
                _parameter(fields, "stripe_rule_overlap_enabled", affects=["belt_stripes_mask", "belt_stripe_filter_debug"], tuning_hint="Disable to test the belt_stripe classification without the altitude-overlap gate. Does not affect belt_bg plateau-overlap, which always applies."),
                _parameter(fields, "stripe_component_min_overlap_fraction", affects=["belt_bg_mask", "belt_stripes_mask", "belt_stripe_filter_debug"], tuning_hint="Lower this when a real belt-background component is being classified as unknown instead of belt_bg."),
            ],
            diagnostics=["belt_stripe_filter_debug", "belt_altitude_histogram"],
            views=[
                _view("pre_stripe_support", "Pre-stripe support", ["selected_blob_cluster_refined_mask"]),
                _view("belt_bg", "Belt background", ["belt_bg_mask"]),
                _view("belt_base", "Belt base", ["belt_base_mask"]),
                _view("stripe_filtered_support", "Stripe-filtered support", ["stripe_filtered_reference_support_mask", "selected_reference_support_mask"]),
                _view("belt_stripes", "Belt stripes", ["belt_stripes_mask"]),
                _view("unknown_low_gradient", "Unknown low-gradient", ["unknown_low_gradient_mask"]),
                _view("surface_suppression", "Surface suppression", ["surface_suppression_mask"]),
                _view("object_search_domain", "Object search domain", ["object_search_domain_mask"]),
                _view("belt_stripes_tophat", "Stripes - top-hat", ["belt_stripes_tophat_mask"]),
                _view("belt_stripes_shape", "Stripes - shape", ["belt_stripes_shape_mask"]),
                _view("belt_above_belt", "Above-belt mask", ["belt_above_belt_mask"]),
                _view("belt_wide_object", "Wide-object guard", ["belt_wide_object_mask"]),
                _view("belt_altitude_plot", "Stripe altitude", ["belt_altitude_histogram"]),
                _view("belt_baseline", "Local-min baseline", ["belt_baseline_local_min"]),
                _view("removed_by_stripe_filter", "Removed by stripe filter", ["support_removed_by_stripe_filter"]),
            ],
            default_view="pre_stripe_support",
            help_markdown="This unit explains the full stripe-suppression chain: the refined support that enters the filter, the stripe candidates found by altitude and shape passes, the belt-base pixels that remain, and the exact support removed before model fitting.",
            strategy_keys=["low_gradient_depth_plateaus", "low_gradient_bg_and_stripes"],
            controlled_by=["detect_belt_plane.candidate_support_refinement", "detect_belt_plane.depth_plateaus"],
            parameter_groups=[
                ProcessingUnitParameterGroup(
                    id="stripe_input",
                    label="Input / support handoff",
                    description="Controls when stripe suppression runs and how broadly it searches relative to the incoming refined support.",
                    param_keys=["belt_stripe_filter_enabled", "belt_stripe_filter_scope", "belt_stripe_filter_warn_removed_fraction"],
                    affects=["Pre-stripe support", "Removed by stripe suppression", "Stripe-filtered support"],
                ),
                ProcessingUnitParameterGroup(
                    id="stripe_altitude",
                    label="Altitude stripe detection",
                    description="Controls the local-baseline altitude pass that catches narrow raised belt texture.",
                    param_keys=[
                        "belt_stripe_filter_window_mm",
                        "belt_stripe_filter_direction",
                        "belt_stripe_filter_threshold_mode",
                        "belt_stripe_filter_min_altitude_mm",
                        "belt_stripe_filter_k_mad",
                        "belt_stripe_filter_fixed_threshold_mm",
                        "belt_stripe_filter_min_stripe_fraction",
                        "belt_stripe_filter_altitude_hist_bins",
                        "belt_stripe_filter_auto_bimodality_margin",
                    ],
                    affects=["Belt stripes", "Stripes - top-hat", "Stripe altitude", "Local-min baseline"],
                ),
                ProcessingUnitParameterGroup(
                    id="stripe_shape",
                    label="Shape / raised-structure guards",
                    description="Controls the optional shape pass that keeps wide objects and catches wider raised stripes near the belt surface.",
                    param_keys=[
                        "belt_stripe_filter_z_floor_enabled",
                        "belt_stripe_filter_z_floor_use_upper_bound",
                        "belt_stripe_filter_z_floor_margin_mm",
                        "belt_stripe_filter_z_floor_upper_percentile",
                        "belt_stripe_filter_z_floor_fallback_lower_percentile",
                        "belt_stripe_filter_z_floor_fallback_upper_percentile",
                        "belt_stripe_filter_max_stripe_height_mm",
                        "belt_stripe_filter_above_belt_close_mm",
                        "belt_stripe_filter_object_kernel_mm",
                        "belt_stripe_filter_object_kernel_shape",
                    ],
                    affects=["Stripes - shape pass", "Above belt mask", "Wide-object guard", "Belt base"],
                ),
                ProcessingUnitParameterGroup(
                    id="stripe_component_classification",
                    label="Component classification (bg_and_stripes)",
                    description="For the low_gradient_bg_and_stripes strategy, every flat connected component (blob) is classified whole: it becomes belt_bg if enough of its own height range overlaps the belt band, otherwise it becomes belt_stripe if enough of its height-over-base overlaps a band around Stripe height (target) +/- tolerance, otherwise unknown. Many blobs can become belt_bg and many can become belt_stripe -- classification is per component, not a single global pixel filter.",
                    param_keys=[
                        "stripe_component_min_overlap_fraction",
                        "stripe_rule_height_range_enabled",
                        "stripe_height_mm",
                        "stripe_height_tolerance_mm",
                        "stripe_rule_overlap_enabled",
                        "stripe_rule_width_range_enabled",
                        "stripe_min_width_mm",
                        "stripe_max_width_mm",
                        "stripe_rule_height_width_ratio_enabled",
                        "stripe_min_height_width_ratio",
                        "stripe_max_height_width_ratio",
                        "stripe_rule_width_cv_enabled",
                        "stripe_max_width_cv",
                        "stripe_rule_length_width_ratio_enabled",
                        "stripe_min_length_width_ratio",
                        "stripe_rule_area_fraction_enabled",
                        "stripe_max_area_fraction",
                    ],
                    affects=["Belt stripes", "Belt background", "Stripe-filter debug"],
                ),
                ProcessingUnitParameterGroup(
                    id="stripe_outputs",
                    label="Outputs to inspect",
                    description="These artifacts tell the stripe-suppression story from input support to kept support.",
                    affects=["Pre-stripe support", "Belt stripes", "Removed by stripe suppression", "Belt base", "Stripe-filtered support"],
                ),
            ],
            downstream_effects=["Stripe removal", "Support cleanup", "Reference-model fit support", "Plane residual quality"],
            tuning_hints=[
                ProcessingUnitTuningHint(condition="Valid belt support disappears", actions=["Lower Min altitude", "Reduce search Scope to bg_plateau when global search overreaches", "Raise warning threshold only after confirming removal is expected"]),
                ProcessingUnitTuningHint(condition="Visible belt ribs remain in support", actions=["Raise Window mm for broader local baseline", "Switch Threshold mode to MAD/auto if fixed threshold is too coarse", "Enable shape pass and tune Z-floor margin"]),
                ProcessingUnitTuningHint(condition="Objects get flagged as stripes", actions=["Raise Object kernel mm", "Lower Max stripe height", "Tighten Z-floor margin so the shape pass stays closer to the belt"]),
                ProcessingUnitTuningHint(condition="On the bg_and_stripes strategy, a real stripe is classified as unknown instead of belt_stripe", actions=["Check Stripe-filter debug's component_decisions for the failing component's reason", "Widen the height/width/ratio gate that failed", "Lower Min overlap fraction if the component barely misses the altitude-evidence overlap"]),
            ],
        ),
        ProcessingUnitDefinition(
            id="detect_belt_plane.reference_model_fit",
            label="Reference model fit",
            kind="substage",
            parent_id=root_id,
            stage_id="detect_belt_plane",
            category="model_fit",
            order=110,
            description="Fit the final reference model and emit inlier/residual diagnostics.",
            inputs=[ProcessingUnitInput(id="selected_support", label="Selected support", artifact_id="selected_reference_support_mask", kind="mask")],
            outputs=[
                ProcessingUnitOutput(id="reference_model", label="Reference model", artifact_id="belt_plane", kind="json"),
                ProcessingUnitOutput(id="plane_inliers", label="Plane inliers", artifact_id="final_plane_inlier_mask", kind="mask"),
            ],
            artifacts=[
                _artifact("detect_belt_plane.reference_model_fit", "reference_model_support_mask", "Reference-model support", role="intermediate"),
                _artifact("detect_belt_plane.reference_model_fit", "belt_plane", "Reference model", kind="json", role="final", renderer="json"),
                _artifact("detect_belt_plane.reference_model_fit", "plane_inlier_mask", "Plane inlier mask", role="diagnostic", feeds_into=["normalized_heightmap_mm", "final_object_mask"]),
                _artifact("detect_belt_plane.reference_model_fit", "expanded_plane_mask", "Expanded plane mask", role="diagnostic", feeds_into=["final_plane_inlier_mask", "final_object_mask"]),
                _artifact("detect_belt_plane.reference_model_fit", "final_plane_inlier_mask", "Final plane inlier mask", role="final"),
                _artifact("detect_belt_plane.reference_model_fit", "plane_fit_debug", "Plane-fit debug", kind="json", role="diagnostic", renderer="json"),
                _artifact("detect_belt_plane.reference_model_fit", "selected_surface_debug", "Selected-surface debug", kind="json", role="diagnostic", renderer="json"),
                _artifact("detect_belt_plane.reference_model_fit", "plane_residual_histogram", "Plane residual histogram", kind="json", role="diagnostic", renderer="json"),
                _artifact("detect_belt_plane.reference_model_fit", "plane_residual_heatmap", "Plane residual heatmap", role="diagnostic", aliases=["belt_plane_residuals"]),
            ],
            parameters=[
                _parameter(fields, "reference_surface_model", affects=["belt_plane", "final_plane_inlier_mask"]),
                _parameter(fields, "reference_suppression_mask_policy", affects=["reference_suppression_mask"]),
                _parameter(fields, "plane_fit_min_inlier_ratio", affects=["final_plane_inlier_mask"], tuning_hint="Lower this only after confirming the selected support is genuinely belt-like but sparse."),
                _parameter(fields, "plane_fit_residual_threshold_mm", affects=["plane_inlier_mask", "final_plane_inlier_mask"], tuning_hint="Widen residual tolerance gradually; large jumps can hide a poor support mask instead of fixing it."),
                _parameter(fields, "plane_background_residual_tolerance_mm", affects=["plane_residual_histogram", "plane_fit_debug"], tuning_hint="Use this to explain why borderline background support was accepted or rejected."),
                _parameter(fields, "plane_fit_roi", affects=["belt_plane", "final_plane_inlier_mask"]),
                _parameter(fields, "plane_fit_downsample", affects=["belt_plane"]),
                _parameter(fields, "plane_fit_min_valid_pixels", affects=["belt_plane"]),
                _parameter(fields, "plane_fit_residual_threshold_adaptive_multiplier", affects=["plane_inlier_mask", "final_plane_inlier_mask"], tuning_hint="Raise this for noisy sensors where a fixed residual threshold is too tight relative to support z-MAD."),
                _parameter(fields, "plane_fit_max_iterations", affects=["belt_plane"]),
                _parameter(fields, "plane_background_residual_tolerance_mode", affects=["plane_residual_histogram", "plane_fit_debug"]),
                _parameter(fields, "plane_background_residual_adaptive_multiplier", affects=["plane_residual_histogram", "plane_fit_debug"]),
                _parameter(fields, "plane_background_min_coverage_ratio", affects=["expanded_plane_mask"]),
                _parameter(fields, "plane_background_fill_holes", affects=["expanded_plane_mask"]),
                _parameter(fields, "plane_background_close_kernel", affects=["expanded_plane_mask"]),
                _parameter(fields, "plane_refit_after_expansion", affects=["belt_plane", "final_plane_inlier_mask"], tuning_hint="Disable to keep the first-pass plane instead of refitting after support expansion."),
                _parameter(fields, "plane_refit_max_iterations", affects=["belt_plane"]),
            ],
            diagnostics=["plane_fit_debug", "plane_residual_histogram", "selected_surface_debug"],
            views=[
                _view("reference_model_support", "Reference-model support", ["reference_model_support_mask"]),
                _view("plane_inliers", "Plane inliers", ["final_plane_inlier_mask", "plane_inlier_mask"]),
                _view("residual_heatmap", "Residual heatmap", ["plane_residual_heatmap", "belt_plane_residuals"]),
                _view("residual_histogram", "Residual histogram", ["plane_residual_histogram"], renderer_type="table"),
                _view("json", "JSON", ["belt_plane", "plane_fit_debug", "selected_surface_debug"], renderer_type="json"),
            ],
            default_view="plane_inliers",
            help_markdown="This unit turns the selected support into the actual reference model used for normalization and segmentation.",
            controlled_by=["detect_belt_plane.stripe_filter", "detect_belt_plane.candidate_support_refinement", "detect_belt_plane.depth_plateaus"],
            parameter_groups=[
                ProcessingUnitParameterGroup(
                    id="reference_model",
                    label="Reference model",
                    param_keys=["reference_surface_model", "reference_suppression_mask_policy", "plane_fit_residual_threshold_mm", "plane_fit_min_inlier_ratio", "plane_background_residual_tolerance_mm"],
                    affects=["Plane inliers", "Residual heatmap", "Segmentation suppression mask"],
                ),
                ProcessingUnitParameterGroup(
                    id="suppression_mask",
                    label="Suppression mask policy",
                    param_keys=["reference_suppression_mask_policy"],
                    affects=["Suppression mask", "Segmentation handoff", "Foreground cleanup"],
                ),
                ProcessingUnitParameterGroup(
                    id="advanced_fitting",
                    label="Advanced fitting",
                    description="Lower-level RANSAC/refit controls for the plane fit; defaults match the pre-existing hardcoded constants.",
                    param_keys=[
                        "plane_fit_roi",
                        "plane_fit_downsample",
                        "plane_fit_min_valid_pixels",
                        "plane_fit_residual_threshold_adaptive_multiplier",
                        "plane_fit_max_iterations",
                        "plane_background_residual_tolerance_mode",
                        "plane_background_residual_adaptive_multiplier",
                        "plane_background_min_coverage_ratio",
                        "plane_background_fill_holes",
                        "plane_background_close_kernel",
                        "plane_refit_after_expansion",
                        "plane_refit_max_iterations",
                    ],
                    affects=["Plane inliers", "Residual heatmap", "Residual histogram"],
                ),
            ],
            downstream_effects=["Reference model", "Residual QA", "Normalization + segmentation handoff"],
            tuning_hints=[
                ProcessingUnitTuningHint(condition="True belt regions are rejected by plane inliers", actions=["Try Constant Z reference model", "Try Suppression = selected support", "Loosen residual threshold only after checking model choice"]),
            ],
        ),
        ProcessingUnitDefinition(
            id="detect_belt_plane.final_support",
            label="Final selected support / selected surface",
            kind="substage",
            parent_id=root_id,
            stage_id="detect_belt_plane",
            category="final_support",
            order=120,
            description="Resolved support and suppression masks handed off to downstream normalization and segmentation.",
            inputs=[ProcessingUnitInput(id="reference_model", label="Reference model", artifact_id="belt_plane", kind="json")],
            outputs=[
                ProcessingUnitOutput(id="selected_support", label="Selected support", artifact_id="selected_reference_support_mask", kind="mask"),
                ProcessingUnitOutput(id="suppression_mask", label="Reference suppression mask", artifact_id="reference_suppression_mask", kind="mask"),
            ],
            artifacts=[
                _artifact("detect_belt_plane.final_support", "selected_reference_support_mask", "Final selected support", role="final"),
                _artifact("detect_belt_plane.final_support", "reference_surface_selected_mask", "Selected surface alias", role="final"),
                _artifact("detect_belt_plane.final_support", "reference_model_support_mask", "Reference-model support", role="intermediate"),
                _artifact("detect_belt_plane.final_support", "reference_suppression_mask", "Reference suppression mask", role="final"),
                _artifact("detect_belt_plane.final_support", "support_loss_waterfall", "Support-loss waterfall", kind="json", role="diagnostic", renderer="table"),
                _artifact("detect_belt_plane.final_support", "final_support_debug", "Final support debug", kind="json", role="diagnostic", renderer="json"),
                _artifact("detect_belt_plane.final_support", "selected_surface_debug", "Selected surface debug", kind="json", role="diagnostic", renderer="json"),
            ],
            parameters=[],
            diagnostics=["support_loss_waterfall", "final_support_debug", "selected_surface_debug"],
            views=[
                _view("selected_surface", "Selected support", ["selected_reference_support_mask", "reference_surface_selected_mask"]),
                _view("reference_suppression_mask", "Suppression mask", ["reference_suppression_mask"]),
                _view("support_loss_waterfall", "Support-loss waterfall", ["support_loss_waterfall"], renderer_type="table"),
            ],
            default_view="selected_surface",
            help_markdown="The final support unit is the handoff point: it explains which support survives for model fitting and which mask downstream segmentation will actually suppress.",
            controlled_by=["detect_belt_plane.reference_model_fit", "detect_belt_plane.stripe_filter", "detect_belt_plane.candidate_support_refinement"],
        ),
    ]
    return units


def _unit(
    *,
    id: str,
    label: str,
    kind: str,
    parent_id: str | None,
    stage_id: str,
    category: str,
    order: int,
    description: str,
    inputs: list[ProcessingUnitInput] | None = None,
    outputs: list[ProcessingUnitOutput] | None = None,
    artifacts: list[ProcessingUnitArtifact] | None = None,
    parameters: list[ProcessingUnitParameter] | None = None,
    diagnostics: list[str] | None = None,
    views: list[ProcessingUnitView] | None = None,
    default_view: str | None = None,
    help_markdown: str | None = None,
    controlled_by: list[str] | None = None,
    parameter_groups: list[ProcessingUnitParameterGroup] | None = None,
    downstream_effects: list[str] | None = None,
    tuning_hints: list[ProcessingUnitTuningHint] | None = None,
) -> ProcessingUnitDefinition:
    return ProcessingUnitDefinition(
        id=id,
        label=label,
        kind=kind,
        parent_id=parent_id,
        stage_id=stage_id,
        category=category,
        order=order,
        description=description,
        inputs=list(inputs or []),
        outputs=list(outputs or []),
        artifacts=list(artifacts or []),
        parameters=list(parameters or []),
        diagnostics=list(diagnostics or []),
        views=list(views or []),
        default_view=default_view,
        help_markdown=help_markdown,
        controlled_by=list(controlled_by or []),
        parameter_groups=list(parameter_groups or []),
        downstream_effects=list(downstream_effects or []),
        tuning_hints=list(tuning_hints or []),
    )


def input_processing_units() -> list[ProcessingUnitDefinition]:
    root_id = "input"
    return [
        _unit(
            id=root_id,
            label="Load heightmap capture",
            kind="stage",
            parent_id=None,
            stage_id="input",
            category="input",
            order=0,
            description="Source capture context for the native 25D pipeline, including raw height preview, metadata, and calibration context.",
            outputs=[
                ProcessingUnitOutput(id="height_preview", label="Height preview", artifact_id="source_heightmap_preview", kind="image"),
                ProcessingUnitOutput(id="metadata", label="Source metadata", artifact_id="source_json", kind="json"),
            ],
            artifacts=[
                _artifact(root_id, "source_heightmap_preview", "Source height preview", role="input"),
                _artifact(root_id, "raw_heightmap_preview", "Raw heightmap preview", role="input"),
                _artifact(root_id, "source_json", "Source JSON", kind="json", role="diagnostic", renderer="json"),
            ],
            views=[
                _view("height_preview", "Height preview", ["source_heightmap_preview", "raw_heightmap_preview"]),
                _view("heightmap", "Heightmap (raw)", ["raw_heightmap_preview", "source_heightmap_preview"]),
                _view("json", "JSON", ["source_json"], renderer_type="json"),
            ],
            default_view="height_preview",
            help_markdown="The input contract exposes source-side context without changing runtime behavior. Missing source-side artifacts fall back to legacy source bindings in Studio.",
        ),
        _unit(
            id="input.acquisition_metadata",
            label="Acquisition metadata",
            kind="substage",
            parent_id=root_id,
            stage_id="input",
            category="input",
            order=10,
            description="Take metadata, modality bindings, and source manifest information.",
            outputs=[ProcessingUnitOutput(id="source_json", label="Source JSON", artifact_id="source_json", kind="json")],
            artifacts=[_artifact("input.acquisition_metadata", "source_json", "Source JSON", kind="json", role="diagnostic", renderer="json")],
            views=[_view("json", "JSON", ["source_json"], renderer_type="json")],
            default_view="json",
        ),
        _unit(
            id="input.raw_heightmap",
            label="Raw heightmap",
            kind="substage",
            parent_id=root_id,
            stage_id="input",
            category="input",
            order=20,
            description="Raw heightmap preview handed to Detect Reference and normalization.",
            outputs=[ProcessingUnitOutput(id="raw_preview", label="Raw heightmap preview", artifact_id="raw_heightmap_preview", kind="image")],
            artifacts=[
                _artifact("input.raw_heightmap", "raw_heightmap_preview", "Raw heightmap preview", role="input"),
                _artifact("input.raw_heightmap", "source_heightmap_preview", "Source height preview", role="input", source_artifact_id="raw_heightmap_preview"),
            ],
            views=[
                _view("height_preview", "Height preview", ["source_heightmap_preview", "raw_heightmap_preview"]),
                _view("heightmap", "Heightmap (raw)", ["raw_heightmap_preview", "source_heightmap_preview"]),
            ],
            default_view="height_preview",
        ),
        _unit(
            id="input.valid_mask",
            label="Valid sensor mask",
            kind="substage",
            parent_id=root_id,
            stage_id="input",
            category="input",
            order=25,
            description="Sensor-valid pixels available to downstream reference detection and normalization.",
            outputs=[ProcessingUnitOutput(id="valid_mask", label="Valid sensor mask", artifact_id="valid_mask", kind="mask")],
            artifacts=[_artifact("input.valid_mask", "valid_mask", "Valid sensor mask", role="input")],
            views=[_view("valid_mask", "Valid mask", ["valid_mask"])],
            default_view="valid_mask",
        ),
        _unit(
            id="input.reflectance_rgb",
            label="Reflectance / RGB",
            kind="substage",
            parent_id=root_id,
            stage_id="input",
            category="input",
            order=30,
            description="Optional secondary source bindings such as reflectance or RGB.",
            artifacts=[
                _artifact("input.reflectance_rgb", "source_reflectance_reflectance", "Reflectance source", role="diagnostic"),
                _artifact("input.reflectance_rgb", "source_rgb_rgb", "RGB source", role="diagnostic"),
            ],
            views=[_view("height_preview", "Height preview", ["source_reflectance_reflectance", "source_rgb_rgb"])],
            default_view="height_preview",
            help_markdown="Secondary source channels are optional. Studio keeps legacy fallback behavior when these bindings are absent.",
        ),
        _unit(
            id="input.calibration_context",
            label="Calibration context",
            kind="substage",
            parent_id=root_id,
            stage_id="input",
            category="input",
            order=40,
            description="Calibration snapshot, parser metadata, and source-side frame context when available.",
            artifacts=[_artifact("input.calibration_context", "source_metadata", "Source metadata", kind="table", role="diagnostic", renderer="table")],
            views=[_view("json", "JSON", ["source_metadata", "source_json"], renderer_type="json")],
            default_view="json",
        ),
    ]


def normalize_processing_units() -> list[ProcessingUnitDefinition]:
    root_id = "normalize_heights_to_plane"
    return [
        _unit(
            id=root_id,
            label="Normalize heights to reference",
            kind="stage",
            parent_id=None,
            stage_id="normalize_heights_to_plane",
            category="normalization",
            order=100,
            description="Transforms raw sensor Z into canonical height-above-belt and emits normalization QA diagnostics.",
            inputs=[
                ProcessingUnitInput(id="reference_model", label="Reference model", artifact_id="belt_plane", kind="json"),
                ProcessingUnitInput(id="raw_heightmap", label="Raw heightmap", artifact_id="raw_heightmap_preview", kind="image"),
            ],
            outputs=[
                ProcessingUnitOutput(id="normalized_height", label="Normalized height", artifact_id="normalized_heightmap", kind="image"),
                ProcessingUnitOutput(id="histogram", label="Normalized-height histogram", artifact_id="normalized_height_histogram", kind="json"),
            ],
            artifacts=[
                _artifact(root_id, "normalized_heightmap", "Normalized heightmap", role="final"),
                _artifact(root_id, "below_reference_mask", "Below/equal reference mask", role="diagnostic"),
                _artifact(root_id, "above_threshold_mask", "Above-threshold mask", role="diagnostic"),
                _artifact(root_id, "normalized_height_histogram", "Normalized-height histogram", kind="json", role="diagnostic", renderer="json"),
                _artifact(root_id, "normalization_debug", "Normalization debug", kind="json", role="diagnostic", renderer="json"),
            ],
            diagnostics=["normalization_debug", "normalized_height_histogram"],
            views=[
                _view("raw_heightmap", "Raw heightmap", ["raw_heightmap_preview"]),
                _view("reference_surface", "Reference surface model", ["belt_plane"], renderer_type="json"),
                _view("normalized_height", "Normalized height", ["normalized_heightmap"]),
                _view("residuals", "Residuals", ["plane_signed_distance_preview", "belt_plane_residuals"]),
                _view("diagnostics", "Diagnostics", ["normalization_debug"], renderer_type="table"),
                _view("below_reference", "Below/equal reference", ["below_reference_mask"]),
                _view("above_threshold", "Above threshold", ["above_threshold_mask"]),
                _view("histogram", "Histogram", ["normalized_height_histogram"], renderer_type="histogram"),
                _view("json", "JSON", ["normalization_debug", "normalized_height_histogram"], renderer_type="json"),
            ],
            default_view="normalized_height",
            help_markdown="Normalization is contract-only here: raw execution is unchanged, but the canonical height-above-belt outputs and QA views are declared explicitly.",
        ),
        _unit(
            id="normalize_heights_to_plane.reference_model_input",
            label="Reference model input",
            kind="substage",
            parent_id=root_id,
            stage_id="normalize_heights_to_plane",
            category="normalization",
            order=110,
            description="Consumes the selected reference model and raw source frame.",
            inputs=[ProcessingUnitInput(id="belt_plane", label="Reference model", artifact_id="belt_plane", kind="json")],
            outputs=[ProcessingUnitOutput(id="reference_model", label="Reference surface model", artifact_id="belt_plane", kind="json")],
            artifacts=[_artifact("normalize_heights_to_plane.reference_model_input", "belt_plane", "Reference surface model", kind="json", role="input", renderer="json")],
            views=[_view("reference_surface", "Reference surface model", ["belt_plane"], renderer_type="json")],
            default_view="reference_surface",
        ),
        _unit(
            id="normalize_heights_to_plane.plane_signed_distance",
            label="Plane signed distance",
            kind="substage",
            parent_id=root_id,
            stage_id="normalize_heights_to_plane",
            category="normalization",
            order=120,
            description="Signed-distance residuals between the raw surface and the resolved reference model.",
            artifacts=[_artifact("normalize_heights_to_plane.plane_signed_distance", "belt_plane_residuals", "Plane residuals", role="diagnostic")],
            views=[_view("residuals", "Residuals", ["belt_plane_residuals"])],
            default_view="residuals",
            controlled_by=["detect_belt_plane.reference_model_fit"],
        ),
        _unit(
            id="normalize_heights_to_plane.height_above_belt",
            label="Height above belt",
            kind="substage",
            parent_id=root_id,
            stage_id="normalize_heights_to_plane",
            category="normalization",
            order=130,
            description="Canonical normalized-height raster used by segmentation, measurement, and classification.",
            outputs=[ProcessingUnitOutput(id="normalized_height", label="Normalized height", artifact_id="normalized_heightmap", kind="image")],
            artifacts=[_artifact("normalize_heights_to_plane.height_above_belt", "normalized_heightmap", "Normalized heightmap", role="final")],
            views=[_view("normalized_height", "Normalized height", ["normalized_heightmap", "plane_fit_debug", "normalized_height_histogram", "normalized_heightmap_display"])],
            default_view="normalized_height",
        ),
        _unit(
            id="normalize_heights_to_plane.normalized_display_preview",
            label="Normalized display preview",
            kind="substage",
            parent_id=root_id,
            stage_id="normalize_heights_to_plane",
            category="normalization",
            order=140,
            description="Display-side masks derived from the normalized height field.",
            artifacts=[
                _artifact("normalize_heights_to_plane.normalized_display_preview", "below_reference_mask", "Below reference mask", role="diagnostic"),
                _artifact("normalize_heights_to_plane.normalized_display_preview", "above_threshold_mask", "Above-threshold mask", role="diagnostic"),
            ],
            views=[
                _view("below_reference", "Below/equal reference", ["below_reference_mask"]),
                _view("above_threshold", "Above threshold", ["above_threshold_mask"]),
            ],
            default_view="below_reference",
            controlled_by=["remove_belt_segment_objects.foreground_thresholding"],
        ),
        _unit(
            id="normalize_heights_to_plane.normalization_diagnostics",
            label="Normalization diagnostics",
            kind="substage",
            parent_id=root_id,
            stage_id="normalize_heights_to_plane",
            category="normalization",
            order=150,
            description="Histogram, render-context, and QA/debug payloads for normalized-height validation.",
            outputs=[ProcessingUnitOutput(id="histogram", label="Histogram", artifact_id="normalized_height_histogram", kind="json")],
            artifacts=[
                _artifact("normalize_heights_to_plane.normalization_diagnostics", "normalized_height_histogram", "Normalized-height histogram", kind="json", role="diagnostic", renderer="json"),
                _artifact("normalize_heights_to_plane.normalization_diagnostics", "normalization_debug", "Normalization debug", kind="json", role="diagnostic", renderer="json"),
            ],
            diagnostics=["normalized_height_histogram", "normalization_debug"],
            views=[
                _view("histogram", "Histogram", ["normalized_height_histogram"], renderer_type="histogram"),
                _view("diagnostics", "Diagnostics", ["normalization_debug"], renderer_type="table"),
                _view("json", "JSON", ["normalization_debug", "normalized_height_histogram"], renderer_type="json"),
            ],
            default_view="diagnostics",
        ),
    ]


def segmentation_processing_units(stage_parameter_schema: Mapping[str, Any]) -> list[ProcessingUnitDefinition]:
    fields = stage_parameter_schema.get("fields") if isinstance(stage_parameter_schema.get("fields"), Mapping) else {}
    root_id = "remove_belt_segment_objects"
    return [
        _unit(
            id=root_id,
            label="Remove reference + segment objects",
            kind="stage",
            parent_id=None,
            stage_id="remove_belt_segment_objects",
            category="segmentation",
            order=200,
            description="Foreground segmentation over normalized height, including reference suppression, morphology cleanup, and connected-component preparation.",
            inputs=[
                ProcessingUnitInput(id="normalized_height", label="Normalized height", artifact_id="normalized_heightmap", kind="image"),
                ProcessingUnitInput(id="reference_suppression_mask", label="Reference suppression mask", artifact_id="reference_suppression_mask", kind="mask", required=False),
            ],
            outputs=[
                ProcessingUnitOutput(id="final_mask", label="Final object mask", artifact_id="final_object_mask", kind="mask"),
                ProcessingUnitOutput(id="connected_components", label="Connected components overlay", artifact_id="connected_components_overlay", kind="overlay"),
            ],
            artifacts=[
                _artifact(root_id, "foreground_before_plane_suppression", "Foreground before reference suppression", role="diagnostic"),
                _artifact(root_id, "below_reference_mask", "Below reference mask", role="diagnostic"),
                _artifact(root_id, "above_threshold_mask", "Above-threshold mask", role="diagnostic"),
                _artifact(root_id, "plane_suppressed_mask", "Reference-suppressed mask", role="diagnostic"),
                _artifact(root_id, "cleaned_object_mask", "Cleaned object mask", role="diagnostic"),
                _artifact(root_id, "final_object_mask", "Final object mask", role="final"),
                _artifact(root_id, "connected_components_overlay", "Connected components overlay", kind="overlay", role="final", renderer="overlay"),
                _artifact(root_id, "segmentation_debug", "Segmentation debug", kind="json", role="diagnostic", renderer="json"),
            ],
            parameters=[
                _parameter(fields, "min_height_mm", affects=["above_threshold_mask", "final_object_mask"], tuning_hint="Raise this to suppress low residual belt texture; lower it to recover shallow objects."),
                _parameter(fields, "max_height_mm", affects=["above_threshold_mask", "final_object_mask"], tuning_hint="Use only when extreme tall outliers should be rejected from object segmentation."),
                _parameter(fields, "suppress_plane_mask_in_segmentation", affects=["plane_suppressed_mask", "final_object_mask"], tuning_hint="Keep this enabled unless you are explicitly auditing suppression behavior."),
                _parameter(fields, "reference_tolerance_mm", affects=["below_reference_mask"]),
                _parameter(fields, "ignore_small_residual_background", affects=["plane_suppressed_mask"]),
                _parameter(fields, "morphology_kernel", affects=["cleaned_object_mask", "final_object_mask"], tuning_hint="Adjust alongside minimum component area to balance speckle cleanup against edge preservation."),
                _parameter(fields, "min_component_area", affects=["cleaned_object_mask", "final_object_mask"], tuning_hint="Increase to drop noise islands; decrease to retain small fragments."),
                _parameter(fields, "fill_holes", affects=["cleaned_object_mask", "final_object_mask"], tuning_hint="Useful when segmented objects should form solid masks for downstream geometry."),
                _parameter(fields, "smoothing_kernel", affects=["cleaned_object_mask", "final_object_mask"]),
            ],
            diagnostics=["segmentation_debug"],
            views=[
                _view("threshold_mask", "Threshold mask", ["above_threshold_mask", "normalized_height_threshold_mask"]),
                _view("cleaned_mask", "Cleaned mask", ["cleaned_object_mask", "final_object_mask"]),
                _view("segmentation", "Connected components", ["connected_components", "segmentation_debug"], renderer_type="table"),
                _view("overlay", "Overlay", ["connected_components_overlay", "height_segmentation_overlay", "height_segmentation"], renderer_type="overlay"),
                _view("json", "JSON", ["segmentation_debug"], renderer_type="json"),
            ],
            default_view="overlay",
            help_markdown="The segmentation contract scopes foreground thresholding, reference suppression, and cleanup into explicit units while preserving the existing segmentation runtime.",
            parameter_groups=[
                ProcessingUnitParameterGroup(
                    id="foreground_thresholding",
                    label="Foreground thresholding",
                    param_keys=["min_height_mm", "max_height_mm"],
                    description="Height cutoff separating candidate objects from the belt/background before any suppression or cleanup runs.",
                    affects=["Threshold mask", "Cleaned mask", "Overlay"],
                ),
                ProcessingUnitParameterGroup(
                    id="reference_suppression",
                    label="Reference suppression",
                    param_keys=["suppress_plane_mask_in_segmentation", "reference_tolerance_mm", "ignore_small_residual_background"],
                    description="Removes pixels still coincident with the detected belt/reference surface and belt-stripe texture before final object extraction.",
                    affects=["Cleaned mask", "Overlay"],
                ),
                ProcessingUnitParameterGroup(
                    id="morphology_cleanup",
                    label="Morphology cleanup",
                    param_keys=["morphology_kernel", "fill_holes", "smoothing_kernel"],
                    description="Open/close morphology and optional hole-filling and smoothing that clean the thresholded mask before connected-component extraction.",
                    affects=["Cleaned mask", "Connected components", "Overlay"],
                ),
                ProcessingUnitParameterGroup(
                    id="connected_component_preparation",
                    label="Connected component preparation",
                    param_keys=["min_component_area"],
                    description="Minimum pixel area for a connected component to be kept as a real object, dropping small noise islands.",
                    affects=["Connected components", "Overlay"],
                ),
            ],
            downstream_effects=[
                "Final object mask -> Geometry connected-component extraction, contour/hull/ellipse fitting",
                "Final object mask -> every downstream Measure/Diagnostics/Classification metric",
                "Reference suppression -> footprint and volume accuracy (belt leakage inflates both)",
            ],
            tuning_hints=[
                ProcessingUnitTuningHint(
                    condition="Real objects are missing from Geometry/Measure entirely",
                    actions=["Lower Min height (mm) to recover shallow objects", "Lower Min component area if objects are small in pixel area", "Check the Foreground thresholding substage's mask directly before blaming later steps"],
                ),
                ProcessingUnitTuningHint(
                    condition="Object count is inflated, or belt/stripe texture appears as spurious objects",
                    actions=["Confirm Suppress plane mask in segmentation is enabled", "Inspect the Reference suppression substage's rejected-residual/rejected-stripe chips to see what's actually being removed", "Raise Min component area to filter residual noise instead"],
                ),
                ProcessingUnitTuningHint(
                    condition="Object masks look fragmented or have jagged edges",
                    actions=["Increase Morphology kernel to smooth boundaries", "Enable Fill holes if objects have small interior gaps"],
                ),
            ],
        ),
        _unit(
            id="remove_belt_segment_objects.foreground_thresholding",
            label="Foreground thresholding",
            kind="substage",
            parent_id=root_id,
            stage_id="remove_belt_segment_objects",
            category="segmentation",
            order=210,
            description="Height-based thresholding over normalized height.",
            outputs=[ProcessingUnitOutput(id="threshold_mask", label="Threshold mask", artifact_id="above_threshold_mask", kind="mask")],
            artifacts=[
                _artifact("remove_belt_segment_objects.foreground_thresholding", "foreground_before_plane_suppression", "Foreground before reference suppression", role="diagnostic"),
                _artifact("remove_belt_segment_objects.foreground_thresholding", "above_threshold_mask", "Above-threshold mask", role="diagnostic"),
            ],
            parameters=[
                _parameter(fields, "min_height_mm", affects=["above_threshold_mask", "foreground_before_plane_suppression"]),
                _parameter(fields, "max_height_mm", affects=["above_threshold_mask", "foreground_before_plane_suppression"]),
            ],
            views=[_view("threshold_mask", "Threshold mask", ["above_threshold_mask", "foreground_before_plane_suppression"])],
            default_view="threshold_mask",
            tuning_hints=[
                ProcessingUnitTuningHint(condition="Shallow real objects are being excluded", actions=["Lower Min height (mm)"]),
                ProcessingUnitTuningHint(condition="Tall outlier artifacts are appearing as objects", actions=["Set Max height (mm) to reject them"]),
            ],
        ),
        _unit(
            id="remove_belt_segment_objects.reference_suppression",
            label="Reference suppression",
            kind="substage",
            parent_id=root_id,
            stage_id="remove_belt_segment_objects",
            category="segmentation",
            order=220,
            description="Consumes the selected support-derived reference suppression mask and removes belt-support pixels from the segmentation domain.",
            outputs=[ProcessingUnitOutput(id="suppressed_mask", label="Reference-suppressed mask", artifact_id="plane_suppressed_mask", kind="mask")],
            artifacts=[
                _artifact("remove_belt_segment_objects.reference_suppression", "below_reference_mask", "Below reference mask", role="diagnostic"),
                _artifact("remove_belt_segment_objects.reference_suppression", "plane_suppressed_mask", "Reference-suppressed mask", role="diagnostic"),
                _artifact("remove_belt_segment_objects.reference_suppression", "rejected_background_residuals", "Rejected background residuals", role="diagnostic"),
                _artifact("remove_belt_segment_objects.reference_suppression", "rejected_belt_stripes", "Rejected belt stripes", role="diagnostic"),
            ],
            parameters=[
                _parameter(fields, "suppress_plane_mask_in_segmentation", affects=["plane_suppressed_mask", "rejected_belt_stripes"]),
                _parameter(fields, "reference_tolerance_mm", affects=["below_reference_mask"]),
                _parameter(fields, "ignore_small_residual_background", affects=["rejected_background_residuals"]),
            ],
            views=[_view("cleaned_mask", "Cleaned mask", ["plane_suppressed_mask", "rejected_background_residuals", "rejected_belt_stripes"])],
            default_view="cleaned_mask",
            controlled_by=["detect_belt_plane.final_support"],
            tuning_hints=[
                ProcessingUnitTuningHint(
                    condition="Belt surface or texture is leaking into the object mask",
                    actions=["Confirm Suppress plane mask in segmentation is enabled", "Check the rejected-residual and rejected-stripe chips to see what is actually being removed"],
                ),
            ],
        ),
        _unit(
            id="remove_belt_segment_objects.stripe_suppression_consumption",
            label="Stripe suppression consumption",
            kind="substage",
            parent_id=root_id,
            stage_id="remove_belt_segment_objects",
            category="segmentation",
            order=230,
            description="Tracks stripe-filter removals consumed from Detect Reference before object cleanup.",
            artifacts=[_artifact("remove_belt_segment_objects.stripe_suppression_consumption", "rejected_belt_stripes", "Rejected belt stripes", role="diagnostic")],
            views=[_view("cleaned_mask", "Cleaned mask", ["rejected_belt_stripes", "plane_suppressed_mask"])],
            default_view="cleaned_mask",
            controlled_by=["detect_belt_plane.stripe_filter"],
        ),
        _unit(
            id="remove_belt_segment_objects.morphology_cleanup",
            label="Morphology cleanup",
            kind="substage",
            parent_id=root_id,
            stage_id="remove_belt_segment_objects",
            category="segmentation",
            order=240,
            description="Morphological cleanup and hole filling before final component extraction.",
            outputs=[ProcessingUnitOutput(id="cleaned_mask", label="Cleaned object mask", artifact_id="cleaned_object_mask", kind="mask")],
            artifacts=[_artifact("remove_belt_segment_objects.morphology_cleanup", "cleaned_object_mask", "Cleaned object mask", role="diagnostic")],
            parameters=[
                _parameter(fields, "morphology_kernel", affects=["cleaned_object_mask", "final_object_mask"]),
                _parameter(fields, "fill_holes", affects=["cleaned_object_mask", "final_object_mask"]),
                _parameter(fields, "smoothing_kernel", affects=["cleaned_object_mask", "final_object_mask"]),
            ],
            views=[_view("cleaned_mask", "Cleaned mask", ["cleaned_object_mask"])],
            default_view="cleaned_mask",
            tuning_hints=[
                ProcessingUnitTuningHint(condition="Masks look fragmented or edges are jagged", actions=["Increase Morphology kernel", "Enable Fill holes for objects with interior gaps"]),
            ],
        ),
        _unit(
            id="remove_belt_segment_objects.connected_component_preparation",
            label="Connected component preparation",
            kind="substage",
            parent_id=root_id,
            stage_id="remove_belt_segment_objects",
            category="segmentation",
            order=250,
            description="Final object mask and component-level preparation for geometry.",
            outputs=[
                ProcessingUnitOutput(id="final_mask", label="Final object mask", artifact_id="final_object_mask", kind="mask"),
                ProcessingUnitOutput(id="components_overlay", label="Connected components overlay", artifact_id="connected_components_overlay", kind="overlay"),
            ],
            artifacts=[
                _artifact("remove_belt_segment_objects.connected_component_preparation", "final_object_mask", "Final object mask", role="final"),
                _artifact("remove_belt_segment_objects.connected_component_preparation", "connected_components_overlay", "Connected components overlay", kind="overlay", role="final", renderer="overlay"),
                _artifact("remove_belt_segment_objects.connected_component_preparation", "segmentation_debug", "Segmentation debug", kind="json", role="diagnostic", renderer="json"),
            ],
            parameters=[_parameter(fields, "min_component_area", affects=["final_object_mask", "connected_components_overlay"])],
            diagnostics=["segmentation_debug"],
            views=[
                _view("segmentation", "Connected components", ["connected_components", "segmentation_debug"], renderer_type="table"),
                _view("overlay", "Overlay", ["connected_components_overlay", "height_segmentation_overlay", "height_segmentation"], renderer_type="overlay"),
                _view("json", "JSON", ["segmentation_debug"], renderer_type="json"),
            ],
            default_view="overlay",
            tuning_hints=[
                ProcessingUnitTuningHint(condition="Small real objects are being dropped, or noise islands are surviving", actions=["Lower Min component area to keep small objects", "Raise it to drop noise"]),
            ],
        ),
    ]


def geometry_processing_units() -> list[ProcessingUnitDefinition]:
    root_id = "geometry"
    return [
        _unit(
            id=root_id,
            label="Footprint geometry",
            kind="stage",
            parent_id=None,
            stage_id="geometry",
            category="geometry",
            order=300,
            description="Component extraction, contour/hull/ellipse geometry, and geometry summaries used by measurement and diagnostics.",
            inputs=[ProcessingUnitInput(id="segmentation_components", label="Connected components", artifact_id="connected_components", kind="json")],
            outputs=[ProcessingUnitOutput(id="geometry_summary", label="Geometry summary", artifact_id="geometry_debug_summary", kind="json")],
            artifacts=[
                _artifact(root_id, "connected_components", "Connected components", kind="json", role="input", renderer="json"),
                _artifact(root_id, "connected_components_overlay", "Connected components overlay", kind="overlay", role="diagnostic", renderer="overlay"),
                _artifact(root_id, "contour", "Contour geometry", kind="json", role="diagnostic", renderer="json"),
                _artifact(root_id, "convex_hull", "Convex hull geometry", kind="json", role="diagnostic", renderer="json"),
                _artifact(root_id, "fitted_ellipse", "Fitted ellipse geometry", kind="json", role="diagnostic", renderer="json"),
                _artifact(root_id, "geometry_debug_summary", "Geometry debug summary", kind="json", role="final", renderer="json"),
            ],
            diagnostics=["geometry_debug_summary"],
            views=[
                _view("measurements_25d", "Measurements", ["geometry_debug_summary"], renderer_type="table"),
                _view("residuals", "Residuals", ["connected_components_overlay", "height_segmentation"], renderer_type="overlay"),
                _view("profiles", "Profiles", ["contour", "fitted_ellipse"], renderer_type="json"),
                _view("provenance", "Provenance", ["geometry_debug_summary"], renderer_type="table"),
                _view("geometry_debug", "Geometry debug", ["geometry_debug_summary", "contour", "convex_hull", "fitted_ellipse"], renderer_type="table"),
                _view("json", "JSON", ["geometry_debug_summary", "contour", "convex_hull", "fitted_ellipse"], renderer_type="json"),
            ],
            default_view="measurements_25d",
            help_markdown="Geometry currently derives from connected components and is only partially materialized as artifacts. The contract keeps this stage inspectable without rewriting geometry execution.",
        ),
        _unit(
            id="geometry.connected_component_extraction",
            label="Connected component extraction",
            kind="substage",
            parent_id=root_id,
            stage_id="geometry",
            category="geometry",
            order=310,
            description="Consumes segmentation connected components for downstream contour and ellipse work.",
            outputs=[ProcessingUnitOutput(id="components", label="Connected components", artifact_id="connected_components", kind="json")],
            artifacts=[
                _artifact("geometry.connected_component_extraction", "connected_components", "Connected components", kind="json", role="input", renderer="json"),
                _artifact("geometry.connected_component_extraction", "connected_components_overlay", "Connected components overlay", kind="overlay", role="diagnostic", renderer="overlay"),
            ],
            views=[
                _view("measurements_25d", "Measurements", ["connected_components"], renderer_type="table"),
                _view("residuals", "Residuals", ["connected_components_overlay", "height_segmentation"], renderer_type="overlay"),
            ],
            default_view="measurements_25d",
        ),
        _unit(
            id="geometry.contour_extraction",
            label="Contour extraction",
            kind="substage",
            parent_id=root_id,
            stage_id="geometry",
            category="geometry",
            order=320,
            description="Extracted contour payloads derived from connected components.",
            outputs=[ProcessingUnitOutput(id="contour", label="Contour geometry", artifact_id="contour", kind="json")],
            artifacts=[_artifact("geometry.contour_extraction", "contour", "Contour geometry", kind="json", role="diagnostic", renderer="json")],
            views=[_view("profiles", "Profiles", ["contour"], renderer_type="json")],
            default_view="profiles",
        ),
        _unit(
            id="geometry.hull_fitting",
            label="Hull fitting",
            kind="substage",
            parent_id=root_id,
            stage_id="geometry",
            category="geometry",
            order=330,
            description="Convex hull payloads for footprint-shape summaries.",
            outputs=[ProcessingUnitOutput(id="convex_hull", label="Convex hull geometry", artifact_id="convex_hull", kind="json")],
            artifacts=[_artifact("geometry.hull_fitting", "convex_hull", "Convex hull geometry", kind="json", role="diagnostic", renderer="json")],
            views=[_view("geometry_debug", "Geometry debug", ["convex_hull"], renderer_type="table")],
            default_view="geometry_debug",
        ),
        _unit(
            id="geometry.ellipse_fitting",
            label="Ellipse fitting",
            kind="substage",
            parent_id=root_id,
            stage_id="geometry",
            category="geometry",
            order=340,
            description="Ellipse-fit payloads and object-level footprint shape summaries.",
            outputs=[ProcessingUnitOutput(id="ellipse", label="Fitted ellipse geometry", artifact_id="fitted_ellipse", kind="json")],
            artifacts=[_artifact("geometry.ellipse_fitting", "fitted_ellipse", "Fitted ellipse geometry", kind="json", role="diagnostic", renderer="json")],
            views=[_view("geometry_debug", "Geometry debug", ["fitted_ellipse"], renderer_type="table")],
            default_view="geometry_debug",
        ),
        _unit(
            id="geometry.geometry_summary",
            label="Geometry summary",
            kind="substage",
            parent_id=root_id,
            stage_id="geometry",
            category="geometry",
            order=350,
            description="Object-level geometry debug summary and invariant context.",
            outputs=[ProcessingUnitOutput(id="summary", label="Geometry debug summary", artifact_id="geometry_debug_summary", kind="json")],
            artifacts=[_artifact("geometry.geometry_summary", "geometry_debug_summary", "Geometry debug summary", kind="json", role="final", renderer="json")],
            diagnostics=["geometry_debug_summary"],
            views=[
                _view("measurements_25d", "Measurements", ["geometry_debug_summary"], renderer_type="table"),
                _view("provenance", "Provenance", ["geometry_debug_summary"], renderer_type="table"),
                _view("geometry_debug", "Geometry debug", ["geometry_debug_summary"], renderer_type="table"),
                _view("json", "JSON", ["geometry_debug_summary"], renderer_type="json"),
            ],
            default_view="measurements_25d",
        ),
    ]


def measurement_processing_units() -> list[ProcessingUnitDefinition]:
    root_id = "measurement"
    return [
        _unit(
            id=root_id,
            label="Height + volume metrics",
            kind="stage",
            parent_id=None,
            stage_id="measurement",
            category="measurement",
            order=400,
            description="Per-object height statistics, volume proxy, dimensions, and surface-shape metrics.",
            inputs=[ProcessingUnitInput(id="components", label="Connected components", artifact_id="connected_components", kind="json")],
            outputs=[ProcessingUnitOutput(id="geometry_debug", label="Geometry debug summary", artifact_id="geometry_debug_summary", kind="json")],
            artifacts=[
                _artifact(root_id, "measurement_overlay", "Measurement overlay", kind="overlay", role="diagnostic", renderer="overlay"),
                _artifact(root_id, "geometry_debug_summary", "Geometry debug summary", kind="json", role="final", renderer="json"),
                _artifact(root_id, "known_object_scale_validation", "Known-object scale validation", kind="json", role="diagnostic", renderer="json"),
            ],
            diagnostics=["geometry_debug_summary", "known_object_scale_validation"],
            views=[
                _view("measurements_25d", "Measurements", ["geometry_debug_summary"], renderer_type="table"),
                _view("residuals", "Residuals", ["measurement_overlay"]),
                _view("profiles", "Profiles", ["radial_profile", "diagnostics_normalized_height_histogram"], renderer_type="json"),
                _view("provenance", "Provenance", ["geometry_debug_summary", "known_object_scale_validation"], renderer_type="table"),
                _view("geometry_debug", "Geometry debug", ["geometry_debug_summary"], renderer_type="table"),
                _view("json", "JSON", ["geometry_debug_summary", "known_object_scale_validation"], renderer_type="json"),
            ],
            default_view="measurements_25d",
            help_markdown="Measurement remains artifact-light in the current runtime. The contract surfaces the existing object-level summaries and correction context without altering computation.",
        ),
        _unit(
            id="measurement.height_metrics",
            label="Height metrics",
            kind="substage",
            parent_id=root_id,
            stage_id="measurement",
            category="measurement",
            order=410,
            description="Visible-surface height statistics over normalized height.",
            artifacts=[_artifact("measurement.height_metrics", "geometry_debug_summary", "Geometry debug summary", kind="json", role="diagnostic", renderer="json")],
            views=[_view("measurements_25d", "Measurements", ["geometry_debug_summary"], renderer_type="table")],
            default_view="measurements_25d",
        ),
        _unit(
            id="measurement.volume_proxy",
            label="Volume proxy",
            kind="substage",
            parent_id=root_id,
            stage_id="measurement",
            category="measurement",
            order=420,
            description="Integrated height and convex volume proxy features.",
            artifacts=[_artifact("measurement.volume_proxy", "geometry_debug_summary", "Geometry debug summary", kind="json", role="diagnostic", renderer="json")],
            views=[_view("measurements_25d", "Measurements", ["geometry_debug_summary"], renderer_type="table")],
            default_view="measurements_25d",
        ),
        _unit(
            id="measurement.dimensions",
            label="Dimensions",
            kind="substage",
            parent_id=root_id,
            stage_id="measurement",
            category="measurement",
            order=430,
            description="Footprint and dimension summaries after geometry correction.",
            artifacts=[_artifact("measurement.dimensions", "geometry_debug_summary", "Geometry debug summary", kind="json", role="diagnostic", renderer="json")],
            views=[_view("measurements_25d", "Measurements", ["geometry_debug_summary"], renderer_type="table")],
            default_view="measurements_25d",
        ),
        _unit(
            id="measurement.deformation_sphericity",
            label="Deformation / sphericity",
            kind="substage",
            parent_id=root_id,
            stage_id="measurement",
            category="measurement",
            order=440,
            description="Shape, sphericity, and surface-consistency feature groups used by classification.",
            artifacts=[_artifact("measurement.deformation_sphericity", "geometry_debug_summary", "Geometry debug summary", kind="json", role="diagnostic", renderer="json")],
            views=[_view("geometry_debug", "Geometry debug", ["geometry_debug_summary"], renderer_type="table")],
            default_view="geometry_debug",
        ),
        _unit(
            id="measurement.correction_calibration_context",
            label="Correction / calibration context",
            kind="substage",
            parent_id=root_id,
            stage_id="measurement",
            category="measurement",
            order=450,
            description="Known-object scale validation and correction provenance when present.",
            outputs=[ProcessingUnitOutput(id="scale_validation", label="Known-object scale validation", artifact_id="known_object_scale_validation", kind="json")],
            artifacts=[_artifact("measurement.correction_calibration_context", "known_object_scale_validation", "Known-object scale validation", kind="json", role="diagnostic", renderer="json")],
            diagnostics=["known_object_scale_validation"],
            views=[
                _view("provenance", "Provenance", ["known_object_scale_validation"], renderer_type="table"),
                _view("json", "JSON", ["known_object_scale_validation"], renderer_type="json"),
            ],
            default_view="provenance",
        ),
    ]


def diagnostics_processing_units(stage_parameter_schema: Mapping[str, Any]) -> list[ProcessingUnitDefinition]:
    fields = stage_parameter_schema.get("fields") if isinstance(stage_parameter_schema.get("fields"), Mapping) else {}
    root_id = "measurement_diagnostics"
    return [
        _unit(
            id=root_id,
            label="Measurement diagnostics",
            kind="stage",
            parent_id=None,
            stage_id="measurement_diagnostics",
            category="measurement_diagnostics",
            order=500,
            description="Feature vector, provenance, quality flags, and known-object validation context.",
            inputs=[ProcessingUnitInput(id="measurement_summary", label="Geometry debug summary", artifact_id="geometry_debug_summary", kind="json", required=False)],
            outputs=[
                ProcessingUnitOutput(id="feature_vector", label="Feature vector", artifact_id="feature_vector", kind="json"),
                ProcessingUnitOutput(id="quality_flags", label="Quality flags", artifact_id="quality_flags", kind="json"),
            ],
            artifacts=[
                _artifact(root_id, "measurement_diagnostics", "Measurement diagnostics", kind="json", role="diagnostic", renderer="json"),
                _artifact(root_id, "feature_vector", "Feature vector", kind="json", role="final", renderer="json"),
                _artifact(root_id, "feature_provenance", "Feature provenance", kind="json", role="diagnostic", renderer="json"),
                _artifact(root_id, "quality_flags", "Quality flags", kind="json", role="diagnostic", renderer="json"),
            ],
            parameters=[
                _parameter(fields, "enabled", affects=["measurement_diagnostics", "quality_flags"], tuning_hint="Enable only when comparing against a known reference object or applying correction context."),
                _parameter(fields, "target_selection", affects=["measurement_diagnostics"], tuning_hint="Use manual component selection only when the automatic primary object choice is wrong."),
                _parameter(fields, "manual_component_id", affects=["measurement_diagnostics"], tuning_hint="Choose the object id that should be treated as the known calibration object."),
                _parameter(fields, "known_width_mm", affects=["feature_provenance"], tuning_hint="Enter the physical X width of the known reference object."),
                _parameter(fields, "known_depth_mm", affects=["feature_provenance"], tuning_hint="Enter the physical Y depth of the known reference object."),
                _parameter(fields, "known_height_mm", affects=["feature_provenance"], tuning_hint="Enter the physical Z height of the known reference object."),
                _parameter(fields, "tolerance_percent", affects=["quality_flags"], tuning_hint="Use a tighter tolerance to surface subtle scale drift."),
                _parameter(fields, "apply_correction", affects=["feature_vector", "feature_provenance"], tuning_hint="Keep this enabled to feed corrected metrics downstream when the known-object validation is trusted."),
            ],
            diagnostics=["measurement_diagnostics", "feature_vector", "feature_provenance", "quality_flags"],
            views=[
                _view("diagnostics", "Diagnostics", ["measurement_diagnostics"], renderer_type="table"),
                _view("feature_vector", "Feature Vector", ["feature_vector"], renderer_type="table"),
                _view("quality_flags", "Quality Flags", ["quality_flags"], renderer_type="table"),
                _view("provenance", "Provenance", ["feature_provenance"], renderer_type="table"),
                _view("json", "JSON", ["measurement_diagnostics", "feature_vector", "feature_provenance", "quality_flags"], renderer_type="json"),
            ],
            default_view="diagnostics",
            help_markdown="Diagnostics parameters come from the known-object calibration contract. This layer scopes those controls to the units that consume them.",
            parameter_groups=[
                ProcessingUnitParameterGroup(
                    id="enable_and_target",
                    label="Enable & target selection",
                    param_keys=["enabled", "target_selection", "manual_component_id"],
                    description="Turn on known-object validation and choose which detected object is the calibration reference.",
                    affects=["Diagnostics", "Quality flags"],
                ),
                ProcessingUnitParameterGroup(
                    id="known_dimensions",
                    label="Known physical dimensions",
                    param_keys=["known_width_mm", "known_depth_mm", "known_height_mm"],
                    description="Real-world X/Y/Z dimensions of the known calibration object, used to compute the scale-correction factor.",
                    affects=["Provenance"],
                ),
                ProcessingUnitParameterGroup(
                    id="validation_and_correction",
                    label="Validation & correction behavior",
                    param_keys=["tolerance_percent", "apply_correction"],
                    description="How strictly to flag scale drift, and whether the computed correction actually gets applied to downstream metrics.",
                    affects=["Quality flags", "Feature Vector", "Provenance"],
                ),
            ],
            downstream_effects=[
                "Apply correction -> Feature Vector and Provenance values Classification later reads",
                "Tolerance percent -> Quality flags drift warnings, independent of whether geometry actually changed",
                "With Enabled off, Feature Vector/Provenance pass through uncorrected and the dimension/tolerance fields have no effect",
            ],
            tuning_hints=[
                ProcessingUnitTuningHint(
                    condition="Correction doesn't seem to be applying / measurements look unscaled",
                    actions=["Confirm Enabled is checked", "Confirm Apply correction is also checked -- enabling validation alone does not apply it", "Confirm Target selection / Manual component id actually points at the real calibration object in this take"],
                ),
                ProcessingUnitTuningHint(
                    condition="Too many or too few scale-drift quality flags",
                    actions=["Widen Tolerance percent to reduce false-positive drift warnings", "Narrow it to catch subtler drift"],
                ),
                ProcessingUnitTuningHint(
                    condition="Scale correction looks wrong for this object",
                    actions=["Double check Known width/depth/height (mm) match the physical calibration object placed in this take", "If using manual selection, confirm Manual component id points at the correct detected object"],
                ),
            ],
        ),
        _unit(
            id="measurement_diagnostics.feature_vector_generation",
            label="Feature vector generation",
            kind="substage",
            parent_id=root_id,
            stage_id="measurement_diagnostics",
            category="measurement_diagnostics",
            order=510,
            description="Canonical feature vector generation from measurement outputs.",
            outputs=[ProcessingUnitOutput(id="feature_vector", label="Feature vector", artifact_id="feature_vector", kind="json")],
            artifacts=[_artifact("measurement_diagnostics.feature_vector_generation", "feature_vector", "Feature vector", kind="json", role="final", renderer="json")],
            views=[_view("feature_vector", "Feature Vector", ["feature_vector"], renderer_type="table")],
            default_view="feature_vector",
        ),
        _unit(
            id="measurement_diagnostics.quality_flags",
            label="Quality flags",
            kind="substage",
            parent_id=root_id,
            stage_id="measurement_diagnostics",
            category="measurement_diagnostics",
            order=520,
            description="Quality and readiness flags over the feature vector.",
            outputs=[ProcessingUnitOutput(id="quality_flags", label="Quality flags", artifact_id="quality_flags", kind="json")],
            artifacts=[_artifact("measurement_diagnostics.quality_flags", "quality_flags", "Quality flags", kind="json", role="diagnostic", renderer="json")],
            views=[_view("quality_flags", "Quality Flags", ["quality_flags"], renderer_type="table")],
            default_view="quality_flags",
        ),
        _unit(
            id="measurement_diagnostics.invariant_checks",
            label="Invariant checks",
            kind="substage",
            parent_id=root_id,
            stage_id="measurement_diagnostics",
            category="measurement_diagnostics",
            order=530,
            description="Feature provenance and invariant-check payloads.",
            outputs=[ProcessingUnitOutput(id="provenance", label="Feature provenance", artifact_id="feature_provenance", kind="json")],
            artifacts=[_artifact("measurement_diagnostics.invariant_checks", "feature_provenance", "Feature provenance", kind="json", role="diagnostic", renderer="json")],
            views=[_view("provenance", "Provenance", ["feature_provenance"], renderer_type="table")],
            default_view="provenance",
        ),
        _unit(
            id="measurement_diagnostics.known_object_validation",
            label="Known-object validation",
            kind="substage",
            parent_id=root_id,
            stage_id="measurement_diagnostics",
            category="measurement_diagnostics",
            order=540,
            description="Known-object validation settings and diagnostics context.",
            parameters=[
                _parameter(fields, "enabled", affects=["measurement_diagnostics", "quality_flags"]),
                _parameter(fields, "target_selection", affects=["measurement_diagnostics"]),
                _parameter(fields, "manual_component_id", affects=["measurement_diagnostics"]),
                _parameter(fields, "known_width_mm", affects=["feature_provenance"]),
                _parameter(fields, "known_depth_mm", affects=["feature_provenance"]),
                _parameter(fields, "known_height_mm", affects=["feature_provenance"]),
                _parameter(fields, "tolerance_percent", affects=["quality_flags"]),
                _parameter(fields, "apply_correction", affects=["feature_vector", "feature_provenance"]),
            ],
            artifacts=[_artifact("measurement_diagnostics.known_object_validation", "measurement_diagnostics", "Measurement diagnostics", kind="json", role="diagnostic", renderer="json")],
            views=[_view("diagnostics", "Diagnostics", ["measurement_diagnostics"], renderer_type="table")],
            default_view="diagnostics",
            parameter_groups=[
                ProcessingUnitParameterGroup(
                    id="enable_and_target",
                    label="Enable & target selection",
                    param_keys=["enabled", "target_selection", "manual_component_id"],
                    description="Turn on known-object validation and choose which detected object is the calibration reference.",
                    affects=["Diagnostics", "Quality flags"],
                ),
                ProcessingUnitParameterGroup(
                    id="known_dimensions",
                    label="Known physical dimensions",
                    param_keys=["known_width_mm", "known_depth_mm", "known_height_mm"],
                    description="Real-world X/Y/Z dimensions of the known calibration object, used to compute the scale-correction factor.",
                    affects=["Provenance"],
                ),
                ProcessingUnitParameterGroup(
                    id="validation_and_correction",
                    label="Validation & correction behavior",
                    param_keys=["tolerance_percent", "apply_correction"],
                    description="How strictly to flag scale drift, and whether the computed correction actually gets applied to downstream metrics.",
                    affects=["Quality flags", "Feature Vector", "Provenance"],
                ),
            ],
            downstream_effects=[
                "Apply correction -> Feature Vector and Provenance values Classification later reads",
                "Tolerance percent -> Quality flags drift warnings, independent of whether geometry actually changed",
                "This is the substage that owns known-object calibration -- Measure's own Correction / calibration context view only displays the result",
            ],
            tuning_hints=[
                ProcessingUnitTuningHint(
                    condition="Correction doesn't seem to be applying / measurements look unscaled",
                    actions=["Confirm Enabled is checked", "Confirm Apply correction is also checked -- enabling validation alone does not apply it", "Confirm Target selection / Manual component id actually points at the real calibration object in this take"],
                ),
                ProcessingUnitTuningHint(
                    condition="Too many or too few scale-drift quality flags",
                    actions=["Widen Tolerance percent to reduce false-positive drift warnings", "Narrow it to catch subtler drift"],
                ),
                ProcessingUnitTuningHint(
                    condition="Scale correction looks wrong for this object",
                    actions=["Double check Known width/depth/height (mm) match the physical calibration object placed in this take", "If using manual selection, confirm Manual component id points at the correct detected object"],
                ),
            ],
        ),
    ]


def classification_processing_units() -> list[ProcessingUnitDefinition]:
    root_id = "classification"
    return [
        _unit(
            id=root_id,
            label="Mining-ball classification",
            kind="stage",
            parent_id=None,
            stage_id="classification",
            category="classification",
            order=600,
            description="Primary 25D classifier, superclass aggregation, explanations, and runtime diagnostics.",
            inputs=[ProcessingUnitInput(id="feature_vector", label="Feature vector", artifact_id="feature_vector", kind="json", required=False)],
            outputs=[ProcessingUnitOutput(id="classification", label="Classification result", artifact_id="classification_result_25d", kind="json")],
            artifacts=[
                _artifact(root_id, "classification_result_25d", "Classification result", kind="json", role="final", renderer="json"),
                _artifact(root_id, "classification_explanation", "Classification explanation", kind="json", role="diagnostic", renderer="json"),
                _artifact(root_id, "classification_runtime_diagnostics", "Classification runtime diagnostics", kind="json", role="diagnostic", renderer="json"),
            ],
            diagnostics=["classification_explanation", "classification_runtime_diagnostics"],
            views=[
                _view("classification_25d", "Classification", ["classification_result_25d"], renderer_type="table"),
                _view("rule_explanation", "Rule explanation", ["classification_explanation"], renderer_type="table"),
                _view("metric_details", "Metric details", ["classification_explanation"], renderer_type="table"),
                _view("diagnostics", "Diagnostics", ["classification_runtime_diagnostics"], renderer_type="table"),
                _view("json", "JSON", ["classification_result_25d", "classification_explanation", "classification_runtime_diagnostics"], renderer_type="json"),
            ],
            default_view="classification_25d",
        ),
        _unit(
            id="classification.primary_heuristic_classifier",
            label="Primary heuristic classifier",
            kind="substage",
            parent_id=root_id,
            stage_id="classification",
            category="classification",
            order=610,
            description="Rule-based 25D classification over the canonical feature vector.",
            outputs=[ProcessingUnitOutput(id="classification_result", label="Classification result", artifact_id="classification_result_25d", kind="json")],
            artifacts=[_artifact("classification.primary_heuristic_classifier", "classification_result_25d", "Classification result", kind="json", role="final", renderer="json")],
            views=[_view("classification_25d", "Classification", ["classification_result_25d"], renderer_type="table")],
            default_view="classification_25d",
        ),
        _unit(
            id="classification.sph3d_fallback",
            label="SPH3D fallback",
            kind="substage",
            parent_id=root_id,
            stage_id="classification",
            category="classification",
            order=620,
            description="Fallback or supporting classifier diagnostics when spherical-shape features dominate.",
            artifacts=[_artifact("classification.sph3d_fallback", "classification_runtime_diagnostics", "Classification runtime diagnostics", kind="json", role="diagnostic", renderer="json")],
            views=[_view("diagnostics", "Diagnostics", ["classification_runtime_diagnostics"], renderer_type="table")],
            default_view="diagnostics",
        ),
        _unit(
            id="classification.superclass_aggregation",
            label="Superclass aggregation",
            kind="substage",
            parent_id=root_id,
            stage_id="classification",
            category="classification",
            order=630,
            description="Aggregates object-level classifications into the public superclass result.",
            artifacts=[_artifact("classification.superclass_aggregation", "classification_result_25d", "Classification result", kind="json", role="final", renderer="json")],
            views=[_view("classification_25d", "Classification", ["classification_result_25d"], renderer_type="table")],
            default_view="classification_25d",
        ),
        _unit(
            id="classification.explanation_generation",
            label="Explanation generation",
            kind="substage",
            parent_id=root_id,
            stage_id="classification",
            category="classification",
            order=640,
            description="Classification explanation and rule-firing context.",
            outputs=[ProcessingUnitOutput(id="explanation", label="Classification explanation", artifact_id="classification_explanation", kind="json")],
            artifacts=[_artifact("classification.explanation_generation", "classification_explanation", "Classification explanation", kind="json", role="diagnostic", renderer="json")],
            views=[
                _view("rule_explanation", "Rule explanation", ["classification_explanation"], renderer_type="table"),
                _view("metric_details", "Metric details", ["classification_explanation"], renderer_type="table"),
            ],
            default_view="rule_explanation",
        ),
    ]


def overlay_processing_units() -> list[ProcessingUnitDefinition]:
    root_id = "overlay"
    return [
        _unit(
            id=root_id,
            label="Overlay rendering",
            kind="stage",
            parent_id=None,
            stage_id="overlay",
            category="overlay",
            order=700,
            description="Human-facing height, measurement, and classification overlays.",
            outputs=[ProcessingUnitOutput(id="classification_overlay", label="Classification overlay", artifact_id="classification_overlay", kind="overlay")],
            artifacts=[
                _artifact(root_id, "height_overlay", "Height overlay", kind="overlay", role="diagnostic", renderer="overlay"),
                _artifact(root_id, "measurement_overlay", "Measurement overlay", kind="overlay", role="diagnostic", renderer="overlay"),
                _artifact(root_id, "classification_overlay", "Classification overlay", kind="overlay", role="final", renderer="overlay"),
                _artifact(root_id, "classification_overlay_metadata", "Classification overlay metadata", kind="json", role="diagnostic", renderer="json"),
            ],
            diagnostics=["classification_overlay_metadata"],
            views=[
                _view("overlays_25d", "Overlays", ["classification_overlay", "measurement_overlay", "height_overlay"], renderer_type="overlay"),
                _view("json", "JSON", ["classification_overlay_metadata"], renderer_type="json"),
            ],
            default_view="overlays_25d",
        ),
        _unit(
            id="overlay.height_overlay",
            label="Height overlay",
            kind="substage",
            parent_id=root_id,
            stage_id="overlay",
            category="overlay",
            order=705,
            description="Height-only human-facing overlay preview over the normalized field.",
            outputs=[ProcessingUnitOutput(id="height_overlay", label="Height overlay", artifact_id="height_overlay", kind="overlay")],
            artifacts=[_artifact("overlay.height_overlay", "height_overlay", "Height overlay", kind="overlay", role="diagnostic", renderer="overlay")],
            views=[_view("overlays_25d", "Overlays", ["height_overlay"], renderer_type="overlay")],
            default_view="overlays_25d",
        ),
        _unit(
            id="overlay.measurement_overlay",
            label="Measurement overlay",
            kind="substage",
            parent_id=root_id,
            stage_id="overlay",
            category="overlay",
            order=710,
            description="Measurement-centric human-facing overlay.",
            outputs=[ProcessingUnitOutput(id="measurement_overlay", label="Measurement overlay", artifact_id="measurement_overlay", kind="overlay")],
            artifacts=[_artifact("overlay.measurement_overlay", "measurement_overlay", "Measurement overlay", kind="overlay", role="diagnostic", renderer="overlay")],
            views=[_view("overlays_25d", "Overlays", ["measurement_overlay"], renderer_type="overlay")],
            default_view="overlays_25d",
        ),
        _unit(
            id="overlay.classification_overlay",
            label="Classification overlay",
            kind="substage",
            parent_id=root_id,
            stage_id="overlay",
            category="overlay",
            order=720,
            description="Final operator-facing classification overlay.",
            outputs=[ProcessingUnitOutput(id="classification_overlay", label="Classification overlay", artifact_id="classification_overlay", kind="overlay")],
            artifacts=[_artifact("overlay.classification_overlay", "classification_overlay", "Classification overlay", kind="overlay", role="final", renderer="overlay")],
            views=[_view("overlays_25d", "Overlays", ["classification_overlay"], renderer_type="overlay")],
            default_view="overlays_25d",
        ),
        _unit(
            id="overlay.summary_overlay_metadata",
            label="Summary overlay metadata",
            kind="substage",
            parent_id=root_id,
            stage_id="overlay",
            category="overlay",
            order=730,
            description="Overlay metadata and human-facing summary context.",
            outputs=[ProcessingUnitOutput(id="overlay_metadata", label="Overlay metadata", artifact_id="classification_overlay_metadata", kind="json")],
            artifacts=[_artifact("overlay.summary_overlay_metadata", "classification_overlay_metadata", "Classification overlay metadata", kind="json", role="diagnostic", renderer="json")],
            views=[_view("json", "JSON", ["classification_overlay_metadata"], renderer_type="json")],
            default_view="json",
        ),
    ]


def processing_units_for_pipeline(
    pipeline_id: str,
    *,
    stage_parameter_schemas: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if pipeline_id != "mining_steel_ball_classification_25d" or not isinstance(stage_parameter_schemas, Mapping):
        return []
    detect_schema = stage_parameter_schemas.get("detect_belt_plane")
    segmentation_schema = stage_parameter_schemas.get("remove_belt_segment_objects")
    diagnostics_schema = stage_parameter_schemas.get("measurement_diagnostics")
    if not isinstance(detect_schema, Mapping) or not isinstance(segmentation_schema, Mapping) or not isinstance(diagnostics_schema, Mapping):
        return []
    units = [
        *input_processing_units(),
        *detect_reference_processing_units(detect_schema),
        *normalize_processing_units(),
        *segmentation_processing_units(segmentation_schema),
        *geometry_processing_units(),
        *measurement_processing_units(),
        *diagnostics_processing_units(diagnostics_schema),
        *classification_processing_units(),
        *overlay_processing_units(),
    ]
    return [asdict(unit) for unit in units]


def processing_unit_contract_fingerprint(units: list[dict[str, Any]]) -> str:
    payload = json.dumps(units, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stage_params_key_for_unit_stage(stage_id: str) -> str:
    if stage_id == "measurement_diagnostics":
        return "known_object_25d"
    if stage_id == "classification":
        return "classify_25d"
    return stage_id


def recipe_parameters_by_unit(
    units: list[dict[str, Any]],
    stage_params: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    if not isinstance(stage_params, Mapping):
        return grouped
    for unit in units:
        if not isinstance(unit, Mapping):
            continue
        unit_id = str(unit.get("id") or "")
        stage_id = str(unit.get("stage_id") or "")
        if not unit_id or not stage_id:
            continue
        unit_stage_params = stage_params.get(stage_params_key_for_unit_stage(stage_id))
        if not isinstance(unit_stage_params, Mapping):
            continue
        params = [item for item in unit.get("parameters") or [] if isinstance(item, Mapping)]
        param_values: dict[str, Any] = {}
        for param in params:
            param_id = str(param.get("id") or "")
            if not param_id:
                continue
            if param_id in unit_stage_params:
                param_values[param_id] = unit_stage_params[param_id]
        if param_values:
            grouped[unit_id] = param_values
    return grouped


def stage_params_from_recipe_parameters(
    units: list[dict[str, Any]],
    parameters_by_unit: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    if not isinstance(parameters_by_unit, Mapping):
        return merged
    unit_by_id = {
        str(unit.get("id") or ""): unit
        for unit in units
        if isinstance(unit, Mapping) and unit.get("id")
    }
    for unit_id, values in parameters_by_unit.items():
        unit = unit_by_id.get(str(unit_id))
        if unit is None or not isinstance(values, Mapping):
            continue
        stage_key = stage_params_key_for_unit_stage(str(unit.get("stage_id") or ""))
        if not stage_key:
            continue
        bucket = merged.setdefault(stage_key, {})
        for key, value in values.items():
            bucket[str(key)] = value
    return merged


def validate_processing_unit_contracts(units: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    unit_ids: list[str] = []
    unit_by_id: dict[str, dict[str, Any]] = {}
    artifact_owner: dict[str, str] = {}

    for unit in units:
        if not isinstance(unit, Mapping):
            errors.append("Processing unit entry must be an object.")
            continue
        unit_id = str(unit.get("id") or "")
        if not unit_id:
            errors.append("Processing unit id must not be empty.")
            continue
        unit_ids.append(unit_id)
        unit_by_id[unit_id] = dict(unit)

    if len(unit_ids) != len(set(unit_ids)):
        seen: set[str] = set()
        for unit_id in unit_ids:
            if unit_id in seen:
                errors.append(f"Duplicate processing unit id: {unit_id}")
            seen.add(unit_id)

    stage_ids = {
        str(unit.get("stage_id") or "")
        for unit in units
        if isinstance(unit, Mapping) and unit.get("stage_id")
    }
    declared_stage_roots = {
        str(unit.get("id") or "")
        for unit in units
        if isinstance(unit, Mapping) and str(unit.get("kind") or "") == "stage"
    }
    for unit in units:
        if not isinstance(unit, Mapping):
            continue
        unit_id = str(unit.get("id") or "")
        for artifact in unit.get("artifacts") or []:
            if isinstance(artifact, Mapping) and artifact.get("artifact_id"):
                artifact_owner.setdefault(str(artifact.get("artifact_id")), unit_id)

    for unit in units:
        if not isinstance(unit, Mapping):
            continue
        unit_id = str(unit.get("id") or "")
        kind = str(unit.get("kind") or "")
        parent_id = unit.get("parent_id")
        stage_id = str(unit.get("stage_id") or "")
        if parent_id is not None and str(parent_id) not in unit_by_id:
            errors.append(f"Unit {unit_id} references missing parent_id {parent_id}.")
        if not stage_id:
            errors.append(f"Unit {unit_id} is missing stage_id.")
        elif stage_id not in stage_ids:
            errors.append(f"Unit {unit_id} references unknown stage_id {stage_id}.")
        if kind == "stage" and unit_id != stage_id:
            warnings.append(f"Stage unit {unit_id} does not match its stage_id {stage_id}.")
        if kind != "stage" and stage_id not in declared_stage_roots:
            errors.append(f"Unit {unit_id} references stage_id {stage_id} that has no stage root definition.")

        param_ids: list[str] = []
        for param in unit.get("parameters") or []:
            if not isinstance(param, Mapping):
                errors.append(f"Unit {unit_id} contains a non-object parameter entry.")
                continue
            param_id = str(param.get("id") or "")
            if not param_id:
                errors.append(f"Unit {unit_id} contains a parameter with empty id.")
                continue
            param_ids.append(param_id)
            param_type = str(param.get("type") or "")
            if param_type not in SUPPORTED_PROCESSING_UNIT_PARAMETER_TYPES:
                errors.append(f"Unit {unit_id} parameter {param_id} uses unsupported type {param_type}.")
            active_when = param.get("active_when")
            if active_when is not None and not isinstance(active_when, Mapping):
                errors.append(f"Unit {unit_id} parameter {param_id} has invalid active_when; expected an object.")
        if len(param_ids) != len(set(param_ids)):
            seen_params: set[str] = set()
            for param_id in param_ids:
                if param_id in seen_params:
                    errors.append(f"Unit {unit_id} declares duplicate parameter id {param_id}.")
                seen_params.add(param_id)

        unit_artifact_ids: set[str] = set()
        for artifact in unit.get("artifacts") or []:
            if not isinstance(artifact, Mapping):
                errors.append(f"Unit {unit_id} contains a non-object artifact entry.")
                continue
            artifact_id = str(artifact.get("artifact_id") or "")
            if not artifact_id:
                errors.append(f"Unit {unit_id} declares an artifact with empty artifact_id.")
                continue
            if artifact_id in unit_artifact_ids:
                errors.append(f"Unit {unit_id} declares duplicate artifact_id {artifact_id}.")
            unit_artifact_ids.add(artifact_id)

        view_ids_list: list[str] = []
        for view in unit.get("views") or []:
            if not isinstance(view, Mapping):
                errors.append(f"Unit {unit_id} contains a non-object view entry.")
                continue
            view_id = str(view.get("id") or "")
            if view_id:
                view_ids_list.append(view_id)
            renderer_type = str(view.get("renderer_type") or "image")
            if renderer_type not in SUPPORTED_VIEW_RENDERER_TYPES:
                errors.append(
                    f"Unit {unit_id} view {view.get('id')} uses unsupported renderer_type {renderer_type}."
                )
            for artifact_id in view.get("artifact_ids") or []:
                artifact_key = str(artifact_id)
                if artifact_key not in unit_artifact_ids:
                    if artifact_key not in artifact_owner:
                        warnings.append(
                            f"Unit {unit_id} view {view.get('id')} references runtime or undeclared artifact_id {artifact_key}."
                        )
                    else:
                        warnings.append(
                            f"Unit {unit_id} view {view.get('id')} references cross-unit artifact_id {artifact_key}."
                        )
        if len(view_ids_list) != len(set(view_ids_list)):
            seen_views: set[str] = set()
            for view_id in view_ids_list:
                if view_id in seen_views:
                    errors.append(f"Unit {unit_id} declares duplicate view id {view_id}.")
                seen_views.add(view_id)
        view_ids = set(view_ids_list)
        default_view = unit.get("default_view")
        if default_view is not None and str(default_view) not in view_ids:
            errors.append(f"Unit {unit_id} declares default_view {default_view} that does not exist in its views.")

        for artifact in unit.get("artifacts") or []:
            if not isinstance(artifact, Mapping):
                continue
            artifact_id = str(artifact.get("artifact_id") or "")
            renderer = str(artifact.get("renderer") or "image")
            if renderer not in SUPPORTED_ARTIFACT_RENDERERS:
                errors.append(f"Unit {unit_id} artifact {artifact_id} uses unsupported renderer {renderer}.")
            diff_type = artifact.get("diff_type")
            if diff_type is not None and str(diff_type) not in SUPPORTED_ARTIFACT_DIFF_TYPES:
                errors.append(f"Unit {unit_id} artifact {artifact_id} uses unsupported diff_type {diff_type}.")
            if bool(artifact.get("diffable")):
                if not diff_type or str(diff_type) not in SUPPORTED_ARTIFACT_DIFF_TYPES:
                    errors.append(f"Unit {unit_id} artifact {artifact_id} is diffable but missing a valid diff_type.")

    sibling_orders: dict[tuple[str | None, int], list[str]] = {}
    for unit in units:
        if not isinstance(unit, Mapping):
            continue
        if str(unit.get("kind") or "") != "substage":
            continue
        unit_id = str(unit.get("id") or "")
        parent_key = str(unit.get("parent_id")) if unit.get("parent_id") is not None else None
        order = int(unit.get("order") or 0)
        sibling_orders.setdefault((parent_key, order), []).append(unit_id)
    for (parent_key, order), siblings in sibling_orders.items():
        if len(siblings) > 1:
            parent_label = parent_key or "root"
            errors.append(
                f"Unit order conflict under parent {parent_label}: order {order} used by {', '.join(siblings)}."
            )

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
    }


def build_recipe_snapshot(
    *,
    pipeline_id: str,
    pipeline_version: str | None,
    registry_version: str,
    contract_fingerprint: str | None,
    units: list[dict[str, Any]] | None,
    stage_params: Mapping[str, Any] | None,
    recipe_version_id: str | None,
    recipe_name: str | None = None,
    recipe_version: int | None = None,
    enabled_units: list[str],
    provenance: Mapping[str, Any] | None = None,
    artifact_policy: Mapping[str, Any] | None = None,
    calibration_snapshot_reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    detect_params = dict(stage_params.get("detect_belt_plane") or {}) if isinstance(stage_params, Mapping) else {}
    public_stage_ids = [
        "input",
        "detect_belt_plane",
        "normalize_heights_to_plane",
        "remove_belt_segment_objects",
        "geometry",
        "measurement",
        "measurement_diagnostics",
        "classification",
        "overlay",
    ]
    parameters_by_unit = recipe_parameters_by_unit(units or [], stage_params)
    return {
        "recipe_id": recipe_version_id or None,
        "recipe_name": recipe_name or recipe_version_id,
        "version": recipe_version,
        "pipeline_id": pipeline_id,
        "pipeline_version": pipeline_version,
        "registry_version": registry_version,
        "processing_unit_contract_version": registry_version,
        "processing_unit_contract_fingerprint": contract_fingerprint,
        "parameter_values": dict(stage_params or {}),
        "parameters_by_unit": parameters_by_unit,
        "stage_params": dict(stage_params or {}),
        "stage_enabled_state": {stage_id: True for stage_id in public_stage_ids},
        "unit_enabled_state": {unit_id: True for unit_id in enabled_units},
        "strategy_branch": str(detect_params.get("background_detection_strategy") or "low_gradient_surface"),
        "artifact_policy": dict(artifact_policy or {"diagnostic_level": "full"}),
        "diagnostic_level": str((artifact_policy or {}).get("diagnostic_level") or "full"),
        "calibration_snapshot_reference": dict(calibration_snapshot_reference or {}),
        "provenance": dict(provenance or {}),
    }


def summarize_processing_unit_trace_coverage(unit_results: Mapping[str, Any]) -> dict[str, Any]:
    entries = [dict(entry) for entry in unit_results.values() if isinstance(entry, Mapping)]
    total_units = len(entries)
    runtime_traced_units = sum(
        1 for entry in entries if str(entry.get("trace_source") or "") == "runtime_unit_callbacks"
    )
    inferred_units = sum(
        1 for entry in entries if str(entry.get("trace_source") or "") == "best_effort_artifact_registry"
    )
    failed_units = sum(
        1 for entry in entries if str(entry.get("status") or "") in {"failed", "error"}
    )
    warning_units = sum(
        1
        for entry in entries
        if str(entry.get("status") or "") == "warning" or bool(entry.get("warnings"))
    )
    trace_coverage_percent = round((runtime_traced_units / total_units) * 100.0, 1) if total_units else 0.0
    coverage_by_stage: dict[str, dict[str, Any]] = {}
    for entry in entries:
        stage_id = str(entry.get("stage_id") or "unknown")
        bucket = coverage_by_stage.setdefault(
            stage_id,
            {
                "total_units": 0,
                "runtime_traced_units": 0,
                "inferred_units": 0,
                "failed_units": 0,
                "warning_units": 0,
                "trace_coverage_percent": 0.0,
            },
        )
        bucket["total_units"] += 1
        if str(entry.get("trace_source") or "") == "runtime_unit_callbacks":
            bucket["runtime_traced_units"] += 1
        if str(entry.get("trace_source") or "") == "best_effort_artifact_registry":
            bucket["inferred_units"] += 1
        if str(entry.get("status") or "") in {"failed", "error"}:
            bucket["failed_units"] += 1
        if str(entry.get("status") or "") == "warning" or bool(entry.get("warnings")):
            bucket["warning_units"] += 1
    for bucket in coverage_by_stage.values():
        total = int(bucket["total_units"] or 0)
        runtime_count = int(bucket["runtime_traced_units"] or 0)
        bucket["trace_coverage_percent"] = round((runtime_count / total) * 100.0, 1) if total else 0.0
    return {
        "total_units": total_units,
        "runtime_traced_units": runtime_traced_units,
        "inferred_units": inferred_units,
        "failed_units": failed_units,
        "warning_units": warning_units,
        "trace_coverage_percent": trace_coverage_percent,
        "coverage_by_stage": coverage_by_stage,
    }


def build_processing_unit_trace(
    *,
    pipeline_id: str,
    units: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    stage_params: Mapping[str, Any] | None,
    runtime_trace: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if pipeline_id != "mining_steel_ball_classification_25d":
        return {"pipeline_id": pipeline_id, "registry_version": PROCESSING_UNIT_REGISTRY_VERSION, "unit_results": {}}
    artifact_ids = {str(item.get("artifact_id")) for item in artifacts if item.get("artifact_id")}
    stage_param_values = dict(stage_params or {}) if isinstance(stage_params, Mapping) else {}
    results: dict[str, Any] = {}
    runtime_units_raw = {}
    if isinstance(runtime_trace, Mapping):
        raw_units = runtime_trace.get("unit_results")
        if not isinstance(raw_units, Mapping):
            raw_units = runtime_trace.get("units")
        if isinstance(raw_units, Mapping):
            runtime_units_raw = {
                str(unit_id): dict(payload)
                for unit_id, payload in raw_units.items()
                if isinstance(payload, Mapping)
            }
    for unit in units:
        unit_id = str(unit.get("id") or "")
        stage_id = str(unit.get("stage_id") or "")
        parent_id = str(unit.get("parent_id") or "") or None
        params = [item for item in unit.get("parameters") or [] if isinstance(item, dict)]
        artifacts_for_unit = [item for item in unit.get("artifacts") or [] if isinstance(item, dict)]
        stage_key = stage_params_key_for_unit_stage(stage_id)
        unit_stage_params = dict(stage_param_values.get(stage_key) or {}) if isinstance(stage_param_values.get(stage_key), Mapping) else {}
        outputs = [str(item.get("artifact_id")) for item in unit.get("outputs") or [] if isinstance(item, dict) and item.get("artifact_id")]
        output_artifacts = [artifact_id for artifact_id in outputs if artifact_id in artifact_ids]
        if not output_artifacts:
            output_artifacts = [
                str(item.get("artifact_id"))
                for item in artifacts_for_unit
                if item.get("artifact_id") and str(item.get("artifact_id")) in artifact_ids and str(item.get("role") or "") != "diagnostic"
            ]
        input_artifacts = [
            str(item.get("artifact_id"))
            for item in unit.get("inputs") or []
            if isinstance(item, dict) and item.get("artifact_id") and str(item.get("artifact_id")) in artifact_ids
        ]
        diagnostic_artifacts = [
            artifact_id
            for artifact_id in [str(item) for item in unit.get("diagnostics") or []]
            if artifact_id in artifact_ids
        ]
        matched_artifacts = [
            str(item.get("artifact_id"))
            for item in artifacts_for_unit
            if item.get("artifact_id") and str(item.get("artifact_id")) in artifact_ids
        ]
        status = "inferred" if output_artifacts or diagnostic_artifacts or matched_artifacts or str(unit.get("kind") or "") == "stage" else "not_emitted"
        inferred_entry = {
            "unit_id": unit_id,
            "stage_id": stage_id,
            "parent_id": parent_id,
            "status": status,
            "parameters_used": {param["id"]: unit_stage_params.get(param["id"], param.get("default")) for param in params if param.get("id")},
            "input_artifacts": input_artifacts,
            "output_artifacts": output_artifacts,
            "metrics": {},
            "diagnostics": {"artifact_ids": diagnostic_artifacts},
            "warnings": [],
            "errors": [],
            "trace_source": "best_effort_artifact_registry",
            "trace_precision": "artifact_level",
        }
        runtime_entry = runtime_units_raw.get(unit_id)
        results[unit_id] = dict(inferred_entry)
        if runtime_entry:
            merged = dict(inferred_entry)
            merged.update(runtime_entry)
            merged["unit_id"] = unit_id
            merged["stage_id"] = str(merged.get("stage_id") or stage_id)
            merged["parent_id"] = merged.get("parent_id") if merged.get("parent_id") is not None else parent_id
            merged["parameters_used"] = dict(inferred_entry["parameters_used"]) | dict(merged.get("parameters_used") or {})
            merged["input_artifacts"] = list(dict.fromkeys([*inferred_entry["input_artifacts"], *list(merged.get("input_artifacts") or [])]))
            merged["output_artifacts"] = list(dict.fromkeys([*inferred_entry["output_artifacts"], *list(merged.get("output_artifacts") or [])]))
            merged["diagnostics"] = dict(inferred_entry["diagnostics"]) | dict(merged.get("diagnostics") or {})
            merged["warnings"] = list(merged.get("warnings") or [])
            merged["errors"] = list(merged.get("errors") or [])
            merged["metrics"] = dict(merged.get("metrics") or {})
            merged["trace_source"] = str(merged.get("trace_source") or "runtime_unit_callbacks")
            merged["trace_precision"] = str(merged.get("trace_precision") or "unit_level")
            results[unit_id] = merged
    top_source = "best_effort_artifact_registry"
    top_precision = "artifact_level"
    if runtime_units_raw:
        runtime_count = sum(1 for entry in results.values() if str(entry.get("trace_source") or "") == "runtime_unit_callbacks")
        if runtime_count == len(results):
            top_source = "runtime_unit_callbacks"
            top_precision = "unit_level"
        else:
            top_source = "mixed"
            top_precision = "mixed"
    return {
        "pipeline_id": pipeline_id,
        "registry_version": PROCESSING_UNIT_REGISTRY_VERSION,
        "trace_source": top_source,
        "trace_precision": top_precision,
        "trace_summary": summarize_processing_unit_trace_coverage(results),
        "unit_results": results,
        "units": results,
    }
