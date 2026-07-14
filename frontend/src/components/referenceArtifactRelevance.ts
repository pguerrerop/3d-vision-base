import type { TakeDetail } from "../api/client";
import type { StudioArtifact } from "./studioWorkspaceModel";

export type ReferenceStrategy =
  | "low_gradient_depth_plateaus"
  | "low_gradient_bg_and_stripes"
  | "low_gradient_blob_height_clusters"
  | "low_gradient_surface"
  | "percentile_or_legacy"
  | "unknown";

export type BlobComponentMode =
  | "xy_only"
  | "height_aware"
  | "unknown";

export type BlobSplitMethod =
  | "disabled"
  | "histogram_gap"
  | "height_borders"
  | "height_borders_then_histogram"
  | "unknown";

export type SupportSelectionMethod =
  | "low_gradient_depth_plateaus"
  | "low_gradient_bg_and_stripes"
  | "low_gradient_surface"
  | "blob_height_clusters"
  | "percentile_or_legacy"
  | "unknown";

export type ReferenceSurfaceModel =
  | "auto"
  | "plane"
  | "constant_z"
  | "unknown";

export type SuppressionMaskPolicy =
  | "auto"
  | "selected_support"
  | "expanded_support"
  | "final_model_inliers"
  | "unknown";

export type ReferenceMethodPresetId =
  | "current_default_plateau_plane"
  | "blob_xy_height_borders"
  | "blob_height_aware_auto"
  | "blob_height_aware_constant_z"
  | "blob_height_aware_strict_plane_qa";

export type ArtifactRelevance =
  | "active"
  | "debug"
  | "inactive_strategy"
  | "legacy"
  | "fallback"
  | "stale"
  | "missing";

export type ArtifactGroup = "main" | "debug" | "stripe" | "legacy" | "all";

export type ArtifactRelevanceTone = "positive" | "neutral" | "warning" | "negative";

export function artifactRelevanceLabel(relevance: ArtifactRelevance): string {
  if (relevance === "active") return "Active";
  if (relevance === "fallback") return "Fallback";
  if (relevance === "debug") return "Used";
  if (relevance === "inactive_strategy") return "Unused";
  if (relevance === "legacy") return "Legacy";
  if (relevance === "stale") return "Stale";
  return "Missing";
}

export function artifactRelevanceTone(relevance: ArtifactRelevance): ArtifactRelevanceTone {
  if (relevance === "active" || relevance === "fallback") return "positive";
  if (relevance === "stale") return "warning";
  if (relevance === "missing") return "negative";
  return "neutral";
}

export type DetectReferenceRunState = {
  strategy: ReferenceStrategy;
  supportSelectionMethod: SupportSelectionMethod;
  componentMode: BlobComponentMode;
  splitMethod: BlobSplitMethod;
  referenceSurfaceModel: ReferenceSurfaceModel;
  resolvedReferenceSurfaceModel: ReferenceSurfaceModel;
  suppressionMaskPolicy: SuppressionMaskPolicy;
  presetId: ReferenceMethodPresetId | null;
  fallbackUsed: boolean;
  fallbackStrategy: string | null;
  processedAt: string | null;
};

export type DetectViewArtifactResolution = {
  primary: StudioArtifact | null;
  reference: StudioArtifact | null;
  category: ArtifactRelevance;
  reason: string;
};

type ResolveArgs = {
  viewId: string;
  artifacts: StudioArtifact[];
  activeStrategy: ReferenceStrategy;
  componentMode: BlobComponentMode;
  splitMethod: BlobSplitMethod;
  fallbackUsed?: boolean;
  fallbackStrategy?: string | null;
  runUpdatedAt?: string | null;
};

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" ? value as Record<string, unknown> : {};
}

export function normalizeReferenceStrategy(raw: unknown): ReferenceStrategy {
  const value = String(raw ?? "").trim().toLowerCase();
  if (value === "low_gradient_depth_plateaus") return "low_gradient_depth_plateaus";
  if (value === "low_gradient_bg_and_stripes") return "low_gradient_bg_and_stripes";
  if (value === "low_gradient_blob_height_clusters") return "low_gradient_blob_height_clusters";
  if (value === "low_gradient_surface") return "low_gradient_surface";
  if (value === "nearest_percentile" || value === "farthest_percentile" || value === "automatic" || value === "depth_percentile_plane" || value === "percentile") {
    return "percentile_or_legacy";
  }
  return "unknown";
}

export function normalizeBlobComponentMode(raw: unknown): BlobComponentMode {
  const value = String(raw ?? "").trim().toLowerCase();
  if (value === "xy_only") return "xy_only";
  if (value === "height_aware") return "height_aware";
  return "unknown";
}

export function normalizeBlobSplitMethod(raw: unknown): BlobSplitMethod {
  const value = String(raw ?? "").trim().toLowerCase();
  if (value === "disabled") return "disabled";
  if (value === "histogram_gap") return "histogram_gap";
  if (value === "height_borders") return "height_borders";
  if (value === "height_borders_then_histogram") return "height_borders_then_histogram";
  return "unknown";
}

export function normalizeSupportSelectionMethod(raw: unknown): SupportSelectionMethod {
  const value = String(raw ?? "").trim().toLowerCase();
  if (value === "low_gradient_depth_plateaus") return "low_gradient_depth_plateaus";
  if (value === "low_gradient_bg_and_stripes") return "low_gradient_bg_and_stripes";
  if (value === "low_gradient_surface") return "low_gradient_surface";
  if (value === "blob_height_clusters" || value === "low_gradient_blob_height_clusters") return "blob_height_clusters";
  if (value === "nearest_percentile" || value === "farthest_percentile" || value === "automatic" || value === "depth_percentile_plane" || value === "percentile" || value === "percentile_or_legacy") return "percentile_or_legacy";
  return "unknown";
}

export function normalizeReferenceSurfaceModel(raw: unknown): ReferenceSurfaceModel {
  const value = String(raw ?? "").trim().toLowerCase();
  if (value === "auto") return "auto";
  if (value === "plane") return "plane";
  if (value === "constant_z") return "constant_z";
  return "unknown";
}

export function normalizeSuppressionMaskPolicy(raw: unknown): SuppressionMaskPolicy {
  const value = String(raw ?? "").trim().toLowerCase();
  if (value === "auto") return "auto";
  if (value === "selected_support") return "selected_support";
  if (value === "expanded_support") return "expanded_support";
  if (value === "final_model_inliers") return "final_model_inliers";
  return "unknown";
}

export function referencePresetLabel(presetId: ReferenceMethodPresetId | null): string | null {
  if (presetId === "current_default_plateau_plane") return "Current default: low-gradient plateaus + plane";
  if (presetId === "blob_xy_height_borders") return "Blob clusters: XY components + height borders";
  if (presetId === "blob_height_aware_auto") return "Blob clusters: height-aware components";
  if (presetId === "blob_height_aware_constant_z") return "Blob clusters: height-aware + constant Z";
  if (presetId === "blob_height_aware_strict_plane_qa") return "Blob clusters: height-aware + strict plane QA";
  return null;
}

export function supportSelectionMethodLabel(value: SupportSelectionMethod): string {
  if (value === "low_gradient_depth_plateaus") return "Low-gradient depth plateaus";
  if (value === "low_gradient_bg_and_stripes") return "Low-gradient background + stripes";
  if (value === "low_gradient_surface") return "Low-gradient surface";
  if (value === "blob_height_clusters") return "Blob height clusters";
  if (value === "percentile_or_legacy") return "Percentile / legacy";
  return "Unknown";
}

export function blobComponentModeLabel(value: BlobComponentMode): string {
  if (value === "xy_only") return "XY components";
  if (value === "height_aware") return "Height-aware components";
  return "Unknown";
}

export function blobSplitMethodLabel(value: BlobSplitMethod): string {
  if (value === "disabled") return "Disabled";
  if (value === "height_borders") return "Height borders";
  if (value === "histogram_gap") return "Histogram gap";
  if (value === "height_borders_then_histogram") return "Height borders then histogram";
  return "Unknown";
}

export function referenceSurfaceModelLabel(value: ReferenceSurfaceModel): string {
  if (value === "auto") return "Auto";
  if (value === "plane") return "Plane";
  if (value === "constant_z") return "Constant Z";
  return "Unknown";
}

export function suppressionMaskPolicyLabel(value: SuppressionMaskPolicy): string {
  if (value === "auto") return "Auto";
  if (value === "selected_support") return "Selected support mask";
  if (value === "expanded_support") return "Expanded/model support mask";
  if (value === "final_model_inliers") return "Final model inliers";
  return "Unknown";
}

export type ReferenceMethodPreset = {
  id: ReferenceMethodPresetId;
  label: string;
  values: Record<string, unknown>;
};

export const REFERENCE_METHOD_PRESETS: ReferenceMethodPreset[] = [
  {
    id: "current_default_plateau_plane",
    label: "Current default: low-gradient plateaus + plane",
    values: {
      background_detection_strategy: "low_gradient_depth_plateaus",
      reference_surface_model: "auto",
      reference_suppression_mask_policy: "auto",
    },
  },
  {
    id: "blob_xy_height_borders",
    label: "Blob clusters: XY components + height borders",
    values: {
      background_detection_strategy: "low_gradient_blob_height_clusters",
      blob_component_mode: "xy_only",
      blob_split_method: "height_borders",
      reference_surface_model: "auto",
      reference_suppression_mask_policy: "auto",
    },
  },
  {
    id: "blob_height_aware_auto",
    label: "Blob clusters: height-aware components",
    values: {
      background_detection_strategy: "low_gradient_blob_height_clusters",
      blob_component_mode: "height_aware",
      blob_split_method: "disabled",
      blob_component_use_smoothed_z: false,
      blob_neighbor_z_tolerance_mm: 2.0,
      reference_surface_model: "auto",
      reference_suppression_mask_policy: "auto",
    },
  },
  {
    id: "blob_height_aware_constant_z",
    label: "Blob clusters: height-aware + constant Z",
    values: {
      background_detection_strategy: "low_gradient_blob_height_clusters",
      blob_component_mode: "height_aware",
      blob_split_method: "disabled",
      blob_component_use_smoothed_z: false,
      blob_neighbor_z_tolerance_mm: 2.0,
      reference_surface_model: "constant_z",
      reference_suppression_mask_policy: "selected_support",
    },
  },
  {
    id: "blob_height_aware_strict_plane_qa",
    label: "Blob clusters: height-aware + strict plane QA",
    values: {
      background_detection_strategy: "low_gradient_blob_height_clusters",
      blob_component_mode: "height_aware",
      blob_split_method: "disabled",
      reference_surface_model: "plane",
      reference_suppression_mask_policy: "final_model_inliers",
    },
  },
];

export function detectReferencePresetById(id: string | null | undefined): ReferenceMethodPreset | null {
  return REFERENCE_METHOD_PRESETS.find((preset) => preset.id === id) ?? null;
}

export function detectReferencePresetForValues(values: Record<string, unknown> | null | undefined): ReferenceMethodPreset | null {
  if (!values) return null;
  for (const preset of REFERENCE_METHOD_PRESETS) {
    const matches = Object.entries(preset.values).every(([key, expected]) => values[key] === expected);
    if (matches) return preset;
  }
  return null;
}

function findArtifact(artifacts: StudioArtifact[], ids: string[]): StudioArtifact | null {
  for (const id of ids) {
    const found = artifacts.find((artifact) => artifact.artifact_id === id);
    if (found) return found;
  }
  return null;
}

function firstImageArtifactByIds(artifacts: StudioArtifact[], ids: string[]): StudioArtifact | null {
  for (const id of ids) {
    const found = artifacts.find((artifact) => artifact.kind === "image" && artifact.artifact_id === id);
    if (found) return found;
  }
  return null;
}

export function resolveDetectReferenceRunState(
  detail: TakeDetail | null,
  artifacts: StudioArtifact[],
  formState?: Record<string, unknown> | null,
): DetectReferenceRunState {
  const planeDebugMeta = asRecord(findArtifact(artifacts, ["plane_fit_debug"])?.metadata);
  const backgroundSelection = asRecord(planeDebugMeta.background_selection);
  const referenceMethod = asRecord(
    planeDebugMeta.reference_method
    ?? backgroundSelection.reference_method
    ?? asRecord(findArtifact(artifacts, ["selected_surface_debug"])?.metadata).reference_method,
  );
  const gradientDebug = asRecord(backgroundSelection.gradient_debug);
  const blobDebug = asRecord(findArtifact(artifacts, ["blob_cluster_selection_debug"])?.metadata);
  const connectivityDebug = asRecord(findArtifact(artifacts, ["height_aware_connectivity_debug"])?.metadata);
  const heightSplitDebug = asRecord(findArtifact(artifacts, ["height_split_debug"])?.metadata);
  const heightBorderDebug = asRecord(findArtifact(artifacts, ["height_border_split_debug"])?.metadata);
  const params = asRecord(asRecord(detail?.result).stage_params);
  const detectParams = asRecord(params.detect_belt_plane);

  const strategy = normalizeReferenceStrategy(
    referenceMethod.background_detection_strategy
    ?? referenceMethod.support_selection_method
    ?? gradientDebug.strategy
    ?? backgroundSelection.background_detection_strategy
    ?? planeDebugMeta.background_detection_strategy
    ?? blobDebug.strategy
    ?? detectParams.background_detection_strategy
    ?? formState?.background_detection_strategy,
  );
  const supportSelectionMethod = normalizeSupportSelectionMethod(
    referenceMethod.support_selection_method
    ?? referenceMethod.background_detection_strategy
    ?? detectParams.support_selection_method
    ?? detectParams.background_detection_strategy
    ?? formState?.support_selection_method
    ?? formState?.background_detection_strategy,
  );
  const componentMode = normalizeBlobComponentMode(
    referenceMethod.blob_component_mode
    ?? referenceMethod.component_mode
    ?? gradientDebug.component_mode
    ?? connectivityDebug.component_mode
    ?? blobDebug.component_mode
    ?? detectParams.blob_component_mode
    ?? formState?.blob_component_mode,
  );
  const splitMethod = normalizeBlobSplitMethod(
    referenceMethod.blob_split_method
    ?? gradientDebug.split_method
    ?? heightSplitDebug.method
    ?? heightBorderDebug.method
    ?? detectParams.blob_split_method
    ?? formState?.blob_split_method,
  );
  return {
    strategy,
    supportSelectionMethod,
    componentMode,
    splitMethod,
    referenceSurfaceModel: normalizeReferenceSurfaceModel(
      referenceMethod.reference_surface_model
      ?? detectParams.reference_surface_model
      ?? formState?.reference_surface_model,
    ),
    resolvedReferenceSurfaceModel: normalizeReferenceSurfaceModel(
      referenceMethod.resolved_reference_surface_model
      ?? planeDebugMeta.reference_surface_model_type
      ?? asRecord(findArtifact(artifacts, ["selected_surface_debug"])?.metadata).reference_surface_model_type
      ?? detectParams.reference_surface_model
      ?? formState?.reference_surface_model,
    ),
    suppressionMaskPolicy: normalizeSuppressionMaskPolicy(
      referenceMethod.reference_suppression_mask_policy
      ?? detectParams.reference_suppression_mask_policy
      ?? formState?.reference_suppression_mask_policy,
    ),
    presetId: referenceMethod.preset_id == null ? null : String(referenceMethod.preset_id) as ReferenceMethodPresetId,
    fallbackUsed: Boolean(referenceMethod.fallback_used ?? gradientDebug.fallback_used ?? blobDebug.fallback_used ?? false),
    fallbackStrategy: referenceMethod.fallback_strategy == null
      ? (gradientDebug.fallback_strategy == null ? (blobDebug.fallback_strategy == null ? null : String(blobDebug.fallback_strategy)) : String(gradientDebug.fallback_strategy))
      : String(referenceMethod.fallback_strategy),
    processedAt: detail?.result?.processed_at ?? null,
  };
}

export function detectReferenceFormDiffersFromRun(state: DetectReferenceRunState, formState?: Record<string, unknown> | null): boolean {
  if (!formState) return false;
  const formStrategy = normalizeReferenceStrategy(formState.background_detection_strategy);
  const formComponentMode = normalizeBlobComponentMode(formState.blob_component_mode);
  const formSplitMethod = normalizeBlobSplitMethod(formState.blob_split_method);
  const formReferenceSurfaceModel = normalizeReferenceSurfaceModel(formState.reference_surface_model);
  const formSuppressionMaskPolicy = normalizeSuppressionMaskPolicy(formState.reference_suppression_mask_policy);
  return (
    (formStrategy !== "unknown" && formStrategy !== state.strategy)
    || (formComponentMode !== "unknown" && formComponentMode !== state.componentMode)
    || (formSplitMethod !== "unknown" && formSplitMethod !== state.splitMethod)
    || (formReferenceSurfaceModel !== "unknown" && formReferenceSurfaceModel !== state.referenceSurfaceModel)
    || (formSuppressionMaskPolicy !== "unknown" && formSuppressionMaskPolicy !== state.suppressionMaskPolicy)
  );
}

export function preferredDetectArtifactsForView(
  viewId: string,
  artifacts: StudioArtifact[],
  state: DetectReferenceRunState,
): StudioArtifact[] {
  const byIds = (ids: string[]) => ids.map((id) => artifacts.find((artifact) => artifact.artifact_id === id)).filter((artifact): artifact is StudioArtifact => Boolean(artifact));
  if (viewId === "blob_components") {
    return state.componentMode === "height_aware"
      ? byIds(["height_aware_blob_components_overlay", "height_aware_blob_id_mask"])
      : byIds(["low_gradient_blob_components_overlay", "low_gradient_blob_id_mask"]);
  }
  if (viewId === "height_aware_blob_components") return byIds(["low_gradient_blob_components_overlay", "low_gradient_blob_id_mask"]);
  if (viewId === "height_aware_connectivity_rejected_edges") return byIds(["height_aware_connectivity_rejected_edges", "height_aware_connectivity_debug"]);
  if (viewId === "height_borders") return byIds(["height_border_strength", "height_border_cut_mask", "height_border_split_debug"]);
  if (viewId === "height_border_fragments") return byIds(["height_border_fragments_overlay", "height_border_fragments_mask", "height_border_fragments_id_mask", "fragment_merge_debug"]);
  if (viewId === "height_split_blobs") return byIds(["height_split_blob_fragments_overlay", "height_split_blob_fragments_mask", "height_split_blob_id_mask", "height_split_debug", "height_consistent_blob_summary"]);
  if (viewId === "blob_clusters") return byIds(["blob_cluster_score_table", "blob_height_clusters", "blob_cluster_selection_debug"]);
  if (viewId === "blob_cluster_scores") return byIds(["blob_cluster_score_table", "blob_cluster_selection_debug", "blob_height_clusters"]);
  if (viewId === "selection_bridge") return byIds(["blob_height_clusters", "blob_cluster_selection_debug", "blob_cluster_score_table"]);
  if (viewId === "selected_support_lineage") return byIds(["selected_support_lineage"]);
  if (viewId === "selected_blob_cluster") return byIds(["selected_blob_cluster_overlay", "selected_blob_cluster_mask", "selected_blob_cluster_pre_refine_mask"]);
  if (viewId === "selected_cluster_pre_refine") return byIds(["selected_blob_cluster_pre_refine_mask", "selected_blob_cluster_mask"]);
  if (viewId === "selected_cluster_refined") return byIds(["selected_blob_cluster_refined_mask", "selected_reference_support_mask", "selected_blob_cluster_pre_refine_mask"]);
  if (viewId === "removed_by_refinement") return byIds(["support_removed_by_candidate_refinement"]);
  if (viewId === "rejected_blob_clusters") return byIds(["rejected_blob_clusters_overlay", "rejected_blob_clusters_mask"]);
  if (viewId === "plateau_plot") return byIds(["background_plateau_plot"]);
  if (viewId === "filtered_depth_plot") return byIds(["flat_candidate_depth_plot"]);
  if (viewId === "surface_candidates") return byIds(["reference_surface_candidates", "low_gradient_components", "reference_surface_plateaus"]);
  if (viewId === "belt_bg") return byIds(["belt_bg_mask"]);
  if (viewId === "belt_base") return byIds(["belt_base_mask"]);
  if (viewId === "belt_stripes") return byIds(["belt_stripes_mask"]);
  if (viewId === "unknown_low_gradient") return byIds(["unknown_low_gradient_mask"]);
  if (viewId === "surface_suppression") return byIds(["surface_suppression_mask", "reference_suppression_mask"]);
  if (viewId === "object_search_domain") return byIds(["object_search_domain_mask"]);
  if (viewId === "belt_stripes_tophat") return byIds(["belt_stripes_tophat_mask"]);
  if (viewId === "belt_stripes_shape") return byIds(["belt_stripes_shape_mask"]);
  if (viewId === "belt_above_belt") return byIds(["belt_above_belt_mask"]);
  if (viewId === "belt_wide_object") return byIds(["belt_wide_object_mask"]);
  if (viewId === "removed_by_stripe_filter") return byIds(["support_removed_by_stripe_filter"]);
  if (viewId === "belt_altitude_plot") return byIds(["belt_altitude_histogram_image", "belt_altitude_local_min"]);
  if (viewId === "belt_baseline") return byIds(["belt_baseline_local_min"]);
  if (viewId === "reference_model_support") return byIds(["reference_model_support_mask", "selected_reference_support_mask"]);
  if (viewId === "reference_suppression_mask") return byIds(["reference_suppression_mask"]);
  if (viewId === "support_loss_waterfall") return byIds(["support_loss_waterfall"]);
  return [];
}

export function resolveDetectViewArtifacts(
  viewId: string,
  artifacts: StudioArtifact[],
  state: DetectReferenceRunState,
): DetectViewArtifactResolution {
  const reference = firstImageArtifactByIds(artifacts, [
    "raw_heightmap_preview",
    "heightmap_input_preview",
    "source_heightmap_preview",
    "replay_heightmap_preview",
    "raw_heightmap",
    "heightmap",
    "input_preview",
  ]);
  const byIds = (ids: string[]) => firstImageArtifactByIds(artifacts, ids);
  const relevance = resolveDetectReferenceArtifactRelevance({
    viewId,
    artifacts,
    activeStrategy: state.strategy,
    componentMode: state.componentMode,
    splitMethod: state.splitMethod,
    fallbackUsed: state.fallbackUsed,
    fallbackStrategy: state.fallbackStrategy,
    runUpdatedAt: state.processedAt,
  });

  if (viewId === "selected_surface") {
    const primary = state.strategy === "low_gradient_blob_height_clusters"
      ? byIds([
          "selected_reference_support_mask",
          "reference_surface_selected_mask",
          "selected_surface",
          "selected_reference_surface",
          "selected_blob_cluster_refined_mask",
          "selected_blob_cluster_pre_refine_mask",
          "selected_blob_cluster_overlay",
          "selected_blob_cluster_mask",
          "expanded_plane_mask",
        ])
      : byIds([
          "selected_reference_support_mask",
          "reference_surface_selected_mask",
          "selected_surface",
          "selected_reference_surface",
          "expanded_plane_mask",
        ]);
    return {
      primary,
      reference,
      category: primary ? relevance.relevance : "missing",
      reason: primary ? relevance.reason : "No selected-surface artifact was emitted for this run.",
    };
  }
  if (viewId === "selected_support_lineage") {
    const primary = findArtifact(artifacts, ["selected_support_lineage"]);
    return {
      primary,
      reference,
      category: primary ? relevance.relevance : "missing",
      reason: primary ? relevance.reason : "No selected-support lineage artifact was emitted for this run.",
    };
  }
  if (viewId === "plane_inliers") {
    const primary = byIds(["reference_model_inlier_mask", "final_plane_inlier_mask", "plane_inlier_mask", "expanded_plane_mask"]);
    return {
      primary,
      reference,
      category: primary ? relevance.relevance : "missing",
      reason: primary ? relevance.reason : (state.resolvedReferenceSurfaceModel === "constant_z" ? "No reference-model support artifact was emitted for this run." : "No plane-inlier artifact was emitted for this run."),
    };
  }
  if (viewId === "residual_heatmap") {
    const primary = byIds(["belt_plane_residuals"]);
    return { primary, reference, category: primary ? relevance.relevance : "missing", reason: primary ? relevance.reason : "No residual heatmap artifact was emitted for this run." };
  }
  if (viewId === "raw_heightmap") {
    return { primary: reference, reference: null, category: reference ? "active" : "missing", reason: reference ? "Raw heightmap preview resolved." : "No raw heightmap preview was emitted for this run." };
  }
  if (viewId === "depth_gradient") {
    const primary = byIds(["depth_gradient_magnitude"]);
    return { primary, reference, category: primary ? "active" : "missing", reason: primary ? "Depth-gradient artifact resolved." : "No depth-gradient artifact was emitted for this run." };
  }
  if (viewId === "low_gradient_mask") {
    const primary = byIds(["low_gradient_mask"]);
    return { primary, reference, category: primary ? "active" : "missing", reason: primary ? "Low-gradient mask resolved." : "No low-gradient mask was emitted for this run." };
  }
  const primary = preferredDetectArtifactsForView(viewId, artifacts, state)[0] ?? null;
  return { primary, reference, category: primary ? relevance.relevance : "missing", reason: primary ? relevance.reason : "No stage-specific artifact was emitted for this run." };
}

export function resolveDetectReferenceArtifactRelevance(args: ResolveArgs): {
  relevance: ArtifactRelevance;
  reason: string;
  priority: number;
  group: ArtifactGroup;
} {
  const state: DetectReferenceRunState = {
    strategy: args.activeStrategy,
    supportSelectionMethod: "unknown",
    componentMode: args.componentMode,
    splitMethod: args.splitMethod,
    referenceSurfaceModel: "unknown",
    resolvedReferenceSurfaceModel: "unknown",
    suppressionMaskPolicy: "unknown",
    presetId: null,
    fallbackUsed: Boolean(args.fallbackUsed),
    fallbackStrategy: args.fallbackStrategy ?? null,
    processedAt: args.runUpdatedAt ?? null,
  };
  const preferred = preferredDetectArtifactsForView(args.viewId, args.artifacts, state);
  const hasPreferred = preferred.length > 0;
  const isBlobStrategy = args.activeStrategy === "low_gradient_blob_height_clusters";
  const isPlateauStrategy = args.activeStrategy === "low_gradient_depth_plateaus";
  const isBgAndStripesStrategy = args.activeStrategy === "low_gradient_bg_and_stripes";
  const splitUsesBorders = args.splitMethod === "height_borders" || args.splitMethod === "height_borders_then_histogram";
  const splitUsesHistogram = args.splitMethod === "histogram_gap" || args.splitMethod === "height_borders_then_histogram";

  if (args.viewId === "blob_components") {
    if (!isBlobStrategy) return { relevance: hasPreferred ? "inactive_strategy" : "missing", reason: "Blob components are inactive for the current strategy.", priority: 60, group: hasPreferred ? "legacy" : "all" };
    if (args.componentMode === "height_aware") {
      return hasPreferred
        ? { relevance: "active", reason: "Current run uses height-aware blob components.", priority: 1, group: "main" }
        : { relevance: "missing", reason: "Height-aware blob components were not emitted for this run.", priority: 99, group: "main" };
    }
    return hasPreferred
      ? { relevance: "active", reason: "Current run uses XY-only blob components.", priority: 1, group: "main" }
      : { relevance: "missing", reason: "Blob components were not emitted for this run.", priority: 99, group: "main" };
  }
  if (args.viewId === "height_aware_blob_components") {
    return hasPreferred
      ? { relevance: args.componentMode === "height_aware" ? "debug" : "legacy", reason: args.componentMode === "height_aware" ? "Current main blob-components view already uses the height-aware artifact." : "XY-only mode is active; this height-aware artifact is comparison-only.", priority: 30, group: "debug" }
      : { relevance: "missing", reason: "No XY-only comparison artifact was emitted.", priority: 99, group: "debug" };
  }
  if (args.viewId === "height_aware_connectivity_rejected_edges") {
    if (!isBlobStrategy || args.componentMode !== "height_aware") {
      return { relevance: hasPreferred ? "inactive_strategy" : "missing", reason: "Rejected Z-connections only apply in height-aware blob mode.", priority: 60, group: hasPreferred ? "legacy" : "debug" };
    }
    return hasPreferred
      ? { relevance: "debug", reason: "Shows neighboring low-gradient pixels blocked by the Z-compatibility rule.", priority: 20, group: "debug" }
      : { relevance: "missing", reason: "No rejected Z-connection diagnostics were emitted for this run.", priority: 99, group: "debug" };
  }
  if (args.viewId === "blob_clusters" || args.viewId === "blob_cluster_scores" || args.viewId === "selected_blob_cluster" || args.viewId === "selected_cluster_pre_refine" || args.viewId === "selected_cluster_refined" || args.viewId === "removed_by_refinement" || args.viewId === "rejected_blob_clusters" || args.viewId === "selection_bridge") {
    if (!isBlobStrategy) return { relevance: hasPreferred ? "inactive_strategy" : "missing", reason: "Blob-cluster diagnostics are inactive for the current strategy.", priority: 60, group: hasPreferred ? "legacy" : "all" };
    return hasPreferred
      ? { relevance: "active", reason: "Blob-cluster diagnostics were emitted for the current run.", priority: 10, group: "main" }
      : { relevance: "missing", reason: "No blob-cluster diagnostics were emitted for this run.", priority: 99, group: "main" };
  }
  if (args.viewId === "selected_support_lineage") {
    if (!isBlobStrategy) return { relevance: hasPreferred ? "inactive_strategy" : "missing", reason: "Selected-support lineage is inactive for the current strategy.", priority: 60, group: hasPreferred ? "legacy" : "all" };
    return hasPreferred
      ? { relevance: "active", reason: "Selected-support lineage explains how the chosen cluster became support, model support, and suppression.", priority: 10, group: "main" }
      : { relevance: "missing", reason: "No selected-support lineage artifact was emitted for this run.", priority: 99, group: "main" };
  }
  if (args.viewId === "height_borders" || args.viewId === "height_border_fragments") {
    if (!isBlobStrategy) return { relevance: hasPreferred ? "inactive_strategy" : "missing", reason: "Height-border diagnostics are inactive for the current strategy.", priority: 60, group: hasPreferred ? "legacy" : "debug" };
    if (!splitUsesBorders) {
      return { relevance: hasPreferred ? "debug" : "missing", reason: "Current split method does not use height-border splitting.", priority: 50, group: "debug" };
    }
    return hasPreferred
      ? { relevance: "active", reason: "Height-border splitting is active for this run.", priority: 12, group: "main" }
      : { relevance: "missing", reason: "No height-border diagnostics were emitted for this run.", priority: 99, group: "main" };
  }
  if (args.viewId === "height_split_blobs") {
    if (!isBlobStrategy) return { relevance: hasPreferred ? "inactive_strategy" : "missing", reason: "Height-split fragment diagnostics are inactive for the current strategy.", priority: 60, group: hasPreferred ? "legacy" : "debug" };
    if (args.splitMethod === "disabled") {
      return { relevance: hasPreferred ? "debug" : "missing", reason: "Blob splitting is disabled for the current run.", priority: 50, group: "debug" };
    }
    if (!splitUsesHistogram && !splitUsesBorders) {
      return { relevance: hasPreferred ? "legacy" : "missing", reason: "Final fragment diagnostics are not part of the active split flow.", priority: 70, group: hasPreferred ? "legacy" : "debug" };
    }
    return hasPreferred
      ? { relevance: splitUsesBorders ? "debug" : "active", reason: splitUsesBorders ? "Final fragment layer is available as a comparison/debug artifact for the current split flow." : "Histogram-based split fragments are active for this run.", priority: splitUsesBorders ? 30 : 12, group: splitUsesBorders ? "debug" : "main" }
      : { relevance: "missing", reason: "No final fragment diagnostics were emitted for this run.", priority: 99, group: splitUsesBorders ? "debug" : "main" };
  }
  if (args.viewId === "surface_candidates" || args.viewId === "plateau_plot" || args.viewId === "filtered_depth_plot" || args.viewId === "depth_plot") {
    if (args.viewId === "surface_candidates" && isBgAndStripesStrategy) {
      return hasPreferred
        ? { relevance: "active", reason: "Low-gradient component role classification is active for the current strategy.", priority: 15, group: "main" }
        : { relevance: "missing", reason: "No low-gradient component role diagnostics were emitted for this run.", priority: 99, group: "main" };
    }
    if (isPlateauStrategy) {
      return hasPreferred || args.viewId === "depth_plot"
        ? { relevance: "active", reason: "Plateau diagnostics are active for the current strategy.", priority: 15, group: "main" }
        : { relevance: "missing", reason: "No plateau diagnostics were emitted for this run.", priority: 99, group: "main" };
    }
    return { relevance: hasPreferred ? "inactive_strategy" : "missing", reason: "Plateau diagnostics belong to the inactive strategy.", priority: 70, group: hasPreferred ? "legacy" : "all" };
  }
  if (args.viewId.startsWith("belt_") || args.viewId === "removed_by_stripe_filter") {
    return hasPreferred
      ? { relevance: "debug", reason: "Belt-stripe diagnostics are available for inspection.", priority: 40, group: "stripe" }
      : { relevance: "missing", reason: "No belt-stripe diagnostics were emitted for this run.", priority: 99, group: "stripe" };
  }
  if (args.viewId === "unknown_low_gradient" || args.viewId === "surface_suppression" || args.viewId === "object_search_domain") {
    return hasPreferred
      ? { relevance: isBgAndStripesStrategy ? "active" : "debug", reason: isBgAndStripesStrategy ? "Explicit surface-role masks are active for the current strategy." : "Surface-role masks are available as diagnostics.", priority: isBgAndStripesStrategy ? 18 : 45, group: isBgAndStripesStrategy ? "main" : "debug" }
      : { relevance: "missing", reason: "No explicit surface-role mask was emitted for this run.", priority: 99, group: isBgAndStripesStrategy ? "main" : "debug" };
  }
  if (args.viewId === "reference_model_support" || args.viewId === "reference_suppression_mask") {
    return hasPreferred
      ? { relevance: "active", reason: "Reference-model support and suppression-mask artifacts were emitted for this run.", priority: 12, group: "main" }
      : { relevance: "missing", reason: "No reference-model support or suppression-mask artifact was emitted for this run.", priority: 99, group: "main" };
  }
  if (args.viewId === "support_loss_waterfall") {
    return hasPreferred
      ? { relevance: "active", reason: "Support-loss waterfall is available for this run.", priority: 12, group: "main" }
      : { relevance: "missing", reason: "No support-loss waterfall was emitted for this run.", priority: 99, group: "main" };
  }
  if (args.fallbackUsed && ["selected_surface", "plane_inliers", "residual_heatmap", "diagnostics"].includes(args.viewId)) {
    return { relevance: "fallback", reason: `Current run used fallback strategy ${args.fallbackStrategy ?? "-"}.`, priority: 5, group: "main" };
  }
  return { relevance: "active", reason: "Active detect-stage artifact.", priority: 5, group: "main" };
}
