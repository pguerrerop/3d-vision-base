import type { TakeDetail } from "../api/client";
import type { StudioArtifact } from "./studioWorkspaceModel";
import { canonicalStageId, type StageSemanticDefinition, type StageViewDefinition } from "./stageSemantics.js";
import {
  resolveDetectReferenceArtifactRelevance,
  resolveDetectReferenceRunState,
  resolveDetectViewArtifacts,
  type ArtifactRelevance,
  type DetectReferenceRunState,
  type ReferenceStrategy,
} from "./referenceArtifactRelevance.js";

export type DisplayRole =
  | "input"
  | "candidate_signal"
  | "candidate_filter"
  | "strategy_intermediate"
  | "selected_reference_surface"
  | "fit_support"
  | "qa"
  | "supporting"
  | "debug"
  | "legacy"
  | "fallback";

export type StrategyDisplayItem = {
  viewId: string;
  label: string;
  role: DisplayRole;
  primaryArtifactIds?: string[];
  referenceArtifactIds?: string[];
  fallbackArtifactIds?: string[];
  description?: string;
};

export type StageDisplayPlan = {
  stageId: string;
  strategy: string;
  source: "run_metadata" | "frontend_registry" | "stage_semantic" | "generic_fallback";
  displaySequence: StrategyDisplayItem[];
};

export type ResolvedStageDisplayItem = StrategyDisplayItem & {
  view: StageViewDefinition;
  category: ArtifactRelevance;
  primaryArtifactId: string | null;
  referenceArtifactId: string | null;
  expectedPrimaryArtifactId: string | null;
  fallbackArtifactId: string | null;
  reason: string;
};

export type StageDisplayPlanResolution = {
  plan: StageDisplayPlan;
  items: ResolvedStageDisplayItem[];
  supportingViews: Array<{ view: StageViewDefinition; category: ArtifactRelevance; reason: string }>;
  debugViews: Array<{ view: StageViewDefinition; category: ArtifactRelevance; reason: string }>;
};

function firstArtifactByIds(artifacts: StudioArtifact[], ids: string[] | undefined): StudioArtifact | null {
  if (!ids?.length) return null;
  for (const id of ids) {
    const found = artifacts.find((artifact) => artifact.artifact_id === id);
    if (found) return found;
  }
  return null;
}

function parseDisplayPlanCandidate(value: unknown, stageId: string): StageDisplayPlan | null {
  if (!value || typeof value !== "object") return null;
  const record = value as Record<string, unknown>;
  const candidateStageId = String(record.stage_id ?? "").trim();
  const sequence = Array.isArray(record.display_sequence) ? record.display_sequence : null;
  if (!sequence?.length) return null;
  if (candidateStageId && canonicalStageId(candidateStageId) !== canonicalStageId(stageId)) return null;
  const displaySequence = sequence
    .filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === "object")
    .map((item) => ({
      viewId: normalizeDisplayPlanViewId(String(item.view_id ?? item.viewId ?? "").trim()),
      label: String(item.label ?? item.view_id ?? item.viewId ?? "").trim(),
      role: String(item.role ?? "supporting").trim() as DisplayRole,
      primaryArtifactIds: Array.isArray(item.primary_artifact_ids)
        ? item.primary_artifact_ids.map(String)
        : item.primary_artifact_id == null
          ? undefined
          : [String(item.primary_artifact_id)],
      referenceArtifactIds: Array.isArray(item.reference_artifact_ids)
        ? item.reference_artifact_ids.map(String)
        : item.reference_artifact_id == null
          ? undefined
          : [String(item.reference_artifact_id)],
      fallbackArtifactIds: Array.isArray(item.fallback_artifact_ids)
        ? item.fallback_artifact_ids.map(String)
        : item.fallback_artifact_id == null
          ? undefined
          : [String(item.fallback_artifact_id)],
      description: item.description == null ? undefined : String(item.description),
    }))
    .filter((item) => item.viewId.length > 0);
  if (!displaySequence.length) return null;
  return {
    stageId: candidateStageId || canonicalStageId(stageId),
    strategy: String(record.strategy ?? "unknown"),
    source: "run_metadata",
    displaySequence,
  };
}

function normalizeDisplayPlanViewId(viewId: string): string {
  const normalized = viewId.trim().toLowerCase();
  if (normalized === "raw") return "raw_heightmap";
  if (normalized === "gradient") return "depth_gradient";
  if (normalized === "low_gradient") return "low_gradient_mask";
  if (normalized === "residuals") return "residual_heatmap";
  return viewId;
}

function runProvidedDisplayPlan(detail: TakeDetail | null, stageArtifacts: StudioArtifact[], stageId: string): StageDisplayPlan | null {
  const resultMetadata = detail?.result && typeof detail.result === "object"
    ? (detail.result as unknown as Record<string, unknown>).metadata
    : null;
  const resultDisplayPlans = resultMetadata && typeof resultMetadata === "object"
    ? ((resultMetadata as Record<string, unknown>).stage_display_plans as Record<string, unknown> | undefined)
    : undefined;
  const candidates: unknown[] = [
    resultDisplayPlans?.[stageId],
    resultDisplayPlans?.[canonicalStageId(stageId)],
    stageArtifacts.find((artifact) => artifact.artifact_id === "plane_fit_debug")?.metadata && (stageArtifacts.find((artifact) => artifact.artifact_id === "plane_fit_debug")?.metadata as Record<string, unknown>).display_plan,
    stageArtifacts.find((artifact) => artifact.artifact_id === "background_selection_debug")?.metadata && (stageArtifacts.find((artifact) => artifact.artifact_id === "background_selection_debug")?.metadata as Record<string, unknown>).display_plan,
  ];
  for (const candidate of candidates) {
    const parsed = parseDisplayPlanCandidate(candidate, stageId);
    if (parsed) return parsed;
  }
  return null;
}

const RAW_REFERENCE_IDS = [
  "heightmap_input_preview",
  "raw_heightmap_preview",
  "source_heightmap_preview",
  "replay_heightmap_preview",
  "raw_heightmap",
  "heightmap",
  "input_preview",
];

function detectPlanRegistry(state: DetectReferenceRunState): StrategyDisplayItem[] {
  const strategy = state.strategy;
  const fitSupportLabel = state.resolvedReferenceSurfaceModel === "constant_z" ? "Model support" : "Plane inliers";
  if (strategy === "low_gradient_blob_height_clusters") {
    return [
      { viewId: "raw_heightmap", label: "Raw", role: "input", primaryArtifactIds: RAW_REFERENCE_IDS },
      { viewId: "valid_mask", label: "Valid mask", role: "candidate_filter", primaryArtifactIds: ["valid_mask"] },
      { viewId: "low_gradient_mask", label: "Low-gradient", role: "candidate_filter", primaryArtifactIds: ["low_gradient_mask"] },
      { viewId: "blob_components", label: "Blob components", role: "strategy_intermediate", primaryArtifactIds: ["height_aware_blob_components_overlay", "height_aware_blob_id_mask"], fallbackArtifactIds: ["low_gradient_blob_components_overlay", "low_gradient_blob_id_mask"] },
      { viewId: "selection_bridge", label: "Selection bridge", role: "strategy_intermediate", primaryArtifactIds: ["blob_height_clusters", "blob_cluster_selection_debug"] },
      { viewId: "blob_clusters", label: "Blob clusters", role: "strategy_intermediate", primaryArtifactIds: ["blob_height_clusters", "blob_cluster_selection_debug"] },
      { viewId: "blob_cluster_scores", label: "Cluster score table", role: "strategy_intermediate", primaryArtifactIds: ["blob_cluster_score_table", "blob_cluster_selection_debug"] },
      { viewId: "selected_blob_cluster", label: "Selected blob cluster", role: "strategy_intermediate", primaryArtifactIds: ["selected_blob_cluster_overlay", "selected_blob_cluster_mask"], fallbackArtifactIds: ["selected_blob_cluster_pre_refine_mask"] },
      { viewId: "selected_support_lineage", label: "Selected support lineage", role: "strategy_intermediate", primaryArtifactIds: ["selected_support_lineage"] },
      { viewId: "selected_surface", label: state.resolvedReferenceSurfaceModel === "constant_z" ? "Model support" : "Selected surface", role: "selected_reference_surface", primaryArtifactIds: ["selected_reference_support_mask", "reference_surface_selected_mask", "selected_surface", "selected_reference_surface"], referenceArtifactIds: RAW_REFERENCE_IDS, fallbackArtifactIds: ["selected_blob_cluster_refined_mask", "selected_blob_cluster_pre_refine_mask", "selected_blob_cluster_overlay", "selected_blob_cluster_mask", "expanded_plane_mask"] },
      { viewId: "plane_inliers", label: fitSupportLabel, role: "fit_support", primaryArtifactIds: ["reference_model_inlier_mask", "final_plane_inlier_mask", "plane_inlier_mask"], referenceArtifactIds: RAW_REFERENCE_IDS, fallbackArtifactIds: ["expanded_plane_mask"] },
      { viewId: "residual_heatmap", label: "Residuals", role: "qa", primaryArtifactIds: ["belt_plane_residuals"] },
      { viewId: "json", label: "JSON", role: "debug" },
    ];
  }
  if (strategy === "low_gradient_depth_plateaus") {
    return [
      { viewId: "raw_heightmap", label: "Raw", role: "input", primaryArtifactIds: RAW_REFERENCE_IDS },
      { viewId: "depth_gradient", label: "Gradient", role: "candidate_signal", primaryArtifactIds: ["depth_gradient_magnitude"] },
      { viewId: "low_gradient_mask", label: "Low-gradient", role: "candidate_filter", primaryArtifactIds: ["low_gradient_mask"] },
      { viewId: "surface_candidates", label: "Surface candidates", role: "strategy_intermediate", primaryArtifactIds: ["reference_surface_candidates"], fallbackArtifactIds: ["reference_surface_plateaus"] },
      { viewId: "selected_surface", label: "Selected surface", role: "selected_reference_surface", primaryArtifactIds: ["selected_reference_support_mask", "reference_surface_selected_mask"], referenceArtifactIds: RAW_REFERENCE_IDS, fallbackArtifactIds: ["expanded_plane_mask"] },
      { viewId: "plane_inliers", label: fitSupportLabel, role: "fit_support", primaryArtifactIds: ["reference_model_inlier_mask", "final_plane_inlier_mask", "plane_inlier_mask"], referenceArtifactIds: RAW_REFERENCE_IDS, fallbackArtifactIds: ["expanded_plane_mask"] },
      { viewId: "residual_heatmap", label: "Residuals", role: "qa", primaryArtifactIds: ["belt_plane_residuals"] },
      { viewId: "json", label: "JSON", role: "debug" },
    ];
  }
  if (strategy === "low_gradient_bg_and_stripes") {
    return [
      { viewId: "raw_heightmap", label: "Raw", role: "input", primaryArtifactIds: RAW_REFERENCE_IDS },
      { viewId: "depth_gradient", label: "Gradient", role: "candidate_signal", primaryArtifactIds: ["depth_gradient_magnitude"] },
      { viewId: "low_gradient_mask", label: "Low-gradient", role: "candidate_filter", primaryArtifactIds: ["low_gradient_mask"] },
      { viewId: "surface_candidates", label: "Surface roles", role: "strategy_intermediate", primaryArtifactIds: ["low_gradient_components", "reference_surface_candidates"] },
      { viewId: "belt_bg", label: "Belt background", role: "selected_reference_surface", primaryArtifactIds: ["belt_bg_mask"], referenceArtifactIds: RAW_REFERENCE_IDS },
      { viewId: "belt_stripes", label: "Belt stripes", role: "strategy_intermediate", primaryArtifactIds: ["belt_stripes_mask"], referenceArtifactIds: RAW_REFERENCE_IDS },
      { viewId: "unknown_low_gradient", label: "Unknown low-gradient", role: "strategy_intermediate", primaryArtifactIds: ["unknown_low_gradient_mask"], referenceArtifactIds: RAW_REFERENCE_IDS },
      { viewId: "surface_suppression", label: "Surface suppression", role: "fit_support", primaryArtifactIds: ["surface_suppression_mask"], referenceArtifactIds: RAW_REFERENCE_IDS },
      { viewId: "object_search_domain", label: "Object domain", role: "strategy_intermediate", primaryArtifactIds: ["object_search_domain_mask"], referenceArtifactIds: RAW_REFERENCE_IDS },
      { viewId: "plane_inliers", label: fitSupportLabel, role: "fit_support", primaryArtifactIds: ["reference_model_inlier_mask", "final_plane_inlier_mask", "plane_inlier_mask"], referenceArtifactIds: RAW_REFERENCE_IDS, fallbackArtifactIds: ["expanded_plane_mask"] },
      { viewId: "residual_heatmap", label: "Residuals", role: "qa", primaryArtifactIds: ["belt_plane_residuals"] },
      { viewId: "json", label: "JSON", role: "debug" },
    ];
  }
  if (strategy === "low_gradient_surface") {
    return [
      { viewId: "raw_heightmap", label: "Raw", role: "input", primaryArtifactIds: RAW_REFERENCE_IDS },
      { viewId: "depth_gradient", label: "Gradient", role: "candidate_signal", primaryArtifactIds: ["depth_gradient_magnitude"] },
      { viewId: "low_gradient_mask", label: "Low-gradient", role: "candidate_filter", primaryArtifactIds: ["low_gradient_mask"] },
      { viewId: "selected_surface", label: "Selected surface", role: "selected_reference_surface", primaryArtifactIds: ["selected_reference_support_mask", "reference_surface_selected_mask"], referenceArtifactIds: RAW_REFERENCE_IDS, fallbackArtifactIds: ["expanded_plane_mask"] },
      { viewId: "plane_inliers", label: fitSupportLabel, role: "fit_support", primaryArtifactIds: ["reference_model_inlier_mask", "final_plane_inlier_mask", "plane_inlier_mask"], referenceArtifactIds: RAW_REFERENCE_IDS, fallbackArtifactIds: ["expanded_plane_mask"] },
      { viewId: "residual_heatmap", label: "Residuals", role: "qa", primaryArtifactIds: ["belt_plane_residuals"] },
      { viewId: "json", label: "JSON", role: "debug" },
    ];
  }
  return [
    { viewId: "raw_heightmap", label: "Raw", role: "input", primaryArtifactIds: RAW_REFERENCE_IDS },
    { viewId: "selected_surface", label: "Selected surface", role: "selected_reference_surface", primaryArtifactIds: ["selected_reference_support_mask", "reference_surface_selected_mask"], referenceArtifactIds: RAW_REFERENCE_IDS, fallbackArtifactIds: ["expanded_plane_mask"] },
    { viewId: "plane_inliers", label: fitSupportLabel, role: "fit_support", primaryArtifactIds: ["reference_model_inlier_mask", "final_plane_inlier_mask", "plane_inlier_mask"], referenceArtifactIds: RAW_REFERENCE_IDS, fallbackArtifactIds: ["expanded_plane_mask"] },
    { viewId: "residual_heatmap", label: "Residuals", role: "qa", primaryArtifactIds: ["belt_plane_residuals"] },
    { viewId: "json", label: "JSON", role: "debug" },
  ];
}

function resolveDetectPlanItem(
  item: StrategyDisplayItem,
  view: StageViewDefinition,
  stageArtifacts: StudioArtifact[],
  detail: TakeDetail | null,
) : ResolvedStageDisplayItem {
  const state = resolveDetectReferenceRunState(detail, stageArtifacts, null);
  if (item.viewId === "json") {
    return {
      ...item,
      view,
      category: "debug",
      primaryArtifactId: "json",
      referenceArtifactId: null,
      expectedPrimaryArtifactId: "json",
      fallbackArtifactId: null,
      reason: "Result/debug JSON view from display plan.",
    };
  }
  const exact = firstArtifactByIds(stageArtifacts, item.primaryArtifactIds);
  const fallback = exact ? null : firstArtifactByIds(stageArtifacts, item.fallbackArtifactIds);
  const detected = exact || fallback ? null : resolveDetectViewArtifacts(item.viewId, stageArtifacts, state);
  const primary = exact ?? fallback ?? detected?.primary ?? null;
  const reference = firstArtifactByIds(stageArtifacts, item.referenceArtifactIds) ?? detected?.reference ?? null;
  const expectedPrimaryArtifactId = item.primaryArtifactIds?.[0] ?? null;
  const fallbackArtifactId = fallback?.artifact_id ?? (exact == null && detected?.primary && detected.primary.artifact_id !== expectedPrimaryArtifactId ? detected.primary.artifact_id : null);
  const category: ArtifactRelevance = exact
    ? "active"
    : fallback || fallbackArtifactId
      ? "fallback"
      : primary
        ? (detected?.category ?? "active")
        : "missing";
  const reason = exact
    ? `${item.label} resolved from strategy display plan.`
    : fallback
      ? `Expected primary artifact missing. Showing fallback ${fallback.artifact_id}.`
      : detected?.primary
        ? `Expected primary artifact missing. Showing fallback ${detected.primary.artifact_id}.`
        : `No active artifact found for this strategy view. Expected: ${expectedPrimaryArtifactId ?? item.viewId}.`;
  return {
    ...item,
    view,
    category,
    primaryArtifactId: primary?.artifact_id ?? null,
    referenceArtifactId: reference?.artifact_id ?? null,
    expectedPrimaryArtifactId,
    fallbackArtifactId,
    reason,
  };
}

export function resolveStageDisplayPlan(
  stageId: string,
  semantic: StageSemanticDefinition,
  detail: TakeDetail | null,
  stageArtifacts: StudioArtifact[],
): StageDisplayPlanResolution {
  const canonical = canonicalStageId(stageId);
  if (canonical !== "detect_belt_plane") {
    const displaySequence = semantic.views.map((view) => ({ viewId: view.id, label: view.label, role: (view.id === "json" ? "debug" : "supporting") as DisplayRole }));
    const items: ResolvedStageDisplayItem[] = displaySequence.flatMap((item) => {
      const view = semantic.views.find((candidate) => candidate.id === item.viewId);
      if (!view) return [];
      return [{
        ...item,
        view,
        category: item.viewId === "json" ? "debug" as ArtifactRelevance : "active" as ArtifactRelevance,
        primaryArtifactId: item.viewId,
        referenceArtifactId: null,
        expectedPrimaryArtifactId: item.viewId,
        fallbackArtifactId: null,
        reason: "Stage semantic fallback.",
      }];
    });
    return {
      plan: { stageId: canonical, strategy: "stage_semantic", source: "stage_semantic", displaySequence },
      items,
      supportingViews: [],
      debugViews: [],
    };
  }

  const state = resolveDetectReferenceRunState(detail, stageArtifacts, null);
  const plan = runProvidedDisplayPlan(detail, stageArtifacts, canonical) ?? {
    stageId: canonical,
    strategy: state.strategy,
    source: "frontend_registry" as const,
    displaySequence: detectPlanRegistry(state),
  };
  const items = plan.displaySequence
    .map((item) => {
      const view = semantic.views.find((candidate) => candidate.id === item.viewId);
      if (!view) return null;
      return resolveDetectPlanItem(item, view, stageArtifacts, detail);
    })
    .filter((item): item is ResolvedStageDisplayItem => Boolean(item));
  const strategyPathIds = new Set(items.map((item) => item.viewId));
  const secondaryViews = semantic.views
    .filter((view) => !strategyPathIds.has(view.id))
    .map((view) => {
      const result = resolveDetectReferenceArtifactRelevance({
        viewId: view.id,
        artifacts: stageArtifacts,
        activeStrategy: state.strategy,
        componentMode: state.componentMode,
        splitMethod: state.splitMethod,
        fallbackUsed: state.fallbackUsed,
        fallbackStrategy: state.fallbackStrategy,
        runUpdatedAt: state.processedAt,
      });
      return { view, category: result.relevance, reason: result.reason };
    })
    .filter((entry) => entry.category !== "missing");
  return {
    plan,
    items,
    supportingViews: secondaryViews.filter((entry) => entry.category === "active" || entry.category === "debug"),
    debugViews: secondaryViews.filter((entry) => entry.category !== "active" && entry.category !== "debug"),
  };
}
