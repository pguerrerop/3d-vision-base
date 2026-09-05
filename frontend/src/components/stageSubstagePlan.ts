import type { PipelineInfo, ProcessingUnitDefinition, TakeDetail } from "../api/client";
import { childProcessingUnits, processingUnitsForStage, processingUnitViewIds, resolveStageViewForUnit, rootProcessingUnit } from "./processingUnitModel.js";
import {
  resolveDetectReferenceRunState,
  type ArtifactRelevance,
  type DetectReferenceRunState,
} from "./referenceArtifactRelevance.js";
import { canonicalStageId, type StageSemanticDefinition, type StageViewDefinition } from "./stageSemantics.js";
import type { StudioArtifact } from "./studioWorkspaceModel";

export type SubstageRenderMode = "overlay" | "boundary" | "mask" | "side_by_side";

export type StageSubstageItem = {
  unitId?: string;
  substageId: string;
  label: string;
  description?: string;
  order: number;
  viewIds: string[];
  artifactIds: string[];
  recommendedViewId?: string;
  defaultRenderMode?: SubstageRenderMode;
  referenceArtifactIds?: string[];
};

export type StageSubstagePlan = {
  stageId: string;
  processLabel: string;
  processDescription?: string;
  source: "frontend_registry" | "processing_unit_registry";
  strategyPathViewIds: string[];
  substages: StageSubstageItem[];
};

export type ArtifactIoRole = "input" | "intermediate" | "output";

export type ResolvedSubstageArtifact = {
  artifactId: string;
  artifact: StudioArtifact | null;
  category: ArtifactRelevance;
  reason?: string;
  ioRole?: ArtifactIoRole;
};

export type ResolvedStageSubstage = StageSubstageItem & {
  views: Array<{
    view: StageViewDefinition;
    category: ArtifactRelevance;
    reason: string;
  }>;
  supportingArtifacts: ResolvedSubstageArtifact[];
  missingArtifactIds: string[];
};

export type StageSubstagePlanResolution = {
  plan: StageSubstagePlan | null;
  substages: ResolvedStageSubstage[];
};

export type ReferenceViewScope = "strategy_path" | "substage";

export type ReferenceViewSelection = {
  activeScope: ReferenceViewScope;
  activeSubstageId: string | null;
  inferredSubstageId: string | null;
};

function substageIdFromUnitId(unitId: string): string {
  return unitId.split(".").slice(1).join(".") || unitId;
}

// Per-run activity is decided by real output, not by a static strategy declaration.
// The `low_gradient_bg_and_stripes` strategy is a hybrid that reuses the plateau and
// blob machinery inline, so neither the units' `strategy_keys` nor the runtime trace
// report it accurately (both mark those units inactive/skipped even though their
// artifacts are emitted). Artifact presence is the only ground-truth signal that holds
// across every strategy, so a unit is "active for this run" iff it emitted at least one
// of its declared artifacts.
function isUnitActiveByPresence(unit: ProcessingUnitDefinition, presentArtifactIds: Set<string>): boolean {
  if (unit.artifacts.length === 0) return true;
  return unit.artifacts.some((artifact) => presentArtifactIds.has(artifact.artifact_id));
}

function categoryForUnitRole(role: string | undefined, present: boolean, unitActive: boolean): ArtifactRelevance {
  if (!unitActive) return "inactive_strategy";
  if (!present) return "missing";
  if (role === "diagnostic") return "debug";
  return "active";
}

function ioRoleForUnitRole(role: string | undefined, feedsInto: string[] | undefined, isDeclaredOutput: boolean): ArtifactIoRole | undefined {
  // A unit's own declared `outputs` contract is the ground truth for "this is what this step
  // hands off" -- it wins over `role`, which sometimes marks a stage-root's own output artifacts
  // as role="input"/"diagnostic" to describe the *kind* of data (raw capture, informational JSON)
  // rather than whether it's consumed from elsewhere.
  if (isDeclaredOutput) return "output";
  if (role === "input") return "input";
  if (role === "diagnostic" && (feedsInto?.length ?? 0) > 0) return "output";
  if (role === "final") return "output";
  return undefined;
}

function unitOutputArtifactIds(unit: ProcessingUnitDefinition): Set<string> {
  return new Set(unit.outputs.map((entry) => entry.artifact_id).filter((id): id is string => Boolean(id)));
}

function primaryRoleForView(unit: ProcessingUnitDefinition, viewId: string): string | undefined {
  const artifactId = unit.views.find((entry) => entry.id === viewId)?.artifact_ids?.[0];
  if (!artifactId) return undefined;
  return unit.artifacts.find((entry) => entry.artifact_id === artifactId)?.role;
}

const INACTIVE_UNIT_REASON = "This substage did not run for the current take (no artifacts emitted).";
const INACTIVE_ARTIFACT_REASON = "This artifact belongs to a branch that did not run for the current strategy.";

const BLOB_CLUSTER_ONLY_ARTIFACTS = new Set([
  "low_gradient_blob_components_overlay",
  "low_gradient_blob_id_mask",
  "low_gradient_blob_summary",
  "height_consistent_blob_summary",
  "height_aware_blob_components_overlay",
  "height_aware_blob_id_mask",
  "height_aware_connectivity_rejected_edges",
  "height_aware_connectivity_debug",
  "height_border_strength",
  "height_border_cut_mask",
  "height_border_split_debug",
  "height_border_fragments_overlay",
  "height_border_fragments_mask",
  "height_border_fragments_id_mask",
  "height_split_blob_fragments_overlay",
  "height_split_blob_fragments_mask",
  "height_split_blob_fragments_pre_merge_mask",
  "height_split_blob_id_mask",
  "height_split_debug",
  "blob_cluster_score_table",
  "blob_height_clusters",
  "blob_cluster_selection_debug",
  "selected_blob_cluster_mask",
  "selected_blob_cluster_pre_refine_mask",
  "selected_blob_cluster_refined_mask",
  "selected_blob_cluster_overlay",
  "rejected_blob_clusters_mask",
  "rejected_blob_clusters_overlay",
  "support_removed_by_candidate_refinement",
  "selected_support_lineage",
]);

const HEIGHT_AWARE_ONLY_ARTIFACTS = new Set([
  "height_aware_blob_components_overlay",
  "height_aware_blob_id_mask",
  "height_aware_connectivity_rejected_edges",
  "height_aware_connectivity_debug",
]);

const HEIGHT_BORDER_ONLY_ARTIFACTS = new Set([
  "height_border_strength",
  "height_border_cut_mask",
  "height_border_split_debug",
  "height_border_fragments_overlay",
  "height_border_fragments_mask",
  "height_border_fragments_id_mask",
]);

const HEIGHT_HISTOGRAM_ONLY_ARTIFACTS = new Set([
  "height_split_blob_fragments_overlay",
  "height_split_blob_fragments_mask",
  "height_split_blob_fragments_pre_merge_mask",
  "height_split_blob_id_mask",
  "height_split_debug",
  "height_consistent_blob_summary",
]);

const PLATEAU_STRIPE_FILTER_ONLY_ARTIFACTS = new Set([
  "belt_baseline_local_min",
  "belt_altitude_local_min",
  "belt_altitude_histogram_image",
  "belt_altitude_histogram",
  "belt_stripe_filter_debug",
  "belt_stripes_tophat_mask",
  "belt_stripes_shape_mask",
  "belt_above_belt_mask",
  "belt_wide_object_mask",
  "belt_base_mask",
  "support_removed_by_stripe_filter",
]);

function splitUsesBorders(splitMethod: DetectReferenceRunState["splitMethod"]): boolean {
  return splitMethod === "height_borders" || splitMethod === "height_borders_then_histogram";
}

function splitUsesHistogram(splitMethod: DetectReferenceRunState["splitMethod"]): boolean {
  return splitMethod === "histogram_gap" || splitMethod === "height_borders_then_histogram";
}

function artifactInactiveForDetectStrategy(artifactId: string, state: DetectReferenceRunState): boolean {
  if (BLOB_CLUSTER_ONLY_ARTIFACTS.has(artifactId) && state.strategy !== "low_gradient_blob_height_clusters") return true;
  if (HEIGHT_AWARE_ONLY_ARTIFACTS.has(artifactId) && state.componentMode !== "height_aware") return true;
  if (HEIGHT_BORDER_ONLY_ARTIFACTS.has(artifactId) && !splitUsesBorders(state.splitMethod)) return true;
  if (HEIGHT_HISTOGRAM_ONLY_ARTIFACTS.has(artifactId) && !splitUsesHistogram(state.splitMethod)) return true;
  if (PLATEAU_STRIPE_FILTER_ONLY_ARTIFACTS.has(artifactId) && state.strategy !== "low_gradient_depth_plateaus") return true;
  return false;
}

function detectUnitActiveByApplicableArtifact(unit: ProcessingUnitDefinition, presentArtifactIds: Set<string>, state: DetectReferenceRunState): boolean {
  const applicableArtifacts = unit.artifacts.filter((artifact) => !artifactInactiveForDetectStrategy(artifact.artifact_id, state));
  if (applicableArtifacts.length === 0) return false;
  return applicableArtifacts.some((artifact) => presentArtifactIds.has(artifact.artifact_id));
}

function detectSubstagePlanFromUnits(
  semantic: StageSemanticDefinition,
  units: ProcessingUnitDefinition[],
  detail: TakeDetail | null,
  stageArtifacts: StudioArtifact[],
  takeArtifacts: StudioArtifact[],
): StageSubstagePlanResolution | null {
  const root = rootProcessingUnit(units);
  if (!root) return null;
  const children = childProcessingUnits(units, root.id);
  const presentArtifactIds = new Set(stageArtifacts.map((item) => item.artifact_id));
  const runState = resolveDetectReferenceRunState(detail, stageArtifacts, null);
  const substages: ResolvedStageSubstage[] = children.map((unit) => {
    const unitActive = detectUnitActiveByApplicableArtifact(unit, presentArtifactIds, runState);
    const declaredOutputIds = unitOutputArtifactIds(unit);
    const supportingArtifacts = unit.artifacts.map((artifact) => {
      const found = stageArtifacts.find((item) => item.artifact_id === artifact.artifact_id)
        ?? takeArtifacts.find((item) => item.artifact_id === artifact.artifact_id)
        ?? null;
      const inactiveForStrategy = artifactInactiveForDetectStrategy(artifact.artifact_id, runState);
      return {
        artifactId: artifact.artifact_id,
        artifact: found,
        category: inactiveForStrategy ? "inactive_strategy" : categoryForUnitRole(artifact.role, Boolean(found), unitActive),
        reason: inactiveForStrategy ? INACTIVE_ARTIFACT_REASON : (unitActive ? (artifact.description ?? unit.description) : INACTIVE_UNIT_REASON),
        ioRole: ioRoleForUnitRole(artifact.role, artifact.feeds_into, declaredOutputIds.has(artifact.artifact_id)),
      } as ResolvedSubstageArtifact;
    });
    return {
      unitId: unit.id,
      substageId: substageIdFromUnitId(unit.id),
      label: unit.label,
      description: unit.description,
      order: unit.order,
      viewIds: processingUnitViewIds(unit),
      artifactIds: unit.artifacts.map((artifact) => artifact.artifact_id),
      recommendedViewId: unit.default_view ?? unit.views[0]?.id,
      views: resolveStageViewForUnit(unit, semantic.views).map(({ view }) => {
        const declaredArtifactIds = unit.views.find((entry) => entry.id === view.id)?.artifact_ids ?? [];
        const inactiveForStrategy = declaredArtifactIds.some((artifactId) => artifactInactiveForDetectStrategy(artifactId, runState));
        return {
          view,
          category: inactiveForStrategy
            ? "inactive_strategy" as const
            : categoryForUnitRole(primaryRoleForView(unit, view.id), true, unitActive),
          reason: inactiveForStrategy ? INACTIVE_ARTIFACT_REASON : (unitActive ? unit.description : INACTIVE_UNIT_REASON),
        };
      }),
      supportingArtifacts,
      missingArtifactIds: supportingArtifacts.filter((item) => item.category === "missing").map((item) => item.artifactId),
    };
  });
  return {
    plan: {
      stageId: "detect_belt_plane",
      processLabel: root.label,
      processDescription: root.description,
      source: "processing_unit_registry",
      strategyPathViewIds: root.views.map((view) => view.id),
      substages: substages.map((substage) => ({
        unitId: substage.unitId,
        substageId: substage.substageId,
        label: substage.label,
        description: substage.description,
        order: substage.order,
        viewIds: substage.viewIds,
        artifactIds: substage.artifactIds,
        recommendedViewId: substage.recommendedViewId,
        defaultRenderMode: substage.defaultRenderMode,
        referenceArtifactIds: substage.referenceArtifactIds,
      })),
    },
    substages,
  };
}

function genericSubstagePlanFromUnits(
  stageId: string,
  semantic: StageSemanticDefinition,
  units: ProcessingUnitDefinition[],
  stageArtifacts: StudioArtifact[],
  takeArtifacts: StudioArtifact[],
): StageSubstagePlanResolution | null {
  const root = rootProcessingUnit(units);
  if (!root) return null;
  const children = childProcessingUnits(units, root.id);
  const substages: ResolvedStageSubstage[] = children.map((unit) => {
    const declaredOutputIds = unitOutputArtifactIds(unit);
    const supportingArtifacts = unit.artifacts.map((artifact) => {
      const found = stageArtifacts.find((item) => item.artifact_id === artifact.artifact_id)
        ?? takeArtifacts.find((item) => item.artifact_id === artifact.artifact_id)
        ?? null;
      return {
        artifactId: artifact.artifact_id,
        artifact: found,
        category: categoryForUnitRole(artifact.role, Boolean(found), true),
        reason: artifact.description ?? unit.description,
        ioRole: ioRoleForUnitRole(artifact.role, artifact.feeds_into, declaredOutputIds.has(artifact.artifact_id)),
      } as ResolvedSubstageArtifact;
    });
    return {
      unitId: unit.id,
      substageId: substageIdFromUnitId(unit.id),
      label: unit.label,
      description: unit.description,
      order: unit.order,
      viewIds: processingUnitViewIds(unit),
      artifactIds: unit.artifacts.map((artifact) => artifact.artifact_id),
      recommendedViewId: unit.default_view ?? unit.views[0]?.id,
      views: resolveStageViewForUnit(unit, semantic.views).map(({ view, missing }) => ({
        view,
        category: missing ? "missing" as const : categoryForUnitRole(primaryRoleForView(unit, view.id), true, true),
        reason: missing ? "Declared by processing-unit contract; falling back to semantic renderer metadata." : unit.description,
      })),
      supportingArtifacts,
      missingArtifactIds: supportingArtifacts.filter((item) => item.category === "missing").map((item) => item.artifactId),
    };
  });
  return {
    plan: {
      stageId,
      processLabel: root.label,
      processDescription: root.description,
      source: "processing_unit_registry",
      strategyPathViewIds: root.views.map((view) => view.id),
      substages: substages.map((substage) => ({
        unitId: substage.unitId,
        substageId: substage.substageId,
        label: substage.label,
        description: substage.description,
        order: substage.order,
        viewIds: substage.viewIds,
        artifactIds: substage.artifactIds,
        recommendedViewId: substage.recommendedViewId,
        defaultRenderMode: substage.defaultRenderMode,
        referenceArtifactIds: substage.referenceArtifactIds,
      })),
    },
    substages,
  };
}

export function resolveStageSubstagePlan(
  stageId: string,
  semantic: StageSemanticDefinition,
  pipeline: PipelineInfo | null,
  detail: TakeDetail | null,
  stageArtifacts: StudioArtifact[],
  takeArtifacts: StudioArtifact[] = stageArtifacts,
): StageSubstagePlanResolution {
  const canonical = canonicalStageId(stageId);
  const stageUnits = processingUnitsForStage(pipeline, detail, canonical);
  if (stageUnits.length > 0) {
    if (canonical === "detect_belt_plane") {
      const detectUnitPlan = detectSubstagePlanFromUnits(semantic, stageUnits, detail, stageArtifacts, takeArtifacts);
      if (detectUnitPlan) return detectUnitPlan;
    }
    const genericPlan = genericSubstagePlanFromUnits(canonical, semantic, stageUnits, stageArtifacts, takeArtifacts);
    if (genericPlan) return genericPlan;
  }
  return { plan: null, substages: [] };
}

export function inferDetectSubstageId(
  resolution: StageSubstagePlanResolution,
  viewId: string | null,
): string | null {
  if (!viewId) return null;
  const match = resolution.substages.find((substage) => substage.viewIds.includes(viewId) || substage.artifactIds.includes(viewId));
  return match?.substageId ?? null;
}

export function resolveReferenceViewSelection(
  resolution: StageSubstagePlanResolution,
  explicitScope: ReferenceViewScope | null,
  selectedSubstageId: string | null,
  viewId: string | null,
): ReferenceViewSelection {
  const inferredSubstageId = inferDetectSubstageId(resolution, viewId);
  const validSelectedSubstageId = selectedSubstageId && resolution.substages.some((substage) => substage.substageId === selectedSubstageId)
    ? selectedSubstageId
    : null;

  if (explicitScope === "strategy_path") {
    return {
      activeScope: "strategy_path",
      activeSubstageId: null,
      inferredSubstageId,
    };
  }

  if (explicitScope === "substage" && validSelectedSubstageId) {
    return {
      activeScope: "substage",
      activeSubstageId: validSelectedSubstageId,
      inferredSubstageId,
    };
  }

  const strategyPathIds = new Set(resolution.plan?.strategyPathViewIds ?? []);
  if (inferredSubstageId && viewId && !strategyPathIds.has(viewId)) {
    return {
      activeScope: "substage",
      activeSubstageId: inferredSubstageId,
      inferredSubstageId,
    };
  }

  return {
    activeScope: "strategy_path",
    activeSubstageId: null,
    inferredSubstageId,
  };
}
