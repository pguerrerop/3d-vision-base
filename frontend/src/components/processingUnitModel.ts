import type { PipelineInfo, ProcessingUnitDefinition, TakeDetail } from "../api/client";
import type { StageViewDefinition } from "./stageSemantics";
import { canonicalStageId } from "./stageSemantics";
import type { StudioArtifact } from "./studioWorkspaceModel";
import { sourceArtifactsForTake } from "./stage_sources/index.js";

export type ResolvedProcessingUnitArtifactRef = {
  id: string;
  label: string;
  artifactId: string | null;
  kind: string | null;
  description: string | null;
  artifact: StudioArtifact | null;
  present: boolean;
};

export type ResolvedProcessingUnitView = {
  unitId: string;
  viewId: string;
  label: string;
  rendererType: StageViewDefinition["rendererType"];
  declaredArtifactIds: string[];
  resolvedArtifacts: StudioArtifact[];
  missingArtifactIds: string[];
  fallbackUsed: boolean;
  fallbackReason?: string;
  emptyState?: string;
  supportedRenderer: boolean;
};

export function processingUnitsForDetectReference(
  pipeline: PipelineInfo | null,
  detail: TakeDetail | null,
): ProcessingUnitDefinition[] {
  return processingUnitsForStage(pipeline, detail, "detect_belt_plane");
}

export function processingUnitsForPipeline(
  pipeline: PipelineInfo | null,
  detail: TakeDetail | null,
): ProcessingUnitDefinition[] {
  const fromResult = detail?.result?.processing_pipeline?.processing_units;
  if (Array.isArray(fromResult) && fromResult.length > 0) return fromResult;
  const fromPipeline = pipeline?.processing_units;
  if (Array.isArray(fromPipeline) && fromPipeline.length > 0) return fromPipeline;
  return [];
}

export function processingUnitsForStage(
  pipeline: PipelineInfo | null,
  detail: TakeDetail | null,
  stageId: string,
): ProcessingUnitDefinition[] {
  const canonical = canonicalStageId(stageId);
  return processingUnitsForPipeline(pipeline, detail).filter((unit) => canonicalStageId(unit.stage_id) === canonical);
}

export function rootProcessingUnit(units: ProcessingUnitDefinition[]): ProcessingUnitDefinition | null {
  return units.find((unit) => unit.kind === "stage" && unit.parent_id == null) ?? null;
}

export function childProcessingUnits(units: ProcessingUnitDefinition[], parentId: string): ProcessingUnitDefinition[] {
  return units
    .filter((unit) => unit.parent_id === parentId)
    .slice()
    .sort((left, right) => left.order - right.order || left.id.localeCompare(right.id));
}

export function processingUnitById(units: ProcessingUnitDefinition[], unitId: string | null | undefined): ProcessingUnitDefinition | null {
  if (!unitId) return null;
  return units.find((unit) => unit.id === unitId) ?? null;
}

export function processingUnitViewIds(unit: ProcessingUnitDefinition): string[] {
  if (isStripeFilterUnit(unit)) {
    return orderViewIds(STRIPE_FILTER_VIEW_ORDER, unit.views.map((view) => view.id));
  }
  return unit.views.map((view) => view.id);
}

export const STRIPE_FILTER_VIEW_ORDER = [
  "pre_stripe_support",
  "belt_bg",
  "belt_baseline",
  "belt_altitude_plot",
  "belt_stripes_tophat",
  "belt_stripes_shape",
  "belt_wide_object",
  "belt_above_belt",
  "belt_base",
  "belt_stripes",
  "unknown_low_gradient",
  "surface_suppression",
  "object_search_domain",
  "removed_by_stripe_filter",
  "stripe_filtered_support",
] as const;

export function isStripeFilterUnit(unit: ProcessingUnitDefinition): boolean {
  return unit.id.endsWith(".stripe_filter") || unit.category === "stripe_suppression";
}

export function orderViewIds(plan: readonly string[], viewIds: string[]): string[] {
  const planIndex = new Map(plan.map((viewId, index) => [viewId, index]));
  return viewIds
    .slice()
    .sort((left, right) => {
      const leftIndex = planIndex.get(left);
      const rightIndex = planIndex.get(right);
      if (leftIndex == null && rightIndex == null) return 0;
      if (leftIndex == null) return 1;
      if (rightIndex == null) return -1;
      return leftIndex - rightIndex;
    });
}

export function resolveStageViewForUnit(
  unit: ProcessingUnitDefinition,
  semanticViews: StageViewDefinition[],
): Array<{ view: StageViewDefinition; missing: boolean }> {
  const byId = new Map(semanticViews.map((view) => [view.id, view]));
  const resolved = unit.views.map((entry) => {
    const semanticView = byId.get(entry.id);
    const view: StageViewDefinition = {
      id: entry.id,
      label: entry.label,
      rendererType: normalizeRendererType(entry.renderer_type),
      emptyState: entry.empty_state ? { title: entry.empty_state } : semanticView?.emptyState,
    };
    return { view, missing: !semanticView };
  });
  if (!isStripeFilterUnit(unit)) return resolved;
  const order = orderViewIds(STRIPE_FILTER_VIEW_ORDER, resolved.map((entry) => entry.view.id));
  const byViewId = new Map(resolved.map((entry) => [entry.view.id, entry]));
  return order.map((viewId) => byViewId.get(viewId)).filter((entry): entry is { view: StageViewDefinition; missing: boolean } => Boolean(entry));
}

export function normalizeRendererType(value: string): StageViewDefinition["rendererType"] {
  if (value === "overlay" || value === "table" || value === "metrics" || value === "json" || value === "gallery" || value === "histogram" || value === "mask") {
    return value;
  }
  return "image";
}

function artifactByDeclaredId(
  artifacts: StudioArtifact[],
  artifactId: string,
  aliases: string[] = [],
): StudioArtifact | null {
  return artifacts.find((artifact) => artifact.artifact_id === artifactId || aliases.includes(artifact.artifact_id)) ?? null;
}

function unitArtifactAliases(unit: ProcessingUnitDefinition, artifactId: string): string[] {
  return unit.artifacts.find((artifact) => artifact.artifact_id === artifactId)?.aliases ?? [];
}

export function resolveProcessingUnitView(
  args: {
    detail: TakeDetail | null;
    unit: ProcessingUnitDefinition | null;
    viewId: string | null;
    semanticViews: StageViewDefinition[];
    stageArtifacts: StudioArtifact[];
    takeArtifacts?: StudioArtifact[];
  },
): ResolvedProcessingUnitView | null {
  const { detail, unit, viewId, semanticViews, stageArtifacts, takeArtifacts = [] } = args;
  if (!unit) return null;
  const view = unit.views.find((item) => item.id === (viewId || unit.default_view || unit.views[0]?.id));
  if (!view) {
    return {
      unitId: unit.id,
      viewId: viewId || unit.default_view || "unknown",
      label: viewId || unit.label,
      rendererType: "image",
      declaredArtifactIds: [],
      resolvedArtifacts: [],
      missingArtifactIds: [],
      fallbackUsed: true,
      fallbackReason: "Selected unit has no declared view for this selection.",
      emptyState: "No contract-declared view is available for this unit selection.",
      supportedRenderer: false,
    };
  }
  const declaredArtifactIds = [...view.artifact_ids];
  const sourceArtifacts = sourceArtifactsForTake(detail, unit.stage_id);
  const allArtifacts = [...stageArtifacts, ...sourceArtifacts, ...takeArtifacts];
  const resolvedArtifacts = declaredArtifactIds
    .map((artifactId) => artifactByDeclaredId(allArtifacts, artifactId, unitArtifactAliases(unit, artifactId)))
    .filter((artifact): artifact is StudioArtifact => Boolean(artifact));
  const missingArtifactIds = declaredArtifactIds.filter((artifactId) => !resolvedArtifacts.some((artifact) => artifact.artifact_id === artifactId || unitArtifactAliases(unit, artifactId).includes(artifact.artifact_id)));
  const rendererType = normalizeRendererType(view.renderer_type);
  const supportedRenderer = semanticViews.some((item) => item.id === view.id) || ["image", "overlay", "table", "metrics", "json", "gallery", "histogram", "mask"].includes(rendererType);
  const fallbackUsed = resolvedArtifacts.length === 0 || missingArtifactIds.length > 0 || !supportedRenderer;
  const fallbackReason = !supportedRenderer
    ? `Renderer type ${view.renderer_type} is not supported by the current viewport contract path.`
    : resolvedArtifacts.length === 0
      ? "Contract-declared artifacts are missing for this run, so legacy fallback is required."
      : missingArtifactIds.length > 0
        ? `Some contract-declared artifacts are missing: ${missingArtifactIds.join(", ")}.`
        : undefined;
  return {
    unitId: unit.id,
    viewId: view.id,
    label: view.label,
    rendererType,
    declaredArtifactIds,
    resolvedArtifacts,
    missingArtifactIds,
    fallbackUsed,
    fallbackReason,
    emptyState: view.empty_state ?? fallbackReason,
    supportedRenderer,
  };
}

export function resolveProcessingUnitArtifactRefs(
  detail: TakeDetail | null,
  unit: ProcessingUnitDefinition | null,
  stageArtifacts: StudioArtifact[],
  takeArtifacts: StudioArtifact[] = [],
): { inputs: ResolvedProcessingUnitArtifactRef[]; outputs: ResolvedProcessingUnitArtifactRef[] } {
  if (!unit) return { inputs: [], outputs: [] };
  const sourceArtifacts = sourceArtifactsForTake(detail, unit.stage_id);
  const allArtifacts = [...stageArtifacts, ...sourceArtifacts, ...takeArtifacts];
  const toRef = (
    entry: {
      id: string;
      label: string;
      artifact_id?: string | null;
      kind?: string | null;
      description?: string | null;
    },
  ): ResolvedProcessingUnitArtifactRef => {
    const artifact = entry.artifact_id
      ? artifactByDeclaredId(allArtifacts, entry.artifact_id, unitArtifactAliases(unit, entry.artifact_id))
      : null;
    return {
      id: entry.id,
      label: entry.label,
      artifactId: entry.artifact_id ?? null,
      kind: entry.kind ?? null,
      description: entry.description ?? null,
      artifact,
      present: Boolean(artifact),
    };
  };
  return {
    inputs: unit.inputs.map(toRef),
    outputs: unit.outputs.map(toRef),
  };
}
