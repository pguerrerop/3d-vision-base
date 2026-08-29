import type { PartialRerunPlanResponse, PipelineComparisonResponse, PipelineRecipe, ProcessingUnitDefinition, TakeDetail } from "../api/client";
import type { RecipeDiffGroup } from "./recipeModel";

export type ProcessingUnitGraphNodeStatus = "completed" | "missing_artifacts" | "warning" | "failed" | "skipped" | "inferred" | "not_run" | "unknown";
export type ProcessingUnitGraphDirtyState = "clean" | "dirty" | "unknown";
export type ProcessingUnitGraphComparisonState = "changed" | "unchanged" | "unknown";
export type ProcessingUnitGraphPlanState = "reused" | "rerun" | "invalidated" | "blocked" | "none";
export type ProcessingUnitGraphSelectionRole = "selected" | "boundary" | "none";

export type ProcessingUnitGraphNode = {
  id: string;
  label: string;
  kind: string;
  stageId: string;
  parentId?: string;
  order: number;
  parameterCount: number;
  artifactCount: number;
  description?: string;
  status: ProcessingUnitGraphNodeStatus;
  dirtyState: ProcessingUnitGraphDirtyState;
  lastRunDirtyState: ProcessingUnitGraphDirtyState;
  comparisonState: ProcessingUnitGraphComparisonState;
  planState: ProcessingUnitGraphPlanState;
  selectionRole: ProcessingUnitGraphSelectionRole;
  warningCount: number;
  missingArtifactCount: number;
  defaultView?: string | null;
  inputArtifacts: string[];
  outputArtifacts: string[];
  artifactIds: string[];
  tunableParameters: string[];
  hasRecipeValues: boolean;
  comparisonSummary?: {
    parameterChanges: number;
    artifactChanges: number;
    metricChanges: number;
    diagnosticChanges: number;
  };
  traceSource?: string | null;
  tracePrecision?: string | null;
  durationMs?: number | null;
  parametersUsed?: Record<string, unknown>;
  metricsSummary?: Record<string, unknown>;
  diagnosticsSummary?: Record<string, unknown>;
  errors?: string[];
  warnings?: string[];
  planIsStale?: boolean;
};

export type ProcessingUnitGraphEdge = {
  id: string;
  source: string;
  target: string;
  kind: "order" | "artifact" | "dependency";
  confidence: "explicit" | "inferred";
  label?: string;
};

export type ProcessingUnitGraphGroup = {
  id: string;
  label: string;
  stageId: string;
  order: number;
  nodeIds: string[];
};

export type ProcessingUnitGraph = {
  nodes: ProcessingUnitGraphNode[];
  edges: ProcessingUnitGraphEdge[];
  groups: ProcessingUnitGraphGroup[];
  warnings: string[];
};

export function graphSelectionForUnit(unit: ProcessingUnitDefinition): {
  stageContractId: string;
  unitId: string;
  substageId: string | null;
} {
  return {
    stageContractId: unit.stage_id,
    unitId: unit.id,
    substageId: unit.kind === "substage" ? unit.id.split(".").slice(1).join(".") || unit.id : null,
  };
}

function stageLabelFromUnit(unit: ProcessingUnitDefinition): string {
  const raw = `${unit.stage_id ?? ""} ${unit.label ?? ""}`.toLowerCase();
  const canonical = raw.includes("normalize_heights_to_plane")
    ? "normalize_heights_to_plane"
    : raw.includes("detect_belt_plane")
      ? "detect_belt_plane"
      : raw.includes("remove_belt_segment_objects")
        ? "remove_belt_segment_objects"
        : raw.includes("load_heightmap")
          || raw.includes("loadheightmapcapture")
          || raw.includes("heightmap capture")
          ? "load_heightmap"
          : raw.includes("fit_object_geometry") || raw === "geometry"
            ? "fit_object_geometry"
            : raw.includes("compute_height_metrics") || raw === "measurement"
              ? "compute_height_metrics"
              : raw.includes("measurement_diagnostics")
                ? "measurement_diagnostics"
                : raw.includes("classify_25d")
                  ? "classify_25d"
                  : raw.includes("overlays_25d")
                    ? "overlays_25d"
                    : unit.stage_id;
  if (canonical === "load_heightmap") return "Load heightmap capture";
  if (canonical === "detect_belt_plane") return "Detect reference surface";
  if (canonical === "normalize_heights_to_plane") return "Normalize heights to reference";
  if (canonical === "remove_belt_segment_objects") return "Remove reference + segment objects";
  if (canonical === "fit_object_geometry") return "Geometry";
  if (canonical === "compute_height_metrics") return "Measurement";
  if (canonical === "measurement_diagnostics") return "Measurement diagnostics";
  if (canonical === "classify_25d") return "Classification";
  if (canonical === "overlays_25d") return "Overlay";
  return unit.label;
}

function graphNodeStatus(
  unit: ProcessingUnitDefinition,
  traceEntry: Record<string, unknown> | null,
  availableArtifactIds: Set<string>,
  hasRunResult: boolean,
): { status: ProcessingUnitGraphNodeStatus; warningCount: number; missingArtifactCount: number } {
  const warningCount = Array.isArray(traceEntry?.warnings) ? traceEntry.warnings.length : 0;
  const errorCount = Array.isArray(traceEntry?.errors) ? traceEntry.errors.length : 0;
  const missingArtifactCount = unit.outputs.filter((output) => output.artifact_id && !availableArtifactIds.has(output.artifact_id)).length;
  const traceStatus = String(traceEntry?.status ?? "");
  const traceSource = String(traceEntry?.trace_source ?? "");
  if (traceStatus === "skipped") return { status: "skipped", warningCount, missingArtifactCount };
  if (traceStatus === "failed" || traceStatus === "error") return { status: "failed", warningCount, missingArtifactCount };
  if (traceSource === "best_effort_artifact_registry" && traceStatus && traceStatus !== "not_emitted") {
    return { status: "inferred", warningCount, missingArtifactCount };
  }
  if (errorCount > 0) return { status: "failed", warningCount, missingArtifactCount };
  if (warningCount > 0) return { status: "warning", warningCount, missingArtifactCount };
  if (traceStatus === "completed") {
    if (missingArtifactCount > 0 && unit.outputs.length > 0) return { status: "missing_artifacts", warningCount, missingArtifactCount };
    return { status: "completed", warningCount, missingArtifactCount };
  }
  if (traceStatus === "not_emitted") return { status: "not_run", warningCount, missingArtifactCount };
  if (hasRunResult) {
    const hasAnyArtifact = unit.artifacts.some((artifact) => availableArtifactIds.has(artifact.artifact_id));
    if (hasAnyArtifact) return { status: "completed", warningCount, missingArtifactCount };
    if (missingArtifactCount > 0 && unit.outputs.length > 0) return { status: "missing_artifacts", warningCount, missingArtifactCount };
    return { status: "not_run", warningCount, missingArtifactCount };
  }
  return { status: "unknown", warningCount, missingArtifactCount };
}

export type ProcessingUnitPersistedTrace = {
  reusedUnitIds: string[];
  boundaryStageId: string | null;
  selectedUnitId: string | null;
};

export function persistedTraceFromDetail(detail?: TakeDetail | null): ProcessingUnitPersistedTrace | null {
  const result = detail?.result;
  if (!result || result.execution_mode !== "rerun_from_public_stage_boundary") return null;
  const trace = result.processing_unit_trace;
  const unitResults = ((trace?.unit_results ?? trace?.units) ?? {}) as Record<string, Record<string, unknown>>;
  const reusedUnitIds = Object.entries(unitResults)
    .filter(([, entry]) => entry?.status === "reused" || entry?.execution_role === "reused")
    .map(([unitId]) => unitId);
  const recipeSnapshot = result.recipe_snapshot as Record<string, unknown> | undefined;
  const boundaryStageId = typeof recipeSnapshot?.boundary_stage_id === "string" ? recipeSnapshot.boundary_stage_id : null;
  const persistedPlan = (result as unknown as Record<string, unknown>).partial_rerun_plan as Record<string, unknown> | undefined;
  const selectedUnitId = typeof persistedPlan?.selected_unit_id === "string" ? persistedPlan.selected_unit_id : boundaryStageId;
  return {
    reusedUnitIds,
    boundaryStageId,
    selectedUnitId,
  };
}

export function buildProcessingUnitGraph(args: {
  units: ProcessingUnitDefinition[];
  detail?: TakeDetail | null;
  recipe?: PipelineRecipe | null;
  recipeDiffGroups?: RecipeDiffGroup[];
  lastRunDiffGroups?: RecipeDiffGroup[];
  comparison?: PipelineComparisonResponse | null;
  partialRerunPlan?: PartialRerunPlanResponse | null;
  partialRerunPlanStale?: boolean;
  persistedTrace?: ProcessingUnitPersistedTrace | null;
}): ProcessingUnitGraph {
  const { units, detail, recipe, recipeDiffGroups = [], lastRunDiffGroups = [], comparison, partialRerunPlan, partialRerunPlanStale = false, persistedTrace = null } = args;
  if (!units.length) return { nodes: [], edges: [], groups: [], warnings: ["No processing units available."] };

  const trace = detail?.result?.processing_unit_trace;
  const unitResults = ((trace?.unit_results ?? trace?.units) ?? {}) as Record<string, Record<string, unknown>>;
  const availableArtifactIds = new Set(
    ((detail?.result?.artifacts ?? []) as Array<{ artifact_id?: string | null }>)
      .map((artifact) => String(artifact.artifact_id ?? ""))
      .filter(Boolean),
  );
  const dirtyUnitIds = new Set(recipeDiffGroups.map((group) => group.unitId));
  const lastRunDirtyUnitIds = new Set(lastRunDiffGroups.map((group) => group.unitId));
  const recipeValueUnitIds = new Set(Object.keys(recipe?.parameters_by_unit ?? {}));
  const comparisonUnits = comparison?.units ?? {};
  const planReuseUnitIds = new Set([...(partialRerunPlan?.units_to_reuse ?? []), ...(persistedTrace?.reusedUnitIds ?? [])]);
  const planRerunUnitIds = new Set(partialRerunPlan?.units_to_rerun ?? []);
  const planInvalidatedUnitIds = new Set(partialRerunPlan?.units_invalidated ?? []);
  const planBlocked = Boolean(partialRerunPlan && !partialRerunPlan.safe);
  const boundaryUnitId = partialRerunPlan?.execution_boundary_unit_id ?? persistedTrace?.boundaryStageId ?? null;
  const selectedUnitIdForRole = partialRerunPlan?.selected_unit_id ?? persistedTrace?.selectedUnitId ?? null;
  const warnings: string[] = [];
  const unitById = new Map(units.map((unit) => [unit.id, unit]));
  const nodes = units
    .slice()
    .sort((left, right) => left.order - right.order || left.id.localeCompare(right.id))
    .map((unit) => {
      const traceEntry = unitResults[unit.id] ?? null;
      const status = graphNodeStatus(unit, traceEntry, availableArtifactIds, Boolean(detail?.result));
      const comparisonUnit = comparisonUnits[unit.id];
      return {
        id: unit.id,
        label: unit.label,
        kind: unit.kind,
        stageId: unit.stage_id,
        parentId: unit.parent_id ?? undefined,
        order: unit.order,
        parameterCount: unit.parameters.length,
        artifactCount: unit.artifacts.length,
        description: unit.description,
        status: status.status,
        dirtyState: recipe ? (dirtyUnitIds.has(unit.id) ? "dirty" : "clean") : "unknown",
        lastRunDirtyState: detail?.result ? (lastRunDirtyUnitIds.has(unit.id) ? "dirty" : "clean") : "unknown",
        comparisonState: comparison ? (comparisonUnit?.has_changes ? "changed" : "unchanged") : "unknown",
        planState: planBlocked && unit.id === partialRerunPlan?.selected_unit_id
          ? "blocked"
          : planInvalidatedUnitIds.has(unit.id)
            ? "invalidated"
            : planRerunUnitIds.has(unit.id)
              ? "rerun"
              : planReuseUnitIds.has(unit.id)
                ? "reused"
                : "none",
        selectionRole: unit.id === selectedUnitIdForRole
          ? "selected"
          : unit.id === boundaryUnitId
            ? "boundary"
            : "none",
        warningCount: status.warningCount,
        missingArtifactCount: status.missingArtifactCount,
        defaultView: unit.default_view ?? null,
        inputArtifacts: unit.inputs.map((item) => item.artifact_id ?? item.id).filter(Boolean) as string[],
        outputArtifacts: unit.outputs.map((item) => item.artifact_id ?? item.id).filter(Boolean) as string[],
        artifactIds: unit.artifacts.map((artifact) => artifact.artifact_id),
        tunableParameters: unit.parameters.map((parameter) => parameter.id),
        hasRecipeValues: recipeValueUnitIds.has(unit.id),
        traceSource: typeof traceEntry?.trace_source === "string" ? traceEntry.trace_source : trace?.trace_source ?? null,
        tracePrecision: typeof traceEntry?.trace_precision === "string" ? traceEntry.trace_precision : trace?.trace_precision ?? null,
        durationMs: typeof traceEntry?.duration_ms === "number" ? traceEntry.duration_ms : null,
        parametersUsed: (traceEntry?.parameters_used && typeof traceEntry.parameters_used === "object") ? traceEntry.parameters_used as Record<string, unknown> : {},
        metricsSummary: (traceEntry?.metrics && typeof traceEntry.metrics === "object") ? traceEntry.metrics as Record<string, unknown> : {},
        diagnosticsSummary: (traceEntry?.diagnostics && typeof traceEntry.diagnostics === "object") ? traceEntry.diagnostics as Record<string, unknown> : {},
        errors: Array.isArray(traceEntry?.errors) ? traceEntry.errors.map((item) => String(item)) : [],
        warnings: Array.isArray(traceEntry?.warnings) ? traceEntry.warnings.map((item) => String(item)) : [],
        planIsStale: partialRerunPlanStale,
        comparisonSummary: comparisonUnit
          ? {
              parameterChanges: comparisonUnit.parameter_diff.filter((row) => row.status !== "unchanged").length,
              artifactChanges: comparisonUnit.artifact_diff.filter((row) => row.status !== "unchanged").length,
              metricChanges: comparisonUnit.metric_diff.filter((row) => row.status !== "unchanged").length,
              diagnosticChanges: comparisonUnit.diagnostic_diff.filter((row) => row.status !== "unchanged").length,
            }
          : undefined,
      } satisfies ProcessingUnitGraphNode;
    });

  const roots = units
    .filter((unit) => unit.kind === "stage" && !unit.parent_id)
    .slice()
    .sort((left, right) => left.order - right.order || left.id.localeCompare(right.id));
  if (!roots.length) warnings.push("No stage roots were found in the processing-unit contract.");

  const groups = roots.map((root) => ({
    id: root.id,
    label: stageLabelFromUnit(root),
    stageId: root.stage_id,
    order: root.order,
    nodeIds: units
      .filter((unit) => unit.id === root.id || unit.parent_id === root.id)
      .slice()
      .sort((left, right) => left.order - right.order || left.id.localeCompare(right.id))
      .map((unit) => unit.id),
  }));

  const edges: ProcessingUnitGraphEdge[] = [];
  for (let index = 1; index < roots.length; index += 1) {
    edges.push({
      id: `order:${roots[index - 1].id}:${roots[index].id}`,
      source: roots[index - 1].id,
      target: roots[index].id,
      kind: "order",
      confidence: "explicit",
    });
  }
  for (const root of roots) {
    const children = units
      .filter((unit) => unit.parent_id === root.id)
      .slice()
      .sort((left, right) => left.order - right.order || left.id.localeCompare(right.id));
    for (const child of children) {
      edges.push({
        id: `dependency:${root.id}:${child.id}`,
        source: root.id,
        target: child.id,
        kind: "dependency",
        confidence: "explicit",
      });
    }
    for (let index = 1; index < children.length; index += 1) {
      edges.push({
        id: `order:${children[index - 1].id}:${children[index].id}`,
        source: children[index - 1].id,
        target: children[index].id,
        kind: "order",
        confidence: "explicit",
      });
    }
  }
  for (const source of units) {
    const outputIds = new Set(source.outputs.map((output) => output.artifact_id).filter(Boolean) as string[]);
    if (!outputIds.size) continue;
    for (const target of units) {
      if (source.id === target.id) continue;
      const shared = target.inputs
        .map((input) => input.artifact_id)
        .filter((artifactId): artifactId is string => typeof artifactId === "string" && outputIds.has(artifactId));
      if (!shared.length) continue;
      edges.push({
        id: `artifact:${source.id}:${target.id}:${shared.join(",")}`,
        source: source.id,
        target: target.id,
        kind: "artifact",
        confidence: "explicit",
        label: shared[0],
      });
    }
  }
  for (const unit of units) {
    if (unit.parent_id && !unitById.has(unit.parent_id)) {
      warnings.push(`Missing parent for unit ${unit.id}: ${unit.parent_id}`);
    }
  }

  return { nodes, edges, groups, warnings };
}

export type ProcessingUnitTraceSummary = {
  total_units: number;
  runtime_traced_units: number;
  inferred_units: number;
  failed_units: number;
  warning_units: number;
  trace_coverage_percent: number;
  coverage_by_stage?: Record<string, {
    total_units: number;
    runtime_traced_units: number;
    inferred_units: number;
    failed_units: number;
    warning_units: number;
    trace_coverage_percent: number;
  }>;
};

export function traceSummaryFromDetail(detail?: TakeDetail | null): ProcessingUnitTraceSummary | null {
  const summary = detail?.result?.processing_unit_trace?.trace_summary;
  if (!summary || typeof summary !== "object") return null;
  const typed = summary as Record<string, unknown>;
  if (typeof typed.total_units !== "number") return null;
  return {
    total_units: Number(typed.total_units ?? 0),
    runtime_traced_units: Number(typed.runtime_traced_units ?? 0),
    inferred_units: Number(typed.inferred_units ?? 0),
    failed_units: Number(typed.failed_units ?? 0),
    warning_units: Number(typed.warning_units ?? 0),
    trace_coverage_percent: Number(typed.trace_coverage_percent ?? 0),
    coverage_by_stage: typeof typed.coverage_by_stage === "object" && typed.coverage_by_stage
      ? Object.fromEntries(
          Object.entries(typed.coverage_by_stage as Record<string, Record<string, unknown>>).map(([stageId, value]) => [
            stageId,
            {
              total_units: Number(value.total_units ?? 0),
              runtime_traced_units: Number(value.runtime_traced_units ?? 0),
              inferred_units: Number(value.inferred_units ?? 0),
              failed_units: Number(value.failed_units ?? 0),
              warning_units: Number(value.warning_units ?? 0),
              trace_coverage_percent: Number(value.trace_coverage_percent ?? 0),
            },
          ]),
        )
      : undefined,
  };
}

export function formatTraceCoverageLine(summary: ProcessingUnitTraceSummary | null): string | null {
  if (!summary || summary.total_units <= 0) return null;
  return `Runtime trace coverage: ${summary.runtime_traced_units} / ${summary.total_units} units, ${summary.trace_coverage_percent.toFixed(0)}%`;
}

export function traceBadgeClass(traceSource?: string | null): string {
  if (traceSource === "runtime_unit_callbacks") return "trace-badge trace-badge-runtime";
  if (traceSource === "best_effort_artifact_registry") return "trace-badge trace-badge-inferred";
  if (traceSource === "mixed") return "trace-badge trace-badge-mixed";
  return "trace-badge trace-badge-unknown";
}

export function traceBadgeLabel(traceSource?: string | null): string {
  if (traceSource === "runtime_unit_callbacks") return "runtime traced";
  if (traceSource === "best_effort_artifact_registry") return "inferred";
  if (traceSource === "mixed") return "mixed trace";
  return "unknown trace";
}
