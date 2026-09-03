import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  fileUrl,
  type CaptureModality,
  type DatasetSessionSummary,
  type DatasetSummary,
  type PipelineInfo,
  type PipelineComparisonResponse,
  type PipelineComparisonIndexEntry,
  type PipelineBatchComparisonResponse,
  type PartialRerunExecuteResponse,
  type PartialRerunPlanResponse,
  type PipelineRecipe,
  type PipelineInstance,
  type ProcessingJobRecord,
  type ProcessTemplate,
  type RuntimeState,
  type SegmentationPreviewResponse,
  type SessionSummary,
  type TakeDetail,
  type FusionRunRecord,
  type FusionRunResult,
  type FusionInputBundle,
  type FusionPreviewResult,
  type InspectionPublishedResult,
  type ProcessBinding,
  type ProcessingUnitDefinition,
  type RuntimeWorkerStatus,
  type TakeSummary,
  type PipelineRunEntry,
  type ValidationBaseline,
} from "../api/client";
import PipelineExecutionGraph from "../components/PipelineExecutionGraph";
import StudioInspector from "../components/StudioInspector";
import StageParameterPanel from "../components/StageParameterPanel";
import DetectReferenceTunePanel from "../components/DetectReferenceTunePanel";
import ContractTunePanel from "../components/ContractTunePanel";
import ProcessingUnitGraphViewer from "../components/ProcessingUnitGraphViewer";
import ProcessingUnitGraphWorkspace from "../components/ProcessingUnitGraphWorkspace";
import StageSubstageTree from "../components/StageSubstageTree";
import { StageSubstageRailList, StageSubstageDetailStrip } from "../components/StageSubstageRail";
import InfoHint from "../components/InfoHint";
import SelectedSurfaceRefinementCard from "../components/SelectedSurfaceRefinementCard";
import type { OverlayDebugInfo } from "../components/overlayModel";
import { resolveOverlayTarget } from "../components/overlayModel";
import {
  chooseBestPipelineForTake,
  incompatibleModalityMessage,
  modalityCompatible,
  stageTimingMs,
  type StageViewTabId
} from "../components/pipelineModel";
import { artifactLineage, artifactList, artifactsForObject, artifactsForStage, objectsForTake, selectedObject, stageInspectorSummary, type StudioArtifact } from "../components/studioWorkspaceModel";
import { renderStageSemanticView } from "../components/stage_view_renderers";
import { canonicalStageId, stageSemanticDefinition, type StageViewDefinition } from "../components/stageSemantics";
import { resolveStageDisplayPlan, type DisplayRole } from "../components/stageDisplayPlan";
import {
  artifactRelevanceLabel,
  artifactRelevanceTone,
  blobComponentModeLabel,
  blobSplitMethodLabel,
  detectReferencePresetById,
  detectReferencePresetForValues,
  detectReferenceFormDiffersFromRun,
  normalizeBlobSplitMethod,
  referencePresetLabel,
  referenceSurfaceModelLabel,
  resolveDetectReferenceArtifactRelevance,
  resolveDetectReferenceRunState,
  resolveDetectViewArtifacts,
  suppressionMaskPolicyLabel,
  supportSelectionMethodLabel,
  type ArtifactRelevance,
} from "../components/referenceArtifactRelevance";
import type { RoiPolygon, RoiRect } from "../components/roiMapping";
import type { RoiType } from "../components/ProjectionArtifactViewer";
import type { HoverInspectionSample } from "../components/hoverInspectionModel";
import { buildCapturePayload, captureDisabledReason, nextSelectedTakeAfterCapture } from "../components/captureWorkflowModel";
import { resolveNextSelectedTakeId } from "../components/takeSelectionModel";
import { buildStudioDeepLink, graphWorkspaceEnabled, parseStudioDeepLink, type StudioDeepLinkParams } from "../studioDeepLink";
import {
  availableBinaryMaskViewModes,
  defaultBinaryMaskViewMode,
  resolveBinaryMaskReference,
  resolveBinaryMaskReferenceCandidates,
} from "../components/binaryMaskArtifacts";
import { shouldUseBinaryMaskRenderer } from "../components/binaryMaskArtifacts";
import { sourceArtifactsForTake } from "../components/stage_sources";
import { preferredViewportModeForArtifact, type StudioArtifactViewState } from "../components/studioViewState";
import {
  resolveReferenceViewSelection,
  resolveStageSubstagePlan,
  type ArtifactIoRole,
  type ReferenceViewScope,
  type ResolvedStageSubstage,
} from "../components/stageSubstagePlan";
import {
  processingUnitsForPipeline,
  processingUnitById,
  processingUnitsForStage,
  resolveProcessingUnitArtifactRefs,
  resolveProcessingUnitView,
  rootProcessingUnit,
} from "../components/processingUnitModel";
import { buildProcessingUnitGraph, formatTraceCoverageLine, graphSelectionForUnit, persistedTraceFromDetail, traceSummaryFromDetail } from "../components/processingUnitGraphModel";
import RunHistoryPanel from "../components/RunHistoryPanel";
import RunLineageBlock from "../components/RunLineageBlock";
import PartialRunNoticeBanner from "../components/PartialRunNoticeBanner";
import { isPartialRerunExecutionMode, partialRunNoticeText } from "../components/partialRunNoticeModel";
import { computeStaleWarnings, type PlanContext } from "../components/staleWarningsModel";
import { formatPartialExecutionSummaryLine } from "../components/partialExecutionSummaryModel";
import {
  buildParameterDiffGroups,
  buildRecipeDiff,
  recipeDirty,
  recipeStageParamsToFlat25d,
  recipeValuesForUnit,
  stageParamsToRecipeParametersByUnit,
} from "../components/recipeModel";
import {
  comparisonSourceLabel,
  formatChangeSummary,
  formatComparisonSummaryLine,
  orderedComparisonUnits,
} from "../components/comparisonModel";
import {
  detectReferenceArtifactRoleHelpKey,
  detectReferenceDisplayRoleHelpKey,
  detectReferenceSubstageHelpKey,
  detectReferenceViewHelpKey,
  resolveStudioHelpEntry,
} from "../components/studioHelp";

type TakeFilter = "all" | "processed" | "unprocessed" | "warnings";
type SegmentationTuningParams = Record<string, unknown>;
type PlaneQaTuningParams = Record<string, unknown>;
type RoiEditStatus = "idle" | "active" | "unsaved" | "saved" | "rerun_required";
type StudioMode = "inspect" | "tune" | "compare" | "debug";
type ControlPanelTab = "inspect" | "tune" | "view" | "graph" | "compare" | "actions" | "debug";
type ScaleCorrection = { x: number; y: number; z: number };
type CompareSourceType = "run" | "recipe" | "current";
const SENSOR_STUDIO_25D_DEFAULTS_KEY = "sensor_studio.25d.stage_params.defaults.v1";
const SENSOR_STUDIO_SHOW_ALL_VIEWS_KEY = "sensor_studio.stage_workspace.show_all_views.v1";

type WorkspaceViewEntry = {
  id: string;
  label: string;
  reason?: string;
  relevance?: ArtifactRelevance;
};

type WorkspaceViewSplit = {
  primary: WorkspaceViewEntry[];
  secondary: WorkspaceViewEntry[];
};

type SourceContextBanner = {
  label: string;
  unresolved: boolean;
  analyticsHref: string | null;
};

function contractRuntimeStageParamsKey(stageId: string): string {
  if (stageId === "measurement_diagnostics") return "known_object_25d";
  return stageId;
}

function contractParamFlatKey(stageId: string, paramId: string): string {
  if (stageId === "remove_belt_segment_objects") {
    if (paramId === "min_height_mm") return "object_min_height_mm";
    if (paramId === "max_height_mm") return "object_max_height_mm";
  }
  if (stageId === "measurement_diagnostics") {
    if (paramId === "enabled") return "known_object_enabled";
    if (paramId === "apply_correction") return "known_object_apply_correction";
    if (paramId === "target_selection") return "known_object_target_selection";
    if (paramId === "manual_component_id") return "known_object_manual_component_id";
    if (paramId === "tolerance_percent") return "known_tolerance_percent";
  }
  return paramId;
}

function stableValueString(value: unknown): string {
  if (Array.isArray(value)) return JSON.stringify(value);
  if (value && typeof value === "object") return JSON.stringify(value);
  return JSON.stringify(value ?? null);
}

function planeQaParamsEqual(a: Record<string, unknown> | null, b: Record<string, unknown> | null): boolean {
  const left = a ?? {};
  const right = b ?? {};
  const keys = new Set([...Object.keys(left), ...Object.keys(right)]);
  for (const key of keys) {
    if (stableValueString(left[key]) !== stableValueString(right[key])) return false;
  }
  return true;
}

function formatComparisonValue(value: unknown): string {
  if (value == null) return "null";
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function formatPercent(value: number | null | undefined, digits = 1): string {
  if (value == null || !Number.isFinite(value)) return "-";
  return `${value.toFixed(digits)}%`;
}

function formatRatio(value: number | null | undefined, digits = 3): string {
  if (value == null || !Number.isFinite(value)) return "-";
  return value.toFixed(digits);
}

function read25dSavedDefaults(): Record<string, unknown> {
  if (typeof window === "undefined") return {};
  try {
    const raw = window.localStorage.getItem(SENSOR_STUDIO_25D_DEFAULTS_KEY);
    return raw ? JSON.parse(raw) as Record<string, unknown> : {};
  } catch {
    return {};
  }
}

function readShowAllStageViews(): boolean {
  if (typeof window === "undefined") return false;
  return window.localStorage.getItem(SENSOR_STUDIO_SHOW_ALL_VIEWS_KEY) === "true";
}

function scaleCorrectionFromRecord(record: Record<string, unknown> | null | undefined): ScaleCorrection | null {
  if (!record) return null;
  const knownX = Number(record.known_width_mm);
  const knownY = Number(record.known_depth_mm);
  const knownZ = Number(record.known_height_mm);
  const measuredX = Number(record.measured_width_mm);
  const measuredY = Number(record.measured_depth_mm);
  const measuredZ = Number(record.measured_height_mm);
  if (
    Number.isFinite(knownX) && Number.isFinite(knownY) && Number.isFinite(knownZ) &&
    Number.isFinite(measuredX) && Number.isFinite(measuredY) && Number.isFinite(measuredZ) &&
    measuredX > 1e-9 && measuredY > 1e-9 && measuredZ > 1e-9
  ) {
    return { x: knownX / measuredX, y: knownY / measuredY, z: knownZ / measuredZ };
  }
  const x = Number(record.recommended_scale_correction_x ?? record.persisted_scale_correction_x ?? record.x);
  const y = Number(record.recommended_scale_correction_y ?? record.persisted_scale_correction_y ?? record.y);
  const z = Number(record.recommended_scale_correction_z ?? record.persisted_scale_correction_z ?? record.z);
  return Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z) ? { x, y, z } : null;
}

function isIdentityScaleCorrection(scale: ScaleCorrection | null): boolean {
  if (!scale) return false;
  return Math.abs(scale.x - 1) < 1e-6 && Math.abs(scale.y - 1) < 1e-6 && Math.abs(scale.z - 1) < 1e-6;
}

function persistedScaleFromKnown(known: Record<string, unknown>): ScaleCorrection | null {
  return scaleCorrectionFromRecord({
    persisted_scale_correction_x: known.persisted_scale_correction_x,
    persisted_scale_correction_y: known.persisted_scale_correction_y,
    persisted_scale_correction_z: known.persisted_scale_correction_z,
  });
}

function studioDeepLinkObjectId(value: string | null | undefined): number | null {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
}

function resolveStudioStageId(pipeline: PipelineInfo | null, requestedStage: string | null | undefined): string | null {
  const stage = String(requestedStage ?? "").trim();
  if (!pipeline || !stage) return null;
  const canonicalRequested = canonicalStageId(stage, stage);
  const exact = pipeline.stages.find((item) => item.id === stage);
  if (exact) return exact.id;
  const compatible = pipeline.stages.find((item) => canonicalStageId(item.id, item.display_name ?? null) === canonicalRequested);
  return compatible?.id ?? null;
}

function buildStudioDeepLinkMessage(params: StudioDeepLinkParams, objectResolved: boolean): string {
  const details: string[] = [];
  if (params.feature) details.push(params.feature);
  if (params.stage) details.push(`${params.stage} stage`);
  if (params.physical_object_id) details.push(`physical object ${params.physical_object_id}`);
  if ((params.object_id || params.physical_object_id) && !objectResolved) details.push("object context unresolved");
  return details.length ? `Opened from Feature Analytics: ${details.join(" · ")}` : "Opened from Feature Analytics.";
}

type ReferenceStrategy =
  | "low_gradient_depth_plateaus"
  | "low_gradient_bg_and_stripes"
  | "low_gradient_blob_height_clusters"
  | "low_gradient_surface"
  | "percentile_or_legacy"
  | "unknown";

type DetectViewGroup = {
  id: string;
  label: string;
  collapsible?: boolean;
  views: StageViewDefinition[];
};

type DetectResolvedView = {
  view: StageViewDefinition;
  relevance: "active" | "debug" | "inactive_strategy" | "legacy" | "fallback" | "stale" | "missing";
  reason: string;
};

type DetectResolvedViewGroup = {
  id: string;
  label: string;
  collapsible?: boolean;
  views: DetectResolvedView[];
};

type DetectArtifactAuditRow = {
  viewId: string;
  viewLabel: string;
  resolvedArtifact: string;
  relevance: string;
  reason: string;
  provenance: string;
  strategy: string;
  componentMode: string;
  splitMethod: string;
};

type DetectStrategySummary = {
  strategy: ReferenceStrategy;
  strategyLabel: string;
  presetLabel: string | null;
  supportLabel: string;
  componentLabel: string | null;
  splitLabel: string | null;
  referenceModelLabel: string;
  resolvedReferenceModelLabel: string;
  suppressionLabel: string;
  detailsAvailable: boolean;
  selectedClusterId: string | null;
  selectedBlobCount: number | null;
  totalBlobCount: number | null;
  fragmentCount: number | null;
  selectedFragmentCount: number | null;
  splitMethod: string | null;
  componentMode: string | null;
  blobConnectivity: number | null;
  neighborZToleranceMm: number | null;
  rejectedNeighborEdges: number | null;
  eligibleBlobCount: number | null;
  heightBorderSplitBlobCount: number | null;
  mergedFragmentCount: number | null;
  fallbackUsed: boolean | null;
  fallbackStrategy: string | null;
  fallbackReason: string | null;
  selectedClusterRatio: number | null;
  selectedSupportRatio: number | null;
  referenceModelSupportRatio: number | null;
  referenceSuppressionRatio: number | null;
  largestSupportLossStep: string | null;
  largestSupportLossFraction: number | null;
  stripeRemovedRatio: number | null;
  stripeFilterEnabled: boolean | null;
  selectedPlateauText: string | null;
  plateauMode: string | null;
};

type StageTabEntry = {
  id: StageViewTabId;
  label: string;
  hiddenReason?: string;
  relevance?: ArtifactRelevance;
};

function ArtifactRelevanceBadge({ relevance, reason }: { relevance?: ArtifactRelevance; reason?: string }) {
  if (!relevance || relevance === "active") return null;
  const tone = artifactRelevanceTone(relevance);
  const title = reason ? `${artifactRelevanceLabel(relevance)}: ${reason}` : artifactRelevanceLabel(relevance);
  return (
    <span className={`artifact-relevance-badge tone-${tone}`} title={title}>
      {artifactRelevanceLabel(relevance)}
    </span>
  );
}

function IoRoleTag({ ioRole }: { ioRole?: ArtifactIoRole }) {
  if (ioRole !== "input" && ioRole !== "output") return null;
  const title = ioRole === "input" ? "Consumed from an earlier substage" : "Used by a later step";
  return (
    <span className={`artifact-io-role-tag role-${ioRole}`} title={title}>
      {ioRole === "input" ? "in" : "out"}
    </span>
  );
}

function substageRelevanceSummary(substage: ResolvedStageSubstage): ArtifactRelevance | null {
  if (substage.views.length === 0) return null;
  const allInactive = substage.views.every((entry) => entry.category === "inactive_strategy" || entry.category === "legacy");
  if (allInactive) return substage.views[0].category;
  const allMissing = substage.views.every((entry) => entry.category === "missing");
  if (allMissing) return "missing";
  return null;
}

type ShellLayoutMode = "wide" | "medium" | "narrow" | "very_narrow";

function normalizeReferenceStrategy(raw: unknown): ReferenceStrategy {
  const value = String(raw ?? "").trim().toLowerCase();
  if (value === "low_gradient_depth_plateaus") return "low_gradient_depth_plateaus";
  if (value === "low_gradient_blob_height_clusters") return "low_gradient_blob_height_clusters";
  if (value === "low_gradient_surface") return "low_gradient_surface";
  if (
    value === "nearest_percentile"
    || value === "farthest_percentile"
    || value === "automatic"
    || value === "depth_percentile_plane"
    || value === "percentile"
  ) return "percentile_or_legacy";
  return "unknown";
}

function referenceStrategyLabel(strategy: ReferenceStrategy): string {
  if (strategy === "low_gradient_blob_height_clusters") return "Low-gradient blobs + height clusters";
  if (strategy === "low_gradient_depth_plateaus") return "Low-gradient depth plateaus";
  if (strategy === "low_gradient_bg_and_stripes") return "Low-gradient background + stripes";
  if (strategy === "low_gradient_surface") return "Low-gradient surface";
  if (strategy === "percentile_or_legacy") return "Percentile / legacy";
  return "Unknown";
}

function detectComponentModeLabel(value: string | null): string {
  if (value === "height_aware") return "height-aware";
  if (value === "xy_only") return "XY-only";
  return value ?? "-";
}

function compactStageLabel(stageId: string, fallback: string): string {
  const canonical = canonicalStageId(stageId, fallback);
  if (canonical === "load_heightmap") return "Input";
  if (canonical === "detect_belt_plane") return "Reference";
  if (canonical === "normalize_heights_to_plane") return "Normalize";
  if (canonical === "remove_belt_segment_objects") return "Segment";
  if (canonical === "fit_object_geometry") return "Geometry";
  if (canonical === "compute_height_metrics") return "Measure";
  if (canonical === "measurement_diagnostics") return "Diagnostics";
  if (canonical === "classify_25d") return "Classify";
  if (canonical === "overlays_25d") return "Overlay";
  return fallback;
}

function compactDetectViewLabel(viewId: string, fallback: string): string {
  if (viewId === "raw_heightmap") return "Raw";
  if (viewId === "depth_gradient") return "Gradient";
  if (viewId === "low_gradient_mask") return "Low-gradient";
  if (viewId === "selected_blob_cluster") return "Selected blob cluster";
  if (viewId === "selected_support_lineage") return "Selected support lineage";
  if (viewId === "selected_surface") return "Selected surface";
  if (viewId === "plane_inliers") return "Plane inliers";
  if (viewId === "residual_heatmap") return "Residuals";
  return fallback;
}

function shouldShowSelectedSurfaceRefinement(activeTab: string, focusedArtifact: StudioArtifact | null): boolean {
  if (activeTab === "selected_blob_cluster" || activeTab === "selected_support_lineage" || activeTab === "selected_surface") return true;
  const artifactId = String(focusedArtifact?.artifact_id ?? "");
  return artifactId === "selected_blob_cluster_overlay"
    || artifactId === "selected_blob_cluster_mask"
    || artifactId === "selected_blob_cluster_pre_refine_mask"
    || artifactId === "selected_blob_cluster_refined_mask"
    || artifactId === "selected_support_lineage"
    || artifactId === "selected_reference_support_mask"
    || artifactId === "reference_surface_selected_mask";
}

function displayRoleLabel(role: DisplayRole): string {
  if (role === "candidate_signal") return "Candidate signal";
  if (role === "candidate_filter") return "Candidate filter";
  if (role === "strategy_intermediate") return "Strategy path";
  if (role === "selected_reference_surface") return "Selected reference surface";
  if (role === "fit_support") return "Fit support";
  return role.replace(/_/g, " ");
}

function detectArtifactRoleLabel(role: unknown): string {
  const value = String(role ?? "");
  if (value === "logical_support") return "Logical support";
  if (value === "fit_support") return "Fit support";
  if (value === "suppression_mask") return "Suppression mask";
  if (value === "connectivity_support") return "Connectivity support";
  if (value === "diagnostic") return "Diagnostic";
  if (value === "final") return "Final output";
  if (value === "intermediate") return "Intermediate";
  if (value === "input") return "Input";
  return value ? value.replace(/_/g, " ") : "Unclassified";
}

function detectSubstageLabel(substageId: unknown): string {
  const value = String(substageId ?? "");
  if (!value) return "Unknown";
  return value.replace(/_/g, " ");
}

function ratioToPercentLabel(value: number | null | undefined, digits = 1): string | null {
  if (value == null || !Number.isFinite(value)) return null;
  return `${(value * 100).toFixed(digits)}%`;
}

function buildSourceContextBanner(params: StudioDeepLinkParams, objectResolved: boolean): SourceContextBanner | null {
  if (!params.feature && !params.stage && !params.physical_object_id && !params.object_id) return null;
  const parts = [
    "Feature Analytics",
    params.feature,
    params.physical_object_id ?? params.object_id,
  ].filter((value): value is string => Boolean(value));
  return {
    label: parts.join(" -> "),
    unresolved: Boolean((params.object_id || params.physical_object_id) && !objectResolved),
    analyticsHref: "/feature-analytics",
  };
}

function workspaceTabLabel(tab: ControlPanelTab, detectMode: boolean): string {
  if (!detectMode) return tab[0].toUpperCase() + tab.slice(1);
  if (tab === "inspect") return "Explain";
  if (tab === "tune") return "Tune";
  if (tab === "view") return "Artifacts";
  if (tab === "debug") return "JSON";
  return tab[0].toUpperCase() + tab.slice(1);
}

function shouldShowWorkspaceTab(tab: ControlPanelTab, detectMode: boolean): boolean {
  if (!detectMode) return ["inspect", "tune", "view", "graph", "compare", "actions", "debug"].includes(tab);
  return tab === "inspect" || tab === "tune" || tab === "view" || tab === "debug";
}

function resolveDetectWorkspaceViewGroups(args: {
  activeSubstage: ResolvedStageSubstage | null;
  activeReferenceViewScope: ReferenceViewScope;
  stageRootUnit: ProcessingUnitDefinition | null;
  activeProcessingUnitView: ReturnType<typeof resolveProcessingUnitView> | null;
  stageDisplayPlan: ReturnType<typeof resolveStageDisplayPlan>;
}): WorkspaceViewSplit {
  const { activeSubstage, activeReferenceViewScope, stageRootUnit, activeProcessingUnitView, stageDisplayPlan } = args;
  const toEntry = (view: StageViewDefinition, reason?: string, relevance?: ArtifactRelevance) => ({
    id: view.id,
    label: compactDetectViewLabel(view.id, view.label),
    reason,
    relevance,
  });

  if (activeSubstage) {
    const orderedViews = activeSubstage.views.map((entry) => toEntry(entry.view, entry.reason, entry.category));
    const primary = orderedViews.filter((entry) => entry.relevance === "active" || entry.relevance === "fallback");
    const secondary = orderedViews.filter((entry) => entry.relevance !== "active" && entry.relevance !== "fallback");
    return { primary: primary.length ? primary : orderedViews, secondary: primary.length ? secondary : [] };
  }

  if (activeReferenceViewScope === "strategy_path" && stageRootUnit) {
    const rootViews = stageRootUnit.views.map((view) => toEntry(
      stageDisplayPlan.items.find((item) => item.view.id === view.id)?.view ?? {
        id: view.id,
        label: view.label,
        rendererType: "json",
      },
      activeProcessingUnitView?.viewId === view.id && activeProcessingUnitView?.fallbackUsed ? activeProcessingUnitView.fallbackReason : undefined,
      "active",
    ));
    const secondary = [
      ...stageDisplayPlan.items
        .filter((item) => !rootViews.some((entry: { id: string }) => entry.id === item.view.id))
        .map((item) => toEntry(item.view, item.reason, item.category)),
      ...stageDisplayPlan.supportingViews.map((item) => toEntry(item.view, item.reason, item.category)),
      ...stageDisplayPlan.debugViews.map((item) => toEntry(item.view, item.reason, item.category)),
    ];
    return { primary: rootViews, secondary };
  }

  return { primary: [], secondary: [] };
}

function controlPanelTabToMode(tab: ControlPanelTab): StudioMode {
  if (tab === "tune") return "tune";
  if (tab === "compare") return "compare";
  if (tab === "debug") return "debug";
  return "inspect";
}

function resolveDetectArtifactAuditRows(
  allViews: StageViewDefinition[],
  stageArtifacts: Array<{ artifact_id?: string; metadata?: Record<string, unknown> | null; created_at?: string | null }>,
  detail: TakeDetail | null,
  formState: Record<string, unknown> | null,
): DetectArtifactAuditRow[] {
  const resolvedArtifacts = stageArtifacts as StudioArtifact[];
  const state = resolveDetectReferenceRunState(detail, resolvedArtifacts, formState);
  return allViews.map((view) => {
    const relevance = resolveDetectReferenceArtifactRelevance({
      viewId: view.id,
      artifacts: resolvedArtifacts,
      activeStrategy: state.strategy,
      componentMode: state.componentMode,
      splitMethod: state.splitMethod,
      fallbackUsed: state.fallbackUsed,
      fallbackStrategy: state.fallbackStrategy,
      runUpdatedAt: state.processedAt,
    });
    const resolution = resolveDetectViewArtifacts(view.id, resolvedArtifacts, state);
    const resolvedArtifact = resolution.primary?.artifact_id ?? (relevance.relevance === "active" || relevance.relevance === "fallback" ? "renderer default" : "-");
    const metadata = (resolution.primary?.metadata ?? {}) as Record<string, unknown>;
    const provenanceBits = [
      metadata.run_id,
      metadata.stage_execution_id,
      metadata.pipeline_execution_id,
      resolution.primary?.created_at,
      metadata.created_at,
    ].filter((value) => value != null && String(value).trim().length > 0).map((value) => String(value));
    return {
      viewId: view.id,
      viewLabel: view.label,
      resolvedArtifact,
      relevance: relevance.relevance,
      reason: relevance.reason,
      provenance: provenanceBits[0] ?? "legacy / unverified provenance",
      strategy: String(metadata.background_detection_strategy ?? metadata.strategy ?? state.strategy),
      componentMode: String(metadata.blob_component_mode ?? metadata.component_mode ?? state.componentMode),
      splitMethod: String(metadata.blob_split_method ?? metadata.split_method ?? state.splitMethod),
    };
  });
}

function detectStrategySummary(
  detail: TakeDetail | null,
  stageArtifacts: Array<{ artifact_id?: string; metadata?: Record<string, unknown> | null }>,
  formState: Record<string, unknown> | null,
): DetectStrategySummary {
  const runState = resolveDetectReferenceRunState(detail, stageArtifacts as StudioArtifact[], formState);
  const fit = ((stageArtifacts.find((item) => item.artifact_id === "plane_fit_debug")?.metadata ?? {}) as Record<string, unknown>);
  const selection = (((fit.background_selection as Record<string, unknown> | undefined) ?? {}) as Record<string, unknown>);
  const gradientDebug = (((selection.gradient_debug as Record<string, unknown> | undefined) ?? {}) as Record<string, unknown>);
  const blobDebug = ((stageArtifacts.find((item) => item.artifact_id === "blob_cluster_selection_debug")?.metadata ?? {}) as Record<string, unknown>);
  const selectedPlateau = (gradientDebug.selected_plateau as Record<string, unknown> | undefined) ?? null;
  const selectedSurfaceDebug = ((stageArtifacts.find((item) => item.artifact_id === "selected_surface_debug")?.metadata ?? {}) as Record<string, unknown>);
  const lineageSummary = ((selectedSurfaceDebug.reference_support_lineage as Record<string, unknown> | undefined) ?? {});
  const supportLossArtifact = stageArtifacts.find((item) => item.artifact_id === "support_loss_waterfall");
  const supportLossRows = Array.isArray((supportLossArtifact?.metadata ?? {}).rows)
    ? (((supportLossArtifact?.metadata ?? {}) as Record<string, unknown>).rows as Array<Record<string, unknown>>)
    : [];
  const supportLossById = new Map(
    supportLossRows
      .filter((row) => row && row.row_id != null)
      .map((row) => [String(row.row_id), row] as const),
  );
  const largestWaterfallLoss = supportLossRows
    .filter((row) => Number.isFinite(Number(row.delta_percent_valid_roi_from_previous)) && Number(row.delta_percent_valid_roi_from_previous) < 0)
    .sort((a, b) => Number(a.delta_percent_valid_roi_from_previous) - Number(b.delta_percent_valid_roi_from_previous))[0] ?? null;
  const zMin = Number(selectedPlateau?.z_min);
  const zMax = Number(selectedPlateau?.z_max);
  const selectedPlateauText = Number.isFinite(zMin) && Number.isFinite(zMax)
    ? `z ${zMin.toFixed(1)}-${zMax.toFixed(1)} mm`
    : null;
  const selectedBlobIds = Array.isArray(gradientDebug.selected_blob_ids)
    ? gradientDebug.selected_blob_ids.map((item) => String(item)).filter(Boolean)
    : [];
  const selectedSourceBlobIds = Array.isArray(gradientDebug.selected_source_blob_ids)
    ? gradientDebug.selected_source_blob_ids.map((item) => String(item)).filter(Boolean)
    : [];
  const stripeInfo = ((gradientDebug.belt_stripe_filter as Record<string, unknown> | undefined) ?? {});
  const selectedClusterRatio = Number(
    supportLossById.get("selected_height_cluster")?.percent_valid_roi
      ?? supportLossById.get("selected_cluster_pre_refine")?.percent_valid_roi
      ?? lineageSummary.selected_cluster_valid_roi_ratio,
  );
  const selectedSupportRatio = Number(gradientDebug.selected_component_area_ratio);
  const selectedReferenceSupportRatio = Number(
    supportLossById.get("final_selected_support")?.percent_valid_roi
      ?? lineageSummary.selected_reference_support_valid_roi_ratio,
  );
  const referenceModelSupportRatio = Number(
    supportLossById.get("reference_model_fit_support")?.percent_valid_roi
      ?? supportLossById.get("plane_inliers")?.percent_valid_roi
      ?? lineageSummary.reference_model_support_valid_roi_ratio,
  );
  const referenceSuppressionRatio = Number(
    supportLossById.get("reference_suppression_mask")?.percent_valid_roi
      ?? lineageSummary.reference_suppression_valid_roi_ratio,
  );
  const blobCount = Number(gradientDebug.blob_count);
  const fragmentCount = Number(gradientDebug.height_consistent_fragment_count);
  const formBlobConnectivity = formState == null ? Number.NaN : Number(formState.blob_connectivity);
  const formNeighborZToleranceMm = formState == null ? Number.NaN : Number(formState.blob_neighbor_z_tolerance_mm);
  const selectedClusterId = gradientDebug.selected_cluster_id == null
    ? null
    : String(gradientDebug.selected_cluster_id);
  const fallbackUsed = typeof gradientDebug.fallback_used === "boolean"
    ? Boolean(gradientDebug.fallback_used)
    : null;
  const stripeFilterEnabled = typeof stripeInfo.enabled === "boolean"
    ? Boolean(stripeInfo.enabled)
    : (formState?.belt_stripe_filter_enabled == null ? null : Boolean(formState.belt_stripe_filter_enabled));
  const stripeRemovedRatio = Number(
    supportLossById.get("support_removed_by_stripe_filter")?.percent_valid_roi
      ?? lineageSummary.support_removed_by_stripe_filter_valid_roi_ratio,
  );
  const detailsAvailable = Boolean(
    Object.keys(gradientDebug).length
    || Object.keys(blobDebug).length
    || selectedPlateauText
    || selectedBlobIds.length
    || selectedClusterId
    || supportLossRows.length
  );
  return {
    strategy: runState.strategy,
    strategyLabel: referenceStrategyLabel(runState.strategy),
    presetLabel: referencePresetLabel(runState.presetId),
    supportLabel: supportSelectionMethodLabel(runState.supportSelectionMethod),
    componentLabel: runState.strategy === "low_gradient_blob_height_clusters" ? blobComponentModeLabel(runState.componentMode) : null,
    splitLabel: runState.strategy === "low_gradient_blob_height_clusters" ? blobSplitMethodLabel(runState.splitMethod) : null,
    referenceModelLabel: referenceSurfaceModelLabel(runState.referenceSurfaceModel),
    resolvedReferenceModelLabel: referenceSurfaceModelLabel(runState.resolvedReferenceSurfaceModel),
    suppressionLabel: suppressionMaskPolicyLabel(runState.suppressionMaskPolicy),
    detailsAvailable,
    selectedClusterId,
    selectedBlobCount: selectedSourceBlobIds.length || selectedBlobIds.length || null,
    totalBlobCount: Number.isFinite(blobCount) ? blobCount : null,
    fragmentCount: Number.isFinite(fragmentCount) ? fragmentCount : null,
    selectedFragmentCount: selectedBlobIds.length || null,
    splitMethod: runState.splitMethod === "unknown" ? null : runState.splitMethod,
    componentMode: runState.componentMode === "unknown" ? null : runState.componentMode,
    blobConnectivity: Number.isFinite(Number(gradientDebug.blob_connectivity))
      ? Number(gradientDebug.blob_connectivity)
      : (Number.isFinite(formBlobConnectivity) ? formBlobConnectivity : null),
    neighborZToleranceMm: Number.isFinite(Number(gradientDebug.blob_neighbor_z_tolerance_mm))
      ? Number(gradientDebug.blob_neighbor_z_tolerance_mm)
      : (Number.isFinite(formNeighborZToleranceMm) ? formNeighborZToleranceMm : null),
    rejectedNeighborEdges: Number.isFinite(Number(gradientDebug.rejected_neighbor_edges))
      ? Number(gradientDebug.rejected_neighbor_edges)
      : null,
    eligibleBlobCount: Number.isFinite(Number(gradientDebug.eligible_blob_count)) ? Number(gradientDebug.eligible_blob_count) : null,
    heightBorderSplitBlobCount: Number.isFinite(Number(gradientDebug.height_border_split_blob_count)) ? Number(gradientDebug.height_border_split_blob_count) : null,
    mergedFragmentCount: Number.isFinite(Number(gradientDebug.merged_fragment_count)) ? Number(gradientDebug.merged_fragment_count) : null,
    fallbackUsed,
    fallbackStrategy: gradientDebug.fallback_strategy == null ? null : String(gradientDebug.fallback_strategy),
    fallbackReason: gradientDebug.fallback_reason == null ? null : String(gradientDebug.fallback_reason),
    selectedClusterRatio: Number.isFinite(selectedClusterRatio) ? selectedClusterRatio : null,
    selectedSupportRatio: Number.isFinite(selectedReferenceSupportRatio) ? selectedReferenceSupportRatio : (Number.isFinite(selectedSupportRatio) ? selectedSupportRatio : null),
    referenceModelSupportRatio: Number.isFinite(referenceModelSupportRatio) ? referenceModelSupportRatio : null,
    referenceSuppressionRatio: Number.isFinite(referenceSuppressionRatio) ? referenceSuppressionRatio : null,
    largestSupportLossStep: largestWaterfallLoss?.row_id != null
      ? String(largestWaterfallLoss.row_id)
      : (lineageSummary.largest_support_loss_step == null ? null : String(lineageSummary.largest_support_loss_step)),
    largestSupportLossFraction: Number.isFinite(Math.abs(Number(largestWaterfallLoss?.delta_percent_valid_roi_from_previous)))
      ? Math.abs(Number(largestWaterfallLoss?.delta_percent_valid_roi_from_previous))
      : (Number.isFinite(Number(lineageSummary.largest_support_loss_fraction)) ? Number(lineageSummary.largest_support_loss_fraction) : null),
    stripeRemovedRatio: Number.isFinite(stripeRemovedRatio) ? stripeRemovedRatio : null,
    stripeFilterEnabled,
    selectedPlateauText,
    plateauMode: gradientDebug.plateau_parameters && typeof gradientDebug.plateau_parameters === "object"
      ? String((gradientDebug.plateau_parameters as Record<string, unknown>).selection_mode ?? "")
      : null,
  };
}

function detectViewGroups(strategy: ReferenceStrategy, views: StageViewDefinition[]): DetectViewGroup[] {
  const byId = new Map(views.map((view) => [view.id, view]));
  const pick = (ids: string[]) => ids.map((id) => byId.get(id)).filter((view): view is StageViewDefinition => Boolean(view));
  const groups: DetectViewGroup[] = [];
  if (strategy === "low_gradient_blob_height_clusters") {
    groups.push({
      id: "main_workflow",
      label: "Main workflow",
      views: pick([
        "raw_heightmap",
        "valid_mask",
        "depth_gradient",
        "low_gradient_mask",
        "blob_components",
        "height_borders",
        "height_border_fragments",
        "blob_clusters",
        "selected_blob_cluster",
        "rejected_blob_clusters",
        "blob_cluster_scores",
        "selected_surface",
        "plane_inliers",
        "residual_heatmap",
        "diagnostics",
        "json",
      ]),
    });
    groups.push({
      id: "blob_split_advanced",
      label: "Blob splitting (advanced)",
      collapsible: true,
      views: pick([
        "height_aware_blob_components",
        "height_aware_connectivity_rejected_edges",
        "height_borders",
        "height_border_fragments",
        "height_split_blobs",
      ]),
    });
    groups.push({
      id: "belt_stripes",
      label: "Belt stripe filter",
      collapsible: true,
      views: pick([
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
      ]),
    });
    groups.push({
      id: "legacy_plateau",
      label: "Legacy / plateau artifacts",
      collapsible: true,
      views: pick([
        "surface_candidates",
        "depth_plot",
        "plateau_plot",
        "filtered_depth_plot",
        "residual_histogram",
      ]),
    });
    return groups.filter((group) => group.views.length > 0);
  }
  if (strategy === "low_gradient_depth_plateaus") {
    groups.push({
      id: "main_workflow",
      label: "Main workflow",
      views: pick([
        "raw_heightmap",
        "valid_mask",
        "depth_gradient",
        "low_gradient_mask",
        "surface_candidates",
        "selected_surface",
        "plane_inliers",
        "residual_heatmap",
        "residual_histogram",
        "diagnostics",
        "depth_plot",
        "plateau_plot",
        "filtered_depth_plot",
        "belt_base",
        "belt_stripes",
        "json",
      ]),
    });
    groups.push({
      id: "belt_stripes",
      label: "Belt stripe filter",
      collapsible: true,
      views: pick([
        "belt_baseline",
        "belt_altitude_plot",
        "belt_stripes_tophat",
        "belt_stripes_shape",
        "belt_wide_object",
        "belt_above_belt",
        "belt_base",
        "belt_stripes",
      ]),
    });
    groups.push({
      id: "blob_artifacts",
      label: "Blob-cluster artifacts",
      collapsible: true,
      views: pick([
        "blob_components",
        "blob_clusters",
        "selected_blob_cluster",
        "rejected_blob_clusters",
        "blob_cluster_scores",
      ]),
    });
    return groups.filter((group) => group.views.length > 0);
  }
  groups.push({
    id: "main_workflow",
    label: "Main workflow",
    views: pick([
      "raw_heightmap",
      "valid_mask",
      "depth_gradient",
      "low_gradient_mask",
      "surface_candidates",
      "selected_surface",
      "plane_inliers",
      "residual_heatmap",
      "diagnostics",
      "json",
    ]),
  });
  groups.push({
    id: "advanced_artifacts",
    label: "Advanced / strategy-specific",
    collapsible: true,
    views: pick([
      "depth_plot",
      "plateau_plot",
      "filtered_depth_plot",
      "blob_components",
      "blob_clusters",
      "selected_blob_cluster",
      "rejected_blob_clusters",
      "blob_cluster_scores",
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
      "residual_histogram",
    ]),
  });
  return groups.filter((group) => group.views.length > 0);
}

function resolveDetectViewGroups(
  groups: DetectViewGroup[],
  allViews: StageViewDefinition[],
  stageArtifacts: Array<{ artifact_id?: string; metadata?: Record<string, unknown> | null; created_at?: string | null }>,
  detail: TakeDetail | null,
  formState: Record<string, unknown> | null,
): DetectResolvedViewGroup[] {
  const resolvedArtifacts = stageArtifacts as StudioArtifact[];
  const state = resolveDetectReferenceRunState(detail, resolvedArtifacts, formState);
  const sourceGroups = groups.length ? groups : [{ id: "all", label: "All artifacts", views: allViews }];
  const resolved = sourceGroups.map((group) => ({
    id: group.id,
    label: group.label,
    collapsible: group.collapsible,
    views: group.views.map((view) => {
      const result = resolveDetectReferenceArtifactRelevance({
        viewId: view.id,
        artifacts: resolvedArtifacts,
        activeStrategy: state.strategy,
        componentMode: state.componentMode,
        splitMethod: state.splitMethod,
        fallbackUsed: state.fallbackUsed,
        fallbackStrategy: state.fallbackStrategy,
        runUpdatedAt: state.processedAt,
      });
      return { view, relevance: result.relevance, reason: result.reason };
    }),
  }));
  const filteredGroups = resolved
    .map((group) => {
      const filtered = group.views.filter((entry) => {
        if (group.id === "main_workflow") return entry.relevance === "active" || entry.relevance === "fallback";
        if (group.id === "belt_stripes") return entry.relevance !== "missing";
        if (group.id === "blob_split_advanced") return entry.relevance === "debug" || entry.relevance === "legacy" || entry.relevance === "inactive_strategy";
        if (group.id === "legacy_plateau") return entry.relevance === "inactive_strategy" || entry.relevance === "legacy";
        return entry.relevance !== "missing";
      });
      return { ...group, views: filtered };
    })
    .filter((group) => group.views.length > 0);
  const coveredIds = new Set(filteredGroups.flatMap((group) => group.views.map((entry) => entry.view.id)));
  const allGroupViews = allViews
    .filter((view) => !coveredIds.has(view.id))
    .map((view) => {
      const result = resolveDetectReferenceArtifactRelevance({
        viewId: view.id,
        artifacts: resolvedArtifacts,
        activeStrategy: state.strategy,
        componentMode: state.componentMode,
        splitMethod: state.splitMethod,
        fallbackUsed: state.fallbackUsed,
        fallbackStrategy: state.fallbackStrategy,
        runUpdatedAt: state.processedAt,
      });
      return { view, relevance: result.relevance, reason: result.reason };
    })
    .filter((entry) => entry.relevance !== "missing");
  if (allGroupViews.length) {
    filteredGroups.push({ id: "all_artifacts", label: "All artifacts", collapsible: true, views: allGroupViews });
  }
  return filteredGroups;
}

function prettyStageLabel(stageId: string, fallback: string): string {
  const canonical = canonicalStageId(stageId, fallback);
  if (canonical === "detect_belt_plane") return "Detect reference surface";
  if (canonical === "normalize_heights_to_plane") return "Normalize heights to reference";
  if (canonical === "remove_belt_segment_objects") return "Remove reference + segment objects";
  return fallback;
}

export default function ProcessingLabPage() {
  const [takes, setTakes] = useState<TakeSummary[]>([]);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [datasetSessions, setDatasetSessions] = useState<DatasetSessionSummary[]>([]);
  const [pipelines, setPipelines] = useState<PipelineInfo[]>([]);
  const [runtimeState, setRuntimeState] = useState<RuntimeState | null>(null);
  const [processTemplates, setProcessTemplates] = useState<ProcessTemplate[]>([]);
  const [pipelineInstances, setPipelineInstances] = useState<PipelineInstance[]>([]);
  const [selectedSession, setSelectedSession] = useState("");
  const [selectedDataset, setSelectedDataset] = useState("");
  const [selectedExperimentSession, setSelectedExperimentSession] = useState("");
  const [selectedTakeId, setSelectedTakeId] = useState("");
  const [selectedPipelineId, setSelectedPipelineId] = useState("3d_ball_inspection");
  const [selectedStageId, setSelectedStageId] = useState("");
  const [activeTab, setActiveTab] = useState<StageViewTabId>("overview");
  const [modalityFilter, setModalityFilter] = useState<CaptureModality | "all">("all");
  const [takeFilter, setTakeFilter] = useState<TakeFilter>("all");
  const [takeSearch, setTakeSearch] = useState("");
  const [takesHasMore, setTakesHasMore] = useState(false);
  const [takesOffset, setTakesOffset] = useState(0);
  const [takesLoading, setTakesLoading] = useState(false);
  const [takesLoadingMore, setTakesLoadingMore] = useState(false);
  const [detail, setDetail] = useState<TakeDetail | null>(null);
  const [artifactBaseline, setArtifactBaseline] = useState<ValidationBaseline | null>(null);
  const [validationMessage, setValidationMessage] = useState<string | null>(null);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [pipelineRuns, setPipelineRuns] = useState<PipelineRunEntry[]>([]);
  const [pipelineRunsLoading, setPipelineRunsLoading] = useState(false);
  const [pipelineRunsRefreshToken, setPipelineRunsRefreshToken] = useState(0);
  const refreshPipelineRuns = useCallback(() => setPipelineRunsRefreshToken((token) => token + 1), []);
  const [selectedObjectId, setSelectedObjectId] = useState<number | null>(null);
  const [activeBlobClusterId, setActiveBlobClusterId] = useState<string | null>(null);
  const [selectedArtifactId, setSelectedArtifactId] = useState<string | null>(null);
  const [hoveredObjectId, setHoveredObjectId] = useState<number | null>(null);
  const [hoveredArtifactId, setHoveredArtifactId] = useState<string | null>(null);
  const [hoverSample, setHoverSample] = useState<HoverInspectionSample | null>(null);
  const [overlayDebug, setOverlayDebug] = useState<OverlayDebugInfo | null>(null);
  const [viewState, setViewState] = useState<StudioArtifactViewState>({
    viewportMode: "fit_all",
    overlayMode: "all",
    binaryMaskMode: "mask",
    binaryMaskOpacity: 0.52,
    binaryMaskReferenceOpacity: 1,
    binaryMaskReferenceArtifactId: null,
  });
  const [selectedSubstageId, setSelectedSubstageId] = useState<string | null>(null);
  const [selectedContractUnitId, setSelectedContractUnitId] = useState<string | null>(null);
  const [graphWorkspaceExpanded, setGraphWorkspaceExpanded] = useState(false);
  const [partialRerunPlan, setPartialRerunPlan] = useState<PartialRerunPlanResponse | null>(null);
  const [partialRerunPlanning, setPartialRerunPlanning] = useState(false);
  const [partialRerunPlanStale, setPartialRerunPlanStale] = useState(false);
  const [planContext, setPlanContext] = useState<PlanContext | null>(null);
  const [partialRerunExecution, setPartialRerunExecution] = useState<PartialRerunExecuteResponse | null>(null);
  const [partialRerunExecuting, setPartialRerunExecuting] = useState(false);
  const [referenceViewScope, setReferenceViewScope] = useState<ReferenceViewScope | null>(null);
  const [selectedTemplateId, setSelectedTemplateId] = useState("mining_steel_ball_classification_2d_reflectance_mvp");
  const [selectedInputType, setSelectedInputType] = useState("rgb_image");
  const [selectedPipelineInstanceId, setSelectedPipelineInstanceId] = useState<string>("");
  const [manualImagePath, setManualImagePath] = useState("");
  const [pipelineSearch, setPipelineSearch] = useState("");
  const [pipelineFamilyFilter, setPipelineFamilyFilter] = useState<"all" | "2d" | "3d" | "25d" | "fusion" | "generic">("all");
  const [lastUsedPipelineByTake, setLastUsedPipelineByTake] = useState<Record<string, string>>({});
  const [tabFallbackMessage, setTabFallbackMessage] = useState<string | null>(null);
  const [deepLinkMessage, setDeepLinkMessage] = useState<string | null>(null);
  const [pipelineRecipes, setPipelineRecipes] = useState<PipelineRecipe[]>([]);
  const [selectedRecipeId, setSelectedRecipeId] = useState<string>("");
  const [recipePanelMessage, setRecipePanelMessage] = useState<string | null>(null);
  const [recipeLoading, setRecipeLoading] = useState(false);
  const [comparisonBusy, setComparisonBusy] = useState(false);
  const [comparisonError, setComparisonError] = useState<string | null>(null);
  const [comparisonResult, setComparisonResult] = useState<PipelineComparisonResponse | null>(null);
  const [batchComparisonBusy, setBatchComparisonBusy] = useState(false);
  const [batchComparisonError, setBatchComparisonError] = useState<string | null>(null);
  const [batchComparisonResult, setBatchComparisonResult] = useState<PipelineBatchComparisonResponse | null>(null);
  const [comparisonLeftType, setComparisonLeftType] = useState<CompareSourceType>("current");
  const [comparisonRightType, setComparisonRightType] = useState<CompareSourceType>("recipe");
  const [comparisonLeftRunId, setComparisonLeftRunId] = useState<string>("");
  const [comparisonRightRunId, setComparisonRightRunId] = useState<string>("");
  const [comparisonLeftRecipeId, setComparisonLeftRecipeId] = useState<string>("");
  const [comparisonRightRecipeId, setComparisonRightRecipeId] = useState<string>("");
  const [comparisonHistory, setComparisonHistory] = useState<PipelineComparisonIndexEntry[]>([]);
  const [comparisonHistoryLoading, setComparisonHistoryLoading] = useState(false);
  const [captureFriendlyName, setCaptureFriendlyName] = useState("");
  const [captureTags, setCaptureTags] = useState("");
  const [captureExpectedClass, setCaptureExpectedClass] = useState("");
  const [captureExpectedDiameter, setCaptureExpectedDiameter] = useState("");
  const [captureNotes, setCaptureNotes] = useState("");
  const [captureLoading, setCaptureLoading] = useState(false);
  const [captureError, setCaptureError] = useState<string | null>(null);
  const [showArchived, setShowArchived] = useState(false);
  const [segmentationPreviewParams, setSegmentationPreviewParams] = useState<SegmentationTuningParams | null>(null);
  const [segmentationPersistedParams, setSegmentationPersistedParams] = useState<SegmentationTuningParams | null>(null);
  const [segmentationPreviewDirty, setSegmentationPreviewDirty] = useState(false);
  const [segmentationPreview, setSegmentationPreview] = useState<SegmentationPreviewResponse | null>(null);
  const [segmentationPreviewLoading, setSegmentationPreviewLoading] = useState(false);
  const [segmentationPreviewError, setSegmentationPreviewError] = useState<string | null>(null);
  const [ellipsePreviewParams, setEllipsePreviewParams] = useState<Record<string, unknown> | null>(null);
  const [ellipsePersistedParams, setEllipsePersistedParams] = useState<Record<string, unknown> | null>(null);
  const [ellipsePreviewDirty, setEllipsePreviewDirty] = useState(false);
  const [ellipsePreviewLoading, setEllipsePreviewLoading] = useState(false);
  const [ellipsePreviewError, setEllipsePreviewError] = useState<string | null>(null);
  const [planeQaParams, setPlaneQaParams] = useState<PlaneQaTuningParams | null>(null);
  const [planeQaPersistedParams, setPlaneQaPersistedParams] = useState<PlaneQaTuningParams | null>(null);
  const [planeQaDirty, setPlaneQaDirty] = useState(false);
  const [measurementConfigOpen, setMeasurementConfigOpen] = useState(false);
  const [pipelineDefaults25d, setPipelineDefaults25d] = useState<Record<string, unknown>>({});
  const [fusionInputs, setFusionInputs] = useState<FusionInputBundle | null>(null);
  const [fusionPreview, setFusionPreview] = useState<FusionPreviewResult | null>(null);
  const [fusionRuns, setFusionRuns] = useState<FusionRunRecord[]>([]);
  const [latestFusionRun, setLatestFusionRun] = useState<FusionRunRecord | null>(null);
  const [latestFusionResult, setLatestFusionResult] = useState<FusionRunResult | null>(null);
  const [fusionLoading, setFusionLoading] = useState(false);
  const [fusionError, setFusionError] = useState<string | null>(null);
  const [runtimeWorkers, setRuntimeWorkers] = useState<RuntimeWorkerStatus[]>([]);
  const [latestPublishedResult, setLatestPublishedResult] = useState<InspectionPublishedResult | null>(null);
  const [processBindings, setProcessBindings] = useState<ProcessBinding[]>([]);
  const [bindingPurpose, setBindingPurpose] = useState<"acquisition_inspection" | "fusion">("acquisition_inspection");
  const [processingJobs, setProcessingJobs] = useState<ProcessingJobRecord[]>([]);
  const [processingJobCounts, setProcessingJobCounts] = useState<Record<string, number>>({});
  const [processingJobsLoading, setProcessingJobsLoading] = useState(false);
  const [processingJobsUnavailable, setProcessingJobsUnavailable] = useState<string | null>(null);
  const [pipelineActionBusy, setPipelineActionBusy] = useState(false);
  const [pipelineActionMessage, setPipelineActionMessage] = useState<string | null>(null);
  const [roiEditModeActive, setRoiEditModeActive] = useState(false);
  const [roiEditStatus, setRoiEditStatus] = useState<RoiEditStatus>("idle");
  const [resultStale, setResultStale] = useState(false);
  const [studioMode, setStudioMode] = useState<StudioMode>("inspect");
  const [leftRailExpanded, setLeftRailExpanded] = useState(false);
  const [leftRailCollapsed, setLeftRailCollapsed] = useState(false);
  const [inspectorExpanded, setInspectorExpanded] = useState(false);
  const [inspectorCollapsed, setInspectorCollapsed] = useState(false);
  const [jobsPopoverOpen, setJobsPopoverOpen] = useState(false);
  const [focusMode, setFocusMode] = useState(false);
  const [shellLayoutMode, setShellLayoutMode] = useState<ShellLayoutMode>("wide");
  const [controlPanelTab, setControlPanelTab] = useState<ControlPanelTab>("inspect");
  const [showAllStageViews, setShowAllStageViews] = useState<boolean>(() => readShowAllStageViews());
  const [showAllSubstageArtifacts, setShowAllSubstageArtifacts] = useState(false);
  const [diagnosticsExpanded, setDiagnosticsExpanded] = useState(false);
  const [sourceContextDismissed, setSourceContextDismissed] = useState(false);
  const [selectedTakeIds, setSelectedTakeIds] = useState<Set<string>>(new Set());
  const lastTakeClickIndexRef = useRef<number | null>(null);
  const selectedTakeIdRef = useRef(selectedTakeId);
  const takesRequestIdRef = useRef(0);
  const studioScopeAppliedRef = useRef(false);
  const studioTakeAppliedRef = useRef(false);
  const studioPipelineAppliedRef = useRef(false);
  const studioStageAppliedRef = useRef(false);
  const studioObjectAppliedRef = useRef(false);
  const studioGraphDeepLinkAppliedRef = useRef(false);
  const studioRunDeepLinkAppliedRef = useRef(false);

  const studioDeepLink = useMemo(
    () => parseStudioDeepLink(typeof window === "undefined" ? "" : window.location.search),
    [],
  );
  const requestedStudioObjectId = useMemo(
    () => studioDeepLinkObjectId(studioDeepLink.object_id),
    [studioDeepLink.object_id],
  );

  const [debouncedTakeSearch, setDebouncedTakeSearch] = useState("");

  useEffect(() => {
    const timer = window.setTimeout(() => setDebouncedTakeSearch(takeSearch), 220);
    return () => window.clearTimeout(timer);
  }, [takeSearch]);

  useEffect(() => {
    function syncShellLayout() {
      const width = window.innerWidth;
      if (width < 900) setShellLayoutMode("very_narrow");
      else if (width < 1180) setShellLayoutMode("narrow");
      else if (width < 1540) setShellLayoutMode("medium");
      else setShellLayoutMode("wide");
    }
    syncShellLayout();
    window.addEventListener("resize", syncShellLayout);
    return () => window.removeEventListener("resize", syncShellLayout);
  }, []);

  useEffect(() => {
    selectedTakeIdRef.current = selectedTakeId;
  }, [selectedTakeId]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(SENSOR_STUDIO_SHOW_ALL_VIEWS_KEY, String(showAllStageViews));
  }, [showAllStageViews]);

  useEffect(() => {
    if (studioScopeAppliedRef.current) return;
    if (studioDeepLink.dataset_id) setSelectedDataset(studioDeepLink.dataset_id);
    if (studioDeepLink.session_id) setSelectedSession(studioDeepLink.session_id);
    studioScopeAppliedRef.current = true;
  }, [studioDeepLink.dataset_id, studioDeepLink.session_id]);

  useEffect(() => {
    if (studioTakeAppliedRef.current || !studioDeepLink.take_id) return;
    if (selectedTakeId !== studioDeepLink.take_id) {
      setSelectedTakeId(studioDeepLink.take_id);
      return;
    }
    studioTakeAppliedRef.current = true;
  }, [selectedTakeId, studioDeepLink.take_id]);


  const takePageParams = useMemo(() => ({
    session_id: selectedSession || undefined,
    dataset_id: selectedDataset || undefined,
    search: debouncedTakeSearch || undefined,
    show_archived: showArchived,
  }), [selectedSession, selectedDataset, debouncedTakeSearch, showArchived]);

  const loadTakePage = useCallback(async (offset: number, append: boolean) => {
    const requestId = ++takesRequestIdRef.current;
    if (append) setTakesLoadingMore(true);
    else setTakesLoading(true);
    try {
      const page = await api.pagedTakes({
        ...takePageParams,
        limit: 50,
        offset,
      });
      if (takesRequestIdRef.current !== requestId) return;
      setTakes((prev) => (append ? [...prev, ...page.items] : page.items));
      setTakesOffset(page.next_offset ?? page.offset + page.items.length);
      setTakesHasMore(Boolean(page.has_more));
      if (!append) {
        const nextTakeId = resolveNextSelectedTakeId({
          currentSelectedTakeId: selectedTakeIdRef.current,
          availableTakeIds: page.items.map((take) => take.take_id),
          preferredTakeId: studioDeepLink.take_id ?? page.items[0]?.take_id ?? null,
        }) ?? "";
        setSelectedTakeId(nextTakeId);
      }
    } finally {
      if (takesRequestIdRef.current === requestId) {
        setTakesLoading(false);
        setTakesLoadingMore(false);
      }
    }
  }, [studioDeepLink.take_id, takePageParams]);

  const load = useCallback(async () => {
    const [
      nextSessions,
      nextPipelines,
      nextRuntime,
      nextTemplates,
      nextInstances,
      nextDatasets,
      nextWorkers,
      nextBindings,
      nextLatestPublished,
      nextJobs,
    ] = await Promise.all([
      api.sessions(),
      api.pipelines(),
      api.state(),
      api.processTemplates(),
      api.pipelineInstances(),
      api.datasets(),
      api.runtimeWorkers().catch(() => ({ workers: [] })),
      api.processBindings().catch(() => [] as ProcessBinding[]),
      api.latestPublishedInspectionResult().catch(() => null as InspectionPublishedResult | null),
      api.processingJobs().catch((error) => {
        setProcessingJobsUnavailable(error instanceof Error ? error.message : "Processing jobs unavailable");
        return { jobs: [], counts: {} };
      }),
    ]);
    const nextDatasetSessions = selectedDataset ? await api.datasetSessions(selectedDataset) : [];
    setSessions(nextSessions);
    setDatasets(nextDatasets);
    setDatasetSessions(nextDatasetSessions);
    setPipelines(nextPipelines);
    setRuntimeState(nextRuntime);
    setProcessTemplates(nextTemplates);
    setPipelineInstances(nextInstances);
    setRuntimeWorkers(nextWorkers.workers ?? []);
    setProcessBindings(nextBindings);
    setLatestPublishedResult(nextLatestPublished);
    setProcessingJobs(nextJobs.jobs ?? []);
    setProcessingJobCounts(nextJobs.counts ?? {});
    setProcessingJobsUnavailable(null);
    setSelectedPipelineInstanceId((current) => current || nextInstances[0]?.id || "");
    setSelectedPipelineId((current) => current || nextPipelines[0]?.id || "3d_ball_inspection");
    await loadTakePage(0, false);
  }, [selectedDataset, loadTakePage]);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    if (selectedPipelineId !== "mining_steel_ball_classification_25d") {
      setPipelineRecipes([]);
      setSelectedRecipeId("");
      setRecipePanelMessage(null);
      return;
    }
    let cancelled = false;
    setRecipeLoading(true);
    void api.pipelineRecipes(selectedPipelineId).then((response) => {
      if (cancelled) return;
      setPipelineRecipes(response.recipes ?? []);
      setSelectedRecipeId((current) => {
        if (current && (response.recipes ?? []).some((recipe) => recipe.recipe_id === current)) return current;
        return "";
      });
    }).catch((error: unknown) => {
      if (cancelled) return;
      setPipelineRecipes([]);
      setRecipePanelMessage(error instanceof Error ? error.message : "Failed to load recipes.");
    }).finally(() => {
      if (!cancelled) setRecipeLoading(false);
    });
    return () => {
      cancelled = true;
    };
  }, [selectedPipelineId]);

  useEffect(() => {
    if (selectedPipelineId !== "mining_steel_ball_classification_25d") {
      setComparisonHistory([]);
      return;
    }
    let cancelled = false;
    setComparisonHistoryLoading(true);
    void api.pipelineComparisons(selectedPipelineId, { take_id: detail?.take_id ?? null, limit: 12 }).then((response) => {
      if (cancelled) return;
      setComparisonHistory(response.comparisons ?? []);
    }).catch(() => {
      if (cancelled) return;
      setComparisonHistory([]);
    }).finally(() => {
      if (!cancelled) setComparisonHistoryLoading(false);
    });
    return () => {
      cancelled = true;
    };
  }, [selectedPipelineId, detail?.take_id, comparisonResult?.comparison_id]);

  useEffect(() => {
    let cancelled = false;
    let pollingStopped = false;
    const refreshJobs = async () => {
      if (pollingStopped) return;
      setProcessingJobsLoading(true);
      try {
        const nextJobs = await api.processingJobs();
        if (cancelled) return;
        setProcessingJobs(nextJobs.jobs ?? []);
        setProcessingJobCounts(nextJobs.counts ?? {});
        setProcessingJobsUnavailable(null);
      } catch (error) {
        if (cancelled) return;
        const message = error instanceof Error ? error.message : "Processing jobs unavailable";
        setProcessingJobsUnavailable(message);
        if (message.startsWith("404") || message.startsWith("500")) {
          pollingStopped = true;
        }
      } finally {
        if (!cancelled) setProcessingJobsLoading(false);
      }
    };
    void refreshJobs();
    const interval = window.setInterval(() => {
      if (pollingStopped) return;
      void refreshJobs();
    }, 3000);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, []);

  const loadMoreTakes = useCallback(async () => {
    if (!takesHasMore || takesLoadingMore) return;
    await loadTakePage(takesOffset, true);
  }, [takesHasMore, takesLoadingMore, takesOffset, loadTakePage]);

  useEffect(() => {
    let cancelled = false;
    if (!selectedTakeId) {
      setDetail(null);
      return () => { cancelled = true; };
    }
    void api.take(selectedTakeId, activeRunId ?? undefined).then((nextDetail) => {
      if (cancelled) return;
      setDetail(nextDetail);
    }).catch(() => {
      if (cancelled) return;
      setDetail(null);
    });
    return () => { cancelled = true; };
  }, [selectedTakeId, activeRunId]);

  useEffect(() => {
    setActiveRunId(null);
  }, [selectedTakeId]);

  useEffect(() => {
    setActiveBlobClusterId(null);
  }, [selectedTakeId]);

  const selectedPipeline = useMemo(
    () => pipelines.find((pipeline) => pipeline.id === selectedPipelineId) ?? pipelines[0] ?? null,
    [pipelines, selectedPipelineId]
  );

  useEffect(() => {
    let cancelled = false;
    if (!selectedPipeline?.id || !selectedTakeId) {
      setPipelineRuns([]);
      return () => { cancelled = true; };
    }
    setPipelineRunsLoading(true);
    void api.pipelineRuns(selectedPipeline.id, { take_id: selectedTakeId, limit: 50 }).then((response) => {
      if (cancelled) return;
      setPipelineRuns(response.runs ?? []);
    }).catch(() => {
      if (cancelled) return;
      setPipelineRuns([]);
    }).finally(() => {
      if (cancelled) return;
      setPipelineRunsLoading(false);
    });
    return () => { cancelled = true; };
  }, [selectedPipeline?.id, selectedTakeId, activeRunId, pipelineRunsRefreshToken]);

  const comparisonSourceLabelContext = useMemo(
    () => ({
      parentRunId: (detail?.result?.parent_run_id as string | null | undefined) ?? null,
      childRunIds: pipelineRuns.filter((run) => run.parent_run_id === detail?.result?.run_id).map((run) => run.run_id),
    }),
    [detail?.result?.parent_run_id, detail?.result?.run_id, pipelineRuns],
  );
  const activePipelineFamily = selectedPipeline?.pipeline_family ?? "3d";
  const compatible = selectedPipeline ? modalityCompatible(selectedPipeline, detail?.modalities) : false;
  const selectedPipelineInstance = useMemo(
    () => pipelineInstances.find((instance) => instance.id === selectedPipelineInstanceId) ?? pipelineInstances[0] ?? null,
    [pipelineInstances, selectedPipelineInstanceId]
  );
  const processed = Boolean(
    ((detail?.processing_by_family ?? []).find((item) => item.family === activePipelineFamily)?.hasCompletedOutput)
    || (detail?.has_done && detail.result)
  );
  const selectedStage = selectedPipeline?.stages.find((stage) => stage.id === selectedStageId) ?? selectedPipeline?.stages[0] ?? null;
  const canonicalSelectedStageId = useMemo(
    () => canonicalStageId(selectedStage?.id ?? selectedStageId, selectedStage?.display_name ?? null),
    [selectedStage?.id, selectedStage?.display_name, selectedStageId]
  );
  useEffect(() => {
    if (!selectedPipeline) return;
    const nextStageId = selectedPipeline.stages[0]?.id ?? "";
    if (!selectedPipeline.stages.some((stage) => stage.id === selectedStageId)) {
      setSelectedStageId(nextStageId);
      setActiveTab("overview");
    }
  }, [selectedPipeline, selectedStageId]);
  const stageSemantic = useMemo(
    () => stageSemanticDefinition(canonicalSelectedStageId),
    [canonicalSelectedStageId]
  );
  const stageTabs = useMemo(
    () => stageSemantic.views.map((view) => ({ id: view.id as StageViewTabId, label: view.label })),
    [stageSemantic.views]
  );
  useEffect(() => {
    const fallback = (stageSemantic.defaultViewId || stageTabs[0]?.id || "overview") as StageViewTabId;
    setActiveTab(fallback);
    setTabFallbackMessage(null);
  }, [canonicalSelectedStageId]);
  useEffect(() => {
    let cancelled = false;
    void api.pipelineDefaults25d().then((response) => {
      if (cancelled) return;
      setPipelineDefaults25d(response.defaults || {});
    }).catch(() => {
      if (cancelled) return;
      setPipelineDefaults25d({});
    });
    return () => { cancelled = true; };
  }, []);
  useEffect(() => {
    if (selectedPipeline?.id !== "mining_steel_ball_classification_25d" || !detail) return;
    const localSaved = read25dSavedDefaults();
    const saved: Record<string, unknown> = { ...localSaved, ...pipelineDefaults25d };
    const fromResult = (((detail.result as unknown as Record<string, unknown> | null) ?? {})["stage_params"] as Record<string, unknown> | undefined) ?? {};
    const detect = {
      ...((saved.detect_belt_plane as Record<string, unknown> | undefined) ?? {}),
      ...((fromResult.detect_belt_plane as Record<string, unknown> | undefined) ?? {}),
    };
    const seg = {
      ...((saved.remove_belt_segment_objects as Record<string, unknown> | undefined) ?? {}),
      ...((fromResult.remove_belt_segment_objects as Record<string, unknown> | undefined) ?? {}),
    };
    const savedKnown = (saved.known_object_25d as Record<string, unknown> | undefined) ?? {};
    const resultKnown = (fromResult.known_object_25d as Record<string, unknown> | undefined) ?? {};
    const savedPersistedScale = persistedScaleFromKnown(savedKnown);
    const resultPersistedScale = persistedScaleFromKnown(resultKnown);
    const persistedScale =
      savedPersistedScale && !isIdentityScaleCorrection(savedPersistedScale)
        ? savedPersistedScale
        : resultPersistedScale ?? savedPersistedScale;
    const known: Record<string, unknown> = {
      ...savedKnown,
      ...resultKnown,
      apply_persisted_correction: savedKnown.apply_persisted_correction ?? resultKnown.apply_persisted_correction ?? true,
      persisted_scale_correction_x: persistedScale?.x ?? savedKnown.persisted_scale_correction_x ?? resultKnown.persisted_scale_correction_x,
      persisted_scale_correction_y: persistedScale?.y ?? savedKnown.persisted_scale_correction_y ?? resultKnown.persisted_scale_correction_y,
      persisted_scale_correction_z: persistedScale?.z ?? savedKnown.persisted_scale_correction_z ?? resultKnown.persisted_scale_correction_z,
    };
    const defaults: PlaneQaTuningParams = {
      background_detection_strategy: String(detect.background_detection_strategy ?? "low_gradient_surface"),
      gradient_smoothing_kernel: Number(detect.gradient_smoothing_kernel ?? 3),
      gradient_threshold_mode: String(detect.gradient_threshold_mode ?? "percentile"),
      gradient_threshold_value: Number(detect.gradient_threshold_value ?? 2.0),
      gradient_threshold_percentile: Number(detect.gradient_threshold_percentile ?? 70),
      low_gradient_morphology_enabled: Boolean(detect.low_gradient_morphology_enabled ?? true),
      low_gradient_open_kernel: Number(detect.low_gradient_open_kernel ?? 3),
      low_gradient_close_kernel: Number(detect.low_gradient_close_kernel ?? 5),
      low_gradient_fill_holes: Boolean(detect.low_gradient_fill_holes ?? true),
      low_gradient_min_component_area: Number(detect.low_gradient_min_component_area ?? 1500),
      reference_surface_selection_mode: String(detect.reference_surface_selection_mode ?? "largest_constant_z"),
      reference_surface_model: String(detect.reference_surface_model ?? "auto"),
      reference_suppression_mask_policy: String(detect.reference_suppression_mask_policy ?? "auto"),
      reference_surface_max_plane_residual_p95_mm: Number(detect.reference_surface_max_plane_residual_p95_mm ?? 3.0),
      // --- Plateau detection (low_gradient_depth_plateaus strategy) ---
      low_gradient_plateau_use_hessian_filter: Boolean(detect.low_gradient_plateau_use_hessian_filter ?? true),
      low_gradient_plateau_hessian_percentile: Number(detect.low_gradient_plateau_hessian_percentile ?? 70),
      low_gradient_plateau_hist_bins: Number(detect.low_gradient_plateau_hist_bins ?? 96),
      low_gradient_plateau_min_fraction: Number(detect.low_gradient_plateau_min_fraction ?? 0.05),
      low_gradient_plateau_min_pixels: Number(detect.low_gradient_plateau_min_pixels ?? 500),
      low_gradient_plateau_smoothing_sigma_bins: Number(detect.low_gradient_plateau_smoothing_sigma_bins ?? 2.5),
      low_gradient_plateau_peak_drop_ratio: Number(detect.low_gradient_plateau_peak_drop_ratio ?? 0.40),
      low_gradient_plateau_select_min_area_fraction: Number(detect.low_gradient_plateau_select_min_area_fraction ?? 0.20),
      low_gradient_plateau_selection_mode: String(detect.low_gradient_plateau_selection_mode ?? "lowest_dominant"),
      low_gradient_plateau_robust_band_mad_k: Number(detect.low_gradient_plateau_robust_band_mad_k ?? 3.0),
      low_gradient_plateau_detection_min_count_floor: Number(detect.low_gradient_plateau_detection_min_count_floor ?? 25),
      low_gradient_plateau_detection_min_count_fraction: Number(detect.low_gradient_plateau_detection_min_count_fraction ?? 0.25),
      blob_cluster_min_component_area: Number(detect.blob_cluster_min_component_area ?? 250),
      blob_component_mode: String(detect.blob_component_mode ?? "xy_only"),
      blob_connectivity: Number(detect.blob_connectivity ?? 8),
      blob_neighbor_z_tolerance_mm: Number(detect.blob_neighbor_z_tolerance_mm ?? 3.0),
      blob_component_z_tolerance_mm: Number(detect.blob_component_z_tolerance_mm ?? 8.0),
      blob_component_tolerance_mode: String(detect.blob_component_tolerance_mode ?? "fixed"),
      blob_component_mad_k: Number(detect.blob_component_mad_k ?? 3.0),
      blob_component_mad_floor_mm: Number(detect.blob_component_mad_floor_mm ?? 1.0),
      blob_component_allow_gradual_slope: Boolean(detect.blob_component_allow_gradual_slope ?? true),
      blob_component_max_local_slope_mm_per_px: Number(detect.blob_component_max_local_slope_mm_per_px ?? 3.0),
      blob_component_min_area_px: Number(detect.blob_component_min_area_px ?? 300),
      blob_component_use_smoothed_z: Boolean(detect.blob_component_use_smoothed_z ?? true),
      blob_component_z_smoothing_kernel: Number(detect.blob_component_z_smoothing_kernel ?? 3),
      blob_split_by_height_enabled: Boolean(detect.blob_split_by_height_enabled ?? true),
      blob_split_method: String(detect.blob_split_method ?? "height_borders"),
      blob_split_min_height_range_mm: Number(detect.blob_split_min_height_range_mm ?? 8.0),
      blob_split_min_pixels: Number(detect.blob_split_min_pixels ?? 300),
      blob_split_gap_mm: Number(detect.blob_split_gap_mm ?? 4.0),
      blob_split_mode: String(detect.blob_split_mode ?? "histogram_gap"),
      blob_split_hist_bins: Number(detect.blob_split_hist_bins ?? 48),
      blob_split_min_band_fraction: Number(detect.blob_split_min_band_fraction ?? 0.08),
      blob_split_min_band_pixels: Number(detect.blob_split_min_band_pixels ?? 150),
      blob_split_merge_gap_mm: Number(detect.blob_split_merge_gap_mm ?? 2.0),
      blob_split_refine_morphology: Boolean(detect.blob_split_refine_morphology ?? true),
      blob_split_open_kernel: Number(detect.blob_split_open_kernel ?? 1),
      blob_split_close_kernel: Number(detect.blob_split_close_kernel ?? 3),
      blob_split_height_border_enabled: Boolean(detect.blob_split_height_border_enabled ?? true),
      blob_split_height_border_mode: String(detect.blob_split_height_border_mode ?? "morphological_gradient"),
      blob_split_height_border_smoothing_kernel: Number(detect.blob_split_height_border_smoothing_kernel ?? 5),
      blob_split_height_border_threshold_mode: String(detect.blob_split_height_border_threshold_mode ?? "percentile"),
      blob_split_height_border_percentile: Number(detect.blob_split_height_border_percentile ?? 85.0),
      blob_split_height_border_min_delta_mm: Number(detect.blob_split_height_border_min_delta_mm ?? 4.0),
      blob_split_height_border_dilate_kernel: Number(detect.blob_split_height_border_dilate_kernel ?? 3),
      blob_split_height_border_close_kernel: Number(detect.blob_split_height_border_close_kernel ?? 3),
      blob_split_height_border_min_length_px: Number(detect.blob_split_height_border_min_length_px ?? 20),
      blob_split_min_fragment_area_px: Number(detect.blob_split_min_fragment_area_px ?? 300),
      blob_split_merge_weak_boundaries: Boolean(detect.blob_split_merge_weak_boundaries ?? true),
      blob_split_merge_max_median_z_gap_mm: Number(detect.blob_split_merge_max_median_z_gap_mm ?? 3.0),
      blob_split_merge_max_boundary_strength_mm: Number(detect.blob_split_merge_max_boundary_strength_mm ?? 6.0),
      blob_split_merge_background_like_only: Boolean(detect.blob_split_merge_background_like_only ?? true),
      blob_cluster_height_gap_mm: Number(detect.blob_cluster_height_gap_mm ?? 8.0),
      blob_cluster_height_gap_mode: String(detect.blob_cluster_height_gap_mode ?? "adaptive"),
      blob_cluster_min_area_fraction: Number(detect.blob_cluster_min_area_fraction ?? 0.10),
      blob_cluster_min_total_pixels: Number(detect.blob_cluster_min_total_pixels ?? 1200),
      blob_cluster_max_z_mad_mm: Number(detect.blob_cluster_max_z_mad_mm ?? 12.0),
      blob_cluster_refine_by_mad: Boolean(detect.blob_cluster_refine_by_mad ?? true),
      blob_cluster_refine_mad_k: Number(detect.blob_cluster_refine_mad_k ?? 2.5),
      blob_cluster_refine_floor_mm: Number(detect.blob_cluster_refine_floor_mm ?? 1.0),
      blob_cluster_refine_keep_border_support: Boolean(detect.blob_cluster_refine_keep_border_support ?? true),
      blob_cluster_fallback_strategy: String(detect.blob_cluster_fallback_strategy ?? "low_gradient_depth_plateaus"),
      plane_fit_min_inlier_ratio: Number(detect.plane_fit_min_inlier_ratio ?? 0.35),
      plane_fit_residual_threshold_mm: Number(detect.plane_fit_residual_threshold_mm ?? 1.25),
      plane_background_residual_tolerance_mm: Number(detect.plane_background_residual_tolerance_mm ?? 2.5),
      plane_background_residual_adaptive_multiplier: Number(detect.plane_background_residual_adaptive_multiplier ?? 1.5),
      plane_refit_after_expansion: Boolean(detect.plane_refit_after_expansion ?? true),
      // --- Belt stripe filter ---
      belt_stripe_filter_enabled: Boolean(detect.belt_stripe_filter_enabled ?? true),
      belt_stripe_filter_scope: String(detect.belt_stripe_filter_scope ?? "global"),
      belt_stripe_filter_window_mm: Number(detect.belt_stripe_filter_window_mm ?? 30.0),
      belt_stripe_filter_direction: String(detect.belt_stripe_filter_direction ?? "auto"),
      belt_stripe_filter_threshold_mode: String(detect.belt_stripe_filter_threshold_mode ?? "otsu"),
      belt_stripe_filter_min_altitude_mm: Number(detect.belt_stripe_filter_min_altitude_mm ?? 10.0),
      belt_stripe_filter_k_mad: Number(detect.belt_stripe_filter_k_mad ?? 3.0),
      belt_stripe_filter_fixed_threshold_mm: Number(detect.belt_stripe_filter_fixed_threshold_mm ?? 10.0),
      belt_stripe_filter_min_stripe_fraction: Number(detect.belt_stripe_filter_min_stripe_fraction ?? 0.02),
      belt_stripe_filter_z_floor_enabled: Boolean(detect.belt_stripe_filter_z_floor_enabled ?? true),
      belt_stripe_filter_z_floor_use_upper_bound: Boolean(detect.belt_stripe_filter_z_floor_use_upper_bound ?? true),
      belt_stripe_filter_z_floor_margin_mm: Number(detect.belt_stripe_filter_z_floor_margin_mm ?? 20.0),
      belt_stripe_filter_max_stripe_height_mm: Number(detect.belt_stripe_filter_max_stripe_height_mm ?? 500.0),
      belt_stripe_filter_above_belt_close_mm: Number(detect.belt_stripe_filter_above_belt_close_mm ?? 30.0),
      belt_stripe_filter_object_kernel_mm: Number(detect.belt_stripe_filter_object_kernel_mm ?? 100.0),
      belt_stripe_filter_object_kernel_shape: String(detect.belt_stripe_filter_object_kernel_shape ?? "ellipse"),
      belt_stripe_filter_altitude_hist_bins: Number(detect.belt_stripe_filter_altitude_hist_bins ?? 64),
      belt_stripe_filter_auto_bimodality_margin: Number(detect.belt_stripe_filter_auto_bimodality_margin ?? 1.10),
      belt_stripe_filter_z_floor_upper_percentile: Number(detect.belt_stripe_filter_z_floor_upper_percentile ?? 99.0),
      belt_stripe_filter_z_floor_fallback_lower_percentile: Number(detect.belt_stripe_filter_z_floor_fallback_lower_percentile ?? 10.0),
      belt_stripe_filter_z_floor_fallback_upper_percentile: Number(detect.belt_stripe_filter_z_floor_fallback_upper_percentile ?? 25.0),
      belt_stripe_filter_warn_removed_fraction: Number(detect.belt_stripe_filter_warn_removed_fraction ?? 0.40),
      // --- Low-gradient surface (legacy strategy) ---
      low_gradient_surface_support_z_mad_multiplier: Number(detect.low_gradient_surface_support_z_mad_multiplier ?? 2.5),
      low_gradient_surface_support_z_floor_mm: Number(detect.low_gradient_surface_support_z_floor_mm ?? 1.0),
      low_gradient_surface_support_z_mad_floor_mm: Number(detect.low_gradient_surface_support_z_mad_floor_mm ?? 0.25),
      low_gradient_surface_ridge_percentile: Number(detect.low_gradient_surface_ridge_percentile ?? 90.0),
      // --- Height gate ---
      reference_surface_height_gate_enabled: Boolean(detect.reference_surface_height_gate_enabled ?? true),
      reference_surface_height_gate_margin_mm: Number(detect.reference_surface_height_gate_margin_mm ?? 8.0),
      reference_surface_height_gate_min_coverage_ratio: Number(detect.reference_surface_height_gate_min_coverage_ratio ?? 0.10),
      reference_surface_height_gate_max_coverage_ratio: Number(detect.reference_surface_height_gate_max_coverage_ratio ?? 0.95),
      reference_surface_height_gate_gap_floor_mm: Number(detect.reference_surface_height_gate_gap_floor_mm ?? 1.0),
      reference_surface_height_gate_gap_ratio: Number(detect.reference_surface_height_gate_gap_ratio ?? 8.0),
      // --- Plot rendering ---
      plot_depth_plot_max_render_samples: Number(detect.plot_depth_plot_max_render_samples ?? 60000),
      plot_y_robust_percentile: Number(detect.plot_y_robust_percentile ?? 98.0),
      reference_surface_region_mode: String(
        detect.reference_surface_region_mode
        ?? (
          Boolean((detect.plane_fit_roi as Record<string, unknown> | undefined)?.enabled ?? false)
            ? String((detect.plane_fit_roi as Record<string, unknown> | undefined)?.type ?? "rectangle")
            : "none"
        ),
      ).replace("vertical_band", "full_height_x_band"),
      plane_fit_roi_enabled: Boolean((detect.plane_fit_roi as Record<string, unknown> | undefined)?.enabled ?? false),
      plane_fit_roi_type: String((detect.plane_fit_roi as Record<string, unknown> | undefined)?.type ?? "rectangle").replace("vertical_band", "full_height_x_band"),
      plane_fit_roi_x: Number((detect.plane_fit_roi as Record<string, unknown> | undefined)?.x ?? 0),
      plane_fit_roi_y: Number((detect.plane_fit_roi as Record<string, unknown> | undefined)?.y ?? 0),
      plane_fit_roi_width: Number((detect.plane_fit_roi as Record<string, unknown> | undefined)?.width ?? 0),
      plane_fit_roi_height: Number((detect.plane_fit_roi as Record<string, unknown> | undefined)?.height ?? 0),
      plane_fit_roi_points: ((detect.plane_fit_roi as Record<string, unknown> | undefined)?.points as Array<[number, number]> | undefined) ?? [],
      object_min_height_mm: Number(seg.min_height_mm ?? 8.0),
      object_max_height_mm: seg.max_height_mm == null ? null : Number(seg.max_height_mm),
      reference_tolerance_mm: Number(seg.reference_tolerance_mm ?? 0.0),
      suppress_plane_mask_in_segmentation: Boolean(seg.suppress_plane_mask_in_segmentation ?? true),
      morphology_kernel: Number(seg.morphology_kernel ?? 5),
      min_component_area: Number(seg.min_component_area ?? 120),
      fill_holes: Boolean(seg.fill_holes ?? true),
      smoothing_kernel: Number(seg.smoothing_kernel ?? 3),
      known_object_enabled: Boolean(known.enabled ?? false),
      known_object_apply_correction: Boolean(known.apply_correction ?? true),
      known_object_label: String(known.object_label ?? "cube"),
      known_object_target_selection: String(known.target_selection ?? "largest_component"),
      known_object_manual_component_id: Number(known.manual_component_id ?? 1),
      known_width_mm: Number(known.known_width_mm ?? 40),
      known_depth_mm: Number(known.known_depth_mm ?? 40),
      known_height_mm: Number(known.known_height_mm ?? 25),
      known_tolerance_percent: Number(known.tolerance_percent ?? 5),
      known_object_apply_persisted_correction: Boolean(known.apply_persisted_correction ?? true),
      known_object_persisted_scale_correction_x: known.persisted_scale_correction_x == null ? null : Number(known.persisted_scale_correction_x),
      known_object_persisted_scale_correction_y: known.persisted_scale_correction_y == null ? null : Number(known.persisted_scale_correction_y),
      known_object_persisted_scale_correction_z: known.persisted_scale_correction_z == null ? null : Number(known.persisted_scale_correction_z),
    };
    setPlaneQaParams(defaults);
    setPlaneQaPersistedParams(defaults);
    setPlaneQaDirty(false);
  }, [selectedPipeline?.id, detail?.take_id, detail?.result, pipelineDefaults25d]);
  useEffect(() => {
    if (selectedPipeline?.id !== "mining_steel_ball_classification_25d" || !selectedRecipeId) return;
    const selectedRecipeEntry = pipelineRecipes.find((recipe) => recipe.recipe_id === selectedRecipeId);
    if (!selectedRecipeEntry) return;
    const nextParams = recipeStageParamsToFlat25d((selectedRecipeEntry.stage_params as Record<string, unknown> | undefined) ?? {});
    setPlaneQaParams(nextParams);
    setPlaneQaPersistedParams(nextParams);
    setPlaneQaDirty(false);
  }, [selectedPipeline?.id, pipelineRecipes, selectedRecipeId]);
  const detectionViewResolutionError = useMemo(() => {
    if (canonicalSelectedStageId !== "detection") return null;
    const ids = new Set(stageSemantic.views.map((view) => view.id));
    if (ids.has("candidate_overlay") && ids.has("blob_metrics") && ids.has("candidates_table")) return null;
    return "Detection artifacts found, but no detection views were registered.";
  }, [canonicalSelectedStageId, stageSemantic.views]);
  const templateById = useMemo(() => new Map(processTemplates.map((template) => [template.id, template])), [processTemplates]);
  const thresholdStep = selectedPipelineInstance?.configured_steps.find((step) => step.step_id === "threshold");
  const morphologyStep = selectedPipelineInstance?.configured_steps.find((step) => step.step_id === "morphology");
  const ellipseStep = selectedPipelineInstance?.configured_steps.find((step) => step.step_id === "ellipse_fitting");
  const blobStep = selectedPipelineInstance?.configured_steps.find((step) => step.step_id === "blob_detection");
  const regionModeRaw = String(planeQaParams?.reference_surface_region_mode ?? "none").replace("vertical_band", "full_height_x_band");
  const roiEnabled = regionModeRaw !== "none";
  const roiType = (() => {
    const raw = regionModeRaw === "none" ? String(planeQaParams?.plane_fit_roi_type ?? "rectangle") : regionModeRaw;
    if (raw === "polygon" || raw === "full_height_x_band") return raw;
    return "rectangle";
  })() as RoiType;
  const activeRoi = roiEnabled && roiType !== "polygon"
    ? {
      x: Number(planeQaParams?.plane_fit_roi_x ?? 0),
      y: roiType === "full_height_x_band" ? 0 : Number(planeQaParams?.plane_fit_roi_y ?? 0),
      width: Number(planeQaParams?.plane_fit_roi_width ?? 0),
      height: Number(planeQaParams?.plane_fit_roi_height ?? 0),
    }
    : null;
  const activeRoiPolygon = roiEnabled && roiType === "polygon"
    ? (((planeQaParams?.plane_fit_roi_points as Array<[number, number]> | undefined) ?? []).map((point) => [Number(point[0]), Number(point[1])] as [number, number]))
    : null;
  useEffect(() => {
    if (!thresholdStep) {
      setSegmentationPersistedParams(null);
      setSegmentationPreviewParams(null);
      setSegmentationPreviewDirty(false);
      setSegmentationPreview(null);
      return;
    }
    const mode = String(thresholdStep.params?.mode ?? thresholdStep.params?.threshold_mode ?? "fixed").toLowerCase();
    const next: SegmentationTuningParams = {
      threshold: Math.max(0, Math.min(255, Number(thresholdStep.params?.value ?? thresholdStep.params?.threshold_value ?? 125))),
      auto_threshold: mode === "otsu",
      invert: Boolean(thresholdStep.params?.invert),
      roi_enabled: Boolean(thresholdStep.params?.roi_enabled),
      morph_op: String(morphologyStep?.params?.morph_op ?? morphologyStep?.params?.operation ?? "open_close"),
      erode_kernel_size: Number(morphologyStep?.params?.erode_kernel_size ?? morphologyStep?.params?.open_kernel ?? 3),
      erode_iterations: Number(morphologyStep?.params?.erode_iterations ?? morphologyStep?.params?.iterations ?? 1),
      dilate_kernel_size: Number(morphologyStep?.params?.dilate_kernel_size ?? morphologyStep?.params?.open_kernel ?? 3),
      dilate_iterations: Number(morphologyStep?.params?.dilate_iterations ?? morphologyStep?.params?.iterations ?? 1),
      close_kernel_size: Number(morphologyStep?.params?.close_kernel_size ?? morphologyStep?.params?.close_kernel ?? 5),
      fill_holes: Boolean(morphologyStep?.params?.fill_holes ?? true),
      min_area_px: Number(morphologyStep?.params?.min_area_px ?? morphologyStep?.params?.cleanup_min_area ?? 120),
      max_area_px: morphologyStep?.params?.max_area_px ?? morphologyStep?.params?.cleanup_max_area ?? null,
    };
    setSegmentationPersistedParams(next);
    setSegmentationPreviewParams(next);
    setSegmentationPreviewDirty(false);
    setSegmentationPreview(null);
    setSegmentationPreviewError(null);
  }, [selectedPipelineInstance?.id, thresholdStep?.params, morphologyStep?.params]);
  useEffect(() => {
    const schema = (selectedStage?.parameter_schema ?? {}) as { fields?: Record<string, { default?: unknown }> };
    const defaults = Object.fromEntries(Object.entries(schema.fields ?? {}).map(([key, meta]) => [key, meta.default]));
    const stepParams = ellipseStep?.params ?? {};
    const next = {
      ...defaults,
      ...stepParams,
      min_contour_points: stepParams.min_contour_points ?? stepParams.min_fit_points ?? defaults.min_contour_points,
    };
    setEllipsePersistedParams(next);
    setEllipsePreviewParams(next);
    setEllipsePreviewDirty(false);
    setEllipsePreviewError(null);
  }, [ellipseStep?.params, selectedStage?.parameter_schema]);

  const previewEnabled = Boolean(
    detail?.take_id
    && selectedPipeline?.id
    && selectedPipeline?.execution_backend === "process_service"
    && thresholdStep
    && segmentationPreviewParams
  );
  const activeAcquisitionGroupId = useMemo(() => {
    const takeMeta = detail?.take_metadata as Record<string, unknown> | undefined;
    const sourceMeta = detail?.metadata as Record<string, unknown> | undefined;
    const fromTake = takeMeta?.acquisition_group_id;
    if (typeof fromTake === "string" && fromTake) return fromTake;
    const fromSource = sourceMeta?.acquisition_group_id;
    if (typeof fromSource === "string" && fromSource) return fromSource;
    return null;
  }, [detail?.take_metadata, detail?.metadata]);
  const selectedSourceId = useMemo(() => {
    const sourceMeta = detail?.metadata as Record<string, unknown> | undefined;
    const sourceFromMeta = sourceMeta?.source;
    if (typeof sourceFromMeta === "string" && sourceFromMeta) return sourceFromMeta;
    const sourceFromRuntime = runtimeState?.acquisition_source;
    if (typeof sourceFromRuntime === "string" && sourceFromRuntime) return sourceFromRuntime;
    return null;
  }, [detail?.metadata, runtimeState?.acquisition_source]);
  const selectedModalityForBinding = useMemo(() => {
    if (bindingPurpose === "fusion") return "acquisition_group";
    return detail?.modalities?.includes("rgb") ? "rgb" : detail?.modalities?.includes("heightmap") ? "heightmap" : "rgb";
  }, [bindingPurpose, detail?.modalities]);
  const selectedBindingSource = useMemo(() => {
    if (bindingPurpose === "fusion") return activeAcquisitionGroupId || selectedSourceId;
    return selectedSourceId;
  }, [bindingPurpose, activeAcquisitionGroupId, selectedSourceId]);
  const activeBinding = useMemo(() => {
    if (!selectedBindingSource) return null;
    return processBindings.find(
      (binding) =>
        binding.enabled
        && binding.source_id === selectedBindingSource
        && binding.modality === selectedModalityForBinding
        && binding.purpose === bindingPurpose
    ) ?? null;
  }, [processBindings, selectedBindingSource, selectedModalityForBinding, bindingPurpose]);
  const rgbWorkerStatus = useMemo(
    () => runtimeWorkers.find((item) => item.worker_type.includes("rgb")) ?? null,
    [runtimeWorkers]
  );
  const worker25dStatus = useMemo(
    () => runtimeWorkers.find((item) => item.worker_type.includes("25d") || item.worker_type.includes("heightmap")) ?? null,
    [runtimeWorkers]
  );
  const fusionWorkerStatus = useMemo(
    () => runtimeWorkers.find((item) => item.worker_type.includes("fusion")) ?? null,
    [runtimeWorkers]
  );
  useEffect(() => {
    let active = true;
    if (!activeAcquisitionGroupId) {
      setFusionInputs(null);
      setFusionPreview(null);
      setFusionError(null);
      return;
    }
    setFusionLoading(true);
    setFusionError(null);
    void Promise.all([
      api.fusionInputs(activeAcquisitionGroupId),
      api.fusionPreview(activeAcquisitionGroupId),
      api.fusionRuns(activeAcquisitionGroupId),
      api.latestFusionResult(activeAcquisitionGroupId),
    ])
      .then(([inputs, preview, runs, latest]) => {
        if (!active) return;
        setFusionInputs(inputs);
        setFusionPreview(preview);
        setFusionRuns(runs.runs ?? []);
        setLatestFusionRun(latest.run ?? null);
        setLatestFusionResult(latest.result ?? null);
      })
      .catch((error: unknown) => {
        if (!active) return;
        setFusionInputs(null);
        setFusionPreview(null);
        setFusionRuns([]);
        setLatestFusionRun(null);
        setLatestFusionResult(null);
        setFusionError(error instanceof Error ? error.message : "Failed to load fusion readiness.");
      })
      .finally(() => {
        if (!active) return;
        setFusionLoading(false);
      });
    return () => {
      active = false;
    };
  }, [activeAcquisitionGroupId]);
  async function runFusionForGroup(force = false) {
    if (!activeAcquisitionGroupId) return;
    setFusionLoading(true);
    setFusionError(null);
    try {
      const created = await api.createFusionRun(activeAcquisitionGroupId, { force });
      const [runs, latest] = await Promise.all([
        api.fusionRuns(activeAcquisitionGroupId),
        api.latestFusionResult(activeAcquisitionGroupId),
      ]);
      setFusionRuns(runs.runs ?? []);
      setLatestFusionRun(latest.run ?? null);
      setLatestFusionResult(latest.result ?? created.result ?? null);
    } catch (error) {
      setFusionError(error instanceof Error ? error.message : "Fusion execution failed.");
    } finally {
      setFusionLoading(false);
    }
  }
  async function publish25dForSelectedTake() {
    if (!detail?.take_id) return;
    setFusionLoading(true);
    setFusionError(null);
    try {
      await api.publish25dResult(detail.take_id);
      const latest = await api.latestPublishedInspectionResult().catch(() => null as InspectionPublishedResult | null);
      setLatestPublishedResult(latest);
    } catch (error) {
      setFusionError(error instanceof Error ? error.message : "25D publish failed.");
    } finally {
      setFusionLoading(false);
    }
  }
  const ellipsePreviewEnabled = Boolean(
    detail?.take_id
    && selectedPipeline?.id
    && selectedPipeline?.execution_backend === "process_service"
    && ellipsePreviewParams
  );

  const workspaceDetail = useMemo(() => {
    if (!detail || !segmentationPreview || (canonicalSelectedStageId !== "segmentation" && canonicalSelectedStageId !== "ellipse_fitting")) return detail;
    if (!detail.result) return detail;
    return {
      ...detail,
      result: {
        ...detail.result,
        artifacts: segmentationPreview.artifacts,
      },
    };
  }, [detail, segmentationPreview, canonicalSelectedStageId]);
  const allObjects = objectsForTake(workspaceDetail);
  const focusedObject = selectedObject(allObjects, selectedObjectId);
  const allArtifacts = useMemo(
    () => artifactsForStage(workspaceDetail, canonicalSelectedStageId),
    [workspaceDetail, canonicalSelectedStageId],
  );
  const takeArtifacts = useMemo(
    () => artifactList(workspaceDetail),
    [workspaceDetail],
  );
  const currentArtifacts = useMemo(
    () => artifactsForObject(allArtifacts, selectedObjectId),
    [allArtifacts, selectedObjectId],
  );
  const stageDisplayPlan = useMemo(
    () => resolveStageDisplayPlan(canonicalSelectedStageId, stageSemantic, workspaceDetail, allArtifacts),
    [allArtifacts, canonicalSelectedStageId, stageSemantic, workspaceDetail]
  );
  const stageSubstagePlan = useMemo(
    () => resolveStageSubstagePlan(canonicalSelectedStageId, stageSemantic, selectedPipeline, workspaceDetail, allArtifacts, takeArtifacts),
    [allArtifacts, canonicalSelectedStageId, selectedPipeline, stageSemantic, takeArtifacts, workspaceDetail],
  );
  const artifactRelevanceById = useMemo(() => {
    const entries: Record<string, { relevance: ArtifactRelevance; reason?: string }> = {};
    for (const substage of stageSubstagePlan.substages) {
      for (const artifact of substage.supportingArtifacts) {
        entries[artifact.artifactId] = { relevance: artifact.category, reason: artifact.reason };
      }
    }
    return entries;
  }, [stageSubstagePlan.substages]);
  const pipelineProcessingUnits = useMemo(
    () => processingUnitsForPipeline(selectedPipeline, detail),
    [detail, selectedPipeline],
  );
  const stageProcessingUnits = useMemo(
    () => processingUnitsForStage(selectedPipeline, workspaceDetail, canonicalSelectedStageId),
    [canonicalSelectedStageId, selectedPipeline, workspaceDetail],
  );
  const stageRootUnit = useMemo(
    () => rootProcessingUnit(stageProcessingUnits),
    [stageProcessingUnits],
  );
  const detectStageSummary = useMemo(
    () => canonicalSelectedStageId === "detect_belt_plane"
      ? detectStrategySummary(workspaceDetail, allArtifacts, planeQaParams)
      : null,
    [allArtifacts, canonicalSelectedStageId, planeQaParams, workspaceDetail]
  );
  const detectReferenceRunState = useMemo(
    () => canonicalSelectedStageId === "detect_belt_plane"
      ? resolveDetectReferenceRunState(workspaceDetail, allArtifacts, planeQaParams)
      : null,
    [allArtifacts, canonicalSelectedStageId, planeQaParams, workspaceDetail]
  );
  const activeReferencePreset = useMemo(
    () => detectReferencePresetForValues(planeQaParams),
    [planeQaParams],
  );
  const persistedReferencePreset = useMemo(
    () => detectReferencePresetForValues(planeQaPersistedParams),
    [planeQaPersistedParams],
  );
  const detectActiveViewResolution = useMemo(
    () => canonicalSelectedStageId === "detect_belt_plane" && detectReferenceRunState
      ? resolveDetectViewArtifacts(activeTab, currentArtifacts, detectReferenceRunState)
      : null,
    [activeTab, canonicalSelectedStageId, currentArtifacts, detectReferenceRunState]
  );
  const activeDisplayItem = useMemo(
    () => stageDisplayPlan.items.find((item) => item.viewId === activeTab) ?? null,
    [activeTab, stageDisplayPlan.items]
  );
  const referenceViewSelection = useMemo(
    () => stageSubstagePlan.substages.length > 0
      ? resolveReferenceViewSelection(stageSubstagePlan, referenceViewScope, selectedSubstageId, activeTab)
      : { activeScope: "strategy_path" as const, activeSubstageId: null, inferredSubstageId: null },
    [activeTab, referenceViewScope, selectedSubstageId, stageSubstagePlan],
  );
  const activeReferenceViewScope = useMemo(
    () => stageSubstagePlan.substages.length > 0
      ? referenceViewSelection.activeScope
      : "strategy_path",
    [referenceViewSelection.activeScope, stageSubstagePlan.substages.length],
  );
  const activeSubstage = useMemo<ResolvedStageSubstage | null>(
    () => stageSubstagePlan.substages.find((substage) => substage.substageId === referenceViewSelection.activeSubstageId) ?? null,
    [referenceViewSelection.activeSubstageId, stageSubstagePlan.substages],
  );
  const tuneFocusedSubstage = useMemo<ResolvedStageSubstage | null>(
    () => activeSubstage
      ?? stageSubstagePlan.substages.find((substage) => substage.substageId === referenceViewSelection.inferredSubstageId)
      ?? null,
    [activeSubstage, referenceViewSelection.inferredSubstageId, stageSubstagePlan.substages],
  );
  const tuneFocusedProcessingUnit = useMemo(
    () => processingUnitById(
      stageProcessingUnits,
      tuneFocusedSubstage?.unitId ?? selectedContractUnitId ?? stageRootUnit?.id,
    ),
    [selectedContractUnitId, stageProcessingUnits, stageRootUnit?.id, tuneFocusedSubstage],
  );
  const activeProcessingUnit = useMemo(
    () => processingUnitById(
      stageProcessingUnits,
      activeSubstage?.unitId ?? selectedContractUnitId ?? stageRootUnit?.id,
    ),
    [activeSubstage?.unitId, selectedContractUnitId, stageProcessingUnits, stageRootUnit?.id],
  );
  const activeProcessingUnitView = useMemo(
    () => activeProcessingUnit
      ? resolveProcessingUnitView({
        detail: workspaceDetail,
        unit: activeProcessingUnit,
        viewId: activeTab,
        semanticViews: stageSemantic.views,
        stageArtifacts: allArtifacts,
        takeArtifacts,
      })
      : null,
    [activeProcessingUnit, activeTab, allArtifacts, stageSemantic.views, takeArtifacts, workspaceDetail],
  );
  const activeProcessingUnitIO = useMemo(
    () => activeProcessingUnit
      ? resolveProcessingUnitArtifactRefs(workspaceDetail, activeProcessingUnit, allArtifacts, takeArtifacts)
      : { inputs: [], outputs: [] },
    [activeProcessingUnit, allArtifacts, takeArtifacts, workspaceDetail],
  );
  const activeTraceEntry = useMemo(
    () => activeProcessingUnit?.id
      ? workspaceDetail?.result?.processing_unit_trace?.unit_results?.[activeProcessingUnit.id]
        ?? workspaceDetail?.result?.processing_unit_trace?.units?.[activeProcessingUnit.id]
        ?? null
      : null,
    [activeProcessingUnit?.id, workspaceDetail?.result?.processing_unit_trace?.unit_results, workspaceDetail?.result?.processing_unit_trace?.units],
  );
  const activeTraceQuality = useMemo(
    () => stageProcessingUnits.length > 0
      ? {
        source: workspaceDetail?.result?.processing_unit_trace?.trace_source ?? "best_effort_artifact_registry",
        precision: workspaceDetail?.result?.processing_unit_trace?.trace_precision ?? "artifact_level",
      }
      : null,
    [stageProcessingUnits.length, workspaceDetail?.result?.processing_unit_trace?.trace_precision, workspaceDetail?.result?.processing_unit_trace?.trace_source],
  );
  const activeLastRunStageParams = useMemo(
    () => {
      const stageParams = (workspaceDetail?.result?.recipe_snapshot?.stage_params as Record<string, unknown> | undefined) ?? {};
      const key = contractRuntimeStageParamsKey(activeProcessingUnit?.stage_id ?? canonicalSelectedStageId);
      const raw = stageParams[key];
      return raw && typeof raw === "object" ? raw as Record<string, unknown> : null;
    },
    [activeProcessingUnit?.stage_id, canonicalSelectedStageId, workspaceDetail?.result?.recipe_snapshot],
  );
  const activeContractStageValues = useMemo(() => {
    if (!planeQaParams || !activeProcessingUnit) return null;
    const next: Record<string, unknown> = {};
    const keys = new Set<string>();
    for (const param of activeProcessingUnit.parameters ?? []) keys.add(param.id);
    for (const param of stageRootUnit?.parameters ?? []) keys.add(param.id);
    for (const key of keys) {
      next[key] = planeQaParams[contractParamFlatKey(activeProcessingUnit.stage_id, key)];
    }
    return next;
  }, [activeProcessingUnit, planeQaParams, stageRootUnit?.parameters]);
  const activeContractPersistedValues = useMemo(() => {
    if (!planeQaPersistedParams || !activeProcessingUnit) return null;
    const next: Record<string, unknown> = {};
    const keys = new Set<string>();
    for (const param of activeProcessingUnit.parameters ?? []) keys.add(param.id);
    for (const param of stageRootUnit?.parameters ?? []) keys.add(param.id);
    for (const key of keys) {
      next[key] = planeQaPersistedParams[contractParamFlatKey(activeProcessingUnit.stage_id, key)];
    }
    return next;
  }, [activeProcessingUnit, planeQaPersistedParams, stageRootUnit?.parameters]);
  const activeLastRunContractValues = useMemo(() => {
    if (!activeLastRunStageParams || !activeProcessingUnit) return null;
    const next: Record<string, unknown> = {};
    const keys = new Set<string>();
    for (const param of activeProcessingUnit.parameters ?? []) keys.add(param.id);
    for (const param of stageRootUnit?.parameters ?? []) keys.add(param.id);
    for (const key of keys) {
      next[key] = activeLastRunStageParams[contractParamFlatKey(activeProcessingUnit.stage_id, key)];
    }
    return next;
  }, [activeLastRunStageParams, activeProcessingUnit, stageRootUnit?.parameters]);
  const current25dStageParams = useMemo(
    () => build25dStageParams(),
    [planeQaParams, selectedPipeline?.id],
  );
  const selectedRecipe = useMemo(
    () => pipelineRecipes.find((recipe) => recipe.recipe_id === selectedRecipeId) ?? null,
    [pipelineRecipes, selectedRecipeId],
  );
  const pipelineRecipeDiff = useMemo(
    () => buildRecipeDiff(selectedRecipe, current25dStageParams, pipelineProcessingUnits),
    [current25dStageParams, pipelineProcessingUnits, selectedRecipe],
  );
  const currentParametersByUnit = useMemo(
    () => stageParamsToRecipeParametersByUnit(current25dStageParams, pipelineProcessingUnits),
    [current25dStageParams, pipelineProcessingUnits],
  );
  const lastRunParametersByUnit = useMemo(() => {
    const snapshot = workspaceDetail?.result?.recipe_snapshot;
    if (!snapshot) return {};
    const grouped = snapshot.parameters_by_unit;
    if (grouped && typeof grouped === "object") return grouped as Record<string, Record<string, unknown>>;
    return stageParamsToRecipeParametersByUnit(
      (snapshot.stage_params as Record<string, unknown> | undefined) ?? null,
      pipelineProcessingUnits,
    );
  }, [pipelineProcessingUnits, workspaceDetail?.result?.recipe_snapshot]);
  const pipelineLastRunDiff = useMemo(
    () => buildParameterDiffGroups(lastRunParametersByUnit, currentParametersByUnit, pipelineProcessingUnits),
    [currentParametersByUnit, lastRunParametersByUnit, pipelineProcessingUnits],
  );
  const activeRecipeUnitValues = useMemo(
    () => recipeValuesForUnit(selectedRecipe, activeProcessingUnit, stageProcessingUnits),
    [selectedRecipe, activeProcessingUnit, stageProcessingUnits],
  );
  const currentRecipeDiff = useMemo(
    () => buildRecipeDiff(selectedRecipe, current25dStageParams, stageProcessingUnits),
    [current25dStageParams, selectedRecipe, stageProcessingUnits],
  );
  const currentRecipeDirty = useMemo(
    () => recipeDirty(selectedRecipe, current25dStageParams, stageProcessingUnits),
    [current25dStageParams, selectedRecipe, stageProcessingUnits],
  );
  const currentVsLastRunDirty = useMemo(
    () => pipelineLastRunDiff.length > 0,
    [pipelineLastRunDiff.length],
  );
  const runActionSummary = useMemo(() => {
    if (selectedPipeline?.id !== "mining_steel_ball_classification_25d") {
      return "Run uses current pipeline parameters.";
    }
    if (!selectedRecipe) {
      return currentRecipeDirty
        ? "Run will use edited form overrides only (no recipe selected)."
        : "Run will use current form values only (no recipe selected).";
    }
    return currentRecipeDirty
      ? `Run will use ${selectedRecipe.name} v${selectedRecipe.version} plus edited overrides.`
      : `Run will use ${selectedRecipe.name} v${selectedRecipe.version} without edited overrides.`;
  }, [selectedPipeline?.id, selectedRecipe, currentRecipeDirty]);
  const orderedComparisonUnitEntries = useMemo(
    () => orderedComparisonUnits(comparisonResult),
    [comparisonResult],
  );
  const processingUnitGraph = useMemo(
    () => buildProcessingUnitGraph({
      units: pipelineProcessingUnits,
      detail,
      recipe: selectedRecipe,
      recipeDiffGroups: pipelineRecipeDiff,
      lastRunDiffGroups: pipelineLastRunDiff,
      comparison: comparisonResult,
      partialRerunPlan,
      partialRerunPlanStale,
      persistedTrace: persistedTraceFromDetail(detail),
    }),
    [comparisonResult, detail, partialRerunPlan, partialRerunPlanStale, pipelineLastRunDiff, pipelineProcessingUnits, pipelineRecipeDiff, selectedRecipe],
  );
  const graphTraceSummary = useMemo(
    () => traceSummaryFromDetail(workspaceDetail),
    [workspaceDetail],
  );
  const graphWorkspaceContext = useMemo(
    () => ({
      hasContract: pipelineProcessingUnits.length > 0,
      hasTake: Boolean(selectedTakeId),
      hasRunResult: Boolean(workspaceDetail?.result),
      hasRecipe: Boolean(selectedRecipe),
      hasComparison: Boolean(comparisonResult),
      traceSummary: graphTraceSummary,
    }),
    [comparisonResult, graphTraceSummary, pipelineProcessingUnits.length, selectedRecipe, selectedTakeId, workspaceDetail?.result],
  );
  const graphWorkspaceLineageSummary = useMemo(() => {
    const executionMode = workspaceDetail?.result?.execution_mode as string | null | undefined;
    if (!isPartialRerunExecutionMode(executionMode)) return null;
    const parentRunId = workspaceDetail?.result?.parent_run_id as string | null | undefined;
    const boundaryStageId = (workspaceDetail?.result?.recipe_snapshot as Record<string, unknown> | undefined)?.boundary_stage_id as string | null | undefined;
    return partialRunNoticeText({ parentRunId, boundaryStageId });
  }, [workspaceDetail?.result]);
  const graphWorkspaceActiveRunTypeLabel = useMemo(
    () => (isPartialRerunExecutionMode(workspaceDetail?.result?.execution_mode as string | null | undefined) ? "Partial rerun" : "Full run"),
    [workspaceDetail?.result?.execution_mode],
  );
  const graphWorkspaceStaleWarnings = useMemo(() => {
    const activeRunId = typeof workspaceDetail?.result?.run_id === "string" ? workspaceDetail.result.run_id : null;
    const activeRunEntry = pipelineRuns.find((run) => run.run_id === activeRunId) ?? null;
    return computeStaleWarnings({
      planExists: partialRerunPlan != null,
      planIsStale: partialRerunPlanStale,
      planContext,
      currentRecipeId: selectedRecipe?.recipe_id ?? null,
      currentRecipeVersion: selectedRecipe?.version ?? null,
      activeRunId,
      activeRunExecutionMode: (workspaceDetail?.result?.execution_mode as string | null | undefined) ?? null,
      activeRunParentRunId: (workspaceDetail?.result?.parent_run_id as string | null | undefined) ?? activeRunEntry?.parent_run_id ?? null,
      activeRunComparisonId: activeRunEntry?.comparison_id ?? null,
      activeRunContractFingerprint:
        (workspaceDetail?.result?.recipe_snapshot as Record<string, unknown> | undefined)?.processing_unit_contract_fingerprint as string | undefined ?? null,
      currentContractFingerprint: selectedRecipe?.processing_unit_contract_fingerprint ?? null,
    });
  }, [partialRerunPlan, partialRerunPlanStale, planContext, pipelineRuns, selectedRecipe, workspaceDetail?.result]);
  const selectedGraphNodeId = useMemo(
    () => activeProcessingUnit?.id ?? stageRootUnit?.id ?? null,
    [activeProcessingUnit?.id, stageRootUnit?.id],
  );
  const activateStrategyPath = useCallback(() => {
    setReferenceViewScope("strategy_path");
    setSelectedSubstageId(null);
    if (stageDisplayPlan.items.some((item) => item.view.id === activeTab)) return;
    const fallbackViewId = stageDisplayPlan.items[0]?.view.id;
    if (fallbackViewId) setActiveTab(fallbackViewId as StageViewTabId);
  }, [activeTab, stageDisplayPlan.items]);
  const activateReferenceSubstage = useCallback((substageId: string) => {
    setReferenceViewScope("substage");
    setSelectedSubstageId(substageId);
  }, []);
  const focusProcessingUnitFromGraph = useCallback((unitId: string) => {
    const unit = pipelineProcessingUnits.find((entry) => entry.id === unitId);
    if (!unit) return;
    const resolvedStageId = resolveStudioStageId(selectedPipeline, unit.stage_id);
    if (resolvedStageId) setSelectedStageId(resolvedStageId);
    const selection = graphSelectionForUnit(unit);
    if (selection.substageId) {
      setReferenceViewScope("substage");
      setSelectedSubstageId(selection.substageId);
    } else {
      setReferenceViewScope("strategy_path");
      setSelectedSubstageId(null);
    }
    setSelectedContractUnitId(selection.unitId);
    setInspectorExpanded(true);
  }, [pipelineProcessingUnits, selectedPipeline]);
  const focusSelectedSupportLineageStep = useCallback((stepId: string, options?: { openTune?: boolean }) => {
    const substageId = stepId === "reference_model_support" || stepId === "suppression_mask"
      ? "reference_model"
      : "selected_support";
    setReferenceViewScope("substage");
    setSelectedSubstageId(substageId);
    if (options?.openTune ?? true) {
      setControlPanelTab("tune");
      setStudioMode("tune");
      setInspectorExpanded(true);
    }
  }, []);
  const focusArtifactForTuning = useCallback((artifactId: string) => {
    const substage = stageSubstagePlan.substages.find((entry) => entry.artifactIds.includes(artifactId));
    if (!substage) return;
    setReferenceViewScope("substage");
    setSelectedSubstageId(substage.substageId);
    setSelectedArtifactId(artifactId);
    setControlPanelTab("tune");
    setStudioMode("tune");
    setInspectorExpanded(true);
    if (substage.recommendedViewId) setActiveTab(substage.recommendedViewId as StageViewTabId);
  }, [stageSubstagePlan.substages]);
  useEffect(() => {
    if (partialRerunPlan == null) return;
    setPartialRerunPlanStale(true);
  }, [current25dStageParams, selectedGraphNodeId, selectedPipeline?.id, selectedTakeId]);
  useEffect(() => {
    setPartialRerunExecution(null);
    setPlanContext(null);
  }, [selectedPipeline?.id, selectedTakeId]);
  const focusedArtifact = currentArtifacts.find((artifact) => artifact.artifact_id === selectedArtifactId)
    ?? (activeDisplayItem?.primaryArtifactId ? allArtifacts.find((artifact) => artifact.artifact_id === activeDisplayItem.primaryArtifactId) ?? null : null)
    ?? detectActiveViewResolution?.primary
    ?? currentArtifacts[0]
    ?? null;
  const previewTargetArtifact = useMemo(
    () => resolveOverlayTarget(currentArtifacts, focusedArtifact).target ?? focusedArtifact,
    [currentArtifacts, focusedArtifact],
  );
  const overlayTargetResolution = useMemo(
    () => resolveOverlayTarget(currentArtifacts, focusedArtifact),
    [currentArtifacts, focusedArtifact],
  );
  const activeComparisonUnit = useMemo(
    () => activeProcessingUnit?.id ? comparisonResult?.units?.[activeProcessingUnit.id] ?? null : null,
    [activeProcessingUnit?.id, comparisonResult],
  );
  const openArtifactFromWorkspace = useCallback((artifactId: string) => {
    setSelectedArtifactId(artifactId);
    setControlPanelTab("view");
    setStudioMode("inspect");
    setGraphWorkspaceExpanded(false);
  }, []);
  const graphWorkspaceTuneContent = useMemo(() => {
    if (!planeQaParams || !activeProcessingUnit) {
      return <div className="empty-state compact">Select a unit with declared parameters to edit it from the graph.</div>;
    }
    if (canonicalSelectedStageId === "detect_belt_plane") {
      return (
        <DetectReferenceTunePanel
          activePreset={activeReferencePreset}
          availableUnits={stageProcessingUnits}
          dirty={planeQaDirty}
          formDiffersFromRun={detectReferenceRunState ? detectReferenceFormDiffersFromRun(detectReferenceRunState, planeQaParams) : false}
          hasTuneChanges={Boolean(planeQaDirty || segmentationPreviewDirty || ellipsePreviewDirty)}
          loading={pipelineActionBusy}
          onApply={() => applyPlaneQaParams(false)}
          onApplyPreset={(presetId, rerun) => applyReferenceMethodPreset(presetId, rerun)}
          onApplyRun={() => applyPlaneQaParams(true)}
          onChange={(key, value) => {
            setPlaneQaParams((current) => {
              if (!current) return current;
              const next = { ...current, [key]: value };
              setPlaneQaDirty(!planeQaParamsEqual(next, planeQaPersistedParams));
              return next;
            });
          }}
          onCopyPresetJson={() => void copyReferenceMethodPresetJson()}
          onDiscard={() => { setPlaneQaParams(planeQaPersistedParams); setPlaneQaDirty(false); }}
          onResetFieldToRecipe={(key) => resetRecipeFieldToPersisted("detect_belt_plane", key)}
          onResetUnitToRecipe={() => resetRecipeUnitToPersisted(tuneFocusedProcessingUnit)}
          onResetPresetOverrides={() => resetReferenceMethodPresetOverrides()}
          onStartVisualRoiEdit={() => startVisualRoiEdit()}
          onUseFullFrame={() => void clearRoi()}
          persistedPreset={persistedReferencePreset}
          persistedValues={planeQaPersistedParams}
          recipeValues={activeRecipeUnitValues}
          runState={detectReferenceRunState}
          lastRunValues={activeLastRunStageParams}
          selectedArtifactId={focusedArtifact?.artifact_id ?? null}
          selectedSubstage={tuneFocusedSubstage}
          selectedUnit={tuneFocusedProcessingUnit}
          selectedViewLabel={activeTab}
          stage={selectedStage}
          strategyLabel={detectStageSummary?.strategyLabel ?? "Unknown"}
          values={planeQaParams}
        />
      );
    }
    if (!activeContractStageValues) {
      return <div className="empty-state compact">This unit has no tunable parameters. It is controlled by upstream contract state.</div>;
    }
    return (
      <ContractTunePanel
        availableUnits={stageProcessingUnits}
        dirty={planeQaDirty}
        lastRunValues={activeLastRunStageParams}
        loading={pipelineActionBusy}
        onApply={() => applyPlaneQaParams(false)}
        onApplyRun={() => applyPlaneQaParams(true)}
        onChange={(key, value) => {
          setPlaneQaParams((current) => {
            if (!current) return current;
            const next = { ...current, [contractParamFlatKey(activeProcessingUnit.stage_id, key)]: value };
            setPlaneQaDirty(!planeQaParamsEqual(next, planeQaPersistedParams));
            return next;
          });
        }}
        onDiscard={() => { setPlaneQaParams(planeQaPersistedParams); setPlaneQaDirty(false); }}
        onResetFieldToRecipe={(key) => resetRecipeFieldToPersisted(activeProcessingUnit.stage_id, key)}
        onResetUnitToRecipe={() => resetRecipeUnitToPersisted(activeProcessingUnit)}
        persistedValues={activeContractPersistedValues}
        recipeValues={activeRecipeUnitValues}
        selectedUnit={activeProcessingUnit}
        stageLabel={selectedStage?.display_name ?? canonicalSelectedStageId}
        values={activeContractStageValues}
      />
    );
  }, [
    activeContractPersistedValues,
    activeContractStageValues,
    activeLastRunStageParams,
    activeProcessingUnit,
    activeRecipeUnitValues,
    activeReferencePreset,
    activeTab,
    canonicalSelectedStageId,
    detectReferenceRunState,
    detectStageSummary?.strategyLabel,
    ellipsePreviewDirty,
    focusedArtifact?.artifact_id,
    persistedReferencePreset,
    pipelineActionBusy,
    planeQaDirty,
    planeQaParams,
    planeQaPersistedParams,
    selectedStage,
    segmentationPreviewDirty,
    stageProcessingUnits,
    tuneFocusedProcessingUnit,
    tuneFocusedSubstage,
  ]);
  const useBinaryMaskView = useMemo(
    () => (focusedArtifact ? shouldUseBinaryMaskRenderer(focusedArtifact, currentArtifacts) : false),
    [currentArtifacts, focusedArtifact],
  );
  const binaryMaskSourceArtifacts = useMemo(
    () => sourceArtifactsForTake(workspaceDetail, focusedArtifact?.stage_id || canonicalSelectedStageId || "input"),
    [workspaceDetail, focusedArtifact?.stage_id, canonicalSelectedStageId],
  );
  const binaryMaskResolverInput = useMemo(
    () => focusedArtifact ? ({
      maskArtifact: focusedArtifact,
      selectedStageId: focusedArtifact.stage_id,
      selectedTake: workspaceDetail,
      currentStageArtifacts: currentArtifacts,
      allRunArtifacts: allArtifacts,
      sourceArtifacts: binaryMaskSourceArtifacts,
      resolvedArtifacts: [...currentArtifacts, ...binaryMaskSourceArtifacts],
    }) : null,
    [focusedArtifact, workspaceDetail, currentArtifacts, allArtifacts, binaryMaskSourceArtifacts],
  );
  const binaryMaskReferenceCandidates = useMemo(
    () => binaryMaskResolverInput && useBinaryMaskView ? resolveBinaryMaskReferenceCandidates(binaryMaskResolverInput) : [],
    [binaryMaskResolverInput, useBinaryMaskView],
  );
  const binaryMaskSelectedReference = useMemo(
    () => {
      if (!useBinaryMaskView) return null;
      return binaryMaskReferenceCandidates.find((candidate) => candidate.artifact.artifact_id === viewState.binaryMaskReferenceArtifactId)
        ?? binaryMaskReferenceCandidates[0]
        ?? null;
    },
    [binaryMaskReferenceCandidates, useBinaryMaskView, viewState.binaryMaskReferenceArtifactId],
  );
  const binaryMaskSelectedReferenceUrl = useMemo(
    () => {
      if (!workspaceDetail || !binaryMaskSelectedReference) return null;
      if (binaryMaskSelectedReference.sourceUrl) return binaryMaskSelectedReference.sourceUrl;
      return binaryMaskSelectedReference.artifact.path
        ? fileUrl(workspaceDetail.take_id, binaryMaskSelectedReference.artifact.path)
        : null;
    },
    [binaryMaskSelectedReference, workspaceDetail],
  );
  const binaryMaskReferenceResolution = useMemo(
    () => binaryMaskResolverInput && useBinaryMaskView
      ? resolveBinaryMaskReference(
          binaryMaskResolverInput,
          binaryMaskSelectedReference?.artifact.artifact_id ?? viewState.binaryMaskReferenceArtifactId,
          binaryMaskSelectedReferenceUrl,
        )
      : null,
    [binaryMaskResolverInput, binaryMaskSelectedReference?.artifact.artifact_id, binaryMaskSelectedReferenceUrl, useBinaryMaskView, viewState.binaryMaskReferenceArtifactId],
  );
  const binaryMaskAvailableModes = useMemo(
    () => focusedArtifact && useBinaryMaskView && binaryMaskReferenceResolution
      ? availableBinaryMaskViewModes(focusedArtifact, currentArtifacts, binaryMaskReferenceResolution)
      : [],
    [focusedArtifact, useBinaryMaskView, currentArtifacts, binaryMaskReferenceResolution],
  );
  const selectionDrivenViewState = useMemo<StudioArtifactViewState>(() => {
    const preferredViewportMode = preferredViewportModeForArtifact(previewTargetArtifact);
    return {
      viewportMode: preferredViewportMode,
      overlayMode: "all",
      binaryMaskMode: focusedArtifact && useBinaryMaskView && binaryMaskReferenceResolution
        ? defaultBinaryMaskViewMode(focusedArtifact, currentArtifacts, binaryMaskReferenceResolution)
        : "mask",
      binaryMaskOpacity: 0.52,
      binaryMaskReferenceOpacity: 1,
      binaryMaskReferenceArtifactId: binaryMaskReferenceCandidates[0]?.artifact.artifact_id ?? null,
    };
  }, [
    previewTargetArtifact,
    focusedArtifact,
    useBinaryMaskView,
    binaryMaskReferenceResolution,
    currentArtifacts,
    binaryMaskReferenceCandidates,
  ]);
  const detectFormDiffersFromRun = useMemo(
    () => canonicalSelectedStageId === "detect_belt_plane" && detectReferenceRunState
      ? detectReferenceFormDiffersFromRun(detectReferenceRunState, planeQaParams)
      : false,
    [canonicalSelectedStageId, detectReferenceRunState, planeQaParams]
  );
  const detectArtifactAuditRows = useMemo(
    () => canonicalSelectedStageId === "detect_belt_plane"
      ? resolveDetectArtifactAuditRows(stageSemantic.views, allArtifacts, workspaceDetail, planeQaParams)
      : [],
    [allArtifacts, canonicalSelectedStageId, planeQaParams, stageSemantic.views, workspaceDetail]
  );
  const requestedStudioPhysicalObjectMatchId = useMemo(() => {
    const requestedPhysicalObjectId = String(studioDeepLink.physical_object_id ?? "").trim();
    if (!requestedPhysicalObjectId || !Array.isArray(detail?.object_annotations)) return null;
    for (const entry of detail.object_annotations) {
      if (!entry || typeof entry !== "object") continue;
      const row = entry as Record<string, unknown>;
      const candidatePhysicalObjectId = String(
        row.physical_object_id
        ?? row.object_physical_id
        ?? row.matched_physical_object_id
        ?? "",
      ).trim();
      if (candidatePhysicalObjectId !== requestedPhysicalObjectId) continue;
      const candidateObjectId = studioDeepLinkObjectId(String(row.matched_candidate_id ?? row.candidate_id ?? row.object_id ?? ""));
      if (candidateObjectId != null) return candidateObjectId;
    }
    return null;
  }, [detail?.object_annotations, studioDeepLink.physical_object_id]);
  const studioDeepLinkObjectResolved = useMemo(() => {
    if (!studioDeepLink.object_id && !studioDeepLink.physical_object_id) return true;
    if (requestedStudioObjectId != null) return allObjects.some((object) => object.object_id === requestedStudioObjectId);
    if (requestedStudioPhysicalObjectMatchId != null) return allObjects.some((object) => object.object_id === requestedStudioPhysicalObjectMatchId);
    return false;
  }, [allObjects, requestedStudioObjectId, requestedStudioPhysicalObjectMatchId, studioDeepLink.object_id, studioDeepLink.physical_object_id]);
  useEffect(() => {
    if (focusedArtifact?.object_id != null && focusedArtifact.object_id !== selectedObjectId) {
      setSelectedObjectId(focusedArtifact.object_id);
    }
  }, [focusedArtifact?.artifact_id, selectedObjectId]);
  const viewStateSelectionKey = useMemo(
    () => [
      focusedArtifact?.artifact_id ?? "",
      activeTab,
      canonicalSelectedStageId,
      selectedObjectId ?? "",
      useBinaryMaskView ? "binary-mask" : "standard",
      binaryMaskReferenceCandidates[0]?.artifact.artifact_id ?? "",
    ].join("|"),
    [
      activeTab,
      binaryMaskReferenceCandidates,
      canonicalSelectedStageId,
      focusedArtifact?.artifact_id,
      selectedObjectId,
      useBinaryMaskView,
    ],
  );
  const lastViewStateSelectionKeyRef = useRef(viewStateSelectionKey);
  useEffect(() => {
    if (lastViewStateSelectionKeyRef.current === viewStateSelectionKey) return;
    lastViewStateSelectionKeyRef.current = viewStateSelectionKey;
    setViewState(selectionDrivenViewState);
  }, [selectionDrivenViewState, viewStateSelectionKey]);
  useEffect(() => {
    if (!useBinaryMaskView) return;
    const nextReferenceArtifactId = binaryMaskReferenceCandidates[0]?.artifact.artifact_id ?? null;
    if (viewState.binaryMaskReferenceArtifactId && binaryMaskReferenceCandidates.some((candidate) => candidate.artifact.artifact_id === viewState.binaryMaskReferenceArtifactId)) return;
    if (viewState.binaryMaskReferenceArtifactId === nextReferenceArtifactId) return;
    setViewState((current) => ({
      ...current,
      binaryMaskReferenceArtifactId: nextReferenceArtifactId,
    }));
  }, [binaryMaskReferenceCandidates, useBinaryMaskView, viewState.binaryMaskReferenceArtifactId]);
  useEffect(() => {
    if (!useBinaryMaskView || !binaryMaskAvailableModes.length || binaryMaskAvailableModes.includes(viewState.binaryMaskMode)) return;
    const nextMode = binaryMaskAvailableModes[0] ?? "mask";
    if (viewState.binaryMaskMode === nextMode) return;
    setViewState((current) => ({ ...current, binaryMaskMode: nextMode }));
  }, [binaryMaskAvailableModes, useBinaryMaskView, viewState.binaryMaskMode]);

  useEffect(() => {
    if (!previewEnabled || !segmentationPreviewDirty || !segmentationPreviewParams || !detail || !selectedPipeline) return;
    const timer = window.setTimeout(() => {
      setSegmentationPreviewLoading(true);
      setSegmentationPreviewError(null);
      void api.previewSegmentation({
        take_id: detail.take_id,
        pipeline_id: selectedPipeline.id,
        params: segmentationPreviewParams,
      }).then((response) => {
        setSegmentationPreview(response);
      }).catch((err: unknown) => {
        setSegmentationPreviewError(err instanceof Error ? err.message : "Preview failed");
      }).finally(() => {
        setSegmentationPreviewLoading(false);
      });
    }, 320);
    return () => window.clearTimeout(timer);
  }, [previewEnabled, segmentationPreviewDirty, segmentationPreviewParams, detail?.take_id, selectedPipeline?.id]);
  useEffect(() => {
    if (!ellipsePreviewEnabled || !ellipsePreviewDirty || !ellipsePreviewParams || !detail || !selectedPipeline) return;
    if (canonicalSelectedStageId !== "ellipse_fitting") return;
    const timer = window.setTimeout(() => {
      setEllipsePreviewLoading(true);
      setEllipsePreviewError(null);
      void api.previewSegmentation({
        take_id: detail.take_id,
        pipeline_id: selectedPipeline.id,
        stage_id: "ellipse_fitting",
        params: ellipsePreviewParams,
      }).then((response) => {
        setSegmentationPreview(response);
      }).catch((err: unknown) => {
        setEllipsePreviewError(err instanceof Error ? err.message : "Preview failed");
      }).finally(() => {
        setEllipsePreviewLoading(false);
      });
    }, 320);
    return () => window.clearTimeout(timer);
  }, [ellipsePreviewEnabled, ellipsePreviewDirty, ellipsePreviewParams, detail?.take_id, selectedPipeline?.id, canonicalSelectedStageId]);
  const filteredTakes = takes.filter((take) => {
    return takeMatchesStudioFilters(take);
  });
  const selectedDatasetInfo = useMemo(
    () => datasets.find((item) => item.id === selectedDataset) ?? null,
    [datasets, selectedDataset],
  );
  const selectedDatasetSessionInfo = useMemo(
    () => datasetSessions.find((item) => item.id === selectedExperimentSession) ?? null,
    [datasetSessions, selectedExperimentSession],
  );
  const selectedSessionTags = useMemo(
    () => (selectedDatasetSessionInfo?.tags ?? []).map((item) => String(item)).filter(Boolean),
    [selectedDatasetSessionInfo?.tags],
  );
  const selectedTakeSummary = useMemo(
    () => takes.find((item) => item.take_id === selectedTakeId) ?? null,
    [takes, selectedTakeId],
  );
  const selectedContextDatasetName = selectedTakeSummary?.dataset_name || selectedDatasetInfo?.name || (selectedDataset || "All datasets");
  const selectedContextSessionName = selectedTakeSummary?.experiment_session_name || selectedDatasetSessionInfo?.name || (selectedExperimentSession || "All experiment sessions");
  const selectedContextSessionType = String(selectedTakeSummary?.experiment_session_type || selectedDatasetSessionInfo?.session_type || "engineering");
  const activeFilterChips = useMemo(() => {
    const chips: string[] = [];
    if (selectedDataset) chips.push(`dataset:${selectedDataset}`);
    if (selectedExperimentSession) chips.push(`experiment_session:${selectedExperimentSession}`);
    if (selectedSession) chips.push(`acquisition_session:${selectedSession}`);
    if (takeSearch) chips.push(`search:${takeSearch}`);
    if (showArchived) chips.push("show_archived");
    return chips;
  }, [selectedDataset, selectedExperimentSession, selectedSession, takeSearch, showArchived]);
  const allVisibleTakeIds = useMemo(() => filteredTakes.map((take) => take.take_id), [filteredTakes]);
  const selectedVisibleCount = useMemo(
    () => allVisibleTakeIds.filter((takeId) => selectedTakeIds.has(takeId)).length,
    [allVisibleTakeIds, selectedTakeIds]
  );
  const allVisibleSelected = allVisibleTakeIds.length > 0 && selectedVisibleCount === allVisibleTakeIds.length;
  const bulkSelectionCount = selectedTakeIds.size;
  useEffect(() => {
    const visible = new Set(allVisibleTakeIds);
    setSelectedTakeIds((current) => {
      const next = new Set<string>();
      current.forEach((takeId) => {
        if (visible.has(takeId)) next.add(takeId);
      });
      return next.size === current.size ? current : next;
    });
  }, [allVisibleTakeIds]);
  const familyStatus = useMemo(
    () => (detail?.processing_by_family ?? []).find((item) => item.family === activePipelineFamily) ?? null,
    [detail?.processing_by_family, activePipelineFamily]
  );
  const contextualRunHistory = useMemo(() => {
    if (!detail || !selectedPipeline) return [];
    const explicitHistory = (detail.run_history ?? []).map((item) => ({
      id: String(item.run_id ?? "run"),
      status: String(item.status ?? "unknown"),
      timestamp: String(item.timestamp ?? ""),
      duration: Number(item.duration_ms ?? 0),
    }));
    if (explicitHistory.length) return explicitHistory;
    if ((selectedPipeline.pipeline_family ?? "3d") === "2d") {
      const history = selectedPipelineInstance?.execution_history ?? [];
      return history
        .filter((item) => ((item as { summary?: { take_id?: string } }).summary?.take_id ?? null) === detail.take_id)
        .map((item) => ({ id: String((item as { run_id?: string }).run_id ?? "run"), status: String((item as { status?: string }).status ?? "unknown"), timestamp: String((item as { timestamp?: string }).timestamp ?? ""), duration: Number(((item as { step_reports?: Array<{ duration_ms?: number }> }).step_reports ?? []).reduce((acc, cur) => acc + Number(cur.duration_ms ?? 0), 0)) }));
    }
    if (detail.result) {
      return [{ id: "3d_result", status: detail.result.status, timestamp: detail.result.processed_at, duration: Number(detail.result.timing_ms?.total ?? 0) }];
    }
    return [];
  }, [detail, selectedPipeline, selectedPipelineInstance?.execution_history]);
  const selectedRunTrace = useMemo(() => contextualRunHistory[0] ?? null, [contextualRunHistory]);
  const comparisonRunOptions = useMemo(
    () => contextualRunHistory.filter((item, index, list) => list.findIndex((candidate) => candidate.id === item.id) === index),
    [contextualRunHistory],
  );
  useEffect(() => {
    if (!comparisonRunOptions.length) {
      setComparisonLeftRunId("");
      setComparisonRightRunId("");
      return;
    }
    setComparisonLeftRunId((current) => current || comparisonRunOptions[0]?.id || "");
    setComparisonRightRunId((current) => current || comparisonRunOptions[1]?.id || comparisonRunOptions[0]?.id || "");
  }, [comparisonRunOptions]);
  useEffect(() => {
    if (!pipelineRecipes.length) {
      setComparisonLeftRecipeId("");
      setComparisonRightRecipeId("");
      return;
    }
    const preferredRecipeId = selectedRecipeId || pipelineRecipes[0]?.recipe_id || "";
    setComparisonLeftRecipeId((current) => current || preferredRecipeId);
    setComparisonRightRecipeId((current) => current || preferredRecipeId);
  }, [pipelineRecipes, selectedRecipeId]);
  const stageNavigator = useMemo(
    () => (selectedPipeline?.stages ?? []).map((stage) => {
      const canonical = canonicalStageId(stage.id, stage.display_name ?? null);
      const executionStage = detail?.result?.pipeline_execution?.stages?.find((item) => item.stage_id === canonical || item.stage_id === stage.id) ?? null;
      const status = String(executionStage?.status ?? (detail?.result ? "idle" : "pending"));
      const durationMs = executionStage?.duration_ms ?? stageTimingMs(detail?.result, canonical);
      return {
        id: stage.id,
        canonical,
        shortLabel: compactStageLabel(stage.id, prettyStageLabel(stage.id, stage.display_name)),
        label: prettyStageLabel(stage.id, stage.display_name),
        status,
        durationMs,
        selected: stage.id === selectedStageId,
        warningCount: Array.isArray(executionStage?.warnings) ? executionStage.warnings.length : 0,
      };
    }),
    [detail?.result, selectedPipeline?.stages, selectedStageId]
  );
  const selectedView = stageSemantic.views.find((item) => item.id === activeTab) ?? stageSemantic.views[0] ?? null;
  const tuneModeActive = studioMode === "tune";
  const debugPanelsVisible = studioMode === "debug";
  const currentStageSummary = stageInspectorSummary(workspaceDetail, selectedPipeline, canonicalSelectedStageId);
  const selectedArtifactLineage = useMemo(
    () => artifactLineage(allArtifacts, focusedArtifact).slice(0, 3),
    [allArtifacts, focusedArtifact]
  );
  const hasTuneChanges = Boolean(planeQaDirty || segmentationPreviewDirty || ellipsePreviewDirty);
  const stageHasEditableParams = useMemo(
    () => ["detect_belt_plane", "segmentation", "ellipse_fitting", "load_heightmap", "remove_belt_segment_objects", "fit_object_geometry", "compute_height_metrics"].includes(canonicalSelectedStageId),
    [canonicalSelectedStageId]
  );
  const rightPanelOverlay = shellLayoutMode === "narrow" || shellLayoutMode === "very_narrow";
  const leftRailDrawer = shellLayoutMode === "narrow" || shellLayoutMode === "very_narrow";
  const inspectorCanCollapse = rightPanelOverlay;
  // Desktop inspection remains visible; collapsing is only a narrow-overlay affordance.
  const inspectorOpen = inspectorCanCollapse ? inspectorExpanded : true;
  const leftRailCanCollapse = shellLayoutMode === "wide";
  const leftRailReallyCollapsed = leftRailCanCollapse && leftRailCollapsed;
  const leftRailVisible = !focusMode && (!leftRailDrawer || leftRailExpanded);
  const stageRailVisible = !focusMode;
  const inspectorVisible = !focusMode && (!rightPanelOverlay || inspectorOpen);
  const overlayControlsVisible = Boolean(focusedArtifact && (focusedArtifact.kind === "overlay" || overlayTargetResolution.overlays.length > 0));
  const workspaceViewSplit = useMemo(
    () => stageSubstagePlan.substages.length > 0
      ? resolveDetectWorkspaceViewGroups({
        activeSubstage,
        activeReferenceViewScope,
        stageRootUnit,
        activeProcessingUnitView,
        stageDisplayPlan,
      })
      : { primary: [], secondary: [] },
    [activeProcessingUnitView, activeReferenceViewScope, activeSubstage, stageDisplayPlan, stageRootUnit, stageSubstagePlan.substages.length],
  );
  const workspacePrimaryViews = workspaceViewSplit.primary;
  const workspaceSecondaryViews = workspaceViewSplit.secondary;
  const workspaceSecondaryViewIds = useMemo(
    () => new Set(workspaceSecondaryViews.map((view) => view.id)),
    [workspaceSecondaryViews],
  );
  const workspaceAllViews = useMemo(
    () => [...workspacePrimaryViews, ...workspaceSecondaryViews],
    [workspacePrimaryViews, workspaceSecondaryViews],
  );
  const panelWidthClass = !inspectorOpen
    ? "panel-collapsed"
    : controlPanelTab === "tune" || controlPanelTab === "debug" || controlPanelTab === "compare" || controlPanelTab === "graph"
      ? "panel-wide"
      : controlPanelTab === "actions"
        ? "panel-medium"
        : "panel-compact";
  const visibleStageTabs = useMemo<StageTabEntry[]>(
    () => {
      if (canonicalSelectedStageId === "detect_belt_plane" && workspacePrimaryViews.length > 0) {
        return workspacePrimaryViews.map((view) => ({
          id: view.id as StageViewTabId,
          label: view.label,
          hiddenReason: view.reason,
          relevance: view.relevance,
        }));
      }
      if (stageRootUnit) {
        if (activeSubstage) {
          return activeSubstage.views.map((entry) => ({
            id: entry.view.id as StageViewTabId,
            label: compactDetectViewLabel(entry.view.id, entry.view.label),
            hiddenReason: entry.reason,
            relevance: entry.category,
          }));
        }
        return stageRootUnit.views.map((view) => ({
            id: view.id as StageViewTabId,
            label: compactDetectViewLabel(view.id, view.label),
            hiddenReason: activeProcessingUnitView?.viewId === view.id && activeProcessingUnitView.fallbackUsed
              ? activeProcessingUnitView.fallbackReason
              : undefined,
        }));
      }
      return stageSemantic.views.map((view) => ({ id: view.id as StageViewTabId, label: view.label }));
    },
    [activeProcessingUnitView, activeSubstage, canonicalSelectedStageId, stageRootUnit, stageSemantic.views, workspacePrimaryViews]
  );
  const overflowStageTabs = useMemo<StageTabEntry[]>(
    () => {
      if (canonicalSelectedStageId === "detect_belt_plane" && workspaceAllViews.length > workspacePrimaryViews.length) {
        const visibleIds = new Set(workspacePrimaryViews.map((view) => view.id));
        return workspaceAllViews
          .filter((view) => !visibleIds.has(view.id))
          .map((view) => ({
            id: view.id as StageViewTabId,
            label: view.label,
            hiddenReason: view.reason,
            relevance: view.relevance,
          }));
      }
      if (stageRootUnit) {
        if (activeSubstage) {
          const scopedIds = new Set(activeSubstage.views.map((entry) => entry.view.id));
          return [
            ...stageDisplayPlan.items
              .filter((item) => !scopedIds.has(item.view.id))
              .map((item) => ({
                id: item.view.id as StageViewTabId,
                label: item.label,
                hiddenReason: item.reason,
                relevance: item.category,
              })),
            ...stageDisplayPlan.supportingViews
              .filter((entry) => !scopedIds.has(entry.view.id))
              .map((entry) => ({
                id: entry.view.id as StageViewTabId,
                label: compactDetectViewLabel(entry.view.id, entry.view.label),
                hiddenReason: entry.reason,
                relevance: entry.category,
              })),
          ];
        }
        const rootIds = new Set(stageRootUnit.views.map((view) => view.id));
        return [
          ...stageDisplayPlan.items
            .filter((item) => !rootIds.has(item.view.id))
            .map((item) => ({
              id: item.view.id as StageViewTabId,
              label: item.label,
              hiddenReason: item.reason,
              relevance: item.category,
            })),
          ...stageDisplayPlan.supportingViews
            .filter((entry) => !rootIds.has(entry.view.id))
            .map((entry) => ({
              id: entry.view.id as StageViewTabId,
              label: compactDetectViewLabel(entry.view.id, entry.view.label),
              hiddenReason: entry.reason,
              relevance: entry.category,
            })),
          ...stageDisplayPlan.debugViews.map((entry) => ({
            id: entry.view.id as StageViewTabId,
            label: compactDetectViewLabel(entry.view.id, entry.view.label),
            hiddenReason: entry.reason,
            relevance: entry.category,
          })),
        ];
      }
      const visibleIds = new Set(visibleStageTabs.map((tab) => tab.id));
      return stageSemantic.views
        .filter((view) => !visibleIds.has(view.id as StageViewTabId))
        .map((view) => ({
          id: view.id as StageViewTabId,
          label: compactDetectViewLabel(view.id, view.label),
        }));
    },
    [activeSubstage, canonicalSelectedStageId, stageDisplayPlan.debugViews, stageDisplayPlan.items, stageDisplayPlan.supportingViews, stageRootUnit, stageSemantic.views, visibleStageTabs, workspaceAllViews, workspacePrimaryViews]
  );
  useEffect(() => {
    if (visibleStageTabs.some((tab) => tab.id === activeTab) || overflowStageTabs.some((tab) => tab.id === activeTab)) return;
    const fallback = visibleStageTabs[0]?.id ?? stageTabs[0]?.id ?? "overview";
    setActiveTab(fallback);
  }, [activeTab, overflowStageTabs, stageTabs, visibleStageTabs]);
  useEffect(() => {
    if (stageSubstagePlan.substages.length === 0) {
      setReferenceViewScope(null);
      setSelectedSubstageId(null);
      return;
    }
    if (!selectedSubstageId) return;
    if (stageSubstagePlan.substages.some((substage) => substage.substageId === selectedSubstageId)) return;
    setSelectedSubstageId(null);
    if (referenceViewScope === "substage") setReferenceViewScope("strategy_path");
  }, [referenceViewScope, selectedSubstageId, stageSubstagePlan.substages]);
  useEffect(() => {
    if (!selectedContractUnitId) return;
    if (canonicalSelectedStageId === "detect_belt_plane") {
      setSelectedContractUnitId(null);
      return;
    }
    if (stageProcessingUnits.some((unit) => unit.id === selectedContractUnitId)) return;
    setSelectedContractUnitId(null);
  }, [canonicalSelectedStageId, selectedContractUnitId, stageProcessingUnits]);
  useEffect(() => {
    if (!activeSubstage) return;
    if (activeSubstage.viewIds.includes(activeTab)) return;
    const fallbackViewId = activeSubstage.recommendedViewId ?? activeSubstage.viewIds[0];
    if (fallbackViewId) setActiveTab(fallbackViewId as StageViewTabId);
  }, [activeSubstage, activeTab]);
  useEffect(() => {
    if (activeSubstage || !stageRootUnit) return;
    const rootViewIds = stageRootUnit.views.map((view) => view.id);
    if (rootViewIds.includes(activeTab)) return;
    const fallbackViewId = stageRootUnit.default_view ?? stageRootUnit.views[0]?.id;
    if (fallbackViewId) setActiveTab(fallbackViewId as StageViewTabId);
  }, [activeSubstage, activeTab, stageRootUnit]);
  useEffect(() => {
    if (studioMode === "tune") {
      setControlPanelTab("tune");
      setInspectorExpanded(true);
      return;
    }
    if (studioMode === "debug") {
      setControlPanelTab("debug");
      setInspectorExpanded(true);
      return;
    }
    if (studioMode === "compare") {
      setControlPanelTab("compare");
      setInspectorExpanded(true);
      return;
    }
    setControlPanelTab("inspect");
  }, [studioMode]);
  const compactInspectorItems = [
    { label: "Artifact", value: activeDisplayItem?.primaryArtifactId ?? focusedArtifact?.artifact_id ?? "None" },
    { label: "Object", value: focusedObject ? `#${focusedObject.object_id}` : "None" },
    { label: "Warnings", value: String(currentStageSummary.warnings.length) },
    { label: "Strategy", value: detectStageSummary?.strategyLabel ?? stageSemantic.category },
    { label: "Status", value: currentStageSummary.status },
  ];
  const activeViewHelp = useMemo(
    () => resolveStudioHelpEntry(canonicalSelectedStageId === "detect_belt_plane" ? detectReferenceViewHelpKey(activeTab) : null),
    [activeTab, canonicalSelectedStageId],
  );
  const activeSubstageHelp = useMemo(
    () => canonicalSelectedStageId === "detect_belt_plane" && activeSubstage
      ? resolveStudioHelpEntry(detectReferenceSubstageHelpKey(activeSubstage.substageId))
      : null,
    [activeSubstage, canonicalSelectedStageId],
  );
  const showSelectedSurfaceRefinement = useMemo(
    () => canonicalSelectedStageId === "detect_belt_plane" && shouldShowSelectedSurfaceRefinement(activeTab, focusedArtifact),
    [activeTab, canonicalSelectedStageId, focusedArtifact],
  );
  const detectSourceContextBanner = useMemo(
    () => buildSourceContextBanner(studioDeepLink, studioDeepLinkObjectResolved),
    [studioDeepLink, studioDeepLinkObjectResolved],
  );
  const stripeHeaderSummary = useMemo(() => {
    if (activeSubstage?.substageId !== "stripes" || !detectStageSummary) return null;
    const bits = [
      `Stripe filter: ${detectStageSummary.stripeFilterEnabled ? "enabled" : "disabled"}`,
      detectStageSummary.selectedSupportRatio != null ? `final support ${ratioToPercentLabel(detectStageSummary.selectedSupportRatio) ?? "-" } ROI` : null,
      detectStageSummary.stripeRemovedRatio != null ? `removed ${ratioToPercentLabel(detectStageSummary.stripeRemovedRatio) ?? "-" } ROI` : null,
      `fallback ${detectReferenceRunState?.fallbackUsed ? "yes" : "no"}`,
    ].filter(Boolean);
    return bits.join(" · ");
  }, [activeSubstage?.substageId, detectReferenceRunState?.fallbackUsed, detectStageSummary]);

  useEffect(() => {
    if (!pipelines.length) return;
    if (studioDeepLink.pipeline_id) return;
    const takeKey = selectedTakeId || "__global__";
    const auto = chooseBestPipelineForTake(pipelines, detail?.modalities, lastUsedPipelineByTake[takeKey] ?? null);
    if (auto && auto.id !== selectedPipelineId) {
      setSelectedPipelineId(auto.id);
    }
  }, [selectedTakeId, detail?.modalities, pipelines, lastUsedPipelineByTake, selectedPipelineId, studioDeepLink.pipeline_id]);

  useEffect(() => {
    if (studioPipelineAppliedRef.current || !studioDeepLink.pipeline_id || !pipelines.length) return;
    if (pipelines.some((pipeline) => pipeline.id === studioDeepLink.pipeline_id)) {
      setSelectedPipelineId(studioDeepLink.pipeline_id);
    }
    studioPipelineAppliedRef.current = true;
  }, [pipelines, studioDeepLink.pipeline_id]);

  useEffect(() => {
    if (studioStageAppliedRef.current || !studioDeepLink.stage) return;
    if (studioDeepLink.pipeline_id && selectedPipeline?.id !== studioDeepLink.pipeline_id) return;
    const nextStageId = resolveStudioStageId(selectedPipeline, studioDeepLink.stage);
    if (!selectedPipeline?.stages.length) return;
    if (nextStageId) {
      setSelectedStageId(nextStageId);
      setActiveTab("overview");
    }
    studioStageAppliedRef.current = true;
  }, [selectedPipeline, studioDeepLink.pipeline_id, studioDeepLink.stage]);

  useEffect(() => {
    if (studioObjectAppliedRef.current) return;
    const nextObjectId = requestedStudioObjectId ?? requestedStudioPhysicalObjectMatchId;
    if (nextObjectId == null) {
      if (studioDeepLink.object_id || studioDeepLink.physical_object_id) studioObjectAppliedRef.current = true;
      return;
    }
    if (!allObjects.some((object) => object.object_id === nextObjectId)) return;
    setSelectedObjectId(nextObjectId);
    studioObjectAppliedRef.current = true;
  }, [
    allObjects,
    requestedStudioObjectId,
    requestedStudioPhysicalObjectMatchId,
    studioDeepLink.object_id,
    studioDeepLink.physical_object_id,
  ]);

  useEffect(() => {
    if (!studioDeepLink.take_id) return;
    setDeepLinkMessage(buildStudioDeepLinkMessage(studioDeepLink, studioDeepLinkObjectResolved));
  }, [studioDeepLink, studioDeepLinkObjectResolved]);

  useEffect(() => {
    setSourceContextDismissed(false);
  }, [studioDeepLink.feature, studioDeepLink.object_id, studioDeepLink.physical_object_id, studioDeepLink.stage, selectedTakeId]);

  useEffect(() => {
    if (graphWorkspaceEnabled(studioDeepLink.graph_workspace)) {
      setGraphWorkspaceExpanded(true);
      setControlPanelTab("graph");
    }
    if (studioDeepLink.recipe_id && pipelineRecipes.some((recipe) => recipe.recipe_id === studioDeepLink.recipe_id)) {
      setSelectedRecipeId(studioDeepLink.recipe_id);
    }
    if (studioDeepLink.unit_id && pipelineProcessingUnits.some((unit) => unit.id === studioDeepLink.unit_id)) {
      focusProcessingUnitFromGraph(studioDeepLink.unit_id);
    }
    if (studioDeepLink.tab === "tune") {
      setControlPanelTab("tune");
      setStudioMode(controlPanelTabToMode("tune"));
      setInspectorExpanded(true);
    }
  }, [
    focusProcessingUnitFromGraph,
    pipelineProcessingUnits,
    pipelineRecipes,
    studioDeepLink.graph_workspace,
    studioDeepLink.recipe_id,
    studioDeepLink.tab,
    studioDeepLink.unit_id,
  ]);

  useEffect(() => {
    if (studioRunDeepLinkAppliedRef.current) return;
    if (!selectedTakeId) return;
    if (!studioDeepLink.run_id) {
      studioRunDeepLinkAppliedRef.current = true;
      return;
    }
    studioRunDeepLinkAppliedRef.current = true;
    setActiveRunId(studioDeepLink.run_id === "latest" ? null : studioDeepLink.run_id);
  }, [selectedTakeId, studioDeepLink.run_id]);

  useEffect(() => {
    if (studioGraphDeepLinkAppliedRef.current) return;
    if (!studioDeepLink.comparison_id) {
      studioGraphDeepLinkAppliedRef.current = true;
      return;
    }
    if (selectedPipeline?.id !== "mining_steel_ball_classification_25d") return;
    studioGraphDeepLinkAppliedRef.current = true;
    let cancelled = false;
    setComparisonBusy(true);
    void api.pipelineComparison(selectedPipeline.id, studioDeepLink.comparison_id).then((response) => {
      if (cancelled) return;
      setComparisonResult(response);
      setControlPanelTab("compare");
    }).catch((error: unknown) => {
      if (cancelled) return;
      setComparisonError(error instanceof Error ? error.message : "Could not reopen comparison from link.");
    }).finally(() => {
      if (!cancelled) setComparisonBusy(false);
    });
    return () => {
      cancelled = true;
    };
  }, [selectedPipeline?.id, studioDeepLink.comparison_id]);

  useEffect(() => {
    if (typeof window === "undefined" || window.location.pathname !== "/studio") return;
    const current = parseStudioDeepLink(window.location.search);
    const nextHref = buildStudioDeepLink({
      ...current,
      take_id: selectedTakeId ?? current.take_id,
      pipeline_id: selectedPipelineId ?? current.pipeline_id,
      run_id: activeRunId ?? current.run_id,
      graph_workspace: graphWorkspaceExpanded ? "1" : null,
      unit_id: selectedGraphNodeId ?? current.unit_id,
      comparison_id: comparisonResult?.comparison_id ?? current.comparison_id,
      recipe_id: selectedRecipeId || current.recipe_id,
    });
    const nextSearch = nextHref.includes("?") ? nextHref.slice(nextHref.indexOf("?")) : "";
    if (nextSearch !== window.location.search) {
      window.history.replaceState(null, "", `/studio${nextSearch}`);
    }
  }, [
    activeRunId,
    comparisonResult?.comparison_id,
    graphWorkspaceExpanded,
    selectedGraphNodeId,
    selectedPipelineId,
    selectedRecipeId,
    selectedTakeId,
  ]);

  function selectStage(stageId: string) {
    setSelectedStageId(stageId);
    setSelectedArtifactId(null);
    setRoiEditModeActive(false);
    setRoiEditStatus("idle");
  }

  function clearStudioSourceContext() {
    setSourceContextDismissed(true);
    if (typeof window === "undefined" || window.location.pathname !== "/studio") return;
    const current = parseStudioDeepLink(window.location.search);
    const nextHref = buildStudioDeepLink({
      ...current,
      feature: null,
      object_id: null,
      physical_object_id: null,
      stage: current.stage,
    });
    const nextSearch = nextHref.includes("?") ? nextHref.slice(nextHref.indexOf("?")) : "";
    window.history.replaceState(null, "", `/studio${nextSearch}`);
  }

  function detectStageIdFromPipeline(): string | null {
    const stages = selectedPipeline?.stages ?? [];
    const found = stages.find((stage) => canonicalStageId(stage.id, stage.display_name) === "detect_belt_plane");
    return found?.id ?? null;
  }

  function startVisualRoiEdit() {
    const detectStageId = detectStageIdFromPipeline();
    if (detectStageId && detectStageId !== selectedStageId) {
      setSelectedStageId(detectStageId);
      setSelectedArtifactId(null);
    }
    setActiveTab("raw_heightmap");
    setRoiEditModeActive(true);
    setRoiEditStatus("active");
    setPipelineActionMessage("Draw reference-surface ROI");
  }

  function cancelVisualRoiEdit() {
    setRoiEditModeActive(false);
    setRoiEditStatus("idle");
    setPipelineActionMessage("ROI edit cancelled.");
  }

  function selectPipeline(pipelineId: string) {
    setSelectedPipelineId(pipelineId);
    const takeKey = selectedTakeId || "__global__";
    setLastUsedPipelineByTake((current) => ({ ...current, [takeKey]: pipelineId }));
  }

  async function createProcessPipeline() {
    await api.createPipelineInstance({
      name: `POC ${selectedTemplateId}`,
      template_id: selectedTemplateId,
      input_type: selectedInputType,
    });
    await load();
  }

  async function toggleStep(stepId: string, enabled: boolean) {
    if (!selectedPipelineInstance) return;
    const configured_steps = selectedPipelineInstance.configured_steps.map((step) =>
      step.step_id === stepId ? { ...step, enabled } : step
    );
    const next = await api.updatePipelineInstance(selectedPipelineInstance.id, { configured_steps });
    setPipelineInstances((items) => items.map((item) => (item.id === next.id ? next : item)));
  }

  async function updateStepParam(stepId: string, key: string, value: string | number | boolean) {
    if (!selectedPipelineInstance) return;
    const configured_steps = selectedPipelineInstance.configured_steps.map((step) =>
      step.step_id === stepId ? { ...step, params: { ...step.params, [key]: value } } : step
    );
    const next = await api.updatePipelineInstance(selectedPipelineInstance.id, { configured_steps });
    setPipelineInstances((items) => items.map((item) => (item.id === next.id ? next : item)));
  }

  async function runInstance() {
    if (!selectedPipelineInstance || !detail) return;
    const rgb = detail.assets?.rgb?.rgb;
    const reflectance = detail.assets?.reflectance?.reflectance;
    const source = rgb || reflectance;
    const imagePath = manualImagePath.trim() || (source ? `incoming/${detail.take_id}/${source}` : "");
    if (!imagePath) return;
    await api.executePipelineInstance(selectedPipelineInstance.id, { image_path: imagePath });
    await load();
  }

  function build25dStageParams(sourceParams?: PlaneQaTuningParams | null): Record<string, unknown> | null {
    const paramsSource = sourceParams ?? planeQaParams;
    if (selectedPipeline?.id !== "mining_steel_ball_classification_25d" || !paramsSource) return null;
    return {
      detect_belt_plane: {
        background_detection_strategy: String(paramsSource.background_detection_strategy ?? "low_gradient_surface"),
        gradient_smoothing_kernel: Number(paramsSource.gradient_smoothing_kernel ?? 3),
        gradient_threshold_mode: String(paramsSource.gradient_threshold_mode ?? "percentile"),
        gradient_threshold_value: Number(paramsSource.gradient_threshold_value ?? 2.0),
        gradient_threshold_percentile: Number(paramsSource.gradient_threshold_percentile ?? 70),
        low_gradient_morphology_enabled: Boolean(paramsSource.low_gradient_morphology_enabled ?? true),
        low_gradient_open_kernel: Number(paramsSource.low_gradient_open_kernel ?? 3),
        low_gradient_close_kernel: Number(paramsSource.low_gradient_close_kernel ?? 5),
        low_gradient_fill_holes: Boolean(paramsSource.low_gradient_fill_holes ?? true),
        low_gradient_min_component_area: Number(paramsSource.low_gradient_min_component_area ?? 1500),
        reference_surface_selection_mode: String(paramsSource.reference_surface_selection_mode ?? "largest_constant_z"),
        reference_surface_model: String(paramsSource.reference_surface_model ?? "auto"),
        reference_suppression_mask_policy: String(paramsSource.reference_suppression_mask_policy ?? "auto"),
        reference_surface_max_plane_residual_p95_mm: Number(paramsSource.reference_surface_max_plane_residual_p95_mm ?? 3.0),
        low_gradient_plateau_use_hessian_filter: Boolean(paramsSource.low_gradient_plateau_use_hessian_filter ?? true),
        low_gradient_plateau_hessian_percentile: Number(paramsSource.low_gradient_plateau_hessian_percentile ?? 70),
        low_gradient_plateau_hist_bins: Number(paramsSource.low_gradient_plateau_hist_bins ?? 96),
        low_gradient_plateau_min_fraction: Number(paramsSource.low_gradient_plateau_min_fraction ?? 0.05),
        low_gradient_plateau_min_pixels: Number(paramsSource.low_gradient_plateau_min_pixels ?? 500),
        low_gradient_plateau_smoothing_sigma_bins: Number(paramsSource.low_gradient_plateau_smoothing_sigma_bins ?? 2.5),
        low_gradient_plateau_peak_drop_ratio: Number(paramsSource.low_gradient_plateau_peak_drop_ratio ?? 0.40),
        low_gradient_plateau_select_min_area_fraction: Number(paramsSource.low_gradient_plateau_select_min_area_fraction ?? 0.20),
        low_gradient_plateau_selection_mode: String(paramsSource.low_gradient_plateau_selection_mode ?? "lowest_dominant"),
        low_gradient_plateau_robust_band_mad_k: Number(paramsSource.low_gradient_plateau_robust_band_mad_k ?? 3.0),
        low_gradient_plateau_detection_min_count_floor: Number(paramsSource.low_gradient_plateau_detection_min_count_floor ?? 25),
        low_gradient_plateau_detection_min_count_fraction: Number(paramsSource.low_gradient_plateau_detection_min_count_fraction ?? 0.25),
        blob_cluster_min_component_area: Number(paramsSource.blob_cluster_min_component_area ?? 250),
        blob_component_mode: String(paramsSource.blob_component_mode ?? "xy_only"),
        blob_connectivity: Number(paramsSource.blob_connectivity ?? 8),
        blob_neighbor_z_tolerance_mm: Number(paramsSource.blob_neighbor_z_tolerance_mm ?? 3.0),
        blob_component_z_tolerance_mm: Number(paramsSource.blob_component_z_tolerance_mm ?? 8.0),
        blob_component_tolerance_mode: String(paramsSource.blob_component_tolerance_mode ?? "fixed"),
        blob_component_mad_k: Number(paramsSource.blob_component_mad_k ?? 3.0),
        blob_component_mad_floor_mm: Number(paramsSource.blob_component_mad_floor_mm ?? 1.0),
        blob_component_allow_gradual_slope: Boolean(paramsSource.blob_component_allow_gradual_slope ?? true),
        blob_component_max_local_slope_mm_per_px: Number(paramsSource.blob_component_max_local_slope_mm_per_px ?? 3.0),
        blob_component_min_area_px: Number(paramsSource.blob_component_min_area_px ?? 300),
        blob_component_use_smoothed_z: Boolean(paramsSource.blob_component_use_smoothed_z ?? true),
        blob_component_z_smoothing_kernel: Number(paramsSource.blob_component_z_smoothing_kernel ?? 3),
        blob_split_by_height_enabled: Boolean(paramsSource.blob_split_by_height_enabled ?? true),
        blob_split_method: String(paramsSource.blob_split_method ?? "height_borders"),
        blob_cluster_height_gap_mm: Number(paramsSource.blob_cluster_height_gap_mm ?? 8.0),
        blob_cluster_height_gap_mode: String(paramsSource.blob_cluster_height_gap_mode ?? "adaptive"),
        blob_cluster_min_area_fraction: Number(paramsSource.blob_cluster_min_area_fraction ?? 0.10),
        blob_cluster_min_total_pixels: Number(paramsSource.blob_cluster_min_total_pixels ?? 1200),
        blob_cluster_max_z_mad_mm: Number(paramsSource.blob_cluster_max_z_mad_mm ?? 12.0),
        blob_cluster_refine_by_mad: Boolean(paramsSource.blob_cluster_refine_by_mad ?? true),
        blob_cluster_refine_mad_k: Number(paramsSource.blob_cluster_refine_mad_k ?? 2.5),
        blob_cluster_refine_floor_mm: Number(paramsSource.blob_cluster_refine_floor_mm ?? 1.0),
        blob_cluster_refine_keep_border_support: Boolean(paramsSource.blob_cluster_refine_keep_border_support ?? true),
        blob_cluster_fallback_strategy: String(paramsSource.blob_cluster_fallback_strategy ?? "low_gradient_depth_plateaus"),
        belt_stripe_filter_enabled: Boolean(paramsSource.belt_stripe_filter_enabled ?? true),
        belt_stripe_filter_scope: String(paramsSource.belt_stripe_filter_scope ?? "global"),
        belt_stripe_filter_window_mm: Number(paramsSource.belt_stripe_filter_window_mm ?? 30.0),
        belt_stripe_filter_direction: String(paramsSource.belt_stripe_filter_direction ?? "auto"),
        belt_stripe_filter_threshold_mode: String(paramsSource.belt_stripe_filter_threshold_mode ?? "otsu"),
        belt_stripe_filter_min_altitude_mm: Number(paramsSource.belt_stripe_filter_min_altitude_mm ?? 10.0),
        belt_stripe_filter_k_mad: Number(paramsSource.belt_stripe_filter_k_mad ?? 3.0),
        belt_stripe_filter_fixed_threshold_mm: Number(paramsSource.belt_stripe_filter_fixed_threshold_mm ?? 10.0),
        belt_stripe_filter_min_stripe_fraction: Number(paramsSource.belt_stripe_filter_min_stripe_fraction ?? 0.02),
        belt_stripe_filter_z_floor_enabled: Boolean(paramsSource.belt_stripe_filter_z_floor_enabled ?? true),
        belt_stripe_filter_z_floor_use_upper_bound: Boolean(paramsSource.belt_stripe_filter_z_floor_use_upper_bound ?? true),
        belt_stripe_filter_z_floor_margin_mm: Number(paramsSource.belt_stripe_filter_z_floor_margin_mm ?? 20.0),
        belt_stripe_filter_max_stripe_height_mm: Number(paramsSource.belt_stripe_filter_max_stripe_height_mm ?? 500.0),
        belt_stripe_filter_above_belt_close_mm: Number(paramsSource.belt_stripe_filter_above_belt_close_mm ?? 30.0),
        belt_stripe_filter_object_kernel_mm: Number(paramsSource.belt_stripe_filter_object_kernel_mm ?? 100.0),
        belt_stripe_filter_object_kernel_shape: String(paramsSource.belt_stripe_filter_object_kernel_shape ?? "ellipse"),
        belt_stripe_filter_altitude_hist_bins: Number(paramsSource.belt_stripe_filter_altitude_hist_bins ?? 64),
        belt_stripe_filter_auto_bimodality_margin: Number(paramsSource.belt_stripe_filter_auto_bimodality_margin ?? 1.10),
        belt_stripe_filter_z_floor_upper_percentile: Number(paramsSource.belt_stripe_filter_z_floor_upper_percentile ?? 99.0),
        belt_stripe_filter_z_floor_fallback_lower_percentile: Number(paramsSource.belt_stripe_filter_z_floor_fallback_lower_percentile ?? 10.0),
        belt_stripe_filter_z_floor_fallback_upper_percentile: Number(paramsSource.belt_stripe_filter_z_floor_fallback_upper_percentile ?? 25.0),
        belt_stripe_filter_warn_removed_fraction: Number(paramsSource.belt_stripe_filter_warn_removed_fraction ?? 0.40),
        low_gradient_surface_support_z_mad_multiplier: Number(paramsSource.low_gradient_surface_support_z_mad_multiplier ?? 2.5),
        low_gradient_surface_support_z_floor_mm: Number(paramsSource.low_gradient_surface_support_z_floor_mm ?? 1.0),
        low_gradient_surface_support_z_mad_floor_mm: Number(paramsSource.low_gradient_surface_support_z_mad_floor_mm ?? 0.25),
        low_gradient_surface_ridge_percentile: Number(paramsSource.low_gradient_surface_ridge_percentile ?? 90.0),
        reference_surface_height_gate_enabled: Boolean(paramsSource.reference_surface_height_gate_enabled ?? true),
        reference_surface_height_gate_margin_mm: Number(paramsSource.reference_surface_height_gate_margin_mm ?? 8.0),
        reference_surface_height_gate_min_coverage_ratio: Number(paramsSource.reference_surface_height_gate_min_coverage_ratio ?? 0.10),
        reference_surface_height_gate_max_coverage_ratio: Number(paramsSource.reference_surface_height_gate_max_coverage_ratio ?? 0.95),
        reference_surface_height_gate_gap_floor_mm: Number(paramsSource.reference_surface_height_gate_gap_floor_mm ?? 1.0),
        reference_surface_height_gate_gap_ratio: Number(paramsSource.reference_surface_height_gate_gap_ratio ?? 8.0),
        plot_depth_plot_max_render_samples: Number(paramsSource.plot_depth_plot_max_render_samples ?? 60000),
        plot_y_robust_percentile: Number(paramsSource.plot_y_robust_percentile ?? 98.0),
        plane_fit_roi: String(paramsSource.reference_surface_region_mode ?? "none") === "none"
          ? {
            enabled: false,
            type: "rectangle",
            x: 0,
            y: 0,
            width: 0,
            height: 0,
          }
          : (
            String(paramsSource.reference_surface_region_mode ?? "none") === "polygon"
              ? {
                enabled: true,
                type: "polygon",
                points: ((paramsSource.plane_fit_roi_points as Array<[number, number]> | undefined) ?? []).map(([x, y]) => [Math.round(Number(x)), Math.round(Number(y))]),
              }
              : String(paramsSource.reference_surface_region_mode ?? "none") === "full_height_x_band"
                ? {
                  enabled: true,
                  type: "full_height_x_band",
                  x: Number(paramsSource.plane_fit_roi_x ?? 0),
                  width: Number(paramsSource.plane_fit_roi_width ?? 0),
                }
              : {
                enabled: true,
                type: "rectangle",
                x: Number(paramsSource.plane_fit_roi_x ?? 0),
                y: Number(paramsSource.plane_fit_roi_y ?? 0),
                width: Number(paramsSource.plane_fit_roi_width ?? 0),
                height: Number(paramsSource.plane_fit_roi_height ?? 0),
              }
          ),
      },
      remove_belt_segment_objects: {
        min_height_mm: Number(paramsSource.object_min_height_mm ?? 8.0),
        max_height_mm: paramsSource.object_max_height_mm == null || paramsSource.object_max_height_mm === "" ? null : Number(paramsSource.object_max_height_mm),
        reference_tolerance_mm: Number(paramsSource.reference_tolerance_mm ?? 0.0),
        suppress_plane_mask_in_segmentation: Boolean(paramsSource.suppress_plane_mask_in_segmentation ?? true),
        morphology_kernel: Number(paramsSource.morphology_kernel ?? 5),
        min_component_area: Number(paramsSource.min_component_area ?? 120),
        fill_holes: Boolean(paramsSource.fill_holes ?? true),
        smoothing_kernel: Number(paramsSource.smoothing_kernel ?? 3),
      },
      known_object_25d: {
        enabled: Boolean(paramsSource.known_object_enabled ?? false),
        apply_correction: Boolean(paramsSource.known_object_apply_correction ?? true),
        object_label: String(paramsSource.known_object_label ?? "cube"),
        target_selection: String(paramsSource.known_object_target_selection ?? "largest_component"),
        manual_component_id: Number(paramsSource.known_object_manual_component_id ?? 1),
        known_width_mm: Number(paramsSource.known_width_mm ?? 40),
        known_depth_mm: Number(paramsSource.known_depth_mm ?? 40),
        known_height_mm: Number(paramsSource.known_height_mm ?? 25),
        tolerance_percent: Number(paramsSource.known_tolerance_percent ?? 5),
        apply_persisted_correction: Boolean(paramsSource.known_object_apply_persisted_correction ?? true),
        persisted_scale_correction_x: paramsSource.known_object_persisted_scale_correction_x == null ? null : Number(paramsSource.known_object_persisted_scale_correction_x),
        persisted_scale_correction_y: paramsSource.known_object_persisted_scale_correction_y == null ? null : Number(paramsSource.known_object_persisted_scale_correction_y),
        persisted_scale_correction_z: paramsSource.known_object_persisted_scale_correction_z == null ? null : Number(paramsSource.known_object_persisted_scale_correction_z),
      },
    };
  }

  function buildComparisonSource(sourceType: CompareSourceType, side: "left" | "right", options?: { requireTakeId?: boolean }) {
    const requireTakeId = options?.requireTakeId ?? true;
    const recipeId = side === "left" ? comparisonLeftRecipeId : comparisonRightRecipeId;
    const runId = side === "left" ? comparisonLeftRunId : comparisonRightRunId;
    if (sourceType === "run") {
      if (!runId) return null;
      if (requireTakeId && !detail?.take_id) return null;
      return {
        type: "run" as const,
        label: `Run ${runId}`,
        ...(requireTakeId ? { take_id: detail!.take_id } : {}),
        run_id: runId,
      };
    }
    if (sourceType === "recipe") {
      const recipe = pipelineRecipes.find((item) => item.recipe_id === recipeId);
      if (!recipe) return null;
      return {
        type: "recipe" as const,
        label: recipe.name,
        recipe_id: recipe.recipe_id,
        version: recipe.version,
      };
    }
    const stageParams = build25dStageParams();
    if (!stageParams) return null;
    return {
      type: "current" as const,
      label: "Current edits",
      stage_params: stageParams,
    };
  }

  async function runComparison() {
    if (!selectedPipeline || selectedPipeline.id !== "mining_steel_ball_classification_25d") return;
    const left = buildComparisonSource(comparisonLeftType, "left");
    const right = buildComparisonSource(comparisonRightType, "right");
    if (!left || !right) {
      setComparisonError("Select valid comparison sources first.");
      return;
    }
    setComparisonBusy(true);
    setComparisonError(null);
    try {
      const response = await api.comparePipeline(selectedPipeline.id, { left, right });
      setComparisonResult(response);
      const history = await api.pipelineComparisons(selectedPipeline.id, { take_id: detail?.take_id ?? null, limit: 12 });
      setComparisonHistory(history.comparisons ?? []);
    } catch (error) {
      setComparisonResult(null);
      setComparisonError(error instanceof Error ? error.message : "Comparison failed.");
    } finally {
      setComparisonBusy(false);
    }
  }

  async function runBatchComparison() {
    if (!selectedPipeline || selectedPipeline.id !== "mining_steel_ball_classification_25d") return;
    const takeIds = Array.from(selectedTakeIds);
    if (!takeIds.length) {
      setBatchComparisonError("Select at least one take first.");
      return;
    }
    // Left stays whatever the Compare tab has configured (a fixed reference:
    // a recipe, current edits, or one pinned run). Right resolves per take
    // when it's a "run" source, so the same right-side run label is looked up
    // independently in each selected take instead of being pinned to the
    // currently open one.
    const left = buildComparisonSource(comparisonLeftType, "left");
    const right = buildComparisonSource(comparisonRightType, "right", { requireTakeId: false });
    if (!left || !right) {
      setBatchComparisonError("Select valid comparison sources first in the Compare tab.");
      return;
    }
    setBatchComparisonBusy(true);
    setBatchComparisonError(null);
    try {
      const response = await api.compareBatchPipeline(selectedPipeline.id, { take_ids: takeIds, left, right });
      setBatchComparisonResult(response);
    } catch (error) {
      setBatchComparisonResult(null);
      setBatchComparisonError(error instanceof Error ? error.message : "Batch comparison failed.");
    } finally {
      setBatchComparisonBusy(false);
    }
  }

  async function reopenComparison(comparisonId: string) {
    if (!selectedPipeline || selectedPipeline.id !== "mining_steel_ball_classification_25d") return;
    setComparisonBusy(true);
    setComparisonError(null);
    try {
      const response = await api.pipelineComparison(selectedPipeline.id, comparisonId);
      setComparisonResult(response);
    } catch (error) {
      setComparisonError(error instanceof Error ? error.message : "Could not reopen comparison.");
    } finally {
      setComparisonBusy(false);
    }
  }

  function applyReferenceMethodPreset(presetId: string, rerun: boolean) {
    if (!planeQaParams) return;
    const preset = detectReferencePresetById(presetId);
    if (!preset) return;
    const nextParams = { ...planeQaParams, ...preset.values };
    setPlaneQaParams(nextParams);
    setPlaneQaDirty(!planeQaParamsEqual(nextParams, planeQaPersistedParams));
    setPipelineActionMessage(`Preset ready: ${preset.label}`);
    if (rerun) {
      void applyPlaneQaParams(true, nextParams);
    }
  }

  function resetReferenceMethodPresetOverrides() {
    if (!planeQaParams) return;
    const preset = detectReferencePresetForValues(planeQaPersistedParams);
    if (!preset) return;
    const nextParams = { ...planeQaParams, ...preset.values };
    setPlaneQaParams(nextParams);
    setPlaneQaDirty(!planeQaParamsEqual(nextParams, planeQaPersistedParams));
    setPipelineActionMessage(`Reset preset overrides: ${preset.label}`);
  }

  async function copyReferenceMethodPresetJson() {
    const stageParams = build25dStageParams();
    if (!stageParams) return;
    const payload = JSON.stringify({ stage_params: stageParams }, null, 2);
    try {
      await navigator.clipboard.writeText(payload);
      setPipelineActionMessage("Copied preset JSON.");
    } catch {
      setPipelineActionMessage("Could not copy preset JSON.");
    }
  }

  function resetRecipeFieldToPersisted(stageId: string, key: string) {
    const persistedKey = contractParamFlatKey(stageId, key);
    setPlaneQaParams((current) => {
      if (!current || !planeQaPersistedParams) return current;
      const next = { ...current, [persistedKey]: planeQaPersistedParams[persistedKey] };
      setPlaneQaDirty(!planeQaParamsEqual(next, planeQaPersistedParams));
      return next;
    });
  }

  function resetRecipeUnitToPersisted(unit: typeof activeProcessingUnit) {
    if (!unit || !planeQaPersistedParams) return;
    const keys = new Set<string>();
    for (const param of unit.parameters ?? []) keys.add(param.id);
    for (const param of stageRootUnit?.parameters ?? []) keys.add(param.id);
    setPlaneQaParams((current) => {
      if (!current) return current;
      const next = { ...current };
      for (const key of keys) {
        const flatKey = contractParamFlatKey(unit.stage_id, key);
        next[flatKey] = planeQaPersistedParams[flatKey];
      }
      setPlaneQaDirty(!planeQaParamsEqual(next, planeQaPersistedParams));
      return next;
    });
  }

  function resetUnitToLastRunValues(unit: typeof activeProcessingUnit) {
    if (!unit || !activeLastRunStageParams) return;
    const keys = new Set<string>();
    for (const param of unit.parameters ?? []) keys.add(param.id);
    for (const param of stageRootUnit?.parameters ?? []) keys.add(param.id);
    setPlaneQaParams((current) => {
      if (!current) return current;
      const next = { ...current };
      for (const key of keys) {
        const flatKey = contractParamFlatKey(unit.stage_id, key);
        next[flatKey] = activeLastRunStageParams[key];
      }
      setPlaneQaDirty(!planeQaParamsEqual(next, planeQaPersistedParams));
      return next;
    });
  }

  function resetUnitToContractDefaults(unit: typeof activeProcessingUnit) {
    if (!unit) return;
    const parameterById = new Map<string, { default?: unknown }>();
    for (const param of stageRootUnit?.parameters ?? []) parameterById.set(param.id, param);
    for (const param of unit.parameters ?? []) parameterById.set(param.id, param);
    setPlaneQaParams((current) => {
      if (!current) return current;
      const next = { ...current };
      for (const [key, param] of parameterById.entries()) {
        const flatKey = contractParamFlatKey(unit.stage_id, key);
        next[flatKey] = param.default;
      }
      setPlaneQaDirty(!planeQaParamsEqual(next, planeQaPersistedParams));
      return next;
    });
  }

  async function planPartialRerunFromSelectedUnit() {
    if (selectedPipeline?.id !== "mining_steel_ball_classification_25d" || !activeProcessingUnit) return;
    setPartialRerunPlanning(true);
    try {
      const snapshot = {
        recipe_id: selectedRecipe?.recipe_id ?? null,
        version: selectedRecipe?.version ?? null,
        stage_params: current25dStageParams,
        parameters_by_unit: currentParametersByUnit,
        processing_unit_contract_fingerprint:
          selectedRecipe?.processing_unit_contract_fingerprint
          ?? (workspaceDetail?.result?.recipe_snapshot?.processing_unit_contract_fingerprint as string | undefined)
          ?? null,
        calibration_snapshot_reference:
          selectedRecipe?.calibration_snapshot_reference
          ?? (workspaceDetail?.result?.recipe_snapshot?.calibration_snapshot_reference as Record<string, unknown> | undefined)
          ?? null,
      };
      const plan = await api.planPipelinePartialRerun(selectedPipeline.id, {
        selected_unit_id: activeProcessingUnit.id,
        changed_parameters: currentParametersByUnit,
        current_recipe_snapshot: snapshot,
        parent_run_result: (workspaceDetail?.result as Record<string, unknown> | null | undefined) ?? null,
      }, true);
      setPartialRerunExecution(null);
      setPartialRerunPlan(plan);
      setPartialRerunPlanStale(false);
      setPlanContext({
        parentRunId: typeof workspaceDetail?.result?.run_id === "string" ? workspaceDetail.result.run_id : null,
        recipeId: selectedRecipe?.recipe_id ?? null,
        recipeVersion: selectedRecipe?.version ?? null,
      });
    } catch (error) {
      setPartialRerunPlan({
        pipeline_id: selectedPipeline.id,
        safe: false,
        mode: "full_run",
        selected_unit_id: activeProcessingUnit.id,
        execution_boundary_unit_id: activeProcessingUnit.stage_id,
        units_to_reuse: [],
        units_to_rerun: [],
        units_invalidated: [],
        required_missing_artifacts: [],
        warnings: [],
        blocking_reasons: [error instanceof Error ? error.message : "Could not plan partial rerun."],
        fallback_mode: "full_run",
      });
      setPartialRerunPlanStale(false);
    } finally {
      setPartialRerunPlanning(false);
    }
  }

  async function executePartialRerunFromPlan() {
    if (
      selectedPipeline?.id !== "mining_steel_ball_classification_25d"
      || !partialRerunPlan?.plan_id
      || !workspaceDetail?.take_id
    ) return;
    const snapshot = {
      recipe_id: selectedRecipe?.recipe_id ?? null,
      version: selectedRecipe?.version ?? null,
      stage_params: current25dStageParams,
      parameters_by_unit: currentParametersByUnit,
      processing_unit_contract_fingerprint:
        selectedRecipe?.processing_unit_contract_fingerprint
        ?? (workspaceDetail?.result?.recipe_snapshot?.processing_unit_contract_fingerprint as string | undefined)
        ?? null,
      calibration_snapshot_reference:
        selectedRecipe?.calibration_snapshot_reference
        ?? (workspaceDetail?.result?.recipe_snapshot?.calibration_snapshot_reference as Record<string, unknown> | undefined)
        ?? null,
    };
    const parentRunId = typeof workspaceDetail.result?.run_id === "string" && workspaceDetail.result.run_id
      ? workspaceDetail.result.run_id
      : null;
    setPartialRerunExecuting(true);
    setComparisonError(null);
    try {
      const execution = await api.executePipelinePartialRerun(selectedPipeline.id, {
        plan_id: partialRerunPlan.plan_id,
        parent_run_ref: {
          take_id: workspaceDetail.take_id,
          run_id: parentRunId,
        },
        effective_recipe_snapshot: snapshot,
        overrides: {
          stage_params: current25dStageParams,
          parameters_by_unit: currentParametersByUnit,
        },
        options: { compare_to_parent: true },
      });
      setPartialRerunExecution(execution);
      if (execution.ok && execution.child_run_id) {
        setActiveRunId(execution.child_run_id);
        setPlanContext(null);
      }
      if (execution.partial_rerun_plan && !Array.isArray(execution.partial_rerun_plan)) {
        setPartialRerunPlan(execution.partial_rerun_plan as PartialRerunPlanResponse);
        setPartialRerunPlanStale(false);
      }
      if (execution.comparison_id) {
        const response = await api.pipelineComparison(selectedPipeline.id, execution.comparison_id);
        setComparisonResult(response);
        const history = await api.pipelineComparisons(selectedPipeline.id, { take_id: workspaceDetail.take_id, limit: 12 });
        setComparisonHistory(history.comparisons ?? []);
      }
    } catch (error) {
      setPartialRerunExecution({
        ok: false,
        safe: false,
        can_execute: false,
        pipeline_id: selectedPipeline.id,
        take_id: workspaceDetail.take_id,
        child_run_id: null,
        parent_run_id: parentRunId,
        execution_mode: "full_run",
        boundary_unit_id: partialRerunPlan.execution_boundary_unit_id ?? partialRerunPlan.effective_start_stage_id ?? null,
        boundary_stage_id: partialRerunPlan.execution_boundary_unit_id ?? partialRerunPlan.effective_start_stage_id ?? null,
        reused_units: [],
        rerun_units: [],
        invalidated_units: [],
        comparison_id: null,
        result_path: null,
        warnings: [],
        blocking_reasons: [error instanceof Error ? error.message : "Could not execute partial rerun."],
        fallback_mode: "full_run",
        partial_rerun_plan_id: partialRerunPlan.plan_id,
        partial_rerun_plan: partialRerunPlan,
        result: null,
      });
      setComparisonError(error instanceof Error ? error.message : "Could not execute partial rerun.");
    } finally {
      setPartialRerunExecuting(false);
    }
  }

  async function saveCurrentRecipeAsNew() {
    if (selectedPipeline?.id !== "mining_steel_ball_classification_25d") return;
    const stageParams = build25dStageParams();
    if (!stageParams) return;
    const name = window.prompt("Recipe name", selectedRecipe?.name ? `${selectedRecipe.name} copy` : "25D recipe");
    if (!name) return;
    setRecipeLoading(true);
    try {
      const recipe = await api.createPipelineRecipeFromCurrent(selectedPipeline.id, {
        name,
        stage_params: stageParams,
        parameters_by_unit: stageParamsToRecipeParametersByUnit(stageParams, stageProcessingUnits),
        artifact_policy: selectedRecipe?.artifact_policy ?? { diagnostic_level: "engineering", persist_intermediates: true },
        provenance: { source: "processing_lab_save_as_recipe" },
        parent_recipe_id: selectedRecipe?.recipe_id ?? null,
      });
      const response = await api.pipelineRecipes(selectedPipeline.id);
      setPipelineRecipes(response.recipes ?? []);
      setSelectedRecipeId(recipe.recipe_id);
      setRecipePanelMessage(`Saved recipe ${recipe.name} v${recipe.version}.`);
    } catch (error) {
      setRecipePanelMessage(error instanceof Error ? error.message : "Could not save recipe.");
    } finally {
      setRecipeLoading(false);
    }
  }

  async function updateSelectedRecipeVersion() {
    if (selectedPipeline?.id !== "mining_steel_ball_classification_25d" || !selectedRecipe) return;
    const stageParams = build25dStageParams();
    if (!stageParams) return;
    setRecipeLoading(true);
    try {
      const recipe = await api.updatePipelineRecipe(selectedPipeline.id, selectedRecipe.recipe_id, {
        name: selectedRecipe.name,
        description: selectedRecipe.description ?? null,
        tags: selectedRecipe.tags,
        notes: selectedRecipe.notes ?? null,
        stage_params: stageParams,
        parameters_by_unit: stageParamsToRecipeParametersByUnit(stageParams, stageProcessingUnits),
        artifact_policy: selectedRecipe.artifact_policy ?? { diagnostic_level: "engineering", persist_intermediates: true },
        provenance: { source: "processing_lab_update_recipe" },
      });
      const response = await api.pipelineRecipes(selectedPipeline.id);
      setPipelineRecipes(response.recipes ?? []);
      setSelectedRecipeId(recipe.recipe_id);
      setRecipePanelMessage(`Saved ${recipe.name} as v${recipe.version}.`);
    } catch (error) {
      setRecipePanelMessage(error instanceof Error ? error.message : "Could not update recipe.");
    } finally {
      setRecipeLoading(false);
    }
  }

  async function cloneSelectedRecipe() {
    if (selectedPipeline?.id !== "mining_steel_ball_classification_25d" || !selectedRecipe) return;
    const name = window.prompt("Cloned recipe name", `${selectedRecipe.name} copy`);
    if (!name) return;
    setRecipeLoading(true);
    try {
      const recipe = await api.clonePipelineRecipe(selectedPipeline.id, selectedRecipe.recipe_id, { name });
      const response = await api.pipelineRecipes(selectedPipeline.id);
      setPipelineRecipes(response.recipes ?? []);
      setSelectedRecipeId(recipe.recipe_id);
      setRecipePanelMessage(`Cloned recipe ${recipe.name}.`);
    } catch (error) {
      setRecipePanelMessage(error instanceof Error ? error.message : "Could not clone recipe.");
    } finally {
      setRecipeLoading(false);
    }
  }

  function resetEditsToSelectedRecipe() {
    if (!selectedRecipe) return;
    const nextParams = recipeStageParamsToFlat25d((selectedRecipe.stage_params as Record<string, unknown> | undefined) ?? {});
    setPlaneQaParams(nextParams);
    setPlaneQaPersistedParams(nextParams);
    setPlaneQaDirty(false);
    setRecipePanelMessage(`Reset edits to ${selectedRecipe.name}.`);
  }

  async function runSelectedPipeline(reprocess = false, stageParamsOverride?: Record<string, unknown> | null) {
    if (!detail || !selectedPipeline) return;
    setPipelineActionBusy(true);
    setPipelineActionMessage(stageParamsOverride ? "Applying recipe and rerunning…" : "Running pipeline…");
    setResultStale(true);
    try {
      const response = await api.processTake(detail.take_id, {
        pipeline_id: selectedPipeline.id,
        reprocess,
        stage_params: stageParamsOverride ?? build25dStageParams(),
        recipe_id: selectedRecipe?.recipe_id ?? null,
        recipe_version: selectedRecipe?.version ?? null,
      }) as { pipeline_instance_id?: string; run_id?: string; result_path?: string };
      if (response.pipeline_instance_id) {
        setSelectedPipelineInstanceId(response.pipeline_instance_id);
      }
      await load();
      setActiveRunId(null);
      const refreshed = await api.take(detail.take_id);
      setDetail(refreshed);
      setPipelineActionMessage(`Run updated: ${response.run_id ?? refreshed?.result?.processed_at ?? "completed"}`);
      if (roiEditStatus === "rerun_required" || roiEditStatus === "saved") {
        setRoiEditStatus("saved");
      }
    } catch (err) {
      setPipelineActionMessage(`Apply + rerun failed: ${err instanceof Error ? err.message : "unknown error"}`);
    } finally {
      setPipelineActionBusy(false);
      setResultStale(false);
    }
  }

  async function processTakeWithSelectedPipeline(
    takeId: string,
    reprocess = false,
    stageParamsOverride?: Record<string, unknown> | null
  ) {
    if (!selectedPipeline) return null;
    return await api.processTake(takeId, {
      pipeline_id: selectedPipeline.id,
      reprocess,
      stage_params: stageParamsOverride ?? build25dStageParams(),
      recipe_id: selectedRecipe?.recipe_id ?? null,
      recipe_version: selectedRecipe?.version ?? null,
    }) as { pipeline_instance_id?: string; run_id?: string; result_path?: string };
  }

  function applyPlaneQaParams(rerun = false, nextParams?: PlaneQaTuningParams | null) {
    const params = nextParams ?? planeQaParams;
    if (!params) return;
    setPlaneQaPersistedParams(params);
    setPlaneQaDirty(false);
    if (rerun) {
      void runSelectedPipeline(false, build25dStageParams(params));
    }
  }

  async function save25dParamsAsDefaults() {
    const params = build25dStageParams();
    if (!params || typeof window === "undefined") return;
    const existingSaved = read25dSavedDefaults();
    const existingSavedScale = persistedScaleFromKnown((existingSaved.known_object_25d as Record<string, unknown> | undefined) ?? {});
    const knownScale = detail?.result?.artifacts?.find((item) => item.artifact_id === "known_object_scale_validation");
    const knownMeta = (knownScale?.metadata ?? {}) as Record<string, unknown>;
    let scale = scaleCorrectionFromRecord(knownMeta);
    if (!scale && knownScale?.path && detail?.take_id) {
      try {
        const response = await fetch(fileUrl(detail.take_id, knownScale.path));
        if (response.ok) {
          scale = scaleCorrectionFromRecord(await response.json() as Record<string, unknown>);
        }
      } catch {
        scale = null;
      }
    }
    if (!scale) {
      const correctedObject = detail?.result?.objects?.find((item) => {
        const applied = (item as Record<string, unknown>).scale_correction_applied;
        return applied && typeof applied === "object";
      }) as Record<string, unknown> | undefined;
      scale = scaleCorrectionFromRecord(correctedObject?.scale_correction_applied as Record<string, unknown> | undefined);
    }
    if (isIdentityScaleCorrection(scale) && existingSavedScale && !isIdentityScaleCorrection(existingSavedScale)) {
      scale = existingSavedScale;
    }
    const nextKnown = { ...(params.known_object_25d as Record<string, unknown>) };
    if (scale) {
      nextKnown.enabled = false;
      nextKnown.apply_persisted_correction = true;
      nextKnown.persisted_scale_correction_x = scale.x;
      nextKnown.persisted_scale_correction_y = scale.y;
      nextKnown.persisted_scale_correction_z = scale.z;
    }
    const nextParams = { ...params, known_object_25d: nextKnown };
    window.localStorage.setItem(SENSOR_STUDIO_25D_DEFAULTS_KEY, JSON.stringify(nextParams));
    let nextPlaneQaParams = planeQaParams;
    if (scale && planeQaParams) {
      nextPlaneQaParams = {
        ...planeQaParams,
        known_object_enabled: false,
        known_object_apply_persisted_correction: true,
        known_object_persisted_scale_correction_x: scale.x,
        known_object_persisted_scale_correction_y: scale.y,
        known_object_persisted_scale_correction_z: scale.z,
      };
      setPlaneQaParams(nextPlaneQaParams);
    }
    setPipelineActionBusy(true);
    setPipelineActionMessage("Saving 25D defaults to server…");
    try {
      const saved = await api.savePipelineDefaults25d(nextParams);
      setPipelineDefaults25d(saved.defaults || nextParams);
      setPipelineActionMessage(scale
        ? `Saved 25D defaults to server with XYZ scale ${scale.x.toFixed(4)}, ${scale.y.toFixed(4)}, ${scale.z.toFixed(4)}.`
        : "Saved 25D tuning defaults to server. No XYZ scale factors detected in this run.");
    } catch (err) {
      setPipelineActionMessage(`Save defaults failed: ${err instanceof Error ? err.message : "unknown error"}`);
    } finally {
      setPipelineActionBusy(false);
    }
    setPlaneQaPersistedParams(nextPlaneQaParams);
    setPlaneQaDirty(false);
  }

  async function clearSelectedOutputs() {
    if (!detail) return;
    await api.clearTakeOutputs(detail.take_id);
    await load();
  }

  async function editTakeMetadata(field: "friendly_name" | "notes" | "expected_class" | "expected_diameter_mm" | "physical_object_id" | "tags") {
    if (!detail) return;
    const meta = (detail.take_metadata ?? {}) as Record<string, unknown>;
    const current = field === "tags" ? String((meta.tags as string[] | undefined)?.join(",") ?? "") : String(meta[field] ?? "");
    const next = window.prompt(`Set ${field.replace(/_/g, " ")}`, current);
    if (next == null) return;
    const payload: Record<string, unknown> = {
      dataset_id: (meta.dataset_id as string | undefined) ?? "legacy",
      session_id: (meta.session_id as string | undefined) ?? detail.session_id ?? "legacy",
    };
    if (field === "expected_diameter_mm") {
      payload.expected_diameter_mm = next.trim() ? Number(next) : null;
    } else if (field === "tags") {
      payload.tags = next.split(",").map((item) => item.trim()).filter(Boolean);
    } else if (field === "physical_object_id") {
      payload.physical_object_id = next.trim() || null;
    } else {
      payload[field] = next.trim() || null;
    }
    await api.updateTakeMetadata(detail.take_id, payload);
    await load();
    setDetail(await api.take(detail.take_id, activeRunId ?? undefined));
  }

  async function setValidationStatus(status: "unreviewed" | "valid" | "invalid" | "needs_review") {
    if (!detail) return;
    const meta = (detail.take_metadata ?? {}) as Record<string, unknown>;
    await api.updateTakeMetadata(detail.take_id, {
      validation_status: status,
      dataset_id: (meta.dataset_id as string | undefined) ?? "legacy",
      session_id: (meta.session_id as string | undefined) ?? detail.session_id ?? "legacy",
    });
    await load();
    setDetail(await api.take(detail.take_id, activeRunId ?? undefined));
  }

  async function toggleSuggestedTag(tag: string) {
    if (!detail) return;
    const meta = (detail.take_metadata ?? {}) as Record<string, unknown>;
    const current = new Set(((meta.tags as string[] | undefined) ?? []).map((item) => String(item)));
    if (current.has(tag)) current.delete(tag);
    else current.add(tag);
    await api.updateTakeMetadata(detail.take_id, {
      tags: Array.from(current),
      dataset_id: (meta.dataset_id as string | undefined) ?? "legacy",
      session_id: (meta.session_id as string | undefined) ?? detail.session_id ?? "legacy",
    });
    await load();
    setDetail(await api.take(detail.take_id, activeRunId ?? undefined));
  }

  async function rerunLatestPipeline() {
    if (!detail || !selectedPipeline) return;
    await runSelectedPipeline(false);
  }

  function handleTakeSelection(
    takeId: string,
    index: number,
    additive: boolean,
    range: boolean,
  ) {
    setSelectedTakeIds((current) => {
      const next = additive ? new Set(current) : new Set<string>();
      if (range && lastTakeClickIndexRef.current != null) {
        const lo = Math.min(lastTakeClickIndexRef.current, index);
        const hi = Math.max(lastTakeClickIndexRef.current, index);
        for (let i = lo; i <= hi; i += 1) {
          next.add(filteredTakes[i].take_id);
        }
      } else if (next.has(takeId)) {
        next.delete(takeId);
      } else {
        next.add(takeId);
      }
      if (!additive && !range) {
        lastTakeClickIndexRef.current = index;
      } else if (range) {
        lastTakeClickIndexRef.current = index;
      }
      return next;
    });
  }

  function takeMatchesStudioFilters(take: TakeSummary): boolean {
    if (selectedExperimentSession && take.experiment_session_id !== selectedExperimentSession) return false;
    if (modalityFilter !== "all" && !(take.modalities ?? []).includes(modalityFilter)) return false;
    const statusForFamily = (take.processing_by_family ?? []).find((item) => item.family === activePipelineFamily);
    const processedForFamily = Boolean(statusForFamily?.hasCompletedOutput);
    if (takeFilter === "processed" && !processedForFamily) return false;
    if (takeFilter === "unprocessed" && processedForFamily) return false;
    if (takeFilter === "warnings" && !(take.warning_count ?? 0)) return false;
    return true;
  }

  async function resolveCurrentFilterTakeIds(): Promise<string[]> {
    const ids: string[] = [];
    let offset = 0;
    while (true) {
      const page = await api.pagedTakes({
        ...takePageParams,
        limit: 200,
        offset,
      });
      const matching = (page.items ?? []).filter((item) => takeMatchesStudioFilters(item));
      ids.push(...matching.map((item) => item.take_id));
      if (!page.has_more) break;
      offset = page.next_offset ?? (offset + (page.items?.length ?? 0));
    }
    return Array.from(new Set(ids));
  }

  async function submitProcessingJob(payload: Parameters<typeof api.createProcessingJob>[0]) {
    setPipelineActionBusy(true);
    setPipelineActionMessage("Creating processing job...");
    try {
      const created = await api.createProcessingJob(payload);
      setPipelineActionMessage(`Processing job ${created.job.id} queued.`);
      await load();
      return created.job;
    } catch (err) {
      setPipelineActionMessage(`Processing job failed: ${err instanceof Error ? err.message : "unknown error"}`);
      return null;
    } finally {
      setPipelineActionBusy(false);
    }
  }

  async function cancelProcessingJob(jobId: string) {
    setPipelineActionBusy(true);
    setPipelineActionMessage(`Cancelling processing job ${jobId}...`);
    try {
      await api.cancelProcessingJob(jobId);
      await load();
      setPipelineActionMessage(`Processing job ${jobId} cancelled.`);
    } catch (err) {
      setPipelineActionMessage(`Cancel failed: ${err instanceof Error ? err.message : "unknown error"}`);
    } finally {
      setPipelineActionBusy(false);
    }
  }

  async function reprocessSelectedTakes() {
    if (!selectedPipeline || selectedTakeIds.size === 0) return;
    const takeIds = Array.from(selectedTakeIds);
    const stageParams = build25dStageParams();
    setResultStale(true);
    await submitProcessingJob({
      take_ids: takeIds,
      pipeline_id: selectedPipeline.id,
      reprocess_mode: "full",
      skip_completed: false,
      force: false,
      selected_stage: selectedStage?.id ?? null,
      classifier_rules_path: selectedPipeline.id === "mining_steel_ball_classification_25d"
        ? (stageParams?.classify_25d as Record<string, unknown> | undefined)?.classifier_rules_path as string | undefined
        : undefined,
      filters_snapshot: {
        dataset_id: selectedDataset || null,
        dataset_session_id: selectedExperimentSession || null,
        acquisition_session_id: selectedSession || null,
        search: takeSearch || null,
        show_archived: showArchived,
        modality: modalityFilter,
        take_filter: takeFilter,
      },
      created_by: "studio",
      options: {
        stage_params: stageParams,
      },
    });
    setResultStale(false);
  }

  async function reprocessFilteredTakes(mode: "failed_only" | "missing_outputs") {
    if (!selectedPipeline) return;
    const takeIds = await resolveCurrentFilterTakeIds();
    if (!takeIds.length) return;
    const stageParams = build25dStageParams();
    await submitProcessingJob({
      take_ids: takeIds,
      pipeline_id: selectedPipeline.id,
      reprocess_mode: mode,
      skip_completed: false,
      force: false,
      selected_stage: selectedStage?.id ?? null,
      classifier_rules_path: selectedPipeline.id === "mining_steel_ball_classification_25d"
        ? (stageParams?.classify_25d as Record<string, unknown> | undefined)?.classifier_rules_path as string | undefined
        : undefined,
      filters_snapshot: {
        dataset_id: selectedDataset || null,
        dataset_session_id: selectedExperimentSession || null,
        acquisition_session_id: selectedSession || null,
        search: takeSearch || null,
        show_archived: showArchived,
        modality: modalityFilter,
        take_filter: takeFilter,
      },
      created_by: "studio",
      options: {
        stage_params: stageParams,
      },
    });
  }

  async function reprocessCurrentExperimentSession() {
    if (!selectedPipeline || !selectedExperimentSession) return;
    const stageParams = build25dStageParams();
    await submitProcessingJob({
      dataset_session_id: selectedExperimentSession,
      pipeline_id: selectedPipeline.id,
      reprocess_mode: "full",
      skip_completed: false,
      force: false,
      selected_stage: selectedStage?.id ?? null,
      classifier_rules_path: selectedPipeline.id === "mining_steel_ball_classification_25d"
        ? (stageParams?.classify_25d as Record<string, unknown> | undefined)?.classifier_rules_path as string | undefined
        : undefined,
      filters_snapshot: {
        dataset_id: selectedDataset || null,
        dataset_session_id: selectedExperimentSession || null,
        acquisition_session_id: selectedSession || null,
        search: takeSearch || null,
        show_archived: showArchived,
      },
      created_by: "studio",
      options: {
        stage_params: stageParams,
      },
    });
  }

  async function reprocessCurrentDataset() {
    if (!selectedPipeline || !selectedDataset) return;
    const stageParams = build25dStageParams();
    await submitProcessingJob({
      dataset_id: selectedDataset,
      pipeline_id: selectedPipeline.id,
      reprocess_mode: "full",
      skip_completed: false,
      force: false,
      selected_stage: selectedStage?.id ?? null,
      classifier_rules_path: selectedPipeline.id === "mining_steel_ball_classification_25d"
        ? (stageParams?.classify_25d as Record<string, unknown> | undefined)?.classifier_rules_path as string | undefined
        : undefined,
      filters_snapshot: {
        dataset_id: selectedDataset || null,
        dataset_session_id: selectedExperimentSession || null,
        acquisition_session_id: selectedSession || null,
        search: takeSearch || null,
        show_archived: showArchived,
      },
      created_by: "studio",
      options: {
        stage_params: stageParams,
      },
    });
  }

  async function clearSelectedOutputFiles() {
    if (!selectedTakeIds.size) return;
    setPipelineActionBusy(true);
    setPipelineActionMessage(`Clearing outputs for ${selectedTakeIds.size} takes...`);
    try {
      for (const takeId of selectedTakeIds) {
        await api.clearTakeOutputs(takeId);
      }
      await load();
      setPipelineActionMessage(`Cleared outputs for ${selectedTakeIds.size} takes.`);
    } catch (err) {
      setPipelineActionMessage(`Clear outputs failed: ${err instanceof Error ? err.message : "unknown error"}`);
    } finally {
      setPipelineActionBusy(false);
    }
  }

  async function upsertObjectAnnotation(payload: Record<string, unknown>) {
    if (!detail) return;
    await api.upsertObjectAnnotation(detail.take_id, payload);
    await load();
    setDetail(await api.take(detail.take_id, activeRunId ?? undefined));
  }

  async function removeFromDataset() {
    if (!detail) return;
    await api.removeTakeFromDataset(detail.take_id);
    await load();
    setDetail(await api.take(detail.take_id, activeRunId ?? undefined));
  }

  async function archiveCurrentTake() {
    if (!detail) return;
    const reason = window.prompt("Archive reason (optional)", "");
    await api.archiveTake(detail.take_id, reason);
    await load();
    setSelectedTakeId((current) => (current === detail.take_id ? "" : current));
    setDetail(null);
  }

  async function restoreCurrentTake() {
    if (!detail) return;
    await api.restoreTake(detail.take_id);
    await load();
    setDetail(await api.take(detail.take_id, activeRunId ?? undefined));
  }

  async function permanentDeleteCurrentTake() {
    if (!detail) return;
    const confirmation = window.prompt(`Type the take id to permanently delete raw files, processed outputs, and linked metadata:\n${detail.take_id}`, "");
    if (!confirmation) return;
    await api.permanentDeleteTake(detail.take_id, confirmation, true);
    await load();
    setSelectedTakeId("");
    setDetail(null);
  }

  async function captureNewTakeInline(runAfterCapture = false) {
    const disabled = captureDisabledReason(selectedDataset, selectedExperimentSession);
    if (disabled) return;
    setCaptureLoading(true);
    setCaptureError(null);
    try {
      const payload = buildCapturePayload({
        datasetId: selectedDataset,
        sessionId: selectedExperimentSession,
        friendlyName: captureFriendlyName,
        tagsText: captureTags,
        expectedClass: captureExpectedClass,
        expectedDiameterMm: captureExpectedDiameter,
        notes: captureNotes,
      });
      const created = await api.captureTake(payload);
      const takeId = created.take_id;
      await load();
      setSelectedTakeId((current) => nextSelectedTakeAfterCapture(current, takeId));
      setActiveRunId(null);
      const nextDetail = await api.take(takeId);
      setDetail(nextDetail);
      setCaptureFriendlyName("");
      setCaptureTags("");
      setCaptureExpectedClass("");
      setCaptureExpectedDiameter("");
      setCaptureNotes("");
      if (runAfterCapture) {
        setTimeout(() => {
          void runSelectedPipeline(false);
        }, 0);
      }
    } catch (err) {
      setCaptureError(err instanceof Error ? err.message : "Capture failed");
    } finally {
      setCaptureLoading(false);
    }
  }

  async function applyRoi(roi: RoiRect, rerun = false) {
    const nextType = (String(planeQaParams?.reference_surface_region_mode ?? "rectangle") === "full_height_x_band" ? "full_height_x_band" : "rectangle") as RoiType;
    setPlaneQaParams((current) => current ? {
      ...current,
      reference_surface_region_mode: nextType,
      plane_fit_roi_type: nextType,
      plane_fit_roi_x: roi.x,
      plane_fit_roi_y: nextType === "full_height_x_band" ? 0 : roi.y,
      plane_fit_roi_width: roi.width,
      plane_fit_roi_height: roi.height,
      plane_fit_roi_points: [],
    } : current);
    setPlaneQaDirty(true);
    setRoiEditStatus("unsaved");
    if (rerun && detail && selectedPipeline) {
      await runSelectedPipeline(false, {
        ...(build25dStageParams() ?? {}),
        detect_belt_plane: {
          ...((build25dStageParams()?.detect_belt_plane as Record<string, unknown> | undefined) ?? {}),
          plane_fit_roi: {
            enabled: true,
            type: nextType,
            x: roi.x,
            width: roi.width,
            ...(nextType === "full_height_x_band"
              ? {}
              : { y: roi.y, height: roi.height }),
          },
        },
      });
      setRoiEditModeActive(false);
      setRoiEditStatus("saved");
      return;
    }
    setRoiEditModeActive(false);
    setRoiEditStatus("rerun_required");
  }

  async function applyPolygonRoi(points: RoiPolygon, rerun = false) {
    if (points.length < 3) return;
    const normalized = points.map(([x, y]) => [Math.round(Number(x)), Math.round(Number(y))] as [number, number]);
    setPlaneQaParams((current) => current ? {
      ...current,
      reference_surface_region_mode: "polygon",
      plane_fit_roi_type: "polygon",
      plane_fit_roi_points: normalized,
    } : current);
    setPlaneQaDirty(true);
    setRoiEditStatus("unsaved");
    if (rerun && detail && selectedPipeline) {
      await runSelectedPipeline(false, {
        ...(build25dStageParams() ?? {}),
        detect_belt_plane: {
          ...((build25dStageParams()?.detect_belt_plane as Record<string, unknown> | undefined) ?? {}),
          plane_fit_roi: {
            enabled: true,
            type: "polygon",
            points: normalized,
          },
        },
      });
      setRoiEditModeActive(false);
      setRoiEditStatus("saved");
      return;
    }
    setRoiEditModeActive(false);
    setRoiEditStatus("rerun_required");
  }

  async function clearRoi() {
    setPlaneQaParams((current) => current ? { ...current, reference_surface_region_mode: "none", plane_fit_roi_points: [] } : current);
    setPlaneQaDirty(true);
    setRoiEditModeActive(false);
    setRoiEditStatus("rerun_required");
  }

  async function applyBlobParams(params: Record<string, unknown>, rerun = false) {
    if (!selectedPipelineInstance) return;
    const configured_steps = selectedPipelineInstance.configured_steps.map((step) =>
      step.step_id === "blob_detection"
        ? { ...step, params: { ...step.params, ...params } }
        : step
    );
    const next = await api.updatePipelineInstance(selectedPipelineInstance.id, { configured_steps });
    setPipelineInstances((items) => items.map((item) => (item.id === next.id ? next : item)));
    if (rerun && detail && selectedPipeline) {
      await runSelectedPipeline(false);
    }
  }

  async function triggerSegmentationPreview() {
    if (!previewEnabled || !detail || !selectedPipeline || !segmentationPreviewParams) return;
    setSegmentationPreviewLoading(true);
    setSegmentationPreviewError(null);
    try {
      const response = await api.previewSegmentation({
        take_id: detail.take_id,
        pipeline_id: selectedPipeline.id,
        params: segmentationPreviewParams,
      });
      setSegmentationPreview(response);
      setSegmentationPreviewDirty(false);
    } catch (err) {
      setSegmentationPreviewError(err instanceof Error ? err.message : "Preview failed");
    } finally {
      setSegmentationPreviewLoading(false);
    }
  }

  async function applySegmentationParams(rerun = false) {
    if (!selectedPipelineInstance || !segmentationPreviewParams) return;
    const configured_steps = selectedPipelineInstance.configured_steps.map((step) => {
      if (step.step_id === "threshold") {
        return {
          ...step,
          params: {
            ...step.params,
            mode: Boolean(segmentationPreviewParams["auto_threshold"]) ? "otsu" : "fixed",
            value: Number(segmentationPreviewParams["threshold"] ?? 125),
            invert: Boolean(segmentationPreviewParams["invert"]),
            roi_enabled: Boolean(segmentationPreviewParams["roi_enabled"]),
          },
        };
      }
      if (step.step_id === "morphology") {
        const maxArea = segmentationPreviewParams["max_area_px"] == null || segmentationPreviewParams["max_area_px"] === ""
          ? null
          : Number(segmentationPreviewParams["max_area_px"]);
        return {
          ...step,
          params: {
            ...step.params,
            morph_op: String(segmentationPreviewParams["morph_op"] ?? "open_close"),
            operation: String(segmentationPreviewParams["morph_op"] ?? "open_close"),
            erode_kernel_size: Number(segmentationPreviewParams["erode_kernel_size"] ?? 3),
            erode_iterations: Number(segmentationPreviewParams["erode_iterations"] ?? 1),
            dilate_kernel_size: Number(segmentationPreviewParams["dilate_kernel_size"] ?? 3),
            dilate_iterations: Number(segmentationPreviewParams["dilate_iterations"] ?? 1),
            close_kernel_size: Number(segmentationPreviewParams["close_kernel_size"] ?? 5),
            fill_holes: Boolean(segmentationPreviewParams["fill_holes"] ?? true),
            min_area_px: Number(segmentationPreviewParams["min_area_px"] ?? 120),
            max_area_px: maxArea,
            cleanup_min_area: Number(segmentationPreviewParams["min_area_px"] ?? 120),
            cleanup_max_area: maxArea,
          },
        };
      }
      return step;
    });
    const next = await api.updatePipelineInstance(selectedPipelineInstance.id, { configured_steps });
    setPipelineInstances((items) => items.map((item) => (item.id === next.id ? next : item)));
    setSegmentationPersistedParams(segmentationPreviewParams);
    setSegmentationPreviewDirty(false);
    if (rerun && detail && selectedPipeline) {
      await runSelectedPipeline(false);
      setSegmentationPreview(null);
    }
  }

  async function triggerEllipsePreview() {
    if (!ellipsePreviewEnabled || !detail || !selectedPipeline || !ellipsePreviewParams) return;
    setEllipsePreviewLoading(true);
    setEllipsePreviewError(null);
    try {
      const response = await api.previewSegmentation({
        take_id: detail.take_id,
        pipeline_id: selectedPipeline.id,
        stage_id: "ellipse_fitting",
        params: ellipsePreviewParams,
      });
      setSegmentationPreview(response);
      setEllipsePreviewDirty(false);
    } catch (err) {
      setEllipsePreviewError(err instanceof Error ? err.message : "Preview failed");
    } finally {
      setEllipsePreviewLoading(false);
    }
  }

  async function applyEllipseParams(rerun = false) {
    if (!selectedPipelineInstance || !ellipsePreviewParams) return;
    const configured_steps = selectedPipelineInstance.configured_steps.map((step) =>
      step.step_id === "ellipse_fitting"
        ? { ...step, params: { ...step.params, ...ellipsePreviewParams } }
        : step
    );
    const next = await api.updatePipelineInstance(selectedPipelineInstance.id, { configured_steps });
    setPipelineInstances((items) => items.map((item) => (item.id === next.id ? next : item)));
    setEllipsePersistedParams(ellipsePreviewParams);
    setEllipsePreviewDirty(false);
    if (rerun && detail && selectedPipeline) {
      await runSelectedPipeline(false);
      setSegmentationPreview(null);
    }
  }

  function resetEllipseDefaults() {
    const schema = (selectedStage?.parameter_schema ?? {}) as { fields?: Record<string, { default?: unknown }> };
    const defaults = Object.fromEntries(Object.entries(schema.fields ?? {}).map(([key, meta]) => [key, meta.default]));
    setEllipsePreviewParams(defaults);
    setEllipsePreviewDirty(true);
  }

  function renderStepParamEditor(step: PipelineInstance["configured_steps"][number]) {
    const template = selectedPipelineInstance ? templateById.get(selectedPipelineInstance.template_id) : null;
    const definition = template?.steps.find((item) => item.step_id === step.step_id);
    const schema = definition?.params_schema ?? {};
    const keys = Object.keys(schema);
    return keys.map((key) => {
      const meta = (schema[key] ?? {}) as Record<string, unknown>;
      const raw = step.params[key];
      if (Array.isArray(meta.enum)) {
        return (
          <label key={`${step.step_id}_${key}`} className="field-label">
            {key}
            <select value={String(raw ?? meta.enum[0])} onChange={(event) => void updateStepParam(step.step_id, key, event.target.value)}>
              {meta.enum.map((value) => (
                <option key={String(value)} value={String(value)}>{String(value)}</option>
              ))}
            </select>
          </label>
        );
      }
      if (meta.type === "boolean") {
        return (
          <label key={`${step.step_id}_${key}`}>
            <input checked={Boolean(raw)} type="checkbox" onChange={(event) => void updateStepParam(step.step_id, key, event.target.checked)} />
            {key}
          </label>
        );
      }
      const isInt = meta.type === "integer";
      return (
        <label key={`${step.step_id}_${key}`} className="field-label">
          {key}
          <input
            type="number"
            step={isInt ? 1 : 0.01}
            value={Number(raw ?? 0)}
            onChange={(event) => void updateStepParam(step.step_id, key, isInt ? Math.trunc(Number(event.target.value)) : Number(event.target.value))}
          />
        </label>
      );
    });
  }

  function workerBadgeLabel(worker: RuntimeWorkerStatus | null): string {
    if (!worker) return "stopped";
    if (worker.state === "running" && String(worker.last_event_summary ?? "").toLowerCase().includes("dry-run")) return "dry_run";
    return worker.state;
  }

  function formatProcessingJobScope(job: ProcessingJobRecord): string {
    const filters = job.filters_summary ?? {};
    const summaryLabel = typeof job.summary?.scope_label === "string" ? job.summary.scope_label : null;
    if (summaryLabel) return summaryLabel;
    if (job.scope_type === "dataset_session") {
      return `Experiment session ${String(filters.dataset_session_id ?? filters.session_id ?? job.summary?.dataset_session_id ?? "selected")}`;
    }
    if (job.scope_type === "acquisition_session") {
      return `Runtime acquisition group ${String(filters.acquisition_session_id ?? job.summary?.acquisition_session_id ?? "selected")}`;
    }
    if (job.scope_type === "dataset") {
      return `Dataset ${String(filters.dataset_id ?? job.summary?.dataset_id ?? "selected")}`;
    }
    if (job.scope_type === "take_selection") {
      return "Selected takes";
    }
    return "Filtered takes";
  }

  const activeProcessingJobs = processingJobs.filter((job) => job.status === "queued" || job.status === "running");
  const processingJobStatusCounts = {
    queued: processingJobCounts.queued ?? processingJobs.filter((job) => job.status === "queued").length,
    running: processingJobCounts.running ?? processingJobs.filter((job) => job.status === "running").length,
    completed: processingJobCounts.completed ?? processingJobs.filter((job) => job.status === "completed").length,
    failed: processingJobCounts.failed ?? processingJobs.filter((job) => job.status === "failed").length,
    cancelled: processingJobCounts.cancelled ?? processingJobs.filter((job) => job.status === "cancelled").length,
    partial: processingJobCounts.partial ?? processingJobs.filter((job) => job.status === "partial").length,
  };

  const hasSubstagePlan = stageSubstagePlan.substages.length > 0;
  const hasSubstageRail = hasSubstagePlan && shellLayoutMode === "wide";

  return (
    <main className={`studio-workspace studio-shell mode-${studioMode} ${panelWidthClass} shell-${shellLayoutMode} ${focusMode ? "focus-mode" : ""} ${rightPanelOverlay ? "right-panel-overlay-mode" : ""} ${leftRailVisible ? "left-rail-visible" : "left-rail-hidden"} ${hasSubstageRail ? "has-substage-rail" : ""} ${leftRailReallyCollapsed ? "left-rail-collapsed" : ""}`}>
      <header className="studio-context-bar">
        <div className="studio-context-main">
          <span className="toolbar-chip identity-chip">Take: {selectedTakeSummary?.friendly_name || detail?.take_id || "none"}</span>
          <span className="toolbar-chip context-chip">{selectedContextDatasetName || "No dataset"} / {selectedContextSessionName || "No session"}</span>
          <label className="toolbar-chip pipeline-chip">
            Pipeline
            <select value={selectedPipelineId} onChange={(event) => selectPipeline(event.target.value)}>
              {pipelines.map((pipeline) => (
                <option key={pipeline.id} value={pipeline.id}>{pipeline.display_name}</option>
              ))}
            </select>
          </label>
          <span className={`toolbar-chip compatibility-chip ${compatible ? "ok" : "warn"}`}>{compatible ? "compatible" : "incompatible"}</span>
          <span className="toolbar-chip run-chip">
            Run
            <strong>{selectedRunTrace?.status ?? "not run"}</strong>
            <small>{selectedRunTrace?.timestamp || "latest selection"}</small>
          </span>
          {canonicalSelectedStageId === "detect_belt_plane" && activeDisplayItem && (
            <>
              <span className={`toolbar-chip ${activeDisplayItem.category === "active" ? "compatibility-chip ok" : "compatibility-chip warn"}`}>{activeDisplayItem.category.toUpperCase()}</span>
              <span className="tab-with-help">
                <span className="toolbar-chip">{displayRoleLabel(activeDisplayItem.role)}</span>
                <InfoHint
                  content={resolveStudioHelpEntry(detectReferenceDisplayRoleHelpKey(activeDisplayItem.role))}
                  label="Open role help"
                />
              </span>
            </>
          )}
        </div>
        <div className="studio-context-actions">
          <div className="studio-actions toolbar-actions">
            {leftRailDrawer && (
              <button type="button" onClick={() => setLeftRailExpanded((current) => !current)}>
                {leftRailExpanded ? "Hide takes" : "Takes"}
              </button>
            )}
            {rightPanelOverlay && (
              <button type="button" onClick={() => setInspectorExpanded((current) => !current)}>
                {inspectorExpanded ? "Hide controls" : "Controls"}
              </button>
            )}
            <button type="button" onClick={() => setFocusMode((current) => !current)}>
              {focusMode ? "Exit focus" : "Focus view"}
            </button>
            <button disabled={!compatible || pipelineActionBusy} title={!compatible ? "Cannot execute: incompatible modalities." : ""} onClick={() => void runSelectedPipeline(false)} type="button">Run</button>
            <button disabled={!detail || !compatible || pipelineActionBusy} title={!compatible ? "Cannot execute: incompatible modalities." : ""} onClick={() => void runSelectedPipeline(true)} type="button">Reprocess</button>
            <button type="button" onClick={() => setJobsPopoverOpen((current) => !current)}>
              Jobs {processingJobStatusCounts.running > 0 ? `(${processingJobStatusCounts.running})` : ""}
            </button>
            {detail?.take_id && (
              <a
                className="studio-report-link"
                href={`/studio/report?take_id=${encodeURIComponent(detail.take_id)}&pipeline_id=${encodeURIComponent(selectedPipelineId)}`}
                title="Printable step-by-step walkthrough of every stage, substage, and parameter for this take"
              >
                Report
              </a>
            )}
          </div>
        </div>
      </header>
      <aside className={`studio-sidebar studio-left-rail ${leftRailExpanded ? "expanded" : "compact"} ${leftRailDrawer ? "drawer" : ""} ${leftRailVisible ? "is-visible" : "is-hidden"} ${leftRailReallyCollapsed ? "is-collapsed" : ""}`}>
        <div className="studio-sidebar-title studio-left-rail-header">
          <div>
            <span className="eyebrow">Studio</span>
            <strong>Processing Lab</strong>
          </div>
          <div className="studio-left-rail-header-actions">
            <button
              type="button"
              className="panel-collapse-toggle rail-detail-toggle"
              onClick={() => setLeftRailExpanded((current) => !current)}
              aria-expanded={leftRailExpanded}
              aria-label={leftRailExpanded ? "Show less detail" : "Show more detail"}
              title={leftRailExpanded ? "Show less detail" : "Show more detail"}
            >
              {leftRailExpanded ? "▾" : "▸"}
            </button>
            {leftRailCanCollapse && (
              <button
                type="button"
                className="panel-collapse-toggle rail-visibility-toggle"
                onClick={() => setLeftRailCollapsed((current) => !current)}
                aria-expanded={!leftRailReallyCollapsed}
                aria-label={leftRailReallyCollapsed ? "Expand takes panel" : "Collapse takes panel"}
                title={leftRailReallyCollapsed ? "Expand takes panel" : "Collapse takes panel"}
              >
                {leftRailReallyCollapsed ? "›" : "‹"}
              </button>
            )}
          </div>
        </div>
        <div className="studio-handoff-note">
          <small>Semantic curation is managed in <a href="/datasets">Datasets</a>.</small>
        </div>
        <div className="studio-curation-context">
          <small className="studio-curation-context-title">Selected take</small>
          <div className="studio-selected-take-link">
            <span className="semantic-chip active-take-chip">{selectedTakeSummary?.friendly_name || detail?.take_id || "No take selected"}</span>
            {selectedTakeSummary?.thumbnail_path && (
              <img
                className="selected-context-thumb"
                alt={selectedTakeSummary?.friendly_name || selectedTakeSummary?.take_id}
                src={fileUrl(selectedTakeSummary.take_id, selectedTakeSummary.thumbnail_path)}
              />
            )}
          </div>
          <div className="studio-selected-take-meta">
            <small>{selectedContextDatasetName}</small>
            <small>{selectedContextSessionName}</small>
          </div>
          {activeFilterChips.length > 0 && (
            <div className="studio-filters-applied">
              <small>Operational filters</small>
              <div className="semantic-chip-row">
                {activeFilterChips.map((chip) => <span key={chip} className="semantic-chip filter-chip">{chip}</span>)}
              </div>
            </div>
          )}
        </div>
        <label className="field-label">
          Dataset
          <select value={selectedDataset} onChange={(event) => { setSelectedDataset(event.target.value); setSelectedExperimentSession(""); }}>
            <option value="">All datasets</option>
            {datasets.map((dataset) => (
              <option key={dataset.id} value={dataset.id}>{dataset.name}</option>
            ))}
          </select>
          {leftRailExpanded && (
            <div className="studio-inline-action-row">
              <button type="button" disabled={pipelineActionBusy || !selectedPipeline || !selectedDataset} onClick={() => void reprocessCurrentDataset()}>Reprocess dataset</button>
            </div>
          )}
        </label>
        <label className="field-label">
          Experiment session
          <select value={selectedExperimentSession} onChange={(event) => setSelectedExperimentSession(event.target.value)} disabled={!selectedDataset}>
            <option value="">All experiment sessions</option>
            {datasetSessions.map((session) => (
              <option key={session.id} value={session.id}>{session.name}</option>
            ))}
          </select>
          {leftRailExpanded && <small className="field-help">Engineering/acquisition context grouping used for replay, calibration, and processing workflows.</small>}
          {leftRailExpanded && (
            <div className="studio-inline-action-row">
              <button type="button" disabled={pipelineActionBusy || !selectedPipeline || !selectedExperimentSession} onClick={() => void reprocessCurrentExperimentSession()}>Reprocess session</button>
            </div>
          )}
        </label>
        {leftRailExpanded && (
        <details className="studio-runtime-grouping">
          <summary>Advanced runtime grouping</summary>
          <div className="studio-runtime-grouping-body">
            <small className="field-help">Low-level runtime/replay grouping primarily used for ingestion and operational replay workflows.</small>
            <label className="field-label">
              Runtime acquisition group
              <select value={selectedSession} onChange={(event) => setSelectedSession(event.target.value)}>
                <option value="">All runtime groups</option>
                {sessions.map((session) => (
                  <option key={session.session_id} value={session.session_id}>{session.session_id}</option>
                ))}
              </select>
            </label>
          </div>
        </details>
        )}
        {leftRailExpanded && (
        <details className="studio-pipeline-card studio-capture-card">
          <summary>Capture take</summary>
          {!selectedDataset && <small>Select a dataset/session to enable capture.</small>}
          {!!captureDisabledReason(selectedDataset, selectedExperimentSession) && (
            <small>{captureDisabledReason(selectedDataset, selectedExperimentSession)}</small>
          )}
          <div className="inline-create-card">
            <input value={captureFriendlyName} onChange={(event) => setCaptureFriendlyName(event.target.value)} placeholder="friendly name (optional)" />
            <input value={captureTags} onChange={(event) => setCaptureTags(event.target.value)} placeholder="tags comma-separated (optional)" />
            <input value={captureExpectedClass} onChange={(event) => setCaptureExpectedClass(event.target.value)} placeholder="expected class (optional)" />
            <input value={captureExpectedDiameter} onChange={(event) => setCaptureExpectedDiameter(event.target.value)} placeholder="expected diameter mm (optional)" />
            <input value={captureNotes} onChange={(event) => setCaptureNotes(event.target.value)} placeholder="notes (optional)" />
            <div className="inline-create-actions">
              <button
                type="button"
                disabled={captureLoading || !!captureDisabledReason(selectedDataset, selectedExperimentSession)}
                onClick={() => void captureNewTakeInline(false)}
                title={captureDisabledReason(selectedDataset, selectedExperimentSession) ?? ""}
              >
                {captureLoading ? "Capturing..." : "Capture"}
              </button>
              <button
                type="button"
                disabled={captureLoading || !!captureDisabledReason(selectedDataset, selectedExperimentSession)}
                onClick={() => void captureNewTakeInline(true)}
                title={captureDisabledReason(selectedDataset, selectedExperimentSession) ?? ""}
              >
                {captureLoading ? "Capturing..." : "Capture + Run"}
              </button>
            </div>
            {captureError && <small>{captureError}</small>}
          </div>
        </details>
        )}
        <label className="field-label">
          Search
          <input value={takeSearch} onChange={(event) => setTakeSearch(event.target.value)} placeholder="take id or name" />
        </label>
        <label>
          <input checked={showArchived} type="checkbox" onChange={(event) => setShowArchived(event.target.checked)} />
          Show archived
        </label>
        <div className="studio-filter-grid">
          <label className="field-label">
            Modality
            <select value={modalityFilter} onChange={(event) => setModalityFilter(event.target.value as CaptureModality | "all")}>
              <option value="all">All</option>
              <option value="rgb">RGB</option>
              <option value="point_cloud">Point Cloud</option>
              <option value="rgb_video">RGB Video</option>
              <option value="heightmap">Heightmap</option>
            </select>
          </label>
          <label className="field-label">
            Status
            <select value={takeFilter} onChange={(event) => setTakeFilter(event.target.value as TakeFilter)}>
              <option value="all">All</option>
              <option value="processed">Processed</option>
              <option value="unprocessed">Needs {activePipelineFamily.toUpperCase()} processing</option>
              <option value="warnings">Warnings</option>
            </select>
          </label>
        </div>
        {leftRailExpanded && (
        <details className="studio-take-bulkbar">
          <summary>Batch ({bulkSelectionCount})</summary>
          <div className="studio-take-bulkbar-body">
            <label>
              <input
                type="checkbox"
                checked={allVisibleSelected}
                onChange={(event) => {
                  const checked = event.target.checked;
                  setSelectedTakeIds(checked ? new Set(allVisibleTakeIds) : new Set());
                  if (!checked) lastTakeClickIndexRef.current = null;
                }}
              />
              Select visible
            </label>
            <button
              type="button"
              disabled={pipelineActionBusy || !selectedPipeline || bulkSelectionCount === 0}
              onClick={() => void reprocessSelectedTakes()}
            >
              Reprocess selected
            </button>
            <button
              type="button"
              disabled={
                batchComparisonBusy ||
                !selectedPipeline ||
                selectedPipeline.id !== "mining_steel_ball_classification_25d" ||
                bulkSelectionCount === 0
              }
              title="Compares the Compare tab's left source (fixed) against each selected take's right source. Configure sources in the Compare tab first."
              onClick={() => void runBatchComparison()}
            >
              {batchComparisonBusy ? "Comparing…" : `Compare selected (${bulkSelectionCount})`}
            </button>
            <button
              type="button"
              disabled={pipelineActionBusy || !selectedPipeline || takesLoading}
              onClick={() => void reprocessFilteredTakes("failed_only")}
            >
              Reprocess failed
            </button>
            <button
              type="button"
              disabled={pipelineActionBusy || !selectedPipeline || takesLoading}
              onClick={() => void reprocessFilteredTakes("missing_outputs")}
            >
              Reprocess missing outputs
            </button>
            <button
              type="button"
              disabled={pipelineActionBusy || bulkSelectionCount === 0}
              onClick={() => void clearSelectedOutputFiles()}
            >
              Clear outputs
            </button>
            <button
              type="button"
              disabled={bulkSelectionCount === 0}
              onClick={() => {
                setSelectedTakeIds(new Set());
                lastTakeClickIndexRef.current = null;
              }}
            >
              Clear
            </button>
          </div>
        </details>
        )}
        <div className="studio-take-list">
          {takesLoading && !filteredTakes.length && <div className="empty-state compact">Loading takes…</div>}
          {filteredTakes.map((take, index) => {
            const isSelectedForBulk = selectedTakeIds.has(take.take_id);
            return (
              <div className="take-card-row" key={take.take_id}>
                <input
                  aria-label={`Select ${take.friendly_name || take.take_id}`}
                  checked={isSelectedForBulk}
                  onChange={() => undefined}
                  onClick={(event) => {
                    event.stopPropagation();
                    handleTakeSelection(take.take_id, index, true, event.shiftKey);
                  }}
                  type="checkbox"
                />
                <button
                  className={`${take.take_id === selectedTakeId ? "active" : ""} ${isSelectedForBulk ? "multi-selected" : ""}`}
                  onClick={(event) => {
                    if (event.shiftKey || event.metaKey || event.ctrlKey) {
                      event.preventDefault();
                      handleTakeSelection(take.take_id, index, event.metaKey || event.ctrlKey, event.shiftKey);
                      return;
                    }
                    setSelectedTakeId(take.take_id);
                    lastTakeClickIndexRef.current = index;
                  }}
                  type="button"
                >
                  <strong>{take.friendly_name || take.take_id}</strong>
                  {take.friendly_name && take.friendly_name !== take.take_id && <small>{take.take_id}</small>}
                  <div className="take-card-main">
                    {take.thumbnail_path ? (
                      <img alt={take.friendly_name || take.take_id} className="take-thumb" src={fileUrl(take.take_id, take.thumbnail_path)} />
                    ) : (
                      <div aria-hidden className="take-thumb take-thumb-empty" />
                    )}
                    <div className="take-card-meta">
                      <div className="semantic-chip-row">
                        {take.latest_run_status && <span className="semantic-chip">{take.latest_run_status}</span>}
                        {(take.modalities ?? []).slice(0, 2).map((item) => <span key={item} className="semantic-chip tag-chip">{item}</span>)}
                      </div>
                      <div className="take-class-meta">
                        <small>{take.dataset_name || "No dataset"}</small>
                        <small>{take.experiment_session_name || "No session"}</small>
                      </div>
                    </div>
                  </div>
                </button>
              </div>
            );
          })}
          {takesHasMore && (
            <button
              type="button"
              className="load-more-takes-btn"
              disabled={takesLoadingMore}
              onClick={() => void loadMoreTakes()}
            >
              {takesLoadingMore ? "Loading…" : "Load more"}
            </button>
          )}
        </div>
        {leftRailExpanded && <div className="studio-pipeline-card studio-datasets-link-card">
          <small>Labeling, validation, annotations, and dataset composition are in <a href="/datasets">Datasets</a>.</small>
        </div>}
        {leftRailExpanded && <details className="studio-pipeline-card">
          <summary>Process config (advanced)</summary>
          <span>Create Process</span>
          <label className="field-label">
            Application
            <select value={selectedTemplateId} onChange={(event) => setSelectedTemplateId(event.target.value)}>
              {processTemplates.map((template) => (
                <option key={template.id} value={template.id}>{template.name}</option>
              ))}
            </select>
          </label>
          <label className="field-label">
            Input Type
            <select value={selectedInputType} onChange={(event) => setSelectedInputType(event.target.value)}>
              <option value="rgb_image">RGB</option>
              <option value="grayscale_image">Grayscale</option>
              <option value="reflectance_image">Reflectance</option>
            </select>
          </label>
          <label className="field-label">
            Image Path (optional)
            <input value={manualImagePath} onChange={(event) => setManualImagePath(event.target.value)} placeholder="incoming/<take>/image.png or /abs/path/image.jpg" />
          </label>
          <button type="button" onClick={() => void createProcessPipeline()}>Create Pipeline</button>
          <label className="field-label">
            Pipeline Instance
            <select value={selectedPipelineInstanceId} onChange={(event) => setSelectedPipelineInstanceId(event.target.value)}>
              {pipelineInstances.map((instance) => (
                <option key={instance.id} value={instance.id}>{instance.name}</option>
              ))}
            </select>
          </label>
          <button type="button" disabled={!selectedPipelineInstance} onClick={() => void runInstance()}>Rerun Process</button>
          {!!selectedPipelineInstance?.execution_history?.length && (
            <small>
              Last summary: {JSON.stringify((selectedPipelineInstance.execution_history[0] as Record<string, unknown>).summary ?? {})}
            </small>
          )}
          {!!selectedPipelineInstance?.execution_history?.length && (
            <small>
              Persisted artifacts: {
                ((selectedPipelineInstance.execution_history[0] as { artifacts?: Array<{ path?: string | null }> }).artifacts ?? [])
                  .filter((item) => Boolean(item.path)).length
              }
            </small>
          )}
          {!!selectedPipelineInstance && selectedPipelineInstance.configured_steps.map((step) => (
            <div key={step.step_id}>
              <small>{step.step_id}</small>
              <label>
                <input checked={step.enabled} type="checkbox" onChange={(event) => void toggleStep(step.step_id, event.target.checked)} />
                enabled
              </label>
              {renderStepParamEditor(step)}
            </div>
          ))}
        </details>}
        {leftRailExpanded && <div className="studio-pipeline-card">
          <span>Run history (selected take + pipeline)</span>
          {!contextualRunHistory.length && <small>No runs yet for this selection.</small>}
          {contextualRunHistory.map((run) => (
            <small key={run.id}>{run.timestamp || run.id} | {run.status} | {run.duration.toFixed(1)} ms</small>
          ))}
        </div>}
      </aside>

      <aside className={`studio-stage-rail ${shellLayoutMode === "medium" || shellLayoutMode === "narrow" || shellLayoutMode === "very_narrow" ? "compact" : ""} ${stageRailVisible ? "is-visible" : "is-hidden"}`} aria-label="Pipeline stages">
        <div className="studio-stage-rail-header">
          <span className="eyebrow">Stages</span>
          <strong>{selectedStage ? compactStageLabel(selectedStage.id, prettyStageLabel(selectedStage.id, selectedStage.display_name)) : "Pipeline"}</strong>
        </div>
        <div className="studio-stage-rail-list">
          {stageNavigator.map((stage) => (
            <button
              key={stage.id}
              type="button"
              className={`studio-stage-rail-row ${stage.selected ? "active" : ""} status-${stage.status}`}
              onClick={() => selectStage(stage.id)}
              title={stage.label}
            >
              <span className="studio-stage-rail-status" aria-hidden>{stage.status === "completed" ? "✓" : stage.status === "failed" || stage.status === "error" ? "!" : "•"}</span>
              <span className="studio-stage-rail-copy">
                <strong>{stage.shortLabel}</strong>
                <small>{stage.label}</small>
              </span>
              <span className="studio-stage-rail-meta">
                <small>{stage.durationMs == null ? "—" : `${stage.durationMs.toFixed(0)} ms`}</small>
                {stage.warningCount > 0 && <small className="warning-badge">{stage.warningCount}</small>}
              </span>
            </button>
          ))}
        </div>
      </aside>

      {hasSubstageRail && (
        <aside className="studio-substage-rail" aria-label="Detect reference surface subprocesses">
          <div className="studio-stage-rail-header">
            <span className="eyebrow">Substages</span>
            <strong>{activeSubstage?.label ?? "Strategy path"}</strong>
          </div>
          <StageSubstageRailList
            substages={stageSubstagePlan.substages}
            activeScope={activeReferenceViewScope}
            selectedSubstageId={selectedSubstageId}
            onSelectStrategyPath={activateStrategyPath}
            onSelectSubstage={activateReferenceSubstage}
            relevanceSummary={substageRelevanceSummary}
            renderRelevanceBadge={(relevance, reason) => <ArtifactRelevanceBadge relevance={relevance} reason={reason} />}
          />
        </aside>
      )}

      <section className="studio-center">
        {measurementConfigOpen && planeQaParams && (
          <div className="modal-backdrop" role="dialog" aria-modal="true" onClick={() => setMeasurementConfigOpen(false)}>
            <div className="modal-panel measurement-config-modal" onClick={(event) => event.stopPropagation()}>
              <div className="modal-header">
                <h3>Known-Cube Calibration Config</h3>
                <button type="button" onClick={() => setMeasurementConfigOpen(false)}>Close</button>
              </div>
              <div className="measurement-config-modal-body">
                <div className="stage-parameter-panel-header">
                  <strong>Measurement controls</strong>
                  <small>{pipelineActionBusy ? "Applying recipe and rerunning…" : (planeQaDirty ? "Unsaved changes" : "Known-cube calibration config")}</small>
                </div>
                <div className="measurement-controls-row modal-controls">
                  <label>Enable
                    <input type="checkbox" checked={Boolean(planeQaParams.known_object_enabled ?? false)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, known_object_enabled: e.target.checked } : c); setPlaneQaDirty(true); }} />
                  </label>
                  <label>Apply correction
                    <input type="checkbox" checked={Boolean(planeQaParams.known_object_apply_correction ?? true)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, known_object_apply_correction: e.target.checked, known_object_enabled: e.target.checked ? true : c.known_object_enabled } : c); setPlaneQaDirty(true); }} />
                  </label>
                  <label>Target
                    <select value={String(planeQaParams.known_object_target_selection ?? "largest_component")} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, known_object_target_selection: e.target.value } : c); setPlaneQaDirty(true); }}>
                      <option value="largest_component">largest component</option>
                      <option value="manual_component_id">object id</option>
                      <option value="label_match">label match</option>
                    </select>
                  </label>
                  <label>Width X mm
                    <input type="number" min={0} max={1000} step={0.1} value={Number(planeQaParams.known_width_mm ?? 40)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, known_width_mm: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                  </label>
                  <label>Depth Y mm
                    <input type="number" min={0} max={1000} step={0.1} value={Number(planeQaParams.known_depth_mm ?? 40)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, known_depth_mm: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                  </label>
                  <label>Height Z mm
                    <input type="number" min={0} max={1000} step={0.1} value={Number(planeQaParams.known_height_mm ?? 25)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, known_height_mm: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                  </label>
                </div>
                <div className="reference-surface-roi-actions">
                  <button type="button" disabled={pipelineActionBusy} onClick={() => applyPlaneQaParams(false)}>Apply</button>
                  <button type="button" disabled={pipelineActionBusy} onClick={() => applyPlaneQaParams(true)}>Apply + rerun</button>
                  <button type="button" disabled={pipelineActionBusy} onClick={() => void save25dParamsAsDefaults()}>Save defaults</button>
                  <button type="button" disabled={pipelineActionBusy} onClick={() => { setPlaneQaParams(planeQaPersistedParams); setPlaneQaDirty(false); }}>Reset</button>
                </div>
              </div>
            </div>
          </div>
        )}
        {pipelineActionMessage && <small className={resultStale ? "preview-stale" : "section-note"}>{pipelineActionMessage}</small>}
        {!compatible && selectedPipeline && (
          <small className="section-note">{incompatibleModalityMessage(selectedPipeline, detail?.modalities) ?? "Cannot execute: incompatible modalities."}</small>
        )}
        <section className="studio-result-workspace">
        <div className="studio-result-hero">
        <div className="stage-hero-header compact">
          {stripeHeaderSummary ? <small className="stage-hero-note">{stripeHeaderSummary}</small> : <span />}
        </div>
        {hasSubstageRail ? (
          <StageSubstageDetailStrip
            activeLabel={activeSubstage?.label ?? "Strategy path"}
            helpSlot={activeSubstage && activeSubstageHelp ? <InfoHint content={activeSubstageHelp} label="Open subprocess help" /> : null}
            substage={activeSubstage}
            expandedPrimaryViews={workspacePrimaryViews}
            expandedSecondaryViews={workspaceSecondaryViews}
            showAllViews={showAllStageViews}
            activeTab={activeTab}
            onToggleShowAllViews={() => setShowAllStageViews((current) => !current)}
            showAllArtifacts={showAllSubstageArtifacts}
            onToggleShowAllArtifacts={() => setShowAllSubstageArtifacts((current) => !current)}
            onSelectView={(id) => setActiveTab(id as StageViewTabId)}
            onFocusArtifact={focusArtifactForTuning}
            renderRelevanceBadge={(relevance, reason) => <ArtifactRelevanceBadge relevance={relevance} reason={reason} />}
            renderIoRoleTag={(ioRole) => <IoRoleTag ioRole={ioRole} />}
          />
        ) : hasSubstagePlan ? (
          <section className="studio-stage-path-card">
            <div className="stage-path-header">
              <strong>Stage path</strong>
              {activeSubstage && activeSubstageHelp ? <InfoHint content={activeSubstageHelp} label="Open subprocess help" /> : null}
            </div>
            <StageSubstageTree
              substages={stageSubstagePlan.substages}
              activeScope={activeReferenceViewScope}
              selectedSubstageId={selectedSubstageId}
              activeTab={activeTab}
              expandedPrimaryViews={workspacePrimaryViews}
              expandedSecondaryViews={workspaceSecondaryViews}
              showAllViews={showAllStageViews}
              onToggleShowAllViews={() => setShowAllStageViews((current) => !current)}
              showAllArtifacts={showAllSubstageArtifacts}
              onToggleShowAllArtifacts={() => setShowAllSubstageArtifacts((current) => !current)}
              onSelectStrategyPath={activateStrategyPath}
              onSelectSubstage={activateReferenceSubstage}
              onSelectView={(id) => setActiveTab(id as StageViewTabId)}
              onFocusArtifact={focusArtifactForTuning}
              relevanceSummary={substageRelevanceSummary}
              renderRelevanceBadge={(relevance, reason) => <ArtifactRelevanceBadge relevance={relevance} reason={reason} />}
              renderIoRoleTag={(ioRole) => <IoRoleTag ioRole={ioRole} />}
            />
          </section>
        ) : (
          <div className="stage-view-tabs-row">
            <nav className="studio-workspace-tabs compact" aria-label="Stage views">
              {visibleStageTabs.map((tab) => (
                <button className={tab.id === activeTab ? "active" : ""} key={tab.id} onClick={() => setActiveTab(tab.id)} type="button">
                  {tab.label}
                  <ArtifactRelevanceBadge relevance={tab.relevance} reason={tab.hiddenReason} />
                </button>
              ))}
            </nav>
          </div>
        )}
        {tabFallbackMessage && <small className="section-note">{tabFallbackMessage}</small>}
        {detectionViewResolutionError && <small className="section-note">{detectionViewResolutionError}</small>}
        {detectFormDiffersFromRun && <small className="section-note">Form settings differ from last run. Click Apply + rerun to update artifacts.</small>}
        </div>

        <section className="studio-workbench">
          <div className={`${(canonicalSelectedStageId === "fit_object_geometry" || canonicalSelectedStageId === "compute_height_metrics") ? "engineering-workbench-grid measurement-workbench-grid" : "engineering-workbench-grid"} ${tuneModeActive ? "is-tuning" : "is-inspecting"}`}>
          <section className="engineering-visual-workspace">
          <div className="viewer-context-header compact">
            <small>
              Primary: {focusedArtifact?.artifact_id ?? activeDisplayItem?.primaryArtifactId ?? "No artifact selected"}
              {(activeDisplayItem?.referenceArtifactId || binaryMaskReferenceResolution?.sourceArtifact?.artifact_id) ? ` · Reference: ${activeDisplayItem?.referenceArtifactId ?? binaryMaskReferenceResolution?.sourceArtifact?.artifact_id}` : ""}
              {activeViewHelp ? <InfoHint content={activeViewHelp} label="Open artifact help" /> : null}
            </small>
          </div>
          {showSelectedSurfaceRefinement && (
            <details className="control-panel-details">
              <summary>Selected cluster → surface refinement details</summary>
              <SelectedSurfaceRefinementCard artifacts={allArtifacts} detail={workspaceDetail} />
            </details>
          )}
          {renderStageSemanticView({
            detail: workspaceDetail,
            compatible,
            processed,
            stageId: canonicalSelectedStageId,
            view: activeProcessingUnitView
              ? {
                id: activeProcessingUnitView.viewId,
                label: activeProcessingUnitView.label,
                rendererType: activeProcessingUnitView.rendererType,
                emptyState: activeProcessingUnitView.emptyState ? { title: activeProcessingUnitView.emptyState } : undefined,
              }
              : stageSemantic.views.find((item) => item.id === activeTab) ?? stageSemantic.views[0],
            selectedArtifactId,
            onSelectArtifact: setSelectedArtifactId,
            selectedObjectId,
            hoveredObjectId,
            onSelectObject: setSelectedObjectId,
            onHoverObject: setHoveredObjectId,
            hoveredArtifactId,
            onHoverArtifact: setHoveredArtifactId,
            onOverlayDebug: setOverlayDebug,
            onHoverSample: setHoverSample,
            roi: activeRoi,
            roiPolygon: activeRoiPolygon,
            roiType,
            roiEnabled,
            onApplyRoi: (roi, rerun) => void applyRoi(roi, rerun),
            onApplyPolygonRoi: (points, rerun) => void applyPolygonRoi(points, rerun),
            onClearRoi: () => void clearRoi(),
            roiEditModeActive,
            roiEditStatus,
            onCancelRoiEdit: () => cancelVisualRoiEdit(),
            blobParams: blobStep?.params ?? null,
            onApplyBlobParams: (params, rerun) => void applyBlobParams(params, rerun),
            onUpsertObjectAnnotation: (payload) => void upsertObjectAnnotation(payload),
            debugMode: studioMode === "debug" ? "engineering" : "operator",
            viewState,
            onViewportModeChange: (mode) => setViewState((current) => ({ ...current, viewportMode: mode })),
            onOverlayModeChange: (mode) => setViewState((current) => ({ ...current, overlayMode: mode })),
            onBinaryMaskModeChange: (mode) => setViewState((current) => ({ ...current, binaryMaskMode: mode })),
            onBinaryMaskReferenceArtifactIdChange: (artifactId) => setViewState((current) => ({ ...current, binaryMaskReferenceArtifactId: artifactId })),
            onBinaryMaskOpacityChange: (value) => setViewState((current) => ({ ...current, binaryMaskOpacity: value })),
            onBinaryMaskReferenceOpacityChange: (value) => setViewState((current) => ({ ...current, binaryMaskReferenceOpacity: value })),
            onResetViewState: () => setViewState((current) => ({
              ...current,
              ...selectionDrivenViewState,
            })),
            activeBlobClusterId,
            onActiveBlobClusterChange: setActiveBlobClusterId,
            onFocusLineageStep: focusSelectedSupportLineageStep,
            resolvedViewArtifacts: activeProcessingUnitView?.resolvedArtifacts,
            unitViewResolution: activeProcessingUnitView,
            artifactRelevance: artifactRelevanceById,
          })}
          <footer className="studio-workspace-footer">
            {canonicalSelectedStageId === "detect_belt_plane" && activeDisplayItem && (
              <span className={`footer-chip ${activeDisplayItem.category !== "active" ? "warning" : ""}`}>
                View: {activeDisplayItem.label}
                {" · "}Primary: {activeDisplayItem.primaryArtifactId ?? "missing"}
                {" · "}Reference: {activeDisplayItem.referenceArtifactId ?? "none"}
                {" · "}Strategy: {detectStageSummary?.strategyLabel ?? "unknown"}
                {" · "}Role: {activeDisplayItem.role}
                {" · "}Category: {activeDisplayItem.category}
                {" · "}Run: {detectReferenceRunState?.processedAt ?? "unverified"}
              </span>
            )}
            <span className="footer-chip">Artifact: {focusedArtifact?.artifact_id ?? "none"}</span>
            <span className="footer-chip">Stage: {canonicalSelectedStageId}</span>
            {activeSubstage && <span className="footer-chip">Subprocess: {activeSubstage.label}</span>}
            <span className="footer-chip">Status: {currentStageSummary.status}</span>
            <span className="footer-chip">Warnings: {currentStageSummary.warnings.length}</span>
            <span className="footer-chip">Object: {focusedObject ? `#${focusedObject.object_id}` : "None"}</span>
            <span className="footer-chip">Strategy: {detectStageSummary?.strategyLabel ?? stageSemantic.category}</span>
            {detectStageSummary?.selectedSupportRatio != null && (
              <span className="footer-chip">Support: {(detectStageSummary.selectedSupportRatio * 100).toFixed(1)}%</span>
            )}
            {detectStageSummary?.selectedClusterRatio != null && (
              <span className="footer-chip">Cluster: {(detectStageSummary.selectedClusterRatio * 100).toFixed(1)}%</span>
            )}
          </footer>
          {!sourceContextDismissed && detectSourceContextBanner ? (
            <div className={`studio-source-context-banner below-artifact ${detectSourceContextBanner.unresolved ? "warning" : ""}`}>
              <span>Source context: {detectSourceContextBanner.label}{detectSourceContextBanner.unresolved ? " (unresolved)" : ""}</span>
              <div className="studio-source-context-actions">
                {detectSourceContextBanner.analyticsHref ? <a href={detectSourceContextBanner.analyticsHref}>Open analytics</a> : null}
                <button type="button" onClick={() => clearStudioSourceContext()}>Clear context</button>
              </div>
            </div>
          ) : null}
          </section>
          </div>
        </section>
        </section>
      </section>

      <aside className={`studio-inspector-shell ${inspectorOpen ? "expanded" : "collapsed"} ${panelWidthClass} ${rightPanelOverlay ? "overlay" : ""} ${inspectorVisible ? "is-visible" : "is-hidden"}`}>
        <div className="studio-inspector-shell-header">
          <div className="studio-sidebar-title">
            <span className="eyebrow">Stage controls</span>
            <strong>{inspectorOpen ? (focusedArtifact?.title ?? selectedStage?.display_name ?? "Context") : "Controls"}</strong>
          </div>
          {inspectorCanCollapse && (
          <button
            type="button"
            className="panel-collapse-toggle"
            onClick={() => setInspectorExpanded((current) => !current)}
            aria-expanded={inspectorOpen}
            aria-label={inspectorOpen ? "Collapse controls panel" : "Expand controls panel"}
            title={inspectorOpen ? "Collapse controls panel" : "Expand controls panel"}
          >
            {inspectorOpen ? "›" : "‹"}
          </button>
          )}
        </div>
        <div className={`stage-control-tabs ${inspectorOpen ? "expanded" : "collapsed"}`}>
          {(["inspect", "tune", "view", "graph", "compare", "actions", "debug"] as ControlPanelTab[])
            .filter((tab) => shouldShowWorkspaceTab(tab, hasSubstagePlan))
            .map((tab) => (
            <button
              key={tab}
              type="button"
              className={controlPanelTab === tab ? "active" : ""}
              onClick={() => {
                setControlPanelTab(tab);
                setStudioMode(controlPanelTabToMode(tab));
                if (tab !== "inspect") setInspectorExpanded(true);
              }}
            >
              {workspaceTabLabel(tab, hasSubstagePlan)}
              {tab === "tune" && hasTuneChanges ? <small>dirty</small> : null}
            </button>
          ))}
        </div>
        {hasSubstagePlan ? (
          <div className="stage-control-secondary-tabs">
            <button type="button" onClick={() => { setControlPanelTab("graph"); setInspectorExpanded(true); }}>Graph</button>
            <button type="button" onClick={() => { setControlPanelTab("compare"); setStudioMode("compare"); setInspectorExpanded(true); }}>Compare</button>
            <button type="button" onClick={() => { setControlPanelTab("actions"); setStudioMode("inspect"); setInspectorExpanded(true); }}>Actions</button>
          </div>
        ) : null}
        {!inspectorOpen ? (
          <div className="studio-inspector-summary">
            {compactInspectorItems.map((item) => (
              <div key={item.label} className="studio-inspector-summary-card">
                <span>{item.label}</span>
                <strong>{item.value}</strong>
              </div>
            ))}
          </div>
        ) : controlPanelTab === "inspect" ? (
          <div className="stage-control-panel-body">
            <div className="studio-inspector-summary">
              {compactInspectorItems.map((item) => (
                <div key={item.label} className="studio-inspector-summary-card">
                  <span>{item.label}</span>
                  <strong>{item.value}</strong>
                </div>
              ))}
              {detectStageSummary?.selectedSupportRatio != null && (
                <div className="studio-inspector-summary-card">
                  <span>Support</span>
                  <strong>{(detectStageSummary.selectedSupportRatio * 100).toFixed(1)}%</strong>
                </div>
              )}
            </div>
            {canonicalSelectedStageId === "detect_belt_plane" && (
              <section className="control-panel-section">
                <span>Decision</span>
                <strong>{activeSubstage?.label ?? "Stage path"}{" -> "}{selectedView?.label ?? activeTab}</strong>
                <small>{detectStageSummary?.strategyLabel ?? "Unknown strategy"}</small>
                {detectStageSummary?.selectedSupportRatio != null ? <small>Selected support: {ratioToPercentLabel(detectStageSummary.selectedSupportRatio) ?? "-"} valid ROI</small> : null}
                {detectStageSummary?.largestSupportLossFraction != null ? <small>Largest loss: {(detectStageSummary.largestSupportLossStep ?? "unknown").replace(/_/g, " ")} · {ratioToPercentLabel(detectStageSummary.largestSupportLossFraction) ?? "-"}</small> : null}
                {currentStageSummary.warnings.length ? <small>Warnings: {currentStageSummary.warnings.join(" | ")}</small> : <small>No stage warnings for the current selection.</small>}
                {detectReferenceRunState && detectStageSummary ? (
                  <>
                    <button type="button" className="stage-diagnostics-toggle" onClick={() => setDiagnosticsExpanded((current) => !current)}>
                      {diagnosticsExpanded ? "Hide reference support details" : "Reference support details"}
                    </button>
                    {diagnosticsExpanded ? (
                      <div className="stage-diagnostics-details">
                        {detectStageSummary.componentLabel ? <small>Components: {detectStageSummary.componentLabel}</small> : null}
                        {detectStageSummary.splitLabel ? <small>Split: {detectStageSummary.splitLabel}</small> : null}
                        <small>Reference model: {detectStageSummary.referenceModelLabel}</small>
                        <small>Suppression: {detectStageSummary.suppressionLabel}</small>
                        {detectStageSummary.stripeFilterEnabled != null ? <small>Stripe filter: {detectStageSummary.stripeFilterEnabled ? "Enabled" : "Disabled"}</small> : null}
                        <small>Fallback: {detectReferenceRunState.fallbackUsed ? "yes" : "no"}</small>
                        {detectStageSummary.selectedClusterRatio != null ? <small>Selected cluster: {ratioToPercentLabel(detectStageSummary.selectedClusterRatio) ?? "-"} valid ROI</small> : null}
                        {detectStageSummary.referenceModelSupportRatio != null ? <small>Model support: {ratioToPercentLabel(detectStageSummary.referenceModelSupportRatio) ?? "-"} valid ROI</small> : null}
                        {detectStageSummary.referenceSuppressionRatio != null ? <small>Suppression mask: {ratioToPercentLabel(detectStageSummary.referenceSuppressionRatio) ?? "-"} valid ROI</small> : null}
                        {deepLinkMessage ? <small>{deepLinkMessage}</small> : null}
                      </div>
                    ) : null}
                  </>
                ) : null}
              </section>
            )}
            {activeDisplayItem && (
              <section className={`control-panel-section ${activeDisplayItem.category !== "active" ? "warning" : ""}`}>
                <span>Display plan</span>
                <small>Process: {stageSubstagePlan.plan?.processLabel ?? prettyStageLabel(selectedStage?.id ?? canonicalSelectedStageId, selectedStage?.display_name)}</small>
                <small>Subprocess: {activeSubstage?.label ?? "Strategy path"}</small>
                <strong>{activeDisplayItem.label}</strong>
                <small>Primary: {activeDisplayItem.primaryArtifactId ?? "missing"}</small>
                <small>Reference: {activeDisplayItem.referenceArtifactId ?? "none"}</small>
                <small>Role: {displayRoleLabel(activeDisplayItem.role)}</small>
                <small>Category: {activeDisplayItem.category}</small>
                <small>Used by: {activeSubstage?.label ?? "Strategy path"}</small>
                <small>{activeDisplayItem.reason}</small>
              </section>
            )}
            {selectedArtifactLineage.length > 0 && (
              <section className="control-panel-section">
                <span>Lineage</span>
                <small>{selectedArtifactLineage.join(" → ")}</small>
              </section>
            )}
            {detectFormDiffersFromRun && (
              <section className="control-panel-section warning">
                <span>Run mismatch</span>
                <small>Form differs from selected run. Apply + rerun to update artifacts.</small>
              </section>
            )}
          </div>
        ) : controlPanelTab === "tune" ? (
          <div className="stage-control-panel-body">
            <section className="control-panel-section">
              <span>Tuning state</span>
              <strong>{hasTuneChanges ? "Unsaved changes" : "In sync with selected run"}</strong>
              <small>Selected run: {selectedRunTrace?.status ?? "not run"} · Persisted recipe and form state remain unchanged unless you apply.</small>
            </section>
            {detectFormDiffersFromRun && (
              <section className="control-panel-section warning">
                <span>Run mismatch</span>
                <small>Form differs from selected run. Apply + rerun to update artifacts.</small>
              </section>
            )}
            {!stageHasEditableParams && stageProcessingUnits.length === 0 && (
              <div className="empty-state compact">No editable parameters for this stage.</div>
            )}
            {canonicalSelectedStageId === "segmentation" && (
              <StageParameterPanel
                stage={selectedStage}
                values={segmentationPreviewParams}
                persistedValues={segmentationPersistedParams}
                loading={segmentationPreviewLoading}
                dirty={segmentationPreviewDirty}
                previewLabel={`Components: ${segmentationPreview?.metrics.component_count ?? "-"} • ${segmentationPreview?.diagnostics?.preview ? "Preview" : "Persisted"}`}
                error={segmentationPreviewError}
                onChange={(key, value) => {
                  setSegmentationPreviewParams((current) => current ? { ...current, [key]: value } : current);
                  setSegmentationPreviewDirty(true);
                }}
                onPreview={() => void triggerSegmentationPreview()}
                onApply={() => void applySegmentationParams(false)}
                onApplyRun={() => void applySegmentationParams(true)}
                onReset={() => {
                  setSegmentationPreviewParams(segmentationPersistedParams);
                  setSegmentationPreviewDirty(false);
                }}
              />
            )}
            {canonicalSelectedStageId === "ellipse_fitting" && (
              <StageParameterPanel
                stage={selectedStage}
                values={ellipsePreviewParams}
                persistedValues={ellipsePersistedParams}
                loading={ellipsePreviewLoading}
                dirty={ellipsePreviewDirty}
                previewLabel={`${segmentationPreview?.diagnostics?.preview ? "Preview" : "Persisted"} fit state`}
                error={ellipsePreviewError}
                onChange={(key, value) => {
                  setEllipsePreviewParams((current) => current ? { ...current, [key]: value } : current);
                  setEllipsePreviewDirty(true);
                }}
                onPreview={() => void triggerEllipsePreview()}
                onApply={() => void applyEllipseParams(false)}
                onApplyRun={() => void applyEllipseParams(true)}
                onReset={() => resetEllipseDefaults()}
              />
            )}
            {selectedPipeline?.id === "mining_steel_ball_classification_25d" && planeQaParams && (
              <section className="control-panel-section">
                <span>Recipe</span>
                <label>
                  <span>Selected recipe</span>
                  <select disabled={recipeLoading} value={selectedRecipeId} onChange={(event) => { setSelectedRecipeId(event.target.value); setRecipePanelMessage(null); }}>
                    <option value="">No recipe selected</option>
                    {pipelineRecipes.map((recipe) => (
                      <option key={`${recipe.recipe_id}:${recipe.version}`} value={recipe.recipe_id}>
                        {recipe.name} · v{recipe.version}
                      </option>
                    ))}
                  </select>
                </label>
                {selectedRecipe ? (
                  <>
                    <small>
                      Selected: {selectedRecipe.name} · v{selectedRecipe.version}
                      {selectedRecipe.archived ? " · archived" : ""}
                    </small>
                    <small>
                      Compatibility: {selectedRecipe.compatibility ?? "unknown"}
                      {selectedRecipe.validation_warnings?.length ? ` · ${selectedRecipe.validation_warnings.length} warning(s)` : ""}
                    </small>
                    {selectedRecipe.validation_warnings?.length ? (
                      <details className="control-panel-details">
                        <summary>Recipe warnings</summary>
                        {selectedRecipe.validation_warnings.map((warning) => (
                          <small key={warning}>{warning}</small>
                        ))}
                      </details>
                    ) : null}
                    <small>Dirty vs recipe: {currentRecipeDirty ? "yes — overrides will be applied on run" : "no — matches recipe snapshot"} · Contract: {selectedRecipe.processing_unit_contract_version ?? "unknown"}</small>
                    {selectedRecipe.description ? <small>{selectedRecipe.description}</small> : null}
                  </>
                ) : (
                  <small>No recipe selected. Choose a saved recipe to populate the tune form and persist recipe provenance on rerun.</small>
                )}
                <div className="control-panel-button-row">
                  <button type="button" disabled={recipeLoading || pipelineActionBusy} onClick={() => void saveCurrentRecipeAsNew()}>Save current as recipe</button>
                  <button type="button" disabled={!selectedRecipe || recipeLoading || pipelineActionBusy} onClick={() => void updateSelectedRecipeVersion()}>Save new version</button>
                </div>
                <div className="control-panel-button-row">
                  <button type="button" disabled={!selectedRecipe || recipeLoading} onClick={() => void cloneSelectedRecipe()}>Clone recipe</button>
                  <button type="button" disabled={!selectedRecipe || recipeLoading || !currentRecipeDirty} onClick={() => resetEditsToSelectedRecipe()}>Reset edits to recipe</button>
                </div>
                <div className="control-panel-button-row">
                  <button
                    type="button"
                    disabled={!selectedRecipe}
                    onClick={() => {
                      setComparisonLeftType("current");
                      setComparisonRightType("recipe");
                      setComparisonRightRecipeId(selectedRecipe?.recipe_id ?? "");
                      setControlPanelTab("compare");
                      setStudioMode("compare");
                    }}
                  >
                    Compare current to recipe
                  </button>
                  <button
                    type="button"
                    disabled={!selectedRunTrace}
                    onClick={() => {
                      setComparisonLeftType("run");
                      setComparisonLeftRunId(selectedRunTrace?.id ?? "");
                      setComparisonRightType("current");
                      setControlPanelTab("compare");
                      setStudioMode("compare");
                    }}
                  >
                    Compare run to current
                  </button>
                </div>
                {recipePanelMessage ? <small>{recipePanelMessage}</small> : null}
                {selectedRecipe && currentRecipeDiff.length > 0 ? (
                  <details className="control-panel-details">
                    <summary>Recipe diff</summary>
                    {currentRecipeDiff.map((group) => (
                      <div key={group.unitId} className="artifact-reference">
                        <span>{group.unitLabel}</span>
                        {group.changes.map((change) => (
                          <small key={`${group.unitId}:${change.paramId}`}>
                            {change.paramLabel}: {JSON.stringify(change.recipeValue ?? null)} → {JSON.stringify(change.currentValue ?? null)}
                          </small>
                        ))}
                      </div>
                    ))}
                  </details>
                ) : null}
              </section>
            )}
            {canonicalSelectedStageId === "detect_belt_plane" && planeQaParams && (
              <>
                <DetectReferenceTunePanel
                  activePreset={activeReferencePreset}
                  availableUnits={stageProcessingUnits}
                  dirty={planeQaDirty}
                  formDiffersFromRun={detectFormDiffersFromRun}
                  hasTuneChanges={hasTuneChanges}
                  loading={pipelineActionBusy}
                  onApply={() => applyPlaneQaParams(false)}
                  onApplyPreset={(presetId, rerun) => applyReferenceMethodPreset(presetId, rerun)}
                  onApplyRun={() => applyPlaneQaParams(true)}
                  onChange={(key, value) => {
                    setPlaneQaParams((current) => {
                      if (!current) return current;
                      const next = { ...current, [key]: value };
                      setPlaneQaDirty(!planeQaParamsEqual(next, planeQaPersistedParams));
                      return next;
                    });
                  }}
                  onCopyPresetJson={() => void copyReferenceMethodPresetJson()}
                  onDiscard={() => { setPlaneQaParams(planeQaPersistedParams); setPlaneQaDirty(false); }}
                  onResetFieldToRecipe={(key) => resetRecipeFieldToPersisted("detect_belt_plane", key)}
                  onResetUnitToRecipe={() => resetRecipeUnitToPersisted(tuneFocusedProcessingUnit)}
                  onResetPresetOverrides={() => resetReferenceMethodPresetOverrides()}
                  onStartVisualRoiEdit={() => startVisualRoiEdit()}
                  onUseFullFrame={() => void clearRoi()}
                  persistedPreset={persistedReferencePreset}
                  persistedValues={planeQaPersistedParams}
                  recipeValues={activeRecipeUnitValues}
                  runState={detectReferenceRunState}
                  lastRunValues={activeLastRunStageParams}
                  selectedArtifactId={focusedArtifact?.artifact_id ?? null}
                  selectedSubstage={tuneFocusedSubstage}
                  selectedUnit={tuneFocusedProcessingUnit}
                  selectedViewLabel={selectedView?.label ?? activeTab}
                  stage={selectedStage}
                  strategyLabel={detectStageSummary?.strategyLabel ?? "Unknown"}
                  values={planeQaParams}
                />
              </>
            )}
            {canonicalSelectedStageId === "measurement_diagnostics" && planeQaParams && activeProcessingUnit && activeContractStageValues && (
              <ContractTunePanel
                availableUnits={stageProcessingUnits}
                dirty={planeQaDirty}
                lastRunValues={activeLastRunStageParams}
                loading={pipelineActionBusy}
                onApply={() => applyPlaneQaParams(false)}
                onApplyRun={() => applyPlaneQaParams(true)}
                onChange={(key, value) => {
                  setPlaneQaParams((current) => {
                    if (!current) return current;
                    const next = { ...current, [contractParamFlatKey(activeProcessingUnit.stage_id, key)]: value };
                    setPlaneQaDirty(!planeQaParamsEqual(next, planeQaPersistedParams));
                    return next;
                  });
                }}
                onDiscard={() => { setPlaneQaParams(planeQaPersistedParams); setPlaneQaDirty(false); }}
                onResetFieldToRecipe={(key) => resetRecipeFieldToPersisted(activeProcessingUnit.stage_id, key)}
                onResetUnitToRecipe={() => resetRecipeUnitToPersisted(activeProcessingUnit)}
                persistedValues={activeContractPersistedValues}
                recipeValues={activeRecipeUnitValues}
                selectedUnit={activeProcessingUnit}
                stageLabel={selectedStage?.display_name ?? "Measurement diagnostics"}
                values={activeContractStageValues}
              />
            )}
            {planeQaParams && activeProcessingUnit && activeContractStageValues && stageProcessingUnits.length > 0 && ![
              "detect_belt_plane",
              "load_heightmap",
              "remove_belt_segment_objects",
              "measurement_diagnostics",
            ].includes(canonicalSelectedStageId) && (
              <ContractTunePanel
                availableUnits={stageProcessingUnits}
                dirty={planeQaDirty}
                lastRunValues={activeLastRunStageParams}
                loading={pipelineActionBusy}
                onApply={() => applyPlaneQaParams(false)}
                onApplyRun={() => applyPlaneQaParams(true)}
                onChange={(key, value) => {
                  setPlaneQaParams((current) => {
                    if (!current) return current;
                    const next = { ...current, [contractParamFlatKey(activeProcessingUnit.stage_id, key)]: value };
                    setPlaneQaDirty(!planeQaParamsEqual(next, planeQaPersistedParams));
                    return next;
                  });
                }}
                onDiscard={() => { setPlaneQaParams(planeQaPersistedParams); setPlaneQaDirty(false); }}
                onResetFieldToRecipe={(key) => resetRecipeFieldToPersisted(activeProcessingUnit.stage_id, key)}
                onResetUnitToRecipe={() => resetRecipeUnitToPersisted(activeProcessingUnit)}
                persistedValues={activeContractPersistedValues}
                recipeValues={activeRecipeUnitValues}
                selectedUnit={activeProcessingUnit}
                stageLabel={selectedStage?.display_name ?? canonicalSelectedStageId}
                values={activeContractStageValues}
              />
            )}
            {canonicalSelectedStageId === "load_heightmap" && planeQaParams && (
              <>
                <details className="control-panel-details" open>
                  <summary>Reference surface region</summary>
                  <div className="control-panel-form-grid">
                    <label>Mode
                      <select value={String(planeQaParams.reference_surface_region_mode ?? "none")} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, reference_surface_region_mode: e.target.value, plane_fit_roi_type: e.target.value === "full_height_x_band" ? "full_height_x_band" : e.target.value === "none" ? "rectangle" : e.target.value } : c); setPlaneQaDirty(true); }}>
                        <option value="none">Full frame</option>
                        <option value="full_height_x_band">Vertical band ROI</option>
                        <option value="rectangle">Rectangle</option>
                        <option value="polygon">Polygon</option>
                      </select>
                    </label>
                    <label>X
                      <input type="number" min={0} step={1} value={Number(planeQaParams.plane_fit_roi_x ?? 0)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, plane_fit_roi_x: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                    </label>
                    <label>W
                      <input type="number" min={1} step={1} value={Number(planeQaParams.plane_fit_roi_width ?? 0)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, plane_fit_roi_width: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                    </label>
                    {String(planeQaParams.reference_surface_region_mode ?? "none") !== "full_height_x_band" && (
                      <>
                        <label>Y
                          <input type="number" min={0} step={1} value={Number(planeQaParams.plane_fit_roi_y ?? 0)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, plane_fit_roi_y: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                        </label>
                        <label>H
                          <input type="number" min={1} step={1} value={Number(planeQaParams.plane_fit_roi_height ?? 0)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, plane_fit_roi_height: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                        </label>
                      </>
                    )}
                  </div>
                  <div className="control-panel-button-row">
                    <button type="button" disabled={pipelineActionBusy} onClick={() => startVisualRoiEdit()}>Edit visually</button>
                    <button type="button" disabled={pipelineActionBusy} onClick={() => void clearRoi()}>Use full frame</button>
                  </div>
                </details>
                <div className="control-panel-button-row">
                  <button type="button" disabled={pipelineActionBusy} onClick={() => applyPlaneQaParams(false)}>Apply</button>
                  <button type="button" disabled={pipelineActionBusy} onClick={() => applyPlaneQaParams(true)}>Apply + rerun</button>
                  <button type="button" disabled={pipelineActionBusy || !hasTuneChanges} onClick={() => { setPlaneQaParams(planeQaPersistedParams); setPlaneQaDirty(false); }}>Discard</button>
                </div>
              </>
            )}
            {canonicalSelectedStageId === "remove_belt_segment_objects" && planeQaParams && activeProcessingUnit && activeContractStageValues ? (
              <ContractTunePanel
                availableUnits={stageProcessingUnits}
                dirty={planeQaDirty}
                lastRunValues={activeLastRunStageParams}
                loading={pipelineActionBusy}
                onApply={() => applyPlaneQaParams(false)}
                onApplyRun={() => applyPlaneQaParams(true)}
                onChange={(key, value) => {
                  setPlaneQaParams((current) => {
                    if (!current) return current;
                    const next = { ...current, [contractParamFlatKey(activeProcessingUnit.stage_id, key)]: value };
                    setPlaneQaDirty(!planeQaParamsEqual(next, planeQaPersistedParams));
                    return next;
                  });
                }}
                onDiscard={() => { setPlaneQaParams(planeQaPersistedParams); setPlaneQaDirty(false); }}
                onResetFieldToRecipe={(key) => resetRecipeFieldToPersisted(activeProcessingUnit.stage_id, key)}
                onResetUnitToRecipe={() => resetRecipeUnitToPersisted(activeProcessingUnit)}
                persistedValues={activeContractPersistedValues}
                recipeValues={activeRecipeUnitValues}
                selectedUnit={activeProcessingUnit}
                stageLabel={selectedStage?.display_name ?? "Segmentation"}
                values={activeContractStageValues}
              />
            ) : canonicalSelectedStageId === "remove_belt_segment_objects" && planeQaParams && (
              <>
                <details className="control-panel-details" open>
                  <summary>Segmentation</summary>
                  <div className="control-panel-form-grid">
                    <label>Min height
                      <input type="number" min={0} max={200} step={0.5} value={Number(planeQaParams.object_min_height_mm ?? 8)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, object_min_height_mm: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                    </label>
                    <label>Max height
                      <input type="number" min={0} max={500} step={1} value={planeQaParams.object_max_height_mm == null || !Number.isFinite(Number(planeQaParams.object_max_height_mm)) ? "" : Number(planeQaParams.object_max_height_mm)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, object_max_height_mm: e.target.value === "" ? null : Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                    </label>
                    <label>Morph kernel
                      <input type="number" min={1} max={31} step={2} value={Number(planeQaParams.morphology_kernel ?? 5)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, morphology_kernel: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                    </label>
                    <label>Min component
                      <input type="number" min={0} max={500000} step={10} value={Number(planeQaParams.min_component_area ?? 120)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, min_component_area: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                    </label>
                  </div>
                </details>
                <div className="control-panel-button-row">
                  <button type="button" disabled={pipelineActionBusy} onClick={() => applyPlaneQaParams(false)}>Apply</button>
                  <button type="button" disabled={pipelineActionBusy} onClick={() => applyPlaneQaParams(true)}>Apply + rerun</button>
                  <button type="button" disabled={pipelineActionBusy || !hasTuneChanges} onClick={() => { setPlaneQaParams(planeQaPersistedParams); setPlaneQaDirty(false); }}>Discard</button>
                </div>
              </>
            )}
            {(canonicalSelectedStageId === "fit_object_geometry" || canonicalSelectedStageId === "compute_height_metrics") && (
              <section className="control-panel-section">
                <span>Measurement controls</span>
                <small>Known-object correction and calibration context remain available here.</small>
                <button type="button" onClick={() => setMeasurementConfigOpen(true)}>Configure known-cube</button>
              </section>
            )}
          </div>
        ) : controlPanelTab === "view" ? (
          <div className="stage-control-panel-body">
            <section className="control-panel-section">
              <span>Viewport</span>
              <div className="studio-workspace-tabs">
                <button className={viewState.viewportMode === "fit_all" ? "active" : ""} onClick={() => setViewState((current) => ({ ...current, viewportMode: "fit_all" }))} type="button">Fit all</button>
                <button className={viewState.viewportMode === "fit_width" ? "active" : ""} onClick={() => setViewState((current) => ({ ...current, viewportMode: "fit_width" }))} type="button">Fit width</button>
                <button className={viewState.viewportMode === "fit_height" ? "active" : ""} onClick={() => setViewState((current) => ({ ...current, viewportMode: "fit_height" }))} type="button">Fit height</button>
                <button className={viewState.viewportMode === "natural" ? "active" : ""} onClick={() => setViewState((current) => ({ ...current, viewportMode: "natural" }))} type="button">100%</button>
              </div>
              <small>Reset keeps view-only state local to Studio and does not change stage params.</small>
            </section>
            {overlayControlsVisible && !useBinaryMaskView && (
              <section className="control-panel-section">
                <span>Overlay visibility</span>
                <div className="studio-workspace-tabs">
                  <button className={viewState.overlayMode === "all" ? "active" : ""} onClick={() => setViewState((current) => ({ ...current, overlayMode: "all" }))} type="button">Show all</button>
                  <button className={viewState.overlayMode === "selected_only" ? "active" : ""} onClick={() => setViewState((current) => ({ ...current, overlayMode: "selected_only" }))} type="button">Selected only</button>
                  <button className={viewState.overlayMode === "hide" ? "active" : ""} onClick={() => setViewState((current) => ({ ...current, overlayMode: "hide" }))} type="button">Hide</button>
                </div>
              </section>
            )}
            {activeProcessingUnit && (
              <section className="control-panel-section">
                <span>Selected unit I/O</span>
                <small>Unit: {activeProcessingUnit.label}</small>
                {activeTraceQuality ? <small>Provenance: {activeTraceQuality.source} · {activeTraceQuality.precision}</small> : null}
                <small>Inputs</small>
                <div className="substage-artifact-list">
                  {activeProcessingUnitIO.inputs.map((entry) => (
                    <button
                      key={`input_${entry.id}`}
                      type="button"
                      className={`footer-chip ${entry.present ? "" : "warning"}`}
                      disabled={!entry.artifact}
                      onClick={() => entry.artifact && setSelectedArtifactId(entry.artifact.artifact_id)}
                    >
                      {entry.label}: {entry.artifactId ?? entry.id} {entry.present ? "" : "(missing)"}
                    </button>
                  ))}
                </div>
                <small>Outputs</small>
                <div className="substage-artifact-list">
                  {activeProcessingUnitIO.outputs.map((entry) => (
                    <button
                      key={`output_${entry.id}`}
                      type="button"
                      className={`footer-chip ${entry.present ? "" : "warning"}`}
                      disabled={!entry.artifact}
                      onClick={() => entry.artifact && setSelectedArtifactId(entry.artifact.artifact_id)}
                    >
                      {entry.label}: {entry.artifactId ?? entry.id} {entry.present ? "" : "(missing)"}
                    </button>
                  ))}
                </div>
              </section>
            )}
            {canonicalSelectedStageId === "detect_belt_plane" && (activeDisplayItem || detectActiveViewResolution) && (
              <>
                <section className={`control-panel-section ${(activeDisplayItem?.category ?? detectActiveViewResolution?.category) !== "active" ? "warning" : ""}`}>
                  <span>Inspect semantics</span>
                  <strong>{selectedView?.label ?? activeTab}</strong>
                  <small>View: {activeDisplayItem?.viewId ?? activeTab}</small>
                  <small>Primary: {activeDisplayItem?.primaryArtifactId ?? detectActiveViewResolution?.primary?.artifact_id ?? "missing"}</small>
                  <small>Expected: {activeDisplayItem?.expectedPrimaryArtifactId ?? "-"}</small>
                  <small>Fallback: {activeDisplayItem?.fallbackArtifactId ?? "none"}</small>
                  <small>Reference: {activeDisplayItem?.referenceArtifactId ?? detectActiveViewResolution?.reference?.artifact_id ?? "none"}</small>
                  <small>Strategy: {detectStageSummary?.strategyLabel ?? "Unknown"}</small>
                  <small>Role: {activeDisplayItem ? displayRoleLabel(activeDisplayItem.role) : "-"}</small>
                  <small>Category: {activeDisplayItem?.category ?? detectActiveViewResolution?.category ?? "missing"}</small>
                  <small>Run match: {detectFormDiffersFromRun ? "form mismatch" : "selected run"}</small>
                  <small>Reason: {activeDisplayItem?.reason ?? detectActiveViewResolution?.reason ?? "-"}</small>
                  <small>Contract fallback: {activeProcessingUnitView?.fallbackUsed ? activeProcessingUnitView.fallbackReason ?? "yes" : "not used"}</small>
                </section>
              </>
            )}
            {focusedArtifact && (
              <section className="control-panel-section">
                <span>Selected artifact</span>
                <div className="toolbar-chip-row">
                  <span className="toolbar-chip">{String(focusedArtifact.metadata?.display_label ?? focusedArtifact.title ?? focusedArtifact.artifact_id)}</span>
                  <span className="tab-with-help">
                    <span className="toolbar-chip">{detectArtifactRoleLabel(focusedArtifact.metadata?.role)}</span>
                    <InfoHint
                      content={resolveStudioHelpEntry(detectReferenceArtifactRoleHelpKey(String(focusedArtifact.metadata?.role ?? "")))}
                      label="Open role help"
                    />
                  </span>
                  <span className="toolbar-chip">{detectSubstageLabel(focusedArtifact.metadata?.substage_id)}</span>
                </div>
                <small>ID: {focusedArtifact.artifact_id}</small>
                <small>Kind: {focusedArtifact.kind}</small>
                <small>Semantic type: {String(focusedArtifact.metadata?.semantic_type ?? "-")}</small>
                <small>Order: {String(focusedArtifact.metadata?.order_index ?? "-")}</small>
                <small>Target image: {focusedArtifact.target_artifact_id ?? "-"}</small>
                <small>Source artifact: {String(focusedArtifact.metadata?.source_artifact_id ?? "-")}</small>
                <small>Mask artifact: {String(focusedArtifact.metadata?.mask_artifact_id ?? "-")}</small>
                <small>Authoritative for fit: {focusedArtifact.metadata?.is_authoritative_for_fit ? "yes" : "no"}</small>
                <small>Authoritative for suppression: {focusedArtifact.metadata?.is_authoritative_for_suppression ? "yes" : "no"}</small>
                <small>Fallback reason: {String(focusedArtifact.metadata?.fallback_reason ?? "-")}</small>
                <small>Warnings: {Array.isArray(focusedArtifact.metadata?.warnings) && focusedArtifact.metadata?.warnings.length ? focusedArtifact.metadata?.warnings.join(" | ") : "-"}</small>
              </section>
            )}
          </div>
        ) : controlPanelTab === "graph" ? (
          <div className="stage-control-panel-body">
            <section className="control-panel-section">
              <span>Contract graph</span>
              <small>Read-only pipeline graph generated from processing-unit contracts.</small>
              <small>Spans every stage in the pipeline — for navigating this stage's own steps, use the stage tree above the viewer.</small>
              <small>Click a node to select its stage or processing unit in Processing Lab.</small>
              {graphTraceSummary ? <small>{formatTraceCoverageLine(graphTraceSummary)}</small> : null}
              <div className="studio-workspace-tabs compact">
                <button type="button" onClick={() => setGraphWorkspaceExpanded(true)}>Expand graph workspace</button>
              </div>
            </section>
            <details className="control-panel-details workflow-checklist">
              <summary>Workflow checklist</summary>
              <ol className="workflow-checklist-list">
                <li>Select take</li>
                <li>Select recipe</li>
                <li>Tune unit parameters</li>
                <li>Run with recipe + overrides</li>
                <li>Compare against previous run</li>
                <li>Inspect changed units in graph</li>
                <li>Save recipe version</li>
              </ol>
            </details>
            <ProcessingUnitGraphViewer
              graph={processingUnitGraph}
              selectedNodeId={selectedGraphNodeId}
              onSelectNode={focusProcessingUnitFromGraph}
              traceSummary={graphTraceSummary}
            />
          </div>
        ) : controlPanelTab === "compare" ? (
          <div className="stage-control-panel-body">
            {detail?.result?.parent_run_id ? (
              <section className="control-panel-section">
                <span>Compare to parent</span>
                <small>Parent run: {String(detail.result.parent_run_id)}</small>
                <div className="control-panel-button-row">
                  {detail.result.execution_mode === "rerun_from_public_stage_boundary" ? (
                    (() => {
                      const parentComparisonId = pipelineRuns.find((run) => run.run_id === detail.result?.run_id)?.comparison_id;
                      return parentComparisonId ? (
                        <button type="button" disabled={comparisonBusy} onClick={() => void reopenComparison(parentComparisonId)}>
                          Reopen comparison to parent
                        </button>
                      ) : (
                        <small>No comparison to parent was generated for this run.</small>
                      );
                    })()
                  ) : null}
                </div>
              </section>
            ) : null}
            <section className="control-panel-section">
              <span>Compare</span>
              <small>Contract-aware run and recipe comparison for the 25D pipeline.</small>
              <div className="control-panel-form-grid">
                <label>
                  <span>Left source</span>
                  <select value={comparisonLeftType} onChange={(event) => setComparisonLeftType(event.target.value as CompareSourceType)}>
                    <option value="current">Current edits</option>
                    <option value="recipe">Selected recipe</option>
                    <option value="run">Last run</option>
                  </select>
                </label>
                <label>
                  <span>Right source</span>
                  <select value={comparisonRightType} onChange={(event) => setComparisonRightType(event.target.value as CompareSourceType)}>
                    <option value="recipe">Selected recipe</option>
                    <option value="current">Current edits</option>
                    <option value="run">Last run</option>
                  </select>
                </label>
                {comparisonLeftType === "recipe" ? (
                  <label>
                    <span>Left recipe</span>
                    <select value={comparisonLeftRecipeId} onChange={(event) => setComparisonLeftRecipeId(event.target.value)}>
                      <option value="">Select recipe</option>
                      {pipelineRecipes.map((recipe) => (
                        <option key={`compare-left-${recipe.recipe_id}:${recipe.version}`} value={recipe.recipe_id}>
                          {recipe.name} · v{recipe.version}
                        </option>
                      ))}
                    </select>
                  </label>
                ) : null}
                {comparisonRightType === "recipe" ? (
                  <label>
                    <span>Right recipe</span>
                    <select value={comparisonRightRecipeId} onChange={(event) => setComparisonRightRecipeId(event.target.value)}>
                      <option value="">Select recipe</option>
                      {pipelineRecipes.map((recipe) => (
                        <option key={`compare-right-${recipe.recipe_id}:${recipe.version}`} value={recipe.recipe_id}>
                          {recipe.name} · v{recipe.version}
                        </option>
                      ))}
                    </select>
                  </label>
                ) : null}
                {comparisonLeftType === "run" ? (
                  <label>
                    <span>Left run</span>
                    <select value={comparisonLeftRunId} onChange={(event) => setComparisonLeftRunId(event.target.value)}>
                      <option value="">Select run</option>
                      {comparisonRunOptions.map((run) => (
                        <option key={`compare-left-run-${run.id}`} value={run.id}>
                          {run.id} · {run.status} · {run.timestamp || "no timestamp"}
                        </option>
                      ))}
                    </select>
                  </label>
                ) : null}
                {comparisonRightType === "run" ? (
                  <label>
                    <span>Right run</span>
                    <select value={comparisonRightRunId} onChange={(event) => setComparisonRightRunId(event.target.value)}>
                      <option value="">Select run</option>
                      {comparisonRunOptions.map((run) => (
                        <option key={`compare-right-run-${run.id}`} value={run.id}>
                          {run.id} · {run.status} · {run.timestamp || "no timestamp"}
                        </option>
                      ))}
                    </select>
                  </label>
                ) : null}
              </div>
              <small>
                Left: {comparisonLeftType === "current" ? "current edited form" : comparisonLeftType === "recipe" ? (pipelineRecipes.find((item) => item.recipe_id === comparisonLeftRecipeId)?.name ?? "select recipe") : (comparisonLeftRunId || "select run")}
              </small>
              <small>
                Right: {comparisonRightType === "current" ? "current edited form" : comparisonRightType === "recipe" ? (pipelineRecipes.find((item) => item.recipe_id === comparisonRightRecipeId)?.name ?? "select recipe") : (comparisonRightRunId || "select run")}
              </small>
              <div className="control-panel-button-row">
                <button type="button" disabled={comparisonBusy || selectedPipeline?.id !== "mining_steel_ball_classification_25d"} onClick={() => void runComparison()}>
                  Compare now
                </button>
                <button
                  type="button"
                  disabled={!selectedRecipe}
                  onClick={() => {
                    setComparisonLeftType("current");
                    setComparisonRightType("recipe");
                    setComparisonRightRecipeId(selectedRecipe?.recipe_id ?? "");
                    setControlPanelTab("compare");
                    setStudioMode("compare");
                  }}
                >
                  Compare current to recipe
                </button>
              </div>
              {comparisonError ? <small>{comparisonError}</small> : null}
            </section>
            <section className="control-panel-section">
              <span>Batch comparison</span>
              <small>
                Select takes in the left rail's Batch panel, then use "Compare selected" there — it reuses the left/right
                sources configured above, but resolves the right side's run independently for each selected take.
              </small>
              {batchComparisonBusy ? <small>Comparing selected takes…</small> : null}
              {batchComparisonError ? <small>{batchComparisonError}</small> : null}
              {batchComparisonResult && !batchComparisonBusy && (
                <>
                  <div className="studio-inspector-summary">
                    <div className="studio-inspector-summary-card">
                      <span>Takes compared</span>
                      <strong>
                        {batchComparisonResult.compared_take_count} / {batchComparisonResult.requested_take_count}
                      </strong>
                    </div>
                    <div className="studio-inspector-summary-card">
                      <span>Classification changed</span>
                      <strong>
                        {batchComparisonResult.aggregate.classification_changed_count} ({formatRatio(batchComparisonResult.aggregate.classification_changed_fraction ?? null, 2)})
                      </strong>
                    </div>
                    <div className="studio-inspector-summary-card">
                      <span>Takes with mask changes</span>
                      <strong>
                        {batchComparisonResult.aggregate.takes_with_mask_changes_count} ({formatRatio(batchComparisonResult.aggregate.takes_with_mask_changes_fraction ?? null, 2)})
                      </strong>
                    </div>
                    <div className="studio-inspector-summary-card">
                      <span>Mask IoU (min / avg / max)</span>
                      <strong>
                        {formatRatio(batchComparisonResult.aggregate.min_iou ?? null)} / {formatRatio(batchComparisonResult.aggregate.average_iou ?? null)} / {formatRatio(batchComparisonResult.aggregate.max_iou ?? null)}
                      </strong>
                    </div>
                  </div>
                  {batchComparisonResult.aggregate.most_changed_units.length > 0 ? (
                    <div className="artifact-reference">
                      <span>Most-changed units across selected takes</span>
                      {batchComparisonResult.aggregate.most_changed_units.slice(0, 5).map((row) => (
                        <small key={`batch-unit:${row.unit_id}`}>
                          {row.unit_id}: {row.changed_take_count} take(s) ({formatRatio(row.changed_take_fraction, 2)})
                        </small>
                      ))}
                    </div>
                  ) : null}
                  {batchComparisonResult.errors.length > 0 ? (
                    <div className="artifact-reference">
                      <span>Could not compare ({batchComparisonResult.errors.length})</span>
                      {batchComparisonResult.errors.map((row) => (
                        <small key={`batch-error:${row.take_id}`}>
                          {row.take_id}: {row.error}
                        </small>
                      ))}
                    </div>
                  ) : null}
                  <div className="comparison-history-list">
                    {batchComparisonResult.per_take.map((row) => (
                      <div key={`batch-take:${row.take_id}`} className="comparison-history-item">
                        <button type="button" className="link-button" onClick={() => setSelectedTakeId(row.take_id)}>
                          {row.take_id}
                        </button>
                        <small>
                          {row.summary.classification_changed ? "classification changed" : "classification unchanged"}
                          {row.summary.average_iou != null ? ` · IoU ${formatRatio(row.summary.average_iou)}` : ""}
                        </small>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </section>
            <section className="control-panel-section">
              <span>Saved comparisons</span>
              {comparisonHistoryLoading ? <small>Loading comparison history…</small> : null}
              {!comparisonHistoryLoading && comparisonHistory.length === 0 ? (
                <small>No saved comparisons for this pipeline{detail?.take_id ? ` and take ${detail.take_id}` : ""} yet.</small>
              ) : null}
              {comparisonHistory.length > 0 ? (
                <div className="comparison-history-list">
                  {comparisonHistory.map((entry) => (
                    <div key={entry.comparison_id} className="comparison-history-item">
                      <strong>{entry.comparison_id}</strong>
                      <small>{entry.created_at ? new Date(entry.created_at).toLocaleString() : "unknown time"}</small>
                      <small>
                        {comparisonSourceLabel(entry.left, comparisonSourceLabelContext)} vs {comparisonSourceLabel(entry.right, comparisonSourceLabelContext)}
                      </small>
                      <small>{formatChangeSummary(entry.summary?.what_changed_most ?? null) ?? "Open to inspect details."}</small>
                      <button type="button" disabled={comparisonBusy} onClick={() => void reopenComparison(entry.comparison_id)}>Reopen</button>
                    </div>
                  ))}
                </div>
              ) : null}
            </section>
            {!comparisonResult && !comparisonBusy ? (
              <div className="empty-state compact">Run a comparison or reopen a saved one to inspect unit, artifact, and result diffs.</div>
            ) : null}
            {comparisonResult && (
              <>
                <section className="control-panel-section">
                  <span>What changed most?</span>
                  <div className="what-changed-most">
                    <small>{formatComparisonSummaryLine(comparisonResult) ?? "No dominant changes detected."}</small>
                    {comparisonResult.summary.what_changed_most?.top_changed_unit?.parameter_changes?.length ? (
                      <div className="artifact-reference">
                        <span>Parameter changes in {comparisonResult.summary.what_changed_most.top_changed_unit.label}</span>
                        {comparisonResult.summary.what_changed_most.top_changed_unit.parameter_changes.map((row) => (
                          <small key={`top-param:${row.parameter_id}`}>
                            {row.label}: {formatComparisonValue(row.left)} → {formatComparisonValue(row.right)}
                          </small>
                        ))}
                      </div>
                    ) : null}
                  </div>
                </section>
                <section className="control-panel-section">
                  <span>Summary</span>
                  <div className="studio-inspector-summary">
                    <div className="studio-inspector-summary-card">
                      <span>Parameters changed</span>
                      <strong>{comparisonResult.summary.parameter_changes}</strong>
                    </div>
                    <div className="studio-inspector-summary-card">
                      <span>Artifacts changed</span>
                      <strong>{comparisonResult.summary.artifact_changes}</strong>
                    </div>
                    <div className="studio-inspector-summary-card">
                      <span>Metrics changed</span>
                      <strong>{comparisonResult.summary.metric_changes}</strong>
                    </div>
                    <div className="studio-inspector-summary-card">
                      <span>Masks compared</span>
                      <strong>{comparisonResult.summary.mask_artifacts_compared ?? 0}</strong>
                    </div>
                    <div className="studio-inspector-summary-card">
                      <span>Masks changed</span>
                      <strong>{comparisonResult.summary.mask_artifacts_changed ?? 0}</strong>
                    </div>
                    <div className="studio-inspector-summary-card">
                      <span>Classification</span>
                      <strong>{comparisonResult.summary.classification_changed ? "changed" : "unchanged"}</strong>
                    </div>
                    <div className="studio-inspector-summary-card">
                      <span>Objects</span>
                      <strong>{comparisonResult.summary.left_object_count ?? "-"} → {comparisonResult.summary.right_object_count ?? "-"}</strong>
                    </div>
                    <div className="studio-inspector-summary-card">
                      <span>Status</span>
                      <strong>{comparisonResult.summary.left_status ?? "-"} → {comparisonResult.summary.right_status ?? "-"}</strong>
                    </div>
                    <div className="studio-inspector-summary-card">
                      <span>Average IoU</span>
                      <strong>{formatRatio(comparisonResult.summary.average_iou ?? null)}</strong>
                    </div>
                  </div>
                  <small>{comparisonResult.left.label ?? comparisonResult.left.type} vs {comparisonResult.right.label ?? comparisonResult.right.type}</small>
                  {comparisonResult.summary.largest_changed_artifact ? (
                    <small>
                      Largest mask change: {comparisonResult.summary.largest_changed_artifact}
                      {comparisonResult.summary.largest_changed_unit ? ` · ${comparisonResult.summary.largest_changed_unit}` : ""}
                    </small>
                  ) : null}
                  {comparisonResult.warnings?.length ? <small>Warnings: {comparisonResult.warnings.join(" | ")}</small> : null}
                </section>
                {orderedComparisonUnitEntries.map(([unitId, unit]) => (
                  <details key={unitId} className={`control-panel-details ${unit.has_changes ? "" : "compact"}`} open={unit.has_changes}>
                    <summary>{unit.label}{unit.has_changes ? " · changed" : " · unchanged"}</summary>
                    {unit.parameter_diff.length > 0 ? (
                      <div className="artifact-reference">
                        <span>Parameters</span>
                        {unit.parameter_diff.map((row) => (
                          <small key={`${unitId}:param:${row.parameter_id}`}>
                            {row.label}: {formatComparisonValue(row.left)} → {formatComparisonValue(row.right)} ({row.status})
                          </small>
                        ))}
                      </div>
                    ) : null}
                    {unit.metric_diff.length > 0 ? (
                      <div className="artifact-reference">
                        <span>Metrics</span>
                        {unit.metric_diff.map((row) => (
                          <small key={`${unitId}:metric:${row.id ?? row.label}`}>
                            {row.label}: {formatComparisonValue(row.left)} → {formatComparisonValue(row.right)} ({row.status})
                          </small>
                        ))}
                      </div>
                    ) : null}
                    {unit.diagnostic_diff.length > 0 ? (
                      <div className="artifact-reference">
                        <span>Diagnostics</span>
                        {unit.diagnostic_diff.map((row) => (
                          <small key={`${unitId}:diag:${row.id ?? row.label}`}>
                            {row.label}: {formatComparisonValue(row.left)} → {formatComparisonValue(row.right)} ({row.status})
                          </small>
                        ))}
                      </div>
                    ) : null}
                    {unit.artifact_diff.length > 0 ? (
                      <div className="artifact-reference">
                        <span>Artifacts</span>
                        {unit.artifact_diff.map((row) => (
                          <div key={`${unitId}:artifact:${row.artifact_id}`} className="artifact-reference">
                            <span>{row.label}</span>
                            <small>
                              {row.left_present ? "left" : "missing"} / {row.right_present ? "right" : "missing"}
                              {row.pixel_diff
                                ? ` · Changed ${formatPercent(row.pixel_diff.changed_percent)} · IoU ${formatRatio(row.pixel_diff.iou, 2)} · +${row.pixel_diff.added_pixels} px · -${row.pixel_diff.removed_pixels} px`
                                : row.diff_reason
                                  ? ` · ${row.diff_reason}`
                                  : row.diff_available
                                    ? ""
                                    : " · pixel diff not computed"}
                            </small>
                            {row.left_dimensions || row.right_dimensions ? (
                              <small>
                                Dimensions: {row.left_dimensions ? `${row.left_dimensions.width}×${row.left_dimensions.height}` : "-"} / {row.right_dimensions ? `${row.right_dimensions.width}×${row.right_dimensions.height}` : "-"}
                              </small>
                            ) : null}
                            {(row.left_url || row.right_url || row.diff_artifacts?.added_mask?.url || row.diff_artifacts?.removed_mask?.url || row.diff_artifacts?.changed_mask?.url) ? (
                              <div className="toolbar-chip-row">
                                {row.left_url ? <a className="control-link-button" href={row.left_url} target="_blank" rel="noreferrer">Left</a> : <span className="toolbar-chip">Left missing</span>}
                                {row.right_url ? <a className="control-link-button" href={row.right_url} target="_blank" rel="noreferrer">Right</a> : <span className="toolbar-chip">Right missing</span>}
                                {row.diff_artifacts?.added_mask?.url ? <a className="control-link-button" href={row.diff_artifacts.added_mask.url} target="_blank" rel="noreferrer">Added</a> : null}
                                {row.diff_artifacts?.removed_mask?.url ? <a className="control-link-button" href={row.diff_artifacts.removed_mask.url} target="_blank" rel="noreferrer">Removed</a> : null}
                                {row.diff_artifacts?.changed_mask?.url ? <a className="control-link-button" href={row.diff_artifacts.changed_mask.url} target="_blank" rel="noreferrer">Changed</a> : null}
                                {row.diff_artifacts?.overlay?.url ? <a className="control-link-button" href={row.diff_artifacts.overlay.url} target="_blank" rel="noreferrer">Overlay</a> : null}
                              </div>
                            ) : null}
                            {(row.left_url || row.right_url || row.diff_artifacts?.overlay?.url) ? (
                              <div className="toolbar-chip-row">
                                {row.left_url ? <img alt={`${row.label} left`} src={row.left_url} style={{ width: 72, height: 72, objectFit: "cover" }} /> : null}
                                {row.right_url ? <img alt={`${row.label} right`} src={row.right_url} style={{ width: 72, height: 72, objectFit: "cover" }} /> : null}
                                {row.diff_artifacts?.overlay?.url ? <img alt={`${row.label} diff overlay`} src={row.diff_artifacts.overlay.url} style={{ width: 72, height: 72, objectFit: "cover" }} /> : null}
                              </div>
                            ) : null}
                          </div>
                        ))}
                      </div>
                    ) : null}
                  </details>
                ))}
              </>
            )}
          </div>
        ) : controlPanelTab === "actions" ? (
          <div className="stage-control-panel-body">
            <section className="control-panel-section">
              <span>Runs</span>
              <PartialRunNoticeBanner
                executionMode={detail?.result?.execution_mode as string | null | undefined}
                parentRunId={detail?.result?.parent_run_id as string | null | undefined}
                boundaryStageId={(detail?.result?.recipe_snapshot as Record<string, unknown> | undefined)?.boundary_stage_id as string | null | undefined}
              />
              {detail?.result?.partial_execution_summary ? (
                <small className="partial-execution-summary-line">
                  {formatPartialExecutionSummaryLine(detail.result.partial_execution_summary)}
                </small>
              ) : null}
              <RunHistoryPanel
                runs={pipelineRuns}
                activeRunId={activeRunId}
                loading={pipelineRunsLoading}
                onSelectRun={(runId) => setActiveRunId(runId)}
              />
              {selectedPipeline?.id && detail?.result?.run_id ? (
                <RunLineageBlock
                  pipelineId={selectedPipeline.id}
                  runId={String(detail.result.run_id)}
                  onSelectRun={(runId) => setActiveRunId(runId)}
                  onOpenComparison={(comparisonId) => { void reopenComparison(comparisonId); setControlPanelTab("compare"); }}
                  onLineageRefreshed={refreshPipelineRuns}
                />
              ) : null}
            </section>
            <section className="control-panel-section">
              <span>Pipeline</span>
              <small>{runActionSummary}</small>
              <div className="control-panel-button-row">
                <button disabled={!compatible || pipelineActionBusy} onClick={() => void runSelectedPipeline(false)} type="button">Run full pipeline</button>
                <button disabled={!detail || !compatible || pipelineActionBusy} onClick={() => void runSelectedPipeline(true)} type="button">Reprocess</button>
                <button disabled={!detail || pipelineActionBusy} onClick={() => void clearSelectedOutputs()} type="button">Clear outputs</button>
                <button type="button" onClick={() => setJobsPopoverOpen(true)}>Open jobs</button>
              </div>
            </section>
            <section className="control-panel-section">
              <span>Runtime worker health</span>
              <small>RGB worker: {workerBadgeLabel(rgbWorkerStatus)} · 25D worker: {workerBadgeLabel(worker25dStatus)} · Fusion worker: {workerBadgeLabel(fusionWorkerStatus)}</small>
            </section>
            <section className="control-panel-section">
              <span>Active binding visibility</span>
              {activeBinding ? (
                <>
                  <small>Binding: {activeBinding.id}</small>
                  <small>active_recipe_version_id: {activeBinding.active_recipe_version_id ?? "none"}</small>
                </>
              ) : (
                <small>No active binding found for the selected take/modality context.</small>
              )}
            </section>
            <section className="control-panel-section">
              <span>Latest published result</span>
              <small>{latestPublishedResult?.id ?? "No published result linked to this take yet."}</small>
            </section>
            {(canonicalSelectedStageId === "detect_belt_plane" || canonicalSelectedStageId === "load_heightmap") && (
              <section className="control-panel-section">
                <span>Reference surface actions</span>
                <div className="control-panel-button-row">
                  <button type="button" disabled={pipelineActionBusy} onClick={() => startVisualRoiEdit()}>Apply ROI</button>
                  <button type="button" disabled={pipelineActionBusy} onClick={() => applyPlaneQaParams(true)}>Apply ROI + rerun</button>
                  <button type="button" disabled={pipelineActionBusy} onClick={() => void clearRoi()}>Reset ROI</button>
                </div>
              </section>
            )}
            <section className="control-panel-section">
              <span>Artifact</span>
              {artifactBaseline?.status === "active" ? <small>REFERENCE · v{artifactBaseline.version}</small> : null}
              {validationMessage ? <small>{validationMessage}</small> : null}
              <div className="control-panel-button-row">
                {detail?.take_id && focusedArtifact?.path ? (
                  <a className="control-link-button" href={fileUrl(detail.take_id, String(focusedArtifact.path))} target="_blank" rel="noreferrer">Open selected artifact</a>
                ) : null}
                <button type="button" disabled={!focusedArtifact?.artifact_id} onClick={() => { void navigator.clipboard?.writeText(String(focusedArtifact?.artifact_id ?? "")); }}>Copy artifact id</button>
                <button type="button" disabled={!detail?.take_id || !focusedArtifact?.artifact_id} onClick={() => {
                  if (!detail?.take_id || !focusedArtifact?.artifact_id) return;
                  setValidationMessage("Approving reference…");
                  void api.approveValidationBaseline({ take_id: detail.take_id, pipeline_id: selectedPipelineId, run_id: activeRunId ?? (detail.result?.run_id as string | undefined) ?? "latest", approval_scope: "artifact", stage_id: canonicalSelectedStageId, processing_unit_id: selectedContractUnitId, view_id: activeTab, artifact_id: focusedArtifact.artifact_id }).then((result) => {
                    const baseline = result.baselines[0] ?? null; setArtifactBaseline(baseline); setValidationMessage(baseline ? `Reference v${baseline.version} approved.` : "Artifact is not comparison-eligible.");
                  }).catch((error: unknown) => setValidationMessage(error instanceof Error ? error.message : "Reference approval failed."));
                }}>Mark as reference</button>
                <button type="button" disabled={!artifactBaseline} onClick={() => {
                  if (!artifactBaseline) return; setValidationMessage("Comparing with reference…");
                  void api.compareValidationBaseline(artifactBaseline.id, activeRunId ?? (detail?.result?.run_id as string | undefined) ?? "latest").then((result) => setValidationMessage(`${String(result.status)}: ${String(result.summary)}`)).catch((error: unknown) => setValidationMessage(error instanceof Error ? error.message : "Comparison failed."));
                }}>Compare with reference</button>
                <button type="button" disabled={!focusedArtifact?.path} onClick={() => { void navigator.clipboard?.writeText(String(focusedArtifact?.path ?? "")); }}>Copy artifact path</button>
                <button type="button" disabled title="Coming later">Compare with previous run</button>
                <button type="button" disabled={!focusedArtifact?.path} title={focusedArtifact?.path ? "" : "No exportable artifact selected"}>Export selected artifact</button>
              </div>
            </section>
          </div>
        ) : (
          <div className="stage-control-panel-body">
            <StudioInspector
              compatible={compatible}
              detail={workspaceDetail}
              pipeline={selectedPipeline}
              selectedArtifact={focusedArtifact}
              overlayDebug={overlayDebug}
              selectedObjectId={focusedObject?.object_id ?? null}
              hoverSample={hoverSample}
              stageId={canonicalSelectedStageId}
              acquisitionContext={{
                dataset: selectedContextDatasetName || null,
                session: selectedContextSessionName || null,
                sessionType: selectedContextSessionType || null,
                sessionTags: selectedSessionTags,
              }}
              onUpsertObjectAnnotation={(payload) => void upsertObjectAnnotation(payload)}
            />
            <section className="control-panel-section">
              <span>Active display plan</span>
              <small>Source: {stageDisplayPlan.plan.source}</small>
              <small>Strategy: {stageDisplayPlan.plan.strategy}</small>
              <div className="table-wrap">
                <table className="detect-artifact-audit-table">
                  <thead>
                    <tr>
                      <th>View</th>
                      <th>Role</th>
                      <th>Expected</th>
                      <th>Primary</th>
                      <th>Fallback</th>
                      <th>Reference</th>
                      <th>Category</th>
                      <th>Reason</th>
                    </tr>
                  </thead>
                  <tbody>
                    {stageDisplayPlan.items.map((item) => (
                      <tr key={item.viewId}>
                        <td>{item.label}</td>
                        <td>{displayRoleLabel(item.role)}</td>
                        <td><code>{item.expectedPrimaryArtifactId ?? "-"}</code></td>
                        <td><code>{item.primaryArtifactId ?? "-"}</code></td>
                        <td><code>{item.fallbackArtifactId ?? "-"}</code></td>
                        <td><code>{item.referenceArtifactId ?? "-"}</code></td>
                        <td>{item.category}</td>
                        <td>{item.reason}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
            {canonicalSelectedStageId === "detect_belt_plane" && stageSubstagePlan.plan && (
              <section className="control-panel-section">
                <span>Substage plan</span>
                <small>Process: {stageSubstagePlan.plan.processLabel}</small>
                <small>Mode: {activeReferenceViewScope === "substage" ? "substage detail" : "strategy path"}</small>
                <div className="table-wrap">
                  <table className="detect-artifact-audit-table">
                    <thead>
                      <tr>
                        <th>Substage</th>
                        <th>Views</th>
                        <th>Artifacts</th>
                        <th>Missing</th>
                        <th>Default mode</th>
                      </tr>
                    </thead>
                    <tbody>
                      {stageSubstagePlan.substages.map((substage) => (
                        <tr key={substage.substageId}>
                          <td>{substage.label}</td>
                          <td>{substage.views.map((entry) => compactDetectViewLabel(entry.view.id, entry.view.label)).join(" | ") || "-"}</td>
                          <td>{substage.supportingArtifacts.filter((entry) => entry.category !== "missing").map((entry) => entry.artifactId).join(" | ") || "-"}</td>
                          <td>{substage.missingArtifactIds.join(" | ") || "-"}</td>
                          <td>{substage.defaultRenderMode ?? "-"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>
            )}
            {activeProcessingUnit && (
              <details className="control-panel-details">
                <summary>Selected unit contract / trace</summary>
                <pre className="control-panel-json">{JSON.stringify({
                  unit: activeProcessingUnit,
                  resolved_view: activeProcessingUnitView,
                  trace_quality: activeTraceQuality,
                  trace_entry: activeTraceEntry,
                  recipe_snapshot_values: activeLastRunStageParams,
                }, null, 2)}</pre>
              </details>
            )}
            <section className="control-panel-section">
              <span>Execution trace</span>
              <PipelineExecutionGraph execution={workspaceDetail?.result?.pipeline_execution} selectedStageId={canonicalSelectedStageId} />
            </section>
            {canonicalSelectedStageId !== "detect_belt_plane" && overflowStageTabs.length > 0 && (
              <section className="control-panel-section">
                <span>Debug / legacy artifacts</span>
                <div className="studio-workspace-tabs">
                  {overflowStageTabs.map((tab) => (
                    <button className={tab.id === activeTab ? "active" : ""} key={tab.id} onClick={() => setActiveTab(tab.id)} type="button">
                      {tab.label}
                      <ArtifactRelevanceBadge relevance={tab.relevance} reason={tab.hiddenReason} />
                    </button>
                  ))}
                </div>
              </section>
            )}
            {canonicalSelectedStageId === "detect_belt_plane" && detectArtifactAuditRows.length > 0 && (
              <details className="detect-artifact-audit" open>
                <summary>Artifact details</summary>
                <div className="table-wrap">
                  <table className="detect-artifact-audit-table">
                    <thead>
                      <tr>
                        <th>View</th>
                        <th>Resolved artifact</th>
                        <th>Relevance</th>
                        <th>Reason</th>
                        <th>Run id / provenance</th>
                        <th>Strategy</th>
                        <th>Component mode</th>
                        <th>Split method</th>
                      </tr>
                    </thead>
                    <tbody>
                      {detectArtifactAuditRows.map((row) => (
                        <tr key={row.viewId}>
                          <td>{row.viewLabel}</td>
                          <td><code>{row.resolvedArtifact}</code></td>
                          <td>{row.relevance.replace(/_/g, " ")}</td>
                          <td>{row.reason}</td>
                          <td>{row.provenance}</td>
                          <td>{row.strategy.replace(/_/g, " ")}</td>
                          <td>{detectComponentModeLabel(row.componentMode)}</td>
                          <td>{row.splitMethod.replace(/_/g, " ")}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </details>
            )}
            {focusedArtifact && (
              <details className="control-panel-details">
                <summary>Selected artifact metadata</summary>
                <pre className="control-panel-json">{JSON.stringify(focusedArtifact.metadata ?? {}, null, 2)}</pre>
              </details>
            )}
          </div>
        )}
      </aside>
      {jobsPopoverOpen && (
      <section className="studio-processing-job-monitor">
        <div className="studio-processing-job-monitor-header">
          <strong>Processing jobs</strong>
          <span className="studio-processing-job-monitor-counts">
            {processingJobStatusCounts.running} running · {processingJobStatusCounts.queued} queued · {processingJobStatusCounts.completed} completed · {processingJobStatusCounts.failed + processingJobStatusCounts.partial} issues
          </span>
          <button type="button" onClick={() => setJobsPopoverOpen(false)}>Close</button>
        </div>
        <div className="studio-processing-job-monitor-body">
          {processingJobsLoading && !processingJobs.length && <small>Loading jobs…</small>}
          {processingJobsUnavailable && <small>Processing jobs unavailable.</small>}
          {!processingJobs.length && !processingJobsLoading && !processingJobsUnavailable && <small>No processing jobs yet.</small>}
          {activeProcessingJobs.length > 0 && (
            <div className="studio-processing-job-monitor-section">
              <strong>Active</strong>
              {activeProcessingJobs.map((job) => (
                <article className="studio-processing-job-card" key={job.id}>
                  <header>
                    <strong>{formatProcessingJobScope(job)}</strong>
                    <span>{job.pipeline_id}</span>
                  </header>
                  <div className="studio-processing-job-card-progress">
                    <progress max={Math.max(1, job.total_takes)} value={Math.min(job.total_takes, job.completed_takes + job.failed_takes + job.skipped_takes)} />
                    <small>{job.completed_takes + job.failed_takes + job.skipped_takes}/{job.total_takes || 0}</small>
                  </div>
                  <div className="studio-processing-job-card-meta">
                    <small>{job.status}</small>
                    <small>{job.failed_takes} failed</small>
                    <small>{job.started_at ? `started ${job.started_at}` : job.created_at}</small>
                  </div>
                  {job.status !== "cancelled" && job.status !== "completed" && job.status !== "failed" && (
                    <button type="button" disabled={pipelineActionBusy} onClick={() => void cancelProcessingJob(job.id)}>Cancel</button>
                  )}
                </article>
              ))}
            </div>
          )}
          <div className="studio-processing-job-monitor-section">
            <strong>Recent</strong>
            {processingJobs.slice(0, 5).map((job) => (
              <article className="studio-processing-job-card" key={job.id}>
                <header>
                  <strong>{formatProcessingJobScope(job)}</strong>
                  <span>{job.pipeline_id}</span>
                </header>
                <div className="studio-processing-job-card-meta">
                  <small>{job.status}</small>
                  <small>{job.completed_takes}/{job.total_takes} complete</small>
                  <small>{job.failed_takes} failed</small>
                </div>
              </article>
            ))}
          </div>
        </div>
      </section>
      )}
      {graphWorkspaceExpanded && (
        <ProcessingUnitGraphWorkspace
          graph={processingUnitGraph}
          selectedNodeId={selectedGraphNodeId}
          onSelectNode={focusProcessingUnitFromGraph}
          selectedUnit={activeProcessingUnit}
          selectedUnitIO={activeProcessingUnitIO}
          selectedTraceEntry={activeTraceEntry as Record<string, unknown> | null}
          selectedRecipe={selectedRecipe}
          currentUnitValues={activeContractStageValues}
          recipeUnitValues={activeRecipeUnitValues}
          lastRunValues={activeLastRunContractValues}
          comparisonUnit={activeComparisonUnit}
          comparisonSummaryLine={formatComparisonSummaryLine(comparisonResult)}
          takeId={workspaceDetail?.take_id ?? null}
          runSummary={runActionSummary}
          lineageSummary={graphWorkspaceLineageSummary}
          currentRecipeDirty={currentRecipeDirty}
          currentVsLastRunDirty={currentVsLastRunDirty}
          tuneContent={graphWorkspaceTuneContent}
          onResetUnitToRecipe={activeProcessingUnit ? () => resetRecipeUnitToPersisted(activeProcessingUnit) : null}
          onResetUnitToLastRun={activeProcessingUnit ? () => resetUnitToLastRunValues(activeProcessingUnit) : null}
          onResetUnitToDefaults={activeProcessingUnit ? () => resetUnitToContractDefaults(activeProcessingUnit) : null}
          onSaveRecipeAsNew={() => void saveCurrentRecipeAsNew()}
          onSaveRecipeVersion={selectedRecipe ? () => void updateSelectedRecipeVersion() : null}
          onCloneRecipe={selectedRecipe ? () => void cloneSelectedRecipe() : null}
          onPlanRerun={activeProcessingUnit ? () => void planPartialRerunFromSelectedUnit() : null}
          onExecutePartialRerun={partialRerunPlan?.plan_id ? () => void executePartialRerunFromPlan() : null}
          partialRerunPlan={partialRerunPlan}
          partialRerunExecution={partialRerunExecution}
          planningBusy={partialRerunPlanning}
          executingBusy={partialRerunExecuting}
          planStale={partialRerunPlanStale}
          context={graphWorkspaceContext}
          onClose={() => setGraphWorkspaceExpanded(false)}
          onOpenTune={() => {
            setControlPanelTab("tune");
            setStudioMode("tune");
            setInspectorExpanded(true);
            setGraphWorkspaceExpanded(false);
          }}
          onOpenCompare={() => {
            setControlPanelTab("compare");
            setStudioMode("compare");
            setInspectorExpanded(true);
            setGraphWorkspaceExpanded(false);
          }}
          onOpenArtifact={openArtifactFromWorkspace}
          pipelineId={selectedPipeline?.id ?? null}
          activeRunId={typeof workspaceDetail?.result?.run_id === "string" ? workspaceDetail.result.run_id : null}
          activeRunTypeLabel={graphWorkspaceActiveRunTypeLabel}
          onSelectRun={(runId) => setActiveRunId(runId)}
          onOpenComparison={(comparisonId) => { void reopenComparison(comparisonId); }}
          onLineageRefreshed={refreshPipelineRuns}
          staleWarnings={graphWorkspaceStaleWarnings}
          performanceSummaryLine={formatPartialExecutionSummaryLine(workspaceDetail?.result?.partial_execution_summary)}
        />
      )}
    </main>
  );
}
