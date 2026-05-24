import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  fileUrl,
  type CaptureModality,
  type DatasetSessionSummary,
  type DatasetSummary,
  type PipelineInfo,
  type PipelineInstance,
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
  type RuntimeWorkerStatus,
  type TakeSummary,
} from "../api/client";
import PipelineStatusPanel from "../components/PipelineStatusPanel";
import PipelineExecutionGraph from "../components/PipelineExecutionGraph";
import StudioInspector from "../components/StudioInspector";
import StageParameterPanel from "../components/StageParameterPanel";
import type { OverlayDebugInfo } from "../components/overlayModel";
import {
  chooseBestPipelineForTake,
  incompatibleModalityMessage,
  modalityCompatible,
  stageTimingMs,
  type StageViewTabId
} from "../components/pipelineModel";
import { artifactsForObject, artifactsForStage, objectsForTake, selectedObject } from "../components/studioWorkspaceModel";
import { renderStageSemanticView } from "../components/stage_view_renderers";
import { canonicalStageId, stageSemanticDefinition, stageSemanticSummary } from "../components/stageSemantics";
import type { RoiPolygon, RoiRect } from "../components/roiMapping";
import type { RoiType } from "../components/ProjectionArtifactViewer";
import type { StudioDebugMode } from "../components/studioDebugMode";
import type { HoverInspectionSample } from "../components/hoverInspectionModel";
import { nextDatasetSelectionAfterCreate, nextSessionSelectionAfterCreate } from "../components/datasetSidebarModel";
import { buildCapturePayload, captureDisabledReason, nextSelectedTakeAfterCapture } from "../components/captureWorkflowModel";
import { resolveNextSelectedTakeId } from "../components/takeSelectionModel";

type TakeFilter = "all" | "processed" | "unprocessed" | "warnings";
type SegmentationTuningParams = Record<string, unknown>;
type PlaneQaTuningParams = Record<string, unknown>;
type RoiEditStatus = "idle" | "active" | "unsaved" | "saved" | "rerun_required";
type ScaleCorrection = { x: number; y: number; z: number };
const SENSOR_STUDIO_25D_DEFAULTS_KEY = "sensor_studio.25d.stage_params.defaults.v1";

function read25dSavedDefaults(): Record<string, unknown> {
  if (typeof window === "undefined") return {};
  try {
    const raw = window.localStorage.getItem(SENSOR_STUDIO_25D_DEFAULTS_KEY);
    return raw ? JSON.parse(raw) as Record<string, unknown> : {};
  } catch {
    return {};
  }
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
  const [tagFilter, setTagFilter] = useState("");
  const [validationFilter, setValidationFilter] = useState<"all" | "unreviewed" | "valid" | "invalid" | "needs_review">("all");
  const [detail, setDetail] = useState<TakeDetail | null>(null);
  const [selectedObjectId, setSelectedObjectId] = useState<number | null>(null);
  const [selectedArtifactId, setSelectedArtifactId] = useState<string | null>(null);
  const [hoveredObjectId, setHoveredObjectId] = useState<number | null>(null);
  const [hoveredArtifactId, setHoveredArtifactId] = useState<string | null>(null);
  const [hoverSample, setHoverSample] = useState<HoverInspectionSample | null>(null);
  const [overlayDebug, setOverlayDebug] = useState<OverlayDebugInfo | null>(null);
  const [selectedTemplateId, setSelectedTemplateId] = useState("mining_steel_ball_classification_2d_reflectance_mvp");
  const [selectedInputType, setSelectedInputType] = useState("rgb_image");
  const [selectedPipelineInstanceId, setSelectedPipelineInstanceId] = useState<string>("");
  const [manualImagePath, setManualImagePath] = useState("");
  const [pipelineSearch, setPipelineSearch] = useState("");
  const [pipelineFamilyFilter, setPipelineFamilyFilter] = useState<"all" | "2d" | "3d" | "25d" | "fusion" | "generic">("all");
  const [lastUsedPipelineByTake, setLastUsedPipelineByTake] = useState<Record<string, string>>({});
  const [tabFallbackMessage, setTabFallbackMessage] = useState<string | null>(null);
  const [showDatasetCreate, setShowDatasetCreate] = useState(false);
  const [showDatasetSessionCreate, setShowDatasetSessionCreate] = useState(false);
  const [newDatasetName, setNewDatasetName] = useState("");
  const [newDatasetNotes, setNewDatasetNotes] = useState("");
  const [newSessionName, setNewSessionName] = useState("");
  const [newSessionNotes, setNewSessionNotes] = useState("");
  const [creatingDataset, setCreatingDataset] = useState(false);
  const [creatingDatasetSession, setCreatingDatasetSession] = useState(false);
  const [createDatasetError, setCreateDatasetError] = useState<string | null>(null);
  const [createSessionError, setCreateSessionError] = useState<string | null>(null);
  const [showCaptureCreate, setShowCaptureCreate] = useState(false);
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
  const [pipelineActionBusy, setPipelineActionBusy] = useState(false);
  const [pipelineActionMessage, setPipelineActionMessage] = useState<string | null>(null);
  const [roiEditModeActive, setRoiEditModeActive] = useState(false);
  const [roiEditStatus, setRoiEditStatus] = useState<RoiEditStatus>("idle");
  const [resultStale, setResultStale] = useState(false);
  const [debugMode, setDebugMode] = useState<StudioDebugMode>("operator");
  const selectedTakeIdRef = useRef(selectedTakeId);

  useEffect(() => {
    selectedTakeIdRef.current = selectedTakeId;
  }, [selectedTakeId]);


  const load = useCallback(async () => {
    const [
      nextTakes,
      nextSessions,
      nextPipelines,
      nextRuntime,
      nextTemplates,
      nextInstances,
      nextDatasets,
      nextWorkers,
      nextBindings,
      nextLatestPublished,
    ] = await Promise.all([
      api.filteredTakes({
        session_id: selectedSession || undefined,
        dataset_id: selectedDataset || undefined,
        validation_status: validationFilter === "all" ? undefined : validationFilter,
        tag: tagFilter || undefined,
        search: takeSearch || undefined,
        show_archived: showArchived,
      }),
      api.sessions(),
      api.pipelines(),
      api.state(),
      api.processTemplates(),
      api.pipelineInstances(),
      api.datasets(),
      api.runtimeWorkers().catch(() => ({ workers: [] })),
      api.processBindings().catch(() => [] as ProcessBinding[]),
      api.latestPublishedInspectionResult().catch(() => null as InspectionPublishedResult | null),
    ]);
    const nextDatasetSessions = selectedDataset ? await api.datasetSessions(selectedDataset) : [];
    setTakes(nextTakes);
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
    setSelectedPipelineInstanceId((current) => current || nextInstances[0]?.id || "");
    setSelectedPipelineId((current) => current || nextPipelines[0]?.id || "3d_ball_inspection");
    const nextTakeId = resolveNextSelectedTakeId({
      currentSelectedTakeId: selectedTakeIdRef.current,
      availableTakeIds: nextTakes.map((take) => take.take_id),
      preferredTakeId: nextTakes[0]?.take_id ?? null,
    }) ?? "";
    setSelectedTakeId(nextTakeId);
  }, [selectedSession, selectedDataset, validationFilter, tagFilter, takeSearch, showArchived]);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    let cancelled = false;
    if (!selectedTakeId) {
      setDetail(null);
      return () => { cancelled = true; };
    }
    void api.take(selectedTakeId).then((nextDetail) => {
      if (cancelled) return;
      setDetail(nextDetail);
    }).catch(() => {
      if (cancelled) return;
      setDetail(null);
    });
    return () => { cancelled = true; };
  }, [selectedTakeId]);

  const selectedPipeline = useMemo(
    () => pipelines.find((pipeline) => pipeline.id === selectedPipelineId) ?? pipelines[0] ?? null,
    [pipelines, selectedPipelineId]
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
      low_gradient_close_kernel: Number(detect.low_gradient_close_kernel ?? 9),
      low_gradient_fill_holes: Boolean(detect.low_gradient_fill_holes ?? true),
      low_gradient_min_component_area: Number(detect.low_gradient_min_component_area ?? 1500),
      reference_surface_selection_mode: String(detect.reference_surface_selection_mode ?? "largest_constant_z"),
      reference_surface_model: String(detect.reference_surface_model ?? "auto"),
      reference_surface_max_plane_residual_p95_mm: Number(detect.reference_surface_max_plane_residual_p95_mm ?? 3.0),
      plane_fit_roi_enabled: Boolean((detect.plane_fit_roi as Record<string, unknown> | undefined)?.enabled ?? false),
      plane_fit_roi_type: String((detect.plane_fit_roi as Record<string, unknown> | undefined)?.type ?? "rectangle"),
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
  const detectionViewResolutionError = useMemo(() => {
    if (canonicalSelectedStageId !== "detection") return null;
    const ids = new Set(stageSemantic.views.map((view) => view.id));
    if (ids.has("candidate_overlay") && ids.has("blob_metrics") && ids.has("candidates_table")) return null;
    return "Detection artifacts found, but no detection views were registered.";
  }, [canonicalSelectedStageId, stageSemantic.views]);
  useEffect(() => {
    if (stageTabs.some((tab) => tab.id === activeTab)) {
      setTabFallbackMessage(null);
      return;
    }
    const fallback = (stageSemantic.defaultViewId || stageTabs[0]?.id || "overview") as StageViewTabId;
    setActiveTab(fallback);
    const fallbackLabel = stageTabs.find((tab) => tab.id === fallback)?.label ?? "default";
    setTabFallbackMessage(`Selected stage changed. Showing default view: ${fallbackLabel}.`);
  }, [activeTab, stageTabs, stageSemantic.defaultViewId, canonicalSelectedStageId]);
  const templateById = useMemo(() => new Map(processTemplates.map((template) => [template.id, template])), [processTemplates]);
  const thresholdStep = selectedPipelineInstance?.configured_steps.find((step) => step.step_id === "threshold");
  const morphologyStep = selectedPipelineInstance?.configured_steps.find((step) => step.step_id === "morphology");
  const ellipseStep = selectedPipelineInstance?.configured_steps.find((step) => step.step_id === "ellipse_fitting");
  const blobStep = selectedPipelineInstance?.configured_steps.find((step) => step.step_id === "blob_detection");
  const roiEnabled = Boolean(planeQaParams?.plane_fit_roi_enabled ?? false);
  const roiType = (() => {
    const raw = String(planeQaParams?.plane_fit_roi_type ?? "rectangle");
    if (raw === "polygon" || raw === "vertical_band") return raw;
    return "rectangle";
  })() as RoiType;
  const activeRoi = roiEnabled && roiType !== "polygon"
    ? {
      x: Number(planeQaParams?.plane_fit_roi_x ?? 0),
      y: roiType === "vertical_band" ? 0 : Number(planeQaParams?.plane_fit_roi_y ?? 0),
      width: Number(planeQaParams?.plane_fit_roi_width ?? 0),
      height: roiType === "vertical_band" ? Number(planeQaParams?.plane_fit_roi_height ?? 0) : Number(planeQaParams?.plane_fit_roi_height ?? 0),
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
  const allArtifacts = artifactsForStage(workspaceDetail, canonicalSelectedStageId);
  const currentArtifacts = artifactsForObject(allArtifacts, selectedObjectId);
  const focusedArtifact = currentArtifacts.find((artifact) => artifact.artifact_id === selectedArtifactId) ?? currentArtifacts[0] ?? null;
  useEffect(() => {
    if (focusedArtifact?.object_id != null && focusedArtifact.object_id !== selectedObjectId) {
      setSelectedObjectId(focusedArtifact.object_id);
    }
  }, [focusedArtifact?.artifact_id, selectedObjectId]);

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
    if (selectedExperimentSession && take.experiment_session_id !== selectedExperimentSession) return false;
    if (modalityFilter !== "all" && !(take.modalities ?? []).includes(modalityFilter)) return false;
    const statusForFamily = (take.processing_by_family ?? []).find((item) => item.family === activePipelineFamily);
    const processedForFamily = Boolean(statusForFamily?.hasCompletedOutput);
    if (takeFilter === "processed" && !processedForFamily) return false;
    if (takeFilter === "unprocessed" && processedForFamily) return false;
    if (takeFilter === "warnings" && !(take.warning_count ?? 0)) return false;
    return true;
  });
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

  useEffect(() => {
    if (!pipelines.length) return;
    const takeKey = selectedTakeId || "__global__";
    const auto = chooseBestPipelineForTake(pipelines, detail?.modalities, lastUsedPipelineByTake[takeKey] ?? null);
    if (auto && auto.id !== selectedPipelineId) {
      setSelectedPipelineId(auto.id);
    }
  }, [selectedTakeId, detail?.modalities, pipelines, lastUsedPipelineByTake, selectedPipelineId]);

  function selectStage(stageId: string) {
    setSelectedStageId(stageId);
    setSelectedArtifactId(null);
    setRoiEditModeActive(false);
    setRoiEditStatus("idle");
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

  function build25dStageParams(): Record<string, unknown> | null {
    if (selectedPipeline?.id !== "mining_steel_ball_classification_25d" || !planeQaParams) return null;
    return {
      detect_belt_plane: {
        background_detection_strategy: String(planeQaParams.background_detection_strategy ?? "low_gradient_surface"),
        gradient_smoothing_kernel: Number(planeQaParams.gradient_smoothing_kernel ?? 3),
        gradient_threshold_mode: String(planeQaParams.gradient_threshold_mode ?? "percentile"),
        gradient_threshold_value: Number(planeQaParams.gradient_threshold_value ?? 2.0),
        gradient_threshold_percentile: Number(planeQaParams.gradient_threshold_percentile ?? 70),
        low_gradient_morphology_enabled: Boolean(planeQaParams.low_gradient_morphology_enabled ?? true),
        low_gradient_open_kernel: Number(planeQaParams.low_gradient_open_kernel ?? 3),
        low_gradient_close_kernel: Number(planeQaParams.low_gradient_close_kernel ?? 9),
        low_gradient_fill_holes: Boolean(planeQaParams.low_gradient_fill_holes ?? true),
        low_gradient_min_component_area: Number(planeQaParams.low_gradient_min_component_area ?? 1500),
        reference_surface_selection_mode: String(planeQaParams.reference_surface_selection_mode ?? "largest_constant_z"),
        reference_surface_model: String(planeQaParams.reference_surface_model ?? "auto"),
        reference_surface_max_plane_residual_p95_mm: Number(planeQaParams.reference_surface_max_plane_residual_p95_mm ?? 3.0),
        plane_fit_roi: Boolean(planeQaParams.plane_fit_roi_enabled ?? false)
          ? (
            String(planeQaParams.plane_fit_roi_type ?? "rectangle") === "polygon"
              ? {
                enabled: true,
                type: "polygon",
                points: ((planeQaParams.plane_fit_roi_points as Array<[number, number]> | undefined) ?? []).map(([x, y]) => [Math.round(Number(x)), Math.round(Number(y))]),
              }
              : String(planeQaParams.plane_fit_roi_type ?? "rectangle") === "vertical_band"
                ? {
                  enabled: true,
                  type: "vertical_band",
                  x: Number(planeQaParams.plane_fit_roi_x ?? 0),
                  width: Number(planeQaParams.plane_fit_roi_width ?? 0),
                }
              : {
                enabled: true,
                type: "rectangle",
                x: Number(planeQaParams.plane_fit_roi_x ?? 0),
                y: Number(planeQaParams.plane_fit_roi_y ?? 0),
                width: Number(planeQaParams.plane_fit_roi_width ?? 0),
                height: Number(planeQaParams.plane_fit_roi_height ?? 0),
              }
          )
          : null,
      },
      remove_belt_segment_objects: {
        min_height_mm: Number(planeQaParams.object_min_height_mm ?? 8.0),
        max_height_mm: planeQaParams.object_max_height_mm == null || planeQaParams.object_max_height_mm === "" ? null : Number(planeQaParams.object_max_height_mm),
        reference_tolerance_mm: Number(planeQaParams.reference_tolerance_mm ?? 0.0),
        suppress_plane_mask_in_segmentation: Boolean(planeQaParams.suppress_plane_mask_in_segmentation ?? true),
        morphology_kernel: Number(planeQaParams.morphology_kernel ?? 5),
        min_component_area: Number(planeQaParams.min_component_area ?? 120),
        fill_holes: Boolean(planeQaParams.fill_holes ?? true),
        smoothing_kernel: Number(planeQaParams.smoothing_kernel ?? 3),
      },
      known_object_25d: {
        enabled: Boolean(planeQaParams.known_object_enabled ?? false),
        apply_correction: Boolean(planeQaParams.known_object_apply_correction ?? true),
        object_label: String(planeQaParams.known_object_label ?? "cube"),
        target_selection: String(planeQaParams.known_object_target_selection ?? "largest_component"),
        manual_component_id: Number(planeQaParams.known_object_manual_component_id ?? 1),
        known_width_mm: Number(planeQaParams.known_width_mm ?? 40),
        known_depth_mm: Number(planeQaParams.known_depth_mm ?? 40),
        known_height_mm: Number(planeQaParams.known_height_mm ?? 25),
        tolerance_percent: Number(planeQaParams.known_tolerance_percent ?? 5),
        apply_persisted_correction: Boolean(planeQaParams.known_object_apply_persisted_correction ?? true),
        persisted_scale_correction_x: planeQaParams.known_object_persisted_scale_correction_x == null ? null : Number(planeQaParams.known_object_persisted_scale_correction_x),
        persisted_scale_correction_y: planeQaParams.known_object_persisted_scale_correction_y == null ? null : Number(planeQaParams.known_object_persisted_scale_correction_y),
        persisted_scale_correction_z: planeQaParams.known_object_persisted_scale_correction_z == null ? null : Number(planeQaParams.known_object_persisted_scale_correction_z),
      },
    };
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
      }) as { pipeline_instance_id?: string; run_id?: string; result_path?: string };
      if (response.pipeline_instance_id) {
        setSelectedPipelineInstanceId(response.pipeline_instance_id);
      }
      await load();
      const refreshed = await api.take(detail.take_id);
      setDetail(refreshed);
      setPipelineActionMessage(`Run updated: ${response.run_id ?? refreshed?.result?.processed_at ?? "completed"}`);
      if (roiEditStatus === "rerun_required" || roiEditStatus === "saved") {
        setRoiEditStatus("saved");
      }
      if (response.run_id) {
        setSelectedArtifactId(null);
        setSelectedObjectId(null);
      }
    } catch (err) {
      setPipelineActionMessage(`Apply + rerun failed: ${err instanceof Error ? err.message : "unknown error"}`);
    } finally {
      setPipelineActionBusy(false);
      setResultStale(false);
    }
  }

  function applyPlaneQaParams(rerun = false) {
    if (!planeQaParams) return;
    setPlaneQaPersistedParams(planeQaParams);
    setPlaneQaDirty(false);
    if (rerun) {
      void runSelectedPipeline(false, build25dStageParams());
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

  async function editTakeMetadata(field: "friendly_name" | "notes" | "expected_class" | "expected_diameter_mm" | "tags") {
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
    } else {
      payload[field] = next.trim() || null;
    }
    await api.updateTakeMetadata(detail.take_id, payload);
    await load();
    setDetail(await api.take(detail.take_id));
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
    setDetail(await api.take(detail.take_id));
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
    setDetail(await api.take(detail.take_id));
  }

  async function rerunLatestPipeline() {
    if (!detail || !selectedPipeline) return;
    await runSelectedPipeline(false);
  }

  async function upsertObjectAnnotation(payload: Record<string, unknown>) {
    if (!detail) return;
    await api.upsertObjectAnnotation(detail.take_id, payload);
    await load();
    setDetail(await api.take(detail.take_id));
  }

  async function removeFromDataset() {
    if (!detail) return;
    await api.removeTakeFromDataset(detail.take_id);
    await load();
    setDetail(await api.take(detail.take_id));
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
    setDetail(await api.take(detail.take_id));
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

  async function createDatasetInline() {
    if (!newDatasetName.trim()) return;
    setCreatingDataset(true);
    setCreateDatasetError(null);
    try {
      const created = await api.createDataset({ name: newDatasetName.trim(), notes: newDatasetNotes.trim() || null });
      const next = nextDatasetSelectionAfterCreate(datasets, created);
      setDatasets(next.datasets);
      setSelectedDataset(next.selectedDataset);
      setSelectedExperimentSession("");
      setShowDatasetCreate(false);
      setNewDatasetName("");
      setNewDatasetNotes("");
      await load();
    } catch (err) {
      setCreateDatasetError(err instanceof Error ? err.message : "Failed to create dataset");
    } finally {
      setCreatingDataset(false);
    }
  }

  async function createDatasetSessionInline() {
    if (!selectedDataset || !newSessionName.trim()) return;
    setCreatingDatasetSession(true);
    setCreateSessionError(null);
    try {
      const created = await api.createDatasetSession(selectedDataset, { name: newSessionName.trim(), notes: newSessionNotes.trim() || null });
      const next = nextSessionSelectionAfterCreate(datasetSessions, created);
      setDatasetSessions(next.sessions);
      setSelectedExperimentSession(next.selectedSession);
      setShowDatasetSessionCreate(false);
      setNewSessionName("");
      setNewSessionNotes("");
      await load();
    } catch (err) {
      setCreateSessionError(err instanceof Error ? err.message : "Failed to create session");
    } finally {
      setCreatingDatasetSession(false);
    }
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
      const nextDetail = await api.take(takeId);
      setDetail(nextDetail);
      setShowCaptureCreate(false);
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
    const nextType = (String(planeQaParams?.plane_fit_roi_type ?? "rectangle") === "vertical_band" ? "vertical_band" : "rectangle") as RoiType;
    setPlaneQaParams((current) => current ? {
      ...current,
      plane_fit_roi_enabled: true,
      plane_fit_roi_type: nextType,
      plane_fit_roi_x: roi.x,
      plane_fit_roi_y: nextType === "vertical_band" ? 0 : roi.y,
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
            ...(nextType === "vertical_band"
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
      plane_fit_roi_enabled: true,
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
    setPlaneQaParams((current) => current ? { ...current, plane_fit_roi_enabled: false, plane_fit_roi_type: "rectangle", plane_fit_roi_points: [] } : current);
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

  return (
    <main className="studio-workspace">
      <aside className="studio-sidebar">
        <div className="studio-sidebar-title">
          <span className="eyebrow">Studio</span>
          <strong>Data browser</strong>
        </div>
        <label className="field-label">
          <span className="inline-label-row">
            <span>Dataset</span>
            <button type="button" className="inline-plus-btn" onClick={() => setShowDatasetCreate((v) => !v)}>+</button>
          </span>
          <select value={selectedDataset} onChange={(event) => { setSelectedDataset(event.target.value); setSelectedExperimentSession(""); }}>
            <option value="">All datasets</option>
            {datasets.map((dataset) => (
              <option key={dataset.id} value={dataset.id}>{dataset.name}</option>
            ))}
          </select>
          {!datasets.length && <small>No datasets yet</small>}
          {!datasets.length && <small>Create your first dataset</small>}
          {showDatasetCreate && (
            <div className="inline-create-card">
              <input value={newDatasetName} onChange={(event) => setNewDatasetName(event.target.value)} placeholder="dataset name" />
              <input value={newDatasetNotes} onChange={(event) => setNewDatasetNotes(event.target.value)} placeholder="notes (optional)" />
              <div className="inline-create-actions">
                <button type="button" disabled={creatingDataset || !newDatasetName.trim()} onClick={() => void createDatasetInline()}>{creatingDataset ? "Creating..." : "Create"}</button>
                <button type="button" disabled={creatingDataset} onClick={() => setShowDatasetCreate(false)}>Cancel</button>
              </div>
              {createDatasetError && <small>{createDatasetError}</small>}
            </div>
          )}
        </label>
        <label className="field-label">
          <span className="inline-label-row">
            <span>Experiment session</span>
            <button type="button" className="inline-plus-btn" disabled={!selectedDataset} onClick={() => setShowDatasetSessionCreate((v) => !v)}>+</button>
          </span>
          <select value={selectedExperimentSession} onChange={(event) => setSelectedExperimentSession(event.target.value)} disabled={!selectedDataset}>
            <option value="">All dataset sessions</option>
            {datasetSessions.map((session) => (
              <option key={session.id} value={session.id}>{session.name}</option>
            ))}
          </select>
          {showDatasetSessionCreate && (
            <div className="inline-create-card">
              <input value={newSessionName} onChange={(event) => setNewSessionName(event.target.value)} placeholder="session name" />
              <input value={newSessionNotes} onChange={(event) => setNewSessionNotes(event.target.value)} placeholder="notes (optional)" />
              <div className="inline-create-actions">
                <button type="button" disabled={creatingDatasetSession || !newSessionName.trim() || !selectedDataset} onClick={() => void createDatasetSessionInline()}>{creatingDatasetSession ? "Creating..." : "Create"}</button>
                <button type="button" disabled={creatingDatasetSession} onClick={() => setShowDatasetSessionCreate(false)}>Cancel</button>
              </div>
              {createSessionError && <small>{createSessionError}</small>}
            </div>
          )}
        </label>
        <label className="field-label">
          Session
          <select value={selectedSession} onChange={(event) => setSelectedSession(event.target.value)}>
            <option value="">All sessions</option>
            {sessions.map((session) => (
              <option key={session.session_id} value={session.session_id}>{session.session_id}</option>
            ))}
          </select>
        </label>
        <div className="studio-pipeline-card">
          <span className="inline-label-row">
            <span>Capture new take</span>
            <button type="button" className="inline-plus-btn" onClick={() => setShowCaptureCreate((v) => !v)}>+</button>
          </span>
          {!!captureDisabledReason(selectedDataset, selectedExperimentSession) && (
            <small>{captureDisabledReason(selectedDataset, selectedExperimentSession)}</small>
          )}
          {showCaptureCreate && (
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
                <button type="button" disabled={captureLoading} onClick={() => setShowCaptureCreate(false)}>Cancel</button>
              </div>
              {captureError && <small>{captureError}</small>}
            </div>
          )}
        </div>
        <label className="field-label">
          Search
          <input value={takeSearch} onChange={(event) => setTakeSearch(event.target.value)} placeholder="take id or name" />
        </label>
        <label className="field-label">
          Tag
          <input value={tagFilter} onChange={(event) => setTagFilter(event.target.value)} placeholder="good, wet, broken..." />
        </label>
        <label className="field-label">
          Validation
          <select value={validationFilter} onChange={(event) => setValidationFilter(event.target.value as typeof validationFilter)}>
            <option value="all">All</option>
            <option value="unreviewed">Unreviewed</option>
            <option value="valid">Valid</option>
            <option value="invalid">Invalid</option>
            <option value="needs_review">Needs review</option>
          </select>
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
        <div className="studio-take-list">
          {filteredTakes.map((take) => (
            <button className={take.take_id === selectedTakeId ? "active" : ""} key={take.take_id} onClick={() => setSelectedTakeId(take.take_id)} type="button">
              <strong>{take.friendly_name || take.take_id}</strong>
              {take.friendly_name && take.friendly_name !== take.take_id && <small>{take.take_id}</small>}
              <div className="take-card-main">
                {take.thumbnail_path ? (
                  <img alt={take.friendly_name || take.take_id} className="take-thumb" src={fileUrl(take.take_id, take.thumbnail_path)} />
                ) : (
                  <div aria-hidden className="take-thumb take-thumb-empty" />
                )}
                <div className="take-class-meta">
                  <small>Class: {take.semantic_labels?.join(", ") || "-"}</small>
                  <small>Superclass: {take.superclass_labels?.join(", ") || "-"}</small>
                </div>
              </div>
            </button>
          ))}
        </div>
        <div className="studio-pipeline-card">
          <span>Take management</span>
          <button type="button" disabled={!detail} onClick={() => void editTakeMetadata("friendly_name")}>Rename take</button>
          <button type="button" disabled={!detail} onClick={() => void editTakeMetadata("notes")}>Edit notes</button>
          <button type="button" disabled={!detail} onClick={() => void editTakeMetadata("tags")}>Edit tags</button>
          <button type="button" disabled={!detail} onClick={() => void editTakeMetadata("expected_class")}>Set expected class</button>
          <button type="button" disabled={!detail} onClick={() => void editTakeMetadata("expected_diameter_mm")}>Set expected diameter</button>
          <div className="chip-row">
            {["good", "broken", "worn", "wet", "occluded", "touching_border", "bad_lighting", "multiple_objects"].map((chip) => (
              <button key={chip} type="button" onClick={() => void toggleSuggestedTag(chip)}>{chip}</button>
            ))}
          </div>
          <div className="chip-row">
            {(["unreviewed", "valid", "invalid", "needs_review"] as const).map((status) => (
              <button key={status} type="button" disabled={!detail} onClick={() => void setValidationStatus(status)}>{status}</button>
            ))}
          </div>
          <button type="button" disabled={!detail} onClick={() => void rerunLatestPipeline()}>Rerun latest pipeline</button>
          <button
            type="button"
            disabled={!detail || !((detail.processing_by_family ?? []).find((item) => item.family === "25d")?.hasCompletedOutput)}
            onClick={() => void publish25dForSelectedTake()}
          >
            Publish 25D result
          </button>
          <button type="button" disabled={!detail} onClick={() => void removeFromDataset()}>Remove from dataset</button>
          {!Boolean((detail?.take_metadata as Record<string, unknown> | undefined)?.archived) ? (
            <button type="button" disabled={!detail} onClick={() => void archiveCurrentTake()}>Archive take</button>
          ) : (
            <button type="button" disabled={!detail} onClick={() => void restoreCurrentTake()}>Restore archived take</button>
          )}
          <details>
            <summary>Delete permanently (advanced)</summary>
            <small>Deletes raw incoming take files, processed outputs, linked sidecar metadata, and indexed run directories for this take.</small>
            <button type="button" disabled={!detail} onClick={() => void permanentDeleteCurrentTake()}>Delete permanently</button>
          </details>
        </div>
        <details className="studio-pipeline-card">
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
        </details>
        <div className="studio-pipeline-card">
          <span>Run history (selected take + pipeline)</span>
          {!contextualRunHistory.length && <small>No runs yet for this selection.</small>}
          {contextualRunHistory.map((run) => (
            <small key={run.id}>{run.timestamp || run.id} | {run.status} | {run.duration.toFixed(1)} ms</small>
          ))}
        </div>
      </aside>

      <section className="studio-center">
        <header className="studio-workspace-toolbar">
          <span className="toolbar-chip">Take: {detail?.take_id ?? "none"}</span>
          <label className="toolbar-chip">
            Pipeline
            <select value={selectedPipelineId} onChange={(event) => selectPipeline(event.target.value)}>
              {pipelines.map((pipeline) => (
                <option key={pipeline.id} value={pipeline.id}>{pipeline.display_name}</option>
              ))}
            </select>
          </label>
          <label className="toolbar-chip">
            Stage
            <select value={selectedStageId} onChange={(event) => selectStage(event.target.value)}>
              {(selectedPipeline?.stages ?? []).map((stage) => (
                <option key={stage.id} value={stage.id}>{prettyStageLabel(stage.id, stage.display_name)}</option>
              ))}
            </select>
          </label>
          <span className={`toolbar-chip ${compatible ? "ok" : "warn"}`}>{compatible ? "compatible" : "incompatible"}</span>
          <div className="studio-actions toolbar-actions">
            <button disabled={!compatible || pipelineActionBusy} title={!compatible ? "Cannot execute: incompatible modalities." : ""} onClick={() => void runSelectedPipeline(false)} type="button">Run</button>
            <button disabled={!detail || !compatible || pipelineActionBusy} title={!compatible ? "Cannot execute: incompatible modalities." : ""} onClick={() => void runSelectedPipeline(true)} type="button">Reprocess</button>
            <button disabled={!detail || pipelineActionBusy} onClick={() => void clearSelectedOutputs()} type="button">Clear</button>
          </div>
        </header>
        {pipelineActionMessage && <small className={resultStale ? "preview-stale" : "section-note"}>{pipelineActionMessage}</small>}
        {!compatible && selectedPipeline && (
          <small className="section-note">{incompatibleModalityMessage(selectedPipeline, detail?.modalities) ?? "Cannot execute: incompatible modalities."}</small>
        )}
        <section className="studio-result-workspace">
        <nav className="studio-workspace-tabs">
          {stageTabs.map((tab) => (
            <button className={tab.id === activeTab ? "active" : ""} key={tab.id} onClick={() => setActiveTab(tab.id)} type="button">
              {tab.label}
            </button>
          ))}
        </nav>
        {tabFallbackMessage && <small className="section-note">{tabFallbackMessage}</small>}
        {detectionViewResolutionError && <small className="section-note">{detectionViewResolutionError}</small>}

        <section className="studio-workbench">
          <div className="studio-workbench-title">
            <span className="eyebrow">Engineering workspace</span>
            <strong>{selectedStage ? prettyStageLabel(selectedStage.id, selectedStage.display_name) : "Workspace"}</strong>
          </div>
          <div className="engineering-workbench-grid">
          <aside className="engineering-parameter-sidebar">
          {canonicalSelectedStageId === "segmentation" && (
            <StageParameterPanel
              stage={selectedStage}
              values={segmentationPreviewParams}
              persistedValues={segmentationPersistedParams}
              loading={segmentationPreviewLoading}
              dirty={segmentationPreviewDirty}
              previewLabel={`Components: ${segmentationPreview?.metrics.component_count ?? "-"} • Threshold coverage: ${segmentationPreview?.metrics.threshold_coverage == null ? "-" : `${(segmentationPreview.metrics.threshold_coverage * 100).toFixed(1)}%`} • Cleaned coverage: ${segmentationPreview?.metrics.cleaned_coverage == null ? "-" : `${(segmentationPreview.metrics.cleaned_coverage * 100).toFixed(1)}%`} • ${segmentationPreview?.diagnostics?.preview ? "Preview (non-persisted)" : "Persisted"}`}
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
              previewLabel={`Fitted: ${Number((segmentationPreview?.artifacts ?? []).filter((item) => item.artifact_id === "ellipse_overlay").length ? 1 : 0)} • ${segmentationPreview?.diagnostics?.preview ? "Preview (non-persisted)" : "Persisted"}`}
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
          {canonicalSelectedStageId === "detect_belt_plane" && planeQaParams && (
            <section className="segmentation-controls-strip reference-surface-controls-strip">
              <div className="stage-parameter-panel-header">
                <strong>Reference surface tuning</strong>
                <small>{pipelineActionBusy ? "Applying recipe and rerunning…" : (planeQaDirty ? "Unsaved changes" : "In sync with last run payload")}</small>
              </div>
              <div className="reference-surface-roi-block">
                <strong>Reference surface region</strong>
                <label>Enable ROI
                  <input
                    type="checkbox"
                    checked={Boolean(planeQaParams.plane_fit_roi_enabled ?? false)}
                    onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, plane_fit_roi_enabled: e.target.checked } : c); setPlaneQaDirty(true); }}
                  />
                </label>
                <div className="reference-surface-roi-grid">
                  <label>Type
                    <select value={String(planeQaParams.plane_fit_roi_type ?? "rectangle")} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, plane_fit_roi_type: e.target.value } : c); setPlaneQaDirty(true); }}>
                      <option value="rectangle">rectangle</option>
                      <option value="polygon">polygon</option>
                      <option value="vertical_band">Vertical band ROI</option>
                    </select>
                  </label>
                  <label>X
                    <input type="number" min={0} step={1} value={Number(planeQaParams.plane_fit_roi_x ?? 0)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, plane_fit_roi_x: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                  </label>
                  {String(planeQaParams.plane_fit_roi_type ?? "rectangle") !== "vertical_band" && (
                    <label>Y
                      <input type="number" min={0} step={1} value={Number(planeQaParams.plane_fit_roi_y ?? 0)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, plane_fit_roi_y: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                    </label>
                  )}
                  <label>W
                    <input type="number" min={1} step={1} value={Number(planeQaParams.plane_fit_roi_width ?? 0)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, plane_fit_roi_width: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                  </label>
                  {String(planeQaParams.plane_fit_roi_type ?? "rectangle") !== "vertical_band" && (
                    <label>H
                      <input type="number" min={1} step={1} value={Number(planeQaParams.plane_fit_roi_height ?? 0)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, plane_fit_roi_height: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                    </label>
                  )}
                </div>
                <div className="reference-surface-roi-actions">
                  <button type="button" disabled={pipelineActionBusy} onClick={() => startVisualRoiEdit()}>Edit visually</button>
                  <button type="button" disabled={pipelineActionBusy} onClick={() => void clearRoi()}>Use full frame</button>
                </div>
                <div className="reference-surface-roi-status">
                  <small>{roiEditModeActive ? "ROI edit mode active" : "ROI edit mode inactive"}</small>
                  <small>{`ROI status: ${roiEditStatus}`}</small>
                </div>
                <small>
                  ROI constrains only reference-surface estimation. Normalization and segmentation still run on the full valid frame.
                </small>
                {String(planeQaParams.plane_fit_roi_type ?? "rectangle") === "vertical_band" && (
                  <small>Constrains processing to an X-range while preserving the full scan height.</small>
                )}
                <small>
                  Workflow: 1) Set ROI on conveyor/background 2) Tune low-gradient parameters 3) Verify selected surface 4) Verify normalized height 5) Verify segmentation.
                </small>
              </div>
              <div className="segmentation-controls-row">
                <label>Strategy
                  <select value={String(planeQaParams.background_detection_strategy ?? "low_gradient_surface")} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, background_detection_strategy: e.target.value } : c); setPlaneQaDirty(true); }}>
                    <option value="low_gradient_surface">low_gradient_surface</option>
                    <option value="depth_percentile_plane">depth_percentile_plane</option>
                  </select>
                </label>
                <label>Gradient percentile {Number(planeQaParams.gradient_threshold_percentile ?? 70)}
                  <input type="range" min={1} max={99} value={Number(planeQaParams.gradient_threshold_percentile ?? 70)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, gradient_threshold_percentile: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                </label>
                <label>Smoothing
                  <input type="number" min={1} max={21} step={2} value={Number(planeQaParams.gradient_smoothing_kernel ?? 3)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, gradient_smoothing_kernel: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                </label>
                <label>Open
                  <input type="number" min={1} max={31} step={2} value={Number(planeQaParams.low_gradient_open_kernel ?? 3)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, low_gradient_open_kernel: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                </label>
                <label>Close
                  <input type="number" min={1} max={31} step={2} value={Number(planeQaParams.low_gradient_close_kernel ?? 9)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, low_gradient_close_kernel: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                </label>
                <label>Min area
                  <input type="number" min={50} max={500000} step={50} value={Number(planeQaParams.low_gradient_min_component_area ?? 1500)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, low_gradient_min_component_area: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                </label>
                <label>Fill holes
                  <input type="checkbox" checked={Boolean(planeQaParams.low_gradient_fill_holes ?? true)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, low_gradient_fill_holes: e.target.checked } : c); setPlaneQaDirty(true); }} />
                </label>
                <label>Model
                  <select value={String(planeQaParams.reference_surface_model ?? "auto")} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, reference_surface_model: e.target.value } : c); setPlaneQaDirty(true); }}>
                    <option value="auto">auto</option>
                    <option value="plane">plane</option>
                    <option value="constant_z">constant_z</option>
                  </select>
                </label>
                <label>Object min height
                  <input type="number" min={0} max={200} step={0.5} value={Number(planeQaParams.object_min_height_mm ?? 8)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, object_min_height_mm: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                </label>
                <button type="button" disabled={pipelineActionBusy} onClick={() => applyPlaneQaParams(false)}>Apply</button>
                <button type="button" disabled={pipelineActionBusy} onClick={() => applyPlaneQaParams(true)}>Apply + rerun</button>
                <button type="button" disabled={pipelineActionBusy} onClick={() => void save25dParamsAsDefaults()}>Save defaults</button>
                <button type="button" disabled={pipelineActionBusy} onClick={() => { setPlaneQaParams(planeQaPersistedParams); setPlaneQaDirty(false); }}>Reset</button>
              </div>
            </section>
          )}
          {canonicalSelectedStageId === "load_heightmap" && planeQaParams && (
            <section className="segmentation-controls-strip reference-surface-controls-strip">
              <div className="stage-parameter-panel-header">
                <strong>Reference surface region</strong>
                <small>Set ROI here, then tune in Detect reference surface.</small>
              </div>
              <div className="reference-surface-roi-grid">
                <label>Type
                    <select value={String(planeQaParams.plane_fit_roi_type ?? "rectangle")} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, plane_fit_roi_type: e.target.value } : c); setPlaneQaDirty(true); }} >
                    <option value="rectangle">rectangle</option>
                    <option value="polygon">polygon</option>
                    <option value="vertical_band">Vertical band ROI</option>
                  </select>
                </label>
                <label>X
                  <input type="number" min={0} step={1} value={Number(planeQaParams.plane_fit_roi_x ?? 0)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, plane_fit_roi_x: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                </label>
                {String(planeQaParams.plane_fit_roi_type ?? "rectangle") !== "vertical_band" && (
                  <label>Y
                    <input type="number" min={0} step={1} value={Number(planeQaParams.plane_fit_roi_y ?? 0)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, plane_fit_roi_y: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                  </label>
                )}
                <label>W
                  <input type="number" min={1} step={1} value={Number(planeQaParams.plane_fit_roi_width ?? 0)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, plane_fit_roi_width: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                </label>
                {String(planeQaParams.plane_fit_roi_type ?? "rectangle") !== "vertical_band" && (
                  <label>H
                    <input type="number" min={1} step={1} value={Number(planeQaParams.plane_fit_roi_height ?? 0)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, plane_fit_roi_height: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                  </label>
                )}
              </div>
              <div className="reference-surface-roi-actions">
                <button type="button" disabled={pipelineActionBusy} onClick={() => startVisualRoiEdit()}>Edit visually</button>
                <button type="button" disabled={pipelineActionBusy} onClick={() => void clearRoi()}>Use full frame</button>
                <button type="button" disabled={pipelineActionBusy} onClick={() => applyPlaneQaParams(true)}>Apply + rerun</button>
              </div>
              {String(planeQaParams.plane_fit_roi_type ?? "rectangle") === "vertical_band" && (
                <small>Vertical band ROI constrains processing to an X-range while preserving the full scan height.</small>
              )}
              <small>ROI affects only reference-surface estimation (not full-frame normalization/segmentation).</small>
            </section>
          )}
          {canonicalSelectedStageId === "remove_belt_segment_objects" && planeQaParams && (
            <section className="segmentation-controls-strip reference-surface-controls-strip">
              <div className="stage-parameter-panel-header">
                <strong>Segmentation tuning</strong>
                <small>{pipelineActionBusy ? "Applying recipe and rerunning…" : (planeQaDirty ? "Unsaved changes" : "In sync with last run payload")}</small>
              </div>
              <div className="segmentation-controls-row">
                <label>Min height
                  <input type="number" min={0} max={200} step={0.5} value={Number(planeQaParams.object_min_height_mm ?? 8)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, object_min_height_mm: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                </label>
                <label>Max height
                  <input type="number" min={0} max={500} step={1} value={planeQaParams.object_max_height_mm == null || !Number.isFinite(Number(planeQaParams.object_max_height_mm)) ? "" : Number(planeQaParams.object_max_height_mm)} placeholder="none" onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, object_max_height_mm: e.target.value === "" ? null : Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                </label>
                <label>Reference tol
                  <input type="number" min={0} max={50} step={0.5} value={Number(planeQaParams.reference_tolerance_mm ?? 0)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, reference_tolerance_mm: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                </label>
                <label>Suppress reference
                  <input type="checkbox" checked={Boolean(planeQaParams.suppress_plane_mask_in_segmentation ?? true)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, suppress_plane_mask_in_segmentation: e.target.checked } : c); setPlaneQaDirty(true); }} />
                </label>
                <label>Morph kernel
                  <input type="number" min={1} max={31} step={2} value={Number(planeQaParams.morphology_kernel ?? 5)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, morphology_kernel: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                </label>
                <label>Min component
                  <input type="number" min={0} max={500000} step={10} value={Number(planeQaParams.min_component_area ?? 120)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, min_component_area: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                </label>
                <label>Fill holes
                  <input type="checkbox" checked={Boolean(planeQaParams.fill_holes ?? true)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, fill_holes: e.target.checked } : c); setPlaneQaDirty(true); }} />
                </label>
                <label>Smoothing
                  <input type="number" min={1} max={31} step={2} value={Number(planeQaParams.smoothing_kernel ?? 3)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, smoothing_kernel: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                </label>
                <button type="button" disabled={pipelineActionBusy} onClick={() => applyPlaneQaParams(false)}>Apply</button>
                <button type="button" disabled={pipelineActionBusy} onClick={() => applyPlaneQaParams(true)}>Apply + rerun</button>
                <button type="button" disabled={pipelineActionBusy} onClick={() => void save25dParamsAsDefaults()}>Save defaults</button>
                <button type="button" disabled={pipelineActionBusy} onClick={() => { setPlaneQaParams(planeQaPersistedParams); setPlaneQaDirty(false); }}>Reset</button>
              </div>
            </section>
          )}
          {(canonicalSelectedStageId === "fit_object_geometry" || canonicalSelectedStageId === "compute_height_metrics") && planeQaParams && (
            <section className="segmentation-controls-strip reference-surface-controls-strip measurement-controls-strip">
              <div className="stage-parameter-panel-header">
                <strong>Measurement controls</strong>
                <small>{pipelineActionBusy ? "Applying recipe and rerunning…" : (planeQaDirty ? "Unsaved changes" : "Known-cube calibration config")}</small>
              </div>
              <div className="measurement-controls-row">
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
                <button type="button" disabled={pipelineActionBusy} onClick={() => applyPlaneQaParams(false)}>Apply</button>
                <button type="button" disabled={pipelineActionBusy} onClick={() => applyPlaneQaParams(true)}>Apply + rerun</button>
                <button type="button" disabled={pipelineActionBusy} onClick={() => void save25dParamsAsDefaults()}>Save defaults</button>
                <button type="button" disabled={pipelineActionBusy} onClick={() => { setPlaneQaParams(planeQaPersistedParams); setPlaneQaDirty(false); }}>Reset</button>
              </div>
            </section>
          )}
          </aside>
          <section className="engineering-visual-workspace">
          {renderStageSemanticView({
            detail: workspaceDetail,
            compatible,
            processed,
            stageId: canonicalSelectedStageId,
            view: stageSemantic.views.find((item) => item.id === activeTab) ?? stageSemantic.views[0],
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
            debugMode: "engineering",
          })}
          {canonicalSelectedStageId === "detect_belt_plane" && (() => {
            const stageArtifacts = artifactsForStage(workspaceDetail, canonicalSelectedStageId);
            const fit = (stageArtifacts.find((a) => a.artifact_id === "plane_fit_debug")?.metadata ?? {}) as Record<string, unknown>;
            const sel = ((fit.background_selection as Record<string, unknown> | undefined) ?? {}) as Record<string, unknown>;
            const gdbg = ((sel.gradient_debug as Record<string, unknown> | undefined) ?? {}) as Record<string, unknown>;
            const roiDbg = ((fit.roi as Record<string, unknown> | undefined) ?? {}) as Record<string, unknown>;
            const roiWarnings = (Array.isArray(fit.warnings) ? fit.warnings : []).map(String).filter((item) => item.startsWith("roi_"));
            const coeffs = (fit.plane_coefficients as unknown[] | undefined) ?? [];
            return (
              <>
                <section className="pipeline-status-grid">
                  <div><span>Strategy</span><strong>{String(sel.background_detection_strategy ?? fit.background_detection_strategy ?? "-")}</strong></div>
                  <div><span>Reference model</span><strong>{String(fit.reference_surface_model_type ?? "failed")}</strong></div>
                  <div><span>Plane calculated</span><strong>{String(fit.reference_surface_model_type ?? "") === "plane" ? "yes" : "no"}</strong></div>
                  <div><span>Constant Z</span><strong>{fit.constant_z_mm == null ? "-" : `${Number(fit.constant_z_mm).toFixed(3)} mm`}</strong></div>
                  <div><span>Surface coverage</span><strong>{`${(Number(gdbg.selected_component_area_ratio ?? 0) * 100).toFixed(1)}%`}</strong></div>
                  <div><span>Residual p95</span><strong>{fit.model_residual_p95_mm == null ? "-" : `${Number(fit.model_residual_p95_mm).toFixed(2)} mm`}</strong></div>
                  <div><span>Selection reason</span><strong>{String(fit.model_selection_reason ?? "-")}</strong></div>
                  <div><span>Plane coeffs</span><strong>{coeffs.length ? coeffs.map((v) => Number(v).toFixed(3)).join(", ") : "-"}</strong></div>
                </section>
                <section className="pipeline-status-grid">
                  <div><span>ROI enabled</span><strong>{Boolean(roiDbg.enabled ?? false) ? "yes" : "no"}</strong></div>
                  <div><span>ROI coverage</span><strong>{`${(Number(roiDbg.area_ratio ?? 1) * 100).toFixed(1)}%`}</strong></div>
                  <div><span>ROI valid-pixel ratio</span><strong>{`${(Number(roiDbg.valid_pixel_ratio ?? 1) * 100).toFixed(1)}%`}</strong></div>
                  <div><span>ROI candidate coverage</span><strong>{`${(Number(roiDbg.candidate_coverage_ratio ?? 0) * 100).toFixed(1)}%`}</strong></div>
                  <div><span>ROI selected-surface coverage</span><strong>{`${(Number(roiDbg.selected_surface_coverage_ratio ?? 0) * 100).toFixed(1)}%`}</strong></div>
                  <div><span>ROI warnings</span><strong>{roiWarnings.length ? roiWarnings.join(", ") : "-"}</strong></div>
                </section>
              </>
            );
          })()}
          </section>
          </div>
        </section>
        </section>
      </section>

      <StudioInspector
        compatible={compatible}
        detail={workspaceDetail}
        pipeline={selectedPipeline}
        selectedArtifact={focusedArtifact}
        overlayDebug={overlayDebug}
        selectedObjectId={focusedObject?.object_id ?? null}
        hoverSample={hoverSample}
        stageId={canonicalSelectedStageId}
        onUpsertObjectAnnotation={(payload) => void upsertObjectAnnotation(payload)}
      />
    </main>
  );
}
  const prettyStageLabel = (stageId: string, fallback: string) => {
    const canonical = canonicalStageId(stageId, fallback);
    if (canonical === "detect_belt_plane") return "Detect reference surface";
    if (canonical === "normalize_heights_to_plane") return "Normalize heights to reference";
    if (canonical === "remove_belt_segment_objects") return "Remove reference + segment objects";
    return fallback;
  };
