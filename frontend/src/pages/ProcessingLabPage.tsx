import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  fileUrl,
  type CaptureModality,
  type DatasetSessionSummary,
  type DatasetSummary,
  type PipelineInfo,
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
  const [takesHasMore, setTakesHasMore] = useState(false);
  const [takesOffset, setTakesOffset] = useState(0);
  const [takesLoading, setTakesLoading] = useState(false);
  const [takesLoadingMore, setTakesLoadingMore] = useState(false);
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
  const [pipelineActionBusy, setPipelineActionBusy] = useState(false);
  const [pipelineActionMessage, setPipelineActionMessage] = useState<string | null>(null);
  const [roiEditModeActive, setRoiEditModeActive] = useState(false);
  const [roiEditStatus, setRoiEditStatus] = useState<RoiEditStatus>("idle");
  const [resultStale, setResultStale] = useState(false);
  const [debugMode, setDebugMode] = useState<StudioDebugMode>("operator");
  const [selectedTakeIds, setSelectedTakeIds] = useState<Set<string>>(new Set());
  const lastTakeClickIndexRef = useRef<number | null>(null);
  const selectedTakeIdRef = useRef(selectedTakeId);
  const takesRequestIdRef = useRef(0);

  const [debouncedTakeSearch, setDebouncedTakeSearch] = useState("");

  useEffect(() => {
    const timer = window.setTimeout(() => setDebouncedTakeSearch(takeSearch), 220);
    return () => window.clearTimeout(timer);
  }, [takeSearch]);

  useEffect(() => {
    selectedTakeIdRef.current = selectedTakeId;
  }, [selectedTakeId]);


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
          preferredTakeId: page.items[0]?.take_id ?? null,
        }) ?? "";
        setSelectedTakeId(nextTakeId);
      }
    } finally {
      if (takesRequestIdRef.current === requestId) {
        setTakesLoading(false);
        setTakesLoadingMore(false);
      }
    }
  }, [takePageParams]);

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
      api.processingJobs().catch(() => ({ jobs: [], counts: {} })),
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
    setSelectedPipelineInstanceId((current) => current || nextInstances[0]?.id || "");
    setSelectedPipelineId((current) => current || nextPipelines[0]?.id || "3d_ball_inspection");
    await loadTakePage(0, false);
  }, [selectedDataset, loadTakePage]);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    let cancelled = false;
    const refreshJobs = async () => {
      setProcessingJobsLoading(true);
      try {
        const nextJobs = await api.processingJobs();
        if (cancelled) return;
        setProcessingJobs(nextJobs.jobs ?? []);
        setProcessingJobCounts(nextJobs.counts ?? {});
      } catch {
        if (cancelled) return;
      } finally {
        if (!cancelled) setProcessingJobsLoading(false);
      }
    };
    void refreshJobs();
    const interval = window.setInterval(() => {
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
      low_gradient_close_kernel: Number(detect.low_gradient_close_kernel ?? 5),
      low_gradient_fill_holes: Boolean(detect.low_gradient_fill_holes ?? true),
      low_gradient_min_component_area: Number(detect.low_gradient_min_component_area ?? 1500),
      reference_surface_selection_mode: String(detect.reference_surface_selection_mode ?? "largest_constant_z"),
      reference_surface_model: String(detect.reference_surface_model ?? "auto"),
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
      // --- Belt stripe filter ---
      belt_stripe_filter_enabled: Boolean(detect.belt_stripe_filter_enabled ?? true),
      belt_stripe_filter_scope: String(detect.belt_stripe_filter_scope ?? "global"),
      belt_stripe_filter_baseline_mode: String(detect.belt_stripe_filter_baseline_mode ?? "opening"),
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
        low_gradient_close_kernel: Number(planeQaParams.low_gradient_close_kernel ?? 5),
        low_gradient_fill_holes: Boolean(planeQaParams.low_gradient_fill_holes ?? true),
        low_gradient_min_component_area: Number(planeQaParams.low_gradient_min_component_area ?? 1500),
        reference_surface_selection_mode: String(planeQaParams.reference_surface_selection_mode ?? "largest_constant_z"),
        reference_surface_model: String(planeQaParams.reference_surface_model ?? "auto"),
        reference_surface_max_plane_residual_p95_mm: Number(planeQaParams.reference_surface_max_plane_residual_p95_mm ?? 3.0),
        low_gradient_plateau_use_hessian_filter: Boolean(planeQaParams.low_gradient_plateau_use_hessian_filter ?? true),
        low_gradient_plateau_hessian_percentile: Number(planeQaParams.low_gradient_plateau_hessian_percentile ?? 70),
        low_gradient_plateau_hist_bins: Number(planeQaParams.low_gradient_plateau_hist_bins ?? 96),
        low_gradient_plateau_min_fraction: Number(planeQaParams.low_gradient_plateau_min_fraction ?? 0.05),
        low_gradient_plateau_min_pixels: Number(planeQaParams.low_gradient_plateau_min_pixels ?? 500),
        low_gradient_plateau_smoothing_sigma_bins: Number(planeQaParams.low_gradient_plateau_smoothing_sigma_bins ?? 2.5),
        low_gradient_plateau_peak_drop_ratio: Number(planeQaParams.low_gradient_plateau_peak_drop_ratio ?? 0.40),
        low_gradient_plateau_select_min_area_fraction: Number(planeQaParams.low_gradient_plateau_select_min_area_fraction ?? 0.20),
        low_gradient_plateau_selection_mode: String(planeQaParams.low_gradient_plateau_selection_mode ?? "lowest_dominant"),
        low_gradient_plateau_robust_band_mad_k: Number(planeQaParams.low_gradient_plateau_robust_band_mad_k ?? 3.0),
        low_gradient_plateau_detection_min_count_floor: Number(planeQaParams.low_gradient_plateau_detection_min_count_floor ?? 25),
        low_gradient_plateau_detection_min_count_fraction: Number(planeQaParams.low_gradient_plateau_detection_min_count_fraction ?? 0.25),
        belt_stripe_filter_enabled: Boolean(planeQaParams.belt_stripe_filter_enabled ?? true),
        belt_stripe_filter_scope: String(planeQaParams.belt_stripe_filter_scope ?? "global"),
        belt_stripe_filter_baseline_mode: String(planeQaParams.belt_stripe_filter_baseline_mode ?? "opening"),
        belt_stripe_filter_window_mm: Number(planeQaParams.belt_stripe_filter_window_mm ?? 30.0),
        belt_stripe_filter_direction: String(planeQaParams.belt_stripe_filter_direction ?? "auto"),
        belt_stripe_filter_threshold_mode: String(planeQaParams.belt_stripe_filter_threshold_mode ?? "otsu"),
        belt_stripe_filter_min_altitude_mm: Number(planeQaParams.belt_stripe_filter_min_altitude_mm ?? 10.0),
        belt_stripe_filter_k_mad: Number(planeQaParams.belt_stripe_filter_k_mad ?? 3.0),
        belt_stripe_filter_fixed_threshold_mm: Number(planeQaParams.belt_stripe_filter_fixed_threshold_mm ?? 10.0),
        belt_stripe_filter_min_stripe_fraction: Number(planeQaParams.belt_stripe_filter_min_stripe_fraction ?? 0.02),
        belt_stripe_filter_z_floor_enabled: Boolean(planeQaParams.belt_stripe_filter_z_floor_enabled ?? true),
        belt_stripe_filter_z_floor_use_upper_bound: Boolean(planeQaParams.belt_stripe_filter_z_floor_use_upper_bound ?? true),
        belt_stripe_filter_z_floor_margin_mm: Number(planeQaParams.belt_stripe_filter_z_floor_margin_mm ?? 20.0),
        belt_stripe_filter_max_stripe_height_mm: Number(planeQaParams.belt_stripe_filter_max_stripe_height_mm ?? 500.0),
        belt_stripe_filter_above_belt_close_mm: Number(planeQaParams.belt_stripe_filter_above_belt_close_mm ?? 30.0),
        belt_stripe_filter_object_kernel_mm: Number(planeQaParams.belt_stripe_filter_object_kernel_mm ?? 100.0),
        belt_stripe_filter_object_kernel_shape: String(planeQaParams.belt_stripe_filter_object_kernel_shape ?? "ellipse"),
        belt_stripe_filter_altitude_hist_bins: Number(planeQaParams.belt_stripe_filter_altitude_hist_bins ?? 64),
        belt_stripe_filter_auto_bimodality_margin: Number(planeQaParams.belt_stripe_filter_auto_bimodality_margin ?? 1.10),
        belt_stripe_filter_z_floor_upper_percentile: Number(planeQaParams.belt_stripe_filter_z_floor_upper_percentile ?? 99.0),
        belt_stripe_filter_z_floor_fallback_lower_percentile: Number(planeQaParams.belt_stripe_filter_z_floor_fallback_lower_percentile ?? 10.0),
        belt_stripe_filter_z_floor_fallback_upper_percentile: Number(planeQaParams.belt_stripe_filter_z_floor_fallback_upper_percentile ?? 25.0),
        belt_stripe_filter_warn_removed_fraction: Number(planeQaParams.belt_stripe_filter_warn_removed_fraction ?? 0.40),
        low_gradient_surface_support_z_mad_multiplier: Number(planeQaParams.low_gradient_surface_support_z_mad_multiplier ?? 2.5),
        low_gradient_surface_support_z_floor_mm: Number(planeQaParams.low_gradient_surface_support_z_floor_mm ?? 1.0),
        low_gradient_surface_support_z_mad_floor_mm: Number(planeQaParams.low_gradient_surface_support_z_mad_floor_mm ?? 0.25),
        low_gradient_surface_ridge_percentile: Number(planeQaParams.low_gradient_surface_ridge_percentile ?? 90.0),
        reference_surface_height_gate_enabled: Boolean(planeQaParams.reference_surface_height_gate_enabled ?? true),
        reference_surface_height_gate_margin_mm: Number(planeQaParams.reference_surface_height_gate_margin_mm ?? 8.0),
        reference_surface_height_gate_min_coverage_ratio: Number(planeQaParams.reference_surface_height_gate_min_coverage_ratio ?? 0.10),
        reference_surface_height_gate_max_coverage_ratio: Number(planeQaParams.reference_surface_height_gate_max_coverage_ratio ?? 0.95),
        reference_surface_height_gate_gap_floor_mm: Number(planeQaParams.reference_surface_height_gate_gap_floor_mm ?? 1.0),
        reference_surface_height_gate_gap_ratio: Number(planeQaParams.reference_surface_height_gate_gap_ratio ?? 8.0),
        plot_depth_plot_max_render_samples: Number(planeQaParams.plot_depth_plot_max_render_samples ?? 60000),
        plot_y_robust_percentile: Number(planeQaParams.plot_y_robust_percentile ?? 98.0),
        plane_fit_roi: String(planeQaParams.reference_surface_region_mode ?? "none") === "none"
          ? {
            enabled: false,
            type: "rectangle",
            x: 0,
            y: 0,
            width: 0,
            height: 0,
          }
          : (
            String(planeQaParams.reference_surface_region_mode ?? "none") === "polygon"
              ? {
                enabled: true,
                type: "polygon",
                points: ((planeQaParams.plane_fit_roi_points as Array<[number, number]> | undefined) ?? []).map(([x, y]) => [Math.round(Number(x)), Math.round(Number(y))]),
              }
              : String(planeQaParams.reference_surface_region_mode ?? "none") === "full_height_x_band"
                ? {
                  enabled: true,
                  type: "full_height_x_band",
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
          ),
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
    }) as { pipeline_instance_id?: string; run_id?: string; result_path?: string };
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

  return (
    <main className="studio-workspace">
      <aside className="studio-sidebar">
        <div className="studio-sidebar-title">
          <span className="eyebrow">Studio</span>
          <strong>Processing Lab</strong>
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
          <div className="studio-inline-action-row">
            <button type="button" disabled={pipelineActionBusy || !selectedPipeline || !selectedDataset} onClick={() => void reprocessCurrentDataset()}>Reprocess dataset</button>
          </div>
        </label>
        <label className="field-label">
          Experiment session
          <select value={selectedExperimentSession} onChange={(event) => setSelectedExperimentSession(event.target.value)} disabled={!selectedDataset}>
            <option value="">All experiment sessions</option>
            {datasetSessions.map((session) => (
              <option key={session.id} value={session.id}>{session.name}</option>
            ))}
          </select>
          <small className="field-help">Engineering/acquisition context grouping used for replay, calibration, and processing workflows.</small>
          <div className="studio-inline-action-row">
            <button type="button" disabled={pipelineActionBusy || !selectedPipeline || !selectedExperimentSession} onClick={() => void reprocessCurrentExperimentSession()}>Reprocess session</button>
          </div>
        </label>
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
        <div className="studio-pipeline-card studio-datasets-link-card">
          <small>Labeling, validation, annotations, and dataset composition are in <a href="/datasets">Datasets</a>.</small>
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
          <span className="toolbar-chip identity-chip">Take: {detail?.take_id ?? "none"}</span>
          <label className="toolbar-chip pipeline-chip">
            Pipeline
            <select value={selectedPipelineId} onChange={(event) => selectPipeline(event.target.value)}>
              {pipelines.map((pipeline) => (
                <option key={pipeline.id} value={pipeline.id}>{pipeline.display_name}</option>
              ))}
            </select>
          </label>
          <label className="toolbar-chip stage-chip">
            Stage
            <select value={selectedStageId} onChange={(event) => selectStage(event.target.value)}>
              {(selectedPipeline?.stages ?? []).map((stage) => (
                <option key={stage.id} value={stage.id}>{prettyStageLabel(stage.id, stage.display_name)}</option>
              ))}
            </select>
          </label>
          <span className={`toolbar-chip compatibility-chip ${compatible ? "ok" : "warn"}`}>{compatible ? "compatible" : "incompatible"}</span>
          <div className="studio-actions toolbar-actions">
            {(canonicalSelectedStageId === "fit_object_geometry" || canonicalSelectedStageId === "compute_height_metrics") && planeQaParams && (
              <button type="button" onClick={() => setMeasurementConfigOpen(true)}>Configure known-cube</button>
            )}
            <button disabled={!compatible || pipelineActionBusy} title={!compatible ? "Cannot execute: incompatible modalities." : ""} onClick={() => void runSelectedPipeline(false)} type="button">Run</button>
            <button disabled={!detail || !compatible || pipelineActionBusy} title={!compatible ? "Cannot execute: incompatible modalities." : ""} onClick={() => void runSelectedPipeline(true)} type="button">Reprocess</button>
            <button disabled={!detail || pipelineActionBusy} onClick={() => void clearSelectedOutputs()} type="button">Clear</button>
          </div>
        </header>
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
          <div className={(canonicalSelectedStageId === "fit_object_geometry" || canonicalSelectedStageId === "compute_height_metrics") ? "engineering-workbench-grid measurement-workbench-grid" : "engineering-workbench-grid"}>
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
                <div className="reference-surface-roi-grid">
                  <label>Mode
                    <select value={String(planeQaParams.reference_surface_region_mode ?? "none")} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, reference_surface_region_mode: e.target.value, plane_fit_roi_type: e.target.value === "full_height_x_band" ? "full_height_x_band" : e.target.value === "none" ? "rectangle" : e.target.value } : c); setPlaneQaDirty(true); }}>
                      <option value="none">No ROI / full frame</option>
                      <option value="full_height_x_band">full_height_x_band</option>
                      <option value="rectangle">rectangle</option>
                      <option value="polygon">polygon</option>
                    </select>
                  </label>
                  <label>X
                    <input type="number" min={0} step={1} value={Number(planeQaParams.plane_fit_roi_x ?? 0)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, plane_fit_roi_x: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                  </label>
                  {String(planeQaParams.reference_surface_region_mode ?? "none") !== "full_height_x_band" && (
                    <label>Y
                      <input type="number" min={0} step={1} value={Number(planeQaParams.plane_fit_roi_y ?? 0)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, plane_fit_roi_y: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                    </label>
                  )}
                  <label>W
                    <input type="number" min={1} step={1} value={Number(planeQaParams.plane_fit_roi_width ?? 0)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, plane_fit_roi_width: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                  </label>
                  {String(planeQaParams.reference_surface_region_mode ?? "none") !== "full_height_x_band" && (
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
                {String(planeQaParams.reference_surface_region_mode ?? "none") === "full_height_x_band" && (
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
                    <option value="low_gradient_depth_plateaus">low_gradient_depth_plateaus</option>
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
                  <input type="number" min={1} max={31} step={2} value={Number(planeQaParams.low_gradient_close_kernel ?? 5)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, low_gradient_close_kernel: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
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
              <details className="advanced-params-details">
                <summary><strong>Advanced parameters</strong> <small>(plateau, stripe filter, height gate, plot rendering)</small></summary>
                <div className="advanced-params-grid">
                  {String(planeQaParams.background_detection_strategy ?? "low_gradient_surface") === "low_gradient_depth_plateaus" && (
                    <fieldset>
                      <legend>Plateau detection</legend>
                      <div className="segmentation-controls-row">
                        <label>Hessian filter
                          <input type="checkbox" checked={Boolean(planeQaParams.low_gradient_plateau_use_hessian_filter ?? true)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, low_gradient_plateau_use_hessian_filter: e.target.checked } : c); setPlaneQaDirty(true); }} />
                        </label>
                        <label>Hessian pct
                          <input type="number" min={1} max={99} step={1} value={Number(planeQaParams.low_gradient_plateau_hessian_percentile ?? 70)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, low_gradient_plateau_hessian_percentile: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                        </label>
                        <label>Hist bins
                          <input type="number" min={16} max={512} step={1} value={Number(planeQaParams.low_gradient_plateau_hist_bins ?? 96)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, low_gradient_plateau_hist_bins: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                        </label>
                        <label>Min fraction
                          <input type="number" min={0} max={1} step={0.01} value={Number(planeQaParams.low_gradient_plateau_min_fraction ?? 0.05)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, low_gradient_plateau_min_fraction: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                        </label>
                        <label>Min pixels
                          <input type="number" min={16} max={1000000} step={50} value={Number(planeQaParams.low_gradient_plateau_min_pixels ?? 500)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, low_gradient_plateau_min_pixels: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                        </label>
                        <label>Smooth σ (bins)
                          <input type="number" min={0.5} max={20} step={0.1} value={Number(planeQaParams.low_gradient_plateau_smoothing_sigma_bins ?? 2.5)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, low_gradient_plateau_smoothing_sigma_bins: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                        </label>
                        <label>Peak drop
                          <input type="number" min={0.05} max={0.95} step={0.01} value={Number(planeQaParams.low_gradient_plateau_peak_drop_ratio ?? 0.40)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, low_gradient_plateau_peak_drop_ratio: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                        </label>
                        <label>Select min frac
                          <input type="number" min={0} max={1} step={0.01} value={Number(planeQaParams.low_gradient_plateau_select_min_area_fraction ?? 0.20)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, low_gradient_plateau_select_min_area_fraction: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                        </label>
                        <label>Selection
                          <select value={String(planeQaParams.low_gradient_plateau_selection_mode ?? "lowest_dominant")} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, low_gradient_plateau_selection_mode: e.target.value } : c); setPlaneQaDirty(true); }}>
                            <option value="lowest_dominant">lowest_dominant</option>
                            <option value="largest">largest</option>
                            <option value="lowest">lowest</option>
                            <option value="robust_band">robust_band</option>
                          </select>
                        </label>
                        <label>k·MAD
                          <input type="number" min={0.5} max={10} step={0.1} value={Number(planeQaParams.low_gradient_plateau_robust_band_mad_k ?? 3.0)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, low_gradient_plateau_robust_band_mad_k: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                        </label>
                        <label>Detect floor
                          <input type="number" min={1} max={10000} step={1} value={Number(planeQaParams.low_gradient_plateau_detection_min_count_floor ?? 25)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, low_gradient_plateau_detection_min_count_floor: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                        </label>
                        <label>Detect frac
                          <input type="number" min={0} max={1} step={0.05} value={Number(planeQaParams.low_gradient_plateau_detection_min_count_fraction ?? 0.25)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, low_gradient_plateau_detection_min_count_fraction: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                        </label>
                      </div>
                    </fieldset>
                  )}
                  <fieldset>
                    <legend>Belt stripe filter</legend>
                    <div className="segmentation-controls-row">
                      <label>Enabled
                        <input type="checkbox" checked={Boolean(planeQaParams.belt_stripe_filter_enabled ?? true)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, belt_stripe_filter_enabled: e.target.checked } : c); setPlaneQaDirty(true); }} />
                      </label>
                      <label>Scope
                        <select value={String(planeQaParams.belt_stripe_filter_scope ?? "global")} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, belt_stripe_filter_scope: e.target.value } : c); setPlaneQaDirty(true); }}>
                          <option value="global">global</option>
                          <option value="bg_plateau">bg_plateau</option>
                        </select>
                      </label>
                      <label>Baseline
                        <select value={String(planeQaParams.belt_stripe_filter_baseline_mode ?? "opening")} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, belt_stripe_filter_baseline_mode: e.target.value } : c); setPlaneQaDirty(true); }}>
                          <option value="opening">opening</option>
                          <option value="erosion">erosion</option>
                        </select>
                      </label>
                      <label>Direction
                        <select value={String(planeQaParams.belt_stripe_filter_direction ?? "auto")} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, belt_stripe_filter_direction: e.target.value } : c); setPlaneQaDirty(true); }}>
                          <option value="auto">auto</option>
                          <option value="raised">raised</option>
                          <option value="recessed">recessed</option>
                        </select>
                      </label>
                      <label>Window mm
                        <input type="number" min={1} max={500} step={1} value={Number(planeQaParams.belt_stripe_filter_window_mm ?? 30.0)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, belt_stripe_filter_window_mm: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                      </label>
                      <label>Threshold
                        <select value={String(planeQaParams.belt_stripe_filter_threshold_mode ?? "otsu")} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, belt_stripe_filter_threshold_mode: e.target.value } : c); setPlaneQaDirty(true); }}>
                          <option value="otsu">otsu</option>
                          <option value="k_mad">k_mad</option>
                          <option value="fixed">fixed</option>
                        </select>
                      </label>
                      <label>Min alt mm
                        <input type="number" min={0} max={500} step={0.5} value={Number(planeQaParams.belt_stripe_filter_min_altitude_mm ?? 10.0)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, belt_stripe_filter_min_altitude_mm: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                      </label>
                      <label>k·MAD
                        <input type="number" min={0.1} max={20} step={0.1} value={Number(planeQaParams.belt_stripe_filter_k_mad ?? 3.0)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, belt_stripe_filter_k_mad: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                      </label>
                      <label>Fixed thr mm
                        <input type="number" min={0} max={500} step={0.5} value={Number(planeQaParams.belt_stripe_filter_fixed_threshold_mm ?? 10.0)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, belt_stripe_filter_fixed_threshold_mm: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                      </label>
                      <label>Min stripe frac
                        <input type="number" min={0} max={1} step={0.005} value={Number(planeQaParams.belt_stripe_filter_min_stripe_fraction ?? 0.02)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, belt_stripe_filter_min_stripe_fraction: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                      </label>
                      <label>z-floor enabled
                        <input type="checkbox" checked={Boolean(planeQaParams.belt_stripe_filter_z_floor_enabled ?? true)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, belt_stripe_filter_z_floor_enabled: e.target.checked } : c); setPlaneQaDirty(true); }} />
                      </label>
                      <label>Use upper bound
                        <input type="checkbox" checked={Boolean(planeQaParams.belt_stripe_filter_z_floor_use_upper_bound ?? true)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, belt_stripe_filter_z_floor_use_upper_bound: e.target.checked } : c); setPlaneQaDirty(true); }} />
                      </label>
                      <label>z margin mm
                        <input type="number" min={0} max={500} step={0.5} value={Number(planeQaParams.belt_stripe_filter_z_floor_margin_mm ?? 20.0)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, belt_stripe_filter_z_floor_margin_mm: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                      </label>
                      <label>Max stripe h mm
                        <input type="number" min={0} max={5000} step={1} value={Number(planeQaParams.belt_stripe_filter_max_stripe_height_mm ?? 500.0)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, belt_stripe_filter_max_stripe_height_mm: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                      </label>
                      <label>Close mm
                        <input type="number" min={0} max={500} step={1} value={Number(planeQaParams.belt_stripe_filter_above_belt_close_mm ?? 30.0)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, belt_stripe_filter_above_belt_close_mm: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                      </label>
                      <label>Object kernel mm
                        <input type="number" min={1} max={1000} step={1} value={Number(planeQaParams.belt_stripe_filter_object_kernel_mm ?? 100.0)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, belt_stripe_filter_object_kernel_mm: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                      </label>
                      <label>Kernel shape
                        <select value={String(planeQaParams.belt_stripe_filter_object_kernel_shape ?? "ellipse")} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, belt_stripe_filter_object_kernel_shape: e.target.value } : c); setPlaneQaDirty(true); }}>
                          <option value="ellipse">ellipse</option>
                          <option value="rect">rect</option>
                        </select>
                      </label>
                      <label>Alt hist bins
                        <input type="number" min={8} max={512} step={1} value={Number(planeQaParams.belt_stripe_filter_altitude_hist_bins ?? 64)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, belt_stripe_filter_altitude_hist_bins: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                      </label>
                      <label>Auto bimodal m
                        <input type="number" min={1.0} max={5.0} step={0.05} value={Number(planeQaParams.belt_stripe_filter_auto_bimodality_margin ?? 1.10)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, belt_stripe_filter_auto_bimodality_margin: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                      </label>
                      <label>z-floor upper pct
                        <input type="number" min={50} max={100} step={0.5} value={Number(planeQaParams.belt_stripe_filter_z_floor_upper_percentile ?? 99.0)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, belt_stripe_filter_z_floor_upper_percentile: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                      </label>
                      <label>Fallback lower
                        <input type="number" min={0} max={50} step={0.5} value={Number(planeQaParams.belt_stripe_filter_z_floor_fallback_lower_percentile ?? 10.0)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, belt_stripe_filter_z_floor_fallback_lower_percentile: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                      </label>
                      <label>Fallback upper
                        <input type="number" min={0} max={100} step={0.5} value={Number(planeQaParams.belt_stripe_filter_z_floor_fallback_upper_percentile ?? 25.0)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, belt_stripe_filter_z_floor_fallback_upper_percentile: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                      </label>
                      <label>Warn removed
                        <input type="number" min={0} max={1} step={0.05} value={Number(planeQaParams.belt_stripe_filter_warn_removed_fraction ?? 0.40)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, belt_stripe_filter_warn_removed_fraction: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                      </label>
                    </div>
                  </fieldset>
                  {String(planeQaParams.background_detection_strategy ?? "low_gradient_surface") === "low_gradient_surface" && (
                    <fieldset>
                      <legend>Low-gradient surface tuning</legend>
                      <div className="segmentation-controls-row">
                        <label>z MAD ×
                          <input type="number" min={0.5} max={10} step={0.1} value={Number(planeQaParams.low_gradient_surface_support_z_mad_multiplier ?? 2.5)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, low_gradient_surface_support_z_mad_multiplier: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                        </label>
                        <label>z floor mm
                          <input type="number" min={0} max={50} step={0.1} value={Number(planeQaParams.low_gradient_surface_support_z_floor_mm ?? 1.0)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, low_gradient_surface_support_z_floor_mm: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                        </label>
                        <label>MAD floor mm
                          <input type="number" min={0} max={10} step={0.05} value={Number(planeQaParams.low_gradient_surface_support_z_mad_floor_mm ?? 0.25)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, low_gradient_surface_support_z_mad_floor_mm: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                        </label>
                        <label>Ridge pct
                          <input type="number" min={1} max={99} step={1} value={Number(planeQaParams.low_gradient_surface_ridge_percentile ?? 90.0)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, low_gradient_surface_ridge_percentile: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                        </label>
                      </div>
                    </fieldset>
                  )}
                  <fieldset>
                    <legend>Height gate</legend>
                    <div className="segmentation-controls-row">
                      <label>Enabled
                        <input type="checkbox" checked={Boolean(planeQaParams.reference_surface_height_gate_enabled ?? true)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, reference_surface_height_gate_enabled: e.target.checked } : c); setPlaneQaDirty(true); }} />
                      </label>
                      <label>Margin mm
                        <input type="number" min={0} max={200} step={0.5} value={Number(planeQaParams.reference_surface_height_gate_margin_mm ?? 8.0)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, reference_surface_height_gate_margin_mm: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                      </label>
                      <label>Min coverage
                        <input type="number" min={0.01} max={0.95} step={0.01} value={Number(planeQaParams.reference_surface_height_gate_min_coverage_ratio ?? 0.10)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, reference_surface_height_gate_min_coverage_ratio: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                      </label>
                      <label>Max coverage
                        <input type="number" min={0.05} max={0.99} step={0.01} value={Number(planeQaParams.reference_surface_height_gate_max_coverage_ratio ?? 0.95)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, reference_surface_height_gate_max_coverage_ratio: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                      </label>
                      <label>Gap floor mm
                        <input type="number" min={0} max={100} step={0.1} value={Number(planeQaParams.reference_surface_height_gate_gap_floor_mm ?? 1.0)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, reference_surface_height_gate_gap_floor_mm: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                      </label>
                      <label>Gap ratio
                        <input type="number" min={1} max={100} step={0.5} value={Number(planeQaParams.reference_surface_height_gate_gap_ratio ?? 8.0)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, reference_surface_height_gate_gap_ratio: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                      </label>
                    </div>
                  </fieldset>
                  <fieldset>
                    <legend>Plot rendering</legend>
                    <div className="segmentation-controls-row">
                      <label>Depth dots cap
                        <input type="number" min={1000} max={1000000} step={1000} value={Number(planeQaParams.plot_depth_plot_max_render_samples ?? 60000)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, plot_depth_plot_max_render_samples: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                      </label>
                      <label>Y robust pct
                        <input type="number" min={50} max={100} step={0.5} value={Number(planeQaParams.plot_y_robust_percentile ?? 98.0)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, plot_y_robust_percentile: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                      </label>
                    </div>
                  </fieldset>
                </div>
                <div className="segmentation-controls-row">
                  <button type="button" disabled={pipelineActionBusy} onClick={() => applyPlaneQaParams(false)}>Apply</button>
                  <button type="button" disabled={pipelineActionBusy} onClick={() => applyPlaneQaParams(true)}>Apply + rerun</button>
                </div>
              </details>
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
                    <select value={String(planeQaParams.reference_surface_region_mode ?? "none")} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, reference_surface_region_mode: e.target.value, plane_fit_roi_type: e.target.value === "full_height_x_band" ? "full_height_x_band" : e.target.value === "none" ? "rectangle" : e.target.value } : c); setPlaneQaDirty(true); }} >
                    <option value="none">No ROI / full frame</option>
                    <option value="full_height_x_band">full_height_x_band</option>
                    <option value="rectangle">rectangle</option>
                    <option value="polygon">polygon</option>
                  </select>
                </label>
                <label>X
                  <input type="number" min={0} step={1} value={Number(planeQaParams.plane_fit_roi_x ?? 0)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, plane_fit_roi_x: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                </label>
                {String(planeQaParams.reference_surface_region_mode ?? "none") !== "full_height_x_band" && (
                  <label>Y
                    <input type="number" min={0} step={1} value={Number(planeQaParams.plane_fit_roi_y ?? 0)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, plane_fit_roi_y: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                  </label>
                )}
                <label>W
                  <input type="number" min={1} step={1} value={Number(planeQaParams.plane_fit_roi_width ?? 0)} onChange={(e) => { setPlaneQaParams((c) => c ? { ...c, plane_fit_roi_width: Number(e.target.value) } : c); setPlaneQaDirty(true); }} />
                </label>
                {String(planeQaParams.reference_surface_region_mode ?? "none") !== "full_height_x_band" && (
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
              {String(planeQaParams.reference_surface_region_mode ?? "none") === "full_height_x_band" && (
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
        acquisitionContext={{
          dataset: selectedContextDatasetName || null,
          session: selectedContextSessionName || null,
          sessionType: selectedContextSessionType || null,
          sessionTags: selectedSessionTags,
        }}
        onUpsertObjectAnnotation={(payload) => void upsertObjectAnnotation(payload)}
      />
      <details className="studio-processing-job-monitor" open>
        <summary>
          Processing jobs
          <span className="studio-processing-job-monitor-counts">
            {processingJobStatusCounts.running} running · {processingJobStatusCounts.queued} queued · {processingJobStatusCounts.completed} completed · {processingJobStatusCounts.failed + processingJobStatusCounts.partial} issues
          </span>
        </summary>
        <div className="studio-processing-job-monitor-body">
          {processingJobsLoading && !processingJobs.length && <small>Loading jobs…</small>}
          {!processingJobs.length && !processingJobsLoading && <small>No processing jobs yet.</small>}
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
      </details>
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
