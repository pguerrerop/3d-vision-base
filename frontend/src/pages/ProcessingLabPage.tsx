import { useCallback, useEffect, useMemo, useState } from "react";
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
  type SessionSummary,
  type TakeDetail,
  type TakeSummary,
} from "../api/client";
import PipelineStatusPanel from "../components/PipelineStatusPanel";
import PipelineExecutionGraph from "../components/PipelineExecutionGraph";
import StudioInspector from "../components/StudioInspector";
import type { OverlayDebugInfo } from "../components/overlayModel";
import {
  chooseBestPipelineForTake,
  compactTakeFamilyStatus,
  incompatibleModalityMessage,
  modalityCompatible,
  stageTimingMs,
  type StageViewTabId
} from "../components/pipelineModel";
import { artifactsForObject, artifactsForStage, objectsForTake, selectedObject } from "../components/studioWorkspaceModel";
import { renderStageSemanticView } from "../components/stage_view_renderers";
import { canonicalStageId, stageSemanticDefinition, stageSemanticSummary } from "../components/stageSemantics";
import type { RoiPolygon, RoiRect } from "../components/roiMapping";
import { nextDatasetSelectionAfterCreate, nextSessionSelectionAfterCreate } from "../components/datasetSidebarModel";
import { buildCapturePayload, captureDisabledReason, nextSelectedTakeAfterCapture } from "../components/captureWorkflowModel";

type TakeFilter = "all" | "processed" | "unprocessed" | "warnings";

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
  const [overlayDebug, setOverlayDebug] = useState<OverlayDebugInfo | null>(null);
  const [selectedTemplateId, setSelectedTemplateId] = useState("mining_steel_ball_classification_2d_reflectance_mvp");
  const [selectedInputType, setSelectedInputType] = useState("rgb_image");
  const [selectedPipelineInstanceId, setSelectedPipelineInstanceId] = useState<string>("");
  const [manualImagePath, setManualImagePath] = useState("");
  const [pipelineSearch, setPipelineSearch] = useState("");
  const [pipelineFamilyFilter, setPipelineFamilyFilter] = useState<"all" | "2d" | "3d" | "generic">("all");
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

  const load = useCallback(async () => {
    const [nextTakes, nextSessions, nextPipelines, nextRuntime, nextTemplates, nextInstances, nextDatasets] = await Promise.all([
      api.filteredTakes({
        session_id: selectedSession || undefined,
        dataset_id: selectedDataset || undefined,
        validation_status: validationFilter === "all" ? undefined : validationFilter,
        tag: tagFilter || undefined,
        search: takeSearch || undefined,
      }),
      api.sessions(),
      api.pipelines(),
      api.state(),
      api.processTemplates(),
      api.pipelineInstances(),
      api.datasets(),
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
    setSelectedPipelineInstanceId((current) => current || nextInstances[0]?.id || "");
    setSelectedPipelineId((current) => current || nextPipelines[0]?.id || "3d_ball_inspection");
    const nextTakeId = selectedTakeId || nextTakes[0]?.take_id || "";
    setSelectedTakeId(nextTakeId);
    setDetail(nextTakeId ? await api.take(nextTakeId) : null);
  }, [selectedSession, selectedDataset, selectedTakeId, validationFilter, tagFilter, takeSearch]);

  useEffect(() => {
    void load();
  }, [load]);

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
  const allObjects = objectsForTake(detail);
  const focusedObject = selectedObject(allObjects, selectedObjectId);
  const allArtifacts = artifactsForStage(detail, canonicalSelectedStageId);
  const currentArtifacts = artifactsForObject(allArtifacts, selectedObjectId);
  const focusedArtifact = currentArtifacts.find((artifact) => artifact.artifact_id === selectedArtifactId) ?? currentArtifacts[0] ?? null;
  const templateById = useMemo(() => new Map(processTemplates.map((template) => [template.id, template])), [processTemplates]);
  const thresholdStep = selectedPipelineInstance?.configured_steps.find((step) => step.step_id === "threshold");
  const blobStep = selectedPipelineInstance?.configured_steps.find((step) => step.step_id === "blob_detection");
  const roiEnabled = Boolean(thresholdStep?.params?.roi_enabled);
  const roiType = (String(thresholdStep?.params?.roi_type ?? "rectangle") === "polygon" ? "polygon" : "rectangle") as "rectangle" | "polygon";
  const activeRoi = roiEnabled
    ? {
      x: Number(thresholdStep?.params?.roi_x ?? 0),
      y: Number(thresholdStep?.params?.roi_y ?? 0),
      width: Number(thresholdStep?.params?.roi_width ?? 0),
      height: Number(thresholdStep?.params?.roi_height ?? 0),
    }
    : null;
  const activeRoiPolygon = roiEnabled && roiType === "polygon"
    ? (((thresholdStep?.params?.roi_polygon_points as Array<[number, number]> | undefined) ?? []).map((point) => [Number(point[0]), Number(point[1])] as [number, number]))
    : null;
  useEffect(() => {
    if (focusedArtifact?.object_id != null && focusedArtifact.object_id !== selectedObjectId) {
      setSelectedObjectId(focusedArtifact.object_id);
    }
  }, [focusedArtifact?.artifact_id]);
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

  async function runSelectedPipeline(reprocess = false) {
    if (!detail || !selectedPipeline) return;
    const response = await api.processTake(detail.take_id, { pipeline_id: selectedPipeline.id, reprocess }) as { pipeline_instance_id?: string; run_id?: string };
    if (response.pipeline_instance_id) {
      setSelectedPipelineInstanceId(response.pipeline_instance_id);
    }
    await load();
    const refreshed = await api.take(detail.take_id);
    setDetail(refreshed);
    if (response.run_id) {
      setSelectedArtifactId(null);
      setSelectedObjectId(null);
    }
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
    if (!selectedPipelineInstance) return;
    const configured_steps = selectedPipelineInstance.configured_steps.map((step) =>
      step.step_id === "threshold"
        ? {
          ...step,
          params: {
            ...step.params,
            roi_enabled: true,
            roi_type: "rectangle",
            roi_x: roi.x,
            roi_y: roi.y,
            roi_width: roi.width,
            roi_height: roi.height,
          },
        }
        : step
    );
    const next = await api.updatePipelineInstance(selectedPipelineInstance.id, { configured_steps });
    setPipelineInstances((items) => items.map((item) => (item.id === next.id ? next : item)));
    if (rerun && detail && selectedPipeline) {
      await runSelectedPipeline(false);
    }
  }

  async function applyPolygonRoi(points: RoiPolygon, rerun = false) {
    if (!selectedPipelineInstance) return;
    const configured_steps = selectedPipelineInstance.configured_steps.map((step) =>
      step.step_id === "threshold"
        ? {
          ...step,
          params: {
            ...step.params,
            roi_enabled: true,
            roi_type: "polygon",
            roi_polygon_points: points.map(([x, y]) => [Math.round(x), Math.round(y)]),
          },
        }
        : step
    );
    const next = await api.updatePipelineInstance(selectedPipelineInstance.id, { configured_steps });
    setPipelineInstances((items) => items.map((item) => (item.id === next.id ? next : item)));
    if (rerun && detail && selectedPipeline) {
      await runSelectedPipeline(false);
    }
  }

  async function clearRoi() {
    if (!selectedPipelineInstance) return;
    const configured_steps = selectedPipelineInstance.configured_steps.map((step) =>
      step.step_id === "threshold"
        ? {
          ...step,
          params: {
            ...step.params,
            roi_enabled: false,
            roi_type: "rectangle",
            roi_polygon_points: [],
          },
        }
        : step
    );
    const next = await api.updatePipelineInstance(selectedPipelineInstance.id, { configured_steps });
    setPipelineInstances((items) => items.map((item) => (item.id === next.id ? next : item)));
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
              <small>{take.take_id}</small>
              {take.thumbnail_path && <img alt={take.friendly_name || take.take_id} className="take-thumb" src={fileUrl(take.take_id, take.thumbnail_path)} />}
              <small>{take.tags?.join(", ") || "-"}</small>
              <small>{take.modalities?.join(", ") || "no modalities"}</small>
              <small>Validation: {take.validation_status ?? "unreviewed"}</small>
              <small>Latest run: {take.latest_run_status ?? "never"}</small>
              <small>{compactTakeFamilyStatus(take.processing_by_family, take.modalities)}</small>
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
        <header className="studio-workspace-header">
          <div>
            <span className="eyebrow">Selected take</span>
            <h1>{detail?.take_id ?? "No take selected"}</h1>
            <p>{detail?.modalities?.join(", ") || "Select a take to inspect modalities, stages, artifacts, and results."}</p>
          </div>
          <div className="studio-actions">
            <button disabled={!compatible} title={!compatible ? "Cannot execute: incompatible modalities." : ""} onClick={() => void runSelectedPipeline(false)} type="button">Run pipeline</button>
            <button
              disabled={!compatible || !selectedStage || !selectedPipeline?.supports_partial_stages}
              title={
                !compatible
                  ? "Cannot execute: incompatible modalities."
                  : !selectedPipeline?.supports_partial_stages
                    ? "This pipeline does not support partial stage execution."
                    : ""
              }
              onClick={() => void runSelectedPipeline(false)}
              type="button"
            >
              Run until selected stage
            </button>
            <button disabled={!detail || !compatible} title={!compatible ? "Cannot execute: incompatible modalities." : ""} onClick={() => void runSelectedPipeline(true)} type="button">Reprocess</button>
            <button disabled={!detail} onClick={() => void clearSelectedOutputs()} type="button">Clear outputs</button>
          </div>
          {!compatible && selectedPipeline && (
            <p>{incompatibleModalityMessage(selectedPipeline, detail?.modalities) ?? "Cannot execute: incompatible modalities."}</p>
          )}
        </header>

        <PipelineStatusPanel
          compact
          result={detail?.result ?? null}
          runtimeState={runtimeState}
          detail={detail}
          pipelines={pipelines}
          selectedPipelineId={selectedPipelineId}
          onSelectPipeline={selectPipeline}
          pipelineSearch={pipelineSearch}
          onPipelineSearchChange={setPipelineSearch}
          pipelineFamilyFilter={pipelineFamilyFilter}
          onPipelineFamilyFilterChange={setPipelineFamilyFilter}
        />

        <section className="studio-stage-strip" aria-label="Stage navigator">
          {(selectedPipeline?.stages ?? []).map((stage, index) => {
            const timing = stageTimingMs(detail?.result, stage.id);
            const stageArtifacts = artifactsForStage(detail, canonicalStageId(stage.id, stage.display_name));
            const artifactCount = stageArtifacts.length;
            const execStage = detail?.result?.pipeline_execution?.stages?.find((item) => item.stage_id === stage.id);
            const semanticSummary = stageSemanticSummary(canonicalStageId(stage.id, stage.display_name), detail, stageArtifacts, execStage);
            return (
              <button className={stage.id === selectedStageId ? "active" : ""} key={stage.id} onClick={() => selectStage(stage.id)} type="button">
                <span>{index + 1}</span>
                <strong>{stage.display_name}</strong>
                <small>{timing == null ? "not run" : `${timing.toFixed(timing >= 100 ? 0 : 1)} ms`}</small>
                <small>{semanticSummary}</small>
                <small>{artifactCount} artifacts • {execStage?.warnings?.length ?? 0} warnings • {execStage?.errors?.length ?? 0} errors</small>
              </button>
            );
          })}
        </section>
        <section className="concept-panel compact studio-run-summary">
          <div className="concept-heading">
            <span className="eyebrow">Run context</span>
            <strong>
              {!selectedRunTrace ? "Never processed" : `${selectedRunTrace.id} • ${selectedRunTrace.status} • ${selectedRunTrace.duration.toFixed(1)} ms`}
            </strong>
          </div>
          <div className="pipeline-status-grid">
            <div><span>Pipeline</span><strong>{selectedPipeline?.display_name ?? "-"}</strong></div>
            <div><span>Run</span><strong>{selectedRunTrace?.id ?? "-"}</strong></div>
            <div><span>Status</span><strong>{selectedRunTrace?.status ?? familyStatus?.status ?? "-"}</strong></div>
            <div><span>Timestamp</span><strong>{selectedRunTrace?.timestamp ?? familyStatus?.lastRunAt ?? "-"}</strong></div>
          </div>
          <details>
            <summary>Trace details</summary>
            <PipelineExecutionGraph execution={detail?.result?.pipeline_execution} selectedStageId={canonicalSelectedStageId} />
          </details>
        </section>

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
            <span className="eyebrow">Stage workspace</span>
            <strong>{selectedStage?.display_name ?? "Workspace"}</strong>
          </div>
          {renderStageSemanticView({
            detail,
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
            roi: activeRoi,
            roiPolygon: activeRoiPolygon,
            roiType,
            roiEnabled,
            onApplyRoi: (roi, rerun) => void applyRoi(roi, rerun),
            onApplyPolygonRoi: (points, rerun) => void applyPolygonRoi(points, rerun),
            onClearRoi: () => void clearRoi(),
            blobParams: blobStep?.params ?? null,
            onApplyBlobParams: (params, rerun) => void applyBlobParams(params, rerun),
            onUpsertObjectAnnotation: (payload) => void upsertObjectAnnotation(payload),
          })}
        </section>
        </section>
      </section>

      <StudioInspector
        compatible={compatible}
        detail={detail}
        pipeline={selectedPipeline}
        selectedArtifact={focusedArtifact}
        overlayDebug={overlayDebug}
        selectedObjectId={focusedObject?.object_id ?? null}
        stageId={canonicalSelectedStageId}
        onUpsertObjectAnnotation={(payload) => void upsertObjectAnnotation(payload)}
      />
    </main>
  );
}
