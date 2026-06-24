import { useEffect, useMemo, useState } from "react";
import { fileUrl, type PipelineInfo, type TakeDetail } from "../api/client";
import type { OverlayDebugInfo } from "./overlayModel";
import { incompatibleModalityMessage } from "./pipelineModel";
import { canonicalStageId, stageSemanticDefinition } from "./stageSemantics";
import { primarySourceImageUrl } from "./stage_sources";
import { normalizeBlobDetectionArtifacts } from "../studio/blobDetectionModel";
import { artifactLineage, objectGeneratingStages, selectedObject, stageInspectorSummary, type StudioArtifact, type StudioObject } from "./studioWorkspaceModel";
import type { HoverInspectionSample } from "./hoverInspectionModel";
import HeightLegend from "./HeightLegend";
import { resolveHeightColorMapping, toHeightLegendSemantic } from "./heightSemantics";

type Props = {
  detail: TakeDetail | null;
  pipeline: PipelineInfo | null;
  stageId: string;
  compatible: boolean;
  selectedObjectId: number | null;
  selectedArtifact: StudioArtifact | null;
  overlayDebug: OverlayDebugInfo | null;
  hoverSample?: HoverInspectionSample | null;
  acquisitionContext?: {
    dataset?: string | null;
    session?: string | null;
    sessionType?: string | null;
    sessionTags?: string[];
  };
  onUpsertObjectAnnotation?: (payload: Record<string, unknown>) => void;
};

export default function StudioInspector({ detail, pipeline, stageId, compatible, selectedObjectId, selectedArtifact, overlayDebug, hoverSample = null, acquisitionContext, onUpsertObjectAnnotation }: Props) {
  const objects = [...(detail?.result?.objects ?? []), ...(detail?.result?.rejected_objects ?? [])];
  const objectCandidates = detail?.result?.object_candidates ?? [];
  const artifacts = detail?.result?.artifacts ?? [];
  const object = selectedObject(objects, selectedObjectId);
  const summary = stageInspectorSummary(detail, pipeline, stageId);
  const semantic = stageSemanticDefinition(stageId);
  const lineage = artifactLineage(artifacts, selectedArtifact);
  const producingStages = objectGeneratingStages(artifacts, object?.object_id ?? null);
  const stageArtifacts = artifactsForStageLocal(artifacts, stageId);
  const beltPlaneArtifact = artifacts.find((item) => item.artifact_id === "belt_plane");
  const segmentationArtifact = artifacts.find((item) => item.artifact_id === "height_segmentation");
  const componentsArtifact = artifacts.find((item) => item.artifact_id === "connected_components");
  const normalizedBlobs = normalizeBlobDetectionArtifacts(stageArtifacts);
  const blobCandidates = normalizedBlobs.candidates;
  const selectedBlob = blobCandidates.find((item) => Number(item.id) === selectedObjectId) ?? null;
  const blobRejected = normalizedBlobs.summary.rejectedCount;
  const ellipseArtifact = stageArtifacts.find((item) => item.artifact_id === "ellipse_metrics");
  const ellipseDebugArtifact = stageArtifacts.find((item) => item.artifact_id === "ellipse_debug_json");
  const ellipseRows = (((ellipseArtifact?.metadata as Record<string, unknown> | undefined)?.entries ?? []) as unknown[])
    .filter((item): item is Record<string, unknown> => typeof item === "object" && item !== null);
  const ellipseEffectiveParams = ((ellipseDebugArtifact?.metadata as Record<string, unknown> | undefined)?.effective_params as Record<string, unknown> | undefined)
    ?? ((ellipseArtifact?.metadata as Record<string, unknown> | undefined)?.effective_params as Record<string, unknown> | undefined)
    ?? {};
  const selectedEllipse = ellipseRows.find((row) => Number(row.candidate_id ?? 0) === selectedObjectId) ?? null;
  const annotationMap = useMemo(() => {
    const map = new Map<string, Record<string, unknown>>();
    const entries = detail?.object_annotations;
    if (!Array.isArray(entries)) return map;
    for (const item of entries) {
      if (item && typeof item === "object") {
        const row = item as Record<string, unknown>;
        const key = String(row.matched_candidate_id ?? row.candidate_id ?? "");
        if (key) map.set(key, row);
      }
    }
    return map;
  }, [detail?.object_annotations]);
  const selectedCandidateId = selectedObjectId != null ? String(selectedObjectId) : null;
  const selectedAnnotation = selectedCandidateId ? (annotationMap.get(selectedCandidateId) ?? null) : null;
  const morphMetrics = stageArtifacts.find((item) => item.artifact_id === "morphology_metrics");
  const morphDebug = stageArtifacts.find((item) => item.artifact_id === "morphology_debug_json");
  const thresholdMask = stageArtifacts.find((item) => item.artifact_id === "threshold_mask");
  const planeFitDebugArtifact = stageArtifacts.find((item) => item.artifact_id === "plane_fit_debug");
  const normalizationDebugArtifact = stageArtifacts.find((item) => item.artifact_id === "normalization_debug" || item.artifact_id === "normalized_height_debug");
  const planeFitDebug = (planeFitDebugArtifact?.metadata ?? {}) as Record<string, unknown>;
  const normalizationDebug = (normalizationDebugArtifact?.metadata ?? {}) as Record<string, unknown>;
  const bgSelection = ((planeFitDebug.background_selection as Record<string, unknown> | undefined) ?? {}) as Record<string, unknown>;
  const morphMetricsMeta = (morphMetrics?.metadata ?? {}) as Record<string, unknown>;
  const morphDebugMeta = (morphDebug?.metadata ?? {}) as Record<string, unknown>;
  const thresholdMeta = (thresholdMask?.metadata ?? {}) as Record<string, unknown>;
  const morphSection = (morphDebugMeta.morphology as Record<string, unknown> | undefined) ?? {};
  const roiSection = (morphDebugMeta.roi as Record<string, unknown> | undefined) ?? {};
  const effectiveParams = (morphDebugMeta.effective_params as Record<string, unknown> | undefined)
    ?? (morphMetricsMeta.effective_params as Record<string, unknown> | undefined)
    ?? {};
  const inspectorTitle = object
    ? `Object #${object.object_id}`
    : selectedArtifact?.title
      ? selectedArtifact.title
        : detail?.result
          ? stageId || "Stage"
          : "No result selected";
  const selectedCandidate = selectedObjectId == null
    ? null
    : objectCandidates.find((item) => Number(item.local_object_index) === selectedObjectId) ?? null;
  const hasClassificationExplanationArtifact = artifacts.some((item) => item.artifact_id === "classification_explanation");
  const runtimeDiagnosticsArtifact = artifacts.find((item) => item.artifact_id === "classification_runtime_diagnostics") ?? null;
  const explanationArtifact = artifacts.find((item) => item.artifact_id === "classification_explanation") ?? null;
  const explanationRows = Array.isArray((explanationArtifact?.metadata as Record<string, unknown> | undefined)?.objects)
    ? (((explanationArtifact?.metadata as Record<string, unknown> | undefined)?.objects as unknown[])
      .filter((item): item is Record<string, unknown> => typeof item === "object" && item !== null))
    : [];
  const explanationByObjectId = new Map<number, Record<string, unknown>>();
  for (const row of explanationRows) {
    const oid = Number(row.object_id ?? 0);
    if (Number.isFinite(oid) && oid > 0) explanationByObjectId.set(oid, row);
  }
  const artifactMeta = ((selectedArtifact?.metadata ?? {}) as Record<string, unknown>);
  const [runtimeDiagnostics, setRuntimeDiagnostics] = useState<Record<string, unknown> | null>(null);
  useEffect(() => {
    async function loadRuntimeDiagnostics() {
      if (!detail?.take_id || !runtimeDiagnosticsArtifact?.path) {
        setRuntimeDiagnostics(null);
        return;
      }
      try {
        const response = await fetch(fileUrl(detail.take_id, String(runtimeDiagnosticsArtifact.path)));
        if (!response.ok) {
          setRuntimeDiagnostics(null);
          return;
        }
        const payload = await response.json();
        setRuntimeDiagnostics(payload as Record<string, unknown>);
      } catch {
        setRuntimeDiagnostics(null);
      }
    }
    void loadRuntimeDiagnostics();
  }, [detail?.take_id, runtimeDiagnosticsArtifact?.path]);
  const transformMeta = (artifactMeta.transform && typeof artifactMeta.transform === "object") ? (artifactMeta.transform as Record<string, unknown>) : null;
  // Resolve the canonical HeightColorMapping for the current take so we can
  // render the colorbar on every stage (not only when a height image is the
  // selected preview). Falls back to a legacy semantic range when no metadata
  // is available, so the legend is always actionable.
  const heightColorMappingResolution = useMemo(
    () => resolveHeightColorMapping(artifacts, {
      semanticField: String(artifactMeta.semantic_field ?? "height_above_belt"),
      preferredArtifactId: selectedArtifact?.artifact_id ?? null,
    }),
    [artifacts, artifactMeta.semantic_field, selectedArtifact?.artifact_id],
  );
  const heightColorMapping = heightColorMappingResolution.mapping;
  const inspectorLegendSemantic = useMemo(
    () => toHeightLegendSemantic(artifactMeta) ?? null,
    [artifactMeta],
  );
  return (
    <aside className="studio-inspector">
      <div className="studio-sidebar-title">
        <span className="eyebrow">Inspector</span>
        <strong>{inspectorTitle}</strong>
      </div>
      <div className="inspector-section">
        <span>Acquisition context</span>
        <strong>{acquisitionContext?.dataset || (detail?.take_metadata?.dataset_id as string | undefined) || "-"}</strong>
        <small>Session: {acquisitionContext?.session || (detail?.take_metadata?.session_id as string | undefined) || "-"}</small>
        <small>Type: {acquisitionContext?.sessionType || "-"}</small>
        <small>Reference: {String((detail?.take_metadata?.is_reference as boolean | undefined) ? "yes" : "no")} | Golden: {String((detail?.take_metadata?.is_golden_sample as boolean | undefined) ? "yes" : "no")}</small>
        <small>Categories: {Array.isArray(detail?.take_metadata?.categories) && detail?.take_metadata?.categories.length ? (detail?.take_metadata?.categories as string[]).join(", ") : "-"}</small>
      </div>
      <div className="inspector-section">
        <span>Compatibility</span>
        <strong>{compatible ? "Compatible" : "Incompatible"}</strong>
        {!compatible && pipeline && <small>{incompatibleModalityMessage(pipeline, detail?.modalities)}</small>}
      </div>
      <div className="inspector-section">
        <span>Stage diagnostics</span>
        <strong>{summary.status.toUpperCase()}</strong>
        <small>Duration: {summary.timingMs == null ? "not run" : `${summary.timingMs.toFixed(1)} ms`}</small>
        <small>Inputs: {summary.inputArtifactCount} | Outputs: {summary.outputArtifactCount}</small>
        <small>Objects: {summary.objectCount}; rejected: {summary.rejectedCount}</small>
      </div>
      {semantic.category === "classification" && (
        <div className="inspector-section">
          <span>Runtime diagnostics</span>
          {(() => {
            const diag = runtimeDiagnostics ?? {};
            const fallback = Boolean(diag.fallback_used);
            const comp = (diag.compatibility_checks as Record<string, unknown> | undefined) ?? {};
            const errors = Array.isArray(comp.errors) ? comp.errors.map(String) : [];
            const warnings = Array.isArray(comp.warnings) ? comp.warnings.map(String) : [];
            const depStatus = String((((diag.deployment_resolution as Record<string, unknown> | undefined)?.deployment as Record<string, unknown> | undefined)?.status ?? ""));
            const statusBadge = fallback
              ? "FALLBACK"
              : depStatus === "invalid"
                ? "INVALID MODEL"
                : String(diag.inference_backend_used ?? "") === "heuristic"
                  ? "HEURISTIC"
                  : errors.length
                    ? "INCOMPATIBLE"
                    : "ACTIVE MODEL";
            return (
              <>
                <strong>{statusBadge}</strong>
                <small>Deployment: {String(((diag.deployment_resolution as Record<string, unknown> | undefined)?.deployment as Record<string, unknown> | undefined)?.id ?? "-")}</small>
                <small>Model: {String(((diag.deployment_resolution as Record<string, unknown> | undefined)?.model as Record<string, unknown> | undefined)?.id ?? "-")} / v{String(((diag.deployment_resolution as Record<string, unknown> | undefined)?.model as Record<string, unknown> | undefined)?.version ?? "-")}</small>
                <small>Backend: {String(diag.inference_backend_used ?? "-")}</small>
                <small>Inference ms: {String(diag.inference_duration_ms ?? "-")}</small>
                <small>Fallback used: {String(fallback)}</small>
                <small>Fallback reason: {String(diag.fallback_reason ?? "-")}</small>
                <small>Feature schema version/hash: {String(diag.feature_schema_used ?? "-")} / {String(diag.feature_schema_hash ?? "-")}</small>
                <small>Errors: {errors.length ? errors.join(" | ") : "-"}</small>
                <small>Warnings: {warnings.length ? warnings.join(" | ") : "-"}</small>
                <small>Calibration compatibility: {JSON.stringify(((comp.checks as Record<string, unknown> | undefined)?.calibration ?? "-"))}</small>
              </>
            );
          })()}
        </div>
      )}
      <div className="inspector-section">
        <span>Stage semantics</span>
        <strong>{semantic.category}</strong>
        {semantic.category === "segmentation" && <small>Primary visualization uses height preview overlays; masks/morphology are diagnostics.</small>}
        {semantic.category === "geometry" && <small>Contour and ellipse candidate geometry metrics are emphasized for this stage.</small>}
        {semantic.category === "measurement" && <small>Diameter and circularity measurements are emphasized for this stage.</small>}
        {semantic.category === "classification" && <small>Class distribution and rejection reasons are emphasized for this stage.</small>}
        {semantic.category === "input" && <small>Primary visualization prefers human-readable source previews (height preview for 2.5D).</small>}
        {semantic.category === "plane_qa" && <small>Plane-fit quality and near-zero normalization diagnostics are prioritized.</small>}
      </div>
      <div className="inspector-section">
        <span>Object candidates</span>
        <strong>{objectCandidates.length}</strong>
        {selectedCandidate ? (
          <>
            <small>ID: {selectedCandidate.id}</small>
            <small>Source modality: {selectedCandidate.source_modality}</small>
            <small>Confidence: {selectedCandidate.confidence == null ? "-" : String(selectedCandidate.confidence)}</small>
            <small>BBox: {Array.isArray(selectedCandidate.bbox_px) ? selectedCandidate.bbox_px.join(", ") : "-"}</small>
          </>
        ) : (
          <small>Select an object to inspect candidate details.</small>
        )}
      </div>
      {semantic.category === "segmentation" && (
        <div className="inspector-section">
          <span>Morphology diagnostics</span>
          <strong>{String(morphSection.components_after ?? morphMetricsMeta.connected_components ?? "-")} connected components</strong>
          <small>Components before/after: {String(morphSection.components_before ?? morphMetricsMeta.components_before ?? "-")} / {String(morphSection.components_after ?? morphMetricsMeta.components_after ?? "-")}</small>
          <small>Removed components: {String(morphSection.removed_components ?? morphMetricsMeta.removed_components ?? "-")}</small>
          <small>Foreground coverage: {String(morphMetricsMeta.foreground_coverage_percent ?? "-")}%</small>
          <small>Threshold fg coverage: {String(morphSection.threshold_foreground_coverage ?? morphMetricsMeta.threshold_foreground_coverage ?? "-")}</small>
          <small>Cleaned fg coverage: {String(morphSection.cleaned_foreground_coverage ?? morphMetricsMeta.cleaned_foreground_coverage ?? "-")}</small>
          <small>Threshold: {String(thresholdMeta.threshold_mode ?? "-")} @ {String(thresholdMeta.threshold_value ?? "-")}</small>
          <small>Invert: {String(thresholdMeta.invert ?? "-")}</small>
          <small>Blur kernel: {String(thresholdMeta.blur_kernel ?? "-")}</small>
          <small>Morph op: {String(morphSection.operation ?? "-")}</small>
          <small>Erode/Dilate/Close kernels: {String(morphSection.erode_kernel_size ?? "-")} / {String(morphSection.dilate_kernel_size ?? "-")} / {String(morphSection.close_kernel_size ?? "-")}</small>
          <small>Erode/Dilate iterations: {String(morphSection.erode_iterations ?? "-")} / {String(morphSection.dilate_iterations ?? "-")}</small>
          <small>Area min/max: {String(morphSection.min_area_px ?? "-")} / {String(morphSection.max_area_px ?? "-")}</small>
          <small>ROI: {String(roiSection.enabled ?? false)} ({String(roiSection.x ?? "-")}, {String(roiSection.y ?? "-")}, {String(roiSection.width ?? "-")}x{String(roiSection.height ?? "-")})</small>
          {!!Object.keys(effectiveParams).length && <small>Effective params: {JSON.stringify(effectiveParams)}</small>}
        </div>
      )}
      {semantic.category === "input" && (
        <div className="inspector-section">
          <span>Source context</span>
          <strong>{detail?.modalities?.join(", ") || "no modalities"}</strong>
          <small>Timestamp: {detail?.created_at ?? "-"}</small>
          <small>Source image: {primarySourceImageUrl(detail) ?? "-"}</small>
          <small>Metadata keys: {Object.keys((detail?.metadata ?? {}) as Record<string, unknown>).join(", ") || "-"}</small>
        </div>
      )}
      {semantic.category === "plane_qa" && (
        <div className={`inspector-section ${(Array.isArray(planeFitDebug.warnings) && planeFitDebug.warnings.length) || (Array.isArray(normalizationDebug.warnings) && normalizationDebug.warnings.length) ? "warning" : ""}`}>
          <span>Plane QA</span>
          <strong>{String(planeFitDebug.status ?? "unknown")}</strong>
          <small>Strategy: {String((bgSelection.background_detection_strategy ?? planeFitDebug.background_detection_strategy) ?? "-")}</small>
          <small>Reference model: {String(planeFitDebug.reference_surface_model_type ?? "-")}</small>
          <small>Selection reason: {String(planeFitDebug.model_selection_reason ?? "-")}</small>
          <small>Selected component: {String((((bgSelection.gradient_debug as Record<string, unknown> | undefined) ?? {}).selected_component_id) ?? "-")}</small>
          <small>Selected area ratio: {String((((bgSelection.gradient_debug as Record<string, unknown> | undefined) ?? {}).selected_component_area_ratio) ?? "-")}</small>
          <small>Selected z std/mm: {String((((bgSelection.gradient_debug as Record<string, unknown> | undefined) ?? {}).selected_component_z_std_mm) ?? "-")}</small>
          <small>Selected z mad/mm: {String((((bgSelection.gradient_debug as Record<string, unknown> | undefined) ?? {}).selected_component_z_mad_mm) ?? "-")}</small>
          <small>Selected gradient p95: {String((((bgSelection.gradient_debug as Record<string, unknown> | undefined) ?? {}).selected_component_gradient_p95) ?? "-")}</small>
          <small>Selection mode: {String(bgSelection.background_selection_mode ?? "-")}</small>
          <small>Inferred convention: {String(bgSelection.inferred_depth_convention ?? "-")}</small>
          <small>Candidate coverage: {String(bgSelection.candidate_coverage_percent ?? "-")}%</small>
          <small>Border-connected: {String((((bgSelection.border_connected_filtering as Record<string, unknown> | undefined) ?? {}).enabled) ?? "-")}</small>
          <small>Failure reason: {String(planeFitDebug.failure_reason ?? "-")}</small>
          <small>Inlier ratio: {String(planeFitDebug.inlier_ratio ?? "-")}</small>
          <small>Residual p95/mm: {String(planeFitDebug.residual_p95_mm ?? "-")}</small>
          <small>Residual max abs/mm: {String(planeFitDebug.residual_max_abs_mm ?? "-")}</small>
          <small>Background mean/mm: {String(normalizationDebug.background_height_mean_after_normalization_mm ?? "-")}</small>
          <small>Background std/mm: {String(normalizationDebug.background_height_std_after_normalization_mm ?? "-")}</small>
          <small>Background p95 abs/mm: {String(normalizationDebug.background_height_p95_abs_after_normalization_mm ?? "-")}</small>
          <small>Foreground mean/mm: {String(normalizationDebug.foreground_mean_height_mm ?? "-")}</small>
          {Array.isArray(planeFitDebug.warnings) && planeFitDebug.warnings.length ? <small>Fit warnings: {planeFitDebug.warnings.map(String).join(" | ")}</small> : null}
          {Array.isArray(normalizationDebug.warnings) && normalizationDebug.warnings.length ? <small>Normalization warnings: {normalizationDebug.warnings.map(String).join(" | ")}</small> : null}
          <small>Convention: height_above_belt_mm = raw_z_mm - plane_z_at_xy</small>
        </div>
      )}
      {(beltPlaneArtifact || segmentationArtifact || componentsArtifact) && (
        <div className="inspector-section">
          <span>25D diagnostics</span>
          <strong>Belt plane + segmentation</strong>
          <small>Plane coefficients: {JSON.stringify(((beltPlaneArtifact?.metadata as Record<string, unknown> | undefined)?.plane_coefficients) ?? detail?.result?.plane_model ?? "-")}</small>
          <small>Plane residual stats: {JSON.stringify(((beltPlaneArtifact?.metadata as Record<string, unknown> | undefined)?.residual_stats_mm) ?? "-")}</small>
          <small>Height threshold config: {JSON.stringify(((segmentationArtifact?.metadata as Record<string, unknown> | undefined)?.height_threshold_config) ?? "-")}</small>
          <small>Connected components: {String(((componentsArtifact?.metadata as Record<string, unknown> | undefined)?.component_count) ?? detail?.result?.objects?.length ?? "-")}</small>
        </div>
      )}
      <ObjectInspector object={object} producingStages={producingStages} explanationByObjectId={explanationByObjectId} />
      {!object && semantic.category === "geometry" && (
        <div className="inspector-section">
          <span>Blob candidate</span>
          <strong>{selectedBlob ? `Blob #${selectedBlob.id}` : "none selected"}</strong>
          {selectedBlob ? (
            <>
              <small>Area: {String(selectedBlob.areaPx ?? "-")} px</small>
              <small>Centroid: {Array.isArray(selectedBlob.centroid) ? `${Number(selectedBlob.centroid[0]).toFixed(1)}, ${Number(selectedBlob.centroid[1]).toFixed(1)}` : "-"}</small>
              <small>BBox: {Array.isArray(selectedBlob.bbox) ? `${selectedBlob.bbox.join(", ")}` : "-"}</small>
              <small>Perimeter: {String(selectedBlob.perimeter ?? "-")}</small>
              <small>Eq. diameter: {String(selectedBlob.equivalentDiameter ?? "-")}</small>
              <small>Circularity: {String(selectedBlob.circularity ?? "-")}</small>
              <small>Aspect: {String(selectedBlob.aspectRatio ?? "-")}</small>
              <small>Solidity: {String(selectedBlob.solidity ?? "-")}</small>
              <small>Touches border: {selectedBlob.touchesBorder ? "yes" : "no"}</small>
              <small>Status: {selectedBlob.rejectionReason ?? selectedBlob.status ?? "kept"}</small>
            </>
          ) : (
            <>
              <small>Candidates: {blobCandidates.length}</small>
              <small>Rejected: {blobRejected}</small>
              <small>Select a candidate from overlay or table for detailed metrics.</small>
            </>
          )}
        </div>
      )}
      {semantic.stageId === "ellipse_fitting" && (
        <div className="inspector-section">
          <span>Ellipse fit</span>
          <strong>{selectedEllipse ? `Candidate #${Number(selectedEllipse.candidate_id ?? 0)}` : `${ellipseRows.length} fitted candidates`}</strong>
          {selectedEllipse ? (
            <>
              <small>Diameter: {String(selectedEllipse.equivalent_diameter ?? "-")}</small>
              <small>Major/minor: {String(selectedEllipse.major_axis ?? "-")} / {String(selectedEllipse.minor_axis ?? "-")}</small>
              <small>Eccentricity: {String(selectedEllipse.eccentricity ?? "-")}</small>
              <small>Fill ratio: {String(selectedEllipse.ellipse_fill_ratio ?? "-")}</small>
              <small>Circularity: {String(selectedEllipse.circularity ?? "-")}</small>
              <small>Solidity: {String(selectedEllipse.solidity ?? "-")}</small>
              <small>RMSE / max error: {String(selectedEllipse.fit_rmse ?? "-")} / {String(selectedEllipse.fit_max_error ?? "-")}</small>
              <small>Initial/refined mean error: {String(selectedEllipse.initial_mean_error ?? "-")} / {String(selectedEllipse.refined_mean_error ?? "-")}</small>
              <small>Refine iters/converged: {String(selectedEllipse.refinement_iterations ?? "-")} / {Boolean(selectedEllipse.refinement_converged) ? "yes" : "no"}</small>
              <small>Valid fit: {Boolean(selectedEllipse.valid_fit) ? "yes" : "no"}</small>
            </>
          ) : (
            <small>Select a candidate from the table/overlay to inspect fit quality metrics.</small>
          )}
          {!!Object.keys(ellipseEffectiveParams).length && <small>Effective params: {JSON.stringify(ellipseEffectiveParams)}</small>}
        </div>
      )}
      {(semantic.category === "geometry" || semantic.stageId === "ellipse_fitting") && selectedCandidateId && (
        <ObjectAnnotationEditor
          annotation={selectedAnnotation}
          candidateId={selectedCandidateId}
          sourceStage={semantic.stageId}
          onUpsert={onUpsertObjectAnnotation}
          bbox={(selectedBlob?.bbox ?? (Array.isArray(selectedEllipse?.bbox) ? selectedEllipse.bbox as number[] : undefined)) as number[] | undefined}
          centroid={(selectedBlob?.centroid ?? (Array.isArray(selectedEllipse?.centroid) ? selectedEllipse.centroid as number[] : undefined)) as number[] | undefined}
        />
      )}
      <div className="inspector-section">
        <span>Calibration</span>
        <strong>{detail?.result?.calibration_id ?? "No default calibration selected"}</strong>
        <small>{detail?.result?.calibration_status ?? "Processing may use automatic plane estimation."}</small>
      </div>
      <div className="inspector-section">
        <span>Artifact focus</span>
        <strong>{selectedArtifact?.title ?? "none"}</strong>
        <small>Stage: {selectedArtifact?.stage_id ?? "-"}</small>
        <small>Kind: {selectedArtifact?.kind ?? "-"}</small>
        <small>Overlay type: {selectedArtifact?.overlay_type ?? "-"}</small>
        <small>Coordinate space: {selectedArtifact?.coordinate_space ?? "-"}</small>
        <small>Target artifact: {selectedArtifact?.target_artifact_id ?? "-"}</small>
        <small>Path: {selectedArtifact?.path ?? "-"}</small>
        <small>Derived from: {lineage.length ? lineage.join(" | ") : "-"}</small>
        <small>Debug: classification_explanation.json found: {hasClassificationExplanationArtifact ? "yes" : "no"}</small>
      </div>
      <div className="inspector-section">
        <span>Height semantics</span>
        <strong>{String(artifactMeta.semantic_field ?? heightColorMapping.semanticField ?? selectedArtifact?.artifact_id ?? "-")}</strong>
        <small>Source semantic: {String(artifactMeta.derived_from ?? "-")}</small>
        <small>Transform lineage: {transformMeta ? JSON.stringify(transformMeta) : "-"}</small>
        <small>Units: {String(artifactMeta.units ?? heightColorMapping.units ?? "-")}</small>
        <small>Authoritative: {String(artifactMeta.is_measurement_authoritative ?? "-")}</small>
        <small>Hover semantic source: {String(hoverSample?.semanticField ?? hoverSample?.displaySemantic ?? "-")}</small>
        <small>Numeric raster source: {String(artifactMeta.numeric_values_path ?? artifactMeta.numeric_source_artifact ?? "-")}</small>
        <small>Preview source: {String(selectedArtifact?.path ?? "-")}</small>
        <small>Color scale range: {heightColorMapping.colorScaleMin.toFixed(2)} .. {heightColorMapping.colorScaleMax.toFixed(2)} {heightColorMapping.units ?? "mm"}</small>
        <small>Colormap: {heightColorMapping.colorMap} ({heightColorMapping.source})</small>
        {heightColorMappingResolution.warning && (
          <small className="hover-recon-error">{heightColorMappingResolution.warning}</small>
        )}
        {/* Always show the canonical colorbar so users can read the active
            scale on every stage, not only when a height image is selected. */}
        <div className="inspector-height-legend">
          <HeightLegend
            colorMapping={heightColorMapping}
            semanticLabel={inspectorLegendSemantic?.semanticLabel}
            positiveDirection={inspectorLegendSemantic?.positiveDirection}
            authoritative={inspectorLegendSemantic?.authoritative ?? heightColorMapping.source !== "legacy_fallback"}
            compact
            showTicks
            tickCount={5}
            percentileLabels={inspectorLegendSemantic?.percentileLabels}
          />
        </div>
      </div>
      {selectedArtifact?.kind === "overlay" && (
        <div className="inspector-section">
          <span>Overlay debug</span>
          <strong>{overlayDebug?.renderable ? "Renderable" : "Non-renderable"}</strong>
          <small>Target id: {overlayDebug?.targetArtifactId ?? selectedArtifact.target_artifact_id ?? "-"}</small>
          <small>Target title: {overlayDebug?.targetTitle ?? "-"}</small>
          <small>Coordinate space: {overlayDebug?.coordinateSpace ?? selectedArtifact.coordinate_space ?? "-"}</small>
          <small>Approximate: {(overlayDebug?.approximate ?? selectedArtifact.approximate) ? "yes" : "no"}</small>
          <small>Warnings: {(overlayDebug?.warnings ?? selectedArtifact.overlay_warnings ?? []).join(" | ") || "-"}</small>
          <small>Raw geometry: {JSON.stringify(selectedArtifact.geometry ?? {})}</small>
          <small>Transformed SVG geometry: {JSON.stringify(overlayDebug?.transformedGeometry ?? null)}</small>
        </div>
      )}
      {(selectedArtifact?.projection_type || selectedArtifact?.kind === "overlay") && (
        <ProjectionDebug artifacts={artifacts} selectedArtifact={selectedArtifact} overlayDebug={overlayDebug} />
      )}
      {!!summary.warnings.length && (
        <div className="inspector-section warning">
          <span>Warnings</span>
          <strong>{summary.warnings.length}</strong>
          <small>{summary.warnings.join(" | ")}</small>
        </div>
      )}
      {!!summary.errors.length && (
        <div className="inspector-section warning">
          <span>Errors</span>
          <strong>{summary.errors.length}</strong>
          <small>{summary.errors.join(" | ")}</small>
        </div>
      )}
    </aside>
  );
}

function ObjectAnnotationEditor({
  candidateId,
  sourceStage,
  annotation,
  onUpsert,
  bbox,
  centroid,
}: {
  candidateId: string;
  sourceStage: string;
  annotation: Record<string, unknown> | null;
  onUpsert?: (payload: Record<string, unknown>) => void;
  bbox?: number[];
  centroid?: number[];
}) {
  const labelSet = useMemo(() => new Set<string>(Array.isArray(annotation?.labels) ? (annotation?.labels as unknown[]).map(String) : []), [annotation]);
  const quickLabels = ["ball", "non_ball", "broken_ball", "worn_ball", "deformed_ball", "occluded", "false_positive", "partial_object", "touching_border"];
  async function savePatch(patch: Record<string, unknown>) {
    if (!onUpsert) return;
    await onUpsert({
      id: annotation?.id ?? undefined,
      source_stage: sourceStage,
      source_artifact_id: sourceStage === "ellipse_fitting" ? "ellipse_metrics" : "blob_contours",
      candidate_id: candidateId,
      bbox: bbox ?? annotation?.bbox ?? null,
      centroid: centroid ?? annotation?.centroid ?? null,
      labels: Array.from(labelSet),
      validation_status: annotation?.validation_status ?? "unreviewed",
      ...patch,
    });
  }
  return (
    <div className="inspector-section">
      <span>Object annotation</span>
      <strong>Candidate #{candidateId}</strong>
      <small>Ground-truth/review metadata, separate from computed pipeline outputs.</small>
      <div className="chip-row">
        {quickLabels.map((label) => (
          <button
            key={label}
            type="button"
            onClick={() => {
              const next = new Set(labelSet);
              if (next.has(label)) next.delete(label);
              else next.add(label);
              void savePatch({ labels: Array.from(next) });
            }}
          >
            {label}
          </button>
        ))}
      </div>
      <button type="button" onClick={() => {
        const value = window.prompt("Expected class", String(annotation?.expected_class ?? ""));
        if (value == null) return;
        void savePatch({ expected_class: value || null });
      }}>Set expected class</button>
      <button type="button" onClick={() => {
        const value = window.prompt("Expected diameter (mm)", String(annotation?.expected_diameter_mm ?? ""));
        if (value == null) return;
        void savePatch({ expected_diameter_mm: value.trim() ? Number(value) : null });
      }}>Set expected diameter</button>
      <button type="button" onClick={() => {
        const value = window.prompt("Notes", String(annotation?.notes ?? ""));
        if (value == null) return;
        void savePatch({ notes: value || null });
      }}>Edit notes</button>
      <div className="chip-row">
        {["unreviewed", "accepted", "rejected", "needs_review"].map((status) => (
          <button key={status} type="button" onClick={() => { void savePatch({ validation_status: status }); }}>
            {status}
          </button>
        ))}
      </div>
      <small>Labels: {Array.from(labelSet).join(", ") || "-"}</small>
      <small>Expected class: {String(annotation?.expected_class ?? "-")}</small>
      <small>Expected diameter: {String(annotation?.expected_diameter_mm ?? "-")}</small>
      <small>Status: {String(annotation?.validation_status ?? "unreviewed")}</small>
    </div>
  );
}

function artifactsForStageLocal(artifacts: StudioArtifact[], stageId: string): StudioArtifact[] {
  const canonical = canonicalStageId(stageId);
  const aliases = new Set<string>([stageId, canonical]);
  if (canonical === "segmentation") {
    aliases.add("threshold");
    aliases.add("morphology");
  } else if (canonical === "detection") {
    aliases.add("blob_detection");
    aliases.add("blob");
    aliases.add("contour_detection");
    aliases.add("blob_contour_detection");
  } else if (canonical === "measurement") {
    aliases.add("ellipse_fitting");
    aliases.add("metrics");
  } else if (canonical === "classification") {
    aliases.add("overlay");
    aliases.add("summary");
  }
  return artifacts.filter((item) => aliases.has(item.stage_id));
}

function ProjectionDebug({
  artifacts,
  selectedArtifact,
  overlayDebug,
}: {
  artifacts: StudioArtifact[];
  selectedArtifact: StudioArtifact | null;
  overlayDebug: OverlayDebugInfo | null;
}) {
  if (!selectedArtifact) return null;
  const target =
    selectedArtifact.kind === "overlay" && selectedArtifact.target_artifact_id
      ? artifacts.find((item) => item.artifact_id === selectedArtifact.target_artifact_id) ?? null
      : selectedArtifact;
  const coordinate = target?.projection_coordinate_system ?? {};
  return (
    <div className="inspector-section">
      <span>Projection Debug</span>
      <strong>{target?.projection_type ?? "compatibility mode"}</strong>
      <small>Coordinate system: {target?.projection_type ? "projection" : selectedArtifact.coordinate_space ?? "-"}</small>
      <small>Pixel/mm: {String((coordinate.pixel_per_mm as number | undefined) ?? "-")}</small>
      <small>Image size: {String((coordinate.image_width as number | undefined) ?? "-")} x {String((coordinate.image_height as number | undefined) ?? "-")}</small>
      <small>World bounds: {JSON.stringify(coordinate.world_bounds_mm ?? {})}</small>
      <small>Overlay transform: {target?.projection_transform_id ?? overlayDebug?.projectionTransformId ?? "-"}</small>
      <small>Renderable: {overlayDebug ? (overlayDebug.renderable ? "yes" : "no") : "n/a"}</small>
      <small>Warnings: {(overlayDebug?.warnings ?? selectedArtifact.overlay_warnings ?? []).join(" | ") || "-"}</small>
    </div>
  );
}

function ObjectInspector({
  object,
  producingStages,
  explanationByObjectId,
}: {
  object: StudioObject | null;
  producingStages: string[];
  explanationByObjectId: Map<number, Record<string, unknown>>;
}) {
  if (!object) {
    return (
      <div className="inspector-section">
        <span>Selected object</span>
        <strong>none</strong>
        <small>Select a candidate object to inspect classification and measurements.</small>
      </div>
    );
  }
  const explanation = (
    ((object as Record<string, unknown>).classification_explanation ?? null)
    || explanationByObjectId.get(Number(object.object_id))
    || null
  ) as Record<string, unknown> | null;
  const rules = (Array.isArray(explanation?.rules) ? explanation?.rules : [])
    .filter((item): item is Record<string, unknown> => typeof item === "object" && item !== null);
  const decisiveIds = Array.isArray((explanation?.decision_summary as Record<string, unknown> | undefined)?.decisive_rule_ids)
    ? (((explanation?.decision_summary as Record<string, unknown> | undefined)?.decisive_rule_ids as unknown[]).map(String))
    : [];
  const decisiveRules = (
    decisiveIds.length
      ? decisiveIds.map((id) => rules.find((rule) => String(rule.rule_id ?? "") === id)).filter((item): item is Record<string, unknown> => Boolean(item))
      : rules.filter((rule) => (
        (String(rule.contribution ?? "neutral") === "negative" && !Boolean(rule.passed))
        || (String(rule.contribution ?? "neutral") === "positive" && Boolean(rule.passed))
      ))
  ).slice(0, 3);
  const footprintGeometry = ((object as Record<string, unknown>).footprint_geometry ?? null) as Record<string, unknown> | null;
  const surfaceGeometry = ((object as Record<string, unknown>).surface_geometry ?? null) as Record<string, unknown> | null;
  const sphereConsistency = ((object as Record<string, unknown>).sphere_consistency ?? null) as Record<string, unknown> | null;
  const damageMetrics = ((object as Record<string, unknown>).damage_metrics ?? null) as Record<string, unknown> | null;
  const featureReadiness = ((object as Record<string, unknown>).feature_readiness ?? null) as Record<string, unknown> | null;
  const featureWarnings = Array.isArray((object as Record<string, unknown>).feature_warnings)
    ? (((object as Record<string, unknown>).feature_warnings as unknown[])
      .filter((item): item is Record<string, unknown> => typeof item === "object" && item !== null))
    : [];
  const featureGroupSummaries = Array.isArray((object as Record<string, unknown>).feature_group_summaries)
    ? (((object as Record<string, unknown>).feature_group_summaries as unknown[])
      .filter((item): item is Record<string, unknown> => typeof item === "object" && item !== null))
    : [];
  return (
    <div className="inspector-section">
      <span>Selected object</span>
      <strong>{object.class_name}</strong>
      <small>Subclass: {object.subclass_label ?? "-"}</small>
      <small>Class label: {String((object as Record<string, unknown>).label ?? "-")}</small>
      <small>Confidence: {object.confidence == null ? "-" : `${Math.round(object.confidence * 100)}%`}</small>
      <small>Diameter: {object.diameter_mm ?? object.diameter_estimate_mm ?? "-"} mm</small>
      <small>Major/minor diameter: {String((object as Record<string, unknown>).major_axis_mm ?? "-")} / {String((object as Record<string, unknown>).minor_axis_mm ?? "-")} mm</small>
      <small>Sphericity: {object.sphericity_score ?? "-"}</small>
      <small>Roundness: {String((object as Record<string, unknown>).feature_eccentricity != null ? (1 - Number((object as Record<string, unknown>).feature_eccentricity)).toFixed(4) : "-")}</small>
      <small>Fit RMSE: {object.fit_rmse_mm ?? "-"}</small>
      <small>Max/mean/p95 height: {String((object.height_above_belt_mm as Record<string, unknown> | undefined)?.max_height_mm ?? "-")} / {String((object.height_above_belt_mm as Record<string, unknown> | undefined)?.mean_height_mm ?? "-")} / {String((object.height_above_belt_mm as Record<string, unknown> | undefined)?.p95_height_mm ?? "-")} mm</small>
      <small>Footprint area: {String((object as Record<string, unknown>).footprint_area_mm2 ?? "-")} mm²</small>
      <small>Volume proxy: {String((object as Record<string, unknown>).feature_volume_proxy_mm3 ?? "-")} mm³</small>
      <small>Deformation score: {String((object as Record<string, unknown>).feature_local_curvature_proxy ?? "-")}</small>
      <small>Class hint/group: {String((object as Record<string, unknown>).class_group ?? "-")}</small>
      <small>Rejection: {object.filter_reason ?? "none"}</small>
      <small>Reasons: {Array.isArray((object as Record<string, unknown>).decision_reasons) ? ((object as Record<string, unknown>).decision_reasons as string[]).join(", ") : "-"}</small>
      <small>Diameter sources: ellipse={String((object as Record<string, unknown>).diameter_ellipse_mm ?? "-")} / area={String((object as Record<string, unknown>).diameter_equivalent_area_mm ?? "-")} / circ={String((object as Record<string, unknown>).diameter_circumference_mm ?? "-")} / selected={String((object as Record<string, unknown>).diameter_selected_mm ?? object.diameter_mm ?? "-")} ({String((object as Record<string, unknown>).diameter_selected_source ?? "unknown")})</small>
      <small>Geometry space: {String((object as Record<string, unknown>).geometry_coordinate_space ?? "-")} | Metrics space: {String((object as Record<string, unknown>).metrics_coordinate_space ?? "-")}</small>
      <small>Raw vs corrected eccentricity: {String((((object as Record<string, unknown>).raw_metrics as Record<string, unknown> | undefined)?.feature_eccentricity ?? "-"))} / {String((object as Record<string, unknown>).feature_eccentricity ?? "-")}</small>
      <small>Raw vs corrected roundness: {String((((object as Record<string, unknown>).raw_metrics as Record<string, unknown> | undefined)?.sphericity_score ?? "-"))} / {String((object as Record<string, unknown>).sphericity_score ?? "-")}</small>
      <small>Raw vs corrected diameter: {String((((object as Record<string, unknown>).raw_metrics as Record<string, unknown> | undefined)?.diameter_mm ?? "-"))} / {String((object as Record<string, unknown>).diameter_mm ?? "-")} mm</small>
      <small>Correction scales: {JSON.stringify((object as Record<string, unknown>).scale_correction_applied ?? null)}</small>
      <small>Generating stages: {producingStages.length ? producingStages.join(" -> ") : "-"}</small>
      <small>Why this class?: {decisiveRules.length ? "" : "-"}</small>
      {decisiveRules.map((rule, idx) => (
        <small key={`decisive_rule_${idx}`}>[{String(rule.severity ?? "info")}] {String(rule.label ?? rule.rule_id ?? "-")} {"->"} {String(rule.message ?? "-")}</small>
      ))}
      <small>Footprint geometry: circ={String(footprintGeometry?.circularity ?? "-")} | radial_cv={String(footprintGeometry?.radial_cv ?? "-")} | ecc={String(footprintGeometry?.eccentricity ?? "-")}</small>
      <small>Surface geometry: sphere_rmse={String(surfaceGeometry?.sphere_fit_rmse_mm ?? "-")} | sphere_r={String(surfaceGeometry?.sphere_fit_radius_mm ?? "-")} mm | ellipsoid_ar={String(surfaceGeometry?.ellipsoid_aspect_ratio ?? "-")}</small>
      <small>Sphere consistency: radial_rmse={String(sphereConsistency?.radial_height_rmse_mm ?? "-")} | completeness={String(sphereConsistency?.surface_completeness_ratio ?? "-")} | volume_deficit={String(sphereConsistency?.volume_deficit_ratio ?? "-")}</small>
      <small>Damage metrics: roughness={String(damageMetrics?.surface_roughness_mean ?? "-")} | flat_ratio={String(damageMetrics?.flat_region_ratio ?? "-")} | discontinuity={String(damageMetrics?.surface_discontinuity_score ?? "-")}</small>
      {featureReadiness ? (
        <small>
          Feature readiness: overall={String(featureReadiness.overall_readiness ?? "-")} | footprint={String(((featureReadiness.group_readiness as Record<string, unknown> | undefined)?.footprint_geometry) ?? "-")} | surface={String(((featureReadiness.group_readiness as Record<string, unknown> | undefined)?.surface_geometry) ?? "-")} | consistency={String(((featureReadiness.group_readiness as Record<string, unknown> | undefined)?.sphere_consistency) ?? "-")} | damage={String(((featureReadiness.group_readiness as Record<string, unknown> | undefined)?.damage_metrics) ?? "-")}
        </small>
      ) : null}
      {featureGroupSummaries.length ? featureGroupSummaries.map((group, idx) => {
        const evidence = Array.isArray(group.key_evidence_features)
          ? (group.key_evidence_features[0] as Record<string, unknown> | undefined)
          : undefined;
        return (
          <small key={`feature_group_${idx}`}>
            Feature group {String(group.group ?? "-")}: {String(group.status ?? "-")} | readiness={String(group.readiness ?? "-")} | severity={String(group.severity_score ?? "-")} | evidence={String(evidence?.feature ?? "-")}={String(evidence?.value ?? "-")}
          </small>
        );
      }) : null}
      {featureWarnings.length ? (
        <small>Feature warnings: {featureWarnings.map((warning) => `[${String(warning.severity ?? "info")}] ${String(warning.message ?? "-")}`).join(" | ")}</small>
      ) : null}
    </div>
  );
}
