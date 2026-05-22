import { useEffect, useMemo, useState, type MouseEvent, type ReactElement } from "react";
import ObjectTable from "../ObjectTable";
import StudioArtifactExplorer from "../StudioArtifactExplorer";
import StudioObjectList from "../StudioObjectList";
import type { OverlayDebugInfo } from "../overlayModel";
import { api, fileUrl, type SourceHistogramResponse, type TakeDetail } from "../../api/client";
import type { StudioArtifact } from "../studioWorkspaceModel";
import { artifactsForStage, firstArtifactByAlias } from "../studioWorkspaceModel";
import { stageEmptyState, stageSemanticDefinition, type StageViewDefinition } from "../stageSemantics";
import { primarySourceImageUrl, resolveStageViewContext } from "../stage_sources";
import type { RoiPolygon, RoiRect } from "../roiMapping";
import { normalizeBlobDetectionArtifacts } from "../../studio/blobDetectionModel";

type RendererProps = {
  detail: TakeDetail | null;
  compatible: boolean;
  processed: boolean;
  stageId: string;
  view: StageViewDefinition;
  selectedArtifactId: string | null;
  onSelectArtifact: (artifactId: string) => void;
  selectedObjectId: number | null;
  hoveredObjectId: number | null;
  onSelectObject: (objectId: number) => void;
  onHoverObject: (objectId: number | null) => void;
  hoveredArtifactId: string | null;
  onHoverArtifact: (artifactId: string | null) => void;
  onOverlayDebug: (info: OverlayDebugInfo) => void;
  roi?: RoiRect | null;
  roiPolygon?: RoiPolygon | null;
  roiType?: "rectangle" | "polygon";
  roiEnabled?: boolean;
  onApplyRoi?: (roi: RoiRect, rerun: boolean) => void;
  onApplyPolygonRoi?: (points: RoiPolygon, rerun: boolean) => void;
  onClearRoi?: () => void;
  blobParams?: Record<string, unknown> | null;
  onApplyBlobParams?: (params: Record<string, unknown>, rerun: boolean) => void;
  onUpsertObjectAnnotation?: (payload: Record<string, unknown>) => void;
};

function semanticEmpty(stageId: string, viewId: string) {
  const empty = stageEmptyState(stageId, viewId);
  return <div className="empty-state"><strong>{empty.title}</strong>{empty.description ? <small>{empty.description}</small> : null}</div>;
}

function objectAnnotationByCandidate(detail: TakeDetail | null): Map<string, Record<string, unknown>> {
  const map = new Map<string, Record<string, unknown>>();
  const entries = detail?.object_annotations;
  if (!Array.isArray(entries)) return map;
  for (const item of entries) {
    if (!item || typeof item !== "object") continue;
    const row = item as Record<string, unknown>;
    const candidateId = String(row.matched_candidate_id ?? row.candidate_id ?? "").trim();
    if (candidateId) map.set(candidateId, row);
  }
  return map;
}

function explore(
  artifacts: StudioArtifact[],
  props: RendererProps,
  opts?: { disableRoi?: boolean }
) {
  if (!artifacts.length) return semanticEmpty(props.stageId, props.view.id);
  return (
    <StudioArtifactExplorer
      artifacts={artifacts}
      takeId={props.detail?.take_id}
      selectedArtifactId={props.selectedArtifactId}
      onSelect={props.onSelectArtifact}
      detailJson={props.detail?.result}
      selectedObjectId={props.selectedObjectId}
      hoveredObjectId={props.hoveredObjectId}
      hoveredArtifactId={props.hoveredArtifactId}
      onSelectObject={props.onSelectObject}
      onHoverObject={props.onHoverObject}
      onHoverArtifact={props.onHoverArtifact}
      onOverlayDebug={props.onOverlayDebug}
      roi={opts?.disableRoi ? null : props.roi}
      roiPolygon={opts?.disableRoi ? null : props.roiPolygon}
      roiType={opts?.disableRoi ? undefined : props.roiType}
      roiEnabled={opts?.disableRoi ? false : props.roiEnabled}
      onApplyRoi={opts?.disableRoi ? undefined : props.onApplyRoi}
      onApplyPolygonRoi={opts?.disableRoi ? undefined : props.onApplyPolygonRoi}
      onClearRoi={opts?.disableRoi ? undefined : props.onClearRoi}
    />
  );
}

const renderers: Record<string, (props: RendererProps) => ReactElement> = {
  image: (props) => {
    const semantic = stageSemanticDefinition(props.stageId);
    const runArtifacts = artifactsForStage(props.detail, props.stageId);
    const context = resolveStageViewContext(props.detail, props.stageId, props.view.id, semantic.category, runArtifacts);
    const candidateArtifacts = (() => {
      if (props.view.id === "below_reference" || props.view.id === "above_threshold") {
        return [
          ...context.resolvedArtifacts,
          ...(props.detail?.result?.artifacts ?? []).filter((artifact) => !context.resolvedArtifacts.some((item) => item.artifact_id === artifact.artifact_id && item.stage_id === artifact.stage_id)),
        ];
      }
      return context.resolvedArtifacts;
    })();
    const artifacts = candidateArtifacts.filter((artifact) => {
      if (artifact.kind !== "image") return false;
      if (props.view.id === "height_preview") {
        return artifact.artifact_id === "source_heightmap_preview"
          || /heightmap_preview/i.test(String(artifact.path ?? ""))
          || artifact.artifact_id === "heightmap"
          || artifact.artifact_id === "input_preview";
      }
      if (props.view.id === "heightmap" || props.view.id === "raw_heightmap") return artifact.artifact_id === "heightmap" || artifact.artifact_id === "input_preview" || artifact.artifact_id === "raw_heightmap_preview";
      if (props.view.id === "depth_gradient") return artifact.artifact_id === "depth_gradient_magnitude" || /depth_gradient_magnitude/i.test(String(artifact.path ?? ""));
      if (props.view.id === "low_gradient_mask") return artifact.artifact_id === "low_gradient_mask" || /low_gradient_mask/i.test(String(artifact.path ?? ""));
      if (props.view.id === "selected_surface") return artifact.artifact_id === "reference_surface_selected_mask" || artifact.artifact_id === "expanded_plane_mask" || /reference_surface_selected_mask|expanded_plane_mask/i.test(String(artifact.path ?? ""));
      if (props.view.id === "normalized_height") return artifact.artifact_id === "normalized_heightmap" || /normalized_heightmap/i.test(String(artifact.path ?? ""));
      if (props.view.id === "below_reference") return artifact.artifact_id === "below_reference_mask" || /below_reference_mask/i.test(String(artifact.path ?? ""));
      if (props.view.id === "above_threshold") return artifact.artifact_id === "above_threshold_mask" || /above_threshold_mask/i.test(String(artifact.path ?? ""));
      if (props.view.id === "valid_mask") return artifact.artifact_id === "valid_mask" || /valid_mask/i.test(String(artifact.path ?? ""));
      if (props.view.id === "background_candidates" || props.view.id === "background_seeds") return artifact.artifact_id === "background_seed_mask" || artifact.artifact_id === "background_candidate_mask" || /background_(seed|candidate)_mask/i.test(String(artifact.path ?? ""));
      if (props.view.id === "expanded_plane") return artifact.artifact_id === "expanded_plane_mask" || /expanded_plane_mask/i.test(String(artifact.path ?? ""));
      if (props.view.id === "depth_plot") return artifact.artifact_id === "background_depth_plot" || /background_depth_plot/i.test(String(artifact.path ?? ""));
      if (props.view.id === "plane_inliers") return artifact.artifact_id === "final_plane_inlier_mask" || artifact.artifact_id === "plane_inlier_mask" || /final_plane_inlier_mask|plane_inlier_mask/i.test(String(artifact.path ?? ""));
      if (props.view.id === "residual_heatmap" || props.view.id === "belt_plane") {
        return artifact.artifact_id === "raw_heightmap_preview"
          || artifact.artifact_id === "valid_mask"
          || artifact.artifact_id === "plane_fit_roi_mask"
          || artifact.artifact_id === "plane_inlier_mask"
          || artifact.artifact_id === "belt_plane_residuals"
          || artifact.artifact_id === "belt_plane_overlay"
          || /belt_plane_(residuals|overlay)/i.test(String(artifact.path ?? ""))
          || /raw_heightmap_preview|valid_mask|plane_fit_roi_mask|plane_inlier_mask/i.test(String(artifact.path ?? ""));
      }
      if (semantic.category !== "segmentation") return true;
      if (props.view.id === "threshold_mask") return artifact.artifact_id === "foreground_before_plane_suppression" || artifact.artifact_id === "normalized_height_threshold_mask" || artifact.artifact_id === "threshold_mask";
      if (props.view.id === "cleaned_mask") return artifact.artifact_id === "final_object_mask" || artifact.artifact_id === "plane_suppressed_mask" || artifact.artifact_id === "cleaned_object_mask" || artifact.artifact_id === "cleaned_mask";
      return true;
    });
    if (semantic.category === "geometry" && props.view.id === "labels") {
      return explore(context.resolvedArtifacts.filter((artifact) => artifact.artifact_id === "blob_labels" || artifact.artifact_id === "labels" || artifact.kind === "image"), props);
    }
    if (semantic.category === "segmentation" && props.view.id === "cleaned_mask") {
      const metrics = context.runArtifacts.find((item) => item.artifact_id === "morphology_metrics");
      const cleanedPixels = Number((metrics?.metadata as Record<string, unknown> | undefined)?.cleaned_foreground_pixels ?? 0);
      if (artifacts.length && cleanedPixels <= 0) {
        return (
          <div>
            {explore(artifacts, props)}
            <div className="empty-state">
              <strong>Cleaned mask is empty.</strong>
              <small>Try lowering min component area or changing threshold/invert.</small>
            </div>
          </div>
        );
      }
    }
    if (props.view.id === "depth_plot") {
      return explore(artifacts, props, { disableRoi: true });
    }
    return explore(artifacts, props);
  },
  overlay: (props) => {
    const semantic = stageSemanticDefinition(props.stageId);
    const runArtifacts = artifactsForStage(props.detail, props.stageId);
    const context = resolveStageViewContext(props.detail, props.stageId, props.view.id, semantic.category, runArtifacts);
    let artifacts = context.resolvedArtifacts.filter((artifact) => artifact.kind === "overlay" || artifact.kind === "image");
    if (semantic.category === "segmentation" && props.view.id === "overlay") {
      artifacts = artifacts.filter((artifact) =>
        artifact.artifact_id === "connected_components_overlay"
        || artifact.artifact_id === "height_segmentation_overlay"
        || artifact.artifact_id === "height_segmentation"
        || /connected_components_overlay/i.test(String(artifact.path ?? ""))
        || /height_segmentation_overlay/i.test(String(artifact.path ?? ""))
        || /segmentation_overlay/i.test(String(artifact.path ?? ""))
      );
    }
    if (props.view.id === "overlays_25d") {
      artifacts = artifacts.filter((artifact) =>
        artifact.artifact_id === "height_overlay"
        || artifact.artifact_id === "measurement_overlay"
        || artifact.artifact_id === "classification_overlay"
        || artifact.artifact_id === "height_segmentation"
        || /overlay/i.test(String(artifact.artifact_id ?? ""))
        || /overlay/i.test(String(artifact.path ?? ""))
      );
    }
    if (semantic.category === "geometry" && props.view.id === "candidate_overlay") {
      const normalized = normalizeBlobDetectionArtifacts(context.runArtifacts);
      const source = context.resolvedArtifacts.find((item) => item.artifact_id === "source_rgb_image" && item.path)
        ?? context.resolvedArtifacts.find((item) => item.kind === "image" && item.path)
        ?? null;
      if (source?.path) {
        return (
          <BlobCandidateOverlayWorkspace
            src={fileUrl(props.detail!.take_id, source.path)}
            candidates={normalized.candidates}
            rejected={normalized.rejected}
            selectedCandidateId={props.selectedObjectId}
            hoveredCandidateId={props.hoveredObjectId}
            onSelectCandidate={props.onSelectObject}
            onHoverCandidate={props.onHoverObject}
          />
        );
      }
      artifacts = artifacts.filter((artifact) =>
        artifact.artifact_id === "blob_debug_overlay"
        || artifact.artifact_id === "candidate_overlay"
        || artifact.artifact_id === "contour_overlay"
        || artifact.kind === "overlay"
      );
    }
    if (semantic.category === "geometry" && props.view.id === "ellipse_overlay") {
      artifacts = artifacts.filter((artifact) =>
        artifact.artifact_id === "ellipse_overlay"
        || artifact.artifact_id === "ellipse_debug_overlay"
        || artifact.artifact_id === "source_rgb_image"
        || artifact.kind === "overlay"
      );
    }
    if (semantic.category === "classification" && props.view.id === "overlay") {
      const overlayImage = context.runArtifacts.find((artifact) => artifact.artifact_id === "classification_overlay_image");
      const overlayMetaArtifact = context.runArtifacts.find((artifact) => artifact.artifact_id === "classification_overlay_metadata");
      const overlayMeta = (overlayMetaArtifact?.metadata ?? {}) as Record<string, unknown>;
      const overlayObjects = Array.isArray(overlayMeta.objects) ? overlayMeta.objects as Array<Record<string, unknown>> : [];
      if (overlayImage?.path && overlayObjects.length) {
        return (
          <ClassificationOverlayWorkspace
            src={fileUrl(props.detail!.take_id, overlayImage.path)}
            objects={overlayObjects}
            selectedObjectId={props.selectedObjectId}
            hoveredObjectId={props.hoveredObjectId}
            onSelectObject={props.onSelectObject}
            onHoverObject={props.onHoverObject}
          />
        );
      }
      artifacts = artifacts.filter((artifact) =>
        artifact.artifact_id === "classification_overlay_image"
        || artifact.artifact_id === "classification_overlay_metadata"
        || artifact.artifact_id === "classification_result"
        || artifact.artifact_id === "classification_result_artifact"
        || artifact.artifact_id === "source_rgb_image"
        || artifact.kind === "overlay"
      );
    }
    return explore(artifacts, props);
  },
  table: (props) => {
    if (!props.detail) return semanticEmpty(props.stageId, props.view.id);
    const semantic = stageSemanticDefinition(props.stageId);
    if ((semantic.category === "input" || semantic.category === "plane_qa") && props.view.id === "metadata") {
      return <pre className="json-block">{JSON.stringify({ modalities: props.detail.modalities, created_at: props.detail.created_at, metadata: props.detail.metadata, frameset: props.detail.frameset, assets: props.detail.assets }, null, 2)}</pre>;
    }
    if (semantic.category === "segmentation" && props.view.id === "params") {
      const stageArtifacts = artifactsForStage(props.detail, props.stageId);
      const threshold = stageArtifacts.find((item) => item.artifact_id === "threshold_mask");
      const morphMetrics = stageArtifacts.find((item) => item.artifact_id === "morphology_metrics");
      const morphDebug = stageArtifacts.find((item) => item.artifact_id === "morphology_debug_json");
      const thresholdMeta = (threshold?.metadata ?? {}) as Record<string, unknown>;
      const metricsMeta = (morphMetrics?.metadata ?? {}) as Record<string, unknown>;
      const debugMeta = (morphDebug?.metadata ?? {}) as Record<string, unknown>;
      const morphDebugSection = (debugMeta.morphology as Record<string, unknown> | undefined) ?? {};
      const roiDebugSection = (debugMeta.roi as Record<string, unknown> | undefined) ?? {};
      if (!threshold && !morphMetrics && !morphDebug) return semanticEmpty(props.stageId, props.view.id);
      const payload = {
        threshold: {
          mode: thresholdMeta.threshold_mode ?? "-",
          value: thresholdMeta.threshold_value ?? "-",
          adaptive_block_size: thresholdMeta.adaptive_block_size ?? "-",
          adaptive_c: thresholdMeta.adaptive_c ?? "-",
          blur_kernel: thresholdMeta.blur_kernel ?? "-",
        },
        morphology: {
          operation: morphDebugSection.operation ?? "-",
          open_kernel: morphDebugSection.open_kernel ?? "-",
          close_kernel: morphDebugSection.close_kernel ?? "-",
          iterations: morphDebugSection.iterations ?? "-",
          fill_holes: morphDebugSection.fill_holes ?? "-",
          min_component_area: morphDebugSection.min_component_area ?? "-",
          max_component_area: morphDebugSection.max_component_area ?? "-",
        },
        roi: {
          enabled: roiDebugSection.enabled ?? "-",
          x: roiDebugSection.x ?? "-",
          y: roiDebugSection.y ?? "-",
          width: roiDebugSection.width ?? "-",
          height: roiDebugSection.height ?? "-",
        },
        metrics: {
          connected_components: morphDebugSection.components_after ?? metricsMeta.connected_components ?? "-",
          components_before: morphDebugSection.components_before ?? metricsMeta.components_before ?? "-",
          components_after: morphDebugSection.components_after ?? metricsMeta.components_after ?? "-",
          removed_components: morphDebugSection.removed_components ?? metricsMeta.removed_components ?? "-",
          threshold_foreground_coverage: morphDebugSection.threshold_foreground_coverage ?? metricsMeta.threshold_foreground_coverage ?? "-",
          cleaned_foreground_coverage: morphDebugSection.cleaned_foreground_coverage ?? metricsMeta.cleaned_foreground_coverage ?? "-",
          foreground_coverage_percent: metricsMeta.foreground_coverage_percent ?? "-",
          largest_component_area: metricsMeta.largest_component_area ?? "-",
          mean_component_area: metricsMeta.mean_component_area ?? "-",
        },
      };
      return <pre className="json-block">{JSON.stringify(payload, null, 2)}</pre>;
    }
    if ((semantic.category === "input" || semantic.category === "plane_qa") && props.view.id === "json") {
      const stageArtifacts = artifactsForStage(props.detail, props.stageId);
      const planeDebug = stageArtifacts.find((item) => item.artifact_id === "plane_fit_debug");
      const normHist = stageArtifacts.find((item) => item.artifact_id === "normalized_height_histogram");
      const normDebug = stageArtifacts.find((item) => item.artifact_id === "normalization_debug" || item.artifact_id === "normalized_height_debug");
      const bgHist = stageArtifacts.find((item) => item.artifact_id === "background_depth_histogram");
      const bgDebug = stageArtifacts.find((item) => item.artifact_id === "background_selection_debug");
      return (
        <pre className="json-block">
          {JSON.stringify(
            {
              plane_fit_debug: planeDebug?.metadata ?? null,
              background_depth_histogram: bgHist?.metadata ?? null,
              background_selection_debug: bgDebug?.metadata ?? null,
              normalized_height_histogram: normHist?.metadata ?? null,
              normalization_debug: normDebug?.metadata ?? null,
              result_summary: props.detail.result?.summary ?? null,
            },
            null,
            2
          )}
        </pre>
      );
    }
    if (semantic.category === "geometry" && (props.view.id === "blob_metrics" || props.view.id === "candidates_table")) {
      const stageArtifacts = artifactsForStage(props.detail, props.stageId);
      const annotationMap = objectAnnotationByCandidate(props.detail);
      const ellipseRows = ellipseEntries(stageArtifacts);
      const ellipseSummaryArtifact = firstArtifactByAlias(stageArtifacts, ["ellipse_summary", "geometry_summary"]);
      const ellipseDebugArtifact = firstArtifactByAlias(stageArtifacts, ["ellipse_debug_json"]);
      const ellipseSummaryMeta = (ellipseSummaryArtifact?.metadata ?? {}) as Record<string, unknown>;
      const ellipseDebugMeta = (ellipseDebugArtifact?.metadata ?? {}) as Record<string, unknown>;
      const blobContourArtifact = firstArtifactByAlias(stageArtifacts, ["blob_contours", "contours"]);
      const blobMetricArtifact = firstArtifactByAlias(stageArtifacts, ["blob_metrics", "detection_metrics"]);
      const blobRows = normalizeBlobDetectionArtifacts(stageArtifacts).candidates;
      const blobRejectedRows = normalizeBlobDetectionArtifacts(stageArtifacts).rejected;
      const fitWarnings = Array.isArray(ellipseDebugMeta.warnings) ? ellipseDebugMeta.warnings.map(String) : [];
      const diagnostics = (
        <div className="artifact-reference">
          <span>Runtime diagnostics</span>
          <small>Blob candidates received: {blobRows.length}</small>
          <small>Contours valid for fitEllipse: {Number(ellipseSummaryMeta.candidate_count ?? ellipseDebugMeta.candidate_count ?? 0)}</small>
          <small>Candidates rejected before fit: {blobRejectedRows.length}</small>
          <small>Successful fits: {Number(ellipseSummaryMeta.fitted_count ?? ellipseDebugMeta.fitted_count ?? ellipseRows.length)}</small>
          <small>ellipse_metrics.json: {firstArtifactByAlias(stageArtifacts, ["ellipse_metrics", "geometry_metrics"]) ? "yes" : "no"}</small>
          <small>ellipse_overlay.png: {firstArtifactByAlias(stageArtifacts, ["ellipse_overlay"]) ? "yes" : "no"}</small>
          <small>ellipse_debug_overlay.png: {firstArtifactByAlias(stageArtifacts, ["ellipse_debug_overlay"]) ? "yes" : "no"}</small>
          <small>blob_contours present: {blobContourArtifact ? "yes" : "no"} • blob_metrics present: {blobMetricArtifact ? "yes" : "no"}</small>
          {fitWarnings.length ? <small>Warnings: {fitWarnings.join(" | ")}</small> : null}
        </div>
      );
      if (props.stageId === "ellipse_fitting" && props.view.id === "candidates_table" && !ellipseRows.length) {
        return (
          <div>
            {diagnostics}
            <div className="empty-state">
              <strong>No ellipse candidates found.</strong>
              <small>Check blob candidates, contour validity, and ellipse fit thresholds.</small>
            </div>
          </div>
        );
      }
      if (props.view.id !== "blob_metrics" && ellipseRows.length) {
        return (
          <div className="studio-table-pane">
            {diagnostics}
            <table className="compact-candidate-table">
              <thead>
                <tr>
                  <th>ID</th><th>Labels</th><th>Diameter</th><th>Major</th><th>Minor</th><th>Eccentricity</th><th>Fill</th><th>RMSE</th><th>Max error</th><th>Circularity</th><th>Solidity</th><th>Valid</th>
                </tr>
              </thead>
              <tbody>
                {ellipseRows.map((row) => (
                  (() => {
                    const candidateId = Number(row.candidate_id ?? 0);
                    return (
                  <tr
                    key={`ellipse_row_${candidateId}`}
                    className={props.selectedObjectId === candidateId ? "selected" : ""}
                    onClick={() => props.onSelectObject(candidateId)}
                    onMouseEnter={() => props.onHoverObject(candidateId)}
                    onMouseLeave={() => props.onHoverObject(null)}
                  >
                    <td>{candidateId}</td>
                    <td>{Array.isArray(annotationMap.get(String(candidateId))?.labels) ? (annotationMap.get(String(candidateId))?.labels as unknown[]).join(", ") : "-"}</td>
                    <td>{fmtNum(row.equivalent_diameter, 2)}</td>
                    <td>{fmtNum(row.major_axis, 2)}</td>
                    <td>{fmtNum(row.minor_axis, 2)}</td>
                    <td>{fmtNum(row.eccentricity, 3)}</td>
                    <td>{fmtNum(row.ellipse_fill_ratio, 3)}</td>
                    <td>{fmtNum(row.fit_rmse, 3)}</td>
                    <td>{fmtNum(row.fit_max_error, 3)}</td>
                    <td>{fmtNum(row.circularity, 3)}</td>
                    <td>{fmtNum(row.solidity, 3)}</td>
                    <td>{row.valid_fit ? "yes" : "no"}</td>
                  </tr>
                    );
                  })()
                ))}
              </tbody>
            </table>
          </div>
        );
      }
      const normalized = normalizeBlobDetectionArtifacts(stageArtifacts);
      const rows = normalized.candidates;
      const rejected = normalized.summary.rejectedCount;
      if (!rows.length) {
        return (
          <div className="empty-state">
            <strong>No blob candidates found.</strong>
            <small>{rejected > 0 ? "All components were rejected. Review min area, border reject, or aspect ratio filters." : "Check cleaned mask, ROI, and component filtering parameters."}</small>
          </div>
        );
      }
      return (
        <div className="studio-table-pane">
          <div className="pipeline-status-grid">
            <div><span>Objects</span><strong>{rows.length}</strong></div>
            <div><span>Rejected</span><strong>{rejected}</strong></div>
            <div><span>Avg diameter</span><strong>{normalized.summary.avgDiameter == null ? "-" : `${normalized.summary.avgDiameter.toFixed(2)} px`}</strong></div>
            <div><span>Avg circularity</span><strong>{normalized.summary.avgCircularity == null ? "-" : normalized.summary.avgCircularity.toFixed(3)}</strong></div>
            <div><span>Largest area</span><strong>{normalized.summary.largestArea == null ? "-" : `${normalized.summary.largestArea.toFixed(0)} px`}</strong></div>
          </div>
          <div className="artifact-reference">
            <span>Calibration stats</span>
            <small>Area px: {fmtRange(normalized.summary.areaStats)}</small>
            <small>Diameter px: {fmtRange(normalized.summary.diameterStats)}</small>
            <small>Circularity: {fmtRange(normalized.summary.circularityStats)}</small>
            <small>Aspect ratio: {fmtRange(normalized.summary.aspectRatioStats)}</small>
            <small>Rejected reasons: {Object.entries(normalized.summary.rejectedReasonCounts ?? {}).map(([k, v]) => `${k}:${v}`).join(" | ") || "-"}</small>
          </div>
          <form
            className="artifact-reference"
            onSubmit={(event) => {
              event.preventDefault();
              if (!props.onApplyBlobParams) return;
              const form = new FormData(event.currentTarget);
              const params = {
                min_area: Number(form.get("min_area") ?? 0),
                max_area: Number(form.get("max_area") ?? 0),
                min_width: Number(form.get("min_width") ?? 0),
                min_height: Number(form.get("min_height") ?? 0),
                max_aspect_ratio: Number(form.get("max_aspect_ratio") ?? 0),
                min_circularity: Number(form.get("min_circularity") ?? 0),
                reject_border_touching: String(form.get("reject_border_touching") ?? "") === "on",
                keep_largest_n: Number(form.get("keep_largest_n") ?? 0),
              };
              props.onApplyBlobParams(params, false);
            }}
          >
            <span>Blob tuning</span>
            <label>min_area <input name="min_area" type="number" step="1" defaultValue={Number(props.blobParams?.min_area ?? 80)} /></label>
            <label>max_area <input name="max_area" type="number" step="1" defaultValue={Number(props.blobParams?.max_area ?? 250000)} /></label>
            <label>min_width <input name="min_width" type="number" step="1" defaultValue={Number(props.blobParams?.min_width ?? 0)} /></label>
            <label>min_height <input name="min_height" type="number" step="1" defaultValue={Number(props.blobParams?.min_height ?? 0)} /></label>
            <label>max_aspect_ratio <input name="max_aspect_ratio" type="number" step="0.01" defaultValue={Number(props.blobParams?.max_aspect_ratio ?? 0)} /></label>
            <label>min_circularity <input name="min_circularity" type="number" step="0.01" defaultValue={Number(props.blobParams?.min_circularity ?? 0.4)} /></label>
            <label>keep_largest_n <input name="keep_largest_n" type="number" step="1" defaultValue={Number(props.blobParams?.keep_largest_n ?? 0)} /></label>
            <label><input name="reject_border_touching" type="checkbox" defaultChecked={Boolean(props.blobParams?.reject_border_touching ?? false)} />reject_border_touching</label>
            <div className="overlay-controls">
              <button type="submit">Apply params</button>
              <button
                type="button"
                onClick={(event) => {
                  if (!props.onApplyBlobParams) return;
                  const formEl = event.currentTarget.closest("form");
                  if (!(formEl instanceof HTMLFormElement)) return;
                  const form = new FormData(formEl);
                  props.onApplyBlobParams(
                    {
                      min_area: Number(form.get("min_area") ?? 0),
                      max_area: Number(form.get("max_area") ?? 0),
                      min_width: Number(form.get("min_width") ?? 0),
                      min_height: Number(form.get("min_height") ?? 0),
                      max_aspect_ratio: Number(form.get("max_aspect_ratio") ?? 0),
                      min_circularity: Number(form.get("min_circularity") ?? 0),
                      reject_border_touching: String(form.get("reject_border_touching") ?? "") === "on",
                      keep_largest_n: Number(form.get("keep_largest_n") ?? 0),
                    },
                    true
                  );
                }}
              >
                Apply params + rerun
              </button>
            </div>
            <small>Advanced parameters can also be edited in pipeline configuration.</small>
          </form>
          <table className="compact-candidate-table">
            <thead>
              <tr>
                <th>ID</th><th>Labels</th><th>Area</th><th>Centroid</th><th>BBox</th><th>Eq. diameter</th><th>Circularity</th><th>Aspect</th><th>Solidity</th><th>Border</th><th>Status</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr
                  key={`blob_row_${row.id}`}
                  className={props.selectedObjectId === Number(row.id) ? "selected" : ""}
                  onClick={() => props.onSelectObject(Number(row.id))}
                  onMouseEnter={() => props.onHoverObject(Number(row.id))}
                  onMouseLeave={() => props.onHoverObject(null)}
                >
                  <td>{row.id}</td>
                  <td>{Array.isArray(annotationMap.get(String(row.id))?.labels) ? (annotationMap.get(String(row.id))?.labels as unknown[]).join(", ") : "-"}</td>
                  <td>{Number(row.areaPx ?? 0).toFixed(0)}</td>
                  <td>{Array.isArray(row.centroid) ? `${Number(row.centroid[0]).toFixed(1)}, ${Number(row.centroid[1]).toFixed(1)}` : "-"}</td>
                  <td>{Array.isArray(row.bbox) ? `${Number(row.bbox[0]).toFixed(0)},${Number(row.bbox[1]).toFixed(0)} ${Number(row.bbox[2]).toFixed(0)}x${Number(row.bbox[3]).toFixed(0)}` : "-"}</td>
                  <td>{row.equivalentDiameter == null ? "-" : Number(row.equivalentDiameter).toFixed(2)}</td>
                  <td>{row.circularity == null ? "-" : Number(row.circularity).toFixed(3)}</td>
                  <td>{row.aspectRatio == null ? "-" : Number(row.aspectRatio).toFixed(3)}</td>
                  <td>{row.solidity == null ? "-" : Number(row.solidity).toFixed(3)}</td>
                  <td>{row.touchesBorder ? "yes" : "no"}</td>
                  <td>{row.rejectionReason ?? row.status ?? "kept"}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <small>Blob candidates are extracted from cleaned mask artifacts and remain classification-agnostic.</small>
        </div>
      );
    }
    if (!props.compatible || !props.processed) return semanticEmpty(props.stageId, props.view.id);
      if (semantic.category === "geometry" && props.view.id === "ellipse_metrics") {
      const stageArtifacts = artifactsForStage(props.detail, props.stageId);
      const ellipseRows = ellipseEntries(stageArtifacts);
      const summary = firstArtifactByAlias(stageArtifacts, ["ellipse_summary", "geometry_summary"]);
      const debug = firstArtifactByAlias(stageArtifacts, ["ellipse_debug_json"]);
      const summaryMeta = (summary?.metadata ?? {}) as Record<string, unknown>;
      const debugMeta = (debug?.metadata ?? {}) as Record<string, unknown>;
      if (!ellipseRows.length) return semanticEmpty(props.stageId, props.view.id);
      return (
        <div className="studio-table-pane">
          <div className="pipeline-status-grid">
            <div><span>Candidates</span><strong>{Number(summaryMeta.candidate_count ?? ellipseRows.length)}</strong></div>
            <div><span>Fitted</span><strong>{Number(summaryMeta.fitted_count ?? ellipseRows.length)}</strong></div>
            <div><span>Rejected fit</span><strong>{Number(summaryMeta.rejected_fit_count ?? 0)}</strong></div>
            <div><span>Avg diameter</span><strong>{fmtNum(Number(summaryMeta.avg_diameter ?? 0), 2)} px</strong></div>
          </div>
          <small>Avg eccentricity: {fmtNum(Number(summaryMeta.avg_eccentricity ?? 0), 3)} • Avg RMSE: {fmtNum(Number(summaryMeta.avg_fit_rmse ?? 0), 3)}</small>
          {Array.isArray(debugMeta.warnings) && debugMeta.warnings.length > 0 ? (
            <div className="artifact-warning-banner">{(debugMeta.warnings as unknown[]).map(String).join(" | ")}</div>
          ) : null}
        </div>
      );
    }
    if (semantic.category === "geometry" && props.view.id === "blob_metrics") {
      const stageArtifacts = artifactsForStage(props.detail, props.stageId);
      const metricsArtifact = stageArtifacts.find((item) => item.artifact_id === "blob_metrics");
      const rows = ((metricsArtifact?.metadata as Record<string, unknown> | undefined)?.entries ?? []) as unknown[];
      if (!rows.length) return semanticEmpty(props.stageId, props.view.id);
      return <pre className="json-block">{JSON.stringify(rows, null, 2)}</pre>;
    }
    if (semantic.category === "measurement" && (props.view.id === "measurements_25d" || props.view.id === "measurements")) {
      const objects = [...(props.detail.result?.objects ?? []), ...(props.detail.result?.rejected_objects ?? [])];
      const kept = objects.filter((item) => item.filter_status !== "rejected");
      const avgDiameter = kept.length ? kept.reduce((acc, obj) => acc + Number(obj.diameter_mm ?? obj.diameter_estimate_mm ?? 0), 0) / kept.length : 0;
      const avgCircularity = kept.length ? kept.reduce((acc, obj) => acc + Number(obj.sphericity_score ?? 0), 0) / kept.length : 0;
      return (
        <div className="studio-object-workspace">
          <StudioObjectList objects={objects} selectedObjectId={props.selectedObjectId} hoveredObjectId={props.hoveredObjectId} onSelect={props.onSelectObject} onHover={props.onHoverObject} />
          <div className="studio-table-pane">
            <div className="pipeline-status-grid">
              <div><span>Objects</span><strong>{kept.length}</strong></div>
              <div><span>Rejected</span><strong>{objects.length - kept.length}</strong></div>
              <div><span>Avg diameter</span><strong>{kept.length ? `${avgDiameter.toFixed(2)} mm` : "-"}</strong></div>
              <div><span>Avg circularity</span><strong>{kept.length ? avgCircularity.toFixed(2) : "-"}</strong></div>
            </div>
            <ObjectTable objects={objects} selectedObjectId={props.selectedObjectId} hoveredObjectId={props.hoveredObjectId} onSelectObject={props.onSelectObject} onHoverObject={props.onHoverObject} />
          </div>
        </div>
      );
    }
    const objects = [...(props.detail.result?.objects ?? []), ...(props.detail.result?.rejected_objects ?? [])];
    const stageArtifacts = artifactsForStage(props.detail, props.stageId);
    const classificationArtifact = firstArtifactByAlias(stageArtifacts, ["classification_result", "classification_result_artifact"]);
    const classificationMeta = (classificationArtifact?.metadata ?? {}) as Record<string, unknown>;
    const byClass = {
      "Bola buena": objects.filter((item) => item.class_name === "Bola buena"),
      "Scrap de Bola": objects.filter((item) => item.class_name === "Scrap de Bola"),
      "Chatarra": objects.filter((item) => item.class_name === "Chatarra"),
    };
    return (
      <div className="studio-object-workspace">
        <StudioObjectList objects={objects} selectedObjectId={props.selectedObjectId} hoveredObjectId={props.hoveredObjectId} onSelect={props.onSelectObject} onHover={props.onHoverObject} />
        <div className="studio-table-pane">
          {!objects.length && Number(classificationMeta.candidates_from_ellipse_fitting ?? 0) > 0 && (
            <div className="empty-state">
              <strong>Ellipse candidates found, but classification consumed 0 candidates.</strong>
              <small>Candidates from ellipse fitting: {String(classificationMeta.candidates_from_ellipse_fitting ?? "-")}.</small>
            </div>
          )}
          <div className="pipeline-status-grid">
            <div><span>Bola buena</span><strong>{byClass["Bola buena"].length}</strong></div>
            <div><span>Scrap de Bola</span><strong>{byClass["Scrap de Bola"].length}</strong></div>
            <div><span>Chatarra</span><strong>{byClass["Chatarra"].length}</strong></div>
          </div>
          <ObjectTable objects={objects} selectedObjectId={props.selectedObjectId} hoveredObjectId={props.hoveredObjectId} onSelectObject={props.onSelectObject} onHoverObject={props.onHoverObject} />
          {explore(resolveStageViewContext(props.detail, props.stageId, props.view.id, semantic.category, artifactsForStage(props.detail, props.stageId)).resolvedArtifacts, props)}
        </div>
      </div>
    );
  },
  metrics: (props) => {
    if (!props.detail || !props.compatible || !props.processed) return semanticEmpty(props.stageId, props.view.id);
    if (stageSemanticDefinition(props.stageId).category === "geometry") {
      const stageArtifacts = artifactsForStage(props.detail, props.stageId);
      const normalized = normalizeBlobDetectionArtifacts(stageArtifacts);
      const rows = normalized.candidates;
      const rejected = normalized.summary.rejectedCount;
      if (!rows.length) return semanticEmpty(props.stageId, props.view.id);
      const avgDiameter = normalized.summary.avgDiameter ?? 0;
      const avgCircularity = normalized.summary.avgCircularity ?? 0;
      return (
        <div className="pipeline-status-grid">
          <div><span>Objects</span><strong>{rows.length}</strong></div>
          <div><span>Rejected</span><strong>{rejected}</strong></div>
          <div><span>Avg diameter</span><strong>{avgDiameter.toFixed(2)} px</strong></div>
          <div><span>Avg circularity</span><strong>{avgCircularity.toFixed(3)}</strong></div>
        </div>
      );
    }
    const objects = props.detail.result?.objects ?? [];
    const rejected = props.detail.result?.rejected_objects ?? [];
    const avgDiameter = objects.length ? objects.reduce((acc, obj) => acc + Number(obj.diameter_mm ?? obj.diameter_estimate_mm ?? 0), 0) / objects.length : 0;
    const avgCircularity = objects.length ? objects.reduce((acc, obj) => acc + Number(obj.sphericity_score ?? 0), 0) / objects.length : 0;
    return (
      <div className="pipeline-status-grid">
        <div><span>Objects</span><strong>{objects.length}</strong></div>
        <div><span>Rejected</span><strong>{rejected.length}</strong></div>
        <div><span>Avg diameter</span><strong>{objects.length ? `${avgDiameter.toFixed(2)} mm` : "-"}</strong></div>
        <div><span>Avg circularity</span><strong>{objects.length ? avgCircularity.toFixed(2) : "-"}</strong></div>
      </div>
    );
  },
  histogram: (props) => {
    if (!props.detail) return semanticEmpty(props.stageId, props.view.id);
    const semantic = stageSemanticDefinition(props.stageId);
    if (semantic.category === "plane_qa") {
      const runArtifacts = artifactsForStage(props.detail, props.stageId);
      const histogram = runArtifacts.find((item) => item.artifact_id === "normalized_height_histogram");
      if (!histogram?.metadata) return semanticEmpty(props.stageId, props.view.id);
      return <pre className="json-block">{JSON.stringify(histogram.metadata, null, 2)}</pre>;
    }
    if (semantic.category === "input") {
      const runArtifacts = artifactsForStage(props.detail, props.stageId);
      const context = resolveStageViewContext(props.detail, props.stageId, "image", semantic.category, runArtifacts);
      const selectedImage = context.resolvedArtifacts.find((artifact) => artifact.artifact_id === props.selectedArtifactId && artifact.kind === "image");
      const fallback = context.resolvedArtifacts.find((artifact) => artifact.kind === "image");
      const chosen = selectedImage ?? fallback ?? null;
      const src = chosen?.path ? fileUrl(props.detail.take_id, chosen.path) : primarySourceImageUrl(props.detail);
      return <SourceImageHistogram src={src} srcKey={chosen?.artifact_id ?? src ?? "none"} takeId={props.detail.take_id} filename={chosen?.path ?? null} />;
    }
    if (!props.compatible || !props.processed) return semanticEmpty(props.stageId, props.view.id);
    const values = (props.detail.result?.objects ?? [])
      .map((item) => Number(item.diameter_mm ?? item.diameter_estimate_mm ?? 0))
      .filter((item) => Number.isFinite(item) && item > 0);
    if (!values.length) return semanticEmpty(props.stageId, props.view.id);
    const min = Math.min(...values);
    const max = Math.max(...values);
    return (
      <div className="artifact-reference">
        <span>Histogram</span>
        <strong>Diameter distribution</strong>
        <small>Count: {values.length}</small>
        <small>Range: {min.toFixed(2)} - {max.toFixed(2)} mm</small>
        <small>Values: {values.slice(0, 30).map((v) => v.toFixed(2)).join(", ")}{values.length > 30 ? " ..." : ""}</small>
      </div>
    );
  },
  json: (props) => {
    if (!props.detail) return semanticEmpty(props.stageId, props.view.id);
    const semantic = stageSemanticDefinition(props.stageId);
    const context = resolveStageViewContext(props.detail, props.stageId, props.view.id, semantic.category, artifactsForStage(props.detail, props.stageId));
    const payload = semantic.category === "input" || semantic.category === "plane_qa"
      ? { take: { take_id: props.detail.take_id, modalities: props.detail.modalities, metadata: props.detail.metadata, assets: props.detail.assets, frameset: props.detail.frameset }, stage: props.stageId, semantic, resolvedArtifacts: context.resolvedArtifacts }
      : semantic.category === "segmentation"
        ? {
          stage: props.stageId,
          semantic,
          threshold_mask: context.runArtifacts.find((item) => item.artifact_id === "normalized_height_threshold_mask" || item.artifact_id === "threshold_mask") ?? null,
          cleaned_mask: context.runArtifacts.find((item) => item.artifact_id === "cleaned_object_mask" || item.artifact_id === "cleaned_mask") ?? null,
          connected_components_overlay: context.runArtifacts.find((item) => item.artifact_id === "connected_components_overlay") ?? null,
          overlay_image: context.runArtifacts.find((item) => item.artifact_id === "overlay_image") ?? null,
          segmentation_debug: context.runArtifacts.find((item) => item.artifact_id === "segmentation_debug")?.metadata ?? null,
        }
        : semantic.category === "geometry"
          ? {
            stage: props.stageId,
            semantic,
            artifacts: context.runArtifacts,
            ellipse_metrics: firstArtifactByAlias(context.runArtifacts, ["ellipse_metrics", "geometry_metrics"])?.metadata ?? null,
            ellipse_summary: firstArtifactByAlias(context.runArtifacts, ["ellipse_summary", "geometry_summary"])?.metadata ?? null,
            blob_metrics: firstArtifactByAlias(context.runArtifacts, ["blob_metrics", "detection_metrics"])?.metadata ?? null,
            blob_contours: firstArtifactByAlias(context.runArtifacts, ["blob_contours", "contours"])?.metadata ?? null,
            blob_rejected: firstArtifactByAlias(context.runArtifacts, ["blob_rejected"])?.metadata ?? null,
          }
          : { stage: props.stageId, semantic, artifacts: context.runArtifacts, result: props.detail.result };
    if (semantic.category === "plane_qa" && props.view.id === "surface_candidates") {
      const surfaceCandidates = context.runArtifacts.find((item) => item.artifact_id === "reference_surface_candidates");
      const gradientDebug = context.runArtifacts.find((item) => item.artifact_id === "gradient_debug");
      const selectedSurface = context.runArtifacts.find((item) => item.artifact_id === "selected_surface_debug");
      return <pre className="json-block">{JSON.stringify({ surface_candidates: surfaceCandidates?.metadata ?? null, gradient_debug: gradientDebug?.metadata ?? null, selected_surface_debug: selectedSurface?.metadata ?? null }, null, 2)}</pre>;
    }
    if (semantic.category === "plane_qa" && props.view.id === "reference_surface") {
      const selectedSurface = context.runArtifacts.find((item) => item.artifact_id === "selected_surface_debug");
      const planeFit = context.runArtifacts.find((item) => item.artifact_id === "plane_fit_debug");
      return <pre className="json-block">{JSON.stringify({ selected_surface_debug: selectedSurface?.metadata ?? null, plane_fit_debug: planeFit?.metadata ?? null }, null, 2)}</pre>;
    }
    return <pre className="json-block">{JSON.stringify(payload, null, 2)}</pre>;
  },
  gallery: (props) => {
    const semantic = stageSemanticDefinition(props.stageId);
    const artifacts = resolveStageViewContext(props.detail, props.stageId, props.view.id, semantic.category, artifactsForStage(props.detail, props.stageId)).resolvedArtifacts.filter((artifact) => artifact.kind === "image");
    return explore(artifacts, props);
  },
};

export function renderStageSemanticView(props: RendererProps): ReactElement {
  if (!props.detail) return <div className="empty-state">Select a take to inspect stage outputs.</div>;
  const semantic = stageSemanticDefinition(props.stageId);
  if (semantic.category !== "input" && semantic.category !== "plane_qa" && (!props.compatible || !props.processed)) return semanticEmpty(props.stageId, props.view.id);
  return (renderers[props.view.rendererType] ?? renderers.json)(props);
}

function fmtRange(value: { min?: number; median?: number; max?: number } | undefined): string {
  if (!value) return "-";
  const min = value.min == null ? "-" : value.min.toFixed(2);
  const med = value.median == null ? "-" : value.median.toFixed(2);
  const max = value.max == null ? "-" : value.max.toFixed(2);
  return `${min} / ${med} / ${max}`;
}

function ellipseEntries(artifacts: StudioArtifact[]): Array<Record<string, number | boolean | string | null>> {
  const metrics = firstArtifactByAlias(artifacts, ["ellipse_metrics", "geometry_metrics"]);
  const rows = ((metrics?.metadata as Record<string, unknown> | undefined)?.entries ?? []) as unknown[];
  if (!Array.isArray(rows)) return [];
  return rows.filter((item): item is Record<string, number | boolean | string | null> => typeof item === "object" && item !== null);
}

function fmtNum(value: unknown, decimals: number): string {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(decimals) : "-";
}

type OverlayMode = "contours" | "labels" | "bbox" | "filled" | "rejected_only";
type ClassificationOverlayFlags = "ids" | "labels" | "confidence" | "contours" | "geometry";

function BlobCandidateOverlayWorkspace({
  src,
  candidates,
  rejected,
  selectedCandidateId,
  hoveredCandidateId,
  onSelectCandidate,
  onHoverCandidate,
}: {
  src: string;
  candidates: Array<{
    id: string | number;
    contour?: [number, number][];
    centroid?: [number, number];
    bbox?: [number, number, number, number];
    areaPx?: number;
    equivalentDiameter?: number;
    circularity?: number;
    aspectRatio?: number;
  }>;
  rejected: Array<{
    id: string | number;
    contour?: [number, number][];
    centroid?: [number, number];
    bbox?: [number, number, number, number];
    rejectionReason?: string;
  }>;
  selectedCandidateId: number | null;
  hoveredCandidateId: number | null;
  onSelectCandidate: (id: number) => void;
  onHoverCandidate: (id: number | null) => void;
}) {
  const [mode, setMode] = useState<OverlayMode>("contours");
  const [imgSize, setImgSize] = useState<{ width: number; height: number } | null>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; row: (typeof candidates)[number] | (typeof rejected)[number]; rejected: boolean } | null>(null);

  const acceptedCount = candidates.length;
  const rejectedCount = rejected.length;
  const rows = mode === "rejected_only" ? rejected : [...candidates, ...rejected];

  return (
    <section className="image-panel overlay-image-panel blob-overlay-workspace">
      <div className="overlay-controls">
        <button className={mode === "contours" ? "active" : ""} onClick={() => setMode("contours")} type="button">Contours</button>
        <button className={mode === "labels" ? "active" : ""} onClick={() => setMode("labels")} type="button">Labels</button>
        <button className={mode === "bbox" ? "active" : ""} onClick={() => setMode("bbox")} type="button">BBox</button>
        <button className={mode === "filled" ? "active" : ""} onClick={() => setMode("filled")} type="button">Filled mask</button>
        <button className={mode === "rejected_only" ? "active" : ""} onClick={() => setMode("rejected_only")} type="button">Rejected only</button>
      </div>
      <div className="blob-overlay-legend">
        <small><span className="legend-chip accepted" /> accepted: {acceptedCount}</small>
        <small><span className="legend-chip rejected" /> rejected: {rejectedCount}</small>
        <small><span className="legend-chip selected" /> selected</small>
      </div>
      <div className="overlay-frame blob-overlay-frame">
        <img
          alt="Blob candidate overlay"
          className="artifact-image"
          draggable={false}
          onLoad={(event) => setImgSize({ width: event.currentTarget.naturalWidth || 1, height: event.currentTarget.naturalHeight || 1 })}
          src={src}
        />
        {imgSize && (
          <svg className="artifact-overlay-layer" viewBox={`0 0 ${imgSize.width} ${imgSize.height}`}>
            {rows.map((row) => {
              const id = Number(row.id);
              const isRejected = rejected.some((item) => Number(item.id) === id);
              const selected = selectedCandidateId != null && selectedCandidateId === id;
              const hovered = hoveredCandidateId != null && hoveredCandidateId === id;
              const contour = Array.isArray(row.contour) ? row.contour : [];
              const bbox = Array.isArray(row.bbox) ? row.bbox : null;
              const centroid = Array.isArray(row.centroid) ? row.centroid : null;
              const stroke = isRejected ? "#ef4444" : "#22c55e";
              const width = selected ? 4 : hovered ? 3 : isRejected ? 1.5 : 2.4;
              const alpha = selected ? 1 : isRejected ? 0.75 : 0.95;
              const label = `#${id}${isRejected && "rejectionReason" in row && row.rejectionReason ? ` (${row.rejectionReason})` : ""}`;
              const shapeProps = {
                stroke,
                strokeWidth: width,
                strokeOpacity: alpha,
                fill: mode === "filled" ? (isRejected ? "rgba(239,68,68,0.24)" : "rgba(34,197,94,0.22)") : "none",
                onClick: (event: MouseEvent<SVGElement>) => {
                  event.preventDefault();
                  event.stopPropagation();
                  onSelectCandidate(id);
                },
                onMouseEnter: (event: MouseEvent<SVGElement>) => {
                  event.preventDefault();
                  onHoverCandidate(id);
                  setTooltip({
                    x: centroid?.[0] ?? (bbox ? bbox[0] + bbox[2] / 2 : 0),
                    y: centroid?.[1] ?? (bbox ? bbox[1] + bbox[3] / 2 : 0),
                    row,
                    rejected: isRejected,
                  });
                },
                onMouseLeave: () => {
                  onHoverCandidate(null);
                  setTooltip(null);
                },
              };
              const parts: ReactElement[] = [];
              if (contour.length >= 2 && mode !== "bbox") {
                parts.push(
                  <polyline
                    key={`contour_${id}`}
                    {...shapeProps}
                    points={contour.map((p) => `${p[0]},${p[1]}`).join(" ")}
                    className={selected ? "blob-selected-shape" : ""}
                  />
                );
              }
              if (bbox && (mode === "bbox" || mode === "labels" || mode === "rejected_only")) {
                parts.push(
                  <rect
                    key={`bbox_${id}`}
                    {...shapeProps}
                    x={bbox[0]}
                    y={bbox[1]}
                    width={bbox[2]}
                    height={bbox[3]}
                    className={selected ? "blob-selected-shape" : ""}
                  />
                );
              }
              if (centroid && mode !== "bbox") {
                parts.push(
                  <circle
                    key={`centroid_${id}`}
                    cx={centroid[0]}
                    cy={centroid[1]}
                    r={selected ? 5 : 3.2}
                    fill={isRejected ? "#f97316" : "#06b6d4"}
                    opacity={0.95}
                  />
                );
                parts.push(
                  <text
                    key={`label_${id}`}
                    className={`artifact-overlay-text ${selected ? "selected" : ""}`}
                    x={centroid[0] + 7}
                    y={Math.max(14, centroid[1] - 7)}
                    fill={selected ? "#111827" : "#0f172a"}
                    stroke={selected ? "#f8fafc" : "none"}
                    strokeWidth={selected ? 0.8 : 0}
                  >
                    {label}
                  </text>
                );
              }
              return <g key={`row_${id}`}>{parts}</g>;
            })}
            {tooltip && (
              <g className="blob-tooltip">
                <rect x={Math.max(0, tooltip.x + 8)} y={Math.max(0, tooltip.y + 8)} width={210} height={88} rx={8} ry={8} fill="#0f172a" fillOpacity={0.88} />
                <text x={Math.max(0, tooltip.x + 18)} y={Math.max(0, tooltip.y + 28)} fill="#f8fafc">ID: {String(tooltip.row.id)}</text>
                <text x={Math.max(0, tooltip.x + 18)} y={Math.max(0, tooltip.y + 44)} fill="#f8fafc">Area: {"areaPx" in tooltip.row ? (tooltip.row.areaPx ?? "-") : "-"}</text>
                <text x={Math.max(0, tooltip.x + 18)} y={Math.max(0, tooltip.y + 60)} fill="#f8fafc">Diameter: {"equivalentDiameter" in tooltip.row ? (tooltip.row.equivalentDiameter?.toFixed?.(2) ?? "-") : "-"}</text>
                <text x={Math.max(0, tooltip.x + 18)} y={Math.max(0, tooltip.y + 76)} fill="#f8fafc">
                  {tooltip.rejected ? `Rejected: ${("rejectionReason" in tooltip.row && tooltip.row.rejectionReason) || "yes"}` : `Circularity: ${"circularity" in tooltip.row && tooltip.row.circularity != null ? tooltip.row.circularity.toFixed(3) : "-"}`}
                </text>
              </g>
            )}
          </svg>
        )}
      </div>
      {(acceptedCount === 0 || (acceptedCount === 0 && rejectedCount > 0)) && (
        <div className="empty-state">
          <strong>{acceptedCount === 0 && rejectedCount > 0 ? "All components were rejected." : "No blob candidates found."}</strong>
          <small>Try lowering `min_area`, lowering `min_circularity`, disabling border rejection, or reducing segmentation cleanup.</small>
        </div>
      )}
    </section>
  );
}

function ClassificationOverlayWorkspace({
  src,
  objects,
  selectedObjectId,
  hoveredObjectId,
  onSelectObject,
  onHoverObject,
}: {
  src: string;
  objects: Array<Record<string, unknown>>;
  selectedObjectId: number | null;
  hoveredObjectId: number | null;
  onSelectObject: (id: number) => void;
  onHoverObject: (id: number | null) => void;
}) {
  const [imgSize, setImgSize] = useState<{ width: number; height: number } | null>(null);
  const [flags, setFlags] = useState<Record<ClassificationOverlayFlags, boolean>>({
    ids: true,
    labels: true,
    confidence: true,
    contours: true,
    geometry: true,
  });
  return (
    <section className="image-panel overlay-image-panel blob-overlay-workspace">
      <div className="overlay-controls">
        <button className={flags.ids ? "active" : ""} onClick={() => setFlags((v) => ({ ...v, ids: !v.ids }))} type="button">IDs</button>
        <button className={flags.labels ? "active" : ""} onClick={() => setFlags((v) => ({ ...v, labels: !v.labels }))} type="button">Labels</button>
        <button className={flags.confidence ? "active" : ""} onClick={() => setFlags((v) => ({ ...v, confidence: !v.confidence }))} type="button">Confidence</button>
        <button className={flags.contours ? "active" : ""} onClick={() => setFlags((v) => ({ ...v, contours: !v.contours }))} type="button">Contours</button>
        <button className={flags.geometry ? "active" : ""} onClick={() => setFlags((v) => ({ ...v, geometry: !v.geometry }))} type="button">Geometry</button>
      </div>
      <div className="overlay-frame blob-overlay-frame">
        <img
          alt="Classification overlay"
          className="artifact-image"
          draggable={false}
          onLoad={(event) => setImgSize({ width: event.currentTarget.naturalWidth || 1, height: event.currentTarget.naturalHeight || 1 })}
          src={src}
        />
        {imgSize && (
          <svg className="artifact-overlay-layer" viewBox={`0 0 ${imgSize.width} ${imgSize.height}`}>
            {objects.map((row) => {
              const sourceObjectId = Number(row.source_object_id ?? 0);
              const id = Number.isFinite(sourceObjectId) && sourceObjectId > 0 ? sourceObjectId : Number(String(row.object_id ?? "").replace(/\D/g, ""));
              if (!Number.isFinite(id) || id <= 0) return null;
              const status = String(row.status ?? "suspicious");
              const color = status === "accepted" ? "#22c55e" : status === "rejected" ? "#ef4444" : "#f59e0b";
              const selected = selectedObjectId != null && selectedObjectId === id;
              const hovered = hoveredObjectId != null && hoveredObjectId === id;
              const width = selected ? 4 : hovered ? 3 : 2;
              const contour = Array.isArray(row.contour) ? row.contour as Array<[number, number]> : [];
              const bbox = row.bounding_box && typeof row.bounding_box === "object" ? row.bounding_box as Record<string, unknown> : {};
              const ellipse = row.ellipse && typeof row.ellipse === "object" ? row.ellipse as Record<string, unknown> : null;
              const x = Number(bbox.x ?? 0);
              const y = Number(bbox.y ?? 0);
              const w = Number(bbox.width ?? 0);
              const h = Number(bbox.height ?? 0);
              const labelText = [flags.ids ? String(row.object_id ?? `object_${id.toString().padStart(3, "0")}`) : "", flags.labels ? String(row.label ?? "") : "", flags.confidence ? `${Number(row.confidence ?? 0).toFixed(2)}` : ""].filter(Boolean).join(" • ");
              return (
                <g
                  key={`classification_obj_${id}`}
                  onClick={() => onSelectObject(id)}
                  onMouseEnter={() => onHoverObject(id)}
                  onMouseLeave={() => onHoverObject(null)}
                >
                  {flags.contours && contour.length >= 2 ? <polyline points={contour.map((p) => `${Number(p[0])},${Number(p[1])}`).join(" ")} stroke={color} strokeWidth={width} fill="none" opacity={0.95} /> : null}
                  {flags.geometry && w > 0 && h > 0 ? <rect x={x} y={y} width={w} height={h} stroke={color} strokeWidth={selected ? 2.5 : 1.5} fill="none" opacity={0.85} /> : null}
                  {flags.geometry && ellipse ? <ellipse cx={Number(ellipse.cx ?? 0)} cy={Number(ellipse.cy ?? 0)} rx={Number(ellipse.rx ?? 0)} ry={Number(ellipse.ry ?? 0)} transform={`rotate(${Number(ellipse.rotation_deg ?? 0)} ${Number(ellipse.cx ?? 0)} ${Number(ellipse.cy ?? 0)})`} stroke={color} strokeWidth={1.2} fill="none" opacity={0.9} /> : null}
                  {labelText ? <text className={`artifact-overlay-text ${selected ? "selected" : ""}`} x={Math.max(8, x)} y={Math.max(16, y - 6)} fill={selected ? "#111827" : "#0f172a"}>{labelText}</text> : null}
                </g>
              );
            })}
          </svg>
        )}
      </div>
      {!objects.length ? <div className="empty-state"><strong>No classification overlay objects available.</strong></div> : null}
    </section>
  );
}

function SourceImageHistogram({ src, srcKey, takeId, filename }: { src: string | null; srcKey: string; takeId: string; filename: string | null }) {
  const [state, setState] = useState<{ loading: boolean; error: string | null; stats: SourceHistogramResponse | null }>({
    loading: false,
    error: null,
    stats: null,
  });
  useEffect(() => {
    let active = true;
    if (!src) {
      setState({ loading: false, error: "No source image available for histogram.", stats: null });
      return;
    }
    setState({ loading: true, error: null, stats: null });
    const timeoutId = window.setTimeout(() => {
      if (!active) return;
      setState({ loading: false, error: "Histogram computation failed: timeout", stats: null });
    }, 5000);
    void api.takeSourceHistogram(takeId, filename ?? undefined, 512, 32).then((result) => {
      if (!active) return;
      window.clearTimeout(timeoutId);
      setState({ loading: false, error: null, stats: result });
    }).catch(() => {
      if (!active) return;
      window.clearTimeout(timeoutId);
      setState({ loading: false, error: "Histogram computation failed: unexpected error", stats: null });
    });
    return () => {
      active = false;
      window.clearTimeout(timeoutId);
    };
  }, [src, srcKey, takeId, filename]);
  const max = useMemo(() => Math.max(...(state.stats?.bins ?? [0])), [state.stats?.bins]);
  if (!src) return <div className="empty-state">No source image available for histogram.</div>;
  if (state.loading) return <div className="artifact-reference"><strong>Histogram loading...</strong></div>;
  if (state.error) return <div className="empty-state">{state.error}</div>;
  if (!state.stats) return <div className="empty-state">Histogram unavailable for this source image.</div>;
  const stats = state.stats;
  return (
    <div className="artifact-reference">
      <span>Histogram</span>
      <strong>Source image grayscale histogram</strong>
      <small>Image: {stats.sampled_width} x {stats.sampled_height}</small>
      <small>Pixels sampled: {stats.sampled_pixels}</small>
      <small>Min/Max: {stats.min} / {stats.max}</small>
      <small>Mean: {stats.mean.toFixed(2)}</small>
      <div className="histogram-bars">
        {stats.bins.map((value, idx) => (
          <div key={`bin_${idx}`} className="histogram-bar-wrap">
            <div className="histogram-bar" style={{ height: `${max ? Math.max(4, (value / max) * 80) : 4}px` }} />
            <small>{Math.round((idx / stats.bin_count) * 255)}</small>
          </div>
        ))}
      </div>
    </div>
  );
}
