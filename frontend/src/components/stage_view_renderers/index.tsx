import { useEffect, useMemo, useState, type MouseEvent, type ReactElement } from "react";
import ObjectTable from "../ObjectTable";
import StudioArtifactExplorer from "../StudioArtifactExplorer";
import StudioObjectList from "../StudioObjectList";
import type { OverlayDebugInfo } from "../overlayModel";
import { api, fileUrl, type SourceHistogramResponse, type TakeDetail } from "../../api/client";
import type { StudioArtifact } from "../studioWorkspaceModel";
import type { ResolvedProcessingUnitView } from "../processingUnitModel";
import { artifactsForStage, firstArtifactByAlias } from "../studioWorkspaceModel";
import { stageEmptyState, stageSemanticDefinition, type StageViewDefinition } from "../stageSemantics";
import { primarySourceImageUrl, resolveStageViewContext } from "../stage_sources";
import type { RoiPolygon, RoiRect } from "../roiMapping";
import type { RoiType } from "../ProjectionArtifactViewer";
import { normalizeBlobDetectionArtifacts, resolveBlobSetSourceImage } from "../../studio/blobDetectionModel";
import BlobSetViewer from "../BlobSetViewer";
import LowGradientComponentsViewer from "../LowGradientComponentsViewer";
import type { StudioDebugMode } from "../studioDebugMode";
import type { HoverInspectionSample } from "../hoverInspectionModel";
import { resolveHeightArtifact } from "../heightSemantics";
import { objectDisplayId } from "../objectSelectionModel";
import { preferredDetectArtifactsForView, resolveDetectReferenceRunState, resolveDetectViewArtifacts, type ArtifactRelevance } from "../referenceArtifactRelevance";
import MetricDetailsPanel from "../MetricDetailsPanel";
import DetectClusterDecision from "../DetectClusterDecision";
import type { StudioArtifactViewState } from "../studioViewState";
import SelectedSupportLineagePanel from "../SelectedSupportLineagePanel";
import StageMaskArtifactRenderer from "./StageMaskArtifactRenderer";

export type RendererProps = {
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
  roiType?: RoiType;
  roiEnabled?: boolean;
  onApplyRoi?: (roi: RoiRect, rerun: boolean) => void;
  onApplyPolygonRoi?: (points: RoiPolygon, rerun: boolean) => void;
  onClearRoi?: () => void;
  roiEditModeActive?: boolean;
  roiEditStatus?: "idle" | "active" | "unsaved" | "saved" | "rerun_required";
  onCancelRoiEdit?: () => void;
  blobParams?: Record<string, unknown> | null;
  onApplyBlobParams?: (params: Record<string, unknown>, rerun: boolean) => void;
  onUpsertObjectAnnotation?: (payload: Record<string, unknown>) => void;
  debugMode?: StudioDebugMode;
  onHoverSample?: (sample: HoverInspectionSample | null) => void;
  viewState: StudioArtifactViewState;
  onViewportModeChange: (mode: StudioArtifactViewState["viewportMode"]) => void;
  onOverlayModeChange: (mode: StudioArtifactViewState["overlayMode"]) => void;
  onBinaryMaskModeChange: (mode: StudioArtifactViewState["binaryMaskMode"]) => void;
  onBinaryMaskReferenceArtifactIdChange: (artifactId: string | null) => void;
  onBinaryMaskOpacityChange?: (value: number) => void;
  onBinaryMaskReferenceOpacityChange?: (value: number) => void;
  onResetViewState: () => void;
  activeBlobClusterId?: string | null;
  onActiveBlobClusterChange?: (clusterId: string | null) => void;
  onFocusLineageStep?: (stepId: string, options?: { openTune?: boolean }) => void;
  resolvedViewArtifacts?: StudioArtifact[];
  unitViewResolution?: ResolvedProcessingUnitView | null;
  artifactRelevance?: Record<string, { relevance: ArtifactRelevance; reason?: string }>;
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
  if (!artifacts.length) {
    if (props.unitViewResolution?.emptyState) {
      return <div className="empty-state"><strong>{props.unitViewResolution.label}</strong><small>{props.unitViewResolution.emptyState}</small></div>;
    }
    return semanticEmpty(props.stageId, props.view.id);
  }
  return (
    <StudioArtifactExplorer
      artifacts={artifacts}
      takeId={props.detail?.take_id}
      selectedArtifactId={props.selectedArtifactId}
      onSelect={props.onSelectArtifact}
      detailJson={props.detail?.result}
      takeDetail={props.detail}
      currentStageArtifacts={artifacts}
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
      roiEditModeActive={Boolean(props.roiEditModeActive)}
      roiEditStatus={props.roiEditStatus}
      onCancelRoiEdit={props.onCancelRoiEdit}
      debugMode={props.debugMode ?? "operator"}
      onHoverSample={props.onHoverSample}
      viewState={props.viewState}
      onViewportModeChange={props.onViewportModeChange}
      onOverlayModeChange={props.onOverlayModeChange}
      onBinaryMaskModeChange={props.onBinaryMaskModeChange}
      onBinaryMaskReferenceArtifactIdChange={props.onBinaryMaskReferenceArtifactIdChange}
      onBinaryMaskOpacityChange={props.onBinaryMaskOpacityChange}
      onBinaryMaskReferenceOpacityChange={props.onBinaryMaskReferenceOpacityChange}
      onResetViewState={props.onResetViewState}
      artifactRelevance={props.artifactRelevance}
    />
  );
}

const renderers: Record<string, (props: RendererProps) => ReactElement> = {
  image: (props) => {
    const semantic = stageSemanticDefinition(props.stageId);
    const runArtifacts = artifactsForStage(props.detail, props.stageId);
    const context = resolveStageViewContext(props.detail, props.stageId, props.view.id, semantic.category, runArtifacts);
    const candidateArtifacts = (() => {
      if (props.resolvedViewArtifacts?.length) {
        return props.resolvedViewArtifacts;
      }
      if (props.view.id === "below_reference" || props.view.id === "above_threshold") {
        return [
          ...context.resolvedArtifacts,
          ...(props.detail?.result?.artifacts ?? []).filter((artifact) => !context.resolvedArtifacts.some((item) => item.artifact_id === artifact.artifact_id && item.stage_id === artifact.stage_id)),
        ];
      }
      return context.resolvedArtifacts;
    })();
    const detectRunState = props.stageId === "detect_belt_plane"
      ? resolveDetectReferenceRunState(props.detail, runArtifacts, null)
      : null;
    const preferredDetectArtifacts = props.stageId === "detect_belt_plane" && detectRunState
      ? preferredDetectArtifactsForView(props.view.id, candidateArtifacts, detectRunState)
      : [];
    const detectViewResolution = props.stageId === "detect_belt_plane" && detectRunState
      ? resolveDetectViewArtifacts(props.view.id, candidateArtifacts, detectRunState)
      : null;
    const rawPreview = resolveHeightArtifact(candidateArtifacts, { semanticField: "raw_sensor_z", representation: "preview_image" }).artifact;
    const normalizedPreview = resolveHeightArtifact(candidateArtifacts, { semanticField: "height_above_belt", representation: "preview_image" }).artifact;
    const residualPreview = resolveHeightArtifact(candidateArtifacts, { semanticField: "plane_signed_distance", representation: "preview_image" }).artifact;
    const artifacts = candidateArtifacts.filter((artifact) => {
      if (props.stageId === "detect_belt_plane" && preferredDetectArtifacts.length > 0) {
        const preferredIds = new Set(preferredDetectArtifacts.map((item) => item.artifact_id));
        if (
          props.view.id === "blob_components"
          || props.view.id === "height_aware_blob_components"
          || props.view.id === "height_aware_connectivity_rejected_edges"
          || props.view.id === "height_borders"
          || props.view.id === "height_border_fragments"
          || props.view.id === "height_split_blobs"
          || props.view.id === "selected_blob_cluster"
          || props.view.id === "selected_cluster_pre_refine"
          || props.view.id === "selected_cluster_refined"
          || props.view.id === "removed_by_refinement"
          || props.view.id === "rejected_blob_clusters"
          || props.view.id === "belt_base"
          || props.view.id === "belt_stripes"
          || props.view.id === "belt_stripes_tophat"
          || props.view.id === "belt_stripes_shape"
          || props.view.id === "belt_above_belt"
          || props.view.id === "belt_wide_object"
          || props.view.id === "removed_by_stripe_filter"
          || props.view.id === "belt_altitude_plot"
          || props.view.id === "belt_baseline"
          || props.view.id === "reference_model_support"
          || props.view.id === "reference_suppression_mask"
          || props.view.id === "surface_candidates"
          || props.view.id === "plateau_plot"
          || props.view.id === "filtered_depth_plot"
        ) {
          return preferredIds.has(artifact.artifact_id);
        }
      }
      if (artifact.kind !== "image") return false;
      if (props.view.id === "height_preview") {
        return artifact.artifact_id === "source_heightmap_preview"
          || /heightmap_preview/i.test(String(artifact.path ?? ""))
          || artifact.artifact_id === "heightmap"
          || artifact.artifact_id === "input_preview";
      }
      if (props.view.id === "heightmap" || props.view.id === "raw_heightmap") {
        if (rawPreview) return artifact.artifact_id === rawPreview.artifact_id;
        return artifact.artifact_id === "heightmap" || artifact.artifact_id === "input_preview" || artifact.artifact_id === "raw_heightmap_preview";
      }
      if (props.view.id === "depth_gradient") return artifact.artifact_id === "depth_gradient_magnitude" || /depth_gradient_magnitude/i.test(String(artifact.path ?? ""));
      if (props.view.id === "low_gradient_mask") return artifact.artifact_id === "low_gradient_mask" || /low_gradient_mask/i.test(String(artifact.path ?? ""));
      if (props.view.id === "selected_surface") {
        return detectViewResolution?.primary != null
          ? artifact.artifact_id === detectViewResolution.primary.artifact_id
          : artifact.artifact_id === "reference_surface_selected_mask" || artifact.artifact_id === "expanded_plane_mask" || /reference_surface_selected_mask|expanded_plane_mask/i.test(String(artifact.path ?? ""));
      }
      if (props.view.id === "normalized_height") {
        if (normalizedPreview) return artifact.artifact_id === normalizedPreview.artifact_id;
        return artifact.artifact_id === "normalized_heightmap" || /normalized_heightmap/i.test(String(artifact.path ?? ""));
      }
      if (props.view.id === "residuals") {
        if (residualPreview) return artifact.artifact_id === residualPreview.artifact_id;
        return artifact.artifact_id === "belt_plane_residuals"
          || artifact.artifact_id === "sphere_fit_residual_heatmap"
          || /residual/i.test(String(artifact.artifact_id ?? ""))
          || /residual/i.test(String(artifact.path ?? ""));
      }
      if (props.view.id === "below_reference") return artifact.artifact_id === "below_reference_mask" || /below_reference_mask/i.test(String(artifact.path ?? ""));
      if (props.view.id === "above_threshold") return artifact.artifact_id === "above_threshold_mask" || /above_threshold_mask/i.test(String(artifact.path ?? ""));
      if (props.view.id === "valid_mask") return artifact.artifact_id === "valid_mask" || /valid_mask/i.test(String(artifact.path ?? ""));
      if (props.view.id === "background_candidates" || props.view.id === "background_seeds") return artifact.artifact_id === "background_seed_mask" || artifact.artifact_id === "background_candidate_mask" || /background_(seed|candidate)_mask/i.test(String(artifact.path ?? ""));
      if (props.view.id === "expanded_plane") return artifact.artifact_id === "expanded_plane_mask" || /expanded_plane_mask/i.test(String(artifact.path ?? ""));
      if (props.view.id === "depth_plot") return artifact.artifact_id === "background_depth_plot" || /background_depth_plot/i.test(String(artifact.path ?? ""));
      if (props.view.id === "plateau_plot") {
        return artifact.artifact_id === "background_plateau_plot"
          || /background_plateau_plot/i.test(String(artifact.path ?? ""));
      }
      if (props.view.id === "filtered_depth_plot") {
        return artifact.artifact_id === "flat_candidate_depth_plot"
          || /flat_candidate_depth_plot/i.test(String(artifact.path ?? ""));
      }
      if (props.view.id === "blob_components") {
        return artifact.artifact_id === "low_gradient_blob_components_overlay"
          || artifact.artifact_id === "low_gradient_blob_id_mask"
          || /low_gradient_blob_components_overlay|low_gradient_blob_id_mask/i.test(String(artifact.path ?? ""));
      }
      if (props.view.id === "height_aware_blob_components") {
        return artifact.artifact_id === "height_aware_blob_components_overlay"
          || artifact.artifact_id === "height_aware_blob_id_mask"
          || /height_aware_blob_components_overlay|height_aware_blob_id_mask/i.test(String(artifact.path ?? ""));
      }
      if (props.view.id === "height_aware_connectivity_rejected_edges") {
        return artifact.artifact_id === "height_aware_connectivity_rejected_edges"
          || artifact.artifact_id === "height_aware_connectivity_debug"
          || /height_aware_connectivity_rejected_edges|height_aware_connectivity_debug/i.test(String(artifact.path ?? ""));
      }
      if (props.view.id === "height_borders") {
        return artifact.artifact_id === "height_border_strength"
          || artifact.artifact_id === "height_border_cut_mask"
          || artifact.artifact_id === "height_border_split_debug"
          || /height_border_strength|height_border_cut_mask|height_border_split_debug/i.test(String(artifact.path ?? ""));
      }
      if (props.view.id === "height_border_fragments") {
        return artifact.artifact_id === "height_border_fragments_overlay"
          || artifact.artifact_id === "height_border_fragments_mask"
          || artifact.artifact_id === "height_border_fragments_id_mask"
          || artifact.artifact_id === "fragment_merge_debug"
          || /height_border_fragments_overlay|height_border_fragments_mask|height_border_fragments_id_mask|fragment_merge_debug/i.test(String(artifact.path ?? ""));
      }
      if (props.view.id === "height_split_blobs") {
        return artifact.artifact_id === "height_split_blob_fragments_overlay"
          || artifact.artifact_id === "height_split_blob_fragments_mask"
          || artifact.artifact_id === "height_split_blob_fragments_pre_merge_mask"
          || artifact.artifact_id === "height_split_blob_id_mask"
          || artifact.artifact_id === "height_split_debug"
          || artifact.artifact_id === "height_consistent_blob_summary"
          || /height_split_blob_fragments_overlay|height_split_blob_fragments_mask|height_split_blob_fragments_pre_merge_mask|height_split_blob_id_mask|height_split_debug|height_consistent_blob_summary/i.test(String(artifact.path ?? ""));
      }
      if (props.view.id === "blob_clusters") {
        return artifact.artifact_id === "blob_height_clusters"
          || artifact.artifact_id === "low_gradient_blob_summary"
          || /blob_height_clusters|low_gradient_blob_summary/i.test(String(artifact.path ?? ""));
      }
      if (props.view.id === "selected_blob_cluster") {
        return artifact.artifact_id === "selected_blob_cluster_overlay"
          || artifact.artifact_id === "selected_blob_cluster_mask"
          || /selected_blob_cluster_overlay|selected_blob_cluster_mask/i.test(String(artifact.path ?? ""));
      }
      if (props.view.id === "selected_cluster_pre_refine") {
        return artifact.artifact_id === "selected_blob_cluster_pre_refine_mask"
          || artifact.artifact_id === "selected_blob_cluster_mask"
          || /selected_blob_cluster_pre_refine_mask|selected_blob_cluster_mask/i.test(String(artifact.path ?? ""));
      }
      if (props.view.id === "selected_cluster_refined") {
        return artifact.artifact_id === "selected_blob_cluster_refined_mask"
          || artifact.artifact_id === "selected_reference_support_mask"
          || artifact.artifact_id === "selected_blob_cluster_pre_refine_mask"
          || /selected_blob_cluster_refined_mask|selected_reference_support_mask|selected_blob_cluster_pre_refine_mask/i.test(String(artifact.path ?? ""));
      }
      if (props.view.id === "removed_by_refinement") {
        return artifact.artifact_id === "support_removed_by_candidate_refinement"
          || /support_removed_by_candidate_refinement/i.test(String(artifact.path ?? ""));
      }
      if (props.view.id === "rejected_blob_clusters") {
        return artifact.artifact_id === "rejected_blob_clusters_overlay"
          || artifact.artifact_id === "rejected_blob_clusters_mask"
          || /rejected_blob_clusters_overlay|rejected_blob_clusters_mask/i.test(String(artifact.path ?? ""));
      }
      if (props.view.id === "blob_cluster_scores") {
        return artifact.artifact_id === "blob_cluster_score_table"
          || artifact.artifact_id === "blob_cluster_selection_debug"
          || /blob_cluster_score_table|blob_cluster_selection_debug/i.test(String(artifact.path ?? ""));
      }
      if (props.view.id === "belt_base") {
        return artifact.artifact_id === "belt_base_mask"
          || /belt_base_mask/i.test(String(artifact.path ?? ""));
      }
      if (props.view.id === "belt_stripes") {
        return artifact.artifact_id === "belt_stripes_mask"
          || /belt_stripes_mask/i.test(String(artifact.path ?? ""));
      }
      if (props.view.id === "belt_stripes_tophat") {
        return artifact.artifact_id === "belt_stripes_tophat_mask"
          || /belt_stripes_tophat_mask/i.test(String(artifact.path ?? ""));
      }
      if (props.view.id === "belt_stripes_shape") {
        return artifact.artifact_id === "belt_stripes_shape_mask"
          || /belt_stripes_shape_mask/i.test(String(artifact.path ?? ""));
      }
      if (props.view.id === "belt_above_belt") {
        return artifact.artifact_id === "belt_above_belt_mask"
          || /belt_above_belt_mask/i.test(String(artifact.path ?? ""));
      }
      if (props.view.id === "belt_wide_object") {
        return artifact.artifact_id === "belt_wide_object_mask"
          || /belt_wide_object_mask/i.test(String(artifact.path ?? ""));
      }
      if (props.view.id === "removed_by_stripe_filter") {
        return artifact.artifact_id === "support_removed_by_stripe_filter"
          || /support_removed_by_stripe_filter/i.test(String(artifact.path ?? ""));
      }
      if (props.view.id === "belt_altitude_plot") {
        return artifact.artifact_id === "belt_altitude_histogram_image"
          || artifact.artifact_id === "belt_altitude_local_min"
          || /belt_altitude_histogram|belt_altitude_local_min/i.test(String(artifact.path ?? ""));
      }
      if (props.view.id === "belt_baseline") {
        return artifact.artifact_id === "belt_baseline_local_min"
          || /belt_baseline_local_min/i.test(String(artifact.path ?? ""));
      }
      if (props.view.id === "reference_model_support") {
        return artifact.artifact_id === "reference_model_support_mask"
          || artifact.artifact_id === "selected_reference_support_mask"
          || /reference_model_support_mask|selected_reference_support_mask/i.test(String(artifact.path ?? ""));
      }
      if (props.view.id === "reference_suppression_mask") {
        return artifact.artifact_id === "reference_suppression_mask"
          || /reference_suppression_mask/i.test(String(artifact.path ?? ""));
      }
      if (props.view.id === "plane_inliers") {
        return detectViewResolution?.primary != null
          ? artifact.artifact_id === detectViewResolution.primary.artifact_id
          : artifact.artifact_id === "final_plane_inlier_mask" || artifact.artifact_id === "plane_inlier_mask" || /final_plane_inlier_mask|plane_inlier_mask/i.test(String(artifact.path ?? ""));
      }
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
    }).concat(
      props.view.id === "normalized_height"
        ? candidateArtifacts.filter((artifact) =>
          (artifact.artifact_id === "plane_fit_debug" || artifact.artifact_id === "normalized_height_histogram" || artifact.artifact_id === "normalized_heightmap_display")
          && artifact.kind !== "image",
        )
        : [],
    );
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
    if (
      props.view.id === "depth_plot"
      || props.view.id === "plateau_plot"
      || props.view.id === "filtered_depth_plot"
      || props.view.id === "belt_altitude_plot"
    ) {
      return explore(artifacts, props, { disableRoi: true });
    }
    if (props.detail && props.stageId === "detect_belt_plane" && (props.view.id === "selected_blob_cluster" || props.view.id === "rejected_blob_clusters")) {
      const stageArtifacts = artifactsForStage(props.detail, props.stageId);
      return (
        <div className="cluster-decision-with-preview">
          <DetectClusterDecision
            takeId={props.detail.take_id}
            detail={props.detail}
            artifacts={stageArtifacts}
            mode="card"
            showSpatialPreview
            activeClusterId={props.activeBlobClusterId}
            onActiveClusterChange={props.onActiveBlobClusterChange}
          />
          {explore(artifacts, props)}
        </div>
      );
    }
    if (props.detail && props.stageId === "detect_belt_plane" && props.view.id === "selected_surface") {
      const stageArtifacts = artifactsForStage(props.detail, props.stageId);
      const selectedSurfaceDebug = ((stageArtifacts.find((item) => item.artifact_id === "selected_surface_debug")?.metadata ?? {}) as Record<string, unknown>);
      const resolvedArtifact = String(selectedSurfaceDebug.selected_surface_alias_resolves_to ?? detectViewResolution?.primary?.artifact_id ?? "Not available");
      const resolvedRole = String(selectedSurfaceDebug.selected_surface_alias_role ?? "selected_support");
      return (
        <div>
          <div className="artifact-reference">
            <span>Selected surface</span>
            <small>Resolved artifact: {resolvedArtifact}</small>
            <small>Role: {resolvedRole.replace(/_/g, " ")}</small>
          </div>
          {explore(artifacts, props)}
        </div>
      );
    }
    return explore(artifacts, props);
  },
  overlay: (props) => {
    const semantic = stageSemanticDefinition(props.stageId);
    const runArtifacts = artifactsForStage(props.detail, props.stageId);
    const context = resolveStageViewContext(props.detail, props.stageId, props.view.id, semantic.category, runArtifacts);
    let artifacts = (props.resolvedViewArtifacts?.length ? props.resolvedViewArtifacts : context.resolvedArtifacts)
      .filter((artifact) => artifact.kind === "overlay" || artifact.kind === "image");
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
      const source = resolveBlobSetSourceImage(context.resolvedArtifacts);
      if (source?.path) {
        return (
          <BlobSetViewer
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
    if (props.resolvedViewArtifacts?.length && props.stageId !== "detect_belt_plane") {
      return explore(props.resolvedViewArtifacts, props, { disableRoi: true });
    }
    if (props.stageId === "detect_belt_plane" && props.resolvedViewArtifacts?.length && props.view.id !== "selection_bridge" && props.view.id !== "blob_cluster_scores" && props.view.id !== "blob_clusters") {
      return explore(props.resolvedViewArtifacts, props, { disableRoi: true });
    }
    if (props.stageId === "detect_belt_plane" && (props.view.id === "blob_clusters" || props.view.id === "blob_cluster_scores" || props.view.id === "selection_bridge")) {
      const stageArtifacts = artifactsForStage(props.detail, props.stageId);
      return (
        <DetectClusterDecision
          takeId={props.detail.take_id}
          detail={props.detail}
          artifacts={stageArtifacts}
          mode={props.view.id === "selection_bridge" ? "bridge" : "table"}
          showSpatialPreview
          activeClusterId={props.activeBlobClusterId}
          onActiveClusterChange={props.onActiveBlobClusterChange}
        />
      );
    }
    if (props.stageId === "detect_belt_plane" && props.view.id === "selected_support_lineage") {
      const stageArtifacts = artifactsForStage(props.detail, props.stageId);
      return (
        <SelectedSupportLineagePanel
          artifacts={stageArtifacts}
          takeId={props.detail.take_id}
          detail={props.detail}
          selectedArtifactId={props.selectedArtifactId}
          onSelectArtifact={props.onSelectArtifact}
          onFocusLineageStep={props.onFocusLineageStep}
        />
      );
    }
    if (props.stageId === "detect_belt_plane" && props.view.id === "support_loss_waterfall") {
      const stageArtifacts = artifactsForStage(props.detail, props.stageId);
      const waterfallArtifact = stageArtifacts.find((item) => item.artifact_id === "support_loss_waterfall");
      const rows = Array.isArray((waterfallArtifact?.metadata as Record<string, unknown> | undefined)?.rows)
        ? ((waterfallArtifact?.metadata as Record<string, unknown>).rows as Array<Record<string, unknown>>)
        : [];
      const fmtPct = (value: unknown) => Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(1)}%` : "-";
      const largestLoss = rows
        .filter((row) => Number.isFinite(Number(row.delta_percent_valid_roi_from_previous)))
        .sort((a, b) => Number(a.delta_percent_valid_roi_from_previous) - Number(b.delta_percent_valid_roi_from_previous))[0] ?? null;
      if (!rows.length) return semanticEmpty(props.stageId, props.view.id);
      return (
        <div className="studio-table-pane">
          <div className="pipeline-status-grid">
            <div><span>Rows</span><strong>{rows.length}</strong></div>
            <div><span>Largest loss</span><strong>{largestLoss ? String(largestLoss.label ?? largestLoss.row_id ?? "-") : "-"}</strong></div>
            <div><span>Loss amount</span><strong>{largestLoss ? fmtPct(Math.abs(Number(largestLoss.delta_percent_valid_roi_from_previous ?? 0))) : "-"}</strong></div>
            <div><span>Final plane inliers</span><strong>{fmtPct(rows[rows.length - 1]?.percent_valid_roi)}</strong></div>
          </div>
          <table className="compact-candidate-table">
            <thead>
              <tr>
                <th>Step</th><th>Substage</th><th>Area px</th><th>% valid ROI</th><th>% previous</th><th>Delta ROI</th><th>Status</th><th>Reason</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={String(row.row_id ?? row.label ?? Math.random())}>
                  <td>{String(row.label ?? row.row_id ?? "-")}</td>
                  <td>{String(row.substage_id ?? "-").replace(/_/g, " ")}</td>
                  <td>{Number.isFinite(Number(row.area_px)) ? Number(row.area_px).toFixed(0) : "-"}</td>
                  <td>{fmtPct(row.percent_valid_roi)}</td>
                  <td>{fmtPct(row.percent_previous)}</td>
                  <td>{fmtPct(row.delta_percent_valid_roi_from_previous)}</td>
                  <td>{String(row.status ?? "-")}</td>
                  <td>{String(row.largest_loss_reason ?? row.warning ?? row.notes ?? "-")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
    }
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
      const detectDebugArtifacts = stageArtifacts
        .filter((item) => item.kind === "json" || item.kind === "table")
        .filter((item) => typeof ((item.metadata ?? {}) as Record<string, unknown>).compact_summary === "object");
      return (
        <div>
          {detectDebugArtifacts.length > 0 ? (
            <section className="artifact-reference">
              <span>Compact debug summaries</span>
              {detectDebugArtifacts.map((artifact) => {
                const summary = (((artifact.metadata ?? {}) as Record<string, unknown>).compact_summary ?? {}) as Record<string, unknown>;
                return (
                  <small key={artifact.artifact_id}>
                    {String(summary.title ?? artifact.title)}: input {String(summary.input_count ?? "-")} • output {String(summary.output_count ?? "-")} • selected {String(summary.selected_id ?? "-")} • fallback {String(summary.fallback_used ?? false)}
                  </small>
                );
              })}
            </section>
          ) : null}
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
        </div>
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
      const stageArtifacts = artifactsForStage(props.detail, props.stageId);
      const knownScale = firstArtifactByAlias(stageArtifacts, ["known_object_scale_validation"])
        ?? firstArtifactByAlias(props.detail.result?.artifacts ?? [], ["known_object_scale_validation"]);
      const knownMeta = (knownScale?.metadata ?? {}) as Record<string, unknown>;
      const stageParams = ((props.detail.result as unknown as Record<string, unknown> | null)?.stage_params ?? {}) as Record<string, unknown>;
      const knownParams = (stageParams.known_object_25d ?? {}) as Record<string, unknown>;
      const scaleX = knownMeta.recommended_scale_correction_x ?? knownMeta.persisted_scale_correction_x ?? knownParams.persisted_scale_correction_x;
      const scaleY = knownMeta.recommended_scale_correction_y ?? knownMeta.persisted_scale_correction_y ?? knownParams.persisted_scale_correction_y;
      const scaleZ = knownMeta.recommended_scale_correction_z ?? knownMeta.persisted_scale_correction_z ?? knownParams.persisted_scale_correction_z;
      const kept = objects.filter((item) => item.filter_status !== "rejected");
      const avgDiameter = kept.length ? kept.reduce((acc, obj) => acc + Number(obj.diameter_mm ?? obj.diameter_estimate_mm ?? 0), 0) / kept.length : 0;
      const avgCircularity = kept.length ? kept.reduce((acc, obj) => acc + Number(obj.sphericity_score ?? 0), 0) / kept.length : 0;
      const measurementArtifacts = resolveStageViewContext(props.detail, props.stageId, props.view.id, semantic.category, stageArtifacts).resolvedArtifacts;
      const thumbnailSource = measurementThumbnailSource(props.detail, [...measurementArtifacts, ...(props.detail.result?.artifacts ?? [])]);
      return (
        <section className="measurement-stage-layout">
          <section className="measurement-kpi-row" aria-label="Measurement stage summary">
            <div className="measurement-kpi-card"><span title="Objects">Objects</span><strong>{kept.length}</strong></div>
            <div className="measurement-kpi-card"><span title="Rejected">Rejected</span><strong>{objects.length - kept.length}</strong></div>
            <div className="measurement-kpi-card"><span title="Average diameter">Avg Ø</span><strong>{kept.length ? `${avgDiameter.toFixed(2)} mm` : "-"}</strong></div>
            <div className="measurement-kpi-card"><span title="Average circularity">Circ.</span><strong>{kept.length ? avgCircularity.toFixed(2) : "-"}</strong></div>
            <div className="measurement-kpi-card"><span title="Scale X">Scale X</span><strong>{scaleX == null ? "-" : Number(scaleX).toFixed(4)}</strong></div>
            <div className="measurement-kpi-card"><span title="Scale Y">Scale Y</span><strong>{scaleY == null ? "-" : Number(scaleY).toFixed(4)}</strong></div>
            <div className="measurement-kpi-card"><span title="Scale Z">Scale Z</span><strong>{scaleZ == null ? "-" : Number(scaleZ).toFixed(4)}</strong></div>
          </section>
          <section className="measurement-main-split result-table-mode">
            <section className="measurement-selected-workspace">
              <section className="measurement-results-section">
                <div className="measurement-section-heading">
                  <div>
                    <span>Object results</span>
                    <strong>Height, volume, footprint, and classification metrics</strong>
                  </div>
                  <small>{objects.length} object{objects.length === 1 ? "" : "s"}</small>
                </div>
                <MeasurementResultsTable
                  objects={objects}
                  selectedObjectId={props.selectedObjectId}
                  hoveredObjectId={props.hoveredObjectId}
                  thumbnailSrc={thumbnailSource}
                  onSelectObject={props.onSelectObject}
                  onHoverObject={props.onHoverObject}
                />
              </section>
              <section className="measurement-overlay-section">
                <div className="measurement-section-heading compact">
                  <div>
                    <span>Artifact evidence</span>
                    <strong>Height map, residuals, and overlays</strong>
                  </div>
                </div>
                {explore(measurementArtifacts, props)}
              </section>
              <details className="measurement-table-section">
                <summary>Raw object table</summary>
                <ObjectTable objects={objects} selectedObjectId={props.selectedObjectId} hoveredObjectId={props.hoveredObjectId} onSelectObject={props.onSelectObject} onHoverObject={props.onHoverObject} />
              </details>
            </section>
          </section>
        </section>
      );
    }
    if (semantic.category === "measurement" && props.view.id === "geometry_debug") {
      const stageArtifacts = artifactsForStage(props.detail, props.stageId);
      const debugArtifacts = stageArtifacts.filter((item) =>
        item.artifact_id === "geometry_debug_summary"
        || String(item.artifact_id ?? "").startsWith("geometry_debug_object_")
        || String(((item.metadata ?? {}) as Record<string, unknown>).semantic_type ?? "") === "geometry_debug"
      );
      if (!debugArtifacts.length) return semanticEmpty(props.stageId, props.view.id);
      return explore(debugArtifacts, props);
    }
    if (semantic.category === "measurement" && props.view.id === "provenance") {
      const objects = [...(props.detail.result?.objects ?? []), ...(props.detail.result?.rejected_objects ?? [])];
      const rows = objects.map((obj) => ({
        ...(obj as Record<string, unknown>),
        object_id: obj.object_id,
        bbox_px: (obj as Record<string, unknown>).bbox_px ?? null,
        segmented_bbox_mm: { min: obj.bbox_min_mm ?? null, max: obj.bbox_max_mm ?? null },
        segmentation_coverage: (obj as Record<string, unknown>).segmentation_coverage ?? null,
        removed_points: (obj as Record<string, unknown>).removed_points ?? null,
        morphology_removals: (obj as Record<string, unknown>).morphology_removals ?? null,
        contour_metrics: {
          eccentricity: (obj as Record<string, unknown>).feature_eccentricity ?? null,
          footprint_roundness: (obj as Record<string, unknown>).feature_footprint_roundness ?? null,
          roughness: (obj as Record<string, unknown>).feature_edge_roughness ?? null,
        },
        height_stats: obj.height_above_belt_mm ?? null,
        sphere_fit_stats: {
          fit_rmse_mm: obj.fit_rmse_mm ?? null,
          sphericity_score: obj.sphericity_score ?? null,
          feature_sphericity_3d: (obj as Record<string, unknown>).feature_sphericity_3d ?? null,
        },
      }));
      return <pre className="json-block">{JSON.stringify(rows, null, 2)}</pre>;
    }
    if (semantic.category === "plane_qa" && (props.view.id === "diagnostics" || props.view.id === "residual_histogram")) {
      const stageArtifacts = artifactsForStage(props.detail, props.stageId);
      const fit = (stageArtifacts.find((item) => item.artifact_id === "plane_fit_debug")?.metadata ?? {}) as Record<string, unknown>;
      const selected = (stageArtifacts.find((item) => item.artifact_id === "selected_surface_debug")?.metadata ?? {}) as Record<string, unknown>;
      const residualHistogram = (stageArtifacts.find((item) => item.artifact_id === "plane_residual_histogram")?.metadata ?? {}) as Record<string, unknown>;
      return <pre className="json-block">{JSON.stringify({ plane_fit_debug: fit, selected_surface_debug: selected, residual_histogram: residualHistogram }, null, 2)}</pre>;
    }
    if (semantic.category === "classification" && props.view.id === "diagnostics") {
      const stageArtifacts = artifactsForStage(props.detail, props.stageId);
      const result = (stageArtifacts.find((item) => item.artifact_id === "classification_result_25d")?.metadata ?? {}) as Record<string, unknown>;
      const explanations = (props.detail.result?.objects ?? []).map((obj) => ({
        object_id: obj.object_id,
        label: (obj as Record<string, unknown>).label ?? null,
        confidence: obj.confidence,
        explanation: (obj as Record<string, unknown>).classification_explanation ?? null,
      }));
      return <pre className="json-block">{JSON.stringify({ summary: result, explanations }, null, 2)}</pre>;
    }
    if (semantic.category === "classification" && props.view.id === "rule_explanation") {
      const stageArtifacts = artifactsForStage(props.detail, props.stageId);
      const explanationArtifact = resolveClassificationExplanationArtifact(stageArtifacts)
        ?? (
          props.detail.result?.files?.classification_explanation
            ? ({
              artifact_id: "classification_explanation_file_fallback",
              stage_id: "classification",
              kind: "json",
              title: "Classification rule explanation (files fallback)",
              path: props.detail.result.files.classification_explanation,
              preview_available: true,
            } as StudioArtifact)
            : null
        );
      if (!explanationArtifact) {
        const visible = stageArtifacts
          .filter((item) => item.stage_id === "classification" || item.stage_id === "overlay")
          .map((item) => ({ artifact_id: item.artifact_id, kind: item.kind, path: item.path ?? null }));
        return (
          <div className="empty-state">
            <strong>No classification explanation artifact found. Reprocess this take.</strong>
            <small>Available classification artifacts: {visible.length ? JSON.stringify(visible) : "none"}</small>
          </div>
        );
      }
      const explanations = (() => {
        const fromArtifact = explainRowsFromArtifact(explanationArtifact);
        if (fromArtifact.length) return fromArtifact;
        return explainRowsFromResult(props.detail);
      })();
      if (!explanations.length) {
        const visible = stageArtifacts
          .filter((item) => item.stage_id === "classification" || item.stage_id === "overlay")
          .map((item) => ({ artifact_id: item.artifact_id, kind: item.kind, path: item.path ?? null }));
        return (
          <div className="empty-state">
            <strong>No classification explanation artifact found. Reprocess this take.</strong>
            <small>Available classification artifacts: {visible.length ? JSON.stringify(visible) : "none"}</small>
          </div>
        );
      }
      const chosen = explanations.find((item) => Number(item.object_id ?? -1) === props.selectedObjectId) ?? explanations[0];
      const rules = (Array.isArray(chosen.rules) ? chosen.rules : [])
        .filter((rule): rule is Record<string, unknown> => typeof rule === "object" && rule !== null)
        .sort((a, b) => {
          const score = (row: Record<string, unknown>) => {
            const sev = String(row.severity ?? "info");
            const pass = Boolean(row.passed);
            if (!pass && sev === "critical") return 0;
            if (!pass && sev === "warning") return 1;
            if (pass) return 3;
            return 2;
          };
          return score(a) - score(b);
        });
      return (
        <div className="studio-table-pane">
          {explanations.length > 1 ? (
            <div className="overlay-controls">
              <label>
                Object
                <select value={Number(chosen.object_id ?? 0)} onChange={(event) => props.onSelectObject(Number(event.target.value))}>
                  {explanations.map((row) => (
                    <option key={`rule_exp_object_${String(row.object_id ?? "")}`} value={Number(row.object_id ?? 0)}>
                      object_{String(row.object_id ?? "-")}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          ) : null}
          <div className="pipeline-status-grid">
            <div><span>Final class</span><strong>{String(chosen.final_class_name ?? "-")}</strong></div>
            <div><span>Class label</span><strong>{String(chosen.final_class_label ?? "-")}</strong></div>
            <div><span>Superclass</span><strong>{String(chosen.superclass ?? "-")}</strong></div>
            <div><span>Subclass</span><strong>{String(chosen.subclass ?? "-")}</strong></div>
            <div><span>Confidence</span><strong>{chosen.confidence == null ? "-" : `${Math.round(Number(chosen.confidence) * 100)}%`}</strong></div>
          </div>
          <div className="artifact-reference">
            <span>Decision path</span>
            <small>{String(((chosen.decision_summary as Record<string, unknown> | undefined)?.final_decision_path) ?? "-")}</small>
            <small>Debug: classification_explanation.json found: yes</small>
          </div>
          {chosen.semantic_group_summaries && typeof chosen.semantic_group_summaries === "object" ? (
            <div className="artifact-reference">
              <span>Semantic groups</span>
              {Object.entries(chosen.semantic_group_summaries as Record<string, unknown>).map(([key, value]) => {
                const row = (value && typeof value === "object") ? value as Record<string, unknown> : {};
                return (
                  <small key={`group_summary_${key}`}>
                    {key}: {Boolean(row.pass) ? "pass" : "fail"} | {String(row.primary_metric ?? "-")}={String(row.value ?? "-")}
                  </small>
                );
              })}
            </div>
          ) : null}
          <table className="compact-candidate-table">
            <thead>
              <tr>
                <th>Rule</th><th>Value</th><th>Expected/threshold</th><th>Pass/fail</th><th>Effect</th><th>Message</th><th>Metric trace</th>
              </tr>
            </thead>
            <tbody>
              {rules.map((rule, idx) => {
                const expected = (rule.expected_range ?? rule.threshold) as Record<string, unknown> | undefined;
                const expectedLabel = expected ? Object.entries(expected).map(([k, v]) => `${k}:${String(v)}`).join(" ") : "-";
                return (
                  <tr key={`${String(rule.rule_id ?? "rule")}_${idx}`}>
                    <td>{String(rule.label ?? rule.rule_id ?? "-")}</td>
                    <td>{String(rule.value ?? "-")}</td>
                    <td>{expectedLabel}</td>
                    <td>{Boolean(rule.passed) ? "pass" : "fail"}</td>
                    <td>{String(rule.contribution ?? "-")} ({String(rule.severity ?? "-")})</td>
                    <td>{String(rule.message ?? "-")}</td>
                    <td>{String(((rule.metric_trace_ref as Record<string, unknown> | undefined)?.trace_id) ?? "-")}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      );
    }
    if (semantic.category === "classification" && props.view.id === "metric_details") {
      const stageArtifacts = artifactsForStage(props.detail, props.stageId);
      const metricArtifact = resolveMetricExplanationArtifact(stageArtifacts)
        ?? (
          props.detail.result?.files?.metric_explanation
            ? ({
              artifact_id: "metric_explanation_file_fallback",
              stage_id: "classification",
              kind: "json",
              title: "Metric explanation (files fallback)",
              path: props.detail.result.files.metric_explanation,
              preview_available: true,
            } as StudioArtifact)
            : null
        );
      if (!metricArtifact) {
        const visible = stageArtifacts.filter((item) => item.stage_id === "classification" || item.stage_id === "overlay").map((item) => ({ artifact_id: item.artifact_id, kind: item.kind, path: item.path ?? null }));
        return (
          <div className="empty-state">
            <strong>No metric explanation artifact found. Reprocess this take.</strong>
            <small>Available classification artifacts: {visible.length ? JSON.stringify(visible) : "none"}</small>
          </div>
        );
      }
      const objects = Array.isArray(((metricArtifact.metadata ?? {}) as Record<string, unknown>).objects)
        ? ((((metricArtifact.metadata ?? {}) as Record<string, unknown>).objects as unknown[]).filter((item): item is Record<string, unknown> => typeof item === "object" && item !== null))
        : [];
      if (!objects.length) return semanticEmpty(props.stageId, props.view.id);
      return (
        <MetricDetailsPanel
          objects={objects}
          selectedObjectId={props.selectedObjectId}
          onSelectObject={props.onSelectObject}
        />
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
      const runArtifacts = props.resolvedViewArtifacts?.length ? props.resolvedViewArtifacts : artifactsForStage(props.detail, props.stageId);
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
    if (props.resolvedViewArtifacts?.length && props.stageId !== "detect_belt_plane") {
      return explore(props.resolvedViewArtifacts, props, { disableRoi: true });
    }
    const semantic = stageSemanticDefinition(props.stageId);
    const stageArtifacts = artifactsForStage(props.detail, props.stageId);
    const context = resolveStageViewContext(props.detail, props.stageId, props.view.id, semantic.category, stageArtifacts);
    const runArtifacts = props.resolvedViewArtifacts?.length ? props.resolvedViewArtifacts : context.runArtifacts;
    const payload = semantic.category === "input" || semantic.category === "plane_qa"
      ? { take: { take_id: props.detail.take_id, modalities: props.detail.modalities, metadata: props.detail.metadata, assets: props.detail.assets, frameset: props.detail.frameset }, stage: props.stageId, semantic, resolvedArtifacts: props.resolvedViewArtifacts?.length ? props.resolvedViewArtifacts : context.resolvedArtifacts }
      : semantic.category === "segmentation"
        ? {
          stage: props.stageId,
          semantic,
          threshold_mask: runArtifacts.find((item) => item.artifact_id === "normalized_height_threshold_mask" || item.artifact_id === "threshold_mask" || item.artifact_id === "above_threshold_mask") ?? null,
          cleaned_mask: runArtifacts.find((item) => item.artifact_id === "cleaned_object_mask" || item.artifact_id === "cleaned_mask" || item.artifact_id === "final_object_mask") ?? null,
          connected_components_overlay: runArtifacts.find((item) => item.artifact_id === "connected_components_overlay") ?? null,
          overlay_image: runArtifacts.find((item) => item.artifact_id === "overlay_image") ?? null,
          segmentation_debug: runArtifacts.find((item) => item.artifact_id === "segmentation_debug")?.metadata ?? null,
        }
        : semantic.category === "geometry"
          ? {
            stage: props.stageId,
            semantic,
            artifacts: runArtifacts,
            ellipse_metrics: firstArtifactByAlias(runArtifacts, ["ellipse_metrics", "geometry_metrics", "fitted_ellipse"])?.metadata ?? null,
            ellipse_summary: firstArtifactByAlias(runArtifacts, ["ellipse_summary", "geometry_summary", "geometry_debug_summary"])?.metadata ?? null,
            blob_metrics: firstArtifactByAlias(runArtifacts, ["blob_metrics", "detection_metrics", "connected_components"])?.metadata ?? null,
            blob_contours: firstArtifactByAlias(runArtifacts, ["blob_contours", "contours", "contour"])?.metadata ?? null,
            blob_rejected: firstArtifactByAlias(runArtifacts, ["blob_rejected"])?.metadata ?? null,
          }
          : { stage: props.stageId, semantic, artifacts: runArtifacts, result: props.detail.result };
    if (semantic.category === "plane_qa" && props.view.id === "surface_candidates") {
      const surfaceCandidates = context.runArtifacts.find((item) => item.artifact_id === "reference_surface_candidates");
      const plateaus = context.runArtifacts.find((item) => item.artifact_id === "reference_surface_plateaus");
      const flatHistogram = context.runArtifacts.find((item) => item.artifact_id === "flat_candidate_histogram");
      const gradientDebug = context.runArtifacts.find((item) => item.artifact_id === "gradient_debug");
      const selectedSurface = context.runArtifacts.find((item) => item.artifact_id === "selected_surface_debug");
      const jsonDump = <pre className="json-block">{JSON.stringify({ surface_candidates: surfaceCandidates?.metadata ?? null, plateaus: plateaus?.metadata ?? null, flat_candidate_histogram: flatHistogram?.metadata ?? null, gradient_debug: gradientDebug?.metadata ?? null, selected_surface_debug: selectedSurface?.metadata ?? null }, null, 2)}</pre>;
      if (!props.detail?.take_id || !surfaceCandidates) return jsonDump;
      return (
        <LowGradientComponentsViewer
          takeId={props.detail.take_id}
          stageArtifacts={context.runArtifacts}
          artifact={surfaceCandidates}
          selectedCandidateId={props.selectedObjectId}
          hoveredCandidateId={props.hoveredObjectId}
          onSelectCandidate={props.onSelectObject}
          onHoverCandidate={props.onHoverObject}
          fallback={jsonDump}
        />
      );
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
    const artifacts = (props.resolvedViewArtifacts?.length ? props.resolvedViewArtifacts : resolveStageViewContext(props.detail, props.stageId, props.view.id, semantic.category, artifactsForStage(props.detail, props.stageId)).resolvedArtifacts).filter((artifact) => artifact.kind === "image");
    return explore(artifacts, props);
  },
  mask: (props) => <StageMaskArtifactRenderer {...props} />,
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

function toFiniteMetric(value: unknown): number | null {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function fmtMetric(value: unknown, decimals: number, suffix = ""): string {
  const n = toFiniteMetric(value);
  return n == null ? "-" : `${n.toFixed(decimals)}${suffix}`;
}

function explainRowsFromResult(detail: TakeDetail | null): Array<Record<string, unknown>> {
  const objects = [...(detail?.result?.objects ?? []), ...(detail?.result?.rejected_objects ?? [])];
  return objects
    .map((item) => (item as Record<string, unknown>).classification_explanation)
    .filter((item): item is Record<string, unknown> => typeof item === "object" && item !== null);
}

function resolveClassificationExplanationArtifact(artifacts: StudioArtifact[]): StudioArtifact | null {
  const bySemantic = artifacts.find((item) => {
    const meta = (item.metadata ?? {}) as Record<string, unknown>;
    const semanticType = String(meta.semantic_type ?? meta.artifact_type ?? meta.semantic_kind ?? "").toLowerCase();
    return semanticType === "classification_explanation";
  });
  if (bySemantic) return bySemantic;
  const byId = firstArtifactByAlias(artifacts, [
    "classification_explanation",
    "classification_explanation_json",
    "classification_explanation_metadata",
  ]);
  if (byId) return byId;
  return artifacts.find((item) => String(item.path ?? "").toLowerCase().endsWith("classification_explanation.json")) ?? null;
}

function explainRowsFromArtifact(artifact: StudioArtifact | null): Array<Record<string, unknown>> {
  if (!artifact) return [];
  const meta = (artifact.metadata ?? {}) as Record<string, unknown>;
  const rows = meta.objects;
  if (!Array.isArray(rows)) return [];
  return rows.filter((item): item is Record<string, unknown> => typeof item === "object" && item !== null);
}

function resolveMetricExplanationArtifact(artifacts: StudioArtifact[]): StudioArtifact | null {
  const bySemantic = artifacts.find((item) => {
    const meta = (item.metadata ?? {}) as Record<string, unknown>;
    const semanticType = String(meta.semantic_type ?? meta.artifact_type ?? meta.semantic_kind ?? "").toLowerCase();
    return semanticType === "metric_explanation";
  });
  if (bySemantic) return bySemantic;
  const byId = firstArtifactByAlias(artifacts, ["metric_explanation", "metric_explanation_json", "object_geometry_explanation"]);
  if (byId) return byId;
  return artifacts.find((item) => String(item.path ?? "").toLowerCase().endsWith("metric_explanation.json")) ?? null;
}

function heightMetricStats(value: unknown): { min: number | null; mean: number | null; max: number | null; p95: number | null; median: number | null } {
  const record = value && typeof value === "object" ? value as Record<string, unknown> : {};
  return {
    min: toFiniteMetric(record.min ?? record.min_height_mm),
    mean: toFiniteMetric(record.mean ?? record.mean_height_mm),
    max: toFiniteMetric(record.max ?? record.max_height_mm),
    p95: toFiniteMetric(record.p95 ?? record.p95_height_mm),
    median: toFiniteMetric(record.median ?? record.median_height_mm),
  };
}

function measurementThumbnailSource(detail: TakeDetail, artifacts: StudioArtifact[]): string | null {
  const files = detail.result?.files;
  const preferredPath = files?.input_preview
    ?? artifacts.find((artifact) => artifact.artifact_id === "heightmap_preview" && artifact.path)?.path
    ?? artifacts.find((artifact) => artifact.artifact_id === "input_preview" && artifact.path)?.path
    ?? artifacts.find((artifact) => artifact.artifact_id === "source_heightmap_preview" && artifact.path)?.path
    ?? artifacts.find((artifact) => artifact.artifact_id === "raw_heightmap_preview" && artifact.path)?.path
    ?? artifacts.find((artifact) => artifact.artifact_id === "normalized_heightmap" && artifact.path)?.path
    ?? artifacts.find((artifact) => artifact.kind === "image" && artifact.path)?.path
    ?? null;
  return preferredPath ? fileUrl(detail.take_id, preferredPath) : null;
}

function bboxFromObject(object: Record<string, unknown>): [number, number, number, number] | null {
  const direct = object.bbox_px;
  const provenance = object.measurement_provenance && typeof object.measurement_provenance === "object"
    ? (object.measurement_provenance as Record<string, unknown>).segmented_bbox_px
    : null;
  const candidate = Array.isArray(direct) ? direct : Array.isArray(provenance) ? provenance : null;
  if (!candidate || candidate.length < 4) return null;
  const values = candidate.slice(0, 4).map((item) => Number(item));
  if (!values.every((item) => Number.isFinite(item))) return null;
  const [x, y, w, h] = values;
  if (w <= 0 || h <= 0) return null;
  return [x, y, w, h];
}

function MeasurementResultsTable({
  objects,
  selectedObjectId,
  hoveredObjectId,
  thumbnailSrc,
  onSelectObject,
  onHoverObject,
}: {
  objects: NonNullable<TakeDetail["result"]>["objects"];
  selectedObjectId: number | null;
  hoveredObjectId: number | null;
  thumbnailSrc: string | null;
  onSelectObject?: (objectId: number) => void;
  onHoverObject?: (objectId: number | null) => void;
}) {
  if (!objects.length) return <div className="empty-state">No objects reported.</div>;
  return (
    <div className="measurement-results-table-wrap">
      <table className="measurement-results-table">
        <thead>
          <tr>
            <th>Object</th>
            <th>Crop</th>
            <th>Class</th>
            <th>Max H</th>
            <th>P95 H</th>
            <th>Mean H</th>
            <th>Dim X</th>
            <th>Dim Y</th>
            <th>Dim Z</th>
            <th>Volume</th>
            <th>Footprint</th>
            <th>Diameter</th>
            <th>Confidence</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {objects.map((object) => {
            const record = object as typeof object & Record<string, unknown>;
            const height = heightMetricStats(record.height_above_belt_mm);
            const dims = object.dimensions_mm ?? null;
            const bbox = bboxFromObject(record);
            const status = object.filter_status ?? "kept";
            return (
              <tr
                key={`measurement_result_${object.object_id}`}
                className={object.object_id === selectedObjectId ? "selected-object-row" : object.object_id === hoveredObjectId ? "hovered-object-row" : ""}
                onClick={() => onSelectObject?.(object.object_id)}
                onMouseEnter={() => onHoverObject?.(object.object_id)}
                onMouseLeave={() => onHoverObject?.(null)}
              >
                <td><strong>{objectDisplayId(object.object_id)}</strong></td>
                <td><ObjectCropThumbnail src={thumbnailSrc} bbox={bbox} label={objectDisplayId(object.object_id)} /></td>
                <td>
                  <strong>{object.class_name ?? "-"}</strong>
                  <small>{object.subclass_label ?? "-"}</small>
                </td>
                <td>{fmtMetric(height.max, 1, " mm")}</td>
                <td>{fmtMetric(height.p95, 1, " mm")}</td>
                <td>{fmtMetric(height.mean, 1, " mm")}</td>
                <td>{fmtMetric(Array.isArray(dims) ? dims[0] : null, 2, " mm")}</td>
                <td>{fmtMetric(Array.isArray(dims) ? dims[1] : null, 2, " mm")}</td>
                <td>{fmtMetric(Array.isArray(dims) ? dims[2] : null, 2, " mm")}</td>
                <td>{fmtMetric(record.feature_volume_proxy_mm3, 0, " mm3")}</td>
                <td>{fmtMetric(record.footprint_area_mm2, 0, " mm2")}</td>
                <td>{fmtMetric(object.diameter_mm ?? object.diameter_estimate_mm, 2, " mm")}</td>
                <td>{object.confidence == null ? "-" : `${Math.round(object.confidence * 100)}%`}</td>
                <td>{status}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function ObjectCropThumbnail({ src, bbox, label }: { src: string | null; bbox: [number, number, number, number] | null; label: string }) {
  const [naturalSize, setNaturalSize] = useState<{ width: number; height: number } | null>(null);
  if (!src || !bbox) return <div className="object-crop-thumb empty">No crop</div>;
  const [x, y, width, height] = bbox;
  const thumbWidth = 84;
  const thumbHeight = 64;
  const scale = Math.min(thumbWidth / Math.max(1, width), thumbHeight / Math.max(1, height));
  const imageStyle = naturalSize ? {
    width: `${naturalSize.width * scale}px`,
    height: `${naturalSize.height * scale}px`,
    left: `${-x * scale}px`,
    top: `${-y * scale}px`,
  } : undefined;
  return (
    <div className="object-crop-thumb" title={`${label}: ${Math.round(x)},${Math.round(y)} ${Math.round(width)}x${Math.round(height)} px`}>
      <img
        alt={`${label} crop`}
        src={src}
        style={imageStyle}
        onLoad={(event) => setNaturalSize({ width: event.currentTarget.naturalWidth || 1, height: event.currentTarget.naturalHeight || 1 })}
      />
    </div>
  );
}

type ClassificationOverlayFlags = "ids" | "labels" | "confidence" | "contours" | "geometry";

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
