import type { TakeDetail } from "../api/client";
import type { StudioArtifact } from "./studioWorkspaceModel";

export type SelectedSurfaceRefinementSummary = {
  selectedClusterId?: string;
  selectedClusterPixels?: number;
  selectedSurfacePixels?: number;
  keptRatio?: number;
  selectedSurfaceValidRoiRatio?: number;
  madEnabled?: boolean;
  madK?: number;
  madFloor?: number;
  keepBorderSupport?: boolean;
  minPixels?: number;
  fallbackStrategy?: string;
  fallbackUsed?: boolean;
  reason?: string;
};

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" ? value as Record<string, unknown> : {};
}

function asNumber(value: unknown): number | undefined {
  const num = Number(value);
  return Number.isFinite(num) ? num : undefined;
}

function asString(value: unknown): string | undefined {
  return value == null ? undefined : String(value);
}

function asBoolean(value: unknown): boolean | undefined {
  return typeof value === "boolean" ? value : undefined;
}

function firstDefined<T>(...values: Array<T | undefined>): T | undefined {
  return values.find((value) => value !== undefined);
}

function findArtifactMetadata(artifacts: StudioArtifact[], artifactId: string): Record<string, unknown> {
  return asRecord(artifacts.find((artifact) => artifact.artifact_id === artifactId)?.metadata);
}

export function deriveSelectedSurfaceRefinementSummary(
  artifacts: StudioArtifact[],
  detail?: TakeDetail | null,
): SelectedSurfaceRefinementSummary {
  const planeDebug = findArtifactMetadata(artifacts, "plane_fit_debug");
  const selectedSurfaceDebug = findArtifactMetadata(artifacts, "selected_surface_debug");
  const selectedSupportLineage = findArtifactMetadata(artifacts, "selected_support_lineage");
  const lineageSummary = asRecord(selectedSupportLineage.summary);
  const selectedSurfaceMask = findArtifactMetadata(artifacts, "selected_reference_support_mask");
  const referenceSurfaceMask = findArtifactMetadata(artifacts, "reference_surface_selected_mask");
  const blobDebug = findArtifactMetadata(artifacts, "blob_cluster_selection_debug");

  const backgroundSelection = asRecord(planeDebug.background_selection);
  const gradientDebug = asRecord(backgroundSelection.gradient_debug);
  const referenceMethod = asRecord(
    planeDebug.reference_method
    ?? backgroundSelection.reference_method
    ?? selectedSurfaceDebug.reference_method,
  );
  const detectParams = asRecord(asRecord(asRecord(detail?.result).stage_params).detect_belt_plane);

  const selectedClusterPixels = firstDefined(
    asNumber(selectedSurfaceDebug.selected_cluster_pixels),
    asNumber(lineageSummary.selected_cluster_pixels),
    asNumber(selectedSurfaceDebug.selected_blob_cluster_pixels),
    asNumber(gradientDebug.selected_cluster_pixels),
    asNumber(gradientDebug.selected_blob_cluster_pixels),
    asNumber(blobDebug.selected_cluster_pixels),
    asNumber(blobDebug.selected_blob_cluster_pixels),
    asNumber(selectedSurfaceMask.selected_cluster_pixels),
    asNumber(referenceSurfaceMask.selected_cluster_pixels),
  );
  const selectedSurfacePixels = firstDefined(
    asNumber(selectedSurfaceDebug.selected_surface_pixels),
    asNumber(selectedSurfaceDebug.selected_support_pixels),
    asNumber(selectedSurfaceDebug.final_selected_surface_pixels),
    asNumber(selectedSurfaceDebug.selected_reference_support_pixels),
    asNumber(planeDebug.selected_surface_pixels),
    asNumber(selectedSurfaceMask.selected_surface_pixels),
    asNumber(referenceSurfaceMask.selected_surface_pixels),
  );
  const keptRatio = firstDefined(
    selectedClusterPixels && selectedSurfacePixels && selectedClusterPixels > 0 ? selectedSurfacePixels / selectedClusterPixels : undefined,
    asNumber(selectedSurfaceDebug.selected_surface_kept_ratio),
    asNumber(selectedSurfaceDebug.kept_ratio),
    asNumber(planeDebug.selected_surface_kept_ratio),
  );

  return {
    selectedClusterId: firstDefined(
      asString(selectedSurfaceDebug.selected_cluster_id),
      asString(selectedSupportLineage.selected_cluster_id),
      asString(gradientDebug.selected_cluster_id),
      asString(blobDebug.selected_cluster_id),
    ),
    selectedClusterPixels,
    selectedSurfacePixels,
    keptRatio,
    selectedSurfaceValidRoiRatio: firstDefined(
      asNumber(selectedSurfaceDebug.selected_surface_valid_roi_ratio),
      asNumber(lineageSummary.selected_reference_support_valid_roi_ratio),
      asNumber(selectedSurfaceDebug.selected_surface_area_ratio),
      asNumber(planeDebug.selected_surface_valid_roi_ratio),
    ),
    madEnabled: firstDefined(
      asBoolean(selectedSurfaceDebug.refine_by_mad),
      asBoolean(referenceMethod.blob_cluster_refine_by_mad),
      asBoolean(detectParams.blob_cluster_refine_by_mad),
    ),
    madK: firstDefined(
      asNumber(selectedSurfaceDebug.mad_k),
      asNumber(referenceMethod.blob_cluster_refine_mad_k),
      asNumber(detectParams.blob_cluster_refine_mad_k),
    ),
    madFloor: firstDefined(
      asNumber(selectedSurfaceDebug.mad_floor_mm),
      asNumber(referenceMethod.blob_cluster_refine_floor_mm),
      asNumber(detectParams.blob_cluster_refine_floor_mm),
    ),
    keepBorderSupport: firstDefined(
      asBoolean(selectedSurfaceDebug.keep_border_support),
      asBoolean(referenceMethod.blob_cluster_refine_keep_border_support),
      asBoolean(detectParams.blob_cluster_refine_keep_border_support),
    ),
    minPixels: firstDefined(
      asNumber(selectedSurfaceDebug.min_total_pixels),
      asNumber(referenceMethod.blob_cluster_min_total_pixels),
      asNumber(detectParams.blob_cluster_min_total_pixels),
    ),
    fallbackStrategy: firstDefined(
      asString(selectedSurfaceDebug.fallback_strategy),
      asString(referenceMethod.fallback_strategy),
      asString(gradientDebug.fallback_strategy),
      asString(blobDebug.fallback_strategy),
      asString(detectParams.blob_cluster_fallback_strategy),
    ),
    fallbackUsed: firstDefined(
      asBoolean(selectedSurfaceDebug.fallback_used),
      asBoolean(referenceMethod.fallback_used),
      asBoolean(gradientDebug.fallback_used),
      asBoolean(blobDebug.fallback_used),
    ),
    reason: firstDefined(
      asString(selectedSurfaceDebug.fallback_reason),
      asString(referenceMethod.fallback_reason),
      asString(gradientDebug.fallback_reason),
      asString(blobDebug.fallback_reason),
      asString(selectedSurfaceDebug.model_selection_reason),
      asString(planeDebug.model_selection_reason),
    ),
  };
}
