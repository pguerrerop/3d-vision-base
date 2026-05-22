import type { PipelineInfo, ProcessingResult, TakeDetail } from "../api/client";

export type StudioWorkspaceTab = "inputs" | "segmentation" | "classification" | "measurements" | "fusion" | "artifacts" | "json";
export type StudioObject = ProcessingResult["objects"][number];
export type StudioArtifact = NonNullable<ProcessingResult["artifacts"]>[number];
export type StudioExecutionStage = NonNullable<NonNullable<ProcessingResult["pipeline_execution"]>["stages"]>[number];
export type BlobCandidate = {
  blob_id: number;
  area_px?: number;
  centroid?: [number, number] | number[] | null;
  bbox?: [number, number, number, number] | number[] | null;
  contour_perimeter?: number;
  equivalent_diameter?: number;
  circularity?: number;
  aspect_ratio?: number;
  solidity?: number;
  touches_border?: boolean;
  status?: string;
  rejection_reason?: string;
};

export function objectsForTake(detail: TakeDetail | null): StudioObject[] {
  return [...(detail?.result?.objects ?? []), ...(detail?.result?.rejected_objects ?? [])];
}

export function artifactList(detail: TakeDetail | null): StudioArtifact[] {
  return detail?.result?.artifacts ?? [];
}

export function artifactsForStage(detail: TakeDetail | null, stageId: string): StudioArtifact[] {
  const artifacts = artifactList(detail);
  const canonical = canonicalStageIdLocal(stageId);
  const aliases = new Set<string>([stageId, canonical]);
  if (canonical === "segmentation") {
    aliases.add("threshold");
    aliases.add("morphology");
  } else if (canonical === "detection") {
    aliases.add("blob_detection");
    aliases.add("blob");
    aliases.add("contour_detection");
    aliases.add("blob_contour_detection");
  } else if (canonical === "ellipse_fitting") {
    aliases.add("ellipse_fitting");
    aliases.add("ellipse");
    aliases.add("geometry_metrics");
    aliases.add("measurement");
  } else if (canonical === "measurement") {
    aliases.add("ellipse_fitting");
    aliases.add("metrics");
  } else if (canonical === "classification") {
    aliases.add("classification");
    aliases.add("overlay");
    aliases.add("summary");
  } else if (canonical === "input") {
    aliases.add("crop_roi");
    aliases.add("rgb_to_gray");
    aliases.add("normalize_lighting");
  } else if (canonical === "detect_belt_plane" || canonical === "normalize_heights_to_plane") {
    aliases.add("calibration");
    aliases.add("input");
  } else if (canonical === "remove_belt_segment_objects") {
    aliases.add("segmentation");
    aliases.add("height_segmentation");
  }
  const staged = artifacts.filter((artifact) => aliases.has(artifact.stage_id));
  const targetIds = new Set(
    staged
      .filter((artifact) => artifact.kind === "overlay")
      .map((artifact) => artifact.target_artifact_id)
      .filter((value): value is string => typeof value === "string" && value.length > 0)
  );
  const references = artifacts.filter((artifact) => targetIds.has(artifact.artifact_id));
  return [...staged, ...references.filter((reference) => !staged.some((artifact) => artifact.artifact_id === reference.artifact_id))];
}

function canonicalStageIdLocal(stageId: string): string {
  const raw = `${stageId ?? ""}`.toLowerCase();
  if (raw.includes("fitobjectgeometry") || raw.includes("footprint") || raw === "geometry") return "fit_object_geometry";
  if (raw.includes("computeheightmetrics") || raw === "measurement") return "compute_height_metrics";
  if (raw.includes("detectbeltplane") || raw.includes("estimatebeltplane") || raw.includes("belt_plane")) return "detect_belt_plane";
  if (raw.includes("normalizeheightstoplane") || raw.includes("normalizeheightrelativetoplane") || raw.includes("normalized_height")) return "normalize_heights_to_plane";
  if (raw.includes("removebeltandsegmentobjects") || raw.includes("height_segmentation")) return "remove_belt_segment_objects";
  if (
    raw.includes("blob_detection")
    || raw.includes("blob/contour")
    || raw.includes("blob_contour")
    || raw.includes("contour_detection")
    || raw.includes("blob detection")
    || raw.includes("contour detection")
    || raw === "detection"
    || raw === "blob"
  ) return "detection";
  if (raw.includes("threshold") || raw.includes("morph")) return "segmentation";
  if (raw.includes("ellipse_fitting") || raw.includes("ellipse") || raw.includes("geometry_metrics")) return "ellipse_fitting";
  if (raw.includes("measure") || raw.includes("metric")) return "measurement";
  if (raw.includes("class")) return "classification";
  if (raw.includes("input") || raw.includes("normalize") || raw.includes("gray") || raw.includes("roi")) return "input";
  return raw;
}

export function artifactsForObject(artifacts: StudioArtifact[], objectId: number | null): StudioArtifact[] {
  if (objectId == null) return artifacts;
  return artifacts.filter((artifact) => artifact.object_id == null || artifact.object_id === objectId);
}

export function stageInspectorSummary(detail: TakeDetail | null, pipeline: PipelineInfo | null, stageId: string) {
  const result = detail?.result;
  const artifacts = artifactsForStage(detail, stageId);
  const objects = objectsForTake(detail);
  const execution = executionStage(detail, stageId);
  return {
    stageId,
    status: execution?.status ?? "skipped",
    timingMs: execution?.duration_ms ?? stageTimingMsLocal(result, stageId),
    warnings: execution?.warnings ?? result?.poc_summary?.warnings ?? [],
    errors: execution?.errors ?? [],
    inputArtifactCount: execution?.input_artifact_ids?.length ?? 0,
    outputArtifactCount: execution?.output_artifact_ids?.length ?? artifacts.length,
    artifactCount: artifacts.length,
    objectCount: execution?.object_count ?? objects.length,
    rejectedCount: execution?.rejected_count ?? objects.filter((object) => object.filter_status === "rejected").length,
    requiredModalities: pipeline?.required_modalities ?? result?.processing_pipeline?.required_modalities ?? [],
  };
}

function stageTimingMsLocal(result: ProcessingResult | null | undefined, stageId: string): number | null {
  const stages = result?.profiling?.stages ?? [];
  const matches = stages.filter((stage) => stage.category === stageId || stage.name.toLowerCase().includes(stageId));
  if (!matches.length) return null;
  return matches.reduce((total, stage) => total + stage.duration_ms, 0);
}

export function selectedObject(objects: StudioObject[], objectId: number | null): StudioObject | null {
  if (objectId == null) return objects[0] ?? null;
  return objects.find((object) => object.object_id === objectId) ?? objects[0] ?? null;
}

export function executionStage(detail: TakeDetail | null, stageId: string): StudioExecutionStage | null {
  return detail?.result?.pipeline_execution?.stages?.find((stage) => stage.stage_id === stageId) ?? null;
}

export function overlayArtifacts(artifacts: StudioArtifact[], targetArtifactId: string | null | undefined): StudioArtifact[] {
  if (!targetArtifactId) return [];
  return artifacts.filter((artifact) => artifact.kind === "overlay" && artifact.target_artifact_id === targetArtifactId);
}

export function objectArtifacts(artifacts: StudioArtifact[], objectId: number | null): StudioArtifact[] {
  if (objectId == null) return [];
  return artifacts.filter((artifact) => artifact.object_id === objectId);
}

export function artifactLineage(artifacts: StudioArtifact[], artifact: StudioArtifact | null): string[] {
  if (!artifact) return [];
  const sourceIds = artifact.source_artifact_ids ?? [];
  const target = artifact.target_artifact_id ? [artifact.target_artifact_id] : [];
  const fromMetadata = Array.isArray(artifact.metadata?.derived_from) ? artifact.metadata.derived_from.map(String) : [];
  const unique = new Set([...sourceIds, ...target, ...fromMetadata]);
  const labels: string[] = [];
  for (const id of unique) {
    const found = artifacts.find((candidate) => candidate.artifact_id === id);
    labels.push(found ? `${found.title} (${found.artifact_id})` : id);
  }
  return labels;
}

export function objectGeneratingStages(artifacts: StudioArtifact[], objectId: number | null): string[] {
  const byStage = new Set(
    artifacts
      .filter((artifact) => objectId != null && artifact.object_id === objectId)
      .map((artifact) => artifact.stage_id)
  );
  return [...byStage];
}

export function stageTitleForTab(tab: StudioWorkspaceTab): string {
  if (tab === "inputs") return "Source data";
  if (tab === "segmentation") return "Segmentation outputs";
  if (tab === "classification") return "Classification results";
  if (tab === "measurements") return "Measurements and statistics";
  if (tab === "fusion") return "Fusion workspace";
  if (tab === "artifacts") return "Artifact explorer";
  return "Result payload";
}

export function firstArtifactByAlias(artifacts: StudioArtifact[], aliases: string[]): StudioArtifact | null {
  for (const alias of aliases) {
    const found = artifacts.find((item) => {
      if (item.artifact_id === alias) return true;
      const semanticKind = String((item.metadata as Record<string, unknown> | undefined)?.semantic_kind ?? "");
      return semanticKind === alias;
    });
    if (found) return found;
  }
  return null;
}

export function blobCandidatesFromArtifacts(artifacts: StudioArtifact[]): BlobCandidate[] {
  const metrics = firstArtifactByAlias(artifacts, ["blob_metrics", "detection_metrics"]);
  const entries = ((metrics?.metadata as Record<string, unknown> | undefined)?.entries ?? null) as unknown;
  if (Array.isArray(entries)) {
    return entries
      .filter((item): item is Record<string, unknown> => typeof item === "object" && item !== null)
      .map((item) => ({
        blob_id: Number(item.blob_id ?? item.object_id ?? 0),
        area_px: num(item.area_px ?? item.area),
        centroid: (Array.isArray(item.centroid) ? item.centroid as number[] : null),
        bbox: (Array.isArray(item.bbox) ? item.bbox as number[] : null),
        contour_perimeter: num(item.contour_perimeter ?? item.perimeter),
        equivalent_diameter: num(item.equivalent_diameter ?? item.diameter),
        circularity: num(item.circularity),
        aspect_ratio: num(item.aspect_ratio),
        solidity: num(item.solidity),
        touches_border: Boolean(item.touches_border),
        status: typeof item.status === "string" ? item.status : "kept",
        rejection_reason: typeof item.rejection_reason === "string" ? item.rejection_reason : undefined,
      }))
      .filter((item) => item.blob_id > 0);
  }
  return [];
}

export function blobRejectedCountFromArtifacts(artifacts: StudioArtifact[]): number {
  const metrics = firstArtifactByAlias(artifacts, ["blob_metrics", "detection_metrics"]);
  const rejected = Number((metrics?.metadata as Record<string, unknown> | undefined)?.rejected_count ?? 0);
  return Number.isFinite(rejected) ? rejected : 0;
}

function num(value: unknown): number | undefined {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}
