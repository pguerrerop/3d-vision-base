import type { CaptureModality, PipelineInfo, ProcessingResult, TakeDetail } from "../api/client";

export function modalityCompatible(pipeline: PipelineInfo, modalities: CaptureModality[] | string[] | undefined): boolean {
  const available = new Set(modalities ?? []);
  return pipeline.required_modalities.every((modality) => available.has(modality));
}

export function incompatibleModalityMessage(pipeline: PipelineInfo, modalities: CaptureModality[] | string[] | undefined): string | null {
  if (modalityCompatible(pipeline, modalities)) return null;
  const available = modalities?.length ? modalities.join(", ") : "none";
  const required = pipeline.required_modalities.join(", ") || "none";
  return `Selected take has ${available}. ${pipeline.display_name} requires ${required}.`;
}

export function activePipelineName(result: ProcessingResult | null | undefined): string {
  return result?.processing_pipeline?.display_name ?? result?.processing_pipeline?.name ?? "No active pipeline";
}

export function stageTimingMs(result: ProcessingResult | null | undefined, stageId: string): number | null {
  const stages = result?.profiling?.stages ?? [];
  const matches = stages.filter((stage) => stage.category === stageId || stage.name.toLowerCase().includes(stageId));
  if (!matches.length) return null;
  return matches.reduce((total, stage) => total + stage.duration_ms, 0);
}

export function sourceSummary(detail: TakeDetail | null | undefined): string {
  const source = detail?.metadata?.source;
  if (typeof source === "string") return source;
  if (typeof source === "object" && source) {
    const typed = source as Record<string, unknown>;
    return String(typed.type ?? "source");
  }
  return "-";
}

export type StudioWorkspaceTab = "inputs" | "segmentation" | "classification" | "measurements" | "fusion" | "artifacts" | "json";
export type StageViewTabId =
  | "overview"
  | "image"
  | "mask"
  | "overlay"
  | "metrics"
  | "artifacts"
  | "json"
  | "metadata"
  | "histogram"
  | "threshold_mask"
  | "cleaned_mask"
  | "params"
  | "contour_overlay"
  | "candidate_overlay"
  | "candidates"
  | "candidates_table"
  | "labels"
  | "blob_metrics"
  | "ellipse_overlay"
  | "ellipse_metrics"
  | "rejected"
  | "geometry"
  | "fit_metrics"
  | "diameter_histogram"
  | "measurements"
  | "classification"
  | "summary"
  | "export"
  | "heightmap"
  | "raw_heightmap"
  | "valid_mask"
  | "plane_inliers"
  | "background_candidates"
  | "background_seeds"
  | "expanded_plane"
  | "depth_plot"
  | "residual_heatmap"
  | "normalized_height"
  | "belt_plane"
  | "segmentation"
  | "measurements_25d"
  | "classification_25d"
  | "overlays_25d";

export function tabForStage(stageId: string): StudioWorkspaceTab {
  if (stageId.includes("segment")) return "segmentation";
  if (stageId.includes("class")) return "classification";
  if (stageId.includes("measure") || stageId.includes("stat")) return "measurements";
  if (stageId.includes("fusion") || stageId.includes("registration")) return "fusion";
  return "artifacts";
}

export function studioEmptyState(tab: StudioWorkspaceTab, compatible: boolean, processed: boolean): string {
  if (!compatible) return "This pipeline is incompatible with the selected take modalities.";
  if (tab === "fusion") return "Fusion workflows are not implemented yet. Future versions will support RGB/3D synchronization and multimodal processing.";
  if (!processed) return "This take is compatible with the selected pipeline. Run processing to generate stage outputs.";
  return "No outputs generated for this stage yet.";
}

export function pipelineCompatibilityGroups(
  pipelines: PipelineInfo[],
  modalities: CaptureModality[] | string[] | undefined
): { compatible: PipelineInfo[]; incompatible: Array<{ pipeline: PipelineInfo; reason: string }> } {
  const compatible: PipelineInfo[] = [];
  const incompatible: Array<{ pipeline: PipelineInfo; reason: string }> = [];
  for (const pipeline of pipelines) {
    if (modalityCompatible(pipeline, modalities)) compatible.push(pipeline);
    else incompatible.push({ pipeline, reason: incompatibleModalityMessage(pipeline, modalities) ?? "Incompatible modalities" });
  }
  return { compatible, incompatible };
}

export function chooseBestPipelineForTake(
  pipelines: PipelineInfo[],
  modalities: CaptureModality[] | string[] | undefined,
  lastUsedPipelineId: string | null | undefined
): PipelineInfo | null {
  if (!pipelines.length) return null;
  const groups = pipelineCompatibilityGroups(pipelines, modalities);
  if (lastUsedPipelineId) {
    const remembered = groups.compatible.find((item) => item.id === lastUsedPipelineId);
    if (remembered) return remembered;
  }
  const preferred = groups.compatible.find((item) =>
    item.id === "mining_steel_ball_classification_25d"
    || item.id === "3d_ball_inspection"
    || item.id === "mining_steel_ball_classification_2d"
  );
  if (preferred) return preferred;
  if (groups.compatible.length) return groups.compatible[0];
  return pipelines[0];
}

export function compactTakeFamilyStatus(
  processingByFamily: Array<{ family: "3d" | "2d" | "25d" | "generic"; hasCompletedOutput: boolean; status: string }> | undefined,
  modalities: CaptureModality[] | string[] | undefined
): string {
  const items = processingByFamily ?? [];
  const available = new Set(modalities ?? []);
  const status2d = items.find((item) => item.family === "2d");
  const status3d = items.find((item) => item.family === "3d");
  const status25d = items.find((item) => item.family === "25d");
  const tags: string[] = [];
  if (status2d) {
    tags.push(`2D: ${status2d.hasCompletedOutput ? "done" : status2d.status === "never_processed" ? "not run" : status2d.status}`);
  }
  if (status3d) {
    const unavailable3d = !available.has("point_cloud");
    tags.push(`3D: ${unavailable3d ? "unavailable" : status3d.hasCompletedOutput ? "done" : status3d.status === "never_processed" ? "not run" : status3d.status}`);
  }
  if (status25d) {
    const unavailable25d = !available.has("heightmap");
    tags.push(`25D: ${unavailable25d ? "unavailable" : status25d.hasCompletedOutput ? "done" : status25d.status === "never_processed" ? "not run" : status25d.status}`);
  }
  return tags.join(" • ");
}

export function stageScopedViewTabs(
  artifacts: Array<{ kind?: string; title?: string; path?: string | null }>
): Array<{ id: StageViewTabId; label: string }> {
  const tabs: Array<{ id: StageViewTabId; label: string }> = [{ id: "overview", label: "Overview" }];
  const hasMask = artifacts.some((artifact) => artifact.kind === "image" && /mask|morph|binary/i.test(`${artifact.title ?? ""} ${artifact.path ?? ""}`));
  const hasImage = artifacts.some((artifact) => artifact.kind === "image" && !/mask|morph|binary/i.test(`${artifact.title ?? ""} ${artifact.path ?? ""}`));
  const hasOverlay = artifacts.some((artifact) => artifact.kind === "overlay");
  const hasMetrics = artifacts.some((artifact) => artifact.kind === "table" || artifact.kind === "metric");
  if (hasImage) tabs.push({ id: "image", label: "Image" });
  if (hasMask) tabs.push({ id: "mask", label: "Mask" });
  if (hasOverlay) tabs.push({ id: "overlay", label: "Overlay" });
  if (hasMetrics) tabs.push({ id: "metrics", label: "Metrics" });
  tabs.push({ id: "artifacts", label: "Artifacts" });
  tabs.push({ id: "json", label: "JSON" });
  return tabs;
}

export function resolveStageViewSelection(
  current: StageViewTabId,
  available: Array<{ id: StageViewTabId; label: string }>
): { next: StageViewTabId; fallbackMessage: string | null } {
  if (available.some((tab) => tab.id === current)) return { next: current, fallbackMessage: null };
  const fallback = available[1]?.id ?? available[0]?.id ?? "overview";
  const fallbackLabel = available.find((tab) => tab.id === fallback)?.label ?? "Overview";
  return { next: fallback, fallbackMessage: `Selected stage changed. Showing default view: ${fallbackLabel}.` };
}
