import type { StageExecutionReport, TakeDetail } from "../api/client";
import type { StudioArtifact } from "./studioWorkspaceModel";
export function canonicalStageId(stageId: string, stageLabel?: string | null): string {
  const raw = `${stageId ?? ""} ${stageLabel ?? ""}`.toLowerCase();
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
  if (raw.includes("ellipse_fitting") || raw.includes("ellipse") || raw.includes("geometry_metrics")) return "ellipse_fitting";
  if (raw.includes("threshold") || raw.includes("morph")) return "segmentation";
  if (raw.includes("measure") || raw.includes("metric")) return "measurement";
  if (raw.includes("class")) return "classification";
  if (raw.includes("input") || raw.includes("normalize") || raw.includes("gray") || raw.includes("roi")) return "input";
  return (stageId || "").toLowerCase();
}

export type StageViewRendererType = "image" | "overlay" | "table" | "metrics" | "json" | "gallery" | "histogram";

export type StageViewDefinition = {
  id: string;
  label: string;
  icon?: string;
  supportedArtifactKinds?: string[];
  supportedOverlayKinds?: string[];
  priority?: number;
  emptyState?: { title: string; description?: string };
  rendererType: StageViewRendererType;
};

export type StageSemanticCategory = "input" | "segmentation" | "geometry" | "measurement" | "classification" | "fusion";

export type StageSemanticDefinition = {
  stageId: string;
  category: StageSemanticCategory;
  defaultViewId: string;
  views: StageViewDefinition[];
  sourceBindings?: {
    image?: boolean;
    metadata?: boolean;
    histogram?: boolean;
    json?: boolean;
  };
};

function stageKind(stageId: string): StageSemanticCategory {
  const id = canonicalStageId(stageId).toLowerCase();
  if (id.includes("input") || id.includes("crop") || id.includes("gray") || id.includes("normalize")) return "input";
  if (id.includes("threshold") || id.includes("morph") || id.includes("segment")) return "segmentation";
  if (id.includes("blob") || id.includes("contour") || id.includes("ellipse") || id.includes("circle") || id.includes("detect")) return "geometry";
  if (id.includes("measure") || id.includes("metric")) return "measurement";
  if (id.includes("class") || id.includes("summary")) return "classification";
  return "fusion";
}

const VIEWS_BY_CATEGORY: Record<StageSemanticCategory, StageViewDefinition[]> = {
  input: [
    { id: "image", label: "Image", rendererType: "image", priority: 1, emptyState: { title: "No source image available yet." } },
    { id: "metadata", label: "Metadata", rendererType: "table", priority: 2, emptyState: { title: "No source metadata available." } },
    { id: "histogram", label: "Histogram", rendererType: "histogram", priority: 3, emptyState: { title: "No image histogram available yet." } },
    { id: "json", label: "JSON", rendererType: "json", priority: 99 },
  ],
  segmentation: [
    { id: "threshold_mask", label: "Threshold mask", rendererType: "image", priority: 1, emptyState: { title: "No threshold mask generated yet.", description: "Run the pipeline to inspect morphology results." } },
    { id: "cleaned_mask", label: "Cleaned mask", rendererType: "image", priority: 2, emptyState: { title: "No cleaned mask generated yet." } },
    { id: "overlay", label: "Overlay", rendererType: "overlay", priority: 3, emptyState: { title: "No segmentation overlays available yet." } },
    { id: "params", label: "Morphology params", rendererType: "table", priority: 4 },
    { id: "json", label: "JSON", rendererType: "json", priority: 99 },
  ],
  geometry: [
    { id: "candidate_overlay", label: "Candidate overlay", rendererType: "overlay", priority: 1, emptyState: { title: "No contour candidates detected yet." } },
    { id: "labels", label: "Labels", rendererType: "image", priority: 2, emptyState: { title: "No blob labels generated yet." } },
    { id: "blob_metrics", label: "Blob metrics", rendererType: "table", priority: 3 },
    { id: "candidates_table", label: "Candidates table", rendererType: "table", priority: 4 },
    { id: "json", label: "JSON", rendererType: "json", priority: 99 },
  ],
  measurement: [
    { id: "fit_metrics", label: "Fit metrics", rendererType: "metrics", priority: 1, emptyState: { title: "Ellipse fitting has not been executed for this take." } },
    { id: "diameter_histogram", label: "Diameter histogram", rendererType: "histogram", priority: 2 },
    { id: "measurements", label: "Measurements", rendererType: "table", priority: 3 },
    { id: "json", label: "JSON", rendererType: "json", priority: 99 },
  ],
  classification: [
    { id: "classification", label: "Classification", rendererType: "table", priority: 1, emptyState: { title: "No classification summary available yet." } },
    { id: "summary", label: "Summary", rendererType: "metrics", priority: 2 },
    { id: "rejected", label: "Rejected objects", rendererType: "table", priority: 3 },
    { id: "export", label: "Export", rendererType: "table", priority: 4 },
    { id: "json", label: "Result JSON", rendererType: "json", priority: 99 },
  ],
  fusion: [
    { id: "overview", label: "Overview", rendererType: "metrics", priority: 1, emptyState: { title: "Fusion outputs are not available yet." } },
    { id: "json", label: "JSON", rendererType: "json", priority: 99 },
  ],
};

export function stageSemanticDefinition(stageId: string): StageSemanticDefinition {
  const canonical = canonicalStageId(stageId || "");
  if (canonical === "ellipse_fitting") {
    return {
      stageId: canonical,
      category: "geometry",
      defaultViewId: "ellipse_overlay",
      views: [
        { id: "ellipse_overlay", label: "Ellipse overlay", rendererType: "overlay", priority: 1, emptyState: { title: "No ellipse fits generated yet." } },
        { id: "ellipse_metrics", label: "Metrics", rendererType: "table", priority: 2, emptyState: { title: "No ellipse metrics generated yet." } },
        { id: "candidates_table", label: "Candidates table", rendererType: "table", priority: 3, emptyState: { title: "No fitted candidates available yet." } },
        { id: "json", label: "JSON", rendererType: "json", priority: 99 },
      ],
    };
  }
  const category = stageKind(canonical);
  const views = [...VIEWS_BY_CATEGORY[category]].sort((a, b) => (a.priority ?? 100) - (b.priority ?? 100));
  const defaultViewId = category === "segmentation"
    ? (views.find((view) => view.id === "overlay")?.id ?? views[0]?.id ?? "json")
    : (views[0]?.id ?? "json");
  return {
    stageId: canonical,
    category,
    defaultViewId,
    views,
    sourceBindings: category === "input"
      ? { image: true, metadata: true, histogram: true, json: true }
      : undefined,
  };
}

export function stageSemanticSummary(stageId: string, detail: TakeDetail | null, artifacts: StudioArtifact[], execStage: StageExecutionReport | null | undefined): string {
  const category = stageKind(canonicalStageId(stageId || ""));
  const objects = detail?.result?.objects ?? [];
  const rejected = detail?.result?.rejected_objects ?? [];
  if (category === "segmentation") {
    const metricsArtifact = artifacts.find((item) => item.artifact_id === "morphology_metrics");
    const debugArtifact = artifacts.find((item) => item.artifact_id === "morphology_debug_json");
    const metricsMeta = (metricsArtifact?.metadata ?? {}) as Record<string, unknown>;
    const debugMeta = (debugArtifact?.metadata ?? {}) as Record<string, unknown>;
    const debugMorph = (debugMeta.morphology as Record<string, unknown> | undefined) ?? {};
    const debugRoi = (debugMeta.roi as Record<string, unknown> | undefined) ?? {};
    const componentCount = Number(debugMorph.components_after ?? metricsMeta.connected_components ?? 0);
    const coverageRatio = Number(debugMorph.cleaned_foreground_coverage ?? metricsMeta.cleaned_foreground_coverage ?? NaN);
    const coveragePercent = Number.isFinite(coverageRatio) ? coverageRatio * 100.0 : Number(metricsMeta.foreground_coverage_percent ?? 0);
    const roiEnabled = Boolean(debugRoi.enabled ?? metricsMeta.roi_enabled ?? false);
    const masks = artifacts.filter((a) => a.artifact_id === "threshold_mask" || a.artifact_id === "cleaned_mask" || a.kind === "image").length;
    if (masks) {
      const coverageText = Number.isFinite(coveragePercent) ? ` • Foreground: ${coveragePercent.toFixed(1)}%` : "";
      const componentText = componentCount > 0 ? ` • ${componentCount} connected regions` : "";
      const roiText = roiEnabled ? " • ROI enabled" : "";
      return `${masks} masks generated${coverageText}${componentText}${roiText}`;
    }
    return "No masks generated";
  }
  if (category === "geometry") {
    const entries = geometryEntries(artifacts);
    if (entries.length) {
      const rejected = geometryRejected(artifacts);
      const avgCircularity = entries.length
        ? (entries.reduce((acc, row) => acc + Number((row as Record<string, unknown>).circularity ?? 0), 0) / entries.length)
        : 0;
      return `${entries.length} candidates${rejected > 0 ? ` • ${rejected} rejected` : ""} • avg circularity ${avgCircularity.toFixed(2)}`;
    }
    const overlay = artifacts.filter((a) => a.kind === "overlay").length;
    return `${Math.max(objects.length, overlay)} candidates detected`;
  }
  if (category === "measurement") {
    const avgCirc = objects.length ? (objects.reduce((acc, obj) => acc + Number(obj.sphericity_score ?? 0), 0) / objects.length) : 0;
    return objects.length ? `${objects.length} ellipses fitted • avg circularity ${avgCirc.toFixed(2)}` : "No ellipse fits";
  }
  if (category === "classification") {
    return `${objects.length} classified • ${rejected.length} rejected`;
  }
  if (category === "input") {
    return artifacts.some((a) => a.kind === "image") ? "Source image ready" : "Waiting for input artifact";
  }
  return `${artifacts.length} artifacts • ${execStage?.warnings.length ?? 0} warnings`;
}

function geometryEntries(artifacts: StudioArtifact[]): Array<Record<string, unknown>> {
  const ellipse = artifacts.find((item) => item.artifact_id === "ellipse_metrics" || item.artifact_id === "geometry_metrics");
  const fromEllipse = (ellipse?.metadata as Record<string, unknown> | undefined)?.entries;
  if (Array.isArray(fromEllipse) && fromEllipse.length) {
    return fromEllipse.filter((item): item is Record<string, unknown> => typeof item === "object" && item !== null);
  }
  const metrics = artifacts.find((item) => item.artifact_id === "blob_metrics" || item.artifact_id === "detection_metrics");
  const fromMetrics = (metrics?.metadata as Record<string, unknown> | undefined)?.entries;
  if (Array.isArray(fromMetrics) && fromMetrics.length) {
    return fromMetrics.filter((item): item is Record<string, unknown> => typeof item === "object" && item !== null);
  }
  const contours = artifacts.find((item) => item.artifact_id === "blob_contours" || item.artifact_id === "contours");
  const fromContours = (contours?.metadata as Record<string, unknown> | undefined)?.entries;
  if (Array.isArray(fromContours)) {
    return fromContours
      .filter((item): item is Record<string, unknown> => typeof item === "object" && item !== null)
      .map((item) => ({
        ...item,
        circularity: Number(item.circularity ?? 0),
      }));
  }
  return [];
}

function geometryRejected(artifacts: StudioArtifact[]): number {
  const byId = artifacts.find((item) => item.artifact_id === "blob_metrics" || item.artifact_id === "detection_metrics");
  const rejected = Number((byId?.metadata as Record<string, unknown> | undefined)?.rejected_count ?? 0);
  return Number.isFinite(rejected) ? rejected : 0;
}

export function stageEmptyState(stageId: string, viewId: string): { title: string; description?: string } {
  const semantic = stageSemanticDefinition(stageId);
  const view = semantic.views.find((candidate) => candidate.id === viewId);
  return view?.emptyState ?? { title: `No ${view?.label?.toLowerCase() ?? "stage"} output generated yet.` };
}
