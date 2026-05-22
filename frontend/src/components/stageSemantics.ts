import type { StageExecutionReport, TakeDetail } from "../api/client";
import type { StudioArtifact } from "./studioWorkspaceModel";
export function canonicalStageId(stageId: string, stageLabel?: string | null): string {
  const raw = `${stageId ?? ""} ${stageLabel ?? ""}`.toLowerCase();
  if (raw.includes("normalize_heights_to_plane")) return "normalize_heights_to_plane";
  if (raw.includes("detect_belt_plane")) return "detect_belt_plane";
  if (raw.includes("remove_belt_segment_objects")) return "remove_belt_segment_objects";
  if (raw.includes("loadheightmapcapture") || raw.includes("heightmap capture")) return "load_heightmap";
  if (raw.includes("detectbeltplane") || raw.includes("estimatebeltplane") || raw.includes("belt plane")) return "detect_belt_plane";
  if (raw.includes("normalizeheightstoplane") || raw.includes("normalizeheightrelativetoplane") || raw.includes("normalized height")) return "normalize_heights_to_plane";
  if (raw.includes("removebeltandsegmentobjects") || raw.includes("heightsegmentation")) return "remove_belt_segment_objects";
  if (raw.includes("extractconnectedcomponents")) return "connected_components";
  if (raw.includes("fitobjectgeometry") || raw.includes("footprint geometry") || raw === "geometry") return "fit_object_geometry";
  if (raw.includes("computeheightmetrics") || raw.includes("height + volume") || raw === "measurement") return "compute_height_metrics";
  if (raw.includes("classifyminingball25d")) return "classify_25d";
  if (raw.includes("generate25doverlays")) return "overlays_25d";
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

export type StageSemanticCategory = "input" | "plane_qa" | "segmentation" | "geometry" | "measurement" | "classification" | "fusion";

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
  if (id.includes("detect_belt_plane") || id.includes("normalize_heights_to_plane")) return "plane_qa";
  if (id.includes("remove_belt_segment_objects") || id.includes("height_segmentation") || id.includes("connected_components")) return "segmentation";
  if (id.includes("fit_object_geometry")) return "geometry";
  if (id.includes("compute_height_metrics")) return "measurement";
  if (id.includes("classify_25d") || id.includes("overlays_25d")) return "classification";
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
  plane_qa: [
    { id: "image", label: "Image", rendererType: "image", priority: 1, emptyState: { title: "No 25D QA image available yet." } },
    { id: "histogram", label: "Histogram", rendererType: "histogram", priority: 2, emptyState: { title: "No normalized-height histogram available yet." } },
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
    { id: "measurements", label: "Objects", rendererType: "table", priority: 1 },
    { id: "fit_metrics", label: "Fit metrics", rendererType: "metrics", priority: 2, emptyState: { title: "Ellipse fitting has not been executed for this take." } },
    { id: "diameter_histogram", label: "Diameter histogram", rendererType: "histogram", priority: 3 },
    { id: "json", label: "JSON", rendererType: "json", priority: 99 },
  ],
  classification: [
    { id: "overlay", label: "Overlay", rendererType: "overlay", priority: 1, emptyState: { title: "No classification overlay available yet." } },
    { id: "classification", label: "Classification", rendererType: "table", priority: 2, emptyState: { title: "No classification summary available yet." } },
    { id: "summary", label: "Summary", rendererType: "metrics", priority: 3 },
    { id: "rejected", label: "Rejected objects", rendererType: "table", priority: 4 },
    { id: "export", label: "Export", rendererType: "table", priority: 5 },
    { id: "json", label: "Result JSON", rendererType: "json", priority: 99 },
  ],
  fusion: [
    { id: "overview", label: "Overview", rendererType: "metrics", priority: 1, emptyState: { title: "Fusion outputs are not available yet." } },
    { id: "json", label: "JSON", rendererType: "json", priority: 99 },
  ],
};

export function stageSemanticDefinition(stageId: string): StageSemanticDefinition {
  const canonical = canonicalStageId(stageId || "");
  if (canonical === "load_heightmap") {
    return {
      stageId: canonical,
      category: "input",
      defaultViewId: "height_preview",
      views: [
        { id: "height_preview", label: "Height preview", rendererType: "image", priority: 1, emptyState: { title: "No height preview available yet." } },
        { id: "heightmap", label: "Heightmap (raw)", rendererType: "image", priority: 2, emptyState: { title: "No heightmap preview available yet." } },
        { id: "json", label: "JSON", rendererType: "json", priority: 99 },
      ],
    };
  }
  if (canonical === "normalize_heights_to_plane" || canonical === "normalized_height") {
    return {
      stageId: canonical,
      category: "plane_qa",
      defaultViewId: "normalized_height",
      views: [
        { id: "raw_heightmap", label: "Raw heightmap", rendererType: "image", priority: 1, emptyState: { title: "No raw heightmap preview available yet." } },
        { id: "reference_surface", label: "Reference surface model", rendererType: "json", priority: 2, emptyState: { title: "No reference-surface model available yet." } },
        { id: "normalized_height", label: "Normalized height", rendererType: "image", priority: 1, emptyState: { title: "No normalized height preview available yet." } },
        { id: "below_reference", label: "Below/equal reference", rendererType: "image", priority: 4, emptyState: { title: "No below-reference mask available yet." } },
        { id: "above_threshold", label: "Above threshold", rendererType: "image", priority: 5, emptyState: { title: "No above-threshold mask available yet." } },
        { id: "histogram", label: "Histogram", rendererType: "histogram", priority: 6, emptyState: { title: "No normalized-height histogram available yet." } },
        { id: "json", label: "JSON", rendererType: "json", priority: 99 },
      ],
    };
  }
  if (canonical === "detect_belt_plane" || canonical === "belt_plane") {
    return {
      stageId: canonical,
      category: "plane_qa",
      defaultViewId: "selected_surface",
      views: [
        { id: "raw_heightmap", label: "Raw heightmap", rendererType: "image", priority: 1, emptyState: { title: "No raw heightmap preview available yet." } },
        { id: "valid_mask", label: "Valid mask", rendererType: "image", priority: 2, emptyState: { title: "No valid mask available yet." } },
        { id: "depth_gradient", label: "Depth gradient", rendererType: "image", priority: 3, emptyState: { title: "No depth gradient preview available yet." } },
        { id: "low_gradient_mask", label: "Low-gradient mask", rendererType: "image", priority: 4, emptyState: { title: "No low-gradient mask available yet." } },
        { id: "surface_candidates", label: "Surface candidates", rendererType: "json", priority: 5, emptyState: { title: "No reference-surface candidates available yet." } },
        { id: "selected_surface", label: "Selected surface", rendererType: "image", priority: 6, emptyState: { title: "No selected surface mask available yet." } },
        { id: "plane_inliers", label: "Plane inliers", rendererType: "image", priority: 7, emptyState: { title: "No plane inlier mask available yet." } },
        { id: "residual_heatmap", label: "Residual heatmap", rendererType: "image", priority: 8, emptyState: { title: "No residual heatmap available yet." } },
        { id: "depth_plot", label: "Depth plot", rendererType: "image", priority: 9, emptyState: { title: "No depth plot available yet." } },
        { id: "json", label: "JSON", rendererType: "json", priority: 99 },
      ],
    };
  }
  if (canonical === "remove_belt_segment_objects" || canonical === "height_segmentation" || canonical === "connected_components") {
    return {
      stageId: canonical,
      category: "segmentation",
      defaultViewId: "overlay",
      views: [
        { id: "threshold_mask", label: "Threshold mask", rendererType: "image", priority: 1, emptyState: { title: "No threshold mask generated yet." } },
        { id: "cleaned_mask", label: "Cleaned mask", rendererType: "image", priority: 2, emptyState: { title: "No cleaned object mask generated yet." } },
        { id: "segmentation", label: "Connected components", rendererType: "table", priority: 3, emptyState: { title: "No connected components found yet." } },
        { id: "overlay", label: "Overlay", rendererType: "overlay", priority: 4, emptyState: { title: "No segmentation overlay available yet." } },
        { id: "json", label: "JSON", rendererType: "json", priority: 99 },
      ],
    };
  }
  if (canonical === "fit_object_geometry" || canonical === "compute_height_metrics") {
    return {
      stageId: canonical,
      category: "measurement",
      defaultViewId: "measurements_25d",
      views: [
        { id: "measurements_25d", label: "Measurements", rendererType: "table", priority: 1, emptyState: { title: "No 25D measurements available yet." } },
        { id: "json", label: "JSON", rendererType: "json", priority: 99 },
      ],
    };
  }
  if (canonical === "classify_25d") {
    return {
      stageId: canonical,
      category: "classification",
      defaultViewId: "classification_25d",
      views: [
        { id: "classification_25d", label: "Classification", rendererType: "table", priority: 1, emptyState: { title: "No 25D classifications available yet." } },
        { id: "json", label: "JSON", rendererType: "json", priority: 99 },
      ],
    };
  }
  if (canonical === "overlays_25d") {
    return {
      stageId: canonical,
      category: "classification",
      defaultViewId: "overlays_25d",
      views: [
        { id: "overlays_25d", label: "Overlays", rendererType: "overlay", priority: 1, emptyState: { title: "No 25D overlays available yet." } },
        { id: "json", label: "JSON", rendererType: "json", priority: 99 },
      ],
    };
  }
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
  const canonical = canonicalStageId(stageId || "");
  if (canonical === "detect_belt_plane" || canonical === "belt_plane") {
    const debug = artifacts.find((item) => item.artifact_id === "plane_fit_debug");
    const meta = (debug?.metadata ?? {}) as Record<string, unknown>;
    const status = String(meta.status ?? "unknown");
    const p95 = Number(meta.residual_p95_mm ?? 0);
    const sel = (meta.background_selection as Record<string, unknown> | undefined) ?? {};
    const gdbg = (sel.gradient_debug as Record<string, unknown> | undefined) ?? {};
    const area = Number(gdbg.selected_component_area_ratio ?? sel.expanded_plane_coverage_percent ?? 0) * (Number(gdbg.selected_component_area_ratio ?? NaN) <= 1 ? 100 : 1);
    const model = String(meta.reference_surface_model_type ?? "plane");
    if (model === "constant_z") {
      const zstd = Number(gdbg.selected_component_z_std_mm ?? 0);
      return `model ${model} • surface ${area.toFixed(1)}% • z std ${zstd.toFixed(1)} mm`;
    }
    return `model ${model} • surface ${area.toFixed(1)}% • residual p95 ${p95.toFixed(1)} mm`;
  }
  if (canonical === "normalize_heights_to_plane" || canonical === "normalized_height") {
    const debug = artifacts.find((item) => item.artifact_id === "normalization_debug" || item.artifact_id === "normalized_height_debug");
    const hist = artifacts.find((item) => item.artifact_id === "normalized_height_histogram");
    const meta = (debug?.metadata ?? {}) as Record<string, unknown>;
    const bg = Number(meta.background_height_p95_abs_after_normalization_mm ?? 0);
    const above = Number(meta.above_threshold_pixel_count ?? 0);
    const total = Number(((hist?.metadata ?? {}) as Record<string, unknown>).sample_count ?? 0);
    const pct = total > 0 ? (above / total) * 100.0 : 0.0;
    return `background p95 abs ${bg.toFixed(1)} mm • above threshold ${pct.toFixed(1)}%`;
  }
  if (canonical === "remove_belt_segment_objects") {
    const debug = artifacts.find((item) => item.artifact_id === "segmentation_debug");
    const meta = (debug?.metadata ?? {}) as Record<string, unknown>;
    const cc = Number(meta.connected_component_count ?? 0);
    const fg = Number(meta.foreground_coverage_percent ?? 0);
    return `${cc} component${cc === 1 ? "" : "s"} • foreground ${fg.toFixed(1)}%`;
  }
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
