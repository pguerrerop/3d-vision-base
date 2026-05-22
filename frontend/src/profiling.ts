import type { ProcessingProfiling, ProcessingResult, ProfilingCategory, ProfilingStage } from "./api/client";

export type TimelineSegment = ProfilingStage & {
  percentOfTotal: number;
  widthPercent: number;
  isSlowest: boolean;
  isDominant: boolean;
};

export type ProfilingBadge = {
  label: string;
  tone: "neutral" | "good" | "warning";
};

const CATEGORY_LABELS: Record<ProfilingCategory, string> = {
  input: "Input",
  io: "IO",
  calibration: "Calibration",
  preprocessing: "Preprocessing",
  calibration_filtering: "Calibration",
  segmentation: "Segmentation",
  clustering: "Clustering",
  geometry: "Geometry",
  measurement: "Measurement",
  classification: "Classification",
  overlay: "Overlay",
  debug_artifacts: "Debug artifacts",
  output: "Output",
};

export function profilingCategoryLabel(category: string) {
  return CATEGORY_LABELS[category as ProfilingCategory] ?? category.replace(/_/g, " ");
}

export function formatMs(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) {
    return "-";
  }
  return `${value.toFixed(value >= 100 ? 0 : 1)} ms`;
}

export function formatCount(value: number | null | undefined) {
  return value == null ? "-" : value.toLocaleString();
}

export function getSlowestStage(profiling: ProcessingProfiling | null | undefined) {
  if (!profiling?.stages.length) {
    return null;
  }
  return profiling.stages.reduce((slowest, stage) => (stage.duration_ms > slowest.duration_ms ? stage : slowest), profiling.stages[0]);
}

export function buildTimelineSegments(profiling: ProcessingProfiling | null | undefined): TimelineSegment[] {
  if (!profiling?.stages.length) {
    return [];
  }
  const total = profiling.total_ms > 0 ? profiling.total_ms : profiling.stages.reduce((sum, stage) => sum + stage.duration_ms, 0);
  const maxDuration = Math.max(...profiling.stages.map((stage) => stage.duration_ms), 0);
  const slowest = getSlowestStage(profiling);

  return profiling.stages.map((stage) => {
    const percentOfTotal = total > 0 ? (stage.duration_ms / total) * 100 : 0;
    return {
      ...stage,
      percentOfTotal,
      widthPercent: maxDuration > 0 ? Math.max((stage.duration_ms / maxDuration) * 100, 2) : 0,
      isSlowest: slowest?.name === stage.name && slowest.duration_ms === stage.duration_ms,
      isDominant: percentOfTotal > 25,
    };
  });
}

export function sortedProfilingStages(profiling: ProcessingProfiling | null | undefined): TimelineSegment[] {
  return [...buildTimelineSegments(profiling)].sort((a, b) => b.duration_ms - a.duration_ms);
}

export function buildOptimizationHints(profiling: ProcessingProfiling | null | undefined): string[] {
  if (!profiling) {
    return [];
  }
  const stages = profiling.stages;
  const byName = new Map(stages.map((stage) => [stage.name, stage]));
  const hints: string[] = [];
  const loadPointCloud = byName.get("load_point_cloud");
  const fastLoad = byName.get("load_point_cloud_fast");
  const removeOutliers = byName.get("remove_outliers");
  const clustering = byName.get("dbscan_clustering") ?? stages.find((stage) => stage.category === "clustering");

  if (!fastLoad && loadPointCloud && loadPointCloud.duration_ms > 1000) {
    hints.push("PLY loading dominates runtime. Consider NPZ runtime format.");
  }
  if (profiling.debug_artifacts_ms > profiling.production_ms) {
    hints.push("Debug rendering dominates latency.");
  }
  if (removeOutliers && clustering && removeOutliers.duration_ms > clustering.duration_ms) {
    hints.push("Outlier removal is expensive for current point count.");
  }

  return hints;
}

export function buildProfilingBadges(result: ProcessingResult | null | undefined): ProfilingBadge[] {
  if (!result) {
    return [];
  }
  const stages = result.profiling?.stages ?? [];
  const hasSkippedDebug = stages.some((stage) => stage.notes === "skipped by --skip-debug-images");
  const hasFastCloud = stages.some((stage) => stage.name === "load_point_cloud_fast");
  const calibrationActive = Boolean(result.calibration_id);

  return [
    result.algorithm_stage === "calibrated_segmentation" || calibrationActive
      ? { label: "calibrated", tone: "good" }
      : { label: "generic", tone: "neutral" },
    ...(hasSkippedDebug ? [{ label: "debug-images skipped", tone: "good" } as const] : []),
    ...(hasFastCloud ? [{ label: "fast cloud used", tone: "good" } as const] : []),
    calibrationActive ? { label: "calibration active", tone: "good" } : { label: "calibration inactive", tone: "neutral" },
  ];
}
