import type {
  ValidationClassificationObjectResult,
  ValidationClassificationSide,
  ValidationComparisonResult,
  ValidationMeasurementObjectResult,
  ValidationMetricResult,
} from "../../api/client";

/**
 * Mirrors client.ts's own validationFileUrl() and API_BASE_URL, including no
 * value import from client.ts at all. This module is loaded straight by the
 * node test runner, which cannot resolve an extensionless value import the
 * way the Vite bundler does; every cross-module reference in this codebase
 * that node has to load directly is `import type` for the same reason. Keep
 * this in step with client.ts if either changes.
 */
function fileUrl(path: string): string {
  const base =
    (import.meta as { env?: Record<string, string | undefined> }).env?.VITE_API_BASE_URL ?? "";
  const encoded = path
    .split("/")
    .map((segment) => encodeURIComponent(segment))
    .join("/");
  return `${base}/api/validation/files/${encoded}`;
}

// --------------------------------------------------------------------------
// Shared
// --------------------------------------------------------------------------

/** A labelled number for a metrics table, formatted for the unit it is in. */
export interface MetricRow {
  label: string;
  value: string;
  raw: number | undefined;
}

const fmt = (value: number | undefined, digits = 3): string =>
  value === undefined || Number.isNaN(value) ? "—" : value.toFixed(digits);

const pct = (value: number | undefined): string =>
  value === undefined || Number.isNaN(value) ? "—" : `${(value * 100).toFixed(2)}%`;

const REASON_TEXT: Record<string, string> = {
  iou_below_threshold: "IoU fell below the configured minimum.",
  dice_below_threshold: "Dice fell below the configured minimum.",
  added_fraction_above_threshold: "Added pixels exceeded the configured fraction.",
  removed_fraction_above_threshold: "Removed pixels exceeded the configured fraction.",
  component_count_changed: "The connected-component count changed.",
  mae_threshold: "Mean absolute error exceeded the configured maximum.",
  rmse_threshold: "RMSE exceeded the configured maximum.",
  p95_absolute_error_threshold: "P95 absolute error exceeded the configured maximum.",
  valid_overlap_threshold: "Valid-pixel overlap fell below the configured minimum.",
  tolerance_exceeded: "The delta exceeded its effective tolerance.",
  missing_metric: "The metric is absent from one side.",
  non_numeric_metric: "The metric is not a number on at least one side.",
  superclass_changed: "The superclass changed.",
  label_changed: "The label changed without an allowed alternative.",
  confidence_below_minimum: "Confidence fell below the configured minimum.",
  confidence_drop: "Confidence dropped more than the configured maximum.",
  new_critical_warning: "A new warning on the configured critical list appeared.",
};

/** A short, specific sentence for a reason code — never the bare code itself. */
export function describeReason(code: string): string {
  return REASON_TEXT[code] ?? code.replace(/_/g, " ");
}

/** URLs for every diff artifact the comparator persisted, keyed by filename. */
export function diffImageUrls(result: ValidationComparisonResult): Record<string, string> {
  const urls: Record<string, string> = {};
  for (const path of result.diff_artifacts) {
    const name = path.split("/").pop() ?? path;
    urls[name] = fileUrl(path);
  }
  return urls;
}

// --------------------------------------------------------------------------
// Binary mask
// --------------------------------------------------------------------------

export interface MaskImages {
  baseline?: string;
  candidate?: string;
  combined?: string;
  added?: string;
  removed?: string;
}

export function maskImages(result: ValidationComparisonResult): MaskImages {
  const urls = diffImageUrls(result);
  return {
    baseline: urls["baseline_mask.png"],
    candidate: urls["candidate_mask.png"],
    combined: urls["combined_diff.png"],
    added: urls["added_pixels.png"],
    removed: urls["removed_pixels.png"],
  };
}

export function maskMetricRows(result: ValidationComparisonResult): MetricRow[] {
  const m = result.metrics;
  return [
    { label: "IoU", value: fmt(m.iou), raw: m.iou },
    { label: "Dice", value: fmt(m.dice), raw: m.dice },
    {
      label: "Baseline foreground area",
      value: `${m.baseline_foreground_pixels ?? "—"} px`,
      raw: m.baseline_foreground_pixels,
    },
    {
      label: "Candidate foreground area",
      value: `${m.candidate_foreground_pixels ?? "—"} px`,
      raw: m.candidate_foreground_pixels,
    },
    { label: "Added fraction", value: pct(m.added_fraction), raw: m.added_fraction },
    { label: "Removed fraction", value: pct(m.removed_fraction), raw: m.removed_fraction },
    {
      label: "Baseline components",
      value: `${m.baseline_components ?? "—"}`,
      raw: m.baseline_components,
    },
    {
      label: "Candidate components",
      value: `${m.candidate_components ?? "—"}`,
      raw: m.candidate_components,
    },
  ];
}

// --------------------------------------------------------------------------
// Numeric raster
// --------------------------------------------------------------------------

export interface RasterImages {
  absoluteDifference?: string;
  validPixelChange?: string;
}

export function rasterImages(result: ValidationComparisonResult): RasterImages {
  const urls = diffImageUrls(result);
  return {
    absoluteDifference: urls["absolute_difference.png"],
    validPixelChange: urls["valid_pixel_change.png"],
  };
}

export function rasterMetricRows(result: ValidationComparisonResult): MetricRow[] {
  const m = result.metrics;
  return [
    { label: "Valid overlap", value: pct(m.valid_overlap), raw: m.valid_overlap },
    { label: "MAE", value: fmt(m.mae), raw: m.mae },
    { label: "RMSE", value: fmt(m.rmse), raw: m.rmse },
    {
      label: "Median absolute error",
      value: fmt(m.median_absolute_error),
      raw: m.median_absolute_error,
    },
    { label: "P95 absolute error", value: fmt(m.p95_absolute_error), raw: m.p95_absolute_error },
    { label: "P99 absolute error", value: fmt(m.p99_absolute_error), raw: m.p99_absolute_error },
    { label: "Max absolute error", value: fmt(m.max_absolute_error), raw: m.max_absolute_error },
    { label: "Signed mean error", value: fmt(m.signed_mean_error), raw: m.signed_mean_error },
  ];
}

// --------------------------------------------------------------------------
// Measurement table
// --------------------------------------------------------------------------

export type MeasurementFilter =
  "all" | "regressions" | "changed" | "missing" | "added" | "ambiguous" | "missing_metrics";

export const MEASUREMENT_FILTERS: Array<{ value: MeasurementFilter; label: string }> = [
  { value: "all", label: "All" },
  { value: "regressions", label: "Regressions" },
  { value: "changed", label: "Changed" },
  { value: "missing", label: "Missing objects" },
  { value: "added", label: "Added objects" },
  { value: "ambiguous", label: "Ambiguous" },
  { value: "missing_metrics", label: "Missing metrics" },
];

/** A row in the measurement or classification table: a matched pair, or an
 * object that exists on only one side.  Object-count changes stay visible
 * even when there is no metric row to anchor them to. */
export type MeasurementRow =
  | { kind: "matched"; result: ValidationMeasurementObjectResult; metric: ValidationMetricResult }
  | { kind: "unmatched-object"; side: "missing" | "added"; objectId: string | null };

export function measurementRows(result: ValidationComparisonResult): MeasurementRow[] {
  const rows: MeasurementRow[] = [];
  for (const object of result.object_results ?? []) {
    const measurement = object as ValidationMeasurementObjectResult;
    for (const metric of measurement.metric_results ?? []) {
      rows.push({ kind: "matched", result: measurement, metric });
    }
  }
  for (const entry of result.matching?.baseline_only ?? []) {
    rows.push({ kind: "unmatched-object", side: "missing", objectId: entry.object_id });
  }
  for (const entry of result.matching?.candidate_only ?? []) {
    rows.push({ kind: "unmatched-object", side: "added", objectId: entry.object_id });
  }
  return rows;
}

export function filterMeasurementRows(
  rows: MeasurementRow[],
  filter: MeasurementFilter,
): MeasurementRow[] {
  switch (filter) {
    case "all":
      return rows;
    case "regressions":
      return rows.filter((row) => row.kind === "matched" && row.metric.status === "regression");
    case "changed":
      return rows.filter((row) => row.kind === "matched" && row.metric.status === "changed");
    case "missing":
      return rows.filter((row) => row.kind === "unmatched-object" && row.side === "missing");
    case "added":
      return rows.filter((row) => row.kind === "unmatched-object" && row.side === "added");
    case "ambiguous":
      // Ambiguous matches carry no metric rows of their own; they are counted, not listed here.
      return [];
    case "missing_metrics":
      return rows.filter((row) => row.kind === "matched" && row.metric.reason === "missing_metric");
  }
}

export interface MeasurementCounts {
  baselineObjects: number;
  candidateObjects: number;
  matched: number;
  missing: number;
  added: number;
  ambiguous: number;
  regressedMetrics: number;
}

export function measurementCounts(result: ValidationComparisonResult): MeasurementCounts {
  const m = result.metrics;
  return {
    baselineObjects: m.baseline_object_count ?? 0,
    candidateObjects: m.candidate_object_count ?? 0,
    matched: m.matched_object_count ?? 0,
    missing: m.missing_object_count ?? 0,
    added: m.added_object_count ?? 0,
    ambiguous: result.matching?.ambiguous.length ?? 0,
    regressedMetrics: m.regressed_metric_count ?? 0,
  };
}

// --------------------------------------------------------------------------
// Classification table
// --------------------------------------------------------------------------

export type ClassificationFilter =
  "all" | "regressions" | "superclass" | "label" | "confidence" | "missing" | "added" | "warnings";

export const CLASSIFICATION_FILTERS: Array<{ value: ClassificationFilter; label: string }> = [
  { value: "all", label: "All" },
  { value: "regressions", label: "Regressions" },
  { value: "superclass", label: "Superclass changed" },
  { value: "label", label: "Label changed" },
  { value: "confidence", label: "Confidence only" },
  { value: "missing", label: "Missing objects" },
  { value: "added", label: "Added objects" },
  { value: "warnings", label: "New critical warnings" },
];

export type ClassificationRow =
  | { kind: "matched"; result: ValidationClassificationObjectResult }
  | { kind: "unmatched-object"; side: "missing" | "added"; objectId: string | null };

export function classificationRows(result: ValidationComparisonResult): ClassificationRow[] {
  const rows: ClassificationRow[] = [];
  for (const object of result.object_results ?? []) {
    rows.push({ kind: "matched", result: object as ValidationClassificationObjectResult });
  }
  for (const entry of result.matching?.baseline_only ?? []) {
    rows.push({ kind: "unmatched-object", side: "missing", objectId: entry.object_id });
  }
  for (const entry of result.matching?.candidate_only ?? []) {
    rows.push({ kind: "unmatched-object", side: "added", objectId: entry.object_id });
  }
  return rows;
}

/** Confidence changed but nothing else did: neither superclass nor label changed. */
function isConfidenceOnlyChange(row: ValidationClassificationObjectResult): boolean {
  return (
    (row.reasons.includes("confidence_below_minimum") || row.reasons.includes("confidence_drop")) &&
    !row.reasons.includes("superclass_changed") &&
    !row.reasons.includes("label_changed")
  );
}

export function filterClassificationRows(
  rows: ClassificationRow[],
  filter: ClassificationFilter,
): ClassificationRow[] {
  switch (filter) {
    case "all":
      return rows;
    case "regressions":
      return rows.filter((row) => row.kind === "matched" && row.result.status === "regression");
    case "superclass":
      return rows.filter(
        (row) => row.kind === "matched" && row.result.reasons.includes("superclass_changed"),
      );
    case "label":
      return rows.filter(
        (row) => row.kind === "matched" && row.result.reasons.includes("label_changed"),
      );
    case "confidence":
      return rows.filter((row) => row.kind === "matched" && isConfidenceOnlyChange(row.result));
    case "missing":
      return rows.filter((row) => row.kind === "unmatched-object" && row.side === "missing");
    case "added":
      return rows.filter((row) => row.kind === "unmatched-object" && row.side === "added");
    case "warnings":
      return rows.filter(
        (row) => row.kind === "matched" && row.result.reasons.includes("new_critical_warning"),
      );
  }
}

export interface ClassificationCounts {
  baselineObjects: number;
  candidateObjects: number;
  labelChanges: number;
  superclassChanges: number;
  newCriticalWarnings: number;
}

export function classificationCounts(result: ValidationComparisonResult): ClassificationCounts {
  const m = result.metrics;
  return {
    baselineObjects: m.baseline_object_count ?? 0,
    candidateObjects: m.candidate_object_count ?? 0,
    labelChanges: m.label_changes ?? 0,
    superclassChanges: m.superclass_changes ?? 0,
    newCriticalWarnings: m.new_critical_warnings ?? 0,
  };
}

export function classificationSideText(side: ValidationClassificationSide): string {
  const confidence =
    side.confidence === null || side.confidence === undefined
      ? "—"
      : `${(side.confidence * 100).toFixed(1)}%`;
  return `${side.superclass ?? "—"} / ${side.label ?? "—"} (${confidence})`;
}
