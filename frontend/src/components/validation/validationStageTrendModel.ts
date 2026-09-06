import type {
  ValidationStageTrendPoint,
  ValidationStageTrendSeries,
  ValidationStageTrendStageEntry,
} from "../../api/client";

/** A short label for an execution column: the date if known, else the id's tail. */
export function executionLabel(ref: { execution_id: string; created_at?: string | null }): string {
  if (ref.created_at) {
    const parsed = new Date(ref.created_at);
    if (!Number.isNaN(parsed.getTime())) {
      return parsed.toISOString().slice(0, 10);
    }
  }
  return ref.execution_id.slice(-6);
}

/** The fraction of that point's comparisons that regressed, or null for a gap. */
export function pointRegressionRate(point: ValidationStageTrendPoint): number | null {
  if (!point || !point.count) return null;
  return (point.status_counts.regression ?? 0) / point.count;
}

/** Every metric field this series reports in at least one execution, alphabetically. */
export function seriesFields(series: ValidationStageTrendSeries): string[] {
  const fields = new Set<string>();
  for (const point of series.points) {
    if (!point) continue;
    for (const field of Object.keys(point.metrics)) fields.add(field);
  }
  return [...fields].sort((a, b) => a.localeCompare(b));
}

/** One field's mean across the series, aligned to the same points (null = gap). */
export function fieldMeanTrend(
  series: ValidationStageTrendSeries,
  field: string,
): Array<number | null> {
  return series.points.map((point) => point?.metrics[field]?.mean ?? null);
}

/** Whether this series has data in at least one execution of the window. */
export function seriesHasAnyData(series: ValidationStageTrendSeries): boolean {
  return series.points.some((point) => point !== null);
}

/** Stages with at least one comparator series carrying data anywhere in the window. */
export function stagesWithTrendData(
  stages: ValidationStageTrendStageEntry[],
): ValidationStageTrendStageEntry[] {
  return stages.filter((stage) => stage.comparators.some(seriesHasAnyData));
}
