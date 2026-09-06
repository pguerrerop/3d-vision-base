import type {
  ValidationStageComparatorSummary,
  ValidationStageSummaryEntry,
} from "../../api/client";

/**
 * Mirrors the priority order matrix() already uses server-side to pick one
 * status for a cell. Duplicated rather than imported so this file, which the
 * node test runner loads directly, needs no value import of validationStatus.
 */
const STATUS_PRIORITY = [
  "regression",
  "not_comparable",
  "needs_review",
  "changed",
  "blocked",
  "pass",
] as const;

/** status_counts entries in a fixed, severity-first order, not JSON key order. */
export function sortedStatusEntries(counts: Record<string, number>): Array<[string, number]> {
  const known = STATUS_PRIORITY.filter((status) => counts[status]).map(
    (status): [string, number] => [status, counts[status]],
  );
  const rest = Object.entries(counts)
    .filter(([status]) => !STATUS_PRIORITY.includes(status as (typeof STATUS_PRIORITY)[number]))
    .sort(([a], [b]) => a.localeCompare(b));
  return [...known, ...rest];
}

/** The fraction of this bucket's comparisons that regressed, or null with no data. */
export function regressionRate(bucket: ValidationStageComparatorSummary): number | null {
  if (!bucket.count) return null;
  return (bucket.status_counts.regression ?? 0) / bucket.count;
}

/** Whether any case in the sample produced a comparison at this stage at all. */
export function stageHasData(stage: ValidationStageSummaryEntry): boolean {
  return stage.comparators.length > 0;
}

/** A bucket's numeric fields, alphabetically, so render order never depends on JSON key order. */
export function fieldRows(
  bucket: ValidationStageComparatorSummary,
): Array<{ field: string; summary: ValidationStageComparatorSummary["metrics"][string] }> {
  return Object.keys(bucket.metrics)
    .sort((a, b) => a.localeCompare(b))
    .map((field) => ({ field, summary: bucket.metrics[field] }));
}
