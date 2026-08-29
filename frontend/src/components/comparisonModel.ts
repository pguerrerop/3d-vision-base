import type { PipelineComparisonChangeSummary, PipelineComparisonIndexEntry, PipelineComparisonResponse } from "../api/client";

export function orderedComparisonUnits(comparison: PipelineComparisonResponse | null): Array<[string, PipelineComparisonResponse["units"][string]]> {
  if (!comparison) return [];
  return Object.entries(comparison.units).sort((left, right) => {
    if (left[1].has_changes !== right[1].has_changes) return left[1].has_changes ? -1 : 1;
    if (left[1].order !== right[1].order) return left[1].order - right[1].order;
    return left[0].localeCompare(right[0]);
  });
}

export function comparisonHasAnyChanges(comparison: PipelineComparisonResponse | null): boolean {
  if (!comparison) return false;
  return orderedComparisonUnits(comparison).some(([, unit]) => unit.has_changes);
}

export function comparisonSourceLabel(
  source: PipelineComparisonIndexEntry["left"] | PipelineComparisonResponse["left"] | undefined,
  context?: { parentRunId?: string | null; childRunIds?: string[] },
): string {
  if (!source) return "unknown";
  if (source.type === "run" && source.run_id) {
    if (context?.parentRunId && source.run_id === context.parentRunId) return `Parent full run (${source.run_id})`;
    if (context?.childRunIds?.includes(source.run_id)) return `Child partial rerun (${source.run_id})`;
  }
  if (source.label) return source.label;
  if (source.type === "recipe" && source.recipe_id) return `recipe ${source.recipe_id}${source.recipe_version ? ` v${source.recipe_version}` : ""}`;
  if (source.type === "run" && source.run_id) return `run ${source.run_id}`;
  return source.type;
}

export function formatChangeSummary(summary?: PipelineComparisonChangeSummary | null): string | null {
  if (!summary) return null;
  const parts: string[] = [];
  if (summary.top_changed_unit?.label) {
    const mask = summary.top_changed_unit.mask_changed_percent;
    parts.push(
      mask != null && mask > 0
        ? `${summary.top_changed_unit.label} mask ${mask.toFixed(1)}% changed`
        : `${summary.top_changed_unit.label} parameters changed`,
    );
  }
  if (summary.top_changed_artifact?.label) {
    parts.push(`artifact ${summary.top_changed_artifact.label}`);
  }
  if (summary.classification_changed) parts.push("classification changed");
  if (summary.object_count_changed) parts.push("object count changed");
  if (summary.warnings_changed) parts.push("warnings changed");
  return parts.length ? parts.join(" · ") : "No dominant changes detected.";
}

export function formatComparisonSummaryLine(comparison: PipelineComparisonResponse | null): string | null {
  return formatChangeSummary(comparison?.summary?.what_changed_most ?? null);
}
