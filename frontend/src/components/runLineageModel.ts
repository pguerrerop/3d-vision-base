import type { PipelineRunEntry, PipelineRunLineage } from "../api/client";

export function formatParentLineageLine(lineage: PipelineRunLineage): string | null {
  if (!lineage.parent) return null;
  const parent = lineage.parent;
  const parts = [`Parent: ${parent.run_id}`];
  if (parent.boundary_stage_id) parts.push(`Boundary: ${parent.boundary_stage_id}`);
  return parts.join(" — ") || null;
}

export function formatBoundaryLine(entry: PipelineRunEntry | null): string | null {
  if (!entry?.boundary_stage_id) return null;
  return `Boundary: ${entry.boundary_stage_id}`;
}

export function formatReusedRerunLine(entry: PipelineRunEntry | null): string | null {
  const summary = entry?.summary;
  if (!summary) return null;
  const reused = summary.reused_units ?? 0;
  const rerun = summary.rerun_units ?? 0;
  const invalidated = summary.invalidated_units ?? 0;
  if (!reused && !rerun && !invalidated) return null;
  return `Reused: ${reused} units — Rerun: ${rerun} units — Invalidated: ${invalidated} units`;
}

export function formatChildRunsSummary(children: PipelineRunEntry[]): string {
  if (!children.length) return "No child partial runs.";
  return `Child partial runs (${children.length})`;
}
