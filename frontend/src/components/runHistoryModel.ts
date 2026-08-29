import type { PipelineRunEntry } from "../api/client";

export function sortRunsNewestFirst(runs: PipelineRunEntry[]): PipelineRunEntry[] {
  return [...runs].sort((a, b) => String(b.created_at ?? "").localeCompare(String(a.created_at ?? "")));
}

export function runTypeLabel(runType: PipelineRunEntry["run_type"] | string | null | undefined): string {
  switch (runType) {
    case "partial_rerun":
      return "Partial rerun";
    case "legacy_3d_result":
      return "Legacy run";
    case "full_run":
    default:
      return "Full run";
  }
}

export function formatRunTimestamp(createdAt: string | null | undefined): string {
  if (!createdAt) return "Unknown time";
  const parsed = new Date(createdAt);
  if (Number.isNaN(parsed.getTime())) return createdAt;
  return parsed.toLocaleString();
}

export function runRecipeLabel(run: PipelineRunEntry): string | null {
  if (!run.recipe_id) return null;
  return run.recipe_version_id ? `${run.recipe_id} (${run.recipe_version_id})` : run.recipe_id;
}
