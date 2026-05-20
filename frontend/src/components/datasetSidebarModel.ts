import type { DatasetSessionSummary, DatasetSummary } from "../api/client";

export function nextDatasetSelectionAfterCreate(
  datasets: DatasetSummary[],
  created: DatasetSummary,
): { datasets: DatasetSummary[]; selectedDataset: string } {
  const merged = [created, ...datasets.filter((item) => item.id !== created.id)];
  return { datasets: merged, selectedDataset: created.id };
}

export function nextSessionSelectionAfterCreate(
  sessions: DatasetSessionSummary[],
  created: DatasetSessionSummary,
): { sessions: DatasetSessionSummary[]; selectedSession: string } {
  const merged = [created, ...sessions.filter((item) => item.id !== created.id)];
  return { sessions: merged, selectedSession: created.id };
}
