import type { PartialExecutionSummary } from "../api/client";

export function formatBytes(bytes: number): string {
  if (bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

export function formatPartialExecutionSummaryLine(summary: PartialExecutionSummary | null | undefined): string | null {
  if (!summary) return null;
  const parts = [
    `Reused: ${summary.reused_artifact_count} artifacts (${formatBytes(summary.reused_artifact_bytes)})`,
    `Rerun: ${summary.rerun_artifact_count} artifacts`,
    `Duration: ${summary.partial_execution_duration_ms} ms`,
  ];
  return parts.join(" — ");
}
