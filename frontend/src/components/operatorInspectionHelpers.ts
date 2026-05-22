import type { PublishedInspectionResult } from "../api/client";

export function formatDecision(decision: PublishedInspectionResult["overall_decision"] | null | undefined): string {
  switch (decision) {
    case "accept":
      return "ACCEPT";
    case "reject":
      return "REJECT";
    case "review":
      return "REVIEW";
    default:
      return "UNKNOWN";
  }
}

export function formatConfidence(value: number | null | undefined): string {
  if (typeof value !== "number" || Number.isNaN(value)) return "-";
  return `${(value * 100).toFixed(1)}%`;
}

export function formatMeasurements(measurements: Record<string, unknown> | null | undefined): string {
  if (!measurements || typeof measurements !== "object") return "-";
  const entries = Object.entries(measurements);
  if (!entries.length) return "-";
  return entries
    .slice(0, 4)
    .map(([key, val]) => `${key}: ${typeof val === "number" ? val.toFixed(3).replace(/\.?0+$/, "") : String(val)}`)
    .join(" | ");
}

export function getDecisionTone(decision: PublishedInspectionResult["overall_decision"] | null | undefined): "accept" | "reject" | "review" | "unknown" {
  if (decision === "accept" || decision === "reject" || decision === "review") return decision;
  return "unknown";
}
