import type { ActiveCalibration, ProcessingResult } from "../api/client";

export function calibrationStateLabel(active?: ActiveCalibration | null, result?: ProcessingResult | null): string {
  const type = active?.calibration_type ?? result?.calibration?.calibration_type ?? "plane_3d";
  if (active?.status === "incompatible") return `Incompatible ${type}`;
  if (active?.status === "warning") return `Warning ${type}`;
  if (active?.status === "missing") return `Missing ${type}`;
  if (active?.active || result?.calibration_active) return `Active ${type}`;
  return `No ${type}`;
}
