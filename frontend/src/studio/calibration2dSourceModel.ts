import type { CalibrationCapture, SourceInfo } from "../api/client";

export const PREVIEW_STALE_THRESHOLD_SECONDS = 5;

export type Calibration2DWorkflowState = "NO_CAPTURES" | "CAPTURES_READY" | "CORNERS_DETECTED" | "CALIBRATION_VALID";

export function sourceDisplayLabel(source: SourceInfo): string {
  const resolution = source.resolution?.length === 2 ? `${source.resolution[0]}x${source.resolution[1]}` : "-";
  const fps = typeof source.fps === "number" ? `${source.fps.toFixed(1)} fps` : "- fps";
  return `${source.label} • ${source.modality.toUpperCase()} • ${resolution} • ${fps}`;
}

export function isSourceFresh(source: SourceInfo | null | undefined): boolean {
  if (!source) return false;
  if (source.status !== "live") return false;
  if (typeof source.last_frame_age_seconds !== "number") return false;
  return source.last_frame_age_seconds <= PREVIEW_STALE_THRESHOLD_SECONDS;
}

export function isTargetConfigValid(target: {
  squares_x: number;
  squares_y: number;
  square_length_mm: number;
  marker_length_mm: number;
}): boolean {
  if (target.squares_x < 2 || target.squares_y < 2) return false;
  if (target.square_length_mm <= 0 || target.marker_length_mm <= 0) return false;
  return target.marker_length_mm < target.square_length_mm;
}

export function canDetectCorners(captures: CalibrationCapture[], targetValid: boolean): boolean {
  return captures.length > 0 && targetValid;
}

export function canCalibrateFromDetections(detections: Array<Record<string, unknown>>, targetValid: boolean): boolean {
  if (!targetValid) return false;
  return detections.some((item) => Boolean(item.corners_found));
}

export function workflowState(params: {
  captures: CalibrationCapture[];
  detections: Array<Record<string, unknown>>;
  hasCalibrationResult: boolean;
}): Calibration2DWorkflowState {
  if (!params.captures.length) return "NO_CAPTURES";
  if (params.hasCalibrationResult) return "CALIBRATION_VALID";
  if (params.detections.some((item) => Boolean(item.corners_found))) return "CORNERS_DETECTED";
  return "CAPTURES_READY";
}
