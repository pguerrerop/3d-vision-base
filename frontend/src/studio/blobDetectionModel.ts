import type { StudioArtifact } from "../components/studioWorkspaceModel";

export type BlobCandidate = {
  id: string | number;
  areaPx?: number;
  centroid?: [number, number];
  bbox?: [number, number, number, number];
  contour?: [number, number][];
  perimeter?: number;
  equivalentDiameter?: number;
  circularity?: number;
  aspectRatio?: number;
  solidity?: number;
  touchesBorder?: boolean;
  status?: "candidate" | "rejected";
  rejectionReason?: string;
  parentRoi?: unknown;
  // Populated only for reference-detection components (belt_bg/belt_stripe/unknown
  // classification + height stats), never for ball-detection blob candidates.
  role?: string;
  meanHeightMm?: number;
};

export type BlobDetectionSummary = {
  candidateCount: number;
  rejectedCount: number;
  avgDiameter?: number;
  avgCircularity?: number;
  largestArea?: number;
  areaStats?: { min?: number; median?: number; max?: number };
  diameterStats?: { min?: number; median?: number; max?: number };
  circularityStats?: { min?: number; median?: number; max?: number };
  aspectRatioStats?: { min?: number; median?: number; max?: number };
  rejectedReasonCounts?: Record<string, number>;
};

// The blob overlay is always drawn against the plain RGB frame the detector ran on, never
// against a debug/overlay artifact, so contour coordinates line up with the pixels shown.
// Both Studio and the Report walkthrough resolve this the same way so they never drift apart.
export function resolveBlobSetSourceImage(stageArtifacts: StudioArtifact[]): StudioArtifact | null {
  return stageArtifacts.find((item) => item.artifact_id === "source_rgb_image" && item.path)
    ?? stageArtifacts.find((item) => item.kind === "image" && item.path)
    ?? null;
}

export function normalizeBlobDetectionArtifacts(artifacts: StudioArtifact[]): {
  candidates: BlobCandidate[];
  rejected: BlobCandidate[];
  summary: BlobDetectionSummary;
  warnings: string[];
} {
  const metricsArtifact = firstByAlias(artifacts, ["blob_metrics", "detection_metrics"]);
  const contoursArtifact = firstByAlias(artifacts, ["blob_contours", "contours"]);
  const rejectedArtifact = firstByAlias(artifacts, ["blob_rejected", "rejected_candidates"]);

  const metricsPayload = coercePayload(metricsArtifact?.metadata);
  const contoursPayload = coercePayload(contoursArtifact?.metadata);
  const rejectedPayload = coercePayload(rejectedArtifact?.metadata);

  const metricsCandidates = candidateArray(metricsPayload).length
    ? candidateArray(metricsPayload)
    : candidateArray(rec(metricsPayload).entries);
  const contourCandidates = contourArray(contoursPayload).length
    ? contourArray(contoursPayload)
    : contourArray(rec(contoursPayload).entries);
  const rejectedCandidates = rejectedArray(rejectedPayload).length
    ? rejectedArray(rejectedPayload)
    : rejectedArray(rec(rejectedPayload).entries);

  const merged = new Map<string | number, BlobCandidate>();
  for (const row of [...contourCandidates, ...metricsCandidates]) {
    const key = row.id;
    const prev = merged.get(key);
    merged.set(key, { ...prev, ...row, status: "candidate" });
  }
  const rejected = rejectedCandidates.map((row) => ({ ...row, status: "rejected" as const }));
  const candidates = [...merged.values()].filter((item) => item.id !== 0 && item.id !== "");
  const summary = summarize(candidates, rejected, metricsPayload);
  const warnings: string[] = [];
  if (!candidates.length) warnings.push("No blob candidates found. Check cleaned mask, ROI, and cleanup filters.");
  if (!candidates.length && rejected.length) warnings.push("All components were rejected. Review min area, border reject, or aspect ratio filters.");
  return { candidates, rejected, summary, warnings };
}

function firstByAlias(artifacts: StudioArtifact[], aliases: string[]): StudioArtifact | null {
  for (const alias of aliases) {
    const found = artifacts.find((item) => item.artifact_id === alias || String((item.metadata as Record<string, unknown> | undefined)?.semantic_kind ?? "") === alias);
    if (found) return found;
  }
  return null;
}

function coercePayload(meta: unknown): unknown {
  if (meta && typeof meta === "object" && Array.isArray((meta as Record<string, unknown>).entries)) {
    return (meta as Record<string, unknown>).entries;
  }
  return meta;
}

function candidateArray(payload: unknown): BlobCandidate[] {
  if (Array.isArray(payload)) return payload.map(normalizeCandidate).filter((item) => item.id !== 0);
  if (payload && typeof payload === "object") {
    const obj = payload as Record<string, unknown>;
    for (const key of ["candidates", "blobs", "rows"]) {
      if (Array.isArray(obj[key])) return (obj[key] as unknown[]).map(normalizeCandidate).filter((item) => item.id !== 0);
    }
  }
  return [];
}

function contourArray(payload: unknown): BlobCandidate[] {
  if (Array.isArray(payload)) return payload.map(normalizeContour).filter((item) => item.id !== 0);
  if (payload && typeof payload === "object") {
    const obj = payload as Record<string, unknown>;
    for (const key of ["candidates", "blobs", "contours"]) {
      if (Array.isArray(obj[key])) return (obj[key] as unknown[]).map(normalizeContour).filter((item) => item.id !== 0);
    }
  }
  return [];
}

function rejectedArray(payload: unknown): BlobCandidate[] {
  if (Array.isArray(payload)) return payload.map(normalizeCandidate);
  if (payload && typeof payload === "object") {
    const obj = payload as Record<string, unknown>;
    for (const key of ["rejected", "candidates", "rows"]) {
      if (Array.isArray(obj[key])) return (obj[key] as unknown[]).map(normalizeCandidate);
    }
  }
  return [];
}

function normalizeContour(value: unknown): BlobCandidate {
  const obj = rec(value);
  const idValue = obj.id ?? obj.blob_id ?? obj.label ?? obj.object_id;
  return {
    id: (typeof idValue === "string" && idValue.trim()) ? idValue : (num(idValue) ?? 0),
    contour: points(obj.contour ?? obj.contour_points),
    centroid: pair(obj.centroid),
    bbox: quad(obj.bbox),
    parentRoi: obj.parent_roi ?? obj.parentRoi,
  };
}

function normalizeCandidate(value: unknown): BlobCandidate {
  const obj = rec(value);
  const idValue = obj.id ?? obj.blob_id ?? obj.label ?? obj.object_id;
  return {
    id: (typeof idValue === "string" && idValue.trim()) ? idValue : (num(idValue) ?? 0),
    areaPx: num(obj.area_px ?? obj.area ?? obj.pixel_area),
    centroid: pair(obj.centroid),
    bbox: quad(obj.bbox),
    contour: points(obj.contour ?? obj.contour_points),
    perimeter: num(obj.contour_perimeter ?? obj.perimeter),
    equivalentDiameter: num(obj.equivalent_diameter ?? obj.diameter ?? obj.equivalentDiameter),
    circularity: num(obj.circularity),
    aspectRatio: num(obj.aspect_ratio ?? obj.aspectRatio),
    solidity: num(obj.solidity),
    touchesBorder: bool(obj.touches_border ?? obj.touchesBorder),
    rejectionReason: str(obj.rejection_reason ?? obj.reason),
    parentRoi: obj.parent_roi ?? obj.parentRoi,
  };
}

function summarize(candidates: BlobCandidate[], rejected: BlobCandidate[], metricsPayload: unknown): BlobDetectionSummary {
  const areas = candidates.map((item) => item.areaPx).filter((v): v is number => Number.isFinite(v));
  const diameters = candidates.map((item) => item.equivalentDiameter).filter((v): v is number => Number.isFinite(v));
  const circularities = candidates.map((item) => item.circularity).filter((v): v is number => Number.isFinite(v));
  const aspects = candidates.map((item) => item.aspectRatio).filter((v): v is number => Number.isFinite(v));
  const reasonCounts: Record<string, number> = {};
  for (const item of rejected) {
    const key = item.rejectionReason ?? "unknown";
    reasonCounts[key] = (reasonCounts[key] ?? 0) + 1;
  }
  const fromPayload = rec(metricsPayload);
  return {
    candidateCount: candidates.length,
    rejectedCount: rejected.length || num(fromPayload.rejected_count) || 0,
    avgDiameter: avg(diameters),
    avgCircularity: avg(circularities),
    largestArea: areas.length ? Math.max(...areas) : undefined,
    areaStats: stats(areas),
    diameterStats: stats(diameters),
    circularityStats: stats(circularities),
    aspectRatioStats: stats(aspects),
    rejectedReasonCounts: reasonCounts,
  };
}

function stats(values: number[]): { min?: number; median?: number; max?: number } | undefined {
  if (!values.length) return undefined;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted[Math.floor(sorted.length / 2)];
  return { min: sorted[0], median: mid, max: sorted[sorted.length - 1] };
}

function avg(values: number[]): number | undefined {
  if (!values.length) return undefined;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function rec(value: unknown): Record<string, unknown> {
  return (value && typeof value === "object") ? value as Record<string, unknown> : {};
}

function str(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

function bool(value: unknown): boolean | undefined {
  if (typeof value === "boolean") return value;
  return undefined;
}

function num(value: unknown): number | undefined {
  const n = Number(value);
  return Number.isFinite(n) ? n : undefined;
}

function pair(value: unknown): [number, number] | undefined {
  if (!Array.isArray(value) || value.length < 2) return undefined;
  const x = num(value[0]);
  const y = num(value[1]);
  return x == null || y == null ? undefined : [x, y];
}

function quad(value: unknown): [number, number, number, number] | undefined {
  if (!Array.isArray(value) || value.length < 4) return undefined;
  const a = num(value[0]); const b = num(value[1]); const c = num(value[2]); const d = num(value[3]);
  return a == null || b == null || c == null || d == null ? undefined : [a, b, c, d];
}

function points(value: unknown): [number, number][] | undefined {
  if (!Array.isArray(value)) return undefined;
  const out: [number, number][] = [];
  for (const row of value) {
    const p = pair(row);
    if (p) out.push(p);
  }
  return out.length ? out : undefined;
}
