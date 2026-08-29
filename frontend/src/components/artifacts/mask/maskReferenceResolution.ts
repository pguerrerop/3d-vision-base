import type { MaskArtifactDiagnostics, MaskReferenceCandidate } from "./maskRenderingTypes";

type ResolutionLike = {
  sourceLabel?: string;
  sourceArtifact?: { artifact_id?: string | null; coordinate_space?: string | null; metadata?: Record<string, unknown> | null } | null;
  computedFrom?: { label: string; artifact: { artifact_id?: string | null } } | null;
  warning?: string;
  inferred?: boolean;
};

function maskDimensionsLabel(maskArtifact: { metadata?: Record<string, unknown> | null }): string | undefined {
  const meta = (maskArtifact.metadata ?? {}) as Record<string, unknown>;
  const shape = Array.isArray(meta.shape) ? meta.shape : Array.isArray(meta.source_shape) ? meta.source_shape : null;
  if (!shape || shape.length < 2) return undefined;
  const height = Number(shape[0]);
  const width = Number(shape[1]);
  if (!Number.isFinite(height) || !Number.isFinite(width) || height <= 0 || width <= 0) return undefined;
  return `${Math.round(width)} x ${Math.round(height)}`;
}

export function resolveSelectedReferenceSource(
  selectedReferenceId: string | null | undefined,
  defaultReferenceId: string | null | undefined,
  resolution: ResolutionLike | null | undefined,
): "user" | "inferred" | "default" | undefined {
  if (!resolution?.sourceArtifact?.artifact_id) return undefined;
  if (selectedReferenceId && selectedReferenceId !== defaultReferenceId) return "user";
  if (resolution.inferred) return "inferred";
  return "default";
}

export function describeReferenceCandidate(
  candidate: MaskReferenceCandidate,
  selectedReferenceId?: string | null,
  defaultReferenceId?: string | null,
): string {
  const suffix: string[] = [];
  if (candidate.id === selectedReferenceId && selectedReferenceId && selectedReferenceId !== defaultReferenceId) suffix.push("user");
  else if (candidate.inferred) suffix.push("inferred");
  if (candidate.role === "computed_from") suffix.push("computed from");
  return suffix.length ? `${candidate.label} (${suffix.join(", ")})` : candidate.label;
}

export function buildMaskArtifactDiagnostics(args: {
  maskArtifact: { artifact_id?: string | null; metadata?: Record<string, unknown> | null; coordinate_space?: string | null };
  resolution: ResolutionLike | null | undefined;
  selectedReferenceId?: string | null;
  defaultReferenceId?: string | null;
}): MaskArtifactDiagnostics {
  const { maskArtifact, resolution, selectedReferenceId, defaultReferenceId } = args;
  const selectedReferenceSource = resolveSelectedReferenceSource(selectedReferenceId, defaultReferenceId, resolution);
  return {
    selectedReferenceLabel: resolution?.sourceLabel,
    selectedReferenceId: resolution?.sourceArtifact?.artifact_id ?? undefined,
    selectedReferenceSource,
    computedFromLabel: resolution?.computedFrom?.label,
    computedFromArtifactId: resolution?.computedFrom?.artifact.artifact_id ?? undefined,
    coordinateSpace: resolution?.sourceArtifact?.coordinate_space ?? maskArtifact.coordinate_space ?? undefined,
    dimensions: maskDimensionsLabel(maskArtifact),
    artifactId: maskArtifact.artifact_id ?? undefined,
    sourceArtifactId: resolution?.sourceArtifact?.artifact_id ?? undefined,
    note: resolution?.warning,
  };
}
