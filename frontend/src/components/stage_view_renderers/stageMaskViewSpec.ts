import type { MaskReferenceCandidate as BinaryMaskReferenceCandidate } from "../binaryMaskArtifacts";
import type { MaskReferenceCandidate as ViewerMaskReferenceCandidate } from "../artifacts/mask/maskRenderingTypes";
import type { StageMaskViewSpec, StageViewDefinition } from "../stageSemantics";
import type { StudioArtifact } from "../studioWorkspaceModel";

const MASK_METADATA_KEYS = ["semantic_type", "artifact_type", "semantic_kind", "image_type", "value_type"] as const;

const RAW_REFERENCE_IDS = ["raw_heightmap", "raw_heightmap_preview", "heightmap_input_preview", "source_heightmap_preview", "take_thumbnail"];
const NORMALIZED_REFERENCE_IDS = ["normalized_heightmap", "normalized_height", ...RAW_REFERENCE_IDS];

const MASK_DEFAULTS_BY_ARTIFACT_ID: Record<string, Partial<StageMaskViewSpec>> = {
  final_plane_inlier_mask: {
    defaultMode: "boundary",
    defaultReferenceRole: "raw",
    preferredReferenceArtifactIds: [...RAW_REFERENCE_IDS, "reference_surface_selected_mask", "expanded_plane_mask"],
    maskOnLabel: "Plane inlier",
    maskOffLabel: "Other",
    description: "Pixels accepted as final inliers for the fitted reference plane.",
  },
  plane_inlier_mask: {
    defaultMode: "boundary",
    defaultReferenceRole: "raw",
    preferredReferenceArtifactIds: [...RAW_REFERENCE_IDS, "reference_surface_selected_mask", "expanded_plane_mask"],
    maskOnLabel: "Plane inlier",
    maskOffLabel: "Other",
  },
  reference_model_inlier_mask: {
    defaultMode: "boundary",
    defaultReferenceRole: "raw",
    preferredReferenceArtifactIds: [...RAW_REFERENCE_IDS, "reference_surface_selected_mask", "expanded_plane_mask"],
    maskOnLabel: "Plane inlier",
    maskOffLabel: "Other",
  },
  reference_surface_selected_mask: {
    defaultMode: "boundary",
    defaultReferenceRole: "raw",
    preferredReferenceArtifactIds: [...RAW_REFERENCE_IDS, "normalized_heightmap"],
    maskOnLabel: "Selected reference",
    maskOffLabel: "Rejected",
    description: "Pixels selected as the reference support before final fit acceptance.",
  },
  selected_reference_support_mask: {
    defaultMode: "boundary",
    defaultReferenceRole: "raw",
    preferredReferenceArtifactIds: [...RAW_REFERENCE_IDS, "normalized_heightmap"],
    maskOnLabel: "Selected reference",
    maskOffLabel: "Rejected",
    description: "Support pixels retained for the selected reference surface.",
  },
  low_gradient_mask: {
    defaultMode: "overlay",
    defaultReferenceRole: "raw",
    preferredReferenceArtifactIds: RAW_REFERENCE_IDS,
    maskOnLabel: "Low gradient",
    maskOffLabel: "Rejected",
    description: "Pixels that pass the low-gradient candidate filter.",
  },
  belt_stripes_mask: {
    defaultMode: "overlay",
    defaultReferenceRole: "raw",
    preferredReferenceArtifactIds: NORMALIZED_REFERENCE_IDS,
    maskOnLabel: "Suppressed stripe",
    maskOffLabel: "Kept",
    description: "Pixels rejected as belt-stripe structure during reference cleanup.",
  },
  cleaned_object_mask: {
    defaultMode: "overlay",
    defaultReferenceRole: "normalized",
    preferredReferenceArtifactIds: NORMALIZED_REFERENCE_IDS,
    maskOnLabel: "Object",
    maskOffLabel: "Background",
    description: "Cleaned foreground object mask after segmentation filtering.",
  },
  final_object_mask: {
    defaultMode: "overlay",
    defaultReferenceRole: "normalized",
    preferredReferenceArtifactIds: NORMALIZED_REFERENCE_IDS,
    maskOnLabel: "Object",
    maskOffLabel: "Background",
    description: "Final object pixels kept for downstream connected components and measurement.",
  },
  plane_fit_roi_mask: {
    defaultMode: "boundary",
    defaultReferenceRole: "raw",
    preferredReferenceArtifactIds: RAW_REFERENCE_IDS,
    maskOnLabel: "Reference ROI",
    maskOffLabel: "Outside ROI",
    description: "ROI pixels eligible for reference-plane fitting.",
  },
  above_threshold_mask: {
    defaultMode: "overlay",
    defaultReferenceRole: "normalized",
    preferredReferenceArtifactIds: NORMALIZED_REFERENCE_IDS,
    maskOnLabel: "Above threshold",
    maskOffLabel: "Below threshold",
    description: "Pixels above the segmentation height threshold.",
  },
  below_reference_mask: {
    defaultMode: "overlay",
    defaultReferenceRole: "normalized",
    preferredReferenceArtifactIds: NORMALIZED_REFERENCE_IDS,
    maskOnLabel: "Below reference",
    maskOffLabel: "Above reference",
    description: "Pixels at or below the reference surface baseline.",
  },
  plane_suppressed_mask: {
    defaultMode: "overlay",
    defaultReferenceRole: "normalized",
    preferredReferenceArtifactIds: NORMALIZED_REFERENCE_IDS,
    maskOnLabel: "Suppressed reference",
    maskOffLabel: "Kept",
    description: "Pixels removed from foreground because they belong to the reference support.",
  },
  rejected_background_residuals: {
    defaultMode: "overlay",
    defaultReferenceRole: "normalized",
    preferredReferenceArtifactIds: NORMALIZED_REFERENCE_IDS,
    maskOnLabel: "Rejected by residual",
    maskOffLabel: "Kept",
    description: "Pixels rejected from foreground because their background residual stays within the reference band.",
  },
  rejected_belt_stripes: {
    defaultMode: "overlay",
    defaultReferenceRole: "normalized",
    preferredReferenceArtifactIds: NORMALIZED_REFERENCE_IDS,
    maskOnLabel: "Rejected stripe",
    maskOffLabel: "Kept",
    description: "Foreground pixels rejected by belt-stripe suppression.",
  },
  background_candidate_mask: {
    defaultMode: "boundary",
    defaultReferenceRole: "raw",
    preferredReferenceArtifactIds: RAW_REFERENCE_IDS,
    maskOnLabel: "Candidate support",
    maskOffLabel: "Other",
    description: "Persisted candidate support mask for the selected reference surface.",
  },
  background_seed_mask: {
    defaultMode: "boundary",
    defaultReferenceRole: "raw",
    preferredReferenceArtifactIds: RAW_REFERENCE_IDS,
    maskOnLabel: "Seed support",
    maskOffLabel: "Other",
    description: "Initial seed mask before later support refinement and residual expansion.",
  },
  expanded_plane_mask: {
    defaultMode: "boundary",
    defaultReferenceRole: "raw",
    preferredReferenceArtifactIds: [...RAW_REFERENCE_IDS, "reference_surface_selected_mask"],
    maskOnLabel: "Expanded support",
    maskOffLabel: "Other",
    description: "ROI-valid pixels within the plane residual tolerance band.",
  },
  height_gate_mask: {
    defaultMode: "overlay",
    defaultReferenceRole: "raw",
    preferredReferenceArtifactIds: RAW_REFERENCE_IDS,
    maskOnLabel: "Height gate",
    maskOffLabel: "Rejected",
    description: "Pixels that pass the reference height-gate filter.",
  },
};

function artifactMetadataTokens(artifact: StudioArtifact): string[] {
  const meta = (artifact.metadata ?? {}) as Record<string, unknown>;
  return MASK_METADATA_KEYS.map((key) => String(meta[key] ?? "").trim().toLowerCase()).filter(Boolean);
}

export function matchesStageMaskSemanticType(artifact: StudioArtifact, semanticTypes: string[]): boolean {
  if (!semanticTypes.length) return false;
  const tokens = artifactMetadataTokens(artifact);
  return semanticTypes.some((semanticType) => {
    const normalized = semanticType.trim().toLowerCase();
    return tokens.some((token) => token === normalized || token.includes(normalized) || normalized.includes(token));
  });
}

export function uniqueMaskArtifacts(artifacts: StudioArtifact[]): StudioArtifact[] {
  const seen = new Set<string>();
  return artifacts.filter((artifact) => {
    const key = `${artifact.stage_id}:${artifact.artifact_id}:${artifact.path ?? ""}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

export function candidateRoleForViewer(candidate: BinaryMaskReferenceCandidate): ViewerMaskReferenceCandidate["role"] {
  if (candidate.role === "lineage_source") return "computed_from";
  if (candidate.role === "normalized_height") return "normalized";
  if (candidate.role === "original_acquisition" || candidate.role === "raw_heightmap") return "raw";
  if (candidate.role === "derived_diagnostic" || candidate.role === "overlay") return "diagnostic";
  return "source";
}

function artifactIdsForSpec(spec: StageMaskViewSpec): string[] {
  return [
    spec.maskArtifactId,
    ...(spec.maskArtifactIds ?? []),
  ].filter((value): value is string => Boolean(value && value.trim()));
}

export function resolveStageMaskSpec(
  view: StageViewDefinition,
  resolvedArtifact?: StudioArtifact | null,
): StageMaskViewSpec {
  const base = view.maskViewSpec ?? {};
  const registryKeys = [
    ...artifactIdsForSpec(base),
    resolvedArtifact?.artifact_id ?? null,
  ].filter((value): value is string => Boolean(value && value.trim()));
  const merged = registryKeys.reduce<StageMaskViewSpec>(
    (current, key) => ({ ...current, ...(MASK_DEFAULTS_BY_ARTIFACT_ID[key] ?? {}) }),
    {},
  );
  return { ...merged, ...base };
}

export function formatStageMaskDiagnosticsNote(
  spec: StageMaskViewSpec,
  resolutionNote: string,
): string {
  const parts = [spec.description, spec.diagnosticsLabel ? `${spec.diagnosticsLabel}: ${resolutionNote}` : resolutionNote].filter(Boolean);
  return parts.join(" · ");
}

function matchesPreferredReference(candidate: BinaryMaskReferenceCandidate, spec: StageMaskViewSpec): boolean {
  const preferredIds = spec.preferredReferenceArtifactIds ?? [];
  if (preferredIds.includes(candidate.artifact.artifact_id)) return true;
  if (spec.preferredReferenceSemanticTypes?.length && matchesStageMaskSemanticType(candidate.artifact, spec.preferredReferenceSemanticTypes)) return true;
  if (spec.defaultReferenceRole && candidateRoleForViewer(candidate) === spec.defaultReferenceRole) return true;
  return false;
}

export function rankStageMaskReferenceCandidates(
  candidates: BinaryMaskReferenceCandidate[],
  spec: StageMaskViewSpec,
): BinaryMaskReferenceCandidate[] {
  return candidates
    .slice()
    .sort((left, right) => {
      const leftPreferred = matchesPreferredReference(left, spec);
      const rightPreferred = matchesPreferredReference(right, spec);
      if (leftPreferred !== rightPreferred) return leftPreferred ? -1 : 1;
      return left.priority - right.priority;
    });
}

export function resolveStageMaskArtifact(
  view: StageViewDefinition,
  artifacts: StudioArtifact[],
): StudioArtifact | null {
  const spec = resolveStageMaskSpec(view, null);
  const ids = spec.maskArtifactIds?.length ? spec.maskArtifactIds : (spec.maskArtifactId ? [spec.maskArtifactId] : []);
  for (const artifactId of ids) {
    const found = artifacts.find((artifact) => artifact.artifact_id === artifactId);
    if (found) return found;
  }
  if (spec.maskSemanticTypes?.length) {
    const found = artifacts.find((artifact) => matchesStageMaskSemanticType(artifact, spec.maskSemanticTypes ?? []));
    if (found) return found;
  }
  return artifacts.find((artifact) => artifact.kind === "image") ?? null;
}

export function resolveStageMaskDefaultMode(
  view: StageViewDefinition,
  artifact: StudioArtifact,
  artifacts: StudioArtifact[],
  resolution?: {
    sourceArtifact?: unknown;
    sourceUrl?: string | null;
  },
) {
  const spec = resolveStageMaskSpec(view, artifact);
  if (spec.defaultMode) return spec.defaultMode;
  if (!resolution?.sourceArtifact || !resolution?.sourceUrl) return "mask";
  const id = String(artifact.artifact_id ?? "").toLowerCase();
  if (
    id.includes("selected_surface")
    || id.includes("reference_surface_selected_mask")
    || id.includes("selected_blob_cluster")
    || id.includes("plane_inlier")
    || id.includes("expanded_plane")
    || id.includes("low_gradient_mask")
    || id.includes("roi")
  ) {
    return "overlay";
  }
  return "boundary";
}
