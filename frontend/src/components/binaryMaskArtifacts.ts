import type { TakeDetail } from "../api/client";
import type { MaskRenderMode } from "./artifacts/mask/maskRenderingTypes";
import type { StudioArtifact } from "./studioWorkspaceModel";

export type BinaryMaskViewMode = MaskRenderMode;
export type MaskReferenceAlignment =
  | "exact"
  | "same_dimensions"
  | "assumed_stage_context"
  | "dimension_mismatch"
  | "unavailable";
export type MaskReferenceRole =
  | "original_acquisition"
  | "raw_heightmap"
  | "take_thumbnail"
  | "normalized_height"
  | "derived_diagnostic"
  | "lineage_source"
  | "overlay"
  | "other";

export type MaskReferenceCandidate = {
  artifact: StudioArtifact;
  label: string;
  role: MaskReferenceRole;
  priority: number;
  alignment: MaskReferenceAlignment;
  reason?: string;
  warning?: string;
  inferred: boolean;
  sourceUrl?: string;
  matchKey: string;
};

export type MaskReferenceDiagnostics = {
  originalAcquisitionCandidateFound: boolean;
  sourceArtifactsReceived: number;
  runArtifactsScanned: number;
  stageArtifactsScanned: number;
  takeAssetFieldsChecked: string[];
  candidatesFound: string[];
};

export type MaskReferenceResolution = {
  selected?: MaskReferenceCandidate;
  candidates: MaskReferenceCandidate[];
  computedFrom?: MaskReferenceCandidate;
  warning?: string;
  sourceArtifact?: StudioArtifact;
  sourceLabel?: string;
  sourceUrl?: string;
  alignment: MaskReferenceAlignment;
  reason?: string;
  candidateIdsTried?: string[];
  inferred: boolean;
  diagnostics: MaskReferenceDiagnostics;
};

export type MaskReferenceResolverInput = {
  maskArtifact: StudioArtifact;
  selectedStageId?: string;
  selectedTake?: TakeDetail | null;
  currentStageArtifacts: StudioArtifact[];
  allRunArtifacts: StudioArtifact[];
  sourceArtifacts?: StudioArtifact[];
  resolvedArtifacts?: StudioArtifact[];
};

const MASK_META_KEYS = [
  "semantic_type",
  "artifact_type",
  "value_type",
  "image_type",
  "artifact_kind",
  "semantic_kind",
];

const EXPLICIT_BINARY_MASK_VALUES = [
  "mask",
  "binary_mask",
  "boolean_mask",
  "bool_mask",
  "segmentation_mask",
  "valid_mask",
];

const FALLBACK_MASK_PATTERNS = [
  "valid_mask",
  "low_gradient_mask",
  "selected_surface_mask",
  "selected_reference_surface_mask",
  "selected_reference_support_mask",
  "reference_surface_selected_mask",
  "expanded_plane_mask",
  "reference_model_inlier_mask",
  "reference_model_support_mask",
  "reference_suppression_mask",
  "final_plane_inlier_mask",
  "plane_inlier_mask",
  "threshold_mask",
  "cleaned_mask",
  "above_belt_mask",
  "wide_object_mask",
  "background_seed_mask",
  "background_candidate_mask",
  "belt_bg_mask",
  "belt_base_mask",
  "belt_stripes_mask",
  "unknown_low_gradient_mask",
  "surface_suppression_mask",
  "object_search_domain_mask",
  "belt_stripes_tophat_mask",
  "belt_stripes_shape_mask",
  "belt_above_belt_mask",
  "belt_wide_object_mask",
  "below_reference_mask",
  "above_threshold_mask",
  "segmentation_mask",
];

const FRIENDLY_LABELS: Record<string, string> = {
  original_acquisition: "Original acquisition",
  original_take: "Original acquisition",
  raw_heightmap: "Raw heightmap",
  raw_heightmap_preview: "Raw heightmap",
  source_heightmap_preview: "Raw heightmap",
  source_heightmap: "Raw heightmap",
  source_heightmap_heightmap: "Raw heightmap",
  heightmap_preview: "Raw heightmap",
  heightmap_input_preview: "Heightmap input preview",
  replay_heightmap_preview: "Raw heightmap",
  heightmap: "Raw heightmap",
  raw_sensor_z: "Original acquisition / Raw heightmap",
  input_heightmap: "Original acquisition / Raw heightmap",
  input_preview: "Original acquisition / Raw heightmap",
  take_thumbnail: "Original acquisition",
  take_preview: "Original acquisition",
  filtered_depth_plot: "Filtered depth plot",
  flat_candidate_depth_plot: "Filtered depth plot",
  background_depth_plot: "Filtered depth plot",
  depth_gradient: "Depth gradient",
  depth_gradient_magnitude: "Depth gradient",
  surface_candidates: "Surface candidates",
  reference_surface_candidates: "Surface candidates",
  background_candidate_mask: "Surface candidates",
  belt_bg_mask: "Belt background",
  background_seed_mask: "Surface candidates",
  selected_surface: "Selected surface",
  selected_reference_surface: "Selected surface",
  selected_reference_support_mask: "Selected support",
  stripe_filtered_reference_support_mask: "Stripe-filtered support",
  reference_surface_selected_mask: "Selected surface",
  support_removed_by_candidate_refinement: "Removed by candidate refinement",
  support_removed_by_stripe_filter: "Removed by stripe filter",
  reference_model_inlier_mask: "Model support",
  reference_model_support_mask: "Reference model support",
  support_removed_by_model_residual: "Removed by model residual",
  reference_suppression_mask: "Suppression mask",
  surface_suppression_mask: "Surface suppression mask",
  unknown_low_gradient_mask: "Unknown low-gradient",
  object_search_domain_mask: "Object search domain",
  residual_heatmap: "Residual heatmap",
  belt_plane_residuals: "Residual heatmap",
  normalized_heightmap: "Normalized height",
  normalized_height: "Normalized height",
  normalized_heightmap_display: "Normalized height",
  normalized_grayscale_image: "Normalized height",
  height_overlay: "Height overlay",
  height_segmentation_overlay: "Height overlay",
  height_segmentation: "Height overlay",
  measurement_overlay: "Measurement overlay",
};

type SourceGroup = {
  ids: string[];
  label: string;
  role: MaskReferenceRole;
};

const SOURCE_GROUPS = {
  originalAcquisition: {
    ids: [
      "raw_heightmap",
      "heightmap",
      "heightmap_input_preview",
      "raw_sensor_z",
      "input_heightmap",
      "source_heightmap",
      "source_heightmap_heightmap",
      "raw_heightmap_preview",
      "source_heightmap_preview",
      "heightmap_preview",
      "input_preview",
      "replay_heightmap_preview",
    ],
    label: "Original acquisition / Raw heightmap",
    role: "original_acquisition",
  },
  rawHeight: {
    ids: [
      "raw_heightmap",
      "heightmap",
      "heightmap_input_preview",
      "raw_sensor_z",
      "raw_heightmap_preview",
      "source_heightmap_preview",
      "source_heightmap_heightmap",
      "heightmap_preview",
      "input_preview",
      "replay_heightmap_preview",
    ],
    label: "Raw heightmap",
    role: "raw_heightmap",
  },
  filteredDepth: {
    ids: ["filtered_depth_plot", "flat_candidate_depth_plot", "background_depth_plot"],
    label: "Filtered depth plot",
    role: "derived_diagnostic",
  },
  depthGradient: {
    ids: ["depth_gradient", "depth_gradient_magnitude"],
    label: "Depth gradient",
    role: "derived_diagnostic",
  },
  surfaceCandidates: {
    ids: ["surface_candidates", "reference_surface_candidates", "background_candidate_mask", "background_seed_mask"],
    label: "Surface candidates",
    role: "derived_diagnostic",
  },
  selectedSurface: {
    ids: ["selected_surface", "selected_reference_surface", "reference_surface_selected_mask"],
    label: "Selected surface",
    role: "derived_diagnostic",
  },
  residualHeatmap: {
    ids: ["residual_heatmap", "belt_plane_residuals", "plane_inliers"],
    label: "Residual heatmap",
    role: "derived_diagnostic",
  },
  normalizedHeight: {
    ids: ["normalized_heightmap", "normalized_height", "normalized_heightmap_display", "normalized_grayscale_image"],
    label: "Normalized height",
    role: "normalized_height",
  },
  heightOverlay: {
    ids: ["height_overlay", "height_segmentation_overlay", "height_segmentation"],
    label: "Height overlay",
    role: "overlay",
  },
  measurementOverlay: {
    ids: ["measurement_overlay", "classification_overlay_image"],
    label: "Measurement overlay",
    role: "overlay",
  },
  takeThumbnail: {
    ids: ["take_thumbnail", "take_preview"],
    label: "Take thumbnail",
    role: "take_thumbnail",
  },
} satisfies Record<string, SourceGroup>;

export function finiteOrDefault(value: unknown, fallback: number): number {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function normalizeText(value: unknown): string {
  return String(value ?? "").trim().toLowerCase();
}

function includesAny(text: string, needles: string[]): boolean {
  return needles.some((needle) => text.includes(needle));
}

function metadataOf(artifact: StudioArtifact): Record<string, unknown> {
  return (artifact.metadata ?? {}) as Record<string, unknown>;
}

function artifactSearchText(artifact: StudioArtifact): string {
  const meta = metadataOf(artifact);
  return [
    artifact.artifact_id,
    artifact.title,
    artifact.description,
    artifact.path,
    meta.semantic_type,
    meta.artifact_type,
    meta.semantic_kind,
    meta.image_type,
    meta.value_type,
    meta.label,
    meta.name,
    meta.source_group,
    meta.source_key,
  ].map(normalizeText).join(" ");
}

function preferredBinaryMaskMetadata(meta: Record<string, unknown>): boolean {
  const values = MASK_META_KEYS.map((key) => normalizeText(meta[key]));
  if (values.some((value) => EXPLICIT_BINARY_MASK_VALUES.includes(value))) return true;
  return values.some((value) => value.includes("binary") && value.includes("mask"));
}

function heuristicBinaryMaskMatch(text: string): boolean {
  if (includesAny(text, ["overlay", "histogram", "heatmap", "plot"])) return false;
  if (FALLBACK_MASK_PATTERNS.some((pattern) => text.includes(pattern))) return true;
  return /(^|[\s_./-])mask($|[\s_./-])/i.test(text);
}

function explicitArtifactSourceIds(artifact: StudioArtifact): string[] {
  const looseArtifact = artifact as StudioArtifact & { source_artifact_id?: string | null };
  const meta = metadataOf(artifact);
  const derivedFrom = Array.isArray(meta.derived_from)
    ? meta.derived_from.map((value) => String(value))
    : typeof meta.derived_from === "string"
      ? [meta.derived_from]
      : [];
  return [
    looseArtifact.source_artifact_id,
    typeof meta.source_artifact_id === "string" ? meta.source_artifact_id : null,
    typeof meta.target_artifact_id === "string" ? meta.target_artifact_id : null,
    ...derivedFrom,
  ].filter((value): value is string => Boolean(value && value.trim()));
}

function sameImageCandidate(a: StudioArtifact, b: StudioArtifact): boolean {
  return a.artifact_id === b.artifact_id;
}

function labelForArtifact(artifact: StudioArtifact, fallback?: string): string {
  return fallback ?? FRIENDLY_LABELS[artifact.artifact_id] ?? artifact.title;
}

function artifactTokens(artifact: StudioArtifact): string[] {
  const meta = metadataOf(artifact);
  return [
    artifact.artifact_id,
    artifact.title,
    artifact.description,
    artifact.path,
    meta.semantic_type,
    meta.artifact_type,
    meta.semantic_kind,
    meta.image_type,
    meta.value_type,
    meta.label,
    meta.name,
    meta.source_group,
    meta.source_key,
  ].map(normalizeText).filter(Boolean);
}

function parseShape(value: unknown): [number, number] | null {
  if (!Array.isArray(value) || value.length < 2) return null;
  const h = finiteOrDefault(value[0], NaN);
  const w = finiteOrDefault(value[1], NaN);
  if (!Number.isFinite(h) || !Number.isFinite(w) || h <= 0 || w <= 0) return null;
  return [Math.round(h), Math.round(w)];
}

function artifactShape(artifact: StudioArtifact): [number, number] | null {
  const meta = metadataOf(artifact);
  return (
    parseShape(meta.shape)
    ?? parseShape(meta.source_shape)
    ?? parseShape(meta.numeric_array_shape)
    ?? parseShape(meta.valid_mask_shape)
    ?? parseShape((meta.projection_metadata as Record<string, unknown> | undefined)?.shape)
  );
}

function sameCoordinateSpace(mask: StudioArtifact, source: StudioArtifact): boolean {
  if (mask.coordinate_space && source.coordinate_space && mask.coordinate_space === source.coordinate_space) return true;
  const maskProjection = mask.projection_type ?? null;
  const sourceProjection = source.projection_type ?? null;
  return Boolean(maskProjection && sourceProjection && maskProjection === sourceProjection);
}

function stageAwareSourceGroups(artifact: StudioArtifact): SourceGroup[] {
  const text = artifactSearchText(artifact);
  const stage = normalizeText(artifact.stage_id);
  if (includesAny(text, ["reference_surface", "plane_inlier_mask", "expanded_plane_mask", "background_seed_mask", "low_gradient_mask"]) || includesAny(stage, ["detect_belt_plane", "normalize_heights_to_plane"])) {
    return [
      SOURCE_GROUPS.originalAcquisition,
      SOURCE_GROUPS.rawHeight,
      SOURCE_GROUPS.filteredDepth,
      SOURCE_GROUPS.depthGradient,
      SOURCE_GROUPS.surfaceCandidates,
      SOURCE_GROUPS.selectedSurface,
      SOURCE_GROUPS.residualHeatmap,
      SOURCE_GROUPS.takeThumbnail,
    ];
  }
  if (includesAny(text, ["valid_mask", "low_gradient_mask"])) {
    return [SOURCE_GROUPS.originalAcquisition, SOURCE_GROUPS.rawHeight, SOURCE_GROUPS.depthGradient, SOURCE_GROUPS.takeThumbnail];
  }
  if (includesAny(text, ["threshold_mask", "cleaned_mask", "object_mask", "segmentation_mask", "wide_object_mask"]) || includesAny(stage, ["remove_belt_segment_objects", "segmentation"])) {
    return [SOURCE_GROUPS.normalizedHeight, SOURCE_GROUPS.originalAcquisition, SOURCE_GROUPS.rawHeight, SOURCE_GROUPS.heightOverlay, SOURCE_GROUPS.takeThumbnail];
  }
  if (includesAny(text, ["above_belt_mask"])) {
    return [SOURCE_GROUPS.normalizedHeight, SOURCE_GROUPS.originalAcquisition, SOURCE_GROUPS.rawHeight, SOURCE_GROUPS.takeThumbnail];
  }
  if (includesAny(stage, ["classification", "measurement"])) {
    return [SOURCE_GROUPS.normalizedHeight, SOURCE_GROUPS.measurementOverlay, SOURCE_GROUPS.takeThumbnail];
  }
  return [SOURCE_GROUPS.originalAcquisition, SOURCE_GROUPS.rawHeight, SOURCE_GROUPS.normalizedHeight, SOURCE_GROUPS.takeThumbnail];
}

function findAnyImageArtifact(
  artifacts: StudioArtifact[],
  matcher: (artifact: StudioArtifact) => boolean,
): StudioArtifact | null {
  return artifacts.find((artifact) => artifact.kind === "image" && matcher(artifact)) ?? null;
}

function findMatchingArtifact(
  artifacts: StudioArtifact[],
  query: string,
): StudioArtifact | null {
  const normalized = normalizeText(query);
  if (!normalized) return null;
  const exact = findAnyImageArtifact(artifacts, (artifact) =>
    artifactTokens(artifact).some((token) => token === normalized)
  );
  if (exact) return exact;
  return findAnyImageArtifact(artifacts, (artifact) =>
    artifactTokens(artifact).some((token) => token.includes(normalized) || normalized.includes(token))
  );
}

function uniqueCandidates(candidates: MaskReferenceCandidate[]): MaskReferenceCandidate[] {
  const seen = new Set<string>();
  return candidates.filter((candidate) => {
    const key = candidate.artifact.artifact_id;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function toCandidate(
  artifact: StudioArtifact,
  opts: {
    label: string;
    role: MaskReferenceRole;
    priority: number;
    inferred: boolean;
    matchKey: string;
    alignment?: MaskReferenceAlignment;
    reason?: string;
    warning?: string;
    sourceUrl?: string;
  },
): MaskReferenceCandidate {
  return {
    artifact,
    label: opts.label,
    role: opts.role,
    priority: opts.priority,
    inferred: opts.inferred,
    matchKey: opts.matchKey,
    alignment: opts.alignment ?? "same_dimensions",
    reason: opts.reason,
    warning: opts.warning,
    sourceUrl: opts.sourceUrl,
  };
}

function externalSourceUrl(artifact: StudioArtifact): string | undefined {
  const url = metadataOf(artifact).external_source_url;
  return typeof url === "string" && url.trim() ? url : undefined;
}

function checkedTakeAssetFields(): string[] {
  return [
    "thumbnail_url",
    "preview_url",
    "assets.heightmap.heightmap",
    "assets.heightmap.raw_heightmap",
    "assets.heightmap.preview",
    "metadata.files.heightmap_preview",
    "source_artifacts",
    "replay_manifest.assets.heightmap",
    "replay_manifest.assets.preview",
  ];
}

function dedupeArtifacts(artifacts: StudioArtifact[]): StudioArtifact[] {
  const seen = new Set<string>();
  const result: StudioArtifact[] = [];
  for (const artifact of artifacts) {
    const key = `${artifact.stage_id}:${artifact.artifact_id}:${artifact.path ?? ""}`;
    if (seen.has(key)) continue;
    seen.add(key);
    result.push(artifact);
  }
  return result;
}

function normalizeResolverInput(
  artifactOrInput: StudioArtifact | MaskReferenceResolverInput,
  artifacts: StudioArtifact[] = [],
): MaskReferenceResolverInput {
  if ("maskArtifact" in artifactOrInput) return artifactOrInput;
  return {
    maskArtifact: artifactOrInput,
    currentStageArtifacts: artifacts,
    allRunArtifacts: artifacts,
    sourceArtifacts: [],
    resolvedArtifacts: [],
  };
}

function collectArtifactPool(input: MaskReferenceResolverInput): StudioArtifact[] {
  return dedupeArtifacts([
    ...input.currentStageArtifacts,
    ...input.allRunArtifacts,
    ...(input.resolvedArtifacts ?? []),
    ...(input.sourceArtifacts ?? []),
  ]);
}

function sourceDiagnostics(input: MaskReferenceResolverInput, candidates: MaskReferenceCandidate[]): MaskReferenceDiagnostics {
  return {
    originalAcquisitionCandidateFound: candidates.some((candidate) => candidate.role === "original_acquisition" || candidate.role === "raw_heightmap"),
    sourceArtifactsReceived: input.sourceArtifacts?.length ?? 0,
    runArtifactsScanned: input.allRunArtifacts.length,
    stageArtifactsScanned: input.currentStageArtifacts.length,
    takeAssetFieldsChecked: checkedTakeAssetFields(),
    candidatesFound: candidates.map((candidate) => candidate.label),
  };
}

export function isBinaryMaskArtifact(artifact: StudioArtifact): boolean {
  if (artifact.kind !== "image") return false;
  const meta = metadataOf(artifact);
  if (preferredBinaryMaskMetadata(meta)) return true;
  return heuristicBinaryMaskMatch(artifactSearchText(artifact));
}

export function resolveBinaryMaskReferenceCandidates(
  artifactOrInput: StudioArtifact | MaskReferenceResolverInput,
  artifacts: StudioArtifact[] = [],
): MaskReferenceCandidate[] {
  const input = normalizeResolverInput(artifactOrInput, artifacts);
  const artifact = input.maskArtifact;
  const artifactPool = collectArtifactPool(input);
  if (!isBinaryMaskArtifact(artifact)) return [];

  const lineage: MaskReferenceCandidate[] = [];
  for (const [index, sourceId] of explicitArtifactSourceIds(artifact).entries()) {
    const found = findMatchingArtifact(artifactPool, sourceId);
    if (!found || sameImageCandidate(found, artifact)) continue;
    lineage.push(
      toCandidate(found, {
        label: labelForArtifact(found),
        role: "lineage_source",
        priority: 1000 + index,
        inferred: false,
        matchKey: sourceId,
        sourceUrl: externalSourceUrl(found),
      }),
    );
  }

  const visual = stageAwareSourceGroups(artifact)
    .map((group, index) => {
      for (const candidateId of group.ids) {
        const match = findMatchingArtifact(artifactPool, candidateId);
        if (match && !sameImageCandidate(match, artifact)) {
          const role = candidateId.includes("thumbnail") || candidateId.includes("take_preview")
            ? "take_thumbnail"
            : group.role;
          return toCandidate(match, {
            label: labelForArtifact(match, group.label),
            role,
            priority: role === "take_thumbnail" ? 90 + index : index,
            inferred: true,
            matchKey: candidateId,
            sourceUrl: externalSourceUrl(match),
          });
        }
      }
      return null;
    })
    .filter((value): value is MaskReferenceCandidate => value !== null);

  return uniqueCandidates([...visual, ...lineage]).sort((a, b) => a.priority - b.priority);
}

export function resolveBinaryMaskReference(
  artifactOrInput: StudioArtifact | MaskReferenceResolverInput,
  artifactsOrSelectedSourceArtifactId?: StudioArtifact[] | string | null,
  selectedSourceArtifactIdOrSourceUrl?: string | null,
  sourceUrlMaybe?: string | null,
): MaskReferenceResolution {
  const artifacts = Array.isArray(artifactsOrSelectedSourceArtifactId) ? artifactsOrSelectedSourceArtifactId : [];
  const input = normalizeResolverInput(artifactOrInput, artifacts);
  const artifact = input.maskArtifact;
  const selectedSourceArtifactId = Array.isArray(artifactsOrSelectedSourceArtifactId)
    ? selectedSourceArtifactIdOrSourceUrl
    : artifactsOrSelectedSourceArtifactId;
  const overrideSourceUrl = Array.isArray(artifactsOrSelectedSourceArtifactId)
    ? sourceUrlMaybe
    : selectedSourceArtifactIdOrSourceUrl;
  const artifactPool = collectArtifactPool(input);
  const candidates = resolveBinaryMaskReferenceCandidates(input);
  const diagnostics = sourceDiagnostics(input, candidates);
  const computedFromArtifact = explicitArtifactSourceIds(artifact)
    .map((sourceId) => findMatchingArtifact(artifactPool, sourceId))
    .find((candidate): candidate is StudioArtifact => Boolean(candidate && !sameImageCandidate(candidate, artifact)));
  const selected = selectedSourceArtifactId
    ? candidates.find((candidate) => candidate.artifact.artifact_id === selectedSourceArtifactId) ?? candidates[0]
    : candidates[0];
  const computedFrom = computedFromArtifact
    ? toCandidate(computedFromArtifact, {
        label: labelForArtifact(computedFromArtifact),
        role: "lineage_source",
        priority: 2000,
        inferred: false,
        matchKey: computedFromArtifact.artifact_id,
        sourceUrl: externalSourceUrl(computedFromArtifact),
      })
    : candidates.find((candidate) => candidate.role === "lineage_source");
  const candidateIdsTried = [
    ...explicitArtifactSourceIds(artifact),
    ...stageAwareSourceGroups(artifact).flatMap((group) => group.ids),
  ];
  if (!selected) {
    return {
      candidates: [],
      computedFrom,
      alignment: "unavailable",
      reason: "no matching source artifact found",
      candidateIdsTried,
      warning: "Comparison source unavailable. Showing mask-only view.",
      inferred: false,
      diagnostics,
    };
  }

  const resolvedSourceUrl = overrideSourceUrl ?? selected.sourceUrl;
  if (!selected.artifact.path && !resolvedSourceUrl) {
    const unavailable = {
      ...selected,
      alignment: "unavailable" as const,
      reason: "reference artifact found but image URL is unavailable",
      warning: "Reference artifact found but image URL is unavailable.",
      sourceUrl: undefined,
    };
    return {
      selected: unavailable,
      candidates,
      computedFrom,
      sourceArtifact: unavailable.artifact,
      sourceLabel: unavailable.label,
      sourceUrl: undefined,
      alignment: unavailable.alignment,
      reason: unavailable.reason,
      candidateIdsTried,
      warning: unavailable.warning,
      inferred: unavailable.inferred,
      diagnostics,
    };
  }
  const maskShape = artifactShape(artifact);
  const sourceShape = artifactShape(selected.artifact);
  let alignment: MaskReferenceAlignment = selected.inferred ? "assumed_stage_context" : "same_dimensions";
  let reason: string | undefined = selected.inferred ? "matched using stage-aware visual reference ranking" : undefined;
  let warning: string | undefined = selected.inferred ? "Reference inferred from stage context." : undefined;
  if (maskShape && sourceShape) {
    if (maskShape[0] !== sourceShape[0] || maskShape[1] !== sourceShape[1]) {
      alignment = "dimension_mismatch";
      reason = "reference dimensions do not match mask";
      warning = "Reference dimensions do not match mask. Overlay disabled.";
    } else if (sameCoordinateSpace(artifact, selected.artifact)) {
      alignment = "exact";
      reason = undefined;
      warning = undefined;
    } else {
      alignment = selected.inferred ? "assumed_stage_context" : "same_dimensions";
    }
  }
  const selectedCandidate = {
    ...selected,
    alignment,
    reason,
    warning,
    sourceUrl: resolvedSourceUrl ?? undefined,
  };
  return {
    selected: selectedCandidate,
    candidates,
    computedFrom,
    sourceArtifact: selected.artifact,
    sourceLabel: selected.label,
    sourceUrl: resolvedSourceUrl ?? undefined,
    alignment,
    reason,
    candidateIdsTried,
    warning,
    inferred: selected.inferred,
    diagnostics,
  };
}

export function resolveBinaryMaskSourceArtifact(
  artifactOrInput: StudioArtifact | MaskReferenceResolverInput,
  artifacts: StudioArtifact[] = [],
): StudioArtifact | null {
  return resolveBinaryMaskReference(artifactOrInput, artifacts).sourceArtifact ?? null;
}

export function resolveBinaryMaskInvalidArtifact(
  artifact: StudioArtifact,
  artifacts: StudioArtifact[],
): StudioArtifact | null {
  const meta = metadataOf(artifact);
  const explicitId = typeof meta.valid_mask_artifact_id === "string"
    ? meta.valid_mask_artifact_id
    : typeof meta.invalid_mask_artifact_id === "string"
      ? meta.invalid_mask_artifact_id
      : null;
  const explicit = explicitId ? findMatchingArtifact(artifacts, explicitId) : null;
  if (explicit && explicit.artifact_id !== artifact.artifact_id) return explicit;
  return findAnyImageArtifact(artifacts, (candidate) =>
    candidate.artifact_id !== artifact.artifact_id
    && (candidate.artifact_id === "valid_mask" || candidate.artifact_id.endsWith("_valid_mask"))
  );
}

export function validateBinaryMaskReferenceAlignment(
  resolution: MaskReferenceResolution,
  maskSize?: { width: number; height: number } | null,
  sourceSize?: { width: number; height: number } | null,
): MaskReferenceResolution {
  if (!resolution.selected || !resolution.sourceArtifact) return resolution;
  if (!maskSize || !sourceSize) return resolution;
  if (maskSize.width !== sourceSize.width || maskSize.height !== sourceSize.height) {
    const selected = {
      ...resolution.selected,
      alignment: "dimension_mismatch" as const,
      reason: "reference dimensions do not match mask",
      warning: "Reference dimensions do not match mask. Overlay disabled.",
    };
    return {
      ...resolution,
      selected,
      alignment: selected.alignment,
      reason: selected.reason,
      warning: selected.warning,
    };
  }
  return resolution;
}

export function canOverlayBinaryMaskReference(resolution: MaskReferenceResolution): boolean {
  return Boolean(
    resolution.selected
    && resolution.sourceArtifact
    && resolution.sourceUrl
    && (resolution.alignment === "exact" || resolution.alignment === "same_dimensions" || resolution.alignment === "assumed_stage_context")
  );
}

export function defaultBinaryMaskViewMode(
  artifactOrInput: StudioArtifact | MaskReferenceResolverInput,
  artifacts: StudioArtifact[] = [],
  resolution?: MaskReferenceResolution,
): BinaryMaskViewMode {
  const current = resolution ?? resolveBinaryMaskReference(artifactOrInput, artifacts);
  if (!canOverlayBinaryMaskReference(current)) return "mask";
  const artifact = "maskArtifact" in artifactOrInput ? artifactOrInput.maskArtifact : artifactOrInput;
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

export function availableBinaryMaskViewModes(
  artifactOrInput: StudioArtifact | MaskReferenceResolverInput,
  artifacts: StudioArtifact[] = [],
  resolution?: MaskReferenceResolution,
): BinaryMaskViewMode[] {
  const artifact = "maskArtifact" in artifactOrInput ? artifactOrInput.maskArtifact : artifactOrInput;
  if (!isBinaryMaskArtifact(artifact)) return [];
  const current = resolution ?? resolveBinaryMaskReference(artifactOrInput, artifacts);
  if (!current.sourceArtifact || !current.sourceUrl) return ["mask"];
  if (current.alignment === "dimension_mismatch") return ["mask", "side_by_side"];
  if (canOverlayBinaryMaskReference(current)) return ["mask", "overlay", "boundary", "side_by_side"];
  return ["mask"];
}

export function binaryMaskAlignmentLabel(alignment: MaskReferenceAlignment): string {
  if (alignment === "exact") return "same pixel grid";
  if (alignment === "same_dimensions") return "same dimensions";
  if (alignment === "assumed_stage_context") return "inferred from stage context";
  if (alignment === "dimension_mismatch") return "dimension mismatch";
  return "unavailable";
}

export function shouldUseBinaryMaskRenderer(
  artifact: StudioArtifact,
  artifacts: StudioArtifact[],
): boolean {
  return isBinaryMaskArtifact(artifact) && availableBinaryMaskViewModes(artifact, artifacts).length > 0;
}
