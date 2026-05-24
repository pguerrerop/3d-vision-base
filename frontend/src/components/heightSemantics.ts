import type { StudioArtifact } from "./studioWorkspaceModel";

export type HeightRepresentation =
  | "numeric_raster"
  | "preview_image"
  | "debug_overlay"
  | "histogram"
  | "mask"
  | "residual_map";

// Canonical color mapping contract shared by:
//   - image renderer (backend PNG + frontend overlays / scalar->RGB previews)
//   - HeightLegend (colorbar + min/max + direction labels)
//   - hover / debug reconstruction (scalar->RGB diffing)
//   - exported preview metadata (`color_mapping` block written by backend)
// Direction is data-native: the colormap is always anchored at valueMin (cool)
// to valueMax (hot). `direction` only affects how labels/ticks are rendered.
export type HeightColorMap = "turbo" | "viridis" | "magma" | "gray" | string;

export type HeightColorMappingSource =
  | "artifact_metadata"
  | "render_context"
  | "legacy_fallback";

export type HeightColorMapping = {
  semanticField: string;
  units?: string;
  valueMin: number;
  valueMax: number;
  colorScaleMin: number;
  colorScaleMax: number;
  colorMap: HeightColorMap;
  direction: "higher_is_hotter" | "lower_is_hotter";
  clamp: boolean;
  source: HeightColorMappingSource;
};

export type ResolveHeightColorMappingOptions = {
  semanticField?: string | null;
  preferredArtifactId?: string | null;
  fallbackRange?: { min: number; max: number } | null;
};

export type ResolveHeightColorMappingResult = {
  mapping: HeightColorMapping;
  warning: string | null;
};

export type HeightSemanticMatch = {
  ok: boolean;
  warning: string | null;
};

export type ResolveHeightArtifactQuery = {
  semanticField: string;
  representation: HeightRepresentation;
};

export type ResolveHeightArtifactResult = {
  artifact: StudioArtifact | null;
  warning: string | null;
};

type HeightLegendSemantic = {
  semanticField: string;
  semanticLabel?: string;
  units?: string;
  minValue?: number | null;
  maxValue?: number | null;
  colorMap?: string;
  positiveDirection?: string;
  authoritative?: boolean;
  displayOnly?: boolean;
  percentileLabels?: { p2?: number | null; p98?: number | null };
};

export function toHeightLegendSemantic(metadata: Record<string, unknown> | null | undefined): HeightLegendSemantic | null {
  if (!metadata) return null;
  const semanticField = asText(metadata.semantic_field ?? metadata.display_semantic);
  if (!semanticField) return null;
  return {
    semanticField,
    semanticLabel: asText(metadata.semantic_label ?? metadata.display_semantic_label),
    units: asText(metadata.units ?? metadata.display_units),
    minValue: asNumber(metadata.value_min ?? metadata.min_mm ?? metadata.raw_min ?? metadata.display_vmin),
    maxValue: asNumber(metadata.value_max ?? metadata.max_mm ?? metadata.raw_max ?? metadata.display_vmax),
    colorMap: asText(metadata.color_map ?? metadata.colormap),
    positiveDirection: asText(metadata.positive_direction),
    authoritative: asBoolean(metadata.is_measurement_authoritative),
    displayOnly: asBoolean(metadata.display_only),
    percentileLabels: {
      p2: asNumber(metadata.p2),
      p98: asNumber(metadata.p98),
    },
  };
}

export function assertSemanticMatch(viewSemanticField: string | null | undefined, hoverSemanticField: string | null | undefined): HeightSemanticMatch {
  const view = asText(viewSemanticField);
  const hover = asText(hoverSemanticField);
  if (!view || !hover) {
    return { ok: true, warning: null };
  }
  if (view === hover) {
    return { ok: true, warning: null };
  }
  return {
    ok: false,
    warning: `\u26a0 Semantic mismatch: view=${view} hover=${hover}`,
  };
}

export function resolveHeightArtifact(artifacts: StudioArtifact[], query: ResolveHeightArtifactQuery): ResolveHeightArtifactResult {
  const exact = artifacts.find((artifact) => {
    const metadata = asRecord(artifact.metadata);
    return (
      isHeightRepresentationMatch(artifact, query.representation)
      && asText(metadata.semantic_field) === query.semanticField
    );
  });
  if (exact) return { artifact: exact, warning: null };

  const fallback = artifacts.find((artifact) => isLegacyHeightRepresentationMatch(artifact, query));
  if (!fallback) return { artifact: null, warning: null };
  return {
    artifact: fallback,
    warning: `legacy semantic fallback used for ${query.semanticField}/${query.representation}`,
  };
}

function isHeightRepresentationMatch(artifact: StudioArtifact, representation: HeightRepresentation): boolean {
  const id = artifact.artifact_id.toLowerCase();
  const path = String(artifact.path ?? "").toLowerCase();
  const metadata = asRecord(artifact.metadata);
  if (representation === "numeric_raster") {
    return artifact.kind === "file" || Boolean(metadata.numeric_source) || path.endsWith(".values.f32");
  }
  if (representation === "preview_image") {
    return artifact.kind === "image" && Boolean(metadata.display_only ?? true);
  }
  if (representation === "debug_overlay") {
    return artifact.kind === "overlay" || id.includes("overlay");
  }
  if (representation === "histogram") {
    return id.includes("histogram") || path.includes("histogram");
  }
  if (representation === "mask") {
    return id.includes("mask") || path.includes("mask");
  }
  return id.includes("residual") || path.includes("residual");
}

function isLegacyHeightRepresentationMatch(artifact: StudioArtifact, query: ResolveHeightArtifactQuery): boolean {
  const id = artifact.artifact_id.toLowerCase();
  const path = String(artifact.path ?? "").toLowerCase();
  if (query.representation === "numeric_raster") {
    const legacy = legacyNumericRasterNames(query.semanticField);
    return legacy.some((name) => id.includes(name) || path.includes(name));
  }
  if (query.representation === "preview_image") {
    if (query.semanticField === "height_above_belt") return id.includes("normalized_heightmap");
    if (query.semanticField === "raw_sensor_z") return id.includes("raw_heightmap") || id === "heightmap";
    if (query.semanticField === "plane_signed_distance") return id.includes("residual");
  }
  if (query.representation === "residual_map") {
    return id.includes("residual");
  }
  return false;
}

function legacyNumericRasterNames(semanticField: string): string[] {
  if (semanticField === "height_above_belt") return ["height_above_belt.values.f32", "normalized_heightmap.values.f32"];
  if (semanticField === "raw_sensor_z") return ["raw_sensor_z.values.f32", "raw_heightmap.values.f32"];
  if (semanticField === "plane_signed_distance") return ["plane_signed_distance.values.f32", "plane_residuals.values.f32"];
  return [];
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : {};
}

function asText(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  return trimmed ? trimmed : null;
}

function asNumber(value: unknown): number | null {
  const parsed = typeof value === "number" ? value : (typeof value === "string" ? Number(value) : NaN);
  return Number.isFinite(parsed) ? parsed : null;
}

function asBoolean(value: unknown): boolean | undefined {
  if (typeof value === "boolean") return value;
  return undefined;
}

// Canonical priority chain for resolving the shared HeightColorMapping:
//   1. explicit `color_mapping` block on any artifact metadata (preferred)
//   2. render_context artifact (`render_vmin/vmax`, `colormap_id`, `value_min/max`)
//   3. display metadata artifact (`color_scale_min/max`, `color_map`)
//   4. legacy_fallback (semantic defaults; emits a warning)
export function resolveHeightColorMapping(
  artifacts: StudioArtifact[],
  options: ResolveHeightColorMappingOptions = {},
): ResolveHeightColorMappingResult {
  const semanticHint = asText(options.semanticField) ?? "height_above_belt";
  const explicit = findExplicitColorMappingBlock(artifacts, options.preferredArtifactId);
  if (explicit) return { mapping: explicit, warning: null };

  const fromRenderContext = mappingFromRenderContext(artifacts, semanticHint);
  if (fromRenderContext) return { mapping: fromRenderContext, warning: null };

  const fromDisplay = mappingFromDisplayMetadata(artifacts, semanticHint);
  if (fromDisplay) return { mapping: fromDisplay, warning: null };

  // Source-bound heightmap previews (e.g. raw camera SDK PNG on the Input
  // stage) carry `min_mm`/`max_mm`/`units` extracted from parser metadata but
  // no canonical color_mapping block. We still derive a usable mapping so the
  // legend shows the correct VALUE range; the LUT is unknown though, so we
  // emit a clear warning rather than silently fall back to defaults.
  const fromSource = mappingFromSourceArtifact(artifacts, semanticHint);
  if (fromSource) {
    return {
      mapping: fromSource,
      warning: "source preview LUT unknown; canonical gradient is shown for reference",
    };
  }

  const fallbackRange = options.fallbackRange ?? defaultFallbackRange(semanticHint);
  const mapping: HeightColorMapping = {
    semanticField: semanticHint,
    units: "mm",
    valueMin: fallbackRange.min,
    valueMax: fallbackRange.max,
    colorScaleMin: fallbackRange.min,
    colorScaleMax: fallbackRange.max,
    colorMap: "turbo",
    direction: directionForSemantic(semanticHint),
    clamp: true,
    source: "legacy_fallback",
  };
  return {
    mapping,
    warning: `legacy color mapping fallback used for ${semanticHint}`,
  };
}

function findExplicitColorMappingBlock(
  artifacts: StudioArtifact[],
  preferredArtifactId?: string | null,
): HeightColorMapping | null {
  const sorted = preferredArtifactId
    ? [
        ...artifacts.filter((artifact) => artifact.artifact_id === preferredArtifactId),
        ...artifacts.filter((artifact) => artifact.artifact_id !== preferredArtifactId),
      ]
    : artifacts;
  for (const artifact of sorted) {
    const metadata = asRecord(artifact.metadata);
    const block = asRecord(metadata.color_mapping);
    if (!Object.keys(block).length) continue;
    const valueMin = asNumber(block.value_min);
    const valueMax = asNumber(block.value_max);
    const colorScaleMin = asNumber(block.color_scale_min) ?? valueMin;
    const colorScaleMax = asNumber(block.color_scale_max) ?? valueMax;
    if (valueMin == null || valueMax == null || colorScaleMin == null || colorScaleMax == null) continue;
    if (colorScaleMax <= colorScaleMin) continue;
    return {
      semanticField: asText(block.semantic_field) ?? asText(metadata.semantic_field) ?? "height_above_belt",
      units: asText(block.units) ?? asText(metadata.units) ?? "mm",
      valueMin,
      valueMax,
      colorScaleMin,
      colorScaleMax,
      colorMap: (asText(block.color_map) ?? "turbo") as HeightColorMap,
      direction: parseDirection(block.direction)
        ?? directionForSemantic(asText(block.semantic_field) ?? asText(metadata.semantic_field) ?? "height_above_belt"),
      clamp: asBoolean(block.clamp) ?? true,
      source: parseSource(block.source) ?? "artifact_metadata",
    };
  }
  return null;
}

function mappingFromRenderContext(
  artifacts: StudioArtifact[],
  semanticHint: string,
): HeightColorMapping | null {
  const ctx = artifacts.find((artifact) => artifact.artifact_id === "normalized_heightmap_render_context");
  const metadata = asRecord(ctx?.metadata);
  if (!Object.keys(metadata).length) return null;
  const renderMin = asNumber(metadata.render_vmin);
  const renderMax = asNumber(metadata.render_vmax);
  if (renderMin == null || renderMax == null || renderMax <= renderMin) return null;
  const valueMin = asNumber(metadata.value_min) ?? asNumber(metadata.actual_data_min) ?? renderMin;
  const valueMax = asNumber(metadata.value_max) ?? asNumber(metadata.actual_data_max) ?? renderMax;
  return {
    semanticField: asText(metadata.semantic_field) ?? semanticHint,
    units: asText(metadata.units) ?? "mm",
    valueMin,
    valueMax,
    colorScaleMin: renderMin,
    colorScaleMax: renderMax,
    colorMap: (asText(metadata.colormap_id) ?? asText(metadata.color_map) ?? "turbo") as HeightColorMap,
    direction: parseDirection(metadata.direction) ?? directionForSemantic(asText(metadata.semantic_field) ?? semanticHint),
    clamp: asBoolean(metadata.clamp) ?? true,
    source: "render_context",
  };
}

function mappingFromDisplayMetadata(
  artifacts: StudioArtifact[],
  semanticHint: string,
): HeightColorMapping | null {
  const candidates = [
    artifacts.find((artifact) => artifact.artifact_id === "normalized_heightmap_display"),
    artifacts.find((artifact) => asText(asRecord(artifact.metadata).semantic_field) === semanticHint && asRecord(artifact.metadata).color_scale_min != null),
  ].filter(Boolean) as StudioArtifact[];
  for (const candidate of candidates) {
    const metadata = asRecord(candidate.metadata);
    const colorScaleMin = asNumber(metadata.color_scale_min) ?? asNumber(metadata.display_vmin);
    const colorScaleMax = asNumber(metadata.color_scale_max) ?? asNumber(metadata.display_vmax);
    if (colorScaleMin == null || colorScaleMax == null || colorScaleMax <= colorScaleMin) continue;
    const valueMin = asNumber(metadata.value_min) ?? colorScaleMin;
    const valueMax = asNumber(metadata.value_max) ?? colorScaleMax;
    return {
      semanticField: asText(metadata.semantic_field) ?? semanticHint,
      units: asText(metadata.units) ?? "mm",
      valueMin,
      valueMax,
      colorScaleMin,
      colorScaleMax,
      colorMap: (asText(metadata.color_map) ?? asText(metadata.colormap) ?? "turbo") as HeightColorMap,
      direction: parseDirection(metadata.direction) ?? directionForSemantic(asText(metadata.semantic_field) ?? semanticHint),
      clamp: asBoolean(metadata.clamp) ?? true,
      source: "artifact_metadata",
    };
  }
  return null;
}

function mappingFromSourceArtifact(
  artifacts: StudioArtifact[],
  semanticHint: string,
): HeightColorMapping | null {
  for (const artifact of artifacts) {
    const metadata = asRecord(artifact.metadata);
    if (!metadata.source_binding) continue;
    if (asText(metadata.source_group) !== "heightmap") continue;
    const minMm = asNumber(metadata.min_mm);
    const maxMm = asNumber(metadata.max_mm);
    if (minMm == null || maxMm == null || maxMm <= minMm) continue;
    const semantic = asText(metadata.semantic_field) ?? "raw_sensor_z";
    return {
      semanticField: semantic === semanticHint || semantic === "raw_sensor_z" ? semantic : semantic,
      units: asText(metadata.units) ?? "mm",
      valueMin: minMm,
      valueMax: maxMm,
      colorScaleMin: minMm,
      colorScaleMax: maxMm,
      colorMap: "turbo",
      direction: directionForSemantic(semantic),
      clamp: true,
      // Source artifacts encode the canonical RANGE but not the LUT used by
      // the third-party SDK preview. We surface this via the resolver's
      // warning so the inspector can clearly flag the discrepancy.
      source: "artifact_metadata",
    };
  }
  return null;
}

function defaultFallbackRange(semanticField: string): { min: number; max: number } {
  if (semanticField === "plane_signed_distance") return { min: -5, max: 5 };
  if (semanticField === "raw_sensor_z") return { min: 0, max: 320 };
  return { min: -20, max: 320 };
}

function directionForSemantic(semanticField: string): "higher_is_hotter" | "lower_is_hotter" {
  if (semanticField === "plane_signed_distance") return "higher_is_hotter";
  return "higher_is_hotter";
}

function parseDirection(value: unknown): "higher_is_hotter" | "lower_is_hotter" | null {
  const key = asText(value);
  if (!key) return null;
  if (key === "higher_is_hotter" || key === "lower_is_hotter") return key;
  return null;
}

function parseSource(value: unknown): HeightColorMappingSource | null {
  const key = asText(value);
  if (!key) return null;
  if (key === "artifact_metadata" || key === "render_context" || key === "legacy_fallback") return key;
  return null;
}
