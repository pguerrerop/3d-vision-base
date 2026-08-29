import type { TakeDetail } from "../../api/client";
import type { StudioArtifact } from "../studioWorkspaceModel";

export type ResolvedStageViewContext = {
  sourceArtifacts: StudioArtifact[];
  runArtifacts: StudioArtifact[];
  resolvedArtifacts: StudioArtifact[];
  usingSourceFallback: boolean;
};

function nestedRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" ? value as Record<string, unknown> : null;
}

function normalizeSourceSemanticField(source: Record<string, unknown>): string | undefined {
  const raw = String(source.semantic_field ?? source.semantic ?? source.role ?? source.kind ?? "").toLowerCase();
  if (raw.includes("height")) return "raw_sensor_z";
  return undefined;
}

function sourceImageArtifacts(detail: TakeDetail, stageId: string): StudioArtifact[] {
  const groups = detail.assets ?? {};
  const files = typeof detail.metadata?.files === "object" && detail.metadata.files ? detail.metadata.files as Record<string, unknown> : {};
  const items: StudioArtifact[] = [];
  const pushNamed = (artifactId: string, title: string, filename: string, metadata: Record<string, unknown>) => {
    items.push({
      artifact_id: artifactId,
      stage_id: stageId,
      kind: "image",
      title,
      path: filename,
      preview_available: true,
      metadata,
      generated: "derived",
    } as StudioArtifact);
  };
  const pushExternal = (artifactId: string, title: string, filename: string, metadata: Record<string, unknown>) => {
    items.push({
      artifact_id: artifactId,
      stage_id: stageId,
      kind: "image",
      title,
      path: filename,
      preview_available: true,
      metadata: {
        external_source_url: filename,
        ...metadata,
      },
      generated: "derived",
    } as StudioArtifact);
  };
  const addSourceCandidate = (artifactId: string, title: string, value: unknown, metadata: Record<string, unknown>) => {
    if (typeof value !== "string" || !value.trim()) return;
    if (/^https?:\/\//i.test(value)) pushExternal(artifactId, title, value, metadata);
    else pushNamed(artifactId, title, value, metadata);
  };
  const sourceMeta = (detail.metadata?.source && typeof detail.metadata.source === "object")
    ? detail.metadata.source as Record<string, unknown>
    : null;
  const sourceRange = (sourceMeta?.height_preview_range_raw && typeof sourceMeta.height_preview_range_raw === "object")
    ? sourceMeta.height_preview_range_raw as Record<string, unknown>
    : null;
  const sourceRangeMetadata = sourceRange ? {
    min_mm: Number(sourceRange.min ?? 0),
    max_mm: Number(sourceRange.max ?? 0),
    units: String(sourceRange.units ?? "sensor_raw"),
  } : {};
  const heightPreview = files.heightmap_preview;
  const parserMetadataFile = typeof files.parser_metadata === "string" ? files.parser_metadata : null;
  if (typeof heightPreview === "string" && heightPreview) {
    pushNamed(
      "source_heightmap_preview",
      "Height preview",
      heightPreview,
      {
        source_binding: true,
        source_group: "heightmap",
        source_key: "heightmap_preview",
        visualization_role: "primary",
        parser_metadata_file: parserMetadataFile,
        semantic_field: "raw_sensor_z",
        is_measurement_authoritative: false,
        source_preview_lut_unknown: true,
        ...sourceRangeMetadata,
      }
    );
  }
  const pick = (groupKey: string, keys: string[]) => {
    const group = groups[groupKey] ?? {};
    for (const key of keys) {
      const filename = group[key];
      if (!filename) continue;
      const meta: Record<string, unknown> = {
        source_binding: true,
        source_group: groupKey,
        source_key: key,
        visualization_role: groupKey === "heightmap" ? "primary" : "secondary",
      };
      if (groupKey === "heightmap") {
        meta.semantic_field = "raw_sensor_z";
        meta.source_preview_lut_unknown = true;
        Object.assign(meta, sourceRangeMetadata);
      }
      pushNamed(
        `source_${groupKey}_${key}`,
        `${groupKey.toUpperCase()} source`,
        filename,
        meta,
      );
      return;
    }
  };
  pick("heightmap", ["heightmap", "height"]);
  pick("reflectance", ["reflectance"]);
  pick("rgb", ["rgb"]);
  pick("laser_rgb", ["laser_rgb", "laser_overlay", "laser_image"]);

  addSourceCandidate("raw_heightmap", "Raw heightmap", groups.heightmap?.raw_heightmap, {
    source_binding: true,
    source_group: "heightmap",
    source_key: "raw_heightmap",
    visualization_role: "primary",
    semantic_field: "raw_sensor_z",
    source_preview_lut_unknown: true,
    ...sourceRangeMetadata,
  });
  addSourceCandidate("heightmap_input_preview", "Heightmap input preview", groups.heightmap?.preview, {
    source_binding: true,
    source_group: "heightmap",
    source_key: "preview",
    visualization_role: "primary",
    semantic_field: "raw_sensor_z",
    source_preview_lut_unknown: true,
    ...sourceRangeMetadata,
  });

  const replayManifest = nestedRecord(detail.metadata?.replay_manifest)
    ?? nestedRecord(detail.take_metadata?.replay_manifest)
    ?? nestedRecord(sourceMeta?.replay_manifest);
  const replayAssets = nestedRecord(replayManifest?.assets);
  addSourceCandidate("replay_heightmap_preview", "Replay heightmap preview", replayAssets?.heightmap ?? replayAssets?.preview, {
    source_binding: true,
    source_group: "replay_manifest",
    source_key: replayAssets?.heightmap ? "heightmap" : "preview",
    visualization_role: "primary",
    semantic_field: "raw_sensor_z",
    source_preview_lut_unknown: true,
    ...sourceRangeMetadata,
  });
  addSourceCandidate("take_thumbnail", "Take thumbnail", nestedRecord(detail.metadata)?.thumbnail_url ?? nestedRecord(detail.take_metadata)?.thumbnail_url, {
    source_binding: true,
    source_group: "take",
    source_key: "thumbnail_url",
    visualization_role: "fallback",
  });
  addSourceCandidate("take_preview", "Take preview", nestedRecord(detail.metadata)?.preview_url ?? nestedRecord(detail.take_metadata)?.preview_url, {
    source_binding: true,
    source_group: "take",
    source_key: "preview_url",
    visualization_role: "fallback",
  });

  const detailLoose = detail as unknown as Record<string, unknown>;
  const sourceArtifacts = Array.isArray(detailLoose.source_artifacts)
    ? detailLoose.source_artifacts as unknown[]
    : Array.isArray(detailLoose.sourceArtifacts)
      ? detailLoose.sourceArtifacts as unknown[]
      : Array.isArray(nestedRecord(detail.metadata)?.source_artifacts)
        ? nestedRecord(detail.metadata)?.source_artifacts as unknown[]
        : [];
  for (const item of sourceArtifacts) {
    const row = nestedRecord(item);
    if (!row) continue;
    const path = row.path ?? row.url ?? row.preview_url;
    if (typeof path !== "string" || !path.trim()) continue;
    const artifactId = String(row.artifact_id ?? row.id ?? row.key ?? row.name ?? "source_artifact").trim();
    const title = String(row.title ?? row.label ?? row.name ?? artifactId).trim() || "Source artifact";
    addSourceCandidate(artifactId, title, path, {
      source_binding: true,
      source_group: "source_artifacts",
      source_key: artifactId,
      visualization_role: "secondary",
      semantic_field: normalizeSourceSemanticField(row),
    });
  }

  return items;
}

function sourceMetadataArtifacts(detail: TakeDetail, stageId: string): StudioArtifact[] {
  return [
    {
      artifact_id: "source_metadata",
      stage_id: stageId,
      kind: "table",
      title: "Take metadata",
      preview_available: true,
      metadata: {
        source_binding: true,
        modalities: detail.modalities,
        created_at: detail.created_at,
        metadata: detail.metadata,
        frameset: detail.frameset,
      },
      generated: "derived",
    } as StudioArtifact,
    {
      artifact_id: "source_json",
      stage_id: stageId,
      kind: "json",
      title: "Source JSON",
      preview_available: true,
      metadata: {
        source_binding: true,
        take: {
          take_id: detail.take_id,
          modalities: detail.modalities,
          metadata: detail.metadata,
          assets: detail.assets,
          frameset: detail.frameset,
        },
      },
      generated: "derived",
    } as StudioArtifact,
  ];
}

export function sourceArtifactsForTake(detail: TakeDetail | null, stageId = "input"): StudioArtifact[] {
  if (!detail) return [];
  return [
    ...sourceImageArtifacts(detail, stageId),
    ...sourceMetadataArtifacts(detail, stageId),
  ];
}

export function primarySourceImageUrl(detail: TakeDetail | null): string | null {
  if (!detail) return null;
  const artifacts = sourceImageArtifacts(detail, "input");
  const first = artifacts[0];
  if (!first?.path) return null;
  const external = typeof first.metadata?.external_source_url === "string" ? first.metadata.external_source_url : null;
  return external ?? sourceFileUrl(detail.take_id, first.path);
}

export function resolveStageViewContext(
  detail: TakeDetail | null,
  stageId: string,
  viewId: string,
  category: "input" | "plane_qa" | "segmentation" | "geometry" | "measurement" | "classification" | "fusion",
  runArtifacts: StudioArtifact[]
): ResolvedStageViewContext {
  if (!detail) return { sourceArtifacts: [], runArtifacts: [], resolvedArtifacts: [], usingSourceFallback: false };
  if (category !== "input" && category !== "plane_qa") {
    return { sourceArtifacts: [], runArtifacts, resolvedArtifacts: runArtifacts, usingSourceFallback: false };
  }

  const sourceArtifacts = sourceArtifactsForTake(detail, stageId);

  const sourceByView = (() => {
    if (viewId === "image" || viewId === "height_preview" || viewId === "heightmap") return sourceArtifacts.filter((item) => item.kind === "image");
    if (viewId === "metadata") return sourceArtifacts.filter((item) => item.artifact_id === "source_metadata");
    if (viewId === "json") return sourceArtifacts.filter((item) => item.artifact_id === "source_json");
    if (viewId === "histogram") return sourceArtifacts.filter((item) => item.kind === "image");
    return sourceArtifacts;
  })();

  const runPreferred = (() => {
    if (viewId === "image") return runArtifacts.filter((item) => item.kind === "image");
    if (viewId === "json") return runArtifacts.filter((item) => item.kind === "json");
    return runArtifacts;
  })();

  const resolvedArtifacts = runPreferred.length ? runPreferred : sourceByView;
  return {
    sourceArtifacts,
    runArtifacts,
    resolvedArtifacts,
    usingSourceFallback: !runPreferred.length && sourceByView.length > 0,
  };
}

function sourceFileUrl(takeId: string, relativePath: string): string {
  const encodedTake = encodeURIComponent(takeId);
  const encodedPath = relativePath
    .split("/")
    .map((segment) => encodeURIComponent(segment))
    .join("/");
  return `/api/takes/${encodedTake}/files/${encodedPath}`;
}
