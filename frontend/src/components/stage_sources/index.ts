import type { TakeDetail } from "../../api/client";
import type { StudioArtifact } from "../studioWorkspaceModel";

export type ResolvedStageViewContext = {
  sourceArtifacts: StudioArtifact[];
  runArtifacts: StudioArtifact[];
  resolvedArtifacts: StudioArtifact[];
  usingSourceFallback: boolean;
};

function sourceImageArtifacts(detail: TakeDetail, stageId: string): StudioArtifact[] {
  const groups = detail.assets ?? {};
  const items: StudioArtifact[] = [];
  const pick = (groupKey: string, keys: string[]) => {
    const group = groups[groupKey] ?? {};
    for (const key of keys) {
      const filename = group[key];
      if (!filename) continue;
      items.push({
        artifact_id: `source_${groupKey}_${key}`,
        stage_id: stageId,
        kind: "image",
        title: `${groupKey.toUpperCase()} source`,
        path: filename,
        preview_available: true,
        metadata: { source_binding: true, source_group: groupKey, source_key: key },
        generated: "derived",
      } as StudioArtifact);
      return;
    }
  };
  pick("rgb", ["rgb"]);
  pick("reflectance", ["reflectance"]);
  pick("heightmap", ["heightmap", "height"]);
  pick("laser_rgb", ["laser_rgb", "laser_overlay", "laser_image"]);
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

export function primarySourceImageUrl(detail: TakeDetail | null): string | null {
  if (!detail) return null;
  const artifacts = sourceImageArtifacts(detail, "input");
  const first = artifacts[0];
  return first?.path ? sourceFileUrl(detail.take_id, first.path) : null;
}

export function resolveStageViewContext(
  detail: TakeDetail | null,
  stageId: string,
  viewId: string,
  category: "input" | "segmentation" | "geometry" | "measurement" | "classification" | "fusion",
  runArtifacts: StudioArtifact[]
): ResolvedStageViewContext {
  if (!detail) return { sourceArtifacts: [], runArtifacts: [], resolvedArtifacts: [], usingSourceFallback: false };
  if (category !== "input") {
    return { sourceArtifacts: [], runArtifacts, resolvedArtifacts: runArtifacts, usingSourceFallback: false };
  }

  const sourceArtifacts = [
    ...sourceImageArtifacts(detail, stageId),
    ...sourceMetadataArtifacts(detail, stageId),
  ];

  const sourceByView = (() => {
    if (viewId === "image") return sourceArtifacts.filter((item) => item.kind === "image");
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
