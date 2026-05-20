import type { StudioArtifact } from "./studioWorkspaceModel";

export type OverlayDisplayMode = "all" | "selected_only" | "hide";

export type OverlayDebugInfo = {
  overlayId: string;
  targetArtifactId: string | null;
  targetTitle: string | null;
  coordinateSpace: string | null;
  renderable: boolean;
  approximate: boolean;
  warnings: string[];
  transformedGeometry: Record<string, unknown> | null;
  projectionType: string | null;
  projectionTransformId: string | null;
};

type ImageDimensions = { width: number; height: number };

export function resolveOverlayTarget(
  artifacts: StudioArtifact[],
  selectedArtifact: StudioArtifact | null
): { target: StudioArtifact | null; overlays: StudioArtifact[]; warning: string | null } {
  if (!selectedArtifact) return { target: null, overlays: [], warning: null };
  if (selectedArtifact.kind === "overlay") {
    const targetId = selectedArtifact.target_artifact_id ?? null;
    const target = targetId ? artifacts.find((artifact) => artifact.artifact_id === targetId && artifact.kind === "image") ?? null : null;
    if (!target) {
      const projection = artifacts.find((artifact) => artifact.kind === "image" && Boolean(artifact.projection_type)) ?? null;
      if (projection) {
        return {
          target: projection,
          overlays: overlaysForTarget(artifacts, projection.artifact_id),
          warning: "Compatibility mode: overlay target missing; projected against first available projection artifact.",
        };
      }
      return { target: null, overlays: [], warning: "No renderable target artifact available." };
    }
    return { target, overlays: overlaysForTarget(artifacts, target.artifact_id), warning: null };
  }
  if (selectedArtifact.kind === "image") {
    return {
      target: selectedArtifact,
      overlays: overlaysForTarget(artifacts, selectedArtifact.artifact_id),
      warning: null,
    };
  }
  return { target: null, overlays: [], warning: "No renderable target artifact available." };
}

export function overlaysForTarget(artifacts: StudioArtifact[], targetArtifactId: string): StudioArtifact[] {
  return artifacts.filter((artifact) => artifact.kind === "overlay" && artifact.target_artifact_id === targetArtifactId);
}

export function visibleOverlays(
  overlays: StudioArtifact[],
  mode: OverlayDisplayMode,
  selectedObjectId: number | null
): StudioArtifact[] {
  if (mode === "hide") return [];
  if (mode === "selected_only" && selectedObjectId != null) {
    return overlays.filter((overlay) => overlay.object_id === selectedObjectId);
  }
  return overlays;
}

export function overlayDebugInfo(
  overlay: StudioArtifact,
  artifacts: StudioArtifact[],
  dims: ImageDimensions
): OverlayDebugInfo {
  const target = overlay.target_artifact_id
    ? artifacts.find((artifact) => artifact.artifact_id === overlay.target_artifact_id) ?? null
    : null;
  const transform = transformOverlayGeometry(overlay, dims);
  const warnings = [...(overlay.overlay_warnings ?? []), ...transform.warnings];
  return {
    overlayId: overlay.artifact_id,
    targetArtifactId: overlay.target_artifact_id ?? null,
    targetTitle: target?.title ?? null,
    coordinateSpace: overlay.coordinate_space ?? null,
    renderable: transform.renderable,
    approximate: Boolean(overlay.approximate || transform.approximate),
    warnings,
    transformedGeometry: transform.transformedGeometry,
    projectionType: target?.projection_type ?? null,
    projectionTransformId: target?.projection_transform_id ?? null,
  };
}

export function transformOverlayGeometry(
  overlay: StudioArtifact,
  dims: ImageDimensions
): {
  renderable: boolean;
  approximate: boolean;
  warnings: string[];
  transformedGeometry: Record<string, unknown> | null;
} {
  const geometry = overlay.geometry ?? {};
  const space = overlay.coordinate_space ?? "image_pixel";
  if (space === "world_mm") {
    return {
      renderable: false,
      approximate: true,
      warnings: [
        "Overlay is approximate: world coordinates projected onto static debug image.",
      ],
      transformedGeometry: null,
    };
  }
  if (space === "point_cloud_projection") {
    return {
      renderable: false,
      approximate: false,
      warnings: ["No projection transform is available for point cloud overlay rendering."],
      transformedGeometry: null,
    };
  }
  if (space === "projection_pixel") {
    return { renderable: true, approximate: Boolean(overlay.approximate), warnings: [], transformedGeometry: geometry };
  }
  const scaleX = overlayPlotScale(overlay, "x", dims.width);
  const scaleY = overlayPlotScale(overlay, "y", dims.height);
  const warnings: string[] = [];
  const approximate = Boolean(overlay.approximate || space === "plot_pixel");
  if (space === "plot_pixel") {
    warnings.push("Compatibility mode: plot_pixel overlay scaled onto target image.");
  }
  const transformed: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(geometry)) {
    if (isPointArray(value)) {
      transformed[key] = value.map((point) => [point[0] * scaleX, point[1] * scaleY]);
      continue;
    }
    if (typeof value === "number") {
      const lower = key.toLowerCase();
      if (lower.includes("rotation") || lower.includes("angle") || lower === "theta") {
        transformed[key] = value;
      } else if (lower.endsWith("_y") || lower === "y" || lower === "cy" || lower === "ry" || lower === "height") {
        transformed[key] = value * scaleY;
      } else if (
        lower.endsWith("_x") ||
        lower === "x" ||
        lower === "cx" ||
        lower === "rx" ||
        lower === "width" ||
        lower === "anchor_x" ||
        lower === "anchor_y"
      ) {
        transformed[key] = value * (lower === "anchor_y" ? scaleY : scaleX);
      } else {
        transformed[key] = value * scaleX;
      }
      continue;
    }
    transformed[key] = value;
  }
  if (space === "normalized_image") {
    transformed["__space"] = "image_pixel_from_normalized";
    return { renderable: true, approximate: approximate, warnings, transformedGeometry: transformed };
  }
  return { renderable: true, approximate, warnings, transformedGeometry: transformed };
}

function overlayPlotScale(
  overlay: StudioArtifact,
  axis: "x" | "y",
  fallback: number
): number {
  const space = overlay.coordinate_space ?? "image_pixel";
  if (space === "normalized_image") return axis === "x" ? fallback : fallback;
  if (space === "plot_pixel") {
    const from = Number((overlay.metadata?.[`plot_${axis === "x" ? "width" : "height"}`] as number | undefined) ?? fallback);
    if (!Number.isFinite(from) || from <= 0) return 1;
    return fallback / from;
  }
  return 1;
}

function isPointArray(value: unknown): value is number[][] {
  return Array.isArray(value) && value.every((point) => Array.isArray(point) && point.length >= 2 && typeof point[0] === "number" && typeof point[1] === "number");
}
