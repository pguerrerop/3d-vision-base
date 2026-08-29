import type { BinaryMaskViewMode } from "./binaryMaskArtifacts";
import type { OverlayDisplayMode } from "./overlayModel";
import type { StudioArtifact } from "./studioWorkspaceModel";

export type ViewportMode = "fit_all" | "fit_width" | "fit_height" | "natural";

export type StudioArtifactViewState = {
  viewportMode: ViewportMode;
  overlayMode: OverlayDisplayMode;
  binaryMaskMode: BinaryMaskViewMode;
  binaryMaskOpacity: number;
  binaryMaskReferenceOpacity: number;
  binaryMaskReferenceArtifactId: string | null;
};

export function preferredViewportModeForArtifact(artifact: StudioArtifact | null): ViewportMode {
  if (!artifact) return "fit_all";
  const metadata = (artifact.metadata ?? {}) as Record<string, unknown>;
  const shape = Array.isArray(metadata.shape) ? metadata.shape as unknown[] : null;
  const height = toFinite(shape?.[0] ?? metadata.height ?? metadata.source_height ?? metadata.image_height);
  const width = toFinite(shape?.[1] ?? metadata.width ?? metadata.source_width ?? metadata.image_width);
  if (width != null && height != null && width > 0 && height > 0) {
    const aspect = width / height;
    if (aspect < 0.9) return "fit_height";
    if (aspect > 1.6) return "fit_width";
  }
  return "fit_all";
}

function toFinite(value: unknown): number | null {
  const n = typeof value === "number" ? value : (typeof value === "string" ? Number(value) : NaN);
  return Number.isFinite(n) ? n : null;
}
