import type { StudioArtifact } from "../../studioWorkspaceModel";

export type ArtifactLike = StudioArtifact;

export type MaskRenderMode =
  | "mask"
  | "overlay"
  | "boundary"
  | "side_by_side";

export type MaskFitMode =
  | "fit_all"
  | "fit_width"
  | "fit_height"
  | "actual_size";

export type MaskReferenceRole =
  | "source"
  | "raw"
  | "normalized"
  | "diagnostic"
  | "computed_from";

export type MaskReferenceCandidate = {
  id: string;
  label: string;
  artifact?: ArtifactLike;
  url?: string;
  role?: MaskReferenceRole;
  priority?: number;
  computedFromArtifactId?: string;
  inferred?: boolean;
  sourceLabel?: string;
};

export type MaskArtifactDiagnostics = {
  selectedReferenceLabel?: string;
  selectedReferenceId?: string;
  selectedReferenceSource?: "user" | "inferred" | "default";
  computedFromLabel?: string;
  computedFromArtifactId?: string;
  coordinateSpace?: string;
  dimensions?: string;
  artifactId?: string;
  sourceArtifactId?: string;
  note?: string;
};

export type MaskRenderedAssets = {
  maskOnlyUrl: string;
  overlayUrl: string;
  boundaryUrl: string;
  hasInvalidData: boolean;
};
