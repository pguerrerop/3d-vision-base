import type { ReactElement } from "react";
import { fileUrl } from "../api/client";
import { useLowGradientComponents } from "../studio/lowGradientComponentModel";
import type { StudioArtifact } from "./studioWorkspaceModel";
import BlobSetViewer from "./BlobSetViewer";

type Props = {
  takeId: string;
  stageArtifacts: StudioArtifact[];
  artifact: StudioArtifact | null;
  selectedCandidateId: number | null;
  hoveredCandidateId: number | null;
  onSelectCandidate: (id: number) => void;
  onHoverCandidate: (id: number | null) => void;
  fallback: ReactElement;
};

// Prefer the plain heightmap preview as the displayed background: the components overlay encodes
// role classification (belt background / stripe / unknown) as flat colors, not per-component
// identity, and layering our own selection fill on top of it just produces color noise. It's still
// a reasonable background fallback if the preview is missing.
function resolveComponentsSourceImage(stageArtifacts: StudioArtifact[]): StudioArtifact | null {
  return stageArtifacts.find((item) => item.artifact_id === "raw_heightmap_preview" && item.path)
    ?? stageArtifacts.find((item) => item.artifact_id === "low_gradient_components_overlay" && item.path)
    ?? null;
}

// The id-mask assigns one integer label per pixel matching each candidate's own component_id/
// blob_id -- unlike the classification overlay, it lets a candidate's true pixel mask be read
// directly rather than guessed at via color sampling.
function resolveComponentsMaskSourceImage(stageArtifacts: StudioArtifact[]): StudioArtifact | null {
  return stageArtifacts.find((item) => item.artifact_id === "low_gradient_blob_id_mask" && item.path) ?? null;
}

export default function LowGradientComponentsViewer({
  takeId,
  stageArtifacts,
  artifact,
  selectedCandidateId,
  hoveredCandidateId,
  onSelectCandidate,
  onHoverCandidate,
  fallback,
}: Props) {
  const { loading, error, components, rejected } = useLowGradientComponents(artifact?.path ? fileUrl(takeId, artifact.path) : null);
  const hasGeometry = [...components, ...rejected].some((row) => row.bbox && row.centroid);
  if (loading || error || !hasGeometry) return fallback;
  const source = resolveComponentsSourceImage(stageArtifacts);
  if (!source?.path) return fallback;
  const maskSource = resolveComponentsMaskSourceImage(stageArtifacts);
  return (
    <BlobSetViewer
      src={fileUrl(takeId, source.path)}
      maskSourceSrc={maskSource?.path ? fileUrl(takeId, maskSource.path) : null}
      candidates={components}
      rejected={rejected}
      selectedCandidateId={selectedCandidateId}
      hoveredCandidateId={hoveredCandidateId}
      onSelectCandidate={onSelectCandidate}
      onHoverCandidate={onHoverCandidate}
    />
  );
}
