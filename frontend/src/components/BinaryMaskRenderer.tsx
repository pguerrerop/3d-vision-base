import { useEffect, useMemo } from "react";
import { fileUrl } from "../api/client";
import type { TakeDetail } from "../api/client";
import MaskArtifactViewer from "./artifacts/mask/MaskArtifactViewer";
import { useMaskRenderAssets, validateMaskResolutionForLoadedImages } from "./artifacts/mask/maskRenderAssets";
import { buildMaskArtifactDiagnostics } from "./artifacts/mask/maskReferenceResolution";
import type { MaskReferenceCandidate as ViewerMaskReferenceCandidate } from "./artifacts/mask/maskRenderingTypes";
import type { StudioArtifact } from "./studioWorkspaceModel";
import {
  availableBinaryMaskViewModes,
  binaryMaskAlignmentLabel,
  finiteOrDefault,
  resolveBinaryMaskInvalidArtifact,
  resolveBinaryMaskReference,
  resolveBinaryMaskReferenceCandidates,
  type BinaryMaskViewMode,
} from "./binaryMaskArtifacts";
import { sourceArtifactsForTake } from "./stage_sources";

type Props = {
  artifact: StudioArtifact;
  artifacts: StudioArtifact[];
  takeId: string;
  cacheKey: string;
  selectedTake?: TakeDetail | null;
  currentStageArtifacts?: StudioArtifact[];
  mode: BinaryMaskViewMode;
  onModeChange: (mode: BinaryMaskViewMode) => void;
  overlayOpacity: number;
  referenceOpacity: number;
  selectedSourceArtifactId: string | null;
  onSelectedSourceArtifactIdChange: (artifactId: string | null) => void;
  onOverlayOpacityChange?: (value: number) => void;
  onReferenceOpacityChange?: (value: number) => void;
  fitMode?: "fit_all" | "fit_width" | "fit_height" | "natural";
  onFitModeChange?: (mode: "fit_all" | "fit_width" | "fit_height" | "natural") => void;
  onResetView?: () => void;
  openUrl?: string | null;
};

export default function BinaryMaskRenderer({
  artifact,
  artifacts,
  takeId,
  cacheKey,
  selectedTake = null,
  currentStageArtifacts = [],
  mode,
  onModeChange,
  overlayOpacity,
  referenceOpacity,
  selectedSourceArtifactId,
  onSelectedSourceArtifactIdChange,
  onOverlayOpacityChange,
  onReferenceOpacityChange,
  fitMode = "fit_all",
  onFitModeChange,
  onResetView,
  openUrl = null,
}: Props) {
  const sourceArtifacts = useMemo(
    () => sourceArtifactsForTake(selectedTake, artifact.stage_id || "input"),
    [selectedTake, artifact.stage_id],
  );
  const resolverInput = useMemo(
    () => ({
      maskArtifact: artifact,
      selectedStageId: artifact.stage_id,
      selectedTake,
      currentStageArtifacts,
      allRunArtifacts: artifacts,
      sourceArtifacts,
      resolvedArtifacts: [...currentStageArtifacts, ...sourceArtifacts],
    }),
    [artifact, selectedTake, currentStageArtifacts, artifacts, sourceArtifacts],
  );
  const sourceCandidates = useMemo(
    () => resolveBinaryMaskReferenceCandidates(resolverInput),
    [resolverInput],
  );

  const maskUrl = useMemo(
    () => (artifact.path ? `${fileUrl(takeId, artifact.path)}?t=${encodeURIComponent(cacheKey)}` : null),
    [artifact.path, takeId, cacheKey],
  );
  const invalidArtifact = useMemo(
    () => resolveBinaryMaskInvalidArtifact(artifact, artifacts),
    [artifact.artifact_id, artifacts],
  );
  const invalidUrl = useMemo(
    () => (invalidArtifact?.path ? `${fileUrl(takeId, invalidArtifact.path)}?t=${encodeURIComponent(cacheKey)}` : null),
    [invalidArtifact?.path, takeId, cacheKey],
  );

  useEffect(() => {
    const nextSourceArtifactId = sourceCandidates[0]?.artifact.artifact_id ?? null;
    if (selectedSourceArtifactId && sourceCandidates.some((candidate) => candidate.artifact.artifact_id === selectedSourceArtifactId)) return;
    if (selectedSourceArtifactId === nextSourceArtifactId) return;
    onSelectedSourceArtifactIdChange(nextSourceArtifactId);
  }, [onSelectedSourceArtifactIdChange, selectedSourceArtifactId, sourceCandidates]);

  const selectedCandidate = useMemo(
    () => sourceCandidates.find((candidate) => candidate.artifact.artifact_id === selectedSourceArtifactId) ?? sourceCandidates[0] ?? null,
    [sourceCandidates, selectedSourceArtifactId],
  );
  const sourceUrl = useMemo(
    () => selectedCandidate?.sourceUrl ?? (selectedCandidate?.artifact.path ? `${fileUrl(takeId, selectedCandidate.artifact.path)}?t=${encodeURIComponent(cacheKey)}` : null),
    [selectedCandidate?.sourceUrl, selectedCandidate?.artifact.path, takeId, cacheKey],
  );

  const validatedResolution = useMemo(
    () => resolveBinaryMaskReference(resolverInput, selectedCandidate?.artifact.artifact_id ?? null, sourceUrl),
    [resolverInput, selectedCandidate?.artifact.artifact_id, sourceUrl],
  );
  const { rendered, error, loadState } = useMaskRenderAssets({ maskUrl, sourceUrl, invalidUrl });
  const safeResolution = useMemo(() => {
    const next = validateMaskResolutionForLoadedImages(validatedResolution, loadState);
    return loadState.sourceFailed && next.sourceArtifact
      ? {
          ...next,
          alignment: "unavailable" as const,
          reason: "reference artifact found but image URL is unavailable",
          warning: "Reference artifact found but image URL is unavailable.",
          sourceUrl: undefined,
        }
      : next;
  }, [validatedResolution, loadState]);
  const availableModes = useMemo(
    () => availableBinaryMaskViewModes(artifact, artifacts, safeResolution),
    [artifact.artifact_id, artifacts, safeResolution],
  );
  const computedFrom = safeResolution.computedFrom;
  const diagnostics = useMemo(
    () => buildMaskArtifactDiagnostics({
      maskArtifact: artifact,
      resolution: safeResolution,
      selectedReferenceId: selectedCandidate?.artifact.artifact_id ?? selectedSourceArtifactId,
      defaultReferenceId: sourceCandidates[0]?.artifact.artifact_id ?? null,
    }),
    [artifact, safeResolution, selectedCandidate?.artifact.artifact_id, selectedSourceArtifactId, sourceCandidates],
  );
  const viewerCandidates = useMemo<ViewerMaskReferenceCandidate[]>(
    () => sourceCandidates.map((candidate) => {
      const role: ViewerMaskReferenceCandidate["role"] = candidate.role === "lineage_source"
        ? "computed_from"
        : candidate.role === "normalized_height"
          ? "normalized"
          : candidate.role === "original_acquisition" || candidate.role === "raw_heightmap"
            ? "raw"
            : candidate.role === "derived_diagnostic" || candidate.role === "overlay"
              ? "diagnostic"
              : "source";
      return {
        id: candidate.artifact.artifact_id,
        label: candidate.label,
        artifact: candidate.artifact,
        url: candidate.sourceUrl,
        role,
        priority: candidate.priority,
        computedFromArtifactId: computedFrom?.artifact.artifact_id,
        inferred: candidate.inferred,
      } as ViewerMaskReferenceCandidate;
    }),
    [computedFrom?.artifact.artifact_id, sourceCandidates],
  );

  useEffect(() => {
    if (availableModes.includes(mode)) return;
    const nextMode = availableModes[0] ?? "mask";
    if (mode === nextMode) return;
    onModeChange(nextMode);
  }, [availableModes, mode, onModeChange]);

  if (!artifact.path) {
    return <div className="empty-state">Artifact file is unavailable.</div>;
  }

  const overlayOpacitySafe = finiteOrDefault(overlayOpacity, 0.52);
  const referenceOpacitySafe = finiteOrDefault(referenceOpacity, 1);
  const maskFitMode = fitMode === "natural" ? "actual_size" : fitMode;

  return (
    <MaskArtifactViewer
      availableModes={availableModes}
      defaultReferenceId={sourceCandidates[0]?.artifact.artifact_id ?? null}
      diagnostics={{
        ...diagnostics,
        note: safeResolution.warning ?? `Alignment: ${binaryMaskAlignmentLabel(safeResolution.alignment)}`,
      }}
      emptyMessage="Rendering mask preview..."
      errorMessage={error}
      fitMode={maskFitMode}
      maskArtifact={artifact}
      maskOpacity={overlayOpacitySafe}
      mode={mode}
      onFitModeChange={(next) => onFitModeChange?.(next === "actual_size" ? "natural" : next)}
      onMaskOpacityChange={onOverlayOpacityChange}
      onModeChange={onModeChange}
      onResetView={onResetView}
      onReferenceChange={(referenceId) => onSelectedSourceArtifactIdChange(referenceId || null)}
      onReferenceOpacityChange={onReferenceOpacityChange}
      openUrl={openUrl}
      referenceCandidates={viewerCandidates}
      referenceOpacity={referenceOpacitySafe}
      rendered={rendered}
      selectedReferenceId={selectedCandidate?.artifact.artifact_id ?? selectedSourceArtifactId}
      selectedReferenceUrl={safeResolution.sourceUrl ?? null}
      showDiagnostics
      showLegend
    />
  );
}
