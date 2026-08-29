import { useEffect, useMemo, type ReactElement } from "react";
import { fileUrl, type TakeDetail } from "../../api/client";
import {
  availableBinaryMaskViewModes,
  binaryMaskAlignmentLabel,
  finiteOrDefault,
  resolveBinaryMaskInvalidArtifact,
  resolveBinaryMaskReference,
  resolveBinaryMaskReferenceCandidates,
  type MaskReferenceResolverInput,
} from "../binaryMaskArtifacts";
import MaskArtifactViewer from "../artifacts/mask/MaskArtifactViewer";
import { useMaskRenderAssets, validateMaskResolutionForLoadedImages } from "../artifacts/mask/maskRenderAssets";
import { buildMaskArtifactDiagnostics } from "../artifacts/mask/maskReferenceResolution";
import type { MaskReferenceCandidate as ViewerMaskReferenceCandidate } from "../artifacts/mask/maskRenderingTypes";
import { resolveStageViewContext } from "../stage_sources";
import { artifactsForStage } from "../studioWorkspaceModel";
import { stageEmptyState, stageSemanticDefinition } from "../stageSemantics";
import type { RendererProps } from "./index";
import {
  candidateRoleForViewer,
  formatStageMaskDiagnosticsNote,
  rankStageMaskReferenceCandidates,
  resolveStageMaskArtifact,
  resolveStageMaskDefaultMode,
  resolveStageMaskSpec,
  uniqueMaskArtifacts,
} from "./stageMaskViewSpec";

function emptyState(stageId: string, viewId: string, fallback?: string): ReactElement {
  const empty = stageEmptyState(stageId, viewId);
  return <div className="empty-state"><strong>{fallback ?? empty.title}</strong>{empty.description ? <small>{empty.description}</small> : null}</div>;
}

export default function StageMaskArtifactRenderer(props: RendererProps): ReactElement {
  const detail = props.detail;
  const takeId = detail?.take_id ?? "";
  const semantic = stageSemanticDefinition(props.stageId);
  const runArtifacts = artifactsForStage(props.detail, props.stageId);
  const context = resolveStageViewContext(props.detail, props.stageId, props.view.id, semantic.category, runArtifacts);
  const stageArtifacts = useMemo(
    () => props.resolvedViewArtifacts?.length ? uniqueMaskArtifacts([...props.resolvedViewArtifacts, ...context.resolvedArtifacts]) : context.resolvedArtifacts,
    [context.resolvedArtifacts, props.resolvedViewArtifacts],
  );
  const sourceArtifacts = useMemo(
    () => context.sourceArtifacts,
    [context.sourceArtifacts],
  );
  const maskArtifact = useMemo(
    () => resolveStageMaskArtifact(props.view, stageArtifacts),
    [props.view, stageArtifacts],
  );
  const resolvedSpec = useMemo(
    () => resolveStageMaskSpec(props.view, maskArtifact),
    [maskArtifact, props.view],
  );

  if (!detail) return <div className="empty-state">Select a take to inspect stage outputs.</div>;
  if (!maskArtifact) {
    if (props.unitViewResolution?.emptyState) {
      return <div className="empty-state"><strong>{props.unitViewResolution.label}</strong><small>{props.unitViewResolution.emptyState}</small></div>;
    }
    return emptyState(props.stageId, props.view.id, props.view.emptyState?.title);
  }
  if (!maskArtifact.path) return <div className="empty-state">Artifact file is unavailable.</div>;

  const resolverInput = useMemo<MaskReferenceResolverInput>(
    () => ({
      maskArtifact,
      selectedStageId: props.stageId,
      selectedTake: detail as TakeDetail,
      currentStageArtifacts: stageArtifacts,
      allRunArtifacts: runArtifacts,
      sourceArtifacts,
      resolvedArtifacts: uniqueMaskArtifacts([...stageArtifacts, ...sourceArtifacts]),
    }),
    [detail, maskArtifact, props.stageId, runArtifacts, sourceArtifacts, stageArtifacts],
  );
  const rawReferenceCandidates = useMemo(
    () => resolveBinaryMaskReferenceCandidates(resolverInput),
    [resolverInput],
  );
  const rankedReferenceCandidates = useMemo(
    () => rankStageMaskReferenceCandidates(rawReferenceCandidates, resolvedSpec),
    [rawReferenceCandidates, resolvedSpec],
  );

  useEffect(() => {
    const nextReferenceArtifactId = rankedReferenceCandidates[0]?.artifact.artifact_id ?? null;
    if (props.viewState.binaryMaskReferenceArtifactId && rankedReferenceCandidates.some((candidate) => candidate.artifact.artifact_id === props.viewState.binaryMaskReferenceArtifactId)) return;
    if (props.viewState.binaryMaskReferenceArtifactId === nextReferenceArtifactId) return;
    props.onBinaryMaskReferenceArtifactIdChange(nextReferenceArtifactId);
  }, [props, rankedReferenceCandidates]);

  const selectedCandidate = useMemo(
    () => rankedReferenceCandidates.find((candidate) => candidate.artifact.artifact_id === props.viewState.binaryMaskReferenceArtifactId) ?? rankedReferenceCandidates[0] ?? null,
    [props.viewState.binaryMaskReferenceArtifactId, rankedReferenceCandidates],
  );
  const selectedReferenceUrl = useMemo(
    () => selectedCandidate?.sourceUrl ?? (selectedCandidate?.artifact.path && takeId ? `${fileUrl(takeId, selectedCandidate.artifact.path)}` : null),
    [selectedCandidate, takeId],
  );
  const validatedResolution = useMemo(
    () => resolveBinaryMaskReference(resolverInput, selectedCandidate?.artifact.artifact_id ?? null, selectedReferenceUrl),
    [resolverInput, selectedCandidate?.artifact.artifact_id, selectedReferenceUrl],
  );
  const invalidArtifact = useMemo(
    () => resolveBinaryMaskInvalidArtifact(maskArtifact, uniqueMaskArtifacts([...runArtifacts, ...sourceArtifacts, ...stageArtifacts])),
    [maskArtifact, runArtifacts, sourceArtifacts, stageArtifacts],
  );
  const maskUrl = useMemo(
    () => `${fileUrl(takeId, maskArtifact.path ?? "")}`,
    [maskArtifact.path, takeId],
  );
  const invalidUrl = useMemo(
    () => invalidArtifact?.path && takeId ? `${fileUrl(takeId, invalidArtifact.path)}` : null,
    [invalidArtifact?.path, takeId],
  );
  const { rendered, error, loadState } = useMaskRenderAssets({ maskUrl, sourceUrl: selectedReferenceUrl, invalidUrl });
  const safeResolution = useMemo(
    () => validateMaskResolutionForLoadedImages(validatedResolution, loadState),
    [loadState, validatedResolution],
  );
  const availableModes = useMemo(
    () => availableBinaryMaskViewModes(maskArtifact, uniqueMaskArtifacts([...runArtifacts, ...sourceArtifacts, ...stageArtifacts]), safeResolution),
    [maskArtifact, runArtifacts, sourceArtifacts, stageArtifacts, safeResolution],
  );
  const defaultMode = useMemo(
    () => resolveStageMaskDefaultMode(props.view, maskArtifact, uniqueMaskArtifacts([...runArtifacts, ...sourceArtifacts, ...stageArtifacts]), safeResolution),
    [props.view, maskArtifact, runArtifacts, sourceArtifacts, stageArtifacts, safeResolution],
  );

  useEffect(() => {
    if (availableModes.includes(props.viewState.binaryMaskMode)) return;
    const nextMode = availableModes[0] ?? defaultMode ?? "mask";
    if (props.viewState.binaryMaskMode === nextMode) return;
    props.onBinaryMaskModeChange(nextMode);
  }, [
    availableModes,
    defaultMode,
    props.onBinaryMaskModeChange,
    props.viewState.binaryMaskMode,
  ]);

  const diagnostics = useMemo(
    () => buildMaskArtifactDiagnostics({
      maskArtifact,
      resolution: safeResolution,
      selectedReferenceId: selectedCandidate?.artifact.artifact_id ?? props.viewState.binaryMaskReferenceArtifactId,
      defaultReferenceId: rankedReferenceCandidates[0]?.artifact.artifact_id ?? null,
    }),
    [maskArtifact, props.viewState.binaryMaskReferenceArtifactId, rankedReferenceCandidates, safeResolution, selectedCandidate?.artifact.artifact_id],
  );
  const viewerCandidates = useMemo<ViewerMaskReferenceCandidate[]>(
    () => rankedReferenceCandidates.map((candidate) => ({
      id: candidate.artifact.artifact_id,
      label: candidate.label,
      artifact: candidate.artifact,
      url: candidate.sourceUrl,
      role: candidateRoleForViewer(candidate),
      priority: candidate.priority,
      computedFromArtifactId: safeResolution.computedFrom?.artifact.artifact_id,
      inferred: candidate.inferred,
    })),
    [rankedReferenceCandidates, safeResolution.computedFrom?.artifact.artifact_id],
  );

  return (
    <MaskArtifactViewer
      availableModes={availableModes}
      defaultReferenceId={rankedReferenceCandidates[0]?.artifact.artifact_id ?? null}
      diagnostics={{
        ...diagnostics,
        note: formatStageMaskDiagnosticsNote(
          resolvedSpec,
          safeResolution.warning ?? `Alignment: ${binaryMaskAlignmentLabel(safeResolution.alignment)}`,
        ),
      }}
      emptyMessage="Rendering mask preview..."
      errorMessage={error}
      fitMode={props.viewState.viewportMode === "natural" ? "actual_size" : props.viewState.viewportMode}
      invalidLabel="Invalid / no data"
      maskArtifact={maskArtifact}
      maskOffLabel={resolvedSpec.maskOffLabel}
      maskOnLabel={resolvedSpec.maskOnLabel}
      maskOpacity={finiteOrDefault(props.viewState.binaryMaskOpacity, 0.52)}
      mode={props.viewState.binaryMaskMode}
      onFitModeChange={(next) => props.onViewportModeChange(next === "actual_size" ? "natural" : next)}
      onMaskOpacityChange={props.onBinaryMaskOpacityChange}
      onModeChange={props.onBinaryMaskModeChange}
      onReferenceChange={(referenceId) => props.onBinaryMaskReferenceArtifactIdChange(referenceId || null)}
      onReferenceOpacityChange={props.onBinaryMaskReferenceOpacityChange}
      onResetView={props.onResetViewState}
      openUrl={maskArtifact.path ? `${fileUrl(detail.take_id, maskArtifact.path)}` : null}
      referenceCandidates={viewerCandidates}
      referenceOpacity={finiteOrDefault(props.viewState.binaryMaskReferenceOpacity, 1)}
      rendered={rendered}
      selectedReferenceId={selectedCandidate?.artifact.artifact_id ?? props.viewState.binaryMaskReferenceArtifactId}
      selectedReferenceUrl={safeResolution.sourceUrl ?? null}
      showDiagnostics
      showLegend
    />
  );
}
