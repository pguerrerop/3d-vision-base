import { useMemo, useState } from "react";
import MaskViewerToolbar from "./MaskViewerToolbar";
import type {
  ArtifactLike,
  MaskArtifactDiagnostics,
  MaskFitMode,
  MaskReferenceCandidate,
  MaskRenderedAssets,
  MaskRenderMode,
} from "./maskRenderingTypes";

export type MaskArtifactViewerProps = {
  maskArtifact: ArtifactLike;
  rendered: MaskRenderedAssets | null;
  referenceCandidates: MaskReferenceCandidate[];
  selectedReferenceId?: string | null;
  defaultReferenceId?: string | null;
  selectedReferenceUrl?: string | null;
  mode?: MaskRenderMode;
  fitMode?: MaskFitMode;
  availableModes?: MaskRenderMode[];
  maskOpacity?: number;
  referenceOpacity?: number;
  showDiagnostics?: boolean;
  showLegend?: boolean;
  maskOnLabel?: string;
  maskOffLabel?: string;
  invalidLabel?: string;
  diagnostics?: MaskArtifactDiagnostics;
  emptyMessage?: string;
  errorMessage?: string | null;
  openUrl?: string | null;
  onModeChange?: (mode: MaskRenderMode) => void;
  onFitModeChange?: (fitMode: MaskFitMode) => void;
  onResetView?: () => void;
  onReferenceChange?: (referenceId: string) => void;
  onMaskOpacityChange?: (value: number) => void;
  onReferenceOpacityChange?: (value: number) => void;
  onShowDiagnosticsChange?: (visible: boolean) => void;
};

function viewportClass(fitMode: MaskFitMode): string {
  if (fitMode === "fit_width") return "viewport-fit-width";
  if (fitMode === "fit_height") return "viewport-fit-height";
  if (fitMode === "actual_size") return "viewport-natural";
  return "viewport-fit_all";
}

function diagnosticSourceLabel(source?: "user" | "inferred" | "default"): string | null {
  if (source === "user") return "user-selected";
  if (source === "inferred") return "inferred from stage context";
  if (source === "default") return "default";
  return null;
}

function MaskPane({ label, src, showCheckerboard }: { label: string; src: string; showCheckerboard: boolean }) {
  return (
    <div className="binary-mask-side-pane">
      <strong>{label}</strong>
      <div className={`overlay-frame binary-mask-frame ${showCheckerboard ? "" : "checkerboard-hidden"}`}>
        <img alt={label} src={src} />
      </div>
    </div>
  );
}

export default function MaskArtifactViewer({
  maskArtifact,
  rendered,
  referenceCandidates,
  selectedReferenceId = null,
  defaultReferenceId = null,
  selectedReferenceUrl = null,
  mode = "mask",
  fitMode = "fit_all",
  availableModes = ["mask"],
  maskOpacity = 0.52,
  referenceOpacity = 1,
  showDiagnostics = true,
  showLegend = true,
  maskOnLabel = "Mask ON",
  maskOffLabel = "Mask OFF",
  invalidLabel = "Invalid / no data",
  diagnostics,
  emptyMessage = "Rendering mask preview...",
  errorMessage = null,
  openUrl,
  onModeChange,
  onFitModeChange,
  onResetView,
  onReferenceChange,
  onMaskOpacityChange,
  onReferenceOpacityChange,
  onShowDiagnosticsChange,
}: MaskArtifactViewerProps) {
  const [checkerboardVisible, setCheckerboardVisible] = useState(true);
  const [diagnosticsVisible, setDiagnosticsVisible] = useState(showDiagnostics);
  const selectedReferenceLabel = diagnostics?.selectedReferenceLabel ?? referenceCandidates.find((candidate) => candidate.id === selectedReferenceId)?.label ?? "unavailable";
  const sourceBadge = diagnosticSourceLabel(diagnostics?.selectedReferenceSource);
  const canOverlay = Boolean(selectedReferenceUrl && availableModes.includes("overlay"));
  const canBoundary = Boolean(selectedReferenceUrl && availableModes.includes("boundary"));
  const canSideBySide = Boolean(selectedReferenceUrl && availableModes.includes("side_by_side"));

  const diagnosticsRows = useMemo(
    () => [
      diagnostics?.selectedReferenceLabel ? `Selected reference: ${diagnostics.selectedReferenceLabel}${sourceBadge ? ` (${sourceBadge})` : ""}` : null,
      diagnostics?.computedFromLabel ? `Computed from: ${diagnostics.computedFromLabel}` : null,
      diagnostics?.coordinateSpace ? `Coordinate space: ${diagnostics.coordinateSpace}` : null,
      diagnostics?.dimensions ? `Dimensions: ${diagnostics.dimensions}` : null,
      diagnostics?.artifactId ? `Artifact id: ${diagnostics.artifactId}` : null,
      diagnostics?.sourceArtifactId ? `Source artifact id: ${diagnostics.sourceArtifactId}` : null,
      diagnostics?.note ?? null,
    ].filter((value): value is string => Boolean(value)),
    [diagnostics, sourceBadge],
  );

  const toggleDiagnostics = () => {
    const next = !diagnosticsVisible;
    setDiagnosticsVisible(next);
    onShowDiagnosticsChange?.(next);
  };

  const copyArtifactId = async () => {
    if (!maskArtifact.artifact_id || typeof navigator === "undefined" || !navigator.clipboard?.writeText) return;
    await navigator.clipboard.writeText(maskArtifact.artifact_id);
  };

  return (
    <div className="binary-mask-panel mask-artifact-viewer">
      {errorMessage ? <div className="artifact-warning-banner">{errorMessage}</div> : null}
      {!selectedReferenceUrl ? <div className="artifact-reference"><small>Comparison source unavailable. Showing mask-only view.</small></div> : null}
      {!rendered ? (
        <div className="empty-state">{emptyMessage}</div>
      ) : mode === "side_by_side" && selectedReferenceUrl && canSideBySide ? (
        <div className={`mask-artifact-viewport ${viewportClass(fitMode)}`}>
          <div className="binary-mask-side-by-side">
            <MaskPane label={selectedReferenceLabel} showCheckerboard={checkerboardVisible} src={selectedReferenceUrl} />
            <MaskPane label={maskArtifact.title} showCheckerboard={checkerboardVisible} src={rendered.maskOnlyUrl} />
          </div>
        </div>
      ) : (
        <div className={`mask-artifact-viewport ${viewportClass(fitMode)}`}>
          <div className={`overlay-frame binary-mask-frame ${checkerboardVisible ? "" : "checkerboard-hidden"}`}>
            {mode === "overlay" && selectedReferenceUrl && canOverlay ? (
              <>
                <img alt={selectedReferenceLabel} src={selectedReferenceUrl} style={{ opacity: referenceOpacity }} />
                <img alt={`${maskArtifact.title} overlay`} className="binary-mask-layer" src={rendered.overlayUrl} style={{ opacity: maskOpacity }} />
              </>
            ) : mode === "boundary" && selectedReferenceUrl && canBoundary ? (
              <>
                <img alt={selectedReferenceLabel} src={selectedReferenceUrl} style={{ opacity: referenceOpacity }} />
                <img alt={`${maskArtifact.title} boundary`} className="binary-mask-layer" src={rendered.boundaryUrl} />
              </>
            ) : (
              <img alt={maskArtifact.title} src={rendered.maskOnlyUrl} />
            )}
          </div>
        </div>
      )}
      <MaskViewerToolbar
        availableModes={availableModes}
        checkerboardVisible={checkerboardVisible}
        defaultReferenceId={defaultReferenceId}
        diagnosticsVisible={diagnosticsVisible}
        fitMode={fitMode}
        maskOpacity={maskOpacity}
        mode={mode}
        onCopyArtifactId={maskArtifact.artifact_id ? (() => { void copyArtifactId(); }) : undefined}
        onFitModeChange={onFitModeChange}
        onMaskOpacityChange={onMaskOpacityChange}
        onModeChange={onModeChange}
        onReferenceChange={onReferenceChange}
        onReferenceOpacityChange={onReferenceOpacityChange}
        onResetView={onResetView ?? (() => onFitModeChange?.("fit_all"))}
        onToggleCheckerboard={() => setCheckerboardVisible((current) => !current)}
        onToggleDiagnostics={toggleDiagnostics}
        openUrl={openUrl}
        referenceCandidates={referenceCandidates}
        referenceOpacity={referenceOpacity}
        selectedReferenceId={selectedReferenceId}
      />
      {showLegend ? (
        <div className="binary-mask-legend" aria-label="Binary mask legend">
          <span><i className="legend-swatch mask-on" />{maskOnLabel}</span>
          <span><i className="legend-swatch mask-off" />{maskOffLabel}</span>
          {rendered?.hasInvalidData ? <span><i className="legend-swatch mask-invalid" />{invalidLabel}</span> : null}
        </div>
      ) : null}
      {diagnosticsVisible && diagnosticsRows.length ? (
        <div className="binary-mask-status">
          {diagnosticsRows.map((row) => <small key={row}>{row}</small>)}
        </div>
      ) : null}
    </div>
  );
}
