import MaskOpacityControls from "./MaskOpacityControls";
import MaskReferenceSelector from "./MaskReferenceSelector";
import MaskViewerMenu from "./MaskViewerMenu";
import type { MaskFitMode, MaskReferenceCandidate, MaskRenderMode } from "./maskRenderingTypes";

type Props = {
  availableModes: MaskRenderMode[];
  mode: MaskRenderMode;
  fitMode: MaskFitMode;
  selectedReferenceId?: string | null;
  defaultReferenceId?: string | null;
  referenceCandidates: MaskReferenceCandidate[];
  maskOpacity: number;
  referenceOpacity: number;
  openUrl?: string | null;
  diagnosticsVisible: boolean;
  checkerboardVisible: boolean;
  onModeChange?: (mode: MaskRenderMode) => void;
  onFitModeChange?: (fitMode: MaskFitMode) => void;
  onReferenceChange?: (referenceId: string) => void;
  onMaskOpacityChange?: (value: number) => void;
  onReferenceOpacityChange?: (value: number) => void;
  onResetView?: () => void;
  onToggleDiagnostics?: () => void;
  onToggleCheckerboard?: () => void;
  onCopyArtifactId?: () => void;
};

const MODES: MaskRenderMode[] = ["mask", "overlay", "boundary", "side_by_side"];

function labelForMode(mode: MaskRenderMode): string {
  return mode === "side_by_side" ? "Side-by-side" : mode[0].toUpperCase() + mode.slice(1);
}

export default function MaskViewerToolbar({
  availableModes,
  mode,
  fitMode,
  selectedReferenceId,
  defaultReferenceId,
  referenceCandidates,
  maskOpacity,
  referenceOpacity,
  openUrl,
  diagnosticsVisible,
  checkerboardVisible,
  onModeChange,
  onFitModeChange,
  onReferenceChange,
  onMaskOpacityChange,
  onReferenceOpacityChange,
  onResetView,
  onToggleDiagnostics,
  onToggleCheckerboard,
  onCopyArtifactId,
}: Props) {
  return (
    <div className="mask-viewer-toolbar">
      <div className="mask-viewer-toolbar-main">
        <div className="binary-mask-mode-group" role="group" aria-label="Mask appearance">
          {MODES.map((value) => (
            <button
              key={value}
              className={mode === value ? "active" : ""}
              disabled={!availableModes.includes(value)}
              onClick={() => onModeChange?.(value)}
              type="button"
            >
              {labelForMode(value)}
            </button>
          ))}
        </div>
        <MaskReferenceSelector
          candidates={referenceCandidates}
          defaultReferenceId={defaultReferenceId}
          onChange={onReferenceChange}
          selectedReferenceId={selectedReferenceId}
        />
      </div>
      <div className="mask-viewer-toolbar-secondary">
        <MaskOpacityControls
          maskOpacity={maskOpacity}
          maskOpacityDisabled={mode !== "overlay" || !availableModes.includes("overlay")}
          onMaskOpacityChange={onMaskOpacityChange}
          onReferenceOpacityChange={onReferenceOpacityChange}
          referenceOpacity={referenceOpacity}
          referenceOpacityDisabled={!referenceCandidates.length || (mode !== "overlay" && mode !== "boundary")}
        />
        <MaskViewerMenu
          checkerboardVisible={checkerboardVisible}
          diagnosticsVisible={diagnosticsVisible}
          fitMode={fitMode}
          onCopyArtifactId={onCopyArtifactId}
          onFitModeChange={onFitModeChange}
          onResetView={onResetView}
          onToggleCheckerboard={onToggleCheckerboard}
          onToggleDiagnostics={onToggleDiagnostics}
          openUrl={openUrl}
        />
      </div>
    </div>
  );
}
