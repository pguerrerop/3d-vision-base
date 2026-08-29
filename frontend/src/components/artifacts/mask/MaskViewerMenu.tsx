import type { MaskFitMode } from "./maskRenderingTypes";

type Props = {
  diagnosticsVisible: boolean;
  checkerboardVisible: boolean;
  fitMode: MaskFitMode;
  openUrl?: string | null;
  onFitModeChange?: (fitMode: MaskFitMode) => void;
  onResetView?: () => void;
  onToggleDiagnostics?: () => void;
  onToggleCheckerboard?: () => void;
  onCopyArtifactId?: () => void;
};

export default function MaskViewerMenu({
  diagnosticsVisible,
  checkerboardVisible,
  fitMode,
  openUrl,
  onFitModeChange,
  onResetView,
  onToggleDiagnostics,
  onToggleCheckerboard,
  onCopyArtifactId,
}: Props) {
  return (
    <details className="mask-viewer-menu">
      <summary aria-label="Mask viewer options" title="Mask viewer options">More</summary>
      <div className="mask-viewer-menu-popover">
        <div className="mask-viewer-menu-section">
          <span>Fit</span>
          <button className={fitMode === "fit_all" ? "active" : ""} onClick={() => onFitModeChange?.("fit_all")} type="button">Fit all</button>
          <button className={fitMode === "fit_width" ? "active" : ""} onClick={() => onFitModeChange?.("fit_width")} type="button">Fit width</button>
          <button className={fitMode === "fit_height" ? "active" : ""} onClick={() => onFitModeChange?.("fit_height")} type="button">Fit height</button>
          <button className={fitMode === "actual_size" ? "active" : ""} onClick={() => onFitModeChange?.("actual_size")} type="button">100%</button>
          <button onClick={() => onResetView?.()} type="button">Reset view</button>
        </div>
        <div className="mask-viewer-menu-section">
          <span>Options</span>
          <button onClick={() => onToggleDiagnostics?.()} type="button">
            {diagnosticsVisible ? "Hide diagnostics" : "Show diagnostics"}
          </button>
          <button onClick={() => onToggleCheckerboard?.()} type="button">
            {checkerboardVisible ? "Hide checkerboard" : "Show checkerboard"}
          </button>
          {onCopyArtifactId ? <button onClick={() => onCopyArtifactId()} type="button">Copy artifact id</button> : null}
          {openUrl ? <a href={openUrl} rel="noreferrer" target="_blank">Open full size</a> : null}
        </div>
      </div>
    </details>
  );
}
