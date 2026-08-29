import type { TakeDetail } from "../api/client";
import type { StudioArtifact } from "./studioWorkspaceModel";
import { deriveSelectedSurfaceRefinementSummary } from "./selectedSurfaceRefinementModel";

type Props = {
  artifacts: StudioArtifact[];
  detail?: TakeDetail | null;
};

function formatNumber(value: number | undefined, digits = 0): string {
  if (value == null || !Number.isFinite(value)) return "Not available";
  return value.toFixed(digits);
}

function formatPercent(value: number | undefined): string {
  if (value == null || !Number.isFinite(value)) return "Not available";
  return `${(value * 100).toFixed(1)}%`;
}

function formatBoolean(value: boolean | undefined): string {
  if (value == null) return "Not available";
  return value ? "Enabled" : "Disabled";
}

export default function SelectedSurfaceRefinementCard({ artifacts, detail }: Props) {
  const summary = deriveSelectedSurfaceRefinementSummary(artifacts, detail);

  return (
    <section className="selected-surface-refinement-card">
      <div className="selected-surface-refinement-header">
        <div>
          <strong>Selected cluster → surface refinement</strong>
          <small>The selected blob cluster is only the starting point. Candidate refinement, stripe suppression, and reference-model filtering can all remove pixels before the selected support is handed to normalization, residual QA, and suppression.</small>
        </div>
      </div>
      <div className="selected-surface-refinement-steps" aria-label="Selected cluster to surface refinement steps">
        {["Selected cluster", "Candidate refinement", "Stripe suppression", "Selected support", "Model support", "Suppression mask"].map((step) => (
          <span key={step} className="footer-chip">{step}</span>
        ))}
      </div>
      <div className="selected-surface-refinement-grid">
        <div><span>Selected cluster id</span><strong>{summary.selectedClusterId ?? "Not available"}</strong></div>
        <div><span>Selected cluster pixels</span><strong>{formatNumber(summary.selectedClusterPixels)}</strong></div>
        <div><span>Selected surface pixels</span><strong>{formatNumber(summary.selectedSurfacePixels)}</strong></div>
        <div><span>Kept ratio</span><strong>{formatPercent(summary.keptRatio)}</strong></div>
        <div><span>Selected surface % of valid ROI</span><strong>{formatPercent(summary.selectedSurfaceValidRoiRatio)}</strong></div>
        <div><span>Refine by MAD</span><strong>{formatBoolean(summary.madEnabled)}</strong></div>
        <div><span>MAD k</span><strong>{formatNumber(summary.madK, 2)}</strong></div>
        <div><span>MAD floor</span><strong>{summary.madFloor == null ? "Not available" : `${summary.madFloor.toFixed(2)} mm`}</strong></div>
        <div><span>Keep border support</span><strong>{formatBoolean(summary.keepBorderSupport)}</strong></div>
        <div><span>Min total pixels</span><strong>{formatNumber(summary.minPixels)}</strong></div>
        <div><span>Fallback strategy</span><strong>{summary.fallbackStrategy ?? "Not available"}</strong></div>
        <div><span>Fallback used</span><strong>{summary.fallbackUsed == null ? "Not available" : (summary.fallbackUsed ? "Yes" : "No")}</strong></div>
      </div>
      <small>{summary.reason ? `Reason: ${summary.reason}` : "Reason: Not available"}</small>
    </section>
  );
}
