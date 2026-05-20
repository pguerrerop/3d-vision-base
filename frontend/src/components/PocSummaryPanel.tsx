import type { PocRunSummary } from "../api/client";
import StatusBadge from "./StatusBadge";

type Props = {
  summary: PocRunSummary | null | undefined;
  compact?: boolean;
};

export default function PocSummaryPanel({ summary, compact = false }: Props) {
  if (!summary) {
    return <section className="section empty-state compact">No POC summary available.</section>;
  }
  const warningText = summary.warnings.length ? summary.warnings.join(", ") : "none";
  return (
    <section className={compact ? "poc-summary compact" : "poc-summary"}>
      <div className="section-heading">
        <h3>POC run summary</h3>
        <div className="badge-row">
          <StatusBadge value={summary.status.demo_ready ? "demo-ready" : "review"} />
          <StatusBadge value={summary.engine} />
        </div>
      </div>
      <div className="poc-grid">
        <div><span>Engine</span><strong>{summary.engine}</strong></div>
        <div><span>Calibration</span><strong>{summary.calibration.display}</strong></div>
        <div><span>Source</span><strong>{summary.calibration.resolution_source}</strong></div>
        <div><span>Status</span><strong>{summary.calibration.status}</strong></div>
        <div><span>Objects</span><strong>{summary.objects.found}</strong></div>
        <div><span>Balls</span><strong>{summary.objects.estimated_balls}</strong></div>
        <div><span>Rejected</span><strong>{summary.objects.rejected}</strong></div>
        <div><span>Plane</span><strong>{summary.plane.found ? "YES" : "NO"}</strong></div>
        <div><span>Total time</span><strong>{summary.profiling.total_processing_ms.toFixed(0)} ms</strong></div>
        <div><span>Calibration</span><strong>{summary.status.calibration_valid ? "VALID" : "CHECK"}</strong></div>
      </div>
      <div className={summary.warnings.length ? "warning-line active" : "warning-line"}>
        Latest warning: {warningText}
      </div>
    </section>
  );
}
