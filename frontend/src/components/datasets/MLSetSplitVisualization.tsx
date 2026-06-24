import type { MLSetSummaryResponse } from "../../api/client";

export default function MLSetSplitVisualization({ splitSummary }: { splitSummary: MLSetSummaryResponse["split_summary"] }) {
  return (
    <section className="entity-section">
      <h4>Split Visualization</h4>
      <div className="datasets-chip-row">
        {Object.entries(splitSummary.split_counts || {}).map(([split, count]) => <span key={split} className="status-chip">{split} {count}</span>)}
      </div>
      <small>Splitting is governed by physical_object_id, not take_id.</small>
      {splitSummary.leaked_object_ids?.length ? (
        <div className="warning-line active">Leakage conflict in object groups: {splitSummary.leaked_object_ids.slice(0, 8).join(", ")}</div>
      ) : null}
      <details className="entity-diagnostics">
        <summary>Session composition by split</summary>
        {Object.entries(splitSummary.session_composition_by_split || {}).map(([split, sessions]) => (
          <small key={split}>{split}: {Object.entries(sessions).map(([sessionId, count]) => `${sessionId}:${count}`).join(" · ")}</small>
        ))}
      </details>
    </section>
  );
}
