import type { MLSetSummaryResponse } from "../../api/client";

export default function MLSetValidationWarnings({ warnings }: { warnings: MLSetSummaryResponse["warnings"] }) {
  return (
    <section className="entity-section">
      <h4>Validation & Warnings</h4>
      {!warnings.length ? <small>No active warnings.</small> : null}
      {warnings.map((warning) => (
        <div key={`${warning.code}_${warning.severity}`} className={`ml-set-warning severity-${warning.severity}`}>
          <strong>{warning.code}</strong>
          <small>{warning.explanation}</small>
          <small>Affected: {warning.affected_count}</small>
        </div>
      ))}
    </section>
  );
}
