import type { MLSetSummaryResponse } from "../../api/client";

export default function MLSetSessionCoverage({
  coverage,
  onSelectSession,
}: {
  coverage: MLSetSummaryResponse["source_session_coverage"];
  onSelectSession?: (sessionId: string) => void;
}) {
  return (
    <section className="entity-section">
      <h4>Source Session Coverage</h4>
      <table className="datasets-table compact-table">
        <thead><tr><th>session</th><th>included</th><th>uncertain</th><th>classes</th></tr></thead>
        <tbody>
          {coverage.map((row) => (
            <tr key={row.session_id}>
              <td>
                {onSelectSession ? <button type="button" className="link-button" onClick={() => onSelectSession(row.session_id)}>{row.session_id}</button> : row.session_id}
              </td>
              <td>{row.included_takes}</td>
              <td>{row.uncertain_labels}</td>
              <td>{Object.entries(row.class_composition || {}).map(([label, count]) => `${label}:${count}`).join(" · ")}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
