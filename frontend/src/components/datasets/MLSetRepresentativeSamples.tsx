import { fileUrl, type MLSetSummaryResponse } from "../../api/client";

export default function MLSetRepresentativeSamples({
  samples,
}: {
  samples: MLSetSummaryResponse["representative_samples"];
}) {
  return (
    <section className="entity-section">
      <h4>Representative Samples</h4>
      <div className="ml-set-sample-groups">
        {Object.entries(samples).map(([label, rows]) => (
          <div key={label} className="ml-set-sample-group">
            <strong>{label}</strong>
            <div className="ml-set-sample-row">
              {rows.map((row) => (
                <a key={row.take_id} href={`/takes/${encodeURIComponent(row.take_id)}`} className="ml-set-sample-card" title={`${row.take_id} · ${row.validation_status || "-"}`}>
                  {row.thumbnail_path ? <img src={fileUrl(row.take_id, row.thumbnail_path)} alt={row.take_id} loading="lazy" /> : <span className="datasets-thumb-placeholder">-</span>}
                  <small>{row.take_id}</small>
                </a>
              ))}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
