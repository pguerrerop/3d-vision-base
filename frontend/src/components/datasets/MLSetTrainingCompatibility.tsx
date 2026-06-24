import type { MLSetSummaryResponse } from "../../api/client";

export default function MLSetTrainingCompatibility({
  trainingCompatibility,
}: {
  trainingCompatibility: MLSetSummaryResponse["training_compatibility"];
}) {
  return (
    <section className="entity-section">
      <h4>Training Compatibility</h4>
      <div className="entity-stats-grid">
        {Object.entries(trainingCompatibility.feature_families || {}).map(([label, value]) => (
          <div key={label} className="entity-stats-row">
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
        <div className="entity-stats-row"><dt>classifier_ready</dt><dd>{trainingCompatibility.classifier_ready ? "yes" : "no"}</dd></div>
      </div>
    </section>
  );
}
