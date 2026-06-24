import WizardStepLayout from "./WizardStepLayout";

type PreviewRow = {
  raw_label?: string;
  label?: string;
  normalized_class?: string;
  superclass?: string;
  count?: number;
  policy?: string;
  review_required?: boolean;
  warnings?: string[];
  warning_codes?: string[];
  source?: string;
};

function sourceBadge(source: string | undefined) {
  if (source === "table") return "from table";
  if (source === "taxonomy") return "from taxonomy";
  return "needs review";
}

export default function LabelNormalizationStep({ preview }: { preview: unknown }) {
  const rows = Array.isArray(preview) ? (preview as PreviewRow[]) : [];
  const rawLabelDistribution = rows.reduce<Record<string, number>>((acc, row) => {
    const key = row.raw_label || "UNKNOWN";
    acc[key] = (acc[key] || 0) + Number(row.count || 0);
    return acc;
  }, {});
  const normalizedClassDistribution = rows.reduce<Record<string, number>>((acc, row) => {
    const key = row.normalized_class || "UNKNOWN";
    acc[key] = (acc[key] || 0) + Number(row.count || 0);
    return acc;
  }, {});
  const superclassDistribution = rows.reduce<Record<string, number>>((acc, row) => {
    const key = row.superclass || "UNKNOWN";
    acc[key] = (acc[key] || 0) + Number(row.count || 0);
    return acc;
  }, {});
  const reviewRequiredCount = rows.reduce((acc, row) => acc + (row.review_required ? Number(row.count || 0) : 0), 0);
  const excludedReferenceCount = rows.reduce((acc, row) => acc + ((row.policy === "exclude" || row.policy === "reference") ? Number(row.count || 0) : 0), 0);
  const canonicalWarnings = rows.filter((row) => (row.warning_codes || []).some((code) => code === "invalid_normalized_class" || code === "invalid_superclass"));
  const unmappedRawLabels = rows.filter((row) => (row.warning_codes || []).includes("unmapped_raw_label"));
  const sourceCounts = rows.reduce<Record<string, number>>((acc, row) => {
    const key = sourceBadge(row.source);
    acc[key] = (acc[key] || 0) + Number(row.count || 0);
    return acc;
  }, {});
  return (
    <WizardStepLayout title="Label Normalization" description="Normalize operator labels to mining_balls_labels_v1.">
      {rows.length === 0 ? <small>Mapping is backend-driven and versioned.</small> : (
        <>
          <div className="entity-summary-cards ml-set-summary-cards">
            <article><small>review required</small><strong>{reviewRequiredCount}</strong></article>
            <article><small>excluded/reference</small><strong>{excludedReferenceCount}</strong></article>
            <article><small>canonical warnings</small><strong>{canonicalWarnings.length}</strong></article>
            <article><small>unmapped raw labels</small><strong>{unmappedRawLabels.length}</strong></article>
            <article><small>from table</small><strong>{sourceCounts["from table"] || 0}</strong></article>
            <article><small>from taxonomy</small><strong>{sourceCounts["from taxonomy"] || 0}</strong></article>
          </div>
          <details className="entity-diagnostics" open>
            <summary>Distribution summary</summary>
            <pre className="datasets-inline-state">{JSON.stringify({
              raw_label_distribution: rawLabelDistribution,
              normalized_class_distribution: normalizedClassDistribution,
              superclass_distribution: superclassDistribution,
              canonical_warnings: canonicalWarnings.map((row) => ({ raw_label: row.raw_label, warnings: row.warnings || [] })),
              unmapped_raw_labels: unmappedRawLabels.map((row) => ({ raw_label: row.raw_label, warnings: row.warnings || [] })),
              source_counts: sourceCounts,
            }, null, 2)}</pre>
          </details>
          <table className="datasets-table">
            <thead>
              <tr>
                <th>raw label</th>
                <th>label</th>
                <th>normalized class</th>
                <th>superclass</th>
                <th>count</th>
                <th>policy</th>
                <th>review required</th>
                <th>warnings</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row, index) => (
                <tr key={`${row.raw_label || "row"}-${index}`}>
                  <td>{row.raw_label || "-"}</td>
                  <td>{row.label || "-"}</td>
                  <td>{row.normalized_class || "-"} <small>({sourceBadge(row.source)})</small></td>
                  <td>{row.superclass || "-"}</td>
                  <td>{row.count ?? 0}</td>
                  <td>{row.policy || "-"}</td>
                  <td>{row.review_required ? "true" : "false"}</td>
                  <td>{(row.warnings || []).join(", ") || "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div style={{ marginTop: 8 }}><small>Completed automatically from rich TSV fields.</small></div>
        </>
      )}
    </WizardStepLayout>
  );
}
