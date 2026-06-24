import WizardStepLayout from "./WizardStepLayout";

type ReconciledRow = {
  source_row_index?: number;
  take_id?: string;
  image_ref?: string;
  created_from_range?: boolean;
};

type Diagnostics = {
  unresolved_take_refs?: Array<Record<string, unknown>>;
};

export default function RangeExpansionStep({ rows, diagnostics }: { rows?: unknown; diagnostics?: unknown }) {
  const reconciledRows = Array.isArray(rows) ? (rows as ReconciledRow[]) : [];
  const parsedDiagnostics = (diagnostics && typeof diagnostics === "object") ? (diagnostics as Diagnostics) : {};
  const expandedRows = reconciledRows.filter((row) => Boolean(row.created_from_range));
  const invalidRangeCount = Array.isArray(parsedDiagnostics.unresolved_take_refs)
    ? parsedDiagnostics.unresolved_take_refs.filter((item) => String(item.reason || "") === "no_refs").length
    : 0;
  const sampleExpansions = Array.from(
    expandedRows.reduce((acc, row) => {
      const key = String(row.source_row_index || "row");
      const existing = acc.get(key) ?? { source_row_index: row.source_row_index, refs: [] as string[], take_ids: [] as string[] };
      existing.refs.push(String(row.image_ref || "-"));
      existing.take_ids.push(String(row.take_id || "-"));
      acc.set(key, existing);
      return acc;
    }, new Map<string, { source_row_index?: number; refs: string[]; take_ids: string[] }>())
      .values(),
  ).slice(0, 5);

  return (
    <WizardStepLayout title="Range Expansion" description="Expand implicit ranges into explicit take refs.">
      {expandedRows.length === 0 ? (
        <>
          <small>No range expansion needed; all rows used explicit image_ref values.</small>
          {invalidRangeCount > 0 ? <div className="empty-state">Invalid range count: {invalidRangeCount}</div> : null}
          <div style={{ marginTop: 8 }}><small>Completed automatically from rich TSV fields.</small></div>
        </>
      ) : (
        <>
          <div className="entity-summary-cards ml-set-summary-cards">
            <article><small>range rows</small><strong>{new Set(expandedRows.map((row) => row.source_row_index)).size}</strong></article>
            <article><small>generated refs</small><strong>{expandedRows.length}</strong></article>
            <article><small>invalid ranges</small><strong>{invalidRangeCount}</strong></article>
          </div>
          <details className="entity-diagnostics" open>
            <summary>Sample expansions</summary>
            <pre className="datasets-inline-state">{JSON.stringify(sampleExpansions, null, 2)}</pre>
          </details>
          <div style={{ marginTop: 8 }}><small>Completed automatically from rich TSV fields.</small></div>
        </>
      )}
    </WizardStepLayout>
  );
}
