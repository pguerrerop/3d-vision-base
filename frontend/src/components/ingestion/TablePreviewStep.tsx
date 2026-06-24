import WizardStepLayout from "./WizardStepLayout";

export default function TablePreviewStep({ parsedPreview }: { parsedPreview: unknown }) {
  const preview = (parsedPreview && typeof parsedPreview === "object") ? parsedPreview as {
    diagnostics?: Record<string, unknown>;
    schema?: Record<string, unknown>;
    provenance?: Record<string, unknown>;
  } : null;
  const diagnostics = preview?.diagnostics ?? {};
  const required = (diagnostics.required_field_inference && typeof diagnostics.required_field_inference === "object")
    ? diagnostics.required_field_inference as Record<string, unknown>
    : {};
  return (
    <WizardStepLayout title="Operator Table Ingestion" description="Parse CSV/TSV/XLSX/pasted tables and infer schema.">
      {preview?.diagnostics ? (
        <>
          <div className="entity-summary-cards ml-set-summary-cards">
            <article><small>format</small><strong>{String(diagnostics.detected_format || preview.provenance?.detected_format || "-")}</strong></article>
            <article><small>rows</small><strong>{String(diagnostics.row_count || preview.provenance?.row_count || 0)}</strong></article>
            <article><small>columns</small><strong>{String(diagnostics.column_count || preview.provenance?.column_count || 0)}</strong></article>
            <article><small>extra cols</small><strong>{String(diagnostics.ignored_extra_columns_count || 0)}</strong></article>
          </div>
          <div className="datasets-chip-row" style={{ marginTop: 8 }}>
            <span className={`status-chip ${required.label_column ? "active" : ""}`}>label: {required.label_column ? "ok" : "missing"}</span>
            <span className={`status-chip ${required.image_ref_or_range ? "active" : ""}`}>refs: {required.image_ref_or_range ? "ok" : "missing"}</span>
            <span className="status-chip">recognized: {Array.isArray(diagnostics.recognized_semantic_column_names) ? diagnostics.recognized_semantic_column_names.join(", ") || "-" : "-"}</span>
          </div>
        </>
      ) : null}
      <pre className="datasets-inline-state">{JSON.stringify(parsedPreview ?? {}, null, 2)}</pre>
    </WizardStepLayout>
  );
}
