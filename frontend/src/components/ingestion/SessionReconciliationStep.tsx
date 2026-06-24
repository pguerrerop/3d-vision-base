import WizardStepLayout from "./WizardStepLayout";

type ParsedPreview = {
  rows?: Array<Record<string, unknown>>;
  schema?: Record<string, unknown>;
  warnings?: string[];
  diagnostics?: Record<string, unknown>;
};

type ReconciledPreview = {
  rows?: Array<Record<string, unknown>>;
  diagnostics?: {
    unresolved_take_refs?: Array<Record<string, unknown>>;
    ambiguous_take_refs?: Array<Record<string, unknown>>;
    duplicate_take_assignments?: Array<Record<string, unknown>>;
    dataset_session_conflicts?: Array<Record<string, unknown>>;
  };
};

function parseSummary(parsed: ParsedPreview) {
  const diagnostics = (parsed.diagnostics && typeof parsed.diagnostics === "object") ? parsed.diagnostics : {};
  const schema = (parsed.schema && typeof parsed.schema === "object") ? parsed.schema : {};
  const recognizedColumns = Array.isArray((diagnostics as { recognized_semantic_column_names?: unknown }).recognized_semantic_column_names)
    ? ((diagnostics as { recognized_semantic_column_names?: string[] }).recognized_semantic_column_names ?? [])
    : [];
  const rows = Array.isArray(parsed.rows) ? parsed.rows : [];
  const objectIdCount = rows.filter((row) => String(row.object_id || row.physical_object_id || "").trim()).length;
  const inferredRangeColumns = [schema.from_column, schema.to_column].filter(Boolean).join(" / ");
  return {
    rowCount: Number((diagnostics as { row_count?: unknown }).row_count ?? rows.length ?? 0),
    columnCount: Number((diagnostics as { column_count?: unknown }).column_count ?? 0),
    labelColumn: String((schema as { label_column?: unknown }).label_column || "-"),
    imageRefColumn: String((schema as { image_ref_column?: unknown }).image_ref_column || inferredRangeColumns || "-"),
    recognizedColumns,
    objectIdCount,
    parsedRowCount: rows.length,
    warnings: Array.isArray(parsed.warnings) ? parsed.warnings : [],
  };
}

function reconcileSummary(reconciled: ReconciledPreview) {
  const rows = Array.isArray(reconciled.rows) ? reconciled.rows : [];
  const diagnostics = reconciled.diagnostics ?? {};
  return {
    resolvedCount: rows.length,
    unresolvedCount: Array.isArray(diagnostics.unresolved_take_refs) ? diagnostics.unresolved_take_refs.length : 0,
    ambiguousCount: Array.isArray(diagnostics.ambiguous_take_refs) ? diagnostics.ambiguous_take_refs.length : 0,
    duplicateCount: Array.isArray(diagnostics.duplicate_take_assignments) ? diagnostics.duplicate_take_assignments.length : 0,
    datasetSessionConflictCount: Array.isArray(diagnostics.dataset_session_conflicts) ? diagnostics.dataset_session_conflicts.length : 0,
    sampleUnresolved: Array.isArray(diagnostics.unresolved_take_refs) ? diagnostics.unresolved_take_refs.slice(0, 5) : [],
    sampleAmbiguous: Array.isArray(diagnostics.ambiguous_take_refs) ? diagnostics.ambiguous_take_refs.slice(0, 5) : [],
  };
}

export default function SessionReconciliationStep({ parsed, reconciled }: { parsed?: unknown; reconciled?: unknown }) {
  const parsedPreview = (parsed && typeof parsed === "object") ? parsed as ParsedPreview : null;
  const reconciledPreview = (reconciled && typeof reconciled === "object") ? reconciled as ReconciledPreview : null;
  return (
    <WizardStepLayout title="Session Reconciliation" description="Resolve refs to canonical dataset/session/take IDs.">
      {reconciledPreview ? (
        <>
          {(() => {
            const summary = reconcileSummary(reconciledPreview);
            return (
              <div className="entity-summary-cards ml-set-summary-cards">
                <article><small>resolved</small><strong>{summary.resolvedCount}</strong></article>
                <article><small>unresolved</small><strong>{summary.unresolvedCount}</strong></article>
                <article><small>ambiguous</small><strong>{summary.ambiguousCount}</strong></article>
                <article><small>duplicates</small><strong>{summary.duplicateCount}</strong></article>
                <article><small>dataset/session conflicts</small><strong>{summary.datasetSessionConflictCount}</strong></article>
              </div>
            );
          })()}
          <small>Reconciliation complete. Canonical dataset/session/take IDs have been resolved where possible.</small>
          <details className="entity-diagnostics" open>
            <summary>Unresolved / ambiguous samples</summary>
            <pre className="datasets-inline-state">{JSON.stringify({
              unresolved: reconcileSummary(reconciledPreview).sampleUnresolved,
              ambiguous: reconcileSummary(reconciledPreview).sampleAmbiguous,
            }, null, 2)}</pre>
          </details>
          <details className="entity-diagnostics">
            <summary>Advanced diagnostics</summary>
            <pre className="datasets-inline-state">{JSON.stringify(reconciledPreview, null, 2)}</pre>
          </details>
        </>
      ) : parsedPreview ? (
        <>
          {(() => {
            const summary = parseSummary(parsedPreview);
            return (
              <>
                <div className="entity-summary-cards ml-set-summary-cards">
                  <article><small>rows</small><strong>{summary.rowCount}</strong></article>
                  <article><small>columns</small><strong>{summary.columnCount}</strong></article>
                  <article><small>label column</small><strong>{summary.labelColumn}</strong></article>
                  <article><small>ref/range</small><strong>{summary.imageRefColumn}</strong></article>
                  <article><small>semantic cols</small><strong>{summary.recognizedColumns.length}</strong></article>
                  <article><small>rows with object_id</small><strong>{summary.objectIdCount}</strong></article>
                </div>
                <div style={{ marginTop: 8 }}>
                  <small>Run reconciliation to resolve refs to canonical dataset/session/take IDs.</small>
                </div>
                <details className="entity-diagnostics" open>
                  <summary>Parsed input summary</summary>
                  <pre className="datasets-inline-state">{JSON.stringify({
                    recognized_semantic_columns: summary.recognizedColumns,
                    parsed_rows: summary.parsedRowCount,
                    warnings: summary.warnings,
                  }, null, 2)}</pre>
                </details>
                <details className="entity-diagnostics">
                  <summary>Advanced diagnostics</summary>
                  <pre className="datasets-inline-state">{JSON.stringify(parsedPreview, null, 2)}</pre>
                </details>
              </>
            );
          })()}
        </>
      ) : (
        <div className="empty-state">Parse a table first to preview inferred schema and then run reconciliation.</div>
      )}
    </WizardStepLayout>
  );
}
