import WizardStepLayout from "./WizardStepLayout";

type Preflight = {
  mlSetId: string;
  provenance: {
    filename?: string | null;
    sha256?: string | null;
    source_type?: string | null;
  };
  totalRows: number;
  keptInMlSetRows: number;
  physicalObjects: number;
  keptInMlSetObjects: number;
  defaultTrainableRows: number;
  defaultTrainableObjects: number;
  reviewRequiredRows: number;
  reviewRequiredObjects: number;
  excludedRows: number;
  excludedObjects: number;
  splitStatus: string;
  blockingErrors: string[];
};

export default function ManifestGenerationStep({ manifestResult, preflight }: { manifestResult: unknown; preflight?: Preflight }) {
  return (
    <WizardStepLayout title="Manifest Generation" description="Generate canonical manifest and deterministic ML set artifacts.">
      {preflight ? (
        <>
          <div className="entity-summary-cards ml-set-summary-cards">
            <article><small>ML set</small><strong>{preflight.mlSetId}</strong></article>
            <article><small>total rows</small><strong>{preflight.totalRows}</strong></article>
            <article><small>kept in ML set rows/objects</small><strong>{preflight.keptInMlSetRows} / {preflight.keptInMlSetObjects}</strong></article>
            <article><small>physical objects</small><strong>{preflight.physicalObjects}</strong></article>
            <article><small>default trainable rows/objects</small><strong>{preflight.defaultTrainableRows} / {preflight.defaultTrainableObjects}</strong></article>
            <article><small>review rows/objects</small><strong>{preflight.reviewRequiredRows} / {preflight.reviewRequiredObjects}</strong></article>
            <article><small>excluded rows/objects</small><strong>{preflight.excludedRows} / {preflight.excludedObjects}</strong></article>
            <article><small>split status</small><strong>{preflight.splitStatus}</strong></article>
          </div>
          <details className="entity-diagnostics" open>
            <summary>Materialization contract</summary>
            <pre className="datasets-inline-state">{JSON.stringify({
              source_type: preflight.provenance.source_type || "-",
              filename: preflight.provenance.filename || "-",
              sha256: preflight.provenance.sha256 || "-",
              blocking_errors: preflight.blockingErrors,
            }, null, 2)}</pre>
          </details>
          {preflight.blockingErrors.length > 0 ? <div className="empty-state">{preflight.blockingErrors.join(" ")}</div> : <small>Completed automatically from rich TSV fields once reconciliation is valid.</small>}
        </>
      ) : null}
      {manifestResult ? (
        <details className="entity-diagnostics" open>
          <summary>Generated manifest result</summary>
          <pre className="datasets-inline-state">{JSON.stringify(manifestResult, null, 2)}</pre>
        </details>
      ) : null}
    </WizardStepLayout>
  );
}
