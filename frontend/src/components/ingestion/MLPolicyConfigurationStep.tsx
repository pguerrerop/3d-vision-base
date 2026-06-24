import WizardStepLayout from "./WizardStepLayout";

type PolicySummary = {
  totalRows: number;
  keptInMlSetRows: number;
  defaultTrainableRows: number;
  reviewRequiredRows: number;
  excludedRows: number;
  splitStatus: string;
};

export default function MLPolicyConfigurationStep({ summary }: { summary?: PolicySummary }) {
  return (
    <WizardStepLayout title="ML Policy Configuration" description="Configure inclusion policy and split strategy.">
      {!summary ? <small>Default excludes unlabeled and REVIEW_REQUIRED from supervised training.</small> : (
        <>
          <div className="entity-summary-cards ml-set-summary-cards">
            <article><small>total rows</small><strong>{summary.totalRows}</strong></article>
            <article><small>kept in ML set rows</small><strong>{summary.keptInMlSetRows}</strong></article>
            <article><small>default trainable rows</small><strong>{summary.defaultTrainableRows}</strong></article>
            <article><small>review required</small><strong>{summary.reviewRequiredRows}</strong></article>
            <article><small>excluded/reference</small><strong>{summary.excludedRows}</strong></article>
            <article><small>split status</small><strong>{summary.splitStatus}</strong></article>
          </div>
          <div style={{ marginTop: 8 }}><small>Completed automatically from rich TSV fields.</small></div>
        </>
      )}
    </WizardStepLayout>
  );
}
