import WizardStepLayout from "./WizardStepLayout";

export default function SessionReconciliationStep({ preview }: { preview: unknown }) {
  return (
    <WizardStepLayout title="Session Reconciliation" description="Resolve refs to canonical dataset/session/take IDs.">
      <pre className="datasets-inline-state">{JSON.stringify(preview ?? {}, null, 2)}</pre>
    </WizardStepLayout>
  );
}
