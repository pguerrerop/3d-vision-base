import WizardStepLayout from "./WizardStepLayout";

export default function AmbiguityResolutionStep({ diagnostics }: { diagnostics: unknown }) {
  return (
    <WizardStepLayout title="Ambiguity Resolution" description="Review unresolved/ambiguous/duplicate references.">
      <pre className="datasets-inline-state">{JSON.stringify(diagnostics ?? {}, null, 2)}</pre>
    </WizardStepLayout>
  );
}
