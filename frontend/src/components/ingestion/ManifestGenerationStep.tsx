import WizardStepLayout from "./WizardStepLayout";

export default function ManifestGenerationStep({ manifestResult }: { manifestResult: unknown }) {
  return (
    <WizardStepLayout title="Manifest Generation" description="Generate canonical manifest and deterministic ML set artifacts.">
      <pre className="datasets-inline-state">{JSON.stringify(manifestResult ?? {}, null, 2)}</pre>
    </WizardStepLayout>
  );
}
