import WizardStepLayout from "./WizardStepLayout";

export default function TablePreviewStep({ parsedPreview }: { parsedPreview: unknown }) {
  return (
    <WizardStepLayout title="Operator Table Ingestion" description="Parse CSV/TSV/XLSX/pasted tables and infer schema.">
      <pre className="datasets-inline-state">{JSON.stringify(parsedPreview ?? {}, null, 2)}</pre>
    </WizardStepLayout>
  );
}
