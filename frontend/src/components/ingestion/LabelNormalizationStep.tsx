import WizardStepLayout from "./WizardStepLayout";

export default function LabelNormalizationStep() {
  return <WizardStepLayout title="Label Normalization" description="Normalize operator labels to mining_balls_labels_v1."><small>Mapping is backend-driven and versioned.</small></WizardStepLayout>;
}
