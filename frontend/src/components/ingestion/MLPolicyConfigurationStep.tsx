import WizardStepLayout from "./WizardStepLayout";

export default function MLPolicyConfigurationStep() {
  return <WizardStepLayout title="ML Policy Configuration" description="Configure inclusion policy and split strategy."><small>Default excludes unlabeled and REVIEW_REQUIRED from supervised training.</small></WizardStepLayout>;
}
