export type ValidationStatus =
  | "pass"
  | "changed"
  | "regression"
  | "not_comparable"
  | "blocked"
  | "needs_review"
  | "skipped"
  | "execution_failed"
  | "missing_baseline"
  | "not_produced"
  | "unsupported"
  | "disabled";

export const validationStatus: Record<
  ValidationStatus,
  {
    label: string;
    severity: "success" | "info" | "warning" | "danger" | "muted";
    icon: string;
    tooltip: string;
  }
> = {
  pass: { label: "PASS", severity: "success", icon: "✓", tooltip: "Within approved tolerance." },
  changed: {
    label: "CHANGED",
    severity: "info",
    icon: "•",
    tooltip: "Meaningful change without a failing policy.",
  },
  regression: {
    label: "REGRESSION",
    severity: "danger",
    icon: "!",
    tooltip: "Violates an approved expectation.",
  },
  not_comparable: {
    label: "NOT COMPARABLE",
    severity: "warning",
    icon: "?",
    tooltip: "No compatible semantic comparison.",
  },
  blocked: {
    label: "BLOCKED",
    severity: "muted",
    icon: "⊘",
    tooltip: "Upstream regression invalidates interpretation.",
  },
  needs_review: {
    label: "NEEDS REVIEW",
    severity: "warning",
    icon: "◐",
    tooltip: "Automation cannot decide correctness.",
  },
  skipped: { label: "SKIPPED", severity: "muted", icon: "–", tooltip: "Not evaluated." },
  execution_failed: {
    label: "EXECUTION FAILED",
    severity: "danger",
    icon: "×",
    tooltip: "Candidate execution failed.",
  },
  missing_baseline: {
    label: "MISSING BASELINE",
    severity: "muted",
    icon: "…",
    tooltip: "No approved baseline covers this stage.",
  },
  not_produced: {
    label: "NOT PRODUCED",
    severity: "muted",
    icon: "–",
    tooltip: "Artifact was not produced.",
  },
  unsupported: {
    label: "UNSUPPORTED",
    severity: "muted",
    icon: "?",
    tooltip: "Comparator is not supported.",
  },
  disabled: { label: "DISABLED", severity: "muted", icon: "–", tooltip: "Case is disabled." },
};
