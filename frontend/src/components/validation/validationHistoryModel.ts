import type {
  ValidationBaseline,
  ValidationResolutionImpact,
  ValidationSuiteCase,
} from "../../api/client";

/**
 * Same artifact identity: take, pipeline, stage and artifact must all agree.
 *
 * Deliberately defined here rather than imported from validationCaseModel, which
 * exports the same predicate. Every cross-module reference in this codebase is
 * `import type`, because the node test runner strips types but cannot resolve a
 * value import that carries no file extension. Keep the two in step.
 */
function isSameFamily(a: ValidationBaseline, b: ValidationBaseline): boolean {
  return (
    a.take_id === b.take_id &&
    a.pipeline_id === b.pipeline_id &&
    (a.stage_id ?? null) === (b.stage_id ?? null) &&
    (a.artifact_id ?? null) === (b.artifact_id ?? null)
  );
}

export interface IntegritySummary {
  label: string;
  detail: string;
  severity: "success" | "warning" | "danger" | "muted";
  /** Whether this version is safe to configure a case against. */
  usable: boolean;
}

/**
 * Integrity is recomputed by the history endpoint on every read, so an absent
 * value means "not reported" rather than "fine".
 */
export function describeIntegrity(baseline: ValidationBaseline): IntegritySummary {
  if (baseline.integrity_state === "ok") {
    return {
      label: "Verified",
      detail: "The stored snapshot still matches its recorded checksum.",
      severity: "success",
      usable: true,
    };
  }
  if (baseline.integrity_state === "invalid") {
    return {
      label: "Failed",
      detail:
        "The stored snapshot does not match its recorded checksum, or is missing. This version should not be used until it is investigated.",
      severity: "danger",
      usable: false,
    };
  }
  return {
    label: "Not checked",
    detail: "This listing did not report an integrity state for this version.",
    severity: "muted",
    usable: true,
  };
}

/** Newest first, which is how a reviewer reads a version history. */
export function orderHistory(baselines: ValidationBaseline[]): ValidationBaseline[] {
  return baselines.slice().sort((a, b) => {
    if (b.version !== a.version) return b.version - a.version;
    return String(b.created_at ?? "").localeCompare(String(a.created_at ?? ""));
  });
}

/** The version each row supersedes, resolved to a label rather than an id. */
export function supersededByLabel(
  baseline: ValidationBaseline,
  history: ValidationBaseline[],
): string | null {
  const successor = history.find((other) => other.supersedes_baseline_id === baseline.id);
  return successor ? `v${successor.version}` : null;
}

export interface ActivationPreview {
  /** Every version of the family active right now. Activation is not exclusive. */
  currentlyActive: ValidationBaseline[];
  /** The version active-mode cases would resolve to afterwards: the highest active one. */
  resolvesToAfter: ValidationBaseline;
  target: ValidationBaseline;
  /** True when the target is already active, so there is nothing to do. */
  noop: boolean;
  /**
   * Set when activating this version will not change what cases resolve to,
   * because a higher version stays active alongside it.
   */
  ineffective: string | null;
  blocked: string | null;
}

/**
 * What activating this version would mean, before anyone commits to it.
 *
 * Activation is deliberately not exclusive on the server: a family may hold
 * several active versions, and active-mode cases resolve to the highest of them.
 * Activating an older version therefore changes nothing on its own, which is
 * worth saying before someone clicks it and sees no effect.
 */
export function previewActivation(
  target: ValidationBaseline,
  history: ValidationBaseline[],
): ActivationPreview {
  const family = history.filter((other) => isSameFamily(other, target));
  const currentlyActive = family.filter((other) => other.status === "active");
  const afterwards = currentlyActive.filter((other) => other.id !== target.id).concat(target);
  const resolvesToAfter = afterwards.reduce((best, other) =>
    other.version > best.version ? other : best,
  );
  const integrity = describeIntegrity(target);
  return {
    currentlyActive,
    resolvesToAfter,
    target,
    noop: target.status === "active",
    ineffective:
      resolvesToAfter.id === target.id
        ? null
        : `v${resolvesToAfter.version} is also active and is newer, so active-resolution cases will keep comparing against it. Deactivate v${resolvesToAfter.version} if you want v${target.version} to take effect.`,
    blocked: integrity.usable ? null : integrity.detail,
  };
}

export interface DeactivationWarning {
  /** Cases that would be left without an active version to resolve to. */
  strandedActiveCases: number;
  pinnedCases: number;
  allowedCases: number;
  message: string | null;
}

/**
 * Deactivating never rewrites a case, but it can leave active-mode cases with
 * nothing to resolve to.  That is worth saying out loud beforehand.
 */
export function previewDeactivation(
  target: ValidationBaseline,
  history: ValidationBaseline[],
  impact: ValidationResolutionImpact,
): DeactivationWarning {
  const remaining = history.filter(
    (other) => other.id !== target.id && other.status === "active" && isSameFamily(other, target),
  );
  const strandedActiveCases = remaining.length ? 0 : impact.active_cases.length;
  const pinnedCases = impact.pinned_cases_unchanged.length;
  const allowedCases = impact.allowed_version_cases_unchanged.length;

  const notes: string[] = [];
  if (strandedActiveCases) {
    notes.push(
      `${strandedActiveCases} active-resolution ${strandedActiveCases === 1 ? "case has" : "cases have"} no other active version to fall back to, and will report a missing baseline.`,
    );
  }
  if (pinnedCases) {
    notes.push(
      `${pinnedCases} pinned ${pinnedCases === 1 ? "case keeps" : "cases keep"} referencing this version; their configuration is not changed.`,
    );
  }
  if (allowedCases) {
    notes.push(
      `${allowedCases} allowed-version ${allowedCases === 1 ? "case keeps" : "cases keep"} it in their list; their configuration is not changed.`,
    );
  }
  return {
    strandedActiveCases,
    pinnedCases,
    allowedCases,
    message: notes.length ? notes.join(" ") : null,
  };
}

/** A one-line count of who follows the activation and who deliberately does not. */
export function summariseImpact(impact: ValidationResolutionImpact): string[] {
  // Each row carries both verb forms; a count of one otherwise reads "1 case follow".
  const rows: Array<[number, string, string]> = [
    [impact.active_cases.length, "follows this activation", "follow this activation"],
    [
      impact.pinned_cases_unchanged.length,
      "stays pinned to its own version",
      "stay pinned to their own version",
    ],
    [
      impact.allowed_version_cases_unchanged.length,
      "keeps its allowed list unchanged",
      "keep their allowed list unchanged",
    ],
    [
      impact.disabled_cases.length,
      "is disabled and will be skipped",
      "are disabled and will be skipped",
    ],
    [impact.archived_suites.length, "belongs to an archived suite", "belong to archived suites"],
  ];
  return rows
    .filter(([count]) => count > 0)
    .map(
      ([count, one, many]) =>
        `${count} ${count === 1 ? "case" : "cases"} ${count === 1 ? one : many}`,
    );
}

/**
 * Whether a version can be offered to a case as a pin or an allowed version.
 * Mirrors the server's compatibility rule, which is by family, not by suite.
 */
export function canAttachToCase(
  baseline: ValidationBaseline,
  caseItem: ValidationSuiteCase,
  history: ValidationBaseline[],
): boolean {
  if (baseline.take_id !== caseItem.take_id) return false;
  const configured = history.filter((other) => caseItem.baseline_ids.includes(other.id));
  if (!configured.length) return true;
  return configured.some((other) => isSameFamily(baseline, other));
}
