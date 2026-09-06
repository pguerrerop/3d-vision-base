import type {
  ValidationBaseline,
  ValidationPromotionResponse,
  ValidationResolutionImpact,
} from "../../api/client";

/** The version this promotion would create if it isn't a reuse of an existing one. */
export function expectedNextVersion(source: ValidationBaseline): number {
  return source.version + 1;
}

/**
 * Impact, phrased for a promotion rather than an activation. Deliberately not
 * validationHistoryModel's summariseImpact(): that copy says "follow this
 * activation", which promotion.activate defaults to false, would be an
 * outright wrong sentence for the far more common case where the new version
 * is created inactive.
 */
export function summarisePromotionImpact(
  impact: ValidationResolutionImpact,
  willActivate: boolean,
): string[] {
  const rows: Array<{ count: number; noun: string; verb: string; pluralVerb: string }> = [
    {
      count: impact.active_cases.length,
      noun: "active-resolution case",
      verb: willActivate ? "will follow the new version" : "would follow it once it is activated",
      pluralVerb: willActivate
        ? "will follow the new version"
        : "would follow it once it is activated",
    },
    {
      count: impact.pinned_cases_unchanged.length,
      noun: "case",
      verb: "stays pinned to its own version and is unaffected",
      pluralVerb: "stay pinned to their own version and are unaffected",
    },
    {
      count: impact.allowed_version_cases_unchanged.length,
      noun: "case",
      verb: "keeps its allowed list unchanged and is unaffected",
      pluralVerb: "keep their allowed lists unchanged and are unaffected",
    },
    {
      count: impact.disabled_cases.length,
      noun: "case",
      verb: "is disabled and will be skipped either way",
      pluralVerb: "are disabled and will be skipped either way",
    },
    {
      count: impact.archived_suites.length,
      noun: "case",
      verb: "belongs to an archived suite and is unaffected",
      pluralVerb: "belong to archived suites and are unaffected",
    },
  ];
  return rows
    .filter((row) => row.count > 0)
    .map((row) => {
      const plural = row.count !== 1;
      const noun = plural ? `${row.noun}s` : row.noun;
      return `${row.count} ${noun} ${plural ? row.pluralVerb : row.verb}`;
    });
}

export interface PromotionOutcome {
  headline: string;
  detail: string;
  /** True when the created or reused baseline is not the active one yet. */
  needsActivation: boolean;
}

/**
 * The one sentence the spec insists on getting right: a repeat promotion must
 * never read like a fresh success. "This candidate was already promoted as
 * baseline v3. No new version was created." is a different message from
 * "Created baseline v3", not a variation of the same one.
 */
export function describePromotionOutcome(response: ValidationPromotionResponse): PromotionOutcome {
  const version = response.baseline.version;
  const active = response.baseline.status === "active";
  if (response.already_promoted) {
    return {
      headline: `This candidate was already promoted as baseline v${version}.`,
      detail: active
        ? "No new version was created. This version is active."
        : "No new version was created. This version is not active.",
      needsActivation: !active,
    };
  }
  return {
    headline: `Created baseline v${version}.`,
    detail: active
      ? "It is active: active-resolution cases now follow it."
      : "It was created inactive: active-resolution cases still follow the previous version until it is activated.",
    needsActivation: !active,
  };
}
