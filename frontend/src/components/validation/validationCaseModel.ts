import type {
  ValidationBaseline,
  ValidationBaselineResolution,
  ValidationResolutionMode,
  ValidationSuiteCase,
} from "../../api/client";

/** What the case editor holds while the user is still changing it. */
export interface ValidationCaseDraft {
  take_id: string;
  baseline_ids: string[];
  mode: ValidationResolutionMode;
  pinned_baseline_id: string;
  allowed_baseline_ids: string[];
  enabled: boolean;
  tags: string[];
  notes: string;
}

export interface ResolutionSummary {
  /** Compact label for the case table, e.g. "Pinned · v1". */
  label: string;
  /** Why the label says what it says, for a tooltip. */
  detail: string;
  severity: "success" | "info" | "warning" | "danger" | "muted";
}

const versionLabel = (baseline: ValidationBaseline | undefined, fallback: string): string =>
  baseline ? `v${baseline.version}` : fallback;

const byId = (baselines: ValidationBaseline[]): Map<string, ValidationBaseline> =>
  new Map(baselines.map((baseline) => [baseline.id, baseline]));

/** Same artifact identity: take, pipeline, stage and artifact must all agree. */
export function isSameFamily(a: ValidationBaseline, b: ValidationBaseline): boolean {
  return (
    a.take_id === b.take_id &&
    a.pipeline_id === b.pipeline_id &&
    (a.stage_id ?? null) === (b.stage_id ?? null) &&
    (a.artifact_id ?? null) === (b.artifact_id ?? null)
  );
}

/**
 * Baselines a case may legally reference, mirroring the server's compatibility rule.
 * Selectors are built from this so an incompatible version is never offered.
 */
export function compatibleBaselines(
  baselines: ValidationBaseline[],
  caseItem: Pick<ValidationSuiteCase, "take_id" | "baseline_ids">,
  pipelineId: string,
): ValidationBaseline[] {
  const configured = baselines.filter((baseline) => caseItem.baseline_ids.includes(baseline.id));
  return baselines
    .filter(
      (baseline) => baseline.take_id === caseItem.take_id && baseline.pipeline_id === pipelineId,
    )
    .filter(
      (baseline) => !configured.length || configured.some((other) => isSameFamily(baseline, other)),
    )
    .slice()
    .sort((a, b) => b.version - a.version);
}

/** The version an active-mode case resolves to: the highest active one in its family. */
export function activeVersionFor(
  baselines: ValidationBaseline[],
  configuredId: string,
): ValidationBaseline | undefined {
  const configured = byId(baselines).get(configuredId);
  if (!configured) return undefined;
  return baselines
    .filter((baseline) => baseline.status === "active" && isSameFamily(baseline, configured))
    .sort((a, b) => b.version - a.version)[0];
}

/**
 * How a case's configured resolution reads today.  Present governance only — a
 * historical execution reports the versions it actually used, and must not be
 * described through this.
 */
export function describeResolution(
  caseItem: ValidationSuiteCase,
  baselines: ValidationBaseline[],
  matchedBaselineId?: string | null,
): ResolutionSummary {
  const lookup = byId(baselines);
  const resolution: ValidationBaselineResolution = caseItem.baseline_resolution ?? {
    mode: "active",
  };

  if (resolution.mode === "pinned") {
    const pinnedId = resolution.baseline_id ?? "";
    const pinned = lookup.get(pinnedId);
    if (!pinned) {
      return {
        label: "Pinned · unavailable",
        detail: `This case is pinned to baseline ${pinnedId || "(none)"}, which no longer exists. The pin is kept; pick another version to change it.`,
        severity: "danger",
      };
    }
    if (pinned.integrity_state === "invalid") {
      return {
        label: `Pinned · corrupt v${pinned.version}`,
        detail: `Version ${pinned.version} failed integrity verification. Its snapshot does not match the recorded checksum.`,
        severity: "danger",
      };
    }
    if (pinned.status !== "active") {
      return {
        label: `Pinned · inactive v${pinned.version}`,
        detail: `Version ${pinned.version} is inactive. The case stays pinned to it and will not fall back to the active version.`,
        severity: "warning",
      };
    }
    return {
      label: `Pinned · v${pinned.version}`,
      detail: `Always compared against version ${pinned.version}, whichever version is active.`,
      severity: "info",
    };
  }

  if (resolution.mode === "allowed_versions") {
    const allowed = (resolution.allowed_baseline_ids ?? []).map((id) => lookup.get(id));
    if (!allowed.length) {
      return {
        label: "Allowed · none configured",
        detail: "No versions are configured. Add at least one allowed version.",
        severity: "danger",
      };
    }
    const versions = allowed
      .map((baseline, index) =>
        versionLabel(baseline, `(missing ${resolution.allowed_baseline_ids?.[index] ?? ""})`),
      )
      .join(", ");
    const matched = matchedBaselineId ? lookup.get(matchedBaselineId) : undefined;
    if (matchedBaselineId && !matched) {
      return {
        label: `Allowed · ${versions}`,
        detail: `Any of ${versions} counts as a pass. The version matched in the selected execution is no longer in the history.`,
        severity: "info",
      };
    }
    return {
      label: matched
        ? `Allowed · ${versions} · matched v${matched.version}`
        : `Allowed · ${versions}`,
      detail: matched
        ? `Any of ${versions} counts as a pass. The selected execution matched version ${matched.version}.`
        : `Any of ${versions} counts as a pass. Newly promoted versions are not added automatically.`,
      severity: matched ? "success" : "info",
    };
  }

  const resolved = caseItem.baseline_ids
    .map((id) => activeVersionFor(baselines, id))
    .filter((baseline): baseline is ValidationBaseline => Boolean(baseline));
  if (!resolved.length) {
    return {
      label: "Active · unavailable",
      detail:
        "No active approved baseline covers this case. Open baseline history to activate one.",
      severity: "danger",
    };
  }
  const versions = [...new Set(resolved.map((baseline) => `v${baseline.version}`))].join(", ");
  return {
    label: `Active · ${versions}`,
    detail: `Follows whichever version is active. Right now that is ${versions}.`,
    severity: "success",
  };
}

/** Archived suites stay inspectable; the server rejects every case mutation on them. */
export function canMutateCases(suiteStatus: string | undefined): boolean {
  return suiteStatus !== "archived";
}

/** Removal drops suite membership only, and the wording has to say so. */
export function caseRemovalWarning(caseItem: Pick<ValidationSuiteCase, "take_id">): string {
  return (
    `Remove ${caseItem.take_id} from this suite? ` +
    "Its approved baselines, source runs and past executions are all kept — only the suite membership goes away."
  );
}

export function draftFromCase(caseItem: ValidationSuiteCase): ValidationCaseDraft {
  const resolution = caseItem.baseline_resolution ?? { mode: "active" as const };
  return {
    take_id: caseItem.take_id,
    baseline_ids: [...caseItem.baseline_ids],
    mode: resolution.mode ?? "active",
    pinned_baseline_id: resolution.baseline_id ?? caseItem.baseline_ids[0] ?? "",
    allowed_baseline_ids: [...(resolution.allowed_baseline_ids ?? caseItem.baseline_ids)],
    enabled: caseItem.enabled,
    tags: [...caseItem.tags],
    notes: caseItem.notes ?? "",
  };
}

/**
 * Reasons the draft cannot be saved, phrased for the person reading them.
 * These mirror the server's own refusals so the round trip is not the first
 * time the user hears about a problem.
 */
export function draftProblems(draft: ValidationCaseDraft): string[] {
  const problems: string[] = [];
  if (!draft.take_id.trim()) problems.push("Choose a take for this case.");
  if (!draft.baseline_ids.length) problems.push("Choose at least one approved reference.");
  if (draft.mode === "pinned" && !draft.pinned_baseline_id) {
    problems.push("Choose the version to pin to.");
  }
  if (draft.mode === "allowed_versions") {
    if (!draft.allowed_baseline_ids.length)
      problems.push("Select at least one allowed baseline version.");
    if (new Set(draft.allowed_baseline_ids).size !== draft.allowed_baseline_ids.length) {
      problems.push("The same version is listed more than once.");
    }
  }
  return problems;
}

function resolutionOf(draft: ValidationCaseDraft): ValidationBaselineResolution {
  if (draft.mode === "pinned") return { mode: "pinned", baseline_id: draft.pinned_baseline_id };
  if (draft.mode === "allowed_versions") {
    return { mode: "allowed_versions", allowed_baseline_ids: [...draft.allowed_baseline_ids] };
  }
  return { mode: "active" };
}

export function buildCaseRequest(draft: ValidationCaseDraft) {
  return {
    take_id: draft.take_id,
    baseline_ids: [...draft.baseline_ids],
    baseline_resolution: resolutionOf(draft),
    enabled: draft.enabled,
    tags: [...draft.tags],
    notes: draft.notes,
  };
}

/**
 * Only what the user actually changed.  The server preserves omitted fields, so
 * sending a full record would overwrite anything another session edited meanwhile.
 */
export function buildCaseUpdate(draft: ValidationCaseDraft, original: ValidationSuiteCase) {
  const update: Record<string, unknown> = {};
  if (draft.enabled !== original.enabled) update.enabled = draft.enabled;
  if (draft.notes !== (original.notes ?? "")) update.notes = draft.notes;
  if (draft.tags.join(" ") !== original.tags.join(" ")) update.tags = [...draft.tags];
  const next = resolutionOf(draft);
  const current = original.baseline_resolution ?? { mode: "active" as const };
  if (JSON.stringify(next) !== JSON.stringify(normaliseResolution(current))) {
    update.baseline_resolution = next;
  }
  return update;
}

function normaliseResolution(
  resolution: ValidationBaselineResolution,
): ValidationBaselineResolution {
  if (resolution.mode === "pinned")
    return { mode: "pinned", baseline_id: resolution.baseline_id ?? "" };
  if (resolution.mode === "allowed_versions") {
    return {
      mode: "allowed_versions",
      allowed_baseline_ids: [...(resolution.allowed_baseline_ids ?? [])],
    };
  }
  return { mode: "active" };
}

export function parseTags(value: string): string[] {
  return [
    ...new Set(
      value
        .split(",")
        .map((tag) => tag.trim())
        .filter(Boolean),
    ),
  ];
}

/**
 * Turn a server refusal into something a person can act on.  The backend answers
 * with a status code and a plain message rather than typed error codes, so this
 * matches on the message and always falls back to showing what was said.
 */
export function describeApiError(error: unknown): string {
  const raw = error instanceof Error ? error.message : String(error);
  const rules: Array<[RegExp, string]> = [
    [
      /archived suite cannot be changed/i,
      "This suite is archived. Restore it before editing cases.",
    ],
    [
      /baseline is incompatible with case/i,
      "That version belongs to a different take, pipeline or artifact.",
    ],
    [/allowed_versions requires at least one/i, "Select at least one allowed baseline version."],
    [/allowed_versions requires unique/i, "The same version is listed more than once."],
    [/pinned resolution requires a baseline version/i, "Choose the version to pin to."],
    [
      /pipeline does not match/i,
      "That reference was approved for a different pipeline than this suite.",
    ],
    [/invalid baseline resolution mode/i, "Choose active, pinned or allowed versions."],
    [/unknown baseline/i, "That reference no longer exists."],
    [/^404\b/, "That record no longer exists. Reload the suite."],
  ];
  for (const [pattern, message] of rules) {
    if (pattern.test(raw)) return message;
  }
  return raw.replace(/^\d{3} [A-Za-z ]+: /, "");
}
