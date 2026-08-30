import test from "node:test";
import assert from "node:assert/strict";

import type { ValidationBaseline, ValidationSuiteCase } from "../src/api/client.ts";
import {
  buildCaseRequest,
  buildCaseUpdate,
  canMutateCases,
  caseRemovalWarning,
  compatibleBaselines,
  describeApiError,
  describeResolution,
  draftFromCase,
  draftProblems,
  parseTags,
} from "../src/components/validation/validationCaseModel.ts";

const baseline = (
  over: Partial<ValidationBaseline> & { id: string; version: number },
): ValidationBaseline => ({
  status: "inactive",
  take_id: "take_1",
  pipeline_id: "mining_steel_ball_classification_25d",
  source_run_id: "run_1",
  stage_id: "segment",
  artifact_id: "final_object_mask",
  ...over,
});

const V1 = baseline({ id: "b1", version: 1 });
const V2 = baseline({ id: "b2", version: 2, status: "active" });
const OTHER_TAKE = baseline({ id: "b9", version: 1, take_id: "take_2", status: "active" });
const OTHER_ARTIFACT = baseline({
  id: "b8",
  version: 1,
  artifact_id: "belt_plane",
  status: "active",
});

const suiteCase = (over: Partial<ValidationSuiteCase> = {}): ValidationSuiteCase => ({
  id: "case_1",
  take_id: "take_1",
  baseline_ids: ["b1"],
  enabled: true,
  tags: [],
  notes: "",
  ...over,
});

// --- resolution display -----------------------------------------------------

test("active mode reports the version it actually resolves to, not the configured one", () => {
  const summary = describeResolution(suiteCase(), [V1, V2]);
  assert.equal(summary.label, "Active · v2");
  assert.equal(summary.severity, "success");
});

test("active mode is explicit when no active baseline covers the case", () => {
  const summary = describeResolution(suiteCase(), [V1]);
  assert.equal(summary.label, "Active · unavailable");
  assert.equal(summary.severity, "danger");
  assert.match(summary.detail, /baseline history/i);
});

test("a pinned case stays on its version and says so when that version is inactive", () => {
  const pinned = suiteCase({
    baseline_ids: ["b1", "b2"],
    baseline_resolution: { mode: "pinned", baseline_id: "b1" },
  });
  const summary = describeResolution(pinned, [V1, V2]);
  assert.equal(summary.label, "Pinned · inactive v1");
  assert.equal(summary.severity, "warning");
  assert.match(summary.detail, /will not fall back/i);
});

test("a pinned case reports a missing or corrupt version distinctly", () => {
  const missing = describeResolution(
    suiteCase({ baseline_resolution: { mode: "pinned", baseline_id: "gone" } }),
    [V1, V2],
  );
  assert.equal(missing.label, "Pinned · unavailable");
  assert.match(missing.detail, /pin is kept/i);

  const corrupt = describeResolution(
    suiteCase({ baseline_resolution: { mode: "pinned", baseline_id: "b2" } }),
    [V1, { ...V2, integrity_state: "invalid" }],
  );
  assert.equal(corrupt.label, "Pinned · corrupt v2");
  assert.equal(corrupt.severity, "danger");
});

test("allowed versions list their versions and the one a given execution matched", () => {
  const allowed = suiteCase({
    baseline_ids: ["b1", "b2"],
    baseline_resolution: { mode: "allowed_versions", allowed_baseline_ids: ["b1", "b2"] },
  });
  // Configured order, not sorted: it reflects what the user selected.
  assert.equal(describeResolution(allowed, [V1, V2]).label, "Allowed · v1, v2");
  assert.equal(describeResolution(allowed, [V1, V2], "b2").label, "Allowed · v1, v2 · matched v2");
  assert.match(describeResolution(allowed, [V1, V2]).detail, /not added automatically/i);
});

test("an empty allowed list is reported as a problem rather than rendered as active", () => {
  const empty = suiteCase({
    baseline_resolution: { mode: "allowed_versions", allowed_baseline_ids: [] },
  });
  const summary = describeResolution(empty, [V1, V2]);
  assert.equal(summary.label, "Allowed · none configured");
  assert.equal(summary.severity, "danger");
});

test("a legacy case with no resolution is shown as active mode", () => {
  assert.equal(describeResolution(suiteCase(), [V1, V2]).label, "Active · v2");
});

// --- selectors --------------------------------------------------------------

test("selectors only offer versions the server would accept", () => {
  const offered = compatibleBaselines(
    [V1, V2, OTHER_TAKE, OTHER_ARTIFACT],
    suiteCase({ baseline_ids: ["b1"] }),
    "mining_steel_ball_classification_25d",
  );
  assert.deepEqual(
    offered.map((b) => b.id),
    ["b2", "b1"],
    "another take and another artifact family are both excluded, newest first",
  );
});

test("a case with no references yet may choose any version of its take", () => {
  const offered = compatibleBaselines(
    [V1, V2, OTHER_ARTIFACT, OTHER_TAKE],
    suiteCase({ baseline_ids: [] }),
    "mining_steel_ball_classification_25d",
  );
  assert.deepEqual(offered.map((b) => b.id).sort(), ["b1", "b2", "b8"]);
});

// --- editing ----------------------------------------------------------------

test("archived suites block every case mutation", () => {
  assert.equal(canMutateCases("active"), true);
  assert.equal(canMutateCases("archived"), false);
});

test("the removal confirmation says what is kept, not just what goes", () => {
  const warning = caseRemovalWarning({ take_id: "take_1" });
  assert.match(warning, /baselines, source runs and past executions are all kept/i);
  assert.match(warning, /only the suite membership/i);
});

test("an empty or duplicated allowed selection is refused before the request is sent", () => {
  const draft = draftFromCase(
    suiteCase({ baseline_resolution: { mode: "allowed_versions", allowed_baseline_ids: [] } }),
  );
  assert.deepEqual(draftProblems({ ...draft, allowed_baseline_ids: [] }), [
    "Select at least one allowed baseline version.",
  ]);
  assert.deepEqual(draftProblems({ ...draft, allowed_baseline_ids: ["b1", "b1"] }), [
    "The same version is listed more than once.",
  ]);
  assert.deepEqual(draftProblems({ ...draft, allowed_baseline_ids: ["b1", "b2"] }), []);
});

test("an update sends only what changed, so concurrent edits are not overwritten", () => {
  const original = suiteCase({ tags: ["nightly"], notes: "watch" });
  const draft = draftFromCase(original);
  assert.deepEqual(buildCaseUpdate(draft, original), {}, "an untouched draft sends nothing");
  assert.deepEqual(buildCaseUpdate({ ...draft, enabled: false }, original), { enabled: false });
  assert.deepEqual(buildCaseUpdate({ ...draft, tags: ["paused"] }, original), { tags: ["paused"] });
});

test("switching a legacy case to pinned is detected as a change", () => {
  const original = suiteCase({ baseline_ids: ["b1", "b2"] });
  const draft = draftFromCase(original);
  assert.deepEqual(
    buildCaseUpdate({ ...draft, mode: "pinned", pinned_baseline_id: "b1" }, original),
    {
      baseline_resolution: { mode: "pinned", baseline_id: "b1" },
    },
  );
});

test("a new case is sent with its resolution already resolved", () => {
  const draft = draftFromCase(suiteCase({ baseline_ids: ["b1", "b2"] }));
  const request = buildCaseRequest({
    ...draft,
    mode: "allowed_versions",
    allowed_baseline_ids: ["b1", "b2"],
  });
  assert.deepEqual(request.baseline_resolution, {
    mode: "allowed_versions",
    allowed_baseline_ids: ["b1", "b2"],
  });
  assert.deepEqual(request.baseline_ids, ["b1", "b2"]);
});

test("tags are trimmed and de-duplicated", () => {
  assert.deepEqual(parseTags(" nightly , smoke ,nightly, "), ["nightly", "smoke"]);
  assert.deepEqual(parseTags(""), []);
});

// --- errors -----------------------------------------------------------------

test("server refusals become sentences the reader can act on", () => {
  assert.equal(
    describeApiError(new Error("422 Unprocessable Entity: archived suite cannot be changed")),
    "This suite is archived. Restore it before editing cases.",
  );
  assert.equal(
    describeApiError(new Error("422 Unprocessable Entity: baseline is incompatible with case")),
    "That version belongs to a different take, pipeline or artifact.",
  );
  assert.equal(
    describeApiError(
      new Error(
        "422 Unprocessable Entity: allowed_versions requires at least one baseline version",
      ),
    ),
    "Select at least one allowed baseline version.",
  );
});

test("an unrecognised refusal still shows what the server said, minus the status noise", () => {
  assert.equal(
    describeApiError(new Error("422 Unprocessable Entity: something new went wrong")),
    "something new went wrong",
  );
});
