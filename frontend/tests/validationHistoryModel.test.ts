import test from "node:test";
import assert from "node:assert/strict";

import type {
  ValidationBaseline,
  ValidationResolutionImpact,
  ValidationSuiteCase,
} from "../src/api/client.ts";
import {
  canAttachToCase,
  describeIntegrity,
  orderHistory,
  previewActivation,
  previewDeactivation,
  summariseImpact,
  supersededByLabel,
} from "../src/components/validation/validationHistoryModel.ts";

const baseline = (
  over: Partial<ValidationBaseline> & { id: string; version: number },
): ValidationBaseline => ({
  status: "inactive",
  take_id: "take_1",
  pipeline_id: "mining_steel_ball_classification_25d",
  source_run_id: "run_1",
  stage_id: "segment",
  artifact_id: "final_object_mask",
  integrity_state: "ok",
  ...over,
});

const V1 = baseline({ id: "b1", version: 1, created_at: "2026-08-01T00:00:00Z" });
const V2 = baseline({
  id: "b2",
  version: 2,
  created_at: "2026-08-02T00:00:00Z",
  supersedes_baseline_id: "b1",
});
const V3 = baseline({
  id: "b3",
  version: 3,
  status: "active",
  created_at: "2026-08-03T00:00:00Z",
  supersedes_baseline_id: "b2",
});
const OTHER_FAMILY = baseline({
  id: "bx",
  version: 1,
  artifact_id: "belt_plane",
  status: "active",
});

const noImpact: ValidationResolutionImpact = {
  active_cases: [],
  pinned_cases_unchanged: [],
  allowed_version_cases_unchanged: [],
  disabled_cases: [],
  archived_suites: [],
};

const ref = (id: string) => ({ suite_id: "suite_1", case_id: id });

// --- integrity --------------------------------------------------------------

test("integrity distinguishes verified, failed and not reported", () => {
  assert.equal(describeIntegrity(V1).label, "Verified");
  assert.equal(describeIntegrity(V1).usable, true);

  const failed = describeIntegrity({ ...V1, integrity_state: "invalid" });
  assert.equal(failed.label, "Failed");
  assert.equal(failed.usable, false);
  assert.equal(failed.severity, "danger");

  // Absent means the listing did not report it, which is not the same as fine.
  const unknown = describeIntegrity({ ...V1, integrity_state: undefined });
  assert.equal(unknown.label, "Not checked");
  assert.equal(unknown.severity, "muted");
});

// --- ordering and lineage ---------------------------------------------------

test("history reads newest first", () => {
  assert.deepEqual(
    orderHistory([V1, V3, V2]).map((b) => b.id),
    ["b3", "b2", "b1"],
  );
});

test("each version shows what superseded it, by version rather than by id", () => {
  const history = [V3, V2, V1];
  assert.equal(supersededByLabel(V1, history), "v2");
  assert.equal(supersededByLabel(V2, history), "v3");
  assert.equal(supersededByLabel(V3, history), null, "the newest version is superseded by nothing");
});

// --- activation -------------------------------------------------------------

test("activation lists what is active now, because activating does not deactivate", () => {
  const preview = previewActivation(V1, [V1, V2, V3]);
  assert.deepEqual(
    preview.currentlyActive.map((b) => b.id),
    ["b3"],
  );
  assert.equal(preview.noop, false);
  assert.equal(preview.blocked, null);
});

test("activating an older version warns that it will not change what cases resolve to", () => {
  // The server allows several active versions at once; the highest one wins.
  const preview = previewActivation(V1, [V1, V2, V3]);
  assert.equal(preview.resolvesToAfter.id, "b3", "v3 stays active and is newer");
  assert.match(String(preview.ineffective), /v3 is also active and is newer/i);
  assert.match(String(preview.ineffective), /Deactivate v3 if you want v1 to take effect/i);
});

test("activating the newest version does take effect, so no warning is given", () => {
  const preview = previewActivation(V2, [V1, V2, { ...V3, status: "inactive" }]);
  assert.equal(preview.resolvesToAfter.id, "b2");
  assert.equal(preview.ineffective, null);
});

test("activating the version that is already active is a no-op, not an action", () => {
  assert.equal(previewActivation(V3, [V1, V2, V3]).noop, true);
});

test("a family with nothing active reports none rather than guessing", () => {
  assert.deepEqual(previewActivation(V1, [V1, V2]).currentlyActive, []);
});

test("another artifact family is never counted as active for this one", () => {
  const preview = previewActivation(V1, [V1, V2, OTHER_FAMILY]);
  assert.deepEqual(preview.currentlyActive, [], "belt_plane is a different family");
  assert.equal(preview.resolvesToAfter.id, "b1");
});

test("a version that failed integrity cannot be activated", () => {
  const preview = previewActivation({ ...V1, integrity_state: "invalid" }, [V1, V3]);
  assert.match(String(preview.blocked), /does not match its recorded checksum/i);
});

// --- deactivation -----------------------------------------------------------

test("deactivating the last active version warns that active cases are stranded", () => {
  const warning = previewDeactivation(V3, [V1, V2, V3], { ...noImpact, active_cases: [ref("c1")] });
  assert.equal(warning.strandedActiveCases, 1);
  assert.match(String(warning.message), /no other active version to fall back to/i);
});

test("deactivating is silent about active cases when another version stays active", () => {
  const history = [V1, { ...V2, status: "active" }, V3];
  const warning = previewDeactivation(V3, history, { ...noImpact, active_cases: [ref("c1")] });
  assert.equal(warning.strandedActiveCases, 0);
  assert.equal(warning.message, null);
});

test("pinned and allowed cases are reported as unchanged, not as breakage", () => {
  const warning = previewDeactivation(V3, [V1, V2, V3], {
    ...noImpact,
    pinned_cases_unchanged: [ref("c2")],
    allowed_version_cases_unchanged: [ref("c3"), ref("c4")],
  });
  assert.match(String(warning.message), /1 pinned case keeps referencing this version/i);
  assert.match(String(warning.message), /2 allowed-version cases keep it/i);
  assert.match(String(warning.message), /configuration is not changed/i);
});

// --- impact -----------------------------------------------------------------

test("impact is summarised as counts, and buckets with nothing in them are omitted", () => {
  const lines = summariseImpact({
    ...noImpact,
    active_cases: [ref("c1"), ref("c2")],
    pinned_cases_unchanged: [ref("c3")],
    archived_suites: [ref("c4")],
  });
  assert.deepEqual(lines, [
    "2 cases follow this activation",
    "1 case stays pinned to its own version",
    "1 case belongs to an archived suite",
  ]);
  assert.deepEqual(summariseImpact(noImpact), []);
});

// --- attaching to a case ----------------------------------------------------

test("a version can only be attached to a case of the same take and family", () => {
  const caseItem: ValidationSuiteCase = {
    id: "case_1",
    take_id: "take_1",
    baseline_ids: ["b1"],
    enabled: true,
    tags: [],
    notes: "",
  };
  const history = [V1, V2, V3, OTHER_FAMILY];
  assert.equal(canAttachToCase(V2, caseItem, history), true);
  assert.equal(canAttachToCase(OTHER_FAMILY, caseItem, history), false);
  assert.equal(
    canAttachToCase({ ...V2, take_id: "take_2" }, caseItem, history),
    false,
    "another take is never attachable",
  );
});
