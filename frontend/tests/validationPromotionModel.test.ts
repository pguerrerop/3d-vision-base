import test from "node:test";
import assert from "node:assert/strict";

import type {
  ValidationBaseline,
  ValidationPromotionResponse,
  ValidationResolutionImpact,
} from "../src/api/client.ts";
import {
  describePromotionOutcome,
  expectedNextVersion,
  summarisePromotionImpact,
} from "../src/components/validation/validationPromotionModel.ts";

const baseline = (
  over: Partial<ValidationBaseline> & { id: string; version: number },
): ValidationBaseline => ({
  status: "inactive",
  take_id: "take_1",
  pipeline_id: "p",
  source_run_id: "run_1",
  ...over,
});

const noImpact: ValidationResolutionImpact = {
  active_cases: [],
  pinned_cases_unchanged: [],
  allowed_version_cases_unchanged: [],
  disabled_cases: [],
  archived_suites: [],
};
const ref = (id: string) => ({ suite_id: "s1", case_id: id });

test("expected next version is the source's version plus one", () => {
  assert.equal(expectedNextVersion(baseline({ id: "b1", version: 3 })), 4);
});

// --- impact wording -----------------------------------------------------

test("active-case impact wording depends on whether this promotion will activate", () => {
  const impact = { ...noImpact, active_cases: [ref("c1"), ref("c2")] };
  assert.deepEqual(summarisePromotionImpact(impact, true), [
    "2 active-resolution cases will follow the new version",
  ]);
  assert.deepEqual(summarisePromotionImpact(impact, false), [
    "2 active-resolution cases would follow it once it is activated",
  ]);
});

test("pinned, allowed and disabled cases are always reported unaffected, regardless of activation", () => {
  const impact = {
    ...noImpact,
    pinned_cases_unchanged: [ref("c1")],
    disabled_cases: [ref("c2"), ref("c3")],
  };
  const lines = summarisePromotionImpact(impact, true);
  assert.ok(
    lines.some((l) => l.includes("1 case stays pinned to its own version and is unaffected")),
  );
  assert.ok(lines.some((l) => l.includes("2 cases are disabled and will be skipped either way")));
});

test("a row whose count coincidentally matches another row's count keeps its own wording", () => {
  // Each row carries its own noun/verb pair rather than being looked up by
  // matching count values against another row, so two rows sharing a count
  // (both 1, here) cannot bleed into each other's phrasing.
  const impact = { ...noImpact, active_cases: [ref("c1")], disabled_cases: [ref("c2")] };
  const lines = summarisePromotionImpact(impact, false);
  assert.deepEqual(lines, [
    "1 active-resolution case would follow it once it is activated",
    "1 case is disabled and will be skipped either way",
  ]);
});

test("buckets with nothing in them produce no line at all", () => {
  assert.deepEqual(summarisePromotionImpact(noImpact, true), []);
});

// --- outcome wording ------------------------------------------------------

test("a genuinely new promotion reads as created, not as a generic success", () => {
  const response: ValidationPromotionResponse = {
    baseline: baseline({ id: "b2", version: 2, status: "active" }),
    already_promoted: false,
    previous_active_baseline: null,
    resulting_active_baseline: null,
    affected_case_resolution: noImpact,
  };
  const outcome = describePromotionOutcome(response);
  assert.equal(outcome.headline, "Created baseline v2.");
  assert.match(outcome.detail, /active-resolution cases now follow it/);
  assert.equal(outcome.needsActivation, false);
});

test("an idempotent retry never reads like a fresh success", () => {
  const response: ValidationPromotionResponse = {
    baseline: baseline({ id: "b3", version: 3, status: "active" }),
    already_promoted: true,
    previous_active_baseline: null,
    resulting_active_baseline: null,
    affected_case_resolution: noImpact,
  };
  const outcome = describePromotionOutcome(response);
  assert.equal(outcome.headline, "This candidate was already promoted as baseline v3.");
  assert.match(outcome.detail, /No new version was created/);
});

test("an inactive result, new or reused, asks for explicit activation", () => {
  const created: ValidationPromotionResponse = {
    baseline: baseline({ id: "b4", version: 4, status: "inactive" }),
    already_promoted: false,
    previous_active_baseline: null,
    resulting_active_baseline: null,
    affected_case_resolution: noImpact,
  };
  const reused: ValidationPromotionResponse = { ...created, already_promoted: true };
  assert.equal(describePromotionOutcome(created).needsActivation, true);
  assert.equal(describePromotionOutcome(reused).needsActivation, true);
  assert.match(describePromotionOutcome(created).detail, /still follow the previous version/);
  assert.match(describePromotionOutcome(reused).detail, /This version is not active/);
});
