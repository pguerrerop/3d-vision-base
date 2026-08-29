import test from "node:test";
import assert from "node:assert/strict";

import { computeStaleWarnings } from "../src/components/staleWarningsModel.ts";

const baseInputs = {
  planExists: false,
  planIsStale: false,
  planContext: null,
  currentRecipeId: null,
  currentRecipeVersion: null,
  activeRunId: null,
  activeRunExecutionMode: null,
  activeRunParentRunId: null,
  activeRunComparisonId: null,
  activeRunContractFingerprint: null,
  currentContractFingerprint: null,
};

test("computeStaleWarnings returns nothing when there is no plan and no active child run", () => {
  assert.deepEqual(computeStaleWarnings(baseInputs), []);
});

test("computeStaleWarnings flags stale current edits", () => {
  const warnings = computeStaleWarnings({ ...baseInputs, planExists: true, planIsStale: true });
  assert.ok(warnings.some((line) => line.includes("Current edits have changed")));
});

test("computeStaleWarnings flags recipe changed since plan", () => {
  const warnings = computeStaleWarnings({
    ...baseInputs,
    planExists: true,
    planContext: { parentRunId: null, recipeId: "r1", recipeVersion: 1 },
    currentRecipeId: "r1",
    currentRecipeVersion: 2,
  });
  assert.ok(warnings.some((line) => line.includes("Selected recipe changed")));
});

test("computeStaleWarnings flags active run changed since plan", () => {
  const warnings = computeStaleWarnings({
    ...baseInputs,
    planExists: true,
    planContext: { parentRunId: "run_a", recipeId: null, recipeVersion: null },
    activeRunId: "run_b",
  });
  assert.ok(warnings.some((line) => line.includes("Active run has changed since this plan was created (planned against run_a)")));
});

test("computeStaleWarnings flags active child run with a different parent than the plan", () => {
  const warnings = computeStaleWarnings({
    ...baseInputs,
    planExists: true,
    planContext: { parentRunId: "run_a", recipeId: null, recipeVersion: null },
    activeRunExecutionMode: "rerun_from_public_stage_boundary",
    activeRunParentRunId: "run_z",
  });
  assert.ok(warnings.some((line) => line.includes("different parent than the current plan")));
});

test("computeStaleWarnings flags missing comparison for a child run", () => {
  const warnings = computeStaleWarnings({
    ...baseInputs,
    activeRunExecutionMode: "rerun_from_public_stage_boundary",
    activeRunComparisonId: null,
  });
  assert.ok(warnings.some((line) => line.includes("Comparison to parent is missing")));
});

test("computeStaleWarnings does not flag missing comparison when one is present", () => {
  const warnings = computeStaleWarnings({
    ...baseInputs,
    activeRunExecutionMode: "rerun_from_public_stage_boundary",
    activeRunComparisonId: "cmp_1",
  });
  assert.ok(!warnings.some((line) => line.includes("Comparison to parent is missing")));
});

test("computeStaleWarnings flags contract fingerprint drift for a child run", () => {
  const warnings = computeStaleWarnings({
    ...baseInputs,
    activeRunExecutionMode: "rerun_from_public_stage_boundary",
    activeRunComparisonId: "cmp_1",
    activeRunContractFingerprint: "fp_old",
    currentContractFingerprint: "fp_new",
  });
  assert.ok(warnings.some((line) => line.includes("contract fingerprint has changed")));
});

test("computeStaleWarnings ignores full-run-only checks when the active run is a full run", () => {
  const warnings = computeStaleWarnings({
    ...baseInputs,
    activeRunExecutionMode: "full_run",
    activeRunComparisonId: null,
  });
  assert.deepEqual(warnings, []);
});
