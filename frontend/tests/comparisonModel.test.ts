import test from "node:test";
import assert from "node:assert/strict";

import { comparisonHasAnyChanges, comparisonSourceLabel, formatChangeSummary, orderedComparisonUnits } from "../src/components/comparisonModel.ts";

test("orderedComparisonUnits prioritizes changed units and stable order", () => {
  const ordered = orderedComparisonUnits({
    comparison_id: "cmp_1",
    pipeline_id: "mining_steel_ball_classification_25d",
    left: { type: "recipe" },
    right: { type: "recipe" },
    summary: {
      parameter_changes: 1,
      artifact_changes: 0,
      metric_changes: 0,
      classification_changed: false,
      warnings_changed: false,
      key_affected_units: ["b"],
    },
    units: {
      b: { label: "B", stage_id: "s", kind: "substage", order: 2, has_changes: true, parameter_diff: [], artifact_diff: [], metric_diff: [], diagnostic_diff: [] },
      a: { label: "A", stage_id: "s", kind: "substage", order: 1, has_changes: false, parameter_diff: [], artifact_diff: [], metric_diff: [], diagnostic_diff: [] },
    },
  });

  assert.equal(ordered[0]?.[0], "b");
  assert.equal(ordered[1]?.[0], "a");
  assert.equal(comparisonHasAnyChanges({
    comparison_id: "cmp_1",
    pipeline_id: "mining_steel_ball_classification_25d",
    left: { type: "recipe" },
    right: { type: "recipe" },
    summary: {
      parameter_changes: 1,
      artifact_changes: 0,
      metric_changes: 0,
      classification_changed: false,
      warnings_changed: false,
      key_affected_units: ["b"],
    },
    units: {
      b: { label: "B", stage_id: "s", kind: "substage", order: 2, has_changes: true, parameter_diff: [], artifact_diff: [], metric_diff: [], diagnostic_diff: [] },
    },
  }), true);
});

test("formatChangeSummary summarizes dominant comparison changes", () => {
  const line = formatChangeSummary({
    top_changed_unit: { label: "Segmentation", mask_changed_percent: 12.5, parameter_changes: [] },
    top_changed_artifact: { label: "final_object_mask" },
    classification_changed: true,
    object_count_changed: false,
    warnings_changed: false,
  });
  assert.match(line ?? "", /Segmentation mask 12\.5% changed/);
  assert.match(line ?? "", /classification changed/);
});

test("comparisonSourceLabel labels parent and child runs given lineage context", () => {
  const context = { parentRunId: "run_parent", childRunIds: ["run_child_1"] };
  assert.equal(comparisonSourceLabel({ type: "run", run_id: "run_parent" }, context), "Parent full run (run_parent)");
  assert.equal(comparisonSourceLabel({ type: "run", run_id: "run_child_1" }, context), "Child partial rerun (run_child_1)");
  assert.equal(comparisonSourceLabel({ type: "run", run_id: "run_other" }, context), "run run_other");
});

test("comparisonSourceLabel falls back to generic labels without lineage context", () => {
  assert.equal(comparisonSourceLabel({ type: "run", run_id: "run_x" }), "run run_x");
  assert.equal(comparisonSourceLabel({ type: "recipe", recipe_id: "r1", recipe_version: 2 }), "recipe r1 v2");
  assert.equal(comparisonSourceLabel(undefined), "unknown");
});
