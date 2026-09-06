import test from "node:test";
import assert from "node:assert/strict";

import type {
  ValidationStageComparatorSummary,
  ValidationStageSummaryEntry,
} from "../src/api/client.ts";
import {
  fieldRows,
  regressionRate,
  sortedStatusEntries,
  stageHasData,
} from "../src/components/validation/validationStageSummaryModel.ts";

const bucket = (
  over: Partial<ValidationStageComparatorSummary> = {},
): ValidationStageComparatorSummary => ({
  comparator: "binary_mask",
  count: 3,
  status_counts: { pass: 3 },
  metrics: {},
  ...over,
});

test("status entries read severity-first, not in JSON key order", () => {
  const counts = { pass: 5, regression: 2, changed: 1 };
  assert.deepEqual(sortedStatusEntries(counts), [
    ["regression", 2],
    ["changed", 1],
    ["pass", 5],
  ]);
});

test("a status this vocabulary doesn't know about still appears, alphabetically after the known ones", () => {
  const counts = { pass: 1, some_future_status: 4, regression: 1 };
  assert.deepEqual(sortedStatusEntries(counts), [
    ["regression", 1],
    ["pass", 1],
    ["some_future_status", 4],
  ]);
});

test("regression rate divides regressions by the bucket total, not by any other count", () => {
  assert.equal(
    regressionRate(bucket({ count: 4, status_counts: { regression: 1, pass: 3 } })),
    0.25,
  );
  assert.equal(regressionRate(bucket({ count: 4, status_counts: { pass: 4 } })), 0);
});

test("regression rate is null rather than a division by zero for an empty bucket", () => {
  assert.equal(regressionRate(bucket({ count: 0, status_counts: {} })), null);
});

test("a stage with no comparators reports no data", () => {
  const empty: ValidationStageSummaryEntry = {
    stage_id: "overlay",
    label: "Overlay",
    comparators: [],
  };
  const withData: ValidationStageSummaryEntry = {
    stage_id: "classification",
    label: "Classification",
    comparators: [bucket()],
  };
  assert.equal(stageHasData(empty), false);
  assert.equal(stageHasData(withData), true);
});

test("field rows are alphabetical regardless of the order fields were inserted in", () => {
  const withMetrics = bucket({
    metrics: {
      p95_absolute_error: { count: 3, mean: 1, median: 1, p95: 1, min: 1, max: 1 },
      mae: { count: 3, mean: 0.2, median: 0.2, p95: 0.3, min: 0.1, max: 0.3 },
    },
  });
  assert.deepEqual(
    fieldRows(withMetrics).map((row) => row.field),
    ["mae", "p95_absolute_error"],
  );
});
