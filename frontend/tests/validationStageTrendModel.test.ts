import test from "node:test";
import assert from "node:assert/strict";

import type {
  ValidationStageTrendPoint,
  ValidationStageTrendSeries,
  ValidationStageTrendStageEntry,
} from "../src/api/client.ts";
import {
  executionLabel,
  fieldMeanTrend,
  pointRegressionRate,
  seriesFields,
  seriesHasAnyData,
  stagesWithTrendData,
} from "../src/components/validation/validationStageTrendModel.ts";

const point = (
  over: Partial<NonNullable<ValidationStageTrendPoint>> = {},
): NonNullable<ValidationStageTrendPoint> => ({
  count: 3,
  status_counts: { pass: 3 },
  metrics: { iou: { count: 3, mean: 0.9, median: 0.9, p95: 0.95, min: 0.8, max: 0.95 } },
  ...over,
});

test("execution label prefers the date, falling back to the id's tail", () => {
  assert.equal(
    executionLabel({ execution_id: "execution_abcdef123456", created_at: "2026-09-06T12:00:00Z" }),
    "2026-09-06",
  );
  assert.equal(executionLabel({ execution_id: "execution_abcdef123456" }), "123456");
  assert.equal(
    executionLabel({ execution_id: "execution_abcdef123456", created_at: "not a date" }),
    "123456",
  );
});

test("point regression rate is null for a gap, not a fabricated zero", () => {
  assert.equal(pointRegressionRate(null), null);
  assert.equal(
    pointRegressionRate(point({ count: 4, status_counts: { regression: 1, pass: 3 } })),
    0.25,
  );
  assert.equal(pointRegressionRate(point({ count: 0, status_counts: {} })), null);
});

test("series fields are the union across every present point, alphabetical", () => {
  const series: ValidationStageTrendSeries = {
    comparator: "binary_mask",
    points: [
      point({ metrics: { dice: { count: 1, mean: 1, median: 1, p95: 1, min: 1, max: 1 } } }),
      null,
      point({ metrics: { iou: { count: 1, mean: 1, median: 1, p95: 1, min: 1, max: 1 } } }),
    ],
  };
  assert.deepEqual(seriesFields(series), ["dice", "iou"]);
});

test("field mean trend aligns to the points, with a gap read as null rather than skipped", () => {
  const series: ValidationStageTrendSeries = {
    comparator: "binary_mask",
    points: [
      point({
        metrics: { iou: { count: 3, mean: 0.97, median: 0.97, p95: 0.97, min: 0.97, max: 0.97 } },
      }),
      null,
      point({
        metrics: { iou: { count: 3, mean: 0.91, median: 0.91, p95: 0.91, min: 0.91, max: 0.91 } },
      }),
    ],
  };
  assert.deepEqual(fieldMeanTrend(series, "iou"), [0.97, null, 0.91]);
  assert.deepEqual(
    fieldMeanTrend(series, "mae"),
    [null, null, null],
    "a field this series never reports is a gap everywhere",
  );
});

test("a series with data in even one execution counts as having data", () => {
  const empty: ValidationStageTrendSeries = { comparator: "binary_mask", points: [null, null] };
  const partial: ValidationStageTrendSeries = {
    comparator: "binary_mask",
    points: [null, point()],
  };
  assert.equal(seriesHasAnyData(empty), false);
  assert.equal(seriesHasAnyData(partial), true);
});

test("stages with only empty series across the whole window are filtered out of the trend view", () => {
  const withData: ValidationStageTrendStageEntry = {
    stage_id: "detect_belt_plane",
    label: "Detect Belt Plane",
    comparators: [{ comparator: "binary_mask", points: [point()] }],
  };
  const empty: ValidationStageTrendStageEntry = {
    stage_id: "overlay",
    label: "Overlay",
    comparators: [{ comparator: "binary_mask", points: [null, null] }],
  };
  const noComparators: ValidationStageTrendStageEntry = {
    stage_id: "input",
    label: "Input",
    comparators: [],
  };
  assert.deepEqual(
    stagesWithTrendData([withData, empty, noComparators]).map((s) => s.stage_id),
    ["detect_belt_plane"],
  );
});
