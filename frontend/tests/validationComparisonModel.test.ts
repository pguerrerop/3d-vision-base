import test from "node:test";
import assert from "node:assert/strict";

import type { ValidationComparisonResult } from "../src/api/client.ts";
import {
  classificationCounts,
  classificationRows,
  classificationSideText,
  describeReason,
  diffImageUrls,
  filterClassificationRows,
  filterMeasurementRows,
  maskImages,
  maskMetricRows,
  measurementCounts,
  measurementRows,
  rasterImages,
  rasterMetricRows,
} from "../src/components/validation/validationComparisonModel.ts";

// Every fixture below is the literal output of the real Python comparators
// (vision_3d_acquisition/validation/service.py), captured by running them
// directly, not hand-written to match the model. If a comparator's shape
// drifts, these fixtures — not the model's expectations — are what's stale.

const MASK_RESULT: ValidationComparisonResult = {
  status: "regression",
  comparator: "binary_mask",
  summary: "Binary-mask comparison.",
  metrics: {
    iou: 0.5,
    dice: 0.6666666666666666,
    intersection_pixels: 1,
    union_pixels: 2,
    added_pixels: 1,
    removed_pixels: 0,
    added_fraction: 0.25,
    removed_fraction: 0.0,
    baseline_foreground_pixels: 1,
    candidate_foreground_pixels: 2,
    baseline_components: 1,
    candidate_components: 1,
  },
  thresholds: { comparator: "binary_mask" },
  reasons: ["iou_below_threshold", "dice_below_threshold", "added_fraction_above_threshold"],
  diff_artifacts: [
    "executions/e1/diffs/baseline_mask.png",
    "executions/e1/diffs/candidate_mask.png",
    "executions/e1/diffs/added_pixels.png",
    "executions/e1/diffs/removed_pixels.png",
    "executions/e1/diffs/combined_diff.png",
  ],
  baseline_ref: { baseline_id: "baseline_x", version: 1 },
  candidate_ref: { artifact_id: "final_object_mask" },
};

const RASTER_RESULT: ValidationComparisonResult = {
  status: "pass",
  comparator: "numeric_raster",
  summary: "Numeric-raster comparison.",
  metrics: {
    valid_overlap: 1.0,
    candidate_only_valid_pixels: 0,
    baseline_only_valid_pixels: 0,
    mae: 0.125,
    rmse: 0.25,
    median_absolute_error: 0.0,
    p95_absolute_error: 0.4249999999999998,
    p99_absolute_error: 0.4849999999999999,
    max_absolute_error: 0.5,
    signed_mean_error: 0.125,
  },
  thresholds: { comparator: "numeric_raster" },
  reasons: [],
  diff_artifacts: [
    "executions/e2/diffs/absolute_difference.png",
    "executions/e2/diffs/valid_pixel_change.png",
  ],
  baseline_ref: { baseline_id: "baseline_y", version: 1 },
  candidate_ref: { artifact_id: "final_object_mask" },
};

const MEASUREMENT_RESULT: ValidationComparisonResult = {
  status: "regression",
  comparator: "measurement_table",
  summary: "1 measurement metrics exceeded tolerance.",
  metrics: {
    baseline_object_count: 2,
    candidate_object_count: 2,
    matched_object_count: 1,
    missing_object_count: 1,
    added_object_count: 1,
    regressed_metric_count: 1,
  },
  thresholds: {},
  reasons: [],
  diff_artifacts: [],
  baseline_ref: { baseline_id: "baseline_m", version: 1 },
  candidate_ref: { artifact_id: "measure" },
  object_results: [
    {
      baseline_index: 0,
      candidate_index: 0,
      baseline_object_id: "one",
      candidate_object_id: "one",
      match_method: "stable_id",
      match_score: 1.0,
      ambiguous: false,
      status: "regression",
      metric_results: [
        {
          metric: "diameter_mm",
          units: null,
          baseline: 10.0,
          candidate: 10.6,
          delta: 0.5999999999999996,
          relative_delta: 0.05999999999999996,
          absolute_tolerance: 0.5,
          relative_tolerance: 0.0,
          effective_tolerance: 0.5,
          status: "regression",
          reason: "tolerance_exceeded",
        },
      ],
    },
  ],
  matching: {
    matches: [
      {
        baseline_index: 0,
        candidate_index: 0,
        baseline_object_id: "one",
        candidate_object_id: "one",
        match_method: "stable_id",
        match_score: 1.0,
        ambiguous: false,
      },
    ],
    baseline_only: [{ index: 1, object_id: "gone" }],
    candidate_only: [{ index: 1, object_id: "new" }],
    ambiguous: [],
  },
};

const CLASSIFICATION_RESULT: ValidationComparisonResult = {
  status: "regression",
  comparator: "classification",
  summary: "1 classification labels changed.",
  metrics: {
    baseline_object_count: 1,
    candidate_object_count: 1,
    label_changes: 1,
    superclass_changes: 1,
    new_critical_warnings: 1,
  },
  thresholds: {},
  reasons: [],
  diff_artifacts: [],
  baseline_ref: { baseline_id: "baseline_c", version: 1 },
  candidate_ref: { artifact_id: "classify" },
  object_results: [
    {
      baseline_index: 0,
      candidate_index: 0,
      baseline_object_id: "one",
      candidate_object_id: "one",
      match_method: "stable_id",
      match_score: 1.0,
      ambiguous: false,
      baseline: { superclass: "ball", label: "sphere", confidence: 0.9 },
      candidate: { superclass: "fragment", label: "fragment", confidence: 0.7 },
      confidence_delta: -0.20000000000000007,
      status: "regression",
      reasons: ["superclass_changed", "label_changed", "confidence_drop", "new_critical_warning"],
    },
  ],
  matching: { matches: [], baseline_only: [], candidate_only: [], ambiguous: [] },
};

// --- shared -------------------------------------------------------------

test("diff image urls are keyed by filename regardless of the path structure above them", () => {
  const urls = diffImageUrls(MASK_RESULT);
  assert.equal(Object.keys(urls).length, 5);
  assert.match(urls["combined_diff.png"], /\/api\/validation\/files\//);
  assert.match(urls["combined_diff.png"], /combined_diff\.png$/);
});

test("reason codes always resolve to a sentence, known or not", () => {
  assert.equal(describeReason("iou_below_threshold"), "IoU fell below the configured minimum.");
  assert.equal(describeReason("some_future_code"), "some future code");
});

// --- mask -----------------------------------------------------------------

test("mask images map the five persisted diffs to their roles", () => {
  const images = maskImages(MASK_RESULT);
  assert.ok(
    images.baseline && images.candidate && images.combined && images.added && images.removed,
  );
});

test("mask metric rows surface iou, dice, areas, fractions and component counts", () => {
  const rows = maskMetricRows(MASK_RESULT);
  const byLabel = Object.fromEntries(rows.map((row) => [row.label, row]));
  assert.equal(byLabel["IoU"].raw, 0.5);
  assert.equal(byLabel["Dice"].value, "0.667");
  assert.equal(byLabel["Baseline foreground area"].value, "1 px");
  assert.equal(byLabel["Added fraction"].value, "25.00%");
  assert.equal(byLabel["Baseline components"].value, "1");
});

// --- raster -----------------------------------------------------------------

test("raster images map the two persisted diffs", () => {
  const images = rasterImages(RASTER_RESULT);
  assert.match(String(images.absoluteDifference), /absolute_difference\.png$/);
  assert.match(String(images.validPixelChange), /valid_pixel_change\.png$/);
});

test("raster metric rows cover every published error statistic", () => {
  const rows = rasterMetricRows(RASTER_RESULT);
  const byLabel = Object.fromEntries(rows.map((row) => [row.label, row]));
  assert.equal(byLabel["Valid overlap"].value, "100.00%");
  assert.equal(byLabel["MAE"].raw, 0.125);
  assert.equal(byLabel["RMSE"].raw, 0.25);
  assert.equal(byLabel["Signed mean error"].raw, 0.125);
});

// --- measurement -------------------------------------------------------------

test("measurement rows include the matched metric and both unmatched objects", () => {
  const rows = measurementRows(MEASUREMENT_RESULT);
  assert.equal(rows.length, 3, "one metric row, one missing, one added");
  const missing = rows.find((row) => row.kind === "unmatched-object" && row.side === "missing");
  const added = rows.find((row) => row.kind === "unmatched-object" && row.side === "added");
  assert.equal(missing && "objectId" in missing ? missing.objectId : null, "gone");
  assert.equal(added && "objectId" in added ? added.objectId : null, "new");
});

test("object-count changes stay visible with the missing filter even though no metric row anchors them", () => {
  const rows = measurementRows(MEASUREMENT_RESULT);
  const missingFiltered = filterMeasurementRows(rows, "missing");
  assert.equal(missingFiltered.length, 1);
  assert.equal(filterMeasurementRows(rows, "added").length, 1);
  assert.equal(filterMeasurementRows(rows, "regressions").length, 1);
  assert.equal(filterMeasurementRows(rows, "all").length, 3);
});

test("measurement counts read straight from the metrics the comparator already computed", () => {
  const counts = measurementCounts(MEASUREMENT_RESULT);
  assert.deepEqual(counts, {
    baselineObjects: 2,
    candidateObjects: 2,
    matched: 1,
    missing: 1,
    added: 1,
    ambiguous: 0,
    regressedMetrics: 1,
  });
});

// --- classification -------------------------------------------------------------

test("classification rows carry both sides and every reason the comparator found", () => {
  const rows = classificationRows(CLASSIFICATION_RESULT);
  assert.equal(rows.length, 1);
  const row = rows[0];
  assert.equal(row.kind, "matched");
  if (row.kind === "matched") {
    assert.deepEqual(row.result.reasons, [
      "superclass_changed",
      "label_changed",
      "confidence_drop",
      "new_critical_warning",
    ]);
  }
});

test("classification filters distinguish superclass, label and confidence-only changes", () => {
  const rows = classificationRows(CLASSIFICATION_RESULT);
  assert.equal(filterClassificationRows(rows, "superclass").length, 1);
  assert.equal(filterClassificationRows(rows, "label").length, 1);
  // Both superclass and label changed here, so it is not a confidence-only change.
  assert.equal(filterClassificationRows(rows, "confidence").length, 0);
  assert.equal(filterClassificationRows(rows, "warnings").length, 1);
});

test("a confidence-only regression is distinguished from a superclass or label change", () => {
  const confidenceOnly: ValidationComparisonResult = {
    ...CLASSIFICATION_RESULT,
    object_results: [
      {
        ...(CLASSIFICATION_RESULT.object_results![0] as any),
        reasons: ["confidence_drop"],
      },
    ],
  };
  const rows = classificationRows(confidenceOnly);
  assert.equal(filterClassificationRows(rows, "confidence").length, 1);
  assert.equal(filterClassificationRows(rows, "superclass").length, 0);
});

test("classification counts read straight from the comparator's metrics", () => {
  assert.deepEqual(classificationCounts(CLASSIFICATION_RESULT), {
    baselineObjects: 1,
    candidateObjects: 1,
    labelChanges: 1,
    superclassChanges: 1,
    newCriticalWarnings: 1,
  });
});

test("a classification side renders superclass, label and confidence as one readable string", () => {
  assert.equal(
    classificationSideText({ superclass: "ball", label: "sphere", confidence: 0.9 }),
    "ball / sphere (90.0%)",
  );
  assert.equal(
    classificationSideText({ superclass: null, label: null, confidence: null }),
    "— / — (—)",
  );
});
