import test from "node:test";
import assert from "node:assert/strict";

import { featureAnalyticsFeatureCatalogScope, featureAnalyticsQueryToSearchParams } from "../src/featureAnalyticsQuery.ts";

test("feature analytics query serializes physical_object_ids as repeated plural params", () => {
  const qs = featureAnalyticsQueryToSearchParams({
    dataset_id: "bolas-2-5-1",
    ml_set_id: "balls_scrap_2026_05_25_29_table_v1",
    physical_object_ids: ["obj_0001"],
  });

  assert.equal(qs.get("dataset_id"), "bolas-2-5-1");
  assert.equal(qs.get("ml_set_id"), "balls_scrap_2026_05_25_29_table_v1");
  assert.deepEqual(qs.getAll("physical_object_ids"), ["obj_0001"]);
  assert.equal(qs.get("physical_object_id"), null);
});

test("feature analytics feature catalog scope preserves ML-set scope while dropping object-level narrowing", () => {
  const scoped = featureAnalyticsFeatureCatalogScope({
    dataset_id: "bolas-2-5-1",
    ml_set_id: "balls_scrap_2026_05_25_29_table_v1",
    session_id: "bolas_labeled_table_2026_05_25",
    physical_object_ids: ["obj_0001"],
    split: ["train"],
    validation_status: ["approved"],
    normalized_classes: ["BALL_SCRAP"],
  });

  assert.deepEqual(scoped, {
    dataset_id: "bolas-2-5-1",
    ml_set_id: "balls_scrap_2026_05_25_29_table_v1",
    session_id: "bolas_labeled_table_2026_05_25",
    pipeline_id: undefined,
    calibration_id: undefined,
    date_from: undefined,
    date_to: undefined,
  });
});
