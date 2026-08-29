import test from "node:test";
import assert from "node:assert/strict";

import { normalizeLowGradientComponents } from "../src/studio/lowGradientComponentModel.ts";

test("normalizeLowGradientComponents maps role, height, and geometry fields", () => {
  const result = normalizeLowGradientComponents([
    { blob_id: 1, area_px: 500, centroid: [10, 20], bbox: [5, 15, 20, 20], aspect_ratio: 1.2, solidity: 0.9, role: "belt_bg", mean_z: 2.5, rejected: false },
    { component_id: 2, area_px: 12, centroid: [50, 60], bbox: [45, 55, 10, 10], rejected: true, rejected_reason: "small_area" },
  ]);
  assert.equal(result.components.length, 1);
  assert.equal(result.rejected.length, 1);
  assert.equal(result.components[0].id, 1);
  assert.equal(result.components[0].role, "belt_bg");
  assert.equal(result.components[0].meanHeightMm, 2.5);
  assert.deepEqual(result.components[0].centroid, [10, 20]);
  assert.equal(result.rejected[0].id, 2);
  assert.equal(result.rejected[0].rejectionReason, "small_area");
});

test("normalizeLowGradientComponents handles an entries-wrapped payload and missing fields", () => {
  const result = normalizeLowGradientComponents({ entries: [{ component_id: 7, rejected: false }] });
  assert.equal(result.components.length, 1);
  assert.equal(result.components[0].id, 7);
  assert.equal(result.components[0].bbox, undefined);
  assert.equal(result.components[0].role, undefined);
});

test("normalizeLowGradientComponents returns empty sets for non-array payloads", () => {
  const result = normalizeLowGradientComponents(null);
  assert.deepEqual(result.components, []);
  assert.deepEqual(result.rejected, []);
});
