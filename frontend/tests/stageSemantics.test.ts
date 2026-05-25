import test from "node:test";
import assert from "node:assert/strict";

import { stageSemanticDefinition } from "../src/components/stageSemantics.ts";

test("classification stage defaults to overlay view", () => {
  const semantic = stageSemanticDefinition("classification");
  assert.equal(semantic.category, "classification");
  assert.equal(semantic.defaultViewId, "overlay");
  assert.ok(semantic.views.some((view) => view.id === "overlay"));
  assert.ok(semantic.views.some((view) => view.id === "rule_explanation"));
  assert.ok(semantic.views.some((view) => view.id === "metric_details"));
});

test("25D load input defaults to height preview", () => {
  const semantic = stageSemanticDefinition("load_heightmap");
  assert.equal(semantic.category, "input");
  assert.equal(semantic.defaultViewId, "height_preview");
  assert.ok(semantic.views.some((view) => view.id === "height_preview"));
});

test("25D remove-belt segmentation defaults to overlay", () => {
  const semantic = stageSemanticDefinition("remove_belt_segment_objects");
  assert.equal(semantic.category, "segmentation");
  assert.equal(semantic.defaultViewId, "overlay");
  assert.ok(semantic.views.some((view) => view.id === "threshold_mask"));
  assert.ok(semantic.views.some((view) => view.id === "cleaned_mask"));
  assert.ok(semantic.views.some((view) => view.id === "segmentation"));
  assert.ok(semantic.views.some((view) => view.id === "overlay"));
});

test("25D detect belt plane defaults to selected surface view", () => {
  const semantic = stageSemanticDefinition("detect_belt_plane");
  assert.equal(semantic.category, "plane_qa");
  assert.equal(semantic.defaultViewId, "selected_surface");
  assert.ok(semantic.views.some((view) => view.id === "selected_surface"));
  assert.ok(semantic.views.some((view) => view.id === "low_gradient_mask"));
  assert.ok(semantic.views.some((view) => view.id === "depth_gradient"));
  assert.ok(semantic.views.some((view) => view.id === "raw_heightmap"));
  assert.ok(semantic.views.some((view) => view.id === "valid_mask"));
  assert.ok(semantic.views.some((view) => view.id === "residual_heatmap"));
});

test("25D normalize heights exposes histogram view", () => {
  const semantic = stageSemanticDefinition("normalize_heights_to_plane");
  assert.equal(semantic.category, "plane_qa");
  assert.equal(semantic.defaultViewId, "normalized_height");
  assert.ok(semantic.views.some((view) => view.id === "histogram"));
});

test("25D classify stage exposes rule explanation view", () => {
  const semantic = stageSemanticDefinition("classify_25d");
  assert.equal(semantic.category, "classification");
  assert.ok(semantic.views.some((view) => view.id === "rule_explanation"));
});

test("25D measurement stage exposes geometry debug view", () => {
  const semantic = stageSemanticDefinition("compute_height_metrics");
  assert.equal(semantic.category, "measurement");
  assert.ok(semantic.views.some((view) => view.id === "geometry_debug"));
});
