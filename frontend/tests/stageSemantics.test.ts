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

test("migrated 2.5D mask views declare rendererType mask", () => {
  const detect = stageSemanticDefinition("detect_belt_plane");
  assert.equal(detect.views.find((view) => view.id === "selected_surface")?.rendererType, "mask");
  assert.equal(detect.views.find((view) => view.id === "plane_inliers")?.rendererType, "mask");
  assert.equal(detect.views.find((view) => view.id === "low_gradient_mask")?.rendererType, "mask");
  assert.equal(detect.views.find((view) => view.id === "belt_bg")?.rendererType, "mask");
  assert.equal(detect.views.find((view) => view.id === "belt_stripes")?.rendererType, "mask");
  assert.equal(detect.views.find((view) => view.id === "unknown_low_gradient")?.rendererType, "mask");
  assert.equal(detect.views.find((view) => view.id === "surface_suppression")?.rendererType, "mask");
  assert.equal(detect.views.find((view) => view.id === "object_search_domain")?.rendererType, "mask");
  assert.equal(detect.views.find((view) => view.id === "background_candidates")?.rendererType, "mask");
  assert.equal(detect.views.find((view) => view.id === "background_seeds")?.rendererType, "mask");
  assert.equal(detect.views.find((view) => view.id === "expanded_plane")?.rendererType, "mask");
  assert.equal(detect.views.find((view) => view.id === "plane_fit_roi")?.rendererType, "mask");

  const segmentation = stageSemanticDefinition("remove_belt_segment_objects");
  const normalize = stageSemanticDefinition("normalize_heights_to_plane");
  assert.equal(normalize.views.find((view) => view.id === "below_reference")?.rendererType, "mask");
  assert.equal(normalize.views.find((view) => view.id === "above_threshold")?.rendererType, "mask");
  assert.equal(segmentation.views.find((view) => view.id === "cleaned_mask")?.rendererType, "mask");
  assert.equal(segmentation.views.find((view) => view.id === "plane_suppressed_mask")?.rendererType, "mask");
  assert.equal(segmentation.views.find((view) => view.id === "rejected_background_residuals")?.rendererType, "mask");
  assert.equal(segmentation.views.find((view) => view.id === "rejected_belt_stripes")?.rendererType, "mask");
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

test("25D detect belt plane exposes height-aware blob workflow views", () => {
  const semantic = stageSemanticDefinition("detect_belt_plane");
  assert.ok(semantic.views.some((view) => view.id === "height_aware_blob_components"));
  assert.ok(semantic.views.some((view) => view.id === "height_aware_connectivity_rejected_edges"));
  const heightAwareView = semantic.views.find((view) => view.id === "height_aware_blob_components");
  assert.match(String(heightAwareView?.emptyState?.description ?? ""), /height_aware/i);
});

test("25D detect belt plane exposes the selection bridge view", () => {
  const semantic = stageSemanticDefinition("detect_belt_plane");
  const bridge = semantic.views.find((view) => view.id === "selection_bridge");
  assert.ok(bridge);
  assert.equal(bridge?.label, "Selection bridge");
  assert.ok(semantic.views.some((view) => view.id === "blob_clusters"));
  assert.ok(semantic.views.some((view) => view.id === "blob_cluster_scores"));
});

test("25D detect belt plane exposes the support-loss waterfall view", () => {
  const semantic = stageSemanticDefinition("detect_belt_plane");
  const waterfall = semantic.views.find((view) => view.id === "support_loss_waterfall");
  assert.ok(waterfall);
  assert.equal(waterfall?.rendererType, "table");
  assert.match(String(waterfall?.emptyState?.title ?? ""), /waterfall/i);
});

test("25D detect belt plane exposes explicit bg-and-stripes surface-role views", () => {
  const semantic = stageSemanticDefinition("detect_belt_plane");
  assert.ok(semantic.views.some((view) => view.id === "belt_bg"));
  assert.ok(semantic.views.some((view) => view.id === "unknown_low_gradient"));
  assert.ok(semantic.views.some((view) => view.id === "surface_suppression"));
  assert.ok(semantic.views.some((view) => view.id === "object_search_domain"));
});
