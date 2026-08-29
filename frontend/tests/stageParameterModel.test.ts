import test from "node:test";
import assert from "node:assert/strict";
import { visibleStageParameterKeys } from "../src/components/stageParameterModel.ts";

test("RANSAC parameters are hidden unless method is ransac_ellipse", () => {
  const fields = {
    fit_method: { group: "basic" },
    min_contour_points: { group: "basic" },
    ransac_iterations: { group: "advanced", visible_when: { fit_method: "ransac_ellipse" } },
  };
  const withoutAdvanced = visibleStageParameterKeys(fields, { fit_method: "ransac_ellipse" }, false);
  assert.deepEqual(withoutAdvanced.sort(), ["fit_method", "min_contour_points"].sort());

  const nonRansac = visibleStageParameterKeys(fields, { fit_method: "opencv_fitEllipse" }, true);
  assert.deepEqual(nonRansac.sort(), ["fit_method", "min_contour_points"].sort());

  const ransac = visibleStageParameterKeys(fields, { fit_method: "ransac_ellipse" }, true);
  assert.deepEqual(ransac.sort(), ["fit_method", "min_contour_points", "ransac_iterations"].sort());
});

test("advanced flag hides grouped fields until advanced is enabled", () => {
  const fields = {
    background_detection_strategy: { group: "Reference surface" },
    plane_fit_min_inlier_ratio: { group: "Advanced reference tuning", advanced: true },
    belt_stripe_filter_scope: { group: "Belt stripe suppression" },
  };

  const withoutAdvanced = visibleStageParameterKeys(fields, {}, false);
  assert.deepEqual(withoutAdvanced.sort(), ["background_detection_strategy", "belt_stripe_filter_scope"].sort());

  const withAdvanced = visibleStageParameterKeys(fields, {}, true);
  assert.deepEqual(withAdvanced.sort(), ["background_detection_strategy", "plane_fit_min_inlier_ratio", "belt_stripe_filter_scope"].sort());
});

test("strategy-specific fields follow visible_when predicates", () => {
  const fields = {
    background_detection_strategy: { group: "Reference surface" },
    low_gradient_plateau_hist_bins: { group: "Advanced", visible_when: { background_detection_strategy: "low_gradient_depth_plateaus" } },
    blob_cluster_height_gap_mm: { group: "Advanced", visible_when: { background_detection_strategy: "low_gradient_blob_height_clusters" } },
    blob_split_by_height_enabled: { group: "Advanced", visible_when: { background_detection_strategy: "low_gradient_blob_height_clusters" } },
    blob_split_method: { group: "Advanced", visible_when: { background_detection_strategy: "low_gradient_blob_height_clusters" } },
  };

  const plateau = visibleStageParameterKeys(fields, { background_detection_strategy: "low_gradient_depth_plateaus" }, true);
  assert.deepEqual(plateau.sort(), ["background_detection_strategy", "low_gradient_plateau_hist_bins"].sort());

  const blob = visibleStageParameterKeys(fields, { background_detection_strategy: "low_gradient_blob_height_clusters" }, true);
  assert.deepEqual(blob.sort(), ["background_detection_strategy", "blob_cluster_height_gap_mm", "blob_split_by_height_enabled", "blob_split_method"].sort());
});
