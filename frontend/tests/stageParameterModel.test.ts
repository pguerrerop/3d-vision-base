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
