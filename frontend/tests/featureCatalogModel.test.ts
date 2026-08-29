import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

import {
  featureBadgeLabels,
  featureOptionLabel,
  featureStudioStage,
  filterFeatureCatalog,
  groupFeatureCatalog,
} from "../src/featureCatalogModel.ts";
import type { FeatureDefinition } from "../src/api/client.ts";

const axisBalance: FeatureDefinition = {
  feature_key: "feature_sphericity_3d",
  display_name: "3D axis balance",
  semantic_group: "Surface shape",
  family: "surface_shape",
  family_label: "Surface shape",
  unit: "ratio",
  source_stage: "measurement",
  studio_stage: "measurement",
  formula: "min(dim_x_mm, dim_y_mm, dim_z_mm) / max(dim_x_mm, dim_y_mm, dim_z_mm)",
  algorithm_summary: "Axis-balance heuristic.",
  interpretation: "Balanced extents.",
  caveats: ["This is not a true sphere-fit metric."],
  stable_schema: true,
  diagnostic_only: false,
  higher_is_worse: false,
  legacy_aliases: ["Sphericity 3D"],
};

const confidence: FeatureDefinition = {
  feature_key: "confidence",
  display_name: "Classification confidence",
  semantic_group: "Classification diagnostics",
  family: "classification_diagnostics",
  family_label: "Classification diagnostics",
  unit: "score",
  source_stage: "classification",
  formula: "Classifier confidence",
  algorithm_summary: "Diagnostic confidence value.",
  interpretation: "Higher is more confident.",
  stable_schema: false,
  diagnostic_only: true,
  higher_is_worse: false,
};

const sphereFitRmse: FeatureDefinition = {
  feature_key: "surface_sphere_fit_rmse_mm",
  display_name: "Visible surface sphere-fit RMSE",
  semantic_group: "Sphere consistency",
  family: "sphere_consistency",
  family_label: "Sphere consistency",
  unit: "mm",
  source_stage: "measurement",
  studio_stage: "measurement",
  formula: "sqrt(mean((||p_i - c|| - r)^2))",
  algorithm_summary: "Sphere fit residual RMSE.",
  interpretation: "Lower means closer to a sphere.",
  caveats: ["Visible 2.5D surface only."],
  stable_schema: true,
  diagnostic_only: false,
  higher_is_worse: true,
};

const curvatureProxy: FeatureDefinition = {
  feature_key: "feature_local_curvature_proxy",
  display_name: "Local curvature proxy",
  semantic_group: "Surface shape",
  family: "surface_shape",
  family_label: "Surface shape",
  unit: "score",
  source_stage: "measurement",
  formula: "gradient-based curvature proxy",
  algorithm_summary: "Local variation proxy.",
  interpretation: "Higher means more local variation.",
  diagnostic_only: true,
  experimental: true,
  higher_is_worse: true,
};

test("feature catalog groups stable and diagnostic features separately", () => {
  const groups = groupFeatureCatalog([confidence, axisBalance]);
  assert.equal(groups.length, 2);
  assert.equal(groups[0]?.sectionLabel, "Stable ML/export features");
  assert.equal(groups[1]?.sectionLabel, "Diagnostic / engineering features");
});

test("feature selector search matches display name, raw key, and legacy aliases", () => {
  assert.deepEqual(filterFeatureCatalog([axisBalance], "axis").map((item) => item.feature_key), ["feature_sphericity_3d"]);
  assert.deepEqual(filterFeatureCatalog([axisBalance], "feature_sphericity_3d").map((item) => item.feature_key), ["feature_sphericity_3d"]);
  assert.deepEqual(filterFeatureCatalog([axisBalance], "sphericity").map((item) => item.feature_key), ["feature_sphericity_3d"]);
});

test("feature option label includes raw key and badges", () => {
  const label = featureOptionLabel(axisBalance);
  assert.ok(label.includes("3D axis balance"));
  assert.ok(label.includes("feature_sphericity_3d"));
  assert.ok(label.includes("Stable ML"));
});

test("sphere-fit feature and curvature proxy remain separate in the catalog", () => {
  const sphereLabel = featureOptionLabel(sphereFitRmse);
  const curvatureLabel = featureOptionLabel(curvatureProxy);
  assert.ok(sphereLabel.includes("Visible surface sphere-fit RMSE"));
  assert.ok(sphereLabel.includes("surface_sphere_fit_rmse_mm"));
  assert.ok(curvatureLabel.includes("Local curvature proxy"));
  assert.ok(curvatureLabel.includes("feature_local_curvature_proxy"));
  assert.ok(!curvatureLabel.includes("Surface deformation score"));
});

test("feature studio stage prefers explicit studio stage", () => {
  assert.equal(featureStudioStage(axisBalance), "measurement");
  assert.equal(featureStudioStage(null), null);
});

test("Feature Analytics page renders feature catalog search and richer inspector metadata", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/pages/FeatureAnalyticsPage.tsx"), "utf-8");
  assert.ok(source.includes("Search feature catalog"));
  assert.ok(source.includes("feature-catalog-badges"));
  assert.ok(source.includes("<strong>Formula</strong>"));
  assert.ok(source.includes("<strong>Interpretation</strong>"));
  assert.ok(source.includes("Open selected sample in Studio"));
  assert.ok(source.includes("featureStudioStage(featureMeta)"));
});
