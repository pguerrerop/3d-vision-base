import test from "node:test";
import assert from "node:assert/strict";

import type { FeatureDefinition } from "../src/api/client.ts";
import {
  featureFamilyLabel,
  featureSectionLabel,
  groupFeatureCatalog,
} from "../src/featureCatalogModel.ts";
import { featureSourceLabel } from "../src/featureAnalyticsFreshness.ts";

const sphereFeatures: FeatureDefinition[] = [
  {
    feature_key: "surface_sphere_fit_rmse_mm",
    display_name: "Raw sphere-fit RMSE",
    family: "sphere_consistency",
    family_label: "Sphere consistency",
    source_stage: "measurement",
    unit: "mm",
    formula: "RMSE",
    algorithm_summary: "Raw visible-surface sphere-fit RMSE.",
    description: "Raw visible-surface sphere-fit RMSE.",
    stable_schema: true,
    diagnostic_only: false,
    higher_is_worse: true,
    caveats: ["Raw RMSE is scale-dependent and visible-surface-only; prefer normalized residuals and sphere plausibility for classification."],
  },
  {
    feature_key: "surface_sphere_fit_rmse_norm",
    display_name: "Normalized sphere-fit RMSE",
    family: "sphere_consistency",
    family_label: "Sphere consistency",
    source_stage: "measurement",
    unit: "ratio",
    formula: "rmse / diameter",
    algorithm_summary: "Normalized RMSE.",
    description: "Normalized RMSE.",
    stable_schema: true,
    diagnostic_only: false,
    higher_is_worse: true,
  },
];

test("groups sphere consistency features under Sphere consistency", () => {
  const groups = groupFeatureCatalog(sphereFeatures);
  assert.ok(groups.some((group) => group.familyLabel === "Sphere consistency"));
});

test("labels derived feature source badge", () => {
  assert.equal(featureSourceLabel("derived"), "derived");
  assert.equal(featureSourceLabel("pipeline_run"), "pipeline");
  assert.equal(featureSourceLabel("feature_backfill"), "backfill");
});

test("exposes stable vs diagnostic section labels", () => {
  assert.match(featureSectionLabel(sphereFeatures[0]), /Stable ML/);
  assert.equal(featureFamilyLabel(sphereFeatures[1]), "Sphere consistency");
});

test("raw RMSE caveat warns against using raw RMSE for classification", () => {
  const caveat = sphereFeatures[0].caveats?.[0] ?? "";
  assert.match(caveat, /Raw RMSE is scale-dependent/i);
  assert.match(caveat, /prefer normalized residuals/i);
});
