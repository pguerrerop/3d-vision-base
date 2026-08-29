import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import {
  buildFeatureJobCliCommand,
  featureFreshnessState,
  featureJobActionsEnabled,
  featureSourceLabel,
  featureSourceTooltip,
  resolveFeatureFreshnessScope,
} from "../src/featureAnalyticsFreshness.ts";

test("detects all missing registered feature state", () => {
  assert.equal(
    featureFreshnessState({
      featureKey: "surface_sphere_fit_rmse_mm",
      featureRegistered: true,
      scopedObjectCount: 10,
      distribution: {
        feature_key: "surface_sphere_fit_rmse_mm",
        group_by: "processed_superclass",
        bins: 8,
        mode: "count",
        groups: [],
        stats: { count: 0, missing: 10, missing_pct: 100, min: null, max: null, mean: null, std: null },
      },
    }),
    "all_missing",
  );
});

test("resolves ml set scope for feature jobs", () => {
  assert.deepEqual(
    resolveFeatureFreshnessScope({
      datasetId: "bolas-2-5-1",
      mlSetId: "balls_scrap_2026_05_25_29_table_v1",
      takeIds: [],
      mlSetName: "balls_scrap_2026_05_25_29_table_v1",
      scopeSummary: { record_count: 10, take_count: 5, physical_object_count: 5, object_count: 10 },
    }),
    {
      scope: "ml_set",
      datasetId: "bolas-2-5-1",
      mlSetId: "balls_scrap_2026_05_25_29_table_v1",
      takeCount: 5,
      mlSetName: "balls_scrap_2026_05_25_29_table_v1",
    },
  );
});

test("enables reprocess and backfill when feature is partially missing", () => {
  const scope = resolveFeatureFreshnessScope({
    datasetId: "d1",
    mlSetId: "ml1",
    takeIds: [],
    scopeSummary: { record_count: 3, take_count: 2, physical_object_count: 2, object_count: 3 },
  });
  const enabled = featureJobActionsEnabled({
    featureKey: "surface_sphere_fit_rmse_mm",
    featureRegistered: true,
    freshness: "partial_missing",
    scope,
    pipelineId: "mining_steel_ball_classification_25d",
    featureFamily: "sphere_consistency",
  });
  assert.equal(enabled.reprocess, true);
  assert.equal(enabled.backfill, true);
});

test("allows reprocess but not backfill for normalized sphere-consistency features", () => {
  const scope = resolveFeatureFreshnessScope({
    datasetId: "d1",
    mlSetId: "ml1",
    takeIds: [],
    scopeSummary: { record_count: 3, take_count: 2, physical_object_count: 2, object_count: 3 },
  });
  const enabled = featureJobActionsEnabled({
    featureKey: "surface_sphere_fit_residual_mad_norm",
    featureRegistered: true,
    freshness: "all_missing",
    scope,
    pipelineId: "mining_steel_ball_classification_25d",
    featureFamily: "sphere_consistency",
  });
  assert.equal(enabled.reprocess, true);
  assert.equal(enabled.backfill, false);
});

test("builds canonical reprocess CLI command", () => {
  const command = buildFeatureJobCliCommand({
    mode: "reprocess",
    featureKey: "surface_sphere_fit_rmse_mm",
    scope: {
      scope: "ml_set",
      datasetId: "bolas-2-5-1",
      mlSetId: "balls_scrap_2026_05_25_29_table_v1",
      takeCount: 223,
    },
    pipelineId: "mining_steel_ball_classification_25d",
  });
  assert.ok(command.includes("ml compute-feature"));
  assert.ok(command.includes("--mode reprocess --apply"));
  assert.ok(command.includes("surface_sphere_fit_rmse_mm"));
});

test("labels feature source compactly", () => {
  assert.equal(featureSourceLabel("feature_backfill"), "backfill");
  assert.equal(featureSourceLabel("pipeline_run"), "pipeline");
  assert.match(featureSourceTooltip("feature_backfill"), /feature_backfill/);
});

test("Feature Analytics page wires freshness actions and source column", () => {
  const pagePath = path.resolve("src/pages/FeatureAnalyticsPage.tsx");
  const panelPath = path.resolve("src/components/FeatureFreshnessPanel.tsx");
  const page = fs.readFileSync(pagePath, "utf8");
  const panel = fs.readFileSync(panelPath, "utf8");
  assert.ok(page.includes("FeatureFreshnessPanel"));
  assert.ok(page.includes("feature-source-chip"));
  assert.ok(panel.includes("Reprocess ML set"));
  assert.ok(panel.includes("Backfill feature"));
  assert.ok(panel.includes("Start reprocess"));
  assert.ok(panel.includes("Start backfill"));
  assert.ok(panel.includes("Refresh analytics"));
  assert.match(panel, /useful for fast exploration/);
});
