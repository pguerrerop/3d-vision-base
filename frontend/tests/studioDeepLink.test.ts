import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { buildStudioDeepLink, graphWorkspaceEnabled, parseStudioDeepLink } from "../src/studioDeepLink.ts";

test("Studio deep links include take, stage, feature, and object context", () => {
  const href = buildStudioDeepLink({
    take_id: "2026-05-29T222710_073",
    pipeline_id: "3d_ball_inspection",
    run_id: "run_123",
    stage: "measurement",
    object_id: "28",
    physical_object_id: "obj_0028",
    feature: "feature_sphericity_3d",
    dataset_id: "bolas-2-5-1",
    session_id: "session_01",
  });

  assert.equal(
    href,
    "/studio?take_id=2026-05-29T222710_073&pipeline_id=3d_ball_inspection&run_id=run_123&stage=measurement&object_id=28&physical_object_id=obj_0028&feature=feature_sphericity_3d&dataset_id=bolas-2-5-1&session_id=session_01",
  );
});

test("Studio deep links encode graph workspace context", () => {
  const href = buildStudioDeepLink({
    take_id: "take_smoke_compare",
    pipeline_id: "mining_steel_ball_classification_25d",
    graph_workspace: "1",
    unit_id: "segment.thresholding",
    comparison_id: "cmp_demo",
    recipe_id: "recipe_base",
    run_id: "smoke_run_b",
  });
  assert.ok(href.includes("graph_workspace=1"));
  assert.ok(href.includes("unit_id=segment.thresholding"));
  assert.ok(href.includes("comparison_id=cmp_demo"));
  assert.ok(href.includes("recipe_id=recipe_base"));
});

test("Studio deep links parse back into routing hints", () => {
  const parsed = parseStudioDeepLink("?take_id=t1&stage=measurement&physical_object_id=obj_0001&feature=feature_sphericity_3d");
  assert.deepEqual(parsed, {
    take_id: "t1",
    pipeline_id: null,
    run_id: null,
    stage: "measurement",
    object_id: null,
    physical_object_id: "obj_0001",
    feature: "feature_sphericity_3d",
    dataset_id: null,
    session_id: null,
    graph_workspace: null,
    unit_id: null,
    comparison_id: null,
    recipe_id: null,
    tab: null,
  });
});

test("graph workspace flag accepts common truthy values", () => {
  assert.equal(graphWorkspaceEnabled("1"), true);
  assert.equal(graphWorkspaceEnabled("true"), true);
  assert.equal(graphWorkspaceEnabled("0"), false);
});

test("Feature Analytics matching rows link to Studio in a new tab", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/pages/FeatureAnalyticsPage.tsx"), "utf-8");
  assert.ok(source.includes("<th>Inspect</th>"));
  assert.ok(source.includes("buildStudioDeepLink({"));
  assert.ok(source.includes('target="_blank"'));
  assert.ok(source.includes('rel="noreferrer"'));
  assert.ok(source.includes("Studio ↗"));
  assert.ok(source.includes("feature: featureKey"));
  assert.ok(source.includes("stage: featureStudioStage(featureMeta) ?? item.stage_id"));
});

test("Processing Lab parses and applies Studio deep-link hints", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/pages/ProcessingLabPage.tsx"), "utf-8");
  assert.ok(source.includes("parseStudioDeepLink"));
  assert.ok(source.includes("buildStudioDeepLinkMessage"));
  assert.ok(source.includes("resolveStudioStageId"));
  assert.ok(source.includes("studioDeepLink.physical_object_id"));
  assert.ok(source.includes("Opened from Feature Analytics"));
  assert.ok(source.includes("buildStudioDeepLink"));
  assert.ok(source.includes("graphWorkspaceEnabled"));
  assert.ok(source.includes("studioGraphDeepLinkAppliedRef"));
});

test("Processing Lab restores the active run from a run_id deep link", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/pages/ProcessingLabPage.tsx"), "utf-8");
  assert.ok(source.includes("studioRunDeepLinkAppliedRef"));
  assert.ok(source.includes('studioDeepLink.run_id === "latest" ? null : studioDeepLink.run_id'));
  assert.ok(source.includes("setActiveRunId(studioDeepLink.run_id"));
});

test("Processing Lab round-trips run_id into the deep-link URL when the active run changes", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/pages/ProcessingLabPage.tsx"), "utf-8");
  assert.ok(source.includes("run_id: activeRunId ?? current.run_id"));
  assert.ok(source.includes("activeRunId,\n    comparisonResult?.comparison_id"));
});
