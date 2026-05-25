import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

test("Rule explanation view is rendered in stage renderer", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/components/stage_view_renderers/index.tsx"), "utf-8");
  assert.ok(source.includes('props.view.id === "rule_explanation"'));
  assert.ok(source.includes("classification_explanation"));
  assert.ok(source.includes("resolveClassificationExplanationArtifact"));
  assert.ok(source.includes("resolveMetricExplanationArtifact"));
  assert.ok(source.includes('props.view.id === "metric_details"'));
  assert.ok(source.includes("No metric explanation artifact found. Reprocess this take."));
  assert.ok(source.includes("explainRowsFromArtifact"));
  assert.ok(source.includes("meta.objects"));
  assert.ok(source.includes("classification_explanation_json"));
  assert.ok(source.includes("classification_explanation_metadata"));
  assert.ok(source.includes("classification_explanation.json"));
  assert.ok(source.includes("files?.classification_explanation"));
  assert.ok(source.includes("No classification explanation artifact found. Reprocess this take."));
  assert.ok(source.includes("Available classification artifacts"));
  assert.ok(source.includes("Decision path"));
  assert.ok(source.includes("Expected/threshold"));
});

test("Inspector exposes why this class section", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/components/StudioInspector.tsx"), "utf-8");
  assert.ok(source.includes("Why this class?"));
  assert.ok(source.includes("diameter_selected_source"));
});
