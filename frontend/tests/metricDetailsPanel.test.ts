import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

test("MetricDetailsPanel renders metric_trace contract fields", () => {
  const source = fs.readFileSync(
    path.resolve(process.cwd(), "src/components/MetricDetailsPanel.tsx"),
    "utf-8"
  );
  for (const fragment of [
    "Correction context",
    "Geometry metric source",
    "Coordinate space",
    "Correction factor",
    "scale_correction_applied",
    "geometry_invariant_warnings",
    "Formula inputs",
    "Intermediate values",
  ]) {
    assert.ok(
      source.includes(fragment),
      `MetricDetailsPanel.tsx missing fragment: ${fragment}`
    );
  }
});

test("MetricDetailsPanel is wired into the classification metric_details view", () => {
  const source = fs.readFileSync(
    path.resolve(process.cwd(), "src/components/stage_view_renderers/index.tsx"),
    "utf-8"
  );
  assert.ok(source.includes("MetricDetailsPanel"));
  assert.ok(source.includes('props.view.id === "metric_details"'));
  assert.ok(source.includes('No metric explanation artifact found. Reprocess this take.'));
});
