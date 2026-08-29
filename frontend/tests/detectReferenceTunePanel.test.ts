import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

test("detect reference tune panel renders backend-sourced parameter groups and tuning guidance", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/components/DetectReferenceTunePanel.tsx"), "utf-8");
  assert.ok(source.includes('selectedUnit?.parameter_groups'));
  assert.ok(source.includes('selectedUnit?.downstream_effects'));
  assert.ok(source.includes('selectedUnit?.tuning_hints'));
  assert.ok(source.includes('Downstream effects:'));
  assert.ok(source.includes('Affects:'));
  assert.ok(source.includes('What to tune'));
});

test("detect reference tune panel exposes low-gradient background plus stripes option", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/components/DetectReferenceTunePanel.tsx"), "utf-8");
  assert.ok(source.includes('option value="low_gradient_bg_and_stripes"'));
  assert.ok(source.includes("Low-gradient background + stripes"));
});
