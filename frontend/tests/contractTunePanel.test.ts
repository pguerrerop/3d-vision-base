import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

test("ContractTunePanel renders contract-scoped help, lineage, and empty-state copy", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/components/ContractTunePanel.tsx"), "utf-8");
  assert.ok(source.includes("No tunable parameters are registered for this substage."));
  assert.ok(source.includes("Affects:"));
  assert.ok(source.includes("Edited value differs from last run."));
  assert.ok(source.includes('label="Open parameter help"'));
  assert.ok(source.includes("Contract unit:"));
});

test("ContractTunePanel reads backend-sourced parameter groups and tuning guidance", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/components/ContractTunePanel.tsx"), "utf-8");
  assert.ok(source.includes("unit?.parameter_groups"));
  assert.ok(source.includes("unit?.downstream_effects"));
  assert.ok(source.includes("unit?.tuning_hints"));
  assert.ok(source.includes("Downstream effects:"));
  assert.ok(source.includes("What to tune"));
  assert.ok(source.includes("Other parameters for this unit"));
});
