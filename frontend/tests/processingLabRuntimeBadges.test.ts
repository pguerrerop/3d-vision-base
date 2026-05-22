import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

test("Processing Lab includes runtime worker badges and published result status", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/pages/ProcessingLabPage.tsx"), "utf-8");
  assert.ok(source.includes("Runtime worker health"));
  assert.ok(source.includes("Latest published result"));
  assert.ok(source.includes("workerBadgeLabel"));
});

test("Processing Lab shows read-only active process binding visibility", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/pages/ProcessingLabPage.tsx"), "utf-8");
  assert.ok(source.includes("Active binding visibility"));
  assert.ok(source.includes("active_recipe_version_id"));
  assert.ok(source.includes("No active binding found"));
});
