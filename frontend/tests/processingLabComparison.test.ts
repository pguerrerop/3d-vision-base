import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

test("Processing Lab exposes comparison controls", async () => {
  const source = await readFile(new URL("../src/pages/ProcessingLabPage.tsx", import.meta.url), "utf8");
  assert.ok(source.includes("Compare current to recipe"));
  assert.ok(source.includes("Compare run to current"));
  assert.ok(source.includes("Compare now"));
  assert.ok(source.includes("Contract-aware run and recipe comparison"));
  assert.ok(source.includes("What changed most"));
  assert.ok(source.includes("Saved comparisons"));
  assert.ok(source.includes("Reopen"));
  assert.ok(source.includes("Masks compared"));
  assert.ok(source.includes("Average IoU"));
  assert.ok(source.includes("pixel diff not computed"));
});
