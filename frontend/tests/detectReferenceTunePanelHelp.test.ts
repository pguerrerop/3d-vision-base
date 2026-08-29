import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

test("DetectReferenceTunePanel wires info hints into detect-reference parameter labels", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/components/DetectReferenceTunePanel.tsx"), "utf-8");
  assert.ok(source.includes("buildFieldHelpEntry"));
  assert.ok(source.includes("detectReferenceGroupHelpKey"));
  assert.ok(source.includes('label="Open group help"'));
  assert.ok(source.includes('label="Open parameter help"'));
  assert.ok(source.includes("info-inline-label"));
  assert.ok(source.includes("No tunable parameters are registered for this substage."));
  assert.ok(source.includes("Contract unit:"));
  assert.ok(source.includes("reference_surface_region"));
  assert.ok(source.includes("Edit visually"));
  assert.ok(source.includes("Edited value differs from last run."));
  assert.ok(source.includes("Last run:"));
});
