import test from "node:test";
import assert from "node:assert/strict";

import {
  formatConfidence,
  formatDecision,
  formatMeasurements,
  getDecisionTone,
} from "../src/components/operatorInspectionHelpers.ts";

test("ACCEPT result renders dominant accept state helper", () => {
  assert.equal(formatDecision("accept"), "ACCEPT");
  assert.equal(getDecisionTone("accept"), "accept");
});

test("REJECT formatting helpers include class/object row-ready values", () => {
  assert.equal(formatDecision("reject"), "REJECT");
  assert.equal(formatConfidence(0.8123), "81.2%");
});

test("missing overlay fallback helper path remains safe", () => {
  assert.equal(formatConfidence(null), "-");
  assert.equal(formatMeasurements({}), "-");
});

test("measurement formatting compacts key values", () => {
  const text = formatMeasurements({ diameter_mm: 22.1234, roughness: "low" });
  assert.ok(text.includes("diameter_mm"));
  assert.ok(text.includes("roughness"));
});

test("API error copy remains readable in operator page source", async () => {
  const fs = await import("node:fs");
  const path = await import("node:path");
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/pages/OperatorInspectionPage.tsx"), "utf-8");
  assert.ok(source.includes("Unable to load operator inspection results"));
  assert.ok(source.includes("No display image available"));
  assert.ok(source.includes("setSelectedResultId(item.published_result_id)"));
});
