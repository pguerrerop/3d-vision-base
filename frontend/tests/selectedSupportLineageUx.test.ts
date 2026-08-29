import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

test("SelectedSupportLineagePanel renders visual mask-evolution workflow affordances", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/components/SelectedSupportLineagePanel.tsx"), "utf-8");

  assert.ok(source.includes("Mask evolution"));
  assert.ok(source.includes("Selected lineage artifact"));
  assert.ok(source.includes("Removed / Added"));
  assert.ok(source.includes("lineage-film-card"));
  assert.ok(source.includes("lineage-thumb"));
  assert.ok(source.includes('selectStepArtifact(step, "difference"'));
  assert.ok(source.includes("Focused artifact:"));
  assert.ok(source.includes("Legend: Red = removed from previous step"));
  assert.ok(source.includes("No selected-support lineage steps are available."));
});

test("Processing Lab routes lineage-step focus into tune mode", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/pages/ProcessingLabPage.tsx"), "utf-8");

  assert.ok(source.includes("const focusSelectedSupportLineageStep = useCallback"));
  assert.ok(source.includes('setControlPanelTab("tune")'));
  assert.ok(source.includes('setStudioMode("tune")'));
  assert.ok(source.includes('onFocusLineageStep: focusSelectedSupportLineageStep'));
});
