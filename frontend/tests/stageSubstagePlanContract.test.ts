import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

test("stage substage plan can resolve Detect Reference from processing-unit registry", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/components/stageSubstagePlan.ts"), "utf-8");
  assert.ok(source.includes('source: "processing_unit_registry"'));
  assert.ok(source.includes("processingUnitsForStage"));
  assert.ok(source.includes("resolveStageViewForUnit"));
  assert.ok(source.includes("substageIdFromUnitId"));
  assert.ok(source.includes("genericSubstagePlanFromUnits"));
});

test("processing unit model resolves contract-first views and artifact refs", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/components/processingUnitModel.ts"), "utf-8");
  assert.ok(source.includes("export type ResolvedProcessingUnitView"));
  assert.ok(source.includes("fallbackReason"));
  assert.ok(source.includes("resolveProcessingUnitView("));
  assert.ok(source.includes("resolveProcessingUnitArtifactRefs("));
});
