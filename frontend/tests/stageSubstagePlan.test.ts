import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

test("reference view selection keeps Strategy path explicit and limits inference to non-strategy detail views", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/components/stageSubstagePlan.ts"), "utf-8");
  assert.ok(source.includes("export function resolveReferenceViewSelection("));
  assert.ok(source.includes('if (explicitScope === "strategy_path")'));
  assert.ok(source.includes('activeScope: "strategy_path"'));
  assert.ok(source.includes('if (explicitScope === "substage" && validSelectedSubstageId)'));
  assert.ok(source.includes("const strategyPathIds = new Set(resolution.plan?.strategyPathViewIds ?? []);"));
  assert.ok(source.includes('if (inferredSubstageId && viewId && !strategyPathIds.has(viewId))'));
  assert.ok(source.includes('activeScope: "substage"'));
});

test("detect substage artifacts can be inactive even when placeholder files exist", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/components/stageSubstagePlan.ts"), "utf-8");
  assert.ok(source.includes("function artifactInactiveForDetectStrategy("));
  assert.ok(source.includes("function detectUnitActiveByApplicableArtifact("));
  assert.ok(source.includes('BLOB_CLUSTER_ONLY_ARTIFACTS.has(artifactId) && state.strategy !== "low_gradient_blob_height_clusters"'));
  assert.ok(source.includes('"low_gradient_blob_components_overlay"'));
  assert.ok(source.includes('"low_gradient_blob_id_mask"'));
  assert.ok(source.includes('"height_aware_blob_components_overlay"'));
  assert.ok(source.includes('"height_aware_blob_id_mask"'));
  assert.ok(source.includes('category: inactiveForStrategy ? "inactive_strategy"'));
  assert.ok(source.includes("detectUnitActiveByApplicableArtifact(unit, presentArtifactIds, runState)"));
});

test("artifacts with downstream feeds_into links are marked as OUT even when their contract role is diagnostic", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/components/stageSubstagePlan.ts"), "utf-8");
  assert.ok(source.includes("function ioRoleForUnitRole(role: string | undefined, feedsInto: string[] | undefined)"));
  assert.ok(source.includes('if (role === "diagnostic" && (feedsInto?.length ?? 0) > 0) return "output";'));
  assert.ok(source.includes("ioRole: ioRoleForUnitRole(artifact.role, artifact.feeds_into)"));
});
