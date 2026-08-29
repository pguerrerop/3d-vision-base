import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

test("ProcessingUnitGraphWorkspace mounts a dedicated RunLineageBlock panel with lineage props", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/components/ProcessingUnitGraphWorkspace.tsx"), "utf-8");
  assert.ok(source.includes('import RunLineageBlock from "./RunLineageBlock"'));
  assert.ok(source.includes("<span>Run lineage</span>"));
  assert.ok(source.includes("pipelineId={pipelineId}"));
  assert.ok(source.includes("runId={activeRunId}"));
  assert.ok(source.includes("activeRunTypeLabel={activeRunTypeLabel}"));
  assert.ok(source.includes("onLineageRefreshed={onLineageRefreshed}"));
});

test("ProcessingUnitGraphWorkspace surfaces stale warnings and the performance summary line", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/components/ProcessingUnitGraphWorkspace.tsx"), "utf-8");
  assert.ok(source.includes("staleWarnings"));
  assert.ok(source.includes("stale-warning-banner"));
  assert.ok(source.includes("performanceSummaryLine"));
});

test("Processing Lab wires pipeline/run/lineage context into the Graph Workspace", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/pages/ProcessingLabPage.tsx"), "utf-8");
  assert.ok(source.includes("pipelineId={selectedPipeline?.id ?? null}"));
  assert.ok(source.includes("activeRunTypeLabel={graphWorkspaceActiveRunTypeLabel}"));
  assert.ok(source.includes("onLineageRefreshed={refreshPipelineRuns}"));
  assert.ok(source.includes("staleWarnings={graphWorkspaceStaleWarnings}"));
  assert.ok(source.includes("performanceSummaryLine={formatPartialExecutionSummaryLine(workspaceDetail?.result?.partial_execution_summary)}"));
});

test("RunLineageBlock offers a generate-comparison-to-parent action when no comparison exists", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/components/RunLineageBlock.tsx"), "utf-8");
  assert.ok(source.includes("Generate comparison to parent"));
  assert.ok(source.includes("generateComparison"));
  assert.ok(source.includes("onLineageRefreshed?.()"));
});

test("useRunLineage exposes a generateComparison action backed by the generate-comparison API call", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/components/useRunLineage.ts"), "utf-8");
  assert.ok(source.includes("api.generateRunComparisonToParent"));
  assert.ok(source.includes("generateComparison"));
});
