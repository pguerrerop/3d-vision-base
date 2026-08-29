import test from "node:test";
import assert from "node:assert/strict";

import {
  formatBoundaryLine,
  formatChildRunsSummary,
  formatParentLineageLine,
  formatReusedRerunLine,
} from "../src/components/runLineageModel.ts";

test("formatParentLineageLine renders parent run id and boundary stage", () => {
  const line = formatParentLineageLine({
    pipeline_id: "p",
    run_id: "run_child",
    parent: { run_id: "run_parent", run_type: "full_run", boundary_stage_id: "classification" },
    children: [],
    comparison_id: null,
  });
  assert.equal(line, "Parent: run_parent — Boundary: classification");
});

test("formatParentLineageLine returns null when there is no parent", () => {
  assert.equal(
    formatParentLineageLine({ pipeline_id: "p", run_id: "run_a", parent: null, children: [], comparison_id: null }),
    null,
  );
});

test("formatBoundaryLine and formatReusedRerunLine summarize entry fields", () => {
  const entry = { run_id: "run_parent", run_type: "full_run" as const, boundary_stage_id: "classification", summary: { reused_units: 18, rerun_units: 35, invalidated_units: 37 } };
  assert.equal(formatBoundaryLine(entry), "Boundary: classification");
  assert.equal(formatReusedRerunLine(entry), "Reused: 18 units — Rerun: 35 units — Invalidated: 37 units");
  assert.equal(formatBoundaryLine(null), null);
  assert.equal(formatReusedRerunLine(null), null);
});

test("formatChildRunsSummary reports child run count", () => {
  assert.equal(formatChildRunsSummary([]), "No child partial runs.");
  assert.equal(
    formatChildRunsSummary([
      { run_id: "run_b", run_type: "partial_rerun" },
      { run_id: "run_c", run_type: "partial_rerun" },
    ]),
    "Child partial runs (2)",
  );
});
