import test from "node:test";
import assert from "node:assert/strict";

import { formatBytes, formatPartialExecutionSummaryLine } from "../src/components/partialExecutionSummaryModel.ts";

test("formatBytes scales into KB/MB/GB", () => {
  assert.equal(formatBytes(0), "0 B");
  assert.equal(formatBytes(512), "512 B");
  assert.equal(formatBytes(2048), "2.0 KB");
  assert.equal(formatBytes(5 * 1024 * 1024), "5.0 MB");
});

test("formatPartialExecutionSummaryLine renders reused/rerun/duration compactly", () => {
  const line = formatPartialExecutionSummaryLine({
    reused_artifact_count: 12,
    reused_artifact_bytes: 2048,
    rerun_artifact_count: 5,
    partial_execution_duration_ms: 340,
    boundary_stage_id: "classification",
    parent_run_id: "run_parent",
  });
  assert.equal(line, "Reused: 12 artifacts (2.0 KB) — Rerun: 5 artifacts — Duration: 340 ms");
});

test("formatPartialExecutionSummaryLine returns null when summary is missing", () => {
  assert.equal(formatPartialExecutionSummaryLine(null), null);
  assert.equal(formatPartialExecutionSummaryLine(undefined), null);
});
