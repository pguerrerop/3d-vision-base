import test from "node:test";
import assert from "node:assert/strict";

import { formatRunTimestamp, runRecipeLabel, runTypeLabel, sortRunsNewestFirst } from "../src/components/runHistoryModel.ts";

test("sortRunsNewestFirst orders runs by created_at descending", () => {
  const runs = [
    { run_id: "a", run_type: "full_run" as const, created_at: "2026-01-01T00:00:00Z" },
    { run_id: "b", run_type: "partial_rerun" as const, created_at: "2026-01-03T00:00:00Z" },
    { run_id: "c", run_type: "full_run" as const, created_at: "2026-01-02T00:00:00Z" },
  ];
  const ordered = sortRunsNewestFirst(runs);
  assert.deepEqual(ordered.map((run) => run.run_id), ["b", "c", "a"]);
});

test("runTypeLabel maps run types to display labels", () => {
  assert.equal(runTypeLabel("full_run"), "Full run");
  assert.equal(runTypeLabel("partial_rerun"), "Partial rerun");
  assert.equal(runTypeLabel("legacy_3d_result"), "Legacy run");
  assert.equal(runTypeLabel(null), "Full run");
});

test("formatRunTimestamp handles missing and invalid timestamps", () => {
  assert.equal(formatRunTimestamp(null), "Unknown time");
  assert.equal(formatRunTimestamp("not-a-date"), "not-a-date");
  assert.notEqual(formatRunTimestamp("2026-01-01T00:00:00Z"), "Unknown time");
});

test("runRecipeLabel combines recipe id and version when available", () => {
  assert.equal(runRecipeLabel({ run_id: "a", run_type: "full_run", recipe_id: "r1", recipe_version_id: "v2" }), "r1 (v2)");
  assert.equal(runRecipeLabel({ run_id: "a", run_type: "full_run", recipe_id: "r1" }), "r1");
  assert.equal(runRecipeLabel({ run_id: "a", run_type: "full_run" }), null);
});
