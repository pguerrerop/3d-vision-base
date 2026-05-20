import test from "node:test";
import assert from "node:assert/strict";

import { nextDatasetSelectionAfterCreate, nextSessionSelectionAfterCreate } from "../src/components/datasetSidebarModel.ts";

test("create dataset selects created dataset", () => {
  const current = [{ id: "a", name: "A" }, { id: "b", name: "B" }] as any;
  const created = { id: "new_ds", name: "New" } as any;

  const next = nextDatasetSelectionAfterCreate(current, created);

  assert.equal(next.selectedDataset, "new_ds");
  assert.equal(next.datasets[0].id, "new_ds");
});

test("create session selects created session", () => {
  const current = [{ id: "s1", dataset_id: "d1", name: "S1" }, { id: "s2", dataset_id: "d1", name: "S2" }] as any;
  const created = { id: "new_s", dataset_id: "d1", name: "New Session" } as any;

  const next = nextSessionSelectionAfterCreate(current, created);

  assert.equal(next.selectedSession, "new_s");
  assert.equal(next.sessions[0].id, "new_s");
});
