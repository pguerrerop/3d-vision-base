import test from "node:test";
import assert from "node:assert/strict";

import { resolveNextSelectedTakeId } from "../src/components/takeSelectionModel.ts";

test("keeps current selected take when still available", () => {
  const next = resolveNextSelectedTakeId({
    currentSelectedTakeId: "take_b",
    availableTakeIds: ["take_a", "take_b", "take_c"],
    preferredTakeId: "take_a",
  });
  assert.equal(next, "take_b");
});

test("selects preferred/latest only when current is missing", () => {
  const next = resolveNextSelectedTakeId({
    currentSelectedTakeId: "missing_take",
    availableTakeIds: ["take_z", "take_y"],
    preferredTakeId: "take_y",
  });
  assert.equal(next, "take_y");
});

test("returns null when list is empty", () => {
  const next = resolveNextSelectedTakeId({
    currentSelectedTakeId: "take_a",
    availableTakeIds: [],
    preferredTakeId: "take_a",
  });
  assert.equal(next, null);
});

test("does not change selection on reordered/refreshed list", () => {
  const next = resolveNextSelectedTakeId({
    currentSelectedTakeId: "take_2",
    availableTakeIds: ["take_3", "take_1", "take_2"],
    preferredTakeId: "take_3",
  });
  assert.equal(next, "take_2");
});
