import test from "node:test";
import assert from "node:assert/strict";

import { buildLabelLookup } from "../src/studio/labelMaskLookup.ts";

test("buildLabelLookup maps each row's id to itself under the label modulus", () => {
  const lookup = buildLabelLookup([{ id: 1 }, { id: 7 }, { id: 9 }], new Set([9]));
  assert.deepEqual(lookup.get(1), { id: 1, isRejected: false });
  assert.deepEqual(lookup.get(7), { id: 7, isRejected: false });
  assert.deepEqual(lookup.get(9), { id: 9, isRejected: true });
  assert.equal(lookup.size, 3);
});

test("buildLabelLookup skips non-positive or non-numeric ids", () => {
  const lookup = buildLabelLookup([{ id: 0 }, { id: "" }, { id: -3 }, { id: 5 }], new Set());
  assert.deepEqual([...lookup.keys()], [5]);
});

test("buildLabelLookup wraps ids at 255, matching the backend's 8-bit label encoding", () => {
  const lookup = buildLabelLookup([{ id: 300 }], new Set());
  assert.deepEqual(lookup.get(300 % 255), { id: 300, isRejected: false });
});
