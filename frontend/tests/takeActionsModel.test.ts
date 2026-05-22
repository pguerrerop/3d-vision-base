import test from "node:test";
import assert from "node:assert/strict";

import { canConfirmPermanentDelete } from "../src/components/takeActionsModel.ts";

test("permanent delete confirm requires exact take id", () => {
  assert.equal(canConfirmPermanentDelete("take_001", "take_001"), true);
  assert.equal(canConfirmPermanentDelete("take_001", " TAKE_001 "), false);
  assert.equal(canConfirmPermanentDelete("take_001", "take_002"), false);
});
