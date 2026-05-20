import test from "node:test";
import assert from "node:assert/strict";

import { containsRawUuidLabel, objectDisplayId, objectIdFromMeasurementRow, objectIdFromOverlay } from "../src/components/objectSelectionModel.ts";

test("object selection mapping is reciprocal and display IDs are friendly", () => {
  const overlay = { object_id: 7 };
  const row = { object_id: 7 };

  assert.equal(objectIdFromOverlay(overlay), 7);
  assert.equal(objectIdFromMeasurementRow(row), 7);
  assert.equal(objectDisplayId(7), "object_007");
  assert.equal(objectDisplayId("12"), "object_012");
});

test("friendly labels do not expose raw UUIDs", () => {
  assert.equal(containsRawUuidLabel("object_001"), false);
  assert.equal(containsRawUuidLabel("550e8400-e29b-41d4-a716-446655440000"), true);
});
