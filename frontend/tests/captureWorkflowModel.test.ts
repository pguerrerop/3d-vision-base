import test from "node:test";
import assert from "node:assert/strict";

import { buildCapturePayload, captureDisabledReason, nextSelectedTakeAfterCapture } from "../src/components/captureWorkflowModel.ts";

test("capture disabled without dataset/session", () => {
  assert.equal(captureDisabledReason("", ""), "Select a dataset first");
  assert.equal(captureDisabledReason("d1", ""), "Select an experiment session first");
  assert.equal(captureDisabledReason("d1", "s1"), null);
});

test("capture payload includes metadata fields", () => {
  const payload = buildCapturePayload({
    datasetId: "d1",
    sessionId: "s1",
    friendlyName: "  Take A  ",
    tagsText: "good, wet ,  ",
    expectedClass: "ball",
    expectedDiameterMm: "80",
    notes: " hello ",
  });

  assert.equal(payload.dataset_id, "d1");
  assert.equal(payload.dataset_session_id, "s1");
  assert.equal(payload.friendly_name, "Take A");
  assert.deepEqual(payload.tags, ["good", "wet"]);
  assert.equal(payload.expected_class, "ball");
  assert.equal(payload.expected_diameter_mm, 80);
  assert.equal(payload.notes, "hello");
});

test("new captured take is auto-selected", () => {
  assert.equal(nextSelectedTakeAfterCapture("old", "new_take"), "new_take");
});
