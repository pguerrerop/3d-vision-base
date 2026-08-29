import test from "node:test";
import assert from "node:assert/strict";

import { isPartialRerunExecutionMode, partialRunNoticeText } from "../src/components/partialRunNoticeModel.ts";

test("isPartialRerunExecutionMode only recognizes the boundary rerun mode", () => {
  assert.equal(isPartialRerunExecutionMode("rerun_from_public_stage_boundary"), true);
  assert.equal(isPartialRerunExecutionMode("full_run"), false);
  assert.equal(isPartialRerunExecutionMode(null), false);
  assert.equal(isPartialRerunExecutionMode(undefined), false);
});

test("partialRunNoticeText includes parent run and boundary stage when available", () => {
  const text = partialRunNoticeText({ parentRunId: "run_abc123", boundaryStageId: "remove_belt_segment_objects" });
  assert.match(text, /run_abc123/);
  assert.match(text, /remove_belt_segment_objects/);
  assert.match(text, /parent run is unchanged/i);
});

test("partialRunNoticeText falls back to generic wording when fields are missing", () => {
  const text = partialRunNoticeText({});
  assert.match(text, /unknown parent run/);
  assert.match(text, /an earlier stage/);
});
