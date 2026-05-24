import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

test("Processing Lab uses resolver-based auto-selection and preserves user click selection path", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/pages/ProcessingLabPage.tsx"), "utf-8");
  assert.ok(source.includes("resolveNextSelectedTakeId"));
  assert.ok(source.includes("currentSelectedTakeId: selectedTakeIdRef.current"));
  assert.ok(source.includes("onClick={() => setSelectedTakeId(take.take_id)}"));
});
