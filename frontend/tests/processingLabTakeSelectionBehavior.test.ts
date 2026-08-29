import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

test("Processing Lab uses resolver-based auto-selection and preserves user click selection path", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/pages/ProcessingLabPage.tsx"), "utf-8");
  assert.ok(source.includes("resolveNextSelectedTakeId"));
  assert.ok(source.includes("currentSelectedTakeId: selectedTakeIdRef.current"));
  assert.ok(source.includes("setSelectedTakeId(take.take_id);"));
  assert.ok(source.includes("handleTakeSelection(take.take_id, index, event.metaKey || event.ctrlKey, event.shiftKey);"));
});

test("Apply + rerun preserves the selected artifact and object", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/pages/ProcessingLabPage.tsx"), "utf-8");
  const rerunFunction = source.slice(
    source.indexOf("async function runSelectedPipeline"),
    source.indexOf("async function processTakeWithSelectedPipeline")
  );
  assert.ok(!rerunFunction.includes("setSelectedArtifactId(null)"));
  assert.ok(!rerunFunction.includes("setSelectedObjectId(null)"));
});
