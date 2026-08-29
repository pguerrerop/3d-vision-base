import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

test("Processing Lab exposes a contract graph viewer tab and shared node selection wiring", async () => {
  const source = await readFile(new URL("../src/pages/ProcessingLabPage.tsx", import.meta.url), "utf8");
  const viewerSource = await readFile(new URL("../src/components/ProcessingUnitGraphViewer.tsx", import.meta.url), "utf8");
  assert.ok(source.includes('"graph"'));
  assert.ok(source.includes("Contract graph"));
  assert.ok(source.includes("ProcessingUnitGraphViewer"));
  assert.ok(source.includes("selectedNodeId={selectedGraphNodeId}"));
  assert.ok(source.includes("onSelectNode={focusProcessingUnitFromGraph}"));
  assert.ok(viewerSource.includes("Node detail"));
});
