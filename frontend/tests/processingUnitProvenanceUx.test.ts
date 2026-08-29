import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

test("Processing Lab exposes best-effort processing-unit provenance labels", async () => {
  const source = await readFile(new URL("../src/pages/ProcessingLabPage.tsx", import.meta.url), "utf8");
  assert.ok(source.includes("best_effort_artifact_registry"));
  assert.ok(source.includes("artifact_level"));
  assert.ok(source.includes("Selected unit contract / trace"));
  assert.ok(source.includes("Selected unit I/O"));
  assert.ok(source.includes("processingUnitsForStage"));
  assert.ok(source.includes("ContractTunePanel"));
});
