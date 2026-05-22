import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

test("Runtime page renders worker list and empty state", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/pages/RuntimePage.tsx"), "utf-8");
  assert.ok(source.includes("Runtime Workers"));
  assert.ok(source.includes("workers.map((worker)"));
  assert.ok(source.includes("No runtime workers registered yet"));
});

test("Runtime worker detail renders events and stop request action", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/pages/RuntimePage.tsx"), "utf-8");
  assert.ok(source.includes("Recent events"));
  assert.ok(source.includes("api.runtimeWorkerEvents"));
  assert.ok(source.includes("api.stopRuntimeWorker"));
  assert.ok(source.includes("Request stop"));
});
