import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

function runtimeSource(): string {
  return fs.readFileSync(path.resolve(process.cwd(), "src/pages/RuntimePage.tsx"), "utf-8");
}

test("Runtime page keeps title and operations subtitle", () => {
  const source = runtimeSource();
  assert.ok(source.includes("<h2>Runtime</h2>"));
  assert.ok(source.includes("runtime-control workspace for acquisition, routing, processing, and system services"));
});

test("Default Take Destination panel exists with full routing controls", () => {
  const source = runtimeSource();
  assert.ok(source.includes("Default Take Destination"));
  assert.ok(source.includes("Session naming mode"));
  assert.ok(source.includes("Activate immediately"));
  assert.ok(source.includes("Persist as default"));
  assert.ok(source.includes("Use now"));
  assert.ok(source.includes("Save as persistent default"));
  assert.ok(source.includes("Clear default destination"));
});

test("Runtime page supports inline dataset/session creation workflows", () => {
  const source = runtimeSource();
  assert.ok(source.includes("Create dataset"));
  assert.ok(source.includes("Create session"));
  assert.ok(source.includes("api.createDataset"));
  assert.ok(source.includes("api.createDatasetSession"));
});

test("Runtime routing APIs are wired for activation/persistence/clear", () => {
  const source = runtimeSource();
  assert.ok(source.includes("api.runtimeRouting"));
  assert.ok(source.includes("api.activateRuntimeRouting"));
  assert.ok(source.includes("api.updateRuntimeDefaultDestination"));
  assert.ok(source.includes("api.clearRuntimeDefaultDestination"));
  assert.ok(source.includes("api.clearRuntimeRouting"));
});

test("Replay override and fallback messaging remains explicit", () => {
  const source = runtimeSource();
  assert.ok(source.includes("Replay override ACTIVE. Default acquisition destination is temporarily overridden."));
  assert.ok(source.includes("Replay is inactive. Historical replay metadata is stored but is not affecting acquisition routing."));
});

test("Acquisition routing hero still shows active destination and precedence", () => {
  const source = runtimeSource();
  assert.ok(source.includes("Current active destination"));
  assert.ok(source.includes("Routing precedence"));
  assert.ok(source.includes("ownership_explanation"));
});
