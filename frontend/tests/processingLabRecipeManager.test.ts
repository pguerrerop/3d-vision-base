import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

test("Processing Lab exposes compact recipe manager controls", async () => {
  const source = await readFile(new URL("../src/pages/ProcessingLabPage.tsx", import.meta.url), "utf8");
  assert.ok(source.includes("Save current as recipe"));
  assert.ok(source.includes("Save new version"));
  assert.ok(source.includes("Clone recipe"));
  assert.ok(source.includes("Reset edits to recipe"));
  assert.ok(source.includes("Recipe diff"));
});
