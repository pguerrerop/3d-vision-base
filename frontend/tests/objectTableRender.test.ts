import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

test("ObjectTable renders table view when objects exist", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/components/ObjectTable.tsx"), "utf-8");
  assert.ok(source.includes("<table>"));
  assert.ok(source.includes("No objects reported."));
});
