import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

test("processing lab serializes detect reference strategy into 25D stage params", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/pages/ProcessingLabPage.tsx"), "utf-8");
  assert.ok(source.includes("function build25dStageParams"));
  assert.ok(source.includes('background_detection_strategy: String(paramsSource.background_detection_strategy ?? "low_gradient_surface")'));
});
