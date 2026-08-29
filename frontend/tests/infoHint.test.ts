import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

test("InfoHint renders a help button and dialog content when opened", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/components/InfoHint.tsx"), "utf-8");
  assert.match(source, /className=\{`info-hint-button/);
  assert.match(source, /role="dialog"/);
  assert.match(source, /defaultOpen = false/);
  assert.match(source, /When to increase/);
  assert.match(source, /content\.affects\.join\(" • "\)/);
});
