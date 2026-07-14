import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

test("blue artifact relevance badge is labeled Used instead of Debug", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/components/referenceArtifactRelevance.ts"), "utf-8");
  assert.ok(source.includes('if (relevance === "debug") return "Used";'));
  assert.ok(!source.includes('if (relevance === "debug") return "Debug";'));
});
