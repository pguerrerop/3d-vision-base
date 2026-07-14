import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

function readSource(): string {
  return fs.readFileSync(path.resolve(process.cwd(), "src/pages/PipelineWalkthroughPage.tsx"), "utf-8");
}

test("walkthrough stage section separates strategy-path overview from child substeps", () => {
  const source = readSource();
  assert.ok(source.includes('substage.substageId === "strategy_path"'));
  assert.ok(source.includes('variant="overview"'));
  assert.ok(source.includes("walkthrough-substage-group"));
  assert.ok(source.includes("Stage overview"));
  assert.ok(source.includes("Substeps"));
});

test("walkthrough lightbox exposes image compare mode buttons and reuses the binary mask renderer when applicable", () => {
  const source = readSource();
  assert.ok(source.includes('shouldUseBinaryMaskRenderer'));
  assert.ok(source.includes('defaultBinaryMaskViewMode'));
  assert.ok(source.includes('<BinaryMaskRenderer'));
  assert.ok(source.includes('>Overlay</button>'));
  assert.ok(source.includes('>Side-by-side</button>'));
  assert.ok(source.includes('selectedReferenceUrl'));
  assert.ok(source.includes('walkthrough-lightbox-side-by-side'));
  assert.ok(source.includes('setImageMode("overlay")'));
  assert.ok(source.includes('setBinaryMaskMode(defaultBinaryMaskViewMode(selectedArtifact, allArtifacts))'));
});
