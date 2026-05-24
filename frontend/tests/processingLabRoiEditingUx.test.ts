import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

test("Edit visually starts direct ROI edit flow", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/pages/ProcessingLabPage.tsx"), "utf-8");
  assert.ok(source.includes("function startVisualRoiEdit()"));
  assert.ok(source.includes("setActiveTab(\"raw_heightmap\")"));
  assert.ok(source.includes("setRoiEditModeActive(true)"));
  assert.ok(source.includes("Draw reference-surface ROI"));
});

test("ROI edit mode state is passed to artifact explorer", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/pages/ProcessingLabPage.tsx"), "utf-8");
  assert.ok(source.includes("roiEditModeActive"));
  assert.ok(source.includes("roiEditStatus"));
  assert.ok(source.includes("onCancelRoiEdit: () => cancelVisualRoiEdit()"));
});

test("Apply ROI writes detect_belt_plane plane_fit_roi", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/pages/ProcessingLabPage.tsx"), "utf-8");
  assert.ok(source.includes("detect_belt_plane: {"));
  assert.ok(source.includes("plane_fit_roi: {"));
  assert.ok(source.includes("x: roi.x"));
  assert.ok(source.includes("height: roi.height"));
  assert.ok(source.includes("type: \"polygon\""));
  assert.ok(source.includes("plane_fit_roi_points"));
});

test("Heightmap input preview includes ROI controls", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/pages/ProcessingLabPage.tsx"), "utf-8");
  assert.ok(source.includes("canonicalSelectedStageId === \"load_heightmap\""));
  assert.ok(source.includes("Reference surface region"));
  assert.ok(source.includes("Edit visually"));
  assert.ok(source.includes("Use full frame"));
  assert.ok(source.includes("Vertical band ROI"));
});
