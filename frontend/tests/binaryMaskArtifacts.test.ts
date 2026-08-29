import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

import {
  availableBinaryMaskViewModes,
  defaultBinaryMaskViewMode,
  finiteOrDefault,
  isBinaryMaskArtifact,
  resolveBinaryMaskReference,
  resolveBinaryMaskReferenceCandidates,
  resolveBinaryMaskSourceArtifact,
  shouldUseBinaryMaskRenderer,
  validateBinaryMaskReferenceAlignment,
} from "../src/components/binaryMaskArtifacts.ts";
import type { ProcessingResult } from "../src/api/client.ts";
import { sourceArtifactsForTake } from "../src/components/stage_sources/index.ts";

type StudioArtifact = NonNullable<ProcessingResult["artifacts"]>[number];

function artifact(overrides: Partial<StudioArtifact>): StudioArtifact {
  return {
    artifact_id: "artifact",
    stage_id: "detect_belt_plane",
    kind: "image",
    title: "Artifact",
    preview_available: true,
    metadata: {},
    path: "artifact.png",
    ...overrides,
  };
}

test("detects binary masks from metadata first", () => {
  const candidate = artifact({
    artifact_id: "custom_binary",
    title: "Reference surface candidate",
    metadata: { semantic_type: "binary_mask" },
  });
  assert.equal(isBinaryMaskArtifact(candidate), true);
});

test("detects binary masks from fallback id and name heuristics", () => {
  const candidate = artifact({
    artifact_id: "low_gradient_mask",
    title: "Low gradient mask",
  });
  assert.equal(isBinaryMaskArtifact(candidate), true);
});

test("reference resolution honors explicit source_artifact_id first as computational lineage", () => {
  const source = artifact({
    artifact_id: "selected_surface",
    title: "Selected surface",
    metadata: { shape: [32, 48] },
  });
  const raw = artifact({
    artifact_id: "raw_heightmap",
    title: "Raw heightmap",
    metadata: { shape: [32, 48], semantic_type: "raw_sensor_z" },
  });
  const mask = artifact({
    artifact_id: "expanded_plane_mask",
    title: "Expanded plane mask",
    metadata: { shape: [32, 48] },
    source_artifact_id: "selected_surface" as never,
  });
  const resolution = resolveBinaryMaskReference(mask, [mask, source, raw], "raw_heightmap", "/raw.png");
  assert.equal(resolution.computedFrom?.artifact.artifact_id, "selected_surface");
  assert.equal(resolution.selected?.artifact.artifact_id, "raw_heightmap");
});

test("reference selector candidates appear when multiple references exist", () => {
  const raw = artifact({
    artifact_id: "raw_heightmap",
    title: "Raw heightmap",
    metadata: { shape: [20, 20], semantic_type: "raw_sensor_z" },
  });
  const selectedSurface = artifact({
    artifact_id: "selected_surface",
    title: "Selected surface",
    metadata: { shape: [20, 20] },
  });
  const mask = artifact({
    artifact_id: "expanded_plane_mask",
    title: "Expanded plane mask",
    metadata: { shape: [20, 20], source_artifact_id: "selected_surface" },
  });
  const candidates = resolveBinaryMaskReferenceCandidates(mask, [mask, selectedSurface, raw]);
  assert.equal(candidates.length >= 2, true);
});

test("detect reference-surface masks default to original acquisition/raw heightmap over selected surface", () => {
  const raw = artifact({
    artifact_id: "raw_heightmap",
    title: "Raw heightmap",
    metadata: { shape: [24, 24], semantic_type: "raw_sensor_z" },
    path: "raw_sensor_z.png",
  });
  const selectedSurface = artifact({
    artifact_id: "selected_surface",
    title: "Selected surface",
    metadata: { shape: [24, 24] },
  });
  const mask = artifact({
    artifact_id: "expanded_plane_mask",
    title: "Expanded plane mask",
    metadata: { shape: [24, 24], source_artifact_id: "selected_surface" },
  });
  const candidates = resolveBinaryMaskReferenceCandidates(mask, [mask, selectedSurface, raw]);
  assert.equal(candidates[0]?.artifact.artifact_id, "raw_heightmap");
  assert.equal(candidates[0]?.role, "original_acquisition");
  assert.equal(resolveBinaryMaskSourceArtifact(mask, [mask, selectedSurface, raw])?.artifact_id, "raw_heightmap");
});

test("resolver includes sourceArtifacts in candidate list and defaults to source raw heightmap", () => {
  const mask = artifact({
    artifact_id: "expanded_plane_mask",
    title: "Expanded plane mask",
    metadata: { shape: [24, 24], source_artifact_id: "reference_surface_selected_mask" },
  });
  const selectedSurface = artifact({
    artifact_id: "reference_surface_selected_mask",
    title: "Selected surface",
    metadata: { shape: [24, 24] },
  });
  const detail = {
    take_id: "take_25d_001",
    status: "processed",
    has_ready: true,
    has_done: true,
    created_at: "2026-06-26T10:00:00Z",
    metadata: { files: { heightmap_preview: "heightmap_preview.png" } },
    modalities: ["heightmap"],
    assets: { heightmap: { raw_heightmap: "heightmap_raw.png" } },
    result: null,
  } as any;
  const sourceArtifacts = sourceArtifactsForTake(detail, "detect_belt_plane");
  const resolution = resolveBinaryMaskReference({
    maskArtifact: mask,
    selectedTake: detail,
    selectedStageId: "detect_belt_plane",
    currentStageArtifacts: [mask, selectedSurface],
    allRunArtifacts: [mask, selectedSurface],
    sourceArtifacts,
    resolvedArtifacts: [...sourceArtifacts, mask, selectedSurface],
  });
  assert.equal(resolution.selected?.artifact.artifact_id, "raw_heightmap");
  assert.equal(resolution.diagnostics.sourceArtifactsReceived > 0, true);
});

test("selected surface is used only when no original/raw source exists", () => {
  const selectedSurface = artifact({
    artifact_id: "reference_surface_selected_mask",
    title: "Selected surface",
    metadata: { shape: [24, 24] },
  });
  const mask = artifact({
    artifact_id: "expanded_plane_mask",
    title: "Expanded plane mask",
    metadata: { shape: [24, 24], source_artifact_id: "reference_surface_selected_mask" },
  });
  const resolution = resolveBinaryMaskReference({
    maskArtifact: mask,
    selectedStageId: "detect_belt_plane",
    currentStageArtifacts: [mask, selectedSurface],
    allRunArtifacts: [mask, selectedSurface],
    sourceArtifacts: [],
    resolvedArtifacts: [mask, selectedSurface],
  });
  assert.equal(resolution.selected?.artifact.artifact_id, "reference_surface_selected_mask");
});

test("take thumbnail fallback appears after raw/source heightmap candidates", () => {
  const detail = {
    take_id: "take_25d_thumb",
    status: "processed",
    has_ready: true,
    has_done: true,
    created_at: "2026-06-26T10:00:00Z",
    metadata: {
      thumbnail_url: "https://example.com/thumb.png",
      files: { heightmap_preview: "heightmap_preview.png" },
    },
    modalities: ["heightmap"],
    assets: { heightmap: { raw_heightmap: "heightmap_raw.png" } },
    result: null,
  } as any;
  const sourceArtifacts = sourceArtifactsForTake(detail, "detect_belt_plane");
  const mask = artifact({
    artifact_id: "expanded_plane_mask",
    title: "Expanded plane mask",
    metadata: { shape: [24, 24] },
  });
  const candidates = resolveBinaryMaskReferenceCandidates({
    maskArtifact: mask,
    selectedTake: detail,
    selectedStageId: "detect_belt_plane",
    currentStageArtifacts: [mask],
    allRunArtifacts: [mask],
    sourceArtifacts,
    resolvedArtifacts: [...sourceArtifacts, mask],
  });
  const rawIndex = candidates.findIndex((item) => item.artifact.artifact_id === "raw_heightmap");
  const thumbIndex = candidates.findIndex((item) => item.role === "take_thumbnail");
  assert.equal(rawIndex >= 0, true);
  assert.equal(thumbIndex > rawIndex, true);
});

test("diagnostics report source artifact counts and checked take fields", () => {
  const mask = artifact({
    artifact_id: "expanded_plane_mask",
    title: "Expanded plane mask",
    metadata: { shape: [24, 24] },
  });
  const resolution = resolveBinaryMaskReference({
    maskArtifact: mask,
    selectedStageId: "detect_belt_plane",
    currentStageArtifacts: [mask],
    allRunArtifacts: [mask],
    sourceArtifacts: [],
    resolvedArtifacts: [mask],
  });
  assert.equal(resolution.diagnostics.sourceArtifactsReceived, 0);
  assert.equal(resolution.diagnostics.takeAssetFieldsChecked.includes("thumbnail_url"), true);
});

test("original acquisition resolves to raw heightmap when RGB is absent", () => {
  const raw = artifact({
    artifact_id: "hm_001",
    title: "Camera heightmap input preview",
    path: "heightmap_input_preview.png",
    metadata: { semantic_type: "raw_sensor_z", shape: [18, 18] },
  });
  const mask = artifact({
    artifact_id: "background_seed_mask",
    title: "Background seed mask",
    metadata: { shape: [18, 18] },
  });
  const candidates = resolveBinaryMaskReferenceCandidates(mask, [mask, raw]);
  assert.equal(candidates[0]?.label.includes("Raw heightmap"), true);
});

test("stage-aware fallback prefers segmentation normalized height first", () => {
  const normalized = artifact({
    artifact_id: "normalized_height",
    stage_id: "remove_belt_segment_objects",
    title: "Normalized height",
    metadata: { shape: [20, 20], semantic_type: "normalized_height" },
  });
  const raw = artifact({
    artifact_id: "raw_heightmap",
    stage_id: "remove_belt_segment_objects",
    title: "Raw heightmap",
    metadata: { shape: [20, 20] },
  });
  const mask = artifact({
    artifact_id: "cleaned_mask",
    stage_id: "remove_belt_segment_objects",
    title: "Cleaned mask",
    metadata: { shape: [20, 20] },
  });
  assert.equal(resolveBinaryMaskReference(mask, [mask, raw, normalized]).selected?.artifact.artifact_id, "normalized_height");
});

test("no source falls back to mask-only mode", () => {
  const mask = artifact({
    artifact_id: "cleaned_mask",
    stage_id: "remove_belt_segment_objects",
    title: "Cleaned mask",
  });
  const resolution = resolveBinaryMaskReference(mask, [mask]);
  assert.equal(resolution.alignment, "unavailable");
  assert.equal(defaultBinaryMaskViewMode(mask, [mask], resolution), "mask");
  assert.deepEqual(availableBinaryMaskViewModes(mask, [mask], resolution), ["mask"]);
});

test("dimension mismatch disables overlay and boundary while keeping side-by-side", () => {
  const source = artifact({
    artifact_id: "raw_heightmap_preview",
    title: "Raw heightmap",
    metadata: { shape: [30, 30] },
  });
  const mask = artifact({
    artifact_id: "valid_mask",
    title: "Valid mask",
    metadata: { shape: [20, 20] },
  });
  const resolution = resolveBinaryMaskReference(mask, [mask, source], "raw_heightmap_preview", "/raw.png");
  assert.equal(resolution.alignment, "dimension_mismatch");
  assert.deepEqual(availableBinaryMaskViewModes(mask, [mask, source], resolution), ["mask", "side_by_side"]);
});

test("runtime dimension mismatch disables overlay and boundary", () => {
  const source = artifact({
    artifact_id: "raw_heightmap_preview",
    title: "Raw heightmap",
    path: "raw.png",
  });
  const mask = artifact({
    artifact_id: "valid_mask",
    title: "Valid mask",
    metadata: { source_artifact_id: "raw_heightmap_preview" },
  });
  const validated = validateBinaryMaskReferenceAlignment(
    resolveBinaryMaskReference(mask, [mask, source], "raw_heightmap_preview", "/raw.png"),
    { width: 20, height: 20 },
    { width: 18, height: 20 },
  );
  assert.equal(validated.alignment, "dimension_mismatch");
});

test("binary mask renderer defaults to Boundary only when source is safe", () => {
  const source = artifact({
    artifact_id: "normalized_heightmap",
    title: "Normalized height",
    metadata: { shape: [20, 20] },
    path: "normalized.png",
  });
  const mask = artifact({
    artifact_id: "threshold_mask",
    title: "Threshold mask",
    stage_id: "remove_belt_segment_objects",
    metadata: { source_artifact_id: "normalized_heightmap", shape: [20, 20] },
  });
  const resolution = resolveBinaryMaskReference(mask, [mask, source], "normalized_heightmap", "/normalized.png");
  assert.equal(defaultBinaryMaskViewMode(mask, [mask, source], resolution), "boundary");
});

test("selected-surface style masks default to Overlay when a reference is available", () => {
  const source = artifact({
    artifact_id: "raw_heightmap_preview",
    title: "Raw heightmap",
    metadata: { shape: [20, 20], semantic_type: "raw_sensor_z" },
    path: "raw.png",
  });
  const mask = artifact({
    artifact_id: "reference_surface_selected_mask",
    title: "Selected surface",
    metadata: { source_artifact_id: "raw_heightmap_preview", shape: [20, 20] },
  });
  const resolution = resolveBinaryMaskReference(mask, [mask, source], "raw_heightmap_preview", "/raw.png");
  assert.equal(defaultBinaryMaskViewMode(mask, [mask, source], resolution), "overlay");
});

test("binary mask renderer defaults to Mask when source is unavailable", () => {
  const mask = artifact({
    artifact_id: "expanded_plane_mask",
    title: "Expanded plane mask",
  });
  const resolution = resolveBinaryMaskReference(mask, [mask]);
  assert.equal(defaultBinaryMaskViewMode(mask, [mask], resolution), "mask");
});

test("finiteOrDefault normalizes NaN opacity and dimension values", () => {
  assert.equal(finiteOrDefault("NaN", 0.52), 0.52);
  assert.equal(finiteOrDefault(undefined, 1), 1);
  assert.equal(finiteOrDefault(0.35, 1), 0.35);
});

test("generic image artifacts do not show mask comparison controls", () => {
  const heightmap = artifact({
    artifact_id: "normalized_heightmap",
    title: "Normalized heightmap",
    metadata: { semantic_type: "height_preview" },
  });
  assert.equal(isBinaryMaskArtifact(heightmap), false);
  assert.equal(shouldUseBinaryMaskRenderer(heightmap, [heightmap]), false);
});

test("binary mask view controls live in the right panel and renderer stays view-state driven", () => {
  const renderer = fs.readFileSync(path.resolve(process.cwd(), "src/components/BinaryMaskRenderer.tsx"), "utf-8");
  const viewer = fs.readFileSync(path.resolve(process.cwd(), "src/components/artifacts/mask/MaskArtifactViewer.tsx"), "utf-8");
  const page = fs.readFileSync(path.resolve(process.cwd(), "src/pages/ProcessingLabPage.tsx"), "utf-8");
  assert.ok(renderer.includes("mode: BinaryMaskViewMode;"));
  assert.ok(renderer.includes("onModeChange: (mode: BinaryMaskViewMode) => void;"));
  assert.ok(viewer.includes("<MaskViewerToolbar"));
  assert.equal(page.includes("Mask appearance"), false);
  assert.ok(page.includes("binaryMaskMode"));
});

test("processing lab resolves binary mask reference URLs from local artifact paths before validating modes", () => {
  const page = fs.readFileSync(path.resolve(process.cwd(), "src/pages/ProcessingLabPage.tsx"), "utf-8");
  assert.ok(page.includes("const binaryMaskSelectedReferenceUrl = useMemo("));
  assert.ok(page.includes("fileUrl(workspaceDetail.take_id, binaryMaskSelectedReference.artifact.path)"));
  assert.ok(page.includes("binaryMaskSelectedReferenceUrl"));
});

test("reference selector remains available in the shared mask viewer", () => {
  const selector = fs.readFileSync(path.resolve(process.cwd(), "src/components/artifacts/mask/MaskReferenceSelector.tsx"), "utf-8");
  const toolbar = fs.readFileSync(path.resolve(process.cwd(), "src/components/artifacts/mask/MaskViewerToolbar.tsx"), "utf-8");
  assert.ok(selector.includes("Mask visual reference"));
  assert.ok(toolbar.includes("<MaskReferenceSelector"));
});

test("selected reference drives overlay rendering and inferred/computed diagnostics are shown", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/components/artifacts/mask/MaskArtifactViewer.tsx"), "utf-8");
  assert.ok(source.includes("selectedReferenceUrl"));
  assert.ok(source.includes("Computed from:"));
  assert.ok(source.includes("inferred from stage context"));
});

test("binary mask renderer keeps image URL dependencies stable across rerenders", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/components/BinaryMaskRenderer.tsx"), "utf-8");
  assert.ok(source.includes("const maskUrl = useMemo("));
  assert.ok(source.includes("useMaskRenderAssets"));
});

test("binary mask view fallback effects avoid no-op state resets", () => {
  const renderer = fs.readFileSync(path.resolve(process.cwd(), "src/components/BinaryMaskRenderer.tsx"), "utf-8");
  const page = fs.readFileSync(path.resolve(process.cwd(), "src/pages/ProcessingLabPage.tsx"), "utf-8");
  assert.ok(renderer.includes("if (selectedSourceArtifactId === nextSourceArtifactId) return;"));
  assert.ok(renderer.includes("if (mode === nextMode) return;"));
  assert.ok(page.includes("if (viewState.binaryMaskReferenceArtifactId === nextReferenceArtifactId) return;"));
  assert.ok(page.includes("if (viewState.binaryMaskMode === nextMode) return;"));
});

test("processing lab preserves binary mask mode until artifact or view selection changes", () => {
  const page = fs.readFileSync(path.resolve(process.cwd(), "src/pages/ProcessingLabPage.tsx"), "utf-8");
  assert.ok(page.includes("const viewStateSelectionKey = useMemo("));
  assert.ok(page.includes("lastViewStateSelectionKeyRef"));
  assert.ok(page.includes("if (lastViewStateSelectionKeyRef.current === viewStateSelectionKey) return;"));
  assert.equal(page.includes("current.binaryMaskMode === selectionDrivenViewState.binaryMaskMode"), false);
});

test("binary mask renderer includes reference diagnostics and disabled states", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/components/artifacts/mask/MaskArtifactViewer.tsx"), "utf-8");
  const opacity = fs.readFileSync(path.resolve(process.cwd(), "src/components/artifacts/mask/MaskOpacityControls.tsx"), "utf-8");
  assert.ok(source.includes("Comparison source unavailable. Showing mask-only view."));
  assert.ok(source.includes("Selected reference:"));
  assert.ok(opacity.includes("Mask overlay opacity"));
  assert.ok(opacity.includes("Mask reference opacity"));
});

test("reference diagnostics call out inferred references", () => {
  const raw = artifact({
    artifact_id: "raw_heightmap",
    title: "Raw heightmap",
    metadata: { shape: [24, 24], semantic_type: "raw_sensor_z" },
  });
  const mask = artifact({
    artifact_id: "expanded_plane_mask",
    title: "Expanded plane mask",
    metadata: { shape: [24, 24] },
  });
  const resolution = resolveBinaryMaskReference(mask, [mask, raw]);
  assert.equal(resolution.inferred, true);
  assert.equal(resolution.warning, "Reference inferred from stage context.");
});

test("studio artifact explorer still gates binary-mask controls behind the dedicated renderer branch", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/components/StudioArtifactExplorer.tsx"), "utf-8");
  assert.ok(source.includes("shouldUseBinaryMaskRenderer"));
  assert.ok(source.includes("<BinaryMaskRenderer"));
});
