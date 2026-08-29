import test from "node:test";
import assert from "node:assert/strict";

import { resolveBinaryMaskReferenceCandidates } from "../src/components/binaryMaskArtifacts.ts";
import { stageSemanticDefinition, type StageViewDefinition } from "../src/components/stageSemantics.ts";
import {
  formatStageMaskDiagnosticsNote,
  rankStageMaskReferenceCandidates,
  resolveStageMaskArtifact,
  resolveStageMaskDefaultMode,
  resolveStageMaskSpec,
} from "../src/components/stage_view_renderers/stageMaskViewSpec.ts";
import type { ProcessingResult } from "../src/api/client.ts";

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

function view(stageId: string, viewId: string): StageViewDefinition {
  const result = stageSemanticDefinition(stageId).views.find((entry) => entry.id === viewId);
  if (!result) throw new Error(`Missing view ${viewId} for ${stageId}`);
  return result;
}

test("stage mask view spec resolves the correct selected-surface mask artifact", () => {
  const selectedSurfaceView = view("detect_belt_plane", "selected_surface");
  const selected = artifact({ artifact_id: "reference_surface_selected_mask", title: "Selected surface" });
  const fallback = artifact({ artifact_id: "expanded_plane_mask", title: "Expanded plane" });
  const resolved = resolveStageMaskArtifact(selectedSurfaceView, [fallback, selected]);
  assert.equal(resolved?.artifact_id, "reference_surface_selected_mask");
});

test("stage mask reference ranking keeps 2.5D raw source ahead of diagnostics", () => {
  const maskView = view("detect_belt_plane", "selected_surface");
  const raw = artifact({
    artifact_id: "raw_heightmap",
    title: "Raw heightmap",
    metadata: { shape: [24, 24], semantic_type: "raw_sensor_z" },
  });
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
  const ranked = rankStageMaskReferenceCandidates(
    resolveBinaryMaskReferenceCandidates(mask, [mask, selectedSurface, raw]),
    maskView.maskViewSpec ?? {},
  );
  assert.equal(ranked[0]?.artifact.artifact_id, "raw_heightmap");
});

test("stage mask default mode honors per-view defaults", () => {
  const lowGradientView = view("detect_belt_plane", "low_gradient_mask");
  const lowGradientMask = artifact({ artifact_id: "low_gradient_mask", title: "Low gradient mask" });
  assert.equal(resolveStageMaskDefaultMode(lowGradientView, lowGradientMask, [lowGradientMask]), "overlay");

  const cleanedView = view("remove_belt_segment_objects", "cleaned_mask");
  const cleanedMask = artifact({ artifact_id: "final_object_mask", title: "Final object mask", stage_id: "remove_belt_segment_objects" });
  assert.equal(resolveStageMaskDefaultMode(cleanedView, cleanedMask, [cleanedMask]), "overlay");
});

test("stage mask reference ranking falls back cleanly when no preferred reference exists", () => {
  const stripesView = view("detect_belt_plane", "belt_stripes");
  const diagnostic = artifact({
    artifact_id: "selected_surface",
    title: "Selected surface",
    metadata: { shape: [20, 20] },
  });
  const mask = artifact({
    artifact_id: "belt_stripes_mask",
    title: "Belt stripes mask",
    metadata: { shape: [20, 20], source_artifact_id: "selected_surface" },
  });
  const ranked = rankStageMaskReferenceCandidates(
    resolveBinaryMaskReferenceCandidates(mask, [mask, diagnostic]),
    stripesView.maskViewSpec ?? {},
  );
  assert.equal(ranked[0]?.artifact.artifact_id, "selected_surface");
});

test("BinaryMaskRenderer remains a compatibility adapter over shared mask rendering helpers", async () => {
  const fs = await import("node:fs");
  const path = await import("node:path");
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/components/BinaryMaskRenderer.tsx"), "utf-8");
  assert.ok(source.includes("<MaskArtifactViewer"));
  assert.ok(source.includes("useMaskRenderAssets"));
});

test("semantic defaults apply by artifact id", () => {
  const planeInliersView = view("detect_belt_plane", "plane_inliers");
  const resolved = resolveStageMaskSpec(planeInliersView, artifact({ artifact_id: "final_plane_inlier_mask", title: "Plane inliers" }));
  assert.equal(resolved.defaultMode, "boundary");
  assert.equal(resolved.maskOnLabel, "Plane inlier");
  assert.equal(resolved.maskOffLabel, "Other");
});

test("description is surfaced in stage mask diagnostics note", () => {
  const roiView = view("detect_belt_plane", "plane_fit_roi");
  const resolved = resolveStageMaskSpec(roiView, artifact({ artifact_id: "plane_fit_roi_mask", title: "Plane-fit ROI" }));
  const note = formatStageMaskDiagnosticsNote(resolved, "Alignment: same pixel grid");
  assert.match(note, /ROI pixels eligible for reference-plane fitting/i);
  assert.match(note, /ROI mask/i);
});
