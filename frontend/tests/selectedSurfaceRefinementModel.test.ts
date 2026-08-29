import test from "node:test";
import assert from "node:assert/strict";
import { deriveSelectedSurfaceRefinementSummary } from "../src/components/selectedSurfaceRefinementModel.ts";

function artifact(artifact_id: string, metadata: Record<string, unknown>) {
  return {
    artifact_id,
    stage_id: "detect_belt_plane",
    kind: "json",
    title: artifact_id,
    preview_available: true,
    metadata,
  } as any;
}

test("selected-surface refinement summary handles full diagnostics", () => {
  const summary = deriveSelectedSurfaceRefinementSummary([
    artifact("selected_surface_debug", {
      selected_cluster_id: "cluster_007",
      selected_cluster_pixels: 3200,
      selected_surface_pixels: 2400,
      selected_surface_valid_roi_ratio: 0.22,
      refine_by_mad: true,
      mad_k: 2.5,
      mad_floor_mm: 1.0,
      keep_border_support: true,
      min_total_pixels: 1200,
      fallback_strategy: "low_gradient_depth_plateaus",
      fallback_used: false,
      fallback_reason: null,
    }),
  ]);

  assert.equal(summary.selectedClusterId, "cluster_007");
  assert.equal(summary.selectedClusterPixels, 3200);
  assert.equal(summary.selectedSurfacePixels, 2400);
  assert.equal(summary.keptRatio, 0.75);
  assert.equal(summary.madEnabled, true);
  assert.equal(summary.fallbackUsed, false);
});

test("selected-surface refinement summary handles partial diagnostics", () => {
  const summary = deriveSelectedSurfaceRefinementSummary([
    artifact("plane_fit_debug", {
      background_selection: {
        gradient_debug: {
          selected_cluster_id: "cluster_partial",
          fallback_used: false,
        },
        reference_method: {
          blob_cluster_refine_mad_k: 3.1,
          blob_cluster_refine_floor_mm: 0.8,
          blob_cluster_refine_keep_border_support: false,
        },
      },
    }),
  ]);

  assert.equal(summary.selectedClusterId, "cluster_partial");
  assert.equal(summary.madK, 3.1);
  assert.equal(summary.madFloor, 0.8);
  assert.equal(summary.keepBorderSupport, false);
  assert.equal(summary.selectedSurfacePixels, undefined);
});

test("selected-surface refinement summary handles missing diagnostics", () => {
  const summary = deriveSelectedSurfaceRefinementSummary([]);
  assert.equal(summary.selectedClusterId, undefined);
  assert.equal(summary.selectedSurfacePixels, undefined);
  assert.equal(summary.fallbackUsed, undefined);
});

test("selected-surface refinement summary surfaces fallback-used cases", () => {
  const summary = deriveSelectedSurfaceRefinementSummary([
    artifact("blob_cluster_selection_debug", {
      selected_cluster_id: "cluster_fallback",
      fallback_used: true,
      fallback_reason: "cluster_below_min_total_pixels",
      fallback_strategy: "low_gradient_depth_plateaus",
    }),
  ]);

  assert.equal(summary.selectedClusterId, "cluster_fallback");
  assert.equal(summary.fallbackUsed, true);
  assert.equal(summary.fallbackStrategy, "low_gradient_depth_plateaus");
  assert.equal(summary.reason, "cluster_below_min_total_pixels");
});
