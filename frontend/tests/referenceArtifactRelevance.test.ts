import test from "node:test";
import assert from "node:assert/strict";
import {
  preferredDetectArtifactsForView,
  resolveDetectReferenceArtifactRelevance,
  resolveDetectReferenceRunState,
  resolveDetectViewArtifacts,
} from "../src/components/referenceArtifactRelevance.ts";

const blobArtifacts = [
  { artifact_id: "plane_fit_debug", stage_id: "detect_belt_plane", kind: "json", title: "plane", preview_available: true, metadata: { background_selection: { gradient_debug: { strategy: "low_gradient_blob_height_clusters", component_mode: "height_aware", split_method: "disabled" } } } },
  { artifact_id: "height_aware_blob_components_overlay", stage_id: "detect_belt_plane", kind: "image", title: "ha", preview_available: true, metadata: {} },
  { artifact_id: "low_gradient_blob_components_overlay", stage_id: "detect_belt_plane", kind: "image", title: "xy", preview_available: true, metadata: {} },
  { artifact_id: "blob_cluster_score_table", stage_id: "detect_belt_plane", kind: "table", title: "clusters", preview_available: true, metadata: {} },
] as any;

test("blob components prefer height-aware artifact when run state is height-aware", () => {
  const state = resolveDetectReferenceRunState({ result: { processed_at: "2026-06-26T12:00:00Z" } } as any, blobArtifacts, null);
  const preferred = preferredDetectArtifactsForView("blob_components", blobArtifacts, state);
  assert.equal(preferred[0]?.artifact_id, "height_aware_blob_components_overlay");
});

test("height-aware comparison view resolves XY-only artifact", () => {
  const state = resolveDetectReferenceRunState({ result: { processed_at: "2026-06-26T12:00:00Z" } } as any, blobArtifacts, null);
  const preferred = preferredDetectArtifactsForView("height_aware_blob_components", blobArtifacts, state);
  assert.equal(preferred[0]?.artifact_id, "low_gradient_blob_components_overlay");
});

test("blob cluster relevance stays in detect-stage diagnostics instead of generic fallback", () => {
  const result = resolveDetectReferenceArtifactRelevance({
    viewId: "blob_clusters",
    artifacts: blobArtifacts,
    activeStrategy: "low_gradient_blob_height_clusters",
    componentMode: "height_aware",
    splitMethod: "disabled",
    fallbackUsed: false,
    fallbackStrategy: null,
    runUpdatedAt: "2026-06-26T12:00:00Z",
  });
  assert.equal(result.relevance, "active");
  assert.equal(result.group, "main");
});

test("run diagnostics override pending form state", () => {
  const state = resolveDetectReferenceRunState(
    { result: { processed_at: "2026-06-26T12:00:00Z" } } as any,
    blobArtifacts,
    {
      background_detection_strategy: "low_gradient_depth_plateaus",
      blob_component_mode: "xy_only",
      blob_split_method: "height_borders",
    },
  );
  assert.equal(state.strategy, "low_gradient_blob_height_clusters");
  assert.equal(state.componentMode, "height_aware");
  assert.equal(state.splitMethod, "disabled");
});

test("height-aware blob components never silently fall back to XY-only artifacts", () => {
  const artifacts = blobArtifacts.filter((artifact) => artifact.artifact_id !== "height_aware_blob_components_overlay");
  const state = resolveDetectReferenceRunState({ result: { processed_at: "2026-06-26T12:00:00Z" } } as any, artifacts, null);
  const preferred = preferredDetectArtifactsForView("blob_components", artifacts, state);
  assert.deepEqual(preferred, []);
});

test("selected surface prefers final selected-surface support over blob-cluster intermediate", () => {
  const artifacts = [
    ...blobArtifacts,
    { artifact_id: "raw_heightmap_preview", stage_id: "detect_belt_plane", kind: "image", title: "raw", preview_available: true, metadata: {} },
    { artifact_id: "selected_blob_cluster_overlay", stage_id: "detect_belt_plane", kind: "image", title: "selected cluster", preview_available: true, metadata: {} },
    { artifact_id: "reference_surface_selected_mask", stage_id: "detect_belt_plane", kind: "image", title: "selected surface", preview_available: true, metadata: {} },
  ] as any;
  const state = resolveDetectReferenceRunState({ result: { processed_at: "2026-06-26T12:00:00Z" } } as any, artifacts, null);
  const resolution = resolveDetectViewArtifacts("selected_surface", artifacts, state);
  assert.equal(resolution.primary?.artifact_id, "reference_surface_selected_mask");
  assert.equal(resolution.reference?.artifact_id, "raw_heightmap_preview");
});

test("selected surface falls back to reference-surface mask before raw heightmap", () => {
  const artifacts = [
    blobArtifacts[0],
    { artifact_id: "raw_heightmap_preview", stage_id: "detect_belt_plane", kind: "image", title: "raw", preview_available: true, metadata: {} },
    { artifact_id: "reference_surface_selected_mask", stage_id: "detect_belt_plane", kind: "image", title: "selected surface", preview_available: true, metadata: {} },
  ] as any;
  const state = resolveDetectReferenceRunState({ result: { processed_at: "2026-06-26T12:00:00Z" } } as any, artifacts, null);
  const resolution = resolveDetectViewArtifacts("selected_surface", artifacts, state);
  assert.equal(resolution.primary?.artifact_id, "reference_surface_selected_mask");
  assert.equal(resolution.reference?.artifact_id, "raw_heightmap_preview");
});

test("selected support lineage resolves as an active blob-strategy artifact", () => {
  const artifacts = [
    ...blobArtifacts,
    { artifact_id: "selected_support_lineage", stage_id: "detect_belt_plane", kind: "json", title: "Selected support lineage", preview_available: true, metadata: { steps: [] } },
  ] as any;
  const state = resolveDetectReferenceRunState({ result: { processed_at: "2026-06-26T12:00:00Z" } } as any, artifacts, null);
  const relevance = resolveDetectReferenceArtifactRelevance({
    viewId: "selected_support_lineage",
    artifacts,
    activeStrategy: state.strategy,
    componentMode: state.componentMode,
    splitMethod: state.splitMethod,
    fallbackUsed: state.fallbackUsed,
    fallbackStrategy: state.fallbackStrategy,
    runUpdatedAt: state.processedAt,
  });
  const resolution = resolveDetectViewArtifacts("selected_support_lineage", artifacts, state);
  assert.equal(relevance.relevance, "active");
  assert.equal(resolution.primary?.artifact_id, "selected_support_lineage");
});

test("selected cluster refinement views resolve pre/post-refine support masks", () => {
  const artifacts = [
    ...blobArtifacts,
    { artifact_id: "selected_blob_cluster_pre_refine_mask", stage_id: "detect_belt_plane", kind: "image", title: "pre", preview_available: true, metadata: {} },
    { artifact_id: "selected_blob_cluster_refined_mask", stage_id: "detect_belt_plane", kind: "image", title: "post", preview_available: true, metadata: {} },
  ] as any;
  const state = resolveDetectReferenceRunState({ result: { processed_at: "2026-06-26T12:00:00Z" } } as any, artifacts, null);
  assert.equal(resolveDetectViewArtifacts("selected_cluster_pre_refine", artifacts, state).primary?.artifact_id, "selected_blob_cluster_pre_refine_mask");
  assert.equal(resolveDetectViewArtifacts("selected_cluster_refined", artifacts, state).primary?.artifact_id, "selected_blob_cluster_refined_mask");
});

test("reference-model support and suppression-mask views resolve semantic fit-support artifacts", () => {
  const artifacts = [
    ...blobArtifacts,
    { artifact_id: "reference_model_support_mask", stage_id: "detect_belt_plane", kind: "image", title: "fit", preview_available: true, metadata: {} },
    { artifact_id: "reference_suppression_mask", stage_id: "detect_belt_plane", kind: "image", title: "suppression", preview_available: true, metadata: {} },
  ] as any;
  const state = resolveDetectReferenceRunState({ result: { processed_at: "2026-06-26T12:00:00Z" } } as any, artifacts, null);
  assert.equal(resolveDetectViewArtifacts("reference_model_support", artifacts, state).primary?.artifact_id, "reference_model_support_mask");
  assert.equal(resolveDetectViewArtifacts("reference_suppression_mask", artifacts, state).primary?.artifact_id, "reference_suppression_mask");
});

test("bg-and-stripes strategy resolves explicit surface-role masks", () => {
  const artifacts = [
    { artifact_id: "plane_fit_debug", stage_id: "detect_belt_plane", kind: "json", title: "plane", preview_available: true, metadata: { reference_method: { background_detection_strategy: "low_gradient_bg_and_stripes" } } },
    { artifact_id: "belt_bg_mask", stage_id: "detect_belt_plane", kind: "image", title: "bg", preview_available: true, metadata: {} },
    { artifact_id: "belt_stripes_mask", stage_id: "detect_belt_plane", kind: "image", title: "stripes", preview_available: true, metadata: {} },
    { artifact_id: "unknown_low_gradient_mask", stage_id: "detect_belt_plane", kind: "image", title: "unknown", preview_available: true, metadata: {} },
    { artifact_id: "surface_suppression_mask", stage_id: "detect_belt_plane", kind: "image", title: "suppression", preview_available: true, metadata: {} },
    { artifact_id: "object_search_domain_mask", stage_id: "detect_belt_plane", kind: "image", title: "domain", preview_available: true, metadata: {} },
  ] as any;
  const state = resolveDetectReferenceRunState({ result: { processed_at: "2026-07-02T12:00:00Z" } } as any, artifacts, null);
  assert.equal(state.strategy, "low_gradient_bg_and_stripes");
  assert.equal(resolveDetectViewArtifacts("belt_bg", artifacts, state).primary?.artifact_id, "belt_bg_mask");
  assert.equal(resolveDetectViewArtifacts("unknown_low_gradient", artifacts, state).primary?.artifact_id, "unknown_low_gradient_mask");
  assert.equal(resolveDetectViewArtifacts("surface_suppression", artifacts, state).primary?.artifact_id, "surface_suppression_mask");
  assert.equal(resolveDetectViewArtifacts("object_search_domain", artifacts, state).primary?.artifact_id, "object_search_domain_mask");
});

test("support-loss waterfall resolves as an active detect-stage diagnostic", () => {
  const artifacts = [
    ...blobArtifacts,
    { artifact_id: "support_loss_waterfall", stage_id: "detect_belt_plane", kind: "table", title: "waterfall", preview_available: true, metadata: { rows: [] } },
  ] as any;
  const state = resolveDetectReferenceRunState({ result: { processed_at: "2026-06-26T12:00:00Z" } } as any, artifacts, null);
  const relevance = resolveDetectReferenceArtifactRelevance({
    viewId: "support_loss_waterfall",
    artifacts,
    activeStrategy: state.strategy,
    componentMode: state.componentMode,
    splitMethod: state.splitMethod,
    fallbackUsed: state.fallbackUsed,
    fallbackStrategy: state.fallbackStrategy,
    runUpdatedAt: state.processedAt,
  });
  const resolution = resolveDetectViewArtifacts("support_loss_waterfall", artifacts, state);
  assert.equal(relevance.relevance, "active");
  assert.equal(resolution.primary?.artifact_id, "support_loss_waterfall");
});

test("rejected Z-connections stay inactive outside height-aware blob mode", () => {
  const result = resolveDetectReferenceArtifactRelevance({
    viewId: "height_aware_connectivity_rejected_edges",
    artifacts: [{
      artifact_id: "height_aware_connectivity_rejected_edges",
      stage_id: "detect_belt_plane",
      kind: "image",
      title: "rejected",
      preview_available: true,
      metadata: {},
    }] as any,
    activeStrategy: "low_gradient_blob_height_clusters",
    componentMode: "xy_only",
    splitMethod: "disabled",
  });
  assert.equal(result.relevance, "inactive_strategy");
});

test("plateau artifacts are demoted under blob strategy", () => {
  const result = resolveDetectReferenceArtifactRelevance({
    viewId: "plateau_plot",
    artifacts: [{
      artifact_id: "background_plateau_plot",
      stage_id: "detect_belt_plane",
      kind: "image",
      title: "plateau",
      preview_available: true,
      metadata: {},
    }] as any,
    activeStrategy: "low_gradient_blob_height_clusters",
    componentMode: "height_aware",
    splitMethod: "disabled",
  });
  assert.equal(result.relevance, "inactive_strategy");
  assert.equal(result.group, "legacy");
});
