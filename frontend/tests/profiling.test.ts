import assert from "node:assert/strict";
import test from "node:test";
import {
  buildOptimizationHints,
  buildProfilingBadges,
  buildTimelineSegments,
  getSlowestStage,
  sortedProfilingStages,
} from "../src/profiling.ts";
import type { PipelineInfo, ProcessingProfiling, ProcessingResult } from "../src/api/client.ts";
import { availableInputTabs } from "../src/components/inputAssetsModel.ts";
import { calibrationStateLabel } from "../src/components/calibrationStateModel.ts";
import { imageAgeSeconds, livePreviewLabel, livePreviewStatus } from "../src/components/livePreviewModel.ts";
import { activePipelineName, chooseBestPipelineForTake, compactTakeFamilyStatus, incompatibleModalityMessage, modalityCompatible, pipelineCompatibilityGroups, resolveStageViewSelection, stageScopedViewTabs, stageTimingMs, studioEmptyState, tabForStage } from "../src/components/pipelineModel.ts";
import { canonicalStageId, stageEmptyState, stageSemanticDefinition, stageSemanticSummary } from "../src/components/stageSemantics.ts";
import { resolveStageViewContext } from "../src/components/stage_sources/index.ts";
import { clampPolygon, clampRoi, mapClientToImagePoint, pointsToRoi, sourceToDisplay } from "../src/components/roiMapping.ts";
import { normalizeBlobDetectionArtifacts } from "../src/studio/blobDetectionModel.ts";
import { PRODUCT_NAV_ITEMS, productAreaForPath } from "../src/productNavigation.ts";
import { artifactLineage, artifactList, artifactsForStage, blobCandidatesFromArtifacts, blobRejectedCountFromArtifacts, executionStage, firstArtifactByAlias, objectGeneratingStages, objectsForTake, overlayArtifacts, selectedObject, stageInspectorSummary } from "../src/components/studioWorkspaceModel.ts";
import { resolveOverlayTarget, transformOverlayGeometry, visibleOverlays } from "../src/components/overlayModel.ts";

const profiling: ProcessingProfiling = {
  total_ms: 2600,
  production_ms: 450,
  debug_artifacts_ms: 80,
  io_ms: 2108,
  stages: [
    {
      name: "load_point_cloud",
      category: "io",
      duration_ms: 2108,
      input_points: 1200000,
      output_points: 1200000,
      notes: null,
    },
    {
      name: "voxel_downsample",
      category: "preprocessing",
      duration_ms: 167,
      input_points: 1200000,
      output_points: 240000,
      notes: null,
    },
    {
      name: "remove_outliers",
      category: "preprocessing",
      duration_ms: 191,
      input_points: 240000,
      output_points: 210000,
      notes: null,
    },
    {
      name: "dbscan_clustering",
      category: "clustering",
      duration_ms: 40,
      input_points: 210000,
      output_points: 210000,
      notes: null,
    },
  ],
};

test("timeline renders stages with proportional bars", () => {
  const segments = buildTimelineSegments(profiling);

  assert.equal(segments.length, 4);
  assert.equal(segments[0].name, "load_point_cloud");
  assert.equal(segments[0].widthPercent, 100);
  assert.ok(segments[1].widthPercent < segments[0].widthPercent);
});

test("slowest stage is highlighted and sorted first", () => {
  const slowest = getSlowestStage(profiling);
  const sorted = sortedProfilingStages(profiling);

  assert.equal(slowest?.name, "load_point_cloud");
  assert.equal(sorted[0].name, "load_point_cloud");
  assert.equal(sorted[0].isSlowest, true);
  assert.equal(sorted[0].isDominant, true);
});

test("optimization hints appear correctly", () => {
  assert.deepEqual(buildOptimizationHints(profiling), [
    "PLY loading dominates runtime. Consider NPZ runtime format.",
    "Outlier removal is expensive for current point count.",
  ]);

  assert.deepEqual(
    buildOptimizationHints({
      ...profiling,
      production_ms: 50,
      debug_artifacts_ms: 500,
      stages: [{ name: "load_point_cloud_fast", category: "io", duration_ms: 20 }],
    }),
    ["Debug rendering dominates latency."]
  );
});

test("empty profiling is handled gracefully", () => {
  assert.deepEqual(buildTimelineSegments(null), []);
  assert.deepEqual(buildTimelineSegments({ total_ms: 0, production_ms: 0, debug_artifacts_ms: 0, io_ms: 0, stages: [] }), []);
  assert.equal(getSlowestStage(undefined), null);
  assert.deepEqual(buildOptimizationHints(undefined), []);
});

test("mode badges expose production diagnostics", () => {
  const result = {
    algorithm_stage: "calibrated_segmentation",
    calibration_id: "belt_setup",
    profiling: {
      ...profiling,
      stages: [
        { name: "load_point_cloud_fast", category: "io", duration_ms: 20 },
        { name: "render_debug", category: "debug_artifacts", duration_ms: 0, notes: "skipped by --skip-debug-images" },
      ],
    },
  } as ProcessingResult;

  assert.deepEqual(
    buildProfilingBadges(result).map((badge) => badge.label),
    ["calibrated", "debug-images skipped", "fast cloud used", "calibration active"]
  );
});

test("input assets tabs show only available modalities plus metadata", () => {
  assert.deepEqual(
    availableInputTabs({
      take_id: "take_001",
      status: "processed",
      has_ready: true,
      has_done: true,
      created_at: null,
      metadata: { files: { point_cloud: "point_cloud.ply" } },
      result: null,
      modalities: ["point_cloud"],
      assets: { point_cloud: { point_cloud: "point_cloud.ply" } },
    }),
    ["point_cloud", "metadata"]
  );

  assert.deepEqual(
    availableInputTabs({
      take_id: "take_002",
      status: "incoming",
      has_ready: true,
      has_done: false,
      created_at: null,
      metadata: { files: { rgb: "rgb.png" } },
      result: null,
      modalities: ["rgb"],
      assets: { rgb: { rgb: "rgb.png" } },
    }),
    ["rgb", "metadata"]
  );

  assert.deepEqual(
    availableInputTabs({
      take_id: "take_003",
      status: "incoming",
      has_ready: true,
      has_done: false,
      created_at: null,
      metadata: { files: { rgb: "preview.png", rgb_video: "rgb_video.mp4" } },
      result: null,
      modalities: ["rgb", "rgb_video"],
      assets: { rgb: { rgb: "preview.png" }, rgb_video: { rgb_video: "rgb_video.mp4" } },
    }),
    ["rgb", "rgb_video", "metadata"]
  );
});

test("live preview states cover empty, stale, connected, and disconnected", () => {
  const connectedState = {
    status: "acquiring",
    latest_take_id: null,
    latest_processed_take_id: null,
    message: null,
    updated_at: null,
    acquisition_connected: true,
    queue_size: 0,
    dropped_frames: 0,
    acquisition_source: "usb_camera",
    stale: false,
  };

  assert.equal(livePreviewStatus(null, null), "unavailable");
  assert.equal(livePreviewStatus(connectedState, { source: "usb_camera", timestamp: null, stale: true }), "unavailable");
  assert.equal(livePreviewStatus(connectedState, { source: "usb_camera", timestamp: "2026-05-17T12:00:00Z", stale: true }), "stale");
  assert.equal(livePreviewStatus(connectedState, { source: "usb_camera", timestamp: "2026-05-17T12:00:00Z", stale: false }), "connected");
  assert.equal(livePreviewStatus({ ...connectedState, acquisition_connected: false }, { source: "usb_camera", timestamp: "2026-05-17T12:00:00Z", stale: false }), "disconnected");
  assert.equal(livePreviewLabel("stale"), "STALE");
  assert.equal(imageAgeSeconds("2026-05-17T12:00:00Z", Date.parse("2026-05-17T12:00:02Z")), 2);
});

test("calibration state labels cover none active and incompatible", () => {
  assert.equal(calibrationStateLabel({ active: false, path: null, calibration_id: null, status: "none", calibration_type: "plane_3d" }), "No plane_3d");
  assert.equal(calibrationStateLabel({ active: true, path: "config/calibrations/x.json", calibration_id: "x", status: "valid", calibration_type: "plane_3d" }), "Active plane_3d");
  assert.equal(calibrationStateLabel({ active: true, path: "config/calibrations/x.json", calibration_id: "x", status: "incompatible", calibration_type: "plane_3d" }), "Incompatible plane_3d");
});

test("pipeline model separates compatibility and stage timing", () => {
  const pipeline = {
    id: "2d_3d_fusion",
    name: "2d_3d_fusion",
    display_name: "2D/3D Fusion",
    required_modalities: ["point_cloud", "rgb"],
    optional_modalities: ["rgb_video"],
    implemented: false,
    stages: [{ id: "fusion", display_name: "Fused evidence" }],
  };

  assert.equal(modalityCompatible(pipeline, ["point_cloud"]), false);
  assert.equal(modalityCompatible(pipeline, ["point_cloud", "rgb"]), true);
  assert.equal(incompatibleModalityMessage(pipeline, ["rgb"]), "Selected take has rgb. 2D/3D Fusion requires point_cloud, rgb.");
  assert.equal(activePipelineName({ processing_pipeline: { name: "inspection", display_name: "3D Ball Inspection", required_modalities: ["point_cloud"] } } as ProcessingResult), "3D Ball Inspection");
  assert.equal(stageTimingMs({ profiling } as ProcessingResult, "clustering"), 40);
});

test("pipeline discovery groups compatibility and auto-selection chooses best compatible", () => {
  const pipelines = [
    {
      id: "3d_ball_inspection",
      name: "3d_ball_inspection",
      display_name: "3D Ball Inspection",
      required_modalities: ["point_cloud"],
      optional_modalities: ["rgb"],
      implemented: true,
      pipeline_family: "3d",
    },
    {
      id: "mining_steel_ball_classification_2d",
      name: "mining_steel_ball_classification_2d",
      display_name: "RGB Steel Ball Classification MVP",
      required_modalities: ["rgb"],
      optional_modalities: [],
      implemented: true,
      pipeline_family: "2d",
    },
  ] as unknown as PipelineInfo[];

  const rgbGroups = pipelineCompatibilityGroups(pipelines, ["rgb"]);
  assert.equal(rgbGroups.compatible[0].id, "mining_steel_ball_classification_2d");
  assert.equal(rgbGroups.incompatible[0].pipeline.id, "3d_ball_inspection");

  const pointCloudChoice = chooseBestPipelineForTake(pipelines, ["point_cloud"], null);
  assert.equal(pointCloudChoice?.id, "3d_ball_inspection");
  const rememberedRgb = chooseBestPipelineForTake(pipelines, ["rgb"], "mining_steel_ball_classification_2d");
  assert.equal(rememberedRgb?.id, "mining_steel_ball_classification_2d");
});

test("take status compaction hides generic and marks unavailable families", () => {
  const compactRgb = compactTakeFamilyStatus(
    [
      { family: "2d", hasCompletedOutput: false, status: "never_processed" },
      { family: "3d", hasCompletedOutput: true, status: "completed" },
      { family: "generic", hasCompletedOutput: false, status: "never_processed" },
    ],
    ["rgb"]
  );
  assert.equal(compactRgb, "2D: not run • 3D: unavailable");

  const compactPointCloud = compactTakeFamilyStatus(
    [
      { family: "2d", hasCompletedOutput: true, status: "completed" },
      { family: "3d", hasCompletedOutput: true, status: "completed" },
    ],
    ["point_cloud"]
  );
  assert.equal(compactPointCloud, "2D: done • 3D: done");
});

test("stage-scoped tabs are contextual and fallback is explicit", () => {
  const thresholdTabs = stageScopedViewTabs([
    { kind: "image", title: "Threshold mask" },
    { kind: "image", title: "Normalized grayscale" },
    { kind: "overlay", title: "Contours" },
  ]);
  assert.deepEqual(thresholdTabs.map((item) => item.id), ["overview", "image", "mask", "overlay", "artifacts", "json"]);

  const preserved = resolveStageViewSelection("overlay", thresholdTabs);
  assert.equal(preserved.next, "overlay");
  assert.equal(preserved.fallbackMessage, null);

  const fallback = resolveStageViewSelection("metrics", thresholdTabs);
  assert.equal(fallback.next, "image");
  assert.equal(fallback.fallbackMessage, "Selected stage changed. Showing default view: Image.");
});

test("stage semantic definitions and empty states are stage-aware", () => {
  const thresholdSemantic = stageSemanticDefinition("threshold_morphology");
  assert.equal(thresholdSemantic.category, "segmentation");
  assert.equal(thresholdSemantic.defaultViewId, "overlay");
  assert.ok(thresholdSemantic.views.some((view) => view.id === "cleaned_mask"));

  const ellipseSemantic = stageSemanticDefinition("ellipse_fitting_metrics");
  assert.equal(ellipseSemantic.category, "geometry");
  assert.ok(ellipseSemantic.views.some((view) => view.id === "ellipse_overlay"));

  const empty = stageEmptyState("threshold_morphology", "threshold_mask");
  assert.equal(empty.title, "No threshold mask generated yet.");
});

test("stage semantic summaries provide operational wording", () => {
  const detail = {
    result: {
      objects: [{ object_id: 1, class_name: "ball", confidence: 0.8, center_mm: null, diameter_mm: 35, sphericity_score: 0.94, fit_rmse_mm: null }],
      rejected_objects: [{ object_id: 2, class_name: "non_ball", confidence: 0.3, center_mm: null, diameter_mm: 20, sphericity_score: 0.5, fit_rmse_mm: null, filter_status: "rejected" }],
    },
  } as unknown as import("../src/api/client.ts").TakeDetail;
  const segmentationSummary = stageSemanticSummary("threshold_morphology", detail, [{ artifact_id: "a", stage_id: "threshold_morphology", kind: "image", title: "Mask" } as any], null);
  assert.equal(segmentationSummary, "1 masks generated • Foreground: 0.0%");
  const classificationSummary = stageSemanticSummary("classification_summary", detail, [], null);
  assert.equal(classificationSummary, "1 classified • 1 rejected");
  const withMorphMetrics = stageSemanticSummary(
    "threshold_morphology",
    detail,
    [
      { artifact_id: "threshold_mask", stage_id: "segmentation", kind: "image", title: "Threshold" } as any,
      {
        artifact_id: "morphology_metrics",
        stage_id: "segmentation",
        kind: "metric",
        title: "Metrics",
        metadata: { foreground_coverage_percent: 8.4, connected_components: 3, roi_enabled: true },
      } as any,
    ],
    null
  );
  assert.equal(withMorphMetrics, "1 masks generated • Foreground: 8.4% • 3 connected regions • ROI enabled");
});

test("detection stage aliases resolve to canonical detection views", () => {
  assert.equal(canonicalStageId("detection"), "detection");
  assert.equal(canonicalStageId("blob_detection"), "detection");
  assert.equal(canonicalStageId("blob_contour_detection"), "detection");
  assert.equal(canonicalStageId("x", "Blob/contour detection"), "detection");

  const direct = stageSemanticDefinition("detection").views.map((view) => view.id);
  const alias = stageSemanticDefinition("blob_detection").views.map((view) => view.id);
  const displayAlias = stageSemanticDefinition(canonicalStageId("x", "Blob/contour detection")).views.map((view) => view.id);

  assert.deepEqual(direct, ["candidate_overlay", "labels", "blob_metrics", "candidates_table", "json"]);
  assert.deepEqual(alias, direct);
  assert.deepEqual(displayAlias, direct);
});

test("ellipse fitting stage exposes geometry metrics views", () => {
  const semantic = stageSemanticDefinition("ellipse_fitting");
  assert.equal(semantic.category, "geometry");
  assert.deepEqual(semantic.views.map((view) => view.id), ["ellipse_overlay", "ellipse_metrics", "candidates_table", "json"]);
  assert.equal(semantic.defaultViewId, "ellipse_overlay");
});

test("ellipse stage summary uses ellipse metrics entries", () => {
  const summary = stageSemanticSummary(
    "ellipse_fitting",
    null,
    [
      {
        artifact_id: "ellipse_metrics",
        stage_id: "ellipse_fitting",
        kind: "json",
        title: "Ellipse metrics",
        metadata: {
          entries: [{ candidate_id: 1, circularity: 0.88 }],
        },
      } as any,
    ],
    null
  );
  assert.ok(summary.includes("1 candidates"));
});

test("blob metrics binding resolves candidates and aliases", () => {
  const artifacts = [
    {
      artifact_id: "detection_metrics",
      stage_id: "blob_detection",
      kind: "json",
      title: "Detection metrics",
      preview_available: true,
      metadata: {
        semantic_kind: "blob_metrics",
        entries: [
          { blob_id: 1, area_px: 1200, equivalent_diameter: 39.1, circularity: 0.93, aspect_ratio: 1.05, solidity: 0.97, touches_border: false },
          { blob_id: 2, area_px: 980, equivalent_diameter: 35.3, circularity: 0.88, aspect_ratio: 1.12, solidity: 0.95, touches_border: false },
        ],
        rejected_count: 3,
      },
    },
  ] as any;
  const found = firstArtifactByAlias(artifacts, ["blob_metrics"]);
  assert.equal(found?.artifact_id, "detection_metrics");
  const rows = blobCandidatesFromArtifacts(artifacts);
  assert.equal(rows.length, 2);
  assert.equal(blobRejectedCountFromArtifacts(artifacts), 3);
  const summary = stageSemanticSummary("blob_detection", { result: { objects: [], rejected_objects: [] } } as any, artifacts, null);
  assert.equal(summary.includes("2 candidates"), true);
});

test("blob normalizer supports contour array-only schema and computes summary", () => {
  const artifacts = [
    {
      artifact_id: "blob_contours",
      stage_id: "blob_detection",
      kind: "json",
      title: "Contours",
      preview_available: true,
      metadata: {
        entries: [
          { blob_id: 1, contour_points: [[0, 0], [1, 1]], bbox: [0, 0, 10, 11], centroid: [5, 6] },
          { blob_id: 2, contour_points: [[2, 2], [3, 3]], bbox: [10, 10, 8, 9], centroid: [14, 15] },
        ],
      },
    },
  ] as any;
  const normalized = normalizeBlobDetectionArtifacts(artifacts);
  assert.equal(normalized.candidates.length, 2);
  assert.equal(normalized.summary.candidateCount, 2);
});

test("blob normalizer supports object schema with candidates", () => {
  const artifacts = [
    {
      artifact_id: "detection_metrics",
      stage_id: "blob_detection",
      kind: "json",
      title: "Metrics",
      preview_available: true,
      metadata: {
        semantic_kind: "blob_metrics",
        entries: {
          candidates: [
            { id: "a", area: 120, diameter: 12, circularity: 0.81, aspectRatio: 1.1, solidity: 0.9 },
          ],
        },
      },
    },
  ] as any;
  const normalized = normalizeBlobDetectionArtifacts(artifacts);
  assert.equal(normalized.candidates.length, 1);
  assert.equal(normalized.summary.avgCircularity?.toFixed(2), "0.81");
});

test("roi mapping converts display coordinates to source pixels and clamps bounds", () => {
  const p1 = mapClientToImagePoint(50, 25, { left: 0, top: 0, width: 100, height: 50 }, 1920, 1080);
  assert.equal(Math.round(p1.x), 960);
  assert.equal(Math.round(p1.y), 540);
  const roi = pointsToRoi({ x: -20, y: 40 }, { x: 2000, y: 1200 }, 1920, 1080);
  assert.equal(roi.x, 0);
  assert.equal(roi.y, 40);
  assert.equal(roi.width, 1920);
  assert.equal(roi.height, 1040);
  const clamped = clampRoi({ x: 1910, y: 1070, width: 50, height: 50 }, 1920, 1080);
  assert.equal(clamped.width, 10);
  assert.equal(clamped.height, 10);
  const display = sourceToDisplay({ x: 960, y: 540 }, 1920, 1080, 100, 50);
  assert.equal(Math.round(display.x), 50);
  assert.equal(Math.round(display.y), 25);
  const poly = clampPolygon([[-10, -10], [1930, 20], [30, 2000]], 1920, 1080);
  assert.deepEqual(poly, [[0, 0], [1920, 20], [30, 1080]]);
});

test("input stage resolves source artifacts before processing and run artifacts override when present", () => {
  const detail = {
    take_id: "take_rgb_001",
    status: "incoming",
    has_ready: true,
    has_done: false,
    created_at: "2026-05-18T10:00:00Z",
    metadata: { camera: "rgb_cam_1" },
    modalities: ["rgb"],
    assets: { rgb: { rgb: "rgb.png" } },
    result: null,
  } as any;

  const sourceOnly = resolveStageViewContext(detail, "input_image", "image", "input", []);
  assert.equal(sourceOnly.sourceArtifacts.some((item) => item.kind === "image"), true);
  assert.equal(sourceOnly.resolvedArtifacts.some((item) => item.kind === "image"), true);
  assert.equal(sourceOnly.usingSourceFallback, true);

  const withRun = {
    ...detail,
    result: {
      artifacts: [
        { artifact_id: "run_input", stage_id: "input_image", kind: "image", title: "Normalized input", path: "normalized.png", preview_available: true },
      ],
    },
  } as any;
  const preferredRun = resolveStageViewContext(withRun, "input_image", "image", "input", withRun.result.artifacts as any);
  assert.equal(preferredRun.resolvedArtifacts[0]?.artifact_id, "run_input");
  assert.equal(preferredRun.usingSourceFallback, false);
});

test("non-input stages remain run-artifact driven before processing", () => {
  const detail = {
    take_id: "take_rgb_001",
    status: "incoming",
    has_ready: true,
    has_done: false,
    created_at: "2026-05-18T10:00:00Z",
    metadata: {},
    modalities: ["rgb"],
    assets: { rgb: { rgb: "rgb.png" } },
    result: null,
  } as any;
  const threshold = resolveStageViewContext(detail, "threshold_morphology", "threshold_mask", "segmentation", []);
  assert.equal(threshold.resolvedArtifacts.length, 0);
  assert.equal(threshold.sourceArtifacts.length, 0);
});

test("stage resolver maps segmentation artifacts from process runs", () => {
  const detail = {
    take_id: "take_rgb_stage_001",
    status: "incoming",
    has_ready: true,
    has_done: false,
    created_at: "2026-05-18T10:00:00Z",
    metadata: {},
    modalities: ["rgb"],
    assets: { rgb: { rgb: "rgb.png" } },
    result: {
      artifacts: [
        { artifact_id: "threshold_mask", stage_id: "segmentation", kind: "image", title: "Threshold", path: "processes/runs/i/r/threshold_mask.png", preview_available: true },
        { artifact_id: "cleaned_mask", stage_id: "segmentation", kind: "image", title: "Cleaned", path: "processes/runs/i/r/cleaned_mask.png", preview_available: true },
        { artifact_id: "overlay_image", stage_id: "segmentation", kind: "image", title: "Overlay", path: "processes/runs/i/r/overlay_image.png", preview_available: true },
        { artifact_id: "morphology_debug_json", stage_id: "segmentation", kind: "json", title: "Morph Debug", path: "processes/runs/i/r/morphology_debug.json", preview_available: true },
      ],
    },
  } as any;
  const segmentationArtifacts = artifactsForStage(detail, "segmentation");
  const ids = segmentationArtifacts.map((item) => item.artifact_id);
  assert.equal(ids.includes("threshold_mask"), true);
  assert.equal(ids.includes("cleaned_mask"), true);
  assert.equal(ids.includes("overlay_image"), true);
});


test("product navigation exposes Operations Studio Runtime Calibration Diagnostics", () => {
  assert.deepEqual(PRODUCT_NAV_ITEMS.map((item) => item.label), ["Operations", "Studio", "Runtime", "Calibration", "Diagnostics"]);
  assert.equal(productAreaForPath("/operator"), "operations");
  assert.equal(productAreaForPath("/operator/inspection"), "operations");
  assert.equal(productAreaForPath("/processing-lab"), "studio");
  assert.equal(productAreaForPath("/debug"), "diagnostics");
  assert.equal(productAreaForPath("/studio"), "studio");
});

test("studio workspace maps stages to tabs and empty states", () => {
  assert.equal(tabForStage("segmentation"), "segmentation");
  assert.equal(tabForStage("classification"), "classification");
  assert.equal(tabForStage("measurement"), "measurements");
  assert.equal(tabForStage("registration"), "fusion");
  assert.equal(studioEmptyState("fusion", true, true), "Fusion workflows are not implemented yet. Future versions will support RGB/3D synchronization and multimodal processing.");
  assert.equal(studioEmptyState("segmentation", false, false), "This pipeline is incompatible with the selected take modalities.");
  assert.equal(studioEmptyState("segmentation", true, false), "This take is compatible with the selected pipeline. Run processing to generate stage outputs.");
});

test("studio workspace model exposes objects artifacts and inspector summary", () => {
  const detail = {
    take_id: "take_001",
    status: "processed",
    has_ready: true,
    has_done: true,
    created_at: null,
    metadata: null,
    modalities: ["point_cloud"],
    result: {
      objects: [{ object_id: 1, class_name: "unknown", confidence: null, center_mm: null, diameter_mm: null, sphericity_score: null, fit_rmse_mm: null }],
      rejected_objects: [{ object_id: 2, class_name: "unknown", confidence: 0.2, center_mm: null, diameter_mm: null, sphericity_score: null, fit_rmse_mm: null, filter_status: "rejected" }],
      stage_outputs: [
        { stage: "segmentation", display_name: "Object segmentation", artifacts: { clusters: "debug_clusters.png" } },
        { stage: "fusion", display_name: "Fusion", implemented: false, artifacts: {} },
      ],
      artifacts: [
        {
          artifact_id: "foreground_clusters",
          stage_id: "segmentation",
          kind: "image",
          title: "Foreground clusters",
          path: "debug_clusters.png",
          preview_available: true,
        },
      ],
      profiling,
    },
  } as unknown as import("../src/api/client.ts").TakeDetail;

  assert.equal(objectsForTake(detail).length, 2);
  assert.equal(selectedObject(objectsForTake(detail), 2)?.object_id, 2);
  assert.equal(artifactList(detail)[0].title, "Foreground clusters");
  assert.equal(stageInspectorSummary(detail, null, "segmentation").artifactCount, 1);
});

test("overlay and execution helpers expose hover/selection contracts", () => {
  const detail = {
    take_id: "take_004",
    status: "processed",
    has_ready: true,
    has_done: true,
    created_at: null,
    metadata: null,
    modalities: ["point_cloud"],
    result: {
      artifacts: [
        { artifact_id: "foreground_clusters", stage_id: "segmentation", kind: "image", title: "Foreground", path: "debug_clusters.png", preview_available: true },
        {
          artifact_id: "classification_overlay_ellipse_1",
          stage_id: "classification",
          object_id: 1,
          kind: "overlay",
          title: "Object #1 ellipse",
          preview_available: false,
          overlay_type: "ellipse",
          target_artifact_id: "foreground_clusters",
          source_artifact_ids: ["classification_object_1"],
          metadata: { derived_from: ["measurement_object_1"] },
        },
      ],
      pipeline_execution: {
        pipeline_id: "3d_ball_inspection",
        stages: [
          {
            stage_id: "classification",
            status: "success",
            started_at: 1,
            finished_at: 2,
            duration_ms: 1,
            warnings: [],
            errors: [],
            input_artifact_ids: ["foreground_clusters"],
            output_artifact_ids: ["classification_overlay_ellipse_1"],
            object_count: 1,
            rejected_count: 0,
          },
        ],
      },
    },
  } as unknown as import("../src/api/client.ts").TakeDetail;

  const artifacts = artifactList(detail);
  assert.equal(overlayArtifacts(artifacts, "foreground_clusters").length, 1);
  assert.equal(executionStage(detail, "classification")?.status, "success");
  assert.deepEqual(objectGeneratingStages(artifacts, 1), ["classification"]);
  const lineage = artifactLineage(artifacts, artifacts[1]);
  assert.equal(lineage.includes("classification_object_1"), true);
});

test("overlay transform handles image and normalized coordinates", () => {
  const imageOverlay = {
    artifact_id: "bbox",
    stage_id: "segmentation",
    kind: "overlay",
    title: "bbox",
    overlay_type: "bbox",
    coordinate_space: "image_pixel",
    geometry: { x: 100, y: 40, width: 20, height: 10 },
    preview_available: false,
  } as unknown as import("../src/components/studioWorkspaceModel.ts").StudioArtifact;
  const normalizedOverlay = {
    ...imageOverlay,
    artifact_id: "norm",
    coordinate_space: "normalized_image",
    geometry: { x: 0.5, y: 0.5, width: 0.2, height: 0.1 },
  } as unknown as import("../src/components/studioWorkspaceModel.ts").StudioArtifact;

  const imageTransform = transformOverlayGeometry(imageOverlay, { width: 1000, height: 500 });
  const normalizedTransform = transformOverlayGeometry(normalizedOverlay, { width: 1000, height: 500 });
  assert.equal((imageTransform.transformedGeometry as Record<string, number>).x, 100);
  assert.equal((normalizedTransform.transformedGeometry as Record<string, number>).x, 500);
  assert.equal((normalizedTransform.transformedGeometry as Record<string, number>).height, 50);
});

test("missing target handling prevents misleading overlay rendering", () => {
  const artifacts = [
    {
      artifact_id: "orphan_overlay",
      stage_id: "classification",
      kind: "overlay",
      title: "orphan",
      target_artifact_id: "missing",
      overlay_type: "ellipse",
      coordinate_space: "image_pixel",
      geometry: { cx: 100, cy: 100, rx: 10, ry: 10 },
      preview_available: false,
    },
  ] as unknown as import("../src/components/studioWorkspaceModel.ts").StudioArtifact[];
  const resolved = resolveOverlayTarget(artifacts, artifacts[0]);
  assert.equal(resolved.target, null);
  assert.equal(resolved.warning, "No renderable target artifact available.");
});

test("overlay grouping by target and mode works", () => {
  const overlays = [
    { artifact_id: "o1", kind: "overlay", object_id: 1 },
    { artifact_id: "o2", kind: "overlay", object_id: 2 },
  ] as unknown as import("../src/components/studioWorkspaceModel.ts").StudioArtifact[];
  assert.equal(visibleOverlays(overlays, "all", 1).length, 2);
  assert.equal(visibleOverlays(overlays, "selected_only", 1).length, 1);
  assert.equal(visibleOverlays(overlays, "hide", 1).length, 0);
});

test("world-space overlays are marked non-renderable", () => {
  const overlay = {
    artifact_id: "world",
    stage_id: "classification",
    kind: "overlay",
    title: "world",
    overlay_type: "ellipse",
    coordinate_space: "world_mm",
    geometry: { cx: 10, cy: 20, rx: 5, ry: 5 },
    preview_available: false,
  } as unknown as import("../src/components/studioWorkspaceModel.ts").StudioArtifact;
  const transformed = transformOverlayGeometry(overlay, { width: 640, height: 480 });
  assert.equal(transformed.renderable, false);
  assert.equal(transformed.approximate, true);
  assert.equal(
    transformed.warnings[0],
    "Overlay is approximate: world coordinates projected onto static debug image."
  );
});
