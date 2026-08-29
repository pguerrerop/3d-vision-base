import test from "node:test";
import assert from "node:assert/strict";

import { buildProcessingUnitGraph, formatTraceCoverageLine, graphSelectionForUnit, persistedTraceFromDetail, traceBadgeClass, traceBadgeLabel, traceSummaryFromDetail } from "../src/components/processingUnitGraphModel.ts";

const units = [
  {
    id: "input",
    label: "Input",
    kind: "stage",
    parent_id: null,
    stage_id: "load_heightmap",
    category: "input",
    order: 1,
    description: "Load inputs",
    inputs: [],
    outputs: [{ id: "out", label: "Raw", artifact_id: "raw_heightmap_preview" }],
    artifacts: [{ id: "raw_heightmap_preview", label: "Raw", artifact_id: "raw_heightmap_preview", kind: "image", role: "final", renderer: "image" }],
    parameters: [],
    diagnostics: [],
    views: [],
    default_view: null,
  },
  {
    id: "segment",
    label: "Segment",
    kind: "stage",
    parent_id: null,
    stage_id: "remove_belt_segment_objects",
    category: "segmentation",
    order: 2,
    description: "Segment objects",
    inputs: [{ id: "input", label: "Input", artifact_id: "raw_heightmap_preview" }],
    outputs: [{ id: "mask", label: "Mask", artifact_id: "final_object_mask" }],
    artifacts: [{ id: "final_object_mask", label: "Final object mask", artifact_id: "final_object_mask", kind: "image", role: "final", renderer: "image" }],
    parameters: [],
    diagnostics: [],
    views: [],
    default_view: null,
  },
  {
    id: "segment.thresholding",
    label: "Thresholding",
    kind: "substage",
    parent_id: "segment",
    stage_id: "remove_belt_segment_objects",
    category: "thresholding",
    order: 3,
    description: "Threshold",
    inputs: [{ id: "input", label: "Input", artifact_id: "raw_heightmap_preview" }],
    outputs: [{ id: "mask", label: "Mask", artifact_id: "final_object_mask" }],
    artifacts: [{ id: "final_object_mask", label: "Final object mask", artifact_id: "final_object_mask", kind: "image", role: "final", renderer: "image" }],
    parameters: [{ id: "min_height_mm", label: "Min height", type: "number" }],
    diagnostics: [],
    views: [],
    default_view: "overlay",
  },
] as const;

test("graph model creates one node per processing unit and groups substages under roots", () => {
  const graph = buildProcessingUnitGraph({ units: units as never[] });
  assert.equal(graph.nodes.length, 3);
  assert.equal(graph.groups.length, 2);
  assert.deepEqual(graph.groups[1]?.nodeIds, ["segment", "segment.thresholding"]);
});

test("graph model emits stable order and artifact edges", () => {
  const graph = buildProcessingUnitGraph({ units: units as never[] });
  assert.ok(graph.edges.some((edge) => edge.id === "order:input:segment"));
  assert.ok(graph.edges.some((edge) => edge.id === "dependency:segment:segment.thresholding"));
  assert.ok(graph.edges.some((edge) => edge.kind === "artifact" && edge.source === "input" && edge.target === "segment"));
});

test("graph model maps dirty and comparison states to nodes", () => {
  const graph = buildProcessingUnitGraph({
    units: units as never[],
    recipe: { recipe_id: "r1", name: "Recipe", pipeline_id: "p", created_at: "", updated_at: "", version: 1, tags: [], parameters_by_unit: { "segment.thresholding": { min_height_mm: 2 } } },
    recipeDiffGroups: [{ unitId: "segment.thresholding", unitLabel: "Thresholding", changes: [{ paramId: "min_height_mm", paramLabel: "Min height", recipeValue: 2, currentValue: 3 }] }],
    comparison: {
      comparison_id: "cmp_1",
      pipeline_id: "p",
      left: { type: "run" },
      right: { type: "run" },
      summary: { parameter_changes: 1, artifact_changes: 0, metric_changes: 0, classification_changed: false, warnings_changed: false, key_affected_units: ["segment.thresholding"] },
      units: {
        "segment.thresholding": { label: "Thresholding", stage_id: "remove_belt_segment_objects", kind: "substage", order: 3, has_changes: true, parameter_diff: [], artifact_diff: [], metric_diff: [], diagnostic_diff: [] },
      },
    },
  });
  const node = graph.nodes.find((entry) => entry.id === "segment.thresholding");
  assert.equal(node?.dirtyState, "dirty");
  assert.equal(node?.comparisonState, "changed");
});

test("graph model prefers runtime trace status and duration, and marks best-effort entries as inferred", () => {
  const graph = buildProcessingUnitGraph({
    units: units as never[],
    detail: {
      result: {
        artifacts: [{ artifact_id: "raw_heightmap_preview" }, { artifact_id: "final_object_mask" }],
        processing_unit_trace: {
          trace_source: "mixed",
          trace_precision: "mixed",
          unit_results: {
            segment: {
              status: "completed",
              duration_ms: 27,
              trace_source: "runtime_unit_callbacks",
              trace_precision: "unit_level",
              metrics: { component_count: 2 },
            },
            input: {
              status: "inferred",
              trace_source: "best_effort_artifact_registry",
              trace_precision: "artifact_level",
            },
          },
        },
      },
    } as never,
  });
  const segmentNode = graph.nodes.find((entry) => entry.id === "segment");
  const inputNode = graph.nodes.find((entry) => entry.id === "input");
  assert.equal(segmentNode?.status, "completed");
  assert.equal(segmentNode?.durationMs, 27);
  assert.equal(segmentNode?.metricsSummary?.component_count, 2);
  assert.equal(inputNode?.status, "inferred");
});

test("graph model maps last-run dirty state and partial rerun plan badges to nodes", () => {
  const graph = buildProcessingUnitGraph({
    units: units as never[],
    detail: {
      result: {
        artifacts: [{ artifact_id: "raw_heightmap_preview" }, { artifact_id: "final_object_mask" }],
      },
    } as never,
    lastRunDiffGroups: [{ unitId: "segment.thresholding", unitLabel: "Thresholding", changes: [{ paramId: "min_height_mm", paramLabel: "Min height", recipeValue: 2, currentValue: 3 }] }],
    partialRerunPlan: {
      pipeline_id: "p",
      safe: true,
      mode: "rerun_from_unit",
      selected_unit_id: "segment.thresholding",
      execution_boundary_unit_id: "segment",
      units_to_reuse: ["input"],
      units_to_rerun: ["segment", "segment.thresholding"],
      units_invalidated: ["segment", "segment.thresholding"],
      required_missing_artifacts: [],
      warnings: [],
      blocking_reasons: [],
    },
  });
  const inputNode = graph.nodes.find((entry) => entry.id === "input");
  const stageNode = graph.nodes.find((entry) => entry.id === "segment");
  const substageNode = graph.nodes.find((entry) => entry.id === "segment.thresholding");
  assert.equal(inputNode?.planState, "reused");
  assert.equal(stageNode?.planState, "invalidated");
  assert.equal(stageNode?.selectionRole, "boundary");
  assert.equal(substageNode?.lastRunDirtyState, "dirty");
  assert.equal(substageNode?.selectionRole, "selected");
});

test("persisted trace derives reused-unit badges after reload with no live plan present", () => {
  const detail = {
    result: {
      execution_mode: "rerun_from_public_stage_boundary",
      recipe_snapshot: { boundary_stage_id: "segment" },
      partial_rerun_plan: { selected_unit_id: "segment.thresholding" },
      artifacts: [{ artifact_id: "raw_heightmap_preview" }, { artifact_id: "final_object_mask" }],
      processing_unit_trace: {
        trace_source: "mixed",
        unit_results: {
          input: { status: "reused", execution_role: "reused", source_run_id: "parent_run_1" },
          segment: { status: "completed" },
        },
      },
    },
  } as never;

  const persistedTrace = persistedTraceFromDetail(detail);
  assert.deepEqual(persistedTrace?.reusedUnitIds, ["input"]);
  assert.equal(persistedTrace?.boundaryStageId, "segment");
  assert.equal(persistedTrace?.selectedUnitId, "segment.thresholding");

  const graph = buildProcessingUnitGraph({
    units: units as never[],
    detail,
    persistedTrace,
  });
  const inputNode = graph.nodes.find((entry) => entry.id === "input");
  const segmentNode = graph.nodes.find((entry) => entry.id === "segment");
  const substageNode = graph.nodes.find((entry) => entry.id === "segment.thresholding");
  assert.equal(inputNode?.planState, "reused");
  assert.equal(segmentNode?.selectionRole, "boundary");
  assert.equal(substageNode?.selectionRole, "selected");
});

test("persisted reused/rerun badges update correctly when switching the active run", () => {
  const childDetail = {
    result: {
      run_id: "child_run_1",
      execution_mode: "rerun_from_public_stage_boundary",
      recipe_snapshot: { boundary_stage_id: "segment" },
      partial_rerun_plan: { selected_unit_id: "segment.thresholding" },
      artifacts: [{ artifact_id: "raw_heightmap_preview" }, { artifact_id: "final_object_mask" }],
      processing_unit_trace: {
        unit_results: {
          input: { status: "reused", execution_role: "reused" },
        },
      },
    },
  } as never;
  const parentDetail = {
    result: {
      run_id: "parent_run_1",
      execution_mode: "full_run",
      artifacts: [{ artifact_id: "raw_heightmap_preview" }, { artifact_id: "final_object_mask" }],
    },
  } as never;

  const childGraph = buildProcessingUnitGraph({ units: units as never[], detail: childDetail, persistedTrace: persistedTraceFromDetail(childDetail) });
  assert.equal(childGraph.nodes.find((entry) => entry.id === "input")?.planState, "reused");

  // Switching the active run back to the parent (full run) must not leak the child's
  // reused badges onto a graph built for a different, unrelated result.
  const parentGraph = buildProcessingUnitGraph({ units: units as never[], detail: parentDetail, persistedTrace: persistedTraceFromDetail(parentDetail) });
  assert.equal(parentGraph.nodes.find((entry) => entry.id === "input")?.planState, "none");

  // Switching forward again to the child must re-derive the reused badge from its own trace.
  const childGraphAgain = buildProcessingUnitGraph({ units: units as never[], detail: childDetail, persistedTrace: persistedTraceFromDetail(childDetail) });
  assert.equal(childGraphAgain.nodes.find((entry) => entry.id === "input")?.planState, "reused");
});

test("persisted trace is null for full runs and does not affect plan state", () => {
  const detail = {
    result: {
      execution_mode: "full_run",
      artifacts: [{ artifact_id: "raw_heightmap_preview" }],
    },
  } as never;
  assert.equal(persistedTraceFromDetail(detail), null);
  const graph = buildProcessingUnitGraph({ units: units as never[], detail, persistedTrace: persistedTraceFromDetail(detail) });
  const inputNode = graph.nodes.find((entry) => entry.id === "input");
  assert.equal(inputNode?.planState, "none");
});

test("graph model handles empty contracts safely", () => {
  const graph = buildProcessingUnitGraph({ units: [] });
  assert.equal(graph.nodes.length, 0);
  assert.ok(graph.warnings.length > 0);
});

test("graph selection maps substages back to stage and substage ids", () => {
  const selection = graphSelectionForUnit(units[2] as never);
  assert.equal(selection.stageContractId, "remove_belt_segment_objects");
  assert.equal(selection.unitId, "segment.thresholding");
  assert.equal(selection.substageId, "thresholding");
});

test("trace summary helpers format coverage and badge labels", () => {
  const summary = traceSummaryFromDetail({
    result: {
      processing_unit_trace: {
        trace_summary: {
          total_units: 56,
          runtime_traced_units: 18,
          inferred_units: 38,
          failed_units: 0,
          warning_units: 2,
          trace_coverage_percent: 32.1,
          coverage_by_stage: {
            geometry: {
              total_units: 5,
              runtime_traced_units: 4,
              inferred_units: 1,
              failed_units: 0,
              warning_units: 0,
              trace_coverage_percent: 80,
            },
          },
        },
      },
    },
  } as never);
  assert.equal(summary?.runtime_traced_units, 18);
  assert.equal(summary?.coverage_by_stage?.geometry?.runtime_traced_units, 4);
  assert.equal(formatTraceCoverageLine(summary), "Runtime trace coverage: 18 / 56 units, 32%");
  assert.equal(traceBadgeLabel("runtime_unit_callbacks"), "runtime traced");
  assert.equal(traceBadgeLabel("best_effort_artifact_registry"), "inferred");
  assert.equal(traceBadgeClass("runtime_unit_callbacks"), "trace-badge trace-badge-runtime");
  assert.equal(traceBadgeClass("best_effort_artifact_registry"), "trace-badge trace-badge-inferred");
});
