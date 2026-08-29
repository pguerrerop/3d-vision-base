import test from "node:test";
import assert from "node:assert/strict";

import { buildRecipeDiff, recipeDirty, recipeStageParamsToFlat25d, recipeValuesForUnit, stageParamsToRecipeParametersByUnit } from "../src/components/recipeModel.ts";
import type { PipelineRecipe, ProcessingUnitDefinition } from "../src/api/client";

const units: ProcessingUnitDefinition[] = [
  {
    id: "detect_belt_plane.candidate_support_refinement",
    label: "Support refinement",
    kind: "substage",
    parent_id: "detect_belt_plane",
    stage_id: "detect_belt_plane",
    category: "reference",
    order: 1,
    description: "refine",
    inputs: [],
    outputs: [],
    artifacts: [],
    parameters: [
      { id: "blob_cluster_refine_mad_k", label: "MAD multiplier", type: "number" },
      { id: "blob_cluster_refine_keep_border_support", label: "Keep border support", type: "boolean" },
    ],
    diagnostics: [],
    views: [],
    default_view: null,
    help_markdown: null,
  },
  {
    id: "remove_belt_segment_objects",
    label: "Segmentation",
    kind: "stage",
    parent_id: null,
    stage_id: "remove_belt_segment_objects",
    category: "segmentation",
    order: 2,
    description: "segment",
    inputs: [],
    outputs: [],
    artifacts: [],
    parameters: [{ id: "min_height_mm", label: "Min height", type: "number" }],
    diagnostics: [],
    views: [],
    default_view: null,
    help_markdown: null,
  },
];

test("recipeStageParamsToFlat25d restores flat tune values", () => {
  const flat = recipeStageParamsToFlat25d({
    detect_belt_plane: {
      background_detection_strategy: "low_gradient_depth_plateaus",
      plane_fit_roi: { enabled: true, type: "rectangle", x: 10, y: 20, width: 30, height: 40 },
    },
    remove_belt_segment_objects: { min_height_mm: 3.5 },
    known_object_25d: { enabled: true, target_selection: "largest_component" },
  });
  assert.equal(flat.background_detection_strategy, "low_gradient_depth_plateaus");
  assert.equal(flat.reference_surface_region_mode, "rectangle");
  assert.equal(flat.plane_fit_roi_x, 10);
  assert.equal(flat.object_min_height_mm, 3.5);
  assert.equal(flat.known_object_enabled, true);
});

test("recipe diff is grouped by processing unit labels", () => {
  const recipe: PipelineRecipe = {
    recipe_id: "recipe_1",
    name: "Recipe 1",
    pipeline_id: "mining_steel_ball_classification_25d",
    created_at: "2026-07-01T00:00:00Z",
    updated_at: "2026-07-01T00:00:00Z",
    version: 1,
    tags: [],
    parameters_by_unit: {
      "detect_belt_plane.candidate_support_refinement": {
        blob_cluster_refine_mad_k: 2.5,
        blob_cluster_refine_keep_border_support: true,
      },
      "remove_belt_segment_objects": {
        min_height_mm: 2,
      },
    },
  };

  const currentStageParams = {
    detect_belt_plane: {
      blob_cluster_refine_mad_k: 3,
      blob_cluster_refine_keep_border_support: false,
    },
    remove_belt_segment_objects: {
      min_height_mm: 2,
    },
  };

  const diff = buildRecipeDiff(recipe, currentStageParams, units);
  assert.equal(diff.length, 1);
  assert.equal(diff[0].unitLabel, "Support refinement");
  assert.deepEqual(diff[0].changes.map((item) => item.paramLabel), ["MAD multiplier", "Keep border support"]);
  assert.equal(recipeDirty(recipe, currentStageParams, units), true);
});

test("stageParamsToRecipeParametersByUnit groups params by declared unit", () => {
  const grouped = stageParamsToRecipeParametersByUnit(
    {
      detect_belt_plane: { blob_cluster_refine_mad_k: 2.5, blob_cluster_refine_keep_border_support: true },
      remove_belt_segment_objects: { min_height_mm: 2 },
    },
    units,
  );
  assert.equal(grouped["detect_belt_plane.candidate_support_refinement"].blob_cluster_refine_mad_k, 2.5);
  assert.equal(grouped["remove_belt_segment_objects"].min_height_mm, 2);
});

test("recipeValuesForUnit returns grouped recipe snapshot for active unit", () => {
  const recipe: PipelineRecipe = {
    recipe_id: "recipe_a",
    name: "A",
    pipeline_id: "mining_steel_ball_classification_25d",
    version: 1,
    parameters_by_unit: {
      "remove_belt_segment_objects": { min_height_mm: 4 },
    },
  };
  const values = recipeValuesForUnit(recipe, units[1], units);
  assert.equal(values?.min_height_mm, 4);
});
