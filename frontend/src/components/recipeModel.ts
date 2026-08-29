import type { PipelineRecipe, ProcessingUnitDefinition } from "../api/client";

function stableValue(value: unknown): string {
  return JSON.stringify(value ?? null);
}

export function runtimeStageParamsKey(stageId: string): string {
  if (stageId === "measurement_diagnostics") return "known_object_25d";
  if (stageId === "classification") return "classify_25d";
  return stageId;
}

export function recipeStageParamsToFlat25d(stageParams: Record<string, unknown> | null | undefined): Record<string, unknown> {
  const next: Record<string, unknown> = {};
  const detect = (stageParams?.detect_belt_plane as Record<string, unknown> | undefined) ?? {};
  Object.assign(next, detect);
  const planeFitRoi = detect.plane_fit_roi as Record<string, unknown> | undefined;
  if (planeFitRoi) {
    const type = String(planeFitRoi.type ?? "rectangle");
    next.reference_surface_region_mode = planeFitRoi.enabled ? type : "none";
    next.plane_fit_roi_type = type;
    if (type === "polygon") {
      next.plane_fit_roi_points = Array.isArray(planeFitRoi.points) ? planeFitRoi.points : [];
    } else {
      next.plane_fit_roi_x = Number(planeFitRoi.x ?? 0);
      next.plane_fit_roi_y = Number(planeFitRoi.y ?? 0);
      next.plane_fit_roi_width = Number(planeFitRoi.width ?? 0);
      next.plane_fit_roi_height = Number(planeFitRoi.height ?? 0);
    }
  }

  const segmentation = (stageParams?.remove_belt_segment_objects as Record<string, unknown> | undefined) ?? {};
  for (const [key, value] of Object.entries(segmentation)) {
    if (key === "min_height_mm") next.object_min_height_mm = value;
    else if (key === "max_height_mm") next.object_max_height_mm = value;
    else next[key] = value;
  }

  const knownObject = (stageParams?.known_object_25d as Record<string, unknown> | undefined) ?? {};
  for (const [key, value] of Object.entries(knownObject)) {
    if (key === "enabled") next.known_object_enabled = value;
    else if (key === "apply_correction") next.known_object_apply_correction = value;
    else if (key === "target_selection") next.known_object_target_selection = value;
    else if (key === "manual_component_id") next.known_object_manual_component_id = value;
    else if (key === "tolerance_percent") next.known_tolerance_percent = value;
    else if (key === "apply_persisted_correction") next.known_object_apply_persisted_correction = value;
    else if (key === "persisted_scale_correction_x") next.known_object_persisted_scale_correction_x = value;
    else if (key === "persisted_scale_correction_y") next.known_object_persisted_scale_correction_y = value;
    else if (key === "persisted_scale_correction_z") next.known_object_persisted_scale_correction_z = value;
    else if (key === "object_label") next.known_object_label = value;
    else next[key] = value;
  }
  return next;
}

export function stageParamsToRecipeParametersByUnit(
  stageParams: Record<string, unknown> | null | undefined,
  units: ProcessingUnitDefinition[],
): Record<string, Record<string, unknown>> {
  const grouped: Record<string, Record<string, unknown>> = {};
  for (const unit of units) {
    const unitStageParams = stageParams?.[runtimeStageParamsKey(unit.stage_id)] as Record<string, unknown> | undefined;
    if (!unitStageParams) continue;
    const values: Record<string, unknown> = {};
    for (const param of unit.parameters ?? []) {
      if (unitStageParams[param.id] !== undefined) values[param.id] = unitStageParams[param.id];
    }
    if (Object.keys(values).length > 0) grouped[unit.id] = values;
  }
  return grouped;
}

export type RecipeDiffGroup = {
  unitId: string;
  unitLabel: string;
  changes: Array<{
    paramId: string;
    paramLabel: string;
    recipeValue: unknown;
    currentValue: unknown;
  }>;
};

export function buildParameterDiffGroups(
  baselineValues: Record<string, Record<string, unknown>> | null | undefined,
  currentValues: Record<string, Record<string, unknown>> | null | undefined,
  units: ProcessingUnitDefinition[],
): RecipeDiffGroup[] {
  const left = baselineValues ?? {};
  const right = currentValues ?? {};
  const unitById = new Map(units.map((unit) => [unit.id, unit]));
  const groups: RecipeDiffGroup[] = [];
  const unitIds = new Set([...Object.keys(left), ...Object.keys(right)]);
  for (const unitId of unitIds) {
    const unit = unitById.get(unitId);
    const baselineUnitValues = left[unitId] ?? {};
    const currentUnitValues = right[unitId] ?? {};
    const paramIds = new Set([...Object.keys(baselineUnitValues), ...Object.keys(currentUnitValues)]);
    const changes = Array.from(paramIds)
      .filter((paramId) => stableValue(baselineUnitValues[paramId]) !== stableValue(currentUnitValues[paramId]))
      .map((paramId) => ({
        paramId,
        paramLabel: unit?.parameters.find((param) => param.id === paramId)?.label ?? paramId,
        recipeValue: baselineUnitValues[paramId],
        currentValue: currentUnitValues[paramId],
      }));
    if (changes.length === 0) continue;
    groups.push({
      unitId,
      unitLabel: unit?.label ?? unitId,
      changes,
    });
  }
  return groups;
}

export function buildRecipeDiff(
  recipe: PipelineRecipe | null,
  currentStageParams: Record<string, unknown> | null | undefined,
  units: ProcessingUnitDefinition[],
): RecipeDiffGroup[] {
  if (!recipe) return [];
  const recipeValues = recipe.parameters_by_unit ?? stageParamsToRecipeParametersByUnit(recipe.stage_params, units);
  const currentValues = stageParamsToRecipeParametersByUnit(currentStageParams, units);
  return buildParameterDiffGroups(recipeValues, currentValues, units);
}

export function recipeDirty(
  recipe: PipelineRecipe | null,
  currentStageParams: Record<string, unknown> | null | undefined,
  units: ProcessingUnitDefinition[],
): boolean {
  return buildRecipeDiff(recipe, currentStageParams, units).length > 0;
}

export function recipeValuesForUnit(
  recipe: PipelineRecipe | null,
  unit: ProcessingUnitDefinition | null | undefined,
  units: ProcessingUnitDefinition[],
): Record<string, unknown> | null {
  if (!recipe || !unit) return null;
  const grouped = recipe.parameters_by_unit ?? stageParamsToRecipeParametersByUnit(recipe.stage_params, units);
  const unitValues = grouped[unit.id];
  if (!unitValues) return null;
  return { ...unitValues };
}
