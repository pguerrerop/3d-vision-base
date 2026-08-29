import type { PipelineStageInfo, ProcessingUnitDefinition } from "../api/client";
import type { ReactNode } from "react";

import InfoHint from "./InfoHint";
import {
  supportSelectionMethodLabel,
  type DetectReferenceRunState,
  type ReferenceMethodPreset,
} from "./referenceArtifactRelevance";
import type { ResolvedStageSubstage } from "./stageSubstagePlan";
import { detectReferenceGroupHelpKey, detectReferenceParamHelpKey, resolveStudioHelpEntry } from "./studioHelp";
import type { StudioHelpEntry } from "./studioHelp";

type FieldMeta = {
  type?: string;
  label?: string;
  default?: unknown;
  enum?: Array<string | number>;
  minimum?: number;
  maximum?: number;
  step?: number;
  help?: string;
  unit?: string;
  nullable?: boolean;
  group?: string;
  advanced?: boolean;
  visible_when?: Record<string, unknown>;
  affects?: string[];
  tuning_hint?: string;
};

type GroupMeta = {
  id: string;
  label?: string;
};

type Props = {
  stage: PipelineStageInfo | null;
  availableUnits: ProcessingUnitDefinition[];
  values: Record<string, unknown> | null;
  persistedValues: Record<string, unknown> | null;
  recipeValues?: Record<string, unknown> | null;
  lastRunValues: Record<string, unknown> | null;
  selectedSubstage: ResolvedStageSubstage | null;
  selectedUnit: ProcessingUnitDefinition | null;
  selectedViewLabel: string;
  selectedArtifactId: string | null;
  runState: DetectReferenceRunState | null;
  strategyLabel: string;
  dirty: boolean;
  formDiffersFromRun: boolean;
  loading?: boolean;
  hasTuneChanges: boolean;
  activePreset: ReferenceMethodPreset | null;
  persistedPreset: ReferenceMethodPreset | null;
  onChange: (key: string, value: unknown) => void;
  onResetFieldToRecipe?: (key: string) => void;
  onResetUnitToRecipe?: () => void;
  onApply: () => void;
  onApplyRun: () => void;
  onDiscard: () => void;
  onStartVisualRoiEdit: () => void;
  onUseFullFrame: () => void;
  onApplyPreset: (presetId: string, rerun: boolean) => void;
  onResetPresetOverrides: () => void;
  onCopyPresetJson: () => void;
};

const HISTOGRAM_SPLIT_PARAM_KEYS = new Set([
  "blob_split_gap_mm",
  "blob_split_mode",
  "blob_split_hist_bins",
  "blob_split_min_band_fraction",
  "blob_split_min_band_pixels",
  "blob_split_merge_gap_mm",
  "blob_split_refine_morphology",
  "blob_split_open_kernel",
  "blob_split_close_kernel",
]);

const HEIGHT_BORDER_PARAM_KEYS = new Set([
  "blob_split_height_border_enabled",
  "blob_split_height_border_mode",
  "blob_split_height_border_smoothing_kernel",
  "blob_split_height_border_threshold_mode",
  "blob_split_height_border_percentile",
  "blob_split_height_border_min_delta_mm",
  "blob_split_height_border_dilate_kernel",
  "blob_split_height_border_close_kernel",
  "blob_split_height_border_min_length_px",
  "blob_split_min_fragment_area_px",
]);

const WEAK_BOUNDARY_MERGE_PARAM_KEYS = new Set([
  "blob_split_merge_weak_boundaries",
  "blob_split_merge_max_median_z_gap_mm",
  "blob_split_merge_max_boundary_strength_mm",
  "blob_split_merge_background_like_only",
]);

function valuesEqual(a: unknown, b: unknown): boolean {
  if (Array.isArray(a) || Array.isArray(b)) return JSON.stringify(a ?? null) === JSON.stringify(b ?? null);
  if (a && typeof a === "object") return JSON.stringify(a) === JSON.stringify(b ?? null);
  if (b && typeof b === "object") return JSON.stringify(a ?? null) === JSON.stringify(b);
  return a === b;
}

function formatEffectiveValue(value: unknown, meta: FieldMeta): string | null {
  if (value == null || value === "") return null;
  if (typeof value === "number") return meta.unit ? `${value} ${meta.unit}` : String(value);
  if (typeof value === "boolean") return value ? "On" : "Off";
  return String(value).replace(/_/g, " ");
}

function effectiveFieldValue(
  key: string,
  meta: FieldMeta,
  values: Record<string, unknown>,
  persistedValues: Record<string, unknown> | null,
): unknown {
  const current = values[key];
  if (current !== undefined) return current;
  const persisted = persistedValues?.[key];
  if (persisted !== undefined) return persisted;
  return meta.default;
}

function dependencyReason(dep: string, expected: unknown): string {
  return `${dep.replace(/_/g, " ")} must be ${String(expected).replace(/_/g, " ")}.`;
}

function resolveFieldApplicability(
  key: string,
  meta: FieldMeta,
  values: Record<string, unknown>,
): { applicable: boolean; reason: string | null } {
  if (meta.type === "roi") return { applicable: true, reason: null };
  if (meta.visible_when) {
    for (const [dep, expected] of Object.entries(meta.visible_when)) {
      if (values?.[dep] !== expected) {
        return { applicable: false, reason: dependencyReason(dep, expected) };
      }
    }
  }
  const splitEnabled = Boolean(values.blob_split_by_height_enabled ?? true);
  const splitMethod = String(values.blob_split_method ?? "height_borders");
  const usesHistogramSplit = splitMethod === "histogram_gap" || splitMethod === "height_borders_then_histogram";
  const usesHeightBorders = splitMethod === "height_borders" || splitMethod === "height_borders_then_histogram";
  const heightBorderEnabled = Boolean(values.blob_split_height_border_enabled ?? true);
  const gradientThresholdMode = String(values.gradient_threshold_mode ?? "percentile");

  if (key === "gradient_threshold_value" && gradientThresholdMode !== "fixed") {
    return {
      applicable: false,
      reason: gradientThresholdMode === "otsu"
        ? "Computed automatically from Otsu thresholding."
        : "Not used while threshold mode is percentile.",
    };
  }
  if (HISTOGRAM_SPLIT_PARAM_KEYS.has(key)) {
    if (!splitEnabled) return { applicable: false, reason: "Not used while split mixed-height blobs is disabled." };
    if (!usesHistogramSplit) return { applicable: false, reason: `Not used by split method ${splitMethod.replace(/_/g, " ")}.` };
  }
  if (HEIGHT_BORDER_PARAM_KEYS.has(key)) {
    if (!splitEnabled) return { applicable: false, reason: "Not used while split mixed-height blobs is disabled." };
    if (!usesHeightBorders) return { applicable: false, reason: `Not used by split method ${splitMethod.replace(/_/g, " ")}.` };
    if (key !== "blob_split_height_border_enabled" && !heightBorderEnabled) {
      return { applicable: false, reason: "Enable height-border split to use this control." };
    }
  }
  if (WEAK_BOUNDARY_MERGE_PARAM_KEYS.has(key)) {
    if (!splitEnabled) return { applicable: false, reason: "Not used while split mixed-height blobs is disabled." };
    if (!usesHeightBorders) return { applicable: false, reason: "Weak-boundary merge only applies to height-border splitting." };
    if (key !== "blob_split_merge_weak_boundaries" && !Boolean(values.blob_split_merge_weak_boundaries ?? true)) {
      return { applicable: false, reason: "Enable Merge weak boundaries to use this control." };
    }
  }
  return { applicable: true, reason: null };
}

function resolveComputedDetail(
  key: string,
  effectiveValue: unknown,
  runState: DetectReferenceRunState | null,
): string | null {
  if ((key === "reference_surface_model" || key === "background_detection_strategy") && (effectiveValue === "auto" || effectiveValue === "automatic")) {
    if (key === "reference_surface_model" && runState?.resolvedReferenceSurfaceModel && runState.resolvedReferenceSurfaceModel !== "unknown") {
      return `Selected run resolved to ${runState.resolvedReferenceSurfaceModel.replace(/_/g, " ")}.`;
    }
    if (key === "background_detection_strategy" && runState?.strategy && runState.strategy !== "unknown") {
      return `Selected run resolved to ${runState.strategy.replace(/_/g, " ")}.`;
    }
  }
  if (effectiveValue === "otsu") return "Computed automatically at rerun time.";
  return null;
}

function buildFieldHelpEntry(args: {
  key: string;
  meta: FieldMeta;
  statusLabel: string;
  detail: string | null;
  recipeValue?: unknown;
  lastRunValue: unknown;
  currentValue: unknown;
}): StudioHelpEntry | null {
  const base = resolveStudioHelpEntry(detectReferenceParamHelpKey(args.key));
  const recipeSummary = args.recipeValue !== undefined
    ? `Recipe: ${formatEffectiveValue(args.recipeValue, args.meta) ?? "unset"}.`
    : "Recipe: unavailable.";
  const recipeState = args.recipeValue !== undefined && valuesEqual(args.recipeValue, args.currentValue)
    ? "Matches recipe."
    : "Edited value differs from recipe.";
  const lastRunSummary = args.lastRunValue !== undefined
    ? `Last run: ${formatEffectiveValue(args.lastRunValue, args.meta) ?? "unset"}.`
    : "Last run: unavailable.";
  const dirtyState = valuesEqual(args.lastRunValue, args.currentValue)
    ? "Matches last run."
    : "Edited value differs from last run.";
  const defaultSummary = `Default: ${formatEffectiveValue(args.meta.default, args.meta) ?? "unset"}. ${valuesEqual(args.meta.default, args.currentValue) ? "Matches default." : "Edited value differs from default."}`;
  const syncBlock = [
    `${args.statusLabel}${args.detail ? ` · ${args.detail}` : ""}`,
    `${recipeSummary} ${recipeState}`,
    `${lastRunSummary} ${dirtyState}`,
    defaultSummary,
  ].join("\n");
  const summary = base?.summary ?? args.meta.help ?? "Parameter tuning and sync state.";
  const detailsParts = [base?.details, args.meta.tuning_hint, syncBlock].filter(Boolean);
  return {
    ...(base ?? {
      title: args.meta.label ?? args.key,
      summary,
      details: "",
    }),
    title: base?.title ?? args.meta.label ?? args.key,
    summary,
    details: detailsParts.join("\n\n"),
    affects: base?.affects?.length ? base.affects : args.meta.affects,
  };
}

function renderRoiField(
  meta: FieldMeta,
  values: Record<string, unknown>,
  recipeValues: Record<string, unknown> | null | undefined,
  lastRunValues: Record<string, unknown> | null,
  onChange: (key: string, value: unknown) => void,
  onStartVisualRoiEdit: () => void,
  onUseFullFrame: () => void,
  onResetFieldToRecipe?: (key: string) => void,
) {
  const mode = String(values.reference_surface_region_mode ?? "none");
  const polygonPoints = Array.isArray(values.plane_fit_roi_points) ? values.plane_fit_roi_points : [];
  const lastRunMode = String(lastRunValues?.reference_surface_region_mode ?? "unavailable");
  const roiDirty = !valuesEqual(
    {
      mode: values.reference_surface_region_mode ?? "none",
      x: values.plane_fit_roi_x ?? 0,
      y: values.plane_fit_roi_y ?? 0,
      width: values.plane_fit_roi_width ?? 0,
      height: values.plane_fit_roi_height ?? 0,
      points: values.plane_fit_roi_points ?? [],
    },
    {
      mode: lastRunValues?.reference_surface_region_mode ?? "none",
      x: lastRunValues?.plane_fit_roi_x ?? 0,
      y: lastRunValues?.plane_fit_roi_y ?? 0,
      width: lastRunValues?.plane_fit_roi_width ?? 0,
      height: lastRunValues?.plane_fit_roi_height ?? 0,
      points: lastRunValues?.plane_fit_roi_points ?? [],
    },
  );
  return (
    <section className="control-panel-section">
      <span className="info-inline-label">
        <span>{meta.label ?? "Reference ROI"}</span>
        <InfoHint content={resolveStudioHelpEntry(detectReferenceParamHelpKey("reference_surface_region"))} label="Open parameter help" />
      </span>
      <div className="control-panel-form-grid">
        <label>
          Mode
          <select
            value={mode}
            onChange={(event) => {
              const nextMode = event.target.value;
              onChange("reference_surface_region_mode", nextMode);
              onChange("plane_fit_roi_type", nextMode === "full_height_x_band" ? "full_height_x_band" : nextMode === "none" ? "rectangle" : nextMode);
            }}
          >
            <option value="none">Full frame</option>
            <option value="full_height_x_band">Vertical band</option>
            <option value="rectangle">Rectangle</option>
            <option value="polygon">Polygon</option>
          </select>
        </label>
        <label>
          X
          <input type="number" min={0} step={1} value={Number(values.plane_fit_roi_x ?? 0)} onChange={(event) => onChange("plane_fit_roi_x", Number(event.target.value))} />
        </label>
        <label>
          Width
          <input type="number" min={1} step={1} value={Number(values.plane_fit_roi_width ?? 0)} onChange={(event) => onChange("plane_fit_roi_width", Number(event.target.value))} />
        </label>
        {mode !== "full_height_x_band" ? (
          <>
            <label>
              Y
              <input type="number" min={0} step={1} value={Number(values.plane_fit_roi_y ?? 0)} onChange={(event) => onChange("plane_fit_roi_y", Number(event.target.value))} />
            </label>
            <label>
              Height
              <input type="number" min={1} step={1} value={Number(values.plane_fit_roi_height ?? 0)} onChange={(event) => onChange("plane_fit_roi_height", Number(event.target.value))} />
            </label>
          </>
        ) : null}
      </div>
      {mode === "polygon" ? <small className="section-note">Polygon points: {polygonPoints.length}</small> : null}
      <small className="section-note">Last run mode: {lastRunMode.replace(/_/g, " ")}.</small>
      <small className="section-note">{roiDirty ? "Edited ROI differs from last run." : "ROI matches last run."}</small>
      <small className="section-note">Recipe mode: {String(recipeValues?.reference_surface_region_mode ?? "unavailable").replace(/_/g, " ")}.</small>
      {meta.affects?.length ? <small className="section-note">Affects: {meta.affects.join(" • ")}</small> : null}
      {meta.tuning_hint ? <small className="section-note">{meta.tuning_hint}</small> : null}
      <div className="control-panel-button-row">
        {onResetFieldToRecipe ? <button type="button" onClick={() => onResetFieldToRecipe("reference_surface_region")}>Reset to recipe</button> : null}
        <button type="button" onClick={onStartVisualRoiEdit}>Edit visually</button>
        <button type="button" onClick={onUseFullFrame}>Reset to full frame</button>
      </div>
    </section>
  );
}

function renderField(
  key: string,
  meta: FieldMeta,
  values: Record<string, unknown>,
  persistedValues: Record<string, unknown> | null,
  recipeValues: Record<string, unknown> | null | undefined,
  lastRunValues: Record<string, unknown> | null,
  runState: DetectReferenceRunState | null,
  onChange: (key: string, value: unknown) => void,
  onResetFieldToRecipe?: (key: string) => void,
  onStartVisualRoiEdit?: () => void,
  onUseFullFrame?: () => void,
) {
  if (meta.type === "roi") {
    return renderRoiField(
      meta,
      values,
      recipeValues,
      lastRunValues,
      onChange,
      onStartVisualRoiEdit ?? (() => undefined),
      onUseFullFrame ?? (() => undefined),
      onResetFieldToRecipe,
    );
  }
  const applicability = resolveFieldApplicability(key, meta, values);
  const value = effectiveFieldValue(key, meta, values, persistedValues);
  const detail = applicability.applicable ? resolveComputedDetail(key, value, runState) : applicability.reason;
  const statusLabel = !applicability.applicable
    ? "Not applicable"
    : meta.nullable && (value == null || value === "")
      ? "Unset / auto"
      : meta.default !== undefined && valuesEqual(value, meta.default)
        ? "Default"
        : "Recipe override";
  const helpEntry = buildFieldHelpEntry({
    key,
    meta,
    statusLabel,
    detail,
    recipeValue: recipeValues?.[key],
    lastRunValue: lastRunValues?.[key],
    currentValue: value,
  });
  const rawNumberValue = value == null || value === "" ? "" : Number(value);

  if (Array.isArray(meta.enum)) {
    return (
      <label key={key}>
        <span className="info-inline-label">
          <span>{meta.label ?? key}</span>
          <InfoHint content={helpEntry} label="Open parameter help" />
        </span>
        <select disabled={!applicability.applicable} value={String(value ?? meta.default ?? meta.enum[0] ?? "")} onChange={(event) => onChange(key, event.target.value)}>
          {meta.enum.map((item) => <option key={String(item)} value={String(item)}>{String(item)}</option>)}
        </select>
        {onResetFieldToRecipe && recipeValues?.[key] !== undefined && !valuesEqual(recipeValues[key], value) ? <button type="button" onClick={() => onResetFieldToRecipe(key)}>Reset to recipe</button> : null}
      </label>
    );
  }
  if (meta.type === "boolean") {
    return (
      <label key={key} className="control-panel-checkbox">
        <input disabled={!applicability.applicable} checked={Boolean(value)} type="checkbox" onChange={(event) => onChange(key, event.target.checked)} />
        <div className="control-panel-checkbox-copy">
          <span className="info-inline-label">
            <span>{meta.label ?? key}</span>
            <InfoHint content={helpEntry} label="Open parameter help" />
          </span>
          {onResetFieldToRecipe && recipeValues?.[key] !== undefined && !valuesEqual(recipeValues[key], value) ? <button type="button" onClick={() => onResetFieldToRecipe(key)}>Reset to recipe</button> : null}
        </div>
      </label>
    );
  }
  return (
    <label key={key}>
      <span className="info-inline-label">
        <span>{meta.label ?? key}</span>
        <InfoHint content={helpEntry} label="Open parameter help" />
      </span>
      <input
        type="number"
        min={meta.minimum}
        max={meta.maximum}
        step={meta.step}
        disabled={!applicability.applicable}
        placeholder={meta.nullable ? "unset / auto" : undefined}
        value={rawNumberValue === "" ? "" : (Number.isFinite(rawNumberValue) ? rawNumberValue : Number(meta.default ?? 0))}
        onChange={(event) => {
          if (event.target.value === "") {
            onChange(key, meta.nullable ? null : meta.default ?? 0);
            return;
          }
          onChange(key, Number(event.target.value));
        }}
      />
      {meta.unit ? <small>{meta.unit}</small> : null}
      {onResetFieldToRecipe && recipeValues?.[key] !== undefined && !valuesEqual(recipeValues[key], value) ? <button type="button" onClick={() => onResetFieldToRecipe(key)}>Reset to recipe</button> : null}
    </label>
  );
}

function FieldList(props: {
  title: string;
  keys: string[];
  values: Record<string, unknown>;
  persistedValues: Record<string, unknown> | null;
  recipeValues?: Record<string, unknown> | null;
  lastRunValues: Record<string, unknown> | null;
  fields: Record<string, FieldMeta>;
  runState: DetectReferenceRunState | null;
  onChange: (key: string, value: unknown) => void;
  onResetFieldToRecipe?: (key: string) => void;
  children?: ReactNode;
  onStartVisualRoiEdit?: () => void;
  onUseFullFrame?: () => void;
}) {
  if (props.keys.length === 0 && !props.children) return null;
  return (
    <details className="control-panel-details">
      <summary>{props.title}</summary>
      <div className="control-panel-form-grid">
        {props.keys.map((key) => renderField(key, props.fields[key] ?? {}, props.values, props.persistedValues, props.recipeValues, props.lastRunValues, props.runState, props.onChange, props.onResetFieldToRecipe, props.onStartVisualRoiEdit, props.onUseFullFrame))}
      </div>
      {props.children}
    </details>
  );
}

export default function DetectReferenceTunePanel(props: Props) {
  const values = props.values;
  const schema = (props.stage?.parameter_schema ?? {}) as { fields?: Record<string, FieldMeta>; groups?: GroupMeta[] };
  const fields = schema.fields ?? {};
  const groupLabels = new Map((schema.groups ?? []).map((group) => [group.id, group.label ?? group.id]));

  if (!props.stage || !values) return null;

  const selectedUnit = props.selectedUnit;
  const unitFields: Record<string, FieldMeta> = Object.fromEntries(
    (selectedUnit?.parameters ?? []).map((param) => [
      param.id,
      {
        type: param.type,
        label: param.label,
        default: param.default,
        enum: param.options ?? undefined,
        minimum: param.min ?? undefined,
        maximum: param.max ?? undefined,
        step: param.step ?? undefined,
        help: param.description ?? undefined,
        unit: param.unit ?? undefined,
        nullable: Boolean(param.nullable),
        group: param.group ?? undefined,
        advanced: Boolean(param.advanced),
        visible_when: param.active_when ?? undefined,
        affects: param.affects ?? [],
        tuning_hint: param.tuning_hint ?? undefined,
      } satisfies FieldMeta,
    ]),
  );
  const unitGroupIds = Array.from(new Set((selectedUnit?.parameters ?? []).map((param) => param.group || (param.advanced ? "Advanced" : "Parameters"))));
  const visibleUnitKeys = Object.keys(unitFields).filter((key) => {
    const meta = unitFields[key] ?? {};
    return meta.type === "roi" || resolveFieldApplicability(key, meta, values).applicable;
  });
  const groupedPrimaryKeys = new Set<string>();
  const substagePrimaryGroups = (selectedUnit?.parameter_groups ?? []).map((group) => {
    const groupKeys = (group.param_keys ?? []).filter((key) => visibleUnitKeys.includes(key));
    for (const key of groupKeys) groupedPrimaryKeys.add(key);
    return { group, keys: groupKeys };
  });
  const basicUnitKeys = visibleUnitKeys.filter((key) => !unitFields[key]?.advanced && !groupedPrimaryKeys.has(key));
  const advancedUnitKeys = visibleUnitKeys.filter((key) => unitFields[key]?.advanced && !groupedPrimaryKeys.has(key));
  const hasPrimaryGroupContent = substagePrimaryGroups.some(({ group, keys }) => keys.length > 0 || group.description || group.affects?.length);
  const upstreamLabels = (selectedUnit?.controlled_by ?? []).map((unitId) => props.availableUnits.find((candidate) => candidate.id === unitId)?.label ?? unitId);

  return (
    <>
      <section className="control-panel-section">
        <span>Tune context</span>
        <small>Process: Detect reference surface</small>
        <small>Subprocess: {props.selectedSubstage?.label ?? "Strategy path"}</small>
        <small>Result: {props.selectedViewLabel}</small>
        <small>Artifact: {props.selectedArtifactId ?? "none selected"}</small>
        <small>Strategy: {props.strategyLabel}</small>
        <small>Contract unit: {selectedUnit?.id ?? "fallback / legacy"}</small>
      </section>
      {props.selectedSubstage && (
        <section className="control-panel-section">
          <span>Current substage</span>
          {props.selectedSubstage.description ? <small>{props.selectedSubstage.description}</small> : null}
          <small>Views: {props.selectedSubstage.views.map((entry) => entry.view.label).join(" • ") || "none"}</small>
          <small>Supporting artifacts: {props.selectedSubstage.supportingArtifacts.map((entry) => entry.artifactId).join(" • ") || "none"}</small>
        </section>
      )}
      <section className="control-panel-section">
        <span>Tuning state</span>
        <strong>{props.hasTuneChanges ? "Unsaved changes" : "In sync with selected run"}</strong>
        <small>{props.formDiffersFromRun ? "Form differs from selected run. Apply + rerun to update artifacts." : "Form matches the selected run for strategy-critical controls."}</small>
        <small>{props.dirty ? "Recipe form is dirty." : "Recipe form is clean."}</small>
      </section>
      <section className="control-panel-section">
        <span className="info-inline-label">
          <span>Strategy</span>
          <InfoHint content={resolveStudioHelpEntry(detectReferenceParamHelpKey("background_detection_strategy"))} label="Open parameter help" />
        </span>
        <div className="control-panel-form-grid">
          <label>
            <span className="info-inline-label">
              <span>Support selection</span>
              <InfoHint content={resolveStudioHelpEntry(detectReferenceParamHelpKey("background_detection_strategy"))} label="Open parameter help" />
            </span>
            <select value={String(values.background_detection_strategy ?? "low_gradient_surface")} onChange={(event) => props.onChange("background_detection_strategy", event.target.value)}>
              <option value="low_gradient_depth_plateaus">Low-gradient depth plateaus</option>
              <option value="low_gradient_surface">Low-gradient surface</option>
              <option value="low_gradient_bg_and_stripes">Low-gradient background + stripes</option>
              <option value="low_gradient_blob_height_clusters">Blob height clusters</option>
              <option value="depth_percentile_plane">Percentile / legacy</option>
            </select>
            <small>{supportSelectionMethodLabel(props.runState?.supportSelectionMethod ?? "unknown")}</small>
          </label>
        </div>
      </section>
      {selectedUnit?.help_markdown || selectedUnit?.downstream_effects?.length ? (
        <section className="control-panel-section">
          <span>{selectedUnit?.label}</span>
          {selectedUnit?.help_markdown ? <small>{selectedUnit.help_markdown}</small> : null}
          {selectedUnit?.downstream_effects?.length ? <small>Downstream effects: {selectedUnit.downstream_effects.join(" • ")}</small> : null}
        </section>
      ) : null}
      {selectedUnit == null || visibleUnitKeys.length === 0 ? (
        <section className="control-panel-section">
          <span>{selectedUnit?.label ?? "Reference subprocess"}</span>
          <small>
            No tunable parameters are registered for this substage.
            {upstreamLabels.length ? ` It is controlled by upstream units: ${upstreamLabels.join(", ")}.` : ""}
          </small>
        </section>
      ) : (
        <>
          {substagePrimaryGroups.map(({ group, keys }) => {
            if (keys.length === 0 && !(group.description || group.affects?.length)) return null;
            const groupHelp = resolveStudioHelpEntry(detectReferenceGroupHelpKey(group.id) ?? detectReferenceGroupHelpKey(group.label));
            return (
              <section key={group.id} className="control-panel-section">
                <span className="info-inline-label">
                  <span>{group.label}</span>
                  {groupHelp ? <InfoHint content={groupHelp} label="Open group help" /> : null}
                </span>
                {!groupHelp && group.description ? <small>{group.description}</small> : null}
                {!groupHelp && group.affects?.length ? <small>Affects: {group.affects.join(" • ")}</small> : null}
                {keys.length ? (
                  <div className="control-panel-form-grid">
                    {keys.map((key) => renderField(key, unitFields[key] ?? {}, values, props.persistedValues, props.recipeValues, props.lastRunValues, props.runState, props.onChange, props.onResetFieldToRecipe, props.onStartVisualRoiEdit, props.onUseFullFrame))}
                  </div>
                ) : null}
              </section>
            );
          })}
          {(() => {
            const fallbackGroups = unitGroupIds.map((groupId) => {
              const groupKeys = basicUnitKeys.filter((key) => (unitFields[key]?.group ?? "Parameters") === groupId);
              if (groupKeys.length === 0) return null;
              const groupLabel = groupLabels.get(groupId) ?? groupId;
              const groupHelp = resolveStudioHelpEntry(detectReferenceGroupHelpKey(groupId) ?? detectReferenceGroupHelpKey(groupLabel));
              return (
                <section key={groupId} className="control-panel-section">
                  <span className="info-inline-label">
                    <span>{groupLabel}</span>
                    {groupHelp ? <InfoHint content={groupHelp} label="Open group help" /> : null}
                  </span>
                  <div className="control-panel-form-grid">
                    {groupKeys.map((key) => renderField(key, unitFields[key] ?? {}, values, props.persistedValues, props.recipeValues, props.lastRunValues, props.runState, props.onChange, props.onResetFieldToRecipe, props.onStartVisualRoiEdit, props.onUseFullFrame))}
                  </div>
                </section>
              );
            });
            if (!fallbackGroups.some(Boolean)) return null;
            if (!hasPrimaryGroupContent) return fallbackGroups;
            return (
              <details className="control-panel-details">
                <summary>Other parameters for this unit</summary>
                {fallbackGroups}
              </details>
            );
          })()}
        </>
      )}
      {selectedUnit?.tuning_hints?.length ? (
        <section className="control-panel-section">
          <span>What to tune</span>
          {selectedUnit.tuning_hints.map((hint) => (
            <small key={hint.condition}>
              {hint.condition}: {hint.actions.join(" • ")}
            </small>
          ))}
        </section>
      ) : null}
      {selectedUnit?.artifacts?.length ? (
        <section className="control-panel-section">
          <span>Declared artifacts</span>
          <small>{selectedUnit.artifacts.map((artifact) => artifact.artifact_id).join(" • ")}</small>
        </section>
      ) : null}
      {props.recipeValues && selectedUnit && Object.keys(props.recipeValues).length > 0 && props.onResetUnitToRecipe ? (
        <div className="control-panel-button-row">
          <button type="button" disabled={props.loading} onClick={props.onResetUnitToRecipe}>Reset unit to recipe</button>
        </div>
      ) : null}
      <FieldList title="Other reference-stage controls" keys={[]} values={values} persistedValues={props.persistedValues} recipeValues={props.recipeValues} lastRunValues={props.lastRunValues} fields={fields} runState={props.runState} onChange={props.onChange} onResetFieldToRecipe={props.onResetFieldToRecipe}>
        <div className="control-panel-button-row">
          <button type="button" disabled={!props.activePreset || props.loading} onClick={() => props.activePreset && props.onApplyPreset(props.activePreset.id, false)}>Apply preset</button>
          <button type="button" disabled={!props.activePreset || props.loading} onClick={() => props.activePreset && props.onApplyPreset(props.activePreset.id, true)}>Apply preset + rerun</button>
        </div>
        <div className="control-panel-button-row">
          <button type="button" disabled={!props.persistedPreset || props.loading} onClick={props.onResetPresetOverrides}>Reset preset overrides</button>
          <button type="button" disabled={props.loading} onClick={props.onCopyPresetJson}>Copy preset JSON</button>
        </div>
        <small>Preset: {props.activePreset?.label ?? "Custom / modified"}</small>
      </FieldList>
      <FieldList title="Advanced" keys={advancedUnitKeys} values={values} persistedValues={props.persistedValues} recipeValues={props.recipeValues} lastRunValues={props.lastRunValues} fields={unitFields} runState={props.runState} onChange={props.onChange} onResetFieldToRecipe={props.onResetFieldToRecipe} onStartVisualRoiEdit={props.onStartVisualRoiEdit} onUseFullFrame={props.onUseFullFrame} />
      <details className="control-panel-details">
        <summary>Full stage parameters</summary>
        {Array.from(new Set(Object.keys(fields).map((key) => fields[key]?.group ?? "Other"))).map((groupId) => {
          const groupKeys = Object.keys(fields).filter((key) => (fields[key]?.group ?? "Other") === groupId);
          if (groupKeys.length === 0) return null;
          return (
            <div key={groupId}>
              <div className="stage-parameter-panel-header">
                <strong>{groupLabels.get(groupId) ?? groupId}</strong>
              </div>
              <div className="control-panel-form-grid">
                {groupKeys.map((key) => renderField(key, fields[key] ?? {}, values, props.persistedValues, props.recipeValues, props.lastRunValues, props.runState, props.onChange, props.onResetFieldToRecipe, props.onStartVisualRoiEdit, props.onUseFullFrame))}
              </div>
            </div>
          );
        })}
      </details>
      <div className="control-panel-button-row">
        <button type="button" disabled={props.loading} onClick={props.onApply}>Apply</button>
        <button type="button" disabled={props.loading} onClick={props.onApplyRun}>Apply + rerun</button>
        <button type="button" disabled={props.loading || !props.hasTuneChanges} onClick={props.onDiscard}>Discard</button>
      </div>
      {props.activePreset == null ? <small className="section-note">Preset modified</small> : null}
    </>
  );
}
