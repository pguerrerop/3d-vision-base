import type { ProcessingUnitDefinition } from "../api/client";

import InfoHint from "./InfoHint";
import type { StudioHelpEntry } from "./studioHelp";

type ContractField = {
  id: string;
  label: string;
  type: string;
  default?: unknown;
  min?: number | null;
  max?: number | null;
  step?: number | null;
  options?: string[] | null;
  advanced?: boolean;
  affects?: string[];
  description?: string | null;
  tuning_hint?: string | null;
  active_when?: Record<string, unknown> | null;
  unit?: string | null;
  nullable?: boolean;
  group?: string | null;
};

type Props = {
  stageLabel: string;
  selectedUnit: ProcessingUnitDefinition | null;
  availableUnits: ProcessingUnitDefinition[];
  values: Record<string, unknown>;
  persistedValues: Record<string, unknown> | null;
  recipeValues?: Record<string, unknown> | null;
  lastRunValues: Record<string, unknown> | null;
  loading?: boolean;
  dirty?: boolean;
  onChange: (key: string, value: unknown) => void;
  onResetFieldToRecipe?: (key: string) => void;
  onResetUnitToRecipe?: () => void;
  onApply: () => void;
  onApplyRun: () => void;
  onDiscard: () => void;
};

function valuesEqual(a: unknown, b: unknown): boolean {
  return JSON.stringify(a ?? null) === JSON.stringify(b ?? null);
}

function visible(field: ContractField, values: Record<string, unknown>): { visible: boolean; reason: string | null } {
  for (const [dep, expected] of Object.entries(field.active_when ?? {})) {
    if (!valuesEqual(values[dep], expected)) {
      return { visible: false, reason: `${dep.replace(/_/g, " ")} must be ${String(expected).replace(/_/g, " ")}.` };
    }
  }
  return { visible: true, reason: null };
}

function valueLabel(value: unknown, field: ContractField): string {
  if (value == null || value === "") return "unset";
  if (typeof value === "boolean") return value ? "On" : "Off";
  if (typeof value === "number" && field.unit) return `${value} ${field.unit}`;
  return String(value).replace(/_/g, " ");
}

function fieldValue(values: Record<string, unknown>, persistedValues: Record<string, unknown> | null, field: ContractField): unknown {
  if (values[field.id] !== undefined) return values[field.id];
  if (persistedValues?.[field.id] !== undefined) return persistedValues[field.id];
  return field.default;
}

function helpEntry(field: ContractField): StudioHelpEntry | null {
  if (!field.description) return null;
  return {
    title: field.label,
    summary: field.description,
    details: field.tuning_hint ?? field.description,
    affects: field.affects ?? [],
  };
}

function renderField(
  field: ContractField,
  values: Record<string, unknown>,
  persistedValues: Record<string, unknown> | null,
  recipeValues: Record<string, unknown> | null | undefined,
  lastRunValues: Record<string, unknown> | null,
  onChange: (key: string, value: unknown) => void,
  onResetFieldToRecipe?: (key: string) => void,
) {
  const applicability = visible(field, values);
  const value = fieldValue(values, persistedValues, field);
  const status = !applicability.visible
    ? "Inactive"
    : field.nullable && (value == null || value === "")
      ? "Unset / auto"
      : valuesEqual(value, field.default)
        ? "Default"
        : "Recipe override";
  const detail = !applicability.visible ? applicability.reason : (field.description ?? null);
  const lastRunValue = lastRunValues?.[field.id];
  const recipeValue = recipeValues?.[field.id];
  const differsFromRecipe = recipeValue !== undefined && !valuesEqual(recipeValue, value);
  const differsFromDefault = field.default !== undefined && !valuesEqual(value, field.default);
  const differsFromLastRun = lastRunValue !== undefined && !valuesEqual(lastRunValue, value);

  const notes = (
    <>
      <small className="section-note">{status}{detail ? ` · ${detail}` : ""}</small>
      <small className={`section-note tune-baseline-recipe${differsFromRecipe ? " tune-dirty-recipe" : " tune-clean-recipe"}`}>
        Recipe: {recipeValue === undefined ? "unavailable" : valueLabel(recipeValue, field)}. {differsFromRecipe ? "Edited value differs from recipe." : "Matches recipe."}
      </small>
      <small className={`section-note tune-baseline-last-run${differsFromLastRun ? " tune-dirty-last-run" : " tune-clean-last-run"}`}>
        Last run: {lastRunValue === undefined ? "unavailable" : valueLabel(lastRunValue, field)}. {differsFromLastRun ? "Edited value differs from last run." : "Matches last run."}
      </small>
      <small className={`section-note tune-baseline-default${differsFromDefault ? " tune-dirty-default" : " tune-clean-default"}`}>
        Default: {valueLabel(field.default, field)}. {differsFromDefault ? "Edited value differs from default." : "Matches default."}
      </small>
      {field.affects?.length ? <small className="section-note">Affects: {field.affects.join(" • ")}</small> : null}
      {field.tuning_hint ? <small className="section-note">{field.tuning_hint}</small> : null}
      {differsFromRecipe && onResetFieldToRecipe ? <button type="button" onClick={() => onResetFieldToRecipe(field.id)}>Reset to recipe</button> : null}
    </>
  );

  if (Array.isArray(field.options) && field.options.length > 0) {
    return (
      <label key={field.id}>
        <span className="info-inline-label">
          <span>{field.label}</span>
          <InfoHint content={helpEntry(field)} label="Open parameter help" />
        </span>
        <select disabled={!applicability.visible} value={String(value ?? field.default ?? field.options[0] ?? "")} onChange={(event) => onChange(field.id, event.target.value)}>
          {field.options.map((option) => <option key={option} value={option}>{option}</option>)}
        </select>
        {notes}
      </label>
    );
  }
  if (field.type === "boolean") {
    return (
      <label key={field.id} className="control-panel-checkbox">
        <input disabled={!applicability.visible} checked={Boolean(value)} type="checkbox" onChange={(event) => onChange(field.id, event.target.checked)} />
        <div className="control-panel-checkbox-copy">
          <span className="info-inline-label">
            <span>{field.label}</span>
            <InfoHint content={helpEntry(field)} label="Open parameter help" />
          </span>
          {notes}
        </div>
      </label>
    );
  }
  return (
    <label key={field.id}>
      <span className="info-inline-label">
        <span>{field.label}</span>
        <InfoHint content={helpEntry(field)} label="Open parameter help" />
      </span>
      <input
        disabled={!applicability.visible}
        type="number"
        min={field.min ?? undefined}
        max={field.max ?? undefined}
        step={field.step ?? undefined}
        value={value == null || value === "" ? "" : Number(value)}
        placeholder={field.nullable ? "unset / auto" : undefined}
        onChange={(event) => {
          if (event.target.value === "") {
            onChange(field.id, field.nullable ? null : field.default ?? 0);
            return;
          }
          onChange(field.id, Number(event.target.value));
        }}
      />
      {field.unit ? <small>{field.unit}</small> : null}
      {notes}
    </label>
  );
}

export default function ContractTunePanel(props: Props) {
  const unit = props.selectedUnit;
  const fields = (unit?.parameters ?? []) as ContractField[];
  const visibleFields = fields.filter((field) => visible(field, props.values).visible);
  const visibleFieldIds = visibleFields.map((field) => field.id);
  const groupIds = Array.from(new Set(visibleFields.map((field) => field.group ?? (field.advanced ? "Advanced" : "Parameters"))));
  const upstreamLabels = (unit?.controlled_by ?? []).map((unitId) => props.availableUnits.find((candidate) => candidate.id === unitId)?.label ?? unitId);

  const groupedKeys = new Set<string>();
  const primaryGroups = (unit?.parameter_groups ?? []).map((group) => {
    const keys = (group.param_keys ?? []).filter((key) => visibleFieldIds.includes(key));
    for (const key of keys) groupedKeys.add(key);
    return { group, keys };
  });
  const fallbackFields = visibleFields.filter((field) => !groupedKeys.has(field.id));
  const hasPrimaryGroupContent = primaryGroups.some(({ group, keys }) => keys.length > 0 || group.description || group.affects?.length);

  const fallbackGroups = groupIds.map((groupId) => {
    const groupFields = fallbackFields.filter((field) => (field.group ?? (field.advanced ? "Advanced" : "Parameters")) === groupId);
    if (groupFields.length === 0) return null;
    return (
      <section key={groupId} className="control-panel-section">
        <span>{groupId}</span>
        <div className="control-panel-form-grid">
          {groupFields.map((field) => renderField(field, props.values, props.persistedValues, props.recipeValues, props.lastRunValues, props.onChange, props.onResetFieldToRecipe))}
        </div>
      </section>
    );
  });

  return (
    <>
      <section className="control-panel-section">
        <span>Tune context</span>
        <small>Stage: {props.stageLabel}</small>
        <small>Contract unit: {unit?.id ?? "none"}</small>
      </section>
      {unit?.help_markdown || unit?.downstream_effects?.length ? (
        <section className="control-panel-section">
          <span>{unit?.label}</span>
          {unit?.help_markdown ? <small>{unit.help_markdown}</small> : null}
          {unit?.downstream_effects?.length ? <small>Downstream effects: {unit.downstream_effects.join(" • ")}</small> : null}
        </section>
      ) : null}
      {!unit || visibleFields.length === 0 ? (
        <section className="control-panel-section">
          <span>{unit?.label ?? props.stageLabel}</span>
          <small>
            No tunable parameters are registered for this substage.
            {upstreamLabels.length ? ` It is controlled by upstream units: ${upstreamLabels.join(", ")}.` : ""}
          </small>
        </section>
      ) : (
        <>
          {primaryGroups.map(({ group, keys }) => {
            if (keys.length === 0 && !(group.description || group.affects?.length)) return null;
            return (
              <section key={group.id} className="control-panel-section">
                <span>{group.label}</span>
                {group.description ? <small>{group.description}</small> : null}
                {group.affects?.length ? <small>Affects: {group.affects.join(" • ")}</small> : null}
                {keys.length ? (
                  <div className="control-panel-form-grid">
                    {keys.map((key) => {
                      const field = visibleFields.find((candidate) => candidate.id === key);
                      return field ? renderField(field, props.values, props.persistedValues, props.recipeValues, props.lastRunValues, props.onChange, props.onResetFieldToRecipe) : null;
                    })}
                  </div>
                ) : null}
              </section>
            );
          })}
          {fallbackGroups.some(Boolean) ? (
            hasPrimaryGroupContent ? (
              <details className="control-panel-details">
                <summary>Other parameters for this unit</summary>
                {fallbackGroups}
              </details>
            ) : fallbackGroups
          ) : null}
        </>
      )}
      {unit?.tuning_hints?.length ? (
        <section className="control-panel-section">
          <span>What to tune</span>
          {unit.tuning_hints.map((hint) => (
            <small key={hint.condition}>
              {hint.condition}: {hint.actions.join(" • ")}
            </small>
          ))}
        </section>
      ) : null}
      {unit?.artifacts?.length ? (
        <section className="control-panel-section">
          <span>Declared artifacts</span>
          <small>{unit.artifacts.map((artifact) => artifact.artifact_id).join(" • ")}</small>
        </section>
      ) : null}
      {props.recipeValues && unit && Object.keys(props.recipeValues).length > 0 && props.onResetUnitToRecipe ? (
        <div className="control-panel-button-row">
          <button type="button" disabled={props.loading} onClick={props.onResetUnitToRecipe}>Reset unit to recipe</button>
        </div>
      ) : null}
      <div className="control-panel-button-row">
        <button type="button" disabled={props.loading} onClick={props.onApply}>Apply</button>
        <button type="button" disabled={props.loading} onClick={props.onApplyRun}>Apply + rerun</button>
        <button type="button" disabled={props.loading || !props.dirty} onClick={props.onDiscard}>Discard</button>
      </div>
    </>
  );
}
