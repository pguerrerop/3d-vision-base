import type { PipelineStageInfo } from "../api/client";
import { useMemo, useState } from "react";
import { visibleStageParameterKeys } from "./stageParameterModel";

type FieldMeta = {
  type?: string;
  label?: string;
  enum?: Array<string | number>;
  minimum?: number;
  maximum?: number;
  nullable?: boolean;
  group?: string;
  visible_when?: Record<string, unknown>;
};

type Props = {
  stage: PipelineStageInfo | null;
  values: Record<string, unknown> | null;
  persistedValues: Record<string, unknown> | null;
  loading?: boolean;
  dirty?: boolean;
  previewLabel?: string;
  error?: string | null;
  onChange: (key: string, value: unknown) => void;
  onPreview: () => void;
  onApply: () => void;
  onApplyRun: () => void;
  onReset: () => void;
};

export default function StageParameterPanel(props: Props) {
  const schema = (props.stage?.parameter_schema ?? {}) as { fields?: Record<string, FieldMeta> };
  const fields = schema.fields ?? {};
  const keys = Object.keys(fields);
  if (!props.stage || !keys.length || !props.values) return null;
  const [showAdvanced, setShowAdvanced] = useState(false);
  const hasAdvanced = keys.some((key) => (fields[key]?.group ?? "") === "advanced");
  const visibleKeys = useMemo(
    () => visibleStageParameterKeys(fields, props.values ?? {}, showAdvanced),
    [fields, props.values, showAdvanced]
  );

  return (
    <section className="segmentation-controls-strip">
      {visibleKeys.map((key) => {
        const meta = fields[key] ?? {};
        const value = props.values?.[key];
        if (Array.isArray(meta.enum)) {
          return (
            <label key={key} className="field-label">
              {meta.label ?? key}
              <select value={String(value ?? meta.enum[0] ?? "")} onChange={(event) => props.onChange(key, event.target.value)}>
                {meta.enum.map((item) => <option key={String(item)} value={String(item)}>{String(item)}</option>)}
              </select>
            </label>
          );
        }
        if (meta.type === "boolean") {
          return (
            <label key={key}>
              <input checked={Boolean(value)} type="checkbox" onChange={(event) => props.onChange(key, event.target.checked)} />
              {meta.label ?? key}
            </label>
          );
        }
        const numeric = Number(value ?? 0);
        return (
          <label key={key} className="field-label">
            {meta.label ?? key}
            <input
              type="number"
              min={meta.minimum}
              max={meta.maximum}
              value={Number.isFinite(numeric) ? numeric : ""}
              onChange={(event) => props.onChange(key, event.target.value === "" && meta.nullable ? null : Number(event.target.value))}
            />
          </label>
        );
      })}
      {hasAdvanced && (
        <div className="segmentation-controls-row">
          <button type="button" onClick={() => setShowAdvanced((current) => !current)}>{showAdvanced ? "Hide advanced" : "Show advanced"}</button>
        </div>
      )}
      <div className="segmentation-controls-row">
        <button type="button" disabled={props.loading} onClick={props.onPreview}>{props.loading ? "Previewing..." : "Preview"}</button>
        <button type="button" onClick={props.onApply}>Apply to recipe</button>
        <button type="button" onClick={props.onApplyRun}>Apply + run</button>
        <button type="button" onClick={props.onReset}>Reset to defaults</button>
      </div>
      {props.previewLabel && <small>{props.previewLabel}</small>}
      {props.persistedValues && props.values && (
        <small className="section-note">
          {props.dirty ? "Preview differs from persisted recipe." : "Preview matches persisted recipe."}
        </small>
      )}
      {props.error && <small className="section-note">{props.error}</small>}
    </section>
  );
}
