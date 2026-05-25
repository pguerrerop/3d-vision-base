import { Fragment, useState, type ReactElement } from "react";

export type MetricTraceRow = Record<string, unknown>;

type Props = {
  objects: Record<string, unknown>[];
  selectedObjectId: number | null;
  onSelectObject: (objectId: number) => void;
};

function fmtNumber(value: unknown, fractionDigits: number = 4): string {
  if (value === null || value === undefined) return "-";
  if (typeof value === "number" && Number.isFinite(value)) {
    if (Math.abs(value) >= 1000) return value.toFixed(Math.max(0, fractionDigits - 2));
    return value.toFixed(fractionDigits).replace(/\.?0+$/, "");
  }
  return String(value);
}

function fmtMaybeNumber(value: unknown): string {
  if (typeof value === "number") return fmtNumber(value);
  return String(value ?? "-");
}

function scaleSummary(scales: Record<string, unknown> | null | undefined): string {
  if (!scales) return "-";
  const sx = scales.x;
  const sy = scales.y;
  const sz = scales.z;
  const fmt = (v: unknown) => (typeof v === "number" ? v.toFixed(4).replace(/\.?0+$/, "") : "-");
  return `x=${fmt(sx)} y=${fmt(sy)} z=${fmt(sz)}`;
}

function validityClass(status: unknown): string {
  const value = String(status ?? "").toLowerCase();
  if (value === "invalid" || value === "suspicious") return "status-pill bad";
  if (value === "not_applied") return "status-pill warn";
  if (value === "valid") return "status-pill ok";
  return "status-pill";
}

function MetricTraceRowExpanded(props: { row: MetricTraceRow }): ReactElement {
  const row = props.row;
  const inputs = Array.isArray(row.formula_inputs) ? (row.formula_inputs as unknown[]) : [];
  const intermediate = (row.intermediate_values && typeof row.intermediate_values === "object")
    ? (row.intermediate_values as Record<string, unknown>)
    : {};
  const warnings = Array.isArray(row.warnings) ? (row.warnings as unknown[]) : [];
  return (
    <div className="metric-trace-expanded">
      <div className="metric-trace-expanded-section">
        <strong>Formula inputs</strong>
        {inputs.length === 0 ? (
          <em>No inputs recorded.</em>
        ) : (
          <table className="compact-candidate-table metric-trace-inner-table">
            <thead>
              <tr><th>Name</th><th>Value</th></tr>
            </thead>
            <tbody>
              {inputs.map((entry, idx) => {
                const item = (entry && typeof entry === "object") ? entry as Record<string, unknown> : {};
                return (
                  <tr key={`mi_${String(item.name ?? idx)}_${idx}`}>
                    <td>{String(item.name ?? "-")}</td>
                    <td>{fmtMaybeNumber(item.value)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
      <div className="metric-trace-expanded-section">
        <strong>Intermediate values</strong>
        {Object.keys(intermediate).length === 0 ? (
          <em>No intermediates recorded.</em>
        ) : (
          <table className="compact-candidate-table metric-trace-inner-table">
            <thead>
              <tr><th>Key</th><th>Value</th></tr>
            </thead>
            <tbody>
              {Object.entries(intermediate).map(([key, value]) => (
                <tr key={`int_${key}`}>
                  <td>{key}</td>
                  <td>{fmtMaybeNumber(value)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
      {warnings.length > 0 ? (
        <div className="metric-trace-expanded-section">
          <strong>Warnings</strong>
          <ul className="metric-trace-warning-list">
            {warnings.map((value, idx) => (
              <li key={`warn_${idx}`} className="metric-trace-warning-item">{String(value)}</li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}

export function MetricDetailsPanel(props: Props): ReactElement {
  const { objects, selectedObjectId, onSelectObject } = props;
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});

  if (!objects.length) {
    return <div className="empty-state"><strong>No metric explanation objects available.</strong></div>;
  }
  const selected = objects.find((row) => Number(row.object_id ?? -1) === selectedObjectId) ?? objects[0];
  const traces = (Array.isArray(selected.metric_trace) ? selected.metric_trace : [])
    .filter((item): item is MetricTraceRow => typeof item === "object" && item !== null);

  const calibrationRow = traces.find((row) => row.metric_key === "scale_correction_applied");
  const invariantRow = traces.find((row) => row.metric_key === "geometry_invariant_warnings");
  const metricRows = traces.filter((row) => {
    const key = String(row.metric_key ?? "");
    return key !== "scale_correction_applied" && key !== "geometry_invariant_warnings";
  });

  const calibration = (calibrationRow ?? null) as MetricTraceRow | null;
  const calibrationScales = (calibration?.correction_scales ?? null) as Record<string, unknown> | null;
  const calibrationIntermediate = (calibration?.intermediate_values ?? {}) as Record<string, unknown>;
  const calibrationInputs = Array.isArray(calibration?.formula_inputs) ? (calibration?.formula_inputs as unknown[]) : [];
  const correctionSource = calibrationInputs.find((entry) => {
    const it = (entry && typeof entry === "object") ? entry as Record<string, unknown> : {};
    return it.name === "correction_source";
  });
  const correctionContextId = calibrationInputs.find((entry) => {
    const it = (entry && typeof entry === "object") ? entry as Record<string, unknown> : {};
    return it.name === "correction_context_id";
  });
  const correctionSourceValue = correctionSource && typeof correctionSource === "object"
    ? String((correctionSource as Record<string, unknown>).value ?? "-")
    : "-";
  const correctionContextValue = correctionContextId && typeof correctionContextId === "object"
    ? String((correctionContextId as Record<string, unknown>).value ?? "-")
    : "-";
  const geometryMetricSource = String(calibrationIntermediate.geometry_metric_source ?? "-");
  const geometryCoordSpace = String(calibrationIntermediate.geometry_coordinate_space ?? "-");
  const correctionApplied = Boolean(calibration?.correction_applied);

  const invariants = invariantRow ? (Array.isArray(invariantRow.warnings) ? (invariantRow.warnings as unknown[]) : []) : [];
  const invariantIntermediate = (invariantRow?.intermediate_values ?? {}) as Record<string, unknown>;
  const violations = Array.isArray(invariantIntermediate.violations) ? (invariantIntermediate.violations as unknown[]) : [];
  const advisories = Array.isArray(invariantIntermediate.advisories) ? (invariantIntermediate.advisories as unknown[]) : [];

  return (
    <div className="studio-table-pane metric-details-panel">
      {objects.length > 1 ? (
        <div className="overlay-controls">
          <label>
            Object
            <select
              value={Number(selected.object_id ?? 0)}
              onChange={(event) => onSelectObject(Number(event.target.value))}
            >
              {objects.map((row) => (
                <option key={`metric_obj_${String(row.object_id ?? "")}`} value={Number(row.object_id ?? 0)}>
                  object_{String(row.object_id ?? "-")}
                </option>
              ))}
            </select>
          </label>
        </div>
      ) : null}

      <section className="metric-details-context">
        <header className="metric-details-context-header">
          <strong>Correction context</strong>
          <span className={correctionApplied ? "status-pill ok" : "status-pill warn"}>
            {correctionApplied ? "applied" : "not applied"}
          </span>
        </header>
        <dl className="metric-details-context-grid">
          <div><dt>Source</dt><dd>{correctionSourceValue}</dd></div>
          <div><dt>Context id</dt><dd>{correctionContextValue}</dd></div>
          <div><dt>Scales</dt><dd>{scaleSummary(calibrationScales)}</dd></div>
          <div><dt>Geometry metric source</dt><dd>{geometryMetricSource}</dd></div>
          <div><dt>Coordinate space</dt><dd>{geometryCoordSpace}</dd></div>
        </dl>
        {invariants.length > 0 ? (
          <div className="metric-details-invariants">
            <strong>Invariants</strong>
            {violations.length > 0 ? (
              <ul className="metric-trace-warning-list">
                {violations.map((value, idx) => (
                  <li key={`viol_${idx}`} className="metric-trace-warning-item violation">{String(value)}</li>
                ))}
              </ul>
            ) : null}
            {advisories.length > 0 ? (
              <ul className="metric-trace-warning-list">
                {advisories.map((value, idx) => (
                  <li key={`adv_${idx}`} className="metric-trace-warning-item advisory">{String(value)}</li>
                ))}
              </ul>
            ) : null}
          </div>
        ) : null}
      </section>

      <table className="compact-candidate-table metric-trace-table">
        <thead>
          <tr>
            <th></th>
            <th>Metric</th>
            <th>Raw</th>
            <th>Corrected</th>
            <th>Correction factor</th>
            <th>Coord space (before / after)</th>
            <th>Source</th>
            <th>Formula</th>
            <th>Validity</th>
            <th>Used by classifier</th>
          </tr>
        </thead>
        <tbody>
          {metricRows.map((row, idx) => {
            const traceKey = String(row.trace_id ?? row.metric_key ?? `row_${idx}`);
            const isOpen = Boolean(expanded[traceKey]);
            const warnings = Array.isArray(row.warnings) ? (row.warnings as unknown[]) : [];
            const validity = String(row.validity_status ?? "-");
            return (
              <Fragment key={traceKey}>
                <tr className={isOpen ? "metric-trace-row open" : "metric-trace-row"}>
                  <td>
                    <button
                      type="button"
                      className="metric-trace-expand-toggle"
                      onClick={() => setExpanded((current) => ({ ...current, [traceKey]: !current[traceKey] }))}
                      aria-expanded={isOpen}
                      aria-label={isOpen ? "Collapse metric trace" : "Expand metric trace"}
                    >
                      {isOpen ? "v" : ">"}
                    </button>
                  </td>
                  <td>
                    <code>{String(row.metric_key ?? "-")}</code>
                  </td>
                  <td>{fmtMaybeNumber(row.raw_value)}</td>
                  <td>
                    <strong>{fmtMaybeNumber(row.corrected_value)}</strong>
                  </td>
                  <td>{fmtMaybeNumber(row.correction_factor_used)}</td>
                  <td>
                    <code>{String(row.coordinate_space_before ?? "-")}</code>
                    {" -> "}
                    <code>{String(row.coordinate_space_after ?? "-")}</code>
                  </td>
                  <td>{String(row.source_artifact_id ?? "-")}</td>
                  <td className="metric-trace-formula">
                    <div><strong>{String(row.formula_name ?? "-")}</strong></div>
                    <div className="metric-trace-formula-text">{String(row.formula_human_readable ?? "-")}</div>
                  </td>
                  <td>
                    <span className={validityClass(validity)}>{validity}</span>
                    {warnings.length > 0 ? (
                      <small className="metric-trace-warning-count">{warnings.length} warning{warnings.length === 1 ? "" : "s"}</small>
                    ) : null}
                  </td>
                  <td>{Boolean(row.used_by_classifier) ? "yes" : "no"}</td>
                </tr>
                {isOpen ? (
                  <tr className="metric-trace-expanded-row">
                    <td></td>
                    <td colSpan={9}>
                      <MetricTraceRowExpanded row={row} />
                    </td>
                  </tr>
                ) : null}
              </Fragment>
            );
          })}
        </tbody>
      </table>

      <details className="measurement-table-section metric-trace-json">
        <summary>Selected metric traces JSON</summary>
        <pre className="json-block">{JSON.stringify(traces, null, 2)}</pre>
      </details>
    </div>
  );
}

export default MetricDetailsPanel;
