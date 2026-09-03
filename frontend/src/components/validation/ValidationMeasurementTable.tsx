import { useState } from "react";
import type { ValidationComparisonResult } from "../../api/client";
import {
  MEASUREMENT_FILTERS,
  describeReason,
  filterMeasurementRows,
  measurementCounts,
  measurementRows,
  type MeasurementFilter,
} from "./validationComparisonModel";

interface Props {
  result: ValidationComparisonResult;
}

export default function ValidationMeasurementTable({ result }: Props) {
  const [filter, setFilter] = useState<MeasurementFilter>("all");
  const rows = measurementRows(result);
  const filtered = filterMeasurementRows(rows, filter);
  const counts = measurementCounts(result);
  const ambiguous = result.matching?.ambiguous ?? [];

  return (
    <section className="control-panel-section">
      <h4>Measurements</h4>
      <div className="table-wrap">
        <table>
          <tbody>
            <tr>
              <th scope="row">Baseline objects</th>
              <td>{counts.baselineObjects}</td>
              <th scope="row">Candidate objects</th>
              <td>{counts.candidateObjects}</td>
            </tr>
            <tr>
              <th scope="row">Matched</th>
              <td>{counts.matched}</td>
              <th scope="row">Missing / added</th>
              <td>
                {counts.missing} / {counts.added}
              </td>
            </tr>
            <tr>
              <th scope="row">Ambiguous</th>
              <td>{counts.ambiguous}</td>
              <th scope="row">Regressed metrics</th>
              <td>{counts.regressedMetrics}</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div className="control-panel-button-row">
        {MEASUREMENT_FILTERS.map((option) => (
          <button
            key={option.value}
            type="button"
            disabled={option.value === filter}
            onClick={() => setFilter(option.value)}
          >
            {option.label}
          </button>
        ))}
      </div>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Object</th>
              <th>Match</th>
              <th>Metric</th>
              <th>Baseline</th>
              <th>Candidate</th>
              <th>Delta</th>
              <th>Relative Δ</th>
              <th>Tolerance</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((row, index) => {
              if (row.kind === "unmatched-object") {
                return (
                  <tr key={`unmatched-${row.side}-${index}`}>
                    <td>{row.objectId ?? "—"}</td>
                    <td colSpan={7}>
                      <em>
                        {row.side === "missing" ? "Missing from candidate" : "Added in candidate"}
                      </em>
                    </td>
                    <td>
                      <span className="status-badge regression">{row.side}</span>
                    </td>
                  </tr>
                );
              }
              const { result: object, metric } = row;
              return (
                <tr key={`${object.baseline_object_id}-${metric.metric}-${index}`}>
                  <td>{object.baseline_object_id ?? object.candidate_object_id ?? "—"}</td>
                  <td>
                    {object.match_method}
                    {object.match_score < 1 ? ` (${object.match_score.toFixed(2)})` : ""}
                  </td>
                  <td>{metric.metric}</td>
                  <td>
                    {formatValue(metric.baseline)} {metric.units ?? ""}
                  </td>
                  <td>
                    {formatValue(metric.candidate)} {metric.units ?? ""}
                  </td>
                  <td>{metric.delta === undefined ? "—" : metric.delta.toFixed(3)}</td>
                  <td>
                    {metric.relative_delta === undefined
                      ? "—"
                      : `${(metric.relative_delta * 100).toFixed(1)}%`}
                  </td>
                  <td>
                    {metric.effective_tolerance === undefined
                      ? "—"
                      : metric.effective_tolerance.toFixed(3)}
                  </td>
                  <td>
                    <span
                      className={`status-badge ${metric.status}`}
                      title={metric.reason ? describeReason(metric.reason) : undefined}
                    >
                      {metric.status}
                    </span>
                  </td>
                </tr>
              );
            })}
            {!filtered.length ? (
              <tr>
                <td colSpan={9}>Nothing matches this filter.</td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>

      {filter === "ambiguous" && ambiguous.length ? (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Baseline object</th>
                <th>Ambiguous candidates</th>
              </tr>
            </thead>
            <tbody>
              {ambiguous.map((entry, index) => (
                <tr key={`${entry.baseline_object_id}-${index}`}>
                  <td>{entry.baseline_object_id ?? "—"}</td>
                  <td>{entry.candidate_indices.join(", ")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </section>
  );
}

function formatValue(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") return value.toFixed(3);
  return String(value);
}
