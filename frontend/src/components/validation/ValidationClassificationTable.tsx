import { useState } from "react";
import type { ValidationComparisonResult } from "../../api/client";
import {
  CLASSIFICATION_FILTERS,
  classificationCounts,
  classificationRows,
  classificationSideText,
  describeReason,
  filterClassificationRows,
  type ClassificationFilter,
} from "./validationComparisonModel";

interface Props {
  result: ValidationComparisonResult;
}

export default function ValidationClassificationTable({ result }: Props) {
  const [filter, setFilter] = useState<ClassificationFilter>("all");
  const rows = classificationRows(result);
  const filtered = filterClassificationRows(rows, filter);
  const counts = classificationCounts(result);

  return (
    <section className="control-panel-section">
      <h4>Classification</h4>
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
              <th scope="row">Label changes</th>
              <td>{counts.labelChanges}</td>
              <th scope="row">Superclass changes</th>
              <td>{counts.superclassChanges}</td>
            </tr>
            <tr>
              <th scope="row">New critical warnings</th>
              <td colSpan={3}>{counts.newCriticalWarnings}</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div className="control-panel-button-row">
        {CLASSIFICATION_FILTERS.map((option) => (
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
              <th>Baseline superclass</th>
              <th>Baseline label</th>
              <th>Candidate superclass</th>
              <th>Candidate label</th>
              <th>Confidence Δ</th>
              <th>Warnings</th>
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
              const { result: object } = row;
              return (
                <tr
                  key={`${object.baseline_object_id}-${index}`}
                  title={classificationSideText(object.baseline)}
                >
                  <td>{object.baseline_object_id ?? object.candidate_object_id ?? "—"}</td>
                  <td>{object.match_method}</td>
                  <td>{object.baseline.superclass ?? "—"}</td>
                  <td>{object.baseline.label ?? "—"}</td>
                  <td>{object.candidate.superclass ?? "—"}</td>
                  <td>{object.candidate.label ?? "—"}</td>
                  <td>{(object.confidence_delta * 100).toFixed(1)}%</td>
                  <td>{object.reasons.filter((r) => r.includes("warning")).length || "—"}</td>
                  <td>
                    <span
                      className={`status-badge ${object.status}`}
                      title={object.reasons.map(describeReason).join(" ")}
                    >
                      {object.status}
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
    </section>
  );
}
