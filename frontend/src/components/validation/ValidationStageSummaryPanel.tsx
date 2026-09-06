import { useEffect, useState } from "react";
import { api, type ValidationStageSummary } from "../../api/client";
import { describeApiError } from "./validationCaseModel";
import { validationStatus, type ValidationStatus } from "./validationStatus";
import {
  fieldRows,
  regressionRate,
  sortedStatusEntries,
  stageHasData,
} from "./validationStageSummaryModel";

interface Props {
  executionId: string | null;
}

const badge = (status: string) =>
  validationStatus[status as ValidationStatus] ?? {
    label: status,
    severity: "muted" as const,
    icon: "?",
    tooltip: status,
  };

const pct = (value: number | null): string =>
  value === null ? "—" : `${(value * 100).toFixed(0)}%`;

export default function ValidationStageSummaryPanel({ executionId }: Props) {
  const [summary, setSummary] = useState<ValidationStageSummary | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    setSummary(null);
    setError("");
    if (!executionId) return;
    let cancelled = false;
    api
      .validationStageSummary(executionId)
      .then((result) => {
        if (!cancelled) setSummary(result);
      })
      .catch((caught: unknown) => {
        if (!cancelled) setError(describeApiError(caught));
      });
    return () => {
      cancelled = true;
    };
  }, [executionId]);

  if (!executionId) return null;

  const withData = summary?.stages.filter(stageHasData) ?? [];
  const withoutData = summary?.stages.filter((stage) => !stageHasData(stage)) ?? [];

  return (
    <section className="control-panel-section">
      <h2>Stage summary</h2>
      <p>
        <small>
          Metrics rolled up across every enabled case in this execution, per stage and comparator. A
          quiet drift that never fails any single case can still show up here.
        </small>
      </p>
      {error ? <p role="alert">{error}</p> : null}
      {!summary && !error ? (
        <p>
          <small>Loading…</small>
        </p>
      ) : null}

      {withData.map((stage) => (
        <div key={stage.stage_id} style={{ marginBottom: "18px" }}>
          <h3 style={{ marginBottom: "6px" }}>{stage.label}</h3>
          {stage.comparators.map((bucket) => {
            const rate = regressionRate(bucket);
            return (
              <div key={bucket.comparator} className="table-wrap" style={{ marginBottom: "10px" }}>
                <table>
                  <thead>
                    <tr>
                      <th>{bucket.comparator}</th>
                      <th>Compared</th>
                      <th>Regressed</th>
                      <th>Status breakdown</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>
                        <small>sample</small>
                      </td>
                      <td>{bucket.count}</td>
                      <td>{pct(rate)}</td>
                      <td>
                        {sortedStatusEntries(bucket.status_counts).map(([status, count]) => {
                          const presentation = badge(status);
                          return (
                            <span
                              key={status}
                              className={`status-badge ${status}`}
                              title={presentation.tooltip}
                              style={{ marginRight: "6px" }}
                            >
                              {presentation.icon} {count}
                            </span>
                          );
                        })}
                      </td>
                    </tr>
                    {fieldRows(bucket).length ? (
                      <tr>
                        <td colSpan={4}>
                          <table>
                            <thead>
                              <tr>
                                <th>Metric</th>
                                <th>n</th>
                                <th>Mean</th>
                                <th>Median</th>
                                <th>P95</th>
                                <th>Min</th>
                                <th>Max</th>
                              </tr>
                            </thead>
                            <tbody>
                              {fieldRows(bucket).map(({ field, summary: fieldSummary }) => (
                                <tr key={field}>
                                  <td>{field}</td>
                                  <td>{fieldSummary.count}</td>
                                  <td>{fieldSummary.mean.toFixed(3)}</td>
                                  <td>{fieldSummary.median.toFixed(3)}</td>
                                  <td>{fieldSummary.p95.toFixed(3)}</td>
                                  <td>{fieldSummary.min.toFixed(3)}</td>
                                  <td>{fieldSummary.max.toFixed(3)}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </td>
                      </tr>
                    ) : null}
                  </tbody>
                </table>
              </div>
            );
          })}
        </div>
      ))}

      {withoutData.length ? (
        <p>
          <small>No comparisons yet: {withoutData.map((stage) => stage.label).join(", ")}.</small>
        </p>
      ) : null}
    </section>
  );
}
