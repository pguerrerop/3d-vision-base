import { Fragment, useEffect, useState } from "react";
import { api, type ValidationStageTrend } from "../../api/client";
import { describeApiError } from "./validationCaseModel";
import {
  executionLabel,
  fieldMeanTrend,
  pointRegressionRate,
  seriesFields,
  stagesWithTrendData,
} from "./validationStageTrendModel";

interface Props {
  suiteId: string;
}

const fmt = (value: number | null, digits = 3): string =>
  value === null ? "—" : value.toFixed(digits);
const pct = (value: number | null): string =>
  value === null ? "—" : `${(value * 100).toFixed(0)}%`;

export default function ValidationStageTrendPanel({ suiteId }: Props) {
  const [trend, setTrend] = useState<ValidationStageTrend | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    setTrend(null);
    setError("");
    let cancelled = false;
    api
      .validationStageTrend(suiteId, 5)
      .then((result) => {
        if (!cancelled) setTrend(result);
      })
      .catch((caught: unknown) => {
        if (!cancelled) setError(describeApiError(caught));
      });
    return () => {
      cancelled = true;
    };
  }, [suiteId]);

  if (error) {
    return (
      <section className="control-panel-section">
        <h2>Stage trend</h2>
        <p role="alert">{error}</p>
      </section>
    );
  }
  if (!trend) return null;

  const stages = stagesWithTrendData(trend.stages);
  if (trend.executions.length < 2) {
    return (
      <section className="control-panel-section">
        <h2>Stage trend</h2>
        <p>
          <small>Run this suite at least twice to see a trend across executions.</small>
        </p>
      </section>
    );
  }

  return (
    <section className="control-panel-section">
      <h2>Stage trend</h2>
      <p>
        <small>
          The same rollup lined up across the last {trend.executions.length} executions of this
          suite. A dash is a genuine gap — that stage or comparator produced nothing that run —
          never a zero.
        </small>
      </p>
      {!stages.length ? (
        <p>
          <small>No stage has produced a comparison in any of these executions yet.</small>
        </p>
      ) : null}
      {stages.map((stage) => (
        <div key={stage.stage_id} className="table-wrap" style={{ marginBottom: "14px" }}>
          <table>
            <thead>
              <tr>
                <th>{stage.label}</th>
                {trend.executions.map((execution) => (
                  <th key={execution.execution_id}>{executionLabel(execution)}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {stage.comparators.map((series) => (
                <Fragment key={series.comparator}>
                  <tr>
                    <td>{series.comparator} — regressed</td>
                    {series.points.map((point, index) => (
                      <td key={index}>{pct(pointRegressionRate(point))}</td>
                    ))}
                  </tr>
                  {seriesFields(series).map((field) => (
                    <tr key={field}>
                      <td>
                        <small>{field} (mean)</small>
                      </td>
                      {fieldMeanTrend(series, field).map((value, index) => (
                        <td key={index}>{fmt(value)}</td>
                      ))}
                    </tr>
                  ))}
                </Fragment>
              ))}
            </tbody>
          </table>
        </div>
      ))}
    </section>
  );
}
