import type { ProcessingProfiling, ProcessingResult } from "../api/client";
import {
  buildOptimizationHints,
  buildProfilingBadges,
  buildTimelineSegments,
  formatCount,
  formatMs,
  getSlowestStage,
  profilingCategoryLabel,
  sortedProfilingStages,
} from "../profiling";

type Props = {
  result: ProcessingResult | null;
};

function stageKey(stage: { name: string; started_at?: number; duration_ms: number }) {
  return `${stage.name}-${stage.started_at ?? stage.duration_ms}`;
}

function EmptyProfilingFallback({ profiling }: { profiling: ProcessingProfiling | null | undefined }) {
  if (profiling) {
    return <div className="empty-state compact">No profiling stages were recorded for this result.</div>;
  }
  return <div className="empty-state compact">Profiling data is unavailable for this result.</div>;
}

export default function ProfilingBreakdown({ result }: Props) {
  const profiling = result?.profiling ?? null;
  const slowestStage = getSlowestStage(profiling);
  const timeline = buildTimelineSegments(profiling);
  const sortedStages = sortedProfilingStages(profiling);
  const hints = buildOptimizationHints(profiling);
  const badges = buildProfilingBadges(result);

  return (
    <section className="section profiling-section" data-testid="profiling-breakdown">
      <div className="section-heading profiling-heading">
        <h3>Profiling breakdown</h3>
        <div className="profiling-badges" aria-label="profiling mode badges">
          {badges.map((badge) => (
            <span className={`profiling-badge profiling-badge-${badge.tone}`} key={badge.label}>
              {badge.label}
            </span>
          ))}
        </div>
      </div>

      {profiling ? (
        <>
          <div className="profiling-summary-grid">
            <div>
              <span>Total ms</span>
              <strong>{formatMs(profiling.total_ms)}</strong>
            </div>
            <div className={profiling.production_ms > 200 ? "timing-warning-cell" : undefined}>
              <span>Production ms</span>
              <strong>{formatMs(profiling.production_ms)}</strong>
              {profiling.production_ms > 200 && <em>Above target</em>}
            </div>
            <div>
              <span>IO ms</span>
              <strong>{formatMs(profiling.io_ms)}</strong>
            </div>
            <div className={profiling.debug_artifacts_ms > profiling.production_ms ? "timing-warning-cell" : undefined}>
              <span>Debug artifact ms</span>
              <strong>{formatMs(profiling.debug_artifacts_ms)}</strong>
            </div>
            <div>
              <span>Slowest stage</span>
              <strong>{slowestStage ? slowestStage.name : "-"}</strong>
            </div>
          </div>

          {timeline.length ? (
            <>
              <div className="profiling-timeline-wrap" aria-label="stage timeline visualization">
                <div className="profiling-timeline">
                  {timeline.map((stage) => (
                    <div className={stage.isSlowest ? "timeline-row slowest" : "timeline-row"} key={stageKey(stage)}>
                      <div className="timeline-stage-name">{stage.name}</div>
                      <div className="timeline-bar-track">
                        <div
                          className={`timeline-bar category-${stage.category}`}
                          style={{ width: `${stage.widthPercent}%` }}
                          title={`${stage.name}: ${formatMs(stage.duration_ms)}`}
                        />
                      </div>
                      <div className="timeline-duration">{formatMs(stage.duration_ms)}</div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="stage-table-wrap">
                <table className="stage-table compact-stage-table">
                  <thead>
                    <tr>
                      <th>Stage</th>
                      <th>Category</th>
                      <th>Duration ms</th>
                      <th>% total</th>
                      <th>Input points</th>
                      <th>Output points</th>
                      <th>Notes</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sortedStages.map((stage) => (
                      <tr
                        className={[stage.isSlowest ? "stage-row-slowest" : "", stage.isDominant ? "stage-row-dominant" : ""]
                          .filter(Boolean)
                          .join(" ")}
                        key={stageKey(stage)}
                      >
                        <td>{stage.name}</td>
                        <td>{profilingCategoryLabel(stage.category)}</td>
                        <td>{stage.duration_ms.toFixed(1)}</td>
                        <td>{stage.percentOfTotal.toFixed(1)}%</td>
                        <td>{formatCount(stage.input_points)}</td>
                        <td>{formatCount(stage.output_points)}</td>
                        <td>{stage.notes ?? ""}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          ) : (
            <EmptyProfilingFallback profiling={profiling} />
          )}

          <div className="optimization-panel">
            <h4>Optimization hints</h4>
            {hints.length ? (
              <ul>
                {hints.map((hint) => (
                  <li key={hint}>{hint}</li>
                ))}
              </ul>
            ) : (
              <p>No obvious profiling heuristics were triggered.</p>
            )}
          </div>

          <details className="raw-profiler-json">
            <summary>Raw profiling JSON</summary>
            <pre className="json-block">{JSON.stringify(profiling, null, 2)}</pre>
          </details>
        </>
      ) : (
        <EmptyProfilingFallback profiling={profiling} />
      )}
    </section>
  );
}
