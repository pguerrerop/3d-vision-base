import type { PipelineRunEntry } from "../api/client";
import { formatRunTimestamp, runRecipeLabel, runTypeLabel, sortRunsNewestFirst } from "./runHistoryModel";

type Props = {
  runs: PipelineRunEntry[];
  activeRunId: string | null;
  onSelectRun: (runId: string | null) => void;
  loading?: boolean;
};

export default function RunHistoryPanel({ runs, activeRunId, onSelectRun, loading }: Props) {
  const ordered = sortRunsNewestFirst(runs);
  return (
    <div className="run-history-panel">
      <div className="run-history-header">
        <strong>Run history</strong>
        {loading ? <small>Loading runs…</small> : null}
      </div>
      {ordered.length === 0 ? (
        <div className="empty-state compact">No indexed runs for this take yet.</div>
      ) : (
        <ul className="run-history-list">
          {ordered.map((run) => {
            const isActive = run.run_id === activeRunId || (activeRunId == null && ordered[0]?.run_id === run.run_id);
            return (
              <li key={run.run_id} className={isActive ? "run-history-item active" : "run-history-item"}>
                <button type="button" onClick={() => onSelectRun(run.run_id)}>
                  <span className="toolbar-chip">{runTypeLabel(run.run_type)}</span>
                  <span>{formatRunTimestamp(run.created_at)}</span>
                  {runRecipeLabel(run) ? <small>{runRecipeLabel(run)}</small> : null}
                  {run.parent_run_id ? <small>Parent: {run.parent_run_id}</small> : null}
                  {run.boundary_stage_id ? <small>Boundary: {run.boundary_stage_id}</small> : null}
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
