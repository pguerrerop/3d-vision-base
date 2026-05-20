import type { PipelineExecutionTrace } from "../api/client";

type Props = {
  execution: PipelineExecutionTrace | null | undefined;
  selectedStageId: string;
};

export default function PipelineExecutionGraph({ execution, selectedStageId }: Props) {
  const stages = execution?.stages ?? [];
  if (!stages.length) {
    return <div className="empty-state compact">Execution trace will appear after processing runs.</div>;
  }
  return (
    <section className="studio-execution-graph">
      {stages.map((stage, index) => (
        <div className="execution-node-wrap" key={stage.stage_id}>
          <div className={`execution-node ${stage.stage_id === selectedStageId ? "active" : ""} ${stage.status}`}>
            <span>{stage.stage_id}</span>
            <strong>{stage.status}</strong>
            <small>{stage.duration_ms ? `${stage.duration_ms.toFixed(1)} ms` : "not run"}</small>
            <small>out: {stage.output_artifact_ids.length}</small>
            <small>w/e: {stage.warnings.length}/{stage.errors.length}</small>
          </div>
          {index < stages.length - 1 && <div className="execution-arrow">→</div>}
        </div>
      ))}
    </section>
  );
}
