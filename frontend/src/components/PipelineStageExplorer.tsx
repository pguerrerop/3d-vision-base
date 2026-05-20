import type { PipelineInfo, ProcessingResult } from "../api/client";
import StageArtifactViewer from "./StageArtifactViewer";
import { stageTimingMs } from "./pipelineModel";

type Props = {
  pipeline: PipelineInfo | null;
  result: ProcessingResult | null;
  takeId?: string | null;
};

export default function PipelineStageExplorer({ pipeline, result, takeId }: Props) {
  const stages = pipeline?.stages ?? result?.processing_pipeline?.stages ?? [];
  if (!stages.length) {
    return <div className="empty-state">No pipeline stages are registered.</div>;
  }
  return (
    <section className="concept-panel">
      <div className="concept-heading">
        <span className="eyebrow">Pipeline stages</span>
        <strong>{pipeline?.display_name ?? result?.processing_pipeline?.display_name ?? "Pipeline"}</strong>
      </div>
      <div className="stage-explorer">
        {stages.map((stage, index) => {
          const timing = stageTimingMs(result, stage.id);
          return (
            <article className="stage-row-card" key={stage.id}>
              <div>
                <span>{index + 1}</span>
                <strong>{stage.display_name}</strong>
                <small>{timing == null ? "not run or not timed" : `${timing.toFixed(timing >= 100 ? 0 : 1)} ms`}</small>
              </div>
              <button disabled type="button">Run up to stage</button>
            </article>
          );
        })}
      </div>
      {takeId && <StageArtifactViewer takeId={takeId} result={result} />}
    </section>
  );
}
