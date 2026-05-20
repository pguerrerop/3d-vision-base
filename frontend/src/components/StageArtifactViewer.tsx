import { fileUrl, type ProcessingResult } from "../api/client";
import ImagePanel from "./ImagePanel";

type Props = {
  takeId: string;
  result: ProcessingResult | null | undefined;
  stageId?: string;
};

const IMAGE_ARTIFACTS = new Set([
  "input_preview",
  "plane_segmentation",
  "foreground",
  "clusters",
  "calibrated_planes",
  "filtered_foreground",
  "rejected_points",
  "filtered_clusters",
  "overlay",
]);

export default function StageArtifactViewer({ takeId, result, stageId }: Props) {
  const outputs = result?.stage_outputs ?? [];
  const visible = stageId ? outputs.filter((output) => output.stage === stageId) : outputs;
  if (!visible.length) {
    return <div className="empty-state">No stage outputs are recorded for this pipeline yet.</div>;
  }
  return (
    <div className="stage-artifact-list">
      {visible.map((output) => (
        <article className="stage-artifact-card" key={output.stage}>
          <div className="stage-artifact-title">
            <strong>{output.display_name ?? output.stage}</strong>
            {output.implemented === false && <span>Future</span>}
          </div>
          {Object.keys(output.artifacts ?? {}).length ? (
            <div className="stage-artifact-grid">
              {Object.entries(output.artifacts).map(([key, filename]) => (
                IMAGE_ARTIFACTS.has(key) && filename ? (
                  <ImagePanel key={key} title={key.replace(/_/g, " ")} src={fileUrl(takeId, String(filename))} emptyMessage="Artifact missing." />
                ) : (
                  <div className="artifact-reference" key={key}>
                    <span>{key}</span>
                    <strong>{filename ?? "-"}</strong>
                  </div>
                )
              ))}
            </div>
          ) : (
            <p className="section-note">{output.implemented === false ? "Fusion pipeline not implemented yet." : "This stage did not publish visual artifacts."}</p>
          )}
        </article>
      ))}
    </div>
  );
}
