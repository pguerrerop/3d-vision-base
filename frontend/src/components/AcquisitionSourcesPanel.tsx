import type { RuntimeState, TakeDetail } from "../api/client";
import LivePreviewPanel from "./LivePreviewPanel";
import { sourceSummary } from "./pipelineModel";

type Props = {
  runtimeState: RuntimeState | null;
  detail?: TakeDetail | null;
  compact?: boolean;
};

export default function AcquisitionSourcesPanel({ runtimeState, detail = null, compact = false }: Props) {
  const sourceDetails = runtimeState?.acquisition_source_details ?? {};
  const modalities = detail?.modalities?.length ? detail.modalities.join(", ") : "-";

  return (
    <section className={compact ? "concept-panel compact" : "concept-panel"}>
      <div className="concept-heading">
        <span className="eyebrow">Acquisition sources</span>
        <strong>Raw/live inputs</strong>
      </div>
      <div className="source-grid">
        <LivePreviewPanel compact runtimeState={runtimeState} title="RGB source preview" />
        <div className="source-state-card">
          <span>3D/source state</span>
          <strong>{runtimeState?.acquisition_connected ? "CONNECTED" : "DISCONNECTED"}</strong>
          <small>Runtime source: {runtimeState?.acquisition_source ?? "-"}</small>
          <small>Take source: {sourceSummary(detail)}</small>
          <small>Modalities: {modalities}</small>
          <small>Camera index: {typeof sourceDetails.camera_index === "number" ? sourceDetails.camera_index : "-"}</small>
          <small>Latest frame: {runtimeState?.latest_frame_timestamp ?? "-"}</small>
        </div>
      </div>
    </section>
  );
}
