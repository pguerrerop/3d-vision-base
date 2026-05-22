import { fileUrl, type TakeDetail } from "../api/client";
import CalibrationStatusBadge from "../components/CalibrationStatusBadge";
import ImagePanel from "../components/ImagePanel";
import InputAssetsPanel from "../components/InputAssetsPanel";
import ObjectTable from "../components/ObjectTable";
import PocSummaryPanel from "../components/PocSummaryPanel";
import ProfilingBreakdown from "../components/ProfilingBreakdown";
import ProcessingModeBadge from "../components/ProcessingModeBadge";
import StatusBadge from "../components/StatusBadge";

type Props = {
  detail: TakeDetail | null;
};

function JsonBlock({ value }: { value: unknown }) {
  return <pre className="json-block">{JSON.stringify(value ?? {}, null, 2)}</pre>;
}

export default function TakeDetailView({ detail }: Props) {
  if (!detail) {
    return <div className="empty-state">Select a take to inspect.</div>;
  }

  const result = detail.result;
  const image = (filename: string | null | undefined) => (filename ? fileUrl(detail.take_id, filename) : null);
  const waitingMessage = detail.has_ready && !detail.has_done ? "This take has not been processed yet. Waiting for processing." : "No processed output available.";

  return (
    <div className="detail-view">
      <div className="detail-header">
        <div>
          <div className="eyebrow">Take detail</div>
          <h2>{detail.take_id}</h2>
        </div>
        <div className="badge-row">
          <StatusBadge value={detail.status} />
          <ProcessingModeBadge mode={result?.processing_mode} />
          <CalibrationStatusBadge result={result} />
          <span className="calibration-badge">Calibration: {result?.poc_summary?.calibration.display ?? "NONE"}</span>
        </div>
      </div>

      {result?.processing_mode === "mock" && (
        <section className="mode-banner compact">
          <ProcessingModeBadge mode="mock" />
          <span>Showing mock/demo processing results. Detections are fake placeholders until the real algorithm exists.</span>
        </section>
      )}
      {result?.processing_mode === "real" && (
        <section className="mode-banner compact real">
          <ProcessingModeBadge mode="real" />
          <span>
            {result.algorithm_stage === "segmentation"
              ? "Real segmentation output. Objects are geometric clusters with unknown class until classification is implemented."
              : "Real processing output."}
          </span>
        </section>
      )}

      <PocSummaryPanel summary={result?.poc_summary} />
      <section className="metric-strip">
        <div>
          <span>Session</span>
          <strong>{detail.session_id ?? result?.session_id ?? "-"}</strong>
        </div>
        <div>
          <span>FrameSet</span>
          <strong>{detail.frameset_id ?? result?.frameset_id ?? "-"}</strong>
        </div>
        <div>
          <span>Sync mode</span>
          <strong>{String((detail.frameset?.synchronization as { mode?: string } | undefined)?.mode ?? "none").toUpperCase()}</strong>
        </div>
        <div>
          <span>Sync confidence</span>
          <strong>
            {typeof (detail.frameset?.synchronization as { confidence?: number } | undefined)?.confidence === "number"
              ? `${(((detail.frameset?.synchronization as { confidence?: number }).confidence ?? 0) * 100).toFixed(0)}%`
              : "-"}
          </strong>
        </div>
      </section>

      <section className="section-group">
        <div className="section-heading">
          <h3>Raw input</h3>
        </div>
        <InputAssetsPanel detail={detail} result={result} />
      </section>

      <section className="section">
        <h3>Metadata</h3>
        <JsonBlock value={detail.metadata} />
      </section>

      <section className="section">
        <h3>Acquisition processing status</h3>
        <JsonBlock value={detail.acquisition_processing_status ?? { status: "acquired" }} />
      </section>

      <section className="section-group">
        <div className="section-heading">
          <h3>Processed outputs</h3>
          <ProcessingModeBadge mode={result?.processing_mode} />
        </div>
        {result?.algorithm_stage === "calibrated_segmentation" ? (
          <section className="image-grid calibrated-grid">
            <ImagePanel title="Calibrated planes" src={image(result.files.debug_calibrated_planes)} emptyMessage={waitingMessage} />
            <ImagePanel title="Belt polygon top view" src={image(result.files.debug_belt_polygon_topview)} emptyMessage={waitingMessage} />
            <ImagePanel title="Filtered foreground" src={image(result.files.debug_filtered_foreground)} emptyMessage={waitingMessage} />
            <ImagePanel title="Rejected points" src={image(result.files.debug_rejected_points)} emptyMessage={waitingMessage} />
            <ImagePanel title="Filtered clusters" src={image(result.files.debug_clusters_filtered)} emptyMessage={waitingMessage} />
          </section>
        ) : (
          <section className="image-grid">
            <ImagePanel
              title={result?.processing_mode === "mock" ? "Demonstration overlay" : "Plane segmentation"}
              src={image(result?.files.overlay ?? result?.files.debug_plane_segmentation)}
              emptyMessage={waitingMessage}
            />
            <ImagePanel
              title={result?.processing_mode === "real" ? "Foreground extraction" : "Debug height"}
              src={image(result?.files.debug_foreground ?? result?.files.debug_height)}
              emptyMessage={waitingMessage}
            />
            <ImagePanel
              title={result?.processing_mode === "real" ? "Clusters" : "Debug segmentation"}
              src={image(result?.files.debug_clusters ?? result?.files.debug_segmentation)}
              emptyMessage={waitingMessage}
            />
          </section>
        )}
      </section>

      {result?.plane_filtering && (
        <section className="section">
          <h3>Calibrated filtering</h3>
          <div className="filtering-stats-grid">
            <div><span>Input points</span><strong>{result.plane_filtering.input_points.toLocaleString()}</strong></div>
            <div><span>Ignored plane points</span><strong>{result.plane_filtering.ignored_plane_points.toLocaleString()}</strong></div>
            <div><span>Candidate foreground points</span><strong>{result.plane_filtering.candidate_foreground_points.toLocaleString()}</strong></div>
            <div><span>Rejected points</span><strong>{result.plane_filtering.rejected_points.toLocaleString()}</strong></div>
            <div><span>Kept objects</span><strong>{result.plane_filtering.kept_objects.toLocaleString()}</strong></div>
            <div><span>Rejected objects</span><strong>{result.plane_filtering.rejected_objects.toLocaleString()}</strong></div>
          </div>
          <div className="plane-model">
            Belt: {result.plane_filtering.belt_plane_id}; ignored: {result.plane_filtering.ignored_plane_ids.join(", ") || "none"}
          </div>
        </section>
      )}

      <section className="section">
        <h3>Objects</h3>
        {result?.processing_mode === "mock" && <p className="section-note">Mock objects are synthetic placeholders for UI testing.</p>}
        {result?.processing_mode === "real" && result.algorithm_stage === "segmentation" && (
          <p className="section-note">Real clusters from foreground geometry. Class is intentionally unknown.</p>
        )}
        <ObjectTable objects={result?.objects ?? []} />
      </section>

      {(result?.rejected_objects?.length ?? 0) > 0 && (
        <section className="section">
          <h3>Rejected clusters</h3>
          <ObjectTable objects={result?.rejected_objects ?? []} />
        </section>
      )}

      {result?.plane_model && (
        <section className="section">
          <h3>Plane model</h3>
          <div className="plane-model">{result.plane_model.map((value) => value.toFixed(5)).join(", ")}</div>
        </section>
      )}

      <ProfilingBreakdown result={result} />

      <section className="section">
        <h3>Result JSON</h3>
        <JsonBlock value={result} />
      </section>
    </div>
  );
}
