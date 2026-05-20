import { useCallback, useEffect, useMemo, useState } from "react";
import {
  api,
  fileUrl,
  type ActiveCalibration,
  type PipelineInfo,
  type ProcessingResult,
  type RuntimeState,
  type SessionSummary,
  type TakeDetail,
  type TakeSummary
} from "../api/client";
import { connectEvents } from "../api/events";
import CalibrationStatusBadge from "../components/CalibrationStatusBadge";
import CalibrationStateBadge from "../components/CalibrationStateBadge";
import AcquisitionSourcesPanel from "../components/AcquisitionSourcesPanel";
import DecisionCard from "../components/DecisionCard";
import ImagePanel from "../components/ImagePanel";
import PipelineStatusPanel from "../components/PipelineStatusPanel";
import PocSummaryPanel from "../components/PocSummaryPanel";
import ProcessingModeBadge from "../components/ProcessingModeBadge";
import StatusBadge from "../components/StatusBadge";

export default function OperatorPage() {
  const [latest, setLatest] = useState<TakeSummary | null>(null);
  const [detail, setDetail] = useState<TakeDetail | null>(null);
  const [state, setState] = useState<RuntimeState | null>(null);
  const [activeCalibration, setActiveCalibration] = useState<ActiveCalibration | null>(null);
  const [live, setLive] = useState(true);
  const [demoMode, setDemoMode] = useState(false);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [pipelines, setPipelines] = useState<PipelineInfo[]>([]);
  const [selectedSession, setSelectedSession] = useState<string>("");

  const load = useCallback(async () => {
    const [nextState, nextSessions, nextPipelines] = await Promise.all([api.state(), api.sessions(), api.pipelines()]);
    setSessions(nextSessions);
    setPipelines(nextPipelines);
    const nextLatest = selectedSession ? (await api.takes(selectedSession))[0] ?? null : await api.latest();
    setLatest(nextLatest);
    setState(nextState);
    if (nextLatest?.take_id) {
      const [nextDetail, nextCalibration] = await Promise.all([
        api.take(nextLatest.take_id),
        api.activeCalibration(nextLatest.take_id)
      ]);
      setDetail(nextDetail);
      setActiveCalibration(nextCalibration);
    } else {
      setDetail(null);
      setActiveCalibration(await api.activeCalibration());
    }
  }, [selectedSession]);

  useEffect(() => {
    void load();
    const source = connectEvents(
      () => {
        setLive(true);
        void load();
      },
      () => setLive(false)
    );
    return () => source?.close();
  }, [load]);

  useEffect(() => {
    if (live) {
      return;
    }
    const timer = window.setInterval(() => void load(), 2000);
    return () => window.clearInterval(timer);
  }, [live, load]);

  const result: ProcessingResult | null = detail?.result ?? null;
  const productionMs = result?.profiling?.production_ms ?? null;
  const totalMs = result?.profiling?.total_ms ?? result?.timing_ms.total ?? null;
  const processingMode = result?.profiling?.stages.some((stage) => stage.notes === "skipped by --skip-debug-images")
    ? "PRODUCTION"
    : result?.processing_mode
      ? result.processing_mode.toUpperCase()
      : "-";
  const overlay = useMemo(() => {
    const filename = result?.files.overlay ?? result?.files.debug_clusters ?? result?.files.debug_foreground ?? null;
    if (!detail || !filename) {
      return null;
    }
    return fileUrl(detail.take_id, filename);
  }, [detail, result]);
  const sourceDetails = state?.acquisition_source_details ?? {};

  return (
    <main className={demoMode ? "operator-page demo-mode" : "operator-page"}>
      <section className="page-title-block">
        <div className="eyebrow">Operations</div>
        <h1>Production HMI</h1>
        <p>Machine state, live acquisition, active pipeline health, latest decision, production KPIs, and alarms.</p>
        <label className="demo-toggle">
          <input checked={demoMode} onChange={(event) => setDemoMode(event.target.checked)} type="checkbox" />
          Demo mode
        </label>
      </section>
      {result?.processing_mode === "mock" && (
        <section className="mode-banner">
          <ProcessingModeBadge mode="mock" />
          <span>Demo visualization is active. Use this view for layout validation, not production decisions.</span>
        </section>
      )}
      {result?.processing_mode === "real" && result.algorithm_stage === "segmentation" && (
        <section className="mode-banner real">
          <ProcessingModeBadge mode="real" />
          <span>Geometry inspection is active. Final object classification is still pending.</span>
        </section>
      )}
      {result && !result.calibration_active && (
        <section className="mode-banner">
          <span>No calibration selected. Automatic plane estimation only.</span>
        </section>
      )}
      <section className="operator-status">
        <div>
          <div className="eyebrow">System status</div>
          <h1>{state?.status ?? "idle"}</h1>
          <p>{state?.message ?? "Waiting for acquisition and processed output."}</p>
        </div>
        <div className="badge-row">
          <CalibrationStatusBadge result={result} />
          <CalibrationStateBadge active={activeCalibration} result={result} />
          <StatusBadge value={state?.acquisition_connected ? "connected" : "disconnected"} />
          <StatusBadge value={state?.stale ? "stale" : "fresh"} />
          <StatusBadge value={live ? "live" : "polling"} />
        </div>
      </section>
      <section className="metric-strip">
        <div>
          <span>Session</span>
          <strong>{state?.current_session ?? "-"}</strong>
        </div>
        <div>
          <span>Queue size</span>
          <strong>{state?.queue_size ?? "-"}</strong>
        </div>
        <div>
          <span>Dropped frames</span>
          <strong>{state?.dropped_frames ?? "-"}</strong>
        </div>
        <div>
          <span>Processing lag</span>
          <strong>{state?.processing_lag_ms == null ? "-" : `${state.processing_lag_ms.toFixed(0)} ms`}</strong>
        </div>
      </section>
      <section className="latest-meta">
        <div>
          <span>Session filter</span>
          <strong>
            <select value={selectedSession} onChange={(event) => setSelectedSession(event.target.value)}>
              <option value="">All sessions</option>
              {sessions.map((session) => (
                <option key={session.session_id} value={session.session_id}>
                  {session.session_id} ({session.take_count} takes)
                </option>
              ))}
            </select>
          </strong>
        </div>
        <div>
          <span>Acquisition source</span>
          <strong>{state?.acquisition_source ?? "-"}</strong>
        </div>
        <div>
          <span>Camera index</span>
          <strong>{typeof sourceDetails.camera_index === "number" ? sourceDetails.camera_index : "-"}</strong>
        </div>
        <div>
          <span>Active calibration</span>
          <strong>{state?.active_calibration ?? result?.calibration_id ?? "-"}</strong>
        </div>
      </section>
      {!!state?.warnings?.length && (
        <section className="mode-banner">
          <span>{state.warnings.join(" | ")}</span>
        </section>
      )}

      <AcquisitionSourcesPanel runtimeState={state} detail={detail} />

      <PipelineStatusPanel result={result} runtimeState={state} detail={detail} pipelines={pipelines} />

      <section className="section-group">
        <div className="section-heading">
          <h3>Latest inspection result</h3>
          <span className="section-note">Processed pipeline output, separate from raw/live acquisition.</span>
        </div>
        <div className="operator-grid">
          <DecisionCard result={result} />
          <ImagePanel
            title={result?.processing_mode === "mock" ? "Demonstration pipeline artifact" : "Latest processed artifact"}
            src={overlay}
            emptyMessage={detail?.has_ready && !detail?.has_done ? "This take has not been processed yet. Waiting for processing." : "No processed image available."}
          />
        </div>
      </section>

      <PocSummaryPanel compact summary={result?.poc_summary} />

      <section className="metric-strip">
        <div>
          <span>Production time</span>
          <strong>{productionMs == null ? "-" : `${productionMs.toFixed(productionMs >= 100 ? 0 : 1)} ms`}</strong>
        </div>
        <div>
          <span>Total time</span>
          <strong>{totalMs == null ? "-" : `${totalMs.toFixed(totalMs >= 100 ? 0 : 1)} ms`}</strong>
        </div>
        <div>
          <span>Processing mode</span>
          <strong>{processingMode}</strong>
        </div>
        <div>
          <span>Calibration</span>
          <strong>{result?.poc_summary?.calibration.display ?? "NONE"}</strong>
        </div>
      </section>

      <section className="latest-meta">
        <div>
          <span>Latest take</span>
          <strong>{latest?.take_id ?? "-"}</strong>
        </div>
        <div>
          <span>Last update</span>
          <strong>{result?.processed_at ?? state?.updated_at ?? "-"}</strong>
        </div>
        <div>
          <span>Result note</span>
          <strong>
            {result?.processing_mode === "mock"
              ? "Showing mock/demo processing results."
              : result?.algorithm_stage === "segmentation"
                ? "Segmentation pipeline active."
                : result
                  ? "Real processing output."
                  : "Waiting for processing."}
          </strong>
        </div>
      </section>
    </main>
  );
}
