import { useCallback, useEffect, useState } from "react";
import { api, type PipelineInfo, type RuntimeState, type SessionSummary, type TakeDetail, type TakeSummary } from "../api/client";
import { connectEvents } from "../api/events";
import AcquisitionSourcesPanel from "../components/AcquisitionSourcesPanel";
import PipelineStageExplorer from "../components/PipelineStageExplorer";
import PipelineStatusPanel from "../components/PipelineStatusPanel";
import StatusBadge from "../components/StatusBadge";
import TakeDetailView from "./TakeDetailView";

type SortMode = "newest" | "slowest" | "warnings";
type FilterMode = "all" | "failed" | "warnings";

export default function DebugPage() {
  const [takes, setTakes] = useState<TakeSummary[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<TakeDetail | null>(null);
  const [live, setLive] = useState(true);
  const [sortMode, setSortMode] = useState<SortMode>("newest");
  const [filterMode, setFilterMode] = useState<FilterMode>("all");
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [selectedSession, setSelectedSession] = useState<string>("");
  const [runtimeState, setRuntimeState] = useState<RuntimeState | null>(null);
  const [pipelines, setPipelines] = useState<PipelineInfo[]>([]);

  const loadList = useCallback(async () => {
    const [nextTakes, nextSessions, nextRuntimeState, nextPipelines] = await Promise.all([
      api.takes(selectedSession || undefined),
      api.sessions(),
      api.state(),
      api.pipelines()
    ]);
    setTakes(nextTakes);
    setSessions(nextSessions);
    setRuntimeState(nextRuntimeState);
    setPipelines(nextPipelines);
    setSelectedId((current) => current ?? nextTakes[0]?.take_id ?? null);
  }, [selectedSession]);

  const loadDetail = useCallback(async (takeId: string | null) => {
    if (!takeId) {
      setDetail(null);
      return;
    }
    setDetail(await api.take(takeId));
  }, []);

  useEffect(() => {
    void loadList();
    const source = connectEvents(
      () => {
        setLive(true);
        void loadList();
        void loadDetail(selectedId);
      },
      () => setLive(false)
    );
    return () => source?.close();
  }, [loadDetail, loadList, selectedId]);

  useEffect(() => {
    void loadDetail(selectedId);
  }, [loadDetail, selectedId]);

  useEffect(() => {
    if (live) {
      return;
    }
    const timer = window.setInterval(() => {
      void loadList();
      void loadDetail(selectedId);
    }, 2000);
    return () => window.clearInterval(timer);
  }, [live, loadDetail, loadList, selectedId]);

  const visibleTakes = takes
    .filter((take) => {
      if (filterMode === "failed") {
        return take.status === "failed";
      }
      if (filterMode === "warnings") {
        return (take.warning_count ?? 0) > 0;
      }
      return true;
    })
    .sort((a, b) => {
      if (sortMode === "slowest") {
        return (b.processing_time_ms ?? -1) - (a.processing_time_ms ?? -1);
      }
      if (sortMode === "warnings") {
        return (b.warning_count ?? 0) - (a.warning_count ?? 0);
      }
      return String(b.created_at ?? b.take_id).localeCompare(String(a.created_at ?? a.take_id));
    });
  const processingFps =
    runtimeState && runtimeState.throughput && typeof runtimeState.throughput["processing_fps"] === "number"
      ? runtimeState.throughput["processing_fps"]
      : null;
  const sourceDetails = runtimeState?.acquisition_source_details ?? {};

  return (
    <main className="debug-page">
      <section className="debug-sidebar">
        <div className="sidebar-title">
          <span>Takes</span>
          <StatusBadge value={live ? "live" : "polling"} />
        </div>
        <div className="review-controls">
          <select value={selectedSession} onChange={(event) => setSelectedSession(event.target.value)}>
            <option value="">All sessions</option>
            {sessions.map((session) => (
              <option key={session.session_id} value={session.session_id}>
                {session.session_id}
              </option>
            ))}
          </select>
          <select value={sortMode} onChange={(event) => setSortMode(event.target.value as SortMode)}>
            <option value="newest">Newest</option>
            <option value="slowest">Slowest</option>
            <option value="warnings">Most warnings</option>
          </select>
          <select value={filterMode} onChange={(event) => setFilterMode(event.target.value as FilterMode)}>
            <option value="all">All takes</option>
            <option value="failed">Failed only</option>
            <option value="warnings">Warnings only</option>
          </select>
        </div>
        <div className="take-list">
          {visibleTakes.map((take) => (
            <button
              className={take.take_id === selectedId ? "take-row active" : "take-row"}
              key={take.take_id}
              onClick={() => setSelectedId(take.take_id)}
              type="button"
            >
              <span className="take-row-main">
                <strong>{take.take_id}</strong>
                <small>
                  {take.engine ?? "-"} · {take.calibration_resolution_source ?? "none"} · {take.calibration_id ?? "no calibration"} · {take.object_count ?? "-"} obj · {take.ball_count ?? "-"} balls
                </small>
              </span>
              <span className="take-row-meta">
                <StatusBadge value={take.status} />
                <small>{take.processing_time_ms == null ? "-" : `${take.processing_time_ms.toFixed(0)} ms`}</small>
                {(take.warning_count ?? 0) > 0 && <small className="warning-count">{take.warning_count} warnings</small>}
              </span>
            </button>
          ))}
        </div>
      </section>
      <section className="debug-detail">
        <section className="page-title-block">
          <div className="eyebrow">Diagnostics</div>
          <h1>Runtime health console</h1>
          <p>Investigate queue health, FPS, dropped frames, profiling, payloads, stale previews, warnings, and errors.</p>
        </section>
        <section className="metric-strip">
          <div>
            <span>Acquisition</span>
            <strong>{runtimeState?.acquisition_connected ? "CONNECTED" : "DISCONNECTED"}</strong>
          </div>
          <div>
            <span>Source</span>
            <strong>{runtimeState?.acquisition_source ?? "-"}</strong>
          </div>
          <div>
            <span>Camera</span>
            <strong>{typeof sourceDetails.camera_index === "number" ? sourceDetails.camera_index : "-"}</strong>
          </div>
          <div>
            <span>Queue</span>
            <strong>{runtimeState?.queue_size ?? "-"}</strong>
          </div>
          <div>
            <span>Lag</span>
            <strong>{runtimeState?.processing_lag_ms == null ? "-" : `${runtimeState.processing_lag_ms.toFixed(0)} ms`}</strong>
          </div>
          <div>
            <span>Throughput</span>
            <strong>{processingFps == null ? "-" : `${processingFps.toFixed(2)} fps`}</strong>
          </div>
        </section>
        {!!runtimeState?.warnings?.length && (
          <section className="mode-banner">
            <span>{runtimeState.warnings.join(" | ")}</span>
          </section>
        )}
        <AcquisitionSourcesPanel compact runtimeState={runtimeState} detail={detail} />
        <PipelineStatusPanel compact result={detail?.result ?? null} runtimeState={runtimeState} detail={detail} pipelines={pipelines} />
        <PipelineStageExplorer
          pipeline={pipelines.find((pipeline) => pipeline.id === (detail?.result?.processing_pipeline?.id ?? "3d_ball_inspection")) ?? pipelines[0] ?? null}
          result={detail?.result ?? null}
          takeId={detail?.take_id}
        />
        <section className="debug-preview-grid">
          <div className="preview-diagnostics-card">
            <span>Preview diagnostics</span>
            <strong>{runtimeState?.preview_available ? "AVAILABLE" : "UNAVAILABLE"}</strong>
            <small>Timestamp: {runtimeState?.preview_timestamp ?? "-"}</small>
            <small>FPS: {runtimeState?.preview_fps_estimate == null ? "-" : runtimeState.preview_fps_estimate.toFixed(2)}</small>
            <small>Stale: {runtimeState?.preview_stale ? "yes" : "no"}</small>
            <small>Path: {runtimeState?.preview_path ?? "-"}</small>
          </div>
        </section>
        <section className="section-group">
          <div className="section-heading">
            <h3>Artifacts and result JSON</h3>
            <span className="section-note">Raw inputs, processed artifacts, profiling, and full payload.</span>
          </div>
        </section>
        <TakeDetailView detail={detail} />
      </section>
    </main>
  );
}
