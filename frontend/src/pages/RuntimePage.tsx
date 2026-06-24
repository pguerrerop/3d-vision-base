import { useCallback, useEffect, useMemo, useState } from "react";
import {
  api,
  type ActiveCalibration,
  type DatasetSessionSummary,
  type DatasetSummary,
  type ReplaySessionState,
  type RuntimeConfigResponse,
  type RuntimeProcessSummary,
  type RuntimeRoutingState,
  type RuntimeState,
  type RuntimeWorkerStatus,
  type SourceInfo,
} from "../api/client";

function isoAgeLabel(ts: string | null | undefined): string {
  if (!ts) return "-";
  const ms = Date.now() - new Date(ts).getTime();
  if (!Number.isFinite(ms) || ms < 0) return "-";
  const sec = Math.floor(ms / 1000);
  if (sec < 60) return `${sec}s ago`;
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min}m ago`;
  return `${Math.floor(min / 60)}h ago`;
}

function sourceFreshness(source: SourceInfo): "Fresh" | "Delayed" | "Stale" | "Offline" {
  if (source.status === "unavailable") return "Offline";
  if (source.last_frame_at == null) return "Delayed";
  if ((source.last_frame_age_seconds ?? 9999) > 5 || source.status === "stale") return "Stale";
  return "Fresh";
}

function serviceBadge(status: string): "Running" | "Stale" | "Stopped" | "Error" | "Unknown" {
  const s = String(status).toLowerCase();
  if (s.includes("running")) return "Running";
  if (s.includes("stale")) return "Stale";
  if (s.includes("stopped")) return "Stopped";
  if (s.includes("error") || s.includes("failed")) return "Error";
  return "Unknown";
}

function routingBadge(mode: string): string {
  if (mode === "replay_override") return "REPLAY OVERRIDE";
  if (mode === "explicit_acquisition_destination") return "EXPLICIT ACQUISITION DESTINATION";
  if (mode === "persistent_default") return "PERSISTENT DEFAULT";
  if (mode === "metadata_driven") return "METADATA-DRIVEN";
  if (mode === "auto_created") return "AUTO SESSION";
  return "UNASSIGNED";
}

export default function RuntimePage() {
  const [runtimeState, setRuntimeState] = useState<RuntimeState | null>(null);
  const [runtimeConfig, setRuntimeConfig] = useState<RuntimeConfigResponse | null>(null);
  const [configDraft, setConfigDraft] = useState<RuntimeConfigResponse["config"] | null>(null);
  const [routing, setRouting] = useState<RuntimeRoutingState | null>(null);

  const [replay, setReplay] = useState<ReplaySessionState | null>(null);
  const [replayDraft, setReplayDraft] = useState({
    dataset_id: "",
    session_id: "",
    replay_mode: "routing_override",
    persist_on_restart: false,
    auto_start_on_startup: false,
    replay_source: "offline_dataset_replay",
  });

  const [destinationDraft, setDestinationDraft] = useState({
    dataset_id: "",
    session_id: "",
    session_naming_mode: "explicit_session",
    auto_create_session: false,
    activate_immediately: true,
    persist_as_default: true,
  });

  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [datasetSessions, setDatasetSessions] = useState<Record<string, DatasetSessionSummary[]>>({});

  const [sources, setSources] = useState<SourceInfo[]>([]);
  const [services, setServices] = useState<RuntimeProcessSummary[]>([]);
  const [workers, setWorkers] = useState<RuntimeWorkerStatus[]>([]);
  const [activeCalibration, setActiveCalibration] = useState<ActiveCalibration | null>(null);

  const [newDatasetName, setNewDatasetName] = useState("");
  const [newDatasetNotes, setNewDatasetNotes] = useState("");
  const [newSessionName, setNewSessionName] = useState("");
  const [newSessionNotes, setNewSessionNotes] = useState("");

  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const sessionOptions = useMemo(() => datasetSessions[destinationDraft.dataset_id] ?? [], [datasetSessions, destinationDraft.dataset_id]);
  const replaySessionOptions = useMemo(() => datasetSessions[replayDraft.dataset_id] ?? [], [datasetSessions, replayDraft.dataset_id]);

  const loadAll = useCallback(async () => {
    const [state, config, replayPayload, ds, src, procs, workerPayload, cal, routingPayload] = await Promise.all([
      api.state(),
      api.runtimeConfig(),
      api.activeReplaySession(),
      api.datasets(),
      api.sources(),
      api.runtimeProcesses(),
      api.runtimeWorkers(),
      api.activeCalibration().catch(() => null),
      api.runtimeRouting(),
    ]);

    const sessionEntries = await Promise.all(ds.map(async (d) => [d.id, await api.datasetSessions(d.id)] as const));
    const sessionsByDataset = Object.fromEntries(sessionEntries);

    setRuntimeState(state);
    setRuntimeConfig(config);
    setConfigDraft(config.config);
    setReplay(replayPayload.active_session);
    setDatasets(ds);
    setDatasetSessions(sessionsByDataset);
    setSources(src);
    setServices(procs);
    setWorkers(workerPayload.workers ?? []);
    setActiveCalibration(cal);
    setRouting(routingPayload);

    const replayState = replayPayload.active_session;
    if (replayState) {
      setReplayDraft({
        dataset_id: String(replayState.dataset_id ?? ""),
        session_id: String(replayState.session_id ?? ""),
        replay_mode: String(replayState.replay_mode ?? "routing_override"),
        persist_on_restart: Boolean(replayState.persist_on_restart ?? replayState.persisted_on_restart ?? false),
        auto_start_on_startup: Boolean(replayState.auto_start_on_startup ?? false),
        replay_source: String(replayState.replay_source ?? "offline_dataset_replay"),
      });
    }

    const live = routingPayload.live_routing ?? {};
    const persisted = routingPayload.persistent_default ?? {};
    setDestinationDraft((prev) => ({
      ...prev,
      dataset_id: String((live as Record<string, unknown>).active_dataset_id ?? (persisted as Record<string, unknown>).default_dataset_id ?? prev.dataset_id),
      session_id: String((live as Record<string, unknown>).active_session_id ?? (persisted as Record<string, unknown>).default_session_id ?? prev.session_id),
      auto_create_session: Boolean((persisted as Record<string, unknown>).auto_create_session ?? prev.auto_create_session),
      session_naming_mode: String((persisted as Record<string, unknown>).session_naming_policy ?? prev.session_naming_mode),
    }));
  }, []);

  useEffect(() => {
    let mounted = true;
    const run = async () => {
      setLoading(true);
      try {
        await loadAll();
        if (mounted) setError(null);
      } catch (exc) {
        if (mounted) setError(String(exc));
      } finally {
        if (mounted) setLoading(false);
      }
    };
    void run();
    const id = window.setInterval(() => { void loadAll(); }, 4000);
    return () => { mounted = false; window.clearInterval(id); };
  }, [loadAll]);

  const runningServices = services.filter((s) => serviceBadge(s.status) === "Running").length;
  const staleServices = services.filter((s) => serviceBadge(s.status) === "Stale").length;
  const failedServices = services.filter((s) => serviceBadge(s.status) === "Error").length;
  const staleSources = sources.filter((s) => ["Stale", "Offline"].includes(sourceFreshness(s))).length;

  const resolved = routing?.resolved;
  const nextDataset = resolved?.active_dataset_id ?? "-";
  const nextSession = resolved?.active_session_id ?? "-";
  const routingMode = resolved?.routing_mode ?? "unassigned";

  async function saveConfig() {
    if (!configDraft) return;
    setBusy(true);
    try {
      const payload = await api.updateRuntimeConfig(configDraft);
      setRuntimeConfig(payload);
      setConfigDraft(payload.config);
      setError(null);
    } catch (exc) {
      setError(String(exc));
    } finally {
      setBusy(false);
    }
  }

  async function createDatasetInline() {
    if (!newDatasetName.trim()) return;
    setBusy(true);
    try {
      const created = await api.createDataset({ name: newDatasetName.trim(), notes: newDatasetNotes || undefined });
      setNewDatasetName("");
      setNewDatasetNotes("");
      await loadAll();
      setDestinationDraft((p) => ({ ...p, dataset_id: created.id, session_id: "" }));
    } catch (exc) {
      setError(String(exc));
    } finally {
      setBusy(false);
    }
  }

  async function createSessionInline() {
    if (!destinationDraft.dataset_id || !newSessionName.trim()) return;
    setBusy(true);
    try {
      const created = await api.createDatasetSession(destinationDraft.dataset_id, { name: newSessionName.trim(), notes: newSessionNotes || undefined });
      setNewSessionName("");
      setNewSessionNotes("");
      await loadAll();
      setDestinationDraft((p) => ({ ...p, session_id: created.id }));
    } catch (exc) {
      setError(String(exc));
    } finally {
      setBusy(false);
    }
  }

  async function applyDefaultDestination() {
    if (!destinationDraft.dataset_id || !destinationDraft.session_id) return;
    setBusy(true);
    try {
      if (destinationDraft.activate_immediately) {
        await api.activateRuntimeRouting({
          dataset_id: destinationDraft.dataset_id,
          session_id: destinationDraft.session_id,
          routing_mode: "explicit_acquisition_destination",
          source: "runtime_ui",
          updated_by: "runtime_ui",
        });
      }
      if (destinationDraft.persist_as_default) {
        await api.updateRuntimeDefaultDestination({
          default_dataset_id: destinationDraft.dataset_id,
          default_session_id: destinationDraft.session_id,
          auto_create_session: destinationDraft.auto_create_session,
          session_naming_policy: destinationDraft.session_naming_mode,
          updated_by: "runtime_ui",
        });
      }
      await loadAll();
    } catch (exc) {
      setError(String(exc));
    } finally {
      setBusy(false);
    }
  }

  async function clearDefaultDestination() {
    setBusy(true);
    try {
      await api.clearRuntimeDefaultDestination();
      await api.clearRuntimeRouting();
      await loadAll();
    } catch (exc) {
      setError(String(exc));
    } finally {
      setBusy(false);
    }
  }

  async function startReplay() {
    if (!replayDraft.dataset_id || !replayDraft.session_id) return;
    setBusy(true);
    try {
      await api.runtimeReplayStart({
        dataset_id: replayDraft.dataset_id,
        session_id: replayDraft.session_id,
        replay_mode: replayDraft.replay_mode,
        persist_on_restart: replayDraft.persist_on_restart,
        auto_start_on_startup: replayDraft.auto_start_on_startup,
        replay_source: replayDraft.replay_source,
        metadata: { created_by: "runtime_ui" },
      });
      await loadAll();
    } catch (exc) {
      setError(String(exc));
    } finally {
      setBusy(false);
    }
  }

  async function stopReplay() { setBusy(true); try { await api.runtimeReplayStop(); await loadAll(); } catch (exc) { setError(String(exc)); } finally { setBusy(false); } }
  async function pauseReplay() { setBusy(true); try { await api.runtimeReplayPause(); await loadAll(); } catch (exc) { setError(String(exc)); } finally { setBusy(false); } }
  async function clearReplay() { setBusy(true); try { await api.runtimeReplayClear(); await loadAll(); } catch (exc) { setError(String(exc)); } finally { setBusy(false); } }
  async function saveReplayConfig() {
    setBusy(true);
    try {
      await api.updateRuntimeReplayConfig(replayDraft);
      await loadAll();
    } catch (exc) {
      setError(String(exc));
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="runtime-page">
      <section className="page-title-block">
        <h2>Runtime</h2>
        <p>Engineering supervision and runtime-control workspace for acquisition, routing, processing, and system services.</p>
      </section>
      {error && <div className="warning-line active">{error}</div>}

      <section className="runtime-workspace">
        <aside className="runtime-left-rail">
          <article className="section">
            <div className="section-heading"><h3>Replay Status</h3><small className="panel-title">Live</small></div>
            <div className="pipeline-status-grid">
              <div><span>Status</span><strong>{replay?.active ? "ACTIVE" : replay ? "INACTIVE" : "NONE"}</strong></div>
              <div><span>Dataset</span><strong>{replay?.dataset_id ?? "-"}</strong></div>
              <div><span>Session</span><strong>{replay?.session_id ?? "-"}</strong></div>
            </div>
          </article>

          <article className="section">
            <div className="section-heading"><h3>Sources</h3><small className="panel-title">Live</small></div>
            <div className="runtime-compact-list">
              {sources.map((s) => (
                <div key={s.id} className="runtime-compact-item">
                  <strong>{s.label}</strong>
                  <span>{sourceFreshness(s)}</span>
                  <small>{s.fps ?? "-"} FPS • {Array.isArray(s.resolution) ? s.resolution.join("x") : "-"}</small>
                  <small>Last frame: {isoAgeLabel(s.last_frame_at)}</small>
                </div>
              ))}
            </div>
          </article>

          <article className="section">
            <div className="section-heading"><h3>Runtime Health</h3><small className="panel-title">Computed</small></div>
            <div className="pipeline-status-grid">
              <div><span>Services running</span><strong>{runningServices}</strong></div>
              <div><span>Stale services</span><strong>{staleServices}</strong></div>
              <div><span>Critical failures</span><strong>{failedServices}</strong></div>
              <div><span>Stale sources</span><strong>{staleSources}</strong></div>
              <div><span>Active workers</span><strong>{workers.filter((w) => w.state === "running").length}</strong></div>
              <div><span>Queue warnings</span><strong>{(runtimeState?.queue_size ?? 0) > 0 ? "yes" : "no"}</strong></div>
            </div>
          </article>
        </aside>

        <section className="runtime-main-workspace">
          <article className="section runtime-hero">
            <div className="section-heading"><h3>Acquisition Routing</h3><small className="panel-title">Live + Computed</small></div>
            <div className="runtime-routing-destination">
              <p>Current active destination</p>
              <h2>Dataset: {nextDataset}</h2>
              <h2>Session: {nextSession}</h2>
              <div className="mode-banner compact real"><strong>{routingBadge(routingMode)}</strong></div>
            </div>
            <p className="section-note">{resolved?.ownership_explanation ?? "No routing owner resolved."}</p>
            <div className="runtime-precedence-list">
              <h3>Routing precedence</h3>
              <div><span>1. Explicit capture request</span><strong>inactive</strong></div>
              <div><span>2. Replay session</span><strong>{routingMode === "replay_override" ? "ACTIVE" : "inactive"}</strong></div>
              <div><span>3. Acquisition metadata</span><strong>{routingMode === "metadata_driven" ? "ACTIVE" : "inactive"}</strong></div>
              <div><span>4. Persistent defaults</span><strong>{routingMode === "persistent_default" ? "ACTIVE" : "inactive"}</strong></div>
              <div><span>5. Auto-created session</span><strong>{routingMode === "auto_created" ? "ACTIVE" : "inactive"}</strong></div>
              <div><span>6. Unassigned</span><strong>{routingMode === "unassigned" ? "ACTIVE" : "inactive"}</strong></div>
            </div>
            {replay?.active && <div className="warning-line active">Replay override ACTIVE. Default acquisition destination is temporarily overridden.</div>}
          </article>

          <article className="section">
            <div className="section-heading"><h3>Default Take Destination</h3><small className="panel-title">Operational Configuration</small></div>
            {datasets.length === 0 && <div className="empty-state compact">No datasets configured. Create your first dataset to begin acquisition routing.</div>}
            <div className="runtime-form-grid">
              <label>
                Dataset selector
                <select value={destinationDraft.dataset_id} onChange={(e) => setDestinationDraft((p) => ({ ...p, dataset_id: e.target.value, session_id: "" }))}>
                  <option value="">Select dataset</option>
                  {datasets.map((d) => <option key={d.id} value={d.id}>{d.name} ({d.id})</option>)}
                </select>
              </label>
              <label>
                Session selector
                <select value={destinationDraft.session_id} onChange={(e) => setDestinationDraft((p) => ({ ...p, session_id: e.target.value }))}>
                  <option value="">Select session</option>
                  {sessionOptions.map((s) => <option key={s.id} value={s.id}>{s.name} ({s.id})</option>)}
                </select>
              </label>
              <label>
                Session naming mode
                <select value={destinationDraft.session_naming_mode} onChange={(e) => setDestinationDraft((p) => ({ ...p, session_naming_mode: e.target.value }))}>
                  <option value="explicit_session">explicit session</option>
                  <option value="timestamp">auto-create timestamp session</option>
                  <option value="daily">auto-create daily session</option>
                  <option value="reuse_current">reuse current session</option>
                </select>
              </label>
              <label><input checked={destinationDraft.auto_create_session} onChange={(e) => setDestinationDraft((p) => ({ ...p, auto_create_session: e.target.checked }))} type="checkbox" /> Auto-create session</label>
              <label><input checked={destinationDraft.activate_immediately} onChange={(e) => setDestinationDraft((p) => ({ ...p, activate_immediately: e.target.checked }))} type="checkbox" /> Activate immediately</label>
              <label><input checked={destinationDraft.persist_as_default} onChange={(e) => setDestinationDraft((p) => ({ ...p, persist_as_default: e.target.checked }))} type="checkbox" /> Persist as default</label>
            </div>

            {!destinationDraft.dataset_id && datasets.length > 0 && <div className="warning-line active">Select a dataset to manage destination sessions.</div>}
            {destinationDraft.dataset_id && sessionOptions.length === 0 && <div className="empty-state compact">No sessions exist for this dataset. Create a session or enable auto-create mode.</div>}

            <div className="studio-actions">
              <button disabled={busy || !destinationDraft.dataset_id || !destinationDraft.session_id} onClick={() => void applyDefaultDestination()} type="button">Use now + Save as persistent default</button>
              <button disabled={busy || !destinationDraft.dataset_id || !destinationDraft.session_id} onClick={() => void api.activateRuntimeRouting({ dataset_id: destinationDraft.dataset_id, session_id: destinationDraft.session_id, routing_mode: "explicit_acquisition_destination", source: "runtime_ui", updated_by: "runtime_ui" }).then(loadAll)} type="button">Use now</button>
              <button disabled={busy || !destinationDraft.dataset_id || !destinationDraft.session_id} onClick={() => void api.updateRuntimeDefaultDestination({ default_dataset_id: destinationDraft.dataset_id, default_session_id: destinationDraft.session_id, auto_create_session: destinationDraft.auto_create_session, session_naming_policy: destinationDraft.session_naming_mode, updated_by: "runtime_ui" }).then(loadAll)} type="button">Save as persistent default</button>
              <button disabled={busy} onClick={() => void clearDefaultDestination()} type="button">Clear default destination</button>
            </div>

            <div className="runtime-form-grid">
              <label>
                Create dataset: name
                <input placeholder="dataset name" value={newDatasetName} onChange={(e) => setNewDatasetName(e.target.value)} />
              </label>
              <label>
                Create dataset: notes (optional)
                <input placeholder="notes" value={newDatasetNotes} onChange={(e) => setNewDatasetNotes(e.target.value)} />
              </label>
              <label>
                Create session: name
                <input placeholder="session name" value={newSessionName} onChange={(e) => setNewSessionName(e.target.value)} />
              </label>
              <label>
                Create session: notes (optional)
                <input placeholder="notes" value={newSessionNotes} onChange={(e) => setNewSessionNotes(e.target.value)} />
              </label>
            </div>
            <div className="studio-actions">
              <button disabled={busy || !newDatasetName.trim()} onClick={() => void createDatasetInline()} type="button">Create dataset</button>
              <button disabled={busy || !destinationDraft.dataset_id || !newSessionName.trim()} onClick={() => void createSessionInline()} type="button">Create session</button>
            </div>
          </article>

          <article className="section">
            <div className="section-heading"><h3>Replay Configuration</h3><small className="panel-title">Temporary Override</small></div>
            {!replay && <div className="empty-state compact">No replay session configured.</div>}
            {replay && !replay.active && <p className="section-note">Replay is inactive. Historical replay metadata is stored but is not affecting acquisition routing.</p>}
            <div className="runtime-form-grid">
              <label>
                Dataset
                <select value={replayDraft.dataset_id} onChange={(e) => setReplayDraft((p) => ({ ...p, dataset_id: e.target.value, session_id: "" }))}>
                  <option value="">Select dataset</option>
                  {datasets.map((d) => <option key={d.id} value={d.id}>{d.name} ({d.id})</option>)}
                </select>
              </label>
              <label>
                Session
                <select value={replayDraft.session_id} onChange={(e) => setReplayDraft((p) => ({ ...p, session_id: e.target.value }))}>
                  <option value="">Select session</option>
                  {replaySessionOptions.map((s) => <option key={s.id} value={s.id}>{s.name} ({s.id})</option>)}
                </select>
              </label>
              <label>
                Replay mode
                <select value={replayDraft.replay_mode} onChange={(e) => setReplayDraft((p) => ({ ...p, replay_mode: e.target.value }))}>
                  <option value="routing_override">routing override</option>
                  <option value="replay_only_inspection">replay-only inspection</option>
                  <option value="passive_replay">passive replay</option>
                  <option value="offline_validation">offline validation</option>
                </select>
              </label>
              <label>
                Replay source
                <select value={replayDraft.replay_source} onChange={(e) => setReplayDraft((p) => ({ ...p, replay_source: e.target.value }))}>
                  <option value="offline_dataset_replay">offline dataset replay</option>
                  <option value="filesystem_replay">filesystem replay</option>
                  <option value="live_replay">live replay</option>
                  <option value="ftp_replay">FTP replay</option>
                </select>
              </label>
              <label><input checked={replayDraft.persist_on_restart} onChange={(e) => setReplayDraft((p) => ({ ...p, persist_on_restart: e.target.checked }))} type="checkbox" /> Persist on restart</label>
              <label><input checked={replayDraft.auto_start_on_startup} onChange={(e) => setReplayDraft((p) => ({ ...p, auto_start_on_startup: e.target.checked }))} type="checkbox" /> Auto-start replay on startup</label>
            </div>
            <div className="studio-actions">
              <button disabled={busy || !replayDraft.dataset_id || !replayDraft.session_id} onClick={() => void startReplay()} type="button">{replay?.active ? "Restart replay" : "Create replay session"}</button>
              <button disabled={busy || !replay} onClick={() => void stopReplay()} type="button">Stop replay</button>
              <button disabled={busy || !replay?.active} onClick={() => void pauseReplay()} type="button">Pause replay</button>
              <button disabled={busy || !replay} onClick={() => void saveReplayConfig()} type="button">Save replay config</button>
              <button disabled={busy || !replay} onClick={() => void clearReplay()} type="button">Clear replay metadata</button>
            </div>
          </article>

          <article className="section">
            <div className="section-heading"><h3>Runtime Services</h3><small className="panel-title">Live</small></div>
            <div className="runtime-service-list">
              {services.map((svc) => (
                <div key={`${svc.process_name}-${svc.pid ?? "none"}`} className={`runtime-service-item status-${serviceBadge(svc.status).toLowerCase().replace(" ", "-")}`}>
                  <div>
                    <strong>{svc.process_name}</strong>
                    <small>{serviceBadge(svc.status)} • PID {svc.pid ?? "-"} • heartbeat {isoAgeLabel(svc.last_heartbeat)}</small>
                    <small>Host {svc.host ?? "-"} • role {svc.runtime_role}</small>
                  </div>
                  <div className="studio-actions">
                    <button disabled title="Backend action pending" type="button">Logs</button>
                    <button disabled title="Backend action pending" type="button">Restart</button>
                    <button disabled title="Backend action pending" type="button">Stop</button>
                  </div>
                </div>
              ))}
            </div>
            <details><summary>Advanced: Raw payload</summary><pre className="json-block runtime-log-view">{JSON.stringify(services, null, 2)}</pre></details>
          </article>

          <article className="section">
            <div className="section-heading"><h3>Calibration Bindings</h3><small className="panel-title">Live + Default</small></div>
            <div className="pipeline-status-grid">
              <div><span>Status</span><strong>{activeCalibration?.status?.toUpperCase() ?? "MISSING"}</strong></div>
              <div><span>Calibration id</span><strong>{activeCalibration?.calibration_id ?? "-"}</strong></div>
              <div><span>Type</span><strong>{activeCalibration?.calibration_type ?? "-"}</strong></div>
              <div><span>Valid for current take</span><strong>{activeCalibration?.valid_for_take == null ? "-" : activeCalibration.valid_for_take ? "yes" : "no"}</strong></div>
              <div><span>Path</span><strong>{activeCalibration?.path ?? "-"}</strong></div>
            </div>
          </article>

          <article className="section">
            <div className="section-heading"><h3>Persistent Runtime Configuration</h3><small className="panel-title">Persistent</small></div>
            <div className="runtime-form-grid">
              <label><input checked={Boolean(configDraft?.runtime?.trispector_ftp?.enabled)} onChange={(e) => setConfigDraft((p) => p ? ({ ...p, runtime: { ...p.runtime, trispector_ftp: { ...p.runtime?.trispector_ftp, enabled: e.target.checked } } }) : p)} type="checkbox" /> FTP enabled</label>
              <label><input disabled type="checkbox" /> Replay restore on startup (Backend write support pending.)</label>
              <label><input disabled type="checkbox" /> Auto-process new takes (Backend write support pending.)</label>
              <label><input disabled type="checkbox" /> Auto-reconnect devices (Backend write support pending.)</label>
              <label>Preview FPS limit<input disabled placeholder="Backend write support pending." value="8" /></label>
              <label>Stale timeout<input disabled placeholder="Backend write support pending." value="5" /></label>
              <label><input checked={Boolean(configDraft?.require_calibration_for_demo)} onChange={(e) => setConfigDraft((p) => p ? ({ ...p, require_calibration_for_demo: e.target.checked }) : p)} type="checkbox" /> Require calibration for demo</label>
              <label><input checked={Boolean(configDraft?.allow_auto_plane_without_calibration)} onChange={(e) => setConfigDraft((p) => p ? ({ ...p, allow_auto_plane_without_calibration: e.target.checked }) : p)} type="checkbox" /> Auto-plane fallback</label>
            </div>
            <div className="studio-actions">
              <button disabled={busy || !configDraft} onClick={() => void saveConfig()} type="button">Save</button>
              <button disabled={busy || !runtimeConfig} onClick={() => setConfigDraft(runtimeConfig?.config ?? null)} type="button">Reset</button>
            </div>
            <details><summary>Advanced: Config JSON</summary><pre className="json-block runtime-log-view">{JSON.stringify(configDraft, null, 2)}</pre></details>
          </article>

          <article className="section">
            <div className="section-heading"><h3>Automation Policies</h3><small className="panel-title">Persistent + Computed</small></div>
            <div className="runtime-service-list">
              <div className="runtime-service-item status-running"><div><strong>Auto plane without calibration</strong><small>{configDraft?.allow_auto_plane_without_calibration ? "ENABLED" : "DISABLED"}</small><small>Allows runtime plane estimation when no saved calibration exists. May reduce measurement consistency.</small></div></div>
              <div className="runtime-service-item status-running"><div><strong>FTP polling</strong><small>{configDraft?.runtime?.trispector_ftp?.enabled ? "ENABLED" : "DISABLED"}</small><small>Controls FTP runtime intake listener.</small></div></div>
            </div>
          </article>
        </section>
      </section>

      {loading && <div className="empty-state compact">Loading runtime workspace...</div>}
    </main>
  );
}
