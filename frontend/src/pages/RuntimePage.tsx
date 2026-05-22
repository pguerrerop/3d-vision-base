import { useCallback, useEffect, useMemo, useState } from "react";
import { api, type RuntimeHealthChecklist, type RuntimeQueueStatus, type RuntimeWorkerEvent, type RuntimeWorkerState, type RuntimeWorkerStatus } from "../api/client";
import StatusBadge from "../components/StatusBadge";

function runtimeStateLabel(worker: RuntimeWorkerStatus, events: RuntimeWorkerEvent[]): RuntimeWorkerState {
  if (worker.state === "running") {
    const hasDryRun = [...events].reverse().find((eventItem) =>
      String(eventItem.event_type || "").includes("WOULD_")
      || String(eventItem.message || "").toLowerCase().includes("dry-run")
    );
    if (hasDryRun) return "dry_run";
  }
  return worker.state;
}

function latestRefs(events: RuntimeWorkerEvent[]): { take: string | null; group: string | null; result: string | null } {
  let take: string | null = null;
  let group: string | null = null;
  let result: string | null = null;
  for (const eventItem of [...events].reverse()) {
    if (!take) {
      const fromMetaTake = eventItem.metadata?.take_id;
      if (typeof fromMetaTake === "string" && fromMetaTake) take = fromMetaTake;
    }
    if (!group) {
      const fromMetaGroup = eventItem.metadata?.group_id;
      if (typeof fromMetaGroup === "string" && fromMetaGroup) group = fromMetaGroup;
      const fromMetaAcqGroup = eventItem.metadata?.acquisition_group_id;
      if (!group && typeof fromMetaAcqGroup === "string" && fromMetaAcqGroup) group = fromMetaAcqGroup;
    }
    if (!result) {
      const fromMetaResult = eventItem.metadata?.published_result_id;
      if (typeof fromMetaResult === "string" && fromMetaResult) result = fromMetaResult;
    }
    if (take && group && result) break;
  }
  return { take, group, result };
}

function commandHint(worker: RuntimeWorkerStatus): string {
  if (worker.worker_type.includes("rgb")) {
    return `python3 scripts/run_rgb_worker.py --data-dir data --source-id ${worker.source_id ?? "usb_camera_0"} --once --dry-run`;
  }
  if (worker.worker_type.includes("25d") || worker.worker_type.includes("heightmap")) {
    return `python3 scripts/run_25d_worker.py --data-dir data --source-id ${worker.source_id ?? "trispector_ftp_0"} --once --dry-run`;
  }
  if (worker.worker_type.includes("fusion")) {
    return `python3 scripts/run_fusion_publisher_worker.py --data-dir data --once --dry-run`;
  }
  return `python3 scripts/runtime.py list`;
}

export default function RuntimePage() {
  const [workers, setWorkers] = useState<RuntimeWorkerStatus[]>([]);
  const [queue, setQueue] = useState<RuntimeQueueStatus | null>(null);
  const [selectedId, setSelectedId] = useState<string>("");
  const [events, setEvents] = useState<RuntimeWorkerEvent[]>([]);
  const [healthChecklist, setHealthChecklist] = useState<RuntimeHealthChecklist | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [busy, setBusy] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [copiedHint, setCopiedHint] = useState<boolean>(false);

  const selected = useMemo(
    () => workers.find((worker) => worker.worker_id === selectedId) ?? null,
    [workers, selectedId]
  );

  const refreshWorkers = useCallback(async () => {
    const payload = await api.runtimeWorkers();
    setWorkers(payload.workers);
    setQueue(payload.queue ?? null);
    if (payload.workers.length > 0 && !payload.workers.some((item) => item.worker_id === selectedId)) {
      setSelectedId(payload.workers[0].worker_id);
    }
  }, [selectedId]);

  const refreshDetails = useCallback(async () => {
    if (!selectedId) return;
    const [eventsPayload, checklistPayload] = await Promise.all([
      api.runtimeWorkerEvents(selectedId, 120),
      api.runtimeProcessHealthChecklist("trispector_ftp").catch(() => null),
    ]);
    setEvents(eventsPayload.events);
    setHealthChecklist(checklistPayload);
  }, [selectedId]);

  useEffect(() => {
    let active = true;
    const load = async () => {
      setLoading(true);
      try {
        await refreshWorkers();
        await refreshDetails();
        if (active) setError(null);
      } catch (exc) {
        if (active) setError(String(exc));
      } finally {
        if (active) setLoading(false);
      }
    };
    void load();
    const timer = window.setInterval(() => {
      void refreshWorkers();
      void refreshDetails();
    }, 3000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [refreshWorkers, refreshDetails]);

  const requestStop = useCallback(
    async (workerId: string) => {
      setBusy(true);
      try {
        await api.stopRuntimeWorker(workerId);
        await refreshWorkers();
        await refreshDetails();
      } catch (exc) {
        setError(String(exc));
      } finally {
        setBusy(false);
      }
    },
    [refreshWorkers, refreshDetails]
  );

  const copyCommandHint = useCallback(async (command: string) => {
    try {
      await navigator.clipboard.writeText(command);
      setCopiedHint(true);
      window.setTimeout(() => setCopiedHint(false), 1200);
    } catch {
      setCopiedHint(false);
    }
  }, []);

  return (
    <main className="runtime-page">
      <section className="page-title-block">
        <h2>Runtime Workers</h2>
        <p>Engineering supervision panel for automatic runtime workers. Operator decisions stay in the Operator UI.</p>
      </section>
      {error && <div className="warning-line active">{error}</div>}
      <section className="runtime-grid">
        <article className="runtime-process-list section">
          {loading && workers.length === 0 && <div className="empty-state compact">Loading runtime workers...</div>}
          {!loading && workers.length === 0 && (
            <div className="empty-state compact">No runtime workers registered yet. Start a worker from CLI and refresh.</div>
          )}
          {workers.map((worker) => {
            const workerState = runtimeStateLabel(worker, worker.worker_id === selectedId ? events : []);
            return (
            <button
              className={`take-row ${selectedId === worker.worker_id ? "active" : ""}`}
              key={worker.worker_id}
              onClick={() => setSelectedId(worker.worker_id)}
              type="button"
            >
              <span className="take-row-main">
                <strong>{worker.worker_id}</strong>
                <small>{worker.worker_type}</small>
                <small>source: {worker.source_id ?? "-"} | station: {worker.station_id ?? "-"}</small>
                <small>{worker.last_event_summary ?? "No events yet"}</small>
              </span>
              <span className="take-row-meta">
                <StatusBadge value={workerState} />
                <small>heartbeat: {worker.last_heartbeat ?? "-"}</small>
              </span>
            </button>
          );
          })}
        </article>
        <article className="runtime-process-detail section">
          {!selected && <div className="empty-state compact">No runtime worker selected.</div>}
          {selected && (
            <>
              <div className="section-heading">
                <div>
                  <h3>{selected.worker_id}</h3>
                  <p>{selected.worker_type}</p>
                </div>
                <div className="studio-actions">
                  <button disabled={busy} onClick={() => void requestStop(selected.worker_id)} type="button">
                    Request stop
                  </button>
                </div>
              </div>
              {selected.last_error && <div className="warning-line active">{selected.last_error}</div>}
              <div className="pipeline-status-grid">
                <div>
                  <span>Status</span>
                  <strong>{runtimeStateLabel(selected, events)}</strong>
                </div>
                <div>
                  <span>Heartbeat</span>
                  <strong>{selected.last_heartbeat ?? "-"}</strong>
                </div>
                <div>
                  <span>Source</span>
                  <strong>{selected.source_id ?? "-"}</strong>
                </div>
                <div>
                  <span>Pipeline</span>
                  <strong>{selected.pipeline_id ?? "-"}</strong>
                </div>
                <div>
                  <span>Modality</span>
                  <strong>{selected.modality ?? "-"}</strong>
                </div>
                <div>
                  <span>Queue depth</span>
                  <strong>{selected.queue_depth ?? 0}</strong>
                </div>
                <div>
                  <span>Station</span>
                  <strong>{selected.station_id ?? "-"}</strong>
                </div>
                <div>
                  <span>Last event</span>
                  <strong>{selected.last_event_summary ?? "-"}</strong>
                </div>
                <div>
                  <span>Last take</span>
                  <strong>{latestRefs(events).take ?? "-"}</strong>
                </div>
                <div>
                  <span>Last group</span>
                  <strong>{latestRefs(events).group ?? "-"}</strong>
                </div>
                <div>
                  <span>Last result</span>
                  <strong>{latestRefs(events).result ?? "-"}</strong>
                </div>
                <div>
                  <span>Stop requested</span>
                  <strong>{selected.stop_requested ? "yes" : "no"}</strong>
                </div>
                <div>
                  <span>Processed count</span>
                  <strong>{selected.processed_count ?? 0}</strong>
                </div>
                <div>
                  <span>Error count</span>
                  <strong>{selected.error_count ?? 0}</strong>
                </div>
                <div>
                  <span>Retry count</span>
                  <strong>{selected.retry_count ?? 0}</strong>
                </div>
                <div>
                  <span>Last success</span>
                  <strong>{selected.last_success_at ?? "-"}</strong>
                </div>
              </div>
              {queue && (
                <div className="pipeline-status-grid">
                  <div><span>Pending queue</span><strong>{queue.pending_count}</strong></div>
                  <div><span>Active claims</span><strong>{queue.active_claims_count}</strong></div>
                  <div><span>Completed</span><strong>{queue.completed_count}</strong></div>
                  <div><span>Failed</span><strong>{queue.failed_count}</strong></div>
                </div>
              )}
              <div className="runtime-detail-columns">
                <div>
                  <h3>Recent events</h3>
                  <pre className="json-block runtime-log-view">
                    {JSON.stringify(events.slice(-30), null, 2) || "No events available."}
                  </pre>
                </div>
                <div>
                  <h3>Manual command hint</h3>
                  <div className="command-callout">
                    <span>CLI</span>
                    <code>{commandHint(selected)}</code>
                    <button type="button" onClick={() => void copyCommandHint(commandHint(selected))}>
                      {copiedHint ? "Copied" : "Copy command"}
                    </button>
                  </div>
                </div>
              </div>
              {healthChecklist && (
                <div className="section">
                  <h3>Runtime Health Checklist</h3>
                  <div className="runtime-health-checklist">
                    {healthChecklist.items.map((item) => (
                      <div key={item.id}>
                        <span>{item.label}</span>
                        <strong>{item.ok ? "ok" : "warning"}</strong>
                        <small>{item.details}</small>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </article>
      </section>
    </main>
  );
}
