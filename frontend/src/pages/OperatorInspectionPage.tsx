import { useCallback, useEffect, useMemo, useState } from "react";

import { api, fileUrl, type OperationsCard, type RuntimeProcessSummary } from "../api/client";

const AUTO_REFRESH_MS = 2500;

function prettyStatus(value: string | null | undefined): string {
  const key = String(value || "acquired").toLowerCase();
  if (key === "acquired") return "Acquired";
  if (key === "queued") return "Queued";
  if (key === "processing") return "Processing";
  if (key === "completed") return "Completed";
  if (key === "failed") return "Failed";
  return key || "Unknown";
}

function formatTimestamp(value: string | null | undefined): string {
  if (!value) return "-";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString();
}

function formatConfidence(value: number | null | undefined): string {
  if (typeof value !== "number") return "-";
  return `${Math.round(value * 100)}%`;
}

function previewUrl(card: OperationsCard): string | null {
  const path = card.preview_image;
  if (!path) return null;
  if (path.startsWith("http://") || path.startsWith("https://") || path.startsWith("data:")) return path;
  return fileUrl(card.take_id, path);
}

export default function OperatorInspectionPage() {
  const [cards, setCards] = useState<OperationsCard[]>([]);
  const [processes, setProcesses] = useState<RuntimeProcessSummary[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState<boolean>(true);
  const [lastUpdatedAt, setLastUpdatedAt] = useState<string | null>(null);

  const load = useCallback(async () => {
    setError(null);
    try {
      const [cardsPayload, processPayload] = await Promise.all([api.operationsCards(120), api.runtimeProcesses()]);
      setCards(cardsPayload.cards ?? []);
      setProcesses(processPayload ?? []);
      setLastUpdatedAt(cardsPayload.updated_at ?? null);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    if (!autoRefresh) return;
    const timer = window.setInterval(() => {
      void load();
    }, AUTO_REFRESH_MS);
    return () => window.clearInterval(timer);
  }, [autoRefresh, load]);

  const statusCounts = useMemo(() => {
    const bucket: Record<string, number> = { acquired: 0, queued: 0, processing: 0, completed: 0, failed: 0 };
    for (const card of cards) {
      const key = String(card.status || "acquired").toLowerCase();
      if (bucket[key] === undefined) bucket[key] = 0;
      bucket[key] += 1;
    }
    return bucket;
  }, [cards]);

  return (
    <main className="operator-page operator-inspection-page">
      <section className="page-title-block">
        <div className="eyebrow">Operations</div>
        <h1>25D Runtime Monitoring</h1>
        <p>Glanceable runtime status for acquired and processed TriSpector takes.</p>
      </section>

      <section className="operator-toolbar">
        <button type="button" className="primary-button" onClick={() => void load()} disabled={loading}>
          {loading ? "Refreshing..." : "Refresh"}
        </button>
        <label className="demo-toggle">
          <input type="checkbox" checked={autoRefresh} onChange={(event) => setAutoRefresh(event.target.checked)} />
          Auto-refresh ({AUTO_REFRESH_MS / 1000}s)
        </label>
        <small>Index updated: {formatTimestamp(lastUpdatedAt)}</small>
      </section>

      {error && <section className="mode-banner"><span>Unable to load operations cards: {error}</span></section>}

      <section className="section-group">
        <div className="section-heading">
          <h3>Runtime Processes</h3>
        </div>
        <div className="operator-counts-grid">
          {processes.map((item) => (
            <div key={item.process_name}>
              <span>{item.runtime_role === "api" ? "API" : item.runtime_role === "watcher" ? "Watcher" : item.runtime_role === "worker" ? "25D Worker" : item.process_name}</span>
              <strong>{String(item.status || "unknown").toUpperCase()}</strong>
            </div>
          ))}
          {!processes.length && <div><span>No process heartbeat</span><strong>-</strong></div>}
        </div>
      </section>

      <section className="operator-counts-grid">
        <div><span>Acquired</span><strong>{statusCounts.acquired ?? 0}</strong></div>
        <div><span>Queued</span><strong>{statusCounts.queued ?? 0}</strong></div>
        <div><span>Processing</span><strong>{statusCounts.processing ?? 0}</strong></div>
        <div><span>Completed</span><strong>{statusCounts.completed ?? 0}</strong></div>
        <div><span>Failed</span><strong>{statusCounts.failed ?? 0}</strong></div>
      </section>

      <section className="operator-recent-list">
        {cards.map((card) => {
          const image = previewUrl(card);
          return (
            <article key={card.take_id} className="take-row operator-recent-row">
              <div className="take-row-main">
                <strong>{card.take_id}</strong>
                <small>Status: {prettyStatus(card.status)}</small>
                <small>Superclass: {card.superclass || "-"} | Label: {card.label || "-"}</small>
                <small>Confidence: {formatConfidence(card.confidence)} | Objects: {card.object_count ?? 0}</small>
                <small>Acquired: {formatTimestamp(card.acquired_at)} | Processed: {formatTimestamp(card.processed_at)}</small>
                {card.error && <small>Error: {card.error}</small>}
              </div>
              <div className="take-row-meta">
                {image ? <img src={image} alt={`Preview ${card.take_id}`} style={{ width: 180, height: 120, objectFit: "cover", borderRadius: 10 }} /> : <small>No preview</small>}
              </div>
            </article>
          );
        })}
        {!cards.length && !loading && <div className="empty-state compact">No operations cards available yet.</div>}
      </section>
    </main>
  );
}
