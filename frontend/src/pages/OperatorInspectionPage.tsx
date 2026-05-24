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

function superclassLabelEs(value: string | null | undefined): string {
  const key = String(value || "").toUpperCase();
  if (key === "BALL_GOOD") return "Bola Buena";
  if (key === "BALL_SCRAP") return "Scrap de Bola";
  if (key === "SCRAP" || key === "SCRAP_METAL") return "Chatarra";
  return "Desconocido";
}

function superclassToneClass(value: string | null | undefined): string {
  const key = String(value || "").toUpperCase();
  if (key === "BALL_GOOD") return "superclass-tone-ball-good";
  if (key === "BALL_SCRAP") return "superclass-tone-ball-scrap";
  if (key === "SCRAP" || key === "SCRAP_METAL") return "superclass-tone-scrap";
  return "superclass-tone-unknown";
}

function formatVariable(value: number | null | undefined, digits = 3): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "-";
  return value.toFixed(digits);
}

export default function OperatorInspectionPage() {
  const [cards, setCards] = useState<OperationsCard[]>([]);
  const [processes, setProcesses] = useState<RuntimeProcessSummary[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState<boolean>(true);
  const [showClassificationVariables, setShowClassificationVariables] = useState<boolean>(false);
  const [lastUpdatedAt, setLastUpdatedAt] = useState<string | null>(null);
  const [hiddenTakeIds, setHiddenTakeIds] = useState<Set<string>>(new Set());
  const [expandedImage, setExpandedImage] = useState<{ src: string; alt: string } | null>(null);

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

  useEffect(() => {
    if (!expandedImage) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setExpandedImage(null);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [expandedImage]);

  const statusCounts = useMemo(() => {
    const bucket: Record<string, number> = { acquired: 0, queued: 0, processing: 0, completed: 0, failed: 0 };
    for (const card of cards) {
      const key = String(card.status || "acquired").toLowerCase();
      if (bucket[key] === undefined) bucket[key] = 0;
      bucket[key] += 1;
    }
    return bucket;
  }, [cards]);

  const visibleCards = useMemo(
    () => cards.filter((card) => !hiddenTakeIds.has(card.take_id)),
    [cards, hiddenTakeIds]
  );

  const clearPastDetections = useCallback(() => {
    setHiddenTakeIds((previous) => {
      const next = new Set(previous);
      for (const card of cards) next.add(card.take_id);
      return next;
    });
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
        <button type="button" className="secondary-button" onClick={clearPastDetections} disabled={!cards.length}>
          Clean Past Detections
        </button>
        <label className="demo-toggle">
          <input type="checkbox" checked={autoRefresh} onChange={(event) => setAutoRefresh(event.target.checked)} />
          Auto-refresh ({AUTO_REFRESH_MS / 1000}s)
        </label>
        <label className="demo-toggle">
          <input
            type="checkbox"
            checked={showClassificationVariables}
            onChange={(event) => setShowClassificationVariables(event.target.checked)}
          />
          Show classification variables
        </label>
        <small>Index updated: {formatTimestamp(lastUpdatedAt)}</small>
      </section>

      {error && <section className="mode-banner"><span>Unable to load operator inspection results: {error}</span></section>}

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
        {visibleCards.map((card, index) => {
          const image = previewUrl(card);
          const superclass = card.superclass || "-";
          const superclassEs = superclassLabelEs(card.superclass);
          const superclassTone = superclassToneClass(card.superclass);
          return (
            <article
              key={card.take_id}
              className={`take-row operator-recent-row ${index === 0 ? "operator-recent-row-latest" : ""}`}
            >
              <div className={`operator-superclass-center ${superclassTone}`}>{superclassEs}</div>
              <div className="take-row-main">
                <strong>{card.take_id}</strong>
                <small>Status: {prettyStatus(card.status)}</small>
                <small>Superclass: {superclassEs} ({superclass}) | Label: {card.label || "-"}</small>
                <small>Confidence: {formatConfidence(card.confidence)} | Objects: {card.object_count ?? 0}</small>
                {showClassificationVariables && (
                  <small>
                    Vars: max_h={formatVariable(card.classification_variables?.max_height_mm, 2)}mm | p95_h=
                    {formatVariable(card.classification_variables?.p95_height_mm, 2)}mm | ecc=
                    {formatVariable(card.classification_variables?.eccentricity)} | sph3d=
                    {formatVariable(card.classification_variables?.sphericity_3d)} | flat=
                    {formatVariable(card.classification_variables?.flatness)} | rough=
                    {formatVariable(card.classification_variables?.edge_roughness, 2)} | footprint_roundness=
                    {formatVariable(card.classification_variables?.footprint_roundness)} | vol=
                    {formatVariable(card.classification_variables?.volume_proxy_mm3, 1)}mm3
                  </small>
                )}
                <small>Acquired: {formatTimestamp(card.acquired_at)} | Processed: {formatTimestamp(card.processed_at)}</small>
                {card.error && <small>Error: {card.error}</small>}
              </div>
              <div className="take-row-meta">
                {image ? (
                  <button
                    type="button"
                    className="preview-thumb-button"
                    onClick={() => setExpandedImage({ src: image, alt: `Preview ${card.take_id}` })}
                  >
                    <img src={image} alt={`Preview ${card.take_id}`} className="preview-thumb-image" />
                  </button>
                ) : (
                  <small>No display image available</small>
                )}
              </div>
            </article>
          );
        })}
        {!visibleCards.length && !loading && <div className="empty-state compact">No operations cards available yet.</div>}
      </section>

      {expandedImage && (
        <div className="modal-backdrop" role="dialog" aria-modal="true" onClick={() => setExpandedImage(null)}>
          <div className="modal-panel wide image-modal-panel" onClick={(event) => event.stopPropagation()}>
            <div className="modal-header">
              <h3>Image Preview</h3>
              <button type="button" onClick={() => setExpandedImage(null)}>Close</button>
            </div>
            <div className="image-modal-content">
              <img src={expandedImage.src} alt={expandedImage.alt} className="image-modal-preview" />
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
