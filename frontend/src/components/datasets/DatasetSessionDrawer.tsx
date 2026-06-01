import { useEffect, useMemo, useState } from "react";
import { api, fileUrl, type DatasetSessionDetailSummary, type DatasetSessionSummary, type DatasetSummary, type TakeSummary } from "../../api/client";
import EntityDetailDrawer from "./EntityDetailDrawer";
import EntityExportActions from "./EntityExportActions";
import EntityMetadataForm, { type SessionForm } from "./EntityMetadataForm";
import EntityStatisticsGrid from "./EntityStatisticsGrid";
import EntitySummaryCards from "./EntitySummaryCards";

type Props = {
  open: boolean;
  session: DatasetSessionSummary | null;
  dataset: DatasetSummary | null;
  onClose: () => void;
  onSaved: () => Promise<void>;
};

const EMPTY_SUMMARY: DatasetSessionDetailSummary = {
  total_takes: 0,
  reviewed: 0,
  unreviewed: 0,
  rejected: 0,
  needs_review: 0,
  golden_sample: 0,
  benchmark_approved: 0,
  missing_labels: 0,
  missing_split: 0,
  missing_calibration: 0,
  processing_failed: 0,
  processing_incomplete: 0,
  no_objects_detected: 0,
  split_distribution: {},
  processing_by_family: {},
  object_annotation_count: 0,
  object_candidate_count: 0,
  last_acquisition_timestamp: "-",
  calibration_assignment: "-",
};

export default function DatasetSessionDrawer({ open, session, dataset, onClose, onSaved }: Props) {
  const [form, setForm] = useState<SessionForm>({
    name: "",
    notes: "",
    tags: "",
    description: "",
    acquisition_type: "engineering",
    calibration_id: "",
    sensor_metadata: "{}",
    conveyor_metadata: "{}",
    lighting_metadata: "{}",
  });
  const [summary, setSummary] = useState<DatasetSessionDetailSummary>(EMPTY_SUMMARY);
  const [takes, setTakes] = useState<TakeSummary[]>([]);
  const [offset, setOffset] = useState(0);
  const [hasMore, setHasMore] = useState(false);
  const [busy, setBusy] = useState(false);
  const [jsonError, setJsonError] = useState<string | null>(null);

  useEffect(() => {
    if (!open || !session) return;
    setForm({
      name: session.name || "",
      notes: session.notes || "",
      tags: (session.tags || []).join(", "),
      description: session.description || "",
      acquisition_type: session.session_type || "engineering",
      calibration_id: session.calibration_id || "",
      sensor_metadata: JSON.stringify(session.sensor_metadata || {}, null, 2),
      conveyor_metadata: JSON.stringify(session.conveyor_metadata || {}, null, 2),
      lighting_metadata: JSON.stringify(session.lighting_metadata || {}, null, 2),
    });
    setOffset(0);
    void api.datasetSessionSummary(session.id, session.dataset_id)
      .then((payload) => setSummary(payload.summary || EMPTY_SUMMARY))
      .catch(() => setSummary(EMPTY_SUMMARY));
    void api.pagedTakes({ dataset_id: session.dataset_id, session_id: session.id, limit: 20, offset: 0 })
      .then((page) => {
        setTakes(page.items || []);
        setOffset(page.next_offset || (page.items || []).length);
        setHasMore(Boolean(page.has_more));
      })
      .catch(() => {
        setTakes([]);
        setHasMore(false);
      });
  }, [open, session]);

  const cards = useMemo(() => [
    { label: "Total takes", value: Number(summary.total_takes || 0) },
    { label: "Reviewed", value: `${Number(summary.total_takes || 0) ? Math.round((Number(summary.reviewed || 0) / Number(summary.total_takes || 1)) * 100) : 0}%` },
    { label: "Unreviewed", value: Number(summary.unreviewed || 0) },
    { label: "Missing labels", value: Number(summary.missing_labels || 0) },
  ], [summary]);

  const reviewedPct = Number(summary.total_takes || 0) ? Math.round((Number(summary.reviewed || 0) / Number(summary.total_takes || 1)) * 100) : 0;
  const splitCovered = Math.max(0, Number(summary.total_takes || 0) - Number(summary.missing_split || 0));
  const splitCoveragePct = Number(summary.total_takes || 0) ? Math.round((splitCovered / Number(summary.total_takes || 1)) * 100) : 0;
  const processingCovered = Math.max(0, Number(summary.total_takes || 0) - Number(summary.processing_incomplete || 0));
  const processingCoveragePct = Number(summary.total_takes || 0) ? Math.round((processingCovered / Number(summary.total_takes || 1)) * 100) : 0;

  function parseJsonField(raw: string, label: string): Record<string, unknown> {
    try {
      const parsed = JSON.parse(raw || "{}");
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
        throw new Error(`${label} must be a JSON object`);
      }
      return parsed as Record<string, unknown>;
    } catch {
      throw new Error(`Invalid JSON in ${label}`);
    }
  }

  async function save() {
    if (!session || !dataset) return;
    let sensor: Record<string, unknown>;
    let conveyor: Record<string, unknown>;
    let lighting: Record<string, unknown>;
    try {
      sensor = parseJsonField(form.sensor_metadata, "sensor metadata");
      conveyor = parseJsonField(form.conveyor_metadata, "conveyor metadata");
      lighting = parseJsonField(form.lighting_metadata, "lighting metadata");
      setJsonError(null);
    } catch (err) {
      setJsonError(err instanceof Error ? err.message : "Invalid JSON");
      return;
    }
    setBusy(true);
    try {
      await api.updateDatasetSession(dataset.id, session.id, {
        name: form.name,
        description: form.description,
        notes: form.notes,
        tags: form.tags.split(",").map((item) => item.trim()).filter(Boolean),
        session_type: form.acquisition_type,
        calibration_id: form.calibration_id || null,
        sensor_metadata: sensor,
        conveyor_metadata: conveyor,
        lighting_metadata: lighting,
      });
      await onSaved();
    } finally {
      setBusy(false);
    }
  }

  async function loadMore() {
    if (!session || !hasMore) return;
    const page = await api.pagedTakes({ dataset_id: session.dataset_id, session_id: session.id, limit: 20, offset });
    setTakes((prev) => [...prev, ...(page.items || [])]);
    setOffset(page.next_offset || (offset + (page.items || []).length));
    setHasMore(Boolean(page.has_more));
  }

  const splitSummary = Object.entries(summary.split_distribution || {}).map(([k, v]) => `${k}:${v}`).join(" · ") || "-";
  const processingSummary = Object.entries(summary.processing_by_family || {}).map(([k, v]) => `${k}:${v}`).join(" · ") || "-";

  return (
    <EntityDetailDrawer
      open={open}
      title={session?.name || "Session"}
      subtitle={`${dataset?.name || "Dataset"} · ${session?.created_at || "-"}`}
      headerExtras={(
        <div className="datasets-chip-row">
          <span className="status-chip">takes {summary.total_takes}</span>
          <span className="status-chip">reviewed {reviewedPct}%</span>
          <span className="status-chip">type {session?.session_type || "engineering"}</span>
          <span className="status-chip">calibration {session?.calibration_id ? "linked" : "unlinked"}</span>
        </div>
      )}
      onClose={onClose}
      footer={(
        <>
          <button type="button" className="primary" disabled={busy} onClick={() => void save()}>Save changes</button>
          <button type="button" onClick={() => {
            const link = document.createElement("a");
            link.href = `/api/dataset-sessions/${encodeURIComponent(session?.id || "")}/export?dataset_id=${encodeURIComponent(dataset?.id || "")}`;
            link.download = `dataset_session_${session?.id || "session"}_export.json`;
            document.body.appendChild(link);
            link.click();
            link.remove();
          }}>Export JSON</button>
          <button type="button" onClick={() => { window.location.href = `/studio?dataset_id=${encodeURIComponent(dataset?.id || "")}&session_id=${encodeURIComponent(session?.id || "")}`; }}>Open in Studio filtered to session</button>
          <button type="button" className="deemphasized" disabled>Archive session</button>
          <button type="button" className="deemphasized" disabled>Restore session</button>
          <button type="button" className="deemphasized" disabled>Create ML set from session</button>
        </>
      )}
    >
      <EntitySummaryCards cards={cards} />
      <section className="entity-section">
        <h4>Summary</h4>
        <EntityStatisticsGrid rows={[
          { label: "Take count", value: String(summary.total_takes || 0) },
          { label: "Reviewed", value: `${reviewedPct}%` },
          { label: "Split coverage", value: `${splitCoveragePct}%` },
          { label: "Processing coverage", value: `${processingCoveragePct}%` },
          { label: "Object annotations", value: String(summary.object_annotation_count || 0) },
          { label: "Last acquisition", value: String(summary.last_acquisition_timestamp || "-") },
        ]} />
        <details className="entity-diagnostics">
          <summary>Diagnostics</summary>
          <small>Split distribution: {splitSummary}</small>
          <small>Processing by family: {processingSummary}</small>
          <small>Calibration: {String(summary.calibration_assignment || "-")}</small>
        </details>
      </section>
      <EntityMetadataForm value={form} onChange={setForm} />
      {jsonError ? <div className="warning-line active">{jsonError}</div> : null}
      <section className="entity-section">
        <h4>Takes in this session ({summary.total_takes || 0})</h4>
        <div className="entity-takes-table-wrap">
          <table className="datasets-table entity-takes-table">
            <thead>
              <tr><th>Thumb</th><th>Take</th><th>Expected class</th><th>Validation</th><th>Processing</th><th /></tr>
            </thead>
            <tbody>
              {takes.map((take) => (
                <tr key={take.take_id}>
                  <td>{take.thumbnail_path ? <img className="datasets-thumb" src={fileUrl(take.take_id, take.thumbnail_path)} alt={take.take_id} loading="lazy" /> : <span className="datasets-thumb-placeholder">-</span>}</td>
                  <td>{take.friendly_name || take.take_id}</td>
                  <td>{take.expected_class || "-"}</td>
                  <td>{take.validation_status || "unreviewed"}</td>
                  <td>{take.has_done ? "done" : take.has_ready ? "ready" : "pending"}</td>
                  <td><a href={`/takes/${encodeURIComponent(take.take_id)}`}>Inspect</a></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="entity-inline-actions">
          <button type="button" disabled={!hasMore} onClick={() => void loadMore()}>{hasMore ? "View all takes" : "All session takes loaded"}</button>
        </div>
      </section>
      <EntityExportActions onExport={() => {
        const link = document.createElement("a");
        link.href = `/api/dataset-sessions/${encodeURIComponent(session?.id || "")}/export?dataset_id=${encodeURIComponent(dataset?.id || "")}`;
        link.download = `dataset_session_${session?.id || "session"}_export.json`;
        document.body.appendChild(link);
        link.click();
        link.remove();
      }} />
    </EntityDetailDrawer>
  );
}
