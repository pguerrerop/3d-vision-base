import { useEffect, useState } from "react";
import { api, type DatasetSummary, type PhysicalObjectSummary } from "../../api/client";
import EntityDetailDrawer from "./EntityDetailDrawer";
import EntityStatisticsGrid from "./EntityStatisticsGrid";
import PhysicalObjectTakeGallery from "./PhysicalObjectTakeGallery";

type Props = {
  open: boolean;
  physicalObject: PhysicalObjectSummary | null;
  dataset: DatasetSummary | null;
  onClose: () => void;
  onSelectSession?: (sessionId: string) => void;
};

export default function PhysicalObjectDetailDrawer({ open, physicalObject, dataset, onClose, onSelectSession }: Props) {
  const [takes, setTakes] = useState<Array<{ session_id: string; take_id: string; metadata: Record<string, unknown> }>>([]);
  const [repeatability, setRepeatability] = useState<Record<string, unknown> | null>(null);

  useEffect(() => {
    if (!open || !physicalObject) return;
    void api.physicalObjectTakes(physicalObject.physical_object_id, physicalObject.dataset_id).then((payload) => setTakes(payload.takes || [])).catch(() => setTakes([]));
    void api.physicalObjectRepeatability(physicalObject.physical_object_id, physicalObject.dataset_id).then((payload) => setRepeatability(payload)).catch(() => setRepeatability(null));
  }, [open, physicalObject]);

  return (
    <EntityDetailDrawer
      open={open}
      title={physicalObject?.physical_object_id || "Physical Object"}
      subtitle={`${dataset?.name || physicalObject?.dataset_id || "Dataset"} · ${physicalObject?.normalized_class || "unlabeled"}`}
      headerExtras={(
        <div className="datasets-chip-row">
          <span className="status-chip">{physicalObject?.normalized_class || "UNKNOWN"}</span>
          <span className="status-chip">{physicalObject?.superclass || "UNKNOWN"}</span>
          <span className="status-chip">{physicalObject?.observation_take_ids?.length ?? 0} takes</span>
          <span className="status-chip">{physicalObject?.source_session_ids?.length ?? 0} sessions</span>
          <span className="status-chip">{physicalObject?.annotation_confidence || "UNKNOWN"}</span>
        </div>
      )}
      onClose={onClose}
      footer={(
        <>
          <button type="button" className="deemphasized" disabled>Reconcile object</button>
          <button type="button" className="deemphasized" disabled>Open ML memberships</button>
        </>
      )}
    >
      <section className="entity-section">
        <h4>Identity</h4>
        <EntityStatisticsGrid rows={[
          { label: "Normalized class", value: String(physicalObject?.normalized_class || "-") },
          { label: "Superclass", value: String(physicalObject?.superclass || "-") },
          { label: "Raw operator label", value: String(physicalObject?.raw_operator_label || "-") },
          { label: "Source row index", value: String(physicalObject?.source_row_index ?? "-") },
          { label: "Needs review", value: physicalObject?.needs_review ? "yes" : "no" },
        ]} />
      </section>
      <section className="entity-section">
        <h4>Dimensions</h4>
        <EntityStatisticsGrid rows={[
          { label: "D1 / D2 / D3", value: `${physicalObject?.d1_mm ?? "-"} / ${physicalObject?.d2_mm ?? "-"} / ${physicalObject?.d3_mm ?? "-"}` },
          { label: "Diameter mean", value: String(physicalObject?.diameter_mean_mm ?? "-") },
          { label: "Diameter range", value: String(physicalObject?.diameter_range_mm ?? "-") },
        ]} />
      </section>
      <section className="entity-section">
        <h4>Source Provenance</h4>
        <div className="datasets-chip-row">
          {(physicalObject?.source_session_ids || []).map((sessionId) => (
            <button key={sessionId} type="button" className="link-button" onClick={() => onSelectSession?.(sessionId)}>{sessionId}</button>
          ))}
        </div>
      </section>
      <PhysicalObjectTakeGallery takes={takes} />
      <section className="entity-section">
        <h4>Repeatability Metrics</h4>
        {!repeatability ? <small>No repeatability summary yet.</small> : (
          <details className="entity-diagnostics" open>
            <summary>{String(repeatability.take_count || 0)} takes grouped by physical_object_id</summary>
            <pre className="datasets-inline-state">{JSON.stringify(repeatability, null, 2)}</pre>
          </details>
        )}
      </section>
      <section className="entity-section">
        <h4>Warnings / Ambiguities</h4>
        <small>{physicalObject?.needs_review ? "Review required for one or more linked takes." : "No object-level warnings surfaced."}</small>
      </section>
    </EntityDetailDrawer>
  );
}
