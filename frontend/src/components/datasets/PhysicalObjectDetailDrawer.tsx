import { useEffect, useMemo, useState } from "react";
import { api, type DatasetSummary, type LabelTaxonomyEntry, type PhysicalObjectSummary } from "../../api/client";
import EntityDetailDrawer from "./EntityDetailDrawer";
import EntityStatisticsGrid from "./EntityStatisticsGrid";
import PhysicalObjectTakeGallery from "./PhysicalObjectTakeGallery";

type Props = {
  open: boolean;
  physicalObject: PhysicalObjectSummary | null;
  dataset: DatasetSummary | null;
  onClose: () => void;
  onSelectSession?: (sessionId: string) => void;
  onLabelCorrected?: () => void;
};

export default function PhysicalObjectDetailDrawer({ open, physicalObject, dataset, onClose, onSelectSession, onLabelCorrected }: Props) {
  const [takes, setTakes] = useState<Array<{ session_id: string; take_id: string; metadata: Record<string, unknown> }>>([]);
  const [takesLoading, setTakesLoading] = useState(false);
  const [repeatability, setRepeatability] = useState<Record<string, unknown> | null>(null);
  const [labelDraft, setLabelDraft] = useState("");
  const [superclassDraft, setSuperclassDraft] = useState("");
  const [reason, setReason] = useState("");
  const [savingCorrection, setSavingCorrection] = useState(false);
  const [correctionMessage, setCorrectionMessage] = useState("");
  const [taxonomy, setTaxonomy] = useState<LabelTaxonomyEntry[]>([]);

  useEffect(() => {
    if (!open || !physicalObject) return;
    setTakesLoading(true);
    void api.physicalObjectTakes(physicalObject.physical_object_id, physicalObject.dataset_id)
      .then((payload) => setTakes(payload.takes || []))
      .catch(() => setTakes([]))
      .finally(() => setTakesLoading(false));
    void api.physicalObjectRepeatability(physicalObject.physical_object_id, physicalObject.dataset_id).then((payload) => setRepeatability(payload)).catch(() => setRepeatability(null));
    void api.labelTaxonomy().then((payload) => setTaxonomy(payload.entries || [])).catch(() => setTaxonomy([]));
  }, [open, physicalObject]);

  useEffect(() => {
    setLabelDraft(physicalObject?.normalized_class || "");
    setSuperclassDraft(physicalObject?.superclass || "");
    setReason("");
    setCorrectionMessage("");
  }, [physicalObject?.physical_object_id]);

  const matchedTaxonomyEntry = useMemo(
    () => taxonomy.find((entry) => entry.normalized_class === labelDraft.trim()) || null,
    [taxonomy, labelDraft],
  );

  useEffect(() => {
    if (matchedTaxonomyEntry) setSuperclassDraft(matchedTaxonomyEntry.superclass);
  }, [matchedTaxonomyEntry]);

  async function saveCorrection() {
    if (!physicalObject || !labelDraft.trim()) return;
    setSavingCorrection(true);
    setCorrectionMessage("");
    try {
      const result = await api.correctPhysicalObjectLabel(physicalObject.physical_object_id, {
        dataset_id: physicalObject.dataset_id,
        normalized_class: labelDraft.trim(),
        superclass: superclassDraft.trim() || null,
        reason: reason.trim() || null,
      });
      setCorrectionMessage(`Correction ${result.id} applied to ${result.affected_take_ids.length} takes and ${result.affected_ml_set_ids.length} ML sets.`);
      void api.labelTaxonomy().then((payload) => setTaxonomy(payload.entries || [])).catch(() => undefined);
      onLabelCorrected?.();
    } catch (error) {
      setCorrectionMessage(error instanceof Error ? error.message : "Label correction failed.");
    } finally {
      setSavingCorrection(false);
    }
  }

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
        <h4>Correct canonical label</h4>
        <small>This updates the physical object, every linked take and its ML-set memberships in one audited database transaction.</small>
        <label className="field-label">
          Normalized class
          <input value={labelDraft} onChange={(event) => setLabelDraft(event.target.value)} list="label-taxonomy-classes" placeholder="type to search known classes…" />
          <datalist id="label-taxonomy-classes">
            {taxonomy.map((entry) => (
              <option key={entry.normalized_class} value={entry.normalized_class}>{entry.superclass}</option>
            ))}
          </datalist>
        </label>
        <label className="field-label">
          Superclass
          <input
            value={superclassDraft}
            onChange={(event) => setSuperclassDraft(event.target.value)}
            disabled={Boolean(matchedTaxonomyEntry)}
            list="label-taxonomy-superclasses"
          />
          <datalist id="label-taxonomy-superclasses">
            {Array.from(new Set(taxonomy.map((entry) => entry.superclass))).map((sc) => <option key={sc} value={sc} />)}
          </datalist>
        </label>
        <small>
          {labelDraft.trim() === ""
            ? null
            : matchedTaxonomyEntry
              ? `Known class — superclass locked to ${matchedTaxonomyEntry.superclass} from label_taxonomy.${matchedTaxonomyEntry.is_uncertain ? " (flagged as an inherently uncertain class.)" : ""}`
              : "New class — not in label_taxonomy yet. Set a superclass above to register it as part of this correction."}
        </small>
        <label className="field-label">Reason<input value={reason} onChange={(event) => setReason(event.target.value)} placeholder="e.g. cadena → SCRAP_METAL" /></label>
        <button type="button" disabled={savingCorrection || !labelDraft.trim()} onClick={() => void saveCorrection()}>{savingCorrection ? "Saving…" : "Apply audited correction"}</button>
        {correctionMessage ? <small>{correctionMessage}</small> : null}
      </section>
      <PhysicalObjectTakeGallery
        takes={takes}
        loading={takesLoading}
        datasetId={physicalObject?.dataset_id}
        physicalObjectId={physicalObject?.physical_object_id}
      />
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
