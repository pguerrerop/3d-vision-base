import { useEffect, useMemo, useState } from "react";
import { api, type DatasetMlSet, type DatasetSummary, type MLSetSummaryResponse } from "../../api/client";
import EntityDetailDrawer from "./EntityDetailDrawer";
import EntityExportActions from "./EntityExportActions";
import EntityStatisticsGrid from "./EntityStatisticsGrid";
import MLSetClassDistribution from "./MLSetClassDistribution";
import MLSetReadinessCards from "./MLSetReadinessCards";
import MLSetRepresentativeSamples from "./MLSetRepresentativeSamples";
import MLSetSessionCoverage from "./MLSetSessionCoverage";
import MLSetSplitVisualization from "./MLSetSplitVisualization";
import MLSetTaskDefinitions from "./MLSetTaskDefinitions";
import MLSetTrainingCompatibility from "./MLSetTrainingCompatibility";
import MLSetValidationWarnings from "./MLSetValidationWarnings";

type Props = {
  open: boolean;
  mlSet: DatasetMlSet | null;
  dataset: DatasetSummary | null;
  onClose: () => void;
  onSelectSession?: (sessionId: string) => void;
};

export default function MLSetDetailDrawer({ open, mlSet, dataset, onClose, onSelectSession }: Props) {
  const [summary, setSummary] = useState<MLSetSummaryResponse | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!open || !mlSet) return;
    setLoading(true);
    void api.mlSetSummary(mlSet.id, mlSet.dataset_id)
      .then((payload) => setSummary(payload))
      .finally(() => setLoading(false));
  }, [open, mlSet]);

  const chips = useMemo(() => {
    if (!summary) return null;
    return (
      <div className="datasets-chip-row">
        <span className="status-chip">{summary.identity.take_count} members</span>
        <span className="status-chip">{summary.identity.physical_object_count} objects</span>
        <span className="status-chip">{`${summary.identity.default_trainable_count ?? 0} default trainable rows / ${summary.identity.trainable_object_count} objects`}</span>
        <span className="status-chip">{`${summary.identity.review_required_count} review rows / ${summary.identity.review_required_object_count ?? 0} objects`}</span>
        <span className="status-chip">{summary.readiness.split_status}</span>
        <span className="status-chip">{summary.identity.label_schema_version}</span>
        <span className="status-chip">{summary.readiness.split_status === "unassigned" && summary.identity.leakage_safe ? "ready / no split assigned yet" : summary.identity.leakage_safe ? "leakage-safe" : "leakage-risk"}</span>
      </div>
    );
  }, [summary]);

  return (
    <EntityDetailDrawer
      open={open}
      title={mlSet?.name || mlSet?.id || "ML Set"}
      subtitle={`${dataset?.name || mlSet?.dataset_id || "Dataset"} · ${mlSet?.created_at || "-"} · ${mlSet?.membership_mode || "ids"}`}
      headerExtras={chips}
      onClose={onClose}
      mode="wide"
      footer={(
        <>
          <button type="button" onClick={() => { window.location.href = `/api/ml-sets/${encodeURIComponent(mlSet?.id || "")}/export?kind=manifest&dataset_id=${encodeURIComponent(mlSet?.dataset_id || "")}`; }}>Export manifest</button>
          <button type="button" onClick={() => { window.location.href = `/api/ml-sets/${encodeURIComponent(mlSet?.id || "")}/export?kind=splits&dataset_id=${encodeURIComponent(mlSet?.dataset_id || "")}`; }}>Export splits</button>
          <button type="button" onClick={() => { window.location.href = `/api/ml-sets/${encodeURIComponent(mlSet?.id || "")}/export?kind=label_schema&dataset_id=${encodeURIComponent(mlSet?.dataset_id || "")}`; }}>Export label schema</button>
          <button type="button" className="deemphasized" disabled>Freeze snapshot</button>
          <button type="button" className="deemphasized" disabled>Clone ML set</button>
          <button type="button" className="deemphasized" disabled>Create classifier experiment</button>
        </>
      )}
    >
      {loading || !summary ? <div className="empty-state">Loading ML set detail...</div> : null}
      {summary ? (
        <>
          <MLSetReadinessCards readiness={summary.readiness} />
          <section className="entity-section">
            <h4>Identity & Provenance</h4>
            <EntityStatisticsGrid rows={[
              { label: "Membership mode", value: summary.membership_definition.membership_mode },
              { label: "Ingestion run", value: summary.identity.ingestion_run || "-" },
              { label: "Validated / unvalidated", value: `${summary.identity.validated_count} / ${summary.identity.unvalidated_count}` },
              { label: "Kept / default trainable", value: `${summary.identity.kept_in_ml_set_count ?? summary.identity.take_count} / ${summary.identity.default_trainable_count ?? summary.identity.trainable_object_count}` },
              { label: "Review required / excluded", value: `${summary.identity.review_required_count} rows / ${summary.identity.review_required_object_count ?? 0} objects · ${summary.identity.excluded_count} excluded` },
              { label: "Split health", value: summary.readiness.split_status === "unassigned" && summary.identity.leakage_safe ? "ready / no split assigned yet" : summary.readiness.split_health },
              { label: "Source", value: String((summary.membership_definition.semantics.source_filename as string | undefined) || "-") },
              { label: "Source hash", value: String((summary.membership_definition.semantics.source_sha256 as string | undefined) || "-") },
              { label: "Normalization version", value: String((summary.membership_definition.semantics.normalization_version as string | undefined) || summary.identity.label_schema_version || "-") },
            ]} />
          </section>
          <section className="entity-section">
            <h4>Raw Label Distribution</h4>
            <pre className="datasets-inline-state">{JSON.stringify(summary.raw_label_distribution, null, 2)}</pre>
          </section>
          <MLSetClassDistribution classDistribution={summary.class_distribution} superclassDistribution={summary.superclass_distribution} />
          <MLSetRepresentativeSamples samples={summary.representative_samples} />
          <MLSetSplitVisualization splitSummary={summary.split_summary} />
          <MLSetSessionCoverage coverage={summary.source_session_coverage} onSelectSession={onSelectSession} />
          <section className="entity-section">
            <h4>Inclusion Rules / Membership Definition</h4>
            <details className="entity-diagnostics" open>
              <summary>Rules and semantics</summary>
              <pre className="datasets-inline-state">{JSON.stringify(summary.membership_definition, null, 2)}</pre>
            </details>
          </section>
          <MLSetValidationWarnings warnings={summary.warnings} />
          <MLSetTaskDefinitions tasks={summary.derived_tasks} />
          <MLSetTrainingCompatibility trainingCompatibility={summary.training_compatibility} />
          <EntityExportActions onExport={() => { window.location.href = `/api/ml-sets/${encodeURIComponent(mlSet?.id || "")}/export?kind=snapshot&dataset_id=${encodeURIComponent(mlSet?.dataset_id || "")}`; }} />
        </>
      ) : null}
    </EntityDetailDrawer>
  );
}
