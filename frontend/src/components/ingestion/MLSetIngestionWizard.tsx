import { useState } from "react";
import { api, type DatasetSessionSummary, type DatasetSummary } from "../../api/client";
import AmbiguityResolutionStep from "./AmbiguityResolutionStep";
import LabelNormalizationStep from "./LabelNormalizationStep";
import ManifestGenerationStep from "./ManifestGenerationStep";
import MLPolicyConfigurationStep from "./MLPolicyConfigurationStep";
import PhysicalObjectGroupingStep from "./PhysicalObjectGroupingStep";
import RangeExpansionStep from "./RangeExpansionStep";
import SessionReconciliationStep from "./SessionReconciliationStep";
import TablePreviewStep from "./TablePreviewStep";

const STEPS = [
  "Table",
  "Reconcile",
  "Ranges",
  "Normalize",
  "Ambiguities",
  "Grouping",
  "Policy",
  "Manifest",
] as const;

export default function MLSetIngestionWizard({ dataset, sessions, onClose }: { dataset: DatasetSummary | null; sessions: DatasetSessionSummary[]; onClose: () => void }) {
  const [step, setStep] = useState(0);
  const [tableInput, setTableInput] = useState("");
  const [runId, setRunId] = useState("");
  const [parsed, setParsed] = useState<unknown>(null);
  const [reconciled, setReconciled] = useState<Record<string, unknown> | null>(null);
  const [manifest, setManifest] = useState<unknown>(null);
  const [mlSetId, setMlSetId] = useState("mining_balls_v1");
  const [busy, setBusy] = useState(false);

  async function ensureRun(): Promise<string> {
    if (runId) return runId;
    if (!dataset) throw new Error("Select a dataset first");
    const run = await api.createMlIngestionRun({ dataset_id: dataset.id, session_ids: sessions.map((s) => s.id), name: "Mining balls ingestion" });
    setRunId(run.run_id);
    return run.run_id;
  }

  async function ingestTable() {
    setBusy(true);
    try {
      const id = await ensureRun();
      const result = await api.ingestMlOperatorTable(id, { content: tableInput, input_format: "tsv" });
      setParsed(result);
      setStep(1);
    } finally {
      setBusy(false);
    }
  }

  async function reconcile() {
    setBusy(true);
    try {
      const id = await ensureRun();
      const result = await api.reconcileMlIngestionRun(id);
      setReconciled(result);
      setStep(2);
    } finally {
      setBusy(false);
    }
  }

  async function generateManifest() {
    setBusy(true);
    try {
      const id = await ensureRun();
      const result = await api.generateMlIngestionManifest(id);
      setManifest(result);
      setStep(7);
    } finally {
      setBusy(false);
    }
  }

  async function materialize() {
    if (!runId) return;
    setBusy(true);
    try {
      await api.materializeMlIngestionRun(runId, mlSetId);
      await api.applyMlIngestionMetadata(runId);
      onClose();
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="concept-panel" style={{ marginTop: 8 }}>
      <h3>ML Set Ingestion Wizard</h3>
      <small>Human reconciliation first, deterministic materialization second.</small>
      <div className="datasets-chip-row" style={{ marginTop: 8 }}>
        {STEPS.map((name, idx) => <span key={name} className={`status-chip ${idx === step ? "active" : ""}`}>{idx + 1}. {name}</span>)}
      </div>

      {step === 0 && (
        <div style={{ marginTop: 8 }}>
          <label className="field-label">Paste table (TSV preferred)<textarea rows={8} value={tableInput} onChange={(e) => setTableInput(e.target.value)} /></label>
          <button type="button" disabled={busy || !tableInput.trim()} onClick={() => void ingestTable()}>Parse table</button>
          <TablePreviewStep parsedPreview={parsed} />
        </div>
      )}

      {step === 1 && <><SessionReconciliationStep preview={reconciled} /><button type="button" disabled={busy} onClick={() => void reconcile()}>Run reconciliation</button></>}
      {step === 2 && <RangeExpansionStep />}
      {step === 3 && <LabelNormalizationStep />}
      {step === 4 && <AmbiguityResolutionStep diagnostics={reconciled?.diagnostics} />}
      {step === 5 && <PhysicalObjectGroupingStep />}
      {step === 6 && <MLPolicyConfigurationStep />}
      {step === 7 && <ManifestGenerationStep manifestResult={manifest} />}

      <div className="datasets-actions-row" style={{ marginTop: 8 }}>
        <button type="button" disabled={step === 0 || busy} onClick={() => setStep((s) => Math.max(0, s - 1))}>Back</button>
        <button type="button" disabled={busy || step >= 7} onClick={() => setStep((s) => Math.min(7, s + 1))}>Next</button>
        <button type="button" disabled={busy || !runId} onClick={() => void generateManifest()}>Generate canonical manifest</button>
        <input value={mlSetId} onChange={(e) => setMlSetId(e.target.value)} placeholder="mining_balls_v1" />
        <button type="button" disabled={busy || !manifest} onClick={() => void materialize()}>Materialize ML set</button>
        <button type="button" onClick={onClose}>Close</button>
      </div>
    </section>
  );
}
