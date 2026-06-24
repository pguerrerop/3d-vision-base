import { useState } from "react";
import { api, type DatasetSessionSummary, type DatasetSummary, type MaterializeMlIngestionResponse } from "../../api/client";
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

type StepVisualStatus = "pending" | "current" | "complete" | "warning" | "blocked";

type StepDefinition = {
  index: number;
  label: typeof STEPS[number];
  status: StepVisualStatus;
  marker: string;
  disabled: boolean;
  gateReason: string;
};

export default function MLSetIngestionWizard({ dataset, sessions, onClose, onMaterialized }: { dataset: DatasetSummary | null; sessions: DatasetSessionSummary[]; onClose: () => void; onMaterialized?: (result: MaterializeMlIngestionResponse) => void }) {
  const [step, setStep] = useState(0);
  const [tableInput, setTableInput] = useState("");
  const [sourceMode, setSourceMode] = useState<"paste" | "upload">("paste");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [runId, setRunId] = useState("");
  const [parsed, setParsed] = useState<unknown>(null);
  const [reconciled, setReconciled] = useState<Record<string, unknown> | null>(null);
  const [manifest, setManifest] = useState<unknown>(null);
  const [mlSetId, setMlSetId] = useState("mining_balls_v1");
  const [busy, setBusy] = useState(false);
  const [parseError, setParseError] = useState<string>("");
  const [reconcileError, setReconcileError] = useState<string>("");

  const reconciledDiagnostics = (reconciled?.diagnostics && typeof reconciled.diagnostics === "object")
    ? reconciled.diagnostics as Record<string, unknown>
    : null;
  const reconciledRows = Array.isArray(reconciled?.rows) ? reconciled.rows as Array<Record<string, unknown>> : [];
  const normalizePreview = Array.isArray(reconciledDiagnostics?.normalization_preview) ? reconciledDiagnostics.normalization_preview as Array<Record<string, unknown>> : [];
  const groupingSummary = (reconciledDiagnostics?.grouping_summary && typeof reconciledDiagnostics.grouping_summary === "object")
    ? reconciledDiagnostics.grouping_summary as Record<string, unknown>
    : null;
  const objectConflictCount = Array.isArray(reconciledDiagnostics?.object_conflicts) ? reconciledDiagnostics.object_conflicts.length : 0;
  const unresolvedCount = Array.isArray(reconciledDiagnostics?.unresolved_take_refs) ? reconciledDiagnostics.unresolved_take_refs.length : 0;
  const ambiguousCount = Array.isArray(reconciledDiagnostics?.ambiguous_take_refs) ? reconciledDiagnostics.ambiguous_take_refs.length : 0;
  const requiredCanonicalMissing = reconciledRows.filter((row) => !String(row.take_id || "").trim() || !String(row.normalized_class || "").trim()).length;
  const physicalObjectCoverageMissing = reconciledRows.filter((row) => !String(row.physical_object_id || "").trim()).length;
  const invalidCanonicalRows = reconciledRows.filter((row) =>
    Array.isArray(row.normalization_warning_codes)
      && row.normalization_warning_codes.some((code) => code === "invalid_normalized_class" || code === "invalid_superclass"),
  ).length;
  const keptInMlSetRows = reconciledRows.filter((row) => Boolean(row.include ?? true)).length;
  const reviewRequiredRows = reconciledRows.filter((row) => Boolean(row.review_required ?? row.needs_review)).length;
  const excludedRows = reconciledRows.filter((row) => ["exclude", "reference"].includes(String(row.label_policy || "")) || !Boolean(row.include ?? true)).length;
  const defaultTrainableRows = reconciledRows.filter((row) => Boolean(row.default_trainable)).length;
  const keptInMlSetObjects = new Set(reconciledRows.filter((row) => Boolean(row.include ?? true) && String(row.physical_object_id || "").trim()).map((row) => String(row.physical_object_id || ""))).size;
  const reviewRequiredObjects = new Set(reconciledRows.filter((row) => Boolean(row.review_required ?? row.needs_review) && String(row.physical_object_id || "").trim()).map((row) => String(row.physical_object_id || ""))).size;
  const excludedObjects = new Set(reconciledRows.filter((row) => (!Boolean(row.include ?? true) || ["exclude", "reference"].includes(String(row.label_policy || ""))) && String(row.physical_object_id || "").trim()).map((row) => String(row.physical_object_id || ""))).size;
  const defaultTrainableObjects = new Set(reconciledRows.filter((row) => Boolean(row.default_trainable) && String(row.physical_object_id || "").trim()).map((row) => String(row.physical_object_id || ""))).size;
  const manifestBlockingErrors = [
    ...(unresolvedCount > 0 ? [`${unresolvedCount} unresolved refs remain.`] : []),
    ...(ambiguousCount > 0 ? [`${ambiguousCount} ambiguous refs remain.`] : []),
    ...(objectConflictCount > 0 ? [`${objectConflictCount} physical object conflicts remain.`] : []),
    ...(requiredCanonicalMissing > 0 ? [`${requiredCanonicalMissing} rows are missing required canonical fields.`] : []),
    ...(physicalObjectCoverageMissing > 0 ? [`${physicalObjectCoverageMissing} rows are missing physical_object_id coverage.`] : []),
    ...(invalidCanonicalRows > 0 ? [`${invalidCanonicalRows} rows contain invalid normalized_class/superclass values.`] : []),
  ];
  const manifestReady = reconciledRows.length > 0 && manifestBlockingErrors.length === 0;
  const canonicalWarningCount = invalidCanonicalRows;
  const manifestStatus = manifest ? "generated" : manifestReady ? "ready" : "blocked";
  const maxUnlockedStep = parsed ? (reconciled ? 7 : 1) : 0;
  const hasReferenceBlockingIssues = unresolvedCount > 0 || ambiguousCount > 0;
  const hasGroupingBlockingIssues = objectConflictCount > 0 || physicalObjectCoverageMissing > 0;
  const activeStepBlockReason = step === 4 && hasReferenceBlockingIssues
    ? manifestBlockingErrors.filter((message) => message.includes("refs remain")).join(" ")
    : step === 5 && hasGroupingBlockingIssues
      ? manifestBlockingErrors.filter((message) => message.includes("physical object") || message.includes("coverage")).join(" ")
      : step === 7 && !manifestReady
        ? manifestBlockingErrors.join(" ")
        : "";
  const preflight = {
    mlSetId,
    provenance: {
      filename: String((parsed as { provenance?: { filename?: string | null } })?.provenance?.filename || ""),
      sha256: String((parsed as { provenance?: { sha256?: string | null } })?.provenance?.sha256 || ""),
      source_type: String((parsed as { provenance?: { source_type?: string | null } })?.provenance?.source_type || ""),
    },
    totalRows: reconciledRows.length,
    keptInMlSetRows,
    physicalObjects: Number(groupingSummary?.physical_object_count || 0),
    keptInMlSetObjects,
    defaultTrainableRows,
    defaultTrainableObjects,
    reviewRequiredRows,
    reviewRequiredObjects,
    excludedRows,
    excludedObjects,
    splitStatus: "unassigned",
    blockingErrors: manifestBlockingErrors,
  };
  const stepStatuses = {
    table: parsed ? "complete" : step === 0 ? "current" : "pending",
    reconcile: step === 1 ? "current" : reconciled ? "complete" : parsed ? "pending" : "pending",
    ranges: step === 2 ? "current" : reconciled ? "complete" : "pending",
    normalize: step === 3 ? "current" : reconciled ? "complete" : "pending",
    ambiguities: step === 4 ? (hasReferenceBlockingIssues ? "blocked" : "current") : hasReferenceBlockingIssues ? "blocked" : reconciled ? "complete" : "pending",
    grouping: step === 5 ? (hasGroupingBlockingIssues ? "blocked" : "current") : hasGroupingBlockingIssues ? "blocked" : reconciled ? "complete" : "pending",
    policy: step === 6 ? (manifestBlockingErrors.length > 0 ? "warning" : "current") : step > 6 ? "complete" : reconciled ? "pending" : "pending",
    manifest: step === 7 ? (manifest ? "complete" : manifestReady ? "current" : "blocked") : manifest ? "complete" : manifestReady ? "pending" : "blocked",
  } as const;
  const steps: StepDefinition[] = STEPS.map((label, index) => {
    let status: StepVisualStatus = "pending";
    let gateReason = "";
    if (index === 0) {
      status = step === 0 ? "current" : parsed ? "complete" : "pending";
    } else if (index === 1) {
      status = step === 1 ? "current" : reconciled ? "complete" : parsed ? "pending" : "pending";
      gateReason = parsed ? "" : "Parse the source table before reconciliation.";
    } else if (index >= 2 && index <= 3) {
      status = step === index ? "current" : step > index ? "complete" : reconciled ? "pending" : "pending";
      gateReason = reconciled ? "" : "Run reconciliation before opening downstream review steps.";
    } else if (index === 4) {
      status = hasReferenceBlockingIssues ? "blocked" : step === 4 ? "current" : step > 4 ? "complete" : reconciled ? "pending" : "pending";
      gateReason = reconciled
        ? (hasReferenceBlockingIssues ? manifestBlockingErrors.filter((message) => message.includes("refs remain")).join(" ") : "")
        : "Run reconciliation before resolving ambiguities.";
    } else if (index === 5) {
      status = hasGroupingBlockingIssues ? "blocked" : step === 5 ? "current" : step > 5 ? "complete" : reconciled ? "pending" : "pending";
      gateReason = reconciled
        ? (hasGroupingBlockingIssues ? manifestBlockingErrors.filter((message) => message.includes("physical object") || message.includes("coverage")).join(" ") : "")
        : "Run reconciliation before reviewing physical object grouping.";
    } else if (index === 6) {
      status = step === 6 ? (manifestBlockingErrors.length > 0 ? "warning" : "current") : step > 6 ? "complete" : reconciled ? "pending" : "pending";
      gateReason = reconciled ? "" : "Run reconciliation before setting policy.";
    } else {
      status = manifest ? "complete" : manifestReady ? (step === 7 ? "current" : "pending") : "blocked";
      gateReason = manifestReady ? "" : manifestBlockingErrors.join(" ");
    }
    return {
      index,
      label,
      status,
      marker: status === "complete" ? "✓" : status === "current" ? "●" : status === "pending" ? "○" : "⚠",
      disabled: index !== step && index > maxUnlockedStep,
      gateReason,
    };
  });

  function navigateToStep(index: number) {
    const target = steps[index];
    if (!target || target.disabled) return;
    setStep(index);
  }

  async function ensureRun(): Promise<string> {
    if (runId) return runId;
    if (!dataset) throw new Error("Select a dataset first");
    const run = await api.createMlIngestionRun({ dataset_id: dataset.id, session_ids: sessions.map((s) => s.id), name: "Mining balls ingestion" });
    setRunId(run.run_id);
    return run.run_id;
  }

  async function ingestTable() {
    setBusy(true);
    setParseError("");
    setReconcileError("");
    try {
      const id = await ensureRun();
      const result = sourceMode === "upload" && selectedFile
        ? await api.ingestMlOperatorFile(id, selectedFile)
        : await api.ingestMlOperatorTable(id, { content: tableInput, input_format: "tsv" });
      setParsed(result);
      setStep(1);
    } catch (error) {
      setParseError(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }

  async function reconcile() {
    if (!parsed) {
      setReconcileError("Parse input before running reconciliation.");
      return;
    }
    setBusy(true);
    setReconcileError("");
    try {
      const id = await ensureRun();
      const result = await api.reconcileMlIngestionRun(id);
      setReconciled(result);
      setStep(2);
    } catch (error) {
      setReconcileError(error instanceof Error ? error.message : String(error));
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
      const result = await api.materializeMlIngestionRun(runId, mlSetId);
      await api.applyMlIngestionMetadata(runId);
      onMaterialized?.(result);
      onClose();
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="concept-panel" style={{ marginTop: 8 }}>
      <h3>ML Set Ingestion Wizard</h3>
      <small>Human reconciliation first, deterministic materialization second.</small>
      <div className="ml-ingestion-stepper" style={{ marginTop: 8 }}>
        {steps.map((stepDef, idx) => (
          <div key={stepDef.label} className="ml-ingestion-stepper-item-wrap">
            <button
              type="button"
              className={`ml-ingestion-stepper-item is-${stepDef.status}`}
              aria-current={stepDef.status === "current" ? "step" : undefined}
              aria-label={`Step ${stepDef.index + 1}: ${stepDef.label} (${stepDef.status})`}
              disabled={stepDef.disabled}
              title={stepDef.gateReason || `${stepDef.label}: ${stepDef.status}`}
              onClick={() => navigateToStep(stepDef.index)}
            >
              <span className="ml-ingestion-stepper-marker" aria-hidden="true">{stepDef.marker}</span>
              <span className="ml-ingestion-stepper-meta">
                <small>{stepDef.index + 1}</small>
                <strong>{stepDef.label}</strong>
              </span>
            </button>
            {idx < steps.length - 1 ? <span className="ml-ingestion-stepper-arrow" aria-hidden="true">→</span> : null}
          </div>
        ))}
      </div>
      <div className="datasets-chip-row" style={{ marginTop: 8 }}>
        <span className="status-chip">Parsed rows: {Array.isArray((parsed as { rows?: unknown })?.rows) ? ((parsed as { rows: unknown[] }).rows.length) : 0}</span>
        <span className="status-chip">Resolved refs: {reconciledRows.length}</span>
        <span className="status-chip">Physical objects: {Number(groupingSummary?.physical_object_count || 0)}</span>
        <span className="status-chip">Canonical warnings: {canonicalWarningCount}</span>
        <span className="status-chip">Review required: {reviewRequiredRows}</span>
        <span className="status-chip">Manifest status: {manifestStatus}</span>
      </div>
      <div className="datasets-chip-row" style={{ marginTop: 8 }}>
        <span className="status-chip">Workflow status</span>
        <span className="status-chip">Table: {stepStatuses.table}</span>
        <span className="status-chip">Reconcile: {stepStatuses.reconcile}</span>
        <span className="status-chip">Ranges: {stepStatuses.ranges}</span>
        <span className="status-chip">Normalize: {stepStatuses.normalize}</span>
        <span className="status-chip">Ambiguities: {stepStatuses.ambiguities}</span>
        <span className="status-chip">Grouping: {stepStatuses.grouping}</span>
        <span className="status-chip">Policy: {stepStatuses.policy}</span>
        <span className="status-chip">Manifest: {stepStatuses.manifest}</span>
      </div>
      <div className="concept-heading">
        <strong>Step {step + 1} of {STEPS.length} — {STEPS[step]}</strong>
        {activeStepBlockReason ? <small>{activeStepBlockReason}</small> : null}
      </div>

      {step === 0 && (
        <div style={{ marginTop: 8 }}>
          <div className="datasets-chip-row" style={{ marginBottom: 8 }}>
            <button type="button" className={`status-chip ${sourceMode === "paste" ? "active" : ""}`} onClick={() => setSourceMode("paste")}>Paste table</button>
            <button type="button" className={`status-chip ${sourceMode === "upload" ? "active" : ""}`} onClick={() => setSourceMode("upload")}>Upload file</button>
          </div>
          {sourceMode === "paste" ? (
            <label className="field-label">Paste table (TSV preferred)<textarea rows={8} value={tableInput} onChange={(e) => setTableInput(e.target.value)} /></label>
          ) : (
            <div>
              <label className="field-label">
                Upload TSV/CSV/TXT
                <input
                  type="file"
                  accept=".tsv,.csv,.txt,text/tab-separated-values,text/csv,text/plain"
                  onChange={(e) => setSelectedFile(e.target.files?.[0] ?? null)}
                />
              </label>
              {selectedFile ? (
                <div className="datasets-chip-row">
                  <span className="status-chip">{selectedFile.name}</span>
                  <span className="status-chip">{selectedFile.size} bytes</span>
                  <button type="button" onClick={() => setSelectedFile(null)}>Remove file</button>
                </div>
              ) : null}
            </div>
          )}
          {parseError ? <div className="empty-state">{parseError}</div> : null}
          <button type="button" disabled={busy || (sourceMode === "paste" ? !tableInput.trim() : !selectedFile)} onClick={() => void ingestTable()}>Parse table</button>
          <TablePreviewStep parsedPreview={parsed} />
        </div>
      )}

      {step === 1 && <>
        <SessionReconciliationStep parsed={parsed} reconciled={reconciled} />
        {reconcileError ? <div className="empty-state">{reconcileError}</div> : null}
        <button type="button" disabled={busy || !parsed} onClick={() => void reconcile()}>{reconciled ? "Run reconciliation again" : "Run reconciliation"}</button>
      </>}
      {step === 2 && <RangeExpansionStep rows={reconciledRows} diagnostics={reconciledDiagnostics} />}
      {step === 3 && <LabelNormalizationStep preview={normalizePreview} />}
      {step === 4 && <AmbiguityResolutionStep diagnostics={reconciledDiagnostics} />}
      {step === 5 && <PhysicalObjectGroupingStep summary={reconciledDiagnostics?.grouping_summary} conflicts={reconciledDiagnostics?.object_conflicts} />}
      {step === 6 && <MLPolicyConfigurationStep summary={{ totalRows: reconciledRows.length, keptInMlSetRows, defaultTrainableRows, reviewRequiredRows, excludedRows, splitStatus: "unassigned" }} />}
      {step === 7 && <ManifestGenerationStep manifestResult={manifest} preflight={preflight} />}

      <div className="datasets-actions-row" style={{ marginTop: 8 }}>
        <button type="button" disabled={step === 0 || busy} onClick={() => setStep((s) => Math.max(0, s - 1))}>Back</button>
        <button type="button" disabled={busy || step >= maxUnlockedStep} onClick={() => setStep((s) => Math.min(maxUnlockedStep, s + 1))}>Next</button>
        <button type="button" disabled={busy || !runId || !manifestReady} onClick={() => void generateManifest()}>Generate canonical manifest</button>
        <input value={mlSetId} onChange={(e) => setMlSetId(e.target.value)} placeholder="mining_balls_v1" />
        <button type="button" disabled={busy || !manifest || manifestBlockingErrors.length > 0} onClick={() => void materialize()}>Materialize ML set</button>
        <button type="button" onClick={onClose}>Close</button>
      </div>
    </section>
  );
}
