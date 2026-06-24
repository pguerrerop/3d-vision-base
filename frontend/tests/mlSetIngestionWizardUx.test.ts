import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

function wizardSource(): string {
  return fs.readFileSync(path.resolve(process.cwd(), "src/components/ingestion/MLSetIngestionWizard.tsx"), "utf-8");
}

function reconcileSource(): string {
  return fs.readFileSync(path.resolve(process.cwd(), "src/components/ingestion/SessionReconciliationStep.tsx"), "utf-8");
}

function datasetsPageSource(): string {
  return fs.readFileSync(path.resolve(process.cwd(), "src/pages/DatasetsPage.tsx"), "utf-8");
}

test("Wizard passes parsed and reconciled previews into reconciliation step", () => {
  const source = wizardSource();
  assert.ok(source.includes("<SessionReconciliationStep parsed={parsed} reconciled={reconciled} />"));
  assert.ok(source.includes('disabled={busy || !parsed}'));
  assert.ok(source.includes("disabled={busy || !runId || !manifestReady}"));
});

test("Wizard promotes Reconcile as the active step immediately after parsing", () => {
  const source = wizardSource();
  assert.ok(source.includes("setStep(1);"));
  assert.ok(source.includes("Step {step + 1} of {STEPS.length} — {STEPS[step]}"));
  assert.ok(source.includes('aria-current={stepDef.status === "current" ? "step" : undefined}'));
});

test("Wizard marks Policy current and earlier steps complete when the flow reaches step 7", () => {
  const source = wizardSource();
  assert.ok(source.includes('policy: step === 6 ? (manifestBlockingErrors.length > 0 ? "warning" : "current")'));
  assert.ok(source.includes('status = step === 6 ? (manifestBlockingErrors.length > 0 ? "warning" : "current") : step > 6 ? "complete" : reconciled ? "pending" : "pending";'));
  assert.ok(source.includes('status = step === index ? "current" : step > index ? "complete" : reconciled ? "pending" : "pending";'));
});

test("Wizard keeps Manifest pending or blocked until generation is available", () => {
  const source = wizardSource();
  assert.ok(source.includes('manifest: step === 7 ? (manifest ? "complete" : manifestReady ? "current" : "blocked") : manifest ? "complete" : manifestReady ? "pending" : "blocked"'));
  assert.ok(source.includes('const manifestStatus = manifest ? "generated" : manifestReady ? "ready" : "blocked";'));
});

test("Wizard surfaces blocked reasons for unresolved refs and object conflicts", () => {
  const source = wizardSource();
  assert.ok(source.includes('unresolvedCount > 0 ? [`${unresolvedCount} unresolved refs remain.`] : []'));
  assert.ok(source.includes('objectConflictCount > 0 ? [`${objectConflictCount} physical object conflicts remain.`] : []'));
  assert.ok(source.includes('title={stepDef.gateReason || `${stepDef.label}: ${stepDef.status}`}'));
});

test("Wizard renders summary badges for workflow totals", () => {
  const source = wizardSource();
  assert.ok(source.includes("Parsed rows:"));
  assert.ok(source.includes("Resolved refs:"));
  assert.ok(source.includes("Physical objects:"));
  assert.ok(source.includes("Canonical warnings:"));
  assert.ok(source.includes("Review required:"));
  assert.ok(source.includes("Manifest status:"));
  assert.ok(source.includes("const canonicalWarningCount = invalidCanonicalRows;"));
  assert.ok(!source.includes("const canonicalWarningCount = invalidCanonicalRows + reviewRequiredRows + ambiguousCount;"));
});

test("Datasets page refreshes and selects the new ML set after materialization", () => {
  const source = datasetsPageSource();
  assert.ok(source.includes("async function handleWizardMaterialized(result: MaterializeMlIngestionResponse)"));
  assert.ok(source.includes("setSuccessMessage(`Created ML set ${result.ml_set_id} with ${result.membership_count} memberships / ${result.physical_object_count} physical objects.`);"));
  assert.ok(source.includes("const items = await refreshMlSetsForDataset(result.dataset_id);"));
  assert.ok(source.includes("setSelectedMlSetId(created.id);"));
  assert.ok(source.includes('setActiveTab("ml_sets");'));
  assert.ok(source.includes("onMaterialized={(result) => { void handleWizardMaterialized(result); }}"));
});

test("Reconcile step shows pending parse summary instead of raw empty object", () => {
  const source = reconcileSource();
  assert.ok(source.includes("Run reconciliation to resolve refs to canonical dataset/session/take IDs."));
  assert.ok(source.includes("Parsed input summary"));
  assert.ok(source.includes("Advanced diagnostics"));
  assert.ok(!source.includes("JSON.stringify(preview ?? {}, null, 2)"));
});

test("Reconcile step renders resolved and unresolved counts after reconciliation", () => {
  const source = reconcileSource();
  assert.ok(source.includes("<small>resolved</small>"));
  assert.ok(source.includes("<small>unresolved</small>"));
  assert.ok(source.includes("<small>ambiguous</small>"));
  assert.ok(source.includes("dataset/session conflicts"));
});

test("Post-reconciliation steps render compact summaries instead of placeholders", () => {
  const rangeSource = fs.readFileSync(path.resolve(process.cwd(), "src/components/ingestion/RangeExpansionStep.tsx"), "utf-8");
  const normalizeSource = fs.readFileSync(path.resolve(process.cwd(), "src/components/ingestion/LabelNormalizationStep.tsx"), "utf-8");
  const groupingSource = fs.readFileSync(path.resolve(process.cwd(), "src/components/ingestion/PhysicalObjectGroupingStep.tsx"), "utf-8");
  const manifestSource = fs.readFileSync(path.resolve(process.cwd(), "src/components/ingestion/ManifestGenerationStep.tsx"), "utf-8");
  const policySource = fs.readFileSync(path.resolve(process.cwd(), "src/components/ingestion/MLPolicyConfigurationStep.tsx"), "utf-8");
  assert.ok(rangeSource.includes("No range expansion needed; all rows used explicit image_ref values."));
  assert.ok(normalizeSource.includes("superclass_distribution"));
  assert.ok(normalizeSource.includes("canonical warnings"));
  assert.ok(normalizeSource.includes("unmapped raw labels"));
  assert.ok(policySource.includes("kept in ML set rows"));
  assert.ok(policySource.includes("default trainable rows"));
  assert.ok(groupingSource.includes("physical objects"));
  assert.ok(groupingSource.includes("conflicts"));
  assert.ok(manifestSource.includes("Materialization contract"));
  assert.ok(manifestSource.includes("default trainable rows/objects"));
});
