import { useState, type ReactElement, type ReactNode } from "react";

import type { PartialRerunExecuteResponse, PartialRerunPlanResponse, PipelineComparisonUnit, PipelineRecipe, ProcessingUnitDefinition } from "../api/client";
import { fileUrl } from "../api/client";
import type { ResolvedProcessingUnitArtifactRef } from "./processingUnitModel";
import {
  formatTraceCoverageLine,
  traceBadgeClass,
  traceBadgeLabel,
  type ProcessingUnitGraph,
  type ProcessingUnitTraceSummary,
} from "./processingUnitGraphModel";
import ProcessingUnitGraphViewer from "./ProcessingUnitGraphViewer";
import RunLineageBlock from "./RunLineageBlock";

type WorkspaceContext = {
  hasContract: boolean;
  hasTake: boolean;
  hasRunResult: boolean;
  hasRecipe: boolean;
  hasComparison: boolean;
  traceSummary: ProcessingUnitTraceSummary | null;
};

type Props = {
  graph: ProcessingUnitGraph;
  selectedNodeId: string | null;
  onSelectNode: (nodeId: string) => void;
  selectedUnit: ProcessingUnitDefinition | null;
  selectedUnitIO: { inputs: ResolvedProcessingUnitArtifactRef[]; outputs: ResolvedProcessingUnitArtifactRef[] };
  selectedTraceEntry: Record<string, unknown> | null;
  selectedRecipe: PipelineRecipe | null;
  currentUnitValues: Record<string, unknown> | null;
  recipeUnitValues: Record<string, unknown> | null;
  lastRunValues: Record<string, unknown> | null;
  comparisonUnit: PipelineComparisonUnit | null;
  comparisonSummaryLine: string | null;
  takeId: string | null;
  runSummary: string | null;
  lineageSummary?: string | null;
  performanceSummaryLine?: string | null;
  currentRecipeDirty: boolean;
  currentVsLastRunDirty: boolean;
  tuneContent: ReactNode;
  onResetUnitToRecipe?: (() => void) | null;
  onResetUnitToLastRun?: (() => void) | null;
  onResetUnitToDefaults?: (() => void) | null;
  onSaveRecipeAsNew?: (() => void) | null;
  onSaveRecipeVersion?: (() => void) | null;
  onCloneRecipe?: (() => void) | null;
  onPlanRerun?: (() => void) | null;
  onExecutePartialRerun?: (() => void) | null;
  partialRerunPlan?: PartialRerunPlanResponse | null;
  partialRerunExecution?: PartialRerunExecuteResponse | null;
  planningBusy?: boolean;
  executingBusy?: boolean;
  planStale?: boolean;
  context: WorkspaceContext;
  onClose: () => void;
  onOpenTune: () => void;
  onOpenCompare: () => void;
  onOpenArtifact: (artifactId: string) => void;
  pipelineId?: string | null;
  activeRunId?: string | null;
  activeRunTypeLabel?: string | null;
  onSelectRun?: (runId: string) => void;
  onOpenComparison?: (comparisonId: string) => void;
  onLineageRefreshed?: () => void;
  staleWarnings?: string[];
};

function renderValue(value: unknown): string {
  if (value == null) return "-";
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") return String(value);
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function imageLike(path: string | null | undefined): boolean {
  return /\.(png|jpe?g|webp|bmp)$/i.test(String(path ?? ""));
}

function artifactDiffUnavailableReason(row: PipelineComparisonUnit["artifact_diff"][number]): string {
  if (row.diff_available) return "";
  if (row.diff_reason) return String(row.diff_reason);
  if (!row.left_present && !row.right_present) return "Artifact missing on both sides.";
  if (!row.left_present || !row.right_present) return "Artifact present on only one side.";
  if (!row.diffable) return "Artifact is not marked diffable in the contract.";
  return "Mask diff unavailable.";
}

const WORKFLOW_STEPS = [
  "Select take",
  "Select recipe",
  "Tune unit parameters",
  "Run with recipe + overrides",
  "Compare against previous run",
  "Inspect changed units in graph",
  "Save recipe version",
];

export default function ProcessingUnitGraphWorkspace({
  graph,
  selectedNodeId,
  onSelectNode,
  selectedUnit,
  selectedUnitIO,
  selectedTraceEntry,
  selectedRecipe,
  currentUnitValues,
  recipeUnitValues,
  lastRunValues,
  comparisonUnit,
  comparisonSummaryLine,
  takeId,
  runSummary,
  lineageSummary = null,
  performanceSummaryLine = null,
  currentRecipeDirty,
  currentVsLastRunDirty,
  tuneContent,
  onResetUnitToRecipe,
  onResetUnitToLastRun,
  onResetUnitToDefaults,
  onSaveRecipeAsNew,
  onSaveRecipeVersion,
  onCloneRecipe,
  onPlanRerun,
  onExecutePartialRerun,
  partialRerunPlan,
  partialRerunExecution,
  planningBusy = false,
  executingBusy = false,
  planStale = false,
  context,
  onClose,
  onOpenTune,
  onOpenCompare,
  onOpenArtifact,
  pipelineId = null,
  activeRunId = null,
  activeRunTypeLabel = null,
  onSelectRun,
  onOpenComparison,
  onLineageRefreshed,
  staleWarnings = [],
}: Props): ReactElement {
  const [editMode, setEditMode] = useState(false);
  const [copyMessage, setCopyMessage] = useState<string | null>(null);
  const graphNode = graph.nodes.find((node) => node.id === selectedNodeId) ?? null;
  const traceCoverageLine = formatTraceCoverageLine(context.traceSummary);
  const inferredOnlyTrace = graphNode?.traceSource === "best_effort_artifact_registry";
  const parameterRows = (selectedUnit?.parameters ?? []).map((parameter) => ({
    id: parameter.id,
    label: parameter.label,
    current: currentUnitValues?.[parameter.id],
    recipe: recipeUnitValues?.[parameter.id],
    lastRun: lastRunValues?.[parameter.id],
    defaultValue: parameter.default,
  }));
  const hasUnitArtifacts = selectedUnitIO.inputs.some((item) => item.present) || selectedUnitIO.outputs.some((item) => item.present);

  async function copyPlanJson() {
    if (!partialRerunPlan) return;
    try {
      await navigator.clipboard.writeText(JSON.stringify(partialRerunPlan, null, 2));
      setCopyMessage("Copied plan JSON.");
    } catch {
      setCopyMessage("Could not copy plan JSON.");
    }
  }

  return (
    <div className="processing-unit-workspace-overlay">
      <div className="processing-unit-workspace">
        <header className="processing-unit-workspace-topbar">
          <div>
            <strong>Graph / Compare Workspace</strong>
            <small>{selectedUnit?.label ?? "No processing unit selected."}</small>
            <small>{selectedRecipe ? `Selected recipe: ${selectedRecipe.name} v${selectedRecipe.version}` : "No recipe selected."}</small>
            <small>{currentRecipeDirty ? "Current edits differ from recipe." : "Current edits match selected recipe."}</small>
            <small>{currentVsLastRunDirty ? "Current edits differ from last run." : "Current edits match last run values."}</small>
            <small>Planning only — no pipeline execution happens from this panel.</small>
            <small>{comparisonSummaryLine ?? "No active comparison loaded."}</small>
            <small>{runSummary ?? "Run summary unavailable."}</small>
            {lineageSummary ? <small>{lineageSummary}</small> : null}
            {performanceSummaryLine ? <small>{performanceSummaryLine}</small> : null}
            {traceCoverageLine ? <small>{traceCoverageLine}</small> : <small>Runtime trace coverage unavailable until a run result is loaded.</small>}
          </div>
          <div className="studio-workspace-tabs compact">
            <button type="button" onClick={() => setEditMode((current) => !current)}>{editMode ? "Exit edit mode" : "Edit mode"}</button>
            <button type="button" onClick={onOpenTune}>Open tune panel</button>
            <button type="button" onClick={onOpenCompare}>Open compare tab</button>
            <button type="button" onClick={onClose}>Close workspace</button>
          </div>
        </header>
        <details className="control-panel-details workflow-checklist">
          <summary>Workflow checklist</summary>
          <ol className="workflow-checklist-list">
            {WORKFLOW_STEPS.map((step) => (
              <li key={step}>{step}</li>
            ))}
          </ol>
        </details>
        {copyMessage ? <div className="empty-state compact">{copyMessage}</div> : null}
        {!context.hasContract ? (
          <div className="empty-state compact">No processing-unit contract loaded for this pipeline.</div>
        ) : null}
        {!context.hasTake ? (
          <div className="empty-state compact">Select a take to attach run artifacts and runtime trace to the graph.</div>
        ) : null}
        {context.hasTake && !context.hasRunResult ? (
          <div className="empty-state compact">No run result yet for the selected take. Run the pipeline to populate runtime trace and artifacts.</div>
        ) : null}
        {staleWarnings.length > 0 ? (
          <div className="stale-warning-banner empty-state compact">
            {staleWarnings.map((warning) => (
              <div key={warning}>{warning}</div>
            ))}
          </div>
        ) : null}
        <div className="processing-unit-workspace-grid">
          {pipelineId && activeRunId && onSelectRun && onOpenComparison ? (
            <section className="processing-unit-workspace-panel">
              <span>Run lineage</span>
              <RunLineageBlock
                pipelineId={pipelineId}
                runId={activeRunId}
                activeRunTypeLabel={activeRunTypeLabel}
                onSelectRun={onSelectRun}
                onOpenComparison={onOpenComparison}
                onLineageRefreshed={onLineageRefreshed}
              />
            </section>
          ) : null}
          <section className="processing-unit-workspace-panel">
            <span>Contract graph</span>
            {!context.hasContract ? (
              <div className="empty-state compact">Contract graph unavailable.</div>
            ) : (
              <ProcessingUnitGraphViewer
                graph={graph}
                selectedNodeId={selectedNodeId}
                onSelectNode={onSelectNode}
                traceSummary={context.traceSummary}
              />
            )}
          </section>
          <section className="processing-unit-workspace-panel">
            <span>Selected unit</span>
            {!selectedUnit ? (
              <div className="empty-state compact">Pick a stage or substage node from the graph.</div>
            ) : (
              <>
                <strong>{selectedUnit.label}</strong>
                <small>{selectedUnit.description ?? "No unit description in contract."}</small>
                {graphNode ? (
                  <>
                    <div className="toolbar-chip-row">
                      <span className={`toolbar-chip ${graphNode.status === "failed" ? "status-badge-failed" : graphNode.status === "warning" ? "status-badge-warning" : ""}`}>{graphNode.status.replace(/_/g, " ")}</span>
                      <span className={traceBadgeClass(graphNode.traceSource)}>{traceBadgeLabel(graphNode.traceSource)}</span>
                      {graphNode.comparisonState === "changed" ? <span className="toolbar-chip status-badge-changed">changed</span> : null}
                      {graphNode.dirtyState === "dirty" ? <span className="toolbar-chip status-badge-dirty">dirty</span> : null}
                      {graphNode.lastRunDirtyState === "dirty" ? <span className="toolbar-chip status-badge-warning">dirty vs last run</span> : null}
                      {graphNode.planState !== "none" ? <span className="toolbar-chip status-badge-warning">plan {graphNode.planState}</span> : null}
                      {graphNode.selectionRole !== "none" ? <span className="toolbar-chip status-badge-dirty">{graphNode.selectionRole}</span> : null}
                    </div>
                    <small>Duration: {graphNode.durationMs != null ? `${graphNode.durationMs} ms` : "-"}</small>
                    <small>Views: {selectedUnit.views.map((view) => view.label).join(" | ") || "none declared"}</small>
                  </>
                ) : null}
                {inferredOnlyTrace ? (
                  <div className="empty-state compact">This unit only has inferred trace from artifacts/registry fallback, not a runtime callback.</div>
                ) : null}
                {!hasUnitArtifacts ? (
                  <div className="empty-state compact">Selected unit has no present input/output artifacts in the current run.</div>
                ) : null}
                <div className="table-wrap">
                  <table className="detect-artifact-audit-table">
                    <thead>
                      <tr>
                        <th>Input</th>
                        <th>Present</th>
                        <th>Output</th>
                        <th>Present</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Array.from({ length: Math.max(selectedUnitIO.inputs.length, selectedUnitIO.outputs.length, 1) }).map((_, index) => {
                        const input = selectedUnitIO.inputs[index];
                        const output = selectedUnitIO.outputs[index];
                        return (
                          <tr key={`${input?.id ?? "input"}:${output?.id ?? "output"}:${index}`}>
                            <td>{input?.label ?? "-"}</td>
                            <td>{input ? (input.present ? "yes" : "no") : "-"}</td>
                            <td>{output?.label ?? "-"}</td>
                            <td>{output ? (output.present ? "yes" : "no") : "-"}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
                <small>Trace params: {renderValue(selectedTraceEntry?.parameters_used)}</small>
                <small>Trace metrics: {renderValue(selectedTraceEntry?.metrics)}</small>
                <small>Trace diagnostics: {renderValue(selectedTraceEntry?.diagnostics)}</small>
                <small>Warnings: {Array.isArray(selectedTraceEntry?.warnings) && selectedTraceEntry.warnings.length ? selectedTraceEntry.warnings.join(" | ") : "-"}</small>
                <small>Errors: {Array.isArray(selectedTraceEntry?.errors) && selectedTraceEntry.errors.length ? selectedTraceEntry.errors.join(" | ") : "-"}</small>
                {comparisonUnit ? (
                  <>
                    <span>Comparison rows</span>
                    <small>Parameters changed: {comparisonUnit.parameter_diff.filter((row) => row.status !== "unchanged").length}</small>
                    <small>Metrics changed: {comparisonUnit.metric_diff.filter((row) => row.status !== "unchanged").length}</small>
                    <small>Diagnostics changed: {comparisonUnit.diagnostic_diff.filter((row) => row.status !== "unchanged").length}</small>
                  </>
                ) : (
                  <div className="empty-state compact">No active comparison rows for this unit. Run a compare or reopen a saved comparison.</div>
                )}
                {partialRerunPlan ? (
                  <>
                    <span>Partial rerun plan</span>
                    <small>{planStale ? "Stale plan — refresh before using for decision-making." : "Active plan."}</small>
                    <small>{partialRerunPlan.explanation?.summary ?? "No planner summary available."}</small>
                    <small>Mode: {partialRerunPlan.mode}</small>
                    <small>Selected unit: {partialRerunPlan.selected_unit_id}</small>
                    <small>Execution boundary: {partialRerunPlan.execution_boundary_unit_id ?? partialRerunPlan.effective_start_stage_id ?? "-"}</small>
                    <small>Safety: {partialRerunPlan.safe ? "safe" : `unsafe${partialRerunPlan.fallback_mode ? ` → ${partialRerunPlan.fallback_mode}` : ""}`}</small>
                    <small>Quality / confidence: {partialRerunPlan.plan_quality ?? "unknown"} / {partialRerunPlan.confidence ?? "unknown"}</small>
                    <small>{partialRerunPlan.explanation?.boundary_reason ?? "No boundary explanation available."}</small>
                    <small>{partialRerunPlan.explanation?.reuse_reason ?? "No reuse explanation available."}</small>
                    <small>{partialRerunPlan.explanation?.rerun_reason ?? "No rerun explanation available."}</small>
                    <small>Reuse: {(partialRerunPlan.units_to_reuse ?? []).length} units · Rerun: {(partialRerunPlan.units_to_rerun ?? []).length} · Invalidated: {(partialRerunPlan.units_invalidated ?? []).length}</small>
                    {partialRerunPlan.required_missing_artifacts.length ? <small>Missing artifacts: {partialRerunPlan.required_missing_artifacts.join(", ")}</small> : null}
                    {partialRerunPlan.blocking_reasons.length ? <small>Blocking reasons: {partialRerunPlan.blocking_reasons.join(" | ")}</small> : null}
                    {partialRerunPlan.warnings.length ? <small>Warnings: {partialRerunPlan.warnings.join(" | ")}</small> : null}
                    {partialRerunPlan.explanation?.fallback_reason ? <small>{partialRerunPlan.explanation.fallback_reason}</small> : null}
                    <div className="studio-workspace-tabs compact">
                      <button
                        type="button"
                        disabled={!partialRerunPlan.safe || planStale || !(partialRerunPlan.execution_boundary_unit_id ?? "") || !onExecutePartialRerun || executingBusy}
                        onClick={() => onExecutePartialRerun?.()}
                      >
                        Run partial from boundary
                      </button>
                      <button type="button" onClick={() => void copyPlanJson()}>Copy plan JSON</button>
                    </div>
                    <small>This creates a new child run. Parent run will not be modified.</small>
                  </>
                ) : null}
                {partialRerunExecution ? (
                  <>
                    <span>Partial run summary</span>
                    <small>Parent run: {partialRerunExecution.parent_run_id ?? "-"}</small>
                    <small>Child run: {partialRerunExecution.child_run_id ?? "-"}</small>
                    <small>Boundary: {partialRerunExecution.boundary_stage_id ?? partialRerunExecution.boundary_unit_id ?? "-"}</small>
                    <small>Mode: {partialRerunExecution.execution_mode ?? "-"}</small>
                    <small>Compare to parent: {partialRerunExecution.comparison_id ? "opened" : "not generated"}</small>
                  </>
                ) : null}
              </>
            )}
          </section>
          <section className="processing-unit-workspace-panel">
            <span>Recipe / tune</span>
            <small>{selectedRecipe ? `Selected recipe: ${selectedRecipe.name} v${selectedRecipe.version}` : "No selected recipe."}</small>
            <small>{currentRecipeDirty ? "Current edits differ from recipe." : "Current edits match selected recipe."}</small>
            <small>{currentVsLastRunDirty ? "Current edits differ from last run." : "Current edits match last run values."}</small>
            <div className="studio-workspace-tabs compact">
              <button type="button" disabled={!selectedUnit || planningBusy} onClick={() => onPlanRerun?.()}>Plan rerun from this unit</button>
              <button type="button" disabled={!selectedUnit || !onResetUnitToRecipe} onClick={() => onResetUnitToRecipe?.()}>Reset unit to recipe</button>
              <button type="button" disabled={!selectedUnit || !onResetUnitToLastRun} onClick={() => onResetUnitToLastRun?.()}>Reset unit to last run</button>
              <button type="button" disabled={!selectedUnit || !onResetUnitToDefaults} onClick={() => onResetUnitToDefaults?.()}>Reset unit to defaults</button>
            </div>
            <div className="studio-workspace-tabs compact">
              <button type="button" disabled={!onSaveRecipeAsNew} onClick={() => onSaveRecipeAsNew?.()}>Save current as recipe</button>
              <button type="button" disabled={!selectedRecipe || !onSaveRecipeVersion} onClick={() => onSaveRecipeVersion?.()}>Save new version</button>
              <button type="button" disabled={!selectedRecipe || !onCloneRecipe} onClick={() => onCloneRecipe?.()}>Clone recipe</button>
            </div>
            {parameterRows.length ? (
              <div className="table-wrap">
                <table className="detect-artifact-audit-table">
                  <thead>
                    <tr>
                      <th>Parameter</th>
                      <th>Current edits</th>
                      <th>Selected recipe</th>
                      <th>Last run</th>
                      <th>Default</th>
                    </tr>
                  </thead>
                  <tbody>
                    {parameterRows.map((row) => (
                      <tr key={row.id}>
                        <td>{row.label}</td>
                        <td>{renderValue(row.current)}</td>
                        <td>{renderValue(row.recipe)}</td>
                        <td>{renderValue(row.lastRun)}</td>
                        <td>{renderValue(row.defaultValue)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="empty-state compact">Selected unit has no tunable parameters in the contract.</div>
            )}
            {editMode ? (
              <div className="graph-workspace-edit-panel">
                <span>Graph edit mode</span>
                {selectedUnit ? tuneContent : <div className="empty-state compact">Select a node to edit declared unit parameters.</div>}
              </div>
            ) : (
              <div className="empty-state compact">Enable edit mode to use the graph workspace as the main tuning surface.</div>
            )}
          </section>
        </div>
        <section className="processing-unit-workspace-panel">
          <span>Artifact diff gallery</span>
          {!context.hasComparison ? (
            <div className="empty-state compact">No active comparison loaded. Compare runs/recipes or reopen a saved comparison.</div>
          ) : !comparisonUnit?.artifact_diff?.length ? (
            <div className="empty-state compact">No artifact comparison rows for the selected unit.</div>
          ) : (
            <div className="processing-unit-workspace-artifact-grid">
              {comparisonUnit.artifact_diff.map((row) => (
                <article className="processing-unit-workspace-artifact-card" key={row.artifact_id}>
                  <strong>{row.label}</strong>
                  <small>
                    Changed: {row.pixel_diff?.changed_percent?.toFixed?.(2) ?? "-"}% · IoU: {row.pixel_diff?.iou?.toFixed?.(3) ?? "-"}
                  </small>
                  <small>+{row.pixel_diff?.added_pixels ?? "-"} px · -{row.pixel_diff?.removed_pixels ?? "-"} px</small>
                  {!row.diff_available ? <small>{artifactDiffUnavailableReason(row)}</small> : null}
                  <div className="processing-unit-workspace-thumb-row">
                    {row.left_url && imageLike(row.left_path) ? <img alt={`${row.label} left`} src={row.left_url} /> : <span className="datasets-thumb-placeholder">Left missing</span>}
                    {row.right_url && imageLike(row.right_path) ? <img alt={`${row.label} right`} src={row.right_url} /> : <span className="datasets-thumb-placeholder">Right missing</span>}
                    {row.diff_artifacts?.overlay?.url ? <img alt={`${row.label} diff`} src={row.diff_artifacts.overlay.url} /> : <span className="datasets-thumb-placeholder">Diff n/a</span>}
                  </div>
                  <div className="studio-workspace-tabs compact">
                    <button type="button" onClick={() => onOpenArtifact(row.artifact_id)}>Open artifact</button>
                    {row.left_path && takeId ? <a className="control-link-button" href={fileUrl(takeId, row.left_path)} target="_blank" rel="noreferrer">Open left</a> : null}
                    {row.right_path && takeId ? <a className="control-link-button" href={fileUrl(takeId, row.right_path)} target="_blank" rel="noreferrer">Open right</a> : null}
                  </div>
                </article>
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
