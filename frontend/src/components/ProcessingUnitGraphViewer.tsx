import type { ReactElement } from "react";

import type { ProcessingUnitGraph, ProcessingUnitGraphNode, ProcessingUnitTraceSummary } from "./processingUnitGraphModel";
import { formatTraceCoverageLine, traceBadgeClass, traceBadgeLabel } from "./processingUnitGraphModel";

type Props = {
  graph: ProcessingUnitGraph;
  selectedNodeId: string | null;
  onSelectNode: (nodeId: string) => void;
  traceSummary?: ProcessingUnitTraceSummary | null;
};

function statusLabel(status: ProcessingUnitGraphNode["status"]): string {
  if (status === "missing_artifacts") return "missing";
  return status.replace(/_/g, " ");
}

function compactObjectLines(value: Record<string, unknown> | undefined): string {
  const entries = Object.entries(value ?? {});
  if (!entries.length) return "none";
  return entries.map(([key, item]) => `${key}=${String(item)}`).join(", ");
}

function planStateLabel(planState: ProcessingUnitGraphNode["planState"]): string {
  if (planState === "none") return "";
  return planState.replace(/_/g, " ");
}

export default function ProcessingUnitGraphViewer({ graph, selectedNodeId, onSelectNode, traceSummary = null }: Props): ReactElement {
  const selectedNode = graph.nodes.find((node) => node.id === selectedNodeId) ?? graph.nodes[0] ?? null;
  const nodeById = new Map(graph.nodes.map((node) => [node.id, node]));
  const traceCoverageLine = formatTraceCoverageLine(traceSummary);
  return (
    <div className="processing-unit-graph">
      {traceCoverageLine ? <small className="trace-coverage-line">{traceCoverageLine}</small> : null}
      <div className="toolbar-chip-row">
        <span className="toolbar-chip">Legend</span>
        <span className="toolbar-chip">reused</span>
        <span className="toolbar-chip status-badge-changed">rerun</span>
        <span className="toolbar-chip status-badge-warning">invalidated</span>
        <span className="toolbar-chip status-badge-failed">blocked</span>
        <span className="toolbar-chip status-badge-dirty">selected</span>
        <span className="toolbar-chip status-badge-warning">boundary</span>
      </div>
      {graph.warnings.length > 0 ? (
        <div className="empty-state compact">{graph.warnings.join(" | ")}</div>
      ) : null}
      {!graph.nodes.length ? (
        <div className="empty-state compact">No processing-unit contract nodes are available.</div>
      ) : (
        <div className="processing-unit-graph-groups">
          {graph.groups.map((group) => (
            <section key={group.id} className="processing-unit-graph-group">
              <header className="processing-unit-graph-group-header">
                <strong>{group.label}</strong>
                <small>{group.nodeIds.length} units</small>
              </header>
              <div className="processing-unit-graph-node-list">
                {group.nodeIds.map((nodeId, index) => {
                  const node = nodeById.get(nodeId);
                  if (!node) return null;
                  const selected = node.id === selectedNodeId;
                  return (
                    <button
                      key={node.id}
                      type="button"
                      className={`processing-unit-graph-node ${selected ? "selected" : ""} ${node.kind === "stage" ? "stage" : "substage"}`}
                      onClick={() => onSelectNode(node.id)}
                    >
                      <div className="processing-unit-graph-node-header">
                        <strong>{node.label}</strong>
                        <small>{node.kind}</small>
                      </div>
                      <div className="toolbar-chip-row">
                        <span className="toolbar-chip">params {node.parameterCount}</span>
                        <span className="toolbar-chip">artifacts {node.artifactCount}</span>
                        <span className={`toolbar-chip ${node.status === "failed" ? "status-badge-failed" : node.status === "warning" || node.status === "missing_artifacts" ? "status-badge-warning" : ""}`}>
                          {statusLabel(node.status)}
                        </span>
                        <span className={traceBadgeClass(node.traceSource)}>{traceBadgeLabel(node.traceSource)}</span>
                        <span className={`toolbar-chip ${node.dirtyState === "dirty" ? "status-badge-dirty" : ""}`}>{node.dirtyState}</span>
                        <span className={`toolbar-chip ${node.lastRunDirtyState === "dirty" ? "status-badge-warning" : ""}`}>last run {node.lastRunDirtyState}</span>
                        <span className={`toolbar-chip ${node.comparisonState === "changed" ? "status-badge-changed" : ""}`}>{node.comparisonState}</span>
                        {node.planState !== "none" ? (
                          <span className={`toolbar-chip ${node.planState === "blocked" ? "status-badge-failed" : node.planState === "invalidated" ? "status-badge-warning" : node.planState === "rerun" ? "status-badge-changed" : ""}`}>
                            {node.planIsStale ? `stale ${planStateLabel(node.planState)}` : planStateLabel(node.planState)}
                          </span>
                        ) : null}
                        {node.selectionRole !== "none" ? (
                          <span className={`toolbar-chip ${node.selectionRole === "boundary" ? "status-badge-warning" : "status-badge-dirty"}`}>
                            {node.selectionRole}
                          </span>
                        ) : null}
                      </div>
                      {node.warningCount > 0 || node.missingArtifactCount > 0 ? (
                        <small>Warnings {node.warningCount} · Missing outputs {node.missingArtifactCount}</small>
                      ) : null}
                      {index < group.nodeIds.length - 1 ? <div className="processing-unit-graph-connector" aria-hidden="true" /> : null}
                    </button>
                  );
                })}
              </div>
            </section>
          ))}
        </div>
      )}
      {selectedNode ? (
        <section className="control-panel-section">
          <span>Node detail</span>
          <strong>{selectedNode.label}</strong>
          <small>{selectedNode.description ?? "No description."}</small>
          <small>Inputs: {selectedNode.inputArtifacts.length ? selectedNode.inputArtifacts.join(", ") : "none"}</small>
          <small>Outputs: {selectedNode.outputArtifacts.length ? selectedNode.outputArtifacts.join(", ") : "none"}</small>
          <small>Parameters: {selectedNode.tunableParameters.length ? selectedNode.tunableParameters.join(", ") : "none"}</small>
          <small>Default view: {selectedNode.defaultView ?? "none"}</small>
          <span className={traceBadgeClass(selectedNode.traceSource)}>{traceBadgeLabel(selectedNode.traceSource)}</span>
          <small>Duration: {selectedNode.durationMs != null ? `${selectedNode.durationMs} ms` : "unknown"}</small>
          <small>Recipe values: {selectedNode.hasRecipeValues ? "available" : "none"}</small>
          <small>Last-run dirty: {selectedNode.lastRunDirtyState}</small>
          <small>Plan state: {selectedNode.planState === "none" ? "none" : planStateLabel(selectedNode.planState)}</small>
          <small>Selection role: {selectedNode.selectionRole}</small>
          <small>Runtime params: {compactObjectLines(selectedNode.parametersUsed)}</small>
          <small>Metrics: {compactObjectLines(selectedNode.metricsSummary)}</small>
          <small>Diagnostics: {compactObjectLines(selectedNode.diagnosticsSummary)}</small>
          <small>Warnings: {selectedNode.warnings?.length ? selectedNode.warnings.join(" | ") : "none"}</small>
          <small>Errors: {selectedNode.errors?.length ? selectedNode.errors.join(" | ") : "none"}</small>
          {selectedNode.comparisonSummary ? (
            <small>
              Comparison: params {selectedNode.comparisonSummary.parameterChanges} · artifacts {selectedNode.comparisonSummary.artifactChanges}
              {" · "}metrics {selectedNode.comparisonSummary.metricChanges} · diagnostics {selectedNode.comparisonSummary.diagnosticChanges}
            </small>
          ) : null}
        </section>
      ) : (
        <div className="empty-state compact">Select a graph node to inspect unit details.</div>
      )}
    </div>
  );
}
