import { formatBoundaryLine, formatChildRunsSummary, formatParentLineageLine, formatReusedRerunLine } from "./runLineageModel";
import { useRunLineage } from "./useRunLineage";

type Props = {
  pipelineId: string;
  runId: string;
  onSelectRun: (runId: string) => void;
  onOpenComparison: (comparisonId: string) => void;
  onLineageRefreshed?: () => void;
  activeRunTypeLabel?: string | null;
};

export default function RunLineageBlock({ pipelineId, runId, onSelectRun, onOpenComparison, onLineageRefreshed, activeRunTypeLabel = null }: Props) {
  const { lineage, loading, generating, error, generateComparison } = useRunLineage(pipelineId, runId);

  if (loading) return <div className="run-lineage-block empty-state compact">Loading lineage…</div>;
  if (!lineage) return null;

  const parentLine = formatParentLineageLine(lineage);
  const boundaryLine = formatBoundaryLine(lineage.parent ?? null);
  const reusedRerunLine = formatReusedRerunLine(lineage.parent ?? null);

  async function handleGenerateComparison() {
    const comparison = await generateComparison();
    onLineageRefreshed?.();
    if (comparison?.comparison_id) onOpenComparison(comparison.comparison_id);
  }

  return (
    <div className="run-lineage-block">
      {activeRunTypeLabel ? <strong>Active run: {activeRunTypeLabel}</strong> : null}
      {lineage.parent ? (
        <div className="run-lineage-parent">
          {parentLine ? <strong>{parentLine}</strong> : null}
          {boundaryLine ? <small>{boundaryLine}</small> : null}
          {reusedRerunLine ? <small>{reusedRerunLine}</small> : null}
          <button type="button" onClick={() => lineage.parent && onSelectRun(lineage.parent.run_id)}>
            Open parent run
          </button>
          {lineage.comparison_id ? (
            <button type="button" onClick={() => lineage.comparison_id && onOpenComparison(lineage.comparison_id)}>
              Reopen comparison to parent
            </button>
          ) : (
            <button type="button" disabled={generating} onClick={() => void handleGenerateComparison()}>
              {generating ? "Generating comparison…" : "Generate comparison to parent"}
            </button>
          )}
          {error ? <small className="run-lineage-error">{error}</small> : null}
        </div>
      ) : (
        <div className="run-lineage-parent">
          <small>This run has no parent — it is a full run.</small>
        </div>
      )}
      <div className="run-lineage-children">
        <strong>{formatChildRunsSummary(lineage.children)}</strong>
        {lineage.children.length ? (
          <ul className="run-lineage-children-list">
            {lineage.children.map((child) => (
              <li key={child.run_id}>
                <button type="button" onClick={() => onSelectRun(child.run_id)}>
                  {child.run_id} ({child.boundary_stage_id ?? "unknown boundary"})
                </button>
              </li>
            ))}
          </ul>
        ) : null}
      </div>
    </div>
  );
}
