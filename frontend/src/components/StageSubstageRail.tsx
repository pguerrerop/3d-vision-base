import type { ReactElement } from "react";

import {
  StageSubstageDetailBody,
  type StageSubstageDetailBodyProps,
  type StageSubstageRailProps,
} from "./StageSubstageTree";
import type { ResolvedStageSubstage } from "./stageSubstagePlan";

const MAX_OUTPUT_LABELS_SHOWN = 2;

function substageOutputSummary(substage: ResolvedStageSubstage): { shown: string; full: string } | null {
  const seen = new Set<string>();
  const labels = substage.supportingArtifacts
    .filter((entry) => entry.ioRole === "output" && !seen.has(entry.artifactId) && seen.add(entry.artifactId))
    .map((entry) => entry.artifact?.title ?? entry.artifactId);
  if (labels.length === 0) return null;
  const extra = labels.length - MAX_OUTPUT_LABELS_SHOWN;
  const shown = extra > 0 ? `${labels.slice(0, MAX_OUTPUT_LABELS_SHOWN).join(", ")} +${extra} more` : labels.join(", ");
  return { shown, full: labels.join(", ") };
}

export function StageSubstageRailList({
  substages,
  activeScope,
  selectedSubstageId,
  onSelectStrategyPath,
  onSelectSubstage,
  relevanceSummary,
  renderRelevanceBadge,
}: StageSubstageRailProps): ReactElement {
  const strategyPathActive = activeScope === "strategy_path";
  return (
    <div className="studio-stage-rail-list">
      <button
        type="button"
        className={`studio-stage-rail-row ${strategyPathActive ? "active" : ""}`}
        onClick={onSelectStrategyPath}
      >
        <span className="studio-stage-rail-copy">
          <strong>Strategy path</strong>
        </span>
      </button>
      {substages.map((substage) => {
        const active = activeScope === "substage" && selectedSubstageId === substage.substageId;
        const outputSummary = substageOutputSummary(substage);
        return (
          <button
            key={substage.substageId}
            type="button"
            className={`studio-stage-rail-row ${active ? "active" : ""}`}
            onClick={() => onSelectSubstage(substage.substageId)}
            title={substage.description ?? ""}
          >
            <span className="studio-stage-rail-copy">
              <strong>{substage.label}</strong>
              {outputSummary && (
                <small className="studio-stage-rail-output" title={`Output: ${outputSummary.full}`}>
                  → {outputSummary.shown}
                </small>
              )}
            </span>
            {renderRelevanceBadge(relevanceSummary(substage) ?? undefined, undefined)}
          </button>
        );
      })}
    </div>
  );
}

type DetailStripProps = StageSubstageDetailBodyProps & {
  activeLabel: string;
  helpSlot?: ReactElement | null;
};

export function StageSubstageDetailStrip({
  activeLabel,
  helpSlot,
  ...bodyProps
}: DetailStripProps): ReactElement {
  return (
    <section className="studio-substage-detail-strip">
      <div className="stage-path-header">
        <strong>{activeLabel}</strong>
        {helpSlot ?? null}
      </div>
      <StageSubstageDetailBody {...bodyProps} />
    </section>
  );
}
