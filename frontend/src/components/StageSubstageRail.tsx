import type { ReactElement } from "react";

import {
  StageSubstageDetailBody,
  type StageSubstageDetailBodyProps,
  type StageSubstageRailProps,
} from "./StageSubstageTree";

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
