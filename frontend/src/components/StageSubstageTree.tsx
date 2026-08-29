import type { ReactElement } from "react";

import type { ArtifactRelevance } from "./referenceArtifactRelevance";
import type { ReferenceViewScope, ResolvedStageSubstage } from "./stageSubstagePlan";

export type StageSubstageTreeViewEntry = {
  id: string;
  label: string;
  reason?: string;
  relevance?: ArtifactRelevance;
};

type Props = {
  substages: ResolvedStageSubstage[];
  activeScope: ReferenceViewScope;
  selectedSubstageId: string | null;
  activeTab: string;
  expandedPrimaryViews: StageSubstageTreeViewEntry[];
  expandedSecondaryViews: StageSubstageTreeViewEntry[];
  showAllViews: boolean;
  onToggleShowAllViews: () => void;
  showAllArtifacts: boolean;
  onToggleShowAllArtifacts: () => void;
  onSelectStrategyPath: () => void;
  onSelectSubstage: (substageId: string) => void;
  onSelectView: (viewId: string) => void;
  onFocusArtifact: (artifactId: string) => void;
  relevanceSummary: (substage: ResolvedStageSubstage) => ArtifactRelevance | null;
  renderRelevanceBadge: (relevance: ArtifactRelevance | undefined, reason: string | undefined) => ReactElement | null;
  renderIoRoleTag: (ioRole: ResolvedStageSubstage["supportingArtifacts"][number]["ioRole"]) => ReactElement | null;
};

export type StageSubstageDetailBodyProps = Pick<Props, "expandedPrimaryViews" | "expandedSecondaryViews" | "showAllViews" | "activeTab" | "onToggleShowAllViews" | "showAllArtifacts" | "onToggleShowAllArtifacts" | "onSelectView" | "renderRelevanceBadge" | "onFocusArtifact" | "renderIoRoleTag"> & {
  substage: ResolvedStageSubstage | null;
};

export type StageSubstageRailProps = Pick<Props, "substages" | "activeScope" | "selectedSubstageId" | "onSelectStrategyPath" | "onSelectSubstage" | "relevanceSummary" | "renderRelevanceBadge">;

export function StageSubstageDetailBody({
  expandedPrimaryViews,
  expandedSecondaryViews,
  showAllViews,
  activeTab,
  onToggleShowAllViews,
  showAllArtifacts,
  onToggleShowAllArtifacts,
  onSelectView,
  renderRelevanceBadge,
  substage,
  onFocusArtifact,
  renderIoRoleTag,
}: StageSubstageDetailBodyProps): ReactElement {
  const primaryArtifacts = substage?.supportingArtifacts.filter((entry) => entry.category !== "debug") ?? [];
  const secondaryArtifacts = substage?.supportingArtifacts.filter((entry) => entry.category === "debug") ?? [];
  return (
    <div className="stage-substage-tree-node-body">
      <div className="artifacts-group-block">
        <small>Views</small>
        <div className="studio-workspace-tabs">
          {expandedPrimaryViews.map((entry) => (
            <button
              key={entry.id}
              type="button"
              className={entry.id === activeTab ? "active" : ""}
              onClick={() => onSelectView(entry.id)}
            >
              {entry.label}
              {renderRelevanceBadge(entry.relevance, entry.reason)}
            </button>
          ))}
        </div>
      </div>
      {expandedSecondaryViews.length > 0 && (
        <div className="artifacts-group-block">
          <small>
            <button type="button" className="stage-view-show-all-toggle" onClick={onToggleShowAllViews}>
              {showAllViews ? "Hide extra views" : `Show ${expandedSecondaryViews.length} more views`}
            </button>
          </small>
          {showAllViews && (
            <div className="studio-workspace-tabs">
              {expandedSecondaryViews.map((entry) => (
                <button
                  key={entry.id}
                  type="button"
                  className={entry.id === activeTab ? "active" : ""}
                  onClick={() => onSelectView(entry.id)}
                >
                  {entry.label}
                  {renderRelevanceBadge(entry.relevance, entry.reason)}
                </button>
              ))}
            </div>
          )}
        </div>
      )}
      {substage && (
        <div className="artifacts-group-block">
          <small>Supporting artifacts</small>
          <div className="substage-artifact-list">
            {primaryArtifacts.map((entry) => (
              <button
                key={entry.artifactId}
                type="button"
                className={`footer-chip ${entry.category === "missing" ? "warning" : ""}`}
                disabled={!entry.artifact}
                onClick={() => onFocusArtifact(entry.artifactId)}
                title="Jump to the parameters for this artifact"
              >
                {entry.artifactId}
                {renderIoRoleTag(entry.ioRole)}
                {renderRelevanceBadge(entry.category, entry.reason)}
              </button>
            ))}
          </div>
          {secondaryArtifacts.length > 0 && (
            <>
              <small>
                <button type="button" className="stage-view-show-all-toggle" onClick={onToggleShowAllArtifacts}>
                  {showAllArtifacts ? "Hide extra artifacts" : `Show ${secondaryArtifacts.length} more artifacts`}
                </button>
              </small>
              {showAllArtifacts && (
                <div className="substage-artifact-list">
                  {secondaryArtifacts.map((entry) => (
                    <button
                      key={entry.artifactId}
                      type="button"
                      className={`footer-chip ${entry.category === "missing" ? "warning" : ""}`}
                      disabled={!entry.artifact}
                      onClick={() => onFocusArtifact(entry.artifactId)}
                      title="Jump to the parameters for this artifact"
                    >
                      {entry.artifactId}
                      {renderIoRoleTag(entry.ioRole)}
                      {renderRelevanceBadge(entry.category, entry.reason)}
                    </button>
                  ))}
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default function StageSubstageTree({
  substages,
  activeScope,
  selectedSubstageId,
  activeTab,
  expandedPrimaryViews,
  expandedSecondaryViews,
  showAllViews,
  onToggleShowAllViews,
  showAllArtifacts,
  onToggleShowAllArtifacts,
  onSelectStrategyPath,
  onSelectSubstage,
  onSelectView,
  onFocusArtifact,
  relevanceSummary,
  renderRelevanceBadge,
  renderIoRoleTag,
}: Props): ReactElement {
  const strategyPathActive = activeScope === "strategy_path";
  return (
    <nav className="stage-substage-tree" aria-label="Reference-surface subprocesses">
      <div className="stage-substage-tree-node">
        <button
          type="button"
          className={`stage-substage-tree-node-header ${strategyPathActive ? "active" : ""}`}
          onClick={onSelectStrategyPath}
        >
          <strong>Strategy path</strong>
        </button>
        {strategyPathActive && (
          <StageSubstageDetailBody
            substage={null}
            expandedPrimaryViews={expandedPrimaryViews}
            expandedSecondaryViews={expandedSecondaryViews}
            showAllViews={showAllViews}
            activeTab={activeTab}
            onToggleShowAllViews={onToggleShowAllViews}
            showAllArtifacts={showAllArtifacts}
            onToggleShowAllArtifacts={onToggleShowAllArtifacts}
            onSelectView={onSelectView}
            renderRelevanceBadge={renderRelevanceBadge}
            onFocusArtifact={onFocusArtifact}
            renderIoRoleTag={renderIoRoleTag}
          />
        )}
      </div>
      {substages.map((substage) => {
        const active = activeScope === "substage" && selectedSubstageId === substage.substageId;
        return (
          <div key={substage.substageId} className="stage-substage-tree-node">
            <button
              type="button"
              className={`stage-substage-tree-node-header ${active ? "active" : ""}`}
              onClick={() => onSelectSubstage(substage.substageId)}
              title={substage.description ?? ""}
            >
              <strong>{substage.label}</strong>
              {renderRelevanceBadge(relevanceSummary(substage) ?? undefined, undefined)}
            </button>
            {active && (
              <StageSubstageDetailBody
                substage={substage}
                expandedPrimaryViews={expandedPrimaryViews}
                expandedSecondaryViews={expandedSecondaryViews}
                showAllViews={showAllViews}
                activeTab={activeTab}
                onToggleShowAllViews={onToggleShowAllViews}
                showAllArtifacts={showAllArtifacts}
                onToggleShowAllArtifacts={onToggleShowAllArtifacts}
                onSelectView={onSelectView}
                renderRelevanceBadge={renderRelevanceBadge}
                onFocusArtifact={onFocusArtifact}
                renderIoRoleTag={renderIoRoleTag}
              />
            )}
          </div>
        );
      })}
    </nav>
  );
}
