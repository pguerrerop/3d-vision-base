export type PlanContext = {
  parentRunId: string | null;
  recipeId: string | null;
  recipeVersion: number | null;
};

export type StaleWarningInputs = {
  planExists: boolean;
  planIsStale: boolean;
  planContext: PlanContext | null;
  currentRecipeId: string | null;
  currentRecipeVersion: number | null;
  activeRunId: string | null;
  activeRunExecutionMode: string | null;
  activeRunParentRunId: string | null;
  activeRunComparisonId: string | null;
  activeRunContractFingerprint: string | null;
  currentContractFingerprint: string | null;
};

function isPartialRerunMode(executionMode: string | null): boolean {
  return executionMode === "rerun_from_public_stage_boundary";
}

export function computeStaleWarnings(inputs: StaleWarningInputs): string[] {
  const warnings: string[] = [];

  if (inputs.planExists && inputs.planIsStale) {
    warnings.push("Current edits have changed since this partial rerun plan was created — re-plan before executing.");
  }

  if (inputs.planExists && inputs.planContext) {
    const { recipeId, recipeVersion, parentRunId } = inputs.planContext;
    if (recipeId && (recipeId !== inputs.currentRecipeId || recipeVersion !== inputs.currentRecipeVersion)) {
      warnings.push("Selected recipe changed since this plan was created — re-plan before executing.");
    }
    if (parentRunId && inputs.activeRunId && parentRunId !== inputs.activeRunId) {
      warnings.push(`Active run has changed since this plan was created (planned against ${parentRunId}).`);
    }
  }

  if (isPartialRerunMode(inputs.activeRunExecutionMode)) {
    if (inputs.planExists && inputs.planContext?.parentRunId && inputs.activeRunParentRunId
      && inputs.planContext.parentRunId !== inputs.activeRunParentRunId) {
      warnings.push("Active run is a child of a different parent than the current plan.");
    }
    if (!inputs.activeRunComparisonId) {
      warnings.push("Comparison to parent is missing for this child run — generate one to inspect what changed.");
    }
    if (inputs.activeRunContractFingerprint && inputs.currentContractFingerprint
      && inputs.activeRunContractFingerprint !== inputs.currentContractFingerprint) {
      warnings.push("Processing-unit contract fingerprint has changed since this child run was executed.");
    }
  }

  return warnings;
}
