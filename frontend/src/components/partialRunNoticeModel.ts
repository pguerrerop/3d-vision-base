export function partialRunNoticeText(params: {
  parentRunId?: string | null;
  boundaryStageId?: string | null;
}): string {
  const parent = params.parentRunId ?? "unknown parent run";
  const boundary = params.boundaryStageId ?? "an earlier stage";
  return (
    `This is a partial rerun — an independent saved result branching from ${parent} at the ${boundary} stage. `
    + "The parent run is unchanged. Stages before the boundary were reused (artifacts copied, not recomputed)."
  );
}

export function isPartialRerunExecutionMode(executionMode: string | null | undefined): boolean {
  return executionMode === "rerun_from_public_stage_boundary";
}
