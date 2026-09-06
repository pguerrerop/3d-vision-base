import type { ValidationSuiteCase } from "../../api/client";

/** Suites a just-approved reference could join: same pipeline, not archived. */
export function eligibleSuitesForPipeline<T extends { pipeline_id: string; status: string }>(
  suites: T[],
  pipelineId: string,
): T[] {
  return suites.filter((suite) => suite.pipeline_id === pipelineId && suite.status !== "archived");
}

/**
 * add_case() has no dedupe of its own -- calling it twice for the same take
 * creates two cases. Studio checks first so "Add to regression suite" clicked
 * twice reports the existing case instead of silently doubling it.
 */
export function findExistingCaseForTake(
  cases: ValidationSuiteCase[],
  takeId: string,
): ValidationSuiteCase | null {
  return cases.find((item) => item.take_id === takeId) ?? null;
}
