import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { buildValidationDeepLink, parseValidationDeepLink } from "../src/validationDeepLink.ts";

test("validation deep links include suite, execution, case, and comparison context", () => {
  const href = buildValidationDeepLink({
    suite_id: "suite_1",
    execution_id: "execution_1",
    case_id: "case_1",
    comparison_id: "baseline_1",
  });
  assert.equal(
    href,
    "/validation?suite_id=suite_1&execution_id=execution_1&case_id=case_1&comparison_id=baseline_1",
  );
});

test("validation deep links omit blank or missing fields", () => {
  assert.equal(buildValidationDeepLink({}), "/validation");
  assert.equal(buildValidationDeepLink({ suite_id: "  " }), "/validation");
  assert.equal(buildValidationDeepLink({ suite_id: "suite_1" }), "/validation?suite_id=suite_1");
});

test("validation deep links parse back into routing hints", () => {
  const parsed = parseValidationDeepLink("?suite_id=suite_1&case_id=case_1");
  assert.deepEqual(parsed, {
    suite_id: "suite_1",
    execution_id: null,
    case_id: "case_1",
    comparison_id: null,
  });
});

test("validation deep links parse an empty search into all-null hints", () => {
  assert.deepEqual(parseValidationDeepLink(""), {
    suite_id: null,
    execution_id: null,
    case_id: null,
    comparison_id: null,
  });
});

test("Validation page prefers a linked suite over the first suite in the list", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/pages/ValidationPage.tsx"), "utf-8");
  assert.ok(source.includes("parseValidationDeepLink"));
  assert.ok(source.includes("deepLink.suite_id && rows.some((row) => row.id === deepLink.suite_id)"));
  assert.ok(source.includes("suiteId || linkedSuiteId || rows[0]?.id"));
});

test("Validation page prefers a linked execution over the suite's latest one", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/pages/ValidationPage.tsx"), "utf-8");
  assert.ok(source.includes("deepLink.execution_id"));
  assert.ok(source.includes("suiteHistory.find((x) => String(x.id) === deepLink.execution_id)"));
  assert.ok(source.includes("const latest = linked ?? suiteHistory.at(-1);"));
});

test("Validation page auto-opens a linked case/comparison exactly once", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/pages/ValidationPage.tsx"), "utf-8");
  assert.ok(source.includes("deepLinkComparisonAppliedRef"));
  assert.ok(source.includes("deepLinkComparisonAppliedRef.current = true;"));
  assert.ok(source.includes("setSelectedCell({ caseId: deepLink.case_id, comparisonIds: [deepLink.comparison_id] });"));
});

test("Studio links a newly added or matched case straight into Validation", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/pages/ProcessingLabPage.tsx"), "utf-8");
  assert.ok(source.includes('import { buildValidationDeepLink } from "../validationDeepLink";'));
  assert.ok(source.includes("buildValidationDeepLink({ suite_id: suiteId, case_id: existing.id })"));
  assert.ok(source.includes("buildValidationDeepLink({ suite_id: suiteId, case_id: created.id })"));
});
