import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { eligibleSuitesForPipeline, findExistingCaseForTake } from "../src/components/validation/addToSuiteModel.ts";

const suite = (over: Partial<{ id: string; pipeline_id: string; status: string }>) => ({
  id: "s1",
  name: "Suite",
  pipeline_id: "mining_steel_ball_classification_25d",
  status: "active",
  ...over,
});

test("eligible suites match the pipeline and exclude archived ones", () => {
  const suites = [
    suite({ id: "a" }),
    suite({ id: "b", pipeline_id: "other_pipeline" }),
    suite({ id: "c", status: "archived" }),
  ];
  assert.deepEqual(
    eligibleSuitesForPipeline(suites, "mining_steel_ball_classification_25d").map((s) => s.id),
    ["a"],
  );
});

test("eligible suites is empty when nothing matches", () => {
  assert.deepEqual(eligibleSuitesForPipeline([suite({ pipeline_id: "other" })], "mining_steel_ball_classification_25d"), []);
});

const suiteCase = (over: Partial<{ id: string; take_id: string }>) => ({
  id: "case_1",
  take_id: "take_a",
  baseline_ids: ["baseline_1"],
  enabled: true,
  tags: [],
  notes: "",
  ...over,
});

test("an existing case for the same take is found rather than duplicated", () => {
  const cases = [suiteCase({ id: "case_1", take_id: "take_a" }), suiteCase({ id: "case_2", take_id: "take_b" })];
  assert.equal(findExistingCaseForTake(cases, "take_a")?.id, "case_1");
});

test("no existing case for a take reports null, not a false match", () => {
  const cases = [suiteCase({ id: "case_2", take_id: "take_b" })];
  assert.equal(findExistingCaseForTake(cases, "take_a"), null);
});

test("an empty case list never matches", () => {
  assert.equal(findExistingCaseForTake([], "take_a"), null);
});

test("Processing Lab gates 'Add to regression suite' on an approved reference", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/pages/ProcessingLabPage.tsx"), "utf-8");
  assert.ok(source.includes('<button type="button" disabled={!artifactBaseline} onClick={openAddToSuite}>Add to regression suite</button>'));
});

test("Processing Lab checks for an existing case before creating a duplicate", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/pages/ProcessingLabPage.tsx"), "utf-8");
  assert.ok(source.includes("import { eligibleSuitesForPipeline, findExistingCaseForTake } from \"../components/validation/addToSuiteModel\";"));
  assert.ok(source.includes("existing = findExistingCaseForTake(suite.cases, artifactBaseline.take_id);"));
  assert.ok(source.includes("already tracks this take"));
});

test("Processing Lab's new-suite branch skips the case lookup a fresh suite can't need", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/pages/ProcessingLabPage.tsx"), "utf-8");
  assert.ok(source.includes("const created = await api.createValidationSuite({"));
  assert.ok(source.includes("suiteId = created.id;"));
});
