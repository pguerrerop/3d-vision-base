import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

// pipelineWalkthroughModel.ts transitively imports processingUnitModel.ts/stageSubstagePlan.ts,
// whose own internal imports use ".js" extensions pointing at sibling ".ts" files -- a pattern
// Node's --experimental-strip-types does not resolve for real (non-type-only) imports. No test
// in this codebase imports those two files as real modules for the same reason (see
// stageSubstagePlanContract.test.ts, contractTunePanel.test.ts); source assertions are the
// established way to cover this layer instead of a live browser check for every change.

function readSource(): string {
  return fs.readFileSync(path.resolve(process.cwd(), "src/components/pipelineWalkthroughModel.ts"), "utf-8");
}

test("buildPipelineWalkthrough reuses the existing relevance-tagged substage plan instead of a new heuristic", () => {
  const source = readSource();
  assert.ok(source.includes("resolveStageSubstagePlan("));
  assert.ok(source.includes("processingUnitsForStage("));
  assert.ok(source.includes("artifactsForStage(detail, stage.id)"));
  assert.ok(source.includes("artifactList(detail)"));
});

test("buildPipelineWalkthrough computes substage-level relevance from real artifact presence, not the legacy view-registry coverage flag", () => {
  const source = readSource();
  assert.ok(source.includes('entry.category === "inactive_strategy" || entry.category === "legacy"'));
  assert.ok(source.includes("substage.supportingArtifacts.length > 0"));
  assert.ok(source.includes('allMissing = substage.supportingArtifacts.every((entry) => entry.category === "missing")'));
  // views' "missing" means "no legacy semantic-view-registry entry", not "artifact absent" --
  // must not be used to compute the substage-level MISSING badge.
  assert.ok(!source.includes('allMissing = substage.views.every'));
});

test("buildPipelineWalkthrough flags parameters whose active_when doesn't match the take's real stage_params", () => {
  const source = readSource();
  assert.ok(source.includes("function parameterRelevance("));
  assert.ok(source.includes("param.active_when"));
  assert.ok(source.includes("values[dep] !== expected"));
  assert.ok(source.includes("allStageParams(detail)"));
  assert.ok(source.includes('result["stage_params"]'));
});

test("buildPipelineWalkthrough resolves each stage's parameter values under its own runtime_stage_params_key, not a flat namespace", () => {
  const source = readSource();
  assert.ok(source.includes("function valuesForStage("));
  assert.ok(source.includes('schema?.["runtime_stage_params_key"]'));
  assert.ok(source.includes("valuesForStage(stage, stageParams)"));
});

test("buildPipelineWalkthrough reads the take's actual current parameter value, falling back to the contract default", () => {
  const source = readSource();
  assert.ok(source.includes("values[param.id] !== undefined ? values[param.id] : param.default"));
});

test("buildPipelineWalkthrough assigns parameters to their declared group and keeps the rest ungrouped", () => {
  const source = readSource();
  assert.ok(source.includes("claimed.add(key)"));
  assert.ok(source.includes("unit.parameters.filter((p) => !claimed.has(p.id))"));
});

test("walkthrough artifacts retain their contract provenance for the full-size inspector", () => {
  const source = readSource();
  assert.ok(source.includes("export type WalkthroughArtifactProvenance"));
  assert.ok(source.includes("function artifactCalculation("));
  assert.ok(source.includes("function provenanceUnitForArtifact("));
  assert.ok(source.includes("BG_AND_STRIPES_COMPONENT_ARTIFACTS"));
  assert.ok(source.includes("function artifactProvenance("));
  assert.ok(source.includes("function artifactInputs("));
  assert.ok(source.includes("function artifactInputUsage("));
  assert.ok(source.includes("function artifactMethodSteps("));
  assert.ok(source.includes("const selectedParameters = directlyAffecting.length > 0 ? directlyAffecting : artifactParameters(unit, artifact, parameters);"));
  assert.ok(source.includes("methodSteps: artifactMethodSteps(unit, artifact, inputs, selectedParameters)"));
  assert.ok(source.includes("provenance: artifactProvenance(provenanceUnit, artifact, provenanceParameters)"));
});

test("walkthrough root artifacts treat downstream-used diagnostics as outputs", () => {
  const source = readSource();
  assert.ok(source.includes('artifact.role === "input"'));
  assert.ok(source.includes('(artifact.role === "final" || (artifact.role === "diagnostic" && (artifact.feeds_into?.length ?? 0) > 0))'));
  assert.ok(source.includes('? "output"'));
});

test("buildPipelineWalkthrough surfaces downstream_effects and tuning_hints verbatim from the contract unit", () => {
  const source = readSource();
  assert.ok(source.includes("downstreamEffects: unit.downstream_effects ?? []"));
  assert.ok(source.includes("tuningHints: unit.tuning_hints ?? []"));
});

test("buildPipelineWalkthrough orders a substage's artifacts by real production timestamp, not contract declaration order", () => {
  const source = readSource();
  assert.ok(source.includes("function sortArtifactsByProductionOrder("));
  assert.ok(source.includes("entry.artifact?.created_at ?? null"));
  // Date.parse/getTime() truncates to millisecond precision; real per-artifact timestamps are
  // often only microseconds apart within the same millisecond, so a numeric Date.parse comparison
  // would silently collapse to a no-op for exactly the case this sort exists to handle. Must
  // compare the raw fixed-width ISO-8601 strings directly instead.
  assert.ok(!source.includes("Date.parse(entry.artifact.created_at)"));
  assert.ok(source.includes("a.time < b.time ? -1 : a.time > b.time ? 1 : a.index - b.index"));
  // never-produced (missing) artifacts have no timestamp and must sort after every real one,
  // not be silently dropped or interleaved arbitrarily.
  assert.ok(source.includes("if (a.time != null) return -1;"));
  assert.ok(source.includes("if (b.time != null) return 1;"));
  assert.ok(source.includes("artifacts: sortArtifactsByProductionOrder(artifacts).map((artifact) => {"));
});

test("sortArtifactsByProductionOrder sorts by real created_at, correctly ordering microsecond-apart timestamps within the same millisecond", () => {
  // Self-contained behavioral check of the sort comparator, copied verbatim from the source
  // (pipelineWalkthroughModel.ts can't be imported directly here -- see file header) so the
  // actual ordering logic gets a real assertion, not just a text-presence check.
  type Entry = { artifactId: string; artifact: { created_at?: string | null } | null };
  function sortArtifactsByProductionOrder(artifacts: Entry[]): Entry[] {
    return artifacts
      .map((entry, index) => ({ entry, index, time: entry.artifact?.created_at ?? null }))
      .sort((a, b) => {
        if (a.time != null && b.time != null) return a.time < b.time ? -1 : a.time > b.time ? 1 : a.index - b.index;
        if (a.time != null) return -1;
        if (b.time != null) return 1;
        return a.index - b.index;
      })
      .map(({ entry }) => entry);
  }
  // Real-world shape: all three share the same millisecond ("...503...Z"), differing only in the
  // last 3 microsecond digits -- exactly what collapsed to a no-op under Date.parse.
  const input: Entry[] = [
    { artifactId: "declared_third", artifact: { created_at: "2026-07-13T18:20:02.503840Z" } },
    { artifactId: "declared_first", artifact: { created_at: "2026-07-13T18:20:02.503679Z" } },
    { artifactId: "never_produced", artifact: null },
    { artifactId: "declared_second", artifact: { created_at: "2026-07-13T18:20:02.503747Z" } },
  ];
  const result = sortArtifactsByProductionOrder(input).map((entry) => entry.artifactId);
  assert.deepEqual(result, ["declared_first", "declared_second", "declared_third", "never_produced"]);
});
