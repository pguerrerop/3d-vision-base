import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

test("Processing Lab uses explicit reference view scope for Detect reference surface navigation", () => {
  const pageSource = fs.readFileSync(path.resolve(process.cwd(), "src/pages/ProcessingLabPage.tsx"), "utf-8");
  const treeSource = fs.readFileSync(path.resolve(process.cwd(), "src/components/StageSubstageTree.tsx"), "utf-8");

  assert.ok(pageSource.includes('const [referenceViewScope, setReferenceViewScope] = useState<ReferenceViewScope | null>(null);'));
  assert.ok(pageSource.includes('resolveReferenceViewSelection(stageSubstagePlan, referenceViewScope, selectedSubstageId, activeTab)'));
  assert.ok(pageSource.includes('onSelectStrategyPath={activateStrategyPath}'));
  assert.ok(pageSource.includes('onSelectSubstage={activateReferenceSubstage}'));
  assert.ok(treeSource.includes('activeScope === "strategy_path"'));
  assert.ok(treeSource.includes('activeScope === "substage" && selectedSubstageId === substage.substageId'));
  assert.ok(treeSource.includes('onClick={onSelectStrategyPath}'));
  assert.ok(treeSource.includes('onClick={() => onSelectSubstage(substage.substageId)}'));
});
