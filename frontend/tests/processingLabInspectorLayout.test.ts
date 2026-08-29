import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

test("Processing Lab keeps the desktop inspector open and only allows collapsing in overlay layouts", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/pages/ProcessingLabPage.tsx"), "utf-8");

  assert.ok(source.includes("const inspectorCanCollapse = rightPanelOverlay;"));
  assert.ok(source.includes("const inspectorOpen = inspectorCanCollapse ? inspectorExpanded : true;"));
  assert.ok(source.includes('className={`studio-inspector-shell ${inspectorOpen ? "expanded" : "collapsed"} ${panelWidthClass} ${rightPanelOverlay ? "overlay" : ""} ${inspectorVisible ? "is-visible" : "is-hidden"}`}'));
  assert.ok(source.includes("{inspectorCanCollapse && ("));
});
