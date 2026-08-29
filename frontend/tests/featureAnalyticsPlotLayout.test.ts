import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import {
  HISTOGRAM_SCROLL_GROUP_THRESHOLD,
  buildFeatureAnalyticsSummaryBar,
  histogramPlotLayout,
  histogramScrollHint,
  resolveMatchingViewMode,
  summarizeMatchingObjects,
} from "../src/featureAnalyticsPlotLayout.ts";

test("histogramPlotLayout fits up to five groups without scrolling", () => {
  for (let groupCount = 1; groupCount <= HISTOGRAM_SCROLL_GROUP_THRESHOLD; groupCount += 1) {
    const layout = histogramPlotLayout(groupCount);
    assert.equal(layout.scrollable, false);
    assert.equal(layout.contentHeightPx, groupCount * 92 + Math.max(0, groupCount - 1) * 6);
  }
});

test("histogramPlotLayout enables scroll for more than five groups", () => {
  const layout = histogramPlotLayout(12);
  assert.equal(layout.scrollable, true);
  assert.equal(layout.maxScrollHeightPx, histogramPlotLayout(HISTOGRAM_SCROLL_GROUP_THRESHOLD).contentHeightPx);
  assert.ok(layout.contentHeightPx > layout.maxScrollHeightPx);
});

test("histogramScrollHint appears only when scrollable groups are partially visible", () => {
  assert.equal(histogramScrollHint(4, 4), null);
  assert.equal(histogramScrollHint(5, 5), null);
  assert.equal(histogramScrollHint(12, 12), null);
  assert.equal(histogramScrollHint(5, 12), "Showing 5 of 12 groups · scroll to view more");
});

test("Feature Analytics plot keeps histogram bands outside overflow hidden clipping", () => {
  const pageSource = fs.readFileSync(path.resolve(process.cwd(), "src/pages/FeatureAnalyticsPage.tsx"), "utf-8");
  const cssSource = fs.readFileSync(path.resolve(process.cwd(), "src/styles.css"), "utf-8");

  assert.ok(pageSource.includes("feature-histogram-scroll"));
  assert.ok(pageSource.includes("histogramPlotLayout"));
  assert.ok(pageSource.includes('className="feature-band-label"') || pageSource.includes("feature-band-label"));
  assert.ok(!cssSource.match(/\.feature-analytics-plot\s*\{[^}]*overflow:\s*hidden/s));
  assert.ok(cssSource.includes(".feature-histogram-scroll.is-scrollable"));
  assert.ok(cssSource.includes("overflow-y: auto"));
});

test("Feature Analytics plot fixture includes four labeled superclass groups", () => {
  const pageSource = fs.readFileSync(path.resolve(process.cwd(), "src/pages/FeatureAnalyticsPage.tsx"), "utf-8");
  assert.ok(pageSource.includes("feature-histogram-grid"));
  assert.ok(pageSource.includes("group.group"));
  assert.ok(pageSource.includes("feature-analytics-plot-axis"));
});

test("buildFeatureAnalyticsSummaryBar formats compact scope and filtered counts", () => {
  const summary = buildFeatureAnalyticsSummaryBar({
    mlSetName: "balls_scrap_2026_05_25_29_table_v1",
    scopeTakeCount: 223,
    scopePhysicalObjectCount: 42,
    filteredTakeCount: 136,
    filteredPhysicalObjectCount: 26,
    filteredObjectCount: 178,
  });
  assert.equal(
    summary,
    "ML set balls_scrap_2026_05_25_29_table_v1 · Scope 223 takes · 42 physical objects | Filtered 136 takes · 26 physical objects · 178 objects",
  );
});

test("summarizeMatchingObjects counts unique takes and physical objects", () => {
  const summary = summarizeMatchingObjects([
    { take_id: "t1", physical_object_id: "obj1" },
    { take_id: "t1", physical_object_id: "obj1" },
    { take_id: "t2", physical_object_id: "obj2" },
  ]);
  assert.deepEqual(summary, { objectCount: 3, takeCount: 2, physicalObjectCount: 2 });
});

test("resolveMatchingViewMode falls back to all filtered without a selected bin", () => {
  assert.equal(resolveMatchingViewMode("selected_bin", false), "all_filtered");
  assert.equal(resolveMatchingViewMode("selected_bin", true), "selected_bin");
});

test("Feature Analytics workspace uses compact header and matching objects table", () => {
  const pageSource = fs.readFileSync(path.resolve(process.cwd(), "src/pages/FeatureAnalyticsPage.tsx"), "utf-8");
  const cssSource = fs.readFileSync(path.resolve(process.cwd(), "src/styles.css"), "utf-8");

  assert.ok(pageSource.includes("feature-scope-bar"));
  assert.ok(pageSource.includes("feature-analytics-toolbar"));
  assert.ok(pageSource.includes("Matching objects"));
  assert.ok(pageSource.includes("feature-matching-objects"));
  assert.ok(pageSource.includes("feature-analytics-workspace"));
  assert.ok(pageSource.includes("feature-matching-mode-toggle"));
  assert.ok(pageSource.includes("Selected bin"));
  assert.ok(pageSource.includes("All filtered"));
  assert.ok(pageSource.includes("Thumbnail request"));
  assert.ok(pageSource.includes("classification_overlay"));
  assert.ok(pageSource.includes("objectThumbnailInfo"));
  assert.ok(pageSource.includes("has-selected-bin"));
  assert.ok(cssSource.includes(".feature-analytics-workspace"));
  assert.ok(cssSource.includes(".feature-matching-table-wrap"));
  assert.ok(cssSource.includes("flex-direction: column"));
  assert.ok(cssSource.includes("position: sticky"));
  assert.ok(cssSource.includes("100dvh"));
});
