import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

const STRIPE_FILTER_VIEW_ORDER = [
  "pre_stripe_support",
  "belt_bg",
  "belt_baseline",
  "belt_altitude_plot",
  "belt_stripes_tophat",
  "belt_stripes_shape",
  "belt_wide_object",
  "belt_above_belt",
  "belt_base",
  "belt_stripes",
  "unknown_low_gradient",
  "surface_suppression",
  "object_search_domain",
  "removed_by_stripe_filter",
  "stripe_filtered_support",
];

test("processing unit model declares stripe filter semantic view order", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/components/processingUnitModel.ts"), "utf-8");
  assert.ok(source.includes("export const STRIPE_FILTER_VIEW_ORDER"));
  assert.ok(source.includes('"pre_stripe_support"'));
  assert.ok(source.includes('"stripe_filtered_support"'));
  assert.ok(source.includes("isStripeFilterUnit"));
  assert.ok(source.includes("orderViewIds"));
  for (let index = 1; index < STRIPE_FILTER_VIEW_ORDER.length; index += 1) {
    const previous = source.indexOf(`"${STRIPE_FILTER_VIEW_ORDER[index - 1]}"`, source.indexOf("STRIPE_FILTER_VIEW_ORDER"));
    const current = source.indexOf(`"${STRIPE_FILTER_VIEW_ORDER[index]}"`, source.indexOf("STRIPE_FILTER_VIEW_ORDER"));
    assert.ok(previous >= 0 && current > previous, `expected ${STRIPE_FILTER_VIEW_ORDER[index - 1]} before ${STRIPE_FILTER_VIEW_ORDER[index]}`);
  }
});

test("Processing Lab belt stripe groups follow the semantic view order", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/pages/ProcessingLabPage.tsx"), "utf-8");
  assert.ok(source.includes('"pre_stripe_support"'));
  assert.ok(source.includes('"belt_baseline"'));
  assert.ok(source.includes('"belt_altitude_plot"'));
  const blobGroupMatch = source.match(/id: "belt_stripes"[\s\S]*?views: pick\(\[([\s\S]*?)\]\)/);
  assert.ok(blobGroupMatch, "expected belt_stripes view group");
  const listedViews = blobGroupMatch[1]
    .split("\n")
    .map((line) => line.trim().replace(/^"|",?$/g, ""))
    .filter(Boolean);
  assert.deepEqual(listedViews.slice(0, STRIPE_FILTER_VIEW_ORDER.length), STRIPE_FILTER_VIEW_ORDER);
});
