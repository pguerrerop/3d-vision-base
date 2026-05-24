import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

import { assertSemanticMatch, resolveHeightArtifact, resolveHeightColorMapping } from "../src/components/heightSemantics.ts";
import { cssGradientForColorMap, sampleColorMap, scalarToRgb } from "../src/components/colormap.ts";
import type { StudioArtifact } from "../src/components/studioWorkspaceModel.ts";

test("HeightLegend consumes semantic metadata props contract", () => {
  const legendPath = path.resolve(process.cwd(), "src/components/HeightLegend.tsx");
  const source = fs.readFileSync(legendPath, "utf8");
  assert.ok(source.includes("semanticField"));
  assert.ok(source.includes("semanticLabel"));
  assert.ok(source.includes("units"));
  assert.ok(source.includes("minValue"));
  assert.ok(source.includes("maxValue"));
  assert.ok(source.includes("positiveDirection"));
  assert.ok(source.includes("authoritative"));
  assert.ok(source.includes("compact"));
  assert.ok(source.includes("showTicks"));
  assert.ok(source.includes("percentileLabels"));
  assert.ok(source.includes("Authoritative"));
});

test("assertSemanticMatch warns on mismatched hover semantics", () => {
  const mismatch = assertSemanticMatch("height_above_belt", "raw_sensor_z");
  assert.equal(mismatch.ok, false);
  assert.ok(String(mismatch.warning).includes("Semantic mismatch"));
  const ok = assertSemanticMatch("height_above_belt", "height_above_belt");
  assert.equal(ok.ok, true);
  assert.equal(ok.warning, null);
});

test("semantic-first artifact resolution returns canonical artifacts", () => {
  const artifacts: StudioArtifact[] = [
    {
      artifact_id: "height_above_belt_raster",
      stage_id: "normalize_heights_to_plane",
      kind: "file",
      title: "Height above belt semantic raster",
      path: "height_above_belt.values.f32",
      preview_available: false,
      metadata: { semantic_field: "height_above_belt", numeric_source: true },
    } as StudioArtifact,
    {
      artifact_id: "legacy_norm",
      stage_id: "normalize_heights_to_plane",
      kind: "file",
      title: "Legacy normalized raster",
      path: "normalized_heightmap.values.f32",
      preview_available: false,
      metadata: {},
    } as StudioArtifact,
  ];
  const canonical = resolveHeightArtifact(artifacts, { semanticField: "height_above_belt", representation: "numeric_raster" });
  assert.equal(canonical.artifact?.artifact_id, "height_above_belt_raster");
  assert.equal(canonical.warning, null);
});

test("HeightLegend consumes canonical HeightColorMapping contract", () => {
  const legendPath = path.resolve(process.cwd(), "src/components/HeightLegend.tsx");
  const source = fs.readFileSync(legendPath, "utf8");
  assert.ok(source.includes("HeightColorMapping"));
  assert.ok(source.includes("cssGradientForColorMap"));
  assert.ok(source.includes("colorScaleMin"));
  assert.ok(source.includes("colorScaleMax"));
  assert.ok(source.includes("colorMapping"));
});

test("resolveHeightColorMapping prefers explicit color_mapping block (artifact_metadata)", () => {
  const artifacts: StudioArtifact[] = [
    {
      artifact_id: "normalized_heightmap_display",
      stage_id: "normalize_heights_to_plane",
      kind: "json",
      title: "Display transform",
      path: "normalized_heightmap_display.json",
      preview_available: true,
      metadata: {
        semantic_field: "height_above_belt",
        color_mapping: {
          semantic_field: "height_above_belt",
          units: "mm",
          value_min: 0,
          value_max: 120.5,
          color_scale_min: 1.2,
          color_scale_max: 90.8,
          color_map: "turbo",
          direction: "higher_is_hotter",
          clamp: true,
          source: "artifact_metadata",
        },
      },
    } as StudioArtifact,
  ];
  const { mapping, warning } = resolveHeightColorMapping(artifacts, { semanticField: "height_above_belt" });
  assert.equal(warning, null);
  assert.equal(mapping.source, "artifact_metadata");
  assert.equal(mapping.colorScaleMin, 1.2);
  assert.equal(mapping.colorScaleMax, 90.8);
  assert.equal(mapping.colorMap, "turbo");
  assert.equal(mapping.direction, "higher_is_hotter");
});

test("resolveHeightColorMapping falls back to render_context when no explicit block", () => {
  const artifacts: StudioArtifact[] = [
    {
      artifact_id: "normalized_heightmap_render_context",
      stage_id: "normalize_heights_to_plane",
      kind: "json",
      title: "Render context",
      path: "normalized_heightmap.render_context.json",
      preview_available: true,
      metadata: {
        semantic_field: "height_above_belt",
        units: "mm",
        render_vmin: 5,
        render_vmax: 60,
        value_min: 0,
        value_max: 80,
        colormap_id: "viridis",
        direction: "higher_is_hotter",
      },
    } as StudioArtifact,
  ];
  const { mapping } = resolveHeightColorMapping(artifacts);
  assert.equal(mapping.source, "render_context");
  assert.equal(mapping.colorScaleMin, 5);
  assert.equal(mapping.colorScaleMax, 60);
  assert.equal(mapping.colorMap, "viridis");
});

test("resolveHeightColorMapping recognizes source-bound heightmap previews and warns about unknown source LUT", () => {
  // Mirrors what `stage_sources/index.ts` attaches for the Load heightmap
  // capture stage: source_binding=true with min_mm/max_mm from parser metadata
  // but no canonical color_mapping block (the LUT used by the SDK preview is
  // unknown). The resolver should still derive a usable mapping AND warn.
  const artifacts: StudioArtifact[] = [
    {
      artifact_id: "source_heightmap_preview",
      stage_id: "input",
      kind: "image",
      title: "Height preview",
      path: "heightmap_preview.png",
      preview_available: true,
      metadata: {
        source_binding: true,
        source_group: "heightmap",
        source_key: "heightmap_preview",
        visualization_role: "primary",
        semantic_field: "raw_sensor_z",
        source_preview_lut_unknown: true,
        min_mm: 12.5,
        max_mm: 280.0,
        units: "mm",
      },
    } as StudioArtifact,
  ];
  const { mapping, warning } = resolveHeightColorMapping(artifacts, { semanticField: "raw_sensor_z" });
  assert.equal(mapping.source, "artifact_metadata");
  assert.equal(mapping.semanticField, "raw_sensor_z");
  assert.equal(mapping.colorScaleMin, 12.5);
  assert.equal(mapping.colorScaleMax, 280.0);
  assert.equal(mapping.units, "mm");
  assert.ok(String(warning).includes("source preview LUT unknown"));
});

test("resolveHeightColorMapping returns legacy_fallback with warning when nothing usable exists", () => {
  const { mapping, warning } = resolveHeightColorMapping([], { semanticField: "height_above_belt" });
  assert.equal(mapping.source, "legacy_fallback");
  assert.ok(warning != null);
  assert.equal(mapping.colorMap, "turbo");
  assert.ok(mapping.colorScaleMax > mapping.colorScaleMin);
});

test("renderer/legend agree on colormap LUT", () => {
  // Same mapping consumed by renderer (scalarToRgb) and legend (sampleColorMap)
  // must produce identical RGB output at matching t. This is the canonical
  // guarantee that the colorbar can never visually disagree with the preview.
  for (const name of ["turbo", "viridis", "magma", "gray"] as const) {
    const min = 0;
    const max = 100;
    for (const value of [0, 25, 50, 75, 100]) {
      const rendererRgb = scalarToRgb(name, value, min, max, true);
      const legendRgb = sampleColorMap(name, (value - min) / (max - min));
      assert.deepEqual(rendererRgb, legendRgb, `${name}@${value}`);
    }
  }
  // CSS gradient must encode the same LUT (data-native orientation).
  const gradient = cssGradientForColorMap("turbo", 4, "higher_is_hotter");
  assert.ok(gradient.startsWith("linear-gradient(180deg,"));
  assert.ok(gradient.includes("rgb("));
});

test("legacy artifacts still resolve with warning", () => {
  const artifacts: StudioArtifact[] = [
    {
      artifact_id: "legacy_norm",
      stage_id: "normalize_heights_to_plane",
      kind: "file",
      title: "Legacy normalized raster",
      path: "normalized_heightmap.values.f32",
      preview_available: false,
      metadata: {},
    } as StudioArtifact,
  ];
  const fallback = resolveHeightArtifact(artifacts, { semanticField: "height_above_belt", representation: "numeric_raster" });
  assert.equal(fallback.artifact?.artifact_id, "legacy_norm");
  assert.ok(String(fallback.warning).includes("legacy semantic fallback"));
});
