import test from "node:test";
import assert from "node:assert/strict";

import { primarySourceImageUrl } from "../src/components/stage_sources/index.ts";

test("heightmap primary source prefers heightmap_preview", () => {
  const detail = {
    take_id: "take_001",
    modalities: ["heightmap"],
    metadata: { files: { heightmap_preview: "heightmap_preview.png", heightmap: "height16.tif", reflectance: "reflectance.png" } },
    assets: { heightmap: { heightmap: "height16.tif" }, reflectance: { reflectance: "reflectance.png" } },
  } as any;
  const src = primarySourceImageUrl(detail);
  assert.equal(src, "/api/takes/take_001/files/heightmap_preview.png");
});
