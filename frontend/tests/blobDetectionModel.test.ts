import test from "node:test";
import assert from "node:assert/strict";

import { resolveBlobSetSourceImage } from "../src/studio/blobDetectionModel.ts";
import type { StudioArtifact } from "../src/components/studioWorkspaceModel.ts";

function artifact(overrides: Partial<StudioArtifact>): StudioArtifact {
  return {
    artifact_id: "some_artifact",
    stage_id: "detect_blobs",
    kind: "json",
    title: "Some artifact",
    preview_available: true,
    ...overrides,
  } as StudioArtifact;
}

test("resolveBlobSetSourceImage prefers source_rgb_image when present", () => {
  const stageArtifacts = [
    artifact({ artifact_id: "blob_debug_overlay", kind: "overlay", path: "blob_debug_overlay.png" }),
    artifact({ artifact_id: "other_image", kind: "image", path: "other.png" }),
    artifact({ artifact_id: "source_rgb_image", kind: "image", path: "source_rgb.png" }),
  ];
  const resolved = resolveBlobSetSourceImage(stageArtifacts);
  assert.equal(resolved?.artifact_id, "source_rgb_image");
});

test("resolveBlobSetSourceImage falls back to the first image artifact with a path", () => {
  const stageArtifacts = [
    artifact({ artifact_id: "blob_metrics", kind: "json", path: undefined }),
    artifact({ artifact_id: "blob_contours", kind: "table" }),
    artifact({ artifact_id: "candidate_overlay", kind: "image", path: "candidate_overlay.png" }),
  ];
  const resolved = resolveBlobSetSourceImage(stageArtifacts);
  assert.equal(resolved?.artifact_id, "candidate_overlay");
});

test("resolveBlobSetSourceImage ignores image artifacts without a path", () => {
  const stageArtifacts = [
    artifact({ artifact_id: "pathless_image", kind: "image" }),
    artifact({ artifact_id: "blob_metrics", kind: "json", path: "blob_metrics.json" }),
  ];
  const resolved = resolveBlobSetSourceImage(stageArtifacts);
  assert.equal(resolved, null);
});
