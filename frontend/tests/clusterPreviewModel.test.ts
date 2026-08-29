import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import {
  clusterColor,
  memberIdsForClusterPreview,
  parseMemberId,
  resolveBlobIdMaskArtifact,
} from "../src/components/clusterPreviewModel.ts";
import type { ClusterRow } from "../src/components/clusterDecisionModel.ts";

test("parseMemberId accepts numeric and suffixed ids", () => {
  assert.equal(parseMemberId("3"), 3);
  assert.equal(parseMemberId("fragment_003"), 3);
  assert.equal(parseMemberId("component_8"), 8);
  assert.equal(parseMemberId(""), null);
});

test("resolveBlobIdMaskArtifact prefers split mask then height-aware mask", () => {
  const artifacts = [
    { artifact_id: "low_gradient_blob_id_mask", path: "low.png" },
    { artifact_id: "height_aware_blob_id_mask", path: "aware.png" },
    { artifact_id: "height_split_blob_id_mask", path: "split.png" },
  ] as Array<{ artifact_id: string; path: string }>;
  assert.equal(resolveBlobIdMaskArtifact(artifacts as never)?.artifact_id, "height_split_blob_id_mask");
  assert.equal(
    resolveBlobIdMaskArtifact(artifacts.filter((item) => item.artifact_id !== "height_split_blob_id_mask") as never)?.artifact_id,
    "height_aware_blob_id_mask",
  );
});

test("memberIdsForClusterPreview uses fragment ids for split masks", () => {
  const row: ClusterRow = {
    clusterId: "cluster_001",
    selected: true,
    decision: "selected",
    score: 0.8,
    rejectionReason: null,
    componentIds: ["8"],
    fragmentIds: ["1", "3"],
    componentCount: 1,
    fragmentCount: 2,
    areaPx: 100,
    areaFraction: 0.2,
    medianZ: 812,
    zMad: 1.2,
    zStd: null,
    borderContact: true,
    spatialCoverage: 0.6,
    centralityPenalty: 0.1,
    objectLikePenalty: 0.1,
    fragmentationPenalty: 0.0,
    stripeOverlap: 0.0,
    scoreTerms: null,
  };
  const splitMask = { artifact_id: "height_split_blob_id_mask", path: "split.png" } as never;
  assert.deepEqual(memberIdsForClusterPreview(row, splitMask), [1, 3]);
});

test("clusterColor is stable for a cluster id", () => {
  assert.equal(clusterColor("cluster_001"), clusterColor("cluster_001"));
  assert.notEqual(clusterColor("cluster_001"), clusterColor("cluster_002"));
});

test("DetectClusterDecision wires spatial preview and controlled cluster selection", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/components/DetectClusterDecision.tsx"), "utf-8");
  assert.ok(source.includes("BlobClusterPreview"));
  assert.ok(source.includes("showSpatialPreview"));
  assert.ok(source.includes("onActiveClusterChange"));
});

test("cluster score table renderer enables spatial preview", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/components/stage_view_renderers/index.tsx"), "utf-8");
  assert.ok(source.includes("showSpatialPreview"));
  assert.ok(source.includes("activeBlobClusterId"));
});
