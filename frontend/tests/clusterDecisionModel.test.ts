import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import {
  buildBridgeRows,
  parseClusterDiagnostics,
  selectedClusterRow,
  topRejectedClusterRow,
} from "../src/components/clusterDecisionModel.ts";

const clustersJson = [
  {
    cluster_id: "cluster_002",
    selected: false,
    decision: "rejected",
    score: 0.42,
    rejected_reason: "central/object-like",
    component_ids: ["component_003"],
    fragment_ids: ["fragment_003"],
    area_fraction: 0.11,
    median_z: 805.1,
    z_mad: 2.2,
    border_contact: false,
    spatial_coverage: 0.3,
    centrality_penalty: 0.5,
    object_like_penalty: 0.4,
    fragmentation_penalty: 0.1,
    stripe_overlap: 0.05,
    score_terms: { area: 0.11, border: 0.0 },
  },
  {
    cluster_id: "cluster_001",
    selected: true,
    decision: "selected",
    score: 0.74,
    rejected_reason: null,
    component_ids: ["component_001", "component_002"],
    fragment_ids: ["fragment_001", "fragment_002"],
    area_fraction: 0.208,
    median_z: 812.3,
    z_mad: 1.4,
    border_contact: true,
    spatial_coverage: 0.63,
    centrality_penalty: 0.12,
    object_like_penalty: 0.25,
    fragmentation_penalty: 0.05,
    stripe_overlap: 0.02,
    score_terms: { area: 0.18, border: 0.2 },
  },
];

const selectionDebug = {
  strategy: "low_gradient_blob_height_clusters",
  selected_cluster_id: "cluster_001",
};

test("parseClusterDiagnostics sorts rows by score descending", () => {
  const diagnostics = parseClusterDiagnostics(clustersJson, selectionDebug);
  assert.equal(diagnostics.hasData, true);
  assert.deepEqual(diagnostics.rows.map((row) => row.clusterId), ["cluster_001", "cluster_002"]);
});

test("selectedClusterRow returns the selected cluster with detail", () => {
  const diagnostics = parseClusterDiagnostics(clustersJson, selectionDebug);
  const selected = selectedClusterRow(diagnostics);
  assert.ok(selected);
  assert.equal(selected?.clusterId, "cluster_001");
  assert.equal(selected?.decision, "selected");
  assert.equal(selected?.score, 0.74);
  assert.equal(selected?.componentCount, 2);
  assert.equal(selected?.fragmentCount, 2);
  assert.equal(selected?.borderContact, true);
});

test("topRejectedClusterRow surfaces the highest-scoring rejected cluster", () => {
  const diagnostics = parseClusterDiagnostics(clustersJson, selectionDebug);
  const rejected = topRejectedClusterRow(diagnostics);
  assert.ok(rejected);
  assert.equal(rejected?.clusterId, "cluster_002");
  assert.equal(rejected?.rejectionReason, "central/object-like");
});

test("missing diagnostics yields a friendly empty state", () => {
  const diagnostics = parseClusterDiagnostics({ blob_clustering_active: false, clusters: [] }, null);
  assert.equal(diagnostics.hasData, false);
  assert.equal(diagnostics.rows.length, 0);
  assert.equal(selectedClusterRow(diagnostics), null);
});

test("selection derives from selection debug when row flags are absent", () => {
  const rowsWithoutFlags = clustersJson.map(({ selected, decision, ...rest }) => rest);
  const diagnostics = parseClusterDiagnostics(rowsWithoutFlags, selectionDebug);
  const selected = selectedClusterRow(diagnostics);
  assert.equal(selected?.clusterId, "cluster_001");
  assert.equal(selected?.selected, true);
});

test("buildBridgeRows maps components/fragments to clusters and decisions", () => {
  const diagnostics = parseClusterDiagnostics(clustersJson, selectionDebug);
  const bridge = buildBridgeRows(diagnostics);
  const selectedMembers = bridge.filter((row) => row.clusterId === "cluster_001");
  assert.deepEqual(selectedMembers.map((row) => row.memberId), ["component_001", "component_002"]);
  assert.ok(selectedMembers.every((row) => row.decision === "selected"));
  const rejectedMember = bridge.find((row) => row.clusterId === "cluster_002");
  assert.equal(rejectedMember?.decision, "rejected");
});

test("DetectClusterDecision renderer wires why card, decision table, and bridge", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/components/DetectClusterDecision.tsx"), "utf-8");
  assert.ok(source.includes("Why this cluster was selected"));
  assert.ok(source.includes("cluster-decision-table"));
  assert.ok(source.includes("cluster-bridge-table"));
  assert.ok(source.includes("Top rejected cluster"));
  // Clicking a cluster row updates local explanation state, not downstream object selection.
  assert.ok(source.includes("onActiveClusterChange"));
  assert.ok(!source.includes("onSelectObject"));
});

test("detect-stage inspector uses reference-support language without classification cards", () => {
  const source = fs.readFileSync(path.resolve(process.cwd(), "src/components/StudioInspector.tsx"), "utf-8");
  assert.ok(source.includes("Reference support"));
  assert.ok(source.includes("reference_component"));
  assert.ok(source.includes("{!isDetectStage && <ObjectInspector"));
  assert.ok(source.includes("{!isDetectStage && ("));
});
