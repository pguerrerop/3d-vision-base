// Pure helpers for the blob-cluster "candidate-support selection" explanation.
//
// These functions normalize the additive diagnostics emitted by the
// `low_gradient_blob_height_clusters` strategy (blob_height_clusters.json,
// blob_cluster_selection_debug.json, blob_cluster_score_table.csv) into a
// consistent shape so the UI can explain *why* a cluster was selected, bridge
// components/fragments -> clusters -> selected support, and stay robust to old
// runs that lack the newer fields.

export type ClusterDecision = "selected" | "rejected" | "candidate";

export type ClusterRow = {
  clusterId: string;
  selected: boolean;
  decision: ClusterDecision;
  score: number | null;
  rejectionReason: string | null;
  componentIds: string[];
  fragmentIds: string[];
  componentCount: number;
  fragmentCount: number;
  areaPx: number | null;
  areaFraction: number | null;
  medianZ: number | null;
  zMad: number | null;
  zStd: number | null;
  borderContact: boolean | null;
  spatialCoverage: number | null;
  centralityPenalty: number | null;
  objectLikePenalty: number | null;
  fragmentationPenalty: number | null;
  stripeOverlap: number | null;
  scoreTerms: Record<string, number> | null;
};

export type ClusterDiagnostics = {
  hasData: boolean;
  rows: ClusterRow[];
  selectedClusterId: string | null;
};

export type BridgeRow = {
  memberId: string;
  memberKind: "component" | "fragment";
  clusterId: string;
  decision: ClusterDecision;
  rejectionReason: string | null;
  selected: boolean;
};

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
}

function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function toNumberOrNull(value: unknown): number | null {
  if (value == null || value === "") return null;
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
}

function toBoolOrNull(value: unknown): boolean | null {
  if (value == null || value === "") return null;
  if (typeof value === "boolean") return value;
  const str = String(value).trim().toLowerCase();
  if (str === "true" || str === "1" || str === "yes") return true;
  if (str === "false" || str === "0" || str === "no") return false;
  return null;
}

function toIdList(value: unknown): string[] {
  return asArray(value).map((item) => String(item));
}

function toScoreTerms(value: unknown): Record<string, number> | null {
  const record = asRecord(value);
  const keys = Object.keys(record);
  if (!keys.length) return null;
  const terms: Record<string, number> = {};
  for (const key of keys) {
    const num = toNumberOrNull(record[key]);
    if (num != null) terms[key] = num;
  }
  return Object.keys(terms).length ? terms : null;
}

function normalizeDecision(value: unknown, selected: boolean, rejected: boolean): ClusterDecision {
  const raw = String(value ?? "").trim().toLowerCase();
  if (raw === "selected" || raw === "rejected" || raw === "candidate") return raw;
  if (selected) return "selected";
  if (rejected) return "rejected";
  return "candidate";
}

function clusterRowFromRecord(record: Record<string, unknown>, selectedClusterId: string | null): ClusterRow {
  const clusterId = String(record.cluster_id ?? record.clusterId ?? "");
  const componentIds = toIdList(record.component_ids ?? record.source_blob_ids ?? record.componentIds);
  const fragmentIds = toIdList(record.fragment_ids ?? record.blob_ids ?? record.fragmentIds);
  const rejected = toBoolOrNull(record.rejected) ?? Boolean(record.rejected_reason ?? record.rejectionReason);
  const selectedExplicit = toBoolOrNull(record.selected);
  const selected = selectedExplicit ?? (selectedClusterId != null && clusterId === selectedClusterId);
  const componentCount = componentIds.length || toNumberOrNull(record.source_blob_count) || 0;
  const fragmentCount = fragmentIds.length
    || toNumberOrNull(record.fragment_count)
    || toNumberOrNull(record.blob_count)
    || 0;
  return {
    clusterId,
    selected,
    decision: normalizeDecision(record.decision, selected, rejected),
    score: toNumberOrNull(record.score),
    rejectionReason: record.rejected_reason == null && record.rejectionReason == null
      ? null
      : String(record.rejected_reason ?? record.rejectionReason),
    componentIds,
    fragmentIds,
    componentCount: Number(componentCount) || componentIds.length,
    fragmentCount: Number(fragmentCount) || fragmentIds.length,
    areaPx: toNumberOrNull(record.area_px ?? record.total_pixels),
    areaFraction: toNumberOrNull(record.area_fraction),
    medianZ: toNumberOrNull(record.median_z),
    zMad: toNumberOrNull(record.z_mad ?? record.z_mad_weighted),
    zStd: toNumberOrNull(record.z_std),
    borderContact: toBoolOrNull(record.border_contact),
    spatialCoverage: toNumberOrNull(record.spatial_coverage),
    centralityPenalty: toNumberOrNull(record.centrality_penalty),
    objectLikePenalty: toNumberOrNull(record.object_like_penalty),
    fragmentationPenalty: toNumberOrNull(record.fragmentation_penalty),
    stripeOverlap: toNumberOrNull(record.stripe_overlap),
    scoreTerms: toScoreTerms(record.score_terms),
  };
}

/**
 * Build normalized cluster diagnostics from blob_height_clusters.json and
 * blob_cluster_selection_debug.json. Either input may be missing/old-format.
 * Rows are returned sorted by score descending (selected first on ties).
 */
export function parseClusterDiagnostics(clustersJson: unknown, selectionDebugJson: unknown): ClusterDiagnostics {
  const debug = asRecord(selectionDebugJson);
  const selectedClusterId = debug.selected_cluster_id == null ? null : String(debug.selected_cluster_id);

  // blob_height_clusters.json is an array of rows for active runs, or an object
  // (e.g. {clusters: []}) for inactive/old runs. The selection debug also
  // carries a per-cluster `clusters` array.
  let rawRows: Record<string, unknown>[] = [];
  if (Array.isArray(clustersJson)) {
    rawRows = clustersJson.map(asRecord);
  } else {
    const clustersField = asArray(asRecord(clustersJson).clusters);
    if (clustersField.length) rawRows = clustersField.map(asRecord);
  }
  if (!rawRows.length) {
    const debugClusters = asArray(debug.clusters);
    if (debugClusters.length) rawRows = debugClusters.map(asRecord);
  }

  const rows = rawRows
    .map((record) => clusterRowFromRecord(record, selectedClusterId))
    .filter((row) => row.clusterId !== "");

  rows.sort((a, b) => {
    const sa = a.score ?? Number.NEGATIVE_INFINITY;
    const sb = b.score ?? Number.NEGATIVE_INFINITY;
    if (sb !== sa) return sb - sa;
    return a.clusterId.localeCompare(b.clusterId);
  });

  return {
    hasData: rows.length > 0,
    rows,
    selectedClusterId: selectedClusterId ?? rows.find((row) => row.selected)?.clusterId ?? null,
  };
}

export function selectedClusterRow(diagnostics: ClusterDiagnostics): ClusterRow | null {
  return diagnostics.rows.find((row) => row.selected)
    ?? diagnostics.rows.find((row) => row.clusterId === diagnostics.selectedClusterId)
    ?? null;
}

export function topRejectedClusterRow(diagnostics: ClusterDiagnostics): ClusterRow | null {
  // Rows are already sorted by score desc, so the first rejected row is the
  // highest-scoring rejected cluster.
  return diagnostics.rows.find((row) => row.decision === "rejected") ?? null;
}

/**
 * Component/fragment -> cluster -> decision bridge rows. Components are
 * preferred for the member id; falls back to fragments when no component ids
 * are present.
 */
export function buildBridgeRows(diagnostics: ClusterDiagnostics): BridgeRow[] {
  const rows: BridgeRow[] = [];
  for (const cluster of diagnostics.rows) {
    const useComponents = cluster.componentIds.length > 0;
    const members = useComponents ? cluster.componentIds : cluster.fragmentIds;
    const memberKind: BridgeRow["memberKind"] = useComponents ? "component" : "fragment";
    if (!members.length) {
      rows.push({
        memberId: "(none)",
        memberKind,
        clusterId: cluster.clusterId,
        decision: cluster.decision,
        rejectionReason: cluster.rejectionReason,
        selected: cluster.selected,
      });
      continue;
    }
    for (const memberId of members) {
      rows.push({
        memberId,
        memberKind,
        clusterId: cluster.clusterId,
        decision: cluster.decision,
        rejectionReason: cluster.rejectionReason,
        selected: cluster.selected,
      });
    }
  }
  return rows;
}

export function magnitudeBucket(value: number | null): "low" | "medium" | "high" | "—" {
  if (value == null) return "—";
  const abs = Math.abs(value);
  if (abs < 0.15) return "low";
  if (abs < 0.45) return "medium";
  return "high";
}

export const CLUSTER_TABLE_COLUMNS: Array<{ key: keyof ClusterRow | "componentsFragments"; label: string }> = [
  { key: "selected", label: "selected" },
  { key: "clusterId", label: "cluster_id" },
  { key: "score", label: "score" },
  { key: "decision", label: "decision" },
  { key: "rejectionReason", label: "rejection_reason" },
  { key: "componentsFragments", label: "components/fragments" },
  { key: "areaPx", label: "area_px" },
  { key: "areaFraction", label: "area_fraction" },
  { key: "medianZ", label: "median_z" },
  { key: "zMad", label: "z_mad" },
  { key: "zStd", label: "z_std" },
  { key: "borderContact", label: "border_contact" },
  { key: "spatialCoverage", label: "spatial_coverage" },
  { key: "centralityPenalty", label: "centrality_penalty" },
  { key: "objectLikePenalty", label: "object_like_penalty" },
  { key: "fragmentationPenalty", label: "fragmentation_penalty" },
  { key: "stripeOverlap", label: "stripe_overlap" },
];
