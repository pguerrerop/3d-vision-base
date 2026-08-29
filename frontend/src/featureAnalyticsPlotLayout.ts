export const HISTOGRAM_BAND_HEIGHT_PX = 92;
export const HISTOGRAM_BAND_GAP_PX = 6;
export const HISTOGRAM_SCROLL_GROUP_THRESHOLD = 5;

export type HistogramPlotLayout = {
  groupCount: number;
  scrollable: boolean;
  contentHeightPx: number;
  maxScrollHeightPx: number;
};

export function histogramBandBlockHeightPx(groupCount: number): number {
  if (groupCount <= 0) return 0;
  return groupCount * HISTOGRAM_BAND_HEIGHT_PX + (groupCount - 1) * HISTOGRAM_BAND_GAP_PX;
}

export function histogramPlotLayout(groupCount: number): HistogramPlotLayout {
  const normalizedCount = Math.max(0, groupCount);
  const scrollable = normalizedCount > HISTOGRAM_SCROLL_GROUP_THRESHOLD;
  const contentHeightPx = histogramBandBlockHeightPx(normalizedCount);
  const maxScrollHeightPx = histogramBandBlockHeightPx(HISTOGRAM_SCROLL_GROUP_THRESHOLD);
  return {
    groupCount: normalizedCount,
    scrollable,
    contentHeightPx,
    maxScrollHeightPx,
  };
}

export function histogramScrollHint(visibleCount: number, groupCount: number): string | null {
  if (groupCount <= HISTOGRAM_SCROLL_GROUP_THRESHOLD) return null;
  if (visibleCount >= groupCount) return null;
  return `Showing ${visibleCount} of ${groupCount} groups · scroll to view more`;
}

export type FeatureAnalyticsSummaryParts = {
  mlSetName?: string | null;
  scopeTakeCount?: number | null;
  scopePhysicalObjectCount?: number | null;
  filteredTakeCount?: number | null;
  filteredPhysicalObjectCount?: number | null;
  filteredObjectCount?: number | null;
  filterDescriptors?: string[];
};

export function buildFeatureAnalyticsSummaryBar(parts: FeatureAnalyticsSummaryParts): string {
  const scopeTakes = parts.scopeTakeCount ?? 0;
  const scopePhysicalObjects = parts.scopePhysicalObjectCount ?? 0;
  const filteredTakes = parts.filteredTakeCount ?? 0;
  const filteredPhysicalObjects = parts.filteredPhysicalObjectCount ?? 0;
  const filteredObjects = parts.filteredObjectCount ?? 0;
  const filterDetail = (parts.filterDescriptors ?? []).filter(Boolean);
  const mlSetPrefix = parts.mlSetName ? `ML set ${parts.mlSetName} · ` : "";
  const scopePart = `Scope ${scopeTakes} takes · ${scopePhysicalObjects} physical objects`;
  const filteredLead = filterDetail.length ? `${filterDetail.join(" · ")} · ` : "";
  const filteredPart = `Filtered ${filteredLead}${filteredTakes} takes · ${filteredPhysicalObjects} physical objects · ${filteredObjects} objects`;
  return `${mlSetPrefix}${scopePart} | ${filteredPart}`;
}

export function buildMatchingObjectsMetaLine(parts: {
  objectCount?: number | null;
  takeCount?: number | null;
  physicalObjectCount?: number | null;
}): string {
  const objectCount = parts.objectCount ?? 0;
  const takeCount = parts.takeCount ?? 0;
  const physicalObjectCount = parts.physicalObjectCount ?? 0;
  return `${objectCount} objects · ${takeCount} takes · ${physicalObjectCount} physical objects`;
}

export function summarizeMatchingObjects<T extends {
  take_id?: string | null;
  physical_object_id?: string | null;
}>(objects: T[]): {
  objectCount: number;
  takeCount: number;
  physicalObjectCount: number;
} {
  const takeIds = new Set<string>();
  const physicalObjectIds = new Set<string>();
  for (const item of objects) {
    const takeId = String(item.take_id ?? "").trim();
    if (takeId) takeIds.add(takeId);
    const physicalObjectId = String(item.physical_object_id ?? "").trim();
    if (physicalObjectId) physicalObjectIds.add(physicalObjectId);
  }
  return {
    objectCount: objects.length,
    takeCount: takeIds.size,
    physicalObjectCount: physicalObjectIds.size,
  };
}

export type MatchingViewMode = "selected_bin" | "all_filtered";

export function resolveMatchingViewMode(
  mode: MatchingViewMode,
  hasSelectedBin: boolean,
): MatchingViewMode {
  if (!hasSelectedBin) return "all_filtered";
  return mode;
}

export function formatMatchingObjectClassCell(item: {
  normalized_class?: string | null;
  processed_class?: string | null;
}): { primary: string; secondary: string | null } {
  const normalized = String(item.normalized_class ?? "").trim();
  const processed = String(item.processed_class ?? "").trim();
  if (normalized && processed && normalized !== processed) {
    return { primary: normalized, secondary: processed };
  }
  return { primary: normalized || processed || "-", secondary: null };
}

export function matchingObjectRowTitle(item: {
  raw_labels?: string[] | null;
  validation_status?: string | null;
  labeled_superclass?: string | null;
  processed_superclass?: string | null;
  labeled_superclass_unresolved?: boolean | null;
  labeled_superclass_source?: string | null;
  normalized_class?: string | null;
}): string {
  const rawLabels = (item.raw_labels ?? []).filter(Boolean).join(", ");
  const validation = String(item.validation_status ?? "").trim();
  const labeled = String(item.labeled_superclass ?? "").trim();
  const processed = String(item.processed_superclass ?? "").trim();
  const labeledSource = String(item.labeled_superclass_source ?? "").trim();
  const normalizedClass = String(item.normalized_class ?? "").trim();
  return [
    rawLabels ? `Raw labels: ${rawLabels}` : "",
    validation ? `Validation: ${validation}` : "",
    labeled || processed ? `Superclass ${labeled || "-"} → ${processed || "-"}` : "",
    labeledSource ? `Labeled source: ${labeledSource}` : "",
    item.labeled_superclass_unresolved && normalizedClass ? "Labeled superclass unresolved from normalized class" : "",
  ].filter(Boolean).join(" · ");
}
