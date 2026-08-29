import { useCallback, useEffect, useMemo, useRef, useState, type KeyboardEvent, type MouseEvent, type ReactNode } from "react";
import {
  buildFeatureAnalyticsSummaryBar,
  buildMatchingObjectsMetaLine,
  formatMatchingObjectClassCell,
  histogramPlotLayout,
  histogramScrollHint,
  matchingObjectRowTitle,
  resolveMatchingViewMode,
  type MatchingViewMode,
} from "../featureAnalyticsPlotLayout";
import {
  featureBadgeLabels,
  featureFamilyLabel,
  featureOptionLabel,
  featureSectionLabel,
  featureStudioStage,
  filterFeatureCatalog,
  groupFeatureCatalog,
} from "../featureCatalogModel";
import {
  api,
  type DatasetMlSet,
  type DatasetSessionSummary,
  type DatasetSummary,
  type FeatureAnalyticsDistributionResponse,
  type FeatureAnalyticsFeaturesResponse,
  type FeatureAnalyticsFilterOptionsResponse,
  type FeatureAnalyticsObject,
  type FeatureAnalyticsQuery,
  type FeatureAnalyticsThumbnailInfo,
  type MLSetSummaryResponse,
  objectThumbnailUrl,
  type PipelineInfo,
} from "../api/client";
import { featureAnalyticsFeatureCatalogScope } from "../featureAnalyticsQuery";
import { buildStudioDeepLink } from "../studioDeepLink";
import FeatureFreshnessPanel from "../components/FeatureFreshnessPanel";
import {
  defaultFeaturePipelineId,
  featureFreshnessState,
  featureJobActionsEnabled,
  featureSourceLabel,
  featureSourceTooltip,
  resolveFeatureFreshnessScope,
} from "../featureAnalyticsFreshness";

type FilterState = {
  datasetId: string;
  mlSetId: string;
  sessionId: string;
  labeledSuperclasses: string[];
  processedSuperclasses: string[];
  normalizedClasses: string[];
  rawLabels: string[];
  processedClasses: string[];
  physicalObjectIds: string[];
  takeIds: string[];
  validationStatus: string[];
  split: string[];
  pipelineId: string;
  calibrationId: string;
  dateFrom: string;
  dateTo: string;
};

type FeatureGroupBy = "processed_superclass" | "labeled_superclass" | "normalized_class" | "raw_label" | "physical_object_id" | "processed_class";

const DEFAULT_FILTERS: FilterState = {
  datasetId: "",
  mlSetId: "",
  sessionId: "",
  labeledSuperclasses: [],
  processedSuperclasses: [],
  normalizedClasses: [],
  rawLabels: [],
  processedClasses: [],
  physicalObjectIds: [],
  takeIds: [],
  validationStatus: [],
  split: [],
  pipelineId: "",
  calibrationId: "",
  dateFrom: "",
  dateTo: "",
};

const COLORS = ["#1d4ed8", "#dc2626", "#0f766e", "#9333ea", "#f97316", "#374151"];
const DEFAULT_CHART_WIDTH_PX = 860;
const KNOWN_VALIDATION = ["approved", "unreviewed", "rejected"];
const KNOWN_SPLITS = ["train", "validation", "test", "holdout", "calibration", "unassigned"];
const MATCHING_BIN_LIMIT = 500;
const MATCHING_ALL_FILTERED_LIMIT = 1000;

type MatchingScopeSummary = FeatureAnalyticsFeaturesResponse["scope_summary"];
type ThumbnailMode = "auto" | "heightmap" | "reflectance" | "classification_overlay";

function thumbnailModeLabel(mode: ThumbnailMode): string {
  if (mode === "heightmap") return "Heightmap";
  if (mode === "reflectance") return "Source/reflectance";
  if (mode === "classification_overlay") return "Classification overlay";
  return "Auto";
}

function thumbnailResolvedSourceLabel(source: FeatureAnalyticsThumbnailInfo["resolved_source"] | string | null | undefined): string {
  if (source === "normalized_heightmap") return "normalized heightmap";
  if (source === "heightmap") return "heightmap";
  if (source === "reflectance") return "reflectance";
  if (source === "source_crop") return "source crop";
  if (source === "classification_overlay") return "classification overlay";
  if (source === "take_thumbnail") return "take thumbnail";
  if (source === "placeholder") return "placeholder";
  return source ? String(source) : "-";
}

function matchingObjectWarning(item: FeatureAnalyticsObject): string | null {
  if (item.labeled_superclass_unresolved && item.normalized_class) {
    return "Labeled superclass unresolved from normalized class";
  }
  return null;
}

function selectHistogramBin(
  range: { min: number; max: number },
  setSelectedRange: (range: { min: number; max: number }) => void,
  setMatchingViewMode: (mode: MatchingViewMode) => void,
) {
  setSelectedRange(range);
  setMatchingViewMode("selected_bin");
}

function sessionTypeLabel(value: DatasetSessionSummary["session_type"] | string | null | undefined): string {
  if (!value) return "Dataset session";
  const normalized = String(value).trim().toLowerCase();
  if (normalized === "engineering") return "Engineering";
  if (normalized === "curated") return "Curated";
  if (normalized === "benchmark") return "Benchmark";
  if (normalized === "operational") return "Operational";
  return normalized ? normalized.charAt(0).toUpperCase() + normalized.slice(1) : "Dataset session";
}

function compactDate(value: string | null | undefined): string {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return date.toISOString().slice(0, 10);
}

function sessionOptionLabel(session: DatasetSessionSummary): string {
  const parts = [
    sessionTypeLabel(session.session_type),
    compactDate(session.created_at),
    typeof session.take_count === "number" ? `${session.take_count} takes` : "",
  ].filter(Boolean);
  return parts.length ? `${session.name} - ${parts.join(" - ")}` : session.name;
}

function groupVisualStyle(group: string, index: number): { color: string; className: string } {
  const normalized = group.trim().toUpperCase();
  if (normalized === "UNKNOWN") return { color: "#94a3b8", className: "is-uncertain" };
  return { color: COLORS[index % COLORS.length], className: "" };
}

function compactNumber(value: number): string {
  if (!Number.isFinite(value)) return "-";
  const abs = Math.abs(value);
  if (abs >= 100) return value.toFixed(0);
  if (abs >= 10) return value.toFixed(1).replace(/\.0$/, "");
  if (abs >= 1) return value.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
  return value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
}

function adaptiveTickIndexes(edges: number[], binCount: number): number[] {
  if (!edges.length) return [];
  const targetTickCount = Math.max(4, Math.min(10, Math.floor(DEFAULT_CHART_WIDTH_PX / 120)));
  const step = Math.max(1, Math.ceil(binCount / targetTickCount));
  const indexes: number[] = [];
  for (let idx = 0; idx < edges.length; idx += step) indexes.push(idx);
  const lastIdx = edges.length - 1;
  if (!indexes.includes(lastIdx)) indexes.push(lastIdx);
  return indexes;
}

function uniq(values: string[]): string[] {
  return Array.from(new Set(values.map((value) => value.trim()).filter(Boolean)));
}

function buildZeroStateMessage(filters: FilterState): string {
  const constraints = [
    filters.datasetId ? `Dataset ${filters.datasetId}` : "",
    filters.mlSetId ? `ML set ${filters.mlSetId}` : "",
    filters.sessionId ? `Session ${filters.sessionId}` : "",
    filters.physicalObjectIds.length ? `Physical object IDs ${filters.physicalObjectIds.join(", ")}` : "",
    filters.normalizedClasses.length ? `Normalized class ${filters.normalizedClasses.join(", ")}` : "",
    filters.labeledSuperclasses.length ? `Labeled superclass ${filters.labeledSuperclasses.join(", ")}` : "",
    filters.processedSuperclasses.length ? `Processed superclass ${filters.processedSuperclasses.join(", ")}` : "",
    filters.split.length ? `Split ${filters.split.join(", ")}` : "",
    filters.validationStatus.length ? `Validation ${filters.validationStatus.join(", ")}` : "",
  ].filter(Boolean);
  const hints = [
    filters.physicalObjectIds.length ? "Try clearing Physical object ID." : "",
    filters.split.length ? "Try clearing Split." : "",
  ].filter(Boolean);
  return `${constraints.join(" · ") || "No active constraints."}${hints.length ? ` ${hints.join(" ")}` : ""}`;
}

function buildRegisteredFeatureMissingHint(featureLabel: string | null | undefined, featureKey: string, mlSetId: string): string {
  const label = featureLabel || featureKey;
  return `${label} is registered, but no values were found in the current scope. Use Reprocess ML set or Backfill feature below.`;
}

function queryFromFilters(filters: FilterState): FeatureAnalyticsQuery {
  return {
    dataset_id: filters.datasetId || undefined,
    ml_set_id: filters.mlSetId || undefined,
    session_id: filters.sessionId || undefined,
    physical_object_ids: filters.physicalObjectIds,
    take_ids: filters.takeIds,
    labeled_superclasses: filters.labeledSuperclasses,
    processed_superclasses: filters.processedSuperclasses,
    raw_labels: filters.rawLabels,
    normalized_classes: filters.normalizedClasses,
    processed_classes: filters.processedClasses,
    validation_status: filters.validationStatus,
    split: filters.split,
    pipeline_id: filters.pipelineId || undefined,
    calibration_id: filters.calibrationId || undefined,
    date_from: filters.dateFrom || undefined,
    date_to: filters.dateTo || undefined,
  };
}

function parseInitialFilters(): Partial<FilterState> {
  if (typeof window === "undefined") return {};
  const params = new URLSearchParams(window.location.search);
  const list = (...keys: string[]) => uniq(keys.flatMap((key) => params.getAll(key).flatMap((value) => value.split(","))).map((value) => value.trim()).filter(Boolean));
  return {
    datasetId: params.get("dataset_id") || params.get("dataset") || "",
    mlSetId: params.get("ml_set_id") || "",
    sessionId: params.get("session_id") || params.get("session") || "",
    labeledSuperclasses: list("labeled_superclasses"),
    processedSuperclasses: list("processed_superclasses", "superclasses", "superclass"),
    normalizedClasses: list("normalized_classes", "normalized_class", "labels"),
    rawLabels: list("raw_labels"),
    processedClasses: list("processed_classes"),
    physicalObjectIds: list("physical_object_ids", "physical_object_id"),
    takeIds: list("take_ids", "take_id"),
    split: list("split"),
  };
}

export default function FeatureAnalyticsPage() {
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [mlSets, setMlSets] = useState<DatasetMlSet[]>([]);
  const [datasetSessions, setDatasetSessions] = useState<DatasetSessionSummary[]>([]);
  const [pipelines, setPipelines] = useState<PipelineInfo[]>([]);
  const [featureDefinitions, setFeatureDefinitions] = useState<FeatureAnalyticsFeaturesResponse["feature_definitions"]>([]);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [inspectorCollapsed, setInspectorCollapsed] = useState(false);
  const [featureSummary, setFeatureSummary] = useState<FeatureAnalyticsFeaturesResponse | null>(null);
  const [filterOptions, setFilterOptions] = useState<FeatureAnalyticsFilterOptionsResponse | null>(null);
  const [filters, setFilters] = useState<FilterState>({ ...DEFAULT_FILTERS, ...parseInitialFilters() });
  const [selectedMlSetSummary, setSelectedMlSetSummary] = useState<MLSetSummaryResponse | null>(null);
  const [featureKey, setFeatureKey] = useState("");
  const [featureSearch, setFeatureSearch] = useState("");
  const [groupBy, setGroupBy] = useState<FeatureGroupBy>("processed_superclass");
  const [mode, setMode] = useState<"count" | "density">("count");
  const [binCount, setBinCount] = useState(24);
  const [distribution, setDistribution] = useState<FeatureAnalyticsDistributionResponse | null>(null);
  const [selectedRange, setSelectedRange] = useState<{ min: number; max: number } | null>(null);
  const [matchingViewMode, setMatchingViewMode] = useState<MatchingViewMode>("all_filtered");
  const [showBinEdgeLabels, setShowBinEdgeLabels] = useState(false);
  const [thumbnailMode, setThumbnailMode] = useState<ThumbnailMode>("auto");
  const [thumbnailInfoByKey, setThumbnailInfoByKey] = useState<Record<string, FeatureAnalyticsThumbnailInfo>>({});
  const [binObjects, setBinObjects] = useState<FeatureAnalyticsObject[]>([]);
  const [allFilteredObjects, setAllFilteredObjects] = useState<FeatureAnalyticsObject[]>([]);
  const [binScopeSummary, setBinScopeSummary] = useState<MatchingScopeSummary | null>(null);
  const [allFilteredScopeSummary, setAllFilteredScopeSummary] = useState<MatchingScopeSummary | null>(null);
  const [binObjectsLoading, setBinObjectsLoading] = useState(false);
  const [allFilteredObjectsLoading, setAllFilteredObjectsLoading] = useState(false);
  const [selectedObject, setSelectedObject] = useState<FeatureAnalyticsObject | null>(null);
  const [hoverPreview, setHoverPreview] = useState<{
    x: number;
    y: number;
    src: string;
    item: FeatureAnalyticsObject;
  } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [zeroStateMessage, setZeroStateMessage] = useState<string | null>(null);
  const [analyticsRefreshKey, setAnalyticsRefreshKey] = useState(0);
  const histogramScrollRef = useRef<HTMLDivElement | null>(null);
  const histogramBandRefs = useRef<Array<HTMLDivElement | null>>([]);
  const [visibleHistogramGroups, setVisibleHistogramGroups] = useState(0);

  useEffect(() => {
    void Promise.all([api.datasets(), api.pipelines()])
      .then(([ds, pls]) => {
        setDatasets(ds);
        setPipelines(pls);
      })
      .catch(() => {
        setDatasets([]);
        setPipelines([]);
      });
  }, []);

  useEffect(() => {
    if (!filters.datasetId) {
      setDatasetSessions([]);
      setMlSets([]);
      setFilters((prev) => ({
        ...prev,
        sessionId: "",
      }));
      return;
    }
    void Promise.all([api.datasetSessions(filters.datasetId), api.datasetMlSets(filters.datasetId)])
      .then(([sessions, scopedMlSets]) => {
        setDatasetSessions(sessions);
        setMlSets(scopedMlSets);
        setFilters((prev) => ({
          ...prev,
          sessionId: prev.sessionId && sessions.some((item) => item.id === prev.sessionId) ? prev.sessionId : "",
          mlSetId: prev.mlSetId && scopedMlSets.some((item) => item.id === prev.mlSetId) ? prev.mlSetId : "",
        }));
      })
      .catch(() => {
        setDatasetSessions([]);
        setMlSets([]);
      });
  }, [filters.datasetId]);

  useEffect(() => {
    if (!filters.mlSetId) {
      setSelectedMlSetSummary(null);
      return;
    }
    void api.mlSetSummary(filters.mlSetId, filters.datasetId || undefined)
      .then((summary) => {
        setSelectedMlSetSummary(summary);
        if (!filters.datasetId && summary.dataset?.id) {
          setFilters((prev) => ({ ...prev, datasetId: summary.dataset.id }));
        }
      })
      .catch(() => setSelectedMlSetSummary(null));
  }, [filters.mlSetId, filters.datasetId]);

  const query = useMemo(() => queryFromFilters(filters), [filters]);
  const featureCatalogQuery = useMemo(() => featureAnalyticsFeatureCatalogScope(query), [query]);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    setZeroStateMessage(null);
    void Promise.all([
      api.featureAnalyticsFeatures(query),
      api.featureAnalyticsFilterOptions(query),
      api.featureAnalyticsFeatures(featureCatalogQuery),
    ])
      .then(([featuresPayload, optionsPayload, catalogPayload]) => {
        if (cancelled) return;
        const visibleFeatures = featuresPayload.feature_definitions.length ? featuresPayload.feature_definitions : catalogPayload.feature_definitions;
        setFeatureDefinitions(visibleFeatures);
        setFeatureSummary(featuresPayload);
        setFilterOptions(optionsPayload);
        setFeatureKey((current) => visibleFeatures.some((item) => item.feature_key === current) ? current : (visibleFeatures[0]?.feature_key || ""));
        if ((featuresPayload.scope_summary?.object_count ?? 0) === 0) {
          setZeroStateMessage(buildZeroStateMessage(filters));
        }
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Failed loading feature analytics filters");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [query, featureCatalogQuery, filters, analyticsRefreshKey]);

  const refreshAnalytics = useCallback(() => {
    setAnalyticsRefreshKey((value) => value + 1);
  }, []);

  useEffect(() => {
    if (!featureKey) {
      setDistribution(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    void api
      .featureAnalyticsDistributions({ ...query, feature_key: featureKey, group_by: groupBy, bins: binCount, mode })
      .then((res) => {
        if (cancelled) return;
        setDistribution(res);
        setSelectedRange(null);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Failed loading distributions");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [query, featureKey, groupBy, binCount, mode, analyticsRefreshKey]);

  useEffect(() => {
    if (!selectedRange || !featureKey) {
      setBinObjects([]);
      setBinScopeSummary(null);
      return;
    }
    let cancelled = false;
    setBinObjectsLoading(true);
    void api
      .featureAnalyticsObjects({
        ...query,
        feature_key: featureKey,
        min_value: selectedRange.min,
        max_value: selectedRange.max,
        limit: MATCHING_BIN_LIMIT,
      })
      .then((res) => {
        if (!cancelled) {
          setBinObjects(res.objects);
          setBinScopeSummary(res.scope_summary);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setBinObjects([]);
          setBinScopeSummary(null);
        }
      })
      .finally(() => {
        if (!cancelled) setBinObjectsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedRange, featureKey, query, analyticsRefreshKey]);

  useEffect(() => {
    if (!featureKey) {
      setAllFilteredObjects([]);
      setAllFilteredScopeSummary(null);
      return;
    }
    let cancelled = false;
    setAllFilteredObjectsLoading(true);
    void api
      .featureAnalyticsObjects({
        ...query,
        feature_key: featureKey,
        limit: MATCHING_ALL_FILTERED_LIMIT,
      })
      .then((res) => {
        if (!cancelled) {
          setAllFilteredObjects(res.objects);
          setAllFilteredScopeSummary(res.scope_summary);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setAllFilteredObjects([]);
          setAllFilteredScopeSummary(null);
        }
      })
      .finally(() => {
        if (!cancelled) setAllFilteredObjectsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [featureKey, query, analyticsRefreshKey]);

  useEffect(() => {
    if (!selectedRange) setMatchingViewMode("all_filtered");
  }, [selectedRange]);

  const featureMeta = useMemo(() => featureDefinitions.find((item) => item.feature_key === featureKey) ?? null, [featureDefinitions, featureKey]);
  const resolvedPipelineId = useMemo(() => filters.pipelineId || defaultFeaturePipelineId(pipelines), [filters.pipelineId, pipelines]);
  const freshnessScope = useMemo(
    () => resolveFeatureFreshnessScope({
      datasetId: filters.datasetId,
      mlSetId: filters.mlSetId,
      takeIds: filters.takeIds,
      mlSetName: selectedMlSetSummary?.ml_set.name || selectedMlSetSummary?.ml_set.id || null,
      scopeSummary: featureSummary?.scope_summary ?? null,
    }),
    [filters.datasetId, filters.mlSetId, filters.takeIds, selectedMlSetSummary, featureSummary?.scope_summary],
  );
  const filteredFeatureDefinitions = useMemo(
    () => filterFeatureCatalog(featureDefinitions, featureSearch),
    [featureDefinitions, featureSearch],
  );
  const groupedFeatureDefinitions = useMemo(
    () => groupFeatureCatalog(filteredFeatureDefinitions.length ? filteredFeatureDefinitions : featureDefinitions),
    [filteredFeatureDefinitions, featureDefinitions],
  );
  const maxY = useMemo(() => {
    if (!distribution) return 0;
    let max = 0;
    for (const group of distribution.groups) {
      for (const value of group.bins) max = Math.max(max, Number(value) || 0);
    }
    return max;
  }, [distribution]);
  const tickIndexes = useMemo(() => {
    if (!distribution?.range?.edges?.length) return [];
    return adaptiveTickIndexes(distribution.range.edges, binCount);
  }, [distribution, binCount]);
  const histogramLayout = useMemo(
    () => histogramPlotLayout(distribution?.groups.length ?? 0),
    [distribution?.groups.length],
  );
  const histogramScrollMessage = useMemo(
    () => histogramScrollHint(visibleHistogramGroups, histogramLayout.groupCount),
    [visibleHistogramGroups, histogramLayout.groupCount],
  );

  useEffect(() => {
    histogramBandRefs.current = histogramBandRefs.current.slice(0, distribution?.groups.length ?? 0);
  }, [distribution?.groups.length]);

  useEffect(() => {
    const scrollEl = histogramScrollRef.current;
    const groupCount = distribution?.groups.length ?? 0;
    if (!scrollEl || !histogramLayout.scrollable || groupCount === 0) {
      setVisibleHistogramGroups(groupCount);
      return;
    }

    const updateVisibleGroups = () => {
      const scrollRect = scrollEl.getBoundingClientRect();
      let visible = 0;
      for (const bandEl of histogramBandRefs.current) {
        if (!bandEl) continue;
        const bandRect = bandEl.getBoundingClientRect();
        if (bandRect.bottom > scrollRect.top + 2 && bandRect.top < scrollRect.bottom - 2) visible += 1;
      }
      setVisibleHistogramGroups(visible);
    };

    updateVisibleGroups();
    scrollEl.addEventListener("scroll", updateVisibleGroups, { passive: true });
    const resizeObserver = new ResizeObserver(updateVisibleGroups);
    resizeObserver.observe(scrollEl);
    for (const bandEl of histogramBandRefs.current) {
      if (bandEl) resizeObserver.observe(bandEl);
    }
    return () => {
      scrollEl.removeEventListener("scroll", updateVisibleGroups);
      resizeObserver.disconnect();
    };
  }, [distribution?.groups, histogramLayout.scrollable, histogramLayout.groupCount]);
  const selectedSessionMeta = useMemo(
    () => datasetSessions.find((item) => item.id === filters.sessionId) ?? null,
    [datasetSessions, filters.sessionId],
  );
  const selectedDataset = useMemo(
    () => datasets.find((item) => item.id === filters.datasetId) ?? null,
    [datasets, filters.datasetId],
  );

  const validationOptions = useMemo(
    () => uniq([...(filterOptions?.validation_statuses ?? []), ...KNOWN_VALIDATION]),
    [filterOptions],
  );
  const splitOptions = useMemo(
    () => uniq([...(filterOptions?.splits ?? []), ...KNOWN_SPLITS]),
    [filterOptions],
  );

  const filteredSummary = featureSummary?.scope_summary ?? filterOptions?.stats ?? null;
  const scopeTakeCount = selectedMlSetSummary?.identity.take_count ?? filteredSummary?.take_count ?? null;
  const scopePhysicalObjectCount = selectedMlSetSummary?.identity.physical_object_count ?? filteredSummary?.physical_object_count ?? null;
  const mlSetLabel = selectedMlSetSummary?.ml_set.name || selectedMlSetSummary?.ml_set.id || null;

  const filterDescriptors = [
    filters.physicalObjectIds.length === 1 ? filters.physicalObjectIds[0] : filters.physicalObjectIds.length > 1 ? `${filters.physicalObjectIds.length} physical objects` : "",
    filters.takeIds.length === 1 ? filters.takeIds[0] : filters.takeIds.length > 1 ? `${filters.takeIds.length} takes` : "",
    filters.labeledSuperclasses.length === 1 ? `labeled ${filters.labeledSuperclasses[0]}` : filters.labeledSuperclasses.length > 1 ? `${filters.labeledSuperclasses.length} labeled superclasses` : "",
    filters.processedSuperclasses.length === 1 ? `processed ${filters.processedSuperclasses[0]}` : filters.processedSuperclasses.length > 1 ? `${filters.processedSuperclasses.length} processed superclasses` : "",
    filters.normalizedClasses.length === 1 ? filters.normalizedClasses[0] : filters.normalizedClasses.length > 1 ? `${filters.normalizedClasses.length} normalized classes` : "",
    filters.processedClasses.length === 1 ? filters.processedClasses[0] : filters.processedClasses.length > 1 ? `${filters.processedClasses.length} processed classes` : "",
    filters.rawLabels.length === 1 ? filters.rawLabels[0] : filters.rawLabels.length > 1 ? `${filters.rawLabels.length} raw labels` : "",
    filters.validationStatus.length === 1 ? filters.validationStatus[0] : filters.validationStatus.length > 1 ? `${filters.validationStatus.length} validation states` : "",
    filters.split.length === 1 ? `${filters.split[0]} split` : filters.split.length > 1 ? `${filters.split.length} splits` : "",
  ].filter(Boolean);
  const compactSummaryBar = buildFeatureAnalyticsSummaryBar({
    mlSetName: mlSetLabel ?? (selectedDataset?.name && !filters.mlSetId ? selectedDataset.name : null),
    scopeTakeCount,
    scopePhysicalObjectCount,
    filteredTakeCount: filteredSummary?.take_count,
    filteredPhysicalObjectCount: filteredSummary?.physical_object_count,
    filteredObjectCount: filteredSummary?.object_count,
    filterDescriptors,
  });
  const activeMatchingViewMode = resolveMatchingViewMode(matchingViewMode, Boolean(selectedRange));
  const displayedMatchingObjects = activeMatchingViewMode === "selected_bin" ? binObjects : allFilteredObjects;
  const displayedMatchingSummary = activeMatchingViewMode === "selected_bin"
    ? (binScopeSummary ?? { record_count: binObjects.length, take_count: 0, physical_object_count: 0, object_count: binObjects.length })
    : (allFilteredScopeSummary ?? filteredSummary ?? { record_count: allFilteredObjects.length, take_count: 0, physical_object_count: 0, object_count: allFilteredObjects.length });
  const filteredPopulationLine = buildMatchingObjectsMetaLine({
    objectCount: filteredSummary?.object_count,
    takeCount: filteredSummary?.take_count,
    physicalObjectCount: filteredSummary?.physical_object_count,
  });
  const selectedBinLine = selectedRange
    ? `Selected bin [${selectedRange.min.toFixed(4)}, ${selectedRange.max.toFixed(4)}] · ${displayedMatchingSummary.object_count} objects`
    : null;
  const matchingObjectsLoading = activeMatchingViewMode === "selected_bin" ? binObjectsLoading : allFilteredObjectsLoading;
  const matchingScopeLine = activeMatchingViewMode === "selected_bin"
    ? `Selected bin: ${buildMatchingObjectsMetaLine({
      objectCount: displayedMatchingSummary.object_count,
      takeCount: displayedMatchingSummary.take_count,
      physicalObjectCount: displayedMatchingSummary.physical_object_count,
    })}`
    : `All filtered: ${filteredPopulationLine}`;
  const matchingLimit = activeMatchingViewMode === "selected_bin" ? MATCHING_BIN_LIMIT : MATCHING_ALL_FILTERED_LIMIT;
  const matchingLimitNote = displayedMatchingSummary.object_count > displayedMatchingObjects.length
    ? `Showing first ${displayedMatchingObjects.length} of ${displayedMatchingSummary.object_count} objects`
    : `Showing ${displayedMatchingObjects.length} objects`;
  const scopedObjectCount = filteredSummary?.object_count ?? 0;
  const featureValuesMissing =
    Boolean(featureKey)
    && scopedObjectCount > 0
    && !loading
    && !error
    && (distribution?.stats.missing_pct ?? 0) >= 100
    && (distribution?.stats.count ?? 0) === 0;
  const featureFreshness = featureFreshnessState({
    featureKey,
    featureRegistered: Boolean(featureMeta),
    scopedObjectCount,
    distribution,
  });
  const featureActionsEnabled = featureJobActionsEnabled({
    featureKey,
    featureRegistered: Boolean(featureMeta),
    freshness: featureFreshness,
    scope: freshnessScope,
    pipelineId: resolvedPipelineId,
    diagnosticOnly: Boolean(featureMeta?.diagnostic_only),
    featureFamily: featureMeta?.family ?? null,
  });
  const registeredFeatureMissingMessage = featureValuesMissing
    ? buildRegisteredFeatureMissingHint(featureMeta?.display_name, featureKey, filters.mlSetId)
    : null;
  const showFreshnessPanel = featureFreshness === "all_missing" || featureFreshness === "partial_missing";
  const matchingEmptyMessage = !featureKey
    ? "Select a feature to load matching objects."
    : showFreshnessPanel && featureFreshness === "all_missing"
      ? "No matching objects with values yet. Start a reprocess or backfill job above."
    : activeMatchingViewMode === "selected_bin"
      ? "Click a histogram bin to inspect matching objects in that range."
      : matchingObjectsLoading
        ? "Loading matching objects..."
        : scopedObjectCount > 0
          ? "No objects in the current scope have a value for this feature."
          : "No objects match the current filters.";
  const selectedObjectThumbnailInfo = selectedObject ? thumbnailInfoByKey[`${selectedObject.take_id}::${selectedObject.object_id}`] : null;
  const selectedObjectStudioHref = useMemo(() => {
    if (!selectedObject || !featureMeta) return null;
    return buildStudioDeepLink({
      take_id: selectedObject.take_id,
      pipeline_id: selectedObject.pipeline_id,
      run_id: selectedObject.run_id,
      stage: featureStudioStage(featureMeta) ?? selectedObject.stage_id,
      object_id: selectedObject.object_id,
      physical_object_id: selectedObject.physical_object_id,
      feature: featureMeta.feature_key,
      dataset_id: selectedObject.dataset_id ?? filters.datasetId,
      session_id: selectedObject.session_id ?? filters.sessionId,
    });
  }, [selectedObject, featureMeta, filters.datasetId, filters.sessionId]);

  useEffect(() => {
    if (!displayedMatchingObjects.length) {
      setSelectedObject(null);
      return;
    }
    setSelectedObject((current) => {
      if (current && displayedMatchingObjects.some((item) => item.take_id === current.take_id && item.object_id === current.object_id)) return current;
      return displayedMatchingObjects[0] ?? null;
    });
  }, [displayedMatchingObjects]);

  useEffect(() => {
    setThumbnailInfoByKey({});
  }, [thumbnailMode, featureMeta?.semantic_group, displayedMatchingObjects]);

  return (
    <main className="page feature-analytics-page">
      <section className={`feature-analytics-layout ${sidebarCollapsed ? "sidebar-collapsed" : ""} ${inspectorCollapsed ? "inspector-collapsed" : ""}`}>
        <aside className={`feature-analytics-sidebar ${sidebarCollapsed ? "is-collapsed" : ""}`}>
          <div className="panel-collapse-header">
            <h2>Filters</h2>
            <button
              type="button"
              className="panel-collapse-toggle"
              onClick={() => setSidebarCollapsed((current) => !current)}
              aria-expanded={!sidebarCollapsed}
              aria-label={sidebarCollapsed ? "Expand filters panel" : "Collapse filters panel"}
              title={sidebarCollapsed ? "Expand filters panel" : "Collapse filters panel"}
            >
              {sidebarCollapsed ? "›" : "‹"}
            </button>
          </div>
          {!sidebarCollapsed && (
          <>
          <SidebarSection title="Scope">
            <label>Dataset<select value={filters.datasetId} onChange={(event) => setFilters((prev) => ({ ...prev, datasetId: event.target.value }))}><option value="">All datasets</option>{datasets.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label>
            <label>ML set<select value={filters.mlSetId} onChange={(event) => setFilters((prev) => ({ ...prev, mlSetId: event.target.value }))}><option value="">All ML sets</option>{mlSets.map((item) => <option key={item.id} value={item.id}>{item.name || item.id}</option>)}</select></label>
            <label>Dataset session / Acquisition session<select value={filters.sessionId} onChange={(event) => setFilters((prev) => ({ ...prev, sessionId: event.target.value }))}><option value="">All sessions</option>{datasetSessions.map((item) => <option key={item.id} value={item.id}>{sessionOptionLabel(item)}</option>)}</select></label>
            {selectedSessionMeta ? <p className="feature-session-meta">{sessionTypeLabel(selectedSessionMeta.session_type)}{selectedSessionMeta.calibration_id ? ` • calib ${selectedSessionMeta.calibration_id}` : ""}</p> : null}
          </SidebarSection>

          <SidebarSection title="Labels">
            <ChipToggleGroup label="Labeled superclass" values={filterOptions?.labeled_superclasses ?? []} selected={filters.labeledSuperclasses} onChange={(labeledSuperclasses) => setFilters((prev) => ({ ...prev, labeledSuperclasses }))} mutedValue="UNKNOWN" />
            <ChipToggleGroup label="Processed superclass" values={filterOptions?.processed_superclasses ?? []} selected={filters.processedSuperclasses} onChange={(processedSuperclasses) => setFilters((prev) => ({ ...prev, processedSuperclasses }))} mutedValue="UNKNOWN" />
            <MultiValueAutocomplete label="Normalized class" values={filters.normalizedClasses} options={filterOptions?.normalized_classes ?? []} placeholder="Add canonical class" onChange={(normalizedClasses) => setFilters((prev) => ({ ...prev, normalizedClasses }))} />
            <MultiValueAutocomplete label="Processed class" values={filters.processedClasses} options={filterOptions?.processed_classes ?? []} placeholder="Add processed class" onChange={(processedClasses) => setFilters((prev) => ({ ...prev, processedClasses }))} />
            <MultiValueAutocomplete label="Raw labels" values={filters.rawLabels} options={filterOptions?.raw_labels ?? []} placeholder="Add source/operator label" onChange={(rawLabels) => setFilters((prev) => ({ ...prev, rawLabels }))} />
          </SidebarSection>

          <SidebarSection title="Identity">
            <MultiValueAutocomplete label="Physical object ID" values={filters.physicalObjectIds} options={filterOptions?.physical_object_ids ?? []} placeholder="Add physical_object_id" onChange={(physicalObjectIds) => setFilters((prev) => ({ ...prev, physicalObjectIds }))} />
            <MultiValueAutocomplete label="Take ID" values={filters.takeIds} options={filterOptions?.take_ids ?? []} placeholder="Add take_id" onChange={(takeIds) => setFilters((prev) => ({ ...prev, takeIds }))} />
          </SidebarSection>

          <SidebarSection title="Processing">
            <ChipToggleGroup label="Validation" values={validationOptions} selected={filters.validationStatus} onChange={(validationStatus) => setFilters((prev) => ({ ...prev, validationStatus }))} />
            <ChipToggleGroup label="Split" values={splitOptions} selected={filters.split} onChange={(split) => setFilters((prev) => ({ ...prev, split }))} />
            <label>Pipeline<select value={filters.pipelineId} onChange={(event) => setFilters((prev) => ({ ...prev, pipelineId: event.target.value }))}><option value="">All pipelines</option>{pipelines.map((item) => <option key={item.id} value={item.id}>{item.display_name}</option>)}</select></label>
            <label>Calibration<select value={filters.calibrationId} onChange={(event) => setFilters((prev) => ({ ...prev, calibrationId: event.target.value }))}><option value="">All calibrations</option>{(filterOptions?.calibration_ids ?? []).map((item) => <option key={item} value={item}>{item}</option>)}</select></label>
            <label>Date from<input type="date" value={filters.dateFrom} onChange={(event) => setFilters((prev) => ({ ...prev, dateFrom: event.target.value }))} /></label>
            <label>Date to<input type="date" value={filters.dateTo} onChange={(event) => setFilters((prev) => ({ ...prev, dateTo: event.target.value }))} /></label>
          </SidebarSection>
          </>
          )}
        </aside>

        <section className="feature-analytics-main">
          <div className="feature-scope-bar" title={compactSummaryBar}>{compactSummaryBar}</div>

          <header className="feature-analytics-toolbar">
            <label className="feature-toolbar-field feature-toolbar-feature">
              <span>Feature</span>
              <div className="feature-selector-stack">
                <input
                  type="search"
                  value={featureSearch}
                  placeholder="Search feature catalog"
                  aria-label="Search feature catalog"
                  onChange={(event) => setFeatureSearch(event.target.value)}
                />
                <select value={featureKey} onChange={(event) => setFeatureKey(event.target.value)}>
                  {groupedFeatureDefinitions.map((group) => (
                    <optgroup key={`${group.sectionLabel}_${group.familyLabel}`} label={`${group.sectionLabel} / ${group.familyLabel}`}>
                      {group.features.map((item) => (
                        <option key={item.feature_key} value={item.feature_key}>
                          {featureOptionLabel(item)}
                        </option>
                      ))}
                    </optgroup>
                  ))}
                </select>
              </div>
            </label>
            <label className="feature-toolbar-field">
              <span>Group by</span>
              <select value={groupBy} onChange={(event) => setGroupBy(event.target.value as FeatureGroupBy)}><option value="processed_superclass">Processed superclass</option><option value="labeled_superclass">Labeled superclass</option><option value="normalized_class">Normalized class</option><option value="processed_class">Processed class</option><option value="raw_label">Raw label</option><option value="physical_object_id">Physical object ID</option></select>
            </label>
            <label className="feature-toolbar-field">
              <span>Mode</span>
              <select value={mode} onChange={(event) => setMode(event.target.value as "count" | "density")}><option value="count">Count</option><option value="density">Normalized density</option></select>
            </label>
            <label className="feature-toolbar-field feature-toolbar-bins">
              <span>Bins</span>
              <input type="range" min={8} max={64} value={binCount} onChange={(event) => setBinCount(Number(event.target.value))} />
              <output>{binCount}</output>
            </label>
            <label className="feature-toolbar-check">
              <input type="checkbox" checked={showBinEdgeLabels} onChange={(event) => setShowBinEdgeLabels(event.target.checked)} />
              <span>Bin edges</span>
            </label>
            <label className="feature-toolbar-field">
              <span>Thumbs</span>
              <select value={thumbnailMode} onChange={(event) => setThumbnailMode(event.target.value as ThumbnailMode)}>
                <option value="auto">Auto</option>
                <option value="heightmap">Heightmap</option>
                <option value="reflectance">Source/reflectance</option>
                <option value="classification_overlay">Classification overlay</option>
              </select>
            </label>
          </header>

          <div className={`feature-analytics-workspace ${selectedRange ? "has-selected-bin" : ""}`}>
          <section className="feature-analytics-plot">
            {loading ? <div className="empty-state">Loading feature analytics...</div> : null}
            {!loading && error ? <div className="empty-state"><strong>{error}</strong></div> : null}
            {!loading && !error && zeroStateMessage ? <div className="empty-state"><strong>No feature records match the current filters.</strong><p>{zeroStateMessage}</p></div> : null}
            {!loading && !error && !zeroStateMessage && showFreshnessPanel ? (
              <FeatureFreshnessPanel
                featureKey={featureKey}
                featureMeta={featureMeta}
                freshness={featureFreshness}
                scope={freshnessScope}
                pipelineId={resolvedPipelineId}
                distribution={distribution}
                scopedObjectCount={scopedObjectCount}
                actionsEnabled={featureActionsEnabled}
                onJobFinished={refreshAnalytics}
              />
            ) : null}
            {!loading && !error && !zeroStateMessage && !showFreshnessPanel && registeredFeatureMissingMessage ? (
              <div className="empty-state">
                <strong>Registered feature has no values in the current scope.</strong>
                <p>{registeredFeatureMissingMessage}</p>
              </div>
            ) : null}
            {!loading && !error && distribution?.groups.length ? (
              <div className="feature-analytics-plot-chart">
                <div className="feature-legend">{distribution.groups.map((group, idx) => {
                  const visual = groupVisualStyle(group.group, idx);
                  return <span key={group.group} className={visual.className}><i style={{ backgroundColor: visual.color }} />{group.group}</span>;
                })}</div>
                {histogramLayout.scrollable ? (
                  <>
                    <div
                      ref={histogramScrollRef}
                      className="feature-histogram-scroll is-scrollable"
                      style={{ maxHeight: histogramLayout.maxScrollHeightPx }}
                    >
                      <div className="feature-histogram-grid">
                        {distribution.groups.map((group, groupIdx) => (
                          <div
                            key={group.group}
                            ref={(node) => {
                              histogramBandRefs.current[groupIdx] = node;
                            }}
                            className="feature-histogram-band"
                          >
                            <div className={`feature-band-label ${groupVisualStyle(group.group, groupIdx).className}`}>{group.group}</div>
                            <div className={`feature-histogram-row ${groupVisualStyle(group.group, groupIdx).className}`}>
                              {group.bins.map((value, idx) => {
                                const edgeStart = distribution.range?.edges[idx] ?? 0;
                                const edgeEnd = distribution.range?.edges[idx + 1] ?? edgeStart;
                                const density = group.count > 0 ? Number(value) / group.count : 0;
                                const pct = maxY ? (Number(value) / maxY) * 100 : 0;
                                const visual = groupVisualStyle(group.group, groupIdx);
                                return (
                                  <button
                                    type="button"
                                    key={`${group.group}_${idx}`}
                                    className={`feature-bin ${visual.className}`}
                                    title={`${group.group} | ${compactNumber(edgeStart)} - ${compactNumber(edgeEnd)} | count ${compactNumber(Number(value))}${mode === "density" ? ` | density ${compactNumber(Number(value))}` : ` | density ${compactNumber(density)}`}`}
                                    onClick={() => selectHistogramBin({ min: edgeStart, max: edgeEnd }, setSelectedRange, setMatchingViewMode)}
                                  >
                                    <span style={{ height: `${Math.max(2, pct)}%`, backgroundColor: visual.color }} />
                                  </button>
                                );
                              })}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                    {histogramScrollMessage ? <p className="feature-histogram-scroll-hint" aria-live="polite">{histogramScrollMessage}</p> : null}
                  </>
                ) : (
                  <div className="feature-histogram-grid">
                    {distribution.groups.map((group, groupIdx) => (
                      <div key={group.group} className="feature-histogram-band">
                        <div className={`feature-band-label ${groupVisualStyle(group.group, groupIdx).className}`}>{group.group}</div>
                        <div className={`feature-histogram-row ${groupVisualStyle(group.group, groupIdx).className}`}>
                          {group.bins.map((value, idx) => {
                            const edgeStart = distribution.range?.edges[idx] ?? 0;
                            const edgeEnd = distribution.range?.edges[idx + 1] ?? edgeStart;
                            const density = group.count > 0 ? Number(value) / group.count : 0;
                            const pct = maxY ? (Number(value) / maxY) * 100 : 0;
                            const visual = groupVisualStyle(group.group, groupIdx);
                            return (
                              <button
                                type="button"
                                key={`${group.group}_${idx}`}
                                className={`feature-bin ${visual.className}`}
                                title={`${group.group} | ${compactNumber(edgeStart)} - ${compactNumber(edgeEnd)} | count ${compactNumber(Number(value))}${mode === "density" ? ` | density ${compactNumber(Number(value))}` : ` | density ${compactNumber(density)}`}`}
                                onClick={() => selectHistogramBin({ min: edgeStart, max: edgeEnd }, setSelectedRange, setMatchingViewMode)}
                              >
                                <span style={{ height: `${Math.max(2, pct)}%`, backgroundColor: visual.color }} />
                              </button>
                            );
                          })}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
                {distribution.range?.edges?.length ? (
                  <div className="feature-axis feature-analytics-plot-axis" aria-label="Feature value axis">
                    {tickIndexes.map((tickIndex) => {
                      const leftPct = distribution.range?.edges?.length && distribution.range.edges.length > 1
                        ? (tickIndex / (distribution.range.edges.length - 1)) * 100
                        : 0;
                      const tickValue = distribution.range?.edges?.[tickIndex] ?? 0;
                      return (
                        <span key={`tick_${tickIndex}`} className="feature-axis-tick" style={{ left: `${leftPct}%` }}>
                          {compactNumber(tickValue)}
                        </span>
                      );
                    })}
                    {showBinEdgeLabels ? (
                      <div className="feature-axis-edge-grid">
                        {distribution.range.edges.map((edgeValue, edgeIdx) => (
                          <span key={`edge_${edgeIdx}`} className="feature-axis-edge-label">{compactNumber(edgeValue)}</span>
                        ))}
                      </div>
                    ) : null}
                  </div>
                ) : null}
              </div>
            ) : null}
          </section>

          <section className="feature-matching-objects">
            <div className="feature-matching-header">
              <div className="feature-matching-title-row">
                <h3>Matching objects</h3>
                {selectedRange ? (
                  <div className="feature-matching-mode-toggle" role="tablist" aria-label="Matching objects view">
                    <button
                      type="button"
                      role="tab"
                      aria-selected={activeMatchingViewMode === "selected_bin"}
                      className={activeMatchingViewMode === "selected_bin" ? "active" : ""}
                      onClick={() => setMatchingViewMode("selected_bin")}
                    >
                      Selected bin
                    </button>
                    <button
                      type="button"
                      role="tab"
                      aria-selected={activeMatchingViewMode === "all_filtered"}
                      className={activeMatchingViewMode === "all_filtered" ? "active" : ""}
                      onClick={() => setMatchingViewMode("all_filtered")}
                    >
                      All filtered
                    </button>
                  </div>
                ) : null}
              </div>
              <div className="feature-matching-counts">
                {activeMatchingViewMode === "selected_bin" && selectedBinLine ? (
                  <>
                    <span className="feature-matching-count-primary">{selectedBinLine}</span>
                    <span className="feature-matching-count-secondary">{matchingScopeLine}{displayedMatchingSummary.object_count > displayedMatchingObjects.length ? ` · showing first ${displayedMatchingObjects.length}` : ""}</span>
                    <span className="feature-matching-count-tertiary">Filtered total: {filteredPopulationLine}</span>
                  </>
                ) : (
                  <>
                    <span className="feature-matching-count-primary">All filtered · {filteredPopulationLine}</span>
                    <span className="feature-matching-count-secondary">{matchingLimitNote}{displayedMatchingSummary.object_count > displayedMatchingObjects.length ? ` · limit ${matchingLimit}` : ""}</span>
                  </>
                )}
                <span className="feature-matching-count-tertiary">Thumbnail request: {thumbnailModeLabel(thumbnailMode)}</span>
              </div>
            </div>
            {matchingObjectsLoading ? <div className="feature-matching-empty">Loading matching objects...</div> : null}
            {!matchingObjectsLoading && !displayedMatchingObjects.length ? (
              <div className="feature-matching-empty">{matchingEmptyMessage}</div>
            ) : null}
            {!matchingObjectsLoading && displayedMatchingObjects.length ? (
              <div className="feature-matching-table-wrap">
                <table className="feature-matching-table">
                  <thead><tr><th title={`Preview thumbnail · requested ${thumbnailModeLabel(thumbnailMode)}`}>Preview</th><th>Take</th><th>Physical object</th><th title="ML-set / dataset labeled superclass">Labeled</th><th title="Pipeline-result superclass">Processed</th><th title="normalized class / processed class">Class</th><th>Split</th><th>Feature</th><th title="Canonical pipeline vs backfill sidecar">Source</th><th>Conf.</th><th>Inspect</th></tr></thead>
                  <tbody>
                    {displayedMatchingObjects.map((item) => {
                      const classCell = formatMatchingObjectClassCell(item);
                      const isSelected = selectedObject?.take_id === item.take_id && selectedObject?.object_id === item.object_id;
                      const rowWarning = matchingObjectWarning(item);
                      const studioHref = buildStudioDeepLink({
                        take_id: item.take_id,
                        pipeline_id: item.pipeline_id,
                        run_id: item.run_id,
                        stage: featureStudioStage(featureMeta) ?? item.stage_id,
                        object_id: item.object_id,
                        physical_object_id: item.physical_object_id,
                        feature: featureKey,
                        dataset_id: item.dataset_id ?? filters.datasetId,
                        session_id: item.session_id ?? filters.sessionId,
                      });
                      return (
                        <tr
                          key={`${item.take_id}_${item.object_id}`}
                          className={isSelected ? "is-selected" : ""}
                          title={matchingObjectRowTitle(item)}
                          tabIndex={0}
                          onClick={() => setSelectedObject(item)}
                          onKeyDown={(event) => {
                            if (event.key === "Enter" || event.key === " ") {
                              event.preventDefault();
                              setSelectedObject(item);
                            }
                          }}
                        >
                          <td>
                            <ObjectThumbCell
                              item={item}
                              thumbnailMode={thumbnailMode}
                              semanticGroup={featureMeta?.semantic_group ?? null}
                              thumbnailInfo={thumbnailInfoByKey[`${item.take_id}::${item.object_id}`] ?? null}
                              onThumbnailInfo={(info) => setThumbnailInfoByKey((prev) => ({ ...prev, [`${item.take_id}::${item.object_id}`]: info }))}
                              onHover={(payload) => setHoverPreview(payload)}
                              onLeave={() => setHoverPreview(null)}
                            />
                          </td>
                          <td className="feature-cell-mono" title={item.take_id}>{item.take_id}</td>
                          <td className="feature-cell-mono">{item.physical_object_id ?? "-"}</td>
                          <td title={rowWarning ?? item.labeled_superclass_source ?? undefined}>
                            <span>{item.labeled_superclass ?? "-"}</span>
                            {rowWarning ? <small className="feature-warning-inline">warn</small> : null}
                          </td>
                          <td>{item.processed_superclass ?? "-"}</td>
                          <td className="feature-cell-class">
                            <span>{classCell.primary}</span>
                            {classCell.secondary ? <small>{classCell.secondary}</small> : null}
                          </td>
                          <td>
                            <span>{item.split ?? "-"}</span>
                            {item.validation_status ? <span className="feature-validation-chip">{item.validation_status}</span> : null}
                          </td>
                          <td className="feature-cell-num">{compactNumber(item.feature_value)}</td>
                          <td title={featureSourceTooltip(item.feature_source)}><span className={`feature-source-chip source-${featureSourceLabel(item.feature_source)}`}>{featureSourceLabel(item.feature_source)}</span></td>
                          <td className="feature-cell-num">{item.confidence == null ? "-" : compactNumber(item.confidence)}</td>
                          <td>
                            <a
                              className="feature-open-link"
                              href={studioHref}
                              target="_blank"
                              rel="noreferrer"
                              aria-label={`Open ${item.take_id} in Studio`}
                              onClick={(event) => event.stopPropagation()}
                            >
                              Studio ↗
                            </a>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : null}
            {hoverPreview ? (
              <div
                className="feature-thumb-hover-card"
                style={{ left: `${hoverPreview.x + 14}px`, top: `${hoverPreview.y + 14}px` }}
              >
                <img alt={`Object ${hoverPreview.item.object_id} preview`} src={hoverPreview.src} loading="lazy" />
                <small>{hoverPreview.item.take_id} • physical {hoverPreview.item.physical_object_id ?? "-"}</small>
                <small>{hoverPreview.item.labeled_superclass ?? "UNKNOWN"} → {hoverPreview.item.processed_superclass ?? "UNKNOWN"} • {compactNumber(hoverPreview.item.feature_value)}</small>
                <small>Thumbnail request: {thumbnailModeLabel(thumbnailMode)}</small>
                {thumbnailInfoByKey[`${hoverPreview.item.take_id}::${hoverPreview.item.object_id}`] ? (
                  <small>
                    Resolved: {thumbnailResolvedSourceLabel(thumbnailInfoByKey[`${hoverPreview.item.take_id}::${hoverPreview.item.object_id}`].resolved_source)}
                    {thumbnailInfoByKey[`${hoverPreview.item.take_id}::${hoverPreview.item.object_id}`].fallback_used ? ` · fallback: ${thumbnailInfoByKey[`${hoverPreview.item.take_id}::${hoverPreview.item.object_id}`].fallback_reason ?? "yes"}` : ""}
                    {thumbnailInfoByKey[`${hoverPreview.item.take_id}::${hoverPreview.item.object_id}`].colorized ? " · colorized" : " · grayscale"}
                  </small>
                ) : null}
                {matchingObjectWarning(hoverPreview.item) ? <small>{matchingObjectWarning(hoverPreview.item)}</small> : null}
                {(hoverPreview.item.raw_labels ?? []).length ? <small>Raw: {(hoverPreview.item.raw_labels ?? []).join(", ")}</small> : null}
              </div>
            ) : null}
          </section>
          </div>
        </section>

        <aside className={`feature-analytics-inspector ${inspectorCollapsed ? "is-collapsed" : ""}`}>
          <div className="panel-collapse-header">
            <h2>Feature inspector</h2>
            <button
              type="button"
              className="panel-collapse-toggle"
              onClick={() => setInspectorCollapsed((current) => !current)}
              aria-expanded={!inspectorCollapsed}
              aria-label={inspectorCollapsed ? "Expand feature inspector" : "Collapse feature inspector"}
              title={inspectorCollapsed ? "Expand feature inspector" : "Collapse feature inspector"}
            >
              {inspectorCollapsed ? "‹" : "›"}
            </button>
          </div>
          {inspectorCollapsed ? null : !featureMeta ? <div className="empty-state">Select a feature.</div> : (
            <div className="inspector-list">
              <p><strong>{featureMeta.display_name}</strong></p>
              <p>Key: <code>{featureMeta.feature_key}</code></p>
              <p>Display name: {featureMeta.display_name}</p>
              <p>Group: {featureFamilyLabel(featureMeta)}</p>
              <p>Unit: {featureMeta.unit ?? "-"}</p>
              <p>Source stage: {featureMeta.source_stage ?? "-"}</p>
              <p>Higher is worse: {featureMeta.higher_is_worse == null ? "unspecified" : featureMeta.higher_is_worse ? "yes" : "no"}</p>
              <p>Studio stage: {featureStudioStage(featureMeta) ?? "-"}</p>
              <p>Grouping source: {groupBy === "labeled_superclass" ? "ML-set / dataset label metadata" : groupBy === "processed_superclass" ? "Pipeline classification result" : "-"}</p>
              <div className="feature-catalog-badges">
                {featureBadgeLabels(featureMeta).map((badge) => <span key={badge} className="feature-badge">{badge}</span>)}
                <span className="feature-badge subdued">{featureSectionLabel(featureMeta)}</span>
              </div>
              <hr />
              <p><strong>Formula</strong></p>
              <p>{featureMeta.formula ?? "TODO"}</p>
              <p><strong>Description</strong></p>
              <p>{featureMeta.description ?? featureMeta.algorithm_summary ?? "No summary."}</p>
              <p><strong>Algorithm summary</strong></p>
              <p>{featureMeta.algorithm_summary ?? "No summary."}</p>
              <p><strong>Interpretation</strong></p>
              <p>{featureMeta.interpretation ?? "No interpretation documented yet."}</p>
              {(featureMeta.caveats ?? []).length ? (
                <>
                  <p><strong>Caveats</strong></p>
                  <div className="feature-caveat-list">
                    {(featureMeta.caveats ?? []).map((item) => <p key={item}>{item}</p>)}
                  </div>
                </>
              ) : null}
              {featureMeta.feature_key === "surface_sphere_fit_rmse_mm" ? (
                <p className="feature-warning-text">Raw RMSE is scale-dependent and visible-surface-only; prefer normalized residuals and sphere plausibility for classification.</p>
              ) : null}
              <hr />
              <p><strong>Distribution</strong></p>
              <p>Count: {distribution?.stats.count ?? 0}</p>
              <p>Missing: {distribution?.stats.missing ?? 0}</p>
              <p>Missing %: {(distribution?.stats.missing_pct ?? 0).toFixed(2)}%</p>
              {showFreshnessPanel ? (
                <>
                  <p><strong>Freshness</strong></p>
                  <p>{featureFreshness === "all_missing" ? "No persisted values in current scope." : `Missing for ${distribution?.stats.missing ?? 0} / ${scopedObjectCount} objects.`}</p>
                </>
              ) : null}
              <p>Min / Max: {distribution?.stats.min ?? "-"} / {distribution?.stats.max ?? "-"}</p>
              <p>Mean / Std: {distribution?.stats.mean ?? "-"} / {distribution?.stats.std ?? "-"}</p>
              {selectedObject ? (
                <>
                  <hr />
                  <p><strong>Selected sample</strong></p>
                  <p>Take: <code>{selectedObject.take_id}</code></p>
                  <p>Physical object: <code>{selectedObject.physical_object_id ?? "-"}</code></p>
                  <p>Superclass: {(selectedObject.labeled_superclass ?? "UNKNOWN")} → {(selectedObject.processed_superclass ?? "UNKNOWN")}</p>
                  <p>Class: {formatMatchingObjectClassCell(selectedObject).primary}{formatMatchingObjectClassCell(selectedObject).secondary ? ` / ${formatMatchingObjectClassCell(selectedObject).secondary}` : ""}</p>
                  <p>Feature value: {compactNumber(selectedObject.feature_value)}</p>
                  <p title={featureSourceTooltip(selectedObject.feature_source)}>Feature source: {featureSourceLabel(selectedObject.feature_source)}</p>
                  <p>Confidence: {selectedObject.confidence == null ? "-" : compactNumber(selectedObject.confidence)}</p>
                  <p>Thumbnail request: {thumbnailModeLabel(thumbnailMode)}</p>
                  {selectedObjectThumbnailInfo ? (
                    <p>
                      Thumbnail source: {thumbnailResolvedSourceLabel(selectedObjectThumbnailInfo.resolved_source)}
                      {selectedObjectThumbnailInfo.fallback_used ? ` · fallback: ${selectedObjectThumbnailInfo.fallback_reason ?? "yes"}` : ""}
                      {selectedObjectThumbnailInfo.colorized ? " · colorized" : " · grayscale"}
                    </p>
                  ) : null}
                  {matchingObjectWarning(selectedObject) ? <p className="feature-warning-text">{matchingObjectWarning(selectedObject)}</p> : null}
                  {selectedObjectStudioHref ? (
                    <p>
                      <a className="feature-open-link" href={selectedObjectStudioHref} target="_blank" rel="noreferrer">
                        Open selected sample in Studio
                      </a>
                    </p>
                  ) : null}
                </>
              ) : null}
            </div>
          )}
        </aside>
      </section>
    </main>
  );
}

function SidebarSection({ title, children }: { title: string; children: ReactNode }) {
  return <div className="feature-filter-section"><h3>{title}</h3>{children}</div>;
}

function ChipToggleGroup({
  label,
  values,
  selected,
  onChange,
  mutedValue,
}: {
  label: string;
  values: string[];
  selected: string[];
  onChange: (next: string[]) => void;
  mutedValue?: string;
}) {
  if (!values.length) return null;
  return (
    <div className="feature-chip-filter">
      <span>{label}</span>
      <div className="feature-chip-grid">
        {values.map((value) => {
          const active = selected.includes(value);
          const muted = mutedValue && value.toUpperCase() === mutedValue.toUpperCase();
          return (
            <button
              type="button"
              key={value}
              className={`feature-filter-chip ${active ? "active" : ""} ${muted ? "muted" : ""}`}
              onClick={() => onChange(active ? selected.filter((item) => item !== value) : [...selected, value])}
            >
              {value}
            </button>
          );
        })}
      </div>
    </div>
  );
}

function MultiValueAutocomplete({
  label,
  values,
  options,
  placeholder,
  onChange,
}: {
  label: string;
  values: string[];
  options: string[];
  placeholder: string;
  onChange: (next: string[]) => void;
}) {
  const [draft, setDraft] = useState("");
  const datalistId = useMemo(() => `feature_filter_${label.toLowerCase().replace(/[^a-z0-9]+/g, "_")}`, [label]);
  const available = useMemo(() => options.filter((option) => !values.includes(option)), [options, values]);

  function commit(value: string) {
    const next = value.trim();
    if (!next) return;
    onChange(uniq([...values, next]));
    setDraft("");
  }

  function handleKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key === "Enter" || event.key === ",") {
      event.preventDefault();
      commit(draft);
    }
    if (event.key === "Backspace" && !draft && values.length) {
      onChange(values.slice(0, -1));
    }
  }

  return (
    <div className="feature-token-filter">
      <span>{label}</span>
      <div className="feature-token-list">
        {values.map((value) => (
          <button type="button" key={value} className="feature-token" onClick={() => onChange(values.filter((item) => item !== value))}>
            {value} <strong>×</strong>
          </button>
        ))}
        <input
          value={draft}
          list={datalistId}
          placeholder={placeholder}
          onChange={(event) => setDraft(event.target.value)}
          onBlur={() => {
            if (draft.trim()) commit(draft);
          }}
          onKeyDown={handleKeyDown}
        />
      </div>
      <datalist id={datalistId}>
        {available.map((option) => <option key={option} value={option} />)}
      </datalist>
    </div>
  );
}

function ObjectThumbCell({
  item,
  thumbnailMode,
  semanticGroup,
  thumbnailInfo,
  onThumbnailInfo,
  onHover,
  onLeave,
}: {
  item: FeatureAnalyticsObject;
  thumbnailMode: ThumbnailMode;
  semanticGroup: string | null;
  thumbnailInfo: FeatureAnalyticsThumbnailInfo | null;
  onThumbnailInfo: (info: FeatureAnalyticsThumbnailInfo) => void;
  onHover: (payload: { x: number; y: number; src: string; item: FeatureAnalyticsObject }) => void;
  onLeave: () => void;
}) {
  const [failed, setFailed] = useState(false);
  const src = objectThumbnailUrl(item.take_id, item.object_id, 56, thumbnailMode, semanticGroup);

  useEffect(() => {
    let cancelled = false;
    if (thumbnailInfo) return;
    void api.objectThumbnailInfo(item.take_id, item.object_id, { mode: thumbnailMode, semantic_group: semanticGroup ?? undefined })
      .then((info) => {
        if (!cancelled) onThumbnailInfo(info);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [item.take_id, item.object_id, semanticGroup, thumbnailInfo, thumbnailMode]);

  function handleMove(event: MouseEvent<HTMLDivElement>) {
    onHover({ x: event.clientX, y: event.clientY, src, item });
  }

  return (
    <div
      className="feature-object-thumb"
      title={
        thumbnailInfo
          ? `Thumbnail: ${thumbnailModeLabel(thumbnailMode)} -> ${thumbnailResolvedSourceLabel(thumbnailInfo.resolved_source)}${thumbnailInfo.fallback_used ? ` · fallback: ${thumbnailInfo.fallback_reason ?? "yes"}` : ""}${thumbnailInfo.colorized ? " · colorized" : " · grayscale"}`
          : `Thumbnail request: ${thumbnailModeLabel(thumbnailMode)}`
      }
      onMouseEnter={handleMove}
      onMouseMove={handleMove}
      onMouseLeave={onLeave}
    >
      {!failed ? (
        <img
          alt={`Object ${item.object_id} thumbnail`}
          src={src}
          loading="lazy"
          onError={() => setFailed(true)}
        />
      ) : (
        <div className="feature-object-thumb-placeholder" />
      )}
    </div>
  );
}
