import { Fragment, useEffect, useMemo, useState } from "react";
import { api, fileUrl, type DatasetLabelSummary, type DatasetMlSet, type DatasetSessionDetailSummary, type DatasetSessionSummary, type DatasetSummary, type MLSetMemberRow, type MLSetReviewFacetsResponse, type MLSetSummaryResponse, type MaterializeMlIngestionResponse, type PhysicalObjectSummary, type TakeDetail, type TakeSummary } from "../api/client";
import DatasetSessionDrawer from "../components/datasets/DatasetSessionDrawer";
import MLSetDetailDrawer from "../components/datasets/MLSetDetailDrawer";
import PhysicalObjectDetailDrawer from "../components/datasets/PhysicalObjectDetailDrawer";
import MLSetIngestionWizard from "../components/ingestion/MLSetIngestionWizard";
import { buildStudioDeepLink } from "../studioDeepLink";

type SplitBucket = "training" | "validation" | "test" | "benchmark" | "production-shadow" | "cross-site";
type DatasetTab = "overview" | "takes" | "physical_objects" | "objects" | "labels" | "ml_sets" | "splits";

type ObjectAnnotation = {
  id?: string;
  candidate_id?: string;
  labels?: unknown[];
  expected_class?: string;
  notes?: string;
  validation_status?: string;
  confidence?: number;
};

type SavedCollection = {
  id: string;
  name: string;
  filters: {
    dataset: string;
    session: string;
    validation: string;
    split: "all" | SplitBucket;
    search: string;
    tag: string;
    expectedClass: string;
    calibrationOnly: boolean;
    fromDate: string;
    toDate: string;
    datePreset: string;
    governanceFlags: string[];
  };
};

const splitBuckets: SplitBucket[] = ["training", "validation", "test", "benchmark", "production-shadow", "cross-site"];
const TAKES_PAGE_LIMIT = 50;
const DEBUG_DATASETS_TAKES = false;
const DATASETS_COLLECTIONS_KEY = "sensor_studio.datasets.saved_collections.v1";
const KEYBOARD_CURATION_ENABLED = false;

const DATE_PRESETS = ["custom", "today", "last_24h", "last_7d", "last_30d", "this_week", "this_shift"] as const;
type DatePreset = (typeof DATE_PRESETS)[number];

const GOVERNANCE_FILTERS = [
  "missing_expected_class",
  "missing_split",
  "missing_tags",
  "missing_calibration",
  "processing_failed",
  "processing_incomplete",
  "no_objects_detected",
  "low_confidence_candidates",
  "classifier_disagreement",
  "outlier_candidate",
] as const;
type GovernanceFilter = (typeof GOVERNANCE_FILTERS)[number];

type PagedSummaryCounts = {
  validation: Record<string, number>;
  missing_labels: number;
  missing_split: number;
  missing_calibration: number;
  processing_failed: number;
  processing_incomplete: number;
  no_objects_detected: number;
};
type BulkModalKind = "move_dataset" | "move_session" | "ml_set" | "split" | "tags" | "validation" | "expected_class";
type DatasetsPageProfilerSnapshot = {
  captured_at: string;
  append: boolean;
  frontend_request_ms: number;
  frontend_commit_ms: number;
  page_item_count: number;
  total_visible_rows: number;
  backend_profile: NonNullable<Awaited<ReturnType<typeof api.pagedTakes>>["profile"]> | null;
};

type MlSetScopedView = "takes" | "object_groups";
type MlSetMembershipStatusFilter = "all" | "review_required" | "default_trainable" | "excluded" | "uncertain";
type MlSetScopeState = {
  mlSetId: string;
  normalizedClass: string;
  superclass: string;
  physicalObjectId: string;
};
type MlSetObjectGroup = {
  id: string;
  physicalObjectId: string;
  normalizedClass: string;
  superclass: string;
  rawLabels: string[];
  splitValues: string[];
  sessions: string[];
  warningCodes: string[];
  takeCount: number;
  members: MLSetMemberRow[];
};

function countFromValidation(summary: PagedSummaryCounts | null, key: string, fallback = 0): number {
  return Number(summary?.validation?.[key] ?? fallback);
}

function readDatasetsProfilerEnabled() {
  if (typeof window === "undefined") return false;
  const params = new URLSearchParams(window.location.search);
  return params.get("profile") === "datasets" || params.get("datasets_profile") === "1";
}

function formatProfilerMs(value: number | null | undefined) {
  if (typeof value !== "number" || Number.isNaN(value)) return "-";
  return `${value.toFixed(1)} ms`;
}

function formatFacetCountLabel(label: string, count?: number | null) {
  return typeof count === "number" && count >= 0 ? `${label} · ${count}` : label;
}

function formatObjectFacetLabel(option: NonNullable<MLSetReviewFacetsResponse["object_ids"]>[number]) {
  const parts = [`${option.value} · ${option.take_count} takes`];
  if (option.normalized_class) parts.push(option.normalized_class);
  if (option.superclass) parts.push(option.superclass);
  return parts.join(" · ");
}

export default function DatasetsPage() {
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [sessions, setSessions] = useState<DatasetSessionSummary[]>([]);
  const [datasetLabelSummary, setDatasetLabelSummary] = useState<DatasetLabelSummary | null>(null);
  const [mlSets, setMlSets] = useState<DatasetMlSet[]>([]);
  const [physicalObjects, setPhysicalObjects] = useState<PhysicalObjectSummary[]>([]);
  const [leftPanelCollapsed, setLeftPanelCollapsed] = useState(false);
  const [rightPanelCollapsed, setRightPanelCollapsed] = useState(false);
  const [modalDatasetSessions, setModalDatasetSessions] = useState<DatasetSessionSummary[]>([]);
  const [pagedTakes, setPagedTakes] = useState<TakeSummary[]>([]);
  const [offset, setOffset] = useState(0);
  const [hasMore, setHasMore] = useState(false);
  const [selectedDataset, setSelectedDataset] = useState("");
  const [selectedSession, setSelectedSession] = useState("");
  const [sessionDrawerId, setSessionDrawerId] = useState("");
  const [selectedSessionDetailSummary, setSelectedSessionDetailSummary] = useState<DatasetSessionDetailSummary | null>(null);
  const [selectedTakeId, setSelectedTakeId] = useState("");
  const [selectedObjectId, setSelectedObjectId] = useState("");
  const [selectedPhysicalObjectId, setSelectedPhysicalObjectId] = useState("");
  const [selectedMlSetId, setSelectedMlSetId] = useState("");
  const [activeMlSetScope, setActiveMlSetScope] = useState<MlSetScopeState | null>(null);
  const [mlSetScopedView, setMlSetScopedView] = useState<MlSetScopedView>("takes");
  const [mlSetMemberRows, setMlSetMemberRows] = useState<MLSetMemberRow[]>([]);
  const [mlSetMemberOffset, setMlSetMemberOffset] = useState(0);
  const [mlSetMemberHasMore, setMlSetMemberHasMore] = useState(false);
  const [mlSetMemberFilteredCount, setMlSetMemberFilteredCount] = useState<number | null>(null);
  const [mlSetScopeSummary, setMlSetScopeSummary] = useState<MLSetSummaryResponse | null>(null);
  const [mlSetReviewFacets, setMlSetReviewFacets] = useState<MLSetReviewFacetsResponse | null>(null);
  const [mlSetObjectIdFilter, setMlSetObjectIdFilter] = useState("");
  const [mlSetRawLabelFilter, setMlSetRawLabelFilter] = useState("");
  const [mlSetNormalizedClassFilter, setMlSetNormalizedClassFilter] = useState("");
  const [mlSetSuperclassFilter, setMlSetSuperclassFilter] = useState("");
  const [mlSetMembershipStatusFilter, setMlSetMembershipStatusFilter] = useState<MlSetMembershipStatusFilter>("all");
  const [mlSetSplitFilter, setMlSetSplitFilter] = useState("");
  const [mlSetSessionFilter, setMlSetSessionFilter] = useState("");
  const [selectedObjectGroupId, setSelectedObjectGroupId] = useState("");
  const [expandedObjectGroupIds, setExpandedObjectGroupIds] = useState<Set<string>>(new Set());
  const [selectedSplit, setSelectedSplit] = useState<"all" | SplitBucket>("all");
  const [fromDate, setFromDate] = useState("");
  const [toDate, setToDate] = useState("");
  const [datePreset, setDatePreset] = useState<DatePreset>("custom");
  const [governanceFlags, setGovernanceFlags] = useState<Set<GovernanceFilter>>(new Set());
  const [expectedClassFilter, setExpectedClassFilter] = useState("");
  const [tagFilter, setTagFilter] = useState("");
  const [calibrationOnly, setCalibrationOnly] = useState(false);
  const [detail, setDetail] = useState<TakeDetail | null>(null);
  const [selectedTakeIds, setSelectedTakeIds] = useState<Set<string>>(new Set());
  const [selectionScope, setSelectionScope] = useState<"visible" | "filtered">("visible");
  const [excludedTakeIds, setExcludedTakeIds] = useState<Set<string>>(new Set());
  const [validationFilter, setValidationFilter] = useState("all");
  const [search, setSearch] = useState("");
  const [activeTab, setActiveTab] = useState<DatasetTab>("overview");
  const [busy, setBusy] = useState(false);
  const [isInitialLoading, setIsInitialLoading] = useState(false);
  const [isLoadingMore, setIsLoadingMore] = useState(false);
  const [serverFilteredCount, setServerFilteredCount] = useState<number | null>(null);
  const [serverTotalCount, setServerTotalCount] = useState<number | null>(null);
  const [pagedSummaryCounts, setPagedSummaryCounts] = useState<PagedSummaryCounts | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [savedCollections, setSavedCollections] = useState<SavedCollection[]>([]);
  const [newCollectionName, setNewCollectionName] = useState("");
  const [showSessionTree, setShowSessionTree] = useState(true);
  const [compareOpen, setCompareOpen] = useState(false);
  const [bulkModal, setBulkModal] = useState<BulkModalKind | null>(null);
  const [modalDatasetId, setModalDatasetId] = useState("");
  const [modalSessionId, setModalSessionId] = useState("");
  const [modalMlSetId, setModalMlSetId] = useState("");
  const [modalMlSetMode, setModalMlSetMode] = useState<"add" | "remove">("add");
  const [modalCreateMlSet, setModalCreateMlSet] = useState(false);
  const [modalNewMlSetName, setModalNewMlSetName] = useState("");
  const [modalNewMlSetDescription, setModalNewMlSetDescription] = useState("");
  const [modalNewMlSetSplitStrategy, setModalNewMlSetSplitStrategy] = useState("");
  const [modalNewMlSetValidationReq, setModalNewMlSetValidationReq] = useState("");
  const [modalSplit, setModalSplit] = useState<SplitBucket>("training");
  const [modalTagMode, setModalTagMode] = useState<"add" | "remove" | "replace">("add");
  const [modalTagValue, setModalTagValue] = useState("");
  const [modalValidationStatus, setModalValidationStatus] = useState("valid");
  const [modalExpectedClassValue, setModalExpectedClassValue] = useState("");
  const [modalCreateSession, setModalCreateSession] = useState(false);
  const [modalNewSessionName, setModalNewSessionName] = useState("");
  const [modalSessionNotes, setModalSessionNotes] = useState("");
  const [modalAffectedCount, setModalAffectedCount] = useState<number | null>(null);
  const [modalCounting, setModalCounting] = useState(false);
  const [ingestionWizardOpen, setIngestionWizardOpen] = useState(false);
  const [datasetsProfilerEnabled, setDatasetsProfilerEnabled] = useState(readDatasetsProfilerEnabled);
  const [datasetsProfiler, setDatasetsProfiler] = useState<DatasetsPageProfilerSnapshot | null>(null);

  useEffect(() => {
    void api.datasets().then((items) => {
      setDatasets(items);
      if (!selectedDataset && items[0]?.id) setSelectedDataset(items[0].id);
    }).catch((err: unknown) => setError(err instanceof Error ? err.message : "Failed to load datasets"));
  }, [selectedDataset]);

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(DATASETS_COLLECTIONS_KEY);
      if (!raw) return;
      const parsed = JSON.parse(raw) as SavedCollection[];
      setSavedCollections(Array.isArray(parsed) ? parsed : []);
    } catch {
      setSavedCollections([]);
    }
  }, []);

  useEffect(() => {
    window.localStorage.setItem(DATASETS_COLLECTIONS_KEY, JSON.stringify(savedCollections));
  }, [savedCollections]);

  useEffect(() => {
    if (!selectedDataset) {
      setSessions([]);
      setSelectedSession("");
      setSessionDrawerId("");
      setSelectedSessionDetailSummary(null);
      return;
    }
    void api.datasetSessions(selectedDataset).then((items) => {
      setSessions(items);
      if (selectedSession && !items.find((s) => s.id === selectedSession)) {
        setSelectedSession("");
        setSessionDrawerId("");
        setSelectedSessionDetailSummary(null);
      }
    }).catch((err: unknown) => setError(err instanceof Error ? err.message : "Failed to load sessions"));
  }, [selectedDataset, selectedSession]);

  useEffect(() => {
    if (!selectedDataset) {
      setDatasetLabelSummary(null);
      return;
    }
    // Class Distribution and the tag cloud used to be computed client-side
    // over `takes`, which only ever holds one page: a 729-take dataset showed
    // the tags of whichever 50 takes happened to be loaded. This is the same
    // dataset-wide aggregate /api/datasets/{id}/label-summary already
    // computed for other consumers.
    void api.datasetLabelSummary(selectedDataset)
      .then((summary) => setDatasetLabelSummary(summary))
      .catch(() => setDatasetLabelSummary(null));
  }, [selectedDataset]);

  useEffect(() => {
    if (!selectedSession) {
      setSelectedSessionDetailSummary(null);
      return;
    }
    const sessionId = selectedSession;
    let cancelled = false;
    void api.datasetSessionSummary(sessionId, selectedDataset || undefined)
      .then((payload) => {
        if (cancelled) return;
        setSelectedSessionDetailSummary(payload.summary ?? null);
      })
      .catch(() => {
        if (!cancelled) setSelectedSessionDetailSummary(null);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedSession, selectedDataset]);

  useEffect(() => {
    if (!selectedDataset) {
      setMlSets([]);
      return;
    }
    void api.datasetMlSets(selectedDataset)
      .then((items) => setMlSets(items))
      .catch(() => setMlSets([]));
  }, [selectedDataset]);

  useEffect(() => {
    if (!activeMlSetScope?.mlSetId || !selectedDataset) {
      setMlSetScopeSummary(null);
      return;
    }
    void api.mlSetSummary(activeMlSetScope.mlSetId, selectedDataset)
      .then((payload) => setMlSetScopeSummary(payload))
      .catch(() => setMlSetScopeSummary(null));
  }, [activeMlSetScope?.mlSetId, selectedDataset]);

  useEffect(() => {
    if (!activeMlSetScope?.mlSetId || !selectedDataset) {
      setMlSetReviewFacets(null);
      return;
    }
    let cancelled = false;
    void api.mlSetReviewFacets(selectedDataset, activeMlSetScope.mlSetId, {
      physical_object_id: mlSetObjectIdFilter.trim() || activeMlSetScope.physicalObjectId || undefined,
      raw_label: mlSetRawLabelFilter.trim() || undefined,
      normalized_class: mlSetNormalizedClassFilter.trim() || activeMlSetScope.normalizedClass || undefined,
      superclass: mlSetSuperclassFilter.trim() || activeMlSetScope.superclass || undefined,
      membership_status: mlSetMembershipStatusFilter === "all" ? undefined : mlSetMembershipStatusFilter,
      split: mlSetSplitFilter.trim() || undefined,
      session_id: mlSetSessionFilter.trim() || undefined,
      search: search.trim() || undefined,
      validation_status: validationFilter === "all" ? undefined : validationFilter,
    })
      .then((payload) => {
        if (!cancelled) setMlSetReviewFacets(payload);
      })
      .catch(() => {
        if (!cancelled) setMlSetReviewFacets(null);
      });
    return () => {
      cancelled = true;
    };
  }, [
    activeMlSetScope?.mlSetId,
    activeMlSetScope?.normalizedClass,
    activeMlSetScope?.superclass,
    activeMlSetScope?.physicalObjectId,
    selectedDataset,
    mlSetObjectIdFilter,
    mlSetRawLabelFilter,
    mlSetNormalizedClassFilter,
    mlSetSuperclassFilter,
    mlSetMembershipStatusFilter,
    mlSetSplitFilter,
    mlSetSessionFilter,
    search,
    validationFilter,
  ]);

  async function refreshMlSetsForDataset(datasetId: string) {
    const items = await api.datasetMlSets(datasetId);
    setMlSets(items);
    return items;
  }

  async function handleWizardMaterialized(result: MaterializeMlIngestionResponse) {
    try {
      setError(null);
      setSuccessMessage(`Created ML set ${result.ml_set_id} with ${result.membership_count} memberships / ${result.physical_object_count} physical objects.`);
      setActiveTab("ml_sets");
      const items = await refreshMlSetsForDataset(result.dataset_id);
      const created = items.find((item) => item.id === result.ml_set_id);
      if (created) {
        setSelectedMlSetId(created.id);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to refresh ML sets after materialization");
    }
  }

  useEffect(() => {
    if (!selectedDataset) {
      setPhysicalObjects([]);
      return;
    }
    void api.physicalObjects({ dataset_id: selectedDataset })
      .then((items) => setPhysicalObjects(items))
      .catch(() => setPhysicalObjects([]));
  }, [selectedDataset]);

  useEffect(() => {
    setSelectedTakeIds(new Set());
    setExcludedTakeIds(new Set());
    setSelectionScope("visible");
    setSelectedTakeId("");
    setSelectedObjectId("");
    setSelectedPhysicalObjectId("");
    setSelectedObjectGroupId("");
    setDetail(null);
  }, [
    selectedDataset,
    selectedSession,
    validationFilter,
    search,
    tagFilter,
    fromDate,
    toDate,
    datePreset,
    selectedSplit,
    expectedClassFilter,
    calibrationOnly,
    governanceFlags,
    activeMlSetScope?.mlSetId,
    mlSetObjectIdFilter,
    mlSetRawLabelFilter,
    mlSetNormalizedClassFilter,
    mlSetSuperclassFilter,
    mlSetMembershipStatusFilter,
    mlSetSplitFilter,
    mlSetSessionFilter,
  ]);

  useEffect(() => {
    if (datePreset === "custom" || datePreset === "this_shift") return;
    const now = new Date();
    const toIso = now.toISOString().slice(0, 10);
    let from = new Date(now);
    if (datePreset === "today") {
      from = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    }
    if (datePreset === "last_24h") {
      from = new Date(now.getTime() - 24 * 60 * 60 * 1000);
    }
    if (datePreset === "last_7d") {
      from = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
    }
    if (datePreset === "last_30d") {
      from = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
    }
    if (datePreset === "this_week") {
      const day = now.getDay();
      const diff = day === 0 ? 6 : day - 1;
      from = new Date(now.getFullYear(), now.getMonth(), now.getDate() - diff);
    }
    setFromDate(from.toISOString().slice(0, 10));
    setToDate(toIso);
  }, [datePreset]);

  useEffect(() => {
    if (!selectedTakeId) return;
    void api.take(selectedTakeId).then((result) => {
      setDetail(result);
      const first = firstObjectId(result);
      if (first && !selectedObjectId) setSelectedObjectId(first);
    }).catch((err: unknown) => setError(err instanceof Error ? err.message : "Failed to load take detail"));
  }, [selectedTakeId, selectedObjectId]);

  const selectedDatasetSummary = useMemo(() => datasets.find((dataset) => dataset.id === selectedDataset) ?? null, [datasets, selectedDataset]);
  const selectedSessionSummary = useMemo(() => sessions.find((session) => session.id === selectedSession) ?? null, [sessions, selectedSession]);
  const drawerSessionSummary = useMemo(() => sessions.find((session) => session.id === sessionDrawerId) ?? null, [sessions, sessionDrawerId]);
  const selectedPhysicalObjectSummary = useMemo(() => physicalObjects.find((item) => item.physical_object_id === selectedPhysicalObjectId) ?? null, [physicalObjects, selectedPhysicalObjectId]);
  const selectedMlSetSummary = useMemo(() => mlSets.find((item) => item.id === selectedMlSetId) ?? null, [mlSets, selectedMlSetId]);
  const activeScopedMlSet = useMemo(() => mlSets.find((item) => item.id === activeMlSetScope?.mlSetId) ?? null, [mlSets, activeMlSetScope?.mlSetId]);
  const mlSetSessionFacetCounts = useMemo(
    () => new Map((mlSetReviewFacets?.sessions ?? []).map((item) => [item.value, item.count])),
    [mlSetReviewFacets?.sessions]
  );
  const mlSetSessionOptions = useMemo(
    () => {
      const known = new Set<string>();
      const options = sessions.map((session) => {
        known.add(session.id);
        return {
          value: session.id,
          label: formatFacetCountLabel(session.name, mlSetSessionFacetCounts.get(session.id)),
        };
      });
      (mlSetReviewFacets?.sessions ?? []).forEach((item) => {
        if (!item.value || known.has(item.value)) return;
        options.push({
          value: item.value,
          label: formatFacetCountLabel(item.name || item.value, item.count),
        });
      });
      return options;
    },
    [sessions, mlSetSessionFacetCounts, mlSetReviewFacets?.sessions]
  );
  const mlSetObjectOptions = useMemo(
    () => (mlSetReviewFacets?.object_ids ?? []).map((item) => ({
      value: item.value,
      label: formatObjectFacetLabel(item),
    })),
    [mlSetReviewFacets?.object_ids]
  );
  const mlSetRawLabelOptions = useMemo(
    () => (mlSetReviewFacets?.raw_labels ?? []).map((item) => ({ value: item.value, label: formatFacetCountLabel(item.value, item.count) })),
    [mlSetReviewFacets?.raw_labels]
  );
  const mlSetNormalizedClassOptions = useMemo(
    () => (mlSetReviewFacets?.normalized_classes ?? []).map((item) => ({ value: item.value, label: formatFacetCountLabel(item.value, item.count) })),
    [mlSetReviewFacets?.normalized_classes]
  );
  const mlSetSuperclassOptions = useMemo(
    () => (mlSetReviewFacets?.superclasses ?? []).map((item) => ({ value: item.value, label: formatFacetCountLabel(item.value, item.count) })),
    [mlSetReviewFacets?.superclasses]
  );
  const mlSetSplitCounts = useMemo(
    () => new Map((mlSetReviewFacets?.splits ?? []).map((item) => [item.value, item.count])),
    [mlSetReviewFacets?.splits]
  );
  const mlSetSplitOptions = useMemo(
    () => ["train", "validation", "test", "holdout", "calibration", "unassigned"].map((value) => ({
      value,
      label: formatFacetCountLabel(value, mlSetSplitCounts.get(value)),
    })),
    [mlSetSplitCounts]
  );
  const mlSetMembershipCounts = useMemo(
    () => new Map((mlSetReviewFacets?.membership_statuses ?? []).map((item) => [item.value, item.count])),
    [mlSetReviewFacets?.membership_statuses]
  );
  const modalSessions = useMemo(
    () => (modalDatasetId === selectedDataset ? sessions : modalDatasetSessions),
    [modalDatasetId, selectedDataset, sessions, modalDatasetSessions]
  );

  const takes = useMemo(
    () => pagedTakes.filter((take) => {
      if (fromDate || toDate) {
        const created = take.created_at ? new Date(take.created_at) : null;
        if (!created) return false;
        if (fromDate) {
          const start = new Date(`${fromDate}T00:00:00`);
          if (created < start) return false;
        }
        if (toDate) {
          const end = new Date(`${toDate}T23:59:59`);
          if (created > end) return false;
        }
      }
      const split = readSplitFromTake(take);
      for (const flag of governanceFlags) {
        if (flag === "missing_expected_class" && take.expected_class) return false;
        if (flag === "missing_split" && split) return false;
        if (flag === "missing_tags" && (take.tags ?? []).length > 0) return false;
        if (flag === "missing_calibration" && take.calibration_id) return false;
        if (flag === "processing_failed" && String(take.latest_run_status ?? "") !== "failed") return false;
        if (flag === "processing_incomplete" && (take.has_done || String(take.latest_run_status ?? "") === "completed")) return false;
        if (flag === "no_objects_detected" && Number(take.object_count ?? 0) > 0) return false;
      }
      return true;
    }),
    [pagedTakes, fromDate, toDate, governanceFlags]
  );
  const activeReviewRows = useMemo(() => (activeMlSetScope ? mlSetMemberRows : []), [activeMlSetScope, mlSetMemberRows]);
  const objectGroupRows = useMemo<MlSetObjectGroup[]>(() => {
    const groups = new Map<string, MlSetObjectGroup>();
    activeReviewRows.forEach((row) => {
      const key = String(row.physical_object_id || `unassigned:${row.take_id}`);
      const current = groups.get(key) ?? {
        id: key,
        physicalObjectId: String(row.physical_object_id || ""),
        normalizedClass: row.normalized_class || "-",
        superclass: row.superclass || "-",
        rawLabels: [],
        splitValues: [],
        sessions: [],
        warningCodes: [],
        takeCount: 0,
        members: [],
      };
      current.takeCount += 1;
      current.members.push(row);
      if (row.raw_label && !current.rawLabels.includes(row.raw_label)) current.rawLabels.push(row.raw_label);
      if (row.split && !current.splitValues.includes(row.split)) current.splitValues.push(row.split);
      const sessionLabel = row.session_name || row.session_id || "Unassigned";
      if (!current.sessions.includes(sessionLabel)) current.sessions.push(sessionLabel);
      row.warnings.forEach((warning) => {
        if (!current.warningCodes.includes(warning)) current.warningCodes.push(warning);
      });
      if (current.normalizedClass === "-" && row.normalized_class) current.normalizedClass = row.normalized_class;
      if (current.superclass === "-" && row.superclass) current.superclass = row.superclass;
      groups.set(key, current);
    });
    return [...groups.values()].sort((a, b) => b.takeCount - a.takeCount || a.id.localeCompare(b.id));
  }, [activeReviewRows]);
  const hasClientOnlyFilters = useMemo(
    () => governanceFlags.size > 0,
    [governanceFlags]
  );
  const filteredCount = hasClientOnlyFilters
    ? takes.length
    : (serverFilteredCount ?? serverTotalCount ?? takes.length);
  const datasetTotalCount = selectedDatasetSummary?.take_count ?? serverTotalCount ?? filteredCount;
  const hasBaseFilters = Boolean(
    selectedSession ||
    validationFilter !== "all" ||
    search.trim() ||
    tagFilter.trim() ||
    fromDate ||
    toDate ||
    datePreset !== "custom" ||
    hasClientOnlyFilters
  );
  const visibleTakeIds = useMemo(
    () => (activeMlSetScope ? activeReviewRows.map((row) => row.take_id) : takes.map((take) => take.take_id)),
    [activeMlSetScope, activeReviewRows, takes]
  );
  const activeFilteredCount = activeMlSetScope ? (mlSetMemberFilteredCount ?? activeReviewRows.length) : filteredCount;
  const selectedCount = selectionScope === "visible" ? selectedTakeIds.size : Math.max(0, activeFilteredCount - excludedTakeIds.size);
  const selectedSessionTakeCount = selectedSessionDetailSummary?.total_takes ?? selectedSessionSummary?.take_count ?? 0;
  const selectedSessionReviewedPct = selectedSessionTakeCount ? Math.round(((selectedSessionDetailSummary?.reviewed ?? 0) / selectedSessionTakeCount) * 100) : 0;
  const selectedSessionValidatedPct = selectedSessionTakeCount
    ? Math.round((Math.max(0, selectedSessionTakeCount - (selectedSessionDetailSummary?.unreviewed ?? 0)) / selectedSessionTakeCount) * 100)
    : 0;
  const selectedSessionCalibrationStatus = selectedSessionDetailSummary?.calibration_assignment && selectedSessionDetailSummary.calibration_assignment !== "-"
    ? "linked"
    : selectedSessionSummary?.calibration_id
      ? "linked"
      : "unlinked";

  useEffect(() => {
    if (selectedTakeId && !takes.find((t) => t.take_id === selectedTakeId) && !activeReviewRows.find((row) => row.take_id === selectedTakeId)) {
      setSelectedTakeId("");
      setDetail(null);
      setSelectedObjectId("");
    }
  }, [selectedTakeId, takes, activeReviewRows]);

  useEffect(() => {
    if (sessionDrawerId && sessionDrawerId !== selectedSession) {
      setSessionDrawerId("");
    }
  }, [selectedSession, sessionDrawerId]);

  useEffect(() => {
    const visible = new Set(visibleTakeIds);
    if (selectionScope === "visible") {
      setSelectedTakeIds((prev) => {
        const next = new Set<string>();
        prev.forEach((id) => {
          if (visible.has(id)) next.add(id);
        });
        return next;
      });
    } else {
      setExcludedTakeIds((prev) => {
        const next = new Set<string>();
        prev.forEach((id) => {
          if (visible.has(id)) next.add(id);
        });
        return next;
      });
    }
  }, [visibleTakeIds, selectionScope]);

  const stats = useMemo(() => {
    const next = {
      total: takes.length,
      validated: 0,
      unreviewed: 0,
      rejected: 0,
      missingLabels: 0,
      processingReady: 0,
      splitCoverage: 0,
      splits: Object.fromEntries(splitBuckets.map((bucket) => [bucket, 0])) as Record<SplitBucket, number>,
      sessionCounts: new Map<string, number>(),
      recent: [] as TakeSummary[],
    };
    takes.forEach((take) => {
      const status = String(take.validation_status ?? "unreviewed");
      if (status === "valid") next.validated += 1;
      else if (status === "invalid") next.rejected += 1;
      else next.unreviewed += 1;
      if (!take.expected_class && !(take.semantic_labels?.length ?? 0)) next.missingLabels += 1;
      if (take.has_done || take.has_ready) next.processingReady += 1;
      const split = readSplitFromTake(take);
      if (split) {
        next.splits[split] += 1;
        next.splitCoverage += 1;
      }
      const sessionKey = take.experiment_session_name || take.experiment_session_id || "unassigned";
      next.sessionCounts.set(sessionKey, (next.sessionCounts.get(sessionKey) ?? 0) + 1);
    });
    next.recent = [...takes].sort((a, b) => String(b.created_at ?? "").localeCompare(String(a.created_at ?? ""))).slice(0, 6);
    return next;
  }, [takes]);

  // From /api/datasets/{id}/label-summary, over the whole dataset — not from
  // `stats`, which only ever saw the loaded page.
  const sortedClassCounts = useMemo(
    () => Object.entries(datasetLabelSummary?.class_counts ?? {}).slice(0, 8),
    [datasetLabelSummary],
  );
  // From the sessions endpoint rather than from `stats`, which only sees the
  // loaded page: the distribution reported the page size for whichever session
  // the page happened to belong to, and nothing for the others.
  const sortedSessionCounts = useMemo(
    () =>
      sessions
        .map((session) => [session.name || session.id, session.take_count ?? 0] as [string, number])
        .filter(([, count]) => count > 0)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 8),
    [sessions],
  );
  const sortedTags = useMemo(
    () => Object.entries(datasetLabelSummary?.raw_tag_counts ?? {}).slice(0, 12),
    [datasetLabelSummary],
  );
  const validatedCount = Number(pagedSummaryCounts?.validation?.validated ?? stats.validated);
  const unreviewedCount = Number(pagedSummaryCounts?.validation?.unreviewed ?? stats.unreviewed);
  const rejectedCount = Number(pagedSummaryCounts?.validation?.rejected ?? stats.rejected);
  const needsReviewCount = countFromValidation(pagedSummaryCounts, "needs_review");
  const goldenSampleCount = countFromValidation(pagedSummaryCounts, "golden_sample");
  const benchmarkApprovedCount = countFromValidation(pagedSummaryCounts, "benchmark_approved");
  const missingLabelsCount = Number(pagedSummaryCounts?.missing_labels ?? stats.missingLabels);
  const missingSplitCount = Number(pagedSummaryCounts?.missing_split ?? Math.max(0, stats.total - stats.splitCoverage));
  const missingCalibrationCount = Number(pagedSummaryCounts?.missing_calibration ?? takes.filter((take) => !take.calibration_id).length);
  const processingIncompleteCount = Number(pagedSummaryCounts?.processing_incomplete ?? Math.max(0, stats.total - stats.processingReady));
  const processingFailedCount = Number(pagedSummaryCounts?.processing_failed ?? 0);
  const noObjectsDetectedCount = Number(pagedSummaryCounts?.no_objects_detected ?? 0);
  const activeReviewCounts = useMemo(() => {
    if (!activeMlSetScope) {
      return {
        unreviewed: unreviewedCount,
        needsReview: needsReviewCount,
        validated: validatedCount,
        missingLabels: missingLabelsCount,
        reviewRequiredMembers: 0,
        missingSplitMembers: 0,
      };
    }
    return activeReviewRows.reduce((acc, row) => {
      const validation = String(row.validation_status || "unreviewed");
      if (validation === "unreviewed") acc.unreviewed += 1;
      if (validation === "valid") acc.validated += 1;
      if (validation === "needs_review") acc.needsReview += 1;
      if (!row.normalized_class) acc.missingLabels += 1;
      if (row.review_required) acc.reviewRequiredMembers += 1;
      if (row.trainable && row.split === "unassigned") acc.missingSplitMembers += 1;
      return acc;
    }, { unreviewed: 0, needsReview: 0, validated: 0, missingLabels: 0, reviewRequiredMembers: 0, missingSplitMembers: 0 });
  }, [activeMlSetScope, activeReviewRows, missingLabelsCount, needsReviewCount, unreviewedCount, validatedCount]);
  const scopeContextCounts = useMemo(() => ({
    members: mlSetScopeSummary?.identity.take_count ?? Number(activeScopedMlSet?.member_count ?? activeScopedMlSet?.membership_count ?? activeReviewRows.length),
    objects: mlSetScopeSummary?.identity.physical_object_count ?? Number(activeScopedMlSet?.physical_object_count ?? 0),
    reviewRequired: mlSetScopeSummary?.identity.review_required_count ?? Number(activeScopedMlSet?.review_required_count ?? 0),
  }), [activeScopedMlSet?.member_count, activeScopedMlSet?.membership_count, activeScopedMlSet?.physical_object_count, activeScopedMlSet?.review_required_count, activeReviewRows.length, mlSetScopeSummary]);
  const healthScore = useMemo(() => {
    const base = hasBaseFilters ? filteredCount : datasetTotalCount;
    if (!base) return 0;
    const validationCoverage = validatedCount / base;
    const splitCoverage = (base - missingSplitCount) / base;
    const processingCoverage = (base - processingIncompleteCount) / base;
    const calibrationCoverage = (base - missingCalibrationCount) / base;
    return Math.round(((validationCoverage + splitCoverage + processingCoverage + calibrationCoverage) / 4) * 100);
  }, [hasBaseFilters, filteredCount, datasetTotalCount, validatedCount, missingSplitCount, processingIncompleteCount, missingCalibrationCount]);

  const objectRows = useMemo(() => {
    const annotations = (detail?.object_annotations ?? []) as ObjectAnnotation[];
    return annotations.map((annotation, idx) => {
      const objectId = String(annotation.candidate_id ?? annotation.id ?? idx);
      const labels = Array.isArray(annotation.labels) ? annotation.labels.map(String).join(", ") : "-";
      return {
        objectId,
        labels,
        expectedClass: String(annotation.expected_class ?? "-"),
        validation: String(annotation.validation_status ?? "unreviewed"),
        notes: String(annotation.notes ?? ""),
        confidence: Number(annotation.confidence ?? Number.NaN),
      };
    });
  }, [detail?.object_annotations]);

  const selectedObject = useMemo(() => objectRows.find((row) => row.objectId === selectedObjectId) ?? null, [objectRows, selectedObjectId]);
  const selectedTakeSummary = useMemo(() => {
    const direct = takes.find((take) => take.take_id === selectedTakeId);
    if (direct) return direct;
    const scoped = activeReviewRows.find((row) => row.take_id === selectedTakeId);
    if (!scoped) return null;
    return {
      take_id: scoped.take_id,
      status: "processed",
      decision: null,
      friendly_name: scoped.take_name || scoped.take_id,
      thumbnail_path: scoped.thumbnail_path || null,
      validation_status: scoped.validation_status || "unreviewed",
      expected_class: scoped.normalized_class || null,
      physical_object_id: scoped.physical_object_id || null,
      experiment_session_id: scoped.session_id || null,
      experiment_session_name: scoped.session_name || null,
      processed_class_label: scoped.processed_class_label || null,
      processed_superclass: scoped.processed_superclass || null,
      has_done: scoped.processing_ready,
      has_ready: scoped.processing_ready,
      created_at: scoped.created_at || null,
      tags: [],
    } as unknown as TakeSummary;
  }, [takes, selectedTakeId, activeReviewRows]);
  const takePreviewsById = useMemo(() => {
    const next = new Map<string, { url: string | null; placeholder: string; modality: string | null }>();
    takes.forEach((take) => {
      next.set(take.take_id, takePreviewForSummary(take));
    });
    activeReviewRows.forEach((row) => {
      if (next.has(row.take_id)) return;
      next.set(row.take_id, {
        url: row.thumbnail_path ? fileUrl(row.take_id, row.thumbnail_path) : null,
        placeholder: row.physical_object_id ? row.physical_object_id.slice(0, 4).toUpperCase() : "TAKE",
        modality: null,
      });
    });
    return next;
  }, [takes, activeReviewRows]);
  const selectedTakePreview = useMemo(
    () => (selectedTakeSummary ? (takePreviewsById.get(selectedTakeSummary.take_id) ?? { url: null, placeholder: "-", modality: null }) : { url: null, placeholder: "-", modality: null }),
    [selectedTakeSummary, takePreviewsById]
  );
  const selectedInspectorPreview = useMemo(() => {
    if (detail?.take_id && detail.take_id === selectedTakeId) {
      const detailPreviewUrl = takePreviewUrlForDetail(detail);
      if (detailPreviewUrl) {
        return {
          url: detailPreviewUrl,
          placeholder: "IMG",
          modality: selectedTakePreview.modality,
        };
      }
    }
    return selectedTakePreview;
  }, [detail, selectedTakeId, selectedTakePreview]);
  const selectedScopedMember = useMemo(() => activeReviewRows.find((row) => row.take_id === selectedTakeId) ?? null, [activeReviewRows, selectedTakeId]);
  const selectedObjectGroup = useMemo(() => objectGroupRows.find((row) => row.id === selectedObjectGroupId) ?? null, [objectGroupRows, selectedObjectGroupId]);
  const detailTakeMetadata = useMemo(() => ((detail?.take_metadata ?? {}) as Record<string, unknown>), [detail?.take_metadata]);
  const detailTakeSuperclasses = useMemo(
    () => (Array.isArray(detailTakeMetadata["superclass_labels"]) ? detailTakeMetadata["superclass_labels"].map(String) : []),
    [detailTakeMetadata]
  );
  const keyboardCommands = useMemo(
    () => ({ j: "next_take", k: "prev_take", x: "toggle_select", v: "validate", r: "reject", t: "tag", s: "split" }),
    []
  );

  function selectSession(sessionId: string, options?: { openDrawer?: boolean }) {
    setSelectedSession(sessionId);
    setSessionDrawerId(options?.openDrawer ? sessionId : "");
  }

  function openSessionDrawer(sessionId: string) {
    selectSession(sessionId, { openDrawer: true });
  }

  function applyMlSetScope(mlSetId: string, options?: Partial<MlSetScopeState>) {
    setActiveMlSetScope({
      mlSetId,
      normalizedClass: options?.normalizedClass ?? "",
      superclass: options?.superclass ?? "",
      physicalObjectId: options?.physicalObjectId ?? "",
    });
    setMlSetNormalizedClassFilter("");
    setMlSetSuperclassFilter("");
    setMlSetObjectIdFilter("");
    setMlSetRawLabelFilter("");
    setMlSetMembershipStatusFilter("all");
    setMlSetSplitFilter("");
    setMlSetSessionFilter("");
    setSelectedSplit("all");
    setActiveTab("takes");
    setMlSetScopedView("takes");
  }

  function clearMlSetScope() {
    setActiveMlSetScope(null);
    setMlSetMemberRows([]);
    setMlSetMemberFilteredCount(null);
    setMlSetScopeSummary(null);
    setMlSetReviewFacets(null);
    setMlSetNormalizedClassFilter("");
    setMlSetSuperclassFilter("");
    setMlSetObjectIdFilter("");
    setMlSetRawLabelFilter("");
    setMlSetMembershipStatusFilter("all");
    setMlSetSplitFilter("");
    setMlSetSessionFilter("");
    setSelectedObjectGroupId("");
    setExpandedObjectGroupIds(new Set());
  }

  function buildFeatureAnalyticsUrl(extra?: { mlSetId?: string; normalizedClass?: string; superclass?: string; physicalObjectId?: string; rawLabel?: string }) {
    const params = new URLSearchParams();
    if (selectedDataset) params.set("dataset_id", selectedDataset);
    const resolvedMlSetId = extra?.mlSetId ?? activeMlSetScope?.mlSetId ?? "";
    if (resolvedMlSetId) params.set("ml_set_id", resolvedMlSetId);
    const normalizedClass = extra?.normalizedClass ?? (mlSetNormalizedClassFilter.trim() || activeMlSetScope?.normalizedClass || "");
    const superclass = extra?.superclass ?? (mlSetSuperclassFilter.trim() || activeMlSetScope?.superclass || "");
    const physicalObjectId = extra?.physicalObjectId ?? (mlSetObjectIdFilter.trim() || activeMlSetScope?.physicalObjectId || "");
    const rawLabel = extra?.rawLabel ?? mlSetRawLabelFilter.trim();
    const sessionId = mlSetSessionFilter.trim() || selectedSession || "";
    const splitValue = activeMlSetScope ? mlSetSplitFilter.trim() : (selectedSplit === "all" ? "" : selectedSplit);
    if (rawLabel) params.append("raw_labels", rawLabel);
    if (normalizedClass) params.append("normalized_classes", normalizedClass);
    if (superclass) params.append("superclasses", superclass);
    if (physicalObjectId) params.append("physical_object_ids", physicalObjectId);
    if (sessionId) params.set("session_id", sessionId);
    if (splitValue) params.append("split", splitValue);
    return `/feature-analytics?${params.toString()}`;
  }

  function openFeatureAnalyticsForSlice(extra?: { mlSetId?: string; normalizedClass?: string; superclass?: string; physicalObjectId?: string; rawLabel?: string }) {
    window.location.href = buildFeatureAnalyticsUrl(extra);
  }

  function openFeatureAnalyticsForSliceInNewTab(extra?: { mlSetId?: string; normalizedClass?: string; superclass?: string; physicalObjectId?: string; rawLabel?: string }) {
    window.open(buildFeatureAnalyticsUrl(extra), "_blank", "noopener,noreferrer");
  }

  function buildStudioTakeHref(takeId: string) {
    const sessionId = selectedScopedMember?.session_id || selectedTakeSummary?.experiment_session_id || selectedSession || null;
    const datasetId = selectedTakeSummary?.dataset_id || selectedDataset || null;
    const physicalObjectId = selectedScopedMember?.physical_object_id || selectedTakeSummary?.physical_object_id || null;
    return buildStudioDeepLink({
      take_id: takeId,
      dataset_id: datasetId,
      session_id: sessionId,
      physical_object_id: physicalObjectId,
    });
  }

  useEffect(() => {
    if (!KEYBOARD_CURATION_ENABLED || activeTab !== "takes") return;
    const onKeyDown = (event: KeyboardEvent) => {
      if ((event.target as HTMLElement | null)?.tagName === "INPUT") return;
      const key = event.key.toLowerCase();
      if (key in keyboardCommands && DEBUG_DATASETS_TAKES) {
        console.debug("[datasets:keyboard]", keyboardCommands[key as keyof typeof keyboardCommands]);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [activeTab, keyboardCommands]);

  async function loadTakePage(nextOffset: number, append: boolean) {
    if (append) setIsLoadingMore(true);
    else setIsInitialLoading(true);
    if (activeMlSetScope?.mlSetId) {
      try {
        const page = await api.mlSetMembers(activeMlSetScope.mlSetId, {
          dataset_id: selectedDataset || undefined,
          physical_object_id: mlSetObjectIdFilter.trim() || activeMlSetScope.physicalObjectId || undefined,
          raw_label: mlSetRawLabelFilter.trim() || undefined,
          normalized_class: mlSetNormalizedClassFilter.trim() || activeMlSetScope.normalizedClass || undefined,
          superclass: mlSetSuperclassFilter.trim() || activeMlSetScope.superclass || undefined,
          membership_status: mlSetMembershipStatusFilter === "all" ? undefined : mlSetMembershipStatusFilter,
          split: mlSetSplitFilter.trim() || undefined,
          session_id: mlSetSessionFilter.trim() || undefined,
          search: search.trim() || undefined,
          validation_status: validationFilter === "all" ? undefined : validationFilter,
          limit: TAKES_PAGE_LIMIT,
          offset: nextOffset,
        });
        const nextItems = append ? [...mlSetMemberRows, ...page.items] : page.items;
        setMlSetMemberRows(nextItems);
        setMlSetMemberOffset(page.next_offset ?? (page.offset + page.items.length));
        setMlSetMemberHasMore(Boolean(page.has_more));
        setMlSetMemberFilteredCount(page.filtered_count ?? nextItems.length);
        setPagedTakes([]);
        setServerFilteredCount(page.filtered_count ?? null);
        setServerTotalCount(page.total_count ?? null);
        setPagedSummaryCounts(null);
      } finally {
        setIsInitialLoading(false);
        setIsLoadingMore(false);
      }
      return;
    }
    const params = {
      dataset_id: selectedDataset || undefined,
      session_id: selectedSession || undefined,
      validation_status: validationFilter === "all" ? undefined : validationFilter,
      search: search.trim() || undefined,
      tag: tagFilter.trim() || undefined,
      created_from: fromDate || undefined,
      created_to: toDate || undefined,
      expected_class: expectedClassFilter.trim() || undefined,
      split: selectedSplit === "all" ? undefined : selectedSplit,
      calibration_linkage_only: calibrationOnly ? true : undefined,
      limit: TAKES_PAGE_LIMIT,
      offset: nextOffset,
      show_archived: false,
      profile: datasetsProfilerEnabled,
    };
    if (DEBUG_DATASETS_TAKES) {
      console.debug("[datasets:takes] request", {
        dataset_id: params.dataset_id,
        session_id: params.session_id,
        validation_status: params.validation_status,
        search: params.search,
        tag: params.tag,
        created_from: params.created_from,
        created_to: params.created_to,
        offset: nextOffset,
        limit: TAKES_PAGE_LIMIT,
      });
    }
    try {
      const requestStartedAt = typeof performance !== "undefined" ? performance.now() : Date.now();
      const page = await api.pagedTakes(params);
      const responseReceivedAt = typeof performance !== "undefined" ? performance.now() : Date.now();
      if (DEBUG_DATASETS_TAKES) {
        console.debug("[datasets:takes] response", {
          count: page.items.length,
          has_more: page.has_more,
          next_offset: page.next_offset,
        });
      }
      const nextItems = append ? [...pagedTakes, ...page.items] : page.items;
      setPagedTakes(nextItems);
      setMlSetMemberRows([]);
      setOffset(page.next_offset ?? (page.offset + page.items.length));
      setHasMore(Boolean(page.has_more));
      setServerFilteredCount(page.filtered_count ?? page.total_count ?? null);
      setServerTotalCount(page.total_count ?? null);
      setPagedSummaryCounts({
        validation: page.summary_counts?.validation ?? {},
        missing_labels: Number(page.summary_counts?.missing_labels ?? 0),
        missing_split: Number(page.summary_counts?.missing_split ?? 0),
        missing_calibration: Number(page.summary_counts?.missing_calibration ?? 0),
        processing_failed: Number(page.summary_counts?.processing_failed ?? 0),
        processing_incomplete: Number(page.summary_counts?.processing_incomplete ?? 0),
        no_objects_detected: Number(page.summary_counts?.no_objects_detected ?? 0),
      });
      if (datasetsProfilerEnabled) {
        requestAnimationFrame(() => {
          const committedAt = typeof performance !== "undefined" ? performance.now() : Date.now();
          const snapshot = {
            captured_at: new Date().toISOString(),
            append,
            frontend_request_ms: responseReceivedAt - requestStartedAt,
            frontend_commit_ms: committedAt - requestStartedAt,
            page_item_count: page.items.length,
            total_visible_rows: nextItems.length,
            backend_profile: page.profile ?? null,
          };
          setDatasetsProfiler(snapshot);
          console.debug("[datasets:profile]", snapshot);
        });
      } else {
        setDatasetsProfiler(null);
      }
    } finally {
      setIsInitialLoading(false);
      setIsLoadingMore(false);
    }
  }

  useEffect(() => {
    setServerFilteredCount(null);
    setServerTotalCount(selectedDatasetSummary?.take_count ?? null);
    setPagedSummaryCounts(null);
    void loadTakePage(0, false).catch((err: unknown) => {
      setError(err instanceof Error ? err.message : "Failed to load takes");
      setIsInitialLoading(false);
      setIsLoadingMore(false);
    });
  }, [
    selectedDataset,
    selectedSession,
    validationFilter,
    search,
    tagFilter,
    fromDate,
    toDate,
    selectedSplit,
    expectedClassFilter,
    calibrationOnly,
    governanceFlags,
    selectedDatasetSummary?.take_count,
    datasetsProfilerEnabled,
    activeMlSetScope?.mlSetId,
    activeMlSetScope?.normalizedClass,
    activeMlSetScope?.superclass,
    activeMlSetScope?.physicalObjectId,
    mlSetObjectIdFilter,
    mlSetRawLabelFilter,
    mlSetNormalizedClassFilter,
    mlSetSuperclassFilter,
    mlSetMembershipStatusFilter,
    mlSetSplitFilter,
    mlSetSessionFilter,
  ]);

  async function refreshTakes() {
    await loadTakePage(0, false);
  }

  async function loadMoreTakes() {
    const nextHasMore = activeMlSetScope ? mlSetMemberHasMore : hasMore;
    const nextOffset = activeMlSetScope ? mlSetMemberOffset : offset;
    if (!nextHasMore || isLoadingMore || isInitialLoading) return;
    await loadTakePage(nextOffset, true);
  }

  function buildActiveBulkFilters() {
    return {
      dataset_id: selectedDataset || undefined,
      ml_set_id: activeMlSetScope?.mlSetId || undefined,
      session_id: activeMlSetScope ? (mlSetSessionFilter.trim() || undefined) : (selectedSession || undefined),
      validation_status: validationFilter === "all" ? undefined : validationFilter,
      search: search.trim() || undefined,
      tag: tagFilter.trim() || undefined,
      created_from: fromDate || undefined,
      created_to: toDate || undefined,
      split: activeMlSetScope ? (mlSetSplitFilter.trim() || undefined) : (selectedSplit === "all" ? undefined : selectedSplit),
      expected_class: expectedClassFilter.trim() || undefined,
      calibration_linkage_only: calibrationOnly || undefined,
      physical_object_id: activeMlSetScope ? (mlSetObjectIdFilter.trim() || activeMlSetScope.physicalObjectId || undefined) : undefined,
      raw_label: activeMlSetScope ? (mlSetRawLabelFilter.trim() || undefined) : undefined,
      normalized_class: activeMlSetScope ? (mlSetNormalizedClassFilter.trim() || activeMlSetScope.normalizedClass || undefined) : undefined,
      superclass: activeMlSetScope ? (mlSetSuperclassFilter.trim() || activeMlSetScope.superclass || undefined) : undefined,
      membership_status: activeMlSetScope && mlSetMembershipStatusFilter !== "all" ? mlSetMembershipStatusFilter : undefined,
    };
  }

  async function runBulkAction(action: "move_to_dataset" | "move_to_session" | "ml_set_add" | "ml_set_remove" | "tag_add" | "tag_remove" | "tag_replace" | "set_validation_status" | "set_split" | "set_expected_class", extra: Record<string, unknown>) {
    if (selectionScope === "filtered") {
      return api.bulkUpdateTakeMetadata({
        mode: "filter",
        action,
        filters: buildActiveBulkFilters(),
        exclude_take_ids: [...excludedTakeIds],
        ...extra,
      });
    }
    return api.bulkUpdateTakeMetadata({
      mode: "ids",
      action,
      take_ids: [...selectedTakeIds],
      ...extra,
    });
  }

  async function applyBulkValidation(status: string) {
    if (selectedCount === 0) return;
    setBusy(true);
    setError(null);
    try {
      await runBulkAction("set_validation_status", { validation_status: status });
      await refreshTakes();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed bulk validation update");
    } finally {
      setBusy(false);
    }
  }

  async function assignBulkClass(value: string) {
    if (selectedCount === 0) return;
    setBusy(true);
    setError(null);
    try {
      await runBulkAction("set_expected_class", { expected_class: value.trim() || null });
      await refreshTakes();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed bulk class update");
    } finally {
      setBusy(false);
    }
  }

  async function assignBulkTags(tags: string[]) {
    if (selectedCount === 0) return;
    setBusy(true);
    setError(null);
    try {
      await runBulkAction("tag_replace", { tags });
      await refreshTakes();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed bulk tag update");
    } finally {
      setBusy(false);
    }
  }

  async function addBulkTags(addTags: string[]) {
    if (selectedCount === 0) return;
    setBusy(true);
    setError(null);
    try {
      await runBulkAction("tag_add", { tags: addTags });
      await refreshTakes();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed add tags action");
    } finally {
      setBusy(false);
    }
  }

  async function removeBulkTags(tagsToRemove: string[]) {
    if (selectedCount === 0) return;
    setBusy(true);
    setError(null);
    try {
      await runBulkAction("tag_remove", { tags: tagsToRemove });
      await refreshTakes();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed remove tags action");
    } finally {
      setBusy(false);
    }
  }

  async function assignBulkSplit(value: SplitBucket) {
    if (selectedCount === 0) return;
    setBusy(true);
    setError(null);
    try {
      await runBulkAction("set_split", { split: value });
      await refreshTakes();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed bulk split update");
    } finally {
      setBusy(false);
    }
  }

  async function moveToDataset(datasetId: string, sessionId: string) {
    if (selectedCount === 0 || !datasetId.trim()) return;
    setBusy(true);
    setError(null);
    try {
      await runBulkAction("move_to_dataset", {
        dataset_id: datasetId.trim() || null,
        session_id: sessionId.trim() || null,
      });
      await refreshTakes();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed move to dataset action");
    } finally {
      setBusy(false);
    }
  }

  async function moveToSession(sessionId: string) {
    if (selectedCount === 0 || !sessionId.trim()) return;
    setBusy(true);
    setError(null);
    try {
      await runBulkAction("move_to_session", {
        dataset_id: selectedDataset || undefined,
        session_id: sessionId.trim() || null,
      });
      await refreshTakes();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed move to session action");
    } finally {
      setBusy(false);
    }
  }

  async function addToMlSet(mlSetId: string) {
    if (selectedCount === 0 || !mlSetId.trim()) return;
    setBusy(true);
    setError(null);
    try {
      await runBulkAction("ml_set_add", { tags: [mlSetId.trim()] });
      await refreshTakes();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed add to ML set action");
    } finally {
      setBusy(false);
    }
  }

  function saveCurrentCollection() {
    const name = newCollectionName.trim() || window.prompt("Collection name", "") || "";
    if (!name.trim()) return;
    const next: SavedCollection = {
      id: `${Date.now()}`,
      name: name.trim(),
      filters: {
        dataset: selectedDataset,
        session: selectedSession,
        validation: validationFilter,
        split: selectedSplit,
        search,
        tag: tagFilter,
        expectedClass: expectedClassFilter,
        calibrationOnly,
        fromDate,
        toDate,
        datePreset,
        governanceFlags: [...governanceFlags],
      },
    };
    setSavedCollections((prev) => [next, ...prev].slice(0, 16));
    setNewCollectionName("");
  }

  async function submitBulkModal() {
    if (selectedCount === 0 || !bulkModal) return;
    if (!canApplyFilteredScope) return;
    const tags = modalTagValue.split(",").map((item) => item.trim()).filter(Boolean);
    if (bulkModal === "move_dataset") await moveToDataset(modalDatasetId, modalSessionId);
    if (bulkModal === "move_session") {
      let targetSessionId = modalSessionId.trim();
      if (modalCreateSession && modalNewSessionName.trim()) {
        const created = await api.createDatasetSession(selectedDataset, {
          name: modalNewSessionName.trim(),
          notes: modalSessionNotes.trim() || undefined,
        });
        targetSessionId = created.id;
      }
      await moveToSession(targetSessionId);
    }
    if (bulkModal === "ml_set") {
      let targetMlSetId = modalMlSetId.trim();
      if (modalCreateMlSet && modalNewMlSetName.trim()) {
        const created = await api.createDatasetMlSet(selectedDataset, {
          name: modalNewMlSetName.trim(),
          description: modalNewMlSetDescription.trim() || undefined,
          notes: modalNewMlSetDescription.trim() || undefined,
          split_strategy: modalNewMlSetSplitStrategy.trim() || undefined,
          validation_requirements: modalNewMlSetValidationReq.split(",").map((item) => item.trim()).filter(Boolean),
        });
        targetMlSetId = created.id;
      }
      await api.updateDatasetMlSetMembers(selectedDataset, targetMlSetId, {
        mode: selectionScope === "filtered" ? "filter" : "ids",
        action: modalMlSetMode === "add" ? "add" : "remove",
        take_ids: selectionScope === "visible" ? [...selectedTakeIds] : undefined,
        filters: selectionScope === "filtered" ? buildActiveBulkFilters() : undefined,
        exclude_take_ids: selectionScope === "filtered" ? [...excludedTakeIds] : undefined,
      });
      setMlSets(await api.datasetMlSets(selectedDataset));
    }
    if (bulkModal === "split") await assignBulkSplit(modalSplit);
    if (bulkModal === "validation") await applyBulkValidation(modalValidationStatus);
    if (bulkModal === "tags" && modalTagMode === "add") await addBulkTags(tags);
    if (bulkModal === "tags" && modalTagMode === "remove") await removeBulkTags(tags);
    if (bulkModal === "tags" && modalTagMode === "replace") await assignBulkTags(tags);
    if (bulkModal === "expected_class") await assignBulkClass(modalExpectedClassValue);
    clearSelection();
    setSelectionScope("visible");
    selectSession("");
    const refreshedSessions = selectedDataset ? await api.datasetSessions(selectedDataset) : [];
    setSessions(refreshedSessions);
    await refreshTakes();
    setBulkModal(null);
  }

  function applyCollection(collection: SavedCollection) {
    setSelectedDataset(collection.filters.dataset);
    selectSession(collection.filters.session);
    setValidationFilter(collection.filters.validation);
    setSelectedSplit(collection.filters.split);
    setSearch(collection.filters.search);
    setTagFilter(collection.filters.tag);
    setExpectedClassFilter(collection.filters.expectedClass);
    setCalibrationOnly(collection.filters.calibrationOnly);
    setFromDate(collection.filters.fromDate);
    setToDate(collection.filters.toDate);
    setDatePreset((collection.filters.datePreset as DatePreset) || "custom");
    setGovernanceFlags(new Set((collection.filters.governanceFlags ?? []) as GovernanceFilter[]));
  }

  function selectVisibleRows() {
    setSelectionScope("visible");
    setExcludedTakeIds(new Set());
    setSelectedTakeIds(new Set(visibleTakeIds));
  }

  function clearSelection() {
    setSelectionScope("visible");
    setSelectedTakeIds(new Set());
    setExcludedTakeIds(new Set());
  }

  function selectAllMatchingFilters() {
    setSelectionScope("filtered");
    setSelectedTakeIds(new Set());
    setExcludedTakeIds(new Set());
  }

  const filteredScopeActionsSupported = true;
  const canApplyFilteredScope = selectionScope !== "filtered" || filteredScopeActionsSupported;

  function openBulkModal(kind: BulkModalKind) {
    setBulkModal(kind);
    setModalDatasetId(selectedDataset || datasets[0]?.id || "");
    setModalSessionId(selectedSession || "");
    setModalMlSetId(mlSets[0]?.id || "");
    setModalSplit("training");
    setModalTagMode("add");
    setModalTagValue("");
    setModalCreateSession(false);
    setModalSessionNotes("");
    setModalCreateMlSet(false);
    setModalNewMlSetDescription("");
    setModalNewMlSetSplitStrategy("");
    setModalNewMlSetValidationReq("");
    setModalAffectedCount(null);
    const from = fromDate || "";
    const to = toDate || "";
    if (from && to) {
      setModalNewSessionName(from === to ? `Session ${from}` : `Session ${from} to ${to}`);
      const fromFmt = from.replace(/-/g, "_");
      const toFmt = to.replace(/-/g, "_");
      setModalNewMlSetName(from === to ? `ml_set_${fromFmt}` : `ml_set_${fromFmt}_to_${toFmt}`);
    } else if (from) {
      setModalNewSessionName(`Session ${from}`);
      setModalNewMlSetName(`ml_set_${from.replace(/-/g, "_")}`);
    } else {
      setModalNewSessionName("");
      const datasetToken = (selectedDatasetSummary?.name || selectedDataset || "dataset")
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "_")
        .replace(/^_+|_+$/g, "");
      setModalNewMlSetName(`ml_set_${datasetToken}_${Date.now()}`);
    }
  }

  useEffect(() => {
    if (!bulkModal || bulkModal !== "move_dataset" || !modalDatasetId || modalDatasetId === selectedDataset) {
      setModalDatasetSessions([]);
      return;
    }
    void api.datasetSessions(modalDatasetId)
      .then((items) => setModalDatasetSessions(items))
      .catch(() => setModalDatasetSessions([]));
  }, [bulkModal, modalDatasetId, selectedDataset]);

  useEffect(() => {
    if (!bulkModal) return;
    setModalCounting(true);
    setModalAffectedCount(null);
    if (bulkModal === "ml_set" && !modalCreateMlSet && modalMlSetId.trim()) {
      const mlSetDryRun = api.updateDatasetMlSetMembers(selectedDataset, modalMlSetId.trim(), {
        mode: selectionScope === "filtered" ? "filter" : "ids",
        action: modalMlSetMode === "add" ? "add" : "remove",
        take_ids: selectionScope === "visible" ? [...selectedTakeIds] : undefined,
        filters: selectionScope === "filtered" ? buildActiveBulkFilters() : undefined,
        exclude_take_ids: selectionScope === "filtered" ? [...excludedTakeIds] : undefined,
        dry_run: true,
      });
      void mlSetDryRun.then((res) => {
        setModalAffectedCount(res.affected_count);
      }).finally(() => setModalCounting(false));
      return;
    }
    const action =
      bulkModal === "move_dataset" ? "move_to_dataset"
      : bulkModal === "move_session" ? "move_to_session"
      : bulkModal === "ml_set" ? (modalMlSetMode === "add" ? "ml_set_add" : "ml_set_remove")
      : bulkModal === "split" ? "set_split"
      : bulkModal === "tags" ? (modalTagMode === "add" ? "tag_add" : modalTagMode === "remove" ? "tag_remove" : "tag_replace")
      : bulkModal === "validation" ? "set_validation_status"
      : "set_expected_class";
    const payload = selectionScope === "filtered"
      ? api.bulkUpdateTakeMetadata({
          mode: "filter",
          action,
          filters: buildActiveBulkFilters(),
          exclude_take_ids: [...excludedTakeIds],
          dry_run: true,
        })
      : api.bulkUpdateTakeMetadata({
          mode: "ids",
          action,
          take_ids: [...selectedTakeIds],
          dry_run: true,
        });
    void payload.then((res) => {
      setModalAffectedCount(res.affected_count);
    }).finally(() => setModalCounting(false));
  }, [bulkModal, modalCreateMlSet, modalMlSetId, modalMlSetMode, modalTagMode, selectionScope, selectedDataset, selectedSession, validationFilter, search, tagFilter, fromDate, toDate, selectedSplit, expectedClassFilter, calibrationOnly, excludedTakeIds, selectedTakeIds]);

  async function archiveSelected() {
    if (!selectedTakeIds.size) return;
    const reason = window.prompt("Archive reason", "semantic_cleanup") ?? undefined;
    setBusy(true);
    setError(null);
    try {
      await Promise.all([...selectedTakeIds].map((takeId) => api.archiveTake(takeId, reason)));
      setSelectedTakeIds(new Set());
      await refreshTakes();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed archive action");
    } finally {
      setBusy(false);
    }
  }

  async function restoreSelected() {
    if (!selectedTakeIds.size) return;
    setBusy(true);
    setError(null);
    try {
      await Promise.all([...selectedTakeIds].map((takeId) => api.restoreTake(takeId)));
      await refreshTakes();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed restore action");
    } finally {
      setBusy(false);
    }
  }

  async function assignObjectLabel(candidateId: string) {
    if (!selectedTakeId) return;
    const value = window.prompt("Labels (comma separated)", "");
    if (value === null) return;
    const labels = value.split(",").map((item) => item.trim()).filter(Boolean);
    setBusy(true);
    setError(null);
    try {
      await api.upsertObjectAnnotation(selectedTakeId, { candidate_id: candidateId, labels, validation_status: "review" });
      const refreshed = await api.take(selectedTakeId);
      setDetail(refreshed);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed object annotation update");
    } finally {
      setBusy(false);
    }
  }

  async function excludeSelectionFromActiveMlSet(takeIds: string[]) {
    if (!activeMlSetScope?.mlSetId || !selectedDataset || takeIds.length === 0) return;
    setBusy(true);
    setError(null);
    try {
      await api.updateDatasetMlSetMembers(selectedDataset, activeMlSetScope.mlSetId, {
        mode: "ids",
        action: "remove",
        take_ids: takeIds,
      });
      await refreshMlSetsForDataset(selectedDataset);
      await refreshTakes();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to exclude selection from ML set");
    } finally {
      setBusy(false);
    }
  }

  async function updateActiveMlSetSelection(payload: {
    split?: string;
    include?: boolean;
    expected_class?: string;
    expected_subclass?: string;
    review_required?: boolean;
    default_trainable?: boolean;
  }, takeIds: string[]) {
    if (!activeMlSetScope?.mlSetId || !selectedDataset || takeIds.length === 0) return;
    setBusy(true);
    setError(null);
    try {
      await api.updateDatasetMlSetMemberships(selectedDataset, activeMlSetScope.mlSetId, {
        take_ids: takeIds,
        ...payload,
      });
      await refreshTakes();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed ML set membership update");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="datasets-page datasets-command-center">
      <section className="datasets-headline">
        <div>
          <h2>Datasets</h2>
          <p>Organize captures, labels, object reviews, ML sets, and splits.</p>
        </div>
        <button
          type="button"
          className={`datasets-profiler-toggle ${datasetsProfilerEnabled ? "active" : ""}`}
          onClick={() => setDatasetsProfilerEnabled((prev) => !prev)}
        >
          {datasetsProfilerEnabled ? "Profiler on" : "Profiler off"}
        </button>
      </section>

      <section className={`datasets-grid ${leftPanelCollapsed ? "left-collapsed" : ""} ${rightPanelCollapsed ? "right-collapsed" : ""}`}>
        <aside className={`concept-panel datasets-left ${leftPanelCollapsed ? "is-collapsed" : ""}`}>
          <div className="datasets-left-title-row">
            <h3>Dataset Explorer</h3>
            <button
              type="button"
              className="panel-collapse-toggle"
              onClick={() => setLeftPanelCollapsed((current) => !current)}
              aria-expanded={!leftPanelCollapsed}
              aria-label={leftPanelCollapsed ? "Expand dataset explorer" : "Collapse dataset explorer"}
              title={leftPanelCollapsed ? "Expand dataset explorer" : "Collapse dataset explorer"}
            >
              {leftPanelCollapsed ? "›" : "‹"}
            </button>
            {!leftPanelCollapsed && (
              <small>
                {hasBaseFilters
                  ? `${filteredCount} filtered / ${datasetTotalCount} total`
                  : `${datasetTotalCount} takes`}
              </small>
            )}
          </div>
          {!leftPanelCollapsed && (
          <>
          <details className="datasets-section" open>
            <summary>Dataset hierarchy</summary>
            <div className="datasets-hierarchy-block">
            <div className="datasets-hierarchy-header">
              <strong>{selectedDatasetSummary?.name ?? "No dataset selected"}</strong>
              <button
                type="button"
                className="datasets-hierarchy-toggle"
                aria-label={showSessionTree ? "Hide sessions" : "Show sessions"}
                title={showSessionTree ? "Hide sessions" : "Show sessions"}
                onClick={() => setShowSessionTree((prev) => !prev)}
              >
                {showSessionTree ? "▾" : "▸"}
              </button>
            </div>
            <small>{sessions.length} sessions · validation {filteredCount ? Math.round((validatedCount / filteredCount) * 100) : 0}%</small>
            {showSessionTree && (
              <div className="datasets-session-tree">
                {sessions.map((session) => {
                  // Server-side counts, not a filter over `takes`: that array holds
                  // one page, so a session outside it reported 0 takes and the one
                  // inside it reported the page size.
                  const sessionTakeCount = session.take_count ?? 0;
                  const sessionValid = session.validated_take_count ?? 0;
                  return (
                    <div
                      key={session.id}
                      className={`datasets-session-node ${selectedSession === session.id ? "active" : ""}`}
                      role="button"
                      tabIndex={0}
                      onClick={() => selectSession(session.id)}
                      onKeyDown={(event) => {
                        if (event.key === "Enter" || event.key === " ") {
                          event.preventDefault();
                          selectSession(session.id);
                        }
                      }}
                    >
                      <div className="datasets-session-node-main">
                        <span>{session.name}</span>
                        <small>{sessionTakeCount} takes · {sessionTakeCount ? Math.round((sessionValid / sessionTakeCount) * 100) : 0}% valid</small>
                      </div>
                      <button
                        type="button"
                        className="datasets-session-node-menu"
                        aria-label="Edit session"
                        title="Edit session"
                        onClick={(event) => {
                          event.stopPropagation();
                          openSessionDrawer(session.id);
                        }}
                      >
                        ⋯
                      </button>
                    </div>
                  );
                })}
              </div>
            )}
            </div>
          </details>
          <details className="datasets-section" open>
            <summary>Filters</summary>
          <label className="field-label">
            Dataset
            <select value={selectedDataset} onChange={(e) => { setSelectedDataset(e.target.value); selectSession(""); }}>
              <option value="">All datasets</option>
              {datasets.map((dataset) => <option key={dataset.id} value={dataset.id}>{dataset.name}</option>)}
            </select>
          </label>
          <label className="field-label">
            Session
            <select value={selectedSession} onChange={(e) => selectSession(e.target.value)}>
              <option value="">All sessions</option>
              {sessions.map((session) => <option key={session.id} value={session.id}>{session.name}</option>)}
            </select>
          </label>
          <label className="field-label">
            Validation state
            <select value={validationFilter} onChange={(e) => setValidationFilter(e.target.value)}>
              <option value="all">All</option>
              <option value="unreviewed">Unreviewed</option>
              <option value="valid">Valid</option>
              <option value="invalid">Rejected</option>
              <option value="needs_review">Needs review</option>
              <option value="golden_sample">Golden sample</option>
              <option value="benchmark_approved">Benchmark approved</option>
            </select>
          </label>
          <label className="field-label">
            Split
            <select value={selectedSplit} onChange={(e) => setSelectedSplit(e.target.value as "all" | SplitBucket)}>
              <option value="all">All splits</option>
              {splitBuckets.map((split) => <option key={split} value={split}>{split}</option>)}
            </select>
          </label>
          <div className="datasets-temporal-grid">
            <label className="field-label">
              Preset
              <select value={datePreset} onChange={(e) => setDatePreset(e.target.value as DatePreset)}>
                <option value="custom">Custom</option>
                <option value="today">Today</option>
                <option value="last_24h">Last 24h</option>
                <option value="last_7d">Last 7d</option>
                <option value="last_30d">Last 30d</option>
                <option value="this_week">This week</option>
                <option value="this_shift">This shift (planned)</option>
              </select>
            </label>
            <label className="field-label">
              From
              <input type="date" value={fromDate} onChange={(e) => { setFromDate(e.target.value); setDatePreset("custom"); }} />
            </label>
            <label className="field-label">
              To
              <input type="date" value={toDate} onChange={(e) => { setToDate(e.target.value); setDatePreset("custom"); }} />
            </label>
          </div>
          <label className="field-label">
            Search
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder={activeMlSetScope ? "take, object, label, superclass, notes, source row" : "take id or text"}
            />
          </label>
          <label className="field-label">
            Tags
            <input value={tagFilter} onChange={(e) => setTagFilter(e.target.value)} placeholder="tag" />
          </label>
          <label className="field-label">
            Expected class
            <input value={expectedClassFilter} onChange={(e) => setExpectedClassFilter(e.target.value)} placeholder="class" />
          </label>
          <label className="field-inline checkbox-inline">
            <input type="checkbox" checked={calibrationOnly} onChange={(e) => setCalibrationOnly(e.target.checked)} />
            Calibration linkage only
          </label>
          </details>

          <div className="datasets-quick-stats">
            <span className="status-chip">sessions: {sessions.length}</span>
            <span className="status-chip">valid: {validatedCount}</span>
            <span className="status-chip">unreviewed: {unreviewedCount}</span>
            <span className="status-chip">selected split: {selectedSplit}</span>
          </div>
          <details className="datasets-section" open>
            <summary>Needs attention</summary>
          <div className="datasets-roadmap-list concept-panel compact">
            <h3>Needs Attention</h3>
            <div className="datasets-chip-row">
              {GOVERNANCE_FILTERS.map((flag) => {
                const active = governanceFlags.has(flag);
                return (
                  <button
                    key={flag}
                    type="button"
                    className={`datasets-governance-chip ${active ? "active" : ""}`}
                    onClick={() => setGovernanceFlags((prev) => {
                      const next = new Set(prev);
                      if (next.has(flag)) next.delete(flag);
                      else next.add(flag);
                      return next;
                    })}
                  >
                    {flag.replace(/_/g, " ")}
                  </button>
                );
              })}
            </div>
          </div>
          </details>
          <details className="datasets-section" open>
            <summary>Saved collections</summary>
          <div className="datasets-roadmap-list concept-panel compact">
            <h3>Saved Collections</h3>
            <div className="datasets-collection-create">
              <input value={newCollectionName} onChange={(e) => setNewCollectionName(e.target.value)} placeholder="collection name" />
              <button type="button" onClick={saveCurrentCollection}>Save</button>
            </div>
            {savedCollections.slice(0, 8).map((collection) => (
              <button key={collection.id} type="button" className="datasets-collection-pill" onClick={() => applyCollection(collection)}>{collection.name}</button>
            ))}
          </div>
          </details>
          </>
          )}
        </aside>

        <section className="datasets-center">
          <nav className="datasets-tabs" aria-label="Datasets semantic workspace">
            {[
              ["overview", "Overview"],
              ["takes", "Takes"],
              ["physical_objects", "Physical Objects"],
              ["objects", "Objects"],
              ["labels", "Labels"],
              ["ml_sets", "ML Sets"],
              ["splits", "Splits"],
            ].map(([id, label]) => (
              <button key={id} type="button" className={activeTab === id ? "active" : ""} onClick={() => setActiveTab(id as DatasetTab)}>{label}</button>
            ))}
          </nav>

          {activeTab === "overview" && (
            <section className="datasets-tab-panel">
              <div className="datasets-overview-kpis">
                <article><span>takes</span><strong>{hasBaseFilters ? filteredCount : datasetTotalCount}</strong></article>
                <article><span>validated</span><strong>{validatedCount}</strong></article>
                <article><span>unreviewed</span><strong>{unreviewedCount}</strong></article>
                <article><span>rejected</span><strong>{rejectedCount}</strong></article>
                <article><span>needs review</span><strong>{needsReviewCount}</strong></article>
                <article><span>golden sample</span><strong>{goldenSampleCount}</strong></article>
                <article><span>missing labels</span><strong>{missingLabelsCount}</strong></article>
                <article><span>processing coverage</span><strong>{(hasBaseFilters ? filteredCount : datasetTotalCount) ? `${Math.round((((hasBaseFilters ? filteredCount : datasetTotalCount) - processingIncompleteCount) / (hasBaseFilters ? filteredCount : datasetTotalCount)) * 100)}%` : "0%"}</strong></article>
                <article><span>split coverage</span><strong>{(hasBaseFilters ? filteredCount : datasetTotalCount) ? `${Math.round((((hasBaseFilters ? filteredCount : datasetTotalCount) - missingSplitCount) / (hasBaseFilters ? filteredCount : datasetTotalCount)) * 100)}%` : "0%"}</strong></article>
                <article><span>sessions</span><strong>{sessions.length}</strong></article>
              </div>
              <div className="datasets-overview-grid">
                <section className="concept-panel compact">
                  <h3>Class Distribution</h3>
                  <div className="datasets-mini-bars">
                    {sortedClassCounts.map(([label, count]) => (
                      <div key={label} className="datasets-mini-bar">
                        <small>{label}</small>
                        <strong>{count}</strong>
                      </div>
                    ))}
                  </div>
                </section>
                <section className="concept-panel compact">
                  <h3>Session Distribution</h3>
                  <div className="datasets-mini-bars">
                    {sortedSessionCounts.map(([label, count]) => (
                      <div key={label} className="datasets-mini-bar">
                        <small>{label}</small>
                        <strong>{count}</strong>
                      </div>
                    ))}
                  </div>
                </section>
                <section className="concept-panel compact">
                  <h3>Split Readiness</h3>
                  <div className="datasets-chip-row">
                    {splitBuckets.map((bucket) => <span className="status-chip" key={bucket}>{bucket}: {stats.splits[bucket]}</span>)}
                  </div>
                </section>
                <section className="concept-panel compact">
                  <h3>Recent Activity</h3>
                  <div className="datasets-recent-list">
                    {stats.recent.map((take) => <small key={take.take_id}>{take.friendly_name || take.take_id} · {take.validation_status || "unreviewed"}</small>)}
                  </div>
                </section>
                <section className="concept-panel compact">
                  <h3>Dataset Health</h3>
                  <strong>{healthScore}%</strong>
                  <small>validation + split + processing + calibration coverage</small>
                </section>
                <section className="concept-panel compact">
                  <h3>Processing Coverage</h3>
                  <small>2D: {stats.total ? `${Math.round((takes.filter((t) => t.has_done || t.has_ready).length / stats.total) * 100)}%` : "0%"}</small>
                  <small>3D: {stats.total ? `${Math.round((takes.filter((t) => (t.modalities ?? []).includes("point_cloud")).length / stats.total) * 100)}%` : "0%"}</small>
                  <small>classification: {stats.total ? `${Math.round((takes.filter((t) => Boolean(t.processed_class_label)).length / stats.total) * 100)}%` : "0%"}</small>
                  <small>feature extraction: placeholder</small>
                </section>
              </div>
            </section>
          )}

          {activeTab === "takes" && (
            <section className="datasets-tab-panel">
              <div className="datasets-queue-row">
                {[
                  { id: "unreviewed", label: "Unreviewed", count: activeReviewCounts.unreviewed, apply: () => setValidationFilter("unreviewed") },
                  { id: "missing_labels", label: "Missing labels", count: activeReviewCounts.missingLabels, apply: () => activeMlSetScope ? setMlSetNormalizedClassFilter("") : setGovernanceFlags(new Set(["missing_expected_class"])) },
                  { id: "calibration", label: "Calibration review", count: missingCalibrationCount, apply: () => setGovernanceFlags(new Set(["missing_calibration"])) },
                  { id: "benchmark", label: "Benchmark approval", count: benchmarkApprovedCount, apply: () => setValidationFilter("benchmark_approved") },
                  { id: "review_required", label: activeMlSetScope ? "Review required" : "Failed processing", count: activeMlSetScope ? activeReviewCounts.reviewRequiredMembers : processingFailedCount, apply: () => activeMlSetScope ? setMlSetMembershipStatusFilter("review_required") : setGovernanceFlags(new Set(["processing_failed"])) },
                ].map((queue) => (
                  <button key={queue.id} type="button" className="datasets-queue-card" onClick={queue.apply}>
                    <strong>{queue.label}</strong>
                    <small>{queue.count}</small>
                  </button>
                ))}
              </div>
              {activeMlSetScope && (
                <section className="concept-panel compact datasets-session-context datasets-ml-set-context">
                  <div className="datasets-session-context-head">
                    <div className="datasets-session-context-copy">
                      <h3>ML Set: {activeScopedMlSet?.name || activeMlSetScope.mlSetId}</h3>
                      <small>{scopeContextCounts.members} members · {scopeContextCounts.objects} objects · {scopeContextCounts.reviewRequired} review required</small>
                    </div>
                    <div className="datasets-actions-row">
                      <button
                        type="button"
                        title="Open Feature Analytics for the current ML-set slice in a new tab"
                        aria-label="Open Feature Analytics for the current ML-set slice in a new tab"
                        onClick={() => openFeatureAnalyticsForSliceInNewTab()}
                      >
                        Analyze features for current slice ↗
                      </button>
                      <button type="button" className="datasets-session-context-edit" onClick={clearMlSetScope}>Clear scope</button>
                    </div>
                  </div>
                  <div className="datasets-toolbar ml-set-scope-toolbar">
                    <label className="field-label">View<select value={mlSetScopedView} onChange={(event) => setMlSetScopedView(event.target.value as MlSetScopedView)}><option value="takes">Takes</option><option value="object_groups">Object Groups</option></select></label>
                    <label className="field-label">
                      object_id
                      <input
                        value={mlSetObjectIdFilter}
                        list="ml-set-object-options"
                        onChange={(event) => setMlSetObjectIdFilter(event.target.value)}
                        placeholder={activeMlSetScope.physicalObjectId || "Search object id"}
                      />
                      <datalist id="ml-set-object-options">
                        {mlSetObjectOptions.map((option) => <option key={option.value} value={option.value} label={option.label} />)}
                      </datalist>
                    </label>
                    <label className="field-label">raw_label<select value={mlSetRawLabelFilter} onChange={(event) => setMlSetRawLabelFilter(event.target.value)}><option value="">All raw labels</option>{mlSetRawLabelOptions.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}</select></label>
                    <label className="field-label">normalized_class<select value={mlSetNormalizedClassFilter} onChange={(event) => setMlSetNormalizedClassFilter(event.target.value)}><option value="">All normalized classes</option>{mlSetNormalizedClassOptions.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}</select></label>
                    <label className="field-label">superclass<select value={mlSetSuperclassFilter} onChange={(event) => setMlSetSuperclassFilter(event.target.value)}><option value="">All superclasses</option>{mlSetSuperclassOptions.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}</select></label>
                    <label className="field-label">review/default_trainable<select value={mlSetMembershipStatusFilter} onChange={(event) => setMlSetMembershipStatusFilter(event.target.value as MlSetMembershipStatusFilter)}><option value="all">{formatFacetCountLabel("All members", mlSetMembershipCounts.get("all"))}</option><option value="default_trainable">{formatFacetCountLabel("Default trainable", mlSetMembershipCounts.get("default_trainable"))}</option><option value="review_required">{formatFacetCountLabel("Review required", mlSetMembershipCounts.get("review_required"))}</option><option value="excluded">{formatFacetCountLabel("Excluded", mlSetMembershipCounts.get("excluded"))}</option>{mlSetMembershipCounts.has("uncertain") && <option value="uncertain">{formatFacetCountLabel("Uncertain", mlSetMembershipCounts.get("uncertain"))}</option>}</select></label>
                    <label className="field-label">split<select value={mlSetSplitFilter} onChange={(event) => setMlSetSplitFilter(event.target.value)}><option value="">All splits</option>{mlSetSplitOptions.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}</select></label>
                    <label className="field-label">session<select value={mlSetSessionFilter} onChange={(event) => setMlSetSessionFilter(event.target.value)}><option value="">All sessions</option>{mlSetSessionOptions.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}</select></label>
                  </div>
                </section>
              )}
              {!activeMlSetScope && selectedSessionSummary && (
                <section className="concept-panel compact datasets-session-context">
                  <div className="datasets-session-context-head">
                    <div className="datasets-session-context-copy">
                      <h3>Session: {selectedSessionSummary.name}</h3>
                      <small>{selectedSessionTakeCount} takes · reviewed {selectedSessionReviewedPct}% · validation {selectedSessionValidatedPct}% · calibration {selectedSessionCalibrationStatus}{selectedSessionSummary.session_type ? ` · type ${selectedSessionSummary.session_type}` : ""}</small>
                    </div>
                    <button type="button" className="datasets-session-context-edit" onClick={() => openSessionDrawer(selectedSessionSummary.id)}>Edit session</button>
                  </div>
                </section>
              )}
              <div className="datasets-toolbar" role="toolbar" aria-label="Takes selection and bulk actions">
                <span className="datasets-toolbar-status">{selectionScope === "filtered" ? `All ${activeFilteredCount} filtered takes selected${excludedTakeIds.size ? ` · ${excludedTakeIds.size} excluded` : ""}` : `${selectedTakeIds.size} visible selected`}</span>
                <span className="datasets-toolbar-status">{activeMlSetScope ? `Showing ${activeReviewRows.length} of ${activeFilteredCount} scoped members` : hasBaseFilters ? `Showing ${takes.length} of ${filteredCount} filtered takes · ${datasetTotalCount} total in dataset` : `Showing ${takes.length} of ${datasetTotalCount} takes`}</span>
                <button type="button" onClick={selectVisibleRows}>Select visible</button>
                <button type="button" onClick={clearSelection}>Clear selection</button>
                <button type="button" onClick={selectAllMatchingFilters}>Select all matching filters</button>
                <span className="datasets-toolbar-divider" aria-hidden>|</span>
                <button type="button" disabled={busy || selectedCount === 0} onClick={() => { setModalValidationStatus("valid"); openBulkModal("validation"); }}>Validated</button>
                <button type="button" disabled={busy || selectedCount === 0} onClick={() => { setModalValidationStatus("needs_review"); openBulkModal("validation"); }}>Needs review</button>
                <button type="button" disabled={busy || selectedCount === 0} onClick={() => { setModalValidationStatus("invalid"); openBulkModal("validation"); }}>Rejected</button>
                <button type="button" disabled={busy || selectedCount === 0} onClick={() => openBulkModal("tags")}>Add tags</button>
                <button type="button" disabled={busy || selectedCount === 0} onClick={() => openBulkModal("split")}>Split assignment</button>
                <details className="datasets-more-actions">
                  <summary>More ▾</summary>
                  <div className="datasets-more-actions-menu">
                    <div className="datasets-more-actions-group">
                      <small>Organization</small>
                      <button type="button" disabled={busy || selectedCount === 0} onClick={() => openBulkModal("move_dataset")}>Move to dataset</button>
                      <button type="button" disabled={busy || selectedCount === 0} onClick={() => openBulkModal("move_session")}>Move to session</button>
                    </div>
                    <div className="datasets-more-actions-group">
                      <small>ML Sets</small>
                      <button type="button" disabled={busy || selectedCount === 0} onClick={() => { setModalMlSetMode("add"); openBulkModal("ml_set"); }}>Add to ML Set</button>
                      <button type="button" disabled={busy || selectedCount === 0} onClick={() => { setModalMlSetMode("remove"); openBulkModal("ml_set"); }}>Remove from ML Set</button>
                    </div>
                    <div className="datasets-more-actions-group">
                      <small>Lifecycle</small>
                      <button type="button" disabled={busy || selectedCount === 0 || selectionScope === "filtered"} title={selectionScope === "filtered" ? "Archive currently supports visible-id selection only." : ""} onClick={() => void archiveSelected()}>Archive</button>
                      <button type="button" disabled={busy || selectedCount === 0 || selectionScope === "filtered"} title={selectionScope === "filtered" ? "Restore currently supports visible-id selection only." : ""} onClick={() => void restoreSelected()}>Restore</button>
                    </div>
                    <div className="datasets-more-actions-group">
                      <small>Advanced</small>
                      <button type="button" disabled={Boolean(activeMlSetScope) || selectionScope !== "visible" || selectedTakeIds.size < 2 || selectedTakeIds.size > 4} onClick={() => setCompareOpen((prev) => !prev)}>Compare selected</button>
                      <button type="button" disabled={busy || selectedCount === 0} onClick={() => openBulkModal("expected_class")}>Expected class</button>
                      <button type="button" disabled={busy || selectedCount === 0} onClick={() => { setModalValidationStatus("golden_sample"); openBulkModal("validation"); }}>Golden sample</button>
                      <button type="button" disabled={busy || selectedCount === 0} onClick={() => { setModalValidationStatus("benchmark_approved"); openBulkModal("validation"); }}>Benchmark approved</button>
                    </div>
                  </div>
                </details>
              </div>
              {!activeMlSetScope && datasetsProfilerEnabled && (
                <div className="datasets-profiler-panel concept-panel compact">
                  <div className="datasets-profiler-header">
                    <strong>Datasets profiler</strong>
                    <small>{datasetsProfiler?.captured_at ? `last capture ${new Date(datasetsProfiler.captured_at).toLocaleTimeString()}` : "waiting for next paged request"}</small>
                  </div>
                  <div className="datasets-profiler-grid">
                    <div><span>Frontend request</span><strong>{formatProfilerMs(datasetsProfiler?.frontend_request_ms)}</strong></div>
                    <div><span>Frontend commit</span><strong>{formatProfilerMs(datasetsProfiler?.frontend_commit_ms)}</strong></div>
                    <div><span>Backend total</span><strong>{formatProfilerMs(datasetsProfiler?.backend_profile?.phase_ms.total)}</strong></div>
                    <div><span>Scan + filter</span><strong>{formatProfilerMs(datasetsProfiler?.backend_profile?.phase_ms.scan_and_filter)}</strong></div>
                    <div><span>Hydrate rows</span><strong>{formatProfilerMs(datasetsProfiler?.backend_profile?.phase_ms.hydrate_page_items)}</strong></div>
                    <div><span>Scanned takes</span><strong>{datasetsProfiler?.backend_profile?.counters.scanned_take_ids ?? "-"}</strong></div>
                    <div><span>Filtered matches</span><strong>{datasetsProfiler?.backend_profile?.counters.candidate_count ?? "-"}</strong></div>
                    <div><span>Page rows</span><strong>{datasetsProfiler?.page_item_count ?? "-"}</strong></div>
                  </div>
                </div>
              )}
              {!activeMlSetScope && compareOpen && (
                <div className="concept-panel compact datasets-compare-panel">
                  <h3>Compare Selected</h3>
                  <div className="datasets-compare-grid">
                    {[...selectedTakeIds].slice(0, 4).map((takeId) => {
                      const take = takes.find((item) => item.take_id === takeId);
                      if (!take) return null;
                      const preview = takePreviewsById.get(take.take_id) ?? { url: null, placeholder: "TAKE", modality: null };
                      return (
                        <article key={take.take_id} className="datasets-compare-card">
                          {preview.url ? <img className="datasets-thumb" src={preview.url} alt={take.take_id} loading="lazy" decoding="async" /> : <span className="datasets-thumb-placeholder">{preview.placeholder}</span>}
                          <strong>{take.friendly_name || take.take_id}</strong>
                          <small>validation: {take.validation_status || "-"}</small>
                          <small>split: {readSplitFromTake(take) || "-"}</small>
                          <small>processing: {take.has_done ? "done" : take.has_ready ? "ready" : "pending"}</small>
                          <small>objects: {String(take.object_count ?? "-")}</small>
                          <small>calibration: {take.calibration_id || "-"}</small>
                        </article>
                      );
                    })}
                  </div>
                </div>
              )}
              <div className="datasets-table-wrap">
                {isInitialLoading && <div className="datasets-table-skeleton" aria-hidden="true">{Array.from({ length: 8 }).map((_, idx) => <div className="datasets-skeleton-row" key={idx} />)}</div>}
                {!isInitialLoading && activeMlSetScope && mlSetScopedView === "object_groups" && objectGroupRows.length === 0 && <div className="datasets-inline-state">No object groups match the current ML set filters.</div>}
                {!isInitialLoading && activeMlSetScope && mlSetScopedView === "object_groups" && objectGroupRows.length > 0 && (
                  <table className="datasets-table">
                    <thead><tr><th /></tr></thead>
                    <tbody>
                      {objectGroupRows.map((group) => {
                        const expanded = expandedObjectGroupIds.has(group.id);
                        return (
                          <Fragment key={group.id}>
                            <tr key={group.id} className={selectedObjectGroupId === group.id ? "active" : ""} onClick={() => setSelectedObjectGroupId(group.id)}>
                              <td>
                                <div className="datasets-object-group-row">
                                  <button type="button" className="link-button" onClick={(event) => { event.stopPropagation(); setExpandedObjectGroupIds((prev) => { const next = new Set(prev); if (next.has(group.id)) next.delete(group.id); else next.add(group.id); return next; }); }}>{expanded ? "▾" : "▸"}</button>
                                  <strong>{group.physicalObjectId || "unassigned"}</strong>
                                  <div className="datasets-inline-thumbs">
                                    {group.members.slice(0, 3).map((member) => {
                                      const preview = takePreviewsById.get(member.take_id) ?? { url: null, placeholder: "TAKE", modality: null };
                                      return preview.url ? <img key={member.take_id} className="datasets-thumb micro" src={preview.url} alt={member.take_id} loading="lazy" decoding="async" /> : <span key={member.take_id} className="datasets-thumb-placeholder micro">{preview.placeholder}</span>;
                                    })}
                                  </div>
                                  <span>{group.takeCount} takes</span>
                                  <span>{group.rawLabels.join(", ") || "-"}</span>
                                  <span>{group.normalizedClass}</span>
                                  <span>{group.superclass}</span>
                                  <span>{group.warningCodes.join(", ") || "-"}</span>
                                  <span>{group.splitValues.join(", ") || "-"}</span>
                                  <span>{group.sessions.join(", ") || "-"}</span>
                                </div>
                              </td>
                            </tr>
                            {expanded && group.members.map((member) => (
                              <tr key={`${group.id}_${member.take_id}`} className="datasets-subrow" onClick={() => setSelectedTakeId(member.take_id)}>
                                <td>{member.take_id} · {member.raw_label || "-"} · {member.session_name || member.session_id || "-"} · {member.split}</td>
                              </tr>
                            ))}
                          </Fragment>
                        );
                      })}
                    </tbody>
                  </table>
                )}
                {!isInitialLoading && activeMlSetScope && mlSetScopedView === "takes" && activeReviewRows.length === 0 && <div className="datasets-inline-state">No ML set members match the current scope and filters.</div>}
                {!isInitialLoading && activeMlSetScope && mlSetScopedView === "takes" && activeReviewRows.length > 0 && (
                  <table className="datasets-table">
                    <thead>
                      <tr>
                        <th />
                        <th>thumbnail</th>
                        <th>take_id</th>
                        <th>physical_object_id</th>
                        <th>raw_label</th>
                        <th>normalized_class</th>
                        <th>superclass</th>
                        <th>review/default_trainable</th>
                        <th>split</th>
                        <th>session</th>
                        <th>feature/readiness</th>
                      </tr>
                    </thead>
                    <tbody>
                      {activeReviewRows.map((row) => {
                        const selected = selectionScope === "filtered" ? !excludedTakeIds.has(row.take_id) : selectedTakeIds.has(row.take_id);
                        const preview = takePreviewsById.get(row.take_id) ?? { url: null, placeholder: "TAKE", modality: null };
                        return (
                          <tr key={row.take_id} className={selectedTakeId === row.take_id ? "active" : ""} onClick={() => setSelectedTakeId(row.take_id)}>
                            <td><input type="checkbox" checked={selected} onClick={(event) => event.stopPropagation()} onChange={(event) => { if (selectionScope === "filtered") { setExcludedTakeIds((prev) => { const next = new Set(prev); if (event.target.checked) next.delete(row.take_id); else next.add(row.take_id); return next; }); } else { setSelectedTakeIds((prev) => { const next = new Set(prev); if (event.target.checked) next.add(row.take_id); else next.delete(row.take_id); return next; }); } }} /></td>
                            <td>{preview.url ? <img className="datasets-thumb" src={preview.url} alt={row.take_id} loading="lazy" decoding="async" /> : <span className="datasets-thumb-placeholder">{preview.placeholder}</span>}</td>
                            <td><button type="button" className="link-button" onClick={() => setSelectedTakeId(row.take_id)}>{row.take_id}</button></td>
                            <td><button type="button" className="link-button" onClick={() => setMlSetObjectIdFilter(row.physical_object_id || "")}>{row.physical_object_id || "-"}</button></td>
                            <td>{row.raw_label || "-"}</td>
                            <td>{row.normalized_class || "-"}</td>
                            <td>{row.superclass || "-"}</td>
                            <td>{row.review_required ? "review_required" : row.default_trainable ? "default_trainable" : row.include ? "included" : "excluded"}</td>
                            <td>{row.split}</td>
                            <td>{row.session_name || row.session_id || "-"}</td>
                            <td>{[row.processing_ready ? "processing" : "pending", row.feature_ready ? "features" : "no_features"].join(" · ")}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                )}
                {!isInitialLoading && !activeMlSetScope && takes.length === 0 && <div className="datasets-inline-state">No takes match the current filters.</div>}
                {!isInitialLoading && !activeMlSetScope && takes.length > 0 && (
                  <table className="datasets-table">
                    <thead>
                      <tr>
                        <th />
                        <th>thumb</th>
                        <th>take</th>
                        <th>dataset</th>
                        <th>session</th>
                        <th>tags</th>
                        <th>expected class</th>
                        <th>validation</th>
                        <th>processing</th>
                        <th>calibration</th>
                        <th>created_at</th>
                        <th>actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {takes.map((take) => {
                        const selected = selectionScope === "filtered" ? !excludedTakeIds.has(take.take_id) : selectedTakeIds.has(take.take_id);
                        const preview = takePreviewsById.get(take.take_id) ?? { url: null, placeholder: "TAKE", modality: null };
                        return (
                          <tr key={take.take_id} className={selectedTakeId === take.take_id ? "active" : ""} onClick={() => setSelectedTakeId(take.take_id)}>
                            <td><input type="checkbox" checked={selected} onClick={(event) => event.stopPropagation()} onChange={(event) => { if (selectionScope === "filtered") { setExcludedTakeIds((prev) => { const next = new Set(prev); if (event.target.checked) next.delete(take.take_id); else next.add(take.take_id); return next; }); } else { setSelectedTakeIds((prev) => { const next = new Set(prev); if (event.target.checked) next.add(take.take_id); else next.delete(take.take_id); return next; }); } }} /></td>
                            <td><a className="datasets-thumb-link" href={buildStudioTakeHref(take.take_id)} target="_blank" rel="noreferrer" title="Open in Studio" onClick={(event) => event.stopPropagation()}>{preview.modality && <span className="datasets-thumb-modality">{preview.modality}</span>}{preview.url ? <img className="datasets-thumb" src={preview.url} alt={take.friendly_name || take.take_id} loading="lazy" decoding="async" /> : <span className="datasets-thumb-placeholder">{preview.placeholder}</span>}</a></td>
                            <td><a className="datasets-take-link" href="#" onClick={(event) => { event.preventDefault(); event.stopPropagation(); setSelectedTakeId(take.take_id); }}>{take.friendly_name || take.take_id}</a></td>
                            <td>{take.dataset_name || take.dataset_id || "-"}</td>
                            <td>{take.experiment_session_id ? <a href="#" onClick={(event) => { event.preventDefault(); event.stopPropagation(); selectSession(take.experiment_session_id || ""); }}>{take.experiment_session_name || take.experiment_session_id}</a> : "-"}</td>
                            <td>{(take.tags ?? []).slice(0, 3).join(", ") || "-"}</td>
                            <td>{take.expected_class || "-"}</td>
                            <td>{take.validation_status || "unreviewed"}</td>
                            <td>{take.has_done ? "done" : take.has_ready ? "ready" : "pending"}</td>
                            <td>{take.calibration_id || "-"}</td>
                            <td>{take.created_at || "-"}</td>
                            <td><a href={`/takes/${encodeURIComponent(take.take_id)}`} title="Inspect pipeline outputs in Studio">Inspect</a></td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                )}
              </div>
              <div className="datasets-loadmore-row">
                <button type="button" disabled={!(activeMlSetScope ? mlSetMemberHasMore : hasMore) || isInitialLoading || isLoadingMore} onClick={() => void loadMoreTakes()}>
                  {isLoadingMore ? "Loading more..." : (activeMlSetScope ? mlSetMemberHasMore : hasMore) ? "Load more" : "All filtered takes loaded"}
                </button>
              </div>
            </section>
          )}

          {activeTab === "physical_objects" && (
            <section className="datasets-tab-panel">
              <div className="concept-panel compact">
                <h3>Physical Objects</h3>
                <small>Real-world semantic entities observed across one or more takes.</small>
                <div className="datasets-table-wrap">
                  <table className="datasets-table compact-table">
                    <thead>
                      <tr>
                        <th>Object</th>
                        <th>Class</th>
                        <th>Takes</th>
                        <th>Sessions</th>
                        <th>Dimensions</th>
                        <th>Confidence</th>
                      </tr>
                    </thead>
                    <tbody>
                      {physicalObjects.map((item) => (
                        <tr key={item.physical_object_id} className={selectedPhysicalObjectId === item.physical_object_id ? "active" : ""}>
                          <td><button type="button" className="link-button" onClick={() => setSelectedPhysicalObjectId(item.physical_object_id)}>{item.physical_object_id}</button></td>
                          <td>{item.normalized_class || "-"}</td>
                          <td>{item.observation_take_ids?.length ?? 0}</td>
                          <td>{item.source_session_ids?.length ?? 0}</td>
                          <td>{[item.d1_mm, item.d2_mm, item.d3_mm].filter(Boolean).join(" / ") || "-"}</td>
                          <td>{item.annotation_confidence || "-"}</td>
                        </tr>
                      ))}
                      {!physicalObjects.length && (
                        <tr><td colSpan={6}>No physical objects registered yet.</td></tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            </section>
          )}

          {activeTab === "objects" && (
            <section className="datasets-tab-panel">
              <div className="datasets-actions-row">
                <span className="status-chip">uncertain queue (placeholder)</span>
                <span className="status-chip">disagreement queue (placeholder)</span>
                <span className="status-chip">active learning queue (placeholder)</span>
                <span className="status-chip">outlier queue (placeholder)</span>
                <a href="/studio">Review segmentation</a>
              </div>
              {!detail && <div className="empty-state compact">Select a take in the Takes tab to open object review.</div>}
              {detail && objectRows.length === 0 && <div className="empty-state compact">No annotated objects yet.</div>}
              {detail && (
                <div className="datasets-table-wrap">
                  <table className="datasets-table">
                    <thead>
                      <tr>
                        <th>object</th>
                        <th>labels</th>
                        <th>confidence</th>
                        <th>validation</th>
                        <th>expected class</th>
                        <th>notes</th>
                        <th>preview</th>
                        <th />
                      </tr>
                    </thead>
                    <tbody>
                      {objectRows.map((row) => (
                        <tr key={row.objectId} className={selectedObjectId === row.objectId ? "active" : ""}>
                          <td><button type="button" onClick={() => setSelectedObjectId(row.objectId)}>{row.objectId}</button></td>
                          <td>{row.labels}</td>
                          <td>{Number.isFinite(row.confidence) ? row.confidence.toFixed(3) : "-"}</td>
                          <td>{row.validation}</td>
                          <td>{row.expectedClass}</td>
                          <td>{row.notes || "-"}</td>
                          <td>geometry + overlay</td>
                          <td><button type="button" disabled={busy} onClick={() => void assignObjectLabel(row.objectId)}>Edit labels</button></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </section>
          )}

          {activeTab === "labels" && (
            <section className="datasets-tab-panel">
              <div className="datasets-roadmap-list concept-panel compact">
                <h3>Labels</h3>
                <small>usage: {sortedClassCounts.length} active classes</small>
                <small>superclass mapping: placeholder</small>
                <small>aliases and merge/deprecate flows: placeholder</small>
                <small>hierarchy visualization: placeholder</small>
              </div>
            </section>
          )}

          {activeTab === "ml_sets" && (
            <section className="datasets-tab-panel">
              <div className="datasets-roadmap-list concept-panel compact">
                <h3>ML Sets</h3>
                <button type="button" onClick={() => setIngestionWizardOpen((prev) => !prev)}>{ingestionWizardOpen ? "Hide ingestion wizard" : "Open ingestion wizard"}</button>
                <small>Datasets own raw semantic organization.</small>
                <small>ML Sets own curated membership for training/evaluation.</small>
                <small>inclusion filters + validation requirements + allowed labels: placeholder</small>
                <small>balancing constraints + split ownership + classifier compatibility: placeholder</small>
                {ingestionWizardOpen && (
                  <MLSetIngestionWizard dataset={selectedDatasetSummary} sessions={sessions} onClose={() => setIngestionWizardOpen(false)} onMaterialized={(result) => { void handleWizardMaterialized(result); }} />
                )}
                <div className="datasets-table-wrap">
                  <table className="datasets-table">
                    <thead>
                      <tr>
                        <th>name</th>
                        <th>dataset</th>
                        <th>members</th>
                        <th>membership mode</th>
                        <th>created_at</th>
                        <th>readiness</th>
                        <th>actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {mlSets.map((item) => (
                        <tr key={item.id} className={selectedMlSetId === item.id ? "active" : ""}>
                          <td><button type="button" className="link-button" onClick={() => setSelectedMlSetId(item.id)}>{item.name || item.id}</button></td>
                          <td>{item.dataset_id || "-"}</td>
                          <td>{String(item.member_count ?? item.membership_count ?? 0)}</td>
                          <td>{item.membership_mode || "ids"}</td>
                          <td>{item.created_at || "-"}</td>
                          <td>{`${item.split_status || "unassigned"} · ${item.physical_object_count ?? 0} objects · ${item.default_trainable_count ?? 0} trainable · ${item.review_required_count ?? 0} review`}</td>
                          <td>
                            <div className="datasets-row-actions">
                              <button type="button" className="link-button" onClick={() => applyMlSetScope(item.id)}>Review members</button>
                              <button type="button" className="link-button" onClick={() => openFeatureAnalyticsForSlice({ mlSetId: item.id })}>Analyze features</button>
                              <button type="button" className="link-button" onClick={() => { applyMlSetScope(item.id); setMlSetSplitFilter("unassigned"); }}>Assign split</button>
                              <button type="button" className="link-button" onClick={() => { window.location.href = `/api/ml-sets/${encodeURIComponent(item.id)}/export?kind=manifest&dataset_id=${encodeURIComponent(item.dataset_id)}`; }}>Export</button>
                            </div>
                          </td>
                        </tr>
                      ))}
                      {!mlSets.length && (
                        <tr>
                          <td colSpan={7}>No ML sets yet.</td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            </section>
          )}

          {activeTab === "splits" && (
            <section className="datasets-tab-panel">
              <div className="datasets-roadmap-list concept-panel compact">
                <h3>Splits</h3>
                <small>distribution: {splitBuckets.map((bucket) => `${bucket}:${stats.splits[bucket]}`).join(" · ")}</small>
                <small>session/temporal leakage diagnostics: placeholder</small>
                <small>cross-site and class imbalance diagnostics: placeholder</small>
              </div>
            </section>
          )}
        </section>

        <aside className={`concept-panel datasets-right ${rightPanelCollapsed ? "is-collapsed" : ""}`}>
          <div className="datasets-left-title-row">
            <h3>Inspector</h3>
            <button
              type="button"
              className="panel-collapse-toggle"
              onClick={() => setRightPanelCollapsed((current) => !current)}
              aria-expanded={!rightPanelCollapsed}
              aria-label={rightPanelCollapsed ? "Expand inspector" : "Collapse inspector"}
              title={rightPanelCollapsed ? "Expand inspector" : "Collapse inspector"}
            >
              {rightPanelCollapsed ? "‹" : "›"}
            </button>
          </div>
          {!rightPanelCollapsed && (
          <>
          {activeMlSetScope && selectedObjectGroup ? (
            <div className="datasets-inspector-preview-grid">
              {selectedObjectGroup.members.slice(0, 4).map((member) => {
                const preview = takePreviewsById.get(member.take_id) ?? { url: null, placeholder: "TAKE", modality: null };
                return preview.url ? <img key={member.take_id} className="datasets-inspector-thumb multi" src={preview.url} alt={member.take_id} loading="lazy" decoding="async" /> : <span key={member.take_id} className="datasets-thumb-placeholder inspector">{preview.placeholder}</span>;
              })}
            </div>
          ) : (
            <div className="datasets-inspector-preview">
              {selectedTakeSummary && selectedInspectorPreview.url ? <img className="datasets-inspector-thumb" src={selectedInspectorPreview.url} alt={selectedTakeSummary.friendly_name || selectedTakeSummary.take_id} loading="lazy" decoding="async" /> : <span className="datasets-thumb-placeholder inspector">{selectedInspectorPreview.placeholder}</span>}
            </div>
          )}
          {!activeMlSetScope || !selectedObjectGroup ? (
            <button
              type="button"
              className="datasets-inspector-open-studio"
              disabled={!selectedTakeId}
              onClick={() => selectedTakeId && window.open(buildStudioTakeHref(selectedTakeId), "_blank", "noopener,noreferrer")}
            >
              Open in Studio ↗
            </button>
          ) : null}
          <section className="datasets-inspector-section">
            <h4>Preview</h4>
            <small>
              {activeMlSetScope && selectedObjectGroup
                ? `${selectedObjectGroup.physicalObjectId || "Unassigned object group"} · ${selectedObjectGroup.takeCount} takes`
                : selectedTakeSummary?.friendly_name ?? selectedTakeSummary?.take_id ?? "Select a take or object group to review metadata, labels, membership, warnings, and actions."}
            </small>
          </section>
          <div className="datasets-inspector-grid">
            <span>Dataset</span>
            <strong>{selectedDatasetSummary?.name ?? "All datasets"}</strong>
            <span>Session</span>
            <strong>{selectedScopedMember?.session_name || selectedScopedMember?.session_id || selectedSessionSummary?.name || "All sessions"}</strong>
            <span>Take</span>
            <strong>{detail?.take_id ?? selectedTakeId ?? "-"}</strong>
            <span>Object</span>
            <strong>{selectedObjectGroup?.physicalObjectId || selectedScopedMember?.physical_object_id || selectedObject?.objectId || "-"}</strong>
            <span>ML set</span>
            <strong>{activeMlSetScope?.mlSetId || (activeTab === "ml_sets" ? "selection context" : "-")}</strong>
            <span>Split</span>
            <strong>{selectedScopedMember?.split || readSplitFromSummary(selectedTakeSummary) || selectedSplit}</strong>
          </div>

          <div className="datasets-inspector-box">
            <strong>Label provenance</strong>
            {activeMlSetScope && selectedObjectGroup ? (
              <>
                <small>normalized class: {selectedObjectGroup.normalizedClass}</small>
                <small>superclass: {selectedObjectGroup.superclass}</small>
                <small>raw labels present: {selectedObjectGroup.rawLabels.join(", ") || "-"}</small>
                <small>sessions: {selectedObjectGroup.sessions.join(", ") || "-"}</small>
                <small>review warnings: {selectedObjectGroup.warningCodes.join(", ") || "-"}</small>
              </>
            ) : (
              <>
                <small>raw label: {selectedScopedMember?.label_provenance.raw_label || "-"}</small>
                <small>normalized class: {selectedScopedMember?.label_provenance.normalized_class || selectedTakeSummary?.expected_class || "-"}</small>
                <small>superclass: {selectedScopedMember?.label_provenance.superclass || detailTakeSuperclasses[0] || "-"}</small>
                <small>normalization version: {selectedScopedMember?.label_provenance.normalization_version || String(detailTakeMetadata["normalization_version"] || "-")}</small>
                <small>source row: {selectedScopedMember?.label_provenance.source_row || "-"}</small>
              </>
            )}
          </div>

          <div className="datasets-inspector-box">
            <strong>Object group membership</strong>
            <small>physical_object_id: {selectedObjectGroup?.physicalObjectId || selectedScopedMember?.physical_object_id || "-"}</small>
            <small>take count: {selectedObjectGroup?.takeCount ?? (selectedScopedMember ? 1 : 0)}</small>
            <small>warnings: {(selectedObjectGroup?.warningCodes || selectedScopedMember?.warnings || []).join(", ") || "-"}</small>
            <small>feature summary: {selectedScopedMember?.feature_ready ? `${selectedScopedMember.feature_summary.processed_class_label || "processed"} · ${selectedScopedMember.feature_summary.processed_superclass || "-"}` : "not available"}</small>
          </div>

          <div className="datasets-inspector-box">
            <strong>Quick actions</strong>
            <div className="datasets-actions-row">
              <button type="button" disabled={!selectedTakeId} onClick={() => selectedTakeId && void api.updateTakeMetadata(selectedTakeId, { validation_status: "valid" }).then(refreshTakes)}>Validate</button>
              <button type="button" disabled={!selectedTakeId} onClick={() => selectedTakeId && void api.updateTakeMetadata(selectedTakeId, { validation_status: "needs_review" }).then(refreshTakes)}>Mark review</button>
              <button type="button" disabled={!activeMlSetScope || (!selectedTakeId && !selectedObjectGroup)} onClick={() => void excludeSelectionFromActiveMlSet(selectedObjectGroup ? selectedObjectGroup.members.map((member) => member.take_id) : selectedTakeId ? [selectedTakeId] : [])}>Exclude from ML set</button>
              <button type="button" disabled={!selectedTakeId} onClick={() => {
                const value = window.prompt("Correct normalized class", selectedScopedMember?.normalized_class || selectedTakeSummary?.expected_class || "");
                if (value && selectedTakeId) void updateActiveMlSetSelection({ expected_class: value.trim() }, [selectedTakeId]);
              }}>Correct label</button>
            </div>
            <div className="datasets-actions-row">
              <button type="button" disabled={!activeMlSetScope || (!selectedTakeId && !selectedObjectGroup)} onClick={() => {
                const value = window.prompt("Assign ML-set split", selectedScopedMember?.split || "validation");
                if (value) void updateActiveMlSetSelection({ split: value.trim() }, selectedObjectGroup ? selectedObjectGroup.members.map((member) => member.take_id) : selectedTakeId ? [selectedTakeId] : []);
              }}>Assign split</button>
              <button type="button" disabled={!selectedTakeId} onClick={() => {
                if (!selectedTakeId) return;
                window.open(buildStudioTakeHref(selectedTakeId), "_blank", "noopener,noreferrer");
              }}>Open in Studio</button>
              <button type="button" disabled={!selectedTakeId && !selectedObjectGroup} onClick={() => openFeatureAnalyticsForSlice({ normalizedClass: selectedScopedMember?.normalized_class || selectedObjectGroup?.normalizedClass || "", superclass: selectedScopedMember?.superclass || selectedObjectGroup?.superclass || "", physicalObjectId: selectedScopedMember?.physical_object_id || selectedObjectGroup?.physicalObjectId || "" })}>Open in Feature Analytics</button>
            </div>
          </div>

          <div className="datasets-inspector-box">
            <strong>Quick Review Mode</strong>
            <small>next unresolved / unreviewed / missing-label take</small>
            <div className="datasets-actions-row">
              <button type="button" onClick={() => {
                const next = activeMlSetScope ? activeReviewRows.find((row) => String(row.validation_status ?? "unreviewed") !== "valid") : takes.find((take) => String(take.validation_status ?? "unreviewed") !== "valid");
                if (next) setSelectedTakeId(next.take_id);
              }}>Next unresolved</button>
              <button type="button" onClick={() => {
                const next = activeMlSetScope ? activeReviewRows.find((row) => String(row.validation_status ?? "unreviewed") === "unreviewed") : takes.find((take) => String(take.validation_status ?? "unreviewed") === "unreviewed");
                if (next) setSelectedTakeId(next.take_id);
              }}>Next unreviewed</button>
              <button type="button" onClick={() => {
                const next = activeMlSetScope ? activeReviewRows.find((row) => !row.normalized_class) : takes.find((take) => !take.expected_class);
                if (next) setSelectedTakeId(next.take_id);
              }}>Next missing label</button>
            </div>
            <small>Keyboard extensibility: j/k/x/v/r/t/s (foundation only)</small>
          </div>

          <div className="datasets-roadmap-list concept-panel compact">
            <h3>Future Curation Architecture</h3>
            <small>similarity search + embedding neighborhoods (placeholder)</small>
            <small>anomaly/disagreement clustering (placeholder)</small>
            <small>feature-space semantic exploration with Feature Analytics (placeholder)</small>
          </div>
          </>
          )}
        </aside>
      </section>

      {bulkModal && (
        <div className="datasets-modal-backdrop" role="presentation" onClick={() => setBulkModal(null)}>
          <section className="datasets-modal" role="dialog" aria-modal="true" onClick={(event) => event.stopPropagation()}>
            <header className="datasets-modal-head">
              <h3>
                {bulkModal === "move_dataset" && "Move to Dataset"}
                {bulkModal === "move_session" && "Move to Session"}
                {bulkModal === "ml_set" && (modalMlSetMode === "add" ? "Add to ML Set" : "Remove from ML Set")}
                {bulkModal === "split" && "Assign Split"}
                {bulkModal === "tags" && "Tag Management"}
                {bulkModal === "validation" && "Validation Update"}
                {bulkModal === "expected_class" && "Set Expected Class"}
              </h3>
              <small>
                {selectionScope === "filtered"
                  ? `All ${activeFilteredCount} filtered takes selected${excludedTakeIds.size ? ` · ${excludedTakeIds.size} excluded` : ""}`
                  : `${selectedTakeIds.size} visible selected`}
              </small>
            </header>

            {bulkModal === "move_dataset" && (
              <div className="datasets-modal-body">
                <label className="field-label">
                  Dataset
                  <select value={modalDatasetId} onChange={(e) => { setModalDatasetId(e.target.value); setModalSessionId(""); }}>
                    <option value="">Select dataset</option>
                    {datasets.map((dataset) => <option key={dataset.id} value={dataset.id}>{dataset.name}</option>)}
                  </select>
                </label>
                <label className="field-label">
                  Session (optional)
                  <select value={modalSessionId} onChange={(e) => setModalSessionId(e.target.value)}>
                    <option value="">No session change</option>
                    {modalSessions.map((session) => <option key={session.id} value={session.id}>{session.name}</option>)}
                  </select>
                </label>
                <div className="datasets-modal-inline-actions">
                  <button type="button" disabled title="Placeholder: create dataset inline">Create new dataset (planned)</button>
                  <button type="button" disabled title="Placeholder: create session inline">Create new session (planned)</button>
                </div>
                <small>{modalCounting ? "Calculating affected count..." : `This will move ${modalAffectedCount ?? 0} takes.`}</small>
              </div>
            )}

            {bulkModal === "move_session" && (
              <div className="datasets-modal-body">
                <label className="field-label">
                  Session
                  <select value={modalSessionId} onChange={(e) => { setModalSessionId(e.target.value); setModalCreateSession(false); }}>
                    <option value="">Select session</option>
                    {sessions.map((session) => <option key={session.id} value={session.id}>{session.name}</option>)}
                  </select>
                </label>
                <label className="field-inline checkbox-inline">
                  <input type="checkbox" checked={modalCreateSession} onChange={(e) => setModalCreateSession(e.target.checked)} />
                  Create new session
                </label>
                {modalCreateSession && (
                  <>
                    <label className="field-label">
                      New session name
                      <input value={modalNewSessionName} onChange={(e) => setModalNewSessionName(e.target.value)} placeholder="Session 2026-05-29" />
                    </label>
                    <label className="field-label">
                      Notes (optional)
                      <input value={modalSessionNotes} onChange={(e) => setModalSessionNotes(e.target.value)} placeholder="day-based curation batch" />
                    </label>
                  </>
                )}
                <small>
                  {modalCounting
                    ? "Calculating affected count..."
                    : `This will move ${modalAffectedCount ?? 0} ${selectionScope === "filtered" ? "filtered" : "selected"} takes.`}
                </small>
              </div>
            )}

            {bulkModal === "ml_set" && (
              <div className="datasets-modal-body">
                <label className="field-inline checkbox-inline">
                  <input type="checkbox" checked={modalCreateMlSet} onChange={(e) => setModalCreateMlSet(e.target.checked)} />
                  Create new ML Set
                </label>
                {modalCreateMlSet && (
                  <>
                    <label className="field-label">
                      ML set name
                      <input value={modalNewMlSetName} onChange={(e) => setModalNewMlSetName(e.target.value)} placeholder="ml_set_2026_05_29" />
                    </label>
                    <label className="field-label">
                      Description/notes (optional)
                      <input value={modalNewMlSetDescription} onChange={(e) => setModalNewMlSetDescription(e.target.value)} placeholder="day-based curated subset" />
                    </label>
                    <label className="field-label">
                      Initial split strategy (placeholder)
                      <input value={modalNewMlSetSplitStrategy} onChange={(e) => setModalNewMlSetSplitStrategy(e.target.value)} placeholder="stratified_by_session" />
                    </label>
                    <label className="field-label">
                      Validation requirements (placeholder)
                      <input value={modalNewMlSetValidationReq} onChange={(e) => setModalNewMlSetValidationReq(e.target.value)} placeholder="validated, benchmark_approved" />
                    </label>
                  </>
                )}
                <label className="field-label">
                  Membership action
                  <select value={modalMlSetMode} onChange={(e) => setModalMlSetMode(e.target.value as "add" | "remove")}>
                    <option value="add">Add to ML set</option>
                    <option value="remove">Remove from ML set</option>
                  </select>
                </label>
                {!modalCreateMlSet && (
                  <label className="field-label">
                    ML set
                    <select value={modalMlSetId} onChange={(e) => setModalMlSetId(e.target.value)}>
                      <option value="">Select ML set</option>
                      {mlSets.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
                    </select>
                  </label>
                )}
                <small>ML set membership is additive and does not move raw takes.</small>
                <small>
                  {modalCounting
                    ? "Calculating affected count..."
                    : modalCreateMlSet
                      ? `This will create ML set ${modalNewMlSetName || "(unnamed)"} and add ${modalAffectedCount ?? 0} ${selectionScope === "filtered" ? "filtered" : "selected"} takes.`
                      : `This will ${modalMlSetMode} ML set membership for ${modalAffectedCount ?? 0} takes.`}
                </small>
              </div>
            )}

            {bulkModal === "split" && (
              <div className="datasets-modal-body">
                <label className="field-label">
                  Split
                  <select value={modalSplit} onChange={(e) => setModalSplit(e.target.value as SplitBucket)}>
                    {splitBuckets.map((split) => <option key={split} value={split}>{split}</option>)}
                  </select>
                </label>
                <small>{modalCounting ? "Calculating affected count..." : `This will assign split to ${modalAffectedCount ?? 0} takes.`}</small>
              </div>
            )}

            {bulkModal === "tags" && (
              <div className="datasets-modal-body">
                <label className="field-label">
                  Mode
                  <select value={modalTagMode} onChange={(e) => setModalTagMode(e.target.value as "add" | "remove" | "replace")}>
                    <option value="add">Add tags</option>
                    <option value="remove">Remove tags</option>
                    <option value="replace">Replace tags</option>
                  </select>
                </label>
                <label className="field-label">
                  Tags (comma separated)
                  <input value={modalTagValue} onChange={(e) => setModalTagValue(e.target.value)} placeholder="wet_surface, benchmark" />
                </label>
                <small>{modalCounting ? "Calculating affected count..." : `This will ${modalTagMode} tags for ${modalAffectedCount ?? 0} takes.`}</small>
              </div>
            )}

            {bulkModal === "validation" && (
              <div className="datasets-modal-body">
                <label className="field-label">
                  Validation state
                  <select value={modalValidationStatus} onChange={(e) => setModalValidationStatus(e.target.value)}>
                    <option value="valid">validated</option>
                    <option value="needs_review">needs_review</option>
                    <option value="invalid">rejected</option>
                    <option value="golden_sample">golden_sample</option>
                    <option value="benchmark_approved">benchmark_approved</option>
                  </select>
                </label>
                <small>
                  {modalCounting
                    ? "Calculating affected count..."
                    : `This will mark ${modalAffectedCount ?? 0} takes as ${modalValidationStatus}.`}
                </small>
              </div>
            )}

            {bulkModal === "expected_class" && (
              <div className="datasets-modal-body">
                <label className="field-label">
                  Expected class
                  <input value={modalExpectedClassValue} onChange={(e) => setModalExpectedClassValue(e.target.value)} placeholder="BALL_GOOD" />
                </label>
                <small>
                  {modalCounting
                    ? "Calculating affected count..."
                    : `This will set expected class for ${modalAffectedCount ?? 0} takes.`}
                </small>
              </div>
            )}

            {!canApplyFilteredScope && (
              <div className="datasets-inline-state">Filter-wide bulk updates require backend support.</div>
            )}

            <footer className="datasets-modal-foot">
              <button type="button" onClick={() => setBulkModal(null)}>Cancel</button>
              <button
                type="button"
                disabled={
                  busy ||
                  selectedCount === 0 ||
                  (!canApplyFilteredScope ||
                    ((bulkModal === "move_session") && (!modalSessionId && !modalCreateSession || (modalCreateSession && !modalNewSessionName.trim()) || modalCounting)) ||
                    ((bulkModal === "move_dataset") && !modalDatasetId.trim()) ||
                    ((bulkModal === "ml_set") && (modalCreateMlSet ? !modalNewMlSetName.trim() : !modalMlSetId.trim())) ||
                    ((bulkModal === "tags") && !modalTagValue.trim()) ||
                    ((bulkModal === "expected_class") && !modalExpectedClassValue.trim()) ||
                    modalCounting)
                }
                onClick={() => void submitBulkModal()}
              >
                {bulkModal === "move_session" && modalAffectedCount !== null
                    ? `Move ${modalAffectedCount} takes`
                  : bulkModal === "move_dataset" && modalAffectedCount !== null
                    ? `Move ${modalAffectedCount} takes`
                    : bulkModal === "ml_set" && modalAffectedCount !== null
                      ? (modalCreateMlSet
                        ? `Create ML Set and add ${modalAffectedCount} takes`
                        : `${modalMlSetMode === "add" ? "Add" : "Remove"} ${modalAffectedCount} takes`)
                      : bulkModal === "split" && modalAffectedCount !== null
                        ? `Assign split to ${modalAffectedCount} takes`
                        : bulkModal === "tags" && modalAffectedCount !== null
                          ? `Apply tags to ${modalAffectedCount} takes`
                      : bulkModal === "validation" && modalAffectedCount !== null
                        ? `Mark ${modalAffectedCount} takes`
                        : bulkModal === "expected_class" && modalAffectedCount !== null
                          ? `Set class for ${modalAffectedCount} takes`
                          : bulkModal === "move_dataset" || bulkModal === "move_session"
                    ? "Move"
                    : bulkModal === "ml_set"
                      ? "Add"
                    : bulkModal === "split"
                      ? "Assign"
                        : "Apply"}
              </button>
            </footer>
          </section>
        </div>
      )}

      <DatasetSessionDrawer
        open={Boolean(sessionDrawerId && drawerSessionSummary)}
        session={drawerSessionSummary}
        dataset={selectedDatasetSummary}
        onClose={() => setSessionDrawerId("")}
        onSaved={async () => {
          if (!selectedDataset) return;
          const refreshed = await api.datasetSessions(selectedDataset);
          setSessions(refreshed);
        }}
      />
      <MLSetDetailDrawer
        open={Boolean(selectedMlSetId && selectedMlSetSummary)}
        mlSet={selectedMlSetSummary}
        dataset={selectedDatasetSummary}
        onClose={() => setSelectedMlSetId("")}
        onSelectSession={(sessionId) => {
          openSessionDrawer(sessionId);
        }}
        onReviewClass={(value) => {
          if (!selectedMlSetSummary) return;
          setSelectedMlSetId("");
          applyMlSetScope(selectedMlSetSummary.id, { normalizedClass: value });
        }}
        onReviewSuperclass={(value) => {
          if (!selectedMlSetSummary) return;
          setSelectedMlSetId("");
          applyMlSetScope(selectedMlSetSummary.id, { superclass: value });
        }}
      />
      <PhysicalObjectDetailDrawer
        open={Boolean(selectedPhysicalObjectId && selectedPhysicalObjectSummary)}
        physicalObject={selectedPhysicalObjectSummary}
        dataset={selectedDatasetSummary}
        onClose={() => setSelectedPhysicalObjectId("")}
        onSelectSession={(sessionId) => {
          openSessionDrawer(sessionId);
        }}
      />

      {successMessage && <div className="warning-line">{successMessage}</div>}
      {error && <div className="warning-line active">{error}</div>}
    </main>
  );
}

function readSplitFromTake(take: TakeSummary): SplitBucket | null {
  const meta = ((take as unknown as { metadata?: Record<string, unknown> }).metadata ?? {}) as Record<string, unknown>;
  const split = String(meta.split ?? "");
  return splitBuckets.includes(split as SplitBucket) ? split as SplitBucket : null;
}

function readSplitFromSummary(take: TakeSummary | null): SplitBucket | null {
  if (!take) return null;
  return readSplitFromTake(take);
}

function firstObjectId(detail: TakeDetail): string {
  const first = (detail.object_annotations ?? [])[0] as ObjectAnnotation | undefined;
  if (!first) return "";
  return String(first.candidate_id ?? first.id ?? "");
}

function recordOrNull(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" ? value as Record<string, unknown> : null;
}

function resolvedTakePreviewUrl(takeId: string, value: unknown): string | null {
  if (typeof value !== "string" || !value.trim()) return null;
  return /^https?:\/\//i.test(value) ? value : fileUrl(takeId, value);
}

function takePreviewUrlForDetail(detail: TakeDetail): string | null {
  const metadataFiles = recordOrNull(detail.metadata)?.files;
  const sourceFiles = recordOrNull(metadataFiles);
  const resultFiles = recordOrNull((detail.result as unknown as { files?: Record<string, unknown> | null } | null)?.files);
  const metadata = recordOrNull(detail.metadata);
  const takeMetadata = recordOrNull(detail.take_metadata);
  const previewCandidate = [
    sourceFiles?.heightmap_preview,
    resultFiles?.input_preview,
    detail.assets?.heightmap?.heightmap_preview,
    detail.assets?.reflectance?.reflectance,
    resultFiles?.reflectance,
    resultFiles?.overlay,
    metadata?.thumbnail_url,
    takeMetadata?.thumbnail_url,
    metadata?.preview_url,
    takeMetadata?.preview_url,
  ].find((value) => typeof value === "string" && value.trim());
  return resolvedTakePreviewUrl(detail.take_id, previewCandidate);
}

function takePreviewForSummary(take: TakeSummary): { url: string | null; placeholder: string; modality: string | null } {
  const modality = modalityBadgeForTake(take);
  if (take.thumbnail_path) {
    return { url: fileUrl(take.take_id, take.thumbnail_path), placeholder: "IMG", modality };
  }
  const assets = take.assets ?? {};
  const preferred = [
    findAssetPath(assets, ["rgb", "reflectance", "preview", "thumbnail", "image"]),
    findAssetPath(assets, ["heightmap", "height", "preview", "thumbnail", "image"]),
    findAssetPath(assets, ["overlay", "preview", "image"]),
  ].find(Boolean);
  if (preferred) {
    return { url: fileUrl(take.take_id, preferred), placeholder: "IMG", modality };
  }
  const modalities = take.modalities ?? [];
  if (modalities.includes("rgb") || modalities.includes("reflectance") || modalities.includes("rgb_video") || modalities.includes("laser_rgb")) {
    return { url: null, placeholder: "RGB", modality };
  }
  if (modalities.includes("heightmap")) return { url: null, placeholder: "HMP", modality };
  if (modalities.includes("point_cloud")) return { url: null, placeholder: "PCD", modality };
  return { url: null, placeholder: "TAKE", modality };
}

function findAssetPath(assets: Record<string, Record<string, string>>, keywords: string[]): string | null {
  for (const values of Object.values(assets)) {
    for (const [key, value] of Object.entries(values)) {
      const target = `${key} ${value}`.toLowerCase();
      if (keywords.some((keyword) => target.includes(keyword))) return value;
    }
  }
  return null;
}

function modalityBadgeForTake(take: TakeSummary): string | null {
  const modalities = take.modalities ?? [];
  if (modalities.includes("heightmap")) return "2.5D";
  if (modalities.includes("rgb") || modalities.includes("reflectance") || modalities.includes("rgb_video") || modalities.includes("laser_rgb")) return "RGB";
  if (modalities.includes("point_cloud")) return "PCD";
  return null;
}
