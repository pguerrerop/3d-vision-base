import { useEffect, useMemo, useState } from "react";
import { api, fileUrl, type DatasetMlSet, type DatasetSessionSummary, type DatasetSummary, type TakeDetail, type TakeSummary } from "../api/client";
import DatasetSessionDrawer from "../components/datasets/DatasetSessionDrawer";
import MLSetIngestionWizard from "../components/ingestion/MLSetIngestionWizard";

type SplitBucket = "training" | "validation" | "test" | "benchmark" | "production-shadow" | "cross-site";
type DatasetTab = "overview" | "takes" | "objects" | "labels" | "ml_sets" | "splits";

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

function countFromValidation(summary: PagedSummaryCounts | null, key: string, fallback = 0): number {
  return Number(summary?.validation?.[key] ?? fallback);
}

export default function DatasetsPage() {
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [sessions, setSessions] = useState<DatasetSessionSummary[]>([]);
  const [mlSets, setMlSets] = useState<DatasetMlSet[]>([]);
  const [modalDatasetSessions, setModalDatasetSessions] = useState<DatasetSessionSummary[]>([]);
  const [pagedTakes, setPagedTakes] = useState<TakeSummary[]>([]);
  const [offset, setOffset] = useState(0);
  const [hasMore, setHasMore] = useState(false);
  const [selectedDataset, setSelectedDataset] = useState("");
  const [selectedSession, setSelectedSession] = useState("");
  const [sessionDrawerId, setSessionDrawerId] = useState("");
  const [selectedTakeId, setSelectedTakeId] = useState("");
  const [selectedObjectId, setSelectedObjectId] = useState("");
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
      return;
    }
    void api.datasetSessions(selectedDataset).then((items) => {
      setSessions(items);
      if (selectedSession && !items.find((s) => s.id === selectedSession)) setSelectedSession("");
    }).catch((err: unknown) => setError(err instanceof Error ? err.message : "Failed to load sessions"));
  }, [selectedDataset, selectedSession]);

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
    setSelectedTakeIds(new Set());
    setExcludedTakeIds(new Set());
    setSelectionScope("visible");
    setSelectedTakeId("");
    setSelectedObjectId("");
    setDetail(null);
  }, [selectedDataset, selectedSession, validationFilter, search, tagFilter, fromDate, toDate, datePreset, selectedSplit, expectedClassFilter, calibrationOnly, governanceFlags]);

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
  const selectedCount = selectionScope === "visible" ? selectedTakeIds.size : Math.max(0, filteredCount - excludedTakeIds.size);

  useEffect(() => {
    if (selectedTakeId && !takes.find((t) => t.take_id === selectedTakeId)) {
      setSelectedTakeId("");
      setDetail(null);
      setSelectedObjectId("");
    }
  }, [selectedTakeId, takes]);

  useEffect(() => {
    const visible = new Set(takes.map((take) => take.take_id));
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
  }, [takes, selectionScope]);

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
      classes: new Map<string, number>(),
      tags: new Map<string, number>(),
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
      const classLabel = take.expected_class || take.normalized_class || "unassigned";
      next.classes.set(classLabel, (next.classes.get(classLabel) ?? 0) + 1);
      const sessionKey = take.experiment_session_name || take.experiment_session_id || "unassigned";
      next.sessionCounts.set(sessionKey, (next.sessionCounts.get(sessionKey) ?? 0) + 1);
      (take.tags ?? []).forEach((tag) => next.tags.set(tag, (next.tags.get(tag) ?? 0) + 1));
    });
    next.recent = [...takes].sort((a, b) => String(b.created_at ?? "").localeCompare(String(a.created_at ?? ""))).slice(0, 6);
    return next;
  }, [takes]);

  const sortedClassCounts = useMemo(() => [...stats.classes.entries()].sort((a, b) => b[1] - a[1]).slice(0, 8), [stats.classes]);
  const sortedSessionCounts = useMemo(() => [...stats.sessionCounts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 8), [stats.sessionCounts]);
  const sortedTags = useMemo(() => [...stats.tags.entries()].sort((a, b) => b[1] - a[1]).slice(0, 12), [stats.tags]);
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
  const selectedTakeSummary = useMemo(() => takes.find((take) => take.take_id === selectedTakeId) ?? null, [takes, selectedTakeId]);
  const takePreviewsById = useMemo(() => {
    const next = new Map<string, { url: string | null; placeholder: string; modality: string | null }>();
    takes.forEach((take) => {
      next.set(take.take_id, takePreviewForSummary(take));
    });
    return next;
  }, [takes]);
  const selectedTakePreview = useMemo(
    () => (selectedTakeSummary ? (takePreviewsById.get(selectedTakeSummary.take_id) ?? { url: null, placeholder: "-", modality: null }) : { url: null, placeholder: "-", modality: null }),
    [selectedTakeSummary, takePreviewsById]
  );
  const keyboardCommands = useMemo(
    () => ({ j: "next_take", k: "prev_take", x: "toggle_select", v: "validate", r: "reject", t: "tag", s: "split" }),
    []
  );

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
      const page = await api.pagedTakes(params);
      if (DEBUG_DATASETS_TAKES) {
        console.debug("[datasets:takes] response", {
          count: page.items.length,
          has_more: page.has_more,
          next_offset: page.next_offset,
        });
      }
      setPagedTakes((prev) => (append ? [...prev, ...page.items] : page.items));
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
  }, [selectedDataset, selectedSession, validationFilter, search, tagFilter, fromDate, toDate, selectedSplit, expectedClassFilter, calibrationOnly, governanceFlags, selectedDatasetSummary?.take_count]);

  async function refreshTakes() {
    await loadTakePage(0, false);
  }

  async function loadMoreTakes() {
    if (!hasMore || isLoadingMore || isInitialLoading) return;
    await loadTakePage(offset, true);
  }

  function buildActiveBulkFilters() {
    return {
      dataset_id: selectedDataset || undefined,
      session_id: selectedSession || undefined,
      validation_status: validationFilter === "all" ? undefined : validationFilter,
      search: search.trim() || undefined,
      tag: tagFilter.trim() || undefined,
      created_from: fromDate || undefined,
      created_to: toDate || undefined,
      split: selectedSplit === "all" ? undefined : selectedSplit,
      expected_class: expectedClassFilter.trim() || undefined,
      calibration_linkage_only: calibrationOnly || undefined,
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
    setSelectedSession("");
    const refreshedSessions = selectedDataset ? await api.datasetSessions(selectedDataset) : [];
    setSessions(refreshedSessions);
    await refreshTakes();
    setBulkModal(null);
  }

  function applyCollection(collection: SavedCollection) {
    setSelectedDataset(collection.filters.dataset);
    setSelectedSession(collection.filters.session);
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
    setSelectedTakeIds(new Set(takes.map((take) => take.take_id)));
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

  return (
    <main className="datasets-page datasets-command-center">
      <section className="datasets-headline">
        <h2>Datasets</h2>
        <p>Organize captures, labels, object reviews, ML sets, and splits.</p>
      </section>

      <section className="datasets-grid">
        <aside className="concept-panel datasets-left">
          <div className="datasets-left-title-row">
            <h3>Dataset Explorer</h3>
            <small>
              {hasBaseFilters
                ? `${filteredCount} filtered / ${datasetTotalCount} total`
                : `${datasetTotalCount} takes`}
            </small>
          </div>
          <details className="datasets-section" open>
            <summary>Dataset hierarchy</summary>
            <div className="datasets-hierarchy-block">
            <div className="datasets-hierarchy-header">
              <strong>{selectedDatasetSummary?.name ?? "No dataset selected"}</strong>
              <button type="button" onClick={() => setShowSessionTree((prev) => !prev)}>{showSessionTree ? "Hide" : "Show"}</button>
            </div>
            <small>{sessions.length} sessions · validation {filteredCount ? Math.round((validatedCount / filteredCount) * 100) : 0}%</small>
            {showSessionTree && (
              <div className="datasets-session-tree">
                {sessions.map((session) => {
                  const sessionTakes = takes.filter((take) => (take.experiment_session_id ?? "") === session.id);
                  const sessionValid = sessionTakes.filter((take) => take.validation_status === "valid").length;
                  return (
                    <button
                      key={session.id}
                      type="button"
                      className={`datasets-session-node ${selectedSession === session.id ? "active" : ""}`}
                      onClick={() => {
                        setSelectedSession(session.id);
                        setSessionDrawerId(session.id);
                      }}
                    >
                      <span>{session.name}</span>
                      <small>{sessionTakes.length} takes · {sessionTakes.length ? Math.round((sessionValid / sessionTakes.length) * 100) : 0}% valid</small>
                    </button>
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
            <select value={selectedDataset} onChange={(e) => { setSelectedDataset(e.target.value); setSelectedSession(""); }}>
              <option value="">All datasets</option>
              {datasets.map((dataset) => <option key={dataset.id} value={dataset.id}>{dataset.name}</option>)}
            </select>
          </label>
          <label className="field-label">
            Session
            <select value={selectedSession} onChange={(e) => setSelectedSession(e.target.value)}>
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
            <input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="take id or text" />
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
        </aside>

        <section className="datasets-center">
          <nav className="datasets-tabs" aria-label="Datasets semantic workspace">
            {[
              ["overview", "Overview"],
              ["takes", "Takes"],
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
                  { id: "unreviewed", label: "Unreviewed", count: unreviewedCount, apply: () => setValidationFilter("unreviewed") },
                  { id: "missing_labels", label: "Missing labels", count: missingLabelsCount, apply: () => setGovernanceFlags(new Set(["missing_expected_class"])) },
                  { id: "calibration", label: "Calibration review", count: missingCalibrationCount, apply: () => setGovernanceFlags(new Set(["missing_calibration"])) },
                  { id: "benchmark", label: "Benchmark approval", count: benchmarkApprovedCount, apply: () => setValidationFilter("benchmark_approved") },
                  { id: "failed", label: "Failed processing", count: processingFailedCount, apply: () => setGovernanceFlags(new Set(["processing_failed"])) },
                ].map((queue) => (
                  <button key={queue.id} type="button" className="datasets-queue-card" onClick={queue.apply}>
                    <strong>{queue.label}</strong>
                    <small>{queue.count}</small>
                  </button>
                ))}
              </div>
              <div className="datasets-toolbar" role="toolbar" aria-label="Takes selection and bulk actions">
                <span className="datasets-toolbar-status">
                  {selectionScope === "filtered"
                    ? `All ${filteredCount} filtered takes selected${excludedTakeIds.size ? ` · ${excludedTakeIds.size} excluded` : ""}`
                    : `${selectedTakeIds.size} visible selected`}
                </span>
                <span className="datasets-toolbar-status">
                  {hasBaseFilters
                    ? `Showing ${takes.length} of ${filteredCount} filtered takes · ${datasetTotalCount} total in dataset`
                    : `Showing ${takes.length} of ${datasetTotalCount} takes`}
                </span>
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
                      <button type="button" disabled={selectionScope !== "visible" || selectedTakeIds.size < 2 || selectedTakeIds.size > 4} onClick={() => setCompareOpen((prev) => !prev)}>Compare selected</button>
                      <button type="button" disabled={busy || selectedCount === 0} onClick={() => openBulkModal("expected_class")}>Expected class</button>
                      <button type="button" disabled={busy || selectedCount === 0} onClick={() => { setModalValidationStatus("golden_sample"); openBulkModal("validation"); }}>Golden sample</button>
                      <button type="button" disabled={busy || selectedCount === 0} onClick={() => { setModalValidationStatus("benchmark_approved"); openBulkModal("validation"); }}>Benchmark approved</button>
                      <button type="button" disabled title="Future placeholder: copy selected takes to another dataset">Copy to dataset (planned)</button>
                    </div>
                  </div>
                </details>
              </div>
              {compareOpen && (
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
                {isInitialLoading && (
                  <div className="datasets-table-skeleton" aria-hidden="true">
                    {Array.from({ length: 8 }).map((_, idx) => <div className="datasets-skeleton-row" key={idx} />)}
                  </div>
                )}
                {!isInitialLoading && takes.length === 0 && <div className="datasets-inline-state">No takes match the current filters.</div>}
                {!isInitialLoading && takes.length > 0 && (
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
                        <tr
                          key={take.take_id}
                          className={selectedTakeId === take.take_id ? "active" : ""}
                          onClick={() => setSelectedTakeId(take.take_id)}
                        >
                          <td>
                            <input
                              type="checkbox"
                              checked={selected}
                              onClick={(event) => event.stopPropagation()}
                              onChange={(e) => {
                                if (selectionScope === "filtered") {
                                  setExcludedTakeIds((prev) => {
                                    const next = new Set(prev);
                                    if (e.target.checked) next.delete(take.take_id);
                                    else next.add(take.take_id);
                                    return next;
                                  });
                                } else {
                                  setSelectedTakeIds((prev) => {
                                    const next = new Set(prev);
                                    if (e.target.checked) next.add(take.take_id);
                                    else next.delete(take.take_id);
                                    return next;
                                  });
                                }
                              }}
                            />
                          </td>
                          <td>
                            <a
                              className="datasets-thumb-link"
                              href="/studio"
                              title="Open in Studio"
                              onClick={(event) => event.stopPropagation()}
                            >
                              {preview.modality && <span className="datasets-thumb-modality">{preview.modality}</span>}
                              {preview.url ? (
                                <img
                                  className="datasets-thumb"
                                  src={preview.url}
                                  alt={take.friendly_name || take.take_id}
                                  loading="lazy"
                                  decoding="async"
                                />
                              ) : (
                                <span className="datasets-thumb-placeholder">{preview.placeholder}</span>
                              )}
                            </a>
                          </td>
                          <td>
                            <a
                              className="datasets-take-link"
                              href="#"
                              onClick={(event) => {
                                event.preventDefault();
                                event.stopPropagation();
                                setSelectedTakeId(take.take_id);
                              }}
                            >
                              {take.friendly_name || take.take_id}
                            </a>
                          </td>
                          <td>{take.dataset_name || take.dataset_id || "-"}</td>
                          <td>
                            {take.experiment_session_id ? (
                              <a
                                href="#"
                                onClick={(event) => {
                                  event.preventDefault();
                                  event.stopPropagation();
                                  setSelectedSession(take.experiment_session_id || "");
                                  setSessionDrawerId(take.experiment_session_id || "");
                                }}
                              >
                                {take.experiment_session_name || take.experiment_session_id}
                              </a>
                            ) : "-"}
                          </td>
                          <td>{(take.tags ?? []).slice(0, 3).join(", ") || "-"}</td>
                          <td>{take.expected_class || "-"}</td>
                          <td>{take.validation_status || "unreviewed"}</td>
                          <td>{take.has_done ? "done" : take.has_ready ? "ready" : "pending"}</td>
                          <td>{take.calibration_id || "-"}</td>
                          <td>{take.created_at || "-"}</td>
                          <td>
                            <a href={`/takes/${encodeURIComponent(take.take_id)}`} title="Inspect pipeline outputs in Studio">Inspect</a>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
                )}
              </div>
              <div className="datasets-loadmore-row">
                <button type="button" disabled={!hasMore || isInitialLoading || isLoadingMore} onClick={() => void loadMoreTakes()}>
                  {isLoadingMore ? "Loading more..." : hasMore ? "Load more" : "All filtered takes loaded"}
                </button>
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
                  <MLSetIngestionWizard dataset={selectedDatasetSummary} sessions={sessions} onClose={() => setIngestionWizardOpen(false)} />
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
                      </tr>
                    </thead>
                    <tbody>
                      {mlSets.map((item) => (
                        <tr key={item.id}>
                          <td>{item.name || item.id}</td>
                          <td>{item.dataset_id || "-"}</td>
                          <td>{String(item.member_count ?? 0)}</td>
                          <td>{item.membership_mode || "ids"}</td>
                          <td>{item.created_at || "-"}</td>
                          <td>placeholder</td>
                        </tr>
                      ))}
                      {!mlSets.length && (
                        <tr>
                          <td colSpan={6}>No ML sets yet.</td>
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

        <aside className="concept-panel datasets-right">
          <h3>Inspector</h3>
          <div className="datasets-inspector-preview">
            {selectedTakeSummary && selectedTakePreview.url ? (
              <img
                className="datasets-inspector-thumb"
                src={selectedTakePreview.url}
                alt={selectedTakeSummary.friendly_name || selectedTakeSummary.take_id}
                loading="lazy"
                decoding="async"
              />
            ) : (
              <span className="datasets-thumb-placeholder inspector">{selectedTakePreview.placeholder}</span>
            )}
          </div>
          <section className="datasets-inspector-section">
            <h4>Preview</h4>
            <small>{selectedTakeSummary?.friendly_name ?? selectedTakeSummary?.take_id ?? "Select a take to review metadata, labels, split, preview, and semantic actions."}</small>
          </section>
          <div className="datasets-inspector-grid">
            <span>Dataset</span>
            <strong>{selectedDatasetSummary?.name ?? "All datasets"}</strong>
            <span>Session</span>
            <strong>{selectedSessionSummary?.name ?? "All sessions"}</strong>
            <span>Take</span>
            <strong>{detail?.take_id ?? selectedTakeId ?? "-"}</strong>
            <span>Object</span>
            <strong>{selectedObject?.objectId ?? "-"}</strong>
            <span>ML set</span>
            <strong>{activeTab === "ml_sets" ? "selection context" : "-"}</strong>
            <span>Split</span>
            <strong>{selectedSplit}</strong>
          </div>

          <div className="datasets-inspector-box">
            <strong>Processing Summary</strong>
            <small>acquisition type: {selectedSessionSummary?.session_type ?? "-"}</small>
            <small>calibration: {selectedSessionSummary?.calibration_id ?? detail?.result?.calibration_id ?? "-"}</small>
            <small>sensor metadata: {selectedSessionSummary?.sensor_metadata ? "available" : "-"}</small>
            <small>validation: {detail?.take_metadata?.validation_status as string ?? "-"}</small>
            <small>split: {String((detail?.take_metadata?.split as string | undefined) ?? readSplitFromSummary(selectedTakeSummary) ?? "-")}</small>
            <small>tags: {(selectedTakeSummary?.tags ?? []).join(", ") || "-"}</small>
            <small>objects: {String(detail?.result?.objects?.length ?? detail?.object_annotations?.length ?? "-")}</small>
          </div>

          <div className="datasets-inspector-box">
            <strong>Quick Semantic Actions</strong>
            <div className="datasets-actions-row">
              <button type="button" disabled={!selectedTakeId} onClick={() => selectedTakeId && void api.updateTakeMetadata(selectedTakeId, { validation_status: "valid" }).then(refreshTakes)}>Validate</button>
              <button type="button" disabled={!selectedTakeId} onClick={() => selectedTakeId && void api.updateTakeMetadata(selectedTakeId, { validation_status: "invalid" }).then(refreshTakes)}>Reject</button>
              <button type="button" disabled={!selectedTakeId} onClick={() => selectedTakeId && void api.updateTakeMetadata(selectedTakeId, { split: "validation" }).then(refreshTakes)}>Assign split</button>
              <button type="button" disabled={!selectedTakeId} onClick={() => selectedTakeId && void api.updateTakeMetadata(selectedTakeId, { ml_set_ids_add: ["ml_set_default"] }).then(refreshTakes)}>Add to ML set</button>
            </div>
          </div>

          <div className="datasets-inspector-actions">
            <button
              type="button"
              disabled={!selectedDataset && !selectedTakeId}
              title={selectedTakeId ? "Open selected take in Studio" : selectedDataset ? "Open selected dataset in Studio" : "Select a dataset first"}
              onClick={() => { window.location.href = "/studio"; }}
            >
              {selectedTakeId ? "Open selected take in Studio" : "Open selected dataset in Studio"}
            </button>
            <button
              type="button"
              disabled={!selectedTakeId}
              title={selectedTakeId ? "Inspect selected take outputs" : "Select a take first"}
              onClick={() => { if (selectedTakeId) window.location.href = `/takes/${encodeURIComponent(selectedTakeId)}`; }}
            >
              Inspect pipeline outputs
            </button>
            <button
              type="button"
              disabled={!selectedTakeId}
              title={selectedTakeId ? "Open Studio for segmentation review" : "Select a take first"}
              onClick={() => { if (selectedTakeId) window.location.href = "/studio"; }}
            >
              Review segmentation
            </button>
            <button
              type="button"
              disabled={selectedTakeIds.size === 0}
              title={selectedTakeIds.size ? "Open selected takes in Feature Analytics" : "Select one or more takes first"}
              onClick={() => { window.location.href = "/feature-analytics"; }}
            >
              Open selection in Feature Analytics
            </button>
          </div>

          <div className="datasets-inspector-box">
            <strong>Quick Review Mode</strong>
            <small>next unresolved / unreviewed / missing-label take</small>
            <div className="datasets-actions-row">
              <button type="button" onClick={() => {
                const next = takes.find((take) => String(take.validation_status ?? "unreviewed") !== "valid");
                if (next) setSelectedTakeId(next.take_id);
              }}>Next unresolved</button>
              <button type="button" onClick={() => {
                const next = takes.find((take) => String(take.validation_status ?? "unreviewed") === "unreviewed");
                if (next) setSelectedTakeId(next.take_id);
              }}>Next unreviewed</button>
              <button type="button" onClick={() => {
                const next = takes.find((take) => !take.expected_class);
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
                  ? `All ${filteredCount} filtered takes selected${excludedTakeIds.size ? ` · ${excludedTakeIds.size} excluded` : ""}`
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
