import { useCallback, useEffect, useMemo, useState } from "react";
import { API_BASE_URL, api, type ActiveCalibration, type CalibrationCapture, type ObjectFilterConfig, type PlaneCandidate, type PlaneLabel, type SourceControlValue, type SourceControlsResponse, type SourceInfo, type SystemCalibration, type TakeSummary } from "../api/client";
import CalibrationStateBadge from "../components/CalibrationStateBadge";
import LivePreviewPanel from "../components/LivePreviewPanel";
import type { RuntimeState } from "../api/client";
import { canCalibrateFromDetections, canDetectCorners, isTargetConfigValid, sourceDisplayLabel, workflowState } from "../studio/calibration2dSourceModel";

const LABELS: PlaneLabel[] = ["belt", "outer_plane_ignore", "unused"];
type CalibrationTab = "plane_3d" | "camera_2d" | "laser_line" | "fusion";
type TargetType = "charuco" | "checkerboard";
const DICTIONARIES = ["DICT_4X4_50", "DICT_4X4_100", "DICT_5X5_50", "DICT_5X5_100", "DICT_6X6_250"] as const;
const CONTROL_ORDER = ["exposure", "focus", "gain", "brightness", "contrast", "sharpness", "saturation", "white_balance"] as const;

const DEFAULT_FILTER: ObjectFilterConfig = {
  min_height_above_belt_mm: 3,
  max_height_above_belt_mm: 130,
  require_center_inside_belt: true,
  min_fraction_points_inside_belt: 0.6
};

function imageSrc(path: string | null): string | null {
  if (!path) return null;
  if (path.startsWith("http") || path.startsWith("/api/")) return path.startsWith("http") ? path : `${API_BASE_URL}${path}`;
  return `/${path}`;
}

function vector(values: number[]): string { return values.map((value) => value.toFixed(2)).join(", "); }
function labelClass(label: PlaneLabel | undefined): string { return label === "belt" ? "belt" : label === "outer_plane_ignore" ? "ignored" : "unused"; }
function calibrationPath(calibrationId: string): string { return `config/calibrations/${calibrationId}.json`; }
function buildRuntimeSettings(draft: Record<string, { value?: number | null; auto?: boolean | null }>): Record<string, number | boolean | null> {
  const out: Record<string, number | boolean | null> = {};
  Object.entries(draft).forEach(([key, value]) => {
    if (value.value !== undefined) out[key] = value.value ?? null;
    if (value.auto !== undefined && value.auto !== null) out[`auto_${key}`] = value.auto;
  });
  return out;
}
function isControlDirty(name: string, draft: { value?: number | null; auto?: boolean | null } | undefined, cfg: SourceControlValue | undefined): boolean {
  if (!cfg || !cfg.writable) return false;
  const valueDirty = draft?.value !== undefined && Number(draft.value ?? NaN) !== Number(cfg.value ?? NaN);
  const autoDirty = cfg.auto_supported && draft?.auto !== undefined && Boolean(draft.auto) !== Boolean(cfg.auto_value);
  return Boolean(valueDirty || autoDirty);
}

export default function CalibrationPage() {
  const [cameraControlsModalOpen, setCameraControlsModalOpen] = useState(false);
  const [charucoModalOpen, setCharucoModalOpen] = useState(false);
  const [freezeFrame, setFreezeFrame] = useState(false);
  const [tab, setTab] = useState<CalibrationTab>("plane_3d");
  const [takes, setTakes] = useState<TakeSummary[]>([]);
  const [calibrations, setCalibrations] = useState<SystemCalibration[]>([]);
  const [activeCalibration, setActiveCalibrationState] = useState<ActiveCalibration | null>(null);
  const [runtimeState, setRuntimeState] = useState<RuntimeState | null>(null);
  const [sources, setSources] = useState<SourceInfo[]>([]);

  const [selectedTakeId, setSelectedTakeId] = useState("");
  const [selectedCalibrationId, setSelectedCalibrationId] = useState("");
  const [planes, setPlanes] = useState<PlaneCandidate[]>([]);
  const [labels, setLabels] = useState<Record<string, PlaneLabel>>({});
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  const [objectFilter, setObjectFilter] = useState<ObjectFilterConfig>(DEFAULT_FILTER);
  const [message, setMessage] = useState("");
  const [lastSavedCalibrationPath, setLastSavedCalibrationPath] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const [sourceId, setSourceId] = useState("usb_camera_0");
  const [targetType, setTargetType] = useState<TargetType>("charuco");
  const [squaresX, setSquaresX] = useState(7);
  const [squaresY, setSquaresY] = useState(5);
  const [squareLengthMm, setSquareLengthMm] = useState(25);
  const [markerLengthMm, setMarkerLengthMm] = useState(18);
  const [dictionary, setDictionary] = useState("DICT_4X4_50");

  const [captures, setCaptures] = useState<CalibrationCapture[]>([]);
  const [selectedCaptureId, setSelectedCaptureId] = useState<string>("");
  const [detectionRows, setDetectionRows] = useState<Array<Record<string, unknown>>>([]);
  const [calibration2DResult, setCalibration2DResult] = useState<Record<string, unknown> | null>(null);
  const [lastAcquisitionMode, setLastAcquisitionMode] = useState<string | null>(null);
  const [lastCaptureAgeSeconds, setLastCaptureAgeSeconds] = useState<number | null>(null);
  const [dictionaryTestRows, setDictionaryTestRows] = useState<Array<Record<string, unknown>>>([]);
  const [sourceControls, setSourceControls] = useState<SourceControlsResponse | null>(null);
  const [controlDraft, setControlDraft] = useState<Record<string, { value?: number | null; auto?: boolean | null }>>({});
  const [imageMode, setImageMode] = useState<"rgb" | "thresholded" | "marker_debug">("rgb");
  const [liveDiagnostics, setLiveDiagnostics] = useState<Record<string, unknown> | null>(null);

  const selectedSource = useMemo(() => sources.find((item) => item.id === sourceId) ?? null, [sources, sourceId]);
  const selectedCapture = useMemo(() => captures.find((item) => item.id === selectedCaptureId) ?? captures[0] ?? null, [captures, selectedCaptureId]);
  const selectedCaptureImageUrl = useMemo(() => {
    if (!selectedCapture) return null;
    const base = selectedCapture.image_url ?? `${API_BASE_URL}/api/calibration/camera-2d/captures/${encodeURIComponent(selectedCapture.id)}/image`;
    const t = encodeURIComponent(selectedCapture.timestamp || selectedCapture.id);
    return base.includes("?") ? `${base}&t=${t}` : `${base}?t=${t}`;
  }, [selectedCapture]);

  const target = { type: targetType, squares_x: squaresX, squares_y: squaresY, square_length_mm: squareLengthMm, marker_length_mm: markerLengthMm, dictionary } as const;
  const targetValid = isTargetConfigValid(target);
  const detectEnabled = canDetectCorners(captures, targetValid);
  const calibrateEnabled = canCalibrateFromDetections(detectionRows, targetValid);
  const state = workflowState({ captures, detections: detectionRows, hasCalibrationResult: Boolean(calibration2DResult) });
  const latestDetection = detectionRows[0] ?? null;
  const streamAvailable = Boolean(sourceId && selectedSource?.status !== "unavailable");

  const loadLists = useCallback(async () => {
    const [nextTakes, nextCalibrations, nextActiveCalibration, nextRuntimeState, nextSources] = await Promise.all([
      api.calibrationReferenceTakes(),
      api.listCalibrations(),
      api.activeCalibration(),
      api.state(),
      api.sources()
    ]);
    setTakes(nextTakes);
    setCalibrations(nextCalibrations);
    setActiveCalibrationState(nextActiveCalibration);
    setRuntimeState(nextRuntimeState);
    setSources(nextSources);
    setSelectedTakeId((current) => current || nextTakes[0]?.take_id || "");
    setSelectedCalibrationId((current) => current || nextCalibrations[0]?.calibration_id || "");
    setSourceId((current) => current || nextSources[0]?.id || "usb_camera_0");
  }, []);

  const loadCaptures = useCallback(async (preferredCaptureId?: string) => {
    const payload = await api.list2DCalibrationCaptures(sourceId);
    setCaptures(payload.captures);
    setSelectedCaptureId((current) => preferredCaptureId || current || payload.captures[0]?.id || "");
  }, [sourceId]);

  useEffect(() => { void loadLists(); }, [loadLists]);
  useEffect(() => { void loadCaptures(); }, [loadCaptures]);
  useEffect(() => {
    if (!sourceId) return;
    void api.sourceControls(sourceId).then((payload) => {
      setSourceControls(payload);
      const draft: Record<string, { value?: number | null; auto?: boolean | null }> = {};
      Object.entries(payload.controls ?? {}).forEach(([name, cfg]) => {
        draft[name] = { value: cfg.value, auto: cfg.auto_value };
      });
      setControlDraft(draft);
    }).catch(() => {
      setSourceControls(null);
      setControlDraft({});
    });
  }, [sourceId]);

  const beltPlaneId = useMemo(() => Object.entries(labels).find(([, label]) => label === "belt")?.[0] ?? null, [labels]);
  const camera2DCalibrations = useMemo(() => calibrations.filter((item) => item.calibration_type === "camera_2d"), [calibrations]);
  const cameraRuntimeState = useMemo(() => ({
    sourceId,
    source: selectedSource,
    controls: sourceControls,
    controlDraft,
    liveDiagnostics,
    freezeFrame,
  }), [sourceId, selectedSource, sourceControls, controlDraft, liveDiagnostics, freezeFrame]);
  const charucoCalibrationState = useMemo(() => ({
    captures,
    selectedCapture,
    selectedCaptureId,
    detectionRows,
    dictionary,
    squaresX,
    squaresY,
    squareLengthMm,
    markerLengthMm,
    targetType,
    imageMode,
  }), [captures, selectedCapture, selectedCaptureId, detectionRows, dictionary, squaresX, squaresY, squareLengthMm, markerLengthMm, targetType, imageMode]);
  const calibrationManagerState = useMemo(() => ({
    camera2DCalibrations,
    selectedCalibrationId,
    activeCalibration,
    calibration2DResult,
    lastSavedCalibrationPath,
  }), [camera2DCalibrations, selectedCalibrationId, activeCalibration, calibration2DResult, lastSavedCalibrationPath]);
  const hasDirtyWritableControls = useMemo(() => {
    const controls = sourceControls?.controls ?? {};
    return Object.entries(controls).some(([name, cfg]) => isControlDirty(name, controlDraft[name], cfg));
  }, [sourceControls, controlDraft]);
  const hasCalibrationRuntimeSettings = useMemo(() => {
    const calibration = camera2DCalibrations.find((item) => item.calibration_id === selectedCalibrationId);
    return Boolean(calibration?.camera_runtime_settings && Object.keys(calibration.camera_runtime_settings).length);
  }, [camera2DCalibrations, selectedCalibrationId]);

  async function detectPlanes() {
    if (!selectedTakeId) return;
    setBusy(true);
    try {
      const response = await api.detectCalibrationPlanes(selectedTakeId);
      setPlanes(response.planes);
      setPreviewImage(response.preview_image);
      setLabels(Object.fromEntries(response.planes.map((plane, index) => [plane.plane_id, index === 0 ? "belt" : "unused"])));
      setMessage(`Detected ${response.planes.length} plane candidates.`);
    } catch (error) { setMessage(error instanceof Error ? error.message : "Plane detection failed."); }
    finally { setBusy(false); }
  }

  function updateLabel(planeId: string, label: PlaneLabel) {
    setLabels((current) => {
      const next = { ...current, [planeId]: label };
      if (label === "belt") for (const id of Object.keys(next)) if (id !== planeId && next[id] === "belt") next[id] = "unused";
      return next;
    });
  }

  async function saveCalibration3D() {
    if (!selectedTakeId || !beltPlaneId) { setMessage("Select exactly one belt plane before saving."); return; }
    setBusy(true);
    try {
      const response = await api.saveCalibration({ reference_take_id: selectedTakeId, plane_labels: labels, object_filter: objectFilter });
      setMessage(`Saved calibration ${response.calibration_id}.`);
      setLastSavedCalibrationPath(response.path);
      await loadLists();
    } catch (error) { setMessage(error instanceof Error ? error.message : "Save failed."); }
    finally { setBusy(false); }
  }

  async function setRuntimeDefaultCalibration(path: string) { await api.setDefaultCalibration(path); await loadLists(); }

  async function capture2DFrame() {
    setBusy(true);
    try {
      const response = await api.capture2DCalibrationFrame({ source_id: sourceId });
      setLastAcquisitionMode(response.acquisition_mode);
      setLastCaptureAgeSeconds(response.frame_age_seconds ?? null);
      setLiveDiagnostics((response as Record<string, unknown>).image_diagnostics as Record<string, unknown> | null);
      if (response.fallback_used) {
        setMessage(`Captured from preview cache fallback, age ${(response.frame_age_seconds ?? 0).toFixed(1)}s.`);
      } else {
        setMessage("Captured fresh frame (direct USB camera).");
      }
      await Promise.all([loadCaptures(response.capture_id), loadLists()]);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Capture failed.";
      if (message.includes("NO_FRESH_FRAME_AVAILABLE")) {
        setMessage("Capture failed: no fresh frame available.");
      } else {
        setMessage(message);
      }
    } finally { setBusy(false); }
  }

  async function detect2DCorners(detectAll = false) {
    if (!detectEnabled) return;
    const captureId = selectedCapture?.id;
    if (!detectAll && !captureId) {
      setMessage("Select a captured frame first.");
      return;
    }
    setBusy(true);
    try {
      const response = await api.detect2DCalibrationCorners(detectAll ? { detect_all: true, capture_ids: captures.map((item) => item.id), target } : { capture_id: captureId, target });
      setDetectionRows(response.detections);
      setLiveDiagnostics((response.detections[0]?.image_diagnostics as Record<string, unknown> | undefined) ?? null);
      setMessage(`Detected corners in ${response.detected_count}/${response.capture_count} captures.`);
      await loadCaptures();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Corner detection failed.");
    } finally { setBusy(false); }
  }

  async function detect2DCornersForCapture(captureId: string) {
    setBusy(true);
    try {
      const response = await api.detect2DCalibrationCorners({ capture_id: captureId, target });
      setDetectionRows(response.detections);
      setLiveDiagnostics((response.detections[0]?.image_diagnostics as Record<string, unknown> | undefined) ?? null);
      setMessage(`Detected corners in ${response.detected_count}/${response.capture_count} captures.`);
      await loadCaptures(captureId);
    } catch (error) {
      const text = error instanceof Error ? error.message : "Corner detection failed.";
      let markerCount = 0;
      let debugImage = "";
      let errDict = dictionary;
      try {
        const jsonPart = text.slice(text.indexOf("{"));
        const parsed = JSON.parse(jsonPart) as { detections?: Array<Record<string, unknown>> };
        const first = parsed.detections?.[0] ?? {};
        markerCount = Number(first.marker_count ?? 0);
        errDict = String(first.dictionary_name ?? dictionary);
        debugImage = String(first.debug_image_url ?? "");
        if (parsed.detections) setDetectionRows(parsed.detections);
      } catch {
        // keep concise fallback
      }
      setMessage(`Detection failed. markers=${markerCount}, dictionary=${errDict}${debugImage ? `, debug=${debugImage}` : ""}`);
      await loadCaptures(captureId);
    } finally { setBusy(false); }
  }

  async function captureAndDetectFromCharucoModal() {
    if (!sourceId || !targetValid) return;
    setBusy(true);
    try {
      const capture = await api.capture2DCalibrationFrame({ source_id: sourceId });
      setLastAcquisitionMode(capture.acquisition_mode);
      setLastCaptureAgeSeconds(capture.frame_age_seconds ?? null);
      setLiveDiagnostics((capture as Record<string, unknown>).image_diagnostics as Record<string, unknown> | null);
      await loadCaptures(capture.capture_id);
      await detect2DCornersForCapture(capture.capture_id);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Capture + detect failed.";
      setMessage(message.includes("NO_FRESH_FRAME_AVAILABLE") ? "Capture failed: no fresh frame available." : message);
    } finally { setBusy(false); }
  }

  async function testDictionaries() {
    if (!selectedCapture?.id) {
      setMessage("Select a captured frame first.");
      return;
    }
    setBusy(true);
    try {
      const response = await api.test2DCalibrationDictionaries({ capture_id: selectedCapture.id, target });
      setDictionaryTestRows(response.results);
      const best = response.best_candidate;
      setMessage(best ? `Best dictionary candidate: ${String(best.dictionary)} (${String(best.marker_count)} markers).` : "Dictionary test produced no candidates.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Dictionary test failed.");
    } finally { setBusy(false); }
  }

  async function calibrate2D() {
    if (!calibrateEnabled) return;
    setBusy(true);
    try {
      const validCaptureIds = detectionRows.filter((item) => Boolean(item.corners_found)).map((item) => String(item.capture_id));
      const response = await api.calibrate2DCamera({ source_id: sourceId, capture_ids: validCaptureIds, target, reference_plane_z_mm: 0.0 });
      setCalibration2DResult(response);
      setMessage("2D camera calibration solved.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Calibration failed.");
    } finally { setBusy(false); }
  }

  async function save2DCalibration() {
    if (!calibration2DResult) return;
    const calibrationId = `camera2d_${new Date().toISOString().replace(/[:.]/g, "_")}`;
    const intrinsics = calibration2DResult.intrinsics as Record<string, unknown>;
    const beltPlane = calibration2DResult.belt_plane as Record<string, unknown>;
    setBusy(true);
    try {
      const response = await api.save2DCalibration({
        calibration_id: calibrationId,
        source_id: sourceId,
        target,
        intrinsics: {
          camera_matrix: (intrinsics.camera_matrix ?? []) as number[][],
          dist_coeffs: (intrinsics.dist_coeffs ?? []) as number[],
          reprojection_error: Number(intrinsics.reprojection_error ?? 0),
          image_width: Number(intrinsics.image_width ?? 1),
          image_height: Number(intrinsics.image_height ?? 1)
        },
        belt_plane: {
          homography: (beltPlane.homography ?? []) as number[][],
          mm_per_px_x: Number(beltPlane.mm_per_px_x ?? 0),
          mm_per_px_y: Number(beltPlane.mm_per_px_y ?? 0),
          reference_plane_z_mm: Number(beltPlane.reference_plane_z_mm ?? 0)
        },
        camera_runtime_settings: buildRuntimeSettings(controlDraft)
      });
      setLastSavedCalibrationPath(response.path);
      setMessage(`Saved 2D calibration ${response.calibration_id}.`);
      await loadLists();
    } catch (error) { setMessage(error instanceof Error ? error.message : "2D calibration save failed."); }
    finally { setBusy(false); }
  }

  async function applySourceControls() {
    if (!sourceId) return;
    const controls = sourceControls?.controls ?? {};
    const changed: Record<string, { value?: number | null; auto?: boolean | null }> = {};
    Object.entries(controls).forEach(([name, cfg]) => {
      const draft = controlDraft[name];
      if (!isControlDirty(name, draft, cfg)) return;
      const payload: { value?: number | null; auto?: boolean | null } = {};
      if (draft?.value !== undefined) payload.value = draft.value ?? null;
      if (cfg.auto_supported && draft?.auto !== undefined) payload.auto = draft.auto ?? null;
      changed[name] = payload;
    });
    if (!Object.keys(changed).length) {
      setMessage("No writable control changes to apply.");
      return;
    }
    setBusy(true);
    try {
      const response = await api.updateSourceControls(sourceId, { controls: changed });
      setSourceControls(response);
      const warnings = Object.entries((response.applied ?? {}) as Record<string, { warnings?: string[] }>).filter(([, value]) => (value.warnings ?? []).length > 0);
      setMessage(warnings.length ? `Applied controls with warnings on ${warnings.map(([k]) => k).join(", ")}.` : "Applied source controls.");
      const refreshed = await api.sourceControls(sourceId);
      setSourceControls(refreshed);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Failed to apply source controls.");
    } finally { setBusy(false); }
  }

  async function restoreSourceControlsDefaults() {
    if (!sourceControls) return;
    const next: Record<string, { value?: number | null; auto?: boolean | null }> = {};
    Object.entries(sourceControls.controls).forEach(([name, cfg]) => {
      next[name] = { value: cfg.default ?? cfg.min, auto: cfg.auto_value };
    });
    setControlDraft(next);
    setMessage("Controls reset to default baseline values. Click Apply controls.");
  }

  function loadControlsFromCalibration() {
    const calibration = camera2DCalibrations.find((item) => item.calibration_id === selectedCalibrationId);
    const settings = calibration?.camera_runtime_settings;
    if (!settings) {
      setMessage("Selected calibration has no persisted camera runtime settings.");
      return;
    }
    const next = { ...controlDraft };
    Object.entries(settings).forEach(([key, value]) => {
      if (key.startsWith("auto_")) {
        const base = key.replace("auto_", "");
        next[base] = { ...(next[base] ?? {}), auto: typeof value === "boolean" ? value : null };
      } else if (typeof value === "number" || value == null) {
        next[key] = { ...(next[key] ?? {}), value: value as number | null };
      }
    });
    setControlDraft(next);
    setMessage("Loaded camera controls from calibration. Click Apply controls to send them to camera.");
  }

  function detectionHint(): string {
    const row = latestDetection ?? {};
    const sharpness = Number((row.sharpness ?? liveDiagnostics?.sharpness ?? 0) || 0);
    const markerCount = Number(row.marker_count ?? 0);
    const rejected = Number(row.rejected_count ?? 0);
    const clipping = Number((liveDiagnostics?.saturation_clipping_percent as number | undefined) ?? 0);
    const mean = Number((liveDiagnostics?.mean_brightness as number | undefined) ?? 0);
    const coverage = Number(row.board_coverage ?? 0);
    if (rejected > 10 && sharpness < 80) return "Likely blur/focus issue: reduce motion blur and adjust focus.";
    if (markerCount < 4 && sharpness > 120) return "Likely dictionary mismatch or low contrast.";
    if (clipping > 8) return "Exposure too high: reduce exposure/gain to avoid highlight clipping.";
    if (mean < 45) return "Image too dark: increase exposure/gain or improve lighting.";
    if (coverage > 0 && coverage < 0.12) return "Board coverage is low: move target closer or occupy more frame area.";
    return "No major diagnostic warning.";
  }

  async function generatePrintableCharuco() {
    setBusy(true);
    try {
      const response = await api.generateCharucoTarget({ squares_x: squaresX, squares_y: squaresY, square_length_mm: squareLengthMm, marker_length_mm: markerLengthMm, dictionary });
      setMessage(`Generated ChArUco ${response.dictionary_name} (${response.squares_x}x${response.squares_y}, ${response.square_length_mm}/${response.marker_length_mm}mm, ${Number(response.printable_size_mm[0]).toFixed(1)}x${Number(response.printable_size_mm[1]).toFixed(1)}mm): ${response.generated_file_path}`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Target generation failed.");
    } finally { setBusy(false); }
  }

  return (
    <main className="calibration-page">
      <aside className="calibration-sidebar">
        <div className="sidebar-title"><span>Calibration</span></div>
        <div className="active-calibration-card"><span>Active calibration</span><CalibrationStateBadge active={activeCalibration} /><strong title={activeCalibration?.path ?? "None configured"}>{activeCalibration?.path ?? "None configured"}</strong><small>Type: {activeCalibration?.calibration_type ?? "plane_3d"}</small><em>{activeCalibration?.status ?? "none"}</em></div>
        {message && <div className="calibration-message">{message}</div>}
        {!!lastSavedCalibrationPath && <button className="primary-button" onClick={() => setRuntimeDefaultCalibration(lastSavedCalibrationPath)} type="button">Set saved calibration as active/default</button>}
      </aside>

      <section className="calibration-main">
        <section className="page-title-block"><div className="eyebrow">Calibration</div><h1>Source alignment/configuration</h1></section>
        <div className="calibration-tabs">
          <button className={tab === "plane_3d" ? "active" : ""} onClick={() => setTab("plane_3d")} type="button">3D Plane</button>
          <button className={tab === "camera_2d" ? "active" : ""} onClick={() => setTab("camera_2d")} type="button">2D Camera</button>
          <button className={tab === "laser_line" ? "active" : ""} onClick={() => setTab("laser_line")} type="button">Laser Line</button>
          <button className={tab === "fusion" ? "active" : ""} onClick={() => setTab("fusion")} type="button">Fusion</button>
        </div>

        {tab === "plane_3d" && (<>
          <LivePreviewPanel runtimeState={runtimeState} title="Live alignment preview" />
          <label className="field-label">Reference take<select value={selectedTakeId} onChange={(event) => setSelectedTakeId(event.target.value)}>{takes.map((take) => <option key={take.take_id} value={take.take_id}>{take.take_id}</option>)}</select></label>
          <button className="primary-button" disabled={!selectedTakeId || busy} onClick={detectPlanes} type="button">Detect planes</button>
          <div className="calibration-preview">{previewImage ? <img src={imageSrc(previewImage) ?? ""} alt="Plane candidates" /> : <div className="empty-image">Detect planes to view segmented candidates.</div>}</div>
          <div className="plane-card-grid">{planes.map((plane) => (<article className={`plane-card ${labelClass(labels[plane.plane_id])}`} key={plane.plane_id}><div className="plane-card-title"><strong>{plane.plane_id}</strong><select value={labels[plane.plane_id] ?? "unused"} onChange={(event) => updateLabel(plane.plane_id, event.target.value as PlaneLabel)}>{LABELS.map((label) => <option key={label} value={label}>{label}</option>)}</select></div><small>Points: {plane.point_count.toLocaleString()}</small><small>Dimensions (mm): {vector(plane.extent_mm)}</small></article>))}</div>
          <button className="primary-button" disabled={busy || !beltPlaneId} onClick={saveCalibration3D} type="button">Save 3D calibration</button>
        </>)}

        {tab === "camera_2d" && (
          <div className="camera2d-layout manager">
            <div className="camera2d-left">
              <h3>Calibration Manager</h3>
              <label className="field-label">Source
                <select value={sourceId} onChange={(event) => setSourceId(event.target.value)}>
                  {sources.length ? sources.map((source) => <option key={source.id} value={source.id}>{sourceDisplayLabel(source)}</option>) : <option value="">No live sources</option>}
                </select>
              </label>
              <label className="field-label">Saved 2D calibrations<select value={selectedCalibrationId} onChange={(event) => setSelectedCalibrationId(event.target.value)}><option value="">None</option>{camera2DCalibrations.map((calibration) => <option key={calibration.calibration_id} value={calibration.calibration_id}>{calibration.calibration_id}</option>)}</select></label>
              <button className="secondary-button" disabled={!selectedCalibrationId} onClick={() => setRuntimeDefaultCalibration(calibrationPath(selectedCalibrationId))} type="button">Set active/default</button>
              <button className="primary-button" disabled={busy || !sourceId} onClick={capture2DFrame} type="button">Capture Snapshot</button>
              <button className="secondary-button" type="button" onClick={() => setCameraControlsModalOpen(true)}>Open Camera Controls</button>
              <button className="secondary-button" type="button" onClick={() => setCharucoModalOpen(true)}>Open ChArUco Calibration</button>
            </div>
            <div className="camera2d-center">
              <h3>Selected Calibration Summary</h3>
              <small>State model: manager/camera/charuco split</small>
              <small>Active calibration: {calibrationManagerState.activeCalibration?.calibration_id ?? "-"}</small>
              <small>Reprojection error: {calibration2DResult?.intrinsics ? Number((calibration2DResult.intrinsics as Record<string, unknown>).reprojection_error ?? 0).toFixed(4) : "-"}</small>
              <small>mm/px x/y: {calibration2DResult?.belt_plane ? `${Number((calibration2DResult.belt_plane as Record<string, unknown>).mm_per_px_x ?? 0).toFixed(4)} / ${Number((calibration2DResult.belt_plane as Record<string, unknown>).mm_per_px_y ?? 0).toFixed(4)}` : "-"}</small>
              <small>Distortion coeffs: {calibration2DResult?.intrinsics ? JSON.stringify((calibration2DResult.intrinsics as Record<string, unknown>).dist_coeffs ?? []) : "-"}</small>
              <small>Image resolution: {calibration2DResult?.intrinsics ? `${String((calibration2DResult.intrinsics as Record<string, unknown>).image_width ?? "-")}x${String((calibration2DResult.intrinsics as Record<string, unknown>).image_height ?? "-")}` : "-"}</small>
              <div className="calibration-preview">{selectedCaptureImageUrl ? <img src={selectedCaptureImageUrl} alt="Calibration preview" /> : <div className="empty-image">No calibration preview yet.</div>}</div>
            </div>
            <div className="camera2d-right">
              <h3>Diagnostics</h3>
              <small>Workflow state: {state}</small>
              <small>Source: {cameraRuntimeState.sourceId}</small>
              <small>Status: {cameraRuntimeState.source?.status ?? "-"}</small>
              <small>Capture count: {charucoCalibrationState.captures.length}</small>
              <small>Last acquisition mode: {lastAcquisitionMode ?? "-"}</small>
              <small>Diagnostic hint: {detectionHint()}</small>
            </div>
          </div>
        )}

        {cameraControlsModalOpen && (
          <div className="modal-backdrop" role="dialog" aria-modal="true">
            <div className="modal-panel wide">
              <div className="modal-header"><h3>Camera Controls</h3><button type="button" onClick={() => setCameraControlsModalOpen(false)}>Close</button></div>
              <div className="modal-grid">
                <div>
                  <small>{selectedSource?.label ?? sourceId} • {selectedSource?.status?.toUpperCase() ?? "UNKNOWN"} • {selectedSource?.resolution?.join("x") ?? "-"} • {selectedSource?.fps == null ? "-" : `${selectedSource.fps.toFixed(1)} fps`} • age {selectedSource?.last_frame_age_seconds == null ? "-" : `${selectedSource.last_frame_age_seconds.toFixed(2)}s`}</small>
                  <div className="live-stream-wrap">
                    <img
                      src={freezeFrame && selectedCaptureImageUrl ? selectedCaptureImageUrl : `${API_BASE_URL}/api/runtime/stream/mjpeg?source_id=${encodeURIComponent(sourceId)}&fps=10`}
                      alt="Live stream"
                    />
                  </div>
                  <div className="camera2d-image-modes">
                    <button type="button" disabled={!streamAvailable} onClick={() => setFreezeFrame((v) => !v)}>{freezeFrame ? "Unfreeze frame" : "Freeze frame"}</button>
                    <button className="primary-button" disabled={busy || !sourceId || !streamAvailable} onClick={capture2DFrame} type="button">Capture snapshot</button>
                  </div>
                </div>
                <div>
                  <div className="camera-controls-grid">
                    {CONTROL_ORDER.map((name) => {
                      const cfg = sourceControls?.controls?.[name] as SourceControlValue | undefined;
                      const draft = controlDraft[name] ?? {};
                      const supported = Boolean(cfg?.supported);
                      const writable = Boolean(cfg?.writable);
                      const hasAuto = Boolean(cfg?.auto_supported);
                      return (
                        <label key={name} className="field-label">
                          {name.replace("_", " ")} <small>{writable ? "writable" : supported ? "read-only" : "unsupported"}</small>
                          <small>Current: {cfg?.value == null ? "-" : String(cfg.value)}</small>
                          <small>Range: {cfg?.min == null || cfg?.max == null ? "unknown" : `${cfg.min} ... ${cfg.max}`} ({cfg?.range_source ?? "unknown"})</small>
                          {hasAuto ? <label><input type="checkbox" checked={Boolean(draft.auto)} disabled={!writable} onChange={(event) => setControlDraft((current) => ({ ...current, [name]: { ...(current[name] ?? {}), auto: event.target.checked } }))} /> Auto</label> : null}
                          <input type="range" value={Number(draft.value ?? cfg?.value ?? cfg?.min ?? 0)} min={cfg?.min ?? 0} max={cfg?.max ?? 1} step={cfg?.step ?? 1} disabled={!writable} onChange={(event) => setControlDraft((current) => ({ ...current, [name]: { ...(current[name] ?? {}), value: Number(event.target.value) } }))} />
                          <input type="number" value={draft.value ?? ""} min={cfg?.min ?? undefined} max={cfg?.max ?? undefined} step={cfg?.step ?? 1} disabled={!writable} onChange={(event) => setControlDraft((current) => ({ ...current, [name]: { ...(current[name] ?? {}), value: Number(event.target.value) } }))} title={supported ? undefined : "Not supported by current source/backend"} />
                          {!supported && <small>Not supported by this camera/backend</small>}
                          {cfg?.warning && <small>{cfg.warning}</small>}
                        </label>
                      );
                    })}
                  </div>
                  <div className="camera2d-image-modes">
                    <button className="secondary-button" disabled={busy || !sourceId || !hasDirtyWritableControls} onClick={applySourceControls} type="button">Apply controls</button>
                    <button className="secondary-button" disabled={busy || !sourceId} onClick={restoreSourceControlsDefaults} type="button">Restore defaults</button>
                    <button className="secondary-button" disabled={busy || !selectedCalibrationId || !hasCalibrationRuntimeSettings} onClick={loadControlsFromCalibration} type="button">Load from calibration</button>
                  </div>
                  <small>Sharpness: {liveDiagnostics ? String(liveDiagnostics.sharpness ?? "-") : "-"}</small>
                  <small>Mean brightness: {liveDiagnostics ? String(liveDiagnostics.mean_brightness ?? "-") : "-"}</small>
                  <small>Clipping %: {liveDiagnostics ? String(liveDiagnostics.saturation_clipping_percent ?? "-") : "-"}</small>
                  {liveDiagnostics && Array.isArray(liveDiagnostics.histogram_16) && (
                    <div className="histogram-preview">{(liveDiagnostics.histogram_16 as number[]).map((v, i) => <span key={`live_h_${i}`} style={{ height: `${Math.max(2, Math.round(v * 60))}px` }} />)}</div>
                  )}
                  <details>
                    <summary>Backend control diagnostics</summary>
                    <small>backend: {sourceControls?.backend ?? "-"}</small>
                    <small>camera index: {String(sourceControls?.camera_index ?? "-")}</small>
                    <small>raw: {JSON.stringify(sourceControls?.diagnostics?.raw_get ?? {})}</small>
                    <small>unsupported: {JSON.stringify(sourceControls?.diagnostics?.unsupported_properties ?? [])}</small>
                    <small>last apply errors: {JSON.stringify(sourceControls?.diagnostics?.last_apply_errors ?? [])}</small>
                  </details>
                </div>
              </div>
            </div>
          </div>
        )}

        {charucoModalOpen && (
          <div className="modal-backdrop" role="dialog" aria-modal="true">
            <div className="modal-panel wide">
              <div className="modal-header"><h3>ChArUco Calibration</h3><button type="button" onClick={() => setCharucoModalOpen(false)}>Close</button></div>
              <div className="modal-grid">
                <div>
                  <label className="field-label">Target type<select value={targetType} onChange={(event) => setTargetType(event.target.value as TargetType)}><option value="charuco">ChArUco</option><option value="checkerboard">Checkerboard</option></select></label>
                  <label className="field-label">Dictionary<select value={dictionary} onChange={(event) => setDictionary(event.target.value)}>{DICTIONARIES.map((item) => <option key={item} value={item}>{item}</option>)}</select></label>
                  <label className="field-label">Squares X<input type="number" value={squaresX} onChange={(event) => setSquaresX(Number(event.target.value))} /></label>
                  <label className="field-label">Squares Y<input type="number" value={squaresY} onChange={(event) => setSquaresY(Number(event.target.value))} /></label>
                  <label className="field-label">Square size mm<input type="number" value={squareLengthMm} onChange={(event) => setSquareLengthMm(Number(event.target.value))} /></label>
                  <label className="field-label">Marker size mm<input type="number" value={markerLengthMm} onChange={(event) => setMarkerLengthMm(Number(event.target.value))} /></label>
                  {!targetValid && <small className="warning-text">Target parameters invalid: marker length must be smaller than square length.</small>}
                  <div className="camera2d-image-modes">
                    <button className="primary-button" disabled={busy || !sourceId || !targetValid} onClick={capture2DFrame} type="button">Capture snapshot</button>
                    <button className="primary-button" disabled={busy || !sourceId || !targetValid} onClick={captureAndDetectFromCharucoModal} type="button">Capture + detect</button>
                    <button className="secondary-button" disabled={busy || !detectEnabled} onClick={() => detect2DCorners(false)} type="button">Detect corners</button>
                    <button className="secondary-button" disabled={busy || !detectEnabled} onClick={() => detect2DCorners(true)} type="button">Detect all captures</button>
                    <button className="secondary-button" disabled={busy || !selectedCapture} onClick={testDictionaries} type="button">Test dictionaries</button>
                    <button className="secondary-button" onClick={generatePrintableCharuco} type="button">Generate printable target</button>
                    <button className="secondary-button" disabled={busy || !calibrateEnabled} onClick={calibrate2D} type="button">Calibrate</button>
                    <button className="primary-button" disabled={busy || !calibration2DResult} onClick={save2DCalibration} type="button">Save calibration</button>
                  </div>
                  <div className="capture-gallery">
                    {captures.map((capture) => (
                      <button key={capture.id} className={`capture-row ${capture.id === selectedCapture?.id ? "selected" : ""}`} onClick={() => setSelectedCaptureId(capture.id)} type="button">
                        <strong title={capture.id}>{capture.id}</strong><small>{capture.timestamp}</small><small>{capture.resolution?.join("x")}</small>
                      </button>
                    ))}
                  </div>
                </div>
                <div>
                  <div className="camera2d-image-modes">
                    <button className={imageMode === "rgb" ? "active" : ""} type="button" onClick={() => setImageMode("rgb")}>RGB</button>
                    <button className={imageMode === "thresholded" ? "active" : ""} type="button" onClick={() => setImageMode("thresholded")}>Thresholded</button>
                    <button className={imageMode === "marker_debug" ? "active" : ""} type="button" onClick={() => setImageMode("marker_debug")}>Marker debug</button>
                  </div>
                  <div className="calibration-preview">
                    {selectedCapture && selectedCaptureImageUrl ? (
                      <img src={imageMode === "rgb" ? selectedCaptureImageUrl : imageMode === "thresholded" ? `${API_BASE_URL}/api/calibration/camera-2d/captures/${encodeURIComponent(selectedCapture.id)}/threshold-image?t=${encodeURIComponent(selectedCapture.timestamp || selectedCapture.id)}` : `${API_BASE_URL}/api/calibration/camera-2d/captures/${encodeURIComponent(selectedCapture.id)}/debug-image?t=${encodeURIComponent(selectedCapture.timestamp || selectedCapture.id)}`} alt="Selected captured calibration frame" />
                    ) : <div className="empty-image">No captured frames.</div>}
                  </div>
                  <small>Markers: {latestDetection ? String(latestDetection.marker_count ?? "-") : "-"}</small>
                  <small>Marker IDs: {latestDetection ? String((latestDetection.marker_ids as number[] | undefined)?.join(", ") ?? "-") : "-"}</small>
                  <small>ChArUco corners: {latestDetection ? String(latestDetection.charuco_corner_count ?? "-") : "-"}</small>
                  <small>Rejected: {latestDetection ? String(latestDetection.rejected_count ?? "-") : "-"}</small>
                  <small>Board coverage: {latestDetection ? String(latestDetection.board_coverage ?? "-") : "-"}</small>
                  <small>Sharpness: {latestDetection ? String(latestDetection.sharpness ?? "-") : "-"}</small>
                  <small>API mode: {latestDetection ? String(latestDetection.api_mode ?? "-") : "-"}</small>
                  <small>Failure reason: {latestDetection ? String(latestDetection.failure_reason ?? "none") : "-"}</small>
                  {latestDetection && Number(latestDetection.marker_count ?? 0) > 0 && Number(latestDetection.charuco_corner_count ?? 0) === 0 && (
                    <>
                      <small className="warning-text">Markers detected but insufficient consistent corners for ChArUco interpolation. Check dictionary, board dimensions, blur/focus, glare, board size, and visibility.</small>
                      <button className="secondary-button" type="button" onClick={() => { setCharucoModalOpen(false); setCameraControlsModalOpen(true); }}>Open Camera Controls</button>
                    </>
                  )}
                  {dictionaryTestRows.length > 0 && <div className="dictionary-test-block"><strong>Best dictionary candidates</strong>{dictionaryTestRows.slice(0, 5).map((row, idx) => <small key={`${String(row.dictionary)}_${idx}`}>{String(row.dictionary)}: markers {String(row.marker_count)} / corners {String(row.charuco_corner_count)}</small>)}</div>}
                </div>
              </div>
            </div>
          </div>
        )}

        {tab !== "plane_3d" && tab !== "camera_2d" && <div className="section"><p>This workspace will be enabled in a future phase.</p></div>}
      </section>
    </main>
  );
}
