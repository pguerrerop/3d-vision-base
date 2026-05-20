import { useCallback, useEffect, useMemo, useState } from "react";
import { API_BASE_URL, api, type ActiveCalibration, type CalibrationCapture, type ObjectFilterConfig, type PlaneCandidate, type PlaneLabel, type SourceInfo, type SystemCalibration, type TakeSummary } from "../api/client";
import CalibrationStateBadge from "../components/CalibrationStateBadge";
import LivePreviewPanel from "../components/LivePreviewPanel";
import type { RuntimeState } from "../api/client";
import { canCalibrateFromDetections, canDetectCorners, isSourceFresh, isTargetConfigValid, sourceDisplayLabel, workflowState } from "../studio/calibration2dSourceModel";

const LABELS: PlaneLabel[] = ["belt", "outer_plane_ignore", "unused"];
type CalibrationTab = "plane_3d" | "camera_2d" | "laser_line" | "fusion";
type TargetType = "charuco" | "checkerboard";

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

export default function CalibrationPage() {
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
  const [captureImageLoadFailed, setCaptureImageLoadFailed] = useState(false);

  const selectedSource = useMemo(() => sources.find((item) => item.id === sourceId) ?? null, [sources, sourceId]);
  const sourceFresh = isSourceFresh(selectedSource);
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

  const loadCaptures = useCallback(async () => {
    const payload = await api.list2DCalibrationCaptures(sourceId);
    setCaptures(payload.captures);
    setSelectedCaptureId((current) => current || payload.captures[0]?.id || "");
  }, [sourceId]);

  useEffect(() => { void loadLists(); }, [loadLists]);
  useEffect(() => { void loadCaptures(); }, [loadCaptures]);
  useEffect(() => { setCaptureImageLoadFailed(false); }, [selectedCapture?.id]);

  const beltPlaneId = useMemo(() => Object.entries(labels).find(([, label]) => label === "belt")?.[0] ?? null, [labels]);
  const camera2DCalibrations = useMemo(() => calibrations.filter((item) => item.calibration_type === "camera_2d"), [calibrations]);

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

  async function refreshSourcePreview() {
    setBusy(true);
    try {
      await api.refresh2DSource({ source_id: sourceId });
      await loadLists();
      setMessage(`Preview refreshed for ${sourceId}.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Refresh failed.");
    } finally { setBusy(false); }
  }

  async function capture2DFrame() {
    setBusy(true);
    try {
      const response = await api.capture2DCalibrationFrame({ source_id: sourceId });
      setLastAcquisitionMode(response.acquisition_mode);
      setLastCaptureAgeSeconds(response.frame_age_seconds ?? null);
      if (response.fallback_used) {
        setMessage(`Captured from preview cache fallback, age ${(response.frame_age_seconds ?? 0).toFixed(1)}s.`);
      } else {
        setMessage("Captured fresh frame (direct USB camera).");
      }
      await Promise.all([loadCaptures(), loadLists()]);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Capture failed.";
      if (message.includes("NO_FRESH_FRAME_AVAILABLE")) {
        setMessage("Capture failed: no fresh frame available.");
      } else {
        setMessage(message);
      }
    } finally { setBusy(false); }
  }

  async function detect2DCorners() {
    if (!detectEnabled) return;
    setBusy(true);
    try {
      const response = await api.detect2DCalibrationCorners({ capture_ids: captures.map((item) => item.id), target });
      setDetectionRows(response.detections);
      setMessage(`Detected corners in ${response.detected_count}/${response.capture_count} captures.`);
      await loadCaptures();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Corner detection failed.");
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
        }
      });
      setLastSavedCalibrationPath(response.path);
      setMessage(`Saved 2D calibration ${response.calibration_id}.`);
      await loadLists();
    } catch (error) { setMessage(error instanceof Error ? error.message : "2D calibration save failed."); }
    finally { setBusy(false); }
  }

  async function generatePrintableCharuco() {
    setBusy(true);
    try {
      const response = await api.generateCharucoTarget({ squares_x: squaresX, squares_y: squaresY, square_length_mm: squareLengthMm, marker_length_mm: markerLengthMm, dictionary });
      setMessage(`Generated target: ${response.png_path} and ${response.pdf_path}`);
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
          <div className="camera2d-layout">
            <div className="camera2d-left">
              <label className="field-label">Source
                <select value={sourceId} onChange={(event) => setSourceId(event.target.value)}>
                  {sources.length ? sources.map((source) => <option key={source.id} value={source.id}>{sourceDisplayLabel(source)}</option>) : <option value="">No live sources</option>}
                </select>
              </label>
              <small>{selectedSource ? `${selectedSource.status.toUpperCase()} • last frame ${selectedSource.last_frame_age_seconds == null ? "-" : `${selectedSource.last_frame_age_seconds.toFixed(1)}s`} ago` : "No source selected"}</small>
              <label className="field-label">Target type<select value={targetType} onChange={(event) => setTargetType(event.target.value as TargetType)}><option value="charuco">ChArUco</option><option value="checkerboard">Checkerboard</option></select></label>
              <label className="field-label">Squares X<input type="number" value={squaresX} onChange={(event) => setSquaresX(Number(event.target.value))} /></label>
              <label className="field-label">Squares Y<input type="number" value={squaresY} onChange={(event) => setSquaresY(Number(event.target.value))} /></label>
              <label className="field-label">Square length (mm)<input type="number" value={squareLengthMm} onChange={(event) => setSquareLengthMm(Number(event.target.value))} /></label>
              <label className="field-label">Marker length (mm)<input type="number" value={markerLengthMm} onChange={(event) => setMarkerLengthMm(Number(event.target.value))} /></label>
              {!targetValid && <small className="warning-text">Target parameters invalid: marker length must be smaller than square length.</small>}
              <button className="primary-button" disabled={busy || !sourceId} onClick={capture2DFrame} type="button">Capture calibration frame</button>
              <button className="secondary-button" disabled={busy || !detectEnabled} onClick={detect2DCorners} type="button">Detect corners</button>
              <button className="secondary-button" disabled={busy || !calibrateEnabled} onClick={calibrate2D} type="button">Calibrate</button>
              <button className="primary-button" disabled={busy || !calibration2DResult} onClick={save2DCalibration} type="button">Save calibration</button>
              <hr />
              <button className="secondary-button" onClick={generatePrintableCharuco} type="button">Generate printable ChArUco</button>
              <button className="secondary-button" disabled={busy || !sourceId} onClick={refreshSourcePreview} type="button">Refresh frame</button>
              <button className="secondary-button" disabled title="TODO: start live preview backend control not wired yet" type="button">Start live preview</button>
              <button className="secondary-button" disabled title="TODO: start/stop live preview backend control not wired yet" type="button">Stop live preview</button>
              <label className="field-label">Saved 2D calibrations<select value={selectedCalibrationId} onChange={(event) => setSelectedCalibrationId(event.target.value)}><option value="">None</option>{camera2DCalibrations.map((calibration) => <option key={calibration.calibration_id} value={calibration.calibration_id}>{calibration.calibration_id}</option>)}</select></label>
              <button className="secondary-button" disabled={!selectedCalibrationId} onClick={() => setRuntimeDefaultCalibration(calibrationPath(selectedCalibrationId))} type="button">Set active/default</button>
            </div>

            <div className="camera2d-center">
              <div className={`calibration-preview ${!sourceFresh ? "preview-stale" : ""}`}>
                {selectedCapture && selectedCaptureImageUrl && !captureImageLoadFailed ? (
                  <img
                    src={selectedCaptureImageUrl}
                    alt="Selected captured calibration frame"
                    onError={() => setCaptureImageLoadFailed(true)}
                  />
                ) : selectedCapture && captureImageLoadFailed ? (
                  <div className="empty-image">
                    <strong>Captured frame image could not be loaded</strong>
                    <small>capture: {selectedCapture.id}</small>
                    <small>image_url: {selectedCaptureImageUrl ?? "-"}</small>
                    {import.meta.env.DEV ? <small>path: {selectedCapture.image_path}</small> : null}
                  </div>
                ) : <div className="empty-image">No captured calibration frames yet.</div>}
              </div>
              <div className="camera2d-badges"><span className={`status-pill ${selectedSource?.status === "live" ? "ok" : selectedSource?.status === "stale" ? "warn" : "bad"}`}>{selectedSource?.status?.toUpperCase() ?? "UNAVAILABLE"}</span><span title={selectedSource?.label ?? "-"}>{selectedSource?.label ?? "-"}</span></div>
              {!sourceFresh && <div className="source-stale-warning">Source preview is stale. Capture still acquires a fresh frame.</div>}
              <div className="capture-gallery">
                {captures.length === 0 && <div className="empty-image">No captures stored yet.</div>}
                {captures.map((capture) => (
                  <button key={capture.id} className={`capture-row ${capture.id === selectedCapture?.id ? "selected" : ""}`} onClick={() => setSelectedCaptureId(capture.id)} type="button">
                    <strong title={capture.id}>{capture.id}</strong>
                    <small>{capture.timestamp}</small>
                    <small>{capture.resolution?.join("x")}</small>
                    <small>{capture.source_id}</small>
                    <small>{capture.detected_corners?.corners_found ? `Corners: ${capture.detected_corners.corner_count}` : "Corners: pending"}</small>
                  </button>
                ))}
              </div>
              <LivePreviewPanel runtimeState={runtimeState} title="Source preview (informational)" compact />
            </div>

            <div className="camera2d-right">
              <h3>Inspector</h3>
              <small>Workflow state: {state}</small>
              <small>Last acquisition mode: {lastAcquisitionMode ?? "-"}</small>
              <small>Last frame age (s): {lastCaptureAgeSeconds == null ? "-" : lastCaptureAgeSeconds.toFixed(2)}</small>
              {lastAcquisitionMode === "preview_cache_fallback" && <small className="warning-text">Capture used preview cache fallback.</small>}
              <small>Source label: {selectedSource?.label ?? "-"}</small>
              <small>Source id: {sourceId || "-"}</small>
              <small>Modality: {selectedSource?.modality ?? "-"}</small>
              <small>Status: {selectedSource?.status ?? "-"}</small>
              <small>Last frame age: {selectedSource?.last_frame_age_seconds == null ? "-" : `${selectedSource.last_frame_age_seconds.toFixed(2)}s`}</small>
              <small>Capture count: {captures.length}</small>
              <small>Selected capture: {selectedCapture?.timestamp ?? "-"}</small>
              <small>Selected capture freshness: {selectedCapture ? "fresh" : "-"}</small>
              <small>Corner detection status: {selectedCapture?.detected_corners?.corners_found ? `detected (${selectedCapture.detected_corners.corner_count})` : "pending"}</small>
              <small>Markers detected: {latestDetection ? String(latestDetection.marker_count ?? "-") : "-"}</small>
              <small>ChArUco corners detected: {latestDetection ? String(latestDetection.charuco_corner_count ?? latestDetection.corner_count ?? "-") : "-"}</small>
              <small>Detection API mode: {latestDetection ? String(latestDetection.api_mode ?? "-") : "-"}</small>
              <small>Reprojection error: {calibration2DResult?.intrinsics ? Number((calibration2DResult.intrinsics as Record<string, unknown>).reprojection_error ?? 0).toFixed(4) : "-"}</small>
              <small>Focal fx/fy: {calibration2DResult?.intrinsics ? (() => { const matrix = ((calibration2DResult.intrinsics as Record<string, unknown>).camera_matrix ?? []) as number[][]; return `${Number(matrix?.[0]?.[0] ?? 0).toFixed(2)} / ${Number(matrix?.[1]?.[1] ?? 0).toFixed(2)}`; })() : "-"}</small>
              <small>Distortion coeffs: {calibration2DResult?.intrinsics ? JSON.stringify((calibration2DResult.intrinsics as Record<string, unknown>).dist_coeffs ?? []) : "-"}</small>
              <small>mm/px x/y: {calibration2DResult?.belt_plane ? `${Number((calibration2DResult.belt_plane as Record<string, unknown>).mm_per_px_x ?? 0).toFixed(4)} / ${Number((calibration2DResult.belt_plane as Record<string, unknown>).mm_per_px_y ?? 0).toFixed(4)}` : "-"}</small>
              <small>Warnings/errors: {(detectionRows.find((row) => row.error)?.error as string) ?? "none"}</small>
            </div>
          </div>
        )}

        {tab !== "plane_3d" && tab !== "camera_2d" && <div className="section"><p>This workspace will be enabled in a future phase.</p></div>}
      </section>
    </main>
  );
}
