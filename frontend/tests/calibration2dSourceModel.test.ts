import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";

import { canCalibrateFromDetections, canDetectCorners, isSourceFresh, isTargetConfigValid, sourceDisplayLabel, workflowState } from "../src/studio/calibration2dSourceModel.ts";

test("source dropdown renders friendly label", () => {
  const label = sourceDisplayLabel({
    id: "usb_camera_0",
    label: "USB Camera 0",
    type: "usb_camera",
    modality: "rgb",
    status: "live",
    last_frame_at: "2026-05-19T12:00:00Z",
    last_frame_age_seconds: 0.4,
    resolution: [1920, 1080],
    fps: 63,
  });
  assert.match(label, /USB Camera 0/);
  assert.match(label, /1920x1080/);
  assert.match(label, /63.0 fps/);
});

test("preview stale does not block capture-first workflow gating", () => {
  assert.equal(isSourceFresh({
    id: "usb_camera_0",
    label: "USB Camera 0",
    type: "usb_camera",
    modality: "rgb",
    status: "stale",
    last_frame_at: "2026-05-19T12:00:00Z",
    last_frame_age_seconds: 12,
    resolution: [640, 480],
    fps: 10,
  }), false);

  assert.equal(canDetectCorners([], true), false);
  assert.equal(canDetectCorners([{ id: "c1", source_id: "usb_camera_0", timestamp: "2026-05-19", image_path: "a.png", resolution: [640, 480], stale: false }], true), true);
});

test("calibration enabled only after corner detection", () => {
  assert.equal(canCalibrateFromDetections([], true), false);
  assert.equal(canCalibrateFromDetections([{ capture_id: "c1", corners_found: false }], true), false);
  assert.equal(canCalibrateFromDetections([{ capture_id: "c1", corners_found: true }], true), true);
});

test("workflow states progress correctly", () => {
  assert.equal(workflowState({ captures: [], detections: [], hasCalibrationResult: false }), "NO_CAPTURES");
  assert.equal(workflowState({ captures: [{ id: "c1", source_id: "usb_camera_0", timestamp: "2026-05-19", image_path: "a.png", resolution: [640, 480], stale: false }], detections: [], hasCalibrationResult: false }), "CAPTURES_READY");
  assert.equal(workflowState({ captures: [{ id: "c1", source_id: "usb_camera_0", timestamp: "2026-05-19", image_path: "a.png", resolution: [640, 480], stale: false }], detections: [{ corners_found: true }], hasCalibrationResult: false }), "CORNERS_DETECTED");
  assert.equal(workflowState({ captures: [{ id: "c1", source_id: "usb_camera_0", timestamp: "2026-05-19", image_path: "a.png", resolution: [640, 480], stale: false }], detections: [{ corners_found: true }], hasCalibrationResult: true }), "CALIBRATION_VALID");
});

test("target params validity and raw source text input removed", () => {
  assert.equal(isTargetConfigValid({ squares_x: 7, squares_y: 5, square_length_mm: 25, marker_length_mm: 18 }), true);
  assert.equal(isTargetConfigValid({ squares_x: 7, squares_y: 5, square_length_mm: 18, marker_length_mm: 25 }), false);

  const calibrationPage = fs.readFileSync(new URL("../src/pages/CalibrationPage.tsx", import.meta.url), "utf8");
  assert.equal(calibrationPage.includes("<input value={sourceId}"), false);
});

test("stale source warning is inline and capture remains clickable", () => {
  const calibrationPage = fs.readFileSync(new URL("../src/pages/CalibrationPage.tsx", import.meta.url), "utf8");
  assert.equal(calibrationPage.includes("Source preview is stale (informational only)."), false);
  assert.equal(calibrationPage.includes("Source preview is stale. Capture still acquires a fresh frame."), true);
  assert.equal(calibrationPage.includes("disabled={busy || !sourceId} onClick={capture2DFrame}"), true);
});

test("captured frame display uses image_url and has explicit load error state", () => {
  const calibrationPage = fs.readFileSync(new URL("../src/pages/CalibrationPage.tsx", import.meta.url), "utf8");
  assert.equal(calibrationPage.includes("selectedCapture.image_url ??"), true);
  assert.equal(calibrationPage.includes("?t=${t}"), true);
  assert.equal(calibrationPage.includes("Captured frame image could not be loaded"), true);
});

test("capture freshness messages are explicit", () => {
  const calibrationPage = fs.readFileSync(new URL("../src/pages/CalibrationPage.tsx", import.meta.url), "utf8");
  assert.equal(calibrationPage.includes("Captured fresh frame (direct USB camera)."), true);
  assert.equal(calibrationPage.includes("Capture failed: no fresh frame available."), true);
});
