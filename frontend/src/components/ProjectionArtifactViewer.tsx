import { useMemo, useRef, useState } from "react";
import type { StudioArtifact } from "./studioWorkspaceModel";
import { overlayDebugInfo, transformOverlayGeometry, type OverlayDebugInfo, type OverlayDisplayMode } from "./overlayModel";
import { clampPolygon, mapClientToImagePoint, mapClientToImagePointStrict, pointsToRoi, type RoiPolygon, type RoiRect } from "./roiMapping";
import type { HoverInspectionSample } from "./hoverInspectionModel";

export type RoiType = "rectangle" | "polygon" | "full_height_x_band";

type Props = {
  title: string;
  src?: string | null;
  overlays: StudioArtifact[];
  selectedObjectId: number | null;
  hoveredObjectId: number | null;
  selectedOverlayArtifactId: string | null;
  displayMode: OverlayDisplayMode;
  onSelectObject?: (objectId: number) => void;
  onSelectOverlayArtifact?: (artifactId: string) => void;
  onHoverObject?: (objectId: number | null) => void;
  onHoverOverlayArtifact?: (artifactId: string | null) => void;
  onOverlayDebug?: (info: OverlayDebugInfo) => void;
  artifacts?: StudioArtifact[];
  emptyMessage?: string;
  roi?: RoiRect | null;
  roiPolygon?: RoiPolygon | null;
  roiType?: RoiType;
  roiLabel?: string;
  drawRoiMode?: boolean;
  onRoiDrawComplete?: (roi: RoiRect) => void;
  onPolygonRoiComplete?: (points: RoiPolygon) => void;
  engineeringMode?: boolean;
  metricRange?: { min: number; max: number; unit: string } | null;
  hoverValueKind?: "raw_z" | "normalized_height" | "unknown";
  worldCalibration?: { originX: number; originY: number; resolutionX: number; resolutionY: number } | null;
  onHoverSample?: (sample: HoverInspectionSample | null) => void;
  numericHover?: {
    sourceArtifactId: string;
    units: string;
    displaySemantic?: string | null;
    semanticField?: string | null;
    width: number;
    height: number;
    renderedWidth: number;
    renderedHeight: number;
    roiPurpose: "plane_fit_only" | "render_crop";
    renderCoordinateSpace: "full_frame" | "roi_crop";
    sourceShape?: [number, number] | null;
    numericArrayShape?: [number, number] | null;
    validMaskShape?: [number, number] | null;
    displayVmin: number;
    displayVmax: number;
    normalizedValues: Float32Array;
    rawValues?: Float32Array | null;
    validMask?: Uint8Array | null;
    /**
     * Optional dense hover sampler that fills isolated invalid speckle pixels
     * inside the valid region by snapping to the nearest valid neighbor (small
     * search radius). When provided, hover uses this instead of raw indexing
     * to eliminate "alternating no-Z" UX while keeping true background
     * invalid.
     */
    denseSample?: (
      x: number,
      y: number,
      maxRadius?: number,
    ) => { value: number | null; valid: boolean; interpolated: boolean; sourceX: number; sourceY: number };
  } | null;
};

type Size = { width: number; height: number };

export default function ProjectionArtifactViewer(props: Props) {
  return <OverlayRenderer {...props} />;
}

function OverlayRenderer({
  title,
  src,
  overlays,
  selectedObjectId,
  hoveredObjectId,
  selectedOverlayArtifactId,
  displayMode,
  onSelectObject,
  onSelectOverlayArtifact,
  onHoverObject,
  onHoverOverlayArtifact,
  onOverlayDebug,
  artifacts = [],
  emptyMessage = "No image",
  roi = null,
  roiPolygon = null,
  roiType = "rectangle",
  roiLabel = "ROI",
  drawRoiMode = false,
  onRoiDrawComplete,
  onPolygonRoiComplete,
  engineeringMode = false,
  metricRange = null,
  hoverValueKind = "unknown",
  worldCalibration = null,
  onHoverSample,
  numericHover = null,
}: Props) {
  const [size, setSize] = useState<Size>({ width: 960, height: 540 });
  const frameRef = useRef<HTMLDivElement | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const [dragStart, setDragStart] = useState<{ x: number; y: number } | null>(null);
  const [dragNow, setDragNow] = useState<{ x: number; y: number } | null>(null);
  const [polyDraft, setPolyDraft] = useState<RoiPolygon>([]);
  const [bandDraft, setBandDraft] = useState<RoiRect | null>(null);
  const [bandDragMode, setBandDragMode] = useState<"move" | "resize_left" | "resize_right" | null>(null);
  const [bandDragAnchor, setBandDragAnchor] = useState<{ x: number; rect: RoiRect } | null>(null);
  const [polyClosed, setPolyClosed] = useState(false);
  const [hoverPoint, setHoverPoint] = useState<{ x: number; y: number } | null>(null);
  const [profilePoints, setProfilePoints] = useState<Array<{ x: number; y: number }>>([]);
  const [sampleCanvas, setSampleCanvas] = useState<HTMLCanvasElement | null>(null);
  const activeOverlays = useMemo(
    () =>
      overlays.filter((overlay) => {
        if (displayMode === "hide") return false;
        if (displayMode === "selected_only" && selectedObjectId != null) return overlay.object_id === selectedObjectId;
        return true;
      }),
    [displayMode, overlays, selectedObjectId]
  );
  return (
    <section className="image-panel overlay-image-panel">
      <div className="panel-title">{title}</div>
      {src ? (
        <div
          className={`overlay-frame ${drawRoiMode ? "draw-roi-mode" : ""}`}
          ref={frameRef}
          onDragStart={(event) => event.preventDefault()}
          onPointerDown={(event) => {
            if (!drawRoiMode || !frameRef.current) return;
            event.preventDefault();
            event.stopPropagation();
            const bounds = frameRef.current.getBoundingClientRect();
            const point = mapClientToImagePoint(event.clientX, event.clientY, bounds, size.width, size.height);
            if (roiType === "polygon") {
              setPolyClosed(false);
              setPolyDraft((current) => [...current, [Math.round(point.x), Math.round(point.y)]]);
              return;
            }
            if (roiType === "full_height_x_band") {
              const active = bandDraft ?? roi;
              const edgeTol = 8;
              if (active) {
                const left = active.x;
                const right = active.x + active.width;
                if (Math.abs(point.x - left) <= edgeTol) {
                  setBandDragMode("resize_left");
                  setBandDragAnchor({ x: point.x, rect: active });
                  return;
                }
                if (Math.abs(point.x - right) <= edgeTol) {
                  setBandDragMode("resize_right");
                  setBandDragAnchor({ x: point.x, rect: active });
                  return;
                }
                if (point.x >= left && point.x <= right) {
                  setBandDragMode("move");
                  setBandDragAnchor({ x: point.x, rect: active });
                  return;
                }
              }
            }
            if (!onRoiDrawComplete) return;
            setDragStart(point);
            setDragNow(point);
          }}
          onPointerMove={(event) => {
            if (!frameRef.current) return;
            event.preventDefault();
            event.stopPropagation();
            const frameBounds = frameRef.current.getBoundingClientRect();
            const imageBounds = imageRef.current?.getBoundingClientRect() ?? frameBounds;
            const strictPoint = mapClientToImagePointStrict(event.clientX, event.clientY, imageBounds, size.width, size.height);
            if (!strictPoint) {
              if (engineeringMode) {
                setHoverPoint(null);
                onHoverSample?.(null);
              }
              if (!drawRoiMode) return;
            }
            const point = strictPoint ?? mapClientToImagePoint(event.clientX, event.clientY, imageBounds, size.width, size.height);
            if (engineeringMode) {
              setHoverPoint(point);
              if (!numericHover) {
                onHoverSample?.(null);
              } else {
                onHoverSample?.(buildHoverSample({
                  point,
                  canvas: sampleCanvas,
                  metricRange,
                  valueKind: hoverValueKind,
                  worldCalibration,
                  hoveredObjectId,
                  selectedObjectId,
                  numericHover,
                  clientX: event.clientX,
                  clientY: event.clientY,
                  frameBounds,
                  imageBounds,
                  roi,
                }));
              }
            }
            if (!drawRoiMode) return;
            if (roiType === "polygon") {
              setHoverPoint(point);
              return;
            }
            if (roiType === "full_height_x_band" && bandDragMode && bandDragAnchor && onRoiDrawComplete) {
              const dx = Math.round(point.x - bandDragAnchor.x);
              const source = bandDragAnchor.rect;
              let next: RoiRect = source;
              if (bandDragMode === "move") {
                next = {
                  x: Math.max(0, Math.min(size.width - source.width, source.x + dx)),
                  y: 0,
                  width: source.width,
                  height: size.height,
                };
              } else if (bandDragMode === "resize_left") {
                const right = source.x + source.width;
                const x = Math.max(0, Math.min(right - 1, source.x + dx));
                next = { x, y: 0, width: Math.max(1, right - x), height: size.height };
              } else if (bandDragMode === "resize_right") {
                const right = Math.max(source.x + 1, Math.min(size.width, source.x + source.width + dx));
                next = { x: source.x, y: 0, width: Math.max(1, right - source.x), height: size.height };
              }
              setBandDraft(next);
              onRoiDrawComplete(next);
              return;
            }
            if (!dragStart) return;
            setDragNow(point);
          }}
          onPointerUp={() => {
            if (!drawRoiMode) return;
            if (roiType === "polygon") return;
            if (roiType === "full_height_x_band" && bandDragMode) {
              setBandDragMode(null);
              setBandDragAnchor(null);
              return;
            }
            if (!drawRoiMode || !dragStart || !dragNow || !onRoiDrawComplete) return;
            const next = roiType === "full_height_x_band"
              ? { ...pointsToRoi(dragStart, dragNow, size.width, size.height), y: 0, height: size.height }
              : pointsToRoi(dragStart, dragNow, size.width, size.height);
            onRoiDrawComplete(next);
            setBandDraft(next);
            setDragStart(null);
            setDragNow(null);
          }}
          onPointerLeave={() => {
            if (engineeringMode) {
              setHoverPoint(null);
              onHoverSample?.(null);
            }
            if (!drawRoiMode) return;
            if (roiType === "polygon") return;
            if (roiType === "full_height_x_band" && bandDragMode) {
              setBandDragMode(null);
              setBandDragAnchor(null);
              return;
            }
            if (!drawRoiMode || !dragStart || !dragNow || !onRoiDrawComplete) return;
            const next = roiType === "full_height_x_band"
              ? { ...pointsToRoi(dragStart, dragNow, size.width, size.height), y: 0, height: size.height }
              : pointsToRoi(dragStart, dragNow, size.width, size.height);
            onRoiDrawComplete(next);
            setBandDraft(next);
            setDragStart(null);
            setDragNow(null);
          }}
          onDoubleClick={(event) => {
            if (!drawRoiMode && engineeringMode && frameRef.current) {
              event.preventDefault();
              event.stopPropagation();
              const bounds = frameRef.current.getBoundingClientRect();
              const point = mapClientToImagePoint(event.clientX, event.clientY, bounds, size.width, size.height);
              setProfilePoints((prev) => prev.length >= 2 ? [point] : [...prev, point]);
              return;
            }
            if (!drawRoiMode || roiType !== "polygon" || !onPolygonRoiComplete) return;
            event.preventDefault();
            event.stopPropagation();
            const points = clampPolygon(polyDraft, size.width, size.height);
            if (points.length < 3) return;
            setPolyClosed(true);
            onPolygonRoiComplete(points);
          }}
          onKeyDown={(event) => {
            if (!drawRoiMode || roiType !== "polygon") return;
            if (event.key === "Escape") {
              event.preventDefault();
              setPolyDraft([]);
              setHoverPoint(null);
              setPolyClosed(false);
            }
            if (event.key === "Backspace") {
              event.preventDefault();
              setPolyDraft((current) => current.slice(0, -1));
            }
            if (event.key === "Enter" && onPolygonRoiComplete) {
              event.preventDefault();
              const points = clampPolygon(polyDraft, size.width, size.height);
              if (points.length < 3) return;
              setPolyClosed(true);
              onPolygonRoiComplete(points);
            }
          }}
          tabIndex={0}
        >
          <img
            ref={imageRef}
            src={src}
            alt={title}
            draggable={false}
            onLoad={(event) => {
              const image = event.currentTarget;
              setSize({ width: image.naturalWidth || 960, height: image.naturalHeight || 540 });
              const canvas = document.createElement("canvas");
              canvas.width = image.naturalWidth || 960;
              canvas.height = image.naturalHeight || 540;
              const ctx = canvas.getContext("2d");
              if (ctx) ctx.drawImage(image, 0, 0);
              setSampleCanvas(canvas);
            }}
          />
          {!!activeOverlays.length && (
            <svg className="artifact-overlay-layer" viewBox={`0 0 ${size.width} ${size.height}`}>
              {activeOverlays.map((overlay) => {
                const style = overlay.style ?? {};
                const key = overlay.artifact_id;
                const transform = transformOverlayGeometry(overlay, size);
                const geometry = transform.transformedGeometry ?? {};
                const highlight = overlay.object_id != null && (overlay.object_id === selectedObjectId || overlay.object_id === hoveredObjectId);
                const selectedOverlay = overlay.artifact_id === selectedOverlayArtifactId;
                const dimmed = selectedObjectId != null && overlay.object_id != null && overlay.object_id !== selectedObjectId;
                const debug = overlayDebugInfo(overlay, artifacts, size);
                if (!transform.renderable) return null;
                const base = {
                  stroke: String(style.stroke ?? "#00ff88"),
                  strokeWidth: selectedOverlay ? Number(style.line_width ?? 2) + 1 : Number(style.line_width ?? 2),
                  fill: String(style.fill ?? "rgba(0, 255, 136, 0.08)"),
                  opacity: selectedOverlay ? 1 : highlight ? 0.96 : dimmed ? 0.28 : 0.6,
                  className: `artifact-overlay-shape ${highlight ? "highlight" : ""} ${selectedOverlay ? "selected" : ""}`,
                  onClick: () => {
                    if (overlay.object_id != null) onSelectObject?.(overlay.object_id);
                    onSelectOverlayArtifact?.(overlay.artifact_id);
                    onOverlayDebug?.(debug);
                  },
                  onMouseEnter: () => {
                    onHoverObject?.(overlay.object_id ?? null);
                    onHoverOverlayArtifact?.(overlay.artifact_id);
                    onOverlayDebug?.(debug);
                  },
                  onMouseLeave: () => {
                    onHoverObject?.(null);
                    onHoverOverlayArtifact?.(null);
                  },
                };
                if (overlay.overlay_type === "ellipse") {
                  return (
                    <g key={key} transform={`rotate(${num(geometry.rotation_deg)} ${px(geometry.cx, size.width)} ${px(geometry.cy, size.height)})`}>
                      <ellipse {...base} cx={px(geometry.cx, size.width)} cy={px(geometry.cy, size.height)} rx={px(geometry.rx, size.width)} ry={px(geometry.ry, size.height)} />
                    </g>
                  );
                }
                if (overlay.overlay_type === "centroid") {
                  return <circle {...base} key={key} cx={px(geometry.x ?? geometry.cx, size.width)} cy={px(geometry.y ?? geometry.cy, size.height)} r={Number(style.radius ?? 4)} fill={String(style.fill ?? "#ff8a00")} />;
                }
                if (overlay.overlay_type === "text") {
                  const text = String(style.label ?? overlay.title);
                  return (
                    <text key={key} className={`artifact-overlay-text ${selectedOverlay ? "selected" : ""}`} x={num(geometry.x)} y={num(geometry.y)} fill={String(style.fill ?? "#0f172a")} onMouseEnter={base.onMouseEnter} onMouseLeave={base.onMouseLeave} onClick={base.onClick}>
                      {text}
                    </text>
                  );
                }
                return <rect {...base} key={key} x={px(geometry.x, size.width)} y={px(geometry.y, size.height)} width={px(geometry.width, size.width)} height={px(geometry.height, size.height)} />;
              })}
            </svg>
          )}
          {(roi || roiPolygon || bandDraft || polyDraft.length || (dragStart && dragNow)) && (
            <svg className="artifact-overlay-layer roi-layer" viewBox={`0 0 ${size.width} ${size.height}`}>
              {(() => {
                if (roiType === "polygon") {
                  const activePoints: RoiPolygon = polyDraft.length ? polyDraft : (roiPolygon ?? []);
                  if (!activePoints.length) return null;
                  const points = [...activePoints];
                  if (!polyClosed && hoverPoint && polyDraft.length) {
                    points.push([Math.round(hoverPoint.x), Math.round(hoverPoint.y)]);
                  }
                  const pointList = points.map(([x, y]) => `${x},${y}`).join(" ");
                  const first = points[0];
                  return (
                    <g>
                      <polygon
                        points={pointList}
                        fill="rgba(255, 209, 102, 0.12)"
                        stroke="#ffd166"
                        strokeWidth={2}
                        strokeDasharray="8 6"
                      />
                      {activePoints.map(([x, y], idx) => (
                        <circle key={`roi_vertex_${idx}`} cx={x} cy={y} r={4} fill="#ffd166" />
                      ))}
                      {first ? <text x={first[0] + 6} y={Math.max(16, first[1] + 16)} fill="#ffd166" className="artifact-overlay-text">{roiLabel}</text> : null}
                    </g>
                  );
                }
                const rect = dragStart && dragNow
                  ? (roiType === "full_height_x_band"
                    ? { ...pointsToRoi(dragStart, dragNow, size.width, size.height), y: 0, height: size.height }
                    : pointsToRoi(dragStart, dragNow, size.width, size.height))
                  : (roiType === "full_height_x_band" ? (bandDraft ?? roi) : roi);
                if (!rect) return null;
                return (
                  <g>
                    <rect
                      x={rect.x}
                      y={rect.y}
                      width={rect.width}
                      height={rect.height}
                      fill="rgba(255, 209, 102, 0.12)"
                      stroke="#ffd166"
                      strokeWidth={2}
                      strokeDasharray="8 6"
                    />
                    <text x={rect.x + 6} y={Math.max(16, rect.y + 16)} fill="#ffd166" className="artifact-overlay-text">{roiLabel}</text>
                  </g>
                );
              })()}
            </svg>
          )}
        </div>
      ) : (
        <div className="empty-image">{emptyMessage}</div>
      )}
      {engineeringMode && profilePoints.length === 2 && (
        <div className="artifact-reference">
          <small>{describeProfile(profilePoints[0], profilePoints[1], sampleCanvas, metricRange)}</small>
        </div>
      )}
    </section>
  );
}

function buildHoverSample(args: {
  point: { x: number; y: number };
  canvas: HTMLCanvasElement | null;
  metricRange: { min: number; max: number; unit: string } | null;
  valueKind: "raw_z" | "normalized_height" | "unknown";
  worldCalibration: { originX: number; originY: number; resolutionX: number; resolutionY: number } | null;
  hoveredObjectId: number | null;
  selectedObjectId: number | null;
  numericHover: {
    sourceArtifactId: string;
    units: string;
    displaySemantic?: string | null;
    semanticField?: string | null;
    width: number;
    height: number;
    renderedWidth: number;
    renderedHeight: number;
    roiPurpose: "plane_fit_only" | "render_crop";
    renderCoordinateSpace: "full_frame" | "roi_crop";
    sourceShape?: [number, number] | null;
    numericArrayShape?: [number, number] | null;
    validMaskShape?: [number, number] | null;
    displayVmin: number;
    displayVmax: number;
    normalizedValues: Float32Array;
    rawValues?: Float32Array | null;
    validMask?: Uint8Array | null;
    denseSample?: (
      x: number,
      y: number,
      maxRadius?: number,
    ) => { value: number | null; valid: boolean; interpolated: boolean; sourceX: number; sourceY: number };
  } | null;
  clientX: number;
  clientY: number;
  frameBounds: DOMRect | { left: number; top: number; width: number; height: number };
  imageBounds: DOMRect | { left: number; top: number; width: number; height: number };
  roi: RoiRect | null;
}): HoverInspectionSample {
  const { point, canvas, metricRange, valueKind, worldCalibration, hoveredObjectId, selectedObjectId, numericHover, clientX, clientY, frameBounds, imageBounds, roi } = args;
  const pixel = readPixel(point, canvas);
  const x = Math.max(0, Math.round(point.x));
  const y = Math.max(0, Math.round(point.y));
  let normalizedValue: number | null = null;
  let rawValue: number | null = null;
  let valid: boolean | null = pixel ? pixel.valid : null;
  let displayClipped: number | null = null;
  let planeHeight: number | null = null;
  let residualMm: number | null = null;
  let clipped: boolean | null = null;
  const renderW = numericHover?.renderedWidth ?? numericHover?.width ?? 0;
  const renderH = numericHover?.renderedHeight ?? numericHover?.height ?? 0;
  const sourceMap = mapRenderedToSourcePixel(
    x,
    y,
    renderW,
    renderH,
    numericHover?.width ?? 0,
    numericHover?.height ?? 0,
    numericHover?.renderCoordinateSpace ?? "full_frame",
  );
  let denseSampleResult: { value: number | null; valid: boolean; interpolated: boolean; sourceX: number; sourceY: number } | null = null;
  if (numericHover && sourceMap && sourceMap.x >= 0 && sourceMap.y >= 0 && sourceMap.x < numericHover.width && sourceMap.y < numericHover.height) {
    const idx = (sourceMap.y * numericHover.width) + sourceMap.x;
    const rawValid = numericHover.validMask ? numericHover.validMask[idx] > 0 : true;
    // Prefer the dense fallback so isolated invalid speckle pixels inside the
    // body still report a meaningful Z (snapped to nearest valid neighbor,
    // small radius). Falls back to raw indexing when no denseSample is wired.
    if (numericHover.denseSample) {
      const denseFromOrigin = numericHover.denseSample(x, y, 3);
      denseSampleResult = denseFromOrigin;
      const useDenseValue = denseFromOrigin.valid;
      normalizedValue = useDenseValue ? denseFromOrigin.value : Number(numericHover.normalizedValues[idx]);
      rawValue = numericHover.rawValues ? Number(numericHover.rawValues[(denseFromOrigin.sourceY * numericHover.width) + denseFromOrigin.sourceX]) : null;
      valid = useDenseValue ? true : rawValid;
    } else {
      normalizedValue = Number(numericHover.normalizedValues[idx]);
      rawValue = numericHover.rawValues ? Number(numericHover.rawValues[idx]) : null;
      valid = rawValid;
    }
    if (normalizedValue != null) {
      displayClipped = Math.min(numericHover.displayVmax, Math.max(numericHover.displayVmin, normalizedValue));
      residualMm = Math.abs(normalizedValue);
      clipped = Math.abs(displayClipped - normalizedValue) > 1e-9;
    }
    if (rawValue != null && normalizedValue != null) {
      planeHeight = rawValue - normalizedValue;
    }
  }
  const roiOriginX = roi ? Math.max(0, Math.round(roi.x)) : 0;
  const roiOriginY = roi ? Math.max(0, Math.round(roi.y)) : 0;
  const roiWidth = roi ? Math.max(1, Math.round(roi.width)) : (canvas?.width ?? 0);
  const roiHeight = roi ? Math.max(1, Math.round(roi.height)) : (canvas?.height ?? 0);
  const roiCropActive = Boolean(numericHover?.renderCoordinateSpace === "roi_crop");
  const renderMode: "full_image" | "roi_crop" = roiCropActive ? "roi_crop" : "full_image";
  const sourceArrayX = sourceMap?.x ?? x;
  const sourceArrayY = sourceMap?.y ?? y;
  const finalSampleX = sourceArrayX;
  const finalSampleY = sourceArrayY;
  const expectedSampleX = sourceMap?.x ?? x;
  const expectedSampleY = sourceMap?.y ?? y;
  if (finalSampleX !== expectedSampleX || finalSampleY !== expectedSampleY) {
    console.warn("[hover-map] coordinate assertion failed", {
      renderMode,
      finalSampleX,
      finalSampleY,
      expectedSampleX,
      expectedSampleY,
      roiOriginX,
      roiOriginY,
    });
  }
  const renderedRgb: [number, number, number] | null = pixel ? [pixel.r, pixel.g, pixel.b] : null;
  const sampledHeight = normalizedValue ?? rawValue ?? null;
  const interpolatedFromNearest = Boolean(denseSampleResult?.interpolated);
  const trace = {
    clientX,
    clientY,
    frameLeft: frameBounds.left,
    frameTop: frameBounds.top,
    frameWidth: frameBounds.width,
    frameHeight: frameBounds.height,
    imageLeft: imageBounds.left,
    imageTop: imageBounds.top,
    imageWidth: imageBounds.width,
    imageHeight: imageBounds.height,
    renderedX: x,
    renderedY: y,
    naturalX: x,
    naturalY: y,
    sourceArrayX,
    sourceArrayY,
    roiOriginX,
    roiOriginY,
    roiWidth,
    roiHeight,
    finalSampleX,
    finalSampleY,
    renderMode,
    renderedRgb,
    sampledHeight,
    roiPurpose: String(numericHover?.roiPurpose ?? "plane_fit_only"),
    renderCoordinateSpace: String(numericHover?.renderCoordinateSpace ?? "full_frame"),
    sourceShape: (numericHover?.sourceShape ?? [numericHover?.height ?? 0, numericHover?.width ?? 0]) as [number, number] | null,
    renderedPngShape: [renderH, renderW] as [number, number],
    numericArrayShape: (numericHover?.numericArrayShape ?? [numericHover?.height ?? 0, numericHover?.width ?? 0]) as [number, number] | null,
    validMaskShape: (numericHover?.validMaskShape ?? (numericHover?.validMask ? [numericHover.height, numericHover.width] : null)) as [number, number] | null,
    interpolatedFromNearest,
    denseSourceX: denseSampleResult?.sourceX ?? sourceArrayX,
    denseSourceY: denseSampleResult?.sourceY ?? sourceArrayY,
  };
  if (typeof window !== "undefined" && (window as unknown as { __STUDIO_HOVER_TRACE?: boolean }).__STUDIO_HOVER_TRACE) {
    console.debug("[hover-map]", trace);
  }
  const worldX = worldCalibration ? worldCalibration.originX + point.x * worldCalibration.resolutionX : null;
  const worldY = worldCalibration ? worldCalibration.originY + point.y * worldCalibration.resolutionY : null;
  return {
    imageX: x,
    imageY: y,
    worldX: Number.isFinite(worldX ?? NaN) ? worldX : null,
    worldY: Number.isFinite(worldY ?? NaN) ? worldY : null,
    // Canonical rule: millimeter values must come from numeric sources, never from preview RGB/intensity inversion.
    rawZ: rawValue,
    normalizedHeight: normalizedValue,
    planeHeight,
    residualMm,
    displayClippedHeight: displayClipped,
    clipped,
    valid,
    intensity: pixel ? pixel.intensity : null,
    hoveredObjectId,
    selectedObjectId,
    units: numericHover?.units ?? metricRange?.unit ?? null,
    sourceArtifactId: numericHover?.sourceArtifactId ?? null,
    displaySemantic: numericHover?.displaySemantic ?? null,
    semanticField: numericHover?.semanticField ?? null,
    valueKind,
    coordDebug: trace,
  };
}

function readPixel(point: { x: number; y: number }, canvas: HTMLCanvasElement | null): { intensity: number; valid: boolean; r: number; g: number; b: number } | null {
  if (!canvas) return null;
  const x = Math.max(0, Math.min(canvas.width - 1, Math.round(point.x)));
  const y = Math.max(0, Math.min(canvas.height - 1, Math.round(point.y)));
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  const data = ctx.getImageData(x, y, 1, 1).data;
  const alpha = Number(data[3] ?? 255);
  const r = Number(data[0] ?? 0);
  const g = Number(data[1] ?? 0);
  const b = Number(data[2] ?? 0);
  const intensity = (r + g + b) / 3;
  return { intensity, valid: alpha > 0, r, g, b };
}

function mapRenderedToSourcePixel(
  x: number,
  y: number,
  renderWidth: number,
  renderHeight: number,
  sourceWidth: number,
  sourceHeight: number,
  renderCoordinateSpace: "full_frame" | "roi_crop",
): { x: number; y: number } | null {
  if (sourceWidth <= 0 || sourceHeight <= 0) return null;
  if (renderWidth <= 0 || renderHeight <= 0) return null;
  if (renderCoordinateSpace === "full_frame" && renderWidth === sourceWidth && renderHeight === sourceHeight) {
    return {
      x: Math.max(0, Math.min(sourceWidth - 1, x)),
      y: Math.max(0, Math.min(sourceHeight - 1, y)),
    };
  }
  const sx = Math.max(0, Math.min(sourceWidth - 1, Math.round((x / Math.max(1, renderWidth - 1)) * Math.max(1, sourceWidth - 1))));
  const sy = Math.max(0, Math.min(sourceHeight - 1, Math.round((y / Math.max(1, renderHeight - 1)) * Math.max(1, sourceHeight - 1))));
  return { x: sx, y: sy };
}

function describeProfile(
  a: { x: number; y: number },
  b: { x: number; y: number },
  canvas: HTMLCanvasElement | null,
  metricRange: { min: number; max: number; unit: string } | null,
): string {
  if (!canvas) return "profile: unavailable";
  const ctx = canvas.getContext("2d");
  if (!ctx) return "profile: unavailable";
  const distancePx = Math.hypot(b.x - a.x, b.y - a.y);
  const samples = Math.max(16, Math.round(distancePx));
  const values: number[] = [];
  for (let i = 0; i < samples; i += 1) {
    const t = i / Math.max(1, samples - 1);
    const x = Math.max(0, Math.min(canvas.width - 1, Math.round(a.x + (b.x - a.x) * t)));
    const y = Math.max(0, Math.min(canvas.height - 1, Math.round(a.y + (b.y - a.y) * t)));
    const data = ctx.getImageData(x, y, 1, 1).data;
    const intensity = (Number(data[0]) + Number(data[1]) + Number(data[2])) / 3;
    const mapped = !metricRange || metricRange.max <= metricRange.min
      ? intensity
      : metricRange.min + (metricRange.max - metricRange.min) * (intensity / 255);
    values.push(mapped);
  }
  const sorted = [...values].sort((x, y) => x - y);
  const p95 = sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95))] ?? 0;
  const min = sorted[0] ?? 0;
  const max = sorted[sorted.length - 1] ?? 0;
  const unit = metricRange?.unit ?? "level";
  return `distance: ${distancePx.toFixed(1)} px | min/max/p95: ${min.toFixed(2)} / ${max.toFixed(2)} / ${p95.toFixed(2)} ${unit}`;
}

function px(value: unknown, max: number): number {
  const n = num(value);
  return Math.abs(n) <= 1.2 ? n * max : n;
}

function num(value: unknown): number {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return 0;
}
