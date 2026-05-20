import { useMemo, useRef, useState } from "react";
import type { StudioArtifact } from "./studioWorkspaceModel";
import { overlayDebugInfo, transformOverlayGeometry, type OverlayDebugInfo, type OverlayDisplayMode } from "./overlayModel";
import { clampPolygon, mapClientToImagePoint, pointsToRoi, type RoiPolygon, type RoiRect } from "./roiMapping";

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
  roiType?: "rectangle" | "polygon";
  roiLabel?: string;
  drawRoiMode?: boolean;
  onRoiDrawComplete?: (roi: RoiRect) => void;
  onPolygonRoiComplete?: (points: RoiPolygon) => void;
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
}: Props) {
  const [size, setSize] = useState<Size>({ width: 960, height: 540 });
  const frameRef = useRef<HTMLDivElement | null>(null);
  const [dragStart, setDragStart] = useState<{ x: number; y: number } | null>(null);
  const [dragNow, setDragNow] = useState<{ x: number; y: number } | null>(null);
  const [polyDraft, setPolyDraft] = useState<RoiPolygon>([]);
  const [polyClosed, setPolyClosed] = useState(false);
  const [hoverPoint, setHoverPoint] = useState<{ x: number; y: number } | null>(null);
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
            if (!onRoiDrawComplete) return;
            setDragStart(point);
            setDragNow(point);
          }}
          onPointerMove={(event) => {
            if (!drawRoiMode || !frameRef.current) return;
            event.preventDefault();
            event.stopPropagation();
            const bounds = frameRef.current.getBoundingClientRect();
            const point = mapClientToImagePoint(event.clientX, event.clientY, bounds, size.width, size.height);
            if (roiType === "polygon") {
              setHoverPoint(point);
              return;
            }
            if (!dragStart) return;
            setDragNow(point);
          }}
          onPointerUp={() => {
            if (!drawRoiMode) return;
            if (roiType === "polygon") return;
            if (!drawRoiMode || !dragStart || !dragNow || !onRoiDrawComplete) return;
            const next = pointsToRoi(dragStart, dragNow, size.width, size.height);
            onRoiDrawComplete(next);
            setDragStart(null);
            setDragNow(null);
          }}
          onPointerLeave={() => {
            if (!drawRoiMode) return;
            if (roiType === "polygon") return;
            if (!drawRoiMode || !dragStart || !dragNow || !onRoiDrawComplete) return;
            const next = pointsToRoi(dragStart, dragNow, size.width, size.height);
            onRoiDrawComplete(next);
            setDragStart(null);
            setDragNow(null);
          }}
          onDoubleClick={(event) => {
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
            src={src}
            alt={title}
            draggable={false}
            onLoad={(event) => {
              const image = event.currentTarget;
              setSize({ width: image.naturalWidth || 960, height: image.naturalHeight || 540 });
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
          {(roi || roiPolygon || polyDraft.length || (dragStart && dragNow)) && (
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
                const rect = dragStart && dragNow ? pointsToRoi(dragStart, dragNow, size.width, size.height) : roi;
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
    </section>
  );
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
