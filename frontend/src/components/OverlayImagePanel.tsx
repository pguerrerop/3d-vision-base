import { useMemo, useState } from "react";
import type { StudioArtifact } from "./studioWorkspaceModel";
import { overlayDebugInfo, transformOverlayGeometry, type OverlayDebugInfo, type OverlayDisplayMode } from "./overlayModel";

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
};

type Size = { width: number; height: number };

export default function OverlayImagePanel({
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
}: Props) {
  const [size, setSize] = useState<Size>({ width: 960, height: 540 });
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
        <div className="overlay-frame">
          <img
            src={src}
            alt={title}
            onLoad={(event) => {
              const image = event.currentTarget;
              setSize({
                width: image.naturalWidth || 960,
                height: image.naturalHeight || 540,
              });
            }}
          />
          {!!activeOverlays.length && (
            <svg className="artifact-overlay-layer" viewBox={`0 0 ${size.width} ${size.height}`}>
              {activeOverlays.map((overlay) =>
                renderOverlayShape(
                  overlay,
                  size,
                  selectedObjectId,
                  hoveredObjectId,
                  selectedOverlayArtifactId,
                  onSelectObject,
                  onSelectOverlayArtifact,
                  onHoverObject,
                  onHoverOverlayArtifact,
                  onOverlayDebug,
                  artifacts
                )
              )}
            </svg>
          )}
        </div>
      ) : (
        <div className="empty-image">{emptyMessage}</div>
      )}
    </section>
  );
}

function renderOverlayShape(
  overlay: StudioArtifact,
  size: Size,
  selectedObjectId: number | null,
  hoveredObjectId: number | null,
  selectedOverlayArtifactId: string | null,
  onSelectObject?: (objectId: number) => void,
  onSelectOverlayArtifact?: (artifactId: string) => void,
  onHoverObject?: (objectId: number | null) => void,
  onHoverOverlayArtifact?: (artifactId: string | null) => void,
  onOverlayDebug?: (info: OverlayDebugInfo) => void,
  artifacts: StudioArtifact[] = []
) {
  const style = overlay.style ?? {};
  const key = overlay.artifact_id;
  const transform = transformOverlayGeometry(overlay, size);
  const geometry = transform.transformedGeometry ?? {};
  const highlight = overlay.object_id != null && (overlay.object_id === selectedObjectId || overlay.object_id === hoveredObjectId);
  const selectedOverlay = overlay.artifact_id === selectedOverlayArtifactId;
  const dimmed = selectedObjectId != null && overlay.object_id != null && overlay.object_id !== selectedObjectId;
  const debug = overlayDebugInfo(overlay, artifacts, size);
  if (!transform.renderable) {
    return null;
  }
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
    return (
      <circle
        {...base}
        key={key}
        cx={px(geometry.x ?? geometry.cx, size.width)}
        cy={px(geometry.y ?? geometry.cy, size.height)}
        r={Number(style.radius ?? 4)}
        fill={String(style.fill ?? "#ff8a00")}
      />
    );
  }
  if (overlay.overlay_type === "polyline") {
    const points = Array.isArray(geometry.points_2d) ? geometry.points_2d : [];
    const path = points
      .map((point) => {
        if (!Array.isArray(point) || point.length < 2) return null;
        return `${px(point[0], size.width)},${px(point[1], size.height)}`;
      })
      .filter((item): item is string => Boolean(item))
      .join(" ");
    return <polyline {...base} key={key} points={path} fill="none" />;
  }
  if (overlay.overlay_type === "text") {
    const text = String(style.label ?? overlay.title);
    const anchorX = num(geometry.anchor_x);
    const anchorY = num(geometry.anchor_y);
    const tx = num(geometry.x);
    const ty = num(geometry.y);
    return (
      <g key={key}>
        {(anchorX !== 0 || anchorY !== 0) && (
          <line
            {...base}
            x1={anchorX}
            y1={anchorY}
            x2={tx}
            y2={ty}
            fill="none"
            stroke={String(style.stroke ?? "#101820")}
            opacity={selectedOverlay ? 1 : 0.8}
          />
        )}
        <text
          className={`artifact-overlay-text ${selectedOverlay ? "selected" : ""}`}
          x={tx}
          y={ty}
          fill={String(style.fill ?? "#0f172a")}
          onMouseEnter={base.onMouseEnter}
          onMouseLeave={base.onMouseLeave}
          onClick={base.onClick}
        >
          {text}
        </text>
      </g>
    );
  }
  return (
    <rect
      {...base}
      key={key}
      x={px(geometry.x, size.width)}
      y={px(geometry.y, size.height)}
      width={px(geometry.width, size.width)}
      height={px(geometry.height, size.height)}
    />
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
