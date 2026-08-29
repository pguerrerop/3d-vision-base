import { useEffect, useMemo, useRef, useState, type MouseEvent, type ReactElement } from "react";
import type { BlobCandidate } from "../studio/blobDetectionModel";
import { buildLabelLookup, type LabelEntry } from "../studio/labelMaskLookup";

type OverlayMode = "contours" | "labels" | "bbox" | "filled" | "rejected_only";

type Props = {
  src: string | null;
  // Optional per-pixel connected-component label mask (grayscale, one integer id per pixel,
  // background = 0) that a candidate's true pixel shape can be read from when it has no contour
  // polygon of its own. Not a color-classification overlay -- those encode role (e.g. belt
  // background vs. stripe), not per-candidate identity, and can't be reliably matched back to a
  // specific candidate id.
  maskSourceSrc?: string | null;
  candidates: BlobCandidate[];
  rejected: BlobCandidate[];
  selectedCandidateId: number | null;
  hoveredCandidateId: number | null;
  onSelectCandidate: (id: number) => void;
  onHoverCandidate: (id: number | null) => void;
};

function fmtNum(value: number | undefined, decimals: number): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(decimals) : "-";
}

function useMaskSourceCanvas(src: string | null | undefined): HTMLCanvasElement | null {
  const [canvas, setCanvas] = useState<HTMLCanvasElement | null>(null);
  useEffect(() => {
    if (!src) { setCanvas(null); return; }
    let cancelled = false;
    const img = new Image();
    img.onload = () => {
      if (cancelled) return;
      const el = document.createElement("canvas");
      el.width = img.naturalWidth;
      el.height = img.naturalHeight;
      const ctx = el.getContext("2d");
      if (!ctx) return;
      ctx.drawImage(img, 0, 0);
      setCanvas(el);
    };
    img.onerror = () => { if (!cancelled) setCanvas(null); };
    img.src = src;
    return () => { cancelled = true; };
  }, [src]);
  return canvas;
}

function readLabelAt(canvas: HTMLCanvasElement, x: number, y: number): number | null {
  if (x < 0 || y < 0 || x >= canvas.width || y >= canvas.height) return null;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  return ctx.getImageData(x, y, 1, 1).data[0];
}

// One full-image pass that recolors every candidate's true pixels at once: dim for everyone once
// a selection exists (still signaling "a blob lives here"), full strength for the selected one.
function buildCompositeMaskDataUrl(
  sourceCanvas: HTMLCanvasElement,
  labelLookup: Map<number, LabelEntry>,
  selectedId: number | null,
): string | null {
  if (labelLookup.size === 0) return null;
  const ctx = sourceCanvas.getContext("2d");
  if (!ctx) return null;
  const { width, height } = sourceCanvas;
  const source = ctx.getImageData(0, 0, width, height);
  const src = source.data;
  const output = new ImageData(width, height);
  const dst = output.data;
  const hasSelection = selectedId != null;
  for (let i = 0; i < src.length; i += 4) {
    const label = src[i];
    if (label === 0) continue;
    const match = labelLookup.get(label);
    if (!match) continue;
    const isSelected = hasSelection && match.id === selectedId;
    const alpha = hasSelection ? (isSelected ? 175 : 14) : (match.isRejected ? 60 : 56);
    const [r, g, b] = match.isRejected ? [239, 68, 68] : [34, 197, 94];
    dst[i] = r;
    dst[i + 1] = g;
    dst[i + 2] = b;
    dst[i + 3] = alpha;
  }
  const outCanvas = document.createElement("canvas");
  outCanvas.width = width;
  outCanvas.height = height;
  const outCtx = outCanvas.getContext("2d");
  if (!outCtx) return null;
  outCtx.putImageData(output, 0, 0);
  return outCanvas.toDataURL();
}

function fmtCentroid(value: [number, number] | undefined): string {
  return value ? `${value[0].toFixed(1)}, ${value[1].toFixed(1)}` : "-";
}

function fmtBbox(value: [number, number, number, number] | undefined): string {
  return value ? value.map((v) => v.toFixed(0)).join(", ") : "-";
}

export default function BlobSetViewer({
  src,
  maskSourceSrc,
  candidates,
  rejected,
  selectedCandidateId,
  hoveredCandidateId,
  onSelectCandidate,
  onHoverCandidate,
}: Props) {
  const [mode, setMode] = useState<OverlayMode>("filled");
  const [imgSize, setImgSize] = useState<{ width: number; height: number } | null>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; row: BlobCandidate; rejected: boolean } | null>(null);
  const maskSourceCanvas = useMaskSourceCanvas(maskSourceSrc);
  const svgRef = useRef<SVGSVGElement>(null);

  const acceptedCount = candidates.length;
  const rejectedCount = rejected.length;
  const overlayRows = mode === "rejected_only" ? rejected : [...candidates, ...rejected];
  const tableRows = [...candidates, ...rejected];

  const rejectedIds = useMemo(() => new Set(rejected.map((row) => Number(row.id))), [rejected]);
  const labelLookup = useMemo(
    () => buildLabelLookup(tableRows, rejectedIds),
    [candidates, rejected],
  );
  const compositeMaskUrl = useMemo(
    () => (mode === "filled" && maskSourceCanvas ? buildCompositeMaskDataUrl(maskSourceCanvas, labelLookup, selectedCandidateId) : null),
    [mode, maskSourceCanvas, labelLookup, selectedCandidateId],
  );

  // Lets you click any pixel that's part of a blob's true mask, not just its centroid -- these
  // components' bounding boxes routinely nest inside one another, so a bbox-based click target
  // would frequently pick the wrong (outer) blob.
  function handleBackgroundClick(event: MouseEvent<SVGElement>) {
    const svg = svgRef.current;
    if (!svg || !maskSourceCanvas || !imgSize) return;
    const point = svg.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    const ctm = svg.getScreenCTM();
    if (!ctm) return;
    const local = point.matrixTransform(ctm.inverse());
    const scaleX = maskSourceCanvas.width / imgSize.width;
    const scaleY = maskSourceCanvas.height / imgSize.height;
    const label = readLabelAt(maskSourceCanvas, Math.round(local.x * scaleX), Math.round(local.y * scaleY));
    if (label == null || label === 0) return;
    const match = labelLookup.get(label);
    if (match) onSelectCandidate(match.id);
  }

  return (
    <div className="blob-set-viewer">
      <section className="image-panel overlay-image-panel blob-overlay-workspace blob-set-viewer-visual">
        <div className="overlay-controls">
          <button className={mode === "contours" ? "active" : ""} onClick={() => setMode("contours")} type="button">Contours</button>
          <button className={mode === "labels" ? "active" : ""} onClick={() => setMode("labels")} type="button">Labels</button>
          <button className={mode === "bbox" ? "active" : ""} onClick={() => setMode("bbox")} type="button">BBox</button>
          <button className={mode === "filled" ? "active" : ""} onClick={() => setMode("filled")} type="button">Filled mask</button>
          <button className={mode === "rejected_only" ? "active" : ""} onClick={() => setMode("rejected_only")} type="button">Rejected only</button>
        </div>
        <div className="blob-overlay-legend">
          <small><span className="legend-chip accepted" /> accepted: {acceptedCount}</small>
          <small><span className="legend-chip rejected" /> rejected: {rejectedCount}</small>
          <small><span className="legend-chip selected" /> selected</small>
        </div>
        {src ? (
          <div className="overlay-frame blob-overlay-frame">
            <img
              alt="Blob candidate overlay"
              className="artifact-image"
              draggable={false}
              onLoad={(event) => setImgSize({ width: event.currentTarget.naturalWidth || 1, height: event.currentTarget.naturalHeight || 1 })}
              src={src}
            />
            {imgSize && (
              <svg ref={svgRef} className="artifact-overlay-layer" viewBox={`0 0 ${imgSize.width} ${imgSize.height}`}>
                {labelLookup.size > 0 && (
                  <rect
                    x={0}
                    y={0}
                    width={imgSize.width}
                    height={imgSize.height}
                    fill="transparent"
                    onClick={handleBackgroundClick}
                    style={{ cursor: "pointer" }}
                  />
                )}
                {compositeMaskUrl && (
                  <image href={compositeMaskUrl} x={0} y={0} width={imgSize.width} height={imgSize.height} style={{ pointerEvents: "none" }} />
                )}
                {overlayRows.map((row, rowIndex) => {
                  const id = Number(row.id);
                  const isRejected = rejected.some((item) => Number(item.id) === id);
                  const selected = selectedCandidateId != null && selectedCandidateId === id;
                  const hovered = hoveredCandidateId != null && hoveredCandidateId === id;
                  const contour = Array.isArray(row.contour) ? row.contour : [];
                  const bbox = Array.isArray(row.bbox) ? row.bbox : null;
                  const centroid = Array.isArray(row.centroid) ? row.centroid : null;
                  const stroke = isRejected ? "#ef4444" : "#22c55e";
                  const width = selected ? 4 : hovered ? 3 : isRejected ? 1.5 : 2.4;
                  const alpha = selected ? 1 : isRejected ? 0.75 : 0.95;
                  const label = `#${id}${isRejected && row.rejectionReason ? ` (${row.rejectionReason})` : ""}`;
                  // With a selection active, dim every other shape's fill so the selected blob's
                  // full mask reads clearly instead of every candidate's fill blending together.
                  const hasSelection = selectedCandidateId != null;
                  const fillOpacity = mode !== "filled" ? 0 : hasSelection ? (selected ? 0.55 : 0.06) : (isRejected ? 0.24 : 0.22);
                  const fillHue = isRejected ? "239,68,68" : "34,197,94";
                  const shapeProps = {
                    stroke,
                    strokeWidth: width,
                    strokeOpacity: alpha,
                    fill: mode === "filled" ? `rgba(${fillHue},${fillOpacity})` : "none",
                    onClick: (event: MouseEvent<SVGElement>) => {
                      event.preventDefault();
                      event.stopPropagation();
                      onSelectCandidate(id);
                    },
                    onMouseEnter: (event: MouseEvent<SVGElement>) => {
                      event.preventDefault();
                      onHoverCandidate(id);
                      setTooltip({
                        x: centroid?.[0] ?? (bbox ? bbox[0] + bbox[2] / 2 : 0),
                        y: centroid?.[1] ?? (bbox ? bbox[1] + bbox[3] / 2 : 0),
                        row,
                        rejected: isRejected,
                      });
                    },
                    onMouseLeave: () => {
                      onHoverCandidate(null);
                      setTooltip(null);
                    },
                  };
                  const parts: ReactElement[] = [];
                  const hasContour = contour.length >= 2;
                  if (hasContour && mode !== "bbox") {
                    parts.push(
                      <polyline
                        key={`contour_${id}`}
                        {...shapeProps}
                        points={contour.map((p) => `${p[0]},${p[1]}`).join(" ")}
                        className={selected ? "blob-selected-shape" : ""}
                      />
                    );
                  }
                  // Filled mode never draws the bbox itself -- for contour-less candidates it's
                  // approximate at best, and the real per-pixel mask (compositeMaskUrl, sampled
                  // from the classification overlay) already covers that case more honestly.
                  if (bbox && (mode === "bbox" || mode === "labels" || mode === "rejected_only")) {
                    parts.push(
                      <rect
                        key={`bbox_${id}`}
                        {...shapeProps}
                        x={bbox[0]}
                        y={bbox[1]}
                        width={bbox[2]}
                        height={bbox[3]}
                        className={selected ? "blob-selected-shape" : ""}
                      />
                    );
                  }
                  if (centroid && mode !== "bbox") {
                    // Contour points aren't always present in the artifact payload (only centroid +
                    // bbox are guaranteed), so the centroid marker needs its own larger, invisible hit
                    // area to stay clickable/hoverable even when there's no polyline to click.
                    parts.push(
                      <circle
                        key={`centroid_hit_${id}`}
                        {...shapeProps}
                        cx={centroid[0]}
                        cy={centroid[1]}
                        r={10}
                        fill="transparent"
                        stroke="none"
                        style={{ cursor: "pointer" }}
                      />
                    );
                    // The dot + id number are only useful when the user explicitly asked for them
                    // (Labels mode) -- everywhere else they just clutter the mask/contour view.
                    // The invisible hit circle above stays in every mode so hover/click keeps working.
                    if (mode === "labels") {
                      parts.push(
                        <circle
                          key={`centroid_${id}`}
                          cx={centroid[0]}
                          cy={centroid[1]}
                          r={selected ? 5 : 3.2}
                          fill={isRejected ? "#f97316" : "#06b6d4"}
                          opacity={0.95}
                          style={{ pointerEvents: "none" }}
                        />
                      );
                      parts.push(
                        <text
                          key={`label_${id}`}
                          className={`artifact-overlay-text ${selected ? "selected" : ""}`}
                          x={centroid[0] + 7}
                          y={Math.max(14, centroid[1] - 7)}
                          fill={selected ? "#111827" : "#0f172a"}
                          stroke={selected ? "#f8fafc" : "none"}
                          strokeWidth={selected ? 0.8 : 0}
                          style={{ pointerEvents: "none" }}
                        >
                          {label}
                        </text>
                      );
                    }
                  }
                  return <g key={`row_${id}_${rowIndex}`}>{parts}</g>;
                })}
                {tooltip && (
                  <g className="blob-tooltip">
                    <rect x={Math.max(0, tooltip.x + 8)} y={Math.max(0, tooltip.y + 8)} width={210} height={88} rx={8} ry={8} fill="#0f172a" fillOpacity={0.88} />
                    <text x={Math.max(0, tooltip.x + 18)} y={Math.max(0, tooltip.y + 28)} fill="#f8fafc">ID: {String(tooltip.row.id)}</text>
                    <text x={Math.max(0, tooltip.x + 18)} y={Math.max(0, tooltip.y + 44)} fill="#f8fafc">Area: {tooltip.row.areaPx ?? "-"}</text>
                    <text x={Math.max(0, tooltip.x + 18)} y={Math.max(0, tooltip.y + 60)} fill="#f8fafc">Diameter: {tooltip.row.equivalentDiameter?.toFixed?.(2) ?? "-"}</text>
                    <text x={Math.max(0, tooltip.x + 18)} y={Math.max(0, tooltip.y + 76)} fill="#f8fafc">
                      {tooltip.rejected ? `Rejected: ${tooltip.row.rejectionReason || "yes"}` : `Circularity: ${tooltip.row.circularity != null ? tooltip.row.circularity.toFixed(3) : "-"}`}
                    </text>
                  </g>
                )}
              </svg>
            )}
          </div>
        ) : (
          <div className="empty-state">
            <strong>No source image available.</strong>
            <small>The blob overlay needs the RGB frame the detector ran on.</small>
          </div>
        )}
        {acceptedCount === 0 && (
          <div className="empty-state">
            <strong>{rejectedCount > 0 ? "All components were rejected." : "No blob candidates found."}</strong>
            <small>Try lowering `min_area`, lowering `min_circularity`, disabling border rejection, or reducing segmentation cleanup.</small>
          </div>
        )}
      </section>
      <div className="blob-set-viewer-table table-wrap">
        <table className="compact-candidate-table">
          <thead>
            <tr>
              <th>ID</th><th>Area px</th><th>Centroid X,Y</th><th>BBox</th><th>Perimeter</th><th>Eq. diameter</th><th>Circularity</th><th>Aspect</th><th>Solidity</th><th>Border</th><th>Status</th>
            </tr>
          </thead>
          <tbody>
            {tableRows.map((row, rowIndex) => {
              const id = Number(row.id);
              const isRejected = rejected.some((item) => Number(item.id) === id);
              const selected = selectedCandidateId != null && selectedCandidateId === id;
              const hovered = hoveredCandidateId != null && hoveredCandidateId === id;
              return (
                <tr
                  key={`blob_row_${id}_${rowIndex}`}
                  className={`${selected ? "selected-object-row" : hovered ? "hovered-object-row" : ""}${isRejected ? " rejected-candidate-row" : ""}`}
                  onClick={() => onSelectCandidate(id)}
                  onMouseEnter={() => onHoverCandidate(id)}
                  onMouseLeave={() => onHoverCandidate(null)}
                >
                  <td>{id}</td>
                  <td>{fmtNum(row.areaPx, 0)}</td>
                  <td>{fmtCentroid(row.centroid)}</td>
                  <td>{fmtBbox(row.bbox)}</td>
                  <td>{fmtNum(row.perimeter, 1)}</td>
                  <td>{fmtNum(row.equivalentDiameter, 2)}</td>
                  <td>{fmtNum(row.circularity, 3)}</td>
                  <td>{fmtNum(row.aspectRatio, 2)}</td>
                  <td>{fmtNum(row.solidity, 3)}</td>
                  <td>{row.touchesBorder ? "yes" : "no"}</td>
                  <td>{isRejected ? (row.rejectionReason ?? "rejected") : (row.role ?? "kept")}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {tableRows.length === 0 && <div className="empty-state"><strong>No candidates to list.</strong></div>}
      </div>
    </div>
  );
}
