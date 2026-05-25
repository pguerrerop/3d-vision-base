import ProjectionArtifactViewer from "./ProjectionArtifactViewer";
import { fileUrl } from "../api/client";
import { useEffect, useMemo, useRef, useState } from "react";
import { overlayDebugInfo, resolveOverlayTarget, visibleOverlays, type OverlayDebugInfo, type OverlayDisplayMode } from "./overlayModel";
import { type StudioArtifact } from "./studioWorkspaceModel";
import { objectDisplayId } from "./objectSelectionModel";
import type { RoiPolygon, RoiRect } from "./roiMapping";
import type { RoiType } from "./ProjectionArtifactViewer";
import type { StudioDebugMode } from "./studioDebugMode";
import type { HoverInspectionSample } from "./hoverInspectionModel";
import { inferColorMapTFromRgb, scalarToRgb } from "./colormap";
import HeightLegend from "./HeightLegend";
import { assertSemanticMatch, resolveHeightArtifact, resolveHeightColorMapping, toHeightLegendSemantic, type HeightColorMapping } from "./heightSemantics";
type HeightRange = { min: number; max: number; unit: string };
type DisplayRangePreset = "auto_semantic" | "fixed_semantic" | "percentile";
type RangeSource = "persisted_render_context" | "artifact_metadata" | "semantic_fallback";
type HeightViewMode = "raw_sensor_z" | "plane_signed_distance" | "height_above_belt" | "preview_normalized";
type HoverParityStatus = "PERFECT_MATCH" | "VALUE_MISMATCH" | "COORDINATE_MISMATCH" | "NORMALIZATION_MISMATCH" | "INTERPOLATION_MISMATCH";
type ViewerRenderContext = {
  artifactId?: string | null;
  activeViewId?: string | null;
  selectedStageId?: string | null;
  displaySourceId?: string | null;
  numericSourceId?: string | null;
  sourceArtifactId: string;
  width: number;
  height: number;
  sourceWidth: number;
  sourceHeight: number;
  roiPurpose: "plane_fit_only" | "render_crop";
  renderCoordinateSpace: "full_frame" | "roi_crop";
  units: string;
  valueKind: "raw_z" | "normalized_height" | "unknown";
  roi: RoiRect | null;
  normalization: { vmin: number; vmax: number; displaySemantic: string };
  normalizationRanges: {
    actualDataMin: number | null;
    actualDataMax: number | null;
    displayMin: number | null;
    displayMax: number | null;
    percentileMin: number | null;
    percentileMax: number | null;
    autoSemanticMin: number | null;
    autoSemanticMax: number | null;
    colorbarMin: number | null;
    colorbarMax: number | null;
    reconstructionMin: number | null;
    reconstructionMax: number | null;
  };
  rangeSource: RangeSource;
  displayPreset: DisplayRangePreset;
  clipping: { enabled: true; mode: "percentile" | "fixed_semantic" | "auto_semantic" };
  colormap: { name: string; invalidColor: [number, number, number] };
  coordinateChain: "display_px -> source_px";
  interpolationMode: string;
  sourceArrayRef: Float32Array;
  sampler: {
    sampleValue: (x: number, y: number) => number | null;
    sampleValid: (x: number, y: number) => boolean;
    sampleRgbFromValue: (x: number, y: number) => [number, number, number];
    // Dense fallback used by hover so isolated invalid speckle pixels inside
    // an otherwise valid region don't cause "alternating no-Z" UX. When
    // (x, y) is invalid, walks outward up to maxRadius pixels and returns the
    // closest valid sample (with a flag indicating it was interpolated).
    sampleValueDense: (
      x: number,
      y: number,
      maxRadius?: number,
    ) => { value: number | null; valid: boolean; interpolated: boolean; sourceX: number; sourceY: number };
  };
};

type Props = {
  artifacts: StudioArtifact[];
  takeId?: string | null;
  selectedArtifactId: string | null;
  onSelect: (artifactId: string) => void;
  detailJson?: unknown;
  selectedObjectId?: number | null;
  hoveredObjectId?: number | null;
  hoveredArtifactId?: string | null;
  onSelectObject?: (objectId: number) => void;
  onHoverObject?: (objectId: number | null) => void;
  onHoverArtifact?: (artifactId: string | null) => void;
  onOverlayDebug?: (info: OverlayDebugInfo) => void;
  roi?: RoiRect | null;
  roiPolygon?: RoiPolygon | null;
  roiType?: RoiType;
  roiEnabled?: boolean;
  onApplyRoi?: (roi: RoiRect, rerun: boolean) => void;
  onApplyPolygonRoi?: (points: RoiPolygon, rerun: boolean) => void;
  onClearRoi?: () => void;
  roiEditModeActive?: boolean;
  roiEditStatus?: "idle" | "active" | "unsaved" | "saved" | "rerun_required";
  onCancelRoiEdit?: () => void;
  debugMode?: StudioDebugMode;
  onHoverSample?: (sample: HoverInspectionSample | null) => void;
};

export default function StudioArtifactExplorer({
  artifacts,
  takeId,
  selectedArtifactId,
  onSelect,
  detailJson,
  selectedObjectId,
  hoveredObjectId = null,
  hoveredArtifactId = null,
  onSelectObject,
  onHoverObject,
  onHoverArtifact,
  onOverlayDebug,
  roi = null,
  roiPolygon = null,
  roiType = "rectangle",
  roiEnabled = false,
  onApplyRoi,
  onApplyPolygonRoi,
  onClearRoi,
  roiEditModeActive = false,
  roiEditStatus = "idle",
  onCancelRoiEdit,
  debugMode = "operator",
  onHoverSample,
}: Props) {
  const [overlayMode, setOverlayMode] = useState<OverlayDisplayMode>("all");
  const [sourceHeightRange, setSourceHeightRange] = useState<HeightRange | null>(null);
  const [drawRoiMode, setDrawRoiMode] = useState(false);
  const [draftRoi, setDraftRoi] = useState<RoiRect | null>(null);
  const [draftPolygon, setDraftPolygon] = useState<RoiPolygon | null>(null);
  const [localRoiType, setLocalRoiType] = useState<RoiType>(roiType);
  const [numericHover, setNumericHover] = useState<{
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
  } | null>(null);
  const [displayPreset, setDisplayPreset] = useState<DisplayRangePreset>("auto_semantic");
  const [heightViewMode, setHeightViewMode] = useState<HeightViewMode>("preview_normalized");
  const [hoverInspectorMode, setHoverInspectorMode] = useState<"compact" | "debug">("compact");
  const [localHoverSample, setLocalHoverSample] = useState<HoverInspectionSample | null>(null);
  const [hoverReconBusy, setHoverReconBusy] = useState(false);
  const [hoverParityStatus, setHoverParityStatus] = useState<HoverParityStatus | null>(null);
  const [hoverReconMetrics, setHoverReconMetrics] = useState<Record<string, unknown> | null>(null);
  const [hoverReconError, setHoverReconError] = useState<string | null>(null);
  const [hoverReconArtifacts, setHoverReconArtifacts] = useState<{
    reconNearestUrl: string;
    reconBilinearUrl: string;
    diffNearestUrl: string;
    diffBilinearUrl: string;
    diffHeatUrl: string;
    diffMaskUrl: string;
    scalarDiffUrl: string;
    debugUrl: string;
  } | null>(null);
  const [numericSourceWarning, setNumericSourceWarning] = useState<string | null>(null);
  const [semanticMismatchWarning, setSemanticMismatchWarning] = useState<string | null>(null);
  const hoverRafRef = useRef<number | null>(null);
  const pendingHoverSampleRef = useRef<HoverInspectionSample | null>(null);
  useEffect(() => {
    setLocalRoiType(roiType);
  }, [roiType]);
  useEffect(() => {
    setDrawRoiMode(roiEditModeActive);
  }, [roiEditModeActive]);
  if (!artifacts.length) {
    return <div className="empty-state">No artifacts generated for this context yet.</div>;
  }
  const selected = artifacts.find((artifact) => artifact.artifact_id === selectedArtifactId) ?? artifacts[0];
  const selectedOverlayDebug = selected.kind === "overlay" ? overlayDebugInfo(selected, artifacts, { width: 960, height: 540 }) : null;
  const targetResolution = resolveOverlayTarget(artifacts, selected);
  const previewTarget = targetResolution.target;
  const previewCacheKey = String(
    selected.created_at
    ?? previewTarget?.created_at
    ?? `${takeId ?? "take"}:${previewTarget?.artifact_id ?? selected.artifact_id}:${previewTarget?.path ?? selected.path ?? "preview"}`
  );
  const selectedOverlayId = selected.kind === "overlay" ? selected.artifact_id : null;
  const morphDebug = artifacts.find((item) => item.artifact_id === "morphology_debug_json");
  const morphMetrics = artifacts.find((item) => item.artifact_id === "morphology_metrics");
  const debugMeta = (morphDebug?.metadata ?? {}) as Record<string, unknown>;
  const metricsMeta = (morphMetrics?.metadata ?? {}) as Record<string, unknown>;
  const morph = (debugMeta.morphology as Record<string, unknown> | undefined) ?? {};
  const threshold = (debugMeta.threshold as Record<string, unknown> | undefined) ?? {};
  const roiInfo = (debugMeta.roi as Record<string, unknown> | undefined) ?? {};
  const components = Number(morph.components_after ?? metricsMeta.connected_components ?? 0);
  const cleanedCoverage = Number(morph.cleaned_foreground_coverage ?? metricsMeta.cleaned_foreground_coverage ?? 0);
  const thresholdCoverage = Number(threshold.foreground_coverage ?? metricsMeta.threshold_foreground_coverage ?? 0);
  const emptyCleaned = Number(morph.cleaned_foreground_pixels ?? metricsMeta.cleaned_foreground_pixels ?? 0) <= 0;
  const overlays = useMemo(
    () => visibleOverlays(targetResolution.overlays, overlayMode, selectedObjectId ?? null),
    [overlayMode, selectedObjectId, targetResolution.overlays]
  );
  const allowRoiEditing = selected.stage_id === "detect_belt_plane";
  const activeViewId = String(previewTarget?.artifact_id ?? selected.artifact_id ?? "");
  const selectedStageId = selected.stage_id ?? null;
  const displaySourceId = previewTarget?.artifact_id ? String(previewTarget.artifact_id) : null;
  const primaryHoverArtifactId = String(previewTarget?.artifact_id ?? selected.artifact_id ?? "");
  useEffect(() => {
    const id = String(selected.artifact_id ?? "");
    if (id.includes("raw_heightmap") || id.includes("heightmap")) setHeightViewMode("raw_sensor_z");
    else if (id.includes("residual")) setHeightViewMode("plane_signed_distance");
    else if (id.includes("normalized")) setHeightViewMode("height_above_belt");
    else setHeightViewMode("preview_normalized");
  }, [selected.artifact_id]);
  const heightLikeImage = (previewTarget?.kind === "image" && isHeightLikeArtifact(previewTarget)) || (selected.kind === "image" && isHeightLikeArtifact(selected));
  const hoverValueKind: "raw_z" | "normalized_height" | "unknown" = primaryHoverArtifactId.includes("normalized")
    ? "normalized_height"
    : primaryHoverArtifactId.includes("heightmap") || primaryHoverArtifactId.includes("raw_height")
      ? "raw_z"
      : "unknown";
  const worldCalibration = useMemo(() => {
    const src = (previewTarget?.metadata ?? selected.metadata ?? {}) as Record<string, unknown>;
    const fromProjection = ((previewTarget?.projection_coordinate_system as Record<string, unknown> | undefined)
      ?? (previewTarget?.projection_metadata as Record<string, unknown> | undefined)?.coordinate_system
      ?? {}) as Record<string, unknown>;
    const originX = toFinite(src.origin_x_mm) ?? toFinite((fromProjection.origin as unknown[] | undefined)?.[0]);
    const originY = toFinite(src.origin_y_mm) ?? toFinite((fromProjection.origin as unknown[] | undefined)?.[1]);
    const resolutionX = toFinite(src.x_resolution_mm) ?? toFinite(src.pixel_size_x_mm);
    const resolutionY = toFinite(src.y_resolution_mm) ?? toFinite(src.pixel_size_y_mm);
    if (originX == null || originY == null || resolutionX == null || resolutionY == null) return null;
    return { originX, originY, resolutionX, resolutionY };
  }, [previewTarget?.metadata, previewTarget?.projection_coordinate_system, previewTarget?.projection_metadata, selected.metadata]);
  const resolvedHeightRange = useMemo(() => resolveHeightRange(selected, artifacts), [selected, artifacts]);
  const heightRange = resolvedHeightRange ?? sourceHeightRange;
  const display = useMemo(() => displayTransform(artifacts), [artifacts]);
  const persistedRenderContext = useMemo(() => renderContextTransform(artifacts), [artifacts]);
  const selectedSemanticField = useMemo(() => {
    const selectedMeta = (selected.metadata ?? {}) as Record<string, unknown>;
    const previewMeta = (previewTarget?.metadata ?? {}) as Record<string, unknown>;
    const fromDisplayRaw = display ? String(display.semantic_field ?? display.display_source_field ?? "") : "";
    const fromDisplay = fromDisplayRaw === "preview_normalized" ? "" : fromDisplayRaw;
    const fromPreview = String(previewMeta.semantic_field ?? previewMeta.display_semantic ?? "");
    const fromSelected = String(selectedMeta.semantic_field ?? selectedMeta.display_semantic ?? "");
    return (fromPreview || fromSelected || fromDisplay || "").trim() || null;
  }, [display, previewTarget?.metadata, selected.metadata]);
  const legendSemantic = useMemo(() => {
    const selectedMeta = (selected.metadata ?? {}) as Record<string, unknown>;
    const previewMeta = (previewTarget?.metadata ?? {}) as Record<string, unknown>;
    return (
      toHeightLegendSemantic(display)
      ?? toHeightLegendSemantic(previewMeta)
      ?? toHeightLegendSemantic(selectedMeta)
    );
  }, [display, previewTarget?.metadata, selected.metadata]);
  // Canonical color mapping: a single contract shared by image renderer,
  // HeightLegend, hover sampler and debug reconstruction. Resolved with a
  // strict priority chain (artifact_metadata > render_context > legacy_fallback).
  const colorMappingResolution = useMemo(
    () => resolveHeightColorMapping(artifacts, {
      semanticField: selectedSemanticField ?? legendSemantic?.semanticField ?? "height_above_belt",
      preferredArtifactId: previewTarget?.artifact_id ?? selected.artifact_id ?? null,
    }),
    [artifacts, selectedSemanticField, legendSemantic?.semanticField, previewTarget?.artifact_id, selected.artifact_id],
  );
  const heightColorMapping: HeightColorMapping = colorMappingResolution.mapping;
  const rangeResolution = useMemo(() => {
    // The shared HeightColorMapping is the single authority for renderer/legend
    // min/max so the colorbar and the rendered preview can never diverge.
    const mappingSource: RangeSource = heightColorMapping.source === "render_context"
      ? "persisted_render_context"
      : heightColorMapping.source === "artifact_metadata"
        ? "artifact_metadata"
        : "semantic_fallback";
    return {
      range: { min: heightColorMapping.colorScaleMin, max: heightColorMapping.colorScaleMax },
      source: mappingSource,
    };
  }, [heightColorMapping]);
  const effectiveDisplayRange = rangeResolution.range;
  const rangeSource = rangeResolution.source;
  // Canonical colormap name: derived from the shared HeightColorMapping so the
  // legend, renderer and hover diffing never disagree on which LUT is active.
  const effectiveColormapId = String(heightColorMapping.colorMap ?? "turbo");
  const effectiveInterpolationMode = String(
    persistedRenderContext?.interpolation_mode
    ?? display?.interpolation_mode
    ?? "nearest"
  );
  const missingAuthoritativeRenderContext = useMemo(
    () => Boolean(selected.artifact_id === "normalized_heightmap" && previewTarget?.path && !persistedRenderContext),
    [selected.artifact_id, previewTarget?.path, persistedRenderContext],
  );
  const effectiveNumericHover = useMemo(() => {
    if (!numericHover) return null;
    return { ...numericHover, displayVmin: effectiveDisplayRange.min, displayVmax: effectiveDisplayRange.max };
  }, [numericHover, effectiveDisplayRange.min, effectiveDisplayRange.max]);
  const normHist = useMemo(() => histogramArtifact(artifacts), [artifacts]);
  const planeDebug = useMemo(() => planeDebugArtifact(artifacts), [artifacts]);
  const engineering = debugMode === "engineering";
  const viewerRenderContext = useMemo<ViewerRenderContext | null>(() => {
    if (!effectiveNumericHover) return null;
    const sourceWidth = effectiveNumericHover.width;
    const sourceHeight = effectiveNumericHover.height;
    const width = effectiveNumericHover.renderedWidth > 0 ? effectiveNumericHover.renderedWidth : sourceWidth;
    const height = effectiveNumericHover.renderedHeight > 0 ? effectiveNumericHover.renderedHeight : sourceHeight;
    const vmin = effectiveNumericHover.displayVmin;
    const vmax = effectiveNumericHover.displayVmax;
    const source = effectiveNumericHover.normalizedValues;
    const validMask = effectiveNumericHover.validMask ?? null;
    const actualDataRange = computeFiniteRange(source, validMask);
    const displayMin = toFinite(display?.["display_vmin"] ?? display?.["color_scale_min"]);
    const displayMax = toFinite(display?.["display_vmax"] ?? display?.["color_scale_max"]);
    const clippingPercentiles = Array.isArray(persistedRenderContext?.["clipping_percentiles"])
      ? (persistedRenderContext?.["clipping_percentiles"] as unknown[])
      : null;
    const percentileMin = toFinite(clippingPercentiles?.[0] ?? display?.["p2"]);
    const percentileMax = toFinite(clippingPercentiles?.[1] ?? display?.["p98"]);
    const autoSemanticMin = toFinite(display?.["auto_semantic_min"] ?? display?.["semantic_min"] ?? display?.["auto_vmin"]);
    const autoSemanticMax = toFinite(display?.["auto_semantic_max"] ?? display?.["semantic_max"] ?? display?.["auto_vmax"]);
    const mapDisplayToSource = (x: number, y: number) => {
      if (effectiveNumericHover.renderCoordinateSpace === "roi_crop") {
        const sx = Math.max(0, Math.min(sourceWidth - 1, Math.round((x / Math.max(1, width - 1)) * Math.max(1, sourceWidth - 1))));
        const sy = Math.max(0, Math.min(sourceHeight - 1, Math.round((y / Math.max(1, height - 1)) * Math.max(1, sourceHeight - 1))));
        return { x: sx, y: sy };
      }
      if (sourceWidth === width && sourceHeight === height) return { x, y };
      const sx = Math.max(0, Math.min(sourceWidth - 1, Math.round((x / Math.max(1, width - 1)) * Math.max(1, sourceWidth - 1))));
      const sy = Math.max(0, Math.min(sourceHeight - 1, Math.round((y / Math.max(1, height - 1)) * Math.max(1, sourceHeight - 1))));
      return { x: sx, y: sy };
    };
    const sampleValue = (x: number, y: number): number | null => {
      if (x < 0 || y < 0 || x >= width || y >= height) return null;
      const sourcePx = mapDisplayToSource(x, y);
      return Number(source[(sourcePx.y * sourceWidth) + sourcePx.x]);
    };
    const sampleValid = (x: number, y: number): boolean => {
      if (x < 0 || y < 0 || x >= width || y >= height) return false;
      const sourcePx = mapDisplayToSource(x, y);
      return validMask ? validMask[(sourcePx.y * sourceWidth) + sourcePx.x] > 0 : true;
    };
    const sampleRgbFromValue = (x: number, y: number): [number, number, number] => {
      const valid = sampleValid(x, y);
      const value = sampleValue(x, y);
      return scalarToRgb(heightColorMapping.colorMap, value, vmin, vmax, valid);
    };
    // Dense hover fallback: searches outward up to maxRadius source pixels for
    // the nearest valid sample. This eliminates the "alternating no-Z" hover
    // UX caused by isolated invalid speckle pixels inside the body, while
    // keeping the *true* invalid background (outside the object) reported as
    // invalid because the search radius is small.
    const sampleValueDense = (
      x: number,
      y: number,
      maxRadius: number = 3,
    ): { value: number | null; valid: boolean; interpolated: boolean; sourceX: number; sourceY: number } => {
      if (x < 0 || y < 0 || x >= width || y >= height) {
        return { value: null, valid: false, interpolated: false, sourceX: x, sourceY: y };
      }
      const origin = mapDisplayToSource(x, y);
      const originIdx = (origin.y * sourceWidth) + origin.x;
      const originValid = validMask ? validMask[originIdx] > 0 : true;
      if (originValid) {
        return {
          value: Number(source[originIdx]),
          valid: true,
          interpolated: false,
          sourceX: origin.x,
          sourceY: origin.y,
        };
      }
      // Expanding square ring search; favors the nearest valid neighbor.
      for (let r = 1; r <= maxRadius; r += 1) {
        let bestDist = Number.POSITIVE_INFINITY;
        let bestIdx = -1;
        let bestSx = origin.x;
        let bestSy = origin.y;
        for (let dy = -r; dy <= r; dy += 1) {
          for (let dx = -r; dx <= r; dx += 1) {
            if (Math.max(Math.abs(dx), Math.abs(dy)) !== r) continue;
            const sx = origin.x + dx;
            const sy = origin.y + dy;
            if (sx < 0 || sy < 0 || sx >= sourceWidth || sy >= sourceHeight) continue;
            const idx = (sy * sourceWidth) + sx;
            if (validMask && validMask[idx] <= 0) continue;
            const d = (dx * dx) + (dy * dy);
            if (d < bestDist) {
              bestDist = d;
              bestIdx = idx;
              bestSx = sx;
              bestSy = sy;
            }
          }
        }
        if (bestIdx >= 0) {
          return {
            value: Number(source[bestIdx]),
            valid: true,
            interpolated: true,
            sourceX: bestSx,
            sourceY: bestSy,
          };
        }
      }
      return { value: null, valid: false, interpolated: false, sourceX: origin.x, sourceY: origin.y };
    };
    return {
      artifactId: selected.artifact_id,
      activeViewId,
      selectedStageId,
      displaySourceId,
      numericSourceId: effectiveNumericHover.sourceArtifactId,
      sourceArtifactId: effectiveNumericHover.sourceArtifactId,
      width,
      height,
      sourceWidth,
      sourceHeight,
      roiPurpose: effectiveNumericHover.roiPurpose,
      renderCoordinateSpace: effectiveNumericHover.renderCoordinateSpace,
      units: effectiveNumericHover.units,
      valueKind: hoverValueKind,
      roi: roi ?? null,
      normalization: {
        vmin,
        vmax,
        displaySemantic: String(effectiveNumericHover.displaySemantic ?? "height_above_belt"),
      },
      normalizationRanges: {
        actualDataMin: actualDataRange.min,
        actualDataMax: actualDataRange.max,
        displayMin,
        displayMax,
        percentileMin,
        percentileMax,
        autoSemanticMin,
        autoSemanticMax,
        colorbarMin: effectiveDisplayRange?.min ?? null,
        colorbarMax: effectiveDisplayRange?.max ?? null,
        reconstructionMin: vmin,
        reconstructionMax: vmax,
      },
      rangeSource,
      displayPreset,
      clipping: {
        enabled: true,
        mode: rangeSource === "semantic_fallback"
          ? displayPreset
          : ((persistedRenderContext?.clipping_mode as DisplayRangePreset | undefined) ?? "percentile"),
      },
      colormap: { name: effectiveColormapId, invalidColor: [0, 0, 0] },
      coordinateChain: "display_px -> source_px",
      interpolationMode: effectiveInterpolationMode,
      sourceArrayRef: source,
      sampler: { sampleValue, sampleValid, sampleRgbFromValue, sampleValueDense },
    };
  }, [
    effectiveNumericHover,
    selected.artifact_id,
    hoverValueKind,
    roi,
    displayPreset,
    activeViewId,
    selectedStageId,
    displaySourceId,
    display,
    effectiveDisplayRange,
    rangeSource,
    effectiveColormapId,
    effectiveInterpolationMode,
    persistedRenderContext,
    heightColorMapping,
  ]);
  const reconEligibility = useMemo(() => {
    const imageLike = Boolean(heightLikeImage && previewTarget?.path);
    const renderContext = Boolean(viewerRenderContext);
    const sampler = Boolean(viewerRenderContext?.sampler);
    const sourceArray = Boolean(viewerRenderContext?.sourceArrayRef && viewerRenderContext.sourceArrayRef.length > 0);
    let reason = "";
    if (!renderContext) reason = "No active render context";
    else if (!sampler) reason = "No sampler";
    else if (!sourceArray) reason = "No source array";
    else if (!imageLike) reason = "Artifact is not image-like";
    return {
      imageLike,
      renderContext,
      sampler,
      sourceArray,
      enabled: imageLike && renderContext && sampler && sourceArray,
      reason,
      forcedByHoverSignal: false,
    };
  }, [heightLikeImage, previewTarget?.path, viewerRenderContext]);
  const canonicalHoverState = useMemo(() => {
    if (!localHoverSample) return null;
    const hasNorm = localHoverSample.normalizedHeight != null;
    const hasRgb = Boolean(localHoverSample.coordDebug?.renderedRgb);
    const hasSrcPx = localHoverSample.coordDebug?.sourceArrayX != null && localHoverSample.coordDebug?.sourceArrayY != null;
    return {
      sample: localHoverSample,
      renderContext: viewerRenderContext,
      sampler: viewerRenderContext?.sampler ?? null,
      sourceArray: viewerRenderContext?.sourceArrayRef ?? null,
      hasNorm,
      hasRgb,
      hasSrcPx,
      fieldSources: {
        norm: viewerRenderContext ? `${viewerRenderContext.sourceArtifactId}` : "none",
        rgb: localHoverSample.coordDebug?.renderedRgb ? "previewCanvasPixelRead" : "none",
        srcPx: hasSrcPx ? "hoverCoordDebugTrace" : "none",
        imgPx: localHoverSample.coordDebug ? "hoverCoordDebugTrace" : "none",
        finalPx: localHoverSample.coordDebug ? "hoverCoordDebugTrace" : "none",
        roi: localHoverSample.coordDebug ? "hoverCoordDebugTrace" : "none",
        mode: localHoverSample.coordDebug ? "hoverCoordDebugTrace" : "none",
      },
    };
  }, [localHoverSample, viewerRenderContext]);
  const effectiveReconEligibility = useMemo(() => {
    const hoverSignalPresent = Boolean(canonicalHoverState && (canonicalHoverState.hasNorm || canonicalHoverState.hasRgb || canonicalHoverState.hasSrcPx));
    return { ...reconEligibility, hoverSignalPresent };
  }, [reconEligibility, canonicalHoverState]);
  useEffect(() => {
    if (!canonicalHoverState) return;
    if (!effectiveReconEligibility.enabled && (canonicalHoverState.hasNorm || canonicalHoverState.hasRgb || canonicalHoverState.hasSrcPx)) {
      console.error("HOVER STATE PARITY VIOLATION", {
        displayedFields: {
          norm: canonicalHoverState.hasNorm,
          rgb: canonicalHoverState.hasRgb,
          srcPx: canonicalHoverState.hasSrcPx,
        },
        viewerContextPresent: Boolean(canonicalHoverState.renderContext),
        samplerPresent: Boolean(canonicalHoverState.sampler),
        sourceArrayPresent: Boolean(canonicalHoverState.sourceArray),
        fieldSources: canonicalHoverState.fieldSources,
      });
    }
  }, [canonicalHoverState, effectiveReconEligibility.enabled]);
  useEffect(() => {
    if (!effectiveNumericHover) onHoverSample?.(null);
  }, [effectiveNumericHover, onHoverSample]);
  useEffect(() => {
    if (!missingAuthoritativeRenderContext) return;
    console.warn("Rendered preview missing authoritative normalization metadata");
  }, [missingAuthoritativeRenderContext]);
  useEffect(() => {
    if (hoverValueKind === "unknown") onHoverSample?.(null);
  }, [hoverValueKind, selected.artifact_id, previewTarget?.artifact_id, onHoverSample]);
  useEffect(() => {
    if (debugMode !== "engineering") {
      setSemanticMismatchWarning(null);
      return;
    }
    const check = assertSemanticMatch(selectedSemanticField, localHoverSample?.semanticField ?? localHoverSample?.displaySemantic);
    setSemanticMismatchWarning(check.warning);
  }, [debugMode, selectedSemanticField, localHoverSample]);
  useEffect(() => () => {
    if (hoverRafRef.current != null) {
      window.cancelAnimationFrame(hoverRafRef.current);
      hoverRafRef.current = null;
    }
  }, []);
  useEffect(() => {
    let canceled = false;
    async function loadSourceRangeFromParserMetadata() {
      setSourceHeightRange(null);
      if (!takeId) return;
      if (selected.kind !== "image") return;
      const metadata = (selected.metadata ?? {}) as Record<string, unknown>;
      const parserFile = typeof metadata.parser_metadata_file === "string" ? metadata.parser_metadata_file : "";
      if (!parserFile) return;
      try {
        const response = await fetch(fileUrl(takeId, parserFile));
        if (!response.ok) return;
        const payload = await response.json() as Record<string, unknown>;
        const range = (payload.height_preview_range_raw as Record<string, unknown> | undefined) ?? {};
        const min = toFinite(range.min);
        const max = toFinite(range.max);
        if (min == null || max == null || max <= min) return;
        if (!canceled) {
          setSourceHeightRange({
            min,
            max,
            unit: typeof range.units === "string" && range.units.trim() ? range.units : "sensor_raw",
          });
        }
      } catch {
        if (!canceled) setSourceHeightRange(null);
      }
    }
    if (resolvedHeightRange) {
      setSourceHeightRange(null);
      return () => {
        canceled = true;
      };
    }
    void loadSourceRangeFromParserMetadata();
    return () => {
      canceled = true;
    };
  }, [resolvedHeightRange, selected, takeId]);
  useEffect(() => {
    let canceled = false;
    async function loadNumericHover() {
      setNumericSourceWarning(null);
      if (!takeId || !heightLikeImage) {
        setNumericHover(null);
        return;
      }
      try {
        const selectedId = String(selected.artifact_id ?? "");
        const shape = display && Array.isArray(display["shape"]) ? display["shape"] as unknown[] : [];
        let width = Number(shape[1] ?? 0);
        let height = Number(shape[0] ?? 0);
        let renderedWidth = 0;
        let renderedHeight = 0;
        const imagePath = previewTarget?.path ?? selected.path ?? "";
        if (imagePath) {
          const size = await readImageSize(fileUrl(takeId, imagePath));
          renderedWidth = size.width;
          renderedHeight = size.height;
        }
        if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
          if (renderedWidth > 0 && renderedHeight > 0) {
            width = renderedWidth;
            height = renderedHeight;
          }
        }
        if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
          if (!canceled) setNumericSourceWarning("No numeric source loaded: unknown image dimensions.");
          return;
        }
        const semanticField = selectedSemanticField ?? (selectedId.includes("residual") ? "plane_signed_distance" : "height_above_belt");
        const rasterResolution = resolveHeightArtifact(artifacts, { semanticField, representation: "numeric_raster" });
        const candidateNumericPaths = (() => {
          if (display?.["numeric_values_path"]) return [String(display["numeric_values_path"])];
          if (typeof rasterResolution.artifact?.path === "string" && rasterResolution.artifact.path.trim()) {
            return [String(rasterResolution.artifact.path)];
          }
          if (semanticField === "raw_sensor_z") return ["raw_sensor_z.values.f32", "raw_heightmap.values.f32"];
          if (semanticField === "plane_signed_distance") return ["plane_signed_distance.values.f32", "plane_residuals.values.f32"];
          return ["height_above_belt.values.f32", "normalized_heightmap.values.f32"];
        })();
        let normalizedValues: Float32Array | null = null;
        let resolvedNumericPath: string | null = null;
        for (const p of candidateNumericPaths) {
          const resp = await fetch(fileUrl(takeId, p));
          if (!resp.ok) continue;
          const arr = new Float32Array(await resp.arrayBuffer());
          const expectedFromDisplay = Math.max(1, width * height);
          const expectedFromRendered = Math.max(1, renderedWidth * renderedHeight);
          if (arr.length === expectedFromRendered && renderedWidth > 0 && renderedHeight > 0) {
            width = renderedWidth;
            height = renderedHeight;
            normalizedValues = arr;
            resolvedNumericPath = p;
            break;
          }
          if (arr.length >= expectedFromDisplay) {
            normalizedValues = arr;
            resolvedNumericPath = p;
            break;
          }
        }
        if (!normalizedValues) {
          if (!canceled) setNumericSourceWarning(`No numeric source loaded: missing ${candidateNumericPaths.join(" | ")}.`);
          return;
        }
        let rawValues: Float32Array | null = null;
        let rawValuesWarning: string | null = null;
        let validMask: Uint8Array | null = null;
        const rawRasterResolution = resolveHeightArtifact(artifacts, { semanticField: "raw_sensor_z", representation: "numeric_raster" });
        const candidateRawPaths = Array.from(new Set([
          persistedRenderContext?.["raw_values_path"],
          display?.["raw_values_path"],
          rawRasterResolution.artifact?.path,
          "raw_sensor_z.values.f32",
          "raw_heightmap.values.f32",
        ].filter((value): value is string => typeof value === "string" && value.trim().length > 0)));
        for (const rawPath of candidateRawPaths) {
          const rResp = await fetch(fileUrl(takeId, rawPath));
          if (!rResp.ok) continue;
          const arr = new Float32Array(await rResp.arrayBuffer());
          const expectedFromDisplay = Math.max(1, width * height);
          const expectedFromRendered = Math.max(1, renderedWidth * renderedHeight);
          if (arr.length === expectedFromRendered && renderedWidth > 0 && renderedHeight > 0) {
            rawValues = arr;
            break;
          }
          if (arr.length >= expectedFromDisplay) {
            rawValues = arr;
            break;
          }
          rawValuesWarning = `Raw Z source shape mismatch: ${rawPath} has ${arr.length} samples, expected ${expectedFromDisplay}.`;
        }
        if (!rawValues && candidateRawPaths.length > 0 && !rawValuesWarning) {
          rawValuesWarning = `Raw Z source unavailable: tried ${candidateRawPaths.join(" | ")}.`;
        }
        const validMaskPath = String(display?.["valid_mask_path"] ?? "valid_mask.u8");
        if (validMaskPath) {
          const vResp = await fetch(fileUrl(takeId, validMaskPath));
          if (vResp.ok) validMask = new Uint8Array(await vResp.arrayBuffer());
        }
        const expectedCount = width * height;
        if (validMask && validMask.length < expectedCount) {
          validMask = null;
          if (!canceled) setNumericSourceWarning("Valid mask shape mismatch; ignoring mask for hover sampling.");
        } else if (validMask && validMask.length > expectedCount) {
          validMask = validMask.slice(0, expectedCount);
        }
        if (rawValues && rawValues.length < expectedCount) {
          rawValues = null;
        } else if (rawValues && rawValues.length > expectedCount) {
          rawValues = rawValues.slice(0, expectedCount);
        }
        const renderCoordinateSpace = String(persistedRenderContext?.["render_coordinate_space"] ?? display?.["render_coordinate_space"] ?? "full_frame") === "roi_crop"
          ? "roi_crop"
          : "full_frame";
        if (
          !canceled
          && renderCoordinateSpace === "full_frame"
          && renderedWidth > 0
          && renderedHeight > 0
          && (width !== renderedWidth || height !== renderedHeight)
        ) {
          console.warn("Hover numeric/rendered shape mismatch in full_frame mapping", {
            rendered_png_shape: [renderedHeight, renderedWidth],
            numeric_array_shape: [height, width],
          });
        }
        if (!canceled) {
          if (rasterResolution.warning) {
            setNumericSourceWarning(`No semantic raster metadata found; using legacy fallback (${rasterResolution.warning}).`);
          } else if (rawValuesWarning) {
            setNumericSourceWarning(rawValuesWarning);
          }
          setNumericHover({
            sourceArtifactId: String(display?.["numeric_source_artifact"] ?? rasterResolution.artifact?.artifact_id ?? (resolvedNumericPath?.includes("raw") ? "heightmap_frame.npz" : "normalized_heightmap.npz")),
            units: String(display?.["units"] ?? "mm"),
            displaySemantic: String(display?.["display_semantic"] ?? semanticField),
            semanticField: String(display?.["semantic_field"] ?? semanticField),
            width,
            height,
            renderedWidth: renderedWidth > 0 ? renderedWidth : width,
            renderedHeight: renderedHeight > 0 ? renderedHeight : height,
            roiPurpose: String(persistedRenderContext?.["roi_purpose"] ?? display?.["roi_purpose"] ?? "plane_fit_only") === "render_crop" ? "render_crop" : "plane_fit_only",
            renderCoordinateSpace,
            sourceShape: parseShapeTuple(persistedRenderContext?.["source_shape"] ?? display?.["source_shape"]),
            numericArrayShape: parseShapeTuple(persistedRenderContext?.["numeric_array_shape"] ?? display?.["numeric_array_shape"]) ?? [height, width],
            validMaskShape: parseShapeTuple(persistedRenderContext?.["valid_mask_shape"] ?? display?.["valid_mask_shape"]) ?? (validMask ? [height, width] : null),
            displayVmin: Number(display?.["display_vmin"] ?? -20),
            displayVmax: Number(display?.["display_vmax"] ?? 320),
            normalizedValues,
            rawValues,
            validMask,
          });
        }
      } catch {
        if (!canceled) {
          setNumericHover(null);
          setNumericSourceWarning("No numeric source loaded: failed to fetch numeric hover artifacts.");
        }
      }
    }
    void loadNumericHover();
    return () => {
      canceled = true;
    };
  }, [display, persistedRenderContext, heightLikeImage, takeId, selected.artifact_id, selected.path, previewTarget?.path, selectedSemanticField, artifacts]);
  useEffect(() => () => {
    if (hoverReconArtifacts) {
      URL.revokeObjectURL(hoverReconArtifacts.reconNearestUrl);
      URL.revokeObjectURL(hoverReconArtifacts.reconBilinearUrl);
      URL.revokeObjectURL(hoverReconArtifacts.diffNearestUrl);
      URL.revokeObjectURL(hoverReconArtifacts.diffBilinearUrl);
      URL.revokeObjectURL(hoverReconArtifacts.diffHeatUrl);
      URL.revokeObjectURL(hoverReconArtifacts.diffMaskUrl);
      URL.revokeObjectURL(hoverReconArtifacts.scalarDiffUrl);
      URL.revokeObjectURL(hoverReconArtifacts.debugUrl);
    }
  }, [hoverReconArtifacts]);
  useEffect(() => {
    if (!drawRoiMode || !allowRoiEditing) return;
    function onKeyDown(event: KeyboardEvent) {
      if (event.key !== "Escape") return;
      event.preventDefault();
      setDraftRoi(null);
      setDraftPolygon(null);
      onCancelRoiEdit?.();
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [drawRoiMode, allowRoiEditing, onCancelRoiEdit]);
  return (
    <div className="studio-artifact-explorer">
      <aside className="studio-artifact-nav">
        {artifacts.map((artifact) => (
          <button
            className={[
              artifact.artifact_id === selected.artifact_id ? "active" : "",
              hoveredArtifactId === artifact.artifact_id ? "overlay-linked-hover" : "",
            ].join(" ")}
            key={artifact.artifact_id}
            onClick={() => onSelect(artifact.artifact_id)}
            onMouseEnter={() => {
              onHoverObject?.(artifact.object_id ?? null);
              onHoverArtifact?.(artifact.artifact_id);
            }}
            onMouseLeave={() => {
              onHoverObject?.(null);
              onHoverArtifact?.(null);
            }}
            type="button"
          >
            <strong>{artifact.title}</strong>
            <span>{artifact.stage_id}</span>
            <small>{artifact.kind}</small>
            <small>{artifact.object_id == null ? "global" : objectDisplayId(artifact.object_id)}</small>
          </button>
        ))}
      </aside>
      <section className="studio-artifact-preview">
        {(selected.kind === "overlay" || targetResolution.overlays.length > 0) && (
          <div className="overlay-controls">
            <button className={overlayMode === "all" ? "active" : ""} onClick={() => setOverlayMode("all")} type="button">Show all overlays</button>
            <button className={overlayMode === "selected_only" ? "active" : ""} onClick={() => setOverlayMode("selected_only")} type="button">Selected object only</button>
            <button className={overlayMode === "hide" ? "active" : ""} onClick={() => setOverlayMode("hide")} type="button">Hide overlays</button>
          </div>
        )}
        {!!targetResolution.warning && (
          <div className="artifact-warning-banner">
            {targetResolution.warning}
          </div>
        )}
        {!!selectedOverlayDebug?.warnings.length && (
          <div className="artifact-warning-banner">
            {selectedOverlayDebug.warnings.join(" | ")}
          </div>
        )}
        {selectedOverlayDebug?.approximate && (
          <div className="artifact-warning-banner approximate">
            APPROXIMATE
          </div>
        )}
        {(selected.kind === "image" || selected.kind === "overlay") && previewTarget?.path && takeId && (
          <div className="height-image-wrap">
            <ProjectionArtifactViewer
              title={previewTarget.title}
              src={`${fileUrl(takeId, previewTarget.path)}?t=${encodeURIComponent(previewCacheKey)}`}
              overlays={overlays}
              selectedObjectId={selectedObjectId ?? null}
              hoveredObjectId={hoveredObjectId}
              selectedOverlayArtifactId={selectedOverlayId}
              displayMode={overlayMode}
              onSelectObject={onSelectObject}
              onSelectOverlayArtifact={onSelect}
              onHoverObject={onHoverObject}
              onHoverOverlayArtifact={onHoverArtifact}
              onOverlayDebug={onOverlayDebug}
              artifacts={artifacts}
              emptyMessage="Artifact file is unavailable."
              roi={(localRoiType === "rectangle" || localRoiType === "full_height_x_band") ? (draftRoi ?? (roiEnabled && roiType === localRoiType ? roi : null)) : null}
              roiPolygon={localRoiType === "polygon" ? (draftPolygon ?? (roiEnabled && roiType === "polygon" ? roiPolygon : null)) : null}
              roiType={localRoiType}
              roiLabel={localRoiType === "full_height_x_band" ? "Vertical band ROI" : "ROI"}
              drawRoiMode={drawRoiMode}
              onRoiDrawComplete={(next) => setDraftRoi(next)}
              onPolygonRoiComplete={(points) => setDraftPolygon(points)}
              engineeringMode={engineering}
              metricRange={heightRange ? { min: heightRange.min, max: heightRange.max, unit: heightRange.unit } : null}
              hoverValueKind={hoverValueKind}
              worldCalibration={worldCalibration}
              onHoverSample={(sample) => {
                pendingHoverSampleRef.current = sample;
                if (hoverRafRef.current == null) {
                  hoverRafRef.current = window.requestAnimationFrame(() => {
                    hoverRafRef.current = null;
                    const next = pendingHoverSampleRef.current;
                    setLocalHoverSample(next);
                    onHoverSample?.(next);
                  });
                }
              }}
              numericHover={effectiveNumericHover ? {
                ...effectiveNumericHover,
                // Wire the dense (nearest-valid-neighbor) hover sampler so
                // isolated invalid speckle pixels inside the body no longer
                // alternate to "no Z" while moving the cursor.
                denseSample: viewerRenderContext?.sampler.sampleValueDense,
              } : null}
            />
            {heightLikeImage && engineering && (
              <aside className="height-lateral-stack" aria-label="Height tools">
                <div className="height-colorbar-compact" aria-label="Height scale">
                  {!legendSemantic?.semanticField && heightColorMapping.source === "legacy_fallback" && (
                    <small className="hover-recon-error">Legacy height preview: semantic field unknown.</small>
                  )}
                  {colorMappingResolution.warning && (
                    <small className="hover-recon-error">{colorMappingResolution.warning}</small>
                  )}
                {selected.artifact_id === "normalized_heightmap" && (
                  <div className="height-colorbar-compact-controls">
                    <label>
                      View mode
                      <select
                        value={heightViewMode}
                        onChange={(event) => {
                          const mode = event.target.value as HeightViewMode;
                          setHeightViewMode(mode);
                          const candidates = artifacts.filter((a) => a.kind === "image");
                          const pick = mode === "raw_sensor_z"
                            ? candidates.find((a) => String(a.artifact_id).includes("raw_heightmap") || String(a.artifact_id) === "heightmap")
                            : mode === "plane_signed_distance"
                              ? candidates.find((a) => String(a.artifact_id).includes("residual"))
                              : mode === "height_above_belt"
                                ? candidates.find((a) => String(a.artifact_id).includes("normalized_heightmap"))
                                : candidates.find((a) => String(a.artifact_id).includes("normalized")) ?? selected;
                          if (pick) onSelect(pick.artifact_id);
                        }}
                      >
                        <option value="raw_sensor_z">Raw sensor Z</option>
                        <option value="plane_signed_distance">Plane signed distance</option>
                        <option value="height_above_belt">Height above belt</option>
                        <option value="preview_normalized">Preview normalized / debug</option>
                      </select>
                    </label>
                    <label>
                      Range
                      <select
                        value={displayPreset}
                        onChange={(event) => setDisplayPreset(event.target.value as DisplayRangePreset)}
                        disabled={rangeSource !== "semantic_fallback"}
                        title={rangeSource !== "semantic_fallback" ? "Range locked to persisted render metadata" : "Using semantic fallback range"}
                      >
                        <option value="auto_semantic">Auto semantic</option>
                        <option value="fixed_semantic">Fixed semantic</option>
                        <option value="percentile">Percentile (p2-p98)</option>
                      </select>
                    </label>
                    <small>
                      Range source: {rangeSource}
                    </small>
                    {rangeSource === "semantic_fallback" ? (
                      <small className="hover-recon-error">Warning: semantic fallback normalization in use.</small>
                    ) : null}
                    {missingAuthoritativeRenderContext ? (
                      <small className="hover-recon-error">Rendered preview missing authoritative normalization metadata</small>
                    ) : null}
                  </div>
                )}
                <HeightLegend
                  colorMapping={heightColorMapping}
                  semanticField={legendSemantic?.semanticField ?? heightColorMapping.semanticField}
                  semanticLabel={legendSemantic?.semanticLabel}
                  positiveDirection={legendSemantic?.positiveDirection}
                  authoritative={legendSemantic?.authoritative}
                  compact
                  showTicks
                  percentileLabels={legendSemantic?.percentileLabels}
                />
                </div>
                <div className={`height-hover-lateral ${hoverInspectorMode === "debug" ? "debug" : "compact"}`} aria-live="polite">
                  <div className="height-hover-lateral-controls">
                    <button type="button" className={hoverInspectorMode === "compact" ? "active" : ""} onClick={() => setHoverInspectorMode("compact")}>Compact</button>
                    <button type="button" className={hoverInspectorMode === "debug" ? "active" : ""} onClick={() => setHoverInspectorMode("debug")}>Debug</button>
                  </div>
                  {semanticMismatchWarning && hoverInspectorMode === "debug" && (
                    <small className="hover-recon-error">{semanticMismatchWarning}</small>
                  )}
                  {hoverInspectorMode === "debug" && (
                    <div className="height-hover-lateral-controls">
                      <button
                        type="button"
                        disabled={hoverReconBusy || !effectiveReconEligibility.enabled || !takeId}
                        title={!effectiveReconEligibility.enabled ? effectiveReconEligibility.reason : "Generate hover reconstruction diagnostics"}
                        onClick={() => void generateHoverReconstruction({
                          takeId,
                          imagePath: previewTarget?.path ?? "",
                          viewerContext: viewerRenderContext,
                          selectionMetadata: {
                            activeViewId,
                            selectedStageId,
                            displaySourceId,
                            numericSourceId: effectiveNumericHover?.sourceArtifactId ?? null,
                          },
                          onResult: (result) => {
                            setHoverReconError(null);
                            setHoverReconMetrics(result.metrics);
                            setHoverParityStatus(result.status);
                            setHoverReconArtifacts(result.artifacts);
                          },
                          onBusy: setHoverReconBusy,
                          onError: (message) => setHoverReconError(message),
                        })}
                      >
                        {hoverReconBusy ? "Reconstructing..." : "Generate hover reconstruction"}
                      </button>
                    </div>
                  )}
                  {hoverInspectorMode === "debug" && hoverParityStatus && (
                    <div className={`hover-parity-badge ${hoverParityStatus === "PERFECT_MATCH" ? "ok" : "warn"}`}>{hoverParityStatus}</div>
                  )}
                  {hoverInspectorMode === "debug" && (
                    <div className="hover-eligibility">
                      <small>imageLike: {effectiveReconEligibility.imageLike ? "yes" : "no"}</small>
                      <small>renderContext: {effectiveReconEligibility.renderContext ? "yes" : "no"}</small>
                      <small>sampler: {effectiveReconEligibility.sampler ? "yes" : "no"}</small>
                      <small>sourceArray: {effectiveReconEligibility.sourceArray ? "yes" : "no"}</small>
                      <small>reconstructionEligible: {effectiveReconEligibility.enabled ? "yes" : "no"}</small>
                      <small>rangeSource: {rangeSource}</small>
                      <small>colormap: {effectiveColormapId}</small>
                      <small>interpolation: {effectiveInterpolationMode}</small>
                      <small>renderCoordinateSpace: {effectiveNumericHover?.renderCoordinateSpace ?? "-"}</small>
                      <small>roiPurpose: {effectiveNumericHover?.roiPurpose ?? "-"}</small>
                      <small>rendered_png_shape: {effectiveNumericHover ? `${effectiveNumericHover.renderedHeight}x${effectiveNumericHover.renderedWidth}` : "-"}</small>
                      <small>numeric_array_shape: {effectiveNumericHover ? `${effectiveNumericHover.height}x${effectiveNumericHover.width}` : "-"}</small>
                      <small>hoverDensityFallback: nearest-valid (r=3)</small>
                      <small>colormap_lut: cv2.COLORMAP_TURBO (256 entries)</small>
                      {localHoverSample?.coordDebug?.interpolatedFromNearest ? (
                        <small>interpolated: yes (snapped to nearest valid)</small>
                      ) : null}
                      <small>hoverSignalPresent: {effectiveReconEligibility.hoverSignalPresent ? "yes" : "no"}</small>
                      <small>selected: {selected?.artifact_id ?? "-"}</small>
                      {!effectiveReconEligibility.enabled ? <small>disabled: {effectiveReconEligibility.reason}</small> : <small>reason: {effectiveReconEligibility.reason || "-"}</small>}
                      {numericSourceWarning ? <small>numericSourceWarning: {numericSourceWarning}</small> : null}
                    </div>
                  )}
                  {hoverInspectorMode === "debug" && hoverReconArtifacts && (
                    <div className="hover-recon-links">
                      <a href={hoverReconArtifacts.reconNearestUrl} download="reconstructed_hover_image_nearest.png">reconstructed_hover_image_nearest.png</a>
                      <a href={hoverReconArtifacts.reconBilinearUrl} download="reconstructed_hover_image_bilinear.png">reconstructed_hover_image_bilinear.png</a>
                      <a href={hoverReconArtifacts.diffNearestUrl} download="hover_vs_render_diff_nearest.png">hover_vs_render_diff_nearest.png</a>
                      <a href={hoverReconArtifacts.diffBilinearUrl} download="hover_vs_render_diff_bilinear.png">hover_vs_render_diff_bilinear.png</a>
                      <a href={hoverReconArtifacts.diffHeatUrl} download="hover_vs_render_diff_heatmap.png">hover_vs_render_diff_heatmap.png</a>
                      <a href={hoverReconArtifacts.diffMaskUrl} download="hover_vs_render_coord_mismatch_mask.png">hover_vs_render_coord_mismatch_mask.png</a>
                      <a href={hoverReconArtifacts.scalarDiffUrl} download="scalar_diff.png">scalar_diff.png</a>
                      <a href={hoverReconArtifacts.debugUrl} download="hover_reconstruction_debug.json">hover_reconstruction_debug.json</a>
                    </div>
                  )}
                  {hoverInspectorMode === "debug" && hoverReconMetrics && (
                    <code className="height-hover-line debug">
                      <span className="hover-chip secondary fixed"><span className="k">MeanDiff</span><span className="v">{String(hoverReconMetrics.mean_rgb_diff ?? "-")}</span></span>
                      <span className="hover-chip secondary fixed"><span className="k">MaxDiff</span><span className="v">{String(hoverReconMetrics.max_rgb_diff ?? "-")}</span></span>
                      <span className="hover-chip secondary fixed"><span className="k">Mismatch%</span><span className="v">{String(hoverReconMetrics.mismatch_percent ?? "-")}</span></span>
                    </code>
                  )}
                  {hoverInspectorMode === "debug" && hoverReconError ? (
                    <small className="hover-recon-error">{hoverReconError}</small>
                  ) : null}
                </div>
              </aside>
            )}
            {heightLikeImage && engineering && (
              <div className={`height-hover-bottom ${hoverInspectorMode === "debug" ? "debug" : "compact"}`} aria-live="polite">
                {canonicalHoverState ? (
                  <>
                    <code className="height-hover-line">
                      {compactHoverChips(canonicalHoverState.sample, debugMode === "engineering").map((chip) => (
                        <span key={chip.key} className={`hover-chip ${chip.priority ?? "secondary"} ${chip.fixed ? "fixed" : ""}`}>
                          <span className="k">{chip.label}</span>
                          <span className="v">{chip.value}</span>
                        </span>
                      ))}
                    </code>
                    {canonicalHoverState.sample.valueKind === "normalized_height" && !canonicalHoverState.renderContext ? (
                      <small className="hover-recon-error">No numeric source loaded</small>
                    ) : null}
                    {hoverInspectorMode === "debug" && (
                      <code className="height-hover-line debug">
                        {debugHoverChips(canonicalHoverState.sample, canonicalHoverState.fieldSources).map((chip) => (
                          <span key={chip.key} className={`hover-chip ${chip.priority ?? "secondary"} ${chip.fixed ? "fixed" : ""}`}>
                            <span className="k">{chip.label}</span>
                            <span className="v">{chip.value}</span>
                          </span>
                        ))}
                      </code>
                    )}
                  </>
                ) : (
                  <code className="height-hover-line">
                    <span className="hover-chip primary fixed"><span className="k">Hover</span><span className="v">Move cursor over image</span></span>
                  </code>
                )}
                {hoverInspectorMode === "debug" && hoverReconArtifacts && (
                  <div className="hover-recon-thumbs">
                    <a href={hoverReconArtifacts.reconNearestUrl} download="reconstructed_hover_image_nearest.png"><img alt="reconstructed hover nearest" src={hoverReconArtifacts.reconNearestUrl} /></a>
                    <a href={hoverReconArtifacts.reconBilinearUrl} download="reconstructed_hover_image_bilinear.png"><img alt="reconstructed hover bilinear" src={hoverReconArtifacts.reconBilinearUrl} /></a>
                    <a href={hoverReconArtifacts.diffNearestUrl} download="hover_vs_render_diff_nearest.png"><img alt="rgb diff nearest" src={hoverReconArtifacts.diffNearestUrl} /></a>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
        {selected.artifact_id === "normalized_heightmap" && (
          <div className="pipeline-status-grid">
            <div><span>Norm p50</span><strong>{String((normHist?.metadata as Record<string, unknown> | undefined)?.median_mm ?? "-")}</strong></div>
            <div><span>Norm p95</span><strong>{String(display?.["p95"] ?? "-")}</strong></div>
            <div><span>Norm max</span><strong>{String((normHist?.metadata as Record<string, unknown> | undefined)?.max_mm ?? "-")}</strong></div>
            <div><span>Residual p95</span><strong>{String((planeDebug?.metadata as Record<string, unknown> | undefined)?.model_residual_p95_mm ?? "-")}</strong></div>
            <div><span>Inlier ratio</span><strong>{String((planeDebug?.metadata as Record<string, unknown> | undefined)?.inlier_ratio ?? "-")}</strong></div>
            <div><span>Plane mode</span><strong>{String((planeDebug?.metadata as Record<string, unknown> | undefined)?.reference_surface_model_type ?? "-")}</strong></div>
            <div><span>Valid px</span><strong>{String(display?.["valid_count"] ?? "-")}</strong></div>
          </div>
        )}
        {selected.artifact_id === "normalized_heightmap" && (() => {
          const quality = (selected.metadata as Record<string, unknown> | undefined)?.normalization_quality as Record<string, unknown> | undefined;
          if (!quality || String(quality.status) !== "degraded") return null;
          return (
            <div className="artifact-warning-banner">
              Plane fit failed. This normalized view uses constant-Z fallback and is not reliable for semantic height-above-belt measurement.
            </div>
          );
        })()}
        {!engineering && heightLikeImage && (
          <div className="artifact-reference">
            <small>Operator mode: compact height overlay.</small>
          </div>
        )}
        {selected.stage_id === "segmentation" && (
          <div className="pipeline-status-grid">
            <div><span>Components</span><strong>{Number.isFinite(components) ? components : "-"}</strong></div>
            <div><span>Threshold coverage</span><strong>{Number.isFinite(thresholdCoverage) ? `${(thresholdCoverage * 100).toFixed(1)}%` : "-"}</strong></div>
            <div><span>Cleaned coverage</span><strong>{Number.isFinite(cleanedCoverage) ? `${(cleanedCoverage * 100).toFixed(1)}%` : "-"}</strong></div>
            <div><span>ROI</span><strong>{Boolean(roiInfo.enabled ?? roiEnabled) ? "enabled" : "disabled"}</strong></div>
          </div>
        )}
        {allowRoiEditing && (
          <div className="overlay-controls">
            <button className={drawRoiMode ? "active" : ""} type="button">Draw reference-surface ROI</button>
            <button className={roiEditStatus === "unsaved" ? "active" : ""} type="button">{`Status: ${roiEditStatus}`}</button>
            <button className={roiEnabled ? "active" : ""} type="button">{`ROI: ${roiEnabled ? "enabled" : "disabled"}`}</button>
            <button className={localRoiType === "rectangle" ? "active" : ""} onClick={() => setLocalRoiType("rectangle")} type="button">Rectangle</button>
            <button className={localRoiType === "polygon" ? "active" : ""} onClick={() => setLocalRoiType("polygon")} type="button">Polygon</button>
            <button className={localRoiType === "full_height_x_band" ? "active" : ""} onClick={() => setLocalRoiType("full_height_x_band")} type="button">Vertical band ROI</button>
            <button className={drawRoiMode ? "active" : ""} onClick={() => setDrawRoiMode((value) => !value)} type="button">Draw ROI</button>
            <button onClick={() => { setDraftRoi(null); setDraftPolygon(null); onCancelRoiEdit?.(); }} type="button">Cancel</button>
            <button onClick={() => { setDraftRoi(null); setDraftPolygon(null); onClearRoi?.(); }} type="button">Use full frame</button>
            <button
              disabled={localRoiType === "polygon" ? !(draftPolygon && draftPolygon.length >= 3) : !draftRoi}
              onClick={() => {
                if (localRoiType === "rectangle" && draftRoi) onApplyRoi?.(draftRoi, false);
                if (localRoiType === "full_height_x_band" && draftRoi) onApplyRoi?.(draftRoi, false);
                if (localRoiType === "polygon" && draftPolygon && draftPolygon.length >= 3) onApplyPolygonRoi?.(draftPolygon, false);
              }}
              type="button"
            >
              Apply ROI
            </button>
            <button
              disabled={localRoiType === "polygon" ? !(draftPolygon && draftPolygon.length >= 3) : !draftRoi}
              onClick={() => {
                if (localRoiType === "rectangle" && draftRoi) onApplyRoi?.(draftRoi, true);
                if (localRoiType === "full_height_x_band" && draftRoi) onApplyRoi?.(draftRoi, true);
                if (localRoiType === "polygon" && draftPolygon && draftPolygon.length >= 3) onApplyPolygonRoi?.(draftPolygon, true);
              }}
              type="button"
            >
              Apply ROI + rerun
            </button>
          </div>
        )}
        {allowRoiEditing && localRoiType === "full_height_x_band" && (
          <div className="artifact-reference">
            <small>Constrains processing to an X-range while preserving the full scan height.</small>
          </div>
        )}
        {selected.stage_id === "segmentation" && emptyCleaned && (
          <div className="artifact-warning-banner">
            Cleaned mask is empty. Try lowering min component area, changing threshold/invert, or selecting a tighter ROI.
          </div>
        )}
        <details className="artifact-reference">
          <summary>Artifact details</summary>
          <small>Stage: {selected.stage_id}</small>
          <small>Kind: {selected.kind}</small>
          <small>Generated: {selected.generated ?? "derived"}</small>
          <small>Object: {selected.object_id == null ? "-" : objectDisplayId(selected.object_id)}</small>
          <small>Preview: {selected.preview_available ? "available" : "not available"}</small>
          <small>File: {selected.path ?? "-"}</small>
          <small>Focused object: {selectedObjectId == null ? "-" : selectedObjectId}</small>
          <small>Target image: {previewTarget ? `${previewTarget.title} (${previewTarget.artifact_id})` : "-"}</small>
          <small>Coordinate space: {selected.coordinate_space ?? "-"}</small>
          <small>Overlay type: {selected.overlay_type ?? "-"}</small>
          <small>Projection type: {previewTarget?.projection_type ?? "-"}</small>
          <small>Projection transform: {previewTarget?.projection_transform_id ?? "-"}</small>
          <small>Source artifact: {String(selected.metadata?.source_artifact_id ?? "-")}</small>
          <small>Mask artifact: {String(selected.metadata?.mask_artifact_id ?? "-")}</small>
        </details>
        {selected.kind === "json" && <pre className="json-block">{JSON.stringify(detailJson ?? {}, null, 2)}</pre>}
        {selected.kind === "table" && <div className="artifact-reference"><strong>Table artifact ready for stage/object inspection.</strong></div>}
        {selected.kind === "metric" && <div className="artifact-reference"><strong>Metric artifact ready for stage/object inspection.</strong></div>}
        {(selected.kind === "point_cloud" || selected.kind === "file" || selected.kind === "video" || selected.kind === "text") && (
          <div className="empty-state">Preview not available for this artifact kind yet.</div>
        )}
      </section>
    </div>
  );
}

function isHeightLikeArtifact(artifact: StudioArtifact): boolean {
  const id = String(artifact.artifact_id ?? "").toLowerCase();
  const path = String(artifact.path ?? "").toLowerCase();
  return (
    /height|depth|residual/.test(id)
    || /height|depth|residual/.test(path)
  );
}

function toFinite(value: unknown): number | null {
  const n = typeof value === "number" ? value : (typeof value === "string" ? Number(value) : NaN);
  return Number.isFinite(n) ? n : null;
}

function parseShapeTuple(value: unknown): [number, number] | null {
  if (!Array.isArray(value) || value.length < 2) return null;
  const h = toFinite(value[0]);
  const w = toFinite(value[1]);
  if (h == null || w == null || h <= 0 || w <= 0) return null;
  return [Math.round(h), Math.round(w)];
}

function pickRangeFromRecord(record: Record<string, unknown> | null | undefined): HeightRange | null {
  if (!record) return null;
  const minCandidates = [
    record.min_height_mm,
    record.min_mm,
    record.min_value_mm,
    record.z_min_mm,
    record.min,
    record.value_min,
  ];
  const maxCandidates = [
    record.max_height_mm,
    record.max_mm,
    record.max_value_mm,
    record.z_max_mm,
    record.max,
    record.value_max,
  ];
  const min = minCandidates.map(toFinite).find((v) => v != null);
  const max = maxCandidates.map(toFinite).find((v) => v != null);
  if (min == null || max == null) return null;
  if (max <= min) return null;
  const unit = typeof record.units === "string" && record.units.trim() ? record.units.trim() : "mm";
  return { min, max, unit };
}

function resolveHeightRange(selected: StudioArtifact, artifacts: StudioArtifact[]): HeightRange | null {
  if (selected.kind !== "image") return null;
  if (!isHeightLikeArtifact(selected)) return null;
  const renderContext = renderContextTransform(artifacts);
  if (renderContext) {
    const min = toFinite(renderContext.render_vmin);
    const max = toFinite(renderContext.render_vmax);
    if (min != null && max != null && max > min) {
      return { min, max, unit: "mm" };
    }
  }
  const display = displayTransform(artifacts);
  if (display) {
    const min = toFinite(display.display_vmin);
    const max = toFinite(display.display_vmax);
    const unit = typeof display.units === "string" && display.units.trim() ? display.units.trim() : "mm";
    if (min != null && max != null && max > min) return { min, max, unit };
  }
  const selectedMeta = (selected.metadata ?? {}) as Record<string, unknown>;
  const direct = pickRangeFromRecord(selectedMeta);
  if (direct) return direct;
  const nestedCandidates = [
    selectedMeta.height_range,
    selectedMeta.value_range,
    selectedMeta.range_mm,
    selectedMeta.residual_stats_mm,
  ];
  for (const candidate of nestedCandidates) {
    if (candidate && typeof candidate === "object") {
      const nested = pickRangeFromRecord(candidate as Record<string, unknown>);
      if (nested) return nested;
    }
  }
  const histogram = artifacts.find((item) => item.artifact_id === "normalized_height_histogram");
  if (histogram?.metadata && typeof histogram.metadata === "object") {
    const fromHistogram = pickRangeFromRecord(histogram.metadata as Record<string, unknown>);
    if (fromHistogram) return fromHistogram;
    const entries = (histogram.metadata as Record<string, unknown>).bins;
    if (Array.isArray(entries) && entries.length > 1) {
      const nums = entries.map((v) => toFinite(v)).filter((v): v is number => v != null);
      if (nums.length > 1) {
        const min = Math.min(...nums);
        const max = Math.max(...nums);
        if (max > min) return { min, max, unit: "mm" };
      }
    }
  }
  return null;
}

function displayTransform(artifacts: StudioArtifact[]): Record<string, unknown> | null {
  const preferred = resolveHeightArtifact(artifacts, { semanticField: "preview_normalized", representation: "numeric_raster" }).artifact
    ?? artifacts.find((a) => a.artifact_id === "normalized_heightmap_display")
    ?? null;
  const metadata = preferred?.metadata;
  if (metadata && typeof metadata === "object") return metadata as Record<string, unknown>;
  return null;
}

function renderContextTransform(artifacts: StudioArtifact[]): Record<string, unknown> | null {
  const preferred = artifacts.find((a) => a.artifact_id === "normalized_heightmap_render_context")
    ?? null;
  const metadata = preferred?.metadata;
  if (metadata && typeof metadata === "object") return metadata as Record<string, unknown>;
  return null;
}

function histogramArtifact(artifacts: StudioArtifact[]): StudioArtifact | null {
  return artifacts.find((a) => a.artifact_id === "normalized_height_histogram") ?? null;
}

function planeDebugArtifact(artifacts: StudioArtifact[]): StudioArtifact | null {
  return artifacts.find((a) => a.artifact_id === "plane_fit_debug") ?? null;
}

type HoverChip = {
  key: string;
  label: string;
  value: string;
  priority?: "primary" | "secondary";
  fixed?: boolean;
};

function fmtMm(value: number | null | undefined, digits = 1): string {
  if (value == null || !Number.isFinite(value)) return "-";
  const sign = value >= 0 ? "+" : "";
  return `${sign}${value.toFixed(digits)}mm`;
}

function compactHoverChips(sample: HoverInspectionSample, debugMode = false): HoverChip[] {
  const objectId = sample.hoveredObjectId ?? sample.selectedObjectId;
  const chips: HoverChip[] = [
    { key: "x", label: "X", value: `${sample.imageX}`, priority: "primary", fixed: true },
    { key: "y", label: "Y", value: `${sample.imageY}`, priority: "primary", fixed: true },
  ];
  const semantic = String(sample.semanticField ?? sample.displaySemantic ?? "");
  if (semantic === "raw_sensor_z" || sample.valueKind === "raw_z") {
    chips.push({ key: "zraw", label: "Raw Z", value: sample.valid ? fmtMm(sample.rawZ, 1) : "invalid", priority: "primary", fixed: true });
    chips.push({ key: "valid", label: "Valid", value: sample.valid ? "yes" : "no", priority: "secondary", fixed: true });
  } else if (semantic === "height_above_belt" || sample.valueKind === "normalized_height") {
    chips.push({ key: "h", label: "Height above belt", value: sample.valid ? fmtMm(sample.normalizedHeight, 1) : "invalid", priority: "primary", fixed: true });
    if (debugMode) chips.push({ key: "raw", label: "Raw Z", value: sample.valid ? fmtMm(sample.rawZ, 1) : "-", priority: "secondary", fixed: true });
  } else {
    chips.push({ key: "val", label: "Val", value: sample.valid ? fmtMm(sample.normalizedHeight ?? sample.rawZ, 1) : "invalid", priority: "primary", fixed: true });
  }
  chips.push({ key: "obj", label: "Obj", value: objectId == null ? "-" : `object_${String(objectId).padStart(3, "0")}`, priority: "primary", fixed: true });
  return chips;
}

function debugHoverChips(
  sample: HoverInspectionSample,
  sources?: {
    norm: string;
    rgb: string;
    srcPx: string;
    imgPx: string;
    finalPx: string;
    roi: string;
    mode: string;
  },
): HoverChip[] {
  const coord = sample.coordDebug;
  return [
    { key: "semantic", label: "Sem[src=display]", value: sample.displaySemantic ?? "-", priority: "secondary", fixed: true },
    { key: "rawz", label: "RawZ[src=hover]", value: sample.valid ? fmtMm(sample.rawZ, 3) : "invalid", priority: "secondary", fixed: true },
    { key: "norm", label: `Norm[src=${sources?.norm ?? "hover"}]`, value: sample.valid ? fmtMm(sample.normalizedHeight, 3) : "invalid", priority: "secondary", fixed: true },
    { key: "planez", label: "PlaneZ[src=hover]", value: sample.valid ? fmtMm(sample.planeHeight, 3) : "-", priority: "secondary", fixed: true },
    { key: "disp", label: "Disp[src=hover]", value: sample.valid ? fmtMm(sample.displayClippedHeight, 3) : "-", priority: "secondary", fixed: true },
    { key: "res", label: "Residual", value: sample.valid ? fmtMm(sample.residualMm, 3) : "-", priority: "secondary", fixed: true },
    { key: "clip", label: "Clip", value: sample.clipped == null ? "-" : sample.clipped ? "yes" : "no", priority: "secondary", fixed: true },
    { key: "valid", label: "Valid", value: sample.valid == null ? "-" : sample.valid ? "yes" : "no", priority: "secondary", fixed: true },
    { key: "client", label: "Client", value: coord ? `${Math.round(coord.clientX)},${Math.round(coord.clientY)}` : "-", priority: "secondary", fixed: true },
    { key: "imgpx", label: `ImgPx[src=${sources?.imgPx ?? "coord"}]`, value: coord ? `${coord.naturalX},${coord.naturalY}` : "-", priority: "secondary", fixed: true },
    { key: "srcpx", label: `SrcPx[src=${sources?.srcPx ?? "coord"}]`, value: coord ? `${coord.sourceArrayX},${coord.sourceArrayY}` : "-", priority: "secondary", fixed: true },
    { key: "finalpx", label: `FinalPx[src=${sources?.finalPx ?? "coord"}]`, value: coord ? `${coord.finalSampleX},${coord.finalSampleY}` : "-", priority: "secondary", fixed: true },
    { key: "roi", label: `ROI[src=${sources?.roi ?? "coord"}]`, value: coord ? `${coord.roiOriginX},${coord.roiOriginY}:${coord.roiWidth}x${coord.roiHeight}` : "-", priority: "secondary", fixed: true },
    { key: "rgb", label: `RGB[src=${sources?.rgb ?? "canvas"}]`, value: coord?.renderedRgb ? `${coord.renderedRgb[0]},${coord.renderedRgb[1]},${coord.renderedRgb[2]}` : "-", priority: "secondary", fixed: true },
    { key: "mode", label: `Mode[src=${sources?.mode ?? "coord"}]`, value: coord?.renderMode ?? "full_image", priority: "secondary", fixed: true },
  ];
}

async function generateHoverReconstruction(args: {
  takeId: string;
  imagePath: string;
  viewerContext: ViewerRenderContext | null;
  selectionMetadata?: {
    activeViewId?: string | null;
    selectedStageId?: string | null;
    displaySourceId?: string | null;
    numericSourceId?: string | null;
  };
  onResult: (result: {
    status: HoverParityStatus;
    metrics: Record<string, unknown>;
    artifacts: {
      reconNearestUrl: string;
      reconBilinearUrl: string;
      diffNearestUrl: string;
      diffBilinearUrl: string;
      diffHeatUrl: string;
      diffMaskUrl: string;
      scalarDiffUrl: string;
      debugUrl: string;
    };
  }) => void;
  onBusy: (busy: boolean) => void;
  onError: (message: string | null) => void;
}) {
  const { takeId, imagePath, viewerContext, selectionMetadata, onResult, onBusy, onError } = args;
  if (!viewerContext) return;
  onBusy(true);
  onError(null);
  try {
    const width = viewerContext.width;
    const height = viewerContext.height;
    const total = width * height;
    const reconNearest = new Uint8ClampedArray(total * 4);
    const reconBilinear = new Uint8ClampedArray(total * 4);
    const trace: Array<Record<string, unknown>> = [];
    const reconNormScalar = new Float32Array(total);
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const idx = y * width + x;
        const value = viewerContext.sampler.sampleValue(x, y);
        const valid = viewerContext.sampler.sampleValid(x, y);
        const clipped = value == null ? null : clamp(value, viewerContext.normalization.vmin, viewerContext.normalization.vmax);
        const t = clipped == null
          ? 0
          : (clipped - viewerContext.normalization.vmin) / Math.max(1e-9, viewerContext.normalization.vmax - viewerContext.normalization.vmin);
        reconNormScalar[idx] = Number.isFinite(t) ? t : 0;
        const [r, g, b] = scalarToRgb(viewerContext.colormap.name, value, viewerContext.normalization.vmin, viewerContext.normalization.vmax, valid);
        const p = idx * 4;
        reconNearest[p + 0] = r;
        reconNearest[p + 1] = g;
        reconNearest[p + 2] = b;
        reconNearest[p + 3] = 255;
        const bilinear = sampleBilinearFromSampler(viewerContext.sampler, x, y, width, height);
        const [br, bg, bb] = scalarToRgb(
          viewerContext.colormap.name,
          bilinear.value,
          viewerContext.normalization.vmin,
          viewerContext.normalization.vmax,
          bilinear.valid,
        );
        reconBilinear[p + 0] = br;
        reconBilinear[p + 1] = bg;
        reconBilinear[p + 2] = bb;
        reconBilinear[p + 3] = 255;
      }
    }
    const reconNearestCanvas = document.createElement("canvas");
    reconNearestCanvas.width = width;
    reconNearestCanvas.height = height;
    const nearestCtx = reconNearestCanvas.getContext("2d");
    if (!nearestCtx) return;
    nearestCtx.putImageData(new ImageData(reconNearest, width, height), 0, 0);
    const reconBilinearCanvas = document.createElement("canvas");
    reconBilinearCanvas.width = width;
    reconBilinearCanvas.height = height;
    const bilinearCtx = reconBilinearCanvas.getContext("2d");
    if (!bilinearCtx) return;
    bilinearCtx.putImageData(new ImageData(reconBilinear, width, height), 0, 0);

    const originalBitmap = await loadBitmap(fileUrl(takeId, imagePath));
    const origCanvas = document.createElement("canvas");
    origCanvas.width = width;
    origCanvas.height = height;
    const octx = origCanvas.getContext("2d");
    if (!octx) return;
    octx.drawImage(originalBitmap, 0, 0, width, height);
    const origData = octx.getImageData(0, 0, width, height).data;
    const rendererNormScalar = new Float32Array(total);
    for (let i = 0; i < total; i += 1) {
      const p = i * 4;
      rendererNormScalar[i] = inferColorMapTFromRgb(viewerContext.colormap.name, [origData[p + 0], origData[p + 1], origData[p + 2]]);
    }

    const diffNearest = new Uint8ClampedArray(total * 4);
    const diffBilinear = new Uint8ClampedArray(total * 4);
    const heat = new Uint8ClampedArray(total * 4);
    const mask = new Uint8ClampedArray(total * 4);
    const scalarDiff = new Uint8ClampedArray(total * 4);
    let sumDiff = 0;
    let maxDiff = 0;
    let mismatch = 0;
    let identical = 0;
    let sumDiffBilinear = 0;
    let maxDiffBilinear = 0;
    let mismatchBilinear = 0;
    let identicalBilinear = 0;
    let scalarDiffSum = 0;
    let scalarDiffMax = 0;
    for (let i = 0; i < total; i += 1) {
      const p = i * 4;
      const drNearest = Math.abs(origData[p + 0] - reconNearest[p + 0]);
      const dgNearest = Math.abs(origData[p + 1] - reconNearest[p + 1]);
      const dbNearest = Math.abs(origData[p + 2] - reconNearest[p + 2]);
      const dNearest = drNearest + dgNearest + dbNearest;
      sumDiff += dNearest;
      maxDiff = Math.max(maxDiff, dNearest);
      if (dNearest === 0) identical += 1;
      if (dNearest > 6) mismatch += 1;
      diffNearest[p + 0] = drNearest;
      diffNearest[p + 1] = dgNearest;
      diffNearest[p + 2] = dbNearest;
      diffNearest[p + 3] = 255;

      const drBilinear = Math.abs(origData[p + 0] - reconBilinear[p + 0]);
      const dgBilinear = Math.abs(origData[p + 1] - reconBilinear[p + 1]);
      const dbBilinear = Math.abs(origData[p + 2] - reconBilinear[p + 2]);
      const dBilinear = drBilinear + dgBilinear + dbBilinear;
      sumDiffBilinear += dBilinear;
      maxDiffBilinear = Math.max(maxDiffBilinear, dBilinear);
      if (dBilinear === 0) identicalBilinear += 1;
      if (dBilinear > 6) mismatchBilinear += 1;
      diffBilinear[p + 0] = drBilinear;
      diffBilinear[p + 1] = dgBilinear;
      diffBilinear[p + 2] = dbBilinear;
      diffBilinear[p + 3] = 255;

      const scalarD = Math.abs(rendererNormScalar[i] - reconNormScalar[i]);
      scalarDiffSum += scalarD;
      scalarDiffMax = Math.max(scalarDiffMax, scalarD);
      const scalarVis = Math.min(255, Math.round(scalarD * 255));
      scalarDiff[p + 0] = scalarVis;
      scalarDiff[p + 1] = 0;
      scalarDiff[p + 2] = 255 - scalarVis;
      scalarDiff[p + 3] = 255;

      const h = Math.min(255, Math.round((dNearest / 765) * 255));
      heat[p + 0] = h;
      heat[p + 1] = 0;
      heat[p + 2] = 255 - h;
      heat[p + 3] = 255;
      const m = dNearest > 6 ? 255 : 0;
      mask[p + 0] = m;
      mask[p + 1] = m;
      mask[p + 2] = m;
      mask[p + 3] = 255;
    }
    const diffNearestCanvas = document.createElement("canvas");
    diffNearestCanvas.width = width;
    diffNearestCanvas.height = height;
    diffNearestCanvas.getContext("2d")?.putImageData(new ImageData(diffNearest, width, height), 0, 0);
    const diffBilinearCanvas = document.createElement("canvas");
    diffBilinearCanvas.width = width;
    diffBilinearCanvas.height = height;
    diffBilinearCanvas.getContext("2d")?.putImageData(new ImageData(diffBilinear, width, height), 0, 0);
    const heatCanvas = document.createElement("canvas");
    heatCanvas.width = width;
    heatCanvas.height = height;
    heatCanvas.getContext("2d")?.putImageData(new ImageData(heat, width, height), 0, 0);
    const maskCanvas = document.createElement("canvas");
    maskCanvas.width = width;
    maskCanvas.height = height;
    maskCanvas.getContext("2d")?.putImageData(new ImageData(mask, width, height), 0, 0);
    const scalarDiffCanvas = document.createElement("canvas");
    scalarDiffCanvas.width = width;
    scalarDiffCanvas.height = height;
    scalarDiffCanvas.getContext("2d")?.putImageData(new ImageData(scalarDiff, width, height), 0, 0);

    const reconNearestUrl = await canvasToObjectUrl(reconNearestCanvas);
    const reconBilinearUrl = await canvasToObjectUrl(reconBilinearCanvas);
    const diffNearestUrl = await canvasToObjectUrl(diffNearestCanvas);
    const diffBilinearUrl = await canvasToObjectUrl(diffBilinearCanvas);
    const diffHeatUrl = await canvasToObjectUrl(heatCanvas);
    const diffMaskUrl = await canvasToObjectUrl(maskCanvas);
    const scalarDiffUrl = await canvasToObjectUrl(scalarDiffCanvas);
    const activeSelectionViewId = selectionMetadata?.activeViewId ?? viewerContext.activeViewId ?? viewerContext.artifactId ?? null;
    const selectedSelectionStageId = selectionMetadata?.selectedStageId ?? viewerContext.selectedStageId ?? null;
    const selectedDisplaySourceId = selectionMetadata?.displaySourceId ?? viewerContext.displaySourceId ?? activeSelectionViewId ?? null;
    const selectedNumericSourceId = selectionMetadata?.numericSourceId ?? viewerContext.numericSourceId ?? viewerContext.sourceArtifactId;
    const rangeAssertions = {
      reconstruction_equals_colorbar: almostEqual(viewerContext.normalizationRanges.reconstructionMin, viewerContext.normalizationRanges.colorbarMin)
        && almostEqual(viewerContext.normalizationRanges.reconstructionMax, viewerContext.normalizationRanges.colorbarMax),
      reconstruction_equals_display: almostEqual(viewerContext.normalizationRanges.reconstructionMin, viewerContext.normalizationRanges.displayMin)
        && almostEqual(viewerContext.normalizationRanges.reconstructionMax, viewerContext.normalizationRanges.displayMax),
    };
    const points = selectTracePoints(width, height, viewerContext.sampler, viewerContext.normalization.vmin, viewerContext.normalization.vmax);
    for (const point of points) {
      const rendererRgb = readRgbAt(origData, width, point.x, point.y);
      const scalar = viewerContext.sampler.sampleValue(point.x, point.y);
      const valid = viewerContext.sampler.sampleValid(point.x, point.y);
      const clipped = scalar == null ? null : clamp(scalar, viewerContext.normalization.vmin, viewerContext.normalization.vmax);
      const normalizedScalar = clipped == null
        ? null
        : (clipped - viewerContext.normalization.vmin) / Math.max(1e-9, viewerContext.normalization.vmax - viewerContext.normalization.vmin);
      const reconRgb = scalarToRgb(viewerContext.colormap.name, scalar, viewerContext.normalization.vmin, viewerContext.normalization.vmax, valid);
      trace.push({
        label: point.label,
        pixel: [point.x, point.y],
        real_renderer_path: {
          source_scalar: scalar,
          normalization_min: viewerContext.normalization.vmin,
          normalization_max: viewerContext.normalization.vmax,
          percentile_clipping: {
            min: viewerContext.normalizationRanges.percentileMin,
            max: viewerContext.normalizationRanges.percentileMax,
          },
          clipped_scalar: clipped,
          normalized_scalar_01: normalizedScalar,
          colormap_input: normalizedScalar,
          final_rgb: rendererRgb,
        },
        hover_reconstruction_path: {
          source_scalar: scalar,
          normalization_min: viewerContext.normalization.vmin,
          normalization_max: viewerContext.normalization.vmax,
          percentile_clipping: {
            min: viewerContext.normalizationRanges.percentileMin,
            max: viewerContext.normalizationRanges.percentileMax,
          },
          clipped_scalar: clipped,
          normalized_scalar_01: normalizedScalar,
          colormap_input: normalizedScalar,
          final_rgb: reconRgb,
        },
      });
    }
    const quantizationStats = {
      scalar_integer_ratio: estimateIntegerRatio(viewerContext, 4096),
      normalized_255_step_ratio: estimateStepRatio(reconNormScalar, 1 / 255),
      normalized_1024_step_ratio: estimateStepRatio(reconNormScalar, 1 / 1024),
      premature_uint8_conversion_detected: false,
      normalized_scalar_rounded_detected: false,
      integer_scalar_bins_detected: false,
    };
    const debugPayload = {
      metrics: {
        width,
        height,
        identical_pixel_count: identical,
        max_rgb_diff: maxDiff,
        mean_rgb_diff: Number((sumDiff / Math.max(1, total)).toFixed(4)),
        mismatch_percent: Number(((mismatch / Math.max(1, total)) * 100).toFixed(4)),
        bilinear_identical_pixel_count: identicalBilinear,
        bilinear_max_rgb_diff: maxDiffBilinear,
        bilinear_mean_rgb_diff: Number((sumDiffBilinear / Math.max(1, total)).toFixed(4)),
        bilinear_mismatch_percent: Number(((mismatchBilinear / Math.max(1, total)) * 100).toFixed(4)),
        scalar_diff_mean: Number((scalarDiffSum / Math.max(1, total)).toFixed(6)),
        scalar_diff_max: Number(scalarDiffMax.toFixed(6)),
      },
      normalization_ranges: viewerContext.normalizationRanges,
      normalization_assertions: rangeAssertions,
      quantization: quantizationStats,
      hover_path: {
        source_array_id: selectedNumericSourceId,
        normalization_range: { vmin: viewerContext.normalization.vmin, vmax: viewerContext.normalization.vmax },
        normalization_range_source: viewerContext.rangeSource,
        colormap: viewerContext.colormap.name,
        interpolation_mode: viewerContext.interpolationMode,
        canvas_scaling_mode: "contain",
        roi_purpose: viewerContext.roiPurpose,
        render_coordinate_space: viewerContext.renderCoordinateSpace,
        source_shape: [viewerContext.sourceHeight, viewerContext.sourceWidth],
        rendered_png_shape: [viewerContext.height, viewerContext.width],
        numeric_array_shape: [viewerContext.sourceHeight, viewerContext.sourceWidth],
        valid_mask_shape: [viewerContext.sourceHeight, viewerContext.sourceWidth],
        roi: viewerContext.roi ? { x: viewerContext.roi.x, y: viewerContext.roi.y, width: viewerContext.roi.width, height: viewerContext.roi.height } : null,
        roi_mode: viewerContext.roi ? "roi_crop" : "full_image",
        coordinate_chain: viewerContext.coordinateChain,
        render_context_artifact_id: viewerContext.artifactId ?? null,
        active_view_id: activeSelectionViewId,
        selected_stage_id: selectedSelectionStageId,
        display_source_id: selectedDisplaySourceId,
        numeric_source_id: selectedNumericSourceId,
      },
      trace_samples: trace,
    };
    if (!rangeAssertions.reconstruction_equals_colorbar || !rangeAssertions.reconstruction_equals_display) {
      console.error("HOVER NORMALIZATION RANGE MISMATCH", {
        normalization_ranges: viewerContext.normalizationRanges,
        normalization_assertions: rangeAssertions,
      });
    }
    const debugUrl = URL.createObjectURL(new Blob([JSON.stringify(debugPayload, null, 2)], { type: "application/json" }));
    const mean = Number((sumDiff / Math.max(1, total)).toFixed(4));
    const mismatchPercent = Number(((mismatch / Math.max(1, total)) * 100).toFixed(4));
    const status: HoverParityStatus =
      mismatchPercent <= 0.01 ? "PERFECT_MATCH"
        : mismatchPercent <= 0.5 ? "INTERPOLATION_MISMATCH"
          : mean > 40 ? "NORMALIZATION_MISMATCH"
            : "VALUE_MISMATCH";
    onResult({
      status,
      metrics: {
        identical_pixel_count: identical,
        max_rgb_diff: maxDiff,
        mean_rgb_diff: mean,
        mismatch_percent: mismatchPercent,
        bilinear_identical_pixel_count: identicalBilinear,
        bilinear_max_rgb_diff: maxDiffBilinear,
        bilinear_mean_rgb_diff: Number((sumDiffBilinear / Math.max(1, total)).toFixed(4)),
        bilinear_mismatch_percent: Number(((mismatchBilinear / Math.max(1, total)) * 100).toFixed(4)),
        scalar_diff_mean: Number((scalarDiffSum / Math.max(1, total)).toFixed(6)),
        scalar_diff_max: Number(scalarDiffMax.toFixed(6)),
      },
      artifacts: {
        reconNearestUrl,
        reconBilinearUrl,
        diffNearestUrl,
        diffBilinearUrl,
        diffHeatUrl,
        diffMaskUrl,
        scalarDiffUrl,
        debugUrl,
      },
    });
  } catch (error) {
    onError(error instanceof Error ? error.message : "Failed to generate hover reconstruction");
  } finally {
    onBusy(false);
  }
}

async function loadBitmap(url: string): Promise<ImageBitmap> {
  const response = await fetch(url);
  const blob = await response.blob();
  return await createImageBitmap(blob);
}

async function canvasToObjectUrl(canvas: HTMLCanvasElement): Promise<string> {
  const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/png"));
  if (!blob) throw new Error("Failed to encode PNG");
  return URL.createObjectURL(blob);
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function almostEqual(a: number | null, b: number | null, eps = 1e-6): boolean {
  if (a == null && b == null) return true;
  if (a == null || b == null) return false;
  return Math.abs(a - b) <= eps;
}

function computeFiniteRange(values: Float32Array, validMask: Uint8Array | null): { min: number | null; max: number | null } {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < values.length; i += 1) {
    if (validMask && validMask[i] <= 0) continue;
    const v = values[i];
    if (!Number.isFinite(v)) continue;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (!Number.isFinite(min) || !Number.isFinite(max)) return { min: null, max: null };
  return { min, max };
}

function sampleBilinearFromSampler(
  sampler: ViewerRenderContext["sampler"],
  x: number,
  y: number,
  width: number,
  height: number,
): { value: number | null; valid: boolean } {
  const x0 = Math.max(0, Math.min(width - 1, x));
  const y0 = Math.max(0, Math.min(height - 1, y));
  const x1 = Math.max(0, Math.min(width - 1, x + 1));
  const y1 = Math.max(0, Math.min(height - 1, y + 1));
  const w00 = 0.25;
  const w10 = 0.25;
  const w01 = 0.25;
  const w11 = 0.25;
  const v00 = sampler.sampleValue(x0, y0);
  const v10 = sampler.sampleValue(x1, y0);
  const v01 = sampler.sampleValue(x0, y1);
  const v11 = sampler.sampleValue(x1, y1);
  const valid = sampler.sampleValid(x0, y0) && sampler.sampleValid(x1, y0) && sampler.sampleValid(x0, y1) && sampler.sampleValid(x1, y1);
  if (!valid || v00 == null || v10 == null || v01 == null || v11 == null) return { value: null, valid: false };
  return { value: (v00 * w00) + (v10 * w10) + (v01 * w01) + (v11 * w11), valid: true };
}

function readRgbAt(imageData: Uint8ClampedArray, width: number, x: number, y: number): [number, number, number] {
  const xi = Math.max(0, x);
  const yi = Math.max(0, y);
  const p = ((yi * width) + xi) * 4;
  return [imageData[p + 0], imageData[p + 1], imageData[p + 2]];
}

function selectTracePoints(
  width: number,
  height: number,
  sampler: ViewerRenderContext["sampler"],
  vmin: number,
  vmax: number,
): Array<{ label: "ball_center" | "rim" | "background"; x: number; y: number }> {
  let center = { x: Math.floor(width / 2), y: Math.floor(height / 2), value: Number.NEGATIVE_INFINITY };
  const step = Math.max(1, Math.floor(Math.min(width, height) / 256));
  for (let y = 0; y < height; y += step) {
    for (let x = 0; x < width; x += step) {
      if (!sampler.sampleValid(x, y)) continue;
      const v = sampler.sampleValue(x, y);
      if (v == null || !Number.isFinite(v)) continue;
      if (v > center.value) center = { x, y, value: v };
    }
  }
  let rim = { x: center.x, y: center.y, grad: Number.NEGATIVE_INFINITY };
  for (let y = 1; y < height - 1; y += step) {
    for (let x = 1; x < width - 1; x += step) {
      if (!sampler.sampleValid(x, y)) continue;
      const left = sampler.sampleValue(x - 1, y);
      const right = sampler.sampleValue(x + 1, y);
      const up = sampler.sampleValue(x, y - 1);
      const down = sampler.sampleValue(x, y + 1);
      if (left == null || right == null || up == null || down == null) continue;
      const grad = Math.abs(right - left) + Math.abs(down - up);
      const v = sampler.sampleValue(x, y);
      if (v == null || v < vmin || v > vmax) continue;
      if (grad > rim.grad) rim = { x, y, grad };
    }
  }
  let background = { x: 0, y: 0, value: Number.POSITIVE_INFINITY };
  for (let y = 0; y < height; y += step) {
    for (let x = 0; x < width; x += step) {
      if (!sampler.sampleValid(x, y)) continue;
      const v = sampler.sampleValue(x, y);
      if (v == null || !Number.isFinite(v)) continue;
      if (v < background.value) background = { x, y, value: v };
    }
  }
  return [
    { label: "ball_center", x: center.x, y: center.y },
    { label: "rim", x: rim.x, y: rim.y },
    { label: "background", x: background.x, y: background.y },
  ];
}

function estimateIntegerRatio(viewerContext: ViewerRenderContext, sampleLimit: number): number {
  const width = viewerContext.width;
  const height = viewerContext.height;
  let sampled = 0;
  let integerish = 0;
  const stride = Math.max(1, Math.floor(Math.sqrt((width * height) / Math.max(1, sampleLimit))));
  for (let y = 0; y < height && sampled < sampleLimit; y += stride) {
    for (let x = 0; x < width && sampled < sampleLimit; x += stride) {
      if (!viewerContext.sampler.sampleValid(x, y)) continue;
      const v = viewerContext.sampler.sampleValue(x, y);
      if (v == null || !Number.isFinite(v)) continue;
      sampled += 1;
      if (Math.abs(v - Math.round(v)) < 1e-6) integerish += 1;
    }
  }
  return sampled <= 0 ? 0 : Number((integerish / sampled).toFixed(6));
}

function estimateStepRatio(values: Float32Array, step: number): number {
  let sampled = 0;
  let snapped = 0;
  const stride = Math.max(1, Math.floor(values.length / 8192));
  for (let i = 0; i < values.length; i += stride) {
    const v = values[i];
    if (!Number.isFinite(v)) continue;
    sampled += 1;
    const nearest = Math.round(v / step) * step;
    if (Math.abs(v - nearest) < 1e-6) snapped += 1;
  }
  return sampled <= 0 ? 0 : Number((snapped / sampled).toFixed(6));
}

async function readImageSize(url: string): Promise<{ width: number; height: number }> {
  const response = await fetch(url);
  if (!response.ok) throw new Error("failed_image_fetch");
  const blob = await response.blob();
  const bitmap = await createImageBitmap(blob);
  const width = bitmap.width;
  const height = bitmap.height;
  bitmap.close();
  return { width, height };
}
