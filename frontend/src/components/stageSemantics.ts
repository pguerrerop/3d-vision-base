import type { StageExecutionReport, TakeDetail } from "../api/client";
import type { StudioArtifact } from "./studioWorkspaceModel";
export function canonicalStageId(stageId: string, stageLabel?: string | null): string {
  const raw = `${stageId ?? ""} ${stageLabel ?? ""}`.toLowerCase();
  if (raw.includes("normalize_heights_to_plane")) return "normalize_heights_to_plane";
  if (raw.includes("detect_belt_plane")) return "detect_belt_plane";
  if (raw.includes("remove_belt_segment_objects")) return "remove_belt_segment_objects";
  if (raw.includes("loadheightmapcapture") || raw.includes("heightmap capture")) return "load_heightmap";
  if (raw.includes("detectbeltplane") || raw.includes("estimatebeltplane") || raw.includes("belt plane")) return "detect_belt_plane";
  if (raw.includes("normalizeheightstoplane") || raw.includes("normalizeheightrelativetoplane") || raw.includes("normalized height")) return "normalize_heights_to_plane";
  if (raw.includes("removebeltandsegmentobjects") || raw.includes("heightsegmentation")) return "remove_belt_segment_objects";
  if (raw.includes("extractconnectedcomponents")) return "connected_components";
  if (raw.includes("fitobjectgeometry") || raw.includes("footprint geometry") || raw.trim() === "geometry") return "fit_object_geometry";
  if (raw.includes("computeheightmetrics") || raw.includes("height + volume") || raw.includes("compute_height_metrics") || raw.trim() === "measurement") return "compute_height_metrics";
  if (raw.includes("computemeasurementdiagnostics") || raw.includes("measurement diagnostics") || raw.includes("measurement_diagnostics")) return "measurement_diagnostics";
  if (raw.includes("classify_25d")) return "classify_25d";
  if (raw.includes("classifyminingball25d")) return "classify_25d";
  if (raw.includes("generate25doverlays") || raw.trim() === "overlay") return "overlays_25d";
  if (
    raw.includes("blob_detection")
    || raw.includes("blob/contour")
    || raw.includes("blob_contour")
    || raw.includes("contour_detection")
    || raw.includes("blob detection")
    || raw.includes("contour detection")
    || raw === "detection"
    || raw === "blob"
  ) return "detection";
  if (raw.includes("ellipse_fitting") || raw.includes("ellipse") || raw.includes("geometry_metrics")) return "ellipse_fitting";
  if (raw.includes("threshold") || raw.includes("morph")) return "segmentation";
  if (raw.includes("measure") || raw.includes("metric")) return "measurement";
  if (raw.includes("class")) return "classification";
  if (raw.includes("input") || raw.includes("normalize") || raw.includes("gray") || raw.includes("roi")) return "input";
  return (stageId || "").toLowerCase();
}

export type StageMaskViewSpec = {
  maskArtifactId?: string;
  maskArtifactIds?: string[];
  maskSemanticTypes?: string[];
  preferredReferenceArtifactIds?: string[];
  preferredReferenceSemanticTypes?: string[];
  defaultMode?: "mask" | "overlay" | "boundary" | "side_by_side";
  defaultReferenceRole?: "source" | "raw" | "normalized" | "computed_from" | "diagnostic";
  maskOnLabel?: string;
  maskOffLabel?: string;
  diagnosticsLabel?: string;
  description?: string;
};

export type StageViewRendererType = "image" | "overlay" | "table" | "metrics" | "json" | "gallery" | "histogram" | "mask";

export type StageViewDefinition = {
  id: string;
  label: string;
  icon?: string;
  supportedArtifactKinds?: string[];
  supportedOverlayKinds?: string[];
  priority?: number;
  emptyState?: { title: string; description?: string };
  rendererType: StageViewRendererType;
  maskViewSpec?: StageMaskViewSpec;
};

export type StageSemanticCategory = "input" | "plane_qa" | "segmentation" | "geometry" | "measurement" | "classification" | "fusion";

export type StageSemanticDefinition = {
  stageId: string;
  category: StageSemanticCategory;
  defaultViewId: string;
  views: StageViewDefinition[];
  sourceBindings?: {
    image?: boolean;
    metadata?: boolean;
    histogram?: boolean;
    json?: boolean;
  };
};

/*
Renderer choice convention:
- Use `rendererType: "mask"` for binary spatial inclusion/exclusion artifacts.
- Use `rendererType: "image"` for scalar rasters and heatmaps such as residuals, gradients, and height previews.
- Use `rendererType: "overlay"` for contours, labels, and classification overlays.
*/

function stageMaskViewSpec(spec: StageMaskViewSpec): StageMaskViewSpec {
  return spec;
}

function stageKind(stageId: string): StageSemanticCategory {
  const id = canonicalStageId(stageId).toLowerCase();
  if (id.includes("detect_belt_plane") || id.includes("normalize_heights_to_plane")) return "plane_qa";
  if (id.includes("remove_belt_segment_objects") || id.includes("height_segmentation") || id.includes("connected_components")) return "segmentation";
  if (id.includes("fit_object_geometry")) return "geometry";
  if (id.includes("compute_height_metrics")) return "measurement";
  if (id.includes("classify_25d") || id.includes("overlays_25d")) return "classification";
  if (id.includes("input") || id.includes("crop") || id.includes("gray") || id.includes("normalize")) return "input";
  if (id.includes("threshold") || id.includes("morph") || id.includes("segment")) return "segmentation";
  if (id.includes("blob") || id.includes("contour") || id.includes("ellipse") || id.includes("circle") || id.includes("detect")) return "geometry";
  if (id.includes("measure") || id.includes("metric")) return "measurement";
  if (id.includes("class") || id.includes("summary")) return "classification";
  return "fusion";
}

const VIEWS_BY_CATEGORY: Record<StageSemanticCategory, StageViewDefinition[]> = {
  input: [
    { id: "image", label: "Image", rendererType: "image", priority: 1, emptyState: { title: "No source image available yet." } },
    { id: "metadata", label: "Metadata", rendererType: "table", priority: 2, emptyState: { title: "No source metadata available." } },
    { id: "histogram", label: "Histogram", rendererType: "histogram", priority: 3, emptyState: { title: "No image histogram available yet." } },
    { id: "json", label: "JSON", rendererType: "json", priority: 99 },
  ],
  plane_qa: [
    { id: "image", label: "Image", rendererType: "image", priority: 1, emptyState: { title: "No 25D QA image available yet." } },
    { id: "histogram", label: "Histogram", rendererType: "histogram", priority: 2, emptyState: { title: "No normalized-height histogram available yet." } },
    { id: "json", label: "JSON", rendererType: "json", priority: 99 },
  ],
  segmentation: [
    { id: "threshold_mask", label: "Threshold mask", rendererType: "image", priority: 1, emptyState: { title: "No threshold mask generated yet.", description: "Run the pipeline to inspect morphology results." } },
    { id: "cleaned_mask", label: "Cleaned mask", rendererType: "image", priority: 2, emptyState: { title: "No cleaned mask generated yet." } },
    { id: "overlay", label: "Overlay", rendererType: "overlay", priority: 3, emptyState: { title: "No segmentation overlays available yet." } },
    { id: "params", label: "Morphology params", rendererType: "table", priority: 4 },
    { id: "json", label: "JSON", rendererType: "json", priority: 99 },
  ],
  geometry: [
    { id: "candidate_overlay", label: "Candidate overlay", rendererType: "overlay", priority: 1, emptyState: { title: "No contour candidates detected yet." } },
    { id: "labels", label: "Labels", rendererType: "image", priority: 2, emptyState: { title: "No blob labels generated yet." } },
    { id: "blob_metrics", label: "Blob metrics", rendererType: "table", priority: 3 },
    { id: "candidates_table", label: "Candidates table", rendererType: "table", priority: 4 },
    { id: "json", label: "JSON", rendererType: "json", priority: 99 },
  ],
  measurement: [
    { id: "measurements", label: "Objects", rendererType: "table", priority: 1 },
    { id: "fit_metrics", label: "Fit metrics", rendererType: "metrics", priority: 2, emptyState: { title: "Ellipse fitting has not been executed for this take." } },
    { id: "diameter_histogram", label: "Diameter histogram", rendererType: "histogram", priority: 3 },
    { id: "geometry_debug", label: "Geometry debug", rendererType: "table", priority: 4, emptyState: { title: "No geometry debug artifacts available yet." } },
    { id: "json", label: "JSON", rendererType: "json", priority: 99 },
  ],
  classification: [
    { id: "overlay", label: "Overlay", rendererType: "overlay", priority: 1, emptyState: { title: "No classification overlay available yet." } },
    { id: "classification", label: "Classification", rendererType: "table", priority: 2, emptyState: { title: "No classification summary available yet." } },
    { id: "rule_explanation", label: "Rule explanation", rendererType: "table", priority: 3, emptyState: { title: "No classification explanation artifact found. Reprocess this take." } },
    { id: "metric_details", label: "Metric details", rendererType: "table", priority: 4, emptyState: { title: "No metric explanation artifact found. Reprocess this take." } },
    { id: "summary", label: "Summary", rendererType: "metrics", priority: 5 },
    { id: "rejected", label: "Rejected objects", rendererType: "table", priority: 6 },
    { id: "export", label: "Export", rendererType: "table", priority: 7 },
    { id: "json", label: "Result JSON", rendererType: "json", priority: 99 },
  ],
  fusion: [
    { id: "overview", label: "Overview", rendererType: "metrics", priority: 1, emptyState: { title: "Fusion outputs are not available yet." } },
    { id: "json", label: "JSON", rendererType: "json", priority: 99 },
  ],
};

export function stageSemanticDefinition(stageId: string): StageSemanticDefinition {
  const canonical = canonicalStageId(stageId || "");
  if (canonical === "load_heightmap") {
    return {
      stageId: canonical,
      category: "input",
      defaultViewId: "height_preview",
      views: [
        { id: "height_preview", label: "Height preview", rendererType: "image", priority: 1, emptyState: { title: "No height preview available yet." } },
        { id: "heightmap", label: "Heightmap (raw)", rendererType: "image", priority: 2, emptyState: { title: "No heightmap preview available yet." } },
        { id: "json", label: "JSON", rendererType: "json", priority: 99 },
      ],
    };
  }
  if (canonical === "normalize_heights_to_plane" || canonical === "normalized_height") {
    return {
      stageId: canonical,
      category: "plane_qa",
      defaultViewId: "normalized_height",
      views: [
        { id: "raw_heightmap", label: "Raw heightmap", rendererType: "image", priority: 1, emptyState: { title: "No raw heightmap preview available yet." } },
        { id: "reference_surface", label: "Reference surface model", rendererType: "json", priority: 2, emptyState: { title: "No reference-surface model available yet." } },
        { id: "normalized_height", label: "Normalized height", rendererType: "image", priority: 1, emptyState: { title: "No normalized height preview available yet." } },
        { id: "residuals", label: "Residuals", rendererType: "image", priority: 3, emptyState: { title: "No residual visualization available yet." } },
        { id: "diagnostics", label: "Diagnostics", rendererType: "table", priority: 4, emptyState: { title: "No normalization diagnostics available yet." } },
        {
          id: "below_reference",
          label: "Below/equal reference",
          rendererType: "mask",
          priority: 4,
          maskViewSpec: stageMaskViewSpec({
            maskArtifactId: "below_reference_mask",
            diagnosticsLabel: "Normalization mask",
          }),
          emptyState: { title: "No below-reference mask available yet." },
        },
        {
          id: "above_threshold",
          label: "Above threshold",
          rendererType: "mask",
          priority: 5,
          maskViewSpec: stageMaskViewSpec({
            maskArtifactId: "above_threshold_mask",
            diagnosticsLabel: "Normalization mask",
          }),
          emptyState: { title: "No above-threshold mask available yet." },
        },
        { id: "histogram", label: "Histogram", rendererType: "histogram", priority: 6, emptyState: { title: "No normalized-height histogram available yet." } },
        { id: "json", label: "JSON", rendererType: "json", priority: 99 },
      ],
    };
  }
  if (canonical === "detect_belt_plane" || canonical === "belt_plane") {
    return {
      stageId: canonical,
      category: "plane_qa",
      defaultViewId: "selected_surface",
      views: [
        { id: "raw_heightmap", label: "Raw heightmap", rendererType: "image", priority: 1, emptyState: { title: "No raw heightmap preview available yet." } },
        {
          id: "valid_mask",
          label: "Valid mask",
          rendererType: "mask",
          priority: 2,
          maskViewSpec: stageMaskViewSpec({
            maskArtifactId: "valid_mask",
            maskOnLabel: "Valid sample",
            maskOffLabel: "Invalid sample",
            diagnosticsLabel: "Input validity",
            description: "Pixels with valid source data before reference-surface filtering.",
          }),
          emptyState: { title: "No valid mask available yet." },
        },
        { id: "depth_gradient", label: "Depth gradient", rendererType: "image", priority: 3, emptyState: { title: "No depth gradient preview available yet." } },
        {
          id: "low_gradient_mask",
          label: "Low-gradient mask",
          rendererType: "mask",
          priority: 4,
          maskViewSpec: stageMaskViewSpec({
            maskArtifactId: "low_gradient_mask",
            preferredReferenceArtifactIds: ["raw_heightmap", "raw_heightmap_preview", "heightmap_input_preview", "source_heightmap_preview", "take_thumbnail"],
            defaultMode: "overlay",
            defaultReferenceRole: "raw",
          }),
          emptyState: { title: "No low-gradient mask available yet." },
        },
        { id: "surface_candidates", label: "Surface candidates", rendererType: "json", priority: 5, emptyState: { title: "No reference-surface candidates available yet." } },
        {
          id: "background_candidates",
          label: "Background candidates",
          rendererType: "mask",
          priority: 5,
          maskViewSpec: stageMaskViewSpec({
            maskArtifactId: "background_candidate_mask",
            diagnosticsLabel: "Candidate support",
          }),
          emptyState: { title: "No background candidate mask available yet." },
        },
        {
          id: "background_seeds",
          label: "Background seeds",
          rendererType: "mask",
          priority: 5,
          maskViewSpec: stageMaskViewSpec({
            maskArtifactId: "background_seed_mask",
            diagnosticsLabel: "Seed support",
          }),
          emptyState: { title: "No background seed mask available yet." },
        },
        {
          id: "plane_fit_roi",
          label: "Plane-fit ROI",
          rendererType: "mask",
          priority: 5,
          maskViewSpec: stageMaskViewSpec({
            maskArtifactId: "plane_fit_roi_mask",
            diagnosticsLabel: "ROI mask",
          }),
          emptyState: { title: "No plane-fit ROI mask available yet." },
        },
        {
          id: "height_gate",
          label: "Height gate",
          rendererType: "mask",
          priority: 5,
          maskViewSpec: stageMaskViewSpec({
            maskArtifactId: "height_gate_mask",
            diagnosticsLabel: "Height gate",
          }),
          emptyState: { title: "No height-gate mask available yet." },
        },
        {
          id: "selected_surface",
          label: "Selected surface",
          rendererType: "mask",
          priority: 6,
          maskViewSpec: stageMaskViewSpec({
            maskArtifactIds: ["selected_reference_support_mask", "reference_surface_selected_mask", "expanded_plane_mask"],
            preferredReferenceArtifactIds: ["raw_heightmap", "raw_heightmap_preview", "heightmap_input_preview", "source_heightmap_preview", "normalized_heightmap", "take_thumbnail"],
            diagnosticsLabel: "Reference support",
          }),
          emptyState: { title: "No selected surface mask available yet." },
        },
        {
          id: "expanded_plane",
          label: "Expanded plane",
          rendererType: "mask",
          priority: 6,
          maskViewSpec: stageMaskViewSpec({
            maskArtifactId: "expanded_plane_mask",
            diagnosticsLabel: "Residual expansion",
          }),
          emptyState: { title: "No expanded-plane mask available yet." },
        },
        {
          id: "plane_inliers",
          label: "Plane inliers",
          rendererType: "mask",
          priority: 7,
          maskViewSpec: stageMaskViewSpec({
            maskArtifactIds: ["reference_model_inlier_mask", "final_plane_inlier_mask", "plane_inlier_mask", "expanded_plane_mask"],
            preferredReferenceArtifactIds: ["raw_heightmap", "raw_heightmap_preview", "heightmap_input_preview", "source_heightmap_preview", "reference_surface_selected_mask", "expanded_plane_mask"],
            diagnosticsLabel: "Fit support",
          }),
          emptyState: { title: "No plane inlier mask available yet." },
        },
        { id: "residual_heatmap", label: "Residual heatmap", rendererType: "image", priority: 8, emptyState: { title: "No residual heatmap available yet." } },
        { id: "residual_histogram", label: "Residual histogram", rendererType: "table", priority: 9, emptyState: { title: "No residual histogram available yet." } },
        { id: "diagnostics", label: "Diagnostics", rendererType: "table", priority: 10, emptyState: { title: "No plane diagnostics available yet." } },
        { id: "depth_plot", label: "Depth plot", rendererType: "image", priority: 9, emptyState: { title: "No depth plot available yet." } },
        { id: "plateau_plot", label: "Plateau plot", rendererType: "image", priority: 10, emptyState: { title: "Plateau plot not available.", description: "Run the stage with strategy=low_gradient_depth_plateaus to populate this view." } },
        { id: "filtered_depth_plot", label: "Filtered depth plot", rendererType: "image", priority: 10, emptyState: { title: "Filtered depth plot not available.", description: "Run the stage with strategy=low_gradient_depth_plateaus to see the sorted-z distribution of low-gradient candidates only." } },
        { id: "blob_components", label: "Blob components", rendererType: "image", priority: 10, emptyState: { title: "Blob components not available.", description: "Run the stage with strategy=low_gradient_blob_height_clusters to inspect low-gradient connected blobs." } },
        { id: "height_aware_blob_components", label: "XY-only components", rendererType: "image", priority: 10, emptyState: { title: "XY-only components not available.", description: "Classic XY-only blob components were not emitted for this run or are unavailable as a height_aware comparison artifact." } },
        { id: "height_aware_connectivity_rejected_edges", label: "Rejected Z-connections", rendererType: "image", priority: 10, emptyState: { title: "Rejected Z-connections not available.", description: "Run the blob strategy with blob_component_mode=height_aware to inspect neighboring low-gradient pixels blocked from joining the same blob because of Z incompatibility." } },
        { id: "height_borders", label: "Height borders", rendererType: "image", priority: 10, emptyState: { title: "Height borders not available.", description: "Run the blob strategy with height-border splitting to inspect internal height-boundary strength and cut masks." } },
        { id: "height_border_fragments", label: "Height-border fragments", rendererType: "image", priority: 10, emptyState: { title: "Height-border fragments not available.", description: "Run the blob strategy with height-border splitting to inspect the spatial fragments before any optional histogram refinement." } },
        { id: "height_split_blobs", label: "Height-split blobs", rendererType: "image", priority: 10, emptyState: { title: "Height-split blobs not available.", description: "Run the blob strategy to inspect height-consistent fragments and split diagnostics." } },
        { id: "blob_clusters", label: "Blob clusters", rendererType: "table", priority: 10, emptyState: { title: "Blob clusters not available.", description: "Run the stage with strategy=low_gradient_blob_height_clusters to inspect height-cluster summaries." } },
        { id: "selected_blob_cluster", label: "Selected blob cluster", rendererType: "image", priority: 10, emptyState: { title: "Selected blob cluster not available.", description: "Run the stage with strategy=low_gradient_blob_height_clusters to inspect the chosen support cluster." } },
        { id: "selected_cluster_pre_refine", label: "Selected cluster pre-refine", rendererType: "image", priority: 10, emptyState: { title: "Selected cluster pre-refine mask not available.", description: "Run the blob strategy to inspect the selected cluster before candidate refinement." } },
        { id: "selected_cluster_refined", label: "Selected cluster refined", rendererType: "image", priority: 10, emptyState: { title: "Selected cluster refined mask not available.", description: "Run the blob strategy to inspect the selected cluster after candidate refinement." } },
        { id: "pre_stripe_support", label: "Pre-stripe support", rendererType: "image", priority: 10, emptyState: { title: "Pre-stripe support mask not available.", description: "Selected reference support before stripe suppression." } },
        { id: "removed_by_refinement", label: "Removed by refinement", rendererType: "image", priority: 10, emptyState: { title: "Removed-by-refinement mask not available.", description: "Run the blob strategy to inspect what candidate refinement removed from the selected cluster." } },
        { id: "rejected_blob_clusters", label: "Rejected blob clusters", rendererType: "image", priority: 10, emptyState: { title: "Rejected blob clusters not available.", description: "Run the stage with strategy=low_gradient_blob_height_clusters to inspect rejected support clusters." } },
        { id: "blob_cluster_scores", label: "Cluster score table", rendererType: "table", priority: 10, emptyState: { title: "Blob cluster score table not available.", description: "Run the stage with strategy=low_gradient_blob_height_clusters to inspect cluster ranking." } },
        { id: "selection_bridge", label: "Selection bridge", rendererType: "table", priority: 10, emptyState: { title: "Selection bridge not available.", description: "Run the stage with strategy=low_gradient_blob_height_clusters to inspect how components/fragments map to clusters and the selected support." } },
        { id: "selected_support_lineage", label: "Selected support lineage", rendererType: "table", priority: 10, emptyState: { title: "Selected support lineage not available.", description: "Run the blob-cluster strategy to inspect how the selected cluster becomes selected support, model support, and suppression." } },
        {
          id: "belt_bg",
          label: "Belt background",
          rendererType: "mask",
          priority: 11,
          maskViewSpec: stageMaskViewSpec({
            maskArtifactId: "belt_bg_mask",
            maskOnLabel: "Belt background",
            maskOffLabel: "Other",
            diagnosticsLabel: "Background support",
            description: "Stable belt/background pixels used to estimate the height reference.",
          }),
          emptyState: { title: "Belt background mask not available.", description: "Stable belt/background pixels used to estimate the height reference." },
        },
        {
          id: "belt_base",
          label: "Belt base",
          rendererType: "mask",
          priority: 11,
          maskViewSpec: stageMaskViewSpec({
            maskArtifactId: "belt_base_mask",
            maskOnLabel: "Belt base",
            maskOffLabel: "Other",
            diagnosticsLabel: "Stripe suppression",
            description: "Support classified as stable belt/base after stripe reasoning.",
          }),
          emptyState: { title: "Belt base mask not available.", description: "Support classified as stable belt/base after stripe reasoning." },
        },
        {
          id: "stripe_filtered_support",
          label: "Stripe-filtered support",
          rendererType: "mask",
          priority: 11,
          maskViewSpec: stageMaskViewSpec({
            maskArtifactId: "stripe_filtered_reference_support_mask",
            maskOnLabel: "Support kept",
            maskOffLabel: "Suppressed",
            diagnosticsLabel: "Stripe suppression",
            description: "Cleaned reference support passed to later reference fitting and support selection.",
          }),
          emptyState: { title: "Stripe-filtered support mask not available.", description: "Cleaned reference support passed to later reference fitting and support selection." },
        },
        {
          id: "belt_stripes",
          label: "Belt stripes",
          rendererType: "mask",
          priority: 11,
          maskViewSpec: stageMaskViewSpec({
            maskArtifactId: "belt_stripes_mask",
            preferredReferenceArtifactIds: ["raw_heightmap", "raw_heightmap_preview", "heightmap_input_preview", "source_heightmap_preview", "normalized_heightmap", "take_thumbnail"],
            diagnosticsLabel: "Stripe suppression",
            description: "Final stripe mask detected by the stripe filter.",
          }),
          emptyState: { title: "Belt stripes mask not available.", description: "Final stripe mask detected by the stripe filter." },
        },
        {
          id: "unknown_low_gradient",
          label: "Unknown low-gradient",
          rendererType: "mask",
          priority: 11,
          maskViewSpec: stageMaskViewSpec({
            maskArtifactId: "unknown_low_gradient_mask",
            maskOnLabel: "Unknown low-gradient",
            maskOffLabel: "Other",
            diagnosticsLabel: "Preserved candidates",
            description: "Low-gradient components preserved for object detection because they were neither accepted belt background nor stripe surfaces.",
          }),
          emptyState: { title: "Unknown low-gradient mask not available.", description: "Low-gradient components preserved for object detection because they were neither accepted belt background nor stripe surfaces." },
        },
        {
          id: "surface_suppression",
          label: "Surface suppression",
          rendererType: "mask",
          priority: 11,
          maskViewSpec: stageMaskViewSpec({
            maskArtifactId: "surface_suppression_mask",
            maskOnLabel: "Suppressed surface",
            maskOffLabel: "Available",
            diagnosticsLabel: "Segmentation suppression",
            description: "Explicit suppression mask used downstream: belt background plus accepted belt stripes.",
          }),
          emptyState: { title: "Surface suppression mask not available.", description: "Explicit suppression mask used downstream: belt background plus accepted belt stripes." },
        },
        {
          id: "object_search_domain",
          label: "Object search domain",
          rendererType: "mask",
          priority: 11,
          maskViewSpec: stageMaskViewSpec({
            maskArtifactId: "object_search_domain_mask",
            maskOnLabel: "Object search",
            maskOffLabel: "Suppressed surface",
            diagnosticsLabel: "Object search domain",
            description: "Remaining domain passed to object segmentation after explicit surface suppression.",
          }),
          emptyState: { title: "Object search domain mask not available.", description: "Remaining domain passed to object segmentation after explicit surface suppression." },
        },
        {
          id: "belt_stripes_tophat",
          label: "Stripes — top-hat",
          rendererType: "mask",
          priority: 12,
          maskViewSpec: stageMaskViewSpec({
            maskArtifactId: "belt_stripes_tophat_mask",
            maskOnLabel: "Top-hat stripe",
            maskOffLabel: "Other",
            diagnosticsLabel: "Stripe pass",
            description: "Morphology/top-hat response highlighting narrow raised stripe candidates.",
          }),
          emptyState: { title: "Top-hat stripe pass not available.", description: "Morphology/top-hat response highlighting narrow raised stripe candidates." },
        },
        {
          id: "belt_stripes_shape",
          label: "Stripes — shape pass",
          rendererType: "mask",
          priority: 12,
          maskViewSpec: stageMaskViewSpec({
            maskArtifactId: "belt_stripes_shape_mask",
            maskOnLabel: "Shape stripe",
            maskOffLabel: "Other",
            diagnosticsLabel: "Stripe pass",
            description: "Stripe candidates that passed shape/geometry constraints.",
          }),
          emptyState: { title: "Shape stripe pass not available.", description: "Stripe candidates that passed shape/geometry constraints." },
        },
        {
          id: "belt_above_belt",
          label: "Above belt mask",
          rendererType: "mask",
          priority: 12,
          maskViewSpec: stageMaskViewSpec({
            maskArtifactId: "belt_above_belt_mask",
            maskOnLabel: "Above belt",
            maskOffLabel: "Other",
            diagnosticsLabel: "Stripe pass",
            description: "Pixels considered above the belt/reference baseline.",
          }),
          emptyState: { title: "Above-belt mask not available.", description: "Pixels considered above the belt/reference baseline." },
        },
        {
          id: "belt_wide_object",
          label: "Wide-object mask",
          rendererType: "mask",
          priority: 12,
          maskViewSpec: stageMaskViewSpec({
            maskArtifactId: "belt_wide_object_mask",
            maskOnLabel: "Wide object",
            maskOffLabel: "Other",
            diagnosticsLabel: "Stripe pass",
            description: "Mask used to avoid treating wide object-like structures as belt stripes.",
          }),
          emptyState: { title: "Wide-object mask not available.", description: "Mask used to avoid treating wide object-like structures as belt stripes." },
        },
        {
          id: "removed_by_stripe_filter",
          label: "Removed by stripe suppression",
          rendererType: "mask",
          priority: 12,
          maskViewSpec: stageMaskViewSpec({
            maskArtifactId: "support_removed_by_stripe_filter",
            maskOnLabel: "Suppressed",
            maskOffLabel: "Kept",
            diagnosticsLabel: "Stripe suppression",
            description: "Pixels removed from support due to stripe suppression.",
          }),
          emptyState: { title: "Removed-by-stripe mask not available.", description: "Pixels removed from support due to stripe suppression." },
        },
        { id: "belt_altitude_plot", label: "Stripe altitude", rendererType: "image", priority: 11, emptyState: { title: "Belt-stripe altitude histogram not available.", description: "Height/altitude evidence above the local belt baseline." } },
        { id: "belt_baseline", label: "Local-min baseline", rendererType: "image", priority: 11, emptyState: { title: "Local-min baseline not available.", description: "Estimated local belt baseline used to detect raised stripe-like structures." } },
        {
          id: "reference_model_support",
          label: "Selected support / fit mask",
          rendererType: "mask",
          priority: 12,
          maskViewSpec: stageMaskViewSpec({
            maskArtifactIds: ["reference_model_support_mask", "selected_reference_support_mask"],
            diagnosticsLabel: "Reference support",
          }),
          emptyState: { title: "Reference-model support mask not available.", description: "Run the stage to inspect the actual pixels used as fit support for the reference model." },
        },
        {
          id: "reference_suppression_mask",
          label: "Suppression mask",
          rendererType: "mask",
          priority: 12,
          maskViewSpec: stageMaskViewSpec({
            maskArtifactId: "reference_suppression_mask",
            maskOnLabel: "Suppressed reference",
            maskOffLabel: "Kept",
            diagnosticsLabel: "Suppression mask",
            description: "Reference-derived pixels removed before foreground segmentation.",
          }),
          emptyState: { title: "Reference suppression mask not available.", description: "Run the stage to inspect which support-derived pixels will be removed downstream from segmentation." },
        },
        { id: "support_loss_waterfall", label: "Support-loss waterfall", rendererType: "table", priority: 12, emptyState: { title: "Support-loss waterfall not available.", description: "Reprocess this take with the current detect-reference diagnostics to inspect where support was lost." } },
        { id: "json", label: "JSON", rendererType: "json", priority: 99 },
      ],
    };
  }
  if (canonical === "remove_belt_segment_objects" || canonical === "height_segmentation" || canonical === "connected_components") {
    return {
      stageId: canonical,
      category: "segmentation",
      defaultViewId: "overlay",
      views: [
        {
          id: "threshold_mask",
          label: "Threshold mask",
          rendererType: "mask",
          priority: 1,
          maskViewSpec: stageMaskViewSpec({
            maskArtifactIds: ["foreground_before_plane_suppression", "normalized_height_threshold_mask", "threshold_mask", "above_threshold_mask"],
            maskOnLabel: "Foreground candidate",
            maskOffLabel: "Background",
            diagnosticsLabel: "Segmentation threshold",
            description: "Foreground candidate pixels before reference suppression cleanup.",
          }),
          emptyState: { title: "No threshold mask generated yet." },
        },
        {
          id: "cleaned_mask",
          label: "Cleaned mask",
          rendererType: "mask",
          priority: 2,
          maskViewSpec: stageMaskViewSpec({
            maskArtifactIds: ["final_object_mask", "cleaned_object_mask", "cleaned_mask", "plane_suppressed_mask"],
            preferredReferenceArtifactIds: ["normalized_heightmap", "normalized_height", "raw_heightmap", "raw_heightmap_preview", "take_thumbnail"],
            diagnosticsLabel: "Final object mask",
          }),
          emptyState: { title: "No cleaned object mask generated yet." },
        },
        {
          id: "plane_suppressed_mask",
          label: "Plane-suppressed mask",
          rendererType: "mask",
          priority: 3,
          maskViewSpec: stageMaskViewSpec({
            maskArtifactId: "plane_suppressed_mask",
            diagnosticsLabel: "Reference suppression",
          }),
          emptyState: { title: "No plane-suppressed mask generated yet." },
        },
        {
          id: "rejected_background_residuals",
          label: "Rejected background residuals",
          rendererType: "mask",
          priority: 3,
          maskViewSpec: stageMaskViewSpec({
            maskArtifactId: "rejected_background_residuals",
            diagnosticsLabel: "Foreground cleanup",
          }),
          emptyState: { title: "No rejected-background residual mask generated yet." },
        },
        {
          id: "rejected_belt_stripes",
          label: "Rejected belt stripes",
          rendererType: "mask",
          priority: 3,
          maskViewSpec: stageMaskViewSpec({
            maskArtifactId: "rejected_belt_stripes",
            diagnosticsLabel: "Foreground cleanup",
          }),
          emptyState: { title: "No rejected-belt-stripes mask generated yet." },
        },
        { id: "segmentation", label: "Connected components", rendererType: "table", priority: 3, emptyState: { title: "No connected components found yet." } },
        { id: "overlay", label: "Overlay", rendererType: "overlay", priority: 4, emptyState: { title: "No segmentation overlay available yet." } },
        { id: "json", label: "JSON", rendererType: "json", priority: 99 },
      ],
    };
  }
  if (canonical === "fit_object_geometry" || canonical === "compute_height_metrics") {
    return {
      stageId: canonical,
      category: "measurement",
      defaultViewId: "measurements_25d",
      views: [
        { id: "measurements_25d", label: "Measurements", rendererType: "table", priority: 1, emptyState: { title: "No 25D measurements available yet." } },
        { id: "residuals", label: "Residuals", rendererType: "image", priority: 2, emptyState: { title: "No geometry residual view available yet." } },
        { id: "profiles", label: "Profiles", rendererType: "image", priority: 3, emptyState: { title: "Select image artifact to inspect object profiles." } },
        { id: "provenance", label: "Provenance", rendererType: "table", priority: 4, emptyState: { title: "No object provenance available yet." } },
        { id: "geometry_debug", label: "Geometry debug", rendererType: "table", priority: 5, emptyState: { title: "No geometry debug artifacts available yet." } },
        { id: "json", label: "JSON", rendererType: "json", priority: 99 },
      ],
    };
  }
  if (canonical === "measurement_diagnostics") {
    return {
      stageId: canonical,
      category: "measurement",
      defaultViewId: "diagnostics",
      views: [
        { id: "diagnostics", label: "Diagnostics", rendererType: "table", priority: 1, emptyState: { title: "No diagnostics artifact available yet." } },
        { id: "feature_vector", label: "Feature Vector", rendererType: "table", priority: 2, emptyState: { title: "No feature vector available yet." } },
        { id: "quality_flags", label: "Quality Flags", rendererType: "table", priority: 3, emptyState: { title: "No quality flags available yet." } },
        { id: "provenance", label: "Provenance", rendererType: "table", priority: 4, emptyState: { title: "No feature provenance available yet." } },
        { id: "json", label: "JSON", rendererType: "json", priority: 99 },
      ],
    };
  }
  if (canonical === "classify_25d") {
    return {
      stageId: canonical,
      category: "classification",
      defaultViewId: "classification_25d",
      views: [
        { id: "classification_25d", label: "Classification", rendererType: "table", priority: 1, emptyState: { title: "No 25D classifications available yet." } },
        { id: "rule_explanation", label: "Rule explanation", rendererType: "table", priority: 2, emptyState: { title: "No classification rule explanation available yet." } },
        { id: "metric_details", label: "Metric details", rendererType: "table", priority: 3, emptyState: { title: "No metric explanation artifact found. Reprocess this take." } },
        { id: "diagnostics", label: "Diagnostics", rendererType: "table", priority: 4, emptyState: { title: "No classification diagnostics available yet." } },
        { id: "json", label: "JSON", rendererType: "json", priority: 99 },
      ],
    };
  }
  if (canonical === "overlays_25d") {
    return {
      stageId: canonical,
      category: "classification",
      defaultViewId: "overlays_25d",
      views: [
        { id: "overlays_25d", label: "Overlays", rendererType: "overlay", priority: 1, emptyState: { title: "No 25D overlays available yet." } },
        { id: "json", label: "JSON", rendererType: "json", priority: 99 },
      ],
    };
  }
  if (canonical === "ellipse_fitting") {
    return {
      stageId: canonical,
      category: "geometry",
      defaultViewId: "ellipse_overlay",
      views: [
        { id: "ellipse_overlay", label: "Ellipse overlay", rendererType: "overlay", priority: 1, emptyState: { title: "No ellipse fits generated yet." } },
        { id: "ellipse_metrics", label: "Metrics", rendererType: "table", priority: 2, emptyState: { title: "No ellipse metrics generated yet." } },
        { id: "candidates_table", label: "Candidates table", rendererType: "table", priority: 3, emptyState: { title: "No fitted candidates available yet." } },
        { id: "json", label: "JSON", rendererType: "json", priority: 99 },
      ],
    };
  }
  const category = stageKind(canonical);
  const views = [...VIEWS_BY_CATEGORY[category]].sort((a, b) => (a.priority ?? 100) - (b.priority ?? 100));
  const defaultViewId = category === "segmentation"
    ? (views.find((view) => view.id === "overlay")?.id ?? views[0]?.id ?? "json")
    : (views[0]?.id ?? "json");
  return {
    stageId: canonical,
    category,
    defaultViewId,
    views,
    sourceBindings: category === "input"
      ? { image: true, metadata: true, histogram: true, json: true }
      : undefined,
  };
}

export function stageSemanticSummary(stageId: string, detail: TakeDetail | null, artifacts: StudioArtifact[], execStage: StageExecutionReport | null | undefined): string {
  const category = stageKind(canonicalStageId(stageId || ""));
  const canonical = canonicalStageId(stageId || "");
  if (canonical === "detect_belt_plane" || canonical === "belt_plane") {
    const debug = artifacts.find((item) => item.artifact_id === "plane_fit_debug");
    const meta = (debug?.metadata ?? {}) as Record<string, unknown>;
    const status = String(meta.status ?? "unknown");
    const p95 = Number(meta.residual_p95_mm ?? 0);
    const sel = (meta.background_selection as Record<string, unknown> | undefined) ?? {};
    const gdbg = (sel.gradient_debug as Record<string, unknown> | undefined) ?? {};
    const area = Number(gdbg.selected_component_area_ratio ?? sel.expanded_plane_coverage_percent ?? 0) * (Number(gdbg.selected_component_area_ratio ?? NaN) <= 1 ? 100 : 1);
    const model = String(meta.reference_surface_model_type ?? "plane");
    if (model === "constant_z") {
      const zstd = Number(gdbg.selected_component_z_std_mm ?? 0);
      return `model ${model} • surface ${area.toFixed(1)}% • z std ${zstd.toFixed(1)} mm`;
    }
    return `model ${model} • surface ${area.toFixed(1)}% • residual p95 ${p95.toFixed(1)} mm`;
  }
  if (canonical === "normalize_heights_to_plane" || canonical === "normalized_height") {
    const debug = artifacts.find((item) => item.artifact_id === "normalization_debug" || item.artifact_id === "normalized_height_debug");
    const hist = artifacts.find((item) => item.artifact_id === "normalized_height_histogram");
    const meta = (debug?.metadata ?? {}) as Record<string, unknown>;
    const bg = Number(meta.background_height_p95_abs_after_normalization_mm ?? 0);
    const above = Number(meta.above_threshold_pixel_count ?? 0);
    const total = Number(((hist?.metadata ?? {}) as Record<string, unknown>).sample_count ?? 0);
    const pct = total > 0 ? (above / total) * 100.0 : 0.0;
    return `background p95 abs ${bg.toFixed(1)} mm • above threshold ${pct.toFixed(1)}%`;
  }
  if (canonical === "remove_belt_segment_objects") {
    const debug = artifacts.find((item) => item.artifact_id === "segmentation_debug");
    const meta = (debug?.metadata ?? {}) as Record<string, unknown>;
    const cc = Number(meta.connected_component_count ?? 0);
    const fg = Number(meta.foreground_coverage_percent ?? 0);
    return `${cc} component${cc === 1 ? "" : "s"} • foreground ${fg.toFixed(1)}%`;
  }
  const objects = detail?.result?.objects ?? [];
  const rejected = detail?.result?.rejected_objects ?? [];
  if (category === "segmentation") {
    const metricsArtifact = artifacts.find((item) => item.artifact_id === "morphology_metrics");
    const debugArtifact = artifacts.find((item) => item.artifact_id === "morphology_debug_json");
    const metricsMeta = (metricsArtifact?.metadata ?? {}) as Record<string, unknown>;
    const debugMeta = (debugArtifact?.metadata ?? {}) as Record<string, unknown>;
    const debugMorph = (debugMeta.morphology as Record<string, unknown> | undefined) ?? {};
    const debugRoi = (debugMeta.roi as Record<string, unknown> | undefined) ?? {};
    const componentCount = Number(debugMorph.components_after ?? metricsMeta.connected_components ?? 0);
    const coverageRatio = Number(debugMorph.cleaned_foreground_coverage ?? metricsMeta.cleaned_foreground_coverage ?? NaN);
    const coveragePercent = Number.isFinite(coverageRatio) ? coverageRatio * 100.0 : Number(metricsMeta.foreground_coverage_percent ?? 0);
    const roiEnabled = Boolean(debugRoi.enabled ?? metricsMeta.roi_enabled ?? false);
    const masks = artifacts.filter((a) => a.artifact_id === "threshold_mask" || a.artifact_id === "cleaned_mask" || a.kind === "image").length;
    if (masks) {
      const coverageText = Number.isFinite(coveragePercent) ? ` • Foreground: ${coveragePercent.toFixed(1)}%` : "";
      const componentText = componentCount > 0 ? ` • ${componentCount} connected regions` : "";
      const roiText = roiEnabled ? " • ROI enabled" : "";
      return `${masks} masks generated${coverageText}${componentText}${roiText}`;
    }
    return "No masks generated";
  }
  if (category === "geometry") {
    const entries = geometryEntries(artifacts);
    if (entries.length) {
      const rejected = geometryRejected(artifacts);
      const avgCircularity = entries.length
        ? (entries.reduce((acc, row) => acc + Number((row as Record<string, unknown>).circularity ?? 0), 0) / entries.length)
        : 0;
      return `${entries.length} candidates${rejected > 0 ? ` • ${rejected} rejected` : ""} • avg circularity ${avgCircularity.toFixed(2)}`;
    }
    const overlay = artifacts.filter((a) => a.kind === "overlay").length;
    return `${Math.max(objects.length, overlay)} candidates detected`;
  }
  if (category === "measurement") {
    const avgCirc = objects.length ? (objects.reduce((acc, obj) => acc + Number(obj.sphericity_score ?? 0), 0) / objects.length) : 0;
    return objects.length ? `${objects.length} ellipses fitted • avg circularity ${avgCirc.toFixed(2)}` : "No ellipse fits";
  }
  if (category === "classification") {
    return `${objects.length} classified • ${rejected.length} rejected`;
  }
  if (category === "input") {
    return artifacts.some((a) => a.kind === "image") ? "Source image ready" : "Waiting for input artifact";
  }
  return `${artifacts.length} artifacts • ${execStage?.warnings.length ?? 0} warnings`;
}

function geometryEntries(artifacts: StudioArtifact[]): Array<Record<string, unknown>> {
  const ellipse = artifacts.find((item) => item.artifact_id === "ellipse_metrics" || item.artifact_id === "geometry_metrics");
  const fromEllipse = (ellipse?.metadata as Record<string, unknown> | undefined)?.entries;
  if (Array.isArray(fromEllipse) && fromEllipse.length) {
    return fromEllipse.filter((item): item is Record<string, unknown> => typeof item === "object" && item !== null);
  }
  const metrics = artifacts.find((item) => item.artifact_id === "blob_metrics" || item.artifact_id === "detection_metrics");
  const fromMetrics = (metrics?.metadata as Record<string, unknown> | undefined)?.entries;
  if (Array.isArray(fromMetrics) && fromMetrics.length) {
    return fromMetrics.filter((item): item is Record<string, unknown> => typeof item === "object" && item !== null);
  }
  const contours = artifacts.find((item) => item.artifact_id === "blob_contours" || item.artifact_id === "contours");
  const fromContours = (contours?.metadata as Record<string, unknown> | undefined)?.entries;
  if (Array.isArray(fromContours)) {
    return fromContours
      .filter((item): item is Record<string, unknown> => typeof item === "object" && item !== null)
      .map((item) => ({
        ...item,
        circularity: Number(item.circularity ?? 0),
      }));
  }
  return [];
}

function geometryRejected(artifacts: StudioArtifact[]): number {
  const byId = artifacts.find((item) => item.artifact_id === "blob_metrics" || item.artifact_id === "detection_metrics");
  const rejected = Number((byId?.metadata as Record<string, unknown> | undefined)?.rejected_count ?? 0);
  return Number.isFinite(rejected) ? rejected : 0;
}

export function stageEmptyState(stageId: string, viewId: string): { title: string; description?: string } {
  const semantic = stageSemanticDefinition(stageId);
  const view = semantic.views.find((candidate) => candidate.id === viewId);
  return view?.emptyState ?? { title: `No ${view?.label?.toLowerCase() ?? "stage"} output generated yet.` };
}
