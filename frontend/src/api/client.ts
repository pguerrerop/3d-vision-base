export type Decision = "accept" | "reject" | "review";
export type TakeStatus = "incoming" | "processed" | "failed";
export type ProcessingMode = "mock" | "real";
export type AlgorithmStage = "mock" | "segmentation" | "calibrated_segmentation" | "classification" | "production";
export type ProfilingCategory =
  | "io"
  | "preprocessing"
  | "calibration_filtering"
  | "clustering"
  | "measurement"
  | "classification"
  | "debug_artifacts"
  | "output";
export type CaptureModality = "point_cloud" | "heightmap" | "reflectance" | "rgb" | "rgb_video" | "laser_rgb";

export interface ProfilingStage {
  name: string;
  category: ProfilingCategory;
  started_at?: number;
  ended_at?: number | null;
  duration_ms: number;
  input_points?: number | null;
  output_points?: number | null;
  input_objects?: number | null;
  output_objects?: number | null;
  notes?: string | null;
  metadata?: Record<string, unknown>;
}

export interface ProcessingProfiling {
  total_ms: number;
  production_ms: number;
  debug_artifacts_ms: number;
  io_ms: number;
  stages: ProfilingStage[];
}

export interface HealthResponse {
  status: "ok";
  data_dir: string;
  incoming_count: number;
  processed_count: number;
}

export interface RuntimeState {
  status: string;
  latest_take_id: string | null;
  latest_processed_take_id: string | null;
  message: string | null;
  updated_at: string | null;
  acquisition_connected: boolean;
  latest_frame_timestamp?: string | null;
  processing_lag_ms?: number | null;
  queue_size: number;
  dropped_frames: number;
  last_successful_processing?: string | null;
  current_session?: string | null;
  active_calibration?: string | null;
  acquisition_source?: string | null;
  acquisition_source_details?: Record<string, unknown>;
  preview_available?: boolean;
  preview_timestamp?: string | null;
  preview_fps_estimate?: number | null;
  preview_stale?: boolean;
  preview_source?: string | null;
  preview_path?: string | null;
  stale: boolean;
  throughput?: Record<string, unknown>;
  warnings?: string[];
}

export interface RuntimePreviewMetadata {
  source: string;
  camera_index?: number | null;
  timestamp?: string | null;
  resolution?: [number, number] | number[] | null;
  stale: boolean;
  fps_estimate?: number | null;
  path?: string | null;
  error?: string | null;
}

export interface SourceInfo {
  id: string;
  label: string;
  type: "usb_camera" | "sick_3d" | "file_take" | "simulated";
  modality: string;
  status: "live" | "stale" | "unavailable";
  last_frame_at: string | null;
  last_frame_age_seconds: number | null;
  resolution: [number, number] | number[] | null;
  fps: number | null;
  active?: boolean;
  session_id?: string | null;
}

export interface CalibrationCapture {
  id: string;
  source_id: string;
  timestamp: string;
  image_path: string;
  image_url?: string;
  resolution: [number, number] | number[];
  stale: boolean;
  detected_corners?: {
    corners_found: boolean;
    corner_count: number;
    debug_image_path?: string;
    detected_at?: string;
  } | null;
  target_type?: "charuco" | "checkerboard" | null;
}

export interface PipelineStageInfo {
  id: string;
  stage_id?: string;
  display_name: string;
  version?: string;
  description?: string;
  required_modalities?: CaptureModality[];
  optional_modalities?: CaptureModality[];
  produced_artifact_kinds?: Array<"image" | "point_cloud" | "table" | "json" | "metric" | "overlay" | "video" | "text" | "file">;
  object_outputs?: boolean;
  supports_real_time?: boolean;
  dependencies?: string[];
  optional_stage?: boolean;
  condition?: string | null;
  implemented?: boolean;
}

export interface PipelineInfo {
  id: string;
  name: string;
  display_name: string;
  pipeline_family?: "3d" | "2d" | "generic";
  execution_backend?: string;
  supported_modalities?: CaptureModality[];
  supports_live_processing?: boolean;
  supports_batch?: boolean;
  supports_partial_stages?: boolean;
  required_modalities: CaptureModality[];
  optional_modalities: CaptureModality[];
  implemented: boolean;
  description?: string;
  stages: PipelineStageInfo[];
  composition?: {
    execution_order?: string[];
    artifact_flow?: Record<string, string[]>;
    conditional_stages?: string[];
    optional_stages?: string[];
  };
}

export interface StageExecutionReport {
  stage_id: string;
  status: "success" | "warning" | "failed" | "skipped";
  started_at: number | null;
  finished_at: number | null;
  duration_ms: number;
  warnings: string[];
  errors: string[];
  input_artifact_ids: string[];
  output_artifact_ids: string[];
  object_count: number;
  rejected_count: number;
}

export interface PipelineExecutionTrace {
  pipeline_id?: string | null;
  stages: StageExecutionReport[];
}

export interface TakeSummary {
  take_id: string;
  status: TakeStatus;
  has_ready: boolean;
  has_done: boolean;
  created_at: string | null;
  decision: Decision | null;
  engine?: string | null;
  object_count?: number | null;
  ball_count?: number | null;
  plane_found?: boolean | null;
  processing_time_ms?: number | null;
  warning_count?: number;
  warnings?: string[];
  calibration_id?: string | null;
  calibration_file?: string | null;
  calibration_resolution_source?: string | null;
  modalities?: CaptureModality[];
  assets?: Record<string, Record<string, string>>;
  frame_count?: number | null;
  frameset?: Record<string, unknown> | null;
  session_id?: string | null;
  frameset_id?: string | null;
  throughput?: Record<string, unknown> | null;
  processing_by_family?: Array<{
    family: "3d" | "2d" | "generic";
    hasCompletedOutput: boolean;
    lastRunAt?: string | null;
    status: "never_processed" | "completed" | "running" | "failed" | "partial";
    sources: Array<{ type: "3d_done_marker" | "process_run"; path?: string | null; runId?: string | null; pipelineInstanceId?: string | null }>;
  }>;
  friendly_name?: string | null;
  tags?: string[];
  validation_status?: string | null;
  expected_class?: string | null;
  expected_diameter_mm?: number | null;
  thumbnail_path?: string | null;
  dataset_id?: string | null;
  dataset_name?: string | null;
  experiment_session_id?: string | null;
  experiment_session_name?: string | null;
  latest_run_status?: string | null;
}

export interface ProcessingResult {
  take_id: string;
  processed_at: string;
  processing_mode: ProcessingMode;
  processing_engine?: "legacy" | "native" | "mock" | null;
  plane_mode?: "calibrated" | "auto" | "disabled" | null;
  calibration_status?: string | null;
  calibration_resolution_source?: "explicit" | "runtime_default" | "none" | null;
  calibration_active?: boolean | null;
  input_modalities?: CaptureModality[];
  output_modalities?: string[];
  processing_pipeline?: {
    id?: string | null;
    name: string;
    display_name?: string | null;
    required_modalities: CaptureModality[];
    optional_modalities?: CaptureModality[];
    stages?: PipelineStageInfo[];
    processing_units?: PipelineStageInfo[];
    composition?: PipelineInfo["composition"];
  } | null;
  stage_outputs?: Array<{
    stage: string;
    display_name?: string | null;
    implemented?: boolean;
    artifacts: Record<string, string | null>;
    artifact_ids?: string[];
  }>;
  artifacts?: Array<{
    artifact_id: string;
    stage_id: string;
    object_id?: number | null;
    kind: "image" | "point_cloud" | "table" | "json" | "metric" | "overlay" | "video" | "text" | "file";
    title: string;
    description?: string | null;
    path?: string | null;
    mime_type?: string | null;
    preview_available: boolean;
    created_at?: string | null;
    metadata?: Record<string, unknown>;
    generated?: "explicit" | "derived";
    overlay_type?: "bbox" | "mask" | "ellipse" | "sphere_fit" | "centroid" | "polyline" | "text" | null;
    coordinate_space?: "image_pixel" | "normalized_image" | "projection_pixel" | "world_mm" | "plot_pixel" | "point_cloud_projection" | null;
    target_artifact_id?: string | null;
    geometry?: Record<string, unknown>;
    style?: Record<string, unknown>;
    source_artifact_ids?: string[];
    approximate?: boolean;
    overlay_warnings?: string[];
    projection_type?: "xy_topdown" | "xz_side" | "yz_side" | "object_crop" | "heightmap" | "depthmap" | null;
    projection_coordinate_system?: Record<string, unknown>;
    projection_metadata?: Record<string, unknown>;
    projection_transform_id?: string | null;
  }>;
  pipeline_execution?: PipelineExecutionTrace | null;
  calibration?: {
    calibration_type: string;
    source_modalities: CaptureModality[];
    resolution_source?: "explicit" | "runtime_default" | "none" | null;
    file?: string | null;
    active: boolean;
  } | null;
  algorithm_stage: AlgorithmStage;
  status: "ok" | "failed";
  summary: {
    object_count: number;
    ball_count: number;
    non_ball_count: number;
    decision: Decision;
    confidence: number | null;
  };
  plane_model: [number, number, number, number] | null;
  input_stats: {
    point_count: number;
    has_colors: boolean;
    has_normals: boolean;
    min_bound: [number, number, number];
    max_bound: [number, number, number];
    extent: [number, number, number];
    file_size_bytes?: number | null;
  } | null;
  objects: Array<{
    object_id: number;
    class_name: "ball" | "non_ball" | "unknown";
    confidence: number | null;
    point_count?: number | null;
    center_mm: [number, number, number] | null;
    dimensions_mm?: [number, number, number] | null;
    diameter_estimate_mm?: number | null;
    bbox_min_mm?: [number, number, number] | null;
    bbox_max_mm?: [number, number, number] | null;
    diameter_mm: number | null;
    sphericity_score: number | null;
    fit_rmse_mm: number | null;
    height_above_belt_mm?: { min: number; max: number; mean: number } | null;
    fraction_points_inside_belt?: number | null;
    filter_status?: "kept" | "rejected" | null;
    filter_reason?: string | null;
    artifact_ids?: string[];
  }>;
  files: {
    point_cloud: string | null;
    point_cloud_npz?: string | null;
    input_preview: string | null;
    overlay: string | null;
    debug_height: string | null;
    debug_segmentation: string | null;
    debug_plane_segmentation: string | null;
    debug_foreground: string | null;
    debug_clusters: string | null;
    debug_calibrated_planes?: string | null;
    debug_belt_polygon_topview?: string | null;
    debug_filtered_foreground?: string | null;
    debug_rejected_points?: string | null;
    debug_clusters_filtered?: string | null;
  };
  timing_ms: {
    load: number;
    segmentation: number;
    classification: number;
    total: number;
  };
  profiling?: ProcessingProfiling | null;
  error: string | null;
  calibration_id?: string | null;
  calibration_file?: string | null;
  calibration_snapshot?: SystemCalibration | Record<string, unknown> | null;
  plane_filtering?: {
    belt_plane_id: string;
    ignored_plane_ids: string[];
    input_points: number;
    ignored_plane_points: number;
    candidate_foreground_points: number;
    rejected_points: number;
    kept_objects: number;
    rejected_objects: number;
  } | null;
  rejected_objects?: ProcessingResult["objects"];
  poc_summary?: PocRunSummary | null;
  calibration_diagnostics?: Record<string, unknown> | null;
  session_id?: string | null;
  frameset_id?: string | null;
  runtime_acquisition?: Record<string, unknown> | null;
  throughput?: Record<string, unknown> | null;
  synchronization?: Record<string, unknown> | null;
  acquisition_timestamps?: Record<string, string> | null;
}

export interface PocRunSummary {
  take_id: string;
  timestamp: string;
  engine: string;
  calibration: {
    id: string | null;
    path: string | null;
    display: string;
    mode: "calibrated" | "auto" | "disabled";
    status: string;
    active: boolean;
    resolution_source: "explicit" | "runtime_default" | "none";
    valid: boolean;
    diagnostics: Record<string, unknown>;
  };
  status: {
    processing_status: string;
    demo_ready: boolean;
    trustworthy: boolean;
    calibration_valid: boolean;
  };
  input: {
    point_count: number | null;
    valid_point_percent: number | null;
    dimensions_mm: [number, number, number] | null;
    encoder_used: boolean;
    capture_source: string | null;
  };
  plane: {
    found: boolean;
    inlier_percent: number | null;
    angle_deg: number | null;
    filtering_enabled: boolean;
  };
  objects: {
    found: number;
    accepted: number;
    rejected: number;
    estimated_balls: number;
    estimated_non_balls: number;
  };
  classification: {
    average_fit_score: number | null;
    worst_fit_score: number | null;
    confidence_average: number | null;
    confidence_min: number | null;
    confidence_max: number | null;
  };
  profiling: {
    slowest_stage: { name: string; category: string; duration_ms: number } | null;
    total_processing_ms: number;
    bottleneck_category: string | null;
  };
  artifacts: Record<string, unknown>;
  warnings: string[];
}

export interface TakeDetail {
  take_id: string;
  status: TakeStatus;
  has_ready: boolean;
  has_done: boolean;
  created_at: string | null;
  metadata: Record<string, unknown> | null;
  result: ProcessingResult | null;
  labels?: Record<string, unknown> | null;
  modalities?: CaptureModality[];
  assets?: Record<string, Record<string, string>>;
  frame_count?: number | null;
  frameset?: Record<string, unknown> | null;
  session_id?: string | null;
  frameset_id?: string | null;
  throughput?: Record<string, unknown> | null;
  processing_by_family?: TakeSummary["processing_by_family"];
  take_metadata?: Record<string, unknown>;
  run_history?: Array<Record<string, unknown>>;
  object_annotations?: Array<Record<string, unknown>>;
}

export interface DatasetSummary {
  id: string;
  name: string;
  description?: string | null;
  created_at?: string | null;
  tags?: string[];
  notes?: string | null;
  session_count?: number;
  take_count?: number;
}

export interface DatasetSessionSummary {
  id: string;
  dataset_id: string;
  name: string;
  description?: string | null;
  calibration_id?: string | null;
  sensor_metadata?: Record<string, unknown>;
  conveyor_metadata?: Record<string, unknown>;
  lighting_metadata?: Record<string, unknown>;
  created_at?: string | null;
  tags?: string[];
  notes?: string | null;
  take_count?: number;
}

export interface SessionSummary {
  session_id: string;
  operator?: string | null;
  sensor_setup?: string | null;
  acquisition_mode?: string | null;
  calibration_used?: string | null;
  conveyor_speed?: number | null;
  encoder_enabled?: boolean | null;
  notes?: string | null;
  created_at: string;
  take_count: number;
  take_ids: string[];
  processed_count?: number;
  failed_count?: number;
  takes?: TakeSummary[];
}

export type PlaneLabel = "belt" | "outer_plane_ignore" | "unused";

export interface PlaneCandidate {
  plane_id: string;
  plane_model: [number, number, number, number];
  point_count: number;
  centroid_mm: [number, number, number];
  bbox_min_mm: [number, number, number];
  bbox_max_mm: [number, number, number];
  extent_mm: [number, number, number];
  avg_z_mm: number;
  normal: [number, number, number];
  roi_polygon_xy_mm: Array<[number, number]>;
  preview_image: string;
}

export interface ObjectFilterConfig {
  min_height_above_belt_mm: number;
  max_height_above_belt_mm: number;
  require_center_inside_belt: boolean;
  min_fraction_points_inside_belt: number;
}

export interface SystemCalibration {
  calibration_id: string;
  calibration_type?: string;
  source_modalities?: CaptureModality[];
  created_at: string;
  reference_take_id?: string | null;
  source_id?: string | null;
  target?: {
    type: "charuco" | "checkerboard";
    squares_x: number;
    squares_y: number;
    square_length_mm: number;
    marker_length_mm: number;
    dictionary: string;
  } | null;
  intrinsics?: {
    camera_matrix: number[][];
    dist_coeffs: number[];
    reprojection_error: number;
    image_width: number;
    image_height: number;
  } | null;
  belt_plane?: {
    homography: number[][];
    mm_per_px_x: number;
    mm_per_px_y: number;
    reference_plane_z_mm: number;
  } | null;
  planes: Array<{
    plane_id: string;
    label: PlaneLabel;
    plane_model: [number, number, number, number];
    point_count: number;
    bbox_min_mm: [number, number, number];
    bbox_max_mm: [number, number, number];
    extent_mm: [number, number, number];
    avg_z_mm: number;
    normal?: [number, number, number] | null;
    roi_polygon_xy_mm: Array<[number, number]>;
  }>;
  object_filter: ObjectFilterConfig;
}

export interface ActiveCalibration {
  active: boolean;
  path: string | null;
  calibration_id: string | null;
  status: "none" | "valid" | "missing" | "invalid" | "incompatible" | "warning";
  warning?: string | null;
  calibration_type?: string;
  source_modalities?: CaptureModality[];
  required_modalities?: CaptureModality[];
  take_modalities?: CaptureModality[] | null;
  valid_for_take?: boolean | null;
  compatibility?: {
    status: "compatible" | "warning" | "incompatible";
    confidence: number;
    warnings: string[];
    errors: string[];
    calibration_age_seconds: number;
    calibration_created_at: string;
  } | null;
}

export interface RuntimeConfigResponse {
  config: {
    default_calibration_file: string | null;
    require_calibration_for_demo: boolean;
    allow_auto_plane_without_calibration: boolean;
  };
  path: string;
  default_calibration_file: string | null;
  default_calibration_valid: boolean;
  warning: string | null;
}

export interface ProcessTemplate {
  id: string;
  name: string;
  category: "image_2d" | "image_3d" | "hybrid";
  description: string;
  supported_input_types: string[];
  steps: Array<{
    step_id: string;
    title: string;
    namespace: string;
    default_algorithm_key: string;
    description?: string;
    optional?: boolean;
    params_schema?: Record<string, unknown>;
  }>;
  default_parameters: Record<string, Record<string, unknown>>;
  ui_metadata?: Record<string, unknown>;
}

export interface PipelineStepConfig {
  step_id: string;
  algorithm_key: string;
  enabled: boolean;
  params: Record<string, unknown>;
  ui_state: Record<string, unknown>;
}

export interface PipelineInstance {
  id: string;
  name: string;
  template_id: string;
  pipeline_family: "2d" | "3d" | "generic";
  supported_input_type: string;
  configured_steps: PipelineStepConfig[];
  overrides: Record<string, unknown>;
  runtime_state: { status: string; last_run_id?: string | null; last_error?: string | null };
  execution_history: Array<Record<string, unknown>>;
  created_at: string;
  updated_at: string;
}

export interface SourceHistogramResponse {
  source: string;
  sampled_width: number;
  sampled_height: number;
  sampled_pixels: number;
  bins: number[];
  min: number;
  max: number;
  mean: number;
  bin_count: number;
  max_dimension: number;
}

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";

async function request<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${message}`);
  }
  return response.json() as Promise<T>;
}

async function put<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${message}`);
  }
  return response.json() as Promise<T>;
}

async function del<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, { method: "DELETE" });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${message}`);
  }
  return response.json() as Promise<T>;
}

export function fileUrl(takeId: string, filename: string): string {
  const encodedPath = filename
    .split("/")
    .map((segment) => encodeURIComponent(segment))
    .join("/");
  return `${API_BASE_URL}/api/takes/${encodeURIComponent(takeId)}/files/${encodedPath}`;
}

export function runtimePreviewUrl(cacheKey?: string | number): string {
  const suffix = cacheKey == null ? "" : `?t=${encodeURIComponent(String(cacheKey))}`;
  return `${API_BASE_URL}/api/runtime/preview${suffix}`;
}

export const api = {
  health: () => request<HealthResponse>("/api/health"),
  state: () => request<RuntimeState>("/api/state"),
  runtimePreviewMetadata: () => request<RuntimePreviewMetadata>("/api/runtime/preview/metadata"),
  sources: () => request<SourceInfo[]>("/api/sources"),
  pipelines: () => request<PipelineInfo[]>("/api/pipelines"),
  processTemplates: () => request<ProcessTemplate[]>("/api/process-templates"),
  pipelineInstances: () => request<PipelineInstance[]>("/api/pipeline-instances"),
  createPipelineInstance: (payload: { name: string; template_id: string; input_type: string }) =>
    post<PipelineInstance>("/api/pipeline-instances", payload),
  updatePipelineInstance: (instanceId: string, payload: { configured_steps?: PipelineStepConfig[]; overrides?: Record<string, unknown> }) =>
    put<PipelineInstance>(`/api/pipeline-instances/${encodeURIComponent(instanceId)}`, payload),
  executePipelineInstance: (instanceId: string, payload: { image_path: string }) =>
    post<{ pipeline_instance: PipelineInstance; run: Record<string, unknown> }>(`/api/pipeline-instances/${encodeURIComponent(instanceId)}/execute`, payload),
  promoteRecipeVersion: (instanceId: string, payload: { run_id?: string | null; recipe_type?: string; notes?: string | null }) =>
    post<Record<string, unknown>>(`/api/pipeline-instances/${encodeURIComponent(instanceId)}/recipes`, payload),
  runtimeConfig: () => request<RuntimeConfigResponse>("/api/runtime-config"),
  updateRuntimeConfig: (payload: RuntimeConfigResponse["config"]) => put<RuntimeConfigResponse>("/api/runtime-config", payload),
  setDefaultCalibration: (path: string) => post<RuntimeConfigResponse>("/api/runtime-config/default-calibration", { path }),
  clearDefaultCalibration: () => del<RuntimeConfigResponse>("/api/runtime-config/default-calibration"),
  takes: (sessionId?: string) => request<TakeSummary[]>(`/api/takes${sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : ""}`),
  filteredTakes: (params: { session_id?: string; dataset_id?: string; validation_status?: string; tag?: string; search?: string }) => {
    const qs = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value) qs.set(key, value);
    });
    return request<TakeSummary[]>(`/api/takes${qs.toString() ? `?${qs.toString()}` : ""}`);
  },
  latest: () => request<TakeSummary | null>("/api/takes/latest"),
  sessions: () => request<SessionSummary[]>("/api/sessions"),
  datasets: () => request<DatasetSummary[]>("/api/datasets"),
  createDataset: (payload: { name: string; notes?: string | null }) =>
    post<DatasetSummary>("/api/datasets", payload),
  datasetSessions: (datasetId: string) => request<DatasetSessionSummary[]>(`/api/datasets/${encodeURIComponent(datasetId)}/sessions`),
  createDatasetSession: (datasetId: string, payload: { name: string; notes?: string | null }) =>
    post<DatasetSessionSummary>(`/api/datasets/${encodeURIComponent(datasetId)}/sessions`, payload),
  captureTake: (payload: {
    dataset_id: string;
    dataset_session_id: string;
    friendly_name?: string | null;
    tags?: string[];
    expected_class?: string | null;
    expected_diameter_mm?: number | null;
    notes?: string | null;
    camera_index?: number;
  }) => post<{ ok: boolean; take_id: string; take?: TakeDetail | null }>("/api/takes/capture", payload),
  sessionSummary: (sessionId: string) => request<SessionSummary>(`/api/sessions/${encodeURIComponent(sessionId)}/summary`),
  take: (takeId: string) => request<TakeDetail>(`/api/takes/${encodeURIComponent(takeId)}`),
  updateTakeMetadata: (takeId: string, payload: Record<string, unknown>) =>
    put<Record<string, unknown>>(`/api/takes/${encodeURIComponent(takeId)}/metadata`, payload),
  upsertObjectAnnotation: (takeId: string, payload: Record<string, unknown>) =>
    post<Record<string, unknown>>(`/api/takes/${encodeURIComponent(takeId)}/object-annotations`, payload),
  takeSourceHistogram: (takeId: string, filename?: string, maxDim = 512, bins = 32) =>
    request<SourceHistogramResponse>(
      `/api/takes/${encodeURIComponent(takeId)}/source-histogram?max_dim=${encodeURIComponent(String(maxDim))}&bins=${encodeURIComponent(String(bins))}${filename ? `&filename=${encodeURIComponent(filename)}` : ""}`
    ),
  processTake: (takeId: string, payload: { pipeline_id: string; run_until_stage?: string | null; reprocess?: boolean }) =>
    post<Record<string, unknown>>(`/api/takes/${encodeURIComponent(takeId)}/process`, payload),
  clearTakeOutputs: (takeId: string) => del<Record<string, unknown>>(`/api/takes/${encodeURIComponent(takeId)}/outputs`),
  pocSummary: (takeId: string) => request<PocRunSummary>(`/api/takes/${encodeURIComponent(takeId)}/poc-summary`),
  labels: () => request<{ allowed_labels: string[]; takes: Array<Record<string, unknown>> }>("/api/labels"),
  updateLabels: (takeId: string, payload: { labels: string[]; notes?: string | null; reviewer?: string | null }) =>
    post<Record<string, unknown>>(`/api/takes/${encodeURIComponent(takeId)}/labels`, payload),
  objectMetrics: (takeId?: string) =>
    request<Array<Record<string, unknown>>>(`/api/exports/object-metrics${takeId ? `?take_id=${encodeURIComponent(takeId)}` : ""}`),
  calibrationReferenceTakes: () => request<TakeSummary[]>("/api/calibration/reference-takes"),
  detectCalibrationPlanes: (takeId: string) =>
    post<{ take_id: string; preview_image: string; planes: PlaneCandidate[] }>("/api/calibration/detect-planes", { take_id: takeId }),
  saveCalibration: (payload: {
    reference_take_id: string;
    plane_labels: Record<string, PlaneLabel>;
    object_filter: ObjectFilterConfig;
  }) => post<{ calibration_id: string; path: string }>("/api/calibration/save", payload),
  listCalibrations: () => request<SystemCalibration[]>("/api/calibration/list"),
  activeCalibration: (takeId?: string) => request<ActiveCalibration>(`/api/calibration/active${takeId ? `?take_id=${encodeURIComponent(takeId)}` : ""}`),
  calibration: (calibrationId: string) => request<SystemCalibration>(`/api/calibration/${encodeURIComponent(calibrationId)}`),
  recommendedCalibration: (takeId?: string) =>
    request<{ recommended: SystemCalibration | null }>(`/api/calibration/recommended${takeId ? `?take_id=${encodeURIComponent(takeId)}` : ""}`),
  capture2DCalibrationFrame: (payload: { source_id: string }) =>
    post<{
      capture_id: string;
      image_path: string;
      timestamp: string;
      resolution: [number, number] | number[];
      source_id: string;
      acquisition_mode: "direct_usb_camera" | "preview_cache_fallback";
      frame_timestamp?: string | null;
      frame_age_seconds?: number | null;
      freshness_status?: "fresh" | "stale";
      fallback_used?: boolean;
      camera_index?: number | null;
      backend?: string | null;
      direct_capture_success?: boolean;
      image_url?: string;
    }>("/api/calibration/camera-2d/capture", payload),
  list2DCalibrationCaptures: (sourceId?: string) =>
    request<{ captures: CalibrationCapture[] }>(`/api/calibration/camera-2d/captures${sourceId ? `?source_id=${encodeURIComponent(sourceId)}` : ""}`),
  refresh2DSource: (payload: { source_id: string }) =>
    post<{ source_id: string; status: string; metadata: Record<string, unknown> }>("/api/calibration/camera-2d/refresh-source", payload),
  detect2DCalibrationCorners: (payload: {
    capture_ids: string[];
    target: { type: "charuco" | "checkerboard"; squares_x: number; squares_y: number; square_length_mm: number; marker_length_mm: number; dictionary: string };
  }) => post<{ success: boolean; detections: Array<Record<string, unknown>>; capture_count: number; detected_count: number }>("/api/calibration/camera-2d/detect-corners", payload),
  calibrate2DCamera: (payload: {
    source_id: string;
    capture_ids: string[];
    target: { type: "charuco" | "checkerboard"; squares_x: number; squares_y: number; square_length_mm: number; marker_length_mm: number; dictionary: string };
    reference_plane_z_mm: number;
  }) => post<Record<string, unknown>>("/api/calibration/camera-2d/calibrate", payload),
  save2DCalibration: (payload: {
    calibration_id: string;
    source_id: string;
    target: { type: "charuco" | "checkerboard"; squares_x: number; squares_y: number; square_length_mm: number; marker_length_mm: number; dictionary: string };
    intrinsics: { camera_matrix: number[][]; dist_coeffs: number[]; reprojection_error: number; image_width: number; image_height: number };
    belt_plane: { homography: number[][]; mm_per_px_x: number; mm_per_px_y: number; reference_plane_z_mm: number };
  }) => post<{ calibration_id: string; path: string }>("/api/calibration/camera-2d/save", payload),
  generateCharucoTarget: (payload: {
    squares_x: number;
    squares_y: number;
    square_length_mm: number;
    marker_length_mm: number;
    dictionary: string;
  }) => post<{ png_path: string; pdf_path: string }>("/api/calibration/camera-2d/generate-charuco", payload),
};
