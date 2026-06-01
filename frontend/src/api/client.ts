export type Decision = "accept" | "reject" | "review";
export type TakeStatus = "incoming" | "processed" | "failed";
export type ProcessingMode = "mock" | "real";
export type AlgorithmStage = "mock" | "segmentation" | "calibrated_segmentation" | "classification" | "production";
export type ProfilingCategory =
  | "input"
  | "io"
  | "calibration"
  | "preprocessing"
  | "calibration_filtering"
  | "segmentation"
  | "clustering"
  | "geometry"
  | "measurement"
  | "classification"
  | "overlay"
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

export interface RuntimeProcessInstance {
  process_id: string;
  process_type: string;
  pid: number | null;
  desired_state: "running" | "stopped";
  observed_state: "stopped" | "starting" | "running" | "stopping" | "crashed" | "failed_startup" | "orphaned";
  status_label: string;
  status: string;
  started_at: string | null;
  stopped_at: string | null;
  restart_count: number;
  config: Record<string, unknown>;
  source_id: string | null;
  health: string;
  last_heartbeat: string | null;
  last_event_summary: string | null;
}

export interface RuntimeProcessSummary {
  process_name: string;
  runtime_role: string;
  status: "running" | "stale" | "stopped" | "failed" | string;
  pid?: number | null;
  last_heartbeat?: string | null;
  started_at?: string | null;
  host?: string | null;
  version?: string | null;
  is_stale?: boolean;
}

export interface RuntimeProcessEvent {
  timestamp: string;
  process_id: string;
  severity: "debug" | "info" | "warning" | "error";
  event_type: string;
  message: string;
  metadata: Record<string, unknown>;
}

export interface RuntimeHealthChecklistItem {
  id: string;
  ok: boolean;
  label: string;
  details: string;
}

export interface RuntimeHealthChecklist {
  process_id: string;
  items: RuntimeHealthChecklistItem[];
}

export interface RuntimeFtpStatus {
  process_id: string;
  listening: boolean;
  host: string;
  port: number;
  auth_mode: string;
  upload_dir: string;
  writable: boolean;
  active_clients: Array<Record<string, unknown>>;
  recent_uploads: Array<Record<string, unknown>>;
  last_upload_timestamp: string | null;
}

export type RuntimeWorkerState = "idle" | "running" | "error" | "stopped" | "dry_run";

export interface RuntimeWorkerStatus {
  worker_id: string;
  worker_type: string;
  state: "idle" | "running" | "error" | "stopped";
  status?: "idle" | "running" | "error" | "stopped";
  source_id?: string | null;
  pipeline_id?: string | null;
  modality?: string | null;
  station_id?: string | null;
  queue_depth?: number;
  poll_interval_sec?: number;
  started_at?: string | null;
  last_heartbeat?: string | null;
  last_activity_at?: string | null;
  last_success_at?: string | null;
  last_event_at?: string | null;
  last_event_summary?: string | null;
  last_error?: string | null;
  error_count?: number;
  processed_count?: number;
  retry_count?: number;
  stop_requested?: boolean;
}

export interface RuntimeWorkerEvent {
  worker_id: string;
  timestamp: string;
  severity: "debug" | "info" | "warning" | "error";
  event_type: string;
  message: string;
  metadata: Record<string, unknown>;
}

export interface RuntimeQueueStatus {
  pending_count: number;
  active_claims_count: number;
  completed_count: number;
  failed_count: number;
  active_claims: Array<Record<string, unknown>>;
}

export interface ReplaySessionState {
  active: boolean;
  historical?: boolean;
  status?: string;
  dataset_id?: string | null;
  session_id?: string | null;
  started_at?: string | null;
  stopped_at?: string | null;
  paused_at?: string | null;
  created_by?: string | null;
  persisted_on_restart?: boolean | null;
  persist_on_restart?: boolean | null;
  auto_start_on_startup?: boolean | null;
  replay_mode?: string | null;
  replay_source?: string | null;
  dataset_exists?: boolean;
  session_exists?: boolean;
  metadata?: Record<string, unknown>;
}

export interface RuntimeRoutingState {
  live_routing: Record<string, unknown> | null;
  persistent_default: Record<string, unknown> | null;
  resolved: {
    routing_mode: string;
    active_dataset_id?: string | null;
    active_session_id?: string | null;
    source: string;
    ownership_explanation?: string;
  };
  replay_override_active: boolean;
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

export interface SourceControlValue {
  supported: boolean;
  readable: boolean;
  writable: boolean;
  value: number | null;
  step: number | null;
  default: number | null;
  unit: string | null;
  min: number | null;
  max: number | null;
  auto_supported: boolean;
  auto_value: boolean | null;
  backend_property: string;
  range_source?: "backend" | "default_profile" | "unknown";
  warning?: string;
}

export interface SourceControlsResponse {
  source_id: string;
  backend: string | null;
  controls: Record<string, SourceControlValue>;
  warnings?: string[];
  applied?: Record<string, unknown>;
  diagnostics?: {
    raw_get?: Record<string, unknown>;
    unsupported_properties?: string[];
    last_apply_errors?: string[];
  };
  camera_index?: number;
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
  dictionary_name?: string | null;
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
  parameter_schema?: Record<string, unknown>;
}

export interface PipelineInfo {
  id: string;
  name: string;
  display_name: string;
  pipeline_family?: "3d" | "2d" | "25d" | "fusion" | "generic";
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
    family: "3d" | "2d" | "25d" | "generic";
    hasCompletedOutput: boolean;
    lastRunAt?: string | null;
    status: "never_processed" | "completed" | "running" | "failed" | "partial";
    sources: Array<{ type: "3d_done_marker" | "process_run"; path?: string | null; runId?: string | null; pipelineInstanceId?: string | null }>;
  }>;
  friendly_name?: string | null;
  tags?: string[];
  semantic_labels?: string[];
  superclass_labels?: string[];
  processed_class_label?: string | null;
  processed_superclass?: string | null;
  normalized_class?: string | null;
  normalization_version?: string | null;
  validation_status?: string | null;
  expected_class?: string | null;
  expected_diameter_mm?: number | null;
  physical_object_id?: string | null;
  thumbnail_path?: string | null;
  dataset_id?: string | null;
  dataset_name?: string | null;
  experiment_session_id?: string | null;
  experiment_session_name?: string | null;
  experiment_session_type?: string | null;
  latest_run_status?: string | null;
  categories?: string[];
  reference_type?: string | null;
  is_reference?: boolean;
  is_golden_sample?: boolean;
  acquisition_processing_status?: Record<string, unknown> | null;
}

export interface TakeSummaryPage {
  items: TakeSummary[];
  limit: number;
  offset: number;
  has_more: boolean;
  next_offset?: number | null;
  total_count?: number;
  filtered_count?: number;
  summary_counts?: {
    validation?: Record<string, number>;
    missing_labels?: number;
    missing_split?: number;
    missing_calibration?: number;
    processing_failed?: number;
    processing_incomplete?: number;
    no_objects_detected?: number;
  };
}

export interface TakeBulkFilters {
  dataset_id?: string;
  session_id?: string;
  validation_status?: string;
  search?: string;
  tag?: string;
  created_from?: string;
  created_to?: string;
  split?: string;
  expected_class?: string;
  calibration_linkage_only?: boolean;
}

export interface TakeBulkMetadataRequest {
  mode: "ids" | "filter";
  action:
    | "move_to_dataset"
    | "move_to_session"
    | "ml_set_add"
    | "ml_set_remove"
    | "tag_add"
    | "tag_remove"
    | "tag_replace"
    | "set_validation_status"
    | "set_split"
    | "set_expected_class";
  take_ids?: string[];
  filters?: TakeBulkFilters;
  exclude_take_ids?: string[];
  dataset_id?: string;
  session_id?: string;
  tags?: string[];
  validation_status?: string;
  split?: string;
  expected_class?: string;
  dry_run?: boolean;
}

export interface TakeBulkMetadataResponse {
  ok: boolean;
  mode: "ids" | "filter";
  action: string;
  matched_count: number;
  affected_count: number;
  skipped_count: number;
  failed_count: number;
  failed_ids: string[];
}

export interface DatasetMlSet {
  id: string;
  dataset_id: string;
  name: string;
  description?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
  notes?: string | null;
  member_count?: number;
  membership_mode?: string;
}

export interface DatasetSessionDetailSummary {
  total_takes: number;
  reviewed: number;
  unreviewed: number;
  rejected?: number;
  needs_review?: number;
  golden_sample?: number;
  benchmark_approved?: number;
  missing_labels: number;
  missing_split?: number;
  missing_calibration?: number;
  processing_failed?: number;
  processing_incomplete?: number;
  no_objects_detected?: number;
  split_distribution: Record<string, number>;
  processing_by_family: Record<string, number>;
  object_annotation_count: number;
  object_candidate_count?: number;
  last_acquisition_timestamp: string | null;
  calibration_assignment: string | null;
}

export interface DatasetSessionDetailResponse {
  dataset: DatasetSummary;
  session: DatasetSessionSummary;
  summary: DatasetSessionDetailSummary;
}

export interface IngestionRun {
  run_id: string;
  dataset_id: string;
  session_ids: string[];
  name?: string;
  status: string;
  created_at?: string;
  policy?: Record<string, unknown>;
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
    overlay_type?: "bbox" | "mask" | "ellipse" | "footprint_roundness" | "centroid" | "polyline" | "text" | null;
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
    class_name: string;
    subclass_label?: string | null;
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
    source_candidate_id?: string;
    decision_reasons?: string[];
    geometry?: Record<string, unknown>;
  }>;
  object_candidates?: Array<{
    id: string;
    local_object_index: number;
    source_modality: "rgb" | "heightmap" | "point_cloud" | "reflectance" | "derived_25d";
    source_take_id: string;
    source_pipeline_run_id?: string | null;
    acquisition_group_id?: string | null;
    bbox_px?: number[] | null;
    contour_px?: number[][] | null;
    mask_artifact_id?: string | null;
    overlay_artifact_id?: string | null;
    centroid_px?: number[] | null;
    centroid_world?: number[] | null;
    geometry: Record<string, unknown>;
    appearance: Record<string, unknown>;
    measurements: Record<string, unknown>;
    classification_hints: Record<string, unknown>;
    confidence?: number | null;
    diagnostics: Record<string, unknown>;
  }>;
  files: {
    point_cloud: string | null;
    point_cloud_npz?: string | null;
    heightmap?: string | null;
    heightmap_npz?: string | null;
    normalized_heightmap?: string | null;
    classification_explanation?: string | null;
    metric_explanation?: string | null;
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
  acquisition_processing_status?: Record<string, unknown> | null;
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
  environment_metadata?: Record<string, unknown>;
  session_type?: "engineering" | "curated" | "benchmark" | "operational";
  metadata?: Record<string, unknown>;
  created_at?: string | null;
  tags?: string[];
  notes?: string | null;
  take_count?: number;
}

export interface FeatureDefinition {
  feature_key: string;
  display_name: string;
  semantic_group: string;
  unit?: string | null;
  description?: string | null;
  source_stage?: string | null;
  source_metric_path?: string | null;
}

export interface FeatureAnalyticsFeaturesResponse {
  record_count: number;
  object_count: number;
  feature_definitions: FeatureDefinition[];
}

export interface FeatureAnalyticsDistributionGroup {
  group: string;
  count: number;
  bins: number[];
  stats: { min: number; max: number; mean: number; std: number };
}

export interface FeatureAnalyticsDistributionResponse {
  feature_key: string;
  group_by: string;
  bins: number;
  mode: "count" | "density";
  range: { min: number; max: number; edges: number[] } | null;
  groups: FeatureAnalyticsDistributionGroup[];
  stats: {
    count: number;
    missing: number;
    missing_pct: number;
    min: number | null;
    max: number | null;
    mean: number | null;
    std: number | null;
  };
}

export interface FeatureAnalyticsObject {
  take_id: string;
  object_id: string;
  dataset_id?: string | null;
  session_id?: string | null;
  pipeline_id?: string | null;
  run_id?: string | null;
  stage_id?: string | null;
  superclass?: string | null;
  labels?: string[];
  feature_value: number;
  timestamp?: string | null;
  annotation?: Record<string, unknown> | null;
}

export interface FeatureAnalyticsObjectsResponse {
  feature_key: string;
  min_value?: number | null;
  max_value?: number | null;
  limit: number;
  objects: FeatureAnalyticsObject[];
  stats: FeatureAnalyticsDistributionResponse["stats"];
}

export interface MLDatasetSummary {
  id: string;
  name: string;
  description?: string | null;
  source_dataset_ids: string[];
  filters: Record<string, unknown>;
  label_schema: Record<string, unknown>;
  object_count?: number;
  tags?: string[];
  notes?: string | null;
}

export interface MLExperimentSummary {
  id: string;
  name: string;
  classifier_type: string;
  classifier_types?: string[];
  experiment_mode?: string;
  feature_set_id: string;
  dataset_id: string;
  split_strategy: string;
  training_config: Record<string, unknown>;
  status: string;
  metrics_summary?: Record<string, unknown>;
  artifact_paths?: Record<string, string>;
}

export interface MLDeploymentSummary {
  id: string;
  model_id: string;
  target_pipeline_id: string;
  target_stage_id: string;
  pipeline_family: string;
  status: "inactive" | "active" | "superseded" | "rolled_back" | "invalid";
  activated_at?: string | null;
  deactivated_at?: string | null;
  superseded_by?: string | null;
  rollback_from?: string | null;
  runtime_validation?: Record<string, unknown>;
  compatibility_snapshot?: Record<string, unknown>;
}

export interface MLSmokeSummary {
  passed: boolean;
  phases: Array<{ name: string; ok: boolean; detail: string }>;
  duration_ms?: number | null;
  fallback_mode?: boolean;
  failures?: Array<{ name: string; detail: string }>;
  timestamp?: string;
}

export interface MLBackendInfo {
  id: string;
  backend_name: string;
  backend_version: string;
  interpretability_level: string;
  training_speed: string;
  inference_speed: string;
  feature_requirements: Record<string, unknown>;
  calibration_requirements: Record<string, unknown>;
  operational_compatibility?: Record<string, unknown>;
  description?: string;
  enabled?: boolean;
}

export interface MLDatasetInsights {
  dataset_id: string;
  object_count: number;
  class_counts: Record<string, number>;
  session_counts: Record<string, number>;
  source_dataset_counts: Record<string, number>;
  validation_counts: Record<string, number>;
  warnings: string[];
}

export interface FusionObjectCandidate {
  fusion_object_id: string;
  rgb_object_id?: string | null;
  heightmap_object_id?: string | null;
  matching_method: "centroid_projection" | "bbox_overlap" | "object_index_fallback" | "unmatched_rgb" | "unmatched_heightmap";
  confidence: number;
  rgb_features?: Record<string, unknown>;
  heightmap_features?: Record<string, unknown>;
  warnings?: string[];
}

export interface FusionInputBundle {
  acquisition_group_id: string;
  rgb_take_id?: string | null;
  heightmap_take_id?: string | null;
  rgb_run_id?: string | null;
  heightmap_run_id?: string | null;
  rgb_result_payload?: Record<string, unknown> | null;
  heightmap_result_payload?: Record<string, unknown> | null;
  rgb_artifacts?: Array<Record<string, unknown>>;
  heightmap_artifacts?: Array<Record<string, unknown>>;
  readiness_status: "waiting_for_rgb" | "waiting_for_heightmap" | "waiting_for_processing" | "ready_for_fusion" | "incomplete" | "failed_input";
  missing_inputs: string[];
  warnings: string[];
  created_at?: string;
  resolved_at?: string;
  debug?: Record<string, unknown>;
}

export interface FusionPreviewResult {
  acquisition_group_id: string;
  readiness_status: FusionInputBundle["readiness_status"];
  object_candidates: FusionObjectCandidate[];
  merged_feature_summary: Record<string, unknown>;
  recommended_next_action: string;
  warnings: string[];
  inputs?: Record<string, unknown>;
}

export interface FusionRunRecord {
  fusion_run_id: string;
  acquisition_group_id: string;
  pipeline_id: string;
  pipeline_family: "fusion";
  recipe_version_id?: string | null;
  config_snapshot_hash?: string | null;
  rgb_take_id?: string | null;
  heightmap_take_id?: string | null;
  rgb_run_id?: string | null;
  heightmap_run_id?: string | null;
  status: string;
  started_at: string;
  completed_at?: string | null;
  result_path: string;
  artifact_paths: Record<string, string>;
  warnings: string[];
}

export interface FusionRunResult {
  acquisition_group_id: string;
  fusion_run_id: string;
  status: string;
  final_objects: Array<Record<string, unknown>>;
  class_counts: Record<string, number>;
  warnings: string[];
  source_refs: Record<string, unknown>;
  artifact_refs: Record<string, unknown>;
}

export type PublishedOverallDecision = "accept" | "reject" | "review" | "unknown";
export type PublishedInspectionStatus = "pending" | "complete" | "incomplete" | "failed";

export interface PublishedInspectionObject {
  final_object_id: string;
  final_class: string;
  final_class_group: string;
  confidence?: number | null;
  decision_reasons: string[];
  key_measurements: Record<string, unknown>;
  display_label: string;
  sort_order: number;
}

export interface PublishedInspectionResult {
  published_result_id: string;
  acquisition_group_id: string;
  fusion_run_id: string;
  station_id?: string | null;
  session_id?: string | null;
  timestamp: string;
  status: PublishedInspectionStatus;
  overall_decision: PublishedOverallDecision;
  primary_class?: string | null;
  primary_class_group?: string | null;
  confidence?: number | null;
  class_counts: Record<string, number>;
  objects: PublishedInspectionObject[];
  warnings: string[];
  display_artifacts: Record<string, string | null>;
  source_refs: Record<string, unknown>;
}

export interface InspectionPublishedResult {
  id: string;
  acquisition_group_id: string;
  fusion_run_id: string;
  station_id?: string | null;
  created_at: string;
  status: "pending" | "complete" | "incomplete" | "failed";
  summary?: {
    total_objects: number;
    counts_by_class: Record<string, number>;
    counts_by_class_group: Record<string, number>;
    has_warnings: boolean;
  };
  warnings?: string[];
}

export interface OperationsCard {
  take_id: string;
  status: "acquired" | "queued" | "processing" | "completed" | "failed" | string;
  superclass: "BALL_GOOD" | "BALL_SCRAP" | "SCRAP_METAL" | "UNKNOWN" | string;
  label: string;
  confidence?: number | null;
  object_count?: number;
  preview_image?: string | null;
  acquired_at?: string | null;
  processed_at?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
  source?: string | null;
  error?: string | null;
  pipeline_id?: string | null;
  classification_variables?: {
    max_height_mm?: number | null;
    p95_height_mm?: number | null;
    eccentricity?: number | null;
    sphericity_3d?: number | null;
    flatness?: number | null;
    edge_roughness?: number | null;
    footprint_roundness?: number | null;
    volume_proxy_mm3?: number | null;
  } | null;
}

export interface ProcessBinding {
  id: string;
  name: string;
  source_id: string;
  modality: string;
  purpose: "acquisition_inspection" | "manual_debug" | "fusion_input" | "fusion";
  pipeline_id: string;
  active_recipe_version_id: string;
  calibration_profile_id?: string | null;
  enabled: boolean;
  created_at: string;
  updated_at: string;
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
  camera_runtime_settings?: Record<string, number | boolean | null> | null;
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
    runtime?: {
      trispector_ftp?: {
        enabled?: boolean;
        command?: string;
        upload_dir?: string;
        poll_interval_sec?: number;
        stable_checks?: number;
        source_id?: string;
        auth_mode?: string;
        username?: string | null;
      };
    };
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
  pipeline_family: "2d" | "3d" | "25d" | "generic";
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

export interface SegmentationPreviewResponse {
  take_id: string;
  pipeline_id: string;
  pipeline_instance_id: string;
  source_image: string;
  artifacts: NonNullable<ProcessingResult["artifacts"]>;
  diagnostics: {
    status: string;
    warnings: string[];
    step_reports: Array<Record<string, unknown>>;
    preview?: boolean;
    effective_params?: Record<string, unknown>;
  };
  metrics: {
    component_count: number;
    threshold_coverage: number;
    cleaned_coverage: number;
    removed_components: number;
  };
  applied_params: {
    threshold: number;
    auto_threshold: boolean;
    invert: boolean;
    threshold_mode: string;
    morph_op?: string;
    erode_kernel_size?: number;
    erode_iterations?: number;
    dilate_kernel_size?: number;
    dilate_iterations?: number;
    close_kernel_size?: number;
    fill_holes?: boolean;
    min_area_px?: number;
    max_area_px?: number | null;
    roi_enabled?: boolean;
  };
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

export function objectThumbnailUrl(takeId: string, objectId: string | number, size = 56): string {
  return `${API_BASE_URL}/api/takes/${encodeURIComponent(takeId)}/objects/${encodeURIComponent(String(objectId))}/thumbnail?size=${encodeURIComponent(String(size))}`;
}

export const api = {
  health: () => request<HealthResponse>("/api/health"),
  state: () => request<RuntimeState>("/api/state"),
  runtimePreviewMetadata: () => request<RuntimePreviewMetadata>("/api/runtime/preview/metadata"),
  runtimeProcesses: () => request<RuntimeProcessSummary[]>("/api/runtime/processes"),
  runtimeProcess: (processId: string) => request<RuntimeProcessInstance>(`/api/runtime/processes/${encodeURIComponent(processId)}`),
  startRuntimeProcess: (processId: string) => post<RuntimeProcessInstance>(`/api/runtime/processes/${encodeURIComponent(processId)}/start`, {}),
  stopRuntimeProcess: (processId: string) => post<RuntimeProcessInstance>(`/api/runtime/processes/${encodeURIComponent(processId)}/stop`, {}),
  restartRuntimeProcess: (processId: string) => post<RuntimeProcessInstance>(`/api/runtime/processes/${encodeURIComponent(processId)}/restart`, {}),
  runtimeProcessLogs: (processId: string, limit = 200) =>
    request<{ process_id: string; lines: string[] }>(
      `/api/runtime/processes/${encodeURIComponent(processId)}/logs?limit=${encodeURIComponent(String(limit))}`
    ),
  runtimeProcessEvents: (processId: string, limit = 200) =>
    request<{ process_id: string; events: RuntimeProcessEvent[] }>(
      `/api/runtime/processes/${encodeURIComponent(processId)}/events?limit=${encodeURIComponent(String(limit))}`
    ),
  runtimeProcessHealthChecklist: (processId: string) =>
    request<RuntimeHealthChecklist>(`/api/runtime/processes/${encodeURIComponent(processId)}/health-checklist`),
  runtimeProcessFtpStatus: (processId: string) =>
    request<RuntimeFtpStatus>(`/api/runtime/processes/${encodeURIComponent(processId)}/ftp-status`),
  runtimeWorkers: () => request<{ workers: RuntimeWorkerStatus[]; queue?: RuntimeQueueStatus }>("/api/runtime/workers"),
  runtimeQueue: () => request<RuntimeQueueStatus>("/api/runtime/queue"),
  runtimeWorker: (workerId: string) => request<RuntimeWorkerStatus>(`/api/runtime/workers/${encodeURIComponent(workerId)}`),
  runtimeWorkerEvents: (workerId: string, limit = 200) =>
    request<{ worker_id: string; events: RuntimeWorkerEvent[] }>(
      `/api/runtime/workers/${encodeURIComponent(workerId)}/events?limit=${encodeURIComponent(String(limit))}`
    ),
  stopRuntimeWorker: (workerId: string) =>
    post<{ worker_id: string; stop_requested: boolean }>(`/api/runtime/workers/${encodeURIComponent(workerId)}/stop-request`, {}),
  activeReplaySession: () =>
    request<{ active_session: ReplaySessionState | null }>("/api/acquisition/replay/active-session"),
  startReplaySession: (payload: { dataset_id: string; session_id: string; metadata?: Record<string, unknown> }) =>
    post<ReplaySessionState>("/api/acquisition/replay/sessions/start", payload),
  stopReplaySession: () =>
    post<{ stopped_session: ReplaySessionState | null }>("/api/acquisition/replay/sessions/stop", {}),
  runtimeReplayStart: (payload: {
    dataset_id: string;
    session_id: string;
    metadata?: Record<string, unknown>;
    persist_on_restart?: boolean;
    replay_mode?: string;
    auto_start_on_startup?: boolean;
    replay_source?: string;
  }) => post<ReplaySessionState>("/api/runtime/replay/start", payload),
  runtimeReplayStop: () => post<{ stopped_session: ReplaySessionState | null }>("/api/runtime/replay/stop", {}),
  runtimeReplayPause: () => post<{ paused_session: ReplaySessionState | null }>("/api/runtime/replay/pause", {}),
  runtimeReplayClear: () => post<{ cleared: boolean }>("/api/runtime/replay/clear", {}),
  updateRuntimeReplayConfig: (payload: {
    dataset_id?: string;
    session_id?: string;
    persist_on_restart?: boolean;
    auto_start_on_startup?: boolean;
    replay_mode?: string;
    replay_source?: string;
    metadata?: Record<string, unknown>;
  }) => put<{ session: ReplaySessionState }>("/api/runtime/replay/config", payload),
  runtimeRouting: () => request<RuntimeRoutingState>("/api/runtime/routing"),
  updateRuntimeRouting: (payload: {
    dataset_id: string;
    session_id: string;
    routing_mode?: string;
    source?: string;
    updated_by?: string;
  }) => put<{ live_routing: Record<string, unknown> }>("/api/runtime/routing", payload),
  activateRuntimeRouting: (payload: {
    dataset_id: string;
    session_id: string;
    routing_mode?: string;
    source?: string;
    updated_by?: string;
  }) => post<{ live_routing: Record<string, unknown> }>("/api/runtime/routing/activate", payload),
  clearRuntimeRouting: () => post<{ cleared: boolean }>("/api/runtime/routing/clear", {}),
  runtimeDefaultDestination: () => request<{ default_destination: Record<string, unknown> | null }>("/api/runtime/default-destination"),
  updateRuntimeDefaultDestination: (payload: {
    default_dataset_id: string;
    default_session_id: string;
    auto_create_session?: boolean;
    session_naming_policy?: string;
    updated_by?: string;
  }) => put<{ default_destination: Record<string, unknown> }>("/api/runtime/default-destination", payload),
  clearRuntimeDefaultDestination: () => del<{ cleared: boolean }>("/api/runtime/default-destination"),
  sources: () => request<SourceInfo[]>("/api/sources"),
  sourceControls: (sourceId: string) => request<SourceControlsResponse>(`/api/sources/${encodeURIComponent(sourceId)}/controls`),
  updateSourceControls: (sourceId: string, payload: { controls: Record<string, { value?: number | null; auto?: boolean | null }> }) =>
    post<SourceControlsResponse>(`/api/sources/${encodeURIComponent(sourceId)}/controls`, payload),
  pipelines: () => request<PipelineInfo[]>("/api/pipelines"),
  processTemplates: () => request<ProcessTemplate[]>("/api/process-templates"),
  pipelineInstances: () => request<PipelineInstance[]>("/api/pipeline-instances"),
  processBindings: () => request<ProcessBinding[]>("/api/process-bindings"),
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
  filteredTakes: (params: {
    session_id?: string;
    dataset_id?: string;
    validation_status?: string;
    tag?: string;
    semantic_label?: string;
    superclass_label?: string;
    physical_object_id?: string;
    expected_class?: string;
    split?: string;
    calibration_linkage_only?: boolean;
    search?: string;
    show_archived?: boolean;
    session_type?: string;
    category?: string;
    reference_type?: string;
    is_reference?: boolean;
    is_golden_sample?: boolean;
  }) => {
    const qs = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== "") qs.set(key, String(value));
    });
    return request<TakeSummary[]>(`/api/takes${qs.toString() ? `?${qs.toString()}` : ""}`);
  },
  pagedTakes: (params: {
    limit?: number;
    offset?: number;
    session_id?: string;
    dataset_id?: string;
    validation_status?: string;
    tag?: string;
    semantic_label?: string;
    superclass_label?: string;
    physical_object_id?: string;
    search?: string;
    created_from?: string;
    created_to?: string;
    show_archived?: boolean;
    session_type?: string;
    category?: string;
    reference_type?: string;
    is_reference?: boolean;
    is_golden_sample?: boolean;
  }) => {
    const qs = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== "") qs.set(key, String(value));
    });
    return request<TakeSummaryPage>(`/api/takes/paged${qs.toString() ? `?${qs.toString()}` : ""}`);
  },
  latest: () => request<TakeSummary | null>("/api/takes/latest"),
  sessions: () => request<SessionSummary[]>("/api/sessions"),
  datasets: () => request<DatasetSummary[]>("/api/datasets"),
  createDataset: (payload: { name: string; notes?: string | null }) =>
    post<DatasetSummary>("/api/datasets", payload),
  datasetSessions: (datasetId: string) => request<DatasetSessionSummary[]>(`/api/datasets/${encodeURIComponent(datasetId)}/sessions`),
  featureAnalyticsFeatures: (params: {
    datasets?: string;
    sessions?: string;
    labels?: string;
    superclass?: string;
    validation_status?: string;
    split?: string;
    pipeline?: string;
    calibration?: string;
    date_from?: string;
    date_to?: string;
    feature_selection?: string;
  }) => {
    const qs = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== "") qs.set(key, String(value));
    });
    return request<FeatureAnalyticsFeaturesResponse>(`/api/feature-analytics/features${qs.toString() ? `?${qs.toString()}` : ""}`);
  },
  featureAnalyticsDistributions: (params: {
    feature_key: string;
    group_by?: "superclass" | "label";
    bins?: number;
    mode?: "count" | "density";
    datasets?: string;
    sessions?: string;
    labels?: string;
    superclass?: string;
    validation_status?: string;
    split?: string;
    pipeline?: string;
    calibration?: string;
    date_from?: string;
    date_to?: string;
  }) => {
    const qs = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== "") qs.set(key, String(value));
    });
    return request<FeatureAnalyticsDistributionResponse>(`/api/feature-analytics/distributions?${qs.toString()}`);
  },
  featureAnalyticsObjects: (params: {
    feature_key: string;
    min_value?: number;
    max_value?: number;
    limit?: number;
    datasets?: string;
    sessions?: string;
    labels?: string;
    superclass?: string;
    validation_status?: string;
    split?: string;
    pipeline?: string;
    calibration?: string;
    date_from?: string;
    date_to?: string;
  }) => {
    const qs = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== "") qs.set(key, String(value));
    });
    return request<FeatureAnalyticsObjectsResponse>(`/api/feature-analytics/objects?${qs.toString()}`);
  },
  createDatasetSession: (datasetId: string, payload: { name: string; notes?: string | null }) =>
    post<DatasetSessionSummary>(`/api/datasets/${encodeURIComponent(datasetId)}/sessions`, payload),
  updateDatasetSession: (datasetId: string, sessionId: string, payload: Record<string, unknown>) =>
    put<DatasetSessionSummary>(`/api/datasets/${encodeURIComponent(datasetId)}/sessions/${encodeURIComponent(sessionId)}`, payload),
  mlDatasets: () => request<MLDatasetSummary[]>("/api/ml/datasets"),
  createMlDataset: (payload: MLDatasetSummary) => post<MLDatasetSummary>("/api/ml/datasets", payload),
  mlExperiments: () => request<MLExperimentSummary[]>("/api/ml/experiments"),
  mlBackends: () => request<MLBackendInfo[]>("/api/ml/backends"),
  runMlExperiment: (payload: Omit<MLExperimentSummary, "status" | "metrics_summary" | "artifact_paths">) =>
    post<MLExperimentSummary>("/api/ml/experiments", payload),
  mlModels: () => request<Array<Record<string, unknown>>>("/api/ml/models"),
  promoteMlModel: (modelId: string, notes?: string | null) =>
    post<Record<string, unknown>>(`/api/ml/models/${encodeURIComponent(modelId)}/promote`, { notes: notes ?? null }),
  evaluateMl: (payload: Omit<MLExperimentSummary, "status" | "metrics_summary" | "artifact_paths">) =>
    post<MLExperimentSummary>("/api/ml/evaluate", payload),
  mlDeployments: () => request<MLDeploymentSummary[]>("/api/ml/deployments"),
  mlActiveDeployments: () => request<MLDeploymentSummary[]>("/api/ml/deployments/active"),
  createMlDeployment: (payload: {
    id: string;
    model_id: string;
    target_pipeline_id: string;
    target_stage_id: string;
    pipeline_family?: string;
    deployed_by?: string;
    notes?: string | null;
  }) => post<MLDeploymentSummary>("/api/ml/deployments", payload),
  activateMlDeployment: (deploymentId: string) =>
    post<MLDeploymentSummary>(`/api/ml/deployments/${encodeURIComponent(deploymentId)}/activate`, {}),
  deactivateMlDeployment: (deploymentId: string) =>
    post<MLDeploymentSummary>(`/api/ml/deployments/${encodeURIComponent(deploymentId)}/deactivate`, {}),
  rollbackMlDeployment: (deploymentId: string) =>
    post<MLDeploymentSummary>(`/api/ml/deployments/${encodeURIComponent(deploymentId)}/rollback`, {}),
  runMlSmokeTest: (payload?: { force_invalid_model?: boolean; ci_mode?: boolean }) =>
    post<{ ok: boolean; returncode: number; stdout: string; stderr: string; result?: MLSmokeSummary | null }>("/api/ml/smoke-test/run", payload ?? {}),
  mlSmokeLatest: () => request<MLSmokeSummary>("/api/ml/smoke-test/latest"),
  mlSmokeHistory: () => request<{ items: MLSmokeSummary[] }>("/api/ml/smoke-test/history"),
  mlRuntimeHealth: () => request<{
    runtime_stats: Record<string, unknown>;
    totals: {
      inference_count: number;
      fallback_rate: number;
      heuristic_usage_rate: number;
      incompatible_deployment_count: number;
      average_inference_time_ms: number;
    };
    latest_smoke_test: MLSmokeSummary | null;
  }>("/api/ml/runtime-health"),
  mlRuntimeHealthHistory: (days = 14) => request<{ items: Array<{ day: string; stats: Record<string, unknown> }> }>(`/api/ml/runtime-health/history?days=${encodeURIComponent(String(days))}`),
  mlDatasetInsights: (datasetId: string) => request<MLDatasetInsights>(`/api/ml/datasets/${encodeURIComponent(datasetId)}/insights`),
  mlLabelsPreview: (datasetId: string, limit = 200) => request<{ dataset_id: string; count: number; items: Array<Record<string, unknown>> }>(`/api/ml/datasets/${encodeURIComponent(datasetId)}/labels-preview?limit=${encodeURIComponent(String(limit))}`),
  mlFeaturesPreview: (datasetId: string, limit = 200) => request<{
    dataset_id: string;
    count: number;
    schema: Record<string, unknown>;
    diagnostics: Record<string, unknown>;
    rows: Array<Record<string, unknown>>;
  }>(`/api/ml/datasets/${encodeURIComponent(datasetId)}/features-preview?limit=${encodeURIComponent(String(limit))}`),
  mlSplitPreview: (datasetId: string, strategy = "by_session", seed = 42) =>
    request<{ dataset_id: string; strategy: string; seed: number; split_manifest: Record<string, unknown>; leakage_report: Record<string, unknown>; confidence_level: string }>(
      `/api/ml/datasets/${encodeURIComponent(datasetId)}/split-preview?strategy=${encodeURIComponent(strategy)}&seed=${encodeURIComponent(String(seed))}`
    ),
  mlExperimentLogs: (experimentId: string) => request<{ experiment_id: string; logs: Array<{ file: string; line_count: number; tail: string[] }> }>(`/api/ml/experiments/${encodeURIComponent(experimentId)}/logs`),
  mlEvaluationPreview: (experimentId: string) => request<{
    experiment_id: string;
    artifacts: Record<string, string | null>;
    summary: Record<string, unknown>;
    failed_examples: Array<Record<string, unknown>>;
    confusion_matrix_csv: string;
    per_class_metrics_csv: string;
  }>(`/api/ml/experiments/${encodeURIComponent(experimentId)}/evaluation-preview`),
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
  datasetSessionSummary: (sessionId: string, datasetId?: string) =>
    request<DatasetSessionDetailResponse>(`/api/dataset-sessions/${encodeURIComponent(sessionId)}/summary${datasetId ? `?dataset_id=${encodeURIComponent(datasetId)}` : ""}`),
  createMlIngestionRun: (payload: { dataset_id: string; session_ids: string[]; name?: string }) =>
    post<IngestionRun>("/api/ml-ingestion/runs", payload),
  getMlIngestionRun: (runId: string) => request<IngestionRun>(`/api/ml-ingestion/runs/${encodeURIComponent(runId)}`),
  ingestMlOperatorTable: (runId: string, payload: { content: string; input_format: string }) =>
    post<Record<string, unknown>>(`/api/ml-ingestion/runs/${encodeURIComponent(runId)}/table`, payload),
  reconcileMlIngestionRun: (runId: string) =>
    post<Record<string, unknown>>(`/api/ml-ingestion/runs/${encodeURIComponent(runId)}/reconcile`, {}),
  updateMlIngestionPolicy: (runId: string, policy: Record<string, unknown>) =>
    post<Record<string, unknown>>(`/api/ml-ingestion/runs/${encodeURIComponent(runId)}/policy`, { policy }),
  generateMlIngestionManifest: (runId: string) =>
    post<Record<string, unknown>>(`/api/ml-ingestion/runs/${encodeURIComponent(runId)}/manifest`, {}),
  materializeMlIngestionRun: (runId: string, mlSetId: string) =>
    post<Record<string, unknown>>(`/api/ml-ingestion/runs/${encodeURIComponent(runId)}/materialize`, { ml_set_id: mlSetId }),
  applyMlIngestionMetadata: (runId: string) =>
    post<Record<string, unknown>>(`/api/ml-ingestion/runs/${encodeURIComponent(runId)}/apply-metadata`, {}),
  take: (takeId: string) => request<TakeDetail>(`/api/takes/${encodeURIComponent(takeId)}`),
  updateTakeMetadata: (takeId: string, payload: Record<string, unknown>) =>
    put<Record<string, unknown>>(`/api/takes/${encodeURIComponent(takeId)}/metadata`, payload),
  bulkUpdateTakeMetadata: (payload: TakeBulkMetadataRequest) =>
    post<TakeBulkMetadataResponse>("/api/takes/bulk-metadata", payload),
  datasetMlSets: (datasetId: string) => request<DatasetMlSet[]>(`/api/datasets/${encodeURIComponent(datasetId)}/ml-sets`),
  createDatasetMlSet: (datasetId: string, payload: {
    id?: string;
    name: string;
    description?: string;
    notes?: string;
    split_strategy?: string;
    validation_requirements?: string[];
  }) => post<DatasetMlSet>(`/api/datasets/${encodeURIComponent(datasetId)}/ml-sets`, payload),
  updateDatasetMlSetMembers: (datasetId: string, mlSetId: string, payload: {
    mode: "ids" | "filter";
    action: "add" | "remove";
    take_ids?: string[];
    filters?: TakeBulkFilters;
    exclude_take_ids?: string[];
    dry_run?: boolean;
  }) => post<{ ok: boolean; matched_count: number; affected_count: number; skipped_count: number; failed_count: number; failed_ids: string[] }>(
    `/api/datasets/${encodeURIComponent(datasetId)}/ml-sets/${encodeURIComponent(mlSetId)}/members`,
    payload
  ),
  removeTakeFromDataset: (takeId: string) => post<Record<string, unknown>>(`/api/takes/${encodeURIComponent(takeId)}/remove-from-dataset`, {}),
  archiveTake: (takeId: string, reason?: string | null) =>
    post<Record<string, unknown>>(`/api/takes/${encodeURIComponent(takeId)}/archive`, { reason: reason ?? null }),
  restoreTake: (takeId: string) => post<Record<string, unknown>>(`/api/takes/${encodeURIComponent(takeId)}/restore`, {}),
  permanentDeleteTake: (takeId: string, confirmation: string, deleteRuns = true) =>
    post<Record<string, unknown>>(`/api/takes/${encodeURIComponent(takeId)}/permanent-delete`, { confirmation, delete_runs: deleteRuns }),
  upsertObjectAnnotation: (takeId: string, payload: Record<string, unknown>) =>
    post<Record<string, unknown>>(`/api/takes/${encodeURIComponent(takeId)}/object-annotations`, payload),
  takeSourceHistogram: (takeId: string, filename?: string, maxDim = 512, bins = 32) =>
    request<SourceHistogramResponse>(
      `/api/takes/${encodeURIComponent(takeId)}/source-histogram?max_dim=${encodeURIComponent(String(maxDim))}&bins=${encodeURIComponent(String(bins))}${filename ? `&filename=${encodeURIComponent(filename)}` : ""}`
    ),
  processTake: (takeId: string, payload: { pipeline_id: string; run_until_stage?: string | null; reprocess?: boolean; stage_params?: Record<string, unknown> | null }) =>
    post<Record<string, unknown>>(`/api/takes/${encodeURIComponent(takeId)}/process`, payload),
  pipelineDefaults25d: () => request<{ defaults: Record<string, unknown> }>("/api/pipeline-defaults/25d"),
  savePipelineDefaults25d: (defaults: Record<string, unknown>) =>
    put<{ defaults: Record<string, unknown> }>("/api/pipeline-defaults/25d", defaults),
  clearPipelineDefaults25d: () => del<{ defaults: Record<string, unknown> }>("/api/pipeline-defaults/25d"),
  previewSegmentation: (payload: { take_id: string; pipeline_id: string; stage_id?: string; params: Record<string, unknown> }) =>
    post<SegmentationPreviewResponse>("/api/pipelines/preview-segmentation", payload),
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
  fusionInputs: (groupId: string) => request<FusionInputBundle>(`/api/acquisition-groups/${encodeURIComponent(groupId)}/fusion-inputs`),
  fusionPreview: (groupId: string) => request<FusionPreviewResult>(`/api/acquisition-groups/${encodeURIComponent(groupId)}/fusion-preview`),
  createFusionRun: (
    groupId: string,
    payload?: { force?: boolean; auto_publish?: boolean; recipe_version_id?: string | null; rules?: Record<string, unknown> }
  ) =>
    post<{ fusion_run_id: string; result: FusionRunResult; published_result_id?: string; published_result?: PublishedInspectionResult }>(
      `/api/acquisition-groups/${encodeURIComponent(groupId)}/fusion-runs`,
      payload ?? {}
    ),
  fusionRuns: (groupId: string) => request<{ acquisition_group_id: string; runs: FusionRunRecord[] }>(`/api/acquisition-groups/${encodeURIComponent(groupId)}/fusion-runs`),
  fusionRun: (fusionRunId: string) => request<{ run: FusionRunRecord; result: FusionRunResult | null }>(`/api/fusion-runs/${encodeURIComponent(fusionRunId)}`),
  latestFusionResult: (groupId: string) =>
    request<{ acquisition_group_id: string; status?: string; run: FusionRunRecord | null; result: FusionRunResult | null }>(
      `/api/acquisition-groups/${encodeURIComponent(groupId)}/fusion-result/latest`
    ),
  publish25dResult: (takeId: string) =>
    post<PublishedInspectionResult>(`/api/takes/${encodeURIComponent(takeId)}/publish-25d-result`, {}),
  latestPublishedInspectionResult: (stationId?: string) =>
    request<InspectionPublishedResult>(
      `/api/inspection-results/latest${stationId ? `?station_id=${encodeURIComponent(stationId)}` : ""}`
    ),
  inspectionResult: (resultId: string) => request<InspectionPublishedResult>(`/api/inspection-results/${encodeURIComponent(resultId)}`),
  inspectionResults: (params?: { station_id?: string; limit?: number }) => {
    const qs = new URLSearchParams();
    if (params?.station_id) qs.set("station_id", params.station_id);
    if (params?.limit !== undefined && params.limit !== null) qs.set("limit", String(params.limit));
    const suffix = qs.toString() ? `?${qs.toString()}` : "";
    return request<{ results: InspectionPublishedResult[]; station_id?: string | null; limit: number }>(`/api/inspection-results${suffix}`);
  },
  operatorLatestInspectionResult: (stationId?: string) =>
    request<PublishedInspectionResult>(
      `/api/operator/inspection-results/latest${stationId ? `?station_id=${encodeURIComponent(stationId)}` : ""}`
    ),
  operatorInspectionResult: (publishedResultId: string) =>
    request<PublishedInspectionResult>(`/api/operator/inspection-results/${encodeURIComponent(publishedResultId)}`),
  operatorInspectionResults: (params?: { session_id?: string; station_id?: string; limit?: number }) => {
    const qs = new URLSearchParams();
    if (params?.session_id) qs.set("session_id", params.session_id);
    if (params?.station_id) qs.set("station_id", params.station_id);
    if (params?.limit !== undefined && params.limit !== null) qs.set("limit", String(params.limit));
    const suffix = qs.toString() ? `?${qs.toString()}` : "";
    return request<{ results: PublishedInspectionResult[]; station_id?: string | null; session_id?: string | null; limit: number }>(
      `/api/operator/inspection-results${suffix}`
    );
  },
  operationsCards: (limit = 100) =>
    request<{ cards: OperationsCard[]; updated_at?: string | null; latest_take_id?: string | null; limit: number }>(
      `/api/operations/cards?limit=${encodeURIComponent(String(limit))}`
    ),
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
    capture_ids?: string[];
    capture_id?: string;
    detect_all?: boolean;
    target: { type: "charuco" | "checkerboard"; squares_x: number; squares_y: number; square_length_mm: number; marker_length_mm: number; dictionary: string };
  }) => post<{ success: boolean; detections: Array<Record<string, unknown>>; capture_count: number; detected_count: number }>("/api/calibration/camera-2d/detect-corners", payload),
  test2DCalibrationDictionaries: (payload: {
    capture_id: string;
    target: { type: "charuco" | "checkerboard"; squares_x: number; squares_y: number; square_length_mm: number; marker_length_mm: number; dictionary: string };
  }) => post<{ capture_id: string; results: Array<Record<string, unknown>>; best_candidate: Record<string, unknown> | null }>("/api/calibration/camera-2d/test-dictionaries", payload),
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
    camera_runtime_settings?: Record<string, number | boolean | null>;
  }) => post<{ calibration_id: string; path: string }>("/api/calibration/camera-2d/save", payload),
  generateCharucoTarget: (payload: {
    squares_x: number;
    squares_y: number;
    square_length_mm: number;
    marker_length_mm: number;
    dictionary: string;
  }) => post<{
    png_path: string;
    pdf_path: string;
    squares_x: number;
    squares_y: number;
    square_length_mm: number;
    marker_length_mm: number;
    dictionary_name: string;
    generated_file_path: string;
    printable_size_mm: [number, number] | number[];
  }>("/api/calibration/camera-2d/generate-charuco", payload),
};
