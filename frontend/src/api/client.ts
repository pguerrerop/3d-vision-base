import { featureAnalyticsQueryToSearchParams } from "../featureAnalyticsQuery";

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

export type ProcessingJobStatus = "queued" | "running" | "completed" | "failed" | "cancelled" | "partial";
export type ProcessingJobScopeType = "take_selection" | "dataset" | "dataset_session" | "acquisition_session" | "filter_query";
export type ProcessingReprocessMode = "full" | "failed_only" | "missing_outputs";

export interface ProcessingJobItem {
  take_id: string;
  status: "queued" | "running" | "completed" | "failed" | "skipped" | "cancelled";
  run_id?: string | null;
  started_at?: string | null;
  completed_at?: string | null;
  error?: string | null;
  skipped_reason?: string | null;
}

export interface ProcessingJobRecord {
  id: string;
  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
  status: ProcessingJobStatus;
  scope_type: ProcessingJobScopeType;
  pipeline_id: string;
  pipeline_family: string;
  total_takes: number;
  completed_takes: number;
  failed_takes: number;
  skipped_takes: number;
  created_by?: string | null;
  filters_summary?: Record<string, unknown>;
  options?: Record<string, unknown>;
  summary?: Record<string, unknown>;
  error_summary?: Array<Record<string, unknown>>;
  take_items?: ProcessingJobItem[];
  cancelled_at?: string | null;
}

export interface ProcessingJobCreateRequest {
  take_ids?: string[];
  dataset_id?: string | null;
  dataset_session_id?: string | null;
  acquisition_session_id?: string | null;
  filters_snapshot?: Record<string, unknown> | null;
  pipeline_id: string;
  reprocess_mode?: ProcessingReprocessMode;
  skip_completed?: boolean;
  force?: boolean;
  selected_stage?: string | null;
  classifier_rules_path?: string | null;
  created_by?: string | null;
  options?: Record<string, unknown>;
  take_selection?: {
    take_ids?: string[];
    dataset_id?: string | null;
    session_id?: string | null;
    filters?: Record<string, unknown> | null;
  } | null;
  scope_type?: ProcessingJobScopeType | null;
}

export type FeatureJobStatus = "queued" | "running" | "completed" | "failed" | "cancelled" | "partial";
export type FeatureJobScope = "ml_set" | "dataset" | "takes";
export type FeatureJobMode = "reprocess" | "backfill";

export interface FeatureJobRequest {
  scope: FeatureJobScope;
  dataset_id?: string;
  ml_set_id?: string;
  take_ids?: string[];
  feature_key: string;
  mode: FeatureJobMode;
  pipeline_id?: string;
  only_missing?: boolean;
}

export interface FeatureJobItem {
  take_id: string;
  status: "queued" | "running" | "completed" | "failed" | "skipped" | "cancelled";
  started_at?: string | null;
  completed_at?: string | null;
  error?: string | null;
  skipped_reason?: string | null;
  values_written?: number;
}

export interface FeatureJobRecord {
  id: string;
  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
  status: FeatureJobStatus;
  scope: FeatureJobScope;
  dataset_id?: string | null;
  ml_set_id?: string | null;
  feature_key: string;
  feature_display_name: string;
  mode: FeatureJobMode;
  pipeline_id?: string | null;
  only_missing: boolean;
  total_takes: number;
  completed_takes: number;
  failed_takes: number;
  skipped_takes: number;
  values_written: number;
  current_take_id?: string | null;
  summary?: Record<string, unknown>;
  error_summary?: Array<Record<string, unknown>>;
  take_items?: FeatureJobItem[];
  cancelled_at?: string | null;
}

export interface FeatureJobResponse {
  job_id: string;
  status: FeatureJobStatus;
  job: FeatureJobRecord;
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

export interface ProcessingUnitDefinition {
  id: string;
  label: string;
  kind: "stage" | "substage" | "utility" | string;
  parent_id?: string | null;
  stage_id: string;
  category: string;
  order: number;
  description: string;
  inputs: Array<{
    id: string;
    label: string;
    artifact_id?: string | null;
    kind?: string | null;
    coordinate_space?: string | null;
    units?: string | null;
    required?: boolean;
    description?: string | null;
  }>;
  outputs: Array<{
    id: string;
    label: string;
    artifact_id?: string | null;
    kind?: string | null;
    coordinate_space?: string | null;
    units?: string | null;
    description?: string | null;
  }>;
  artifacts: Array<{
    id: string;
    label: string;
    artifact_id: string;
    kind: string;
    role: string;
    renderer: string;
    description?: string | null;
    produced_by?: string | null;
    source_artifact_id?: string | null;
    coordinate_space?: string | null;
    aliases?: string[];
    feeds_into?: string[];
  }>;
  parameters: Array<{
    id: string;
    label: string;
    type: string;
    default?: unknown;
    min?: number | null;
    max?: number | null;
    options?: string[] | null;
    advanced?: boolean;
    affects?: string[];
    description?: string | null;
    tuning_hint?: string | null;
    active_when?: Record<string, unknown> | null;
    group?: string | null;
    unit?: string | null;
    step?: number | null;
    nullable?: boolean;
  }>;
  diagnostics: string[];
  views: Array<{
    id: string;
    label: string;
    renderer_type: string;
    artifact_ids: string[];
    description?: string | null;
    empty_state?: string | null;
  }>;
  default_view?: string | null;
  help_markdown?: string | null;
  enabled_by_default?: boolean;
  strategy_keys?: string[];
  supports_preview?: boolean;
  supports_partial_rerun?: boolean;
  controlled_by?: string[];
  parameter_groups?: Array<{
    id: string;
    label: string;
    param_keys?: string[];
    description?: string | null;
    affects?: string[];
  }>;
  downstream_effects?: string[];
  tuning_hints?: Array<{ condition: string; actions: string[] }>;
}

export interface PipelineRecipe {
  recipe_id: string;
  name: string;
  description?: string | null;
  pipeline_id: string;
  pipeline_family?: string | null;
  created_at: string;
  updated_at: string;
  version: number;
  parent_recipe_id?: string | null;
  tags: string[];
  notes?: string | null;
  processing_unit_contract_version?: string | null;
  processing_unit_contract_fingerprint?: string | null;
  parameters_by_unit?: Record<string, Record<string, unknown>>;
  stage_params?: Record<string, unknown>;
  unit_enabled_state?: Record<string, unknown>;
  artifact_policy?: Record<string, unknown>;
  calibration_snapshot_reference?: Record<string, unknown>;
  provenance?: Record<string, unknown>;
  archived?: boolean;
  compatibility?: "compatible" | "compatible_with_warnings" | "incompatible" | string;
  validation_warnings?: string[];
  validation_errors?: string[];
}

export interface PipelineComparisonSource {
  type: "run" | "recipe" | "current" | string;
  label?: string | null;
  take_id?: string | null;
  run_id?: string | null;
  recipe_id?: string | null;
  recipe_version?: number | null;
}

export interface PipelineComparisonRow {
  id?: string;
  parameter_id?: string;
  artifact_id?: string;
  label: string;
  left: unknown;
  right: unknown;
  status: "changed" | "unchanged" | "added" | "removed" | string;
}

export interface PipelineArtifactComparisonRow {
  artifact_id: string;
  label: string;
  kind?: string | null;
  left_present: boolean;
  right_present: boolean;
  left_path?: string | null;
  right_path?: string | null;
  left_url?: string | null;
  right_url?: string | null;
  diffable?: boolean;
  same_dimensions?: boolean | null;
  left_dimensions?: { width: number; height: number } | null;
  right_dimensions?: { width: number; height: number } | null;
  diff_available: boolean;
  diff_type?: "binary_mask" | "image_metadata" | "json_numeric" | string | null;
  diff_reason?: string | null;
  pixel_diff?: {
    width: number;
    height: number;
    total_pixels: number;
    changed_pixels: number;
    changed_percent: number;
    left_active_pixels: number;
    right_active_pixels: number;
    added_pixels: number;
    removed_pixels: number;
    intersection_pixels: number;
    union_pixels: number;
    iou: number;
    added_percent_of_union?: number;
    removed_percent_of_union?: number;
  } | null;
  diff_artifacts?: Record<string, { path: string; url: string }>;
  status: "changed" | "unchanged" | string;
}

export interface PipelineComparisonUnit {
  label: string;
  stage_id: string;
  kind: string;
  order: number;
  has_changes: boolean;
  parameter_diff: Array<{
    parameter_id: string;
    label: string;
    left: unknown;
    right: unknown;
    status: "changed" | "unchanged" | "added" | "removed" | string;
  }>;
  artifact_diff: PipelineArtifactComparisonRow[];
  metric_diff: PipelineComparisonRow[];
  diagnostic_diff: PipelineComparisonRow[];
}

export interface PipelineComparisonResponse {
  comparison_id: string;
  pipeline_id: string;
  contract_fingerprint?: string | null;
  left: PipelineComparisonSource;
  right: PipelineComparisonSource;
  summary: {
    parameter_changes: number;
    artifact_changes: number;
    metric_changes: number;
    classification_changed: boolean;
    warnings_changed: boolean;
    left_object_count?: number | null;
    right_object_count?: number | null;
    left_status?: string | null;
    right_status?: string | null;
    key_affected_units: string[];
    mask_artifacts_compared?: number;
    mask_artifacts_changed?: number;
    largest_changed_artifact?: string | null;
    largest_changed_unit?: string | null;
    average_iou?: number | null;
    object_count_changed?: boolean;
    what_changed_most?: PipelineComparisonChangeSummary | null;
  };
  units: Record<string, PipelineComparisonUnit>;
  warnings?: string[];
}

export interface PipelineBatchComparisonEntry {
  take_id: string;
  comparison_id?: string | null;
  summary: PipelineComparisonResponse["summary"];
}

export interface PipelineBatchComparisonError {
  take_id: string;
  error: string;
}

export interface PipelineBatchComparisonAggregate {
  compared_take_count: number;
  classification_changed_count: number;
  classification_changed_fraction?: number | null;
  object_count_changed_count: number;
  object_count_changed_fraction?: number | null;
  warnings_changed_count: number;
  warnings_changed_fraction?: number | null;
  takes_with_mask_changes_count: number;
  takes_with_mask_changes_fraction?: number | null;
  average_iou?: number | null;
  min_iou?: number | null;
  max_iou?: number | null;
  most_changed_units: Array<{ unit_id: string; changed_take_count: number; changed_take_fraction: number }>;
}

export interface PipelineBatchComparisonResponse {
  batch_comparison_id: string;
  pipeline_id: string;
  created_at: string;
  left: PipelineComparisonSource;
  right: PipelineComparisonSource;
  requested_take_count: number;
  compared_take_count: number;
  error_take_count: number;
  errors: PipelineBatchComparisonError[];
  aggregate: PipelineBatchComparisonAggregate;
  per_take: PipelineBatchComparisonEntry[];
}

export interface PartialRerunPlanResponse {
  pipeline_id: string;
  safe: boolean;
  mode: string;
  selected_unit_id: string;
  selected_stage_id?: string | null;
  effective_start_stage_id?: string | null;
  execution_boundary_unit_id?: string | null;
  selection_policy?: string;
  suggested_mode?: string;
  fallback_mode?: string | null;
  stages_to_reuse?: string[];
  stages_to_rerun?: string[];
  stages_invalidated?: string[];
  units_to_reuse: string[];
  units_to_rerun: string[];
  units_invalidated: string[];
  required_missing_artifacts: string[];
  reusable_artifacts?: string[];
  warnings: string[];
  blocking_reasons: string[];
  changed_unit_ids?: string[];
  changed_parameters?: Record<string, Record<string, unknown>>;
  contract_fingerprint_match?: boolean | null;
  calibration_match?: boolean | null;
  parent_trace_source?: string | null;
  explanation?: {
    summary?: string | null;
    selected_unit_reason?: string | null;
    boundary_reason?: string | null;
    reuse_reason?: string | null;
    rerun_reason?: string | null;
    fallback_reason?: string | null;
  };
  plan_quality?: string;
  confidence?: string;
  plan_id?: string | null;
  created_at?: string | null;
  plan_path?: string | null;
}

export interface PartialRerunPlanListEntry {
  plan_id?: string | null;
  created_at?: string | null;
  pipeline_id?: string | null;
  selected_unit_id?: string | null;
  execution_boundary_unit_id?: string | null;
  safe?: boolean;
  mode?: string | null;
  fallback_mode?: string | null;
  plan_quality?: string | null;
  confidence?: string | null;
  plan_path?: string | null;
}

export interface PartialRerunExecuteResponse {
  ok: boolean;
  safe: boolean;
  can_execute: boolean;
  pipeline_id: string;
  take_id?: string | null;
  child_run_id?: string | null;
  parent_run_id?: string | null;
  execution_mode?: string | null;
  boundary_unit_id?: string | null;
  boundary_stage_id?: string | null;
  reused_units?: string[];
  rerun_units?: string[];
  invalidated_units?: string[];
  comparison_id?: string | null;
  result_path?: string | null;
  warnings?: string[];
  blocking_reasons?: string[];
  fallback_mode?: string | null;
  partial_rerun_plan_id?: string | null;
  partial_rerun_plan?: PartialRerunPlanResponse | Record<string, unknown> | null;
  result?: Record<string, unknown> | null;
}

export interface PipelineRunEntry {
  run_id: string;
  run_type: "full_run" | "partial_rerun" | "legacy_3d_result";
  execution_mode?: string | null;
  pipeline_family?: string | null;
  pipeline_id?: string | null;
  parent_run_id?: string | null;
  parent_take_id?: string | null;
  boundary_stage_id?: string | null;
  boundary_unit_id?: string | null;
  selected_unit_id?: string | null;
  partial_rerun_plan_id?: string | null;
  comparison_id?: string | null;
  recipe_id?: string | null;
  recipe_version_id?: string | null;
  created_at?: string | null;
  status?: string | null;
  summary?: {
    reused_units?: number;
    rerun_units?: number;
    invalidated_units?: number;
    changed_parameters?: number;
    classification_label?: string | null;
    classification_superclass?: string | null;
    object_count?: number | null;
  } | null;
}

export interface PipelineRunLineage {
  pipeline_id: string;
  run_id: string;
  parent: PipelineRunEntry | null;
  children: PipelineRunEntry[];
  comparison_id: string | null;
}

export interface PipelineComparisonChangeSummary {
  top_changed_unit?: {
    unit_id?: string | null;
    label?: string | null;
    mask_changed_percent?: number | null;
    parameter_changes?: Array<{
      parameter_id?: string | null;
      label?: string | null;
      left?: unknown;
      right?: unknown;
    }>;
  } | null;
  top_changed_artifact?: {
    artifact_id?: string | null;
    label?: string | null;
    unit_id?: string | null;
    unit_label?: string | null;
    changed_percent?: number | null;
  } | null;
  classification_changed?: boolean;
  object_count_changed?: boolean;
  warnings_changed?: boolean;
}

export interface PipelineComparisonIndexEntry {
  comparison_id: string;
  created_at?: string | null;
  pipeline_id?: string | null;
  left?: PipelineComparisonSource;
  right?: PipelineComparisonSource;
  summary?: {
    parameter_changes?: number;
    artifact_changes?: number;
    metric_changes?: number;
    classification_changed?: boolean;
    object_count_changed?: boolean;
    warnings_changed?: boolean;
    what_changed_most?: PipelineComparisonChangeSummary | null;
  };
  comparison_json_path?: string | null;
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
  processing_units?: ProcessingUnitDefinition[];
  processing_unit_registry_version?: string;
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
  profile?: {
    enabled: boolean;
    limit: number;
    offset: number;
    counters: {
      scanned_take_ids: number;
      candidate_count: number;
      page_item_count: number;
      total_count: number;
    };
    phase_ms: {
      list_take_ids: number;
      scan_and_filter: number;
      sort_candidates: number;
      hydrate_page_items: number;
      total: number;
    };
  } | null;
}

export interface TakeBulkFilters {
  dataset_id?: string;
  ml_set_id?: string;
  session_id?: string;
  validation_status?: string;
  search?: string;
  tag?: string;
  created_from?: string;
  created_to?: string;
  split?: string;
  expected_class?: string;
  calibration_linkage_only?: boolean;
  physical_object_id?: string;
  raw_label?: string;
  normalized_class?: string;
  superclass?: string;
  membership_status?: string;
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
  membership_count?: number;
  membership_mode?: string;
  physical_object_count?: number;
  default_trainable_count?: number;
  review_required_count?: number;
  split_status?: string;
  source_type?: string | null;
  source_filename?: string | null;
  source_sha256?: string | null;
  normalization_version?: string | null;
}

export interface MaterializeMlIngestionResponse {
  ok: boolean;
  dataset_id: string;
  ml_set_id: string;
  ml_set_name: string;
  created_or_updated: string;
  ml_set_path: string;
  memberships_path: string;
  membership_count: number;
  physical_object_count: number;
  default_trainable_rows: number;
  default_trainable_objects: number;
  review_required_rows: number;
  review_required_objects: number;
  split_status: string;
  warnings: string[];
  errors: string[];
  output_dir: string;
  validation_report: Record<string, unknown>;
}

export interface MLSetSummaryResponse {
  ml_set: DatasetMlSet & Record<string, unknown>;
  dataset: DatasetSummary;
  identity: {
    take_count: number;
    physical_object_count: number;
    kept_in_ml_set_count?: number;
    default_trainable_count?: number;
    trainable_object_count: number;
    review_required_count: number;
    review_required_object_count?: number;
    excluded_count: number;
    class_count: number;
    split_strategy: string;
    label_schema_version: string;
    leakage_safe: boolean;
    validated_count: number;
    unvalidated_count: number;
    ingestion_run?: string;
  };
  readiness: {
    labeled_coverage_pct: number;
    split_completeness_pct: number;
    leakage_risk: string;
    class_imbalance_ratio: number;
    uncertain_label_count: number;
    take_count: number;
    physical_object_count: number;
    kept_in_ml_set_count?: number;
    default_trainable_count?: number;
    trainable_object_count: number;
    review_required_count: number;
    review_required_object_count?: number;
    excluded_count: number;
    empty_scene_count: number;
    calibration_reference_count: number;
    processing_completeness: string;
    feature_coverage: string;
    split_health: string;
    split_status: string;
  };
  raw_label_distribution: Record<string, number>;
  class_distribution: Record<string, number>;
  superclass_distribution: Record<string, number>;
  split_by_class: Record<string, Record<string, number>>;
  representative_samples: Record<string, Array<{ take_id: string; thumbnail_path?: string | null; validation_status?: string; expected_class?: string; session_id?: string }>>;
  split_summary: {
    split_counts: Record<string, number>;
    session_composition_by_split: Record<string, Record<string, number>>;
    leaked_object_ids: string[];
    warnings: Array<Record<string, unknown>>;
  };
  source_session_coverage: Array<{ session_id: string; included_takes: number; class_composition: Record<string, number>; uncertain_labels: number }>;
  membership_definition: {
    membership_mode: string;
    filter_snapshot: Record<string, unknown>;
    exclude_take_ids: string[];
    semantics: Record<string, unknown>;
  };
  warnings: Array<{ severity: string; code: string; explanation: string; affected_count: number; affected_ids?: string[] }>;
  derived_tasks: Array<{ task_id: string; included_classes: string[]; excluded_classes: string[]; effective_sample_count: number; uncertainty_policy: string; empty_scene_policy: string }>;
  training_compatibility: {
    feature_families: Record<string, string>;
    classifier_ready: boolean;
    processing_complete_count: number;
  };
  exports: Record<string, string>;
}

export interface MLSetMemberRow {
  dataset_id: string;
  ml_set_id: string;
  take_id: string;
  take_name?: string | null;
  created_at?: string | null;
  thumbnail_path?: string | null;
  physical_object_id?: string | null;
  raw_label?: string | null;
  normalized_class?: string | null;
  superclass?: string | null;
  review_required: boolean;
  default_trainable: boolean;
  trainable: boolean;
  include: boolean;
  split: string;
  validation_status?: string | null;
  session_id?: string | null;
  session_name?: string | null;
  processing_ready: boolean;
  feature_ready: boolean;
  processed_class_label?: string | null;
  processed_superclass?: string | null;
  warnings: string[];
  warning_count: number;
  label_provenance: {
    raw_label?: string | null;
    expected_label?: string | null;
    normalized_class?: string | null;
    superclass?: string | null;
    normalization_version?: string | null;
    source_row?: string | null;
    label_policy?: string | null;
  };
  feature_summary: {
    processed_class_label?: string | null;
    processed_superclass?: string | null;
    processing_by_family?: TakeSummary["processing_by_family"];
  };
}

export interface MLSetMembersResponse {
  ml_set: DatasetMlSet & Record<string, unknown>;
  dataset: DatasetSummary;
  items: MLSetMemberRow[];
  filtered_count: number;
  total_count: number;
  limit: number;
  offset: number;
  has_more: boolean;
  next_offset?: number | null;
  stats: {
    member_count: number;
    object_count: number;
    review_required_count: number;
  };
}

export interface MLSetReviewFacetOption {
  value: string;
  count: number;
}

export interface MLSetReviewObjectFacetOption {
  value: string;
  take_count: number;
  normalized_class?: string | null;
  superclass?: string | null;
}

export interface MLSetReviewSessionFacetOption {
  value: string;
  name?: string | null;
  count: number;
}

export interface MLSetReviewFacetsResponse {
  ml_set: DatasetMlSet & Record<string, unknown>;
  dataset: DatasetSummary;
  stats: {
    member_count: number;
    object_count: number;
  };
  object_ids: MLSetReviewObjectFacetOption[];
  raw_labels: MLSetReviewFacetOption[];
  normalized_classes: MLSetReviewFacetOption[];
  superclasses: MLSetReviewFacetOption[];
  splits: MLSetReviewFacetOption[];
  membership_statuses: MLSetReviewFacetOption[];
  sessions: MLSetReviewSessionFacetOption[];
}

export interface PhysicalObjectSummary {
  physical_object_id: string;
  dataset_id: string;
  source_session_ids: string[];
  raw_operator_label?: string | null;
  normalized_class?: string | null;
  superclass?: string | null;
  d1_mm?: number | string | null;
  d2_mm?: number | string | null;
  d3_mm?: number | string | null;
  diameter_mean_mm?: number | null;
  diameter_range_mm?: number | null;
  annotation_confidence?: string | null;
  needs_review?: boolean;
  tags?: string[];
  notes?: string | null;
  source_type?: string | null;
  source_row_index?: number | null;
  observation_take_ids?: string[];
  created_at?: string | null;
  updated_at?: string | null;
}

export interface LabelTaxonomyEntry {
  normalized_class: string;
  superclass: string;
  is_uncertain: boolean;
  notes?: string | null;
  updated_at: string;
  updated_by?: string | null;
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

export interface PartialExecutionSummary {
  reused_artifact_count: number;
  reused_artifact_bytes: number;
  rerun_artifact_count: number;
  partial_execution_duration_ms: number;
  boundary_stage_id: string | null;
  parent_run_id: string | null;
}

export interface ProcessingResult {
  take_id: string;
  run_id?: string | null;
  parent_run_id?: string | null;
  execution_mode?: string | null;
  partial_execution_summary?: PartialExecutionSummary | null;
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
    processing_units?: ProcessingUnitDefinition[];
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
    label?: string | null;
    display_label?: string | null;
    superclass?: string | null;
    class_group?: string | null;
    footprint_geometry?: Record<string, unknown> | null;
    surface_geometry?: Record<string, unknown> | null;
    sphere_consistency?: Record<string, unknown> | null;
    damage_metrics?: Record<string, unknown> | null;
    feature_group_summaries?: Array<{
      group: string;
      status: string;
      severity_score: number;
      readiness: string;
      top_warnings?: Array<Record<string, unknown>>;
      key_evidence_features?: Array<Record<string, unknown>>;
    }>;
    feature_warnings?: Array<{
      feature: string;
      severity: string;
      message: string;
      group: string;
      ux_visibility?: string[];
    }>;
    feature_readiness?: {
      overall_readiness?: string;
      group_readiness?: Record<string, string>;
    } | null;
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
    measurement_diagnostics?: string | null;
    feature_vector?: string | null;
    feature_provenance?: string | null;
    quality_flags?: string | null;
    feature_runtime_summary?: string | null;
    feature_studio_summary?: string | null;
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
  stage_params?: Record<string, unknown>;
  recipe_snapshot?: Record<string, unknown> | null;
  processing_unit_trace?: {
    pipeline_id?: string | null;
    registry_version?: string | null;
    trace_source?: string | null;
    trace_precision?: string | null;
    trace_summary?: {
      total_units?: number;
      runtime_traced_units?: number;
      inferred_units?: number;
      failed_units?: number;
      warning_units?: number;
      trace_coverage_percent?: number;
      coverage_by_stage?: Record<string, {
        total_units?: number;
        runtime_traced_units?: number;
        inferred_units?: number;
        failed_units?: number;
        warning_units?: number;
        trace_coverage_percent?: number;
      }> | null;
    } | null;
    units?: Record<string, {
      unit_id?: string;
      stage_id?: string;
      parent_id?: string | null;
      status: string;
      started_at?: string | null;
      completed_at?: string | null;
      duration_ms?: number | null;
      parameters_used?: Record<string, unknown>;
      input_artifacts?: string[];
      output_artifacts?: string[];
      metrics?: Record<string, unknown>;
      diagnostics?: Record<string, unknown>;
      warnings?: string[];
      errors?: string[];
      trace_source?: string | null;
      trace_precision?: string | null;
    }>;
    unit_results?: Record<string, {
      status: string;
      unit_id?: string;
      stage_id?: string;
      parent_id?: string | null;
      started_at?: string | null;
      completed_at?: string | null;
      duration_ms?: number | null;
      parameters_used?: Record<string, unknown>;
      input_artifacts?: string[];
      output_artifacts?: string[];
      metrics?: Record<string, unknown>;
      diagnostics?: Record<string, unknown>;
      warnings?: string[];
      errors?: string[];
      trace_source?: string | null;
      trace_precision?: string | null;
    }>;
  } | null;
  session_id?: string | null;
  frameset_id?: string | null;
  runtime_acquisition?: Record<string, unknown> | null;
  throughput?: Record<string, unknown> | null;
  synchronization?: Record<string, unknown> | null;
  acquisition_timestamps?: Record<string, string> | null;
  feature_metadata?: Record<string, unknown> | null;
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

export interface DatasetLabelSummary {
  dataset_id: string;
  take_count: number;
  raw_tag_counts: Record<string, number>;
  semantic_label_counts: Record<string, number>;
  superclass_counts: Record<string, number>;
  unmapped_tags: Record<string, number>;
  class_counts: Record<string, number>;
  normalization_version: string | null;
  normalization_versions_seen: Record<string, number>;
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
  validated_take_count?: number;
}

export interface FeatureDefinition {
  key?: string;
  feature_key: string;
  display_name: string;
  semantic_group: string;
  family?: string;
  family_label?: string;
  unit?: string | null;
  description?: string | null;
  source_stage?: string | null;
  studio_stage?: string | null;
  source_metric_path?: string | null;
  formula?: string | null;
  algorithm_summary?: string | null;
  inputs?: string[];
  interpretation?: string | null;
  caveats?: string[];
  stable_schema?: boolean;
  diagnostic_only?: boolean;
  experimental?: boolean;
  higher_is_worse?: boolean | null;
  expected_range?: number[] | null;
  display_precision?: number | null;
  preferred_chart_scale?: "linear" | "log" | string | null;
  legacy_aliases?: string[];
}

export interface FeatureAnalyticsFeaturesResponse {
  record_count: number;
  object_count: number;
  scope_summary: {
    record_count: number;
    take_count: number;
    physical_object_count: number;
    object_count: number;
  };
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
  physical_object_id?: string | null;
  pipeline_id?: string | null;
  run_id?: string | null;
  stage_id?: string | null;
  labeled_superclass?: string | null;
  labeled_superclass_source?: string | null;
  labeled_superclass_unresolved?: boolean | null;
  normalized_class?: string | null;
  processed_superclass?: string | null;
  processed_class?: string | null;
  processed_label?: string | null;
  superclass?: string | null;
  raw_labels?: string[];
  labels?: string[];
  validation_status?: string | null;
  split?: string | null;
  feature_value: number;
  feature_source?: string | null;
  confidence?: number | null;
  timestamp?: string | null;
  annotation?: Record<string, unknown> | null;
}

export interface FeatureAnalyticsThumbnailInfo {
  requested_mode: "auto" | "heightmap" | "reflectance" | "classification_overlay";
  resolved_source:
    | "heightmap"
    | "normalized_heightmap"
    | "reflectance"
    | "source_crop"
    | "classification_overlay"
    | "take_thumbnail"
    | "placeholder";
  resolved_artifact_id?: string | null;
  resolved_artifact_path?: string | null;
  colorized: boolean;
  fallback_used: boolean;
  fallback_reason?: string | null;
}

export interface FeatureAnalyticsObjectsResponse {
  feature_key: string;
  min_value?: number | null;
  max_value?: number | null;
  limit: number;
  objects: FeatureAnalyticsObject[];
  scope_summary: FeatureAnalyticsFeaturesResponse["scope_summary"];
  stats: FeatureAnalyticsDistributionResponse["stats"];
}

export interface FeatureAnalyticsQuery {
  dataset_id?: string;
  ml_set_id?: string;
  session_id?: string;
  physical_object_ids?: string[];
  take_ids?: string[];
  labeled_superclasses?: string[];
  processed_superclasses?: string[];
  raw_labels?: string[];
  normalized_classes?: string[];
  processed_classes?: string[];
  superclasses?: string[];
  validation_status?: string[];
  split?: string[];
  pipeline_id?: string;
  calibration_id?: string;
  date_from?: string;
  date_to?: string;
}

export interface FeatureAnalyticsFilterOptionsResponse {
  raw_labels: string[];
  normalized_classes: string[];
  labeled_superclasses: string[];
  processed_superclasses: string[];
  processed_classes: string[];
  superclasses: string[];
  physical_object_ids: string[];
  take_ids: string[];
  validation_statuses: string[];
  splits: string[];
  pipeline_ids: string[];
  calibration_ids: string[];
  stats: FeatureAnalyticsFeaturesResponse["scope_summary"];
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

/** A path from diff_artifacts or a baseline's snapshot_artifact_path, servable directly. */
export function validationFileUrl(path: string): string {
  const encoded = path.split("/").map((segment) => encodeURIComponent(segment)).join("/");
  return `${API_BASE_URL}/api/validation/files/${encoded}`;
}

export type ValidationComparator =
  | "binary_mask" | "numeric_raster" | "measurement_table" | "classification"
  | "json_fields" | "plane_model" | "visual_only" | "not_comparable";
export type ValidationComparatorSource =
  | "declared_comparator" | "semantic_type" | "role" | "kind" | "path" | "heuristic" | "unresolved";
export interface ValidationBaselinePromotion {
  source_execution_id: string; source_case_id: string; source_comparison_id: string;
  candidate_run_id: string; candidate_artifact_checksum: string;
  promoted_at: string; promoted_by?: string | null; notes?: string;
}
export interface ValidationBaseline {
  id: string; version: number; status: "active" | "inactive" | string;
  take_id: string; pipeline_id: string; source_run_id: string; stage_id?: string | null;
  processing_unit_id?: string | null; substage_id?: string | null; view_id?: string | null; artifact_id?: string | null;
  artifact_kind?: string | null; artifact_semantic_type?: string | null; artifact_role?: string | null;
  comparator?: ValidationComparator | string; comparator_source?: ValidationComparatorSource | string;
  artifact_checksum?: string; recipe_fingerprint?: string | null; recipe_id?: string | null;
  pipeline_contract_fingerprint?: string | null; comparison_policy?: Record<string, unknown>;
  calibration_context?: Record<string, unknown>;
  created_at?: string; supersedes_baseline_id?: string | null;
  promotion?: ValidationBaselinePromotion | null;
  /** Recomputed on read by the history endpoint, not stored on the record. */
  integrity_state?: "ok" | "invalid" | string;
  review?: { status?: string; notes?: string; reviewed_at?: string; reviewed_by?: string | null };
}
export interface ValidationCaseRequest {
  take_id?: string; baseline_ids: string[]; baseline_resolution?: ValidationBaselineResolution;
  enabled?: boolean; tags?: string[]; notes?: string; included_artifacts?: string[];
}
/** Every field is optional: omitted ones are preserved by the server. */
export type ValidationCaseUpdate = Partial<
  Pick<ValidationSuiteCase, "enabled" | "tags" | "notes" | "included_artifacts" | "baseline_resolution">
>;
export type ValidationResolutionMode = "active" | "pinned" | "allowed_versions";
export interface ValidationBaselineResolution { mode: ValidationResolutionMode; baseline_id?: string | null; allowed_baseline_ids?: string[]; }
export interface ValidationSuiteCase { id: string; take_id: string; baseline_ids: string[]; baseline_resolution?: ValidationBaselineResolution; enabled: boolean; tags: string[]; notes: string; included_artifacts?: string[]; }
export interface ValidationSuiteDetail { id: string; name: string; description: string; pipeline_id: string; status: string; cases: ValidationSuiteCase[]; }
export interface ValidationCoverageArea { area: string; approved: number; missing: number; stale: number; unsupported: number; visual_only: number; coverage_fraction: number; }
export interface ValidationCaseRef {
  suite_id: string;
  case_id: string;
}
/** Which cases follow an activation and which are deliberately unaffected. */
export interface ValidationResolutionImpact {
  active_cases: ValidationCaseRef[];
  pinned_cases_unchanged: ValidationCaseRef[];
  allowed_version_cases_unchanged: ValidationCaseRef[];
  disabled_cases: ValidationCaseRef[];
  archived_suites: ValidationCaseRef[];
}
export type ValidationComparisonStatus = "pass" | "changed" | "regression" | "not_comparable" | "blocked" | "needs_review";
export interface ValidationMatch { baseline_index: number; candidate_index: number; baseline_object_id: string | null; candidate_object_id: string | null; match_method: "stable_id" | "centroid" | string; match_score: number; ambiguous: boolean; }
export interface ValidationUnmatchedObject { index: number; object_id: string | null; }
export interface ValidationAmbiguousMatch { baseline_object_id: string | null; candidate_indices: number[]; }
export interface ValidationMatching { matches: ValidationMatch[]; baseline_only: ValidationUnmatchedObject[]; candidate_only: ValidationUnmatchedObject[]; ambiguous: ValidationAmbiguousMatch[]; }
export interface ValidationMetricResult { metric: string; units?: string | null; baseline: unknown; candidate: unknown; delta?: number; relative_delta?: number; absolute_tolerance?: number; relative_tolerance?: number; effective_tolerance?: number; status: "pass" | "regression" | "changed" | "not_comparable"; reason: string | null; }
export interface ValidationMeasurementObjectResult extends ValidationMatch { status: "pass" | "regression"; metric_results: ValidationMetricResult[]; }
export interface ValidationClassificationSide { superclass: string | null; label: string | null; confidence: number | null; }
export interface ValidationClassificationObjectResult extends ValidationMatch { baseline: ValidationClassificationSide; candidate: ValidationClassificationSide; confidence_delta: number; status: "pass" | "regression"; reasons: string[]; }
export interface ValidationComparisonResult {
  status: ValidationComparisonStatus; comparator: string; summary: string;
  /** Field set depends on the comparator; look up the ones that comparator defines. */
  metrics: Record<string, number>; thresholds: Record<string, unknown>; reasons: string[];
  /** Paths under the validation root, servable via validationFileUrl(). */
  diff_artifacts: string[];
  baseline_ref: { baseline_id: string | null; version: number | null }; candidate_ref: { artifact_id: string | null };
  /** Measurement and classification comparators only. */
  object_results?: Array<ValidationMeasurementObjectResult | ValidationClassificationObjectResult>;
  matching?: ValidationMatching;
}
export interface ValidationExecutionCase {
  case_id: string; take_id: string; status: string; skipped?: boolean; summary?: string;
  comparisons?: ValidationComparisonResult[]; matched_baseline_id?: string | null; first_failing_stage?: string | null;
  first_divergence?: { take_id: string; stage_id?: string | null; processing_unit_id?: string | null; substage_id?: string | null; view_id?: string | null; artifact_id?: string | null; status: string; summary: string } | null;
}
export interface ValidationNumericFieldSummary { count: number; mean: number; median: number; p95: number; min: number; max: number; }
export interface ValidationStageComparatorSummary { comparator: string; count: number; status_counts: Record<string, number>; metrics: Record<string, ValidationNumericFieldSummary>; }
export interface ValidationStageSummaryEntry { stage_id: string; label: string; comparators: ValidationStageComparatorSummary[]; }
export interface ValidationStageSummary { execution_id: string; stages: ValidationStageSummaryEntry[]; }

export interface ValidationExecution {
  id: string; suite_id: string; pipeline_id: string; status: string; candidate_run_id: string;
  created_at?: string; started_at?: string; completed_at?: string; cases: ValidationExecutionCase[];
}
export interface ValidationPromotionResponse { baseline: ValidationBaseline; already_promoted: boolean; previous_active_baseline: ValidationBaseline | null; resulting_active_baseline: ValidationBaseline | null; affected_case_resolution: ValidationResolutionImpact; }

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

async function postForm<T>(path: string, body: FormData): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    body,
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

async function patch<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "PATCH",
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

export function objectThumbnailUrl(
  takeId: string,
  objectId: string | number,
  size = 56,
  mode: "auto" | "heightmap" | "reflectance" | "classification_overlay" = "auto",
  semanticGroup?: string | null,
): string {
  const params = new URLSearchParams({
    size: String(size),
    mode,
  });
  if (semanticGroup) params.set("semantic_group", semanticGroup);
  return `${API_BASE_URL}/api/takes/${encodeURIComponent(takeId)}/objects/${encodeURIComponent(String(objectId))}/thumbnail?${params.toString()}`;
}

export const api = {
  validationBaselines: (takeId?: string, pipelineId?: string) => {
    const query = new URLSearchParams(); if (takeId) query.set("take_id", takeId); if (pipelineId) query.set("pipeline_id", pipelineId);
    return request<ValidationBaseline[]>(`/api/validation/baselines${query.size ? `?${query}` : ""}`);
  },
  approveValidationBaseline: (payload: Record<string, unknown>) => post<{ baselines: ValidationBaseline[]; included: string[]; skipped: Array<Record<string, unknown>> }>("/api/validation/baselines", payload),
  compareValidationBaseline: (baselineId: string, candidateRunId: string) => post<ValidationComparisonResult>(`/api/validation/baselines/${encodeURIComponent(baselineId)}/compare?candidate_run_id=${encodeURIComponent(candidateRunId)}`, {}),
  validationBaselineImpact: (baselineId: string) => request<ValidationResolutionImpact>(`/api/validation/baselines/${encodeURIComponent(baselineId)}/impact`),
  activateValidationBaseline: (baselineId: string) => post<ValidationBaseline>(`/api/validation/baselines/${encodeURIComponent(baselineId)}/activate`, {}),
  deactivateValidationBaseline: (baselineId: string) => post<ValidationBaseline>(`/api/validation/baselines/${encodeURIComponent(baselineId)}/deactivate`, {}),
  validationSuites: () => request<Array<{ id: string; name: string; pipeline_id: string; status: string }>>("/api/validation/suites"),
  validationSuite: (suiteId: string) => request<ValidationSuiteDetail>(`/api/validation/suites/${encodeURIComponent(suiteId)}`),
  createValidationSuite: (payload: { name: string; pipeline_id: string; description?: string }) => post<ValidationSuiteDetail>("/api/validation/suites", payload),
  updateValidationSuite: (suiteId: string, payload: Partial<Pick<ValidationSuiteDetail, "name" | "description" | "status">>) => put<ValidationSuiteDetail>(`/api/validation/suites/${encodeURIComponent(suiteId)}`, payload),
  duplicateValidationSuite: (suiteId: string) => post<ValidationSuiteDetail>(`/api/validation/suites/${encodeURIComponent(suiteId)}/duplicate`, {}),
  archiveValidationSuite: (suiteId: string) => post<ValidationSuiteDetail>(`/api/validation/suites/${encodeURIComponent(suiteId)}/archive`, {}),
  restoreValidationSuite: (suiteId: string) => post<ValidationSuiteDetail>(`/api/validation/suites/${encodeURIComponent(suiteId)}/restore`, {}),
  validationCoverage: (suiteId: string) => request<{ suite_id: string; case_count: number; areas: ValidationCoverageArea[] }>(`/api/validation/suites/${encodeURIComponent(suiteId)}/coverage`),
  validationSuiteExecutions: (suiteId: string) => request<Array<Record<string, unknown>>>(`/api/validation/suites/${encodeURIComponent(suiteId)}/executions`),
  addValidationCase: (suiteId: string, payload: ValidationCaseRequest) => post<ValidationSuiteCase>(`/api/validation/suites/${encodeURIComponent(suiteId)}/cases`, payload),
  updateValidationCase: (suiteId: string, caseId: string, payload: ValidationCaseUpdate) => patch<ValidationSuiteCase>(`/api/validation/suites/${encodeURIComponent(suiteId)}/cases/${encodeURIComponent(caseId)}`, payload),
  deleteValidationCase: (suiteId: string, caseId: string) => del<{ deleted: string }>(`/api/validation/suites/${encodeURIComponent(suiteId)}/cases/${encodeURIComponent(caseId)}`),
  validationBaselineHistory: (params?: { take_id?: string; pipeline_id?: string; artifact_id?: string }) => {
    const query=new URLSearchParams(); Object.entries(params ?? {}).forEach(([key,value]) => { if (value) query.set(key,value); });
    return request<ValidationBaseline[]>(`/api/validation/baselines/history${query.size ? `?${query}` : ""}`);
  },
  promoteValidationComparison: (executionId: string, comparisonId: string, payload: { case_id: string; activate?: boolean; notes?: string; carry_forward_policy?: boolean; reviewed_by?: string }) => post<ValidationPromotionResponse>(`/api/validation/executions/${encodeURIComponent(executionId)}/comparisons/${encodeURIComponent(comparisonId)}/promote`, payload),
  validationExecutions: () => request<Array<Record<string, unknown>>>("/api/validation/executions"),
  validationExecution: (executionId: string) => request<ValidationExecution>(`/api/validation/executions/${encodeURIComponent(executionId)}`),
  validationBaseline: (baselineId: string) => request<ValidationBaseline>(`/api/validation/baselines/${encodeURIComponent(baselineId)}`),
  validationMatrix: (executionId: string) => request<{ execution_id: string; columns: Array<{ id:string; label:string; stage_id:string; order:number }>; rows: Array<{ case_id:string; take_id:string; overall_status:string; cells:Array<{ column_id:string; status:string; comparison_ids:Array<string | null>; first_divergence:boolean; artifact_id?:string | null }> }> }>(`/api/validation/executions/${encodeURIComponent(executionId)}/matrix`),
  validationStageSummary: (executionId: string) => request<ValidationStageSummary>(`/api/validation/executions/${encodeURIComponent(executionId)}/stage-summary`),
  runValidationSuite: (suiteId: string, candidateRunId = "latest") => post<ValidationExecution>(`/api/validation/suites/${encodeURIComponent(suiteId)}/execute`, { candidate_run_id: candidateRunId }),
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
    profile?: boolean;
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
  datasetLabelSummary: (datasetId: string) => request<DatasetLabelSummary>(`/api/datasets/${encodeURIComponent(datasetId)}/label-summary`),
  physicalObjects: (params: { dataset_id: string; normalized_class?: string; superclass?: string; session_id?: string; needs_review?: boolean }) => {
    const qs = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== "") qs.set(key, String(value));
    });
    return request<PhysicalObjectSummary[]>(`/api/physical-objects?${qs.toString()}`);
  },
  physicalObject: (physicalObjectId: string, datasetId: string) =>
    request<PhysicalObjectSummary>(`/api/physical-objects/${encodeURIComponent(physicalObjectId)}?dataset_id=${encodeURIComponent(datasetId)}`),
  physicalObjectTakes: (physicalObjectId: string, datasetId: string) =>
    request<{ takes: Array<{ session_id: string; take_id: string; metadata: Record<string, unknown> }> }>(`/api/physical-objects/${encodeURIComponent(physicalObjectId)}/takes?dataset_id=${encodeURIComponent(datasetId)}`),
  physicalObjectRepeatability: (physicalObjectId: string, datasetId: string) =>
    request<Record<string, unknown>>(`/api/physical-objects/${encodeURIComponent(physicalObjectId)}/repeatability?dataset_id=${encodeURIComponent(datasetId)}`),
  correctPhysicalObjectLabel: (physicalObjectId: string, payload: { dataset_id: string; normalized_class: string; superclass?: string | null; raw_label?: string | null; actor?: string; reason?: string | null }) =>
    post<{ id: string; affected_take_ids: string[]; affected_ml_set_ids: string[] }>(`/api/physical-objects/${encodeURIComponent(physicalObjectId)}/label-corrections`, payload),
  labelTaxonomy: (query?: string) =>
    request<{ entries: LabelTaxonomyEntry[] }>(`/api/label-taxonomy${query ? `?query=${encodeURIComponent(query)}` : ""}`),
  upsertLabelTaxonomyEntry: (payload: { normalized_class: string; superclass: string; is_uncertain?: boolean; notes?: string | null; updated_by?: string }) =>
    post<LabelTaxonomyEntry>("/api/label-taxonomy", payload),
  featureAnalyticsFeatures: (params: FeatureAnalyticsQuery & { feature_selection?: string[] }) => {
    const qs = featureAnalyticsQueryToSearchParams(params);
    return request<FeatureAnalyticsFeaturesResponse>(`/api/feature-analytics/features${qs.toString() ? `?${qs.toString()}` : ""}`);
  },
  featureAnalyticsFilterOptions: (params: FeatureAnalyticsQuery) => {
    const qs = featureAnalyticsQueryToSearchParams(params);
    return request<FeatureAnalyticsFilterOptionsResponse>(`/api/feature-analytics/filter-options${qs.toString() ? `?${qs.toString()}` : ""}`);
  },
  featureAnalyticsDistributions: (params: {
    feature_key: string;
    group_by?:
      | "superclass"
      | "processed_superclass"
      | "labeled_superclass"
      | "label"
      | "raw_label"
      | "normalized_class"
      | "physical_object_id"
      | "processed_class";
    bins?: number;
    mode?: "count" | "density";
  } & FeatureAnalyticsQuery) => {
    const qs = featureAnalyticsQueryToSearchParams(params);
    qs.set("feature_key", params.feature_key);
    if (params.group_by) qs.set("group_by", params.group_by);
    if (params.bins !== undefined) qs.set("bins", String(params.bins));
    if (params.mode) qs.set("mode", params.mode);
    return request<FeatureAnalyticsDistributionResponse>(`/api/feature-analytics/distributions?${qs.toString()}`);
  },
  featureAnalyticsObjects: (params: {
    feature_key: string;
    min_value?: number;
    max_value?: number;
    limit?: number;
  } & FeatureAnalyticsQuery) => {
    const qs = featureAnalyticsQueryToSearchParams(params);
    qs.set("feature_key", params.feature_key);
    if (params.min_value !== undefined) qs.set("min_value", String(params.min_value));
    if (params.max_value !== undefined) qs.set("max_value", String(params.max_value));
    if (params.limit !== undefined) qs.set("limit", String(params.limit));
    return request<FeatureAnalyticsObjectsResponse>(`/api/feature-analytics/objects?${qs.toString()}`);
  },
  objectThumbnailInfo: (
    takeId: string,
    objectId: string | number,
    params: {
      mode?: "auto" | "heightmap" | "reflectance" | "classification_overlay";
      semantic_group?: string;
    } = {},
  ) => {
    const qs = new URLSearchParams();
    if (params.mode) qs.set("mode", params.mode);
    if (params.semantic_group) qs.set("semantic_group", params.semantic_group);
    return request<FeatureAnalyticsThumbnailInfo>(`/api/takes/${encodeURIComponent(takeId)}/objects/${encodeURIComponent(String(objectId))}/thumbnail-info${qs.toString() ? `?${qs.toString()}` : ""}`);
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
  startMlLiveTrial: (payload: { deployment_id: string; recipe_snapshot?: Record<string, unknown>; notes?: string }) => post<{ id: string }>("/api/ml/live-trials", payload),
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
  ingestMlOperatorFile: (runId: string, file: File) => {
    const form = new FormData();
    form.append("file", file);
    return postForm<Record<string, unknown>>(`/api/ml-ingestion/runs/${encodeURIComponent(runId)}/table`, form);
  },
  reconcileMlIngestionRun: (runId: string) =>
    post<Record<string, unknown>>(`/api/ml-ingestion/runs/${encodeURIComponent(runId)}/reconcile`, {}),
  updateMlIngestionPolicy: (runId: string, policy: Record<string, unknown>) =>
    post<Record<string, unknown>>(`/api/ml-ingestion/runs/${encodeURIComponent(runId)}/policy`, { policy }),
  generateMlIngestionManifest: (runId: string) =>
    post<Record<string, unknown>>(`/api/ml-ingestion/runs/${encodeURIComponent(runId)}/manifest`, {}),
  materializeMlIngestionRun: (runId: string, mlSetId: string) =>
    post<MaterializeMlIngestionResponse>(`/api/ml-ingestion/runs/${encodeURIComponent(runId)}/materialize`, { ml_set_id: mlSetId }),
  applyMlIngestionMetadata: (runId: string) =>
    post<Record<string, unknown>>(`/api/ml-ingestion/runs/${encodeURIComponent(runId)}/apply-metadata`, {}),
  take: (takeId: string, runId?: string | null) =>
    request<TakeDetail>(
      `/api/takes/${encodeURIComponent(takeId)}${runId ? `?run_id=${encodeURIComponent(runId)}` : ""}`,
    ),
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
  updateDatasetMlSetMemberships: (datasetId: string, mlSetId: string, payload: {
    take_ids: string[];
    split?: string;
    include?: boolean;
    expected_class?: string;
    expected_subclass?: string;
    review_required?: boolean;
    default_trainable?: boolean;
  }) => post<{ ok: boolean; updated_count: number; skipped_count: number; skipped_ids: string[] }>(
    `/api/datasets/${encodeURIComponent(datasetId)}/ml-sets/${encodeURIComponent(mlSetId)}/members/update`,
    payload
  ),
  mlSetSummary: (mlSetId: string, datasetId?: string) =>
    request<MLSetSummaryResponse>(`/api/ml-sets/${encodeURIComponent(mlSetId)}/summary${datasetId ? `?dataset_id=${encodeURIComponent(datasetId)}` : ""}`),
  mlSetMembers: (mlSetId: string, params?: {
    dataset_id?: string;
    physical_object_id?: string;
    raw_label?: string;
    normalized_class?: string;
    superclass?: string;
    membership_status?: string;
    split?: string;
    session_id?: string;
    search?: string;
    validation_status?: string;
    limit?: number;
    offset?: number;
  }) => {
    const qs = new URLSearchParams();
    Object.entries(params ?? {}).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== "") qs.set(key, String(value));
    });
    return request<MLSetMembersResponse>(`/api/ml-sets/${encodeURIComponent(mlSetId)}/members${qs.toString() ? `?${qs.toString()}` : ""}`);
  },
  mlSetReviewFacets: (datasetId: string, mlSetId: string, params?: {
    physical_object_id?: string;
    raw_label?: string;
    normalized_class?: string;
    superclass?: string;
    membership_status?: string;
    split?: string;
    session_id?: string;
    search?: string;
    validation_status?: string;
  }) => {
    const qs = new URLSearchParams();
    Object.entries(params ?? {}).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== "") qs.set(key, String(value));
    });
    return request<MLSetReviewFacetsResponse>(`/api/datasets/${encodeURIComponent(datasetId)}/ml-sets/${encodeURIComponent(mlSetId)}/facets${qs.toString() ? `?${qs.toString()}` : ""}`);
  },
  mlSetClassDistribution: (mlSetId: string, datasetId?: string) =>
    request<Record<string, unknown>>(`/api/ml-sets/${encodeURIComponent(mlSetId)}/class-distribution${datasetId ? `?dataset_id=${encodeURIComponent(datasetId)}` : ""}`),
  mlSetSplitSummary: (mlSetId: string, datasetId?: string) =>
    request<Record<string, unknown>>(`/api/ml-sets/${encodeURIComponent(mlSetId)}/split-summary${datasetId ? `?dataset_id=${encodeURIComponent(datasetId)}` : ""}`),
  mlSetWarnings: (mlSetId: string, datasetId?: string) =>
    request<{ warnings: Array<Record<string, unknown>> }>(`/api/ml-sets/${encodeURIComponent(mlSetId)}/warnings${datasetId ? `?dataset_id=${encodeURIComponent(datasetId)}` : ""}`),
  mlSetRepresentativeSamples: (mlSetId: string, datasetId?: string) =>
    request<{ representative_samples: Record<string, unknown> }>(`/api/ml-sets/${encodeURIComponent(mlSetId)}/representative-samples${datasetId ? `?dataset_id=${encodeURIComponent(datasetId)}` : ""}`),
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
  processTake: (
    takeId: string,
    payload: {
      pipeline_id: string;
      run_until_stage?: string | null;
      reprocess?: boolean;
      stage_params?: Record<string, unknown> | null;
      recipe_id?: string | null;
      recipe_version?: number | null;
    }
  ) =>
    post<Record<string, unknown>>(`/api/takes/${encodeURIComponent(takeId)}/process`, payload),
  pipelineRecipes: (pipelineId: string, includeArchived = false) =>
    request<{ pipeline_id: string; recipes: PipelineRecipe[] }>(
      `/api/pipelines/${encodeURIComponent(pipelineId)}/recipes${includeArchived ? "?include_archived=true" : ""}`
    ),
  pipelineRecipe: (pipelineId: string, recipeId: string, version?: number | null) =>
    request<PipelineRecipe>(
      `/api/pipelines/${encodeURIComponent(pipelineId)}/recipes/${encodeURIComponent(recipeId)}${version != null ? `?version=${encodeURIComponent(String(version))}` : ""}`
    ),
  createPipelineRecipe: (
    pipelineId: string,
    payload: {
      recipe_id?: string | null;
      name: string;
      description?: string | null;
      tags?: string[];
      notes?: string | null;
      parameters_by_unit?: Record<string, unknown> | null;
      stage_params?: Record<string, unknown> | null;
      unit_enabled_state?: Record<string, unknown> | null;
      artifact_policy?: Record<string, unknown> | null;
      calibration_snapshot_reference?: Record<string, unknown> | null;
      provenance?: Record<string, unknown> | null;
      parent_recipe_id?: string | null;
    },
  ) => post<PipelineRecipe>(`/api/pipelines/${encodeURIComponent(pipelineId)}/recipes`, payload),
  createPipelineRecipeFromCurrent: (
    pipelineId: string,
    payload: {
      recipe_id?: string | null;
      name: string;
      description?: string | null;
      tags?: string[];
      notes?: string | null;
      parameters_by_unit?: Record<string, unknown> | null;
      stage_params?: Record<string, unknown> | null;
      unit_enabled_state?: Record<string, unknown> | null;
      artifact_policy?: Record<string, unknown> | null;
      calibration_snapshot_reference?: Record<string, unknown> | null;
      provenance?: Record<string, unknown> | null;
      parent_recipe_id?: string | null;
    },
  ) => post<PipelineRecipe>(`/api/pipelines/${encodeURIComponent(pipelineId)}/recipes/from-current`, payload),
  updatePipelineRecipe: (
    pipelineId: string,
    recipeId: string,
    payload: {
      name: string;
      description?: string | null;
      tags?: string[];
      notes?: string | null;
      parameters_by_unit?: Record<string, unknown> | null;
      stage_params?: Record<string, unknown> | null;
      unit_enabled_state?: Record<string, unknown> | null;
      artifact_policy?: Record<string, unknown> | null;
      calibration_snapshot_reference?: Record<string, unknown> | null;
      provenance?: Record<string, unknown> | null;
    },
  ) => put<PipelineRecipe>(`/api/pipelines/${encodeURIComponent(pipelineId)}/recipes/${encodeURIComponent(recipeId)}`, payload),
  clonePipelineRecipe: (pipelineId: string, recipeId: string, payload?: { name?: string | null; recipe_id?: string | null }) =>
    post<PipelineRecipe>(`/api/pipelines/${encodeURIComponent(pipelineId)}/recipes/${encodeURIComponent(recipeId)}/clone`, payload ?? {}),
  archivePipelineRecipe: (pipelineId: string, recipeId: string) =>
    post<PipelineRecipe>(`/api/pipelines/${encodeURIComponent(pipelineId)}/recipes/${encodeURIComponent(recipeId)}/archive`, {}),
  validatePipelineRecipe: (
    pipelineId: string,
    payload: { recipe_id?: string | null; parameters_by_unit?: Record<string, unknown> | null; stage_params?: Record<string, unknown> | null },
  ) => post<{
    pipeline_id: string;
    recipe_id?: string | null;
    compatibility: string;
    errors: string[];
    warnings: string[];
    normalized_parameters_by_unit?: Record<string, Record<string, unknown>>;
  }>(`/api/pipelines/${encodeURIComponent(pipelineId)}/recipes/validate`, payload),
  comparePipeline: (
    pipelineId: string,
    payload: {
      left: {
        type: "run" | "recipe" | "current" | string;
        label?: string | null;
        take_id?: string | null;
        run_id?: string | null;
        recipe_id?: string | null;
        version?: number | null;
        stage_params?: Record<string, unknown> | null;
        parameters_by_unit?: Record<string, unknown> | null;
      };
      right: {
        type: "run" | "recipe" | "current" | string;
        label?: string | null;
        take_id?: string | null;
        run_id?: string | null;
        recipe_id?: string | null;
        version?: number | null;
        stage_params?: Record<string, unknown> | null;
        parameters_by_unit?: Record<string, unknown> | null;
      };
    },
  ) => post<PipelineComparisonResponse>(`/api/pipelines/${encodeURIComponent(pipelineId)}/compare`, payload),
  compareBatchPipeline: (
    pipelineId: string,
    payload: {
      take_ids: string[];
      left: {
        type: "run" | "recipe" | "current" | string;
        label?: string | null;
        take_id?: string | null;
        run_id?: string | null;
        recipe_id?: string | null;
        version?: number | null;
        stage_params?: Record<string, unknown> | null;
        parameters_by_unit?: Record<string, unknown> | null;
      };
      right: {
        type: "run" | "recipe" | "current" | string;
        label?: string | null;
        take_id?: string | null;
        run_id?: string | null;
        recipe_id?: string | null;
        version?: number | null;
        stage_params?: Record<string, unknown> | null;
        parameters_by_unit?: Record<string, unknown> | null;
      };
    },
  ) => post<PipelineBatchComparisonResponse>(`/api/pipelines/${encodeURIComponent(pipelineId)}/compare-batch`, payload),
  pipelineComparisons: (pipelineId: string, params?: { take_id?: string | null; limit?: number }) => {
    const query = new URLSearchParams();
    if (params?.take_id) query.set("take_id", params.take_id);
    if (params?.limit != null) query.set("limit", String(params.limit));
    const suffix = query.toString() ? `?${query.toString()}` : "";
    return request<{ pipeline_id: string; comparisons: PipelineComparisonIndexEntry[] }>(
      `/api/pipelines/${encodeURIComponent(pipelineId)}/comparisons${suffix}`,
    );
  },
  pipelineComparison: (pipelineId: string, comparisonId: string) =>
    request<PipelineComparisonResponse>(`/api/pipelines/${encodeURIComponent(pipelineId)}/comparisons/${encodeURIComponent(comparisonId)}`),
  pipelineRuns: (pipelineId: string, params?: { take_id?: string | null; limit?: number }) => {
    const query = new URLSearchParams();
    if (params?.take_id) query.set("take_id", params.take_id);
    if (params?.limit != null) query.set("limit", String(params.limit));
    const suffix = query.toString() ? `?${query.toString()}` : "";
    return request<{ pipeline_id: string; runs: PipelineRunEntry[] }>(
      `/api/pipelines/${encodeURIComponent(pipelineId)}/runs${suffix}`,
    );
  },
  pipelineRunLineage: (pipelineId: string, runId: string) =>
    request<PipelineRunLineage>(
      `/api/pipelines/${encodeURIComponent(pipelineId)}/runs/${encodeURIComponent(runId)}/lineage`,
    ),
  generateRunComparisonToParent: (pipelineId: string, runId: string) =>
    post<PipelineComparisonResponse>(
      `/api/pipelines/${encodeURIComponent(pipelineId)}/runs/${encodeURIComponent(runId)}/generate-comparison`,
      {},
    ),
  planPipelinePartialRerun: (
    pipelineId: string,
    payload: {
      selected_unit_id: string;
      changed_parameters?: Record<string, unknown> | null;
      current_recipe_snapshot?: Record<string, unknown> | null;
      parent_run_result_ref?: { take_id: string; run_id?: string | null } | null;
      parent_run_result?: Record<string, unknown> | null;
    },
    persist = false,
  ) => post<PartialRerunPlanResponse>(`/api/pipelines/${encodeURIComponent(pipelineId)}/partial-rerun/plan${persist ? "?persist=true" : ""}`, payload),
  pipelinePartialRerunPlans: (pipelineId: string, limit = 10) =>
    request<{ pipeline_id: string; plans: PartialRerunPlanListEntry[] }>(
      `/api/pipelines/${encodeURIComponent(pipelineId)}/partial-rerun/plans?limit=${encodeURIComponent(String(limit))}`,
    ),
  executePipelinePartialRerun: (
    pipelineId: string,
    payload: {
      plan_id?: string | null;
      plan?: Record<string, unknown> | null;
      parent_run_ref: { take_id: string; run_id?: string | null };
      effective_recipe_snapshot?: Record<string, unknown> | null;
      overrides?: Record<string, unknown> | null;
      options?: { compare_to_parent?: boolean };
    },
  ) => post<PartialRerunExecuteResponse>(`/api/pipelines/${encodeURIComponent(pipelineId)}/partial-rerun/execute`, payload),
  createProcessingJob: (payload: ProcessingJobCreateRequest) =>
    post<{ job: ProcessingJobRecord }>("/api/processing-jobs", payload),
  processingJobs: () => request<{ jobs: ProcessingJobRecord[]; counts: Record<string, number> }>("/api/processing-jobs"),
  processingJob: (jobId: string) => request<{ job: ProcessingJobRecord }>(`/api/processing-jobs/${encodeURIComponent(jobId)}`),
  cancelProcessingJob: (jobId: string) =>
    post<{ ok: boolean; job_id: string; status: ProcessingJobStatus }>(`/api/processing-jobs/${encodeURIComponent(jobId)}/cancel`, {}),
  createFeatureJob: (payload: FeatureJobRequest) => post<FeatureJobResponse>("/api/ml/feature-jobs", payload),
  featureJob: (jobId: string) => request<FeatureJobResponse>(`/api/ml/feature-jobs/${encodeURIComponent(jobId)}`),
  cancelFeatureJob: (jobId: string) =>
    post<{ ok: boolean; job_id: string; status: FeatureJobStatus }>(`/api/ml/feature-jobs/${encodeURIComponent(jobId)}/cancel`, {}),
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
