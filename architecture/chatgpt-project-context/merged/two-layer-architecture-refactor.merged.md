# Merged context: two-layer architecture refactor

This merged context reflects the implemented incremental refactor introducing:

- reusable `vision_core` platform modules
- app split (`acquisition_studio`, `ball_inspection`)
- source abstraction (`SensorSource`, `ReplaySource`, `FileSource`)
- modality-aware capture references with grouped assets and frame counts
- USB RGB image/video acquisition validation with OpenCV discovery, synchronous capture endpoints, session links, and runtime source details
- browser-first live preview via throttled latest JPEG plus metadata polling, with native OpenCV preview kept as explicit engineering mode
- explicit acquisition source vs processing pipeline model, pipeline registry endpoint, stage output grouping, and Studio route
- minimal stage-based pipeline (`PipelineContext`, `PipelineStage`, `PipelineRunner`, `PipelineResult`)
- stage modality requirements and clear missing-modality failures
- compatibility bridge to existing real segmentation implementation (not used by default ball pipeline)
- preserved legacy command behavior alongside new app entry points
- `process_latest_real.py --engine native` compatibility path to the stage-native ball inspection flow
- POC readiness layer for run summaries, calibration diagnostics, labels, validation, and object metrics exports
- product organization around Operations, Studio, Calibration, and Diagnostics
- Studio three-pane workspace model: persistent data browser, central modality/stage workbench, contextual inspector
- Studio workstation evolution: fixed shell with independent pane scrolling, stage-driven rendering, selectable object candidates, typed artifact explorer, and contextual inspector updates
- Studio stage behavior separates segmentation, classification, measurements, fusion placeholders, artifact routing, and selected-object context without adding new processing algorithms
- canonical `result.artifacts` contract with stage/object links, explicit emission in native runs, and legacy backfill normalization
- canonical overlay artifact model (`kind: "overlay"`, `overlay_type`, `target_artifact_id`, geometry/style, lineage links)
- explicit overlay coordinate spaces (`image_pixel`, `normalized_image`, `plot_pixel`; future `world_mm`, `point_cloud_projection`)
- strict overlay target resolution and non-renderable/approximate safeguards to avoid misleading visualization
- `result.pipeline_execution` stage diagnostics and ordered execution trace for skipped/incompatible/failed stage introspection
- Studio execution graph for lightweight stage-flow debugging (without node-editor behavior)
- expanded processing-unit registry metadata and pipeline composition blocks (dependencies, artifact flow, optional/conditional stages)
- forward-compatible point-cloud artifact metadata for future 3D viewer integration and 2D projection references
- canonical projection artifact model (`xy_topdown`, `xz_side`, `yz_side`, `object_crop`) with deterministic render coordinate systems
- overlay targeting rules updated to projection artifacts (`projection_pixel`) with explicit legacy compatibility warnings for screenshot/plot overlays
- Calibration UX direction expands from `plane_3d` toward source alignment, coordinate systems, ROI, encoder/sensor synchronization, and future RGB-to-3D alignment
- modality-aware input tabs in Studio/Diagnostics, including RGB video preview metadata, and `plane_3d` calibration metadata
- lightweight acquisition sessions (`data/sessions/*`) and session-aware API/UI filtering
- extended frameset synchronization metadata contract (`frameset_id`, assets, sync mode/confidence)
- runtime acquisition monitor fields (connectivity, preview freshness, lag, queue, stale, calibration/session context)
- live single-process polling pipeline (`scripts/run_live_pipeline.py`) with graceful shutdown
- rolling throughput diagnostics and warnings for queue buildup / processing bottlenecks
- calibration compatibility diagnostics and recommended calibration endpoint

See source context document:

- `architecture/chatgpt-project-context/two-layer-architecture-refactor.md`
