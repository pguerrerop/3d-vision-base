# ML Stabilization Pass: Safety Tests, Diagnostics Visibility, Calibration Compatibility

## Deployment Invariants and Lifecycle Safety
Automated backend tests were added under `tests/ml/` for deployment lifecycle invariants:
- single active deployment per `(pipeline_id, stage_id, pipeline_family)`
- supersede behavior on activation
- rollback lineage and immutable snapshot behavior
- activation/deactivation transition integrity

## Runtime Diagnostics Flow
Classification runtime now emits and surfaces `classification_runtime_diagnostics.json` with:
- deployment resolution details
- compatibility checks (errors/warnings)
- fallback decision and reason
- backend used and inference duration
- schema hash/version
- calibration compatibility snapshot

Studio Inspector now loads and displays runtime diagnostics for classification runs.

## Calibration and Metric Compatibility Rules
Runtime input validation now enforces model-declared requirements:
- `required_metric_fields`
- `required_calibrated_fields`
- `required_modalities`
- `requires_mm_calibration`

If validation fails, inference is rejected and heuristic fallback is used with explicit diagnostics.

## Schema Integrity Validation
Feature schema integrity now includes:
- feature ordering hash
- schema fingerprint
- feature count and dtype expectations

These are persisted with trained model metadata and included in runtime compatibility diagnostics.

## Split and Leakage Guarantees
Split subsystem tests cover:
- split manifest generation
- by-session and by-dataset leakage checks
- chronological split semantics
- leakage report generation and near-duplicate checks

Artifacts persisted per experiment:
- `split_manifest.json`
- `leakage_report.json`

## Runtime Metadata Traceability
Classification payload and metadata now persist:
- deployment/model identifiers and status
- backend name/version
- schema version/hash
- calibration compatibility
- inference duration
- fallback markers

## Operational Observability Model
Lightweight runtime counters are persisted under `data/ml/runtime_stats/`:
- inference count
- fallback count
- incompatible deployment count
- heuristic usage count
- average inference time

No external monitoring dependency is required.

## Guided Workflow UX (Mining/25D POC)
Classifiers now follows a guided path designed for real datasets:
- `Dataset -> Labels -> Features -> Train -> Evaluate -> Promote`

Top-level sections:
- `Overview`
- `Datasets`
- `Labels`
- `Features`
- `Experiments`
- `Evaluation`
- `Registry`
- `Deployment`
- `Runtime Health`

Key workflow semantics:
- dataset preparation shows object counts, class/session distribution, validation coverage, and data quality warnings
- labels view surfaces labeled object previews for quick QA and deep-link context fields
- features view shows schema fingerprint, ordering hash, feature diagnostics (constant/sparse/missing)
- training flow uses split preview + leakage diagnostics (`split_manifest` + `leakage_report`) before run
- evaluation view loads summary/failed examples/confusion/per-class outputs from persisted artifacts
- promotion/deployment remains explicit and immutable with compatibility snapshots preserved

Runtime health dashboard exposes:
- inference/fallback/heuristic and incompatibility rates
- average inference duration
- smoke-test health and recent runtime-stats history

## Classifiers UX Composition
The Classifiers area now follows a stable layout hierarchy:
1. guided workflow header and global health badge
2. clickable workflow stepper (`Dataset`, `Labels`, `Features`, `Train`, `Evaluate`, `Promote`)
3. styled section navigation tabs
4. summary cards/workspace panels
5. expandable advanced diagnostics

## Stepper Semantics
Each step reflects workflow state as `READY`, `PENDING`, or `WARNING`.
Stepper states are derived from existing artifacts and metadata:
- dataset/object availability
- label quality warnings
- feature preview readiness
- training completion
- evaluation artifact availability
- promotion eligibility

## Runtime Health Dashboard
Runtime Health/Overview includes:
- top-level system badge (`HEALTHY`, `WARNING`, `FAILED`)
- KPI cards (inference count, fallback rate, heuristic usage, average inference latency, incompatible deployments)
- latest smoke-test summary card
- daily runtime history table with compact stats

## Diagnostics Disclosure Model
Engineering detail is preserved via progressive disclosure:
- primary UX surfaces concise summaries and badges
- advanced deployment/runtime compatibility data stays available in collapsible diagnostic sections
- ids/hashes/fingerprints remain visible in monospace blocks for auditability

## Deployment UX Safeguards
Deployment actions now include explicit confirmations for:
- activation (with supersede warning)
- deactivation
- rollback lineage creation
- model promotion and deployment creation

These are UI-only safeguards; existing backend lifecycle invariants remain authoritative.

## Progressive Loading Strategy
Classifiers data hydration uses partial-loading semantics:
- overview and dataset panels fetch via independent `Promise.allSettled` groups
- endpoint failures degrade gracefully and show partial-data warnings
- heavy details (logs/evaluation/diagnostics JSON) load on selected context and stay in bounded scroll containers

## Operator-Density Workspace Composition
The Classifiers UI now targets operator-screen density:
- reduced vertical padding, tighter table rows, compact badges, and denser stepper/tabs
- KPI telemetry presented in a single compact strip
- workspace-first split panes instead of stacked cards:
  - health: smoke summary + runtime history side-by-side
  - experiments: training control + leakage/diagnostic review split
  - deployment: operational table + compact diagnostics drawer split

Density principles:
- prioritize above-the-fold operational signals and primary actions
- keep advanced JSON/details collapsed by default
- preserve fast scanning with stronger heading/header contrast and compact status chips

## Classifier Backend Registry and Comparison Semantics
Classifier selection is now a first-class experiment dimension.

Backend registry module:
- `vision_3d_acquisition/ml/backend_registry.py`
- `get_available_backends()` exposes backend metadata and placeholders
- `load_backend()` resolves supported runtime/trainable backend ids

Initial selectable backends:
- `heuristic_ruleset`
- `logistic_regression`
- `random_forest`
- `gradient_boosting`
- `svm`

Future placeholders:
- `pytorch`
- `onnx`
- `tensorflow`

Exposed backend metadata includes:
- backend name/version
- interpretability level
- training/inference speed profile
- feature and calibration requirements

## Comparison Experiment Workflow
Experiments support:
- single-backend mode
- `comparison_experiment` mode with multiple backends sharing dataset/features/split

Comparison artifacts:
- per-backend evaluation outputs under `data/ml/evaluations/<experiment>/<backend>/`
- `comparison_summary.json` with ranking and best candidate recommendation

Ranking policy:
- `f1_macro`
- `precision_macro`
- `recall_macro`
- `inference_duration_ms` (faster is better)
- `training_duration_ms` (faster is better)

Promotion/deployment semantics remain explicit and unchanged:
- no automatic promotion of recommended backend
- deployment activation/rollback still requires explicit user action
