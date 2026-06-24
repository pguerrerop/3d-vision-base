# Merged Context: Runtime POC for Native 2.5D Operations

## Objective
Add an additive runtime path for tomorrow's demo using only `mining_steel_ball_classification_25d`.

## Final Runtime Flow
TriSpector folder/FTP logging -> watcher process -> 25D worker process -> operations card + runtime index -> Operations UI.

## What Was Added

### 1) Take watcher process
File: `scripts/watch_trispector_folder.py`
- polls a watch directory
- uses safe completion heuristics to avoid partial files
- parses TriSpector 2.5D uploads and registers `data/incoming/<take_id>/`
- writes `metadata.json`, `READY`, and `runtime_state.json`
- runtime state starts as `acquired`
- does not invoke processing

### 2) 25D worker process
File: `scripts/run_25d_worker.py`
- polls incoming 25D takes
- state transitions: `acquired -> queued -> processing -> completed|failed`
- executes only `mining_steel_ball_classification_25d`
- on every run outcome, publishes `operations_card.json`
- updates shared runtime index incrementally

### 3) Operations data contracts
Files:
- `vision_3d_acquisition/operations/classification_superclass.py`
- `vision_3d_acquisition/operations/summary.py`

Capabilities:
- superclass mapping helper
- sph3d fallback classifier (secondary pass for non-`BALL_GOOD`/non-`SCRAP_METAL` objects)
- operations-card generation
- preview fallback resolution
- runtime index persistence (`data/runtime/operations_index.json`)

### sph3d fallback (25D classifier refinement)
Implemented in `classification_superclass.py`, invoked from `ClassifyMiningBall25DStage` after primary `_classify_25d` heuristics.

Execution order:
1. Primary good-ball rules run first (unchanged thresholds in `_classify_25d`).
2. If result is `BALL_GOOD` or `SCRAP_METAL`, skip fallback entirely.
3. Otherwise apply `feature_sphericity_3d` (`sph3d`) thresholds:
   - `< 0.30` → `chatarra` / `SCRAP_METAL`
   - `0.30–0.75` → `bola_con_chip` / `BALL_SCRAP`
   - `>= 0.75` → keep primary label/superclass

Thresholds are heuristic and calibration-oriented. Object payloads include `debug.sph3d_rule` and optional `classification_reason`. The previous `sphere_fit` object metric is now named `feature_footprint_roundness`. Superclass aggregation is unchanged.

### 4) Operations endpoint
File: `vision_3d_acquisition/api/main.py`
- new endpoint: `GET /api/operations/cards?limit=...`
- serves compact cards from runtime index

### 5) Operations UI
Files:
- `frontend/src/pages/OperatorInspectionPage.tsx`
- `frontend/src/api/client.ts`

Behavior:
- lightweight monitoring list/cards for 25D runtime
- shows preview, take id, acquisition/processed time, status, superclass, detailed label, confidence, object count, and errors
- no stage explorer or engineering diagnostics

## Contract Details

### Operations card (per take)
Stored at `data/processed/<take_id>/operations_card.json`.
Includes:
- `take_id`
- `status`
- `superclass`
- `label`
- `confidence`
- `object_count`
- `preview_image`
- `acquired_at`
- `processed_at`
- `error` (when failed)

### Runtime index
Stored at `data/runtime/operations_index.json`.
- contains latest compact card entries
- maintained incrementally by worker
- avoids expensive full-directory scans in Operations path

## Failure Handling
- failed processing does not crash watcher/worker loops
- worker keeps best generated artifacts
- failed cards remain visible in Operations with error summaries

## Studio Separation
No redesign/refactor of:
- Studio UI workflows
- stage semantics
- internal 2.5D pipeline architecture

Operations path is additive and intentionally minimal.

## Canonical height semantics alignment

Operations remains lightweight, but semantic contracts are explicit:

- classification and measurement-facing geometry semantics align to canonical `height_above_belt`;
- preview imagery remains display-only and is not a numeric measurement source;
- semantic lineage metadata (`derived_from`, `transform`) is retained so replay/debug tooling can reconstruct provenance without filename heuristics.

This preserves the additive Operations design while keeping one geometry truth across Studio and runtime.

## Unified Sensor Studio CLI (wrapper only)

A unified wrapper CLI now centralizes runtime/demo operations without changing pipeline internals.

Script:
- `scripts/sensor_studio_cli.py`

Structure:
- `sensor-studio 25d ...`
- `sensor-studio studio ...`
- `sensor-studio acquisition ...`

Main 25D commands:

```bash
python scripts/sensor_studio_cli.py --data-dir data 25d create-synthetic --session-id synthetic_25d_demo
python scripts/sensor_studio_cli.py --data-dir data 25d demo
python scripts/sensor_studio_cli.py --data-dir data 25d process --take-id <take_id>
python scripts/sensor_studio_cli.py --data-dir data 25d validate-api --take-id <take_id>
python scripts/sensor_studio_cli.py --data-dir data 25d inspect-result --take-id <take_id>
python scripts/sensor_studio_cli.py --data-dir data 25d clean --take-id <take_id>
python scripts/sensor_studio_cli.py --data-dir data 25d interactive
```

Support commands:

```bash
python scripts/sensor_studio_cli.py --data-dir data studio serve
python scripts/sensor_studio_cli.py --data-dir data acquisition list-takes --limit 20
python scripts/sensor_studio_cli.py --data-dir data acquisition latest
```

Validation rules:
- `25d process` and `25d validate-api` require `--take-id`.
- `25d clean` requires `--take-id` or `--session-id`.
- Legacy scripts remain available; this command is an additive wrapper.

## Repeatability Analysis Extension (Additive)

An offline repeatability-analysis entrypoint is now part of the architecture:

- `python scripts/analyze_feature_repeatability.py`

It operates on persisted outputs (no pipeline mutation) and provides:
- grouping by `physical_object_id`
- per-feature repeatability metrics across repeated takes
- variability ranking/instability flags
- orientation-sensitivity heuristic classes
- acquisition-quality vs instability correlation tables when quality features are available

Produced artifacts:
- `repeatability_summary.json`
- `repeatability_per_object_feature.csv`
- `repeatability_feature_stability.csv`
- `repeatability_correlations.csv`

These are written under `data/runtime/analysis/repeatability_<timestamp>/` and remain additive to current runtime/operations contracts.

## 25D Diagnostics + Provenance Layer

An additive diagnostics stage is introduced in the 25D pipeline before classification:

- `ComputeMeasurementDiagnostics`

Persisted diagnostic/provenance artifacts:

- `measurement_diagnostics.json`
- `feature_vector.json`
- `feature_provenance.json`
- `quality_flags.json`
- geometry intermediates (`contour`, `convex_hull`, `fitted_ellipse`, `principal_axes`, `radial_profile`, histogram)

Design guarantees:

- existing artifact ids/flows remain valid
- stage semantics remain deterministic and modality-aware
- metadata includes lineage/source references and coordinate-space hints
- confidence fields are reserved as optional (`null`) for future confidence modeling

Integration points:

- ML export can optionally include diagnostics, invalidity flags, and provenance summaries
- repeatability analysis can optionally correlate stability with diagnostics/quality signals

## Pre-demo Explainable Tuning

New additive script:

- `scripts/tune_25d_rules.py`

It tunes threshold parameters of the existing hierarchical ruleset, evaluates object-safe splits, and produces an auditable report/config bundle without introducing ML model training complexity.

## Selectable Rule Sets (Additive)

The 25D classifier architecture now supports optional external rule-set configs while preserving built-in defaults.

Model:

- engine: `mining_steel_ball_classification_25d`
- rule sets: `builtin_default`, tuned configs, client/demo configs

When a rule config is supplied (`classifier_rules_path`), classification runs with loaded params and emits provenance metadata (`rule_set_id`, `rule_set_source`, `rule_path`, `confidence_proxy`, etc.).

Resolution precedence is deterministic:

- runtime override path
- pipeline-config rule-set path
- `SENSOR_STUDIO_DEFAULT_RULE_SET` env default
- built-in default

When omitted, behavior is unchanged.

## Rule Evaluation & Comparison

Added offline evaluator:

- `scripts/evaluate_25d_rules.py`

Outputs include:

- `evaluation_report.json`
- `confusion_matrix.csv`
- `per_class_metrics.csv`
- `predictions.csv`
- `misclassified.csv`
- `disagreements.csv` (when comparing two configs)

Evaluation remains object-safe via `physical_object_id` grouping.
