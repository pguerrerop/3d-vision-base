# Runtime POC: Native 2.5D Operations Flow

## Scope
This POC adds a filesystem-first runtime architecture focused only on `mining_steel_ball_classification_25d` for demo operations monitoring.

Non-goals:
- RGB+25D fusion
- multisource synchronization
- DAG orchestration
- Studio workflow redesign

## Process Separation
Three independent processes are expected:

1. `scripts/watch_trispector_folder.py`
- watches TriSpector FTP/logging folder
- applies safe completion heuristics (stable size, optional done marker)
- registers acquired takes in `data/incoming/<take_id>/`
- writes `runtime_state.json` with `state=acquired`
- does not start processing

python scripts/watch_trispector_folder.py \
  --watch-dir data/trispector_uploads \
  --data-dir data \
  --source-id trispector_ftp \
  --poll-interval-sec 1.0

### FTP server mode (project-hosted TriSpector ingress)
When this repository hosts the FTP endpoint directly (instead of watching an externally written folder), start the supervised FTP runtime process:

source .venv/bin/activate
python scripts/runtime.py start trispector_ftp --foreground

Use this mode instead of `watch_trispector_folder.py` for the same upload flow.

2. Detection: `scripts/run_25d_worker.py`
- polls takes from `data/incoming`
- handles runtime states: `queued`, `processing`, `completed`, `failed`
- executes only `mining_steel_ball_classification_25d`
- preserves existing pipeline stage/artifact semantics
- publishes operations cards and updates runtime index

python scripts/run_25d_worker.py

3. API/UI process (`python -m vision_3d_acquisition.api.main`)
- exposes lightweight operations feed via `/api/operations/cards`
- keeps Operations view focused on monitoring, not engineering internals

## Operational Summary Contract
Each processed take publishes `operations_card.json` (under `data/processed/<take_id>/`) with compact fields:
- take id
- status
- superclass
- detailed label
- confidence
- object count
- preview image reference
- acquired/processed timestamps
- optional error summary

Operations UI consumes only this card contract.

## Runtime Index
Worker updates `data/runtime/operations_index.json` incrementally.

Purpose:
- avoid full scans of all run/stage folders
- keep operations reads lightweight and stable

## Superclass Mapping
Implemented in `vision_3d_acquisition/operations/classification_superclass.py`.

Operational classes:
- `BALL_GOOD`
- `BALL_SCRAP`
- `SCRAP`
- `UNKNOWN`

Mapping is additive and non-breaking for evolving stage outputs.

### sph3d fallback (25D classifier refinement)
After primary `_classify_25d` heuristics, a secondary fallback runs for objects **not** already classified as `BALL_GOOD` or `SCRAP_METAL`:

1. Primary good-ball rules run first (unchanged).
2. If result is `BALL_GOOD` or `SCRAP_METAL`, skip fallback entirely.
3. Otherwise apply `feature_sphericity_3d` (`sph3d`) thresholds:
   - `< 0.30` → `chatarra` / `SCRAP_METAL`
   - `0.30–0.75` → `bola_con_chip` / `BALL_SCRAP`
   - `>= 0.75` → keep primary label/superclass

Thresholds are heuristic and calibration-oriented. Objects include `debug.sph3d_rule` and optional `classification_reason` for traceability. The previous `sphere_fit` object metric is now named `feature_footprint_roundness`.

## Preview Fallback Behavior
Operations card preview resolves in this order:
1. `classification_overlay.png`
2. `height_segmentation_overlay.png`
3. `normalized_heightmap_preview.png`
4. `raw_heightmap_preview.png`

Failed runs still publish cards and attempt best available preview.

## Studio vs Operations Responsibility
Studio remains engineering/debug environment (stage-level views, tuning, diagnostics).

Operations remains production-style monitoring:
- status progression
- quick classification signal
- confidence
- previews
- failure visibility

## Canonical height semantics alignment

Operations cards remain lightweight, but semantic contracts are now explicit:

- production metrics and labels are derived from canonical `height_above_belt` geometry semantics;
- preview imagery remains display-only and does not serve as numeric measurement input;
- semantic lineage metadata (`derived_from`, `transform`) is preserved in processing artifacts so replay/debug tools can reconstruct geometry provenance without filename heuristics.

This keeps Operations additive while preserving a single geometry truth across Studio and runtime views.

## Unified Sensor Studio CLI (wrapper only)

A unified wrapper CLI now centralizes common runtime/demo operations without changing pipeline logic.

Script:
- `scripts/sensor_studio_cli.py`

Top-level structure:
- `sensor-studio 25d ...`
- `sensor-studio studio ...`
- `sensor-studio acquisition ...`

Primary 25D operations:

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

Notes:
- `25d process` and `25d validate-api` require `--take-id`.
- `25d clean` requires either `--take-id` or `--session-id`.
- Existing scripts remain valid; this CLI delegates to the same runtime/application functions.

## Repeatability Analysis Extension (Additive)

To evolve from single-take inspection toward measurement characterization, an offline analysis layer is introduced:

- `python scripts/analyze_feature_repeatability.py`

Capabilities:
- groups repeated takes by `physical_object_id`
- aggregates scalar features from persisted processing results
- computes stability metrics (`mean`, `std`, `cv`, `min/max`, `outlier_count`, `missing_data_ratio`)
- produces feature-level variability ranking and instability flags
- computes orientation-sensitivity heuristic classes:
  - `low_sensitivity`
  - `medium_sensitivity`
  - `high_sensitivity`
- emits correlation summaries against acquisition-quality signals when present

Output artifacts:
- JSON summary + CSV tables under `data/runtime/analysis/repeatability_<timestamp>/`

Architecture constraints preserved:
- no change to existing processing pipeline contracts
- no mutation of raw acquisitions
- no mutation of session/grouping semantics
- no runtime hidden state; outputs are explicit filesystem artifacts

## 25D Measurement Diagnostics Stage (Additive)

New additive stage:

- `ComputeMeasurementDiagnostics`
- placement: after measurement/geometry correction and before classification
- pipeline semantics preserved; prior outputs are unchanged

Persisted artifacts (canonical, stage-owned):

- `measurement_diagnostics.json`
- `feature_vector.json`
- `feature_provenance.json`
- `quality_flags.json`
- `contour.json`
- `convex_hull.json`
- `fitted_ellipse.json`
- `principal_axes.json`
- `radial_profile.json`
- `normalized_height_histogram.json`

Contract expectations:

- artifacts are registered in processing artifacts with:
  - `artifact_id`
  - `stage_id=measurement_diagnostics`
  - `kind=json`
  - coordinate-space and lineage metadata
- no replacement of existing artifact ids
- no mutation of classification flow

Feature provenance model:

- each feature stores:
  - `value`
  - `source_stage`
  - `source_artifact_id`
  - `validity` (`ok|warning|invalid|unavailable`)
  - optional future placeholders: `confidence`, `expected_error`
  - `derived_from`

Quality flags:

- rule-based additive warnings:
  - `LOW_VALID_PIXEL_RATIO`
  - `HIGH_INVALID_REGION_RATIO`
  - `BORDER_TOUCH`
  - `HIGH_PLANE_RESIDUAL`
  - `SMALL_OBJECT`
  - `SEGMENTATION_FRAGMENTED`
  - `EXTREME_HEIGHT_OUTLIER`
- each includes severity, trigger values, and related feature references

## Rule Tuning Workflow (Pre-demo)

Additive operational script:

- `python scripts/tune_25d_rules.py`

Inputs:
- exported 25D feature CSV (preferred)
- optional dataset/ml-set metadata for reporting/discovery

Outputs:
- tuned threshold config (`best_rules.json`)
- confusion matrix and per-class metrics
- per-take predictions with rule-path traces

This is intentionally not model training; it tunes explainable rule parameters only.

## Classifier Engine vs Rule Set

The runtime now explicitly separates:

- classifier engine: `mining_steel_ball_classification_25d`
- rule set: threshold/config payload applied by that engine

Default behavior:

- if no rule config is provided, built-in rules remain active (backward compatible)

Optional behavior:

- if `classifier_rules_path` is provided, classification stage loads params from external JSON and applies the same deterministic evaluator structure

Expected config location:

- `configs/classifiers/*.json`

This enables client/demo/site-specific variants without changing pipeline topology.

## Classification Provenance

Classification outputs now carry rule-set provenance fields:

- `classifier_engine`
- `rule_set_id`
- `rule_set_path`
- `rule_set_version`
- `rule_set_source`
- `rule_path`
- `confidence_proxy`

Resolution precedence is deterministic:

- runtime override path
- pipeline-config rule-set path
- `SENSOR_STUDIO_DEFAULT_RULE_SET` env default
- built-in default

Purpose:

- reproducible comparisons
- regression analysis
- auditable client reporting

## Offline Rule Evaluation

New script:

- `python scripts/evaluate_25d_rules.py`

Capabilities:

- evaluate one rule config against exported feature CSV
- optional comparison between two configs (`--compare-rules-config`)
- object-safe evaluation via `physical_object_id` grouping/splits
- outputs: report, confusion matrix, per-class metrics, predictions, misclassified rows, disagreements
