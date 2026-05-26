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
