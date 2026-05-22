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

2. `scripts/run_25d_worker.py`
- polls takes from `data/incoming`
- handles runtime states: `queued`, `processing`, `completed`, `failed`
- executes only `mining_steel_ball_classification_25d`
- preserves existing pipeline stage/artifact semantics
- publishes operations cards and updates runtime index

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
