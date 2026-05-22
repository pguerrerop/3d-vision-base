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
- operations-card generation
- preview fallback resolution
- runtime index persistence (`data/runtime/operations_index.json`)

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
