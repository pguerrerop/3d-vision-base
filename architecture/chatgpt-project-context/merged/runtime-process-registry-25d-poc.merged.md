# Merged: Lightweight Runtime Process Registry for 2.5D POC

## Goal
Provide operational visibility of independently running runtime processes (watcher, worker, API) without changing pipeline or Studio internals.

## Added Runtime Layer

### 1) Filesystem Registry Helper
File: `vision_3d_acquisition/runtime/process_registry.py`

Capabilities:
- register process record
- periodic heartbeat updates
- status transitions (`running`, `failed`, `stopped`)
- atomic heartbeat writes
- process listing with dynamic stale computation

Registry path:
- `data/runtime/processes/<process_name>.json`

### 2) Heartbeat Integration

#### Watcher
File: `scripts/watch_trispector_folder.py`
- role: `watcher`
- process name: `trispector_take_watcher`
- heartbeat includes acquired/error counters

#### 25D Worker
File: `scripts/run_25d_worker.py`
- role: `worker`
- process name: `run_25d_worker`
- heartbeat includes processing counters

#### API
File: `vision_3d_acquisition/api/main.py`
- role: `api`
- process name: `api_server`
- startup/shutdown hooks start and stop periodic heartbeat

### 3) Stale Detection
Computed at read time in API/registry listing:
- stale when heartbeat older than 15s
- returned `status` resolves to `stale` for stale-running processes
- no permanent stale flag persisted

### 4) Runtime Processes API
Endpoint:
- `GET /api/runtime/processes`

Returns compact process summaries for Operations monitoring (name, role, status, pid, heartbeat, stale flag).

### 5) Operations UI Integration
File: `frontend/src/pages/OperatorInspectionPage.tsx`

Adds compact "Runtime Processes" monitor section:
- Watcher: RUNNING/STALE/FAILED/STOPPED
- 25D Worker: RUNNING/STALE/FAILED/STOPPED
- API: RUNNING/STALE/FAILED/STOPPED

Keeps UI operational and glanceable; no engineering-heavy diagnostics added.

## Operational Validation Flow
1. Start watcher
2. Start worker
3. Start API
4. `curl /api/runtime/processes`
5. Push take into watched folder
6. Verify acquisition and completion flow
7. Kill worker
8. Confirm stale after ~15s
9. Restart worker
10. Confirm return to running and resumed processing

## Separation from Studio/Pipeline
No redesign performed for:
- native 2.5D stages
- stage artifacts/semantics
- Studio engineering workflows

Implementation remains additive, local-runtime, and POC-oriented.
