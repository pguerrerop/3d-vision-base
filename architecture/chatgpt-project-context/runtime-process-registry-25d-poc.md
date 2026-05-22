# Runtime Process Registry (2.5D POC)

## Purpose
Add a lightweight, filesystem-first runtime process visibility layer for operational monitoring of local POC services.

Scope is runtime monitoring only. No pipeline-stage refactors and no Studio semantic changes.

## Registry Storage
Process registry files are written under:
- `data/runtime/processes/<process_name>.json`

Each file stores compact heartbeat metadata, including:
- process name
- role (`watcher`, `worker`, `api`, etc.)
- pid
- status
- started/heartbeat timestamps
- host/version
- optional counters/metadata

## Heartbeat Semantics
Implemented in `vision_3d_acquisition/runtime/process_registry.py`.

Behavior:
- process registers on startup
- periodic heartbeat every ~3s
- atomic overwrite writes (`.tmp` + rename)
- graceful shutdown marks `status=stopped`
- failure path can mark `status=failed`

## Stale Detection
Staleness is computed dynamically in API reads, not persisted:
- stale when `now - last_heartbeat > 15s`
- rendered status set to `stale` only for entries still marked `running`

Runtime status surface:
- `running`
- `stale`
- `stopped`
- `failed`

## Integrated Processes
Integrated heartbeats:
- `scripts/watch_trispector_folder.py` as role `watcher` (`process_name=trispector_take_watcher`)
- `scripts/run_25d_worker.py` as role `worker` (`process_name=run_25d_worker`)
- `vision_3d_acquisition/api/main.py` as role `api` (`process_name=api_server`)

## API Contract
Endpoint:
- `GET /api/runtime/processes`

Returns compact array entries such as:
- `process_name`
- `runtime_role`
- `status`
- `pid`
- `last_heartbeat`
- `is_stale`

## Operations UI
Operations page includes a compact "Runtime Processes" strip with glanceable statuses (Watcher, 25D Worker, API).

No logs view, no orchestration graph, no DAG/metrics dashboard added.

## Smoke Flow
1. Start watcher (`watch_trispector_folder.py`)
2. Start worker (`run_25d_worker.py`)
3. Start API (`python -m vision_3d_acquisition.api.main`)
4. Verify:
   - `curl http://localhost:8000/api/runtime/processes`
5. Drop a valid take into watched folder
6. Verify acquired -> processing -> completed in Operations card flow
7. Kill worker process
8. Verify worker becomes `stale` after ~15s
9. Restart worker
10. Verify processing resumes and worker returns to `running`

## Explicit Non-goals
Not implemented:
- Celery/Redis/Kafka
- distributed orchestration
- Kubernetes
- websocket streaming
- Prometheus/Grafana
- automatic restarts/process supervisor redesign
