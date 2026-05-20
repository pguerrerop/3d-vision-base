# Processes

This document defines each runtime role, its responsibilities, and what it may read or write.

## Overview

| Process | Role | Writes | Reads |
|---------|------|--------|-------|
| Acquisition | Produce takes | `incoming/`, `state/acquisition.json` | Sensor, FTP, or local files |
| Acquisition Studio app | Reusable debug/engineering flow | `processed/<take_id>/` debug artifacts | `incoming/` (with `READY`) |
| Ball Inspection app | Domain classification and statistics | `processed/<take_id>/result.json`, state, events | `incoming/` (with `READY`) |
| Legacy processing | Backward-compatible segmentation flow | `processed/<take_id>/result.json`, state, events | `incoming/` (with `READY`) |
| UI/API | Observe and command (future) | Control channel TBD | `incoming/`, `processed/`, `state/` |
| Output controller | Act on results (future) | Hardware / PLC TBD | `processed/` or commands |

Only **acquisition** publishes new folders under `data/incoming/`. Other processes are consumers or observers unless a future control contract is added.

---

## Acquisition process

**Purpose:** Turn sensor data (or offline inputs) into a versioned **take** on disk.

**Responsibilities:**

- Generate or accept a `take_id`.
- Build valid `metadata.json` (Pydantic-validated).
- Stage payload files (PLY, height TIFF, reflectance PNG, etc.).
- Publish via temp folder + atomic rename + `READY`.
- Update `data/state/acquisition.json` after each successful publish.

**Modes (current and planned):**

| Mode | `metadata.source` | `metadata.mode` | Implementation |
|------|-------------------|-----------------|----------------|
| Offline PLY | `offline_ply` | `offline` | `OfflinePlyAcquisition` |
| Live sensor | `harvesters` (planned) | `live` | Harvesters/GigE (future) |
| FTP drop | `ftp` (planned) | `live` or `batch` | FTP watcher (future) |

**Must not:**

- Write directly to `incoming/<take_id>/` without staging in `.tmp` first.
- Create `READY` before the folder rename completes.
- Modify `processed/` (processing owns that tree).

**Entry points (v1):**

```bash
python scripts/publish_ply_take.py --ply path/to/file.ply --data-dir data
```

---

## Processing process (future)

**Purpose:** Consume complete takes, run algorithms (filtering, meshing, measurement), emit artifacts.

**Responsibilities:**

- Poll or watch `data/incoming/` for directories containing `READY`.
- Validate `metadata.json` against the contract.
- Load referenced files (`point_cloud.ply`, images).
- Write outputs under `data/processed/<take_id>/`.
- Optionally write `DONE` when finished.
- Move or archive the incoming take (policy TBD).

**Must not:**

- Read folders without `READY`.
- Partially modify `incoming/<take_id>/` in place (prefer copy or move after read).

---

## UI/API process (future)

**Purpose:** Expose take lists, previews, and operator controls (start/stop acquisition, reprocess).

**Responsibilities:**

- Read-only access to queue and state for listing and download.
- Optional: write commands to a control file or socket (not defined in v1).
- Serve FastAPI/uvicorn endpoints (dependencies already listed in `requirements.txt`).

**Must not:**

- Bypass acquisition to inject takes without the same publish contract (keeps a single writer semantics).

---

## Output controller (future)

**Purpose:** Drive external equipment (reject gate, robot, printer) from processing results.

**Responsibilities:**

- Subscribe to `processed/` or explicit command files.
- Idempotent handling of duplicate signals.

**Must not:**

- Block acquisition or processing loops.

---

## Inter-process rules of thumb

1. **One writer per tree** — acquisition owns `incoming/` creation; processing owns `processed/` creation.
2. **Markers, not guesses** — presence of `READY` / `DONE` is the contract, not directory mtime alone.
3. **Validate JSON** — use shared Pydantic models in `vision_3d_acquisition.contracts`.
4. **Fail loud** — CLI and services exit non-zero on contract violations; log `take_id` on errors.
