# Dataset Management Architecture (MVP)

## Goal

Add a first-class experiment layer on top of existing take/run processing:

- `Dataset` (project/campaign)
- `Take Session` (acquisition/setup condition)
- `Take metadata` (labels, notes, expected values, validation state)

Raw take data and processing outputs remain unchanged.

## Storage layout

New sidecar metadata root:

```text
data/
  datasets/
    dataset_<id>/
      dataset.json
      sessions/
        session_<id>/
          session.json
          takes/
            <take_id>/
              metadata.json
```

Existing paths are preserved:

- raw takes: `data/incoming/<take_id>/`
- legacy sessions: `data/sessions/...`
- run outputs: `data/processed/<take_id>/` and `data/processes/runs/...`

## Core model

### Dataset

- `id`, `name`, `description`, `created_at`, `tags`, `notes`

### Dataset session

- `id`, `dataset_id`, `name`, `description`, `calibration_id`
- `sensor_metadata`, `conveyor_metadata`, `lighting_metadata`
- `created_at`, `tags`, `notes`

### Take metadata

- `friendly_name`, `labels`, `tags`, `notes`
- `expected_class`, `expected_diameter_mm`, `expected_count`
- `operator_notes`, `validation_status`
- `dataset_id`, `session_id`

## Compatibility strategy

Legacy takes without dataset metadata are loaded with synthesized defaults at runtime:

- `friendly_name = take_id`
- empty `tags/labels`
- `validation_status = unreviewed`

This keeps old takes visible and runnable with no migration step.

## Backend additions

- New lightweight filesystem repository/service: `DatasetService`
- New endpoints:
  - `GET/POST/PUT /api/datasets`
  - `GET/POST/PUT /api/datasets/{dataset_id}/sessions`
  - `PUT /api/takes/{take_id}/metadata`
- `GET /api/takes` now supports optional filters:
  - `dataset_id`, `validation_status`, `tag`, `search`

## Studio UX changes (MVP)

Processing Lab sidebar now supports:

- dataset selector
- dataset session filter
- search, tag filter, validation filter
- richer take cards: friendly name, thumbnail, tags, run status
- lightweight take management actions:
  - rename, notes, tags, expected class, expected diameter
  - quick tag chips
  - validation status update

## Acquisition vs processing preserved

Design invariant remains unchanged:

- take = immutable acquisition container
- run = processing interpretation of a take
- one take can have many runs

No raw payload mutation or run path changes were introduced.

## Object-level annotations (MVP)

Take metadata now supports per-candidate annotations under `object_annotations` in the take sidecar metadata.

Shape:

```json
{
  "object_annotations": [
    {
      "id": "object_001",
      "source_stage": "blob_detection",
      "source_artifact_id": "blob_contours",
      "candidate_id": "1",
      "bbox": [x, y, w, h],
      "centroid": [x, y],
      "contour_ref": null,
      "labels": ["ball", "worn_ball"],
      "expected_class": "ball",
      "expected_diameter_mm": 80,
      "notes": "...",
      "validation_status": "unreviewed|accepted|rejected|needs_review",
      "created_at": "...",
      "updated_at": "..."
    }
  ]
}
```

These annotations are explicitly human review metadata and are not mixed into computed pipeline outputs.

### Rerun survival strategy

When loading take detail:

1. match by `candidate_id` first
2. fallback to geometry matching:
  - bbox IoU (`>= 0.2`)
  - nearest centroid as final fallback

Returned take detail includes matched metadata (`matched_candidate_id`, `matched_by`) so prior annotations remain usable after reruns when candidate ids drift.

### API additions

- `POST /api/takes/{take_id}/object-annotations` for upsert
- `GET /api/takes/{take_id}` now includes `object_annotations`

### Studio behavior

- Blob/Contour candidate tables show annotation labels per candidate.
- Inspector for geometry/ellipse stages exposes object-annotation editing controls:
  - multi-label chips
  - expected class
  - expected diameter
  - notes
  - validation status

## Inline Dataset/Session creation UX

Studio sidebar now supports lightweight inline creation for:

- Dataset selector (`+`)
- Experiment session selector (`+`)

Behavior:

- Inline popover form (no route change, no page modal)
- Fields: `name` + optional `notes`
- Uses existing APIs:
  - `POST /api/datasets`
  - `POST /api/datasets/{dataset_id}/sessions`
- After creation:
  - refreshes selector data
  - auto-selects created dataset/session
  - preserves active filtering context where possible
- Includes optimistic loading and inline error state.

Empty states:

- `No datasets yet`
- `Create your first dataset`

## Capture-to-dataset workflow

Studio now supports capturing new takes directly into the selected dataset/session.

Endpoint:

- `POST /api/takes/capture`

Payload fields:

- `dataset_id` (required)
- `dataset_session_id` (required)
- `friendly_name` (optional)
- `tags` (optional)
- `expected_class` (optional)
- `expected_diameter_mm` (optional)
- `notes` (optional)

Implementation notes:

- Reuses existing RGB capture flow (`capture_image`) rather than duplicating acquisition logic.
- Raw take format remains unchanged under `data/incoming/<take_id>/`.
- Dataset sidecar metadata is written/updated immediately after capture:
  - `dataset_id`, `session_id`, `friendly_name`, `tags`, `expected_class`, `expected_diameter_mm`, `notes`, `validation_status=unreviewed`.

Compatibility:

- Existing capture endpoints and scripts remain functional (`/api/capture/image`, `/api/capture/video`, CLI scripts).

## Safe take lifecycle actions

New safe actions are available for dataset-managed takes:

1. **Remove from dataset/session**
- Removes dataset/session association from take metadata.
- Does **not** delete raw incoming files.
- Does **not** delete processed outputs/runs.

2. **Archive take**
- Sets metadata flags:
  - `archived: true`
  - `archived_at`
  - `archived_reason`
- Archived takes are hidden from default take listing.
- Raw data and run outputs are preserved.

3. **Restore archived take**
- Clears archive flags and returns the take to default listing.

4. **Permanent delete (advanced)**
- Requires explicit typed confirmation that must exactly match `take_id`.
- Deletes:
  - `data/incoming/<take_id>` (raw take)
  - `data/processed/<take_id>` (processed output)
  - linked dataset sidecar take metadata folder
  - optionally indexed run directories linked to the take
- API response includes deleted paths and warning text.

### Metadata additions

Take sidecar metadata now includes:

- `archived: boolean`
- `archived_at: string | null`
- `archived_reason: string | null`

### Listing/filter behavior

- Default listing hides archived takes.
- `show_archived=true` exposes archived takes in listing.

## Replayable 2.5D Acquisition Dataset Layer

### Intent
- TriSpector 2.5D acquisition is now persisted in a replayable, filesystem-first contract.
- Each acquired take is stored exactly in the shape consumed by downstream heightmap pipelines.
- Replay reproduces acquisition state; runs remain immutable processing interpretations.

### Replayable Take Contract
- New immutable sidecar: `data/incoming/<take_id>/replay_manifest.json`
- Manifest includes:
  - dataset/session linkage (`dataset_id`, `session_id`)
  - source/modality metadata (`source_id`, `modalities`)
  - asset references (`raw_upload`, `heightmap`, `reflectance`, preview, parser metadata)
  - parser diagnostics and acquisition metadata
  - calibration snapshot (active runtime calibration reference + config snapshot hash context)
  - multimodal-prep fields (`acquisition_group_id`, `frameset_id`, `sync_metadata`)
- Once written, manifest is treated as immutable.

### Acquisition Replay Sessions
- Runtime-managed active replay session state is persisted under:
  - `data/runtime/acquisition_replay/active_session.json`
- Active session links new TriSpector takes to existing dataset sessions automatically.
- Session metadata is independent from processing results and remains dataset-oriented.

### Live TriSpector Integration
- Existing TriSpector FTP ingestion remains the only ingestion path.
- Existing parser remains the only parser path.
- On successful ingestion:
  - assets are persisted in incoming take folder
  - replay manifest is generated
  - active replay session (if any) is attached
  - normal binding-driven processing trigger still runs unchanged

### Replay Execution
- Replay uses the same processing contract path (`dispatch_take_processing`) used by live/manual paths.
- Binding resolution is reused when replay pipeline is not explicitly provided.
- CLI entrypoints:
  - `python scripts/replay_take.py --take-id <id> [--pipeline <pipeline_id>]`
  - `python scripts/replay_session.py --session-id <id> [--dataset-id <id>] [--pipeline <pipeline_id>]`

### Operator/Studio Separation
- Replay metadata and manifests are acquisition/dataset artifacts.
- Processing run outputs remain separate in processed/process-run storage.
- Publication/operator abstractions remain downstream and independent.

## Canonical Label Normalization (Raw + Semantic)

- Raw operator tags remain preserved in `take_metadata.tags` and are never overwritten.
- Canonical semantic fields are additive:
  - `semantic_labels: string[]`
  - `superclass_labels: string[]`
  - `normalized_class: string | null`
  - `normalization_version: string | null`

### Taxonomy

- Canonical taxonomy file:
  - `config/label_normalization/mining_balls_v1.json`
- The taxonomy maps raw tags to canonical semantic/superclass labels for ML-ready querying and training.
- Unknown tags are preserved and flagged via normalization warnings (e.g., `UNMAPPED_TAG:<tag>`).

### Normalization Service

- Service: `vision_3d_acquisition/datasets/label_normalization.py`
- Responsibilities:
  - load taxonomy
  - normalize tags deterministically
  - preserve unknown tags
  - emit stable normalized class (`normalized_class` = first canonical semantic label)

### Normalization CLI

- `python scripts/normalize_take_labels.py --dataset <id> --taxonomy mining_balls_v1`
- Supports:
  - `--dry-run`
  - `--take-id`
  - `--session-id`
- Output summary includes:
  - normalized count
  - unknown tags
  - semantic/superclass distributions

### API Semantics

- `GET /api/takes` supports filtering by:
  - `semantic_label`
  - `superclass_label`
- `GET /api/datasets/{dataset_id}/label-summary` returns:
  - raw tag counts
  - semantic label counts
  - superclass counts
  - unmapped tags
  - normalization version

### Curation Principle

- Raw tags are historical acquisition/operator annotations.
- Semantic labels are canonical ML taxonomy annotations.
- Both coexist permanently.

### Mining Balls Coverage + Split-Tag Repair

- Taxonomy `mining_balls_v1` now includes observed dataset tags such as:
  - `chica`, `chica 2`, `80%`, `ahuevada`, `mitad 2`, `golilla`, `cubo`, `encoder`, `cámara`, `tarro curry`, `tarro spray`, `zoquete`.
- Known split-tag repair is supported during normalization only (raw tags unchanged):
  - `ok 2 (2` + `5")` => interpreted as `ok 2 (2,5")`.
- Repair is surfaced in normalization warnings as:
  - `REPAIRED_SPLIT_TAG:<left + right -> merged>`.
- Aliases are supported in taxonomy entries to preserve deterministic semantic mapping across spelling/format variants.
