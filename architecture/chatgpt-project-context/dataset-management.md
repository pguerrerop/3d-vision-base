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
