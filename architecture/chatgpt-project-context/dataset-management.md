# Dataset Management Architecture (MVP)

## Goal

Add a first-class experiment layer on top of existing take/run processing:

- `Dataset` (project/campaign)
- `Experiment Session` (canonical engineering/reprocessing scope)
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

### Dataset session / Experiment session

- `id`, `dataset_id`, `name`, `description`, `calibration_id`
- `sensor_metadata`, `conveyor_metadata`, `lighting_metadata`
- `created_at`, `tags`, `notes`

Studio terminology note:

- `Dataset session` is the persisted storage name.
- `Experiment session` is the preferred Studio-facing name.
- This is the canonical scope for replay, calibration context, and bulk reprocessing.

Separate from this, the legacy acquisition/runtime `session_id` attached to raw captures remains available as a low-level operational grouping for ingestion and replay internals.

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

## Responsibility boundary (Studio vs Datasets)

Studio and Datasets now have explicit ownership boundaries:

- Studio (engineering/process-debugging workspace):
  - experiment-session-centric take selection
  - lightweight operational filtering (dataset/experiment-session/search/modality/status)
  - stage-by-stage processing inspection
  - rerun/debug execution flows
- Datasets (semantic curation/ML preparation workspace):
  - labeling/tagging workflows
  - validation governance/review
  - object annotations and dataset composition
  - split management and experiment preparation

Rationale:

- prevent duplicated responsibilities across pages
- keep Studio focused on processing diagnostics
- keep semantic curation in Datasets where governance workflows belong

Final UX boundary notes:

- Studio may expose navigation-time dataset/session scoping for processing selection.
- Studio should treat experiment sessions as the default replay/reprocessing scope and keep acquisition/runtime grouping as an advanced operational control.
- Studio should not foreground curation administration (label edits, validation review flows, semantic bulk ops).
- Datasets remains the primary workspace for metadata authoring, annotation governance, and ML-set preparation.

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

## Curated acquisition organization (additive hardening)

The Dataset/Session/Take semantics are now explicitly hardened for curated acquisition workflows while preserving immutable take and processing contracts.

### Semantic boundaries

- Dataset: project/campaign scope, not a single day/session.
- Session: acquisition context bucket under roughly stable physical/runtime conditions.
- Take: immutable raw acquisition unit with many processing runs over time.

### Session metadata contract (filesystem sidecar)

`dataset_<id>/sessions/session_<id>/session.json` now supports lightweight, non-normalized metadata:

- `session_type`: `engineering | curated | benchmark | operational` (default: `engineering`)
- `sensor_metadata` (model, firmware, profile distance, scan direction, exposure/gain, acquisition mode)
- `conveyor_metadata` (belt speed, encoder enabled, encoder ticks/mm, belt type)
- `lighting_metadata` and `environment_metadata` (ambient light, notes/operator notes)
- `metadata` for extra campaign-specific context

### Take curation metadata contract

`dataset_<id>/sessions/session_<id>/takes/<take_id>/metadata.json` remains the canonical sidecar and now supports:

- `friendly_name` (preferred UI label; `take_id` remains immutable identity)
- `categories` (optional multi-class curation tags), suggested values:
  - `empty_belt_reference`, `calibration_reference`, `golden_sample`, `stress_case`, `benchmark_case`, `ml_training_candidate`, `engineering_debug`, `operational_capture`
- `reference_type` (single primary reference classifier)
- `is_reference`, `is_golden_sample`
- `session_notes` for acquisition observations

### Studio filtering behavior

`GET /api/takes` and Studio sidebar filtering are extended additively with:

- `session_type`
- `category`
- `reference_type`
- `is_reference`
- `is_golden_sample`

Existing filters (`dataset_id`, `session_id`, `tag`, `search`, `validation_status`, label filters, `physical_object_id`) are unchanged.

For performance-sensitive browsing, Studio now uses paginated summary listing:

- `GET /api/takes/paged?limit=&offset=...`
- response: `items`, `limit`, `offset`, `has_more`, `next_offset`
- filters are equivalent to `GET /api/takes`
- selected take detail remains loaded via `GET /api/takes/{take_id}`

### Studio curation UX semantics

- Friendly names are rendered first; `take_id` remains visible as secondary/debug context.
- Compact chips support:
  - curation categories
  - quick reference marking (`empty belt`, `calibration ref`, `known object`, `golden`)
- Session type/reference/golden cues are surfaced in take cards for fast triage.
- A compact persistent curation-context block exposes active dataset/session/session-type/session-tags while browsing.
- Acquisition browsing thumbnails prioritize full-frame context (`contain`, aspect-preserving, letterbox acceptable) instead of aggressive object-centric crop.
- Curation context resolves from the **selected take** when present; filter controls remain a separate browse layer and are shown independently when active.

### Compatibility guarantees

These additions are metadata-only and preserve architecture invariants:

- raw take immutability (`data/incoming/<take_id>/` unchanged)
- many-runs-per-take model unchanged
- processing output layout unchanged
- pipeline execution contracts unchanged
- existing take IDs unchanged
- replay compatibility preserved

### Extensibility rationale

- Sidecar-first schema keeps migration cost low and allows future evolution toward stricter contracts without blocking current acquisition and debugging workflows.
- Session-level classification and take-level curation tags provide a clean bridge for future ML split/benchmark tooling without mixing those concerns into processing artifacts.
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

## Acquisition Grouping vs ML Grouping (Architectural Separation)

### Rationale

Sessions and ML datasets represent different concerns and should not share the same abstraction:

- Acquisition grouping (`Session`): capture run context, setup, calibration, operator workflow, and experiment run boundaries.
- ML grouping (`MLSet`): reusable training/evaluation cohorts, split assignment, benchmark sets, and task-specific views over takes.

Using one concept for both causes ambiguity and leakage-prone workflows. Sensor Studio therefore separates them explicitly.

### Target model

- `Dataset`: top-level project/collection (example: `bolas-2-5-1`).
- `Session`: acquisition context (examples: `morning_poc`, `evening_labeled_objects`, `conveyor_test_01`).
- `MLSet`: ML experiment/training grouping (examples: `balls_scrap_classifier_v1`, `diameter_regression_v1`, `benchmark_june_2026`, `holdout_set_v1`).

### Cardinality rules

- One take belongs to exactly one session.
- One take belongs to zero, one, or many ML sets.
- Session assignment is acquisition provenance.
- ML set membership is experiment curation.

### New entities

`MLSet`

```text
MLSet
- id
- dataset_id
- name
- description
- task_type
- created_at
- updated_at
- notes
```

Supported `task_type` values:

- `classification`
- `regression`
- `detection`
- `segmentation`
- `clustering`
- `benchmark`

`MLSetMembership`

```text
MLSetMembership
- ml_set_id
- take_id
- split
- physical_object_id
- include
- notes
- created_at
- updated_at
```

Supported `split` values:

- `train`
- `validation`
- `test`
- `holdout`
- `calibration`
- `unassigned`

### Persistence layout (filesystem)

ML sets are persisted under the dataset sidecar root:

```text
data/
  datasets/
    dataset_<dataset_id>/
      ml_sets/
        ml_set_<ml_set_id>/
          ml_set.json
          memberships.json
```

`ml_set.json` stores MLSet metadata.

`memberships.json` stores membership rows keyed by `take_id` semantics (no duplicates per MLSet).

### Take metadata shape (extended)

```json
{
  "take_id": "2026-05-25T183433_037",
  "dataset_id": "bolas-2-5-1",
  "session_id": "evening_labeled_objects",
  "ml_set_memberships": [
    {
      "ml_set_id": "balls_scrap_classifier_v1",
      "split": "train",
      "physical_object_id": "obj_0005"
    },
    {
      "ml_set_id": "diameter_regression_v1",
      "split": "validation",
      "physical_object_id": "obj_0005"
    }
  ]
}
```

### Leakage prevention strategy (object-level split integrity)

Split assignment must be object-aware:

- A physical object can have many takes.
- All takes from the same `physical_object_id` must stay in the same split within a given ML set.
- Split assignment is performed over object groups first, then expanded to takes.

This prevents train/test leakage for scenarios such as repeated captures of `obj_0005`.

### Session regrouping CLI remains acquisition-only

Existing command remains valid and intentionally acquisition-scoped:

- `python scripts/sensor_studio_cli.py acquisition update-take-session`

Semantics remain:

- dry-run by default, `--apply` required for writes
- preserves `dataset_id`; only `session_id` changes
- does not mutate raw files or processing outputs

This command should not be used to represent train/validation/test or ML classifier dataset membership.

Usage patterns:

- Explicit take IDs:
  - `python scripts/sensor_studio_cli.py acquisition update-take-session --data-dir data --session-id ball_good_84mm --take-id 2026-05-25T183433_037 --take-id 2026-05-25T183434_038 --dry-run`
- Newline take file:
  - `python scripts/sensor_studio_cli.py acquisition update-take-session --data-dir data --session-id ball_good_84mm --take-file labels/ball_good_84mm_takes.txt --dry-run`
- Create destination session + apply:
  - `python scripts/sensor_studio_cli.py acquisition update-take-session --data-dir data --session-id ball_good_84mm --session-name "Good ball 84 mm" --create-session --take-file labels/ball_good_84mm_takes.txt --apply`

Command behavior:

- Input union: repeated `--take-id` and `--take-file` are merged and deduplicated.
- Validation: prints requested/valid/missing summary and refuses when missing takes exist, unless `--allow-missing`.
- Dataset inference: inferred from take metadata; multi-dataset input requires explicit `--dataset-id`.
- Session creation: destination session is reused when present; created only with `--create-session`.
- Dry-run: default mode; prints proposed old/new session mapping and writes nothing.
- Apply mode: requires `--apply`; updates take sidecar metadata `session_id` only and writes audit log:
  - `data/runtime/logs/update_take_session_<timestamp>.json`

Immutability guarantees:

- No move/duplication/deletion of raw take folders under `data/incoming/<take_id>/`.
- No mutation of processed outputs under `data/processed/<take_id>/`.
- No mutation of labels/classes/annotations except session association in dataset sidecar metadata.

Why this helps supervised workflows:

- Allows fast regrouping into acquisition sessions aligned with operator curation passes.
- Supports labeling/reprocessing batches without recapturing data.
- Preserves acquisition provenance while ML grouping is handled separately through `MLSet` membership.

### Future ML workflow CLI architecture

- Create ML set:
  - `python scripts/sensor_studio_cli.py ml create-set --dataset-id bolas-2-5-1 --ml-set-id balls_scrap_classifier_v1 --name "Balls vs Scrap Classifier v1" --task-type classification`
- Add takes:
  - `python scripts/sensor_studio_cli.py ml add-takes --ml-set-id balls_scrap_classifier_v1 --take-file labels/balls_scrap_labeled_takes.txt --split unassigned`
- List ML set:
  - `python scripts/sensor_studio_cli.py ml list-set --ml-set-id balls_scrap_classifier_v1 --show-memberships`
- Assign splits:
  - `python scripts/sensor_studio_cli.py ml assign-splits`
- Export dataset:
  - `python scripts/sensor_studio_cli.py ml export-dataset`
- Export features:
  - `python scripts/sensor_studio_cli.py ml export-features`
- Train classifier:
  - `python scripts/sensor_studio_cli.py ml train-classifier`
- Evaluate model:
  - `python scripts/sensor_studio_cli.py ml evaluate`

MLSet lookup semantics:

- MLSet IDs are dataset-scoped, not globally unique.
- Commands such as `ml add-takes` and `ml list-set` resolve `--ml-set-id` as follows:
  - with `--dataset-id`: resolve only inside that dataset
  - without `--dataset-id`:
    - zero matches -> fail (`MLSet '<id>' not found.`)
    - one match -> auto-resolve and continue
    - multiple matches -> fail and require `--dataset-id`

`ml assign-splits` semantics:

- metadata-only operation over `MLSetMembership` rows
- default mode is dry-run
- apply mode requires `--apply`
- deterministic assignment from stable grouping + seeded shuffle
- ratio validation:
  - each ratio must be `>= 0`
  - `train + validation + test + holdout + calibration == 1.0`
- leakage-safe grouping:
  - with `--by-physical-object-id` (default), assignment unit is `physical_object_id`
  - all takes for the same physical object are forced into the same split
  - missing `physical_object_id` fails clearly with offending take IDs
- without `--by-physical-object-id`, assignment unit is `take_id`
- on apply, writes:
  - `data/runtime/logs/ml_assign_splits_<timestamp>.json`

`ml reprocess-set` semantics:

- metadata-aware batch processing over MLSet memberships
- supports optional split filtering (`--split` repeatable)
- default mode is dry-run; apply requires `--apply`
- uses explicit `pipeline_id` for all selected takes
- does not mutate MLSet/session labels/splits/raw acquisitions
- updates only processing outputs
- apply writes:
  - `data/runtime/logs/ml_reprocess_set_<timestamp>.json`

`ml export-features` semantics:

- deterministic one-row-per-take export (CSV)
- joins:
  - MLSet membership metadata (`ml_set_id`, `split`, `physical_object_id`)
  - take metadata (`dataset_id`, `session_id`, expected labels when present)
  - pipeline provenance (`pipeline_id`, `pipeline_run_id`, `pipeline_timestamp`)
  - flattened stable scalar numeric features from compatible pipeline results
- supports optional split filtering
- `--require-processed` enforces complete compatible processing coverage
- deterministic column ordering:
  - metadata/provenance columns first
  - feature columns sorted lexicographically

`ml import-manifest` semantics:

- preferred high-level ML curation entrypoint (CSV/JSON)
- each manifest row represents one physical object with one-or-many take IDs
- importer validates:
  - MLSet resolution
  - manifest schema
  - take existence
  - take dataset compatibility with MLSet dataset
  - conflicting physical-object assignment
- importer updates memberships idempotently (add-or-update, no duplicates)
- supported membership metadata updates:
  - `physical_object_id`
  - `split`
  - `include`
  - `notes`
  - `expected_label`, `expected_class`, `expected_subclass`
  - optional numeric measurements (`d1_mm`, `d2_mm`, `d3_mm`)
- default mode is dry-run; apply requires `--apply`
- apply writes:
  - `data/runtime/logs/ml_import_manifest_<timestamp>.json`

### Grinding balls example

- Dataset: `bolas-2-5-1`
- Session: `evening_labeled_objects` (acquisition context)
- ML set A: `balls_scrap_classifier_v1` with split=`train`
- ML set B: `diameter_regression_v1` with split=`validation`
- Shared `physical_object_id`: `obj_0005` across all takes of that ball

Result:

- Acquisition provenance stays stable in session metadata.
- ML experiments can reuse the same takes with independent tasks and splits.
- Leakage guard is enforced per ML set via object-level grouping.

Future path from this foundation:

- curate memberships in `MLSet`
- import manifest-driven object groupings/labels (`ml import-manifest`)
- assign leakage-safe splits using `physical_object_id` (`ml assign-splits`)
- reprocess target MLSet (`ml reprocess-set`)
- export features/datasets (`ml export-features`)
- train classifier
- evaluate model

## Repeatability Platform Additions (Additive)

This architecture now adds a repeatability-analysis path without changing take/run/session fundamentals.

Key additions:
- `physical_object_id` is now a first-class optional take metadata field and API filter.
- Studio can edit `physical_object_id` inline for curation workflows.
- New offline script: `scripts/analyze_feature_repeatability.py`.

Repeatability script behavior:
- groups repeated takes by `physical_object_id` (or fallback per-take when absent)
- computes per-feature statistics across repeated scans:
  - `mean`, `std`, `cv`, `min`, `max`, `outlier_count`, `missing_data_ratio`
- emits per-feature stability ranking and instability flags
- emits orientation-sensitivity heuristic classes:
  - `low_sensitivity`, `medium_sensitivity`, `high_sensitivity`
- emits correlation tables linking instability to acquisition-quality features when available

Outputs:
- `repeatability_summary.json`
- `repeatability_per_object_feature.csv`
- `repeatability_feature_stability.csv`
- `repeatability_correlations.csv`

All additions are metadata/statistics oriented and preserve:
- immutable raw acquisitions
- immutable run lineage semantics
- dataset/session/take separation

### ML export and repeatability diagnostics integration

`ml export-features` supports additive optional columns:

- `--include-diagnostics`
  - appends `diag_*` feature columns from `feature_vector.json`
- `--include-invalidity-flags`
  - appends `diagnostic_flag_count` and `diagnostic_flags`
- `--include-provenance-summary`
  - appends compact provenance summary fields (source-stage count/list and warning/invalid count)

Default export behavior is unchanged when these flags are omitted.

`analyze_feature_repeatability.py` supports additive optional diagnostics analysis:

- `--include-diagnostics`
- `--include-invalidity-flags`
- `--include-provenance-summaries`

These options add quality-association and provenance-validity summaries without changing baseline outputs.

## Explainable Rule Tuning (25D)

New utility script:

- `python scripts/tune_25d_rules.py`

Purpose:
- tune threshold parameters for the existing explainable 25D rule classifier
- keep deterministic, interpretable rule logic
- avoid black-box model replacement

Important distinction:
- rule tuning adjusts geometric thresholds in a fixed rule family
- ML training learns a new model representation

Object-safe evaluation:
- grouping is done by `physical_object_id`
- all takes from the same object stay in one split
- this prevents optimistic leakage when repeated captures exist

Hierarchical evaluation order:
1. detect obvious `SCRAP_METAL`
2. detect high-confidence `BALL_GOOD`
3. detect `BALL_SCRAP` degraded-ball region
4. fallback for unresolved cases

Script outputs:
- `best_rules.json`
- `tuning_report.json`
- `confusion_matrix.csv`
- `per_class_metrics.csv`
- `predictions.csv`
- `feature_ranges.json`

Operational usage:
- export features via `ml export-features` (optionally with diagnostics/provenance columns)
- tune with `tune_25d_rules.py`
- review confusion matrix + per-class metrics
- manually inspect/copy `best_rules.json` into classifier config review flow

### Selectable rule sets

Rule sets are external JSON configs under:

- `configs/classifiers/*.json`

This allows multiple explainable threshold variants to coexist:

- `builtin_default`
- tuned snapshots (for example `tuned_20260529`)
- client/demo variants

Reprocessing can optionally use a selected rule set via:

- `sensor_studio_cli.py ml reprocess-set --classifier-rules <path>`

If omitted, built-in rules remain active.

Rule set discovery from CLI:

- `sensor_studio_cli.py ml list-rule-sets`
- optional `--json` for machine-readable output
- optional `--classifier-id <id>` filter

`ml list-rule-sets` scans `configs/classifiers/*.json` and reports:

- rule set id (file stem)
- version
- classifier id
- description
- config path
- metadata (`dataset_id`, `ml_set_id`, `optimized_metric`, validation score when present)

Operational selection precedence:

- runtime override (`ml reprocess-set --classifier-rules <path>`)
- pipeline recipe classifier rule-set path
- environment default (`SENSOR_STUDIO_DEFAULT_RULE_SET`)
- built-in default rules

Active selection inspection:

- `sensor_studio_cli.py ml show-active-rule-set --pipeline-id mining_steel_ball_classification_25d`
- optional `--json` for automation/runtime integration

Immutability policy:

- rule-set config files are treated as immutable release artifacts
- do not overwrite tuned snapshots in place
- create versioned successors (example: `*_v2.json`) for threshold updates

## Studio vs Datasets Responsibility Split (2026-05-31)

A new top-level semantic workspace is introduced at `/datasets` to separate engineering execution from ML/data governance concerns.

### Top-level navigation ownership

Primary product navigation now follows:

- `Operations`
- `Studio`
- `Datasets`
- `Classifiers`
- `Runtime`
- `Calibration`
- `Diagnostics`

### Mental model boundary

- `Studio` answers: "What happened technically?"
- `Datasets` answers: "What do we know semantically?"

### Studio ownership (kept)

- take -> pipeline -> run -> stage execution workflows
- stage-native visualization and artifact inspection
- segmentation/geometry/debug overlays and calibration validation
- runtime engineering workflows and acquisition troubleshooting
- lightweight dataset/session selection only

### Datasets ownership (new)

- dataset explorer hierarchy: dataset -> sessions -> takes -> object annotations
- dataset dashboard: counts, validation coverage, class/split readiness indicators
- session view: acquisition conditions + sensor/conveyor/lighting/calibration context
- take browser for bulk semantic operations (validation/class assignment/tags)
- object review workspace reusing `object_annotations` contracts
- future ML set and split-management workflows with leakage diagnostics

### Explicit non-goals

This separation does **not** change:

- pipeline execution architecture
- runtime orchestration contracts
- stage artifact/overlay contracts
- take immutability or run immutability
- processing APIs/backends

### UX decomposition rationale

Keeping Studio focused on execution/debugging preserves responsiveness and stage-centric clarity for engineering users. Moving semantic curation into Datasets prevents sidebar overloading, improves discoverability for labeling/review/split tasks, and creates a scalable surface for active-learning and classifier lifecycle workflows.

### Incremental rollout path

- Phase 1: `/datasets` route, explorer/dashboard/session-take-object views, bulk semantic updates.
- Phase 2: explicit ML Set entities + split management + leakage and balance diagnostics.
- Phase 3: training orchestration linkage, classifier lineage, active-learning/disagreement queues, deployment feedback loops.

## Datasets Command Center UX Refinement (2026-05-31, Phase 2A/2B base)

The `/datasets` workspace was refactored from stacked dashboard sections into a multi-context command-center layout optimized for semantic operations.

### Workspace decomposition rationale

The page now follows a stable three-pane operating model:

- left: semantic explorer/filtering/navigation
- center: tabbed operational workspace
- right: contextual inspector

This reduces context switching, improves density, and prevents semantic metadata from competing with core table/review workflows.

### Tab ownership semantics

Center workspace tabs define explicit responsibility ownership:

- `Overview`: dataset health/readiness (counts, coverage, distribution, recent activity)
- `Takes`: canonical bulk semantic take management surface
- `Objects`: object annotation/review queues (contract-compatible placeholders included)
- `Labels`: taxonomy and usage organization
- `ML Sets`: composition/readiness placeholders for training datasets
- `Splits`: split-governance and leakage diagnostics placeholders

The previous independent stacked blocks (dashboard/session/take/object) are now contextual tab content.

### Inspector responsibilities

The right inspector now owns context metadata for selected entities:

- dataset metadata context
- session acquisition/calibration context
- take semantic context
- object annotation context
- ML set context (when active)
- split context (when active)

This keeps center panes focused on actions and review density rather than metadata narration.

### Semantic workflow direction

The refined flow prioritizes operational curation:

- filter in explorer
- act in tabbed workspace (bulk review/edit/govern)
- inspect context in right panel
- jump to Studio only for technical stage/debug detail

Subtle workflow links (`Open in Studio`, `Inspect pipeline outputs`, `Review segmentation`) preserve handoff without merging workflows.

### Dataset governance UX philosophy

The Datasets workspace is positioned as an industrial semantic governance surface, not a generic admin CRUD page:

- compact controls
- denser tables/cards
- progressive disclosure via tabs and inspector
- minimal prose, action-first layout

### Studio vs Datasets workflow separation

Boundary is reinforced at interaction level:

- `Studio`: execution/debugging/stage visualization and engineering diagnosis
- `Datasets`: labeling/review/semantic organization/ML curation/split governance

No backend processing contracts, pipeline execution architecture, or object annotation data contracts were redesigned.

## Datasets Command Center Refinement (2026-05-31, tab/reactivity pass)

### Tab workspace semantics

The center workspace now behaves as a true tabbed operational surface:

- compact horizontal tab strip
- active tab content anchored immediately below tabs
- reduced framing/vertical dead space
- Overview-owned KPI and summary composition (no detached dashboard region)

### Overview tab ownership

Overview now exclusively owns health/readiness summaries:

- KPI strip
- class/session distributions
- split readiness
- recent activity

No pre-tab dashboard region is used.

### Takes rendering/reactivity fix

The takes rendering issue was corrected by separating data acquisition from local semantic filtering:

- source list fetched into `rawTakes` from current dataset/session/server-side filters
- UI-visible list derived reactively from `rawTakes` via `useMemo` for split/class/calibration filters
- selection validity reconciled when derived list changes
- explicit loading and empty states added to Takes tab

Result:

- takes list updates immediately and reliably on dataset/session/filter changes
- no stale memoization path blocks rendering
- explicit empty state text: `No takes match the current filters.`

### Inspector action hierarchy

Inspector actions now have clear priority and availability rules:

- primary: `Open selected dataset in Studio`
- secondary: `Inspect pipeline outputs`, `Review segmentation`
- actions disable when required target context is missing
- disabled states include inline `title` reason

Metadata snapshot and actions are visually separated.

### Operational density rationale

Density was increased with tighter spacing and stronger interaction anchoring:

- tighter tab/panel spacing
- compact KPI cards and tables
- sticky table header retained
- clearer row hover + selected row styling
- selection count surfaced in bulk action bar

### Workspace composition refinement

Placeholder tabs (Labels / ML Sets / Splits) were reduced to compact roadmap-style content to avoid empty-card/admin-dashboard feel while preserving future expansion direction.

## Paginated Takes Loading In Datasets (2026-05-31)

The `/datasets` Takes tab now uses paginated summary loading via `GET /api/takes/paged` instead of full take hydration.

### Why paginated summaries

Datasets is a semantic curation workspace and should remain responsive on large datasets. It now loads lightweight `TakeSummary` pages (`limit=50`) and avoids hydrating all takes/details upfront.

### Loading model

- initial request: `limit=50`, `offset=0`
- server filters passed: `dataset_id`, `session_id`, `validation_status`, `search`, `tag`
- client state tracks: `pagedTakes`, `offset`, `hasMore`, `isInitialLoading`, `isLoadingMore`
- reset pagination on dataset/session/search/tag/validation changes
- append results with explicit `Load more` action
- client-side filters remain for non-paged params (split, expected class, calibration-only)

### Rendering semantics

- first load uses compact row-height skeleton placeholders
- first page renders immediately when returned
- empty result text after load: `No takes match the current filters.`
- loaded/total state is surfaced (`Showing X of Y takes` where total is available)

### Selection semantics

- selection is reconciled against currently visible/loaded rows
- selection is reset when base server-filter context changes
- selection does not persist across non-visible rows

### Separation from Studio detail hydration

Datasets remains summary-first and only fetches take detail for explicitly selected takes/object review. Studio continues to own heavy technical detail and stage-level debugging workflows.

## Compact Thumbnails In Datasets Takes Table (2026-05-31)

The `/datasets` Takes tab now includes a compact thumbnail column for faster semantic scanning of large paginated lists.

### Thumbnail semantics

Each row resolves a lightweight preview using summary metadata only, with this priority order:

1. `take.thumbnail_path` (preferred)
2. source-like preview asset path (RGB/reflectance/image-like summary asset)
3. heightmap-like preview asset path
4. neutral modality-aware placeholder (`RGB`, `HMP`, `PCD`, fallback `TAKE`)

No per-row take detail hydration is performed.

### Paginated thumbnail loading rationale

Thumbnails are loaded only for currently rendered rows in the paginated table and preserve summary-first behavior:

- fixed dimensions to avoid layout shift
- `loading="lazy"` and async decoding
- table rows render immediately; images resolve progressively

This keeps `/datasets` responsive for large sets while improving visual review speed.

### Lightweight preview philosophy

Datasets uses small visual evidence for semantic browsing, not full technical inspection:

- compact thumbnail as row anchor
- take id remains semantic identifier
- lightweight placeholder when no preview exists

### Relationship with Studio deep inspection

Datasets thumbnails support quick recognition and navigation. Deep artifact/stage debugging remains in Studio/Take detail views. This preserves separation:

- Datasets: semantic browsing/curation
- Studio: execution/stage-level inspection

## Selected Row And Inspector Thumbnail Sync (2026-05-31)

Refinement to `/datasets` take review UX adds explicit selection semantics and synchronized preview behavior.

### Row selection semantics

- clicking a take row selects that take for semantic review context
- bulk checkbox selection is preserved and does not trigger row-navigation side effects
- inspector context updates immediately from selected row state

### Thumbnail interaction semantics

- thumbnail and take-id remain explicit interactive affordances
- thumbnail hover title: `Open in Studio`
- pointer affordance remains on explicit interactive elements (thumbnail/link/actions), not on the entire row

### Inspector thumbnail synchronization

Inspector preview now reuses the same resolved preview model used by table rows (same URL/placeholder/modality resolution path), preventing mismatch where row thumbnail exists but inspector shows placeholder.

### Compact modality badge semantics

Each thumbnail may overlay a compact modality badge when summary modalities are available:

- `RGB`
- `HMP`
- `PCD`
- `2.5D`

This is lightweight row-context only and does not trigger extra fetches.

### Performance guarantees

- no per-row take-detail hydration
- no full artifact list fetch for thumbnails
- lazy image loading with fixed dimensions
- paginated summary loading model remains unchanged

## Large-Scale Semantic Curation Expansion (2026-05-31, Phase 3A-3F UX)

### Dataset vs ML Set semantics

The workspace now reinforces explicit ownership separation:

- `Dataset`: operational semantic organization for captured takes and sessions.
- `ML Set`: curated semantic membership for training/evaluation composition.

Datasets do not collapse into ML Sets, and ML Sets do not own raw take storage.

### Bulk semantic organization philosophy

Takes tab now surfaces high-scale curation actions around row selection:

- move selected takes to dataset/session
- add/remove selected takes to/from ML set membership (additive membership semantics)
- split assignment and validation workflows
- add/remove/replace tag flows

These actions are selection-first and optimized for many-row curation loops.

### Hierarchy navigation rationale

Dataset Explorer now includes a lightweight hierarchy browser (dataset -> sessions) with compact session-level status, while preserving filter controls. This separates navigation context from filtering context for faster scanning.

### Semantic preview drawer semantics

Inspector is expanded from metadata-only panel into semantic review drawer sections:

- preview
- metadata/context
- processing summary
- quick semantic actions
- future curation architecture placeholders

It remains non-Studio: no stage-debug controls are introduced.

### Saved collection architecture

Saved filters/smart collections are stored locally as reusable semantic views and can re-apply full filter context quickly. This provides a queue-like operational pattern and future bridge to active-learning collections.

### Future active-learning direction

Explicit placeholders were added for:

- uncertainty/disagreement/outlier queues
- similarity and embedding neighborhood browsing
- anomaly/disagreement clustering
- feature-space exploration linkage

These remain additive UX scaffolding and do not merge Feature Analytics or Classifiers workflows.

### Keyboard workflow extensibility

A keyboard command map foundation was introduced behind a disabled flag for future scalable curation shortcuts (`j/k/x/v/r/t/s`).

### Semantic governance philosophy

The `/datasets` direction is summary-first, incremental, and curation-oriented:

- paginated takes
- no eager per-row detail hydration
- semantic action density over debugging depth
- Studio remains execution/debug authority

### Phase 3A action refinements

Additional bulk workflow refinements in Takes:

- inline dataset creation from curation workflow
- inline session creation from curation workflow
- explicit future placeholder action for dataset copy semantics (`Copy to dataset (planned)`)
- expanded validation states are accepted as semantic status values (`golden_sample`, `benchmark_approved`) without changing backend processing contracts

## Phase 4 Semantic Governance Evolution (2026-05-31)

### Temporal filtering semantics

Datasets now supports operational temporal filtering with:

- explicit `From`/`To` date range
- quick presets (`Today`, `Last 24h`, `Last 7d`, `Last 30d`, `This week`, `This shift` placeholder)
- pagination reset/reload when temporal filters change
- saved collections persisting temporal filters and preset state

Date filters are applied in summary-first browsing flow and integrated with paged loading requests.

### Governance queue philosophy

Queue presets are introduced as operational semantic workflows (not only static saved filters), for example:

- unreviewed captures
- missing labels
- calibration review
- benchmark approval
- failed processing

This establishes a human-in-the-loop curation rhythm and future bridge to active-learning queues.

### Semantic review workflows

The take-review flow now emphasizes high-throughput governance:

- sticky multi-select action bar
- bulk organization actions (dataset/session/ML-set/split/tag/validation)
- queue preset activation
- quick review actions for "next unresolved" / "next unreviewed" / "next missing label"

### Compare-mode semantics

A lightweight semantic compare mode (2–4 selected takes) is available in Takes:

- side-by-side thumbnail and semantic metadata comparison
- validation/split/processing/object/session/calibration context
- no stage/debug artifact exploration

### Dataset health rationale

Overview now includes lightweight governance scoring and coverage summaries:

- validation coverage
- split coverage
- processing coverage
- calibration coverage
- aggregated dataset health percentage (heuristic, intentionally simple)

### ML-set lineage direction

ML Set tab explicitly preserves governance separation and direction:

- datasets own acquisition-semantic organization
- ML sets own curated membership/composition semantics
- placeholders for inclusion/exclusion rules, balancing constraints, split ownership, export and classifier compatibility

### Feature Analytics interoperability boundaries

Soft linkage is provided via "Open selection in Feature Analytics" actions while preserving app boundaries:

- Datasets: semantic governance and operational curation
- Feature Analytics: feature-space understanding and anomaly analysis

No workflow merge is introduced.

### Future object-governance direction

Architecture placeholders maintain future path for object-centric governance without changing take ownership:

- object-level datasets / ML sets / splits / validation (future)
- current invariant preserved: take remains acquisition container

### Keyboard workflow extensibility

A keyboard command map foundation exists behind a disabled feature flag (`j/k/x/v/r/t/s`) to support future high-throughput review loops without forcing immediate behavior changes.

### Semantic governance architecture evolution

Phase 4 continues the summary-first, paginated, large-scale pattern:

- incremental take loading
- reactive governance filtering
- compact visual review controls
- no eager detail hydration
- explicit separation across Datasets / Studio / Feature Analytics / Classifiers

## Selection, Count, And Pagination Semantics Refinement (2026-05-31)

### Visible vs filtered selection semantics

Takes selection now uses explicit scope semantics:

- `selectionScope = visible`:
  - selection applies only to currently loaded/visible rows
  - `selectedTakeIds` tracks selected loaded ids
- `selectionScope = filtered`:
  - selection conceptually represents all takes matching active filters
  - `excludedTakeIds` tracks loaded rows manually unselected from filtered scope
  - does not fetch all matching ids eagerly

Controls include:

- `Select visible`
- `Clear selection`
- `Select all matching filters`

Selection messaging is scope-aware (`0 selected`, `N visible selected`, `All X filtered takes selected`, `All X filtered takes selected · Y excluded`).

### Filtered count semantics

Takes count display is filter-aware:

- filtered view: `Showing loaded of filtered · total in dataset`
- unfiltered view: `Showing loaded of total takes`

Explorer header is also filter-aware (`filtered / total`) when filters are active.

### Pagination terminal state

Load more state is explicit and safe:

- loading: `Loading more...`
- more pages: `Load more`
- terminal: `All filtered takes loaded`

No active load-more action remains once `hasMore=false`.

### Filter-change selection reset semantics

When filters change, the page resets selection and pagination context:

- clear `selectedTakeIds`
- clear `excludedTakeIds`
- reset `selectionScope` to `visible`
- reload first page

This prevents stale selection scope from leaking across filter contexts.

### Backend requirement for filter-wide bulk actions

Filter-wide bulk mutation is intentionally guarded when backend filter-bulk mutation APIs are unavailable:

- filtered-scope selection can be represented in UI
- bulk action execution remains disabled/guarded with explicit messaging
- UI does not silently apply filtered-scope actions only to loaded rows

This preserves safety and prevents false assumptions in large-scale governance workflows.

## Filtered Counts, Selection Scope, And Pagination Semantics (2026-05-31)

### filtered_count vs loaded item count

`/api/takes/paged` now returns authoritative count fields that are independent of the current page size:

- `items`: current page rows only
- `filtered_count`: total rows matching active filters (across all pages)
- `total_count`: total rows in selected dataset scope before active filters

The UI no longer infers filtered totals from loaded rows.

### First-page count availability

Counts are available from the first paged response and immediately power:

- filtered/total count labels
- filtered-scope selection messaging
- bulk-selection scope text

No full-id hydration or full dataset loading is required.

### Visible vs filtered selection scope

Takes selection is explicit and safe:

- `visible` scope: selected loaded rows only (`selectedTakeIds`)
- `filtered` scope: all filtered rows conceptually selected; manual unchecks tracked as `excludedTakeIds`

Filter-wide actions are guarded when backend filter-bulk mutation support is unavailable.

### Filter-change reset semantics

When filters change, selection and paging context reset atomically:

- `selectionScope -> visible`
- clear `selectedTakeIds`
- clear `excludedTakeIds`
- reload first page

This avoids stale cross-filter selection ambiguity.

### Terminal pagination state

`has_more` drives terminal state directly:

- active: `Load more`
- loading: `Loading more...`
- terminal: `All filtered takes loaded`

No active load-more action remains at terminal state.

### Density/layout refinements

To reduce command-center clutter while preserving throughput:

- compact horizontal tabs with low-height active state
- queue cards compressed into compact chips/counters
- primary bulk actions kept visible; secondary actions moved into `More actions`
- sidebar sections made collapsible (hierarchy / filters / needs attention / saved collections)
- inspector primary Studio action label is context-aware (selected take vs selected dataset)
- table row spacing tightened while preserving thumbnail readability

## Overview KPI Authoritative Count Semantics (2026-05-31)

### KPI source of truth

Overview and governance counters in `/datasets` are now sourced from backend paging metadata, not loaded rows:

- `filtered_count`
- `total_count`
- `summary_counts`

This prevents KPI drift on large datasets where only the first page is loaded.

### `/api/takes/paged` summary contract

Paged responses now carry governance-safe aggregates:

- `summary_counts.validation.validated`
- `summary_counts.validation.unreviewed`
- `summary_counts.validation.rejected`
- `summary_counts.validation.needs_review`
- `summary_counts.validation.golden_sample`
- `summary_counts.validation.benchmark_approved`
- `summary_counts.missing_labels`
- `summary_counts.missing_split`
- `summary_counts.missing_calibration`
- `summary_counts.processing_failed`
- `summary_counts.processing_incomplete`
- `summary_counts.no_objects_detected`

### Overview TAKES semantics

- if filters are active: `TAKES = filtered_count`
- if no filters are active: `TAKES = total_count`

`pagedTakes.length` is never used as the authoritative governance total.

### Queue and sidebar count semantics

Queue chips and overview KPI cards prefer `summary_counts` when available. Loaded-row fallbacks are only used if backend summary metadata is unavailable.

### Pagination invariants

Loading additional pages appends `items` only; it does not redefine filtered governance totals. Count semantics remain stable until filters change and page zero is requested again.

## Command-Center Density And Toolbar Ergonomics (2026-05-31)

### Compact unified takes toolbar

The Takes workspace now uses one compact horizontal toolbar that combines:

- selection scope/status text
- count context (`showing X of Y`)
- selection controls (`Select visible`, `Clear selection`, `Select all matching filters`)
- primary bulk actions (`Validated`, `Needs review`, `Rejected`, `Add tags`, `Split assignment`)
- `More` menu for secondary actions

Selection safety semantics are unchanged: visible-vs-filtered scope remains explicit, and filter-wide mutation remains guarded when backend support is unavailable.

### Primary vs secondary bulk actions

Primary actions stay visible for fast repeated review workflows. Secondary actions moved to `More`:

- move/create dataset
- move/create session
- add/remove ML set
- remove/replace tags
- expected class
- golden sample / benchmark approved
- archive / restore
- copy-to-dataset placeholder

### Queue chip density model

Queue controls were reduced from card-like elements to compact pill chips with inline counts. They remain clickable filter presets and wrap only when needed on narrow viewports.

### Tab/workspace whitespace reduction

Workspace density was tightened so active content starts immediately below tabs:

- smaller command-center outer padding
- reduced headline bottom spacing
- tighter center-column and tab-panel gaps
- compact tab bar/button heights

### Inspector empty-state behavior

When no take is selected, inspector preview text now communicates the intended workflow:

- `Select a take to review metadata, labels, split, preview, and semantic actions.`

Dataset-level Studio action remains available; take-specific actions remain disabled with lower visual emphasis.

### Table action copy refinement

Per-row action label changed from verbose `Inspect pipeline outputs` to compact `Inspect` with tooltip clarifying Studio pipeline-output inspection intent.

## Toolbar Density, Sidebar Overflow, And Modal Bulk Workflows (2026-05-31)

### Compact toolbar container rules

The Takes command toolbar remains a single compact flex row with wrap support and no oversized parent framing:

- removed extra vertical padding/margins from toolbar wrappers
- no fixed tall container behavior around disabled controls
- compact button heights with stable inline layout
- table starts immediately after compact queue + toolbar sections

### Sidebar overflow and long-name handling

Dataset Explorer now prevents horizontal overflow and remains usable with long names:

- `overflow-x: hidden`, `min-width: 0`, `max-width: 100%` on sidebar shell
- controls constrained to container width
- hierarchy header title truncates safely with ellipsis
- session node primary text clamps to two lines and metadata stays compact
- helper buttons (for example `Hide`) are non-expanding and do not force horizontal scroll

### Modal-based bulk workflows replacing prompt()

Prompt-based bulk organization flows were replaced with selector-driven modals:

- Move to Dataset: dataset selector + optional session selector
- Move to Session: session selector
- Add to ML Set: ML set selector + additive-membership note
- Split assignment: split selector
- Tag management: mode (`add/remove/replace`) + tag input

All modals show scope-aware summary (`selected` vs `all filtered`) and preserve current selection semantics.

### Selector-based organization semantics

Bulk organization actions now use preloaded selector data where possible:

- datasets from existing datasets state
- sessions from selected dataset state, with on-demand session load for alternate dataset in modal
- ML sets from `api.mlDatasets()`

No raw id typing is required for standard workflows.

### Filter-wide guard behavior in modals

When `selectionScope === "filtered"` and backend filter-wide mutation is unsupported:

- modal shows explicit warning: `Filter-wide bulk updates require backend support.`
- confirm action is disabled
- visible-selection semantics are unchanged

This keeps large-scale curation safe and explicit.

## Center Workspace Non-Stretch Layout Semantics (2026-06-01)

The `/datasets` 3-column command-center layout now explicitly prevents center-column vertical stretching caused by taller sidebars.

### Grid alignment rule

`datasets-grid` now uses non-stretch cross-axis semantics:

- `align-items: start`
- `grid-auto-rows: max-content`
- removed viewport-driven min-height stretching (`min-height: 0`)

This allows left/right panels to be independently tall while center content keeps intrinsic height.

### Center workspace sizing rule

`datasets-center` now opts into content-owned height:

- `align-self: start`
- `align-content: start`
- `height: auto`
- `min-height: 0`

No center fill/stretch behavior is used.

### Tab bar sizing semantics

`datasets-tabs` now explicitly stays compact and non-stretched:

- `align-self: start`
- `height: auto`
- compact `min-height` and padding retained

Tab bar width can remain full-width, but height is owned by tab controls only.

### Active-tab-only layout ownership

`datasets-tab-panel` is constrained to content-owned height:

- `align-self: start`
- `align-content: start`
- `height: auto`

Because only active tab content is rendered in React, inactive tabs do not reserve vertical space.

## Filter-Wide Bulk Metadata Update Semantics (2026-06-01)

### Bulk operation modes

`/datasets` now supports backend bulk metadata mutation through `POST /api/takes/bulk-metadata` with two targeting modes:

- `mode: "ids"`: explicit visible selection (`take_ids`)
- `mode: "filter"`: backend-resolved filtered scope (`filters`) with `exclude_take_ids`

This avoids frontend take-id hydration for large filtered sets.

### Filter-wide safety and confirmation

For filter-wide actions, frontend performs a dry-run request first and surfaces authoritative count confirmation:

- dry-run returns `affected_count` before apply
- modal shows: `This will move N filtered takes.`
- confirm CTA is count-aware (`Move N takes`)
- confirm remains disabled while counting or when required fields are missing

### Exclude behavior

When user selects `all filtered` and manually unchecks rows, unchecks are preserved as `exclude_take_ids` and respected server-side in filter mode.

### Current prioritized workflow

Enabled first-class workflow:

- Move to session for visible selection and filter-wide selection
- Optional inline create-session path in modal
- Date-range-driven default session naming:
  - same day: `Session YYYY-MM-DD`
  - range: `Session YYYY-MM-DD to YYYY-MM-DD`

After success:

- selection is cleared
- `selectionScope` resets to `visible`
- sessions/takes are reloaded
- filtered counts and summary counts refresh via paged reload

### Response contract

Bulk endpoint returns operational result stats:

- `matched_count`
- `affected_count`
- `skipped_count`
- `failed_count`
- `failed_ids` (sample)

### Governance boundaries preserved

No Studio behavior changes, no full take-detail hydration, and paginated summary-first browsing remains the default interaction model.

## Full Filter-Wide Bulk Support Matrix (2026-06-01)

### Enabled filter-wide actions (backend + UI)

`/datasets` now enables filter-wide execution for all actions currently supported by `POST /api/takes/bulk-metadata`:

- Move to session
- Move to dataset
- Add to ML set
- Remove from ML set
- Assign split
- Tag add / remove / replace
- Validation updates (`valid`, `needs_review`, `invalid`, `golden_sample`, `benchmark_approved`)
- Set expected class

No supported action remains disabled solely because `selectionScope === "filtered"`.

### Execution modes

Each action executes through the same dual-mode pathway:

- visible selection: `mode="ids"` + `take_ids`
- filtered selection: `mode="filter"` + active filter object + `exclude_take_ids`

The frontend never loads all matching IDs for filter-wide operations.

### Dry-run and confirmation flow

All bulk modals use dry-run before apply:

- `dry_run: true` call retrieves authoritative `affected_count`
- modal shows affected-count summary
- confirm CTA is count-specific (for example `Move 151 takes`, `Assign split to 151 takes`, `Apply tags to 151 takes`)

### Exclude semantics

When user selects all filtered and manually unchecks rows, `exclude_take_ids` is passed to backend in filter mode and must be respected. Operations do not silently re-include excluded rows.

### Action-specific modal semantics

- Move to session: session selector, optional create-session inline fields (name + notes), date-range default naming
- Move to dataset: dataset/session selectors + placeholders for create dataset/session
- ML set: action mode (`add`/`remove`) + ML set selector, additive-membership note
- Split: split selector
- Tags: mode (`add`/`remove`/`replace`) + tag input
- Validation: target state selector
- Expected class: class input

### Post-apply behavior

After successful bulk apply:

- clear selection
- reset `selectionScope` to `visible`
- clear exclusions
- reload paged takes to refresh `filtered_count`, `total_count`, and `summary_counts`
- refresh sessions when organization operations affect session context

### Guardrails

Unsupported/placeholder workflows (for example copy-to-dataset) remain explicitly disabled and must not silently degrade to visible-only behavior.

## Create-ML-Set From Filtered Selection (2026-06-01)

### Workflow support in Add to ML Set modal

Add-to-ML-Set now supports two paths in the same bulk modal:

- add/remove membership for an existing ML set
- create a new ML set and immediately add the selected scope membership

Selection scope behavior is preserved for both visible and filtered selection modes.

### Dry-run and affected-count confirmation

Before apply, modal runs dry-run and shows authoritative affected counts. Confirm CTA is count-specific and scope-aware, including create-and-add flow:

- `Create ML Set and add N takes`
- `Add N takes`
- `Remove N takes`

Filtered mode always uses backend filter resolution and `exclude_take_ids`; it never degrades to visible-only behavior.

### Dataset-scoped ML set persistence

ML set metadata is stored under dataset-sidecar storage:

- `data/datasets/dataset_<id>/ml_sets/ml_set_<ml_set_id>/ml_set.json`
- `data/datasets/dataset_<id>/ml_sets/ml_set_<ml_set_id>/memberships.json`

Creation enriches metadata with membership + semantics blocks for governance context:

- membership mode (`ids` or `filter_snapshot`)
- filter snapshot + excludes for filter-origin memberships
- semantics source (`manual_bulk_selection`), validation requirements, split-strategy placeholder, notes

### Membership modes and ownership boundary

Dataset vs ML set boundary remains explicit:

- Datasets own acquisition/take organization
- ML sets store curated membership metadata and membership criteria/snapshots
- ML sets do not own raw take storage

### API additions

Added dataset-scoped ML set API surface:

- `GET /api/datasets/{dataset_id}/ml-sets`
- `POST /api/datasets/{dataset_id}/ml-sets`
- `POST /api/datasets/{dataset_id}/ml-sets/{ml_set_id}/members`

Members endpoint supports `ids` and `filter` modes with `exclude_take_ids`, plus dry-run.

### ML Sets tab visibility

ML Sets tab now lists dataset ML sets with operational summary columns:

- name
- source dataset
- member count
- membership mode
- created_at
- readiness placeholder

## Entity Detail Drawer Architecture (Phase: Dataset Session)

- Added additive non-modal right-side `EntityDetailDrawer` as the canonical entity management surface in Datasets.
- First concrete implementation is `DatasetSessionDrawer`, opened from session hierarchy cards and session links in Takes.
- Drawer preserves list/table context behind it, supports keyboard dismiss (`Esc`), and scrolls independently.
- Drawer composition is reusable via shared primitives:
  - `EntityDetailDrawer`
  - `EntitySummaryCards`
  - `EntityStatisticsGrid`
  - `EntityMetadataForm`
  - `EntityExportActions`

### Dataset Session Drawer Semantics

- Header: session name, dataset name, created timestamp.
- Summary: take totals, reviewed/unreviewed, missing labels, split distribution, processing-family summary, annotation totals, last acquisition timestamp, calibration assignment.
- Editable metadata: name, description, notes, tags, acquisition type/session type, calibration id, sensor/conveyor/lighting metadata JSON.
- Session takes section: compact paginated table (thumb, take id/name, expected class, validation, processing, inspect link).
- Footer actions:
  - `Save`
  - `Archive session` (placeholder)
  - `Restore session` (placeholder)
  - `Open in Studio filtered to session`
  - `Create ML set from session` (placeholder)

### Export Semantics

- Added deterministic session export endpoint:
  - `GET /api/dataset-sessions/{session_id}/export?dataset_id=...`
- Export payload contract:
  - `export_type`, `schema_version`, `exported_at`
  - `dataset`, `session`, `summary`
  - `takes[]` sorted deterministically by `(created_at, take_id)`
- Export includes metadata/references and semantic state, not duplicated binaries.

### Summary Endpoint for Drawer

- Added `GET /api/dataset-sessions/{session_id}/summary?dataset_id=...`.
- Provides session-scoped aggregated governance/curation metrics used by the drawer.
- Built compositionally from existing DatasetService + paged/list take contracts.

### Boundary Rationale

- Datasets drawer remains semantic-governance oriented (metadata, curation, organization, export).
- Studio remains execution/debugging oriented; drawer links out to Studio rather than embedding Studio workflows.

## Dataset Session Drawer Refinement (Session-Scoped Correctness + Density)

- Enforced **canonical session ownership invariant** for drawer summary/export:
  - session aggregation must match `dataset_id + canonical session_id` from dataset-managed metadata (`experiment_session_id`), never display-name matching.
- Added deterministic consistency rule:
  - `drawer summary total_takes == export takes.length` for the selected session context.
- Added backend validation test (`tests/test_dataset_session_export.py`) covering:
  - session summary count correctness per selected session
  - export count parity with session summary
  - session switch changes summary totals deterministically

### Drawer Density Guidelines

- `EntityDetailDrawer` default compact width target: ~500px (responsive max-width).
- Optional wide mode exists for future large payload entity drawers.
- Header/footer are sticky; body scrolls independently.
- Session drawer keeps high-signal chips in header (take count, reviewed %, acquisition type, calibration link state).

### Session Drawer UX Rules

- Summary cards use compact governance metrics (takes, reviewed %, missing labels, coverage metrics).
- Detailed processing family breakdown moved to collapsible diagnostics section.
- Advanced JSON metadata fields are collapsed under “Advanced metadata”.
- JSON fields are validated before save; invalid JSON blocks save with inline error.
- Footer action hierarchy:
  - Primary: Save changes
  - Secondary: Export JSON, Open in Studio filtered to session
  - De-emphasized placeholders: Archive/Restore/Create ML set
- Export filename convention:
  - `dataset_session_<session_id>_export.json`

## Human-In-The-Loop ML Set Ingestion Wizard

Added a reusable two-layer ingestion architecture for semi-structured operator tables.

### Layer A: Human Reconciliation Wizard

- New reusable wizard shell and step components:
  - `MLSetIngestionWizard.tsx`
  - `WizardStepLayout.tsx`
  - `TablePreviewStep.tsx`
  - `SessionReconciliationStep.tsx`
  - `RangeExpansionStep.tsx`
  - `LabelNormalizationStep.tsx`
  - `AmbiguityResolutionStep.tsx`
  - `PhysicalObjectGroupingStep.tsx`
  - `MLPolicyConfigurationStep.tsx`
  - `ManifestGenerationStep.tsx`
- Wizard is additive in Datasets -> ML Sets and preserves existing navigation/layout contracts.
- Wizard run state is persisted under `data/ml_ingestion_runs/<run_id>/` for resume/reopen workflows.

### Layer B: Deterministic Materialization

- Canonical manifest is generated first and then consumed by deterministic materialization.
- Output ML set artifacts are immutable references under `data/ml_sets/<ml_set_id>/`.
- Raw acquisition binaries are not duplicated.

### Semantic Reconciliation Semantics

- Ingestion parsing supports `csv`, `tsv`, and pasted tabular text (`xlsx` returns explicit planned/not-yet-supported error).
- Session/take matching uses canonical dataset/session/take identity via DatasetService metadata.
- Abbreviated refs are resolved by deterministic suffix matching; ambiguities are surfaced (never silently guessed).
- Range expansion supports explicit columns (`from/to`, `start/end`, `first/last`) and inline ranges (`139...143`).

### Label Schema + Normalization

- Versioned schema id: `mining_balls_labels_v1`.
- Required classes/superclasses are represented in normalization policy with review-safe fallbacks.
- `Cubo` maps to `CALIBRATION_CUBE` / `REFERENCE_OBJECT`.
- Empty rows/empty labels map to review-safe behavior (`REVIEW_REQUIRED`) unless explicitly labeled `EMPTY_SCENE`.
- Labels ending with `?` are marked uncertain and `needs_review=true`.

### Physical Object Grouping + Split Policy

- Physical object groups are deterministic (`object_000001`, ...), keyed by source row index.
- Split policy materializes by `physical_object_id` (never by take id), preventing leakage across train/val/test.
- Validation report flags any cross-split object leakage.

### Unlabeled + Review Policy

- Default policy excludes unlabeled/review-required rows from supervised splits.
- They are retained in diagnostics (`unlabeled_pool`) for future review/active-learning queues.
- Unlabeled data is never silently converted into negatives.

### Canonical Manifest Role

- Canonical manifest is the authoritative semantic artifact and includes:
  - take/session/object grouping keys
  - raw operator label + normalized class/superclass
  - confidence/review flags
  - range-resolution provenance
  - schema version
- Manifest files written to `data/ml_ingestion_runs/<run_id>/canonical_manifest/` include CSV/JSONL + diagnostics and validation report.

## ML Set Detail Drawer Architecture

- ML Sets now open a reusable right-side detail drawer rather than behaving as a plain member-count row.
- The drawer reuses `EntityDetailDrawer` and becomes the canonical ML-set governance surface for experiment readiness.

### ML Set entity semantics

- Dataset: owns raw semantic organization and acquisition/session structure.
- ML Set: owns curated semantic subset semantics for reproducible ML workflows.
- ML Set detail therefore emphasizes composition, provenance, split integrity, readiness, and exportability rather than raw take browsing alone.

### Detail sections

- Identity + provenance
- Readiness / health
- Class and superclass distribution
- Representative samples
- Split visualization and object-group leakage diagnostics
- Source session coverage
- Membership definition / inclusion rules
- Validation warnings
- Derived tasks
- Training compatibility
- Export / reproducibility actions

### Split integrity rationale

- ML-set integrity is evaluated by `physical_object_id`, not only by `take_id`.
- Leakage warning is raised when the same physical object spans multiple splits.
- Source session composition by split is surfaced to reveal acquisition bias and train/validation imbalance.

### Reproducibility principles

- Membership mode, filter snapshot, exclusion list, and semantics are shown directly in the drawer.
- Export actions expose deterministic manifest/split/schema/snapshot artifacts.
- The drawer is designed so lineage can later extend toward: Dataset -> Ingestion Run -> Canonical Manifest -> ML Set -> Experiment.

## Physical Object Semantic Layer

- Added a first-class `PhysicalObject` semantic entity distinct from:
  - immutable takes/acquisitions
  - pipeline-generated detected objects
  - ML-set memberships
- `PhysicalObject` represents a real-world entity observed across one or more takes.

### Core distinction

- `PhysicalObject`: semantic real-world entity (`object_000005`, `BALL_GOOD`).
- `DetectedObject`: per-take/per-run pipeline candidate or segmentation artifact.
- Takes remain immutable observations; physical-object linkage is sidecar semantic metadata.

### Registry + linkage

- Added filesystem-backed registry under dataset space for physical objects.
- Take metadata continues to support a primary `physical_object_id` linkage.
- Ingestion wizard reconciliation now syncs grouped operator rows into the physical-object registry automatically.

### Dataset UX

- Datasets now include a first-class `Physical Objects` tab.
- Clicking a row opens a `PhysicalObjectDetailDrawer` with:
  - identity/labels
  - dimensions
  - source provenance
  - observed takes
  - repeatability summary
  - warnings/review context

### ML governance rationale

- Physical Objects are the canonical split unit for ML safety.
- Split integrity and leakage checks remain object-centric (`physical_object_id`) instead of take-centric.
