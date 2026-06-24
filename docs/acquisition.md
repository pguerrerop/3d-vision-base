# Acquisition

How the acquisition process works in v1 and how to extend it for live sensor modes.

## Package layout

```
vision_3d_acquisition/
  contracts/metadata.py      # Pydantic models (no I/O)
  storage/publisher.py       # Filesystem queue publish
  acquisition/
    base.py                  # AcquisitionBase ABC
    offline_ply.py           # Offline .ply ingest
    publisher.py             # Re-export (backward compat)
  processing/                # Future: segmentation, fitting, classification
  state/                     # Future: status managers
  utils/
    ids.py, paths.py, time.py
config/
  acquisition.yaml.example
scripts/
  publish_ply_take.py        # Thin CLI
  test_cti.py                # Manual GenTL/CTI hardware validation
samples/pointclouds/
  sample.ply                 # Tracked test asset
vendor/sick/sdk_4_3/
  SICKGigEVisionTL.cti       # GenTL producer for Harvesters (future)
```

## AcquisitionBase

All acquisition modes implement `acquire() -> tuple[str, Path]` returning `(take_id, published_folder)`.

Future implementations:

- **Harvesters** — GigE Vision / GenTL grab (`cti_path` in config), write PLY/TIFF/PNG, then publish.
- **FTP** — watch or poll drops, normalize filenames, publish.

Shared publish path: always use `vision_3d_acquisition.storage.publisher.AcquisitionPublisher.publish_take()`.

## AcquisitionPublisher

Located in **`storage.publisher`** (not `acquisition`).

**Constructor:** `AcquisitionPublisher(data_dir: Path)`

Ensures:

- `data_dir/incoming/`
- `data_dir/state/`

**Method:** `publish_take(take_id, metadata: AcquisitionMetadata, files: dict[str, Path])`

- `files` keys match `metadata.files` (`point_cloud`, `height`, `reflectance`).
- Only keys with non-null filenames in metadata are copied.
- Stages under `incoming/.<take_id>.tmp/`, renames, writes `READY`, updates `acquisition.json`.

Legacy import still works:

```python
from vision_3d_acquisition.acquisition.publisher import AcquisitionPublisher
```

## OfflinePlyAcquisition

**Constructor:** `OfflinePlyAcquisition(publisher: AcquisitionPublisher)`

**Method:** `acquire(ply_path: Path) -> tuple[str, Path]`

1. Resolve and validate path (exists, regular file, `.ply` extension).
2. Generate `take_id` via `generate_take_id()`.
3. Build `AcquisitionMetadata` with `source="offline_ply"`, `mode="offline"`.
4. Publish source file as `point_cloud.ply`.

## CLI

From repository root with venv activated:

```bash
python scripts/publish_ply_take.py \
  --ply samples/pointclouds/sample.ply \
  --data-dir data
```

Output:

- Prints `take_id`
- Prints absolute path to `data/incoming/<take_id>/`
- Exit code 1 on missing/invalid PLY or publish errors

### Batch regroup takes into a new session (metadata-only)

Use the unified CLI to reassign existing takes to another dataset session without touching raw files or processing artifacts.

```bash
python scripts/sensor_studio_cli.py acquisition update-take-session \
  --data-dir data \
  --session-id ball_good_84mm \
  --take-file labels/ball_good_84mm_takes.txt
```

Defaults:

- dry-run by default (no writes)
- validates take existence
- infers dataset from existing take metadata

Apply changes explicitly:

```bash
python scripts/sensor_studio_cli.py acquisition update-take-session \
  --data-dir data \
  --session-id ball_good_84mm \
  --session-name "Good ball 84 mm" \
  --create-session \
  --take-file labels/ball_good_84mm_takes.txt \
  --apply
```

Notes:

- only `session_id` in take sidecar metadata is updated
- `dataset_id`, raw acquisitions, processing outputs, and labels remain unchanged
- operation logs are written on apply at `data/runtime/logs/update_take_session_<timestamp>.json`

### Session terminology and bulk reprocessing scope

Two session concepts now coexist intentionally:

- `Dataset session` / `Experiment session`: dataset-managed engineering scope stored in take sidecar metadata as `session_id`
- `Runtime acquisition group`: low-level acquisition/replay grouping carried by raw capture metadata and runtime state

Operational guidance:

- Use experiment sessions as the default scope for Studio replay, calibration context, and bulk reprocessing.
- Use runtime acquisition groups only when you specifically need ingestion/replay-era grouping behavior.
- Keep ML sets separate from both; they remain curation/export groupings rather than acquisition or replay scope.

### ML set grouping (separate from acquisition sessions)

Acquisition session reassignment and ML grouping are intentionally separate concerns.

- `Session` remains acquisition provenance (`one take -> one session`).
- `MLSet` is ML curation (`one take -> zero/one/many ML sets`).

Create ML set:

```bash
python scripts/sensor_studio_cli.py ml create-set \
  --data-dir data \
  --dataset-id bolas-2-5-1 \
  --ml-set-id balls_scrap_classifier_v1 \
  --name "Balls vs Scrap Classifier v1" \
  --task-type classification
```

Add takes to ML set:

```bash
python scripts/sensor_studio_cli.py ml add-takes \
  --data-dir data \
  --ml-set-id balls_scrap_classifier_v1 \
  --take-file labels/balls_scrap_labeled_takes.txt \
  --split unassigned
```

Inspect ML set:

```bash
python scripts/sensor_studio_cli.py ml list-set \
  --data-dir data \
  --ml-set-id balls_scrap_classifier_v1 \
  --show-memberships
```

Assign deterministic leakage-safe splits:

```bash
python scripts/sensor_studio_cli.py ml assign-splits \
  --data-dir data \
  --ml-set-id balls_scrap_classifier_v1 \
  --by-physical-object-id \
  --train 0.7 \
  --validation 0.15 \
  --test 0.15 \
  --seed 42
```

Apply changes:

```bash
python scripts/sensor_studio_cli.py ml assign-splits \
  --data-dir data \
  --ml-set-id balls_scrap_classifier_v1 \
  --by-physical-object-id \
  --train 0.7 \
  --validation 0.15 \
  --test 0.15 \
  --seed 42 \
  --apply
```

Notes:

- dry-run is the default (no writes)
- when `--by-physical-object-id` is enabled, all takes sharing the same `physical_object_id` are assigned to the same split
- ratios must be non-negative and sum to `1.0`
- apply writes operation log: `data/runtime/logs/ml_assign_splits_<timestamp>.json`

Batch reprocess ML set through a pipeline:

```bash
python scripts/sensor_studio_cli.py ml reprocess-set \
  --data-dir data \
  --ml-set-id balls_scrap_classifier_v1 \
  --pipeline-id mining_steel_ball_classification_25d \
  --split train \
  --apply
```

Export ML features table (CSV):

```bash
python scripts/sensor_studio_cli.py ml export-features \
  --data-dir data \
  --ml-set-id balls_scrap_classifier_v1 \
  --pipeline-id mining_steel_ball_classification_25d \
  --output data/ml_exports/balls_scrap_classifier_v1/features.csv
```

Manifest-driven ML import:

```bash
python scripts/sensor_studio_cli.py ml import-manifest \
  --data-dir data \
  --dataset-id bolas-2-5-1 \
  --ml-set-id balls_scrap_classifier_v1 \
  --manifest labels/balls_manifest.csv \
  --apply
```

Reprocess/export semantics:

- both commands operate on MLSet memberships (optional split filters)
- `reprocess-set` is dry-run by default and writes apply log:
  - `data/runtime/logs/ml_reprocess_set_<timestamp>.json`
- `export-features` emits one row per take and includes:
  - MLSet metadata columns (`take_id`, `split`, `physical_object_id`, etc.)
  - pipeline provenance columns (`pipeline_id`, `pipeline_run_id`, `pipeline_timestamp`)
  - flattened scalar numeric features from compatible processing outputs
- `import-manifest` supports `.csv` and `.json` manifests and updates memberships idempotently

### Explainable 25D Rule Tuning

Use exported ML features to tune rule thresholds (not ML model training):

```bash
python scripts/tune_25d_rules.py \
  --features-csv data/ml_exports/balls_scrap_classifier_v1/features_with_diagnostics.csv \
  --output-dir data/ml_exports/balls_scrap_classifier_v1/rule_tuning \
  --group-column physical_object_id \
  --label-column expected_superclass \
  --search random \
  --max-evals 2000 \
  --seed 42
```

Outputs:
- `best_rules.json`
- `tuning_report.json`
- `confusion_matrix.csv`
- `per_class_metrics.csv`
- `predictions.csv`
- `feature_ranges.json`

### Evaluate Rule Sets (No Reprocessing)

Evaluate an external rule config directly on exported features:

```bash
python scripts/evaluate_25d_rules.py \
  --features-csv data/ml_exports/balls_scrap_classifier_v1/features_with_diagnostics.csv \
  --rules-config configs/classifiers/mining_ball_rules_default.json \
  --output-dir data/ml_exports/balls_scrap_classifier_v1/rule_eval
```

Compare two rule sets:

```bash
python scripts/evaluate_25d_rules.py \
  --features-csv data/ml_exports/balls_scrap_classifier_v1/features_with_diagnostics.csv \
  --rules-config configs/classifiers/mining_ball_rules_default.json \
  --compare-rules-config configs/classifiers/mining_ball_rules_tuned_20260529.json \
  --output-dir data/ml_exports/balls_scrap_classifier_v1/rule_eval_compare
```

### Reprocess with External Rule Set

Optional rule-set override for 25D reprocessing:

```bash
python scripts/sensor_studio_cli.py ml reprocess-set \
  --data-dir data \
  --dataset-id bolas-2-5-1 \
  --ml-set-id balls_scrap_classifier_v1 \
  --pipeline-id mining_steel_ball_classification_25d \
  --classifier-rules configs/classifiers/mining_ball_rules_tuned_20260529.json \
  --apply
```

If `--classifier-rules` is omitted, built-in classifier behavior remains the default.

Rule-set resolution precedence:

- runtime override (`--classifier-rules`)
- pipeline recipe config rule-set path (when configured)
- `SENSOR_STUDIO_DEFAULT_RULE_SET` environment variable (optional)
- built-in defaults

### List Available Rule Sets

Discover selectable classifier rule configs for demo/experiment workflows:

```bash
python scripts/sensor_studio_cli.py ml list-rule-sets \
  --data-dir data
```

Machine-readable output:

```bash
python scripts/sensor_studio_cli.py ml list-rule-sets \
  --data-dir data \
  --json
```

Optional classifier filter:

```bash
python scripts/sensor_studio_cli.py ml list-rule-sets \
  --data-dir data \
  --classifier-id mining_steel_ball_classification_25d_rules
```

Show currently resolved active rule set:

```bash
python scripts/sensor_studio_cli.py ml show-active-rule-set \
  --pipeline-id mining_steel_ball_classification_25d
```

JSON output:

```bash
python scripts/sensor_studio_cli.py ml show-active-rule-set \
  --pipeline-id mining_steel_ball_classification_25d \
  --json
```
- apply import writes:
  - `data/runtime/logs/ml_import_manifest_<timestamp>.json`

MLSet lookup semantics (`ml add-takes`, `ml list-set`):

- MLSet IDs are dataset-scoped (not globally unique).
- If `--dataset-id` is provided, resolution is constrained to that dataset.
- If omitted:
  - zero matches => `MLSet '<id>' not found.`
  - one match => auto-resolved and command continues
  - multiple matches => fails and requires `--dataset-id`

## Automated tests vs hardware validation

`pytest tests/` covers deterministic unit and integration checks that can run
without a SICK SDK install, a CTI producer on the host system, or connected
camera hardware.

GenTL/CTI validation is intentionally handled by a manual script:

```bash
python scripts/test_cti.py \
  --cti vendor/sick/sdk_4_3/SICKGigEVisionTL.cti \
  --list-devices
```

To acquire one buffer from a detected device:

```bash
python scripts/test_cti.py \
  --cti vendor/sick/sdk_4_3/SICKGigEVisionTL.cti \
  --device-index 0 \
  --timeout 5
```

The script prints the GenTL producer path, detected devices, and acquisition
success or failure details. It is expected to fail meaningfully if the SDK,
network interface, camera, permissions, or GenTL runtime are not configured.

Manual CTI validation requires:

- `harvesters` installed from `requirements.txt`
- a SICK GenTL CTI producer, such as
  `vendor/sick/sdk_4_3/SICKGigEVisionTL.cti`
- camera hardware connected and reachable from the host
- any vendor SDK runtime requirements, dynamic library paths, NIC settings, and
  permissions required by the SICK SDK on the target machine

## Configuration

Copy and edit:

```bash
cp config/acquisition.yaml.example config/acquisition.yaml
```

Example keys: `data_dir`, `mode`, `cti_path`, `input_ply` (see `config/acquisition.yaml.example`).

## Adding a live mode (sketch)

1. Subclass `AcquisitionBase`.
2. Load CTI from `vendor/sick/sdk_4_3/SICKGigEVisionTL.cti` via Harvesters.
3. After grab, fill `AcquisitionMetadata` (`source="harvesters"`, `mode="live"`, sensor fields).
4. Pass local paths to `publish_take` for each non-null file in `metadata.files`.
5. Do not create `READY` manually — publisher owns that.

## Operations

**Stale temp folders:** Safe to delete `data/incoming/.<take_id>.tmp/` if acquisition died mid-publish.

**Disk space:** Each take is self-contained; archive or delete processed/incoming per retention policy (not automated in v1).

**Permissions:** Acquisition process needs write access to `data/` only.

## Related docs

- [architecture.md](architecture.md) — system context and package boundaries
- [contracts.md](contracts.md) — JSON and filesystem norms
- [processes.md](processes.md) — process boundaries
