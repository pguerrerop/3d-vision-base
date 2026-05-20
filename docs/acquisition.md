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
