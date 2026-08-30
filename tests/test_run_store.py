"""Process runs stored in SQLite: the concurrency fix stage 2 exists for.

The old append read data/processes/index/runs.json whole, added an entry, and
rewrote the file — no lock, no atomic replace — from four processes at once. The
test that matters here is the concurrent one: it fails against that
implementation and passes against a table.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.processing.status_index import (
    append_process_run_index,
    ensure_runs_migrated,
    index_path,
    load_process_entries,
    process_entries_for_take,
    update_process_run_index_entry,
)
from vision_3d_acquisition.storage import run_store
from vision_3d_acquisition.storage.catalog_index import CatalogIndexer


def _append(data_dir: Path, run_id: str, *, take_id: str = "take_a", **extra) -> None:
    append_process_run_index(
        data_dir,
        take_id=take_id,
        pipeline_instance_id="pipeline_1",
        run_id=run_id,
        pipeline_family="2d",
        status="success",
        run_dir=data_dir / "processes" / "runs" / "pipeline_1" / run_id,
        created_at=f"2026-05-19T11:00:{int(run_id.split('_')[-1]):02d}Z",
        **extra,
    )


def test_appending_runs_concurrently_keeps_every_entry(
    catalog_settings: ApiSettings, write_take
) -> None:
    """Twenty writers at once. The document-rewrite implementation loses entries."""
    write_take("take_a")
    data_dir = catalog_settings.data_dir
    run_ids = [f"run_{index:02d}" for index in range(20)]

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda run_id: _append(data_dir, run_id), run_ids))

    stored = {entry["run_id"] for entry in load_process_entries(data_dir)}
    assert stored == set(run_ids)


def test_a_run_is_readable_right_after_it_is_written(
    catalog_settings: ApiSettings, write_take
) -> None:
    write_take("take_a")
    _append(catalog_settings.data_dir, "run_01")

    entries = process_entries_for_take(catalog_settings.data_dir, "take_a")

    assert len(entries) == 1
    assert entries[0]["run_id"] == "run_01"
    assert entries[0]["status"] == "success"
    assert entries[0]["pipeline_family"] == "2d"
    assert entries[0]["take_id"] == "take_a"


def test_updating_a_run_patches_it_in_place(catalog_settings: ApiSettings, write_take) -> None:
    write_take("take_a")
    _append(catalog_settings.data_dir, "run_01")

    update_process_run_index_entry(
        catalog_settings.data_dir, "run_01", comparison_id="cmp_1", status="warning"
    )

    entry = process_entries_for_take(catalog_settings.data_dir, "take_a")[0]
    assert entry["comparison_id"] == "cmp_1"
    assert entry["status"] == "warning"


def test_the_summary_payload_survives_the_round_trip(
    catalog_settings: ApiSettings, write_take
) -> None:
    write_take("take_a")
    _append(catalog_settings.data_dir, "run_01", run_summary={"decision": "accept", "objects": 3})

    entry = process_entries_for_take(catalog_settings.data_dir, "take_a")[0]
    assert entry["summary"] == {"decision": "accept", "objects": 3}


# ------------------------------------------------------------------- migration


def test_the_legacy_document_is_imported_once(catalog_settings: ApiSettings, write_take) -> None:
    write_take("take_a")
    legacy = index_path(catalog_settings.data_dir)
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "take_id": "take_a",
                        "run_id": "legacy_run",
                        "pipeline_instance_id": "pipeline_0",
                        "pipeline_family": "2d",
                        "status": "success",
                        "created_at": "2026-05-01T00:00:00Z",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert ensure_runs_migrated(catalog_settings.data_dir) == 1
    assert [entry["run_id"] for entry in load_process_entries(catalog_settings.data_dir)] == [
        "legacy_run"
    ]

    # Second call is a no-op: the table is no longer empty, so a stale mirror
    # can never overwrite rows written since.
    assert ensure_runs_migrated(catalog_settings.data_dir) == 0

    _append(catalog_settings.data_dir, "run_01")
    assert ensure_runs_migrated(catalog_settings.data_dir) == 0
    assert len(load_process_entries(catalog_settings.data_dir)) == 2


def test_a_rebuild_does_not_delete_stored_runs(
    catalog_settings: ApiSettings, catalog_conn, write_take
) -> None:
    """process_run is authoritative now; a rebuild must not clear and reload it."""
    write_take("take_a")
    _append(catalog_settings.data_dir, "run_01")
    assert len(load_process_entries(catalog_settings.data_dir)) == 1

    CatalogIndexer(catalog_settings, catalog_conn).rebuild(full=True)

    assert [entry["run_id"] for entry in load_process_entries(catalog_settings.data_dir)] == [
        "run_01"
    ]


def test_a_full_rebuild_refreshes_the_read_only_mirror(
    catalog_settings: ApiSettings, catalog_conn, write_take
) -> None:
    write_take("take_a")
    _append(catalog_settings.data_dir, "run_01")

    CatalogIndexer(catalog_settings, catalog_conn).rebuild(full=True)

    mirror = json.loads(index_path(catalog_settings.data_dir).read_text(encoding="utf-8"))
    assert mirror["readonly"] is True
    assert mirror["source"] == "sqlite:process_run"
    assert [entry["run_id"] for entry in mirror["entries"]] == ["run_01"]


def test_the_flag_reverts_to_the_json_document(
    catalog_settings: ApiSettings, write_take, monkeypatch
) -> None:
    write_take("take_a")
    monkeypatch.setenv("SENSOR_STUDIO_INDEX", "off")

    _append(catalog_settings.data_dir, "run_01")

    payload = json.loads(index_path(catalog_settings.data_dir).read_text(encoding="utf-8"))
    assert [entry["run_id"] for entry in payload["entries"]] == ["run_01"]
    assert run_store.is_empty(catalog_settings.data_dir)
    assert [entry["run_id"] for entry in load_process_entries(catalog_settings.data_dir)] == [
        "run_01"
    ]
