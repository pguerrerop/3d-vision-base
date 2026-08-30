"""The indexed listing must answer exactly what the filesystem scan answers.

Stage 1 swaps the take listing onto SQL. The contract is that nothing about the
response changes: same items in the same order, same counts, same buckets. These
tests run both implementations over the same data directory and diff them, which
is the only way the swap is safe to make.

The matrix deliberately includes filters that look trivial but are not:
case-insensitive comparisons, expected_class as a substring, a date window, and
the session_type filter that compares against the raw session file rather than
the effective default.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from vision_3d_acquisition.api import filesystem as fs
from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.datasets import DatasetService
from vision_3d_acquisition.storage import catalog_queries
from vision_3d_acquisition.storage.catalog_index import CatalogIndexer

FILTER_CASES: list[dict[str, Any]] = [
    {},
    {"limit": 2},
    {"limit": 2, "offset": 2},
    {"dataset_id": "demo"},
    {"dataset_id": "otro"},
    {"session_id": "s1"},
    {"validation_status": "valid"},
    {"validation_status": "VALID"},
    {"tag": "wet"},
    {"tag": "WET"},
    {"semantic_label": "chatarra"},
    {"superclass_label": "SCRAP_METAL"},
    {"category": "revisar"},
    {"search": "take"},
    {"search": "BOLA"},
    {"expected_class": "chat"},
    {"expected_class": "chatarra"},
    {"physical_object_id": "obj_0042"},
    {"physical_object_id": "OBJ_0042"},
    {"session_type": "engineering"},
    {"session_type": "curated"},
    {"reference_type": "patron"},
    {"is_reference": True},
    {"is_reference": False},
    {"is_golden_sample": True},
    {"include_archived": True},
    {"split": "train"},
    {"created_from": "2026-05-20"},
    {"created_to": "2026-05-19"},
    {"created_from": "2026-05-18", "created_to": "2026-05-20"},
    {"calibration_linkage_only": True},
    {"dataset_id": "demo", "validation_status": "valid", "limit": 1},
    {"session_id": "s1", "tag": "wet", "include_archived": True},
]


@pytest.fixture
def populated(catalog_settings: ApiSettings, write_take, monkeypatch) -> ApiSettings:
    """A data directory exercising every branch the filters can take.

    Built in the legacy filesystem layout on purpose. Stage 2 moved the documents
    into the catalog, so comparing the two listings only means something if both
    are looking at the same data: the fixture writes files, each test reads the
    filesystem answer from them, and the catalog answer comes from importing
    those same files. That makes every case a check of the migration too.
    """
    monkeypatch.setenv("SENSOR_STUDIO_INDEX", "off")
    service = DatasetService(catalog_settings.data_dir)
    service.create_dataset(dataset_id="demo", name="Demo")
    service.create_dataset(dataset_id="otro", name="Otro")
    service.create_session(dataset_id="demo", session_id="s1", name="Sesion 1", session_type="curated")
    service.create_session(dataset_id="demo", session_id="s2", name="Sesion 2")
    service.create_session(dataset_id="otro", session_id="s3", name="Sesion 3", session_type="benchmark")

    write_take("take_labeled", created_at="2026-05-19T10:00:00Z", done=True)
    service.upsert_take_metadata(
        take_id="take_labeled",
        dataset_id="demo",
        session_id="s1",
        updates={
            "friendly_name": "Bola buena",
            "tags": ["Wet", "good"],
            "semantic_labels": ["chatarra"],
            "superclass_labels": ["SCRAP_METAL"],
            "categories": ["revisar"],
            "validation_status": "valid",
            "expected_class": "chatarra",
            "physical_object_id": "obj_0042",
            "reference_type": "patron",
            "is_reference": True,
            "split": "train",
            "calibration_id": "calib_a",
        },
    )

    write_take("take_plain", created_at="2026-05-20T10:00:00Z")

    write_take("take_archived", created_at="2026-05-18T10:00:00Z")
    service.upsert_take_metadata(
        take_id="take_archived",
        dataset_id="demo",
        session_id="s2",
        updates={"archived": True, "tags": ["wet"], "validation_status": "rejected"},
    )

    write_take("take_golden", created_at="2026-05-21T10:00:00Z", done=True)
    service.upsert_take_metadata(
        take_id="take_golden",
        dataset_id="otro",
        session_id="s3",
        updates={"is_golden_sample": True, "validation_status": "golden_sample"},
    )

    # No sidecar at all: has to read as default_take_metadata in both paths.
    write_take("take_orphan", created_at="2026-05-17T10:00:00Z", session_id="acq_session")

    # No created_at anywhere: falls back to sorting by take_id.
    write_take("take_undated", created_at="")

    monkeypatch.delenv("SENSOR_STUDIO_INDEX")
    return catalog_settings


def _filesystem_answer(settings: ApiSettings, monkeypatch, call):
    """Run a listing against the legacy layout, before anything is migrated."""
    monkeypatch.setenv("SENSOR_STUDIO_INDEX", "off")
    try:
        return call()
    finally:
        monkeypatch.delenv("SENSOR_STUDIO_INDEX")


def _reindex(settings: ApiSettings, conn: sqlite3.Connection) -> None:
    CatalogIndexer(settings, conn).rebuild(full=True)


def _comparable(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "items": [item.take_id for item in payload["items"]],
        "limit": payload["limit"],
        "offset": payload["offset"],
        "has_more": payload["has_more"],
        "next_offset": payload["next_offset"],
        "filtered_count": payload["filtered_count"],
        "total_count": payload["total_count"],
        "summary_counts": payload["summary_counts"],
    }


@pytest.mark.parametrize("case", FILTER_CASES, ids=lambda case: json.dumps(case, sort_keys=True))
def test_paged_listing_matches_the_filesystem_scan(
    populated: ApiSettings, catalog_conn: sqlite3.Connection, case: dict[str, Any], monkeypatch
) -> None:
    expected = _filesystem_answer(populated, monkeypatch, lambda: fs.list_takes_paged(populated, **case))

    _reindex(populated, catalog_conn)
    actual = catalog_queries.list_takes_paged(catalog_conn, dict(case))

    assert _comparable(actual) == _comparable(expected)


@pytest.mark.parametrize(
    "case",
    [case for case in FILTER_CASES if not {"limit", "offset", "created_from", "created_to", "split", "expected_class", "calibration_linkage_only"} & case.keys()],
    ids=lambda case: json.dumps(case, sort_keys=True),
)
def test_unpaged_listing_matches_the_filesystem_scan(
    populated: ApiSettings, catalog_conn: sqlite3.Connection, case: dict[str, Any], monkeypatch
) -> None:
    expected = _filesystem_answer(
        populated, monkeypatch, lambda: [item.take_id for item in fs.list_takes(populated, **case)]
    )

    _reindex(populated, catalog_conn)
    actual = [item.take_id for item in catalog_queries.list_takes(catalog_conn, dict(case))]

    assert actual == expected


def test_page_items_are_the_indexed_summaries(
    populated: ApiSettings, catalog_conn: sqlite3.Connection
) -> None:
    _reindex(populated, catalog_conn)

    page = catalog_queries.list_takes_paged(catalog_conn, {"limit": 10})

    for item in page["items"]:
        stored = json.loads(
            catalog_conn.execute(
                "SELECT summary_json FROM take_index WHERE take_id = ?", (item.take_id,)
            ).fetchone()["summary_json"]
        )
        assert item.model_dump(mode="json") == stored


def test_the_flag_selects_the_filesystem_path(populated: ApiSettings, monkeypatch) -> None:
    """SENSOR_STUDIO_INDEX=off has to work without an index present at all."""
    monkeypatch.setenv("SENSOR_STUDIO_INDEX", "off")
    assert fs._catalog(populated) is None

    payload = fs.list_takes_paged(populated, limit=5)

    assert payload["items"]
    assert "source" not in payload.get("profile", {})


def test_the_page_items_survive_the_migration(
    populated: ApiSettings, catalog_conn: sqlite3.Connection, monkeypatch
) -> None:
    """Every take that existed as a file is a row afterwards, with its labels."""
    before = _filesystem_answer(
        populated,
        monkeypatch,
        lambda: {
            item.take_id: (item.tags, item.semantic_labels, item.validation_status)
            for item in fs.list_takes_paged(populated, limit=200)["items"]
        },
    )

    _reindex(populated, catalog_conn)
    after = {
        item.take_id: (item.tags, item.semantic_labels, item.validation_status)
        for item in catalog_queries.list_takes_paged(catalog_conn, {"limit": 200})["items"]
    }

    assert after == before


def test_the_heavy_status_field_is_opt_in(
    populated: ApiSettings, catalog_conn: sqlite3.Connection
) -> None:
    """acquisition_processing_status embeds the run's whole result.json.

    It is 159 KB at the median against ~1.6 KB for the rest of the summary, and
    the listing never read it: the frontend only touches it on the take detail.
    So the listing leaves it out unless a caller asks.
    """
    _reindex(populated, catalog_conn)

    default_page = fs.list_takes_paged(populated, limit=10)
    assert all(item.acquisition_processing_status is None for item in default_page["items"])

    opted_in = fs.list_takes_paged(populated, limit=10, include_acquisition_processing_status=True)
    assert [item.take_id for item in opted_in["items"]] == [
        item.take_id for item in default_page["items"]
    ]

    # Same for the unpaged listing.
    assert all(
        item.acquisition_processing_status is None for item in fs.list_takes(populated)
    )


def test_the_take_detail_still_carries_the_status(
    populated: ApiSettings, catalog_conn: sqlite3.Connection
) -> None:
    """The one consumer that reads it is TakeDetailView, off TakeDetail."""
    from vision_3d_acquisition.acquisition.processing_status import write_status

    _reindex(populated, catalog_conn)
    write_status(populated.data_dir, "take_labeled", {"status": "processed", "take_id": "take_labeled"})

    detail = fs.get_take_detail(populated, "take_labeled")

    assert detail is not None
    assert detail.acquisition_processing_status["status"] == "processed"

    # And the listing still omits it, even though it is now on disk.
    page = fs.list_takes_paged(populated, limit=10)
    assert all(item.acquisition_processing_status is None for item in page["items"])

    opted_in = fs.list_takes_paged(populated, limit=10, include_acquisition_processing_status=True)
    match = next(item for item in opted_in["items"] if item.take_id == "take_labeled")
    assert match.acquisition_processing_status["status"] == "processed"
