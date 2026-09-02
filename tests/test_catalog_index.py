"""Stage 0 of the SQLite catalog: migrations, projection, and the equivalence net.

The load-bearing test here is ``test_indexed_summary_matches_filesystem_read``.
Every later stage of the migration assumes an indexed row is interchangeable with
what ``get_take_summary`` returns from disk; if that stops holding, the stages
that read from the index start serving wrong answers silently.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest

from vision_3d_acquisition.api.filesystem import get_take_summary
from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.datasets import DatasetService
from vision_3d_acquisition.storage import db
from vision_3d_acquisition.storage.catalog_index import PROJECTION_VERSION, CatalogIndexer


def _indexer(settings: ApiSettings, conn: sqlite3.Connection) -> CatalogIndexer:
    return CatalogIndexer(settings, conn)


def _touch_later(path: Path) -> None:
    """Move a file's mtime forward without depending on clock resolution."""
    stat = path.stat()
    os.utime(path, (stat.st_atime + 10, stat.st_mtime + 10))


# --------------------------------------------------------------------- schema


def test_migrations_apply_once_and_are_idempotent(catalog_data_dir: Path) -> None:
    conn = db.connect(catalog_data_dir)
    try:
        assert db.schema_version(conn) == 0
        assert db.migrate(conn) == [version for version, _ in db.available_migrations()]
        assert db.migrate(conn) == []
        assert db.schema_version(conn) == db.latest_schema_version()
    finally:
        conn.close()


def test_catalog_pragmas_are_set(catalog_conn: sqlite3.Connection) -> None:
    assert catalog_conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert catalog_conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_transaction_rolls_back_on_error(catalog_conn: sqlite3.Connection) -> None:
    with pytest.raises(RuntimeError):
        with db.transaction(catalog_conn):
            catalog_conn.execute("INSERT INTO dataset(id, name) VALUES ('d', 'D')")
            raise RuntimeError("boom")
    assert catalog_conn.execute("SELECT count(*) FROM dataset").fetchone()[0] == 0


# ----------------------------------------------------------------- projection


def test_full_rebuild_projects_takes_and_authoritative_rows(
    catalog_settings: ApiSettings, catalog_conn: sqlite3.Connection, write_take
) -> None:
    write_take("take_a")
    write_take("take_b", done=True)
    service = DatasetService(catalog_settings.data_dir)
    service.create_dataset(dataset_id="demo", name="Demo")
    service.create_session(dataset_id="demo", session_id="s1", name="Session 1")
    service.upsert_take_metadata(
        take_id="take_b",
        dataset_id="demo",
        session_id="s1",
        updates={"friendly_name": "Bola 1", "tags": ["Wet", "good"], "validation_status": "valid"},
    )

    report = _indexer(catalog_settings, catalog_conn).rebuild(full=True)

    assert report.takes_on_disk == 2
    assert report.takes_projected == 2
    assert report.take_failures == []
    assert report.datasets == 1 and report.sessions == 1

    rows = {
        row["take_id"]: row
        for row in catalog_conn.execute("SELECT take_id, status, has_done FROM take_index")
    }
    assert set(rows) == {"take_a", "take_b"}
    assert rows["take_a"]["status"] == "incoming"
    assert rows["take_b"]["has_done"] == 1

    labels = catalog_conn.execute(
        "SELECT value, value_norm FROM take_label WHERE take_id = 'take_b' AND kind = 'tag'"
        " ORDER BY value"
    ).fetchall()
    assert [(row["value"], row["value_norm"]) for row in labels] == [("Wet", "wet"), ("good", "good")]

    metadata = catalog_conn.execute(
        "SELECT friendly_name, validation_status, dataset_id FROM take_metadata WHERE take_id = 'take_b'"
    ).fetchone()
    assert metadata["friendly_name"] == "Bola 1"
    assert metadata["validation_status"] == "valid"
    assert metadata["dataset_id"] == "demo"


def test_indexed_summary_matches_filesystem_read(
    catalog_settings: ApiSettings, catalog_conn: sqlite3.Connection, write_take
) -> None:
    write_take("take_plain")
    write_take("take_done", done=True)
    service = DatasetService(catalog_settings.data_dir)
    service.create_dataset(dataset_id="demo", name="Demo")
    service.create_session(dataset_id="demo", session_id="s1", name="Session 1")
    service.upsert_take_metadata(
        take_id="take_done",
        dataset_id="demo",
        session_id="s1",
        updates={"tags": ["a"], "semantic_labels": ["bola"], "expected_diameter_mm": 42.5},
    )

    _indexer(catalog_settings, catalog_conn).rebuild(full=True)

    for take_id in ("take_plain", "take_done"):
        stored = json.loads(
            catalog_conn.execute(
                "SELECT summary_json FROM take_index WHERE take_id = ?", (take_id,)
            ).fetchone()["summary_json"]
        )
        assert stored == get_take_summary(catalog_settings, take_id).model_dump(mode="json")


def test_verify_passes_and_detects_a_stale_row(
    catalog_settings: ApiSettings, catalog_conn: sqlite3.Connection, write_take
) -> None:
    write_take("take_a")
    indexer = _indexer(catalog_settings, catalog_conn)
    indexer.rebuild(full=True)

    assert indexer.verify()["ok"] is True

    with db.transaction(catalog_conn):
        catalog_conn.execute(
            "UPDATE take_index SET summary_json = json_set(summary_json, '$.friendly_name', 'wrong')"
        )
    result = indexer.verify()
    assert result["ok"] is False
    assert result["mismatched"] == [{"take_id": "take_a", "fields": ["friendly_name"]}]


# ---------------------------------------------------------------- incremental


def test_stale_rebuild_skips_unchanged_takes(
    catalog_settings: ApiSettings, catalog_conn: sqlite3.Connection, write_take
) -> None:
    write_take("take_a")
    write_take("take_b")
    indexer = _indexer(catalog_settings, catalog_conn)
    indexer.rebuild(full=True)

    report = indexer.rebuild()

    assert report.takes_projected == 0
    assert report.takes_unchanged == 2


def test_changed_result_reprojects_only_that_take(
    catalog_settings: ApiSettings, catalog_conn: sqlite3.Connection, write_take
) -> None:
    write_take("take_a", done=True)
    write_take("take_b", done=True)
    indexer = _indexer(catalog_settings, catalog_conn)
    indexer.rebuild(full=True)

    result_path = catalog_settings.processed_dir / "take_a" / "result.json"
    result_path.write_text(
        json.dumps({"status": "failed", "summary": {"decision": "reject"}}), encoding="utf-8"
    )
    _touch_later(result_path)

    report = indexer.rebuild()

    assert report.takes_projected == 1
    assert report.takes_unchanged == 1
    assert (
        catalog_conn.execute("SELECT status FROM take_index WHERE take_id = 'take_a'").fetchone()[0]
        == "failed"
    )


def test_a_metadata_edit_reaches_the_index_immediately(
    catalog_settings: ApiSettings, catalog_conn: sqlite3.Connection, write_take
) -> None:
    """upsert_take_metadata pushes: no rebuild needed for the edit to be visible."""
    write_take("take_a")
    service = DatasetService(catalog_settings.data_dir)
    service.create_dataset(dataset_id="demo", name="Demo")
    service.create_session(dataset_id="demo", session_id="s1", name="Session 1")
    service.upsert_take_metadata(
        take_id="take_a", dataset_id="demo", session_id="s1", updates={"tags": ["one"]}
    )
    _indexer(catalog_settings, catalog_conn).rebuild(full=True)

    service.upsert_take_metadata(
        take_id="take_a", dataset_id="demo", session_id="s1", updates={"tags": ["one", "two"]}
    )

    summary = json.loads(
        catalog_conn.execute("SELECT summary_json FROM take_index").fetchone()["summary_json"]
    )
    assert summary["tags"] == ["one", "two"]
    assert [
        row[0]
        for row in catalog_conn.execute(
            "SELECT value FROM take_label WHERE take_id='take_a' AND kind='tag' ORDER BY value"
        )
    ] == ["one", "two"]


def test_a_missed_push_is_still_caught_by_the_rebuild(
    catalog_settings: ApiSettings, catalog_conn: sqlite3.Connection, write_take, monkeypatch
) -> None:
    """Labels live under data/datasets/, so mtimes on the take dir never move.

    The push is the fast path, not the guarantee: with it disabled the edit still
    has to be picked up, because that is what stops a missed hook from turning
    into permanent drift.
    """
    write_take("take_a")
    service = DatasetService(catalog_settings.data_dir)
    service.create_dataset(dataset_id="demo", name="Demo")
    service.create_session(dataset_id="demo", session_id="s1", name="Session 1")
    service.upsert_take_metadata(
        take_id="take_a", dataset_id="demo", session_id="s1", updates={"tags": ["one"]}
    )
    indexer = _indexer(catalog_settings, catalog_conn)
    indexer.rebuild(full=True)

    monkeypatch.setenv("SENSOR_STUDIO_INDEX", "off")
    service.upsert_take_metadata(
        take_id="take_a", dataset_id="demo", session_id="s1", updates={"tags": ["one", "two"]}
    )
    monkeypatch.delenv("SENSOR_STUDIO_INDEX")

    report = indexer.rebuild()

    assert report.takes_projected == 1
    summary = json.loads(
        catalog_conn.execute("SELECT summary_json FROM take_index").fetchone()["summary_json"]
    )
    assert summary["tags"] == ["one", "two"]


def test_projection_version_bump_forces_reprojection(
    catalog_settings: ApiSettings, catalog_conn: sqlite3.Connection, write_take, monkeypatch
) -> None:
    write_take("take_a")
    _indexer(catalog_settings, catalog_conn).rebuild(full=True)

    monkeypatch.setattr(
        "vision_3d_acquisition.storage.catalog_index.PROJECTION_VERSION", PROJECTION_VERSION + 1
    )
    report = _indexer(catalog_settings, catalog_conn).rebuild()

    assert report.takes_projected == 1


def test_deleted_take_is_dropped_from_the_index(
    catalog_settings: ApiSettings, catalog_conn: sqlite3.Connection, write_take
) -> None:
    import shutil

    write_take("take_a")
    write_take("take_b")
    indexer = _indexer(catalog_settings, catalog_conn)
    indexer.rebuild(full=True)

    shutil.rmtree(catalog_settings.incoming_dir / "take_b")
    report = indexer.rebuild()

    assert report.takes_removed == 1
    assert [row[0] for row in catalog_conn.execute("SELECT take_id FROM take_index")] == ["take_a"]
    assert indexer.status()["drift"] == 0


# -------------------------------------------------------------------- drift


def test_migrating_duplicate_sidecars_keeps_one_and_records_the_rest(
    catalog_settings: ApiSettings, catalog_conn: sqlite3.Connection, write_take
) -> None:
    """The one-shot import is the only way a duplicate can still show up.

    On disk a take could carry a sidecar under two sessions at once. take_id is
    a primary key, so the state is unrepresentable in the catalog: the import
    keeps the first membership — the one the filesystem resolution surfaced — and
    records the loser rather than dropping it silently.
    """
    from vision_3d_acquisition.storage import catalog_sync

    write_take("take_a")
    datasets_dir = catalog_settings.datasets_dir
    (datasets_dir / "dataset_demo").mkdir(parents=True, exist_ok=True)
    (datasets_dir / "dataset_demo" / "dataset.json").write_text(
        json.dumps({"id": "demo", "name": "Demo", "created_at": "2026-05-01T00:00:00Z"}),
        encoding="utf-8",
    )
    for session_id, tag in (("s1", "from_s1"), ("s2", "from_s2")):
        session_dir = datasets_dir / "dataset_demo" / "sessions" / f"session_{session_id}"
        (session_dir).mkdir(parents=True, exist_ok=True)
        (session_dir / "session.json").write_text(
            json.dumps({"id": session_id, "dataset_id": "demo", "created_at": "2026-05-01T00:00:00Z"}),
            encoding="utf-8",
        )
        take_dir = session_dir / "takes" / "take_a"
        take_dir.mkdir(parents=True, exist_ok=True)
        (take_dir / "metadata.json").write_text(
            json.dumps(
                {"take_id": "take_a", "dataset_id": "demo", "session_id": session_id, "tags": [tag]}
            ),
            encoding="utf-8",
        )

    counts = catalog_sync.ensure_documents_migrated(catalog_settings.data_dir)

    assert counts["takes"] == 1
    assert catalog_conn.execute("SELECT count(*) FROM take_metadata").fetchone()[0] == 1
    conflict = catalog_conn.execute("SELECT take_id, session_id FROM take_metadata_conflict").fetchone()
    assert conflict["take_id"] == "take_a"

    service = DatasetService(catalog_settings.data_dir)
    assert len(service.resolve_all_take_memberships("take_a")) == 1


def test_process_runs_are_projected_from_the_run_index(
    catalog_settings: ApiSettings, catalog_conn: sqlite3.Connection, write_take
) -> None:
    from vision_3d_acquisition.processing.status_index import append_process_run_index

    write_take("take_a")
    append_process_run_index(
        catalog_settings.data_dir,
        take_id="take_a",
        pipeline_instance_id="pipeline_1",
        run_id="run_1",
        pipeline_family="2d",
        status="success",
        run_dir=catalog_settings.data_dir / "processes" / "runs" / "pipeline_1" / "run_1",
        created_at="2026-05-19T11:00:00Z",
    )

    report = _indexer(catalog_settings, catalog_conn).rebuild(full=True)

    assert report.process_runs == 1
    row = catalog_conn.execute("SELECT take_id, pipeline_family, status FROM process_run").fetchone()
    assert (row["take_id"], row["pipeline_family"], row["status"]) == ("take_a", "2d", "success")


def test_a_new_run_reaches_the_index_immediately(
    catalog_settings: ApiSettings, catalog_conn: sqlite3.Connection, write_take
) -> None:
    """append_process_run_index pushes the take it touched."""
    from vision_3d_acquisition.processing.status_index import append_process_run_index

    write_take("take_a")
    indexer = _indexer(catalog_settings, catalog_conn)
    indexer.rebuild(full=True)

    append_process_run_index(
        catalog_settings.data_dir,
        take_id="take_a",
        pipeline_instance_id="pipeline_1",
        run_id="run_1",
        pipeline_family="2d",
        status="success",
        run_dir=catalog_settings.data_dir / "processes" / "runs" / "pipeline_1" / "run_1",
        created_at="2026-05-19T11:00:00Z",
    )

    assert (
        catalog_conn.execute(
            "SELECT latest_run_status FROM take_index WHERE take_id = 'take_a'"
        ).fetchone()[0]
        == "success"
    )
    assert catalog_conn.execute("SELECT count(*) FROM process_run").fetchone()[0] == 1

    # And the pushed row is not left looking stale to the next rebuild.
    assert indexer.rebuild().takes_projected == 0


def test_opening_many_connections_at_once_never_reports_a_locked_database(tmp_path: Path) -> None:
    """Concurrent opens must not fail on the journal-mode pragma.

    ``PRAGMA journal_mode=WAL`` needs a brief exclusive lock, and SQLite does not
    run the busy handler for it: a connection that opens while another holds the
    file gets SQLITE_BUSY at once. The mode is a property of the file, so losing
    that race is not a real conflict — the winner sets the mode the loser wanted.
    Before this was handled, roughly one run in seven of the concurrent append
    test died here rather than in anything it was written to check.
    """
    from concurrent.futures import ThreadPoolExecutor

    from vision_3d_acquisition.storage import db

    db.close_process_connections()

    def open_and_read(data_dir: Path) -> str:
        conn = db.connect(data_dir)
        try:
            return str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
        finally:
            conn.close()

    # Only the very first opener of a file performs the transition, so the race
    # exists once per database. Repeat over fresh ones: a single storm caught the
    # old behaviour about one time in five, which is too weak to guard anything.
    for attempt in range(16):
        data_dir = tmp_path / f"data_{attempt}"
        with ThreadPoolExecutor(max_workers=12) as pool:
            modes = list(pool.map(lambda _: open_and_read(data_dir), range(12)))
        assert modes == ["wal"] * 12, f"storm {attempt}: every connection ends up in WAL, none raises"


def test_a_connection_can_wait_for_a_write_lock_instead_of_failing(tmp_path: Path) -> None:
    """busy_timeout has to be in place before any pragma that can block."""
    from vision_3d_acquisition.storage import db

    data_dir = tmp_path / "data"
    conn = db.connect(data_dir)
    try:
        timeout_ms = int(conn.execute("PRAGMA busy_timeout").fetchone()[0])
        assert timeout_ms > 0, "a zero busy timeout turns every contended write into an error"
    finally:
        conn.close()
