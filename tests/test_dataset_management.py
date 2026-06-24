from __future__ import annotations

import json
from pathlib import Path

from vision_3d_acquisition.api.filesystem import get_take_detail, list_takes, list_takes_paged
from vision_3d_acquisition.api.main import dataset_sessions, update_take_metadata
from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.datasets import DatasetService


def make_settings(data_dir: Path) -> ApiSettings:
    settings = ApiSettings(
        data_dir=data_dir,
        incoming_dir=data_dir / "incoming",
        processed_dir=data_dir / "processed",
        state_dir=data_dir / "state",
        events_dir=data_dir / "events",
        sessions_dir=data_dir / "sessions",
        datasets_dir=data_dir / "datasets",
    )
    settings.ensure_directories()
    return settings


def write_take(data_dir: Path, take_id: str) -> None:
    take_dir = data_dir / "incoming" / take_id
    take_dir.mkdir(parents=True, exist_ok=True)
    (take_dir / "rgb.png").write_bytes(b"fake")
    (take_dir / "metadata.json").write_text(
        json.dumps({"take_id": take_id, "created_at": "2026-05-19T10:00:00Z", "session_id": "session_live", "files": {"rgb": "rgb.png"}}),
        encoding="utf-8",
    )
    (take_dir / "READY").touch()


def test_take_metadata_defaults_for_legacy_take(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    write_take(settings.data_dir, "take_legacy")

    detail = get_take_detail(settings, "take_legacy")

    assert detail is not None
    assert detail.take_metadata["friendly_name"] == "take_legacy"
    assert detail.take_metadata["validation_status"] == "unreviewed"
    assert detail.take_metadata["categories"] == []
    assert detail.take_metadata["is_reference"] is False


def test_take_metadata_persistence_and_tag_editing(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    write_take(settings.data_dir, "take_meta")
    service = DatasetService(settings.data_dir)
    service.create_dataset(dataset_id="demo", name="Demo")
    service.create_session(dataset_id="demo", session_id="s1", name="Session 1")
    service.upsert_take_metadata(
        take_id="take_meta",
        dataset_id="demo",
        session_id="s1",
        updates={"friendly_name": "Take Friendly", "tags": ["good", "wet"], "expected_diameter_mm": 42.5},
        source_metadata={"session_id": "session_live"},
    )

    detail = get_take_detail(settings, "take_meta")
    summary = list_takes(settings)[0]

    assert detail is not None
    assert detail.take_metadata["friendly_name"] == "Take Friendly"
    assert set(detail.take_metadata["tags"]) == {"good", "wet"}
    assert summary.expected_diameter_mm == 42.5


def test_dataset_session_grouping_and_take_filter(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    write_take(settings.data_dir, "take_a")
    write_take(settings.data_dir, "take_b")
    service = DatasetService(settings.data_dir)
    service.create_dataset(dataset_id="set1", name="Set 1")
    service.create_session(dataset_id="set1", session_id="daylight", name="Daylight")
    service.upsert_take_metadata(take_id="take_a", dataset_id="set1", session_id="daylight", updates={"tags": ["good"]}, source_metadata={})

    all_takes = list_takes(settings)
    grouped = list_takes(settings, dataset_id="set1")
    tagged = list_takes(settings, tag="good")

    assert len(all_takes) == 2
    assert [item.take_id for item in grouped] == ["take_a"]
    assert [item.take_id for item in tagged] == ["take_a"]


def test_take_filter_by_semantic_and_superclass_labels(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    write_take(settings.data_dir, "take_sem_a")
    write_take(settings.data_dir, "take_sem_b")
    service = DatasetService(settings.data_dir)
    service.create_dataset(dataset_id="set1", name="Set 1")
    service.create_session(dataset_id="set1", session_id="daylight", name="Daylight")
    service.upsert_take_metadata(
        take_id="take_sem_a",
        dataset_id="set1",
        session_id="daylight",
        updates={"semantic_labels": ["BALL_CHIPPED"], "superclass_labels": ["BALL_DEFECT"]},
        source_metadata={},
    )
    service.upsert_take_metadata(
        take_id="take_sem_b",
        dataset_id="set1",
        session_id="daylight",
        updates={"semantic_labels": ["SCRAP_BOLT"], "superclass_labels": ["SCRAP"]},
        source_metadata={},
    )

    sem = list_takes(settings, semantic_label="BALL_CHIPPED")
    sup = list_takes(settings, superclass_label="SCRAP")

    assert [item.take_id for item in sem] == ["take_sem_a"]
    assert [item.take_id for item in sup] == ["take_sem_b"]


def test_take_filter_by_physical_object_id(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    write_take(settings.data_dir, "take_obj_a")
    write_take(settings.data_dir, "take_obj_b")
    service = DatasetService(settings.data_dir)
    service.create_dataset(dataset_id="set1", name="Set 1")
    service.create_session(dataset_id="set1", session_id="daylight", name="Daylight")
    service.upsert_take_metadata(
        take_id="take_obj_a",
        dataset_id="set1",
        session_id="daylight",
        updates={"physical_object_id": "obj_0005"},
        source_metadata={},
    )
    service.upsert_take_metadata(
        take_id="take_obj_b",
        dataset_id="set1",
        session_id="daylight",
        updates={"physical_object_id": "obj_0006"},
        source_metadata={},
    )

    filtered = list_takes(settings, physical_object_id="obj_0005")
    assert [item.take_id for item in filtered] == ["take_obj_a"]


def test_take_filter_by_session_type_category_and_reference_flags(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    write_take(settings.data_dir, "take_ref")
    write_take(settings.data_dir, "take_eng")
    service = DatasetService(settings.data_dir)
    service.create_dataset(dataset_id="set1", name="Set 1")
    service.create_session(dataset_id="set1", session_id="cur_01", name="Curated session", session_type="curated")
    service.create_session(dataset_id="set1", session_id="eng_01", name="Engineering session", session_type="engineering")
    service.upsert_take_metadata(
        take_id="take_ref",
        dataset_id="set1",
        session_id="cur_01",
        updates={
            "categories": ["golden_sample", "benchmark_case"],
            "reference_type": "golden_sample",
            "is_reference": True,
            "is_golden_sample": True,
        },
        source_metadata={},
    )
    service.upsert_take_metadata(
        take_id="take_eng",
        dataset_id="set1",
        session_id="eng_01",
        updates={"categories": ["engineering_debug"], "is_reference": False},
        source_metadata={},
    )

    curated = list_takes(settings, session_type="curated")
    golden = list_takes(settings, category="golden_sample")
    references = list_takes(settings, is_reference=True)
    golden_only = list_takes(settings, is_golden_sample=True)

    assert [item.take_id for item in curated] == ["take_ref"]
    assert [item.take_id for item in golden] == ["take_ref"]
    assert [item.take_id for item in references] == ["take_ref"]
    assert [item.take_id for item in golden_only] == ["take_ref"]


def test_take_paged_list_returns_has_more_and_respects_filters(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    for take_id in ("p1", "p2", "p3"):
        write_take(settings.data_dir, take_id)
    service = DatasetService(settings.data_dir)
    service.create_dataset(dataset_id="set1", name="Set 1")
    service.create_session(dataset_id="set1", session_id="s1", name="Session 1", session_type="curated")
    service.upsert_take_metadata(take_id="p1", dataset_id="set1", session_id="s1", updates={"is_reference": True}, source_metadata={})
    service.upsert_take_metadata(take_id="p2", dataset_id="set1", session_id="s1", updates={"is_reference": False}, source_metadata={})
    service.upsert_take_metadata(take_id="p3", dataset_id="set1", session_id="s1", updates={"is_reference": True}, source_metadata={})

    first = list_takes_paged(settings, limit=2, offset=0)
    second = list_takes_paged(settings, limit=2, offset=int(first["next_offset"] or 0))
    references = list_takes_paged(settings, limit=10, offset=0, is_reference=True)

    assert len(first["items"]) == 2
    assert first["has_more"] is True
    assert len(second["items"]) >= 1
    assert all(item.is_reference for item in references["items"])


def test_take_paged_list_profile_is_opt_in(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    for take_id in ("prof_1", "prof_2"):
        write_take(settings.data_dir, take_id)

    plain = list_takes_paged(settings, limit=1, offset=0)
    profiled = list_takes_paged(settings, limit=1, offset=0, profile=True)

    assert plain.get("profile") is None
    assert profiled["profile"]["enabled"] is True
    assert profiled["profile"]["counters"]["scanned_take_ids"] == 2
    assert profiled["profile"]["counters"]["page_item_count"] == 1
    assert set(profiled["profile"]["phase_ms"]) == {
        "list_take_ids",
        "scan_and_filter",
        "sort_candidates",
        "hydrate_page_items",
        "total",
    }


def test_dataset_session_summary_take_count_matches_paged_filters(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    for take_id in ("s1_a", "s1_b", "s2_a"):
        write_take(settings.data_dir, take_id)

    service = DatasetService(settings.data_dir)
    service.create_dataset(dataset_id="set1", name="Set 1")
    service.create_session(dataset_id="set1", session_id="session_a", name="Session A")
    service.create_session(dataset_id="set1", session_id="session_b", name="Session B")
    service.upsert_take_metadata(take_id="s1_a", dataset_id="set1", session_id="session_a", updates={}, source_metadata={})
    service.upsert_take_metadata(take_id="s1_b", dataset_id="set1", session_id="session_a", updates={}, source_metadata={})
    service.upsert_take_metadata(take_id="s2_a", dataset_id="set1", session_id="session_b", updates={}, source_metadata={})

    summaries = dataset_sessions("set1", settings)
    summary_by_id = {item.id: item.take_count for item in summaries}

    expected_a = int(list_takes_paged(settings, limit=1, offset=0, dataset_id="set1", session_id="session_a")["filtered_count"])
    expected_b = int(list_takes_paged(settings, limit=1, offset=0, dataset_id="set1", session_id="session_b")["filtered_count"])

    assert summary_by_id["session_a"] == expected_a == 2
    assert summary_by_id["session_b"] == expected_b == 1


def test_dataset_session_summary_excludes_archived_takes(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    for take_id in ("active_take", "archived_take"):
        write_take(settings.data_dir, take_id)

    service = DatasetService(settings.data_dir)
    service.create_dataset(dataset_id="set1", name="Set 1")
    service.create_session(dataset_id="set1", session_id="session_a", name="Session A")
    service.upsert_take_metadata(take_id="active_take", dataset_id="set1", session_id="session_a", updates={}, source_metadata={})
    service.upsert_take_metadata(
        take_id="archived_take",
        dataset_id="set1",
        session_id="session_a",
        updates={"archived": True},
        source_metadata={},
    )

    summaries = dataset_sessions("set1", settings)

    assert len(summaries) == 1
    assert summaries[0].take_count == 1


def test_run_history_lookup_includes_process_entries(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    write_take(settings.data_dir, "take_hist")
    index_dir = settings.data_dir / "processes" / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "runs.json").write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "take_id": "take_hist",
                        "pipeline_instance_id": "pipe_1",
                        "run_id": "run_1",
                        "pipeline_family": "2d",
                        "status": "success",
                        "created_at": "2026-05-19T12:00:00Z",
                        "path": "data/processes/runs/pipe_1/run_1",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    detail = get_take_detail(settings, "take_hist")

    assert detail is not None
    assert detail.run_history
    assert detail.run_history[0]["run_id"] == "run_1"


def test_update_take_session_id_preserves_dataset_and_cleans_old_membership(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    write_take(settings.data_dir, "take_move")
    service = DatasetService(settings.data_dir)
    service.create_dataset(dataset_id="set1", name="Set 1")
    service.create_session(dataset_id="set1", session_id="s_old", name="Old")
    service.create_session(dataset_id="set1", session_id="s_new", name="New")
    service.upsert_take_metadata(
        take_id="take_move",
        dataset_id="set1",
        session_id="s_old",
        updates={"friendly_name": "move me"},
        source_metadata={"session_id": "session_live"},
    )

    updated = service.update_take_session_id(
        take_id="take_move",
        dataset_id="set1",
        new_session_id="s_new",
        source_metadata={"session_id": "session_live"},
    )

    assert updated["dataset_id"] == "set1"
    assert updated["session_id"] == "s_new"
    assert not (settings.data_dir / "datasets" / "dataset_set1" / "sessions" / "session_s_old" / "takes" / "take_move").exists()
    assert (settings.data_dir / "datasets" / "dataset_set1" / "sessions" / "session_s_new" / "takes" / "take_move" / "metadata.json").is_file()


def test_batch_update_take_session_dry_run_only_plans(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    write_take(settings.data_dir, "t1")
    write_take(settings.data_dir, "t2")
    service = DatasetService(settings.data_dir)
    service.create_dataset(dataset_id="d1", name="Dataset 1")
    service.create_session(dataset_id="d1", session_id="s_old", name="Old Session")
    service.upsert_take_metadata(take_id="t1", dataset_id="d1", session_id="s_old", updates={}, source_metadata={"session_id": "session_live"})
    service.upsert_take_metadata(take_id="t2", dataset_id="d1", session_id="s_old", updates={}, source_metadata={"session_id": "session_live"})

    result = service.batch_update_take_session(
        take_ids=["t1", "t2", "t2"],
        new_session_id="s_new",
        create_session=True,
        session_name="Session New",
        apply=False,
    )

    assert result["requested"] == 2
    assert result["missing_take_ids"] == []
    assert result["dataset_id"] == "d1"
    assert result["destination_session_exists"] is False
    assert service.get_session("d1", "s_new") is None
    assert service.load_take_metadata(take_id="t1", source_metadata={"session_id": "session_live"}).get("session_id") == "s_old"


def test_batch_update_take_session_apply_updates_membership(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data")
    write_take(settings.data_dir, "t1")
    write_take(settings.data_dir, "t2")
    service = DatasetService(settings.data_dir)
    service.create_dataset(dataset_id="d1", name="Dataset 1")
    service.create_session(dataset_id="d1", session_id="s_old", name="Old Session")
    service.upsert_take_metadata(take_id="t1", dataset_id="d1", session_id="s_old", updates={}, source_metadata={"session_id": "session_live"})
    service.upsert_take_metadata(take_id="t2", dataset_id="d1", session_id="s_old", updates={}, source_metadata={"session_id": "session_live"})

    result = service.batch_update_take_session(
        take_ids=["t1", "t2"],
        new_session_id="s_new",
        create_session=True,
        session_name="Session New",
        apply=True,
    )

    assert result["dataset_id"] == "d1"
    assert service.get_session("d1", "s_new") is not None
    assert service.resolve_all_take_memberships("t1") == [("d1", "s_new")]
    assert service.resolve_all_take_memberships("t2") == [("d1", "s_new")]
