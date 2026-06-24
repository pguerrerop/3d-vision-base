from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.datasets import DatasetService
from vision_3d_acquisition.ml.ingestion_wizard import IngestionWizardService


FIXTURE_NAME = "balls_scrap_2026_05_25_29_wizard_compliant.tsv"
RICH_TSV = "\ufeff" + "\n".join(
    [
        "object_id\tsource_row\traw_label\tlabel\tnormalized_class\tsuperclass\tlabel_policy\treview_required\timage_ref\td1_mm\td2_mm\td3_mm",
        "obj_0002\t1\tAhuevada\tdeformed\tahuevada\tBALL_SCRAP\tinclude\tfalse\t2026-05-25T150423_025\t91.5\t90.6\t72.4",
        "obj_0002\t1\tAhuevada\tdeformed\tahuevada\tBALL_SCRAP\tinclude\tfalse\t2026-05-25T150423_026\t91.5\t90.6\t72.4",
        "obj_0019\t19\tAhuevada?\tdeformed\tahuevada\tBALL_SCRAP\treview\ttrue\t2026-05-29T221714_022\t86\t86\t74",
    ]
) + "\n"


def _settings(data_dir: Path) -> ApiSettings:
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


def _write_take(settings: ApiSettings, take_id: str, session_id: str) -> None:
    take_dir = settings.incoming_dir / take_id
    take_dir.mkdir(parents=True, exist_ok=True)
    (take_dir / "metadata.json").write_text(json.dumps({"take_id": take_id, "created_at": "2026-06-03T12:00:00Z", "session_id": session_id}), encoding="utf-8")
    (take_dir / "READY").touch()


def _setup_run(tmp_path: Path) -> tuple[IngestionWizardService, dict[str, str]]:
    settings = _settings(tmp_path / "data")
    service = DatasetService(settings.data_dir)
    service.create_dataset(dataset_id="d1", name="Dataset")
    service.create_session(dataset_id="d1", session_id="s1", name="Session 1")
    for take_id in ["2026-05-25T150423_025", "2026-05-25T150423_026", "2026-05-29T221714_022"]:
        _write_take(settings, take_id, "runtime")
        service.upsert_take_metadata(take_id=take_id, dataset_id="d1", session_id="s1", updates={}, source_metadata={"session_id": "runtime"})
    wizard = IngestionWizardService(settings.data_dir)
    run = wizard.create_run(dataset_id="d1", session_ids=["s1"], name="upload")
    return wizard, run


def test_uploaded_tsv_with_rich_columns_and_bom(tmp_path: Path) -> None:
    wizard, run = _setup_run(tmp_path)
    payload = wizard.ingest_uploaded_file(run["run_id"], filename=FIXTURE_NAME, content_bytes=RICH_TSV.encode("utf-8"))
    assert payload["provenance"]["source_type"] == "uploaded_file"
    assert payload["provenance"]["filename"] == FIXTURE_NAME
    assert payload["provenance"]["detected_format"] == "tsv"
    assert payload["provenance"]["row_count"] == 3
    assert payload["provenance"]["column_count"] == 12
    assert payload["diagnostics"]["required_field_inference"]["label_column"] is True
    assert payload["diagnostics"]["required_field_inference"]["image_ref_or_range"] is True


def test_uploaded_csv_fallback_and_delimiter_detection(tmp_path: Path) -> None:
    wizard, run = _setup_run(tmp_path)
    content = "\n".join([
        "label,image_ref,d1_mm",
        "good,2026-05-25T150423_025,91.5",
    ]) + "\n"
    payload = wizard.ingest_uploaded_file(run["run_id"], filename="fixture.csv", content_bytes=content.encode("utf-8"))
    assert payload["provenance"]["detected_format"] == "csv"
    assert payload["rows"][0]["label"] == "good"


def test_uploaded_semicolon_csv_detection(tmp_path: Path) -> None:
    wizard, run = _setup_run(tmp_path)
    content = "\n".join([
        "label;image_ref;d1_mm",
        "good;2026-05-25T150423_025;91.5",
    ]) + "\n"
    payload = wizard.ingest_uploaded_file(run["run_id"], filename="fixture.txt", content_bytes=content.encode("utf-8"))
    assert payload["provenance"]["detected_format"] == "csv_semicolon"
    assert payload["rows"][0]["image_ref"] == "2026-05-25T150423_025"


def test_uploaded_file_missing_required_columns(tmp_path: Path) -> None:
    wizard, run = _setup_run(tmp_path)
    content = "foo\tbar\n1\t2\n"
    with pytest.raises(ValueError, match="Missing label column"):
        wizard.ingest_uploaded_file(run["run_id"], filename="bad.tsv", content_bytes=content.encode("utf-8"))


def test_uploaded_file_empty_rejected(tmp_path: Path) -> None:
    wizard, run = _setup_run(tmp_path)
    with pytest.raises(ValueError, match="empty"):
        wizard.ingest_uploaded_file(run["run_id"], filename="empty.tsv", content_bytes=b"")


def test_uploaded_manifest_includes_provenance(tmp_path: Path) -> None:
    wizard, run = _setup_run(tmp_path)
    wizard.ingest_uploaded_file(run["run_id"], filename=FIXTURE_NAME, content_bytes=RICH_TSV.encode("utf-8"))
    wizard.reconcile(run["run_id"])
    manifest = wizard.generate_manifest(run["run_id"])
    rows = list(csv.DictReader(Path(manifest["files"]["label_manifest_csv"]).open(encoding="utf-8")))
    assert rows[0]["source_type"] == "uploaded_file"
    assert rows[0]["source_filename"] == FIXTURE_NAME
    assert rows[0]["source_detected_format"] == "tsv"
    assert rows[0]["source_row_count"] == "3"
