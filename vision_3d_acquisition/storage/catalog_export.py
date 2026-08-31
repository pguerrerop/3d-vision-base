"""Write the catalog back out as the directory layout it replaced.

The catalog holds documents that exist nowhere else after stage 2 — labels,
validation state, ML set membership, physical objects. This is the way back out:
a readable snapshot, an interchange format, and the exit if the decision to move
to SQLite is ever reversed.

It writes the same layout ``FilesystemDocuments`` reads, so an export can be
restored by pointing a data directory at it with ``SENSOR_STUDIO_INDEX=off``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class ExportReport:
    destination: str = ""
    datasets: int = 0
    sessions: int = 0
    takes: int = 0
    ml_sets: int = 0
    memberships: int = 0
    physical_objects: int = 0
    process_runs: int = 0
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def export_catalog(data_dir: Path, destination: Path) -> ExportReport:
    """Dump every authoritative document under ``destination``."""
    from vision_3d_acquisition.datasets.documents import FilesystemDocuments
    from vision_3d_acquisition.storage import run_store
    from vision_3d_acquisition.storage.dataset_store import SqlDocuments

    source = SqlDocuments(Path(data_dir))
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    target = FilesystemDocuments(destination)

    report = ExportReport(destination=str(destination))

    for dataset in source.list_datasets():
        dataset_id = str(dataset.get("id") or "")
        if not dataset_id:
            report.warnings.append("skipped a dataset with no id")
            continue
        target.write_dataset(dataset_id, dataset)
        report.datasets += 1

        for session in source.list_sessions(dataset_id):
            session_id = str(session.get("id") or "")
            if not session_id:
                continue
            target.write_session(dataset_id, session_id, session)
            report.sessions += 1

        for session_id, take_id, payload in source.iter_dataset_takes(dataset_id):
            target.write_take(dataset_id, session_id, take_id, payload)
            report.takes += 1

        for ml_set in source.list_ml_sets(dataset_id):
            ml_set_id = str(ml_set.get("id") or "")
            if not ml_set_id:
                continue
            target.write_ml_set(dataset_id, ml_set_id, ml_set)
            report.ml_sets += 1
            memberships = source.read_memberships(dataset_id, ml_set_id)
            target.write_memberships(dataset_id, ml_set_id, memberships)
            report.memberships += len(memberships)

        for obj in source.list_physical_objects(dataset_id):
            object_id = str(obj.get("physical_object_id") or obj.get("id") or "")
            if not object_id:
                continue
            target.write_physical_object(dataset_id, object_id, obj)
            report.physical_objects += 1

    runs = run_store.load_runs(Path(data_dir))
    runs_path = destination / "processes" / "index" / "runs.json"
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    runs_path.write_text(json.dumps({"entries": runs}, indent=2), encoding="utf-8")
    report.process_runs = len(runs)

    (destination / "EXPORT.json").write_text(
        json.dumps(
            {
                "exported_at": datetime.now(UTC).isoformat(),
                "source": str(Path(data_dir).resolve()),
                "counts": {
                    key: value
                    for key, value in report.as_dict().items()
                    if isinstance(value, int)
                },
                "restore": (
                    "Copy datasets/ and processes/ into a data directory and run with"
                    " SENSOR_STUDIO_INDEX=off, or delete index.db and let the one-shot"
                    " import pick them up."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return report
