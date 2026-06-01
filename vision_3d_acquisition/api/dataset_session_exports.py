from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from vision_3d_acquisition.api.filesystem import get_take_detail, list_takes_paged
from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.datasets import DatasetService


class DatasetSessionExportService:
    """Build deterministic dataset-session exports without coupling to UI DTOs."""

    def __init__(self, settings: ApiSettings) -> None:
        self.settings = settings
        self.dataset_service = DatasetService(settings.data_dir)

    def resolve_session(self, session_id: str, dataset_id: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        sessions = self.dataset_service.list_sessions(dataset_id=dataset_id)
        session = next((item for item in sessions if str(item.get("id") or "") == session_id), None)
        if not isinstance(session, dict):
            if dataset_id:
                raise KeyError(f"Unknown session: {session_id}")
            # fallback across datasets
            all_sessions = self.dataset_service.list_sessions()
            session = next((item for item in all_sessions if str(item.get("id") or "") == session_id), None)
            if not isinstance(session, dict):
                raise KeyError(f"Unknown session: {session_id}")
        resolved_dataset_id = str(session.get("dataset_id") or dataset_id or "")
        dataset = self.dataset_service.get_dataset(resolved_dataset_id)
        if not isinstance(dataset, dict):
            raise KeyError(f"Unknown dataset for session: {session_id}")
        return dataset, session

    def build_session_summary(self, session_id: str, dataset_id: str | None = None) -> dict[str, Any]:
        dataset, session = self.resolve_session(session_id, dataset_id)
        canonical_session_id = str(session.get("id") or session_id)
        session_dataset_id = str(dataset.get("id") or "")
        page = list_takes_paged(
            self.settings,
            limit=1,
            offset=0,
            dataset_id=session_dataset_id,
            session_id=canonical_session_id,
            include_archived=False,
        )
        filtered_count = int(page.get("filtered_count") or 0)
        summary_counts = page.get("summary_counts") if isinstance(page.get("summary_counts"), dict) else {}
        takes = self._list_canonical_session_takes(dataset_id=session_dataset_id, session_id=canonical_session_id)
        split_distribution: dict[str, int] = {}
        latest_created_at = None
        processing_by_family: dict[str, int] = {"completed": 0, "running": 0, "failed": 0, "partial": 0, "never_processed": 0}
        object_annotation_count = 0
        object_candidate_count = 0
        for take in takes:
            split_value = str(getattr(take, "split", "") or "")
            if not split_value:
                split_value = str(((take.model_dump(mode="json").get("take_metadata") or {}).get("split") or ""))
            split_distribution[split_value or "unassigned"] = int(split_distribution.get(split_value or "unassigned", 0)) + 1
            if take.created_at and (latest_created_at is None or str(take.created_at) > str(latest_created_at)):
                latest_created_at = take.created_at
            for family_state in (take.processing_by_family or []):
                status = str((family_state or {}).get("status") or "never_processed")
                processing_by_family[status] = int(processing_by_family.get(status, 0)) + 1
            detail = get_take_detail(self.settings, take.take_id)
            if detail is not None:
                object_annotation_count += len(detail.object_annotations or [])
                object_candidate_count += len((detail.result or {}).get("object_candidates") or []) if isinstance(detail.result, dict) else 0

        reviewed = int((summary_counts.get("validation") or {}).get("validated") or 0)
        rejected = int((summary_counts.get("validation") or {}).get("rejected") or 0)
        needs_review = int((summary_counts.get("validation") or {}).get("needs_review") or 0)
        benchmark_approved = int((summary_counts.get("validation") or {}).get("benchmark_approved") or 0)
        golden_sample = int((summary_counts.get("validation") or {}).get("golden_sample") or 0)
        unreviewed = int((summary_counts.get("validation") or {}).get("unreviewed") or 0)

        return {
            "dataset": dataset,
            "session": session,
            "summary": {
                "total_takes": filtered_count,
                "reviewed": reviewed,
                "unreviewed": unreviewed,
                "rejected": rejected,
                "needs_review": needs_review,
                "golden_sample": golden_sample,
                "benchmark_approved": benchmark_approved,
                "missing_labels": int(summary_counts.get("missing_labels") or 0),
                "missing_split": int(summary_counts.get("missing_split") or 0),
                "missing_calibration": int(summary_counts.get("missing_calibration") or 0),
                "processing_failed": int(summary_counts.get("processing_failed") or 0),
                "processing_incomplete": int(summary_counts.get("processing_incomplete") or 0),
                "no_objects_detected": int(summary_counts.get("no_objects_detected") or 0),
                "split_distribution": split_distribution,
                "processing_by_family": processing_by_family,
                "object_annotation_count": object_annotation_count,
                "object_candidate_count": object_candidate_count,
                "last_acquisition_timestamp": latest_created_at,
                "calibration_assignment": str(session.get("calibration_id") or ""),
                "session_id": canonical_session_id,
                "dataset_id": session_dataset_id,
            },
        }

    def export_session(self, session_id: str, dataset_id: str | None = None) -> dict[str, Any]:
        summary_payload = self.build_session_summary(session_id, dataset_id)
        dataset = summary_payload["dataset"]
        session = summary_payload["session"]
        dataset_id_value = str(dataset.get("id") or "")
        canonical_session_id = str(session.get("id") or session_id)
        takes = self._list_canonical_session_takes(dataset_id=dataset_id_value, session_id=canonical_session_id)
        ordered = sorted(takes, key=lambda item: (str(item.created_at or ""), item.take_id))
        exported_takes: list[dict[str, Any]] = []
        for take in ordered:
            detail = get_take_detail(self.settings, take.take_id)
            if detail is None:
                continue
            take_meta = detail.take_metadata if isinstance(detail.take_metadata, dict) else {}
            exported_takes.append(
                {
                    "take_id": take.take_id,
                    "metadata": take_meta,
                    "processing_by_family": take.processing_by_family or [],
                    "labels": detail.labels if isinstance(detail.labels, dict) else {},
                    "object_annotations": detail.object_annotations or [],
                    "assets": detail.assets or {},
                    "created_at": take.created_at,
                    "thumbnail_path": take.thumbnail_path,
                }
            )
        return {
            "export_type": "dataset_session",
            "schema_version": "1.0",
            "exported_at": datetime.now(UTC).isoformat(),
            "dataset": dataset,
            "session": session,
            "summary": summary_payload["summary"],
            "takes": exported_takes,
        }

    def _list_canonical_session_takes(self, *, dataset_id: str, session_id: str) -> list[Any]:
        """Return takes strictly matched on dataset/take metadata session ownership."""
        all_dataset_takes = list_takes_paged(
            self.settings,
            dataset_id=dataset_id,
            limit=200,
            offset=0,
            include_archived=False,
        )
        rows = list(all_dataset_takes.get("items") or [])
        next_offset = int(all_dataset_takes.get("next_offset") or len(rows))
        while bool(all_dataset_takes.get("has_more")):
            all_dataset_takes = list_takes_paged(
                self.settings,
                dataset_id=dataset_id,
                limit=200,
                offset=next_offset,
                include_archived=False,
            )
            chunk = list(all_dataset_takes.get("items") or [])
            rows.extend(chunk)
            next_offset = int(all_dataset_takes.get("next_offset") or (next_offset + len(chunk)))
        return [take for take in rows if str(getattr(take, "experiment_session_id", "") or "") == session_id]
