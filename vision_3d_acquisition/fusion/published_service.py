from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vision_3d_acquisition.acquisition.groups import AcquisitionGroupService
from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.fusion.models import FusionResult
from vision_3d_acquisition.fusion.published_models import PublishedInspectionObject, PublishedInspectionResult
from vision_3d_acquisition.fusion.service import FusionService


class PublishedInspectionResultService:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.root = data_dir / "published" / "inspection_results"
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"

    def publish_fusion_result(self, fusion_run_id: str, station_id: str | None = None, persist: bool = True) -> dict[str, Any]:
        fusion_result = FusionService(_build_settings(self.data_dir)).get_run(fusion_run_id)
        if fusion_result is None:
            raise KeyError(f"Unknown fusion run: {fusion_run_id}")
        return self.publish_from_fusion_result(fusion_result=fusion_result, station_id=station_id, persist=persist)

    def publish_from_25d_result(
        self,
        *,
        take_id: str,
        result_payload: dict[str, Any],
        station_id: str | None = None,
        persist: bool = True,
    ) -> dict[str, Any]:
        acquisition_group_id = str(result_payload.get("acquisition_group_id") or f"take::{take_id}")
        fusion_run_id = str(result_payload.get("run_id") or f"25d_take_{take_id}")
        published_result_id = f"published_25d_{take_id}"
        objects_raw = result_payload.get("objects") if isinstance(result_payload.get("objects"), list) else []
        warnings = [str(item) for item in (result_payload.get("warnings") or []) if str(item)] if isinstance(result_payload.get("warnings"), list) else []
        objects: list[dict[str, Any]] = []
        for index, item in enumerate(objects_raw, start=1):
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or item.get("class_name") or "Unknown")
            group = _class_group_from_label(label)
            conf = _as_optional_float(item.get("confidence"))
            measurements = {
                "max_height_mm": (item.get("height_above_belt_mm") or {}).get("max_height_mm"),
                "p95_height_mm": (item.get("height_above_belt_mm") or {}).get("p95_height_mm"),
                "footprint_area_mm2": item.get("footprint_area_mm2"),
                "diameter_mm": item.get("diameter_mm"),
            }
            objects.append(
                PublishedInspectionObject(
                    final_object_id=str(item.get("object_id") or f"obj_{index}"),
                    final_class=label,
                    final_class_group=group,
                    confidence=conf,
                    decision_reasons=["Published from 25D-only run"],
                    key_measurements={k: v for k, v in measurements.items() if v is not None},
                    display_label=label if conf is None else f"{label} ({conf:.2f})",
                    sort_order=index,
                ).model_dump(mode="json")
            )

        class_counts = self._build_class_counts(objects)
        primary_class, primary_group = self._resolve_primary_class(objects, class_counts) if objects else (None, None)
        confidence = self._resolve_confidence(objects)
        status = "complete" if objects else "incomplete"
        overall_decision = self._resolve_overall_decision(status=status, objects=objects, warnings=warnings, confidence=confidence)
        payload = PublishedInspectionResult(
            published_result_id=published_result_id,
            acquisition_group_id=acquisition_group_id,
            fusion_run_id=fusion_run_id,
            station_id=station_id,
            session_id=str(result_payload.get("session_id") or "") or None,
            timestamp=datetime.now(UTC).isoformat(),
            status=status,  # type: ignore[arg-type]
            overall_decision=overall_decision,  # type: ignore[arg-type]
            primary_class=primary_class,
            primary_class_group=primary_group,
            confidence=confidence,
            class_counts=class_counts,
            objects=objects,
            warnings=warnings,
            display_artifacts={
                "main_overlay": (result_payload.get("files") or {}).get("overlay"),
                "heightmap_overlay": (result_payload.get("files") or {}).get("overlay"),
                "summary_json": str((self.root / published_result_id / "display_summary.json").relative_to(self.data_dir)),
            },
            source_refs={
                "result_mode": "25d_only",
                "take_id": take_id,
                "pipeline_id": ((result_payload.get("processing_pipeline") or {}).get("id") or "mining_steel_ball_classification_25d"),
                "acquisition_group_id": acquisition_group_id,
            },
        ).model_dump(mode="json")
        if persist:
            self._persist_published_result(payload)
            self._update_index(payload)
        return payload

    def publish_from_fusion_result(
        self,
        fusion_result: dict[str, Any],
        station_id: str | None = None,
        persist: bool = True,
    ) -> dict[str, Any]:
        parsed = FusionResult(**fusion_result)
        group = AcquisitionGroupService(self.data_dir).get_acquisition_group(parsed.acquisition_group_id) or {}
        resolved_station_id = station_id or (str(group.get("station_id") or "") or None)
        metadata = group.get("metadata") if isinstance(group.get("metadata"), dict) else {}
        session_id = str(metadata.get("session_id") or "") or None

        published_result_id = f"published_{parsed.fusion_run_id}"
        warnings = self._collect_warnings(fusion_result)
        status = self._resolve_status(fusion_result, warnings)
        objects = self._build_objects(parsed.final_objects)
        class_counts = self._build_class_counts(objects)
        primary_class, primary_group = self._resolve_primary_class(objects, class_counts)
        confidence = self._resolve_confidence(objects)
        overall_decision = self._resolve_overall_decision(status=status, objects=objects, warnings=warnings, confidence=confidence)

        payload = PublishedInspectionResult(
            published_result_id=published_result_id,
            acquisition_group_id=parsed.acquisition_group_id,
            fusion_run_id=parsed.fusion_run_id,
            station_id=resolved_station_id,
            session_id=session_id,
            timestamp=datetime.now(UTC).isoformat(),
            status=status,
            overall_decision=overall_decision,
            primary_class=primary_class,
            primary_class_group=primary_group,
            confidence=confidence,
            class_counts=class_counts,
            objects=objects,
            warnings=warnings,
            display_artifacts=self._build_display_artifacts(parsed, published_result_id),
            source_refs=self._build_source_refs(parsed, fusion_result),
        ).model_dump(mode="json")

        if persist:
            self._persist_published_result(payload)
            self._update_index(payload)
        return payload

    def get_published_result(self, published_result_id: str) -> dict[str, Any] | None:
        payload = _read_json(self.root / published_result_id / "published_result.json")
        return payload if isinstance(payload, dict) else None

    def get_latest_published_result(self, station_id: str | None = None, session_id: str | None = None) -> dict[str, Any] | None:
        index = self._read_index()
        if station_id:
            latest_by_station = index.get("latest_by_station")
            if not isinstance(latest_by_station, dict):
                return None
            result_id = latest_by_station.get(station_id)
            return self.get_published_result(result_id) if isinstance(result_id, str) else None
        if session_id:
            latest_by_session = index.get("latest_by_session")
            if not isinstance(latest_by_session, dict):
                return None
            result_id = latest_by_session.get(session_id)
            return self.get_published_result(result_id) if isinstance(result_id, str) else None
        latest_global = index.get("latest_global")
        return self.get_published_result(latest_global) if isinstance(latest_global, str) else None

    def list_published_results(
        self,
        station_id: str | None = None,
        session_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        safe_limit = max(1, min(500, int(limit)))
        entries = self._read_index().get("recent")
        if not isinstance(entries, list):
            return []
        filtered_ids: list[str] = []
        for item in entries:
            if not isinstance(item, dict):
                continue
            if station_id and str(item.get("station_id") or "") != station_id:
                continue
            if session_id and str(item.get("session_id") or "") != session_id:
                continue
            result_id = str(item.get("published_result_id") or "")
            if result_id:
                filtered_ids.append(result_id)
        payloads: list[dict[str, Any]] = []
        for result_id in filtered_ids[:safe_limit]:
            payload = self.get_published_result(result_id)
            if payload is not None:
                payloads.append(payload)
        return payloads

    def latest_for_acquisition_group(self, acquisition_group_id: str) -> dict[str, Any] | None:
        for item in self.list_published_results(limit=500):
            if str(item.get("acquisition_group_id") or "") == acquisition_group_id:
                return item
        return None

    def _update_index(self, payload: dict[str, Any]) -> None:
        entries = self._read_index().get("recent")
        recent = entries if isinstance(entries, list) else []
        recent = [
            item
            for item in recent
            if isinstance(item, dict) and str(item.get("published_result_id") or "") != str(payload.get("published_result_id") or "")
        ]
        recent.append(
            {
                "published_result_id": payload["published_result_id"],
                "acquisition_group_id": payload["acquisition_group_id"],
                "fusion_run_id": payload["fusion_run_id"],
                "station_id": payload.get("station_id"),
                "session_id": payload.get("session_id"),
                "timestamp": payload["timestamp"],
                "status": payload["status"],
                "overall_decision": payload["overall_decision"],
                "primary_class": payload.get("primary_class"),
                "confidence": payload.get("confidence"),
                "path": str(self.root / payload["published_result_id"] / "published_result.json"),
            }
        )
        recent.sort(key=lambda item: str(item.get("timestamp") or ""), reverse=True)

        latest_by_station: dict[str, str] = {}
        latest_by_session: dict[str, str] = {}
        for item in recent:
            pid = str(item.get("published_result_id") or "")
            if not pid:
                continue
            station_id = str(item.get("station_id") or "")
            session_id = str(item.get("session_id") or "")
            if station_id and station_id not in latest_by_station:
                latest_by_station[station_id] = pid
            if session_id and session_id not in latest_by_session:
                latest_by_session[session_id] = pid

        index_payload = {
            "latest_global": recent[0]["published_result_id"] if recent else None,
            "latest_by_station": latest_by_station,
            "latest_by_session": latest_by_session,
            "recent": recent[:500],
            "updated_at": datetime.now(UTC).isoformat(),
        }
        self.index_path.write_text(json.dumps(index_payload, indent=2) + "\n", encoding="utf-8")

    def _read_index(self) -> dict[str, Any]:
        payload = _read_json(self.index_path)
        if not isinstance(payload, dict):
            return {"latest_global": None, "latest_by_station": {}, "latest_by_session": {}, "recent": []}
        return {
            "latest_global": payload.get("latest_global"),
            "latest_by_station": payload.get("latest_by_station") if isinstance(payload.get("latest_by_station"), dict) else {},
            "latest_by_session": payload.get("latest_by_session") if isinstance(payload.get("latest_by_session"), dict) else {},
            "recent": payload.get("recent") if isinstance(payload.get("recent"), list) else [],
        }

    def _persist_published_result(self, payload: dict[str, Any]) -> None:
        result_dir = self.root / payload["published_result_id"]
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / "published_result.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        (result_dir / "display_summary.json").write_text(
            json.dumps(
                {
                    "published_result_id": payload["published_result_id"],
                    "overall_decision": payload["overall_decision"],
                    "primary_class": payload.get("primary_class"),
                    "confidence": payload.get("confidence"),
                    "class_counts": payload.get("class_counts", {}),
                    "warnings": payload.get("warnings", []),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def _build_objects(self, final_objects: list[Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for index, raw in enumerate(final_objects, start=1):
            obj = raw if isinstance(raw, dict) else raw.model_dump(mode="json")
            final_class = str(obj.get("final_class") or "Unknown")
            confidence = _as_optional_float(obj.get("confidence"))
            out.append(
                PublishedInspectionObject(
                    final_object_id=str(obj.get("id") or f"fobj_{index}"),
                    final_class=final_class,
                    final_class_group=str(obj.get("final_class_group") or "Unknown"),
                    confidence=confidence,
                    decision_reasons=[str(v) for v in (obj.get("decision_reasons") or []) if str(v)],
                    key_measurements=dict(obj.get("measurements") or {}) if isinstance(obj.get("measurements"), dict) else {},
                    display_label=final_class if confidence is None else f"{final_class} ({confidence:.2f})",
                    sort_order=index,
                ).model_dump(mode="json")
            )
        return out

    def _build_class_counts(self, objects: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for obj in objects:
            class_name = str(obj.get("final_class") or "Unknown")
            counts[class_name] = counts.get(class_name, 0) + 1
        return counts

    def _resolve_primary_class(self, objects: list[dict[str, Any]], class_counts: dict[str, int]) -> tuple[str | None, str | None]:
        if not objects:
            return None, None
        winner = max(class_counts.items(), key=lambda item: item[1])[0] if class_counts else str(objects[0].get("final_class") or "Unknown")
        primary = next((obj for obj in objects if str(obj.get("final_class") or "") == winner), objects[0])
        return str(primary.get("final_class") or "Unknown"), str(primary.get("final_class_group") or "Unknown")

    def _resolve_confidence(self, objects: list[dict[str, Any]]) -> float | None:
        nums = [_as_optional_float(obj.get("confidence")) for obj in objects]
        numeric = [value for value in nums if value is not None]
        return round(sum(numeric) / len(numeric), 6) if numeric else None

    def _collect_warnings(self, fusion_result: dict[str, Any]) -> list[str]:
        warnings: list[str] = []
        if isinstance(fusion_result.get("warnings"), list):
            warnings.extend(str(item) for item in fusion_result["warnings"] if str(item))
        diagnostics = fusion_result.get("diagnostics") if isinstance(fusion_result.get("diagnostics"), dict) else {}
        message = str(diagnostics.get("message") or "")
        if message and message != "ok":
            warnings.append(message)
        if not isinstance(fusion_result.get("final_objects"), list):
            warnings.append("Fusion result has no final objects list")
        deduped: list[str] = []
        seen: set[str] = set()
        for warning in warnings:
            if warning not in seen:
                seen.add(warning)
                deduped.append(warning)
        return deduped

    def _resolve_status(self, fusion_result: dict[str, Any], warnings: list[str]) -> str:
        raw_status = str(fusion_result.get("status") or "").lower()
        if raw_status in {"pending", "complete", "incomplete", "failed"}:
            return raw_status
        if raw_status in {"error", "failed"}:
            return "failed"
        return "incomplete" if warnings else "complete"

    def _resolve_overall_decision(
        self,
        *,
        status: str,
        objects: list[dict[str, Any]],
        warnings: list[str],
        confidence: float | None,
    ) -> str:
        if status in {"failed", "pending"} or not objects:
            return "unknown"
        if warnings or (confidence is not None and confidence < 0.8):
            return "review"
        classes = {str(obj.get("final_class") or "") for obj in objects}
        groups = {str(obj.get("final_class_group") or "") for obj in objects}
        if any("scrap de bola" in value.lower() or "chatarra" in value.lower() for value in classes | groups):
            return "reject"
        if classes and all(value.lower() == "bola buena" for value in classes):
            return "accept"
        return "review"

    def _build_display_artifacts(self, fusion_result: FusionResult, published_result_id: str) -> dict[str, str | None]:
        fusion_overlay = _extract_fusion_overlay_ref(fusion_result.diagnostics)
        rgb_overlay = _extract_overlay_from_objects(fusion_result.final_objects, "2d_overlay_artifact_id")
        heightmap_overlay = _extract_overlay_from_objects(fusion_result.final_objects, "25d_overlay_artifact_id")
        return {
            "main_overlay": fusion_overlay or rgb_overlay or heightmap_overlay,
            "rgb_overlay": rgb_overlay,
            "heightmap_overlay": heightmap_overlay,
            "summary_json": str((self.root / published_result_id / "display_summary.json").relative_to(self.data_dir)),
        }

    def _build_source_refs(self, fusion_result: FusionResult, raw_payload: dict[str, Any]) -> dict[str, Any]:
        run_dir = self.data_dir / "fusion" / "runs" / fusion_result.fusion_run_id
        return {
            "fusion_result_id": fusion_result.id,
            "fusion_run_id": fusion_result.fusion_run_id,
            "fusion_run_dir": str(run_dir.relative_to(self.data_dir)),
            "fusion_result_path": str((run_dir / "fusion_result.json").relative_to(self.data_dir)),
            "acquisition_group_id": fusion_result.acquisition_group_id,
            "matched_object_count": len(raw_payload.get("matched_objects") or []),
            "final_object_count": len(fusion_result.final_objects),
        }


def publish_fusion_result(
    data_dir: Path,
    fusion_run_id: str,
    station_id: str | None = None,
    persist: bool = True,
) -> PublishedInspectionResult:
    payload = PublishedInspectionResultService(data_dir).publish_fusion_result(
        fusion_run_id=fusion_run_id,
        station_id=station_id,
        persist=persist,
    )
    return PublishedInspectionResult(**payload)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _build_settings(data_dir: Path) -> ApiSettings:
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


def _extract_overlay_from_objects(final_objects: list[Any], key: str) -> str | None:
    for raw in final_objects:
        obj = raw if isinstance(raw, dict) else raw.model_dump(mode="json")
        artifacts = obj.get("artifacts") if isinstance(obj.get("artifacts"), dict) else {}
        value = str(artifacts.get(key) or "")
        if value:
            return value
    return None


def _class_group_from_label(label: str) -> str:
    value = label.lower()
    if "bola buena" in value:
        return "Bola buena"
    if "scrap de bola" in value or "bola con chip" in value or "bola partida" in value or "bola deformada" in value:
        return "Scrap de Bola"
    if "chatarra" in value or "pernos" in value or "planchuelas" in value or "mallas" in value:
        return "Chatarra"
    return "Unknown"


def _extract_fusion_overlay_ref(diagnostics: dict[str, Any]) -> str | None:
    for key in ("fusion_overlay", "fusion_overlay_path", "overlay_path"):
        value = str(diagnostics.get(key) or "")
        if value:
            return value
    artifacts = diagnostics.get("artifacts") if isinstance(diagnostics.get("artifacts"), dict) else {}
    for key in ("fusion_overlay", "rgb_overlay", "heightmap_overlay"):
        value = str(artifacts.get(key) or "")
        if value:
            return value
    return None


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
