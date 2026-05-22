from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from vision_3d_acquisition.acquisition.groups import AcquisitionGroupService
from vision_3d_acquisition.api.filesystem import get_take_detail
from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.fusion.models import FinalObject, FusionResult
from vision_3d_acquisition.processing.status_index import append_process_run_index

DEFAULT_FUSION_CONFIG: dict[str, Any] = {"centroid_distance_threshold_px": 45.0, "min_bbox_iou": 0.15, "use_bbox_iou": True}


class FusionService:
    def __init__(self, settings: ApiSettings) -> None:
        self.settings = settings
        self.runs_dir = settings.data_dir / "fusion" / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def run_fusion(
        self,
        *,
        acquisition_group_id: str,
        recipe_version_id: str | None = None,
        force: bool = False,
        rules: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        group = AcquisitionGroupService(self.settings.data_dir).get_acquisition_group(acquisition_group_id)
        if not group:
            raise KeyError(f"Unknown acquisition group: {acquisition_group_id}")
        config = dict(DEFAULT_FUSION_CONFIG)
        config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        run_id = f"fusion_run_{datetime.now(UTC).strftime('%Y%m%d%H%M%S%f')}"

        takes = AcquisitionGroupService(self.settings.data_dir).list_group_takes(acquisition_group_id)
        rgb_candidates: list[dict[str, Any]] = []
        hm_candidates: list[dict[str, Any]] = []
        for item in takes:
            take_id = str(item.get("take_id") or "")
            if not take_id:
                continue
            detail = get_take_detail(self.settings, take_id)
            if detail is None or not isinstance(detail.result, dict):
                continue
            result = detail.result
            fam = str(((result.get("processing_pipeline") or {}).get("pipeline_family") or ""))
            status = str(result.get("status") or "")
            if status not in {"ok", "success", "warning", "completed"}:
                continue
            candidates = [c for c in (result.get("object_candidates") or []) if isinstance(c, dict)]
            if fam == "2d":
                rgb_candidates = candidates
            elif fam == "25d":
                hm_candidates = candidates

        matched, unmatched_rgb, unmatched_hm = self._match_candidates(rgb_candidates, hm_candidates, config)
        final_objects = [self._fuse_pair(acquisition_group_id, i + 1, p[0], p[1]) for i, p in enumerate(matched)]
        for c in unmatched_rgb:
            final_objects.append(self._fuse_pair(acquisition_group_id, len(final_objects) + 1, c, None))
        for c in unmatched_hm:
            final_objects.append(self._fuse_pair(acquisition_group_id, len(final_objects) + 1, None, c))

        warnings: list[str] = []
        if not rgb_candidates:
            warnings.append("No 2D candidates available for fusion")
        if not hm_candidates:
            warnings.append("No 2.5D candidates available for fusion")
        if force and warnings:
            warnings.append("Fusion forced with missing modality candidates")
        payload = FusionResult(
            id=f"fusion_{run_id}",
            acquisition_group_id=acquisition_group_id,
            fusion_run_id=run_id,
            matched_objects=[{"candidate_2d_id": a.get("id"), "candidate_25d_id": b.get("id")} for a, b in matched],
            unmatched_2d_candidates=unmatched_rgb,
            unmatched_25d_candidates=unmatched_hm,
            final_objects=final_objects,
            diagnostics={
                "recipe_version_id": recipe_version_id,
                "config_hash": config_hash,
                "config": config,
                "candidate_counts": {"2d": len(rgb_candidates), "25d": len(hm_candidates), "matched": len(matched)},
                "rules": rules or {},
                "message": "missing candidates" if (not rgb_candidates or not hm_candidates) else "ok",
            },
        ).model_dump(mode="json")
        payload["status"] = "ok"
        payload["warnings"] = warnings

        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "fusion_result.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        (run_dir / "fusion_run.json").write_text(json.dumps({"fusion_run_id": run_id, "acquisition_group_id": acquisition_group_id, "created_at": datetime.now(UTC).isoformat()}, indent=2) + "\n", encoding="utf-8")
        (run_dir / "fusion_summary.json").write_text(
            json.dumps(
                {
                    "fusion_run_id": run_id,
                    "acquisition_group_id": acquisition_group_id,
                    "matched_count": len(matched),
                    "unmatched_2d_count": len(unmatched_rgb),
                    "unmatched_25d_count": len(unmatched_hm),
                    "final_object_count": len(final_objects),
                    "status": "incomplete" if warnings else "complete",
                    "warnings": warnings,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (run_dir / "fusion_object_table.json").write_text(
            json.dumps([item.model_dump(mode="json") for item in final_objects], indent=2) + "\n",
            encoding="utf-8",
        )
        (run_dir / "fusion_debug_pairing.json").write_text(
            json.dumps(
                {
                    "matched_objects": payload.get("matched_objects") or [],
                    "unmatched_2d_candidates": payload.get("unmatched_2d_candidates") or [],
                    "unmatched_25d_candidates": payload.get("unmatched_25d_candidates") or [],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        append_process_run_index(
            self.settings.data_dir,
            take_id=f"group::{acquisition_group_id}",
            pipeline_instance_id="fusion",
            run_id=run_id,
            pipeline_family="generic",
            status="success",
            run_dir=run_dir,
            created_at=datetime.now(UTC).isoformat(),
            pipeline_id="mining_steel_ball_fusion",
            recipe_version_id=recipe_version_id,
            config_snapshot_hash=config_hash,
            source_id=acquisition_group_id,
            acquisition_group_id=acquisition_group_id,
            calibration_profile_id=None,
        )
        return payload

    def get_run(self, fusion_run_id: str) -> dict[str, Any] | None:
        path = self.runs_dir / fusion_run_id / "fusion_result.json"
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def list_group_runs(self, acquisition_group_id: str) -> list[dict[str, Any]]:
        if not self.runs_dir.is_dir():
            return []
        runs: list[dict[str, Any]] = []
        for child in sorted(self.runs_dir.iterdir(), key=lambda p: p.name, reverse=True):
            path = child / "fusion_result.json"
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            if str(payload.get("acquisition_group_id") or "") != acquisition_group_id:
                continue
            runs.append(
                {
                    "fusion_run_id": payload.get("fusion_run_id"),
                    "acquisition_group_id": payload.get("acquisition_group_id"),
                    "result_id": payload.get("id"),
                    "status": payload.get("status") or "ok",
                    "warnings": payload.get("warnings") or [],
                }
            )
        return runs

    def latest_result_for_group(self, acquisition_group_id: str) -> dict[str, Any] | None:
        latest = self.latest_fusion_for_group(acquisition_group_id)
        if latest is None:
            return None
        return {
            "acquisition_group_id": acquisition_group_id,
            "status": "ok",
            "run": {
                "fusion_run_id": latest.get("fusion_run_id"),
                "result_id": latest.get("id"),
            },
            "result": latest,
        }

    def latest_fusion_for_group(self, acquisition_group_id: str) -> dict[str, Any] | None:
        if not self.runs_dir.is_dir():
            return None
        for child in sorted(self.runs_dir.iterdir(), key=lambda p: p.name, reverse=True):
            path = child / "fusion_result.json"
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if str(payload.get("acquisition_group_id") or "") == acquisition_group_id:
                return payload
        return None

    def _match_candidates(self, rgb: list[dict[str, Any]], hm: list[dict[str, Any]], config: dict[str, Any]) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
        matched: list[tuple[dict[str, Any], dict[str, Any]]] = []
        remaining_hm = hm[:]
        max_dist = float(config.get("centroid_distance_threshold_px") or 45.0)
        min_iou = float(config.get("min_bbox_iou") or 0.15)
        use_iou = bool(config.get("use_bbox_iou", True))
        for rc in rgb:
            rcent = _centroid(rc)
            best = None
            best_score = 1e12
            for hc in remaining_hm:
                hcent = _centroid(hc)
                if rcent and hcent:
                    dist = ((rcent[0] - hcent[0]) ** 2 + (rcent[1] - hcent[1]) ** 2) ** 0.5
                    if dist <= max_dist and dist < best_score:
                        best = hc
                        best_score = dist
                        continue
                if use_iou:
                    iou = _bbox_iou(_bbox(rc), _bbox(hc))
                    if iou >= min_iou and (1000.0 - iou) < best_score:
                        best = hc
                        best_score = 1000.0 - iou
            if best is not None:
                matched.append((rc, best))
                remaining_hm = [i for i in remaining_hm if i.get("id") != best.get("id")]
        matched_rgb_ids = {a.get("id") for a, _ in matched}
        return matched, [i for i in rgb if i.get("id") not in matched_rgb_ids], remaining_hm

    def _fuse_pair(self, group_id: str, idx: int, cand2d: dict[str, Any] | None, cand25d: dict[str, Any] | None) -> FinalObject:
        measurements: dict[str, Any] = {}
        geometry: dict[str, Any] = {}
        appearance: dict[str, Any] = {}
        if cand2d:
            measurements.update(cand2d.get("measurements") or {})
            geometry.update(cand2d.get("geometry") or {})
            appearance.update(cand2d.get("appearance") or {})
        if cand25d:
            measurements.update(cand25d.get("measurements") or {})
            geometry.update(cand25d.get("geometry") or {})
            appearance.update(cand25d.get("appearance") or {})
        cls, cls_group, reasons, conf = _classify(cand2d, cand25d)
        bbox = cand2d.get("bbox_px") if cand2d and cand2d.get("bbox_px") else (cand25d.get("bbox_px") if cand25d else None)
        centroid_px = cand2d.get("centroid_px") if cand2d and cand2d.get("centroid_px") else (cand25d.get("centroid_px") if cand25d else None)
        centroid_world = cand25d.get("centroid_world") if cand25d and cand25d.get("centroid_world") else None
        return FinalObject(
            id=f"fobj_{idx}",
            acquisition_group_id=group_id,
            candidate_2d_id=cand2d.get("id") if cand2d else None,
            candidate_25d_id=cand25d.get("id") if cand25d else None,
            bbox_px=bbox,
            centroid_px=centroid_px,
            centroid_world=centroid_world,
            measurements=measurements,
            appearance=appearance,
            geometry=geometry,
            final_class=cls,
            final_class_group=cls_group,
            confidence=conf,
            decision_reasons=reasons,
            artifacts={"2d_overlay_artifact_id": cand2d.get("overlay_artifact_id") if cand2d else None, "25d_overlay_artifact_id": cand25d.get("overlay_artifact_id") if cand25d else None},
            diagnostics={"has_2d": bool(cand2d), "has_25d": bool(cand25d)},
        )


def _centroid(candidate: dict[str, Any]) -> tuple[float, float] | None:
    c = candidate.get("centroid_px")
    if isinstance(c, list) and len(c) >= 2:
        try:
            return float(c[0]), float(c[1])
        except Exception:
            return None
    return None


def _bbox(candidate: dict[str, Any]) -> tuple[float, float, float, float] | None:
    b = candidate.get("bbox_px")
    if isinstance(b, list) and len(b) >= 4:
        try:
            return float(b[0]), float(b[1]), float(b[2]), float(b[3])
        except Exception:
            return None
    return None


def _bbox_iou(a: tuple[float, float, float, float] | None, b: tuple[float, float, float, float] | None) -> float:
    if a is None or b is None:
        return 0.0
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = max(1e-9, aw * ah + bw * bh - inter)
    return inter / union


def _classify(c2d: dict[str, Any] | None, c25d: dict[str, Any] | None) -> tuple[str, str, list[str], float | None]:
    reasons: list[str] = []
    if c2d and isinstance(c2d.get("classification_hints"), dict):
        label = str((c2d.get("classification_hints") or {}).get("class_label") or "")
        conf = c2d.get("confidence")
        if label in {"Bola buena", "Scrap de Bola", "Chatarra"} and isinstance(conf, (int, float)) and float(conf) >= 0.85:
            reasons.append("Strong 2D class hint preserved")
            return label, "Ball" if "Bola" in label else "Scrap", reasons, float(conf)
    if c25d:
        hints = c25d.get("classification_hints") if isinstance(c25d.get("classification_hints"), dict) else {}
        deform = float(hints.get("deformation_hint") or 0.0)
        flatness = float(hints.get("flatness_hint") or 0.0)
        roundness = float(hints.get("roundness_hint") or 0.0)
        max_h = float((c25d.get("measurements") or {}).get("height_max_mm") or 0.0)
        if deform >= 0.6:
            reasons.append("High 2.5D deformation hint")
            return "Scrap de Bola / Bola deformada", "Scrap de Bola", reasons, c25d.get("confidence")
        if roundness < 0.45:
            reasons.append("Low roundness in 2.5D")
            return "Scrap de Bola", "Scrap de Bola", reasons, c25d.get("confidence")
        if flatness > 0.9 and max_h < 4.0:
            reasons.append("Flat + low profile in 2.5D")
            return "Chatarra / Planchuelas", "Chatarra", reasons, c25d.get("confidence")
        reasons.append("2.5D hint defaulted to Bola buena")
        return "Bola buena", "Bola", reasons, c25d.get("confidence")
    if c2d:
        label = str((c2d.get("classification_hints") or {}).get("class_label") or "Unknown")
        reasons.append("Only 2D candidate available")
        return label or "Unknown", "Unknown", reasons, c2d.get("confidence")
    reasons.append("No candidate features")
    return "Unknown", "Unknown", reasons, None
