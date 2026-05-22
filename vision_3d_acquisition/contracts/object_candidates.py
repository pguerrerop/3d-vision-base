from __future__ import annotations

from typing import Any


def normalize_object_candidates(result_payload: dict[str, Any]) -> list[dict[str, Any]]:
    existing = result_payload.get("object_candidates")
    if isinstance(existing, list):
        return [_normalize_candidate(item, index=i + 1) for i, item in enumerate(existing) if isinstance(item, dict)]

    modalities = result_payload.get("input_modalities")
    if isinstance(modalities, list) and "rgb" in modalities:
        return extract_object_candidates_from_legacy_2d(result_payload)
    if isinstance(modalities, list) and any(item in modalities for item in ("heightmap", "point_cloud")):
        return extract_object_candidates_from_legacy_25d(result_payload)
    return []


def extract_object_candidates_from_legacy_2d(result_payload: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts = result_payload.get("artifacts") if isinstance(result_payload.get("artifacts"), list) else []
    by_id = {
        str(item.get("artifact_id") or ""): item
        for item in artifacts
        if isinstance(item, dict) and str(item.get("artifact_id") or "")
    }
    blob_rows = _artifact_entries(by_id.get("blob_metrics"))
    ellipse_rows = _artifact_entries(by_id.get("ellipse_metrics"))
    class_rows = _artifact_entries(by_id.get("classification_result")) or _artifact_entries(by_id.get("classification_result_artifact"))

    blob_by_id = {int(row.get("blob_id") or row.get("object_id") or 0): row for row in blob_rows if int(row.get("blob_id") or row.get("object_id") or 0) > 0}
    ellipse_by_id = {int(row.get("candidate_id") or row.get("object_id") or 0): row for row in ellipse_rows if int(row.get("candidate_id") or row.get("object_id") or 0) > 0}
    class_by_id = {int(row.get("object_id") or 0): row for row in class_rows if int(row.get("object_id") or 0) > 0}

    known = sorted(set(blob_by_id.keys()) | set(ellipse_by_id.keys()) | set(class_by_id.keys()))
    candidates: list[dict[str, Any]] = []
    for local_idx, key in enumerate(known, start=1):
        blob = blob_by_id.get(key, {})
        ell = ellipse_by_id.get(key, {})
        cls = class_by_id.get(key, {})
        contour = blob.get("contour_points") if isinstance(blob.get("contour_points"), list) else None
        bbox = blob.get("bbox") if isinstance(blob.get("bbox"), list | tuple) else None
        centroid = blob.get("centroid") if isinstance(blob.get("centroid"), list | tuple) else None
        geometry = {
            "area_px": blob.get("area_px"),
            "circularity": blob.get("circularity") if blob else ell.get("circularity"),
            "major_axis_px": ell.get("major_axis"),
            "minor_axis_px": ell.get("minor_axis"),
            "equivalent_diameter_px": ell.get("equivalent_diameter") or blob.get("equivalent_diameter"),
            "aspect_ratio": blob.get("aspect_ratio"),
            "solidity": blob.get("solidity"),
            "fit_rmse": ell.get("fit_rmse"),
            "eccentricity": ell.get("eccentricity"),
        }
        appearance = {
            "touches_border": blob.get("touches_border"),
            "contour_points": len(contour) if isinstance(contour, list) else None,
        }
        measurements = {
            "area_px": blob.get("area_px"),
            "perimeter_px": blob.get("contour_perimeter"),
            "equivalent_diameter_px": ell.get("equivalent_diameter") or blob.get("equivalent_diameter"),
        }
        classification_hints = {
            "class_label": cls.get("class_label"),
            "subclass_label": cls.get("subclass_label"),
            "decision_reasons": cls.get("decision_reasons"),
            "spherical_enough": cls.get("spherical_enough"),
        }
        candidates.append(
            _normalize_candidate(
                {
                    "id": f"rgb_cand_{key}",
                    "local_object_index": local_idx,
                    "source_modality": "rgb",
                    "source_take_id": result_payload.get("take_id"),
                    "source_pipeline_run_id": result_payload.get("run_id"),
                    "acquisition_group_id": result_payload.get("acquisition_group_id"),
                    "bbox_px": list(bbox) if isinstance(bbox, (list, tuple)) else None,
                    "contour_px": contour,
                    "mask_artifact_id": "cleaned_mask" if "cleaned_mask" in by_id else None,
                    "overlay_artifact_id": "classification_overlay_image" if "classification_overlay_image" in by_id else "overlay_preview_image" if "overlay_preview_image" in by_id else None,
                    "centroid_px": list(centroid) if isinstance(centroid, (list, tuple)) else None,
                    "centroid_world": None,
                    "geometry": geometry,
                    "appearance": appearance,
                    "measurements": measurements,
                    "classification_hints": classification_hints,
                    "confidence": cls.get("confidence"),
                    "diagnostics": {
                        "blob_rejection_reason": blob.get("rejection_reason"),
                        "ellipse_valid_fit": ell.get("valid_fit"),
                    },
                },
                index=local_idx,
            )
        )
    return candidates


def extract_object_candidates_from_legacy_25d(result_payload: dict[str, Any]) -> list[dict[str, Any]]:
    objects = result_payload.get("objects") if isinstance(result_payload.get("objects"), list) else []
    artifacts = result_payload.get("artifacts") if isinstance(result_payload.get("artifacts"), list) else []
    by_id = {str(item.get("artifact_id") or ""): item for item in artifacts if isinstance(item, dict)}
    candidates: list[dict[str, Any]] = []
    for idx, item in enumerate(objects, start=1):
        if not isinstance(item, dict):
            continue
        oid = int(item.get("object_id") or idx)
        h = item.get("height_above_belt_mm") if isinstance(item.get("height_above_belt_mm"), dict) else {}
        hints = {
            "deformation_hint": item.get("feature_height_asymmetry"),
            "flatness_hint": item.get("feature_flatness"),
            "roundness_hint": item.get("sphericity_score"),
            "class_label": item.get("label") or item.get("class_name"),
            "class_group": item.get("class_group"),
        }
        candidates.append(
            _normalize_candidate(
                {
                    "id": f"hmap_cand_{oid}",
                    "local_object_index": idx,
                    "source_modality": "derived_25d",
                    "source_take_id": result_payload.get("take_id"),
                    "source_pipeline_run_id": result_payload.get("run_id"),
                    "acquisition_group_id": result_payload.get("acquisition_group_id"),
                    "bbox_px": item.get("bbox_px") if isinstance(item.get("bbox_px"), list) else None,
                    "contour_px": item.get("contour_px") if isinstance(item.get("contour_px"), list) else None,
                    "mask_artifact_id": "cleaned_mask" if "cleaned_mask" in by_id else None,
                    "overlay_artifact_id": "classification_overlay" if "classification_overlay" in by_id else "measurement_overlay" if "measurement_overlay" in by_id else None,
                    "centroid_px": item.get("centroid_px") if isinstance(item.get("centroid_px"), list) else None,
                    "centroid_world": list(item.get("center_mm")) if isinstance(item.get("center_mm"), (list, tuple)) else None,
                    "geometry": {
                        "major_axis_mm": item.get("major_axis_mm"),
                        "minor_axis_mm": item.get("minor_axis_mm"),
                        "diameter_estimate_mm": item.get("diameter_estimate_mm"),
                    },
                    "appearance": {
                        "edge_roughness": item.get("feature_edge_roughness"),
                        "curvature_proxy": item.get("feature_local_curvature_proxy"),
                    },
                    "measurements": {
                        "height_max_mm": h.get("max_height_mm"),
                        "height_mean_mm": h.get("mean_height_mm"),
                        "height_p95_mm": h.get("p95_height_mm"),
                        "footprint_area_mm2": item.get("footprint_area_mm2"),
                        "volume_proxy_mm3": item.get("feature_volume_proxy_mm3"),
                        "diameter_mm": item.get("diameter_mm"),
                    },
                    "classification_hints": hints,
                    "confidence": item.get("confidence"),
                    "diagnostics": {
                        "filter_status": item.get("filter_status"),
                        "filter_reason": item.get("filter_reason"),
                    },
                },
                index=idx,
            )
        )
    return candidates


def _artifact_entries(artifact: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(artifact, dict):
        return []
    metadata = artifact.get("metadata") if isinstance(artifact.get("metadata"), dict) else {}
    entries = metadata.get("entries") if isinstance(metadata.get("entries"), list) else []
    return [item for item in entries if isinstance(item, dict)]


def _normalize_candidate(item: dict[str, Any], *, index: int) -> dict[str, Any]:
    return {
        "id": str(item.get("id") or f"candidate_{index}"),
        "local_object_index": int(item.get("local_object_index") or index),
        "source_modality": str(item.get("source_modality") or "rgb"),
        "source_take_id": str(item.get("source_take_id") or ""),
        "source_pipeline_run_id": item.get("source_pipeline_run_id"),
        "acquisition_group_id": item.get("acquisition_group_id"),
        "bbox_px": item.get("bbox_px"),
        "contour_px": item.get("contour_px"),
        "mask_artifact_id": item.get("mask_artifact_id"),
        "overlay_artifact_id": item.get("overlay_artifact_id"),
        "centroid_px": item.get("centroid_px"),
        "centroid_world": item.get("centroid_world"),
        "geometry": item.get("geometry") if isinstance(item.get("geometry"), dict) else {},
        "appearance": item.get("appearance") if isinstance(item.get("appearance"), dict) else {},
        "measurements": item.get("measurements") if isinstance(item.get("measurements"), dict) else {},
        "classification_hints": item.get("classification_hints") if isinstance(item.get("classification_hints"), dict) else {},
        "confidence": item.get("confidence"),
        "diagnostics": item.get("diagnostics") if isinstance(item.get("diagnostics"), dict) else {},
    }
