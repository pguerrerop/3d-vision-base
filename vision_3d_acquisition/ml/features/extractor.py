from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from vision_3d_acquisition.ml.features.schemas import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION


def _f(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def features_from_object(*, obj: dict[str, Any], take_id: str, dataset_id: str | None, session_id: str | None, labels: list[str], validation_status: str | None) -> dict[str, Any]:
    hab = obj.get("height_above_belt_mm") if isinstance(obj.get("height_above_belt_mm"), dict) else {}
    fg = obj.get("footprint_geometry") if isinstance(obj.get("footprint_geometry"), dict) else {}
    sg = obj.get("surface_geometry") if isinstance(obj.get("surface_geometry"), dict) else {}
    dg = obj.get("damage_metrics") if isinstance(obj.get("damage_metrics"), dict) else {}
    row: dict[str, Any] = {
        "take_id": take_id,
        "object_id": int(obj.get("object_id") or 0),
        "dataset_id": dataset_id,
        "session_id": session_id,
        "labels": labels,
        "validation_status": validation_status,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "provenance": {
            "source_stage": "classification",
            "semantic_field": "height_above_belt",
        },
        "feature_eccentricity": _f(obj.get("feature_eccentricity")),
        "feature_sphericity_3d": _f(obj.get("feature_sphericity_3d")),
        "feature_flatness": _f(obj.get("feature_flatness")),
        "feature_edge_roughness": _f(obj.get("feature_edge_roughness")),
        "feature_volume_proxy_mm3": _f(obj.get("feature_volume_proxy_mm3")),
        "height_max_mm": _f(hab.get("max_height_mm")),
        "height_p95_mm": _f(hab.get("p95_height_mm")),
        "height_mean_mm": _f(hab.get("mean_height_mm")),
        "footprint_radial_cv": _f(fg.get("radial_cv")),
        "surface_sphere_fit_rmse_mm": _f(sg.get("sphere_fit_rmse_mm")),
        "surface_deformation_score": _f(sg.get("deformation_score")),
        "damage_flat_region_ratio": _f(dg.get("flat_region_ratio")),
        "damage_surface_discontinuity_score": _f(dg.get("surface_discontinuity_score")),
    }
    return row


def rows_to_matrix(rows: list[dict[str, Any]]) -> tuple[list[list[float]], list[str]]:
    matrix = [[_f(row.get(col)) for col in FEATURE_COLUMNS] for row in rows]
    return matrix, FEATURE_COLUMNS


def export_rows(rows: list[dict[str, Any]], output_dir: Path, stem: str) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{stem}.jsonl"
    csv_path = output_dir / f"{stem}.csv"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    fieldnames = ["take_id", "object_id", "dataset_id", "session_id", "labels", "validation_status", *FEATURE_COLUMNS]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {k: row.get(k) for k in fieldnames}
            if isinstance(flat.get("labels"), list):
                flat["labels"] = ",".join(str(x) for x in flat["labels"])
            writer.writerow(flat)
    paths = {"jsonl": str(jsonl_path), "csv": str(csv_path)}
    try:
        import pandas as pd  # type: ignore

        parquet_path = output_dir / f"{stem}.parquet"
        pd.DataFrame(rows).to_parquet(parquet_path, index=False)
        paths["parquet"] = str(parquet_path)
    except Exception:
        pass
    return paths
