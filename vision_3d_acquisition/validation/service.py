from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import numpy as np
from PIL import Image

from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.pipelines.comparison import resolve_pipeline_run_source
from vision_3d_acquisition.pipelines.processing_units import processing_unit_contract_fingerprint


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def _json(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _read(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    os.replace(tmp, path)


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Declared semantic types mapped to comparators.  Extend this table rather than
# widening the identifier heuristics below it: an entry here is a reviewed decision,
# a heuristic match is a guess.  Types deliberately absent fall through to weaker
# evidence -- component id maps, for instance, are label rasters and comparing them
# as binary masks would silently under-report.
COMPARATOR_BY_SEMANTIC_TYPE: dict[str, str] = {
    "valid_mask": "binary_mask",
    "roi_mask": "binary_mask",
    "low_gradient_mask": "binary_mask",
    "unknown_low_gradient_mask": "binary_mask",
    "height_gate_mask": "binary_mask",
    "flat_candidate_mask": "binary_mask",
    "belt_background_mask": "binary_mask",
    "belt_base_mask": "binary_mask",
    "belt_above_belt_mask": "binary_mask",
    "belt_stripes_mask": "binary_mask",
    "belt_wide_object_mask": "binary_mask",
    "object_search_domain_mask": "binary_mask",
    "surface_suppression_mask": "binary_mask",
    "reference_suppression_mask": "binary_mask",
    "reference_model_support_mask": "binary_mask",
    "reference_surface_selected_mask": "binary_mask",
    "final_selected_support_mask": "binary_mask",
    "final_plane_inlier_mask": "binary_mask",
    "plane_inlier_mask": "binary_mask",
    "selected_cluster_mask": "binary_mask",
    "height_border_fragments_mask": "binary_mask",
    "height_split_fragments_mask": "binary_mask",
    "raw_heightmap": "numeric_raster",
    "gradient_heatmap": "numeric_raster",
    "plane_residual_heatmap": "numeric_raster",
    "height_border_strength": "numeric_raster",
    "reference_model_json": "plane_model",
    "quality_flags": "json_fields",
    "diagnostics": "json_fields",
}

KNOWN_COMPARATORS: frozenset[str] = frozenset(
    {
        "binary_mask",
        "numeric_raster",
        "measurement_table",
        "classification",
        "json_fields",
        "plane_model",
        "visual_only",
    }
)

# Roles whose artifacts exist to be looked at, not compared numerically.
NON_COMPARABLE_ROLES: frozenset[str] = frozenset(
    {"difference_overlay", "added_pixels", "removed_pixels"}
)


class ValidationService:
    """Filesystem-first repository.  It never writes into a take or run."""

    def __init__(self, settings: ApiSettings):
        self.settings = settings
        self.root = settings.data_dir / "validation"
        for name in ("baselines", "suites", "executions", "indexes"):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def _index(self, name: str) -> Path:
        return self.root / "indexes" / f"{name}.json"

    def _items(self, name: str) -> list[dict[str, Any]]:
        return [x for x in _read(self._index(name), []) if isinstance(x, dict)]

    def _set_items(self, name: str, items: list[dict[str, Any]]) -> None:
        _write(self._index(name), items)

    def list_baselines(
        self, *, take_id: str | None = None, pipeline_id: str | None = None
    ) -> list[dict[str, Any]]:
        rows = self._items("baselines")
        return [
            x
            for x in rows
            if (not take_id or x.get("take_id") == take_id)
            and (not pipeline_id or x.get("pipeline_id") == pipeline_id)
        ]

    def get_baseline(self, baseline_id: str) -> dict[str, Any] | None:
        path = self.root / "baselines" / baseline_id / "baseline.json"
        value = _read(path, None)
        return value if isinstance(value, dict) else None

    def _resolve(self, *, take_id: str, pipeline_id: str, run_id: str) -> dict[str, Any]:
        value = resolve_pipeline_run_source(
            self.settings, take_id=take_id, pipeline_id=pipeline_id, run_id=run_id
        )
        if not value:
            raise ValueError(
                f"Unknown run {run_id!r} for take {take_id!r} and pipeline {pipeline_id!r}."
            )
        return value

    def _eligible(self, artifact: Mapping[str, Any]) -> str:
        return self._classify(artifact)["comparator"]

    def _classify(self, artifact: Mapping[str, Any]) -> dict[str, str]:
        """Resolve an artifact to a comparator, preferring declared contract metadata.

        Later sources are progressively weaker evidence.  ``source`` is recorded so a
        reviewer can tell a declared decision from a guessed one.
        """
        metadata = artifact.get("metadata") if isinstance(artifact.get("metadata"), Mapping) else {}
        aid = str(artifact.get("artifact_id") or "")
        kind = str(artifact.get("kind") or "")
        path = str(artifact.get("path") or "")

        # 1. A stage states the comparator outright.  Nothing overrides this.
        declared = str(metadata.get("comparator") or "")
        if declared:
            return {"comparator": declared, "source": "declared_comparator"}

        # 2. The declared semantic type, mapped through a reviewed table.
        semantic = str(metadata.get("semantic_type") or "")
        if semantic in COMPARATOR_BY_SEMANTIC_TYPE:
            return {"comparator": COMPARATOR_BY_SEMANTIC_TYPE[semantic], "source": "semantic_type"}

        # 3. The declared role, where it settles comparability on its own.
        role = str(metadata.get("role") or "")
        if role in NON_COMPARABLE_ROLES:
            return {"comparator": "visual_only", "source": "role"}

        # 4. Real artifact kinds and the on-disk representation.  ``kind`` is a closed
        # literal in the contract, so only these values can ever appear.
        if kind == "overlay":
            return {"comparator": "visual_only", "source": "kind"}
        if path.endswith(".npy"):
            return {"comparator": "numeric_raster", "source": "path"}

        # 5. Legacy identifier heuristics.  Preserved so nothing that is approvable
        # today stops being approvable, but reported as a guess.
        if aid.endswith("overlay") or "overlay" in aid:
            return {"comparator": "visual_only", "source": "heuristic"}
        if "mask" in aid:
            return {"comparator": "binary_mask", "source": "heuristic"}
        if aid in {"belt_plane", "belt_plane_json"}:
            return {"comparator": "plane_model", "source": "heuristic"}
        if "height" in aid and kind == "image":
            return {"comparator": "numeric_raster", "source": "heuristic"}
        if "classification" in aid:
            return {"comparator": "classification", "source": "heuristic"}
        if "measurement" in aid or "metrics" in aid:
            return {"comparator": "measurement_table", "source": "heuristic"}
        if kind in {"json", "table"}:
            return {"comparator": "json_fields", "source": "kind"}
        return {"comparator": "not_comparable", "source": "unresolved"}

    def _selected(
        self, resolved: Mapping[str, Any], request: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        scope = str(request.get("approval_scope") or "artifact")
        artifacts = [dict(a) for a in resolved.get("artifacts", []) if isinstance(a, Mapping)]
        stage, unit, aid = (
            request.get("stage_id"),
            request.get("processing_unit_id") or request.get("substage_id"),
            request.get("artifact_id"),
        )
        if scope == "artifact":
            artifacts = [
                a
                for a in artifacts
                if str(a.get("artifact_id")) == str(aid)
                and (not stage or a.get("stage_id") == stage)
            ]
        elif scope in {"processing_unit", "substage"}:
            artifacts = [
                a
                for a in artifacts
                if str(a.get("processing_unit_id") or a.get("produced_by") or "") == str(unit)
            ]
        elif scope == "stage":
            artifacts = [a for a in artifacts if str(a.get("stage_id") or "") == str(stage)]
        if scope != "run" and not artifacts:
            raise ValueError("No artifact belongs to the requested run/stage/unit context.")
        return artifacts

    def create_baselines(self, request: Mapping[str, Any]) -> dict[str, Any]:
        take_id, pipeline_id, run_id = (
            str(request.get(k) or "") for k in ("take_id", "pipeline_id", "run_id")
        )
        if not all((take_id, pipeline_id, run_id)):
            raise ValueError("take_id, pipeline_id, and run_id are required.")
        resolved = self._resolve(take_id=take_id, pipeline_id=pipeline_id, run_id=run_id)
        selected = self._selected(resolved, request)
        created, skipped = [], []
        for artifact in selected:
            decision = self._classify(artifact)
            comparator, comparator_source = decision["comparator"], decision["source"]
            if comparator == "not_comparable":
                skipped.append(
                    {
                        "artifact_id": artifact.get("artifact_id"),
                        "reason": "unsupported",
                        "comparator_source": comparator_source,
                    }
                )
                continue
            source = Path(str(artifact.get("path") or ""))
            artifact_root = Path(resolved["artifact_root"]).resolve()
            source = (
                source.resolve() if source.is_absolute() else (artifact_root / source).resolve()
            )
            try:
                source.relative_to(artifact_root)
            except ValueError:
                raise ValueError("Artifact path escapes its resolved run artifact root.")
            if not source.is_file():
                skipped.append(
                    {
                        "artifact_id": artifact.get("artifact_id"),
                        "reason": "missing_source",
                        "comparator_source": comparator_source,
                    }
                )
                continue
            prior = [
                b
                for b in self.list_baselines(take_id=take_id, pipeline_id=pipeline_id)
                if b.get("artifact_id") == artifact.get("artifact_id")
                and b.get("stage_id") == artifact.get("stage_id")
            ]
            version = max([int(b.get("version", 0)) for b in prior] or [0]) + 1
            bid, root = _id("baseline"), None
            root = self.root / "baselines" / bid
            destination = root / "expected" / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

            result_payload = resolved.get("result_payload") or {}
            processing_units = (result_payload.get("processing_pipeline") or {}).get(
                "processing_units"
            ) or []
            recipe_snapshot = result_payload.get("recipe_snapshot") or {}
            metadata = artifact.get("metadata") or {}

            payload = {
                "id": bid,
                "version": version,
                "status": "active",
                "approval_scope": request.get("approval_scope", "artifact"),
                "take_id": take_id,
                "pipeline_id": pipeline_id,
                "source_run_id": run_id,
                "stage_id": artifact.get("stage_id") or request.get("stage_id"),
                "processing_unit_id": artifact.get("processing_unit_id")
                or artifact.get("produced_by")
                or request.get("processing_unit_id"),
                "substage_id": request.get("substage_id"),
                "view_id": request.get("view_id"),
                "artifact_id": artifact.get("artifact_id"),
                "artifact_kind": artifact.get("kind"),
                "artifact_semantic_type": metadata.get("semantic_type")
                or artifact.get("semantic_type")
                or comparator,
                "artifact_role": metadata.get("role"),
                "comparator": comparator,
                "comparator_source": comparator_source,
                "artifact_metadata": _json(metadata),
                "source_artifact_path": str(source),
                "snapshot_artifact_path": str(destination.relative_to(root)),
                "artifact_checksum": _checksum(destination),
                "pipeline_contract_fingerprint": processing_unit_contract_fingerprint(
                    list(processing_units)
                ),
                "recipe_id": resolved.get("recipe_id"),
                "recipe_fingerprint": recipe_snapshot.get("fingerprint"),
                "parameter_snapshot": _json(resolved.get("stage_params") or {}),
                "calibration_context": _json(
                    result_payload.get("calibration_snapshot_reference") or {}
                ),
                "comparison_policy": _json(
                    request.get("comparison_policy") or {"comparator": comparator}
                ),
                "review": {
                    "status": "approved",
                    "reviewed_at": _now(),
                    "reviewed_by": request.get("reviewed_by"),
                    "notes": request.get("notes") or "",
                },
                "created_at": _now(),
                "supersedes_baseline_id": prior[-1]["id"] if prior else None,
            }
            _write(root / "baseline.json", payload)
            rows = self._items("baselines")
            rows.append(
                {
                    k: payload.get(k)
                    for k in (
                        "id",
                        "version",
                        "status",
                        "take_id",
                        "pipeline_id",
                        "artifact_id",
                        "stage_id",
                        "source_run_id",
                        "created_at",
                        "supersedes_baseline_id",
                    )
                }
            )
            self._set_items("baselines", rows)
            created.append(payload)
            # A normal reapproval promotes one active version.  Callers can explicitly
            # retain alternatives for legitimate multi-outcome cases.
            if not request.get("allow_alternative"):
                for previous in prior:
                    if previous.get("status") == "active":
                        self.set_active(str(previous["id"]), False)
        return {
            "baselines": created,
            "included": [b["artifact_id"] for b in created],
            "skipped": skipped,
        }

    def set_active(self, baseline_id: str, active: bool) -> dict[str, Any]:
        baseline = self.get_baseline(baseline_id)
        if not baseline:
            raise KeyError(baseline_id)
        baseline["status"] = "active" if active else "inactive"
        _write(self.root / "baselines" / baseline_id / "baseline.json", baseline)
        rows = self._items("baselines")
        for row in rows:
            if row.get("id") == baseline_id:
                row["status"] = baseline["status"]
        self._set_items("baselines", rows)
        return baseline

    @staticmethod
    def _comparator_of(baseline: Mapping[str, Any]) -> str:
        policy = (
            baseline.get("comparison_policy")
            if isinstance(baseline.get("comparison_policy"), Mapping)
            else {}
        )
        semantic = str(baseline.get("artifact_semantic_type") or "")
        return str(
            policy.get("comparator")
            or baseline.get("comparator")
            or COMPARATOR_BY_SEMANTIC_TYPE.get(semantic)
            or (semantic if semantic in KNOWN_COMPARATORS else "")
            or "not_comparable"
        )

    def compare(
        self, baseline_id: str, *, candidate_run_id: str, diff_dir: Path | None = None
    ) -> dict[str, Any]:
        baseline = self.get_baseline(baseline_id)
        if not baseline:
            raise KeyError(baseline_id)
        candidate = self._resolve(
            take_id=baseline["take_id"],
            pipeline_id=baseline["pipeline_id"],
            run_id=candidate_run_id,
        )
        artifact = next(
            (
                a
                for a in candidate.get("artifacts", [])
                if a.get("artifact_id") == baseline.get("artifact_id")
                and (not baseline.get("stage_id") or a.get("stage_id") == baseline.get("stage_id"))
            ),
            None,
        )
        if not artifact:
            return self._result("not_comparable", baseline, None, "Candidate artifact is absent.")
        path = Path(str(artifact.get("path") or ""))
        path = path if path.is_absolute() else Path(candidate["artifact_root"]) / path
        if not path.is_file():
            return self._result(
                "not_comparable", baseline, artifact, "Candidate artifact file is absent."
            )
        comparator = self._comparator_of(baseline)
        if comparator == "visual_only":
            return self._result(
                "needs_review", baseline, artifact, "Rendered evidence is visual-only."
            )
        if comparator == "binary_mask":
            return self._mask(
                Path(self.root / "baselines" / baseline_id / baseline["snapshot_artifact_path"]),
                path,
                baseline,
                artifact,
                diff_dir,
            )
        if comparator == "numeric_raster":
            return self._raster(
                Path(self.root / "baselines" / baseline_id / baseline["snapshot_artifact_path"]),
                path,
                baseline,
                artifact,
                diff_dir,
            )
        if comparator == "measurement_table":
            return self._measurement_compare(
                Path(self.root / "baselines" / baseline_id / baseline["snapshot_artifact_path"]),
                path,
                baseline,
                artifact,
            )
        if comparator == "classification":
            return self._classification_compare(
                Path(self.root / "baselines" / baseline_id / baseline["snapshot_artifact_path"]),
                path,
                baseline,
                artifact,
            )
        if comparator in {"json_fields", "plane_model"}:
            return self._json_compare(
                Path(self.root / "baselines" / baseline_id / baseline["snapshot_artifact_path"]),
                path,
                baseline,
                artifact,
                comparator,
            )
        return self._result(
            "not_comparable", baseline, artifact, "No comparator is registered for this artifact."
        )

    def _result(
        self,
        status: str,
        baseline: Mapping[str, Any],
        artifact: Mapping[str, Any] | None,
        summary: str,
        **extra: Any,
    ) -> dict[str, Any]:
        return {
            "status": status,
            "comparator": self._comparator_of(baseline),
            "summary": summary,
            "metrics": extra.get("metrics", {}),
            "thresholds": extra.get("thresholds", {}),
            "reasons": extra.get("reasons", []),
            "diff_artifacts": [],
            "baseline_ref": {"baseline_id": baseline.get("id"), "version": baseline.get("version")},
            "candidate_ref": {"artifact_id": (artifact or {}).get("artifact_id")},
        }

    def _mask(
        self,
        a: Path,
        b: Path,
        base: Mapping[str, Any],
        art: Mapping[str, Any],
        diff_dir: Path | None = None,
    ) -> dict[str, Any]:
        try:
            x = np.asarray(Image.open(a).convert("L")) > 0
            y = np.asarray(Image.open(b).convert("L")) > 0
        except Exception:
            return self._result("not_comparable", base, art, "Mask files cannot be decoded.")
        if x.shape != y.shape:
            return self._result("not_comparable", base, art, "Mask dimensions differ.")
        inter = (x & y).sum()
        union = (x | y).sum()
        added = (y & ~x).sum()
        removed = (x & ~y).sum()
        dice = 2 * inter / (x.sum() + y.sum()) if x.sum() + y.sum() else 1.0
        iou = inter / union if union else 1.0
        policy = base.get("comparison_policy") or {}
        total = x.size
        components = lambda z: __import__("scipy.ndimage", fromlist=["label"]).label(z)[1]
        bc, cc = int(components(x)), int(components(y))
        reasons = []
        for cond, name in (
            (iou < float(policy.get("min_iou", 0.97)), "iou_below_threshold"),
            (dice < float(policy.get("min_dice", 0.98)), "dice_below_threshold"),
            (
                added / total > float(policy.get("max_added_fraction", 0.01)),
                "added_fraction_above_threshold",
            ),
            (
                removed / total > float(policy.get("max_removed_fraction", 0.01)),
                "removed_fraction_above_threshold",
            ),
            (
                bool(policy.get("require_component_count_match")) and bc != cc,
                "component_count_changed",
            ),
        ):
            if cond:
                reasons.append(name)
        result = self._result(
            "regression" if reasons else "pass",
            base,
            art,
            "Binary-mask comparison.",
            metrics={
                "iou": float(iou),
                "dice": float(dice),
                "intersection_pixels": int(inter),
                "union_pixels": int(union),
                "added_pixels": int(added),
                "removed_pixels": int(removed),
                "added_fraction": float(added / total),
                "removed_fraction": float(removed / total),
                "baseline_foreground_pixels": int(x.sum()),
                "candidate_foreground_pixels": int(y.sum()),
                "baseline_components": bc,
                "candidate_components": cc,
            },
            thresholds=policy,
            reasons=reasons,
        )
        if diff_dir:
            diff_dir.mkdir(parents=True, exist_ok=True)
            files = {
                "baseline_mask.png": x * 255,
                "candidate_mask.png": y * 255,
                "added_pixels.png": (y & ~x) * 255,
                "removed_pixels.png": (x & ~y) * 255,
            }
            for name, image in files.items():
                Image.fromarray(image.astype(np.uint8)).save(diff_dir / name)
            combined = np.zeros((*x.shape, 3), dtype=np.uint8)
            combined[x & y] = [170, 170, 170]
            combined[y & ~x] = [0, 220, 0]
            combined[x & ~y] = [230, 40, 40]
            Image.fromarray(combined).save(diff_dir / "combined_diff.png")
            result["diff_artifacts"] = [
                str((diff_dir / name).relative_to(self.root))
                for name in (*files, "combined_diff.png")
            ]
        return result

    def _raster(
        self,
        a: Path,
        b: Path,
        base: Mapping[str, Any],
        art: Mapping[str, Any],
        diff_dir: Path | None = None,
    ) -> dict[str, Any]:
        try:
            x = np.load(a) if a.suffix == ".npy" else np.asarray(Image.open(a), dtype=float)
            y = np.load(b) if b.suffix == ".npy" else np.asarray(Image.open(b), dtype=float)
        except Exception:
            return self._result("not_comparable", base, art, "Numeric rasters cannot be decoded.")
        if x.shape != y.shape:
            return self._result("not_comparable", base, art, "Raster dimensions differ.")
        valid = np.isfinite(x) & np.isfinite(y)
        overlap = float(valid.sum() / max(1, (np.isfinite(x) | np.isfinite(y)).sum()))
        d = np.abs(x.astype(float) - y.astype(float))[valid]
        if not d.size:
            return self._result(
                "not_comparable", base, art, "No mutually valid authoritative raster pixels."
            )
        metrics = {
            "valid_overlap": overlap,
            "candidate_only_valid_pixels": int((np.isfinite(y) & ~np.isfinite(x)).sum()),
            "baseline_only_valid_pixels": int((np.isfinite(x) & ~np.isfinite(y)).sum()),
            "mae": float(np.mean(d)),
            "rmse": float(np.sqrt(np.mean(d * d))),
            "median_absolute_error": float(np.median(d)),
            "p95_absolute_error": float(np.percentile(d, 95)),
            "p99_absolute_error": float(np.percentile(d, 99)),
            "max_absolute_error": float(np.max(d)),
            "signed_mean_error": float(np.mean((y - x)[valid])),
        }
        policy = base.get("comparison_policy") or {}
        mapping = (
            ("mae", "max_mae"),
            ("rmse", "max_rmse"),
            ("p95_absolute_error", "max_p95_abs_error"),
            ("valid_overlap", "min_valid_overlap"),
        )
        reasons = [
            f"{k}_threshold"
            for k, t in mapping
            if t in policy
            and (
                (metrics[k] > float(policy[t]))
                if k != "valid_overlap"
                else (metrics[k] < float(policy[t]))
            )
        ]
        result = self._result(
            "regression" if reasons else "pass",
            base,
            art,
            "Numeric-raster comparison.",
            metrics=metrics,
            thresholds=policy,
            reasons=reasons,
        )
        if diff_dir:
            diff_dir.mkdir(parents=True, exist_ok=True)
            heat = np.zeros_like(x, dtype=float)
            heat[valid] = np.abs(x - y)[valid]
            scaled = (255 * heat / max(float(heat.max()), 1e-12)).astype(np.uint8)
            Image.fromarray(scaled).save(diff_dir / "absolute_difference.png")
            Image.fromarray((np.isfinite(x) != np.isfinite(y)).astype(np.uint8) * 255).save(
                diff_dir / "valid_pixel_change.png"
            )
            result["diff_artifacts"] = [
                str((diff_dir / name).relative_to(self.root))
                for name in ("absolute_difference.png", "valid_pixel_change.png")
            ]
        return result

    def _json_compare(
        self, a: Path, b: Path, base: Mapping[str, Any], art: Mapping[str, Any], comparator: str
    ) -> dict[str, Any]:
        left, right = _read(a, None), _read(b, None)
        if left is None or right is None:
            return self._result("not_comparable", base, art, "JSON payload cannot be decoded.")
        policy = base.get("comparison_policy") or {}
        ignored = set(
            policy.get("unstableFields")
            or ["timestamp", "processed_at", "run_id", "elapsed_ms", "path"]
        )

        def clean(v):
            if isinstance(v, dict):
                return {k: clean(x) for k, x in v.items() if k not in ignored}
            if isinstance(v, list):
                return [clean(x) for x in v]
            return v

        same = clean(left) == clean(right)
        return self._result(
            "pass" if same else "regression",
            base,
            art,
            f"{comparator} comparison.",
            metrics={"equal": same},
            thresholds=policy,
        )

    @staticmethod
    def _rows(value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            return [dict(x) for x in value if isinstance(x, dict)]
        if isinstance(value, dict):
            for key in ("objects", "measurements", "rows", "entries", "classifications"):
                if isinstance(value.get(key), list):
                    return [dict(x) for x in value[key] if isinstance(x, dict)]
            if any(key in value for key in ("object_id", "id", "label", "class_name")):
                return [dict(value)]
        return []

    @staticmethod
    def _id(row: Mapping[str, Any]) -> str | None:
        for key in (
            "object_id",
            "stable_object_id",
            "source_object_id",
            "source_candidate_id",
            "lineage_id",
            "id",
        ):
            value = row.get(key)
            if value is not None and str(value):
                return str(value)
        return None

    def _match_objects(
        self, left: list[dict[str, Any]], right: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Conservative one-to-one matching; IDs always outrank geometry."""
        unmatched_right = set(range(len(right)))
        matches = []
        ambiguous = []
        baseline_only = []
        for li, row in enumerate(left):
            identifiers = [
                (key, str(row[key]))
                for key in (
                    "object_id",
                    "stable_object_id",
                    "source_object_id",
                    "source_candidate_id",
                    "lineage_id",
                    "id",
                )
                if row.get(key) is not None
            ]
            candidates = [
                ri
                for ri in unmatched_right
                if any(str(right[ri].get(key)) == value for key, value in identifiers)
            ]
            if len(candidates) == 1:
                ri = candidates[0]
                unmatched_right.remove(ri)
                matches.append(
                    {
                        "baseline_index": li,
                        "candidate_index": ri,
                        "baseline_object_id": self._id(row),
                        "candidate_object_id": self._id(right[ri]),
                        "match_method": "stable_id",
                        "match_score": 1.0,
                        "ambiguous": False,
                    }
                )
                continue
            if len(candidates) > 1:
                ambiguous.append(
                    {"baseline_object_id": self._id(row), "candidate_indices": candidates}
                )
                continue
            # Centroid fallback only where a single candidate is unambiguously close.
            center = row.get("centroid") or row.get("center_mm")
            if isinstance(center, (list, tuple)) and len(center) >= 2:
                scored = []
                for ri in unmatched_right:
                    other = right[ri].get("centroid") or right[ri].get("center_mm")
                    if isinstance(other, (list, tuple)) and len(other) >= 2:
                        scored.append(
                            (
                                float(
                                    sum(
                                        (float(a) - float(b)) ** 2
                                        for a, b in zip(center[:3], other[:3])
                                    )
                                    ** 0.5
                                ),
                                ri,
                            )
                        )
                scored.sort()
                if (
                    scored
                    and scored[0][0] <= 5.0
                    and (len(scored) == 1 or scored[1][0] - scored[0][0] > 1.0)
                ):
                    _, ri = scored[0]
                    unmatched_right.remove(ri)
                    matches.append(
                        {
                            "baseline_index": li,
                            "candidate_index": ri,
                            "baseline_object_id": self._id(row),
                            "candidate_object_id": self._id(right[ri]),
                            "match_method": "centroid",
                            "match_score": 1 / (1 + scored[0][0]),
                            "ambiguous": False,
                        }
                    )
                    continue
                if scored and scored[0][0] <= 5.0:
                    ambiguous.append(
                        {
                            "baseline_object_id": self._id(row),
                            "candidate_indices": [ri for _, ri in scored[:2]],
                        }
                    )
                    continue
            baseline_only.append(li)
        return {
            "matches": matches,
            "baseline_only": baseline_only,
            "candidate_only": sorted(unmatched_right),
            "ambiguous": ambiguous,
            "split_candidates": [],
            "merge_candidates": [],
        }

    @staticmethod
    def _metric_value(row: Mapping[str, Any], metric: str) -> Any:
        aliases = {
            "diameter_selected_mm": ["diameter_selected_mm", "diameter_mm", "diameter_estimate_mm"],
            "p95_height_mm": ["p95_height_mm"],
            "max_height_mm": ["max_height_mm"],
        }
        keys = aliases.get(metric, [metric])
        for key in keys:
            if key in row:
                return row[key]
            for group in ("height_above_belt_mm", "dimensions_mm", "measurement", "metrics"):
                nested = row.get(group)
                if isinstance(nested, dict) and key in nested:
                    return nested[key]
        return None

    def _measurement_compare(
        self, a: Path, b: Path, base: Mapping[str, Any], art: Mapping[str, Any]
    ) -> dict[str, Any]:
        left, right = self._rows(_read(a, None)), self._rows(_read(b, None))
        if not left and not right:
            return self._result(
                "not_comparable", base, art, "Measurement schema is unsupported or empty."
            )
        policy = base.get("comparison_policy") or {}
        configured = policy.get("metrics") if isinstance(policy.get("metrics"), dict) else {}
        if not configured:
            return self._result(
                "not_comparable",
                base,
                art,
                "Measurement comparator requires declared metric tolerances.",
            )
        match = self._match_objects(left, right)
        results = []
        regressions = 0
        for m in match["matches"]:
            l, r = left[m["baseline_index"]], right[m["candidate_index"]]
            metric_results = []
            object_regression = False
            for metric, spec in configured.items():
                spec = spec if isinstance(spec, dict) else {}
                lv, rv = self._metric_value(l, metric), self._metric_value(r, metric)
                if lv is None or rv is None:
                    status = "regression" if spec.get("required") else "changed"
                    metric_results.append(
                        {
                            "metric": metric,
                            "baseline": lv,
                            "candidate": rv,
                            "status": status,
                            "reason": "missing_metric",
                        }
                    )
                    object_regression |= status == "regression"
                    continue
                if not isinstance(lv, (int, float)) or not isinstance(rv, (int, float)):
                    metric_results.append(
                        {
                            "metric": metric,
                            "baseline": lv,
                            "candidate": rv,
                            "status": "not_comparable",
                            "reason": "non_numeric_metric",
                        }
                    )
                    continue
                delta = float(rv) - float(lv)
                rel = abs(delta) / max(abs(float(lv)), 1e-12)
                absolute = float(spec.get("absolute_tolerance", 0))
                relative = float(spec.get("relative_tolerance", 0))
                effective = max(absolute, relative * abs(float(lv)))
                status = "pass" if abs(delta) <= effective else "regression"
                object_regression |= status == "regression"
                metric_results.append(
                    {
                        "metric": metric,
                        "units": spec.get("units"),
                        "baseline": lv,
                        "candidate": rv,
                        "delta": delta,
                        "relative_delta": rel,
                        "absolute_tolerance": absolute,
                        "relative_tolerance": relative,
                        "effective_tolerance": effective,
                        "status": status,
                        "reason": None if status == "pass" else "tolerance_exceeded",
                    }
                )
            regressions += sum(x["status"] == "regression" for x in metric_results)
            results.append(
                {
                    **m,
                    "status": "regression" if object_regression else "pass",
                    "metric_results": metric_results,
                }
            )
        status = (
            "needs_review"
            if match["ambiguous"]
            else (
                "regression"
                if regressions or match["baseline_only"] or match["candidate_only"]
                else "pass"
            )
        )
        return {
            **self._result(
                status,
                base,
                art,
                f"{regressions} measurement metrics exceeded tolerance.",
                metrics={
                    "baseline_object_count": len(left),
                    "candidate_object_count": len(right),
                    "matched_object_count": len(match["matches"]),
                    "missing_object_count": len(match["baseline_only"]),
                    "added_object_count": len(match["candidate_only"]),
                    "regressed_metric_count": regressions,
                },
                thresholds=policy,
            ),
            "object_results": results,
            "matching": match,
        }

    def _classification_compare(
        self, a: Path, b: Path, base: Mapping[str, Any], art: Mapping[str, Any]
    ) -> dict[str, Any]:
        left, right = self._rows(_read(a, None)), self._rows(_read(b, None))
        if not left and not right:
            return self._result(
                "not_comparable", base, art, "Classification schema is unsupported or empty."
            )
        policy = base.get("comparison_policy") or {}
        match = self._match_objects(left, right)
        results = []
        labels = superclasses = warnings = 0
        critical = set(policy.get("critical_warning_codes") or [])
        for m in match["matches"]:
            l, r = left[m["baseline_index"]], right[m["candidate_index"]]
            label = lambda v: v.get("detailed_label") or v.get("label") or v.get("class_name")
            superclass = lambda v: v.get("superclass") or v.get("class_name")
            reasons = []
            if superclass(l) != superclass(r):
                reasons.append("superclass_changed")
                superclasses += 1
            allowed = set((policy.get("allowed_labels") or {}).get(str(self._id(l)), []))
            if label(l) != label(r) and label(r) not in allowed:
                reasons.append("label_changed")
                labels += 1
            confidence_delta = float(r.get("confidence", 0) or 0) - float(
                l.get("confidence", 0) or 0
            )
            if r.get("confidence") is not None and float(r["confidence"]) < float(
                policy.get("min_confidence", 0)
            ):
                reasons.append("confidence_below_minimum")
            if confidence_delta < -float(policy.get("max_confidence_drop", float("inf"))):
                reasons.append("confidence_drop")
            lw = set(l.get("warning_codes") or l.get("warnings") or [])
            rw = set(r.get("warning_codes") or r.get("warnings") or [])
            new_critical = (rw - lw) & critical
            if new_critical:
                reasons.append("new_critical_warning")
                warnings += len(new_critical)
            results.append(
                {
                    **m,
                    "baseline": {
                        "superclass": superclass(l),
                        "label": label(l),
                        "confidence": l.get("confidence"),
                    },
                    "candidate": {
                        "superclass": superclass(r),
                        "label": label(r),
                        "confidence": r.get("confidence"),
                    },
                    "confidence_delta": confidence_delta,
                    "status": "regression" if reasons else "pass",
                    "reasons": reasons,
                }
            )
        status = (
            "needs_review"
            if match["ambiguous"]
            else (
                "regression"
                if labels or superclasses or warnings or match["baseline_only"]
                else "changed" if match["candidate_only"] else "pass"
            )
        )
        return {
            **self._result(
                status,
                base,
                art,
                f"{labels} classification labels changed.",
                metrics={
                    "baseline_object_count": len(left),
                    "candidate_object_count": len(right),
                    "label_changes": labels,
                    "superclass_changes": superclasses,
                    "new_critical_warnings": warnings,
                },
                thresholds=policy,
            ),
            "object_results": results,
            "matching": match,
        }

    def list_suites(self) -> list[dict[str, Any]]:
        return self._items("suites")

    def get_suite(self, suite_id: str) -> dict[str, Any]:
        value = _read(self.root / "suites" / suite_id / "suite.json", None)
        if not isinstance(value, dict):
            raise KeyError(suite_id)
        value["cases"] = [
            x
            for x in _read(self.root / "suites" / suite_id / "cases.json", [])
            if isinstance(x, dict)
        ]
        return value

    def create_suite(self, request: Mapping[str, Any]) -> dict[str, Any]:
        suite = {
            "id": _id("suite"),
            "name": request["name"],
            "description": request.get("description") or "",
            "pipeline_id": request["pipeline_id"],
            "status": "active",
            "case_ids": [],
            "default_comparison_policy": _json(request.get("default_comparison_policy") or {}),
            "created_at": _now(),
            "updated_at": _now(),
        }
        root = self.root / "suites" / suite["id"]
        _write(root / "suite.json", suite)
        _write(root / "cases.json", [])
        rows = self._items("suites")
        rows.append(
            {
                k: suite[k]
                for k in ("id", "name", "pipeline_id", "status", "created_at", "updated_at")
            }
        )
        self._set_items("suites", rows)
        return suite

    def update_suite(self, suite_id: str, request: Mapping[str, Any]) -> dict[str, Any]:
        suite = self.get_suite(suite_id)
        for key in ("name", "description", "status", "default_comparison_policy"):
            if key in request:
                suite[key] = _json(request[key])
        suite["updated_at"] = _now()
        _write(
            self.root / "suites" / suite_id / "suite.json",
            {k: v for k, v in suite.items() if k != "cases"},
        )
        rows = self._items("suites")
        for row in rows:
            if row.get("id") == suite_id:
                row.update({k: suite.get(k) for k in ("name", "status", "updated_at")})
        self._set_items("suites", rows)
        return suite

    def archive_suite(self, suite_id: str) -> dict[str, Any]:
        return self.update_suite(suite_id, {"status": "archived"})

    def restore_suite(self, suite_id: str) -> dict[str, Any]:
        return self.update_suite(suite_id, {"status": "active"})

    def duplicate_suite(self, suite_id: str) -> dict[str, Any]:
        source = self.get_suite(suite_id)
        duplicate = self.create_suite(
            {
                "name": f"{source['name']} copy",
                "pipeline_id": source["pipeline_id"],
                "description": source.get("description") or "",
            }
        )
        for case in source["cases"]:
            self.add_case(duplicate["id"], {k: v for k, v in case.items() if k not in {"id"}})
        return self.get_suite(duplicate["id"])

    def coverage(self, suite_id: str) -> dict[str, Any]:
        suite = self.get_suite(suite_id)
        areas: dict[str, dict[str, Any]] = {}
        for case in suite["cases"]:
            if not case.get("enabled", True):
                continue
            for baseline_id in case.get("baseline_ids") or []:
                baseline = self.get_baseline(str(baseline_id))
                area = str((baseline or {}).get("stage_id") or "unknown")
                row = areas.setdefault(
                    area,
                    {
                        "area": area,
                        "approved": 0,
                        "missing": 0,
                        "unsupported": 0,
                        "stale": 0,
                        "visual_only": 0,
                    },
                )
                if not baseline:
                    row["missing"] += 1
                    continue
                comparator = self._comparator_of(baseline)
                if baseline.get("status") != "active":
                    row["stale"] += 1
                elif comparator == "visual_only":
                    row["visual_only"] += 1
                elif comparator in {"", "not_comparable"}:
                    row["unsupported"] += 1
                else:
                    row["approved"] += 1
        values = []
        for row in areas.values():
            total = sum(
                row[key] for key in ("approved", "missing", "unsupported", "stale", "visual_only")
            )
            row["coverage_fraction"] = row["approved"] / total if total else 0
            values.append(row)
        return {
            "suite_id": suite_id,
            "case_count": len(suite["cases"]),
            "areas": sorted(values, key=lambda x: x["area"]),
        }

    def _require_compatible(
        self, baseline_ids: list[str], *, take_id: str | None, pipeline_id: str
    ) -> None:
        for baseline_id in baseline_ids:
            baseline = self.get_baseline(baseline_id)
            if (
                not baseline
                or (take_id and baseline.get("take_id") != take_id)
                or baseline.get("pipeline_id") != pipeline_id
            ):
                raise ValueError("baseline is incompatible with case")

    @staticmethod
    def _resolution_ids(resolution: Mapping[str, Any], mode: str, fallback: list[str]) -> list[str]:
        """An absent selection defaults to the configured baselines; an explicitly
        empty one is a request the caller must correct, not a silent fallback."""
        key = "allowed_baseline_ids" if mode == "allowed_versions" else "baseline_id"
        if key in resolution:
            value = resolution[key]
            ids = (
                [str(x) for x in value]
                if isinstance(value, (list, tuple))
                else ([str(value)] if value else [])
            )
            if not ids:
                raise ValueError(
                    "allowed_versions requires at least one baseline version"
                    if mode == "allowed_versions"
                    else "pinned resolution requires a baseline version"
                )
            return ids
        return [str(x) for x in fallback]

    def add_case(self, suite_id: str, request: Mapping[str, Any]) -> dict[str, Any]:
        suite = self.get_suite(suite_id)
        if suite.get("status") == "archived":
            raise ValueError("archived suite cannot be changed")
        baseline_ids = request.get("baseline_ids") or (
            [request["baseline_id"]] if request.get("baseline_id") else []
        )
        if not baseline_ids:
            raise ValueError("baseline_id or baseline_ids is required.")
        for bid in baseline_ids:
            b = self.get_baseline(str(bid))
            if not b:
                raise ValueError(f"Unknown baseline {bid}.")
            if b.get("pipeline_id") != suite["pipeline_id"]:
                raise ValueError("Baseline pipeline does not match suite pipeline.")
        resolution = dict(request.get("baseline_resolution") or {"mode": "active"})
        mode = str(resolution.get("mode") or "active")
        if mode not in {"active", "pinned", "allowed_versions"}:
            raise ValueError("invalid baseline resolution mode")
        take_id = str(request.get("take_id") or self.get_baseline(str(baseline_ids[0]))["take_id"])
        if mode == "pinned":
            pinned = self._resolution_ids(resolution, mode, baseline_ids[:1])
            self._require_compatible(pinned, take_id=take_id, pipeline_id=suite["pipeline_id"])
            resolution = {"mode": "pinned", "baseline_id": pinned[0]}
        elif mode == "allowed_versions":
            allowed = self._resolution_ids(resolution, mode, baseline_ids)
            if len(set(allowed)) != len(allowed):
                raise ValueError("allowed_versions requires unique baseline IDs")
            self._require_compatible(allowed, take_id=take_id, pipeline_id=suite["pipeline_id"])
            resolution = {"mode": "allowed_versions", "allowed_baseline_ids": allowed}
        else:
            resolution = {"mode": "active"}
        case = {
            "id": _id("case"),
            "take_id": take_id,
            "baseline_ids": list(baseline_ids),
            "baseline_resolution": resolution,
            "enabled": bool(request.get("enabled", True)),
            "tags": list(request.get("tags") or []),
            "notes": request.get("notes") or "",
            "included_artifacts": request.get("included_artifacts") or [],
        }
        cases = suite["cases"]
        cases.append(case)
        _write(self.root / "suites" / suite_id / "cases.json", cases)
        suite["case_ids"] = [x["id"] for x in cases]
        suite["updated_at"] = _now()
        _write(
            self.root / "suites" / suite_id / "suite.json",
            {k: v for k, v in suite.items() if k != "cases"},
        )
        return case

    def update_case(
        self, suite_id: str, case_id: str, request: Mapping[str, Any]
    ) -> dict[str, Any]:
        suite = self.get_suite(suite_id)
        if suite.get("status") == "archived":
            raise ValueError("archived suite cannot be changed")
        case = next((x for x in suite["cases"] if x.get("id") == case_id), None)
        if not case:
            raise KeyError(case_id)
        for key in ("enabled", "tags", "notes", "included_artifacts"):
            if key in request:
                case[key] = _json(request[key])
        if "baseline_resolution" in request:
            resolution = dict(request["baseline_resolution"] or {})
            mode = str(resolution.get("mode") or "")
            if mode not in {"active", "pinned", "allowed_versions"}:
                raise ValueError("invalid baseline resolution mode")
            ids = self._resolution_ids(resolution, mode, list(case.get("baseline_ids") or []))
            if mode == "allowed_versions" and len(set(ids)) != len(ids):
                raise ValueError("allowed_versions requires unique baseline IDs")
            self._require_compatible(ids, take_id=case["take_id"], pipeline_id=suite["pipeline_id"])
            case["baseline_resolution"] = {
                "mode": mode,
                "baseline_id": ids[0] if mode == "pinned" else None,
                "allowed_baseline_ids": ids if mode == "allowed_versions" else [],
            }
        _write(self.root / "suites" / suite_id / "cases.json", suite["cases"])
        return case

    def delete_case(self, suite_id: str, case_id: str) -> dict[str, Any]:
        suite = self.get_suite(suite_id)
        if suite.get("status") == "archived":
            raise ValueError("archived suite cannot be changed")
        cases = [x for x in suite["cases"] if x.get("id") != case_id]
        if len(cases) == len(suite["cases"]):
            raise KeyError(case_id)
        _write(self.root / "suites" / suite_id / "cases.json", cases)
        suite["case_ids"] = [x["id"] for x in cases]
        suite["updated_at"] = _now()
        _write(
            self.root / "suites" / suite_id / "suite.json",
            {k: v for k, v in suite.items() if k != "cases"},
        )
        return {"deleted": case_id}

    def list_executions(self) -> list[dict[str, Any]]:
        return self._items("executions")

    def rebuild_indexes(self) -> dict[str, int]:
        """Rebuild disposable indexes from canonical immutable records."""
        counts: dict[str, int] = {}
        for name, filename, fields in (
            (
                "baselines",
                "baseline.json",
                (
                    "id",
                    "version",
                    "status",
                    "take_id",
                    "pipeline_id",
                    "artifact_id",
                    "stage_id",
                    "source_run_id",
                    "created_at",
                    "supersedes_baseline_id",
                ),
            ),
            (
                "suites",
                "suite.json",
                ("id", "name", "pipeline_id", "status", "created_at", "updated_at"),
            ),
            (
                "executions",
                "execution.json",
                ("id", "suite_id", "pipeline_id", "status", "created_at", "completed_at"),
            ),
        ):
            rows = []
            for path in (self.root / name).glob(f"*/{filename}"):
                payload = _read(path, None)
                if isinstance(payload, dict) and payload.get("id"):
                    rows.append({key: payload.get(key) for key in fields})
            self._set_items(name, rows)
            counts[name] = len(rows)
        return counts

    def verify_integrity(self) -> dict[str, Any]:
        problems: list[dict[str, str]] = []
        baselines = []
        for path in (self.root / "baselines").glob("*/baseline.json"):
            baseline = _read(path, None)
            if not isinstance(baseline, dict):
                problems.append({"kind": "invalid_baseline_record", "path": str(path)})
                continue
            baselines.append(baseline)
            snapshot = (path.parent / str(baseline.get("snapshot_artifact_path") or "")).resolve()
            try:
                snapshot.relative_to((self.root / "baselines").resolve())
            except ValueError:
                problems.append(
                    {"kind": "snapshot_path_escape", "baseline_id": str(baseline.get("id"))}
                )
                continue
            if not snapshot.is_file():
                problems.append(
                    {"kind": "missing_snapshot", "baseline_id": str(baseline.get("id"))}
                )
            elif baseline.get("artifact_checksum") != _checksum(snapshot):
                problems.append(
                    {"kind": "checksum_mismatch", "baseline_id": str(baseline.get("id"))}
                )
        ids = {str(x.get("id")) for x in baselines}
        for suite_path in (self.root / "suites").glob("*/cases.json"):
            for case in _read(suite_path, []):
                if isinstance(case, dict):
                    for baseline_id in case.get("baseline_ids") or []:
                        if str(baseline_id) not in ids:
                            problems.append(
                                {
                                    "kind": "broken_suite_baseline",
                                    "case_id": str(case.get("id")),
                                    "baseline_id": str(baseline_id),
                                }
                            )
        return {"ok": not problems, "baseline_count": len(baselines), "problems": problems}

    def get_execution(self, eid: str) -> dict[str, Any]:
        value = _read(self.root / "executions" / eid / "execution.json", None)
        if not isinstance(value, dict):
            raise KeyError(eid)
        return value

    def matrix(self, execution_id: str) -> dict[str, Any]:
        execution = self.get_execution(execution_id)
        pipeline = str(execution.get("pipeline_id") or "")
        try:
            from vision_3d_acquisition.pipelines.registry import list_processing_unit_definitions

            units = list_processing_unit_definitions(pipeline)
        except Exception:
            units = []
        stage_order = []
        for unit in sorted(
            [x for x in units if isinstance(x, dict)],
            key=lambda x: (int(x.get("order", 0)), str(x.get("id", ""))),
        ):
            stage = str(unit.get("stage_id") or unit.get("id") or "unknown")
            if stage not in stage_order:
                stage_order.append(stage)
        if not stage_order:
            stage_order = [
                "detect_belt_plane",
                "normalize_heights_to_plane",
                "remove_belt_segment_objects",
                "extract_connected_components",
                "fit_object_geometry",
                "compute_height_metrics",
                "classification",
                "overlay",
            ]
        columns = [
            {
                "id": stage,
                "label": stage.replace("_", " ").title(),
                "stage_id": stage,
                "order": index,
            }
            for index, stage in enumerate(stage_order)
        ]
        rows = []
        for case in execution.get("cases") or []:
            comparisons = [x for x in case.get("comparisons") or [] if isinstance(x, dict)]
            cells = []
            for column in columns:
                matches = [
                    x
                    for x in comparisons
                    if (
                        self.get_baseline(
                            str((x.get("baseline_ref") or {}).get("baseline_id")) or ""
                        )
                        or {}
                    ).get("stage_id")
                    == column["stage_id"]
                ]
                statuses = [str(x.get("status")) for x in matches]
                status = next(
                    (
                        s
                        for s in (
                            "regression",
                            "not_comparable",
                            "needs_review",
                            "changed",
                            "blocked",
                            "pass",
                        )
                        if s in statuses
                    ),
                    "missing_baseline",
                )
                divergence = case.get("first_divergence") or {}
                cells.append(
                    {
                        "column_id": column["id"],
                        "status": status,
                        "comparison_ids": [
                            (x.get("baseline_ref") or {}).get("baseline_id") for x in matches
                        ],
                        "first_divergence": divergence.get("stage_id") == column["stage_id"],
                        "artifact_id": (
                            divergence.get("artifact_id")
                            if divergence.get("stage_id") == column["stage_id"]
                            else None
                        ),
                    }
                )
            rows.append(
                {
                    "case_id": case.get("case_id"),
                    "take_id": case.get("take_id"),
                    "cells": cells,
                    "overall_status": case.get("status"),
                }
            )
        return {"execution_id": execution_id, "columns": columns, "rows": rows}

    def baseline_history(self, **filters: Any) -> list[dict[str, Any]]:
        rows = []
        for baseline in self.list_baselines(
            take_id=filters.get("take_id"), pipeline_id=filters.get("pipeline_id")
        ):
            detail = self.get_baseline(str(baseline.get("id")))
            if not detail:
                continue
            if any(
                filters.get(key) is not None and detail.get(key) != filters[key]
                for key in ("stage_id", "processing_unit_id", "artifact_id", "source_run_id")
            ):
                continue
            if filters.get("active") is not None and (detail.get("status") == "active") != bool(
                filters["active"]
            ):
                continue
            snapshot = (
                self.root
                / "baselines"
                / str(detail["id"])
                / str(detail.get("snapshot_artifact_path") or "")
            )
            detail["integrity_state"] = (
                "ok"
                if snapshot.is_file() and detail.get("artifact_checksum") == _checksum(snapshot)
                else "invalid"
            )
            rows.append(detail)
        return sorted(
            rows,
            key=lambda x: (str(x.get("created_at") or ""), int(x.get("version") or 0)),
            reverse=True,
        )

    def resolution_impact(self, baseline: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
        result = {
            "active_cases": [],
            "pinned_cases_unchanged": [],
            "allowed_version_cases_unchanged": [],
            "disabled_cases": [],
            "archived_suites": [],
        }
        for suite in self.list_suites():
            full = self.get_suite(str(suite["id"]))
            for case in full["cases"]:
                family = [self.get_baseline(str(x)) for x in case.get("baseline_ids") or []]
                related = any(
                    x
                    and x.get("artifact_id") == baseline.get("artifact_id")
                    and x.get("stage_id") == baseline.get("stage_id")
                    and x.get("take_id") == baseline.get("take_id")
                    for x in family
                )
                if not related:
                    continue
                entry = {"suite_id": suite["id"], "case_id": case["id"]}
                if suite.get("status") == "archived":
                    result["archived_suites"].append(entry)
                elif not case.get("enabled", True):
                    result["disabled_cases"].append(entry)
                elif (case.get("baseline_resolution") or {}).get("mode", "active") == "pinned":
                    result["pinned_cases_unchanged"].append(entry)
                elif (case.get("baseline_resolution") or {}).get("mode") == "allowed_versions":
                    result["allowed_version_cases_unchanged"].append(entry)
                else:
                    result["active_cases"].append(entry)
        return result

    def _active_of_family(self, baseline: Mapping[str, Any]) -> dict[str, Any] | None:
        """The version a future active-mode case would resolve to, or None."""
        family = [
            row
            for row in self.list_baselines(
                take_id=baseline.get("take_id"), pipeline_id=baseline.get("pipeline_id")
            )
            if row.get("status") == "active"
            and row.get("artifact_id") == baseline.get("artifact_id")
            and row.get("stage_id") == baseline.get("stage_id")
        ]
        if not family:
            return None
        return self.get_baseline(str(max(family, key=lambda x: int(x.get("version", 0))).get("id")))

    def _promotion_response(
        self, *, baseline: Mapping[str, Any], source: Mapping[str, Any], reused: bool
    ) -> dict[str, Any]:
        """Describe state after the promotion.  ``source`` was read before it and is
        stale by now, so every reported record is re-read from disk."""
        return {
            "baseline": dict(baseline),
            "already_promoted": reused,
            "previous_active_baseline": self.get_baseline(str(source.get("id"))) or dict(source),
            "resulting_active_baseline": self._active_of_family(baseline),
            "affected_case_resolution": self.resolution_impact(baseline),
        }

    def promote_comparison(
        self, execution_id: str, case_id: str, comparison_id: str, request: Mapping[str, Any]
    ) -> dict[str, Any]:
        execution = self.get_execution(execution_id)
        case = next((x for x in execution.get("cases") or [] if x.get("case_id") == case_id), None)
        if not case:
            raise KeyError(case_id)
        comparison = next(
            (
                x
                for x in case.get("comparisons") or []
                if str((x.get("baseline_ref") or {}).get("baseline_id")) == comparison_id
            ),
            None,
        )
        if not comparison:
            raise KeyError(comparison_id)
        source = self.get_baseline(comparison_id)
        if not source:
            raise KeyError(comparison_id)
        candidate_run = str(execution.get("candidate_run_id") or "latest")
        candidate = self._resolve(
            take_id=source["take_id"], pipeline_id=source["pipeline_id"], run_id=candidate_run
        )
        artifact = next(
            (
                x
                for x in candidate.get("artifacts") or []
                if isinstance(x, dict)
                and x.get("artifact_id") == source.get("artifact_id")
                and x.get("stage_id") == source.get("stage_id")
            ),
            None,
        )
        if not artifact:
            raise ValueError("candidate artifact missing")
        path = Path(str(artifact.get("path") or ""))
        path = path if path.is_absolute() else Path(candidate["artifact_root"]) / path
        if not path.is_file():
            raise ValueError("candidate artifact missing")
        checksum = _checksum(path)
        for existing in self.baseline_history(
            take_id=source["take_id"], pipeline_id=source["pipeline_id"]
        ):
            promotion = existing.get("promotion") or {}
            if (
                promotion.get("source_execution_id") == execution_id
                and promotion.get("source_case_id") == case_id
                and promotion.get("source_comparison_id") == comparison_id
                and promotion.get("candidate_artifact_checksum") == checksum
            ):
                return self._promotion_response(baseline=existing, source=source, reused=True)
        policy = (
            source.get("comparison_policy") if request.get("carry_forward_policy", True) else {}
        )
        created = self.create_baselines(
            {
                "take_id": source["take_id"],
                "pipeline_id": source["pipeline_id"],
                "run_id": candidate_run,
                "approval_scope": "artifact",
                "stage_id": source.get("stage_id"),
                "processing_unit_id": source.get("processing_unit_id"),
                "view_id": source.get("view_id"),
                "artifact_id": source["artifact_id"],
                "notes": request.get("notes") or "",
                "reviewed_by": request.get("reviewed_by"),
                "comparison_policy": policy,
                "allow_alternative": not bool(request.get("activate")),
            }
        )["baselines"]
        if not created:
            raise ValueError("candidate artifact cannot be promoted")
        promoted = created[0]
        if promoted.get("artifact_checksum") != checksum:
            raise ValueError("candidate artifact changed while it was being promoted")
        promoted["promotion"] = {
            "source_execution_id": execution_id,
            "source_case_id": case_id,
            "source_comparison_id": comparison_id,
            "candidate_run_id": candidate_run,
            "candidate_artifact_checksum": checksum,
            "promoted_at": _now(),
            "promoted_by": request.get("reviewed_by"),
            "notes": request.get("notes") or "",
        }
        if not request.get("activate"):
            self.set_active(promoted["id"], False)
            promoted["status"] = "inactive"
        _write(self.root / "baselines" / promoted["id"] / "baseline.json", promoted)
        return self._promotion_response(baseline=promoted, source=source, reused=False)

    def _case_baselines(self, case: Mapping[str, Any]) -> list[str]:
        configured = [str(x) for x in case.get("baseline_ids") or []]
        resolution = (
            case.get("baseline_resolution")
            if isinstance(case.get("baseline_resolution"), dict)
            else {"mode": "active"}
        )
        mode = resolution.get("mode", "active")
        if mode == "pinned":
            return [str(resolution.get("baseline_id"))]
        if mode == "allowed_versions":
            return [str(x) for x in resolution.get("allowed_baseline_ids") or []]
        # Legacy cases are active mode. Preserve each artifact identity while
        # resolving the latest explicitly active version.
        resolved = []
        for baseline_id in configured:
            baseline = self.get_baseline(baseline_id)
            if not baseline:
                resolved.append(baseline_id)
                continue
            siblings = [
                b
                for b in self.list_baselines(
                    take_id=baseline.get("take_id"), pipeline_id=baseline.get("pipeline_id")
                )
                if b.get("status") == "active"
                and b.get("artifact_id") == baseline.get("artifact_id")
                and b.get("stage_id") == baseline.get("stage_id")
            ]
            resolved.append(
                str(
                    (
                        max(siblings, key=lambda x: int(x.get("version", 0)))
                        if siblings
                        else baseline
                    ).get("id")
                )
            )
        return resolved

    def execute_suite(self, suite_id: str, request: Mapping[str, Any]) -> dict[str, Any]:
        """Compare already-created candidate runs. Execution is intentionally separate from pipeline mutation."""
        suite = self.get_suite(suite_id)
        eid = _id("execution")
        root = self.root / "executions" / eid
        report = {
            "id": eid,
            "suite_id": suite_id,
            "pipeline_id": suite["pipeline_id"],
            "status": "running",
            "created_at": _now(),
            "started_at": _now(),
            "candidate_run_id": request.get("candidate_run_id", "latest"),
            "cases": [],
            "progress": {"completed": 0, "total": len(suite["cases"]), "status": "running"},
        }
        _write(root / "execution.json", report)
        for case in suite["cases"]:
            if not case.get("enabled", True):
                result = {
                    "case_id": case["id"],
                    "take_id": case["take_id"],
                    "status": "changed",
                    "skipped": True,
                    "summary": "Case disabled.",
                }
            else:
                comparisons = []
                upstream_blocked = False
                for bid in self._case_baselines(case):
                    baseline = self.get_baseline(str(bid))
                    if upstream_blocked:
                        comparisons.append(
                            self._result(
                                "blocked",
                                baseline or {},
                                None,
                                "An upstream comparison invalidates this downstream interpretation.",
                            )
                        )
                        continue
                    try:
                        outcome = self.compare(
                            bid,
                            candidate_run_id=str(request.get("candidate_run_id") or "latest"),
                            diff_dir=root / "cases" / str(case["id"]) / "diffs" / str(bid),
                        )
                        comparisons.append(outcome)
                        if outcome["status"] == "regression" and bool(
                            (baseline or {})
                            .get("comparison_policy", {})
                            .get("blocksDownstreamOnFailure")
                        ):
                            upstream_blocked = True
                    except (KeyError, ValueError) as exc:
                        comparisons.append(
                            {
                                "status": "not_comparable",
                                "summary": str(exc),
                                "baseline_ref": {"baseline_id": bid},
                            }
                        )
                statuses = [x["status"] for x in comparisons]
                allowed_mode = (
                    isinstance(case.get("baseline_resolution"), dict)
                    and case["baseline_resolution"].get("mode") == "allowed_versions"
                )
                overall = (
                    "pass"
                    if allowed_mode and "pass" in statuses
                    else (
                        "regression"
                        if "regression" in statuses
                        else (
                            "needs_review"
                            if "needs_review" in statuses
                            else (
                                "not_comparable"
                                if "not_comparable" in statuses
                                else "changed" if "changed" in statuses else "pass"
                            )
                        )
                    )
                )
                first = next((x for x in comparisons if x["status"] == "regression"), None)
                first_baseline = (
                    self.get_baseline(first["baseline_ref"]["baseline_id"])
                    if first and first.get("baseline_ref", {}).get("baseline_id")
                    else None
                )
                divergence = (
                    {
                        "take_id": case["take_id"],
                        "stage_id": first_baseline.get("stage_id"),
                        "processing_unit_id": first_baseline.get("processing_unit_id"),
                        "substage_id": first_baseline.get("substage_id"),
                        "view_id": first_baseline.get("view_id"),
                        "artifact_id": first_baseline.get("artifact_id"),
                        "status": first["status"],
                        "summary": first["summary"],
                    }
                    if first_baseline
                    else None
                )
                result = {
                    "case_id": case["id"],
                    "take_id": case["take_id"],
                    "status": overall,
                    "comparisons": comparisons,
                    "matched_baseline_id": next(
                        (
                            x.get("baseline_ref", {}).get("baseline_id")
                            for x in comparisons
                            if x.get("status") == "pass"
                        ),
                        None,
                    ),
                    "first_failing_stage": (divergence or {}).get("stage_id"),
                    "first_divergence": divergence,
                }
            report["cases"].append(result)
            report["progress"]["completed"] += 1
            _write(root / "execution.json", report)
        statuses = [x["status"] for x in report["cases"]]
        report["status"] = "completed_with_regressions" if "regression" in statuses else "completed"
        report["progress"]["status"] = report["status"]
        report["completed_at"] = _now()
        _write(root / "execution.json", report)
        rows = self._items("executions")
        rows.append(
            {
                k: report[k]
                for k in ("id", "suite_id", "pipeline_id", "status", "created_at", "completed_at")
            }
        )
        self._set_items("executions", rows)
        return report
