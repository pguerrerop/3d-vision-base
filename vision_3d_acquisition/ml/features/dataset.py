from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    group: str
    display_name: str | None = None
    unit: str | None = None
    higher_is_worse: bool | None = None
    normalization_hint: str | None = None
    description: str | None = None
    short_description: str | None = None
    tooltip: str | None = None
    expected_range: tuple[float, float] | None = None
    recommended_warning_threshold: float | None = None
    recommended_critical_threshold: float | None = None
    ux_visibility: dict[str, bool] = field(default_factory=dict)
    display_precision: int | None = None
    preferred_visualization: str | None = None
    severity_direction: str | None = None


@dataclass
class FeatureSchema:
    feature_schema_version: str
    feature_groups: dict[str, list[str]]
    features: dict[str, FeatureDefinition]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ordered_feature_names(self) -> list[str]:
        out: list[str] = []
        for group in self.feature_groups:
            out.extend(self.feature_groups[group])
        return out

    def validate_vector(self, feature_vector: dict[str, Any]) -> dict[str, list[str]]:
        errors: list[str] = []
        warnings: list[str] = []
        expected = set(self.ordered_feature_names)
        provided = set(feature_vector.keys())
        missing = sorted(expected - provided)
        unknown = sorted(provided - expected)
        if missing:
            errors.append(f"missing_required_features:{','.join(missing)}")
        if unknown:
            warnings.append(f"unknown_features:{','.join(unknown)}")
        for name in expected & provided:
            value = feature_vector.get(name)
            if value is None:
                warnings.append(f"null_feature:{name}")
                continue
            try:
                _ = float(value)
            except Exception:
                errors.append(f"invalid_numeric_feature:{name}")
        return {"errors": errors, "warnings": warnings}

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_schema_version": self.feature_schema_version,
            "feature_groups": self.feature_groups,
            "features": {
                name: {
                    "name": fd.name,
                    "group": fd.group,
                    "display_name": fd.display_name,
                    "unit": fd.unit,
                    "higher_is_worse": fd.higher_is_worse,
                    "normalization_hint": fd.normalization_hint,
                    "description": fd.description,
                    "short_description": fd.short_description,
                    "tooltip": fd.tooltip,
                    "expected_range": list(fd.expected_range) if fd.expected_range is not None else None,
                    "recommended_warning_threshold": fd.recommended_warning_threshold,
                    "recommended_critical_threshold": fd.recommended_critical_threshold,
                    "ux_visibility": fd.ux_visibility,
                    "display_precision": fd.display_precision,
                    "preferred_visualization": fd.preferred_visualization,
                    "severity_direction": fd.severity_direction,
                }
                for name, fd in self.features.items()
            },
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FeatureSchema:
        feature_groups = payload.get("feature_groups") if isinstance(payload.get("feature_groups"), dict) else {}
        features_payload = payload.get("features") if isinstance(payload.get("features"), dict) else {}
        features: dict[str, FeatureDefinition] = {}
        for name, raw in features_payload.items():
            if not isinstance(raw, dict):
                continue
            features[str(name)] = FeatureDefinition(
                name=str(raw.get("name") or name),
                group=str(raw.get("group") or ""),
                display_name=(str(raw.get("display_name")) if raw.get("display_name") is not None else None),
                unit=(str(raw.get("unit")) if raw.get("unit") is not None else None),
                higher_is_worse=(bool(raw.get("higher_is_worse")) if raw.get("higher_is_worse") is not None else None),
                normalization_hint=(str(raw.get("normalization_hint")) if raw.get("normalization_hint") is not None else None),
                description=(str(raw.get("description")) if raw.get("description") is not None else None),
                short_description=(str(raw.get("short_description")) if raw.get("short_description") is not None else None),
                tooltip=(str(raw.get("tooltip")) if raw.get("tooltip") is not None else None),
                expected_range=(
                    (float(raw.get("expected_range")[0]), float(raw.get("expected_range")[1]))
                    if isinstance(raw.get("expected_range"), (list, tuple)) and len(raw.get("expected_range")) >= 2
                    else None
                ),
                recommended_warning_threshold=(
                    float(raw.get("recommended_warning_threshold")) if raw.get("recommended_warning_threshold") is not None else None
                ),
                recommended_critical_threshold=(
                    float(raw.get("recommended_critical_threshold")) if raw.get("recommended_critical_threshold") is not None else None
                ),
                ux_visibility=(
                    {str(k): bool(v) for k, v in dict(raw.get("ux_visibility") or {}).items()}
                    if isinstance(raw.get("ux_visibility"), dict)
                    else {}
                ),
                display_precision=(int(raw.get("display_precision")) if raw.get("display_precision") is not None else None),
                preferred_visualization=(str(raw.get("preferred_visualization")) if raw.get("preferred_visualization") is not None else None),
                severity_direction=(str(raw.get("severity_direction")) if raw.get("severity_direction") is not None else None),
            )
        return cls(
            feature_schema_version=str(payload.get("feature_schema_version") or "unknown"),
            feature_groups={str(k): [str(x) for x in (v or [])] for k, v in feature_groups.items()},
            features=features,
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass
class FeatureSample:
    sample_id: str
    take_id: str | None
    object_id: str | None
    label: str | None
    subclass: str | None
    feature_vector: dict[str, float | None]
    feature_groups: dict[str, dict[str, float | None]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "take_id": self.take_id,
            "object_id": self.object_id,
            "label": self.label,
            "subclass": self.subclass,
            "feature_vector": self.feature_vector,
            "feature_groups": self.feature_groups,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FeatureSample:
        return cls(
            sample_id=str(payload.get("sample_id") or ""),
            take_id=(str(payload.get("take_id")) if payload.get("take_id") is not None else None),
            object_id=(str(payload.get("object_id")) if payload.get("object_id") is not None else None),
            label=(str(payload.get("label")) if payload.get("label") is not None else None),
            subclass=(str(payload.get("subclass")) if payload.get("subclass") is not None else None),
            feature_vector={str(k): _to_float_or_none(v) for k, v in dict(payload.get("feature_vector") or {}).items()},
            feature_groups={
                str(group): {str(k): _to_float_or_none(v) for k, v in dict(vals or {}).items()}
                for group, vals in dict(payload.get("feature_groups") or {}).items()
            },
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass
class FeatureDataset:
    dataset_id: str
    name: str
    feature_schema: FeatureSchema
    created_at: str
    source: str
    samples: list[FeatureSample]
    metadata: dict[str, Any] = field(default_factory=dict)
    splits: dict[str, list[str]] = field(default_factory=lambda: {"train": [], "validation": [], "test": []})

    def get_feature_names(self, *, include_groups: list[str] | None = None) -> list[str]:
        if not include_groups:
            return list(self.feature_schema.ordered_feature_names)
        out: list[str] = []
        for group in include_groups:
            out.extend(self.feature_schema.feature_groups.get(group, []))
        return out

    def get_feature_groups(self) -> dict[str, list[str]]:
        return {k: list(v) for k, v in self.feature_schema.feature_groups.items()}

    def get_labels(self, *, field_name: str = "label") -> list[str | None]:
        if field_name == "subclass":
            return [sample.subclass for sample in self.samples]
        if field_name == "synthetic_object_type":
            return [sample.metadata.get("synthetic_object_type") for sample in self.samples]
        return [sample.label for sample in self.samples]

    def get_feature_matrix(
        self,
        *,
        include_groups: list[str] | None = None,
        exclude_features: list[str] | None = None,
    ) -> tuple[list[list[float]], list[str]]:
        names = self.get_feature_names(include_groups=include_groups)
        excluded = set(exclude_features or [])
        names = [name for name in names if name not in excluded]
        matrix: list[list[float]] = []
        for sample in self.samples:
            row = []
            for name in names:
                value = sample.feature_vector.get(name)
                row.append(float(value) if value is not None else 0.0)
            matrix.append(row)
        return matrix, names

    def select_feature_groups(self, groups: list[str]) -> FeatureDataset:
        selected_names = set(self.get_feature_names(include_groups=groups))
        return self._clone_with_feature_filter(include_feature_names=selected_names)

    def exclude_features(self, names: list[str]) -> FeatureDataset:
        excluded = set(names)
        include = [name for name in self.get_feature_names() if name not in excluded]
        return self._clone_with_feature_filter(include_feature_names=set(include))

    def validate(self) -> dict[str, Any]:
        errors: list[str] = []
        warnings: list[str] = []
        if not self.feature_schema.feature_schema_version:
            errors.append("missing_feature_schema_version")
        known_groups = set(self.feature_schema.feature_groups.keys())
        for sample in self.samples:
            if not sample.label and not sample.metadata.get("synthetic_object_type"):
                errors.append(f"missing_label:{sample.sample_id}")
            res = self.feature_schema.validate_vector(sample.feature_vector)
            errors.extend([f"{sample.sample_id}:{entry}" for entry in res["errors"]])
            warnings.extend([f"{sample.sample_id}:{entry}" for entry in res["warnings"]])
            for group in sample.feature_groups:
                if group not in known_groups:
                    errors.append(f"{sample.sample_id}:unknown_feature_group:{group}")
        return {
            "ok": len(errors) == 0,
            "error_count": len(errors),
            "warning_count": len(warnings),
            "errors": sorted(set(errors)),
            "warnings": sorted(set(warnings)),
            "sample_count": len(self.samples),
            "feature_count": len(self.feature_schema.ordered_feature_names),
        }

    def compute_statistics(self) -> dict[str, Any]:
        feature_stats: dict[str, dict[str, Any]] = {}
        for name in self.get_feature_names():
            values = [sample.feature_vector.get(name) for sample in self.samples]
            nums = [float(v) for v in values if v is not None]
            missing = sum(1 for v in values if v is None)
            if nums:
                mu = _mean(nums)
                std = _stdev(nums)
                feature_stats[name] = {
                    "mean": mu,
                    "stddev": std,
                    "min": min(nums),
                    "max": max(nums),
                    "missing_count": missing,
                }
            else:
                feature_stats[name] = {
                    "mean": None,
                    "stddev": None,
                    "min": None,
                    "max": None,
                    "missing_count": missing,
                }
        group_stats: dict[str, dict[str, Any]] = {}
        for group, names in self.feature_schema.feature_groups.items():
            vec = [feature_stats[name]["stddev"] for name in names if name in feature_stats and feature_stats[name]["stddev"] is not None]
            activation = []
            for sample in self.samples:
                vals = [sample.feature_vector.get(name) for name in names if sample.feature_vector.get(name) is not None]
                if vals:
                    activation.append(float(sum(vals) / len(vals)))
            group_stats[group] = {
                "feature_count": len(names),
                "aggregate_variance": float(sum(v * v for v in vec) / len(vec)) if vec else 0.0,
                "activation_mean": _mean(activation) if activation else 0.0,
                "activation_min": min(activation) if activation else 0.0,
                "activation_max": max(activation) if activation else 0.0,
            }
        return {"features": feature_stats, "groups": group_stats}

    def normalize(self, *, mode: str = "zscore") -> FeatureDataset:
        stats = self.compute_statistics()["features"]
        cloned = self._clone_with_feature_filter(include_feature_names=set(self.get_feature_names()))
        for sample in cloned.samples:
            for name in cloned.get_feature_names():
                v = sample.feature_vector.get(name)
                if v is None:
                    continue
                st = stats.get(name) or {}
                if mode == "minmax":
                    vmin = st.get("min")
                    vmax = st.get("max")
                    if vmin is None or vmax is None or float(vmax) <= float(vmin):
                        sample.feature_vector[name] = 0.0
                    else:
                        sample.feature_vector[name] = (float(v) - float(vmin)) / (float(vmax) - float(vmin))
                else:
                    mu = st.get("mean")
                    sd = st.get("stddev")
                    if mu is None or sd is None or float(sd) <= 1e-12:
                        sample.feature_vector[name] = 0.0
                    else:
                        sample.feature_vector[name] = (float(v) - float(mu)) / float(sd)
        return cloned

    def save_json(self, path: Path) -> None:
        payload = {
            "dataset_id": self.dataset_id,
            "name": self.name,
            "feature_schema": self.feature_schema.to_dict(),
            "created_at": self.created_at,
            "source": self.source,
            "metadata": self.metadata,
            "splits": self.splits,
            "samples": [sample.to_dict() for sample in self.samples],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: Path) -> FeatureDataset:
        payload = json.loads(path.read_text(encoding="utf-8"))
        schema = FeatureSchema.from_dict(dict(payload.get("feature_schema") or {}))
        samples = [FeatureSample.from_dict(dict(raw)) for raw in list(payload.get("samples") or []) if isinstance(raw, dict)]
        return cls(
            dataset_id=str(payload.get("dataset_id") or ""),
            name=str(payload.get("name") or ""),
            feature_schema=schema,
            created_at=str(payload.get("created_at") or ""),
            source=str(payload.get("source") or ""),
            samples=samples,
            metadata=dict(payload.get("metadata") or {}),
            splits=dict(payload.get("splits") or {"train": [], "validation": [], "test": []}),
        )

    @classmethod
    def from_advanced_validation_exports(
        cls,
        *,
        csv_path: Path,
        analysis_json_path: Path,
        dataset_id: str = "advanced_25d_feature_dataset",
        name: str = "Advanced 25D Feature Dataset",
    ) -> FeatureDataset:
        analysis = json.loads(analysis_json_path.read_text(encoding="utf-8"))
        version = str(analysis.get("feature_schema_version") or "v1")
        groups = analysis.get("feature_groups") if isinstance(analysis.get("feature_groups"), dict) else {}
        meta = analysis.get("feature_metadata") if isinstance(analysis.get("feature_metadata"), dict) else {}
        features: dict[str, FeatureDefinition] = {}
        for group, names in groups.items():
            for name in names or []:
                md = meta.get(name) if isinstance(meta.get(name), dict) else {}
                features[str(name)] = FeatureDefinition(
                    name=str(name),
                    group=str(group),
                    display_name=(str(md.get("display_name")) if md.get("display_name") is not None else str(name)),
                    unit=(str(md.get("unit")) if md.get("unit") is not None else None),
                    higher_is_worse=(bool(md.get("higher_is_worse")) if md.get("higher_is_worse") is not None else None),
                    normalization_hint=(str(md.get("normalization_hint")) if md.get("normalization_hint") is not None else "robust_zscore"),
                    description=(str(md.get("description")) if md.get("description") is not None else None),
                    short_description=(str(md.get("short_description")) if md.get("short_description") is not None else None),
                    tooltip=(str(md.get("tooltip")) if md.get("tooltip") is not None else None),
                    expected_range=(
                        (float(md.get("expected_range")[0]), float(md.get("expected_range")[1]))
                        if isinstance(md.get("expected_range"), (list, tuple)) and len(md.get("expected_range")) >= 2
                        else None
                    ),
                    recommended_warning_threshold=(
                        float(md.get("recommended_warning_threshold")) if md.get("recommended_warning_threshold") is not None else None
                    ),
                    recommended_critical_threshold=(
                        float(md.get("recommended_critical_threshold")) if md.get("recommended_critical_threshold") is not None else None
                    ),
                    ux_visibility=(
                        {str(k): bool(v) for k, v in dict(md.get("ux_visibility") or {}).items()}
                        if isinstance(md.get("ux_visibility"), dict)
                        else {"operations": False, "studio": True, "classifier_studio": True}
                    ),
                    display_precision=(int(md.get("display_precision")) if md.get("display_precision") is not None else 4),
                    preferred_visualization=(str(md.get("preferred_visualization")) if md.get("preferred_visualization") is not None else "histogram"),
                    severity_direction=(str(md.get("severity_direction")) if md.get("severity_direction") is not None else "higher_is_worse"),
                )
        schema = FeatureSchema(
            feature_schema_version=version,
            feature_groups={str(k): [str(x) for x in (v or [])] for k, v in groups.items()},
            features=features,
            metadata={"source_files": {"analysis_json": str(analysis_json_path)}},
        )
        samples: list[FeatureSample] = []
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fv: dict[str, float | None] = {}
                fg: dict[str, dict[str, float | None]] = {group: {} for group in schema.feature_groups}
                for feature_name in schema.ordered_feature_names:
                    value = _to_float_or_none(row.get(feature_name))
                    fv[feature_name] = value
                    group = schema.features.get(feature_name).group if schema.features.get(feature_name) else ""
                    if group:
                        fg.setdefault(group, {})[feature_name] = value
                metadata = {
                    "sample_origin": "synthetic",
                    "synthetic_object_type": row.get("synthetic_object_type"),
                    "expected_failure_modes": _parse_json_or_default(row.get("expected_failure_modes"), default=[]),
                    "expected_metric_family_reactions": _parse_json_or_default(row.get("expected_metric_family_reactions"), default={}),
                    "dominant_failure_family": row.get("dominant_failure_family"),
                    "secondary_failure_family": row.get("secondary_failure_family"),
                    "family_severity_ranking": _parse_json_or_default(row.get("family_severity_ranking"), default=[]),
                }
                samples.append(
                    FeatureSample(
                        sample_id=str(row.get("sample_id") or row.get("take_id") or ""),
                        take_id=(str(row.get("sample_id")) if row.get("sample_id") else None),
                        object_id=None,
                        label=(str(row.get("predicted_class")) if row.get("predicted_class") else None),
                        subclass=(str(row.get("predicted_subclass")) if row.get("predicted_subclass") else None),
                        feature_vector=fv,
                        feature_groups=fg,
                        metadata=metadata,
                    )
                )
        created_at = datetime.now(UTC).isoformat()
        return cls(
            dataset_id=dataset_id,
            name=name,
            feature_schema=schema,
            created_at=created_at,
            source="advanced_25d_geometry_validation_exports",
            samples=samples,
            metadata={"analysis_json_path": str(analysis_json_path), "csv_path": str(csv_path)},
            splits={"train": [], "validation": [], "test": []},
        )

    def _clone_with_feature_filter(self, *, include_feature_names: set[str]) -> FeatureDataset:
        new_groups: dict[str, list[str]] = {}
        new_features: dict[str, FeatureDefinition] = {}
        for group, names in self.feature_schema.feature_groups.items():
            kept = [name for name in names if name in include_feature_names]
            if kept:
                new_groups[group] = kept
                for name in kept:
                    if name in self.feature_schema.features:
                        new_features[name] = self.feature_schema.features[name]
        new_samples: list[FeatureSample] = []
        for sample in self.samples:
            fv = {name: sample.feature_vector.get(name) for name in include_feature_names if name in sample.feature_vector}
            fg: dict[str, dict[str, float | None]] = {}
            for group, names in new_groups.items():
                fg[group] = {name: fv.get(name) for name in names}
            new_samples.append(
                FeatureSample(
                    sample_id=sample.sample_id,
                    take_id=sample.take_id,
                    object_id=sample.object_id,
                    label=sample.label,
                    subclass=sample.subclass,
                    feature_vector=fv,
                    feature_groups=fg,
                    metadata=dict(sample.metadata),
                )
            )
        return FeatureDataset(
            dataset_id=self.dataset_id,
            name=self.name,
            feature_schema=FeatureSchema(
                feature_schema_version=self.feature_schema.feature_schema_version,
                feature_groups=new_groups,
                features=new_features,
                metadata=dict(self.feature_schema.metadata),
            ),
            created_at=self.created_at,
            source=self.source,
            samples=new_samples,
            metadata=dict(self.metadata),
            splits={k: list(v) for k, v in self.splits.items()},
        )


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    var = float(sum((x - mu) ** 2 for x in values) / (len(values) - 1))
    return var ** 0.5


def _to_float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _parse_json_or_default(value: Any, *, default: Any) -> Any:
    if value is None or value == "":
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except Exception:
        return default
