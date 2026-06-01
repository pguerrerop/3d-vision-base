from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vision_3d_acquisition.ml.features.dataset import FeatureDataset, FeatureDefinition, FeatureSchema


@dataclass
class FeatureRegistry:
    feature_schema_version: str
    groups: dict[str, list[str]]
    features: dict[str, FeatureDefinition]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_schema(cls, schema: FeatureSchema) -> FeatureRegistry:
        return cls(
            feature_schema_version=schema.feature_schema_version,
            groups={k: list(v) for k, v in schema.feature_groups.items()},
            features=dict(schema.features),
            metadata=dict(schema.metadata),
        )

    @classmethod
    def from_dataset(cls, dataset: FeatureDataset) -> FeatureRegistry:
        return cls.from_schema(dataset.feature_schema)

    def to_schema(self) -> FeatureSchema:
        return FeatureSchema(
            feature_schema_version=self.feature_schema_version,
            feature_groups={k: list(v) for k, v in self.groups.items()},
            features=dict(self.features),
            metadata=dict(self.metadata),
        )

    def get_feature(self, name: str) -> FeatureDefinition | None:
        return self.features.get(name)

    def features_for_group(self, group: str) -> list[FeatureDefinition]:
        return [self.features[name] for name in self.groups.get(group, []) if name in self.features]

    def visible_features(self, *, audience: str) -> list[str]:
        key = audience.strip().lower()
        out: list[str] = []
        for name, fd in self.features.items():
            flags = fd.ux_visibility or {}
            if bool(flags.get(key)):
                out.append(name)
        return sorted(out)

    def validate_metadata(self) -> dict[str, Any]:
        errors: list[str] = []
        warnings: list[str] = []
        required_groups = {"footprint_geometry", "surface_geometry", "sphere_consistency", "damage_metrics"}
        missing_groups = sorted([g for g in required_groups if g not in self.groups])
        if missing_groups:
            errors.append(f"missing_groups:{','.join(missing_groups)}")
        for name, fd in self.features.items():
            if not fd.group:
                errors.append(f"{name}:missing_group")
            if fd.group and fd.group not in self.groups:
                errors.append(f"{name}:unknown_group:{fd.group}")
            if not fd.display_name:
                warnings.append(f"{name}:missing_display_name")
            if fd.higher_is_worse is None:
                warnings.append(f"{name}:missing_higher_is_worse")
            if not fd.normalization_hint:
                warnings.append(f"{name}:missing_normalization_hint")
            if not fd.ux_visibility:
                warnings.append(f"{name}:missing_ux_visibility")
        return {
            "ok": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "feature_count": len(self.features),
            "group_count": len(self.groups),
        }

    def validate_dataset_compatibility(self, dataset: FeatureDataset) -> dict[str, Any]:
        errors: list[str] = []
        warnings: list[str] = []
        if dataset.feature_schema.feature_schema_version != self.feature_schema_version:
            errors.append("schema_version_mismatch")
        expected = set()
        for names in self.groups.values():
            expected.update(names)
        observed = set(dataset.get_feature_names())
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        if missing:
            errors.append(f"missing_dataset_features:{','.join(missing)}")
        if extra:
            warnings.append(f"unknown_dataset_features:{','.join(extra)}")
        return {"ok": len(errors) == 0, "errors": errors, "warnings": warnings}
