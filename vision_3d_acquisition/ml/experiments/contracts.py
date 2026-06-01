from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class DatasetSplitSet:
    dataset_id: str
    split_strategy: str
    seed: int
    train_ids: list[str]
    validation_ids: list[str]
    test_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "split_strategy": self.split_strategy,
            "seed": self.seed,
            "train_ids": sorted(self.train_ids),
            "validation_ids": sorted(self.validation_ids),
            "test_ids": sorted(self.test_ids),
            "metadata": self.metadata,
        }


@dataclass
class LabelTaxonomy:
    classes: list[str]
    aliases: dict[str, str] = field(default_factory=dict)
    groups: dict[str, list[str]] = field(default_factory=dict)
    ignored_labels: list[str] = field(default_factory=list)
    target_mode: str = "multiclass"  # multiclass | binary
    positive_class: str | None = None

    def normalize_label(self, label: str | None) -> str | None:
        if label is None:
            return None
        if label in self.ignored_labels:
            return None
        normalized = self.aliases.get(label, label)
        if self.target_mode == "binary":
            if self.positive_class is None:
                return normalized
            return self.positive_class if normalized == self.positive_class else f"non_{self.positive_class}"
        for group, values in self.groups.items():
            if normalized in values:
                return group
        return normalized

    def to_dict(self) -> dict[str, Any]:
        return {
            "classes": self.classes,
            "aliases": self.aliases,
            "groups": self.groups,
            "ignored_labels": self.ignored_labels,
            "target_mode": self.target_mode,
            "positive_class": self.positive_class,
        }


@dataclass
class ExperimentConfig:
    experiment_id: str
    name: str
    feature_schema_version: str
    dataset_id: str
    split_config: dict[str, Any]
    feature_selection: dict[str, Any]
    label_config: dict[str, Any]
    normalization: dict[str, Any]
    evaluation: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "feature_schema_version": self.feature_schema_version,
            "dataset_id": self.dataset_id,
            "split_config": self.split_config,
            "feature_selection": self.feature_selection,
            "label_config": self.label_config,
            "normalization": self.normalization,
            "evaluation": self.evaluation,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }
