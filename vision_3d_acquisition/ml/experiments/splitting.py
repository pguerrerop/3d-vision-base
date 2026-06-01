from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from vision_3d_acquisition.ml.experiments.contracts import DatasetSplitSet
from vision_3d_acquisition.ml.features.dataset import FeatureDataset


def create_split_set(
    dataset: FeatureDataset,
    *,
    strategy: str = "stratified",
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 123,
) -> DatasetSplitSet:
    if abs((train_ratio + validation_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("ratios must sum to 1.0")
    rng = random.Random(seed)
    sample_ids = [sample.sample_id for sample in dataset.samples]
    by_group: dict[str, list[str]] = {}
    mode = strategy.strip().lower()
    if mode == "random":
        ids = sorted(sample_ids)
        rng.shuffle(ids)
        train, val, test = _cut(ids, train_ratio, validation_ratio)
    else:
        for sample in dataset.samples:
            key = _group_key(sample, mode=mode)
            by_group.setdefault(key, []).append(sample.sample_id)
        train, val, test = [], [], []
        for key in sorted(by_group.keys()):
            ids = sorted(by_group[key])
            rng.shuffle(ids)
            a, b, c = _cut(ids, train_ratio, validation_ratio)
            train.extend(a)
            val.extend(b)
            test.extend(c)
        train.sort()
        val.sort()
        test.sort()
    return DatasetSplitSet(
        dataset_id=dataset.dataset_id,
        split_strategy=mode,
        seed=seed,
        train_ids=train,
        validation_ids=val,
        test_ids=test,
        metadata={
            "ratios": {"train": train_ratio, "validation": validation_ratio, "test": test_ratio},
            "sample_count": len(sample_ids),
        },
    )


def save_split_manifest(split_set: DatasetSplitSet, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split_set.to_dict(), indent=2), encoding="utf-8")


def load_split_manifest(path: Path) -> DatasetSplitSet:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return DatasetSplitSet(
        dataset_id=str(payload.get("dataset_id") or ""),
        split_strategy=str(payload.get("split_strategy") or "random"),
        seed=int(payload.get("seed") or 0),
        train_ids=[str(x) for x in payload.get("train_ids") or []],
        validation_ids=[str(x) for x in payload.get("validation_ids") or []],
        test_ids=[str(x) for x in payload.get("test_ids") or []],
        metadata=dict(payload.get("metadata") or {}),
    )


def _group_key(sample: Any, *, mode: str) -> str:
    if mode == "stratified":
        return str(sample.label or sample.metadata.get("synthetic_object_type") or "unknown")
    if mode == "by_synthetic_object_type":
        return str(sample.metadata.get("synthetic_object_type") or "unknown")
    if mode == "by_failure_family":
        return str(sample.metadata.get("dominant_failure_family") or "unknown")
    return str(sample.sample_id)


def _cut(ids: list[str], train_ratio: float, val_ratio: float) -> tuple[list[str], list[str], list[str]]:
    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = ids[:n_train]
    val = ids[n_train : n_train + n_val]
    test = ids[n_train + n_val :]
    return train, val, test
