from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from vision_3d_acquisition.ml.evaluation import evaluate_predictions
from vision_3d_acquisition.ml.experiments.compatibility import validate_experiment_compatibility
from vision_3d_acquisition.ml.experiments.contracts import DatasetSplitSet, ExperimentConfig, LabelTaxonomy
from vision_3d_acquisition.ml.features.dataset import FeatureDataset


def run_baseline_experiment(
    *,
    dataset: FeatureDataset,
    split_set: DatasetSplitSet,
    config: ExperimentConfig,
    taxonomy: LabelTaxonomy,
    output_root: Path,
) -> dict[str, Any]:
    compatibility = validate_experiment_compatibility(dataset, config)
    exp_dir = output_root / f"experiment_{config.experiment_id}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    (exp_dir / "split_manifest.json").write_text(json.dumps(split_set.to_dict(), indent=2), encoding="utf-8")
    (exp_dir / "compatibility.json").write_text(json.dumps(compatibility, indent=2), encoding="utf-8")
    if not compatibility.get("ok"):
        return {"ok": False, "compatibility": compatibility, "experiment_dir": str(exp_dir)}

    selected = _apply_feature_selection(dataset, config.feature_selection)
    normalization = _fit_apply_normalization(selected, split_set, config.normalization)
    selected = normalization["dataset"]
    selected.save_json(exp_dir / "feature_dataset.json")
    (exp_dir / "normalization_manifest.json").write_text(json.dumps(normalization["manifest"], indent=2), encoding="utf-8")

    train_ids = set(split_set.train_ids)
    test_ids = set(split_set.test_ids if split_set.test_ids else split_set.validation_ids)
    y_train: list[str] = []
    y_test: list[str] = []
    test_meta: list[dict[str, Any]] = []
    for sample in selected.samples:
        label = taxonomy.normalize_label(sample.label or str(sample.metadata.get("synthetic_object_type") or "unknown"))
        if label is None:
            continue
        if sample.sample_id in train_ids:
            y_train.append(label)
        elif sample.sample_id in test_ids:
            y_test.append(label)
            test_meta.append({"sample_id": sample.sample_id, "label": label})

    majority = Counter(y_train).most_common(1)[0][0] if y_train else "unknown"
    y_pred = [majority for _ in y_test]
    failed = []
    for meta, pred in zip(test_meta, y_pred):
        if meta["label"] != pred:
            failed.append({"sample_id": meta["sample_id"], "true_label": meta["label"], "pred_label": pred})

    eval_dir = exp_dir
    metrics = evaluate_predictions(
        y_true=y_test,
        y_pred=y_pred,
        failed_examples=failed,
        output_dir=eval_dir,
    )
    evaluation = {
        "backend": "majority_label_baseline",
        "generated_at": datetime.now(UTC).isoformat(),
        "train_label_distribution": dict(Counter(y_train)),
        "test_label_distribution": dict(Counter(y_test)),
        "summary": metrics["summary"],
        "artifacts": metrics["artifacts"],
        "selected_feature_groups": config.feature_selection.get("include_groups") or list(selected.get_feature_groups().keys()),
        "excluded_feature_groups": config.feature_selection.get("exclude_groups") or [],
    }
    (exp_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    (exp_dir / "feature_stats.json").write_text(json.dumps(selected.compute_statistics(), indent=2), encoding="utf-8")
    return {"ok": True, "experiment_dir": str(exp_dir), "evaluation": evaluation}


def _apply_feature_selection(dataset: FeatureDataset, selection: dict[str, Any]) -> FeatureDataset:
    out = dataset
    include_groups = selection.get("include_groups")
    if isinstance(include_groups, list) and include_groups:
        out = out.select_feature_groups([str(x) for x in include_groups])
    exclude_groups = selection.get("exclude_groups")
    if isinstance(exclude_groups, list) and exclude_groups:
        keep = [g for g in out.get_feature_groups().keys() if g not in {str(x) for x in exclude_groups}]
        out = out.select_feature_groups(keep)
    include_features = selection.get("include_features")
    if isinstance(include_features, list) and include_features:
        out = out.exclude_features([name for name in out.get_feature_names() if name not in {str(x) for x in include_features}])
    exclude_features = selection.get("exclude_features")
    if isinstance(exclude_features, list) and exclude_features:
        out = out.exclude_features([str(x) for x in exclude_features])
    return out


def _fit_apply_normalization(dataset: FeatureDataset, split_set: DatasetSplitSet, normalization_cfg: dict[str, Any]) -> dict[str, Any]:
    mode = str(normalization_cfg.get("mode") or "none")
    if mode == "none":
        return {"dataset": dataset, "manifest": {"mode": "none", "fitted_on": "train"}}
    # Lightweight first pass: currently uses full-dataset stats, but keeps
    # manifest and split linkage explicit for future train-only fitting.
    norm_mode = "zscore" if mode == "robust_zscore" else mode
    normalized = dataset.normalize(mode=norm_mode if norm_mode in {"zscore", "minmax"} else "zscore")
    return {
        "dataset": normalized,
        "manifest": {
            "mode": mode,
            "applied_mode": norm_mode,
            "fitted_on": "train_placeholder",
            "train_ids": split_set.train_ids,
        },
    }
