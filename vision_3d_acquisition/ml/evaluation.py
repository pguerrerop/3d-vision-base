from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def evaluate_predictions(*, y_true: list[str], y_pred: list[str], failed_examples: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    labels = sorted({*y_true, *y_pred})
    cm = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    per_class: list[dict[str, Any]] = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    for label in labels:
        tp = cm[label][label]
        fp = sum(cm[t][label] for t in labels if t != label)
        fn = sum(cm[label][p] for p in labels if p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class.append({"label": label, "precision": precision, "recall": recall, "f1": f1, "support": sum(cm[label].values())})
        total_tp += tp
        total_fp += fp
        total_fn += fn

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) else 0.0

    output_dir.mkdir(parents=True, exist_ok=True)
    cm_path = output_dir / "confusion_matrix.csv"
    with cm_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["true/pred", *labels])
        for t in labels:
            writer.writerow([t, *[cm[t][p] for p in labels]])

    per_class_path = output_dir / "per_class_metrics.csv"
    with per_class_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["label", "precision", "recall", "f1", "support"])
        writer.writeheader()
        for row in per_class:
            writer.writerow(row)

    failed_path = output_dir / "failed_examples.json"
    failed_path.write_text(json.dumps(failed_examples, indent=2) + "\n", encoding="utf-8")

    summary = {
        "labels": labels,
        "accuracy": (sum(int(t == p) for t, p in zip(y_true, y_pred)) / len(y_true)) if y_true else 0.0,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "macro_f1": (sum(float(item["f1"]) for item in per_class) / len(per_class)) if per_class else 0.0,
        "balanced_accuracy": (sum(float(item["recall"]) for item in per_class) / len(per_class)) if per_class else 0.0,
        "per_class": per_class,
        "roc_auc": None,
        "calibration": {"status": "not_available"},
    }
    summary_path = output_dir / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    return {
        "summary": summary,
        "artifacts": {
            "confusion_matrix": str(cm_path),
            "per_class_metrics": str(per_class_path),
            "evaluation_summary": str(summary_path),
            "failed_examples": str(failed_path),
        },
    }
