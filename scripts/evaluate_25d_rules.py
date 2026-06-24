from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vision_3d_acquisition.classifiers.mining_ball_rules import (
    load_classifier_rule_config,
    predict_superclass_from_rules,
)
from scripts.tune_25d_rules import (
    DEFAULT_LABEL_TO_SUPERCLASS,
    compute_metrics,
    make_object_safe_splits,
)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh) if isinstance(row, dict)]


def _expected_superclass(row: dict[str, Any], label_column: str, superclass_column: str) -> str | None:
    sup = str(row.get(superclass_column) or "").strip().upper()
    if sup in {"BALL_GOOD", "BALL_SCRAP", "SCRAP_METAL"}:
        return sup
    lbl = str(row.get(label_column) or "").strip().lower()
    return DEFAULT_LABEL_TO_SUPERCLASS.get(lbl)


def _predict_rows(rows: list[dict[str, Any]], params: dict[str, float], *, group_column: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        pred = predict_superclass_from_rules(row, params)
        out.append(
            {
                **row,
                "predicted_superclass": pred.superclass,
                "rule_path": pred.rule_path,
                "confidence_proxy": pred.confidence_proxy,
                "correct": pred.superclass == row["expected_superclass"],
                "physical_object_id": row.get(group_column),
            }
        )
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols = fieldnames or sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in cols})


def _evaluate(
    rows: list[dict[str, Any]],
    *,
    params: dict[str, float],
    group_column: str,
    split: str | None = None,
) -> dict[str, Any]:
    subset = [row for row in rows if split is None or str(row.get("split") or "") == split]
    preds = _predict_rows(subset, params, group_column=group_column)
    metrics = compute_metrics(preds)
    metrics["predictions"] = preds
    return metrics


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = _load_csv(args.features_csv)
    prepared: list[dict[str, Any]] = []
    for row in rows:
        sup = _expected_superclass(row, label_column=args.label_column, superclass_column=args.superclass_column)
        if sup is None:
            continue
        row["expected_superclass"] = sup
        if not row.get(args.group_column):
            row[args.group_column] = row.get("take_id")
        prepared.append(row)
    if not prepared:
        raise ValueError("no labeled rows available for evaluation")
    has_split = any(str(row.get("split") or "").strip() for row in prepared)
    if not has_split:
        make_object_safe_splits(
            prepared,
            group_column=args.group_column,
            seed=args.seed,
            train_name=args.train_split,
            validation_name=args.validation_split,
            test_name=args.test_split,
        )

    base_cfg = load_classifier_rule_config(args.rules_config)
    base_eval = _evaluate(prepared, params=base_cfg["params"], group_column=args.group_column, split=None)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        out_dir / "predictions.csv",
        [
            {
                "take_id": row.get("take_id"),
                "physical_object_id": row.get(args.group_column),
                "split": row.get("split"),
                "expected_superclass": row.get("expected_superclass"),
                "predicted_superclass": row.get("predicted_superclass"),
                "correct": row.get("correct"),
                "confidence_proxy": row.get("confidence_proxy"),
                "rule_path": row.get("rule_path"),
                "rule_set_id": Path(str(base_cfg.get("path") or args.rules_config)).stem,
                "rule_set_version": base_cfg.get("version"),
                "rule_set_source": "rules_config",
            }
            for row in base_eval["predictions"]
        ],
        [
            "take_id",
            "physical_object_id",
            "split",
            "expected_superclass",
            "predicted_superclass",
            "correct",
            "confidence_proxy",
            "rule_path",
            "rule_set_id",
            "rule_set_version",
            "rule_set_source",
        ],
    )
    _write_csv(
        out_dir / "misclassified.csv",
        [
            row
            for row in base_eval["predictions"]
            if not bool(row.get("correct"))
        ],
    )
    cm_rows = []
    cm = base_eval["confusion_matrix"]
    classes = ["BALL_GOOD", "BALL_SCRAP", "SCRAP_METAL"]
    for exp in classes:
        cm_rows.append({"expected": exp, **{pred: cm.get(exp, {}).get(pred, 0) for pred in classes}})
    _write_csv(out_dir / "confusion_matrix.csv", cm_rows, ["expected", *classes])
    _write_csv(
        out_dir / "per_class_metrics.csv",
        [{"class": cls, **vals} for cls, vals in base_eval["per_class"].items()],
        ["class", "precision", "recall", "f1", "tp", "fp", "fn", "tn"],
    )

    report: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "features_csv": str(args.features_csv),
        "rules_config": str(args.rules_config),
        "rule_set_id": Path(str(base_cfg.get("path") or args.rules_config)).stem,
        "rule_set_version": base_cfg.get("version"),
        "rule_set_source": "rules_config",
        "rows_used": len(prepared),
        "object_count": len({str(r.get(args.group_column) or r.get("take_id") or "") for r in prepared}),
        "metrics": {k: base_eval[k] for k in ("accuracy", "macro_f1", "balanced_accuracy", "rows")},
    }

    if args.compare_rules_config:
        cmp_cfg = load_classifier_rule_config(args.compare_rules_config)
        cmp_eval = _evaluate(prepared, params=cmp_cfg["params"], group_column=args.group_column, split=None)
        delta = {
            "accuracy": float(cmp_eval["accuracy"]) - float(base_eval["accuracy"]),
            "macro_f1": float(cmp_eval["macro_f1"]) - float(base_eval["macro_f1"]),
            "balanced_accuracy": float(cmp_eval["balanced_accuracy"]) - float(base_eval["balanced_accuracy"]),
        }
        base_by_take = {str(r.get("take_id") or ""): r for r in base_eval["predictions"]}
        cmp_by_take = {str(r.get("take_id") or ""): r for r in cmp_eval["predictions"]}
        disagreements: list[dict[str, Any]] = []
        for take_id in sorted(set(base_by_take.keys()) | set(cmp_by_take.keys())):
            b = base_by_take.get(take_id)
            c = cmp_by_take.get(take_id)
            if not b or not c:
                continue
            if b.get("predicted_superclass") != c.get("predicted_superclass"):
                disagreements.append(
                    {
                        "take_id": take_id,
                        "physical_object_id": b.get(args.group_column),
                        "expected_superclass": b.get("expected_superclass"),
                        "base_predicted_superclass": b.get("predicted_superclass"),
                        "compare_predicted_superclass": c.get("predicted_superclass"),
                        "base_rule_path": b.get("rule_path"),
                        "compare_rule_path": c.get("rule_path"),
                    }
                )
        _write_csv(out_dir / "disagreements.csv", disagreements)
        report["compare"] = {
            "compare_rules_config": str(args.compare_rules_config),
            "compare_metrics": {k: cmp_eval[k] for k in ("accuracy", "macro_f1", "balanced_accuracy", "rows")},
            "delta": delta,
            "disagreement_count": len(disagreements),
        }

    (out_dir / "evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate external 25D rule config(s) against exported feature CSV.")
    p.add_argument("--features-csv", type=Path, required=True)
    p.add_argument("--rules-config", type=Path, required=True)
    p.add_argument("--compare-rules-config", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--group-column", type=str, default="physical_object_id")
    p.add_argument("--label-column", type=str, default="expected_label")
    p.add_argument("--superclass-column", type=str, default="expected_superclass")
    p.add_argument("--train-split", type=str, default="train")
    p.add_argument("--validation-split", type=str, default="validation")
    p.add_argument("--test-split", type=str, default="test")
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = run(args)
    except ValueError as exc:
        print(str(exc))
        return 2
    print("Rule evaluation complete")
    print(f"- output_dir: {args.output_dir}")
    print(f"- rows_used: {report['rows_used']}")
    print(f"- macro_f1: {report['metrics']['macro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
