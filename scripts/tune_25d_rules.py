from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vision_3d_acquisition.classifiers.mining_ball_rules import (
    DEFAULT_RULE_PARAMS,
    predict_superclass_from_rules as predict_superclass_from_rules_shared,
)

SUPERCLASSES = ("BALL_GOOD", "BALL_SCRAP", "SCRAP_METAL")

DEFAULT_LABEL_TO_SUPERCLASS = {
    "buena": "BALL_GOOD",
    "buena_menos": "BALL_GOOD",
    "ahuevada": "BALL_SCRAP",
    "mitad": "BALL_SCRAP",
    "bola_con_chip": "BALL_SCRAP",
    "duda": "BALL_SCRAP",
    "chica": "BALL_SCRAP",
    "cubo": "SCRAP_METAL",
    "cadena": "SCRAP_METAL",
    "parece_planchuela": "SCRAP_METAL",
    "planchuela": "SCRAP_METAL",
    "planchuela_doblada": "SCRAP_METAL",
    "perno": "SCRAP_METAL",
    "tuerca": "SCRAP_METAL",
    "chatarra": "SCRAP_METAL",
    "scrap": "SCRAP_METAL",
    "non_ball": "SCRAP_METAL",
}

DEFAULT_PARAMS: dict[str, float] = dict(DEFAULT_RULE_PARAMS)

PARAM_RANGES: dict[str, tuple[float, float]] = {
    "good_min_sphericity": (0.55, 0.90),
    "good_max_eccentricity": (0.35, 0.85),
    "good_min_flatness": (-0.20, 0.30),
    "good_max_edge_roughness": (5.0, 20.0),
    "good_min_height_mm": (4.0, 20.0),
    "good_max_height_mm": (70.0, 130.0),
    "deformed_min_eccentricity": (0.65, 0.95),
    "deformed_max_flatness": (-0.40, 0.20),
    "deformed_min_edge_roughness": (8.0, 25.0),
    "ball_scrap_min_sphericity": (0.40, 0.85),
    "scrap_max_sphericity": (0.10, 0.45),
    "scrap_max_height_mm": (4.0, 15.0),
    "scrap_max_p95_height_mm": (2.0, 10.0),
    "scrap_max_volume_proxy_mm3": (1000.0, 10000.0),
    "scrap_min_flatness": (-1.0, 0.0),
    "fallback_scrap_max_sphericity": (0.10, 0.45),
    "fallback_ball_scrap_max_sphericity": (0.55, 0.85),
}


@dataclass
class Prediction:
    superclass: str
    rule_path: str
    confidence_proxy: float


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool_or_false(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def expected_superclass_from_row(row: dict[str, Any], *, label_column: str = "expected_label", superclass_column: str = "expected_superclass") -> str | None:
    sup = str(row.get(superclass_column) or "").strip().upper()
    if sup in SUPERCLASSES:
        return sup
    label = str(row.get(label_column) or "").strip().lower()
    return DEFAULT_LABEL_TO_SUPERCLASS.get(label)


def _metric(row: dict[str, Any], key: str) -> float | None:
    direct = _safe_float(row.get(key))
    if direct is not None:
        return direct
    return _safe_float(row.get(f"diag_{key}"))


def predict_superclass_from_rules(row: dict[str, Any], params: dict[str, float]) -> Prediction:
    resolved = dict(row)
    for key in ("feature_sphericity_3d", "max_height_mm", "p95_height_mm", "feature_volume_proxy_mm3", "feature_flatness", "feature_eccentricity", "feature_edge_roughness", "border_touch_ratio", "invalid_pixel_ratio"):
        if key not in resolved and f"diag_{key}" in row:
            resolved[key] = row.get(f"diag_{key}")
    shared = predict_superclass_from_rules_shared(resolved, params)
    return Prediction(shared.superclass, shared.rule_path, shared.confidence_proxy)


def _margin_high_is_good(value: float | None, threshold: float) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(1.0, (value - threshold) / max(1e-9, abs(threshold) + 1.0)))


def _margin_high_is_bad(value: float | None, threshold: float) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(1.0, (threshold - value) / max(1e-9, abs(threshold) + 1.0)))


def _margin_range(value: float | None, low: float, high: float) -> float:
    if value is None:
        return 0.0
    if value < low:
        return 0.0
    if value > high:
        return 0.0
    center = (low + high) / 2.0
    half = max(1e-9, (high - low) / 2.0)
    return max(0.0, 1.0 - abs(value - center) / half)


def make_object_safe_splits(
    rows: list[dict[str, Any]],
    *,
    group_column: str,
    seed: int,
    train_name: str,
    validation_name: str,
    test_name: str,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = str(row.get(group_column) or row.get("take_id") or "")
        groups[key].append(row)
    keys = sorted(groups.keys())
    rnd = random.Random(seed)
    rnd.shuffle(keys)
    n = len(keys)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * validation_ratio))
    if n_train + n_val > n:
        n_val = max(0, n - n_train)
    train_set = set(keys[:n_train])
    val_set = set(keys[n_train:n_train + n_val])
    for key, items in groups.items():
        split = train_name if key in train_set else validation_name if key in val_set else test_name
        for row in items:
            row["split"] = split


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cm: dict[str, dict[str, int]] = {exp: {pred: 0 for pred in SUPERCLASSES} for exp in SUPERCLASSES}
    for row in rows:
        exp = str(row["expected_superclass"])
        pred = str(row["predicted_superclass"])
        cm.setdefault(exp, {k: 0 for k in SUPERCLASSES})
        cm[exp][pred] = cm[exp].get(pred, 0) + 1

    per_class = {}
    total = len(rows)
    correct = 0
    recalls = []
    for cls in SUPERCLASSES:
        tp = cm.get(cls, {}).get(cls, 0)
        fp = sum(cm.get(other, {}).get(cls, 0) for other in SUPERCLASSES if other != cls)
        fn = sum(cm.get(cls, {}).get(other, 0) for other in SUPERCLASSES if other != cls)
        tn = total - tp - fp - fn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class[cls] = {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}
        correct += tp
        recalls.append(recall)
    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(SUPERCLASSES)
    balanced_accuracy = sum(recalls) / len(SUPERCLASSES)
    accuracy = correct / total if total else 0.0
    return {"accuracy": accuracy, "macro_f1": macro_f1, "balanced_accuracy": balanced_accuracy, "per_class": per_class, "confusion_matrix": cm, "rows": total}


def evaluate_rule_params(rows: list[dict[str, Any]], params: dict[str, float], *, split: str | None = None) -> dict[str, Any]:
    subset = [row for row in rows if split is None or str(row.get("split") or "") == split]
    predicted_rows: list[dict[str, Any]] = []
    for row in subset:
        pred = predict_superclass_from_rules(row, params)
        predicted_rows.append(
            {
                **row,
                "predicted_superclass": pred.superclass,
                "rule_path": pred.rule_path,
                "confidence_proxy": round(pred.confidence_proxy, 6),
                "correct": pred.superclass == row["expected_superclass"],
            }
        )
    metrics = compute_metrics(predicted_rows)
    metrics["predictions"] = predicted_rows
    return metrics


def _score(metrics: dict[str, Any], metric_name: str) -> float:
    return float(metrics.get(metric_name) or 0.0)


def _sample_params(rnd: random.Random) -> dict[str, float]:
    params = dict(DEFAULT_PARAMS)
    for key, (lo, hi) in PARAM_RANGES.items():
        params[key] = rnd.uniform(lo, hi)
    # keep hierarchy sane
    if params["fallback_scrap_max_sphericity"] > params["fallback_ball_scrap_max_sphericity"]:
        params["fallback_scrap_max_sphericity"], params["fallback_ball_scrap_max_sphericity"] = (
            params["fallback_ball_scrap_max_sphericity"],
            params["fallback_scrap_max_sphericity"],
        )
    return params


def _grid_values(lo: float, hi: float, n: int = 4) -> list[float]:
    if n <= 1:
        return [lo]
    step = (hi - lo) / (n - 1)
    return [lo + i * step for i in range(n)]


def _iter_grid_params(max_evals: int) -> list[dict[str, float]]:
    keys = [
        "good_min_sphericity",
        "good_max_eccentricity",
        "good_min_flatness",
        "good_max_edge_roughness",
        "deformed_min_eccentricity",
        "deformed_max_flatness",
        "scrap_max_sphericity",
        "scrap_max_volume_proxy_mm3",
        "fallback_scrap_max_sphericity",
        "fallback_ball_scrap_max_sphericity",
    ]
    grids = {k: _grid_values(*PARAM_RANGES[k], n=3) for k in keys}
    out: list[dict[str, float]] = []
    for a in grids[keys[0]]:
        for b in grids[keys[1]]:
            for c in grids[keys[2]]:
                for d in grids[keys[3]]:
                    for e in grids[keys[4]]:
                        for f in grids[keys[5]]:
                            params = dict(DEFAULT_PARAMS)
                            params.update(
                                {
                                    keys[0]: a,
                                    keys[1]: b,
                                    keys[2]: c,
                                    keys[3]: d,
                                    keys[4]: e,
                                    keys[5]: f,
                                    keys[6]: grids[keys[6]][0],
                                    keys[7]: grids[keys[7]][1],
                                    keys[8]: grids[keys[8]][0],
                                    keys[9]: grids[keys[9]][1],
                                }
                            )
                            if params["fallback_scrap_max_sphericity"] > params["fallback_ball_scrap_max_sphericity"]:
                                continue
                            out.append(params)
                            if len(out) >= max_evals:
                                return out
    return out


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh) if isinstance(row, dict)]


def _find_latest_features_csv(*, data_dir: Path, ml_set_id: str) -> Path | None:
    candidates = [p for p in (data_dir / "ml_exports").glob("**/*.csv") if ml_set_id in str(p)]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _resolve_features_csv(args: argparse.Namespace) -> Path:
    if args.features_csv:
        return args.features_csv
    if args.data_dir and args.ml_set_id:
        found = _find_latest_features_csv(data_dir=args.data_dir, ml_set_id=args.ml_set_id)
        if found:
            return found
    raise ValueError("features CSV not found. Run `sensor_studio_cli.py ml export-features ...` or pass --features-csv.")


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


def run_tuning(args: argparse.Namespace) -> dict[str, Any]:
    features_csv = _resolve_features_csv(args)
    rows = _load_csv(features_csv)
    if not rows:
        raise ValueError(f"empty features CSV: {features_csv}")

    filtered: list[dict[str, Any]] = []
    for row in rows:
        expected = expected_superclass_from_row(row, label_column=args.label_column, superclass_column=args.superclass_column)
        if expected not in SUPERCLASSES:
            continue
        row["expected_superclass"] = expected
        if not row.get(args.group_column):
            row[args.group_column] = row.get("take_id")
        filtered.append(row)
    if not filtered:
        raise ValueError("no labeled rows found after expected label/superclass mapping")

    missing_columns = sorted(
        col
        for col in [
            "feature_sphericity_3d",
            "feature_eccentricity",
            "feature_flatness",
            "feature_edge_roughness",
            "max_height_mm",
            "p95_height_mm",
            "feature_volume_proxy_mm3",
        ]
        if not any((row.get(col) not in (None, "")) or (row.get(f"diag_{col}") not in (None, "")) for row in filtered)
    )

    has_split = any(str(row.get("split") or "").strip() for row in filtered)
    if not has_split:
        make_object_safe_splits(
            filtered,
            group_column=args.group_column,
            seed=args.seed,
            train_name=args.train_split,
            validation_name=args.validation_split,
            test_name=args.test_split,
        )

    validation_rows = [r for r in filtered if str(r.get("split") or "") == args.validation_split]
    train_rows = [r for r in filtered if str(r.get("split") or "") == args.train_split]
    test_rows = [r for r in filtered if str(r.get("split") or "") == args.test_split]
    target_split = args.validation_split if validation_rows else args.train_split

    best_params = dict(DEFAULT_PARAMS)
    best_eval = evaluate_rule_params(filtered, best_params, split=target_split)
    best_score = _score(best_eval, args.optimize_metric)
    history: list[dict[str, Any]] = [{"score": best_score, "params": dict(best_params)}]

    if args.search == "random":
        rnd = random.Random(args.seed)
        for _ in range(args.max_evals):
            params = _sample_params(rnd)
            ev = evaluate_rule_params(filtered, params, split=target_split)
            score = _score(ev, args.optimize_metric)
            history.append({"score": score, "params": params})
            if score > best_score:
                best_score = score
                best_eval = ev
                best_params = params
    else:
        for params in _iter_grid_params(args.max_evals):
            ev = evaluate_rule_params(filtered, params, split=target_split)
            score = _score(ev, args.optimize_metric)
            history.append({"score": score, "params": params})
            if score > best_score:
                best_score = score
                best_eval = ev
                best_params = params

    full_eval = evaluate_rule_params(filtered, best_params, split=None)
    train_eval = evaluate_rule_params(filtered, best_params, split=args.train_split) if train_rows else None
    val_eval = evaluate_rule_params(filtered, best_params, split=args.validation_split) if validation_rows else None
    test_eval = evaluate_rule_params(filtered, best_params, split=args.test_split) if test_rows else None

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    best_rules = {
        "classifier_id": "mining_steel_ball_classification_25d_rules",
        "version": f"tuned_{now}",
        "target_superclasses": list(SUPERCLASSES),
        "optimized_metric": args.optimize_metric,
        "params": best_params,
    }
    (out_dir / "best_rules.json").write_text(json.dumps(best_rules, indent=2), encoding="utf-8")
    if args.write_config:
        if args.write_config.exists() and not bool(args.overwrite):
            raise ValueError(
                f"refusing to overwrite existing rule config: {args.write_config} "
                "(use --overwrite or write a versioned file like *_v2.json)"
            )
        args.write_config.parent.mkdir(parents=True, exist_ok=True)
        args.write_config.write_text(json.dumps(best_rules, indent=2), encoding="utf-8")

    pred_rows = full_eval["predictions"]
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
            }
            for row in pred_rows
        ],
        ["take_id", "physical_object_id", "split", "expected_superclass", "predicted_superclass", "correct", "confidence_proxy", "rule_path"],
    )

    cm_rows = []
    cm = full_eval["confusion_matrix"]
    for expected in SUPERCLASSES:
        row = {"expected": expected}
        for predicted in SUPERCLASSES:
            row[predicted] = cm.get(expected, {}).get(predicted, 0)
        cm_rows.append(row)
    _write_csv(out_dir / "confusion_matrix.csv", cm_rows, ["expected", *SUPERCLASSES])

    per_class_rows = []
    for cls, vals in full_eval["per_class"].items():
        per_class_rows.append({"class": cls, **vals})
    _write_csv(out_dir / "per_class_metrics.csv", per_class_rows, ["class", "precision", "recall", "f1", "tp", "fp", "fn", "tn"])

    feature_ranges: dict[str, dict[str, float]] = {}
    numeric_cols = sorted({k for row in filtered for k in row.keys() if _safe_float(row.get(k)) is not None})
    for col in numeric_cols:
        vals = [_safe_float(r.get(col)) for r in filtered]
        nums = [v for v in vals if v is not None]
        if not nums:
            continue
        feature_ranges[col] = {"min": min(nums), "max": max(nums), "mean": sum(nums) / len(nums)}
    (out_dir / "feature_ranges.json").write_text(json.dumps(feature_ranges, indent=2), encoding="utf-8")

    split_counts: dict[str, int] = defaultdict(int)
    split_object_counts: dict[str, int] = defaultdict(int)
    object_by_split: dict[str, set[str]] = defaultdict(set)
    for row in filtered:
        split = str(row.get("split") or "")
        split_counts[split] += 1
        object_by_split[split].add(str(row.get(args.group_column) or row.get("take_id") or ""))
    for split, objs in object_by_split.items():
        split_object_counts[split] = len(objs)

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "classifier_engine": "mining_steel_ball_classification_25d_rules",
        "rule_set_source": "tuned",
        "dataset_id": args.dataset_id,
        "ml_set_id": args.ml_set_id,
        "pipeline_id": args.pipeline_id,
        "feature_source_csv": str(features_csv),
        "rows_used": len(filtered),
        "object_count": len({str(r.get(args.group_column) or r.get("take_id") or "") for r in filtered}),
        "split_distribution": {"takes": dict(split_counts), "objects": dict(split_object_counts)},
        "optimized_metric": args.optimize_metric,
        "target_split": target_split,
        "best_validation_metrics": val_eval if val_eval else best_eval,
        "train_metrics": train_eval,
        "test_metrics": test_eval,
        "overall_metrics": full_eval,
        "missing_feature_columns": missing_columns,
        "search_space": {k: {"min": v[0], "max": v[1]} for k, v in PARAM_RANGES.items()},
        "search_settings": {"search": args.search, "max_evals": args.max_evals, "seed": args.seed},
    }
    (out_dir / "tuning_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"best_rules": best_rules, "report": report, "output_dir": str(out_dir)}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tune explainable 25D ball classifier thresholds from labeled feature exports.")
    p.add_argument("--features-csv", type=Path, default=None, help="Input features CSV (from ml export-features)")
    p.add_argument("--data-dir", type=Path, default=Path("data"), help="Data root (used to locate latest export when --features-csv is omitted)")
    p.add_argument("--dataset-id", type=str, default=None, help="Dataset id (report metadata)")
    p.add_argument("--ml-set-id", type=str, default=None, help="ML set id (for CSV auto-discovery)")
    p.add_argument("--pipeline-id", type=str, default="mining_steel_ball_classification_25d", help="Pipeline id (report metadata)")
    p.add_argument("--output-dir", type=Path, required=True, help="Output directory for tuning artifacts")
    p.add_argument("--group-column", type=str, default="physical_object_id", help="Grouping column for object-safe evaluation")
    p.add_argument("--label-column", type=str, default="expected_label", help="Label column used when expected superclass is absent")
    p.add_argument("--superclass-column", type=str, default="expected_superclass", help="Superclass column if already present")
    p.add_argument("--train-split", type=str, default="train")
    p.add_argument("--validation-split", type=str, default="validation")
    p.add_argument("--test-split", type=str, default="test")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--optimize-metric", choices=["macro_f1", "balanced_accuracy", "accuracy"], default="macro_f1")
    p.add_argument("--search", choices=["grid", "random"], default="random")
    p.add_argument("--max-evals", type=int, default=2000)
    p.add_argument("--write-config", type=Path, default=None, help="Optional path to also write best_rules.json-compatible config")
    p.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing --write-config file")
    return p


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = run_tuning(args)
    except ValueError as exc:
        print(str(exc))
        return 2
    report = result["report"]
    print("Rule tuning complete")
    print(f"- output_dir: {result['output_dir']}")
    print(f"- rows_used: {report['rows_used']}")
    print(f"- objects: {report['object_count']}")
    print(f"- optimized_metric: {report['optimized_metric']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
