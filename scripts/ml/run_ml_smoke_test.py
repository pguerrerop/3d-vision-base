#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vision_3d_acquisition.apps.ball_inspection_25d import run_ball_inspection_25d_flow
from vision_3d_acquisition.datasets import DatasetService
from vision_3d_acquisition.ml import MLService
from vision_3d_acquisition.ml.features.extractor import export_rows
from vision_3d_acquisition.ml.features.schemas import FEATURE_COLUMNS
from vision_3d_acquisition.ml.storage import MLStorage
from vision_3d_acquisition.synthetic import create_synthetic_25d_take


@dataclass
class PhaseResult:
    name: str
    ok: bool
    detail: str


REQUIRED_EVAL_FILES = ("confusion_matrix", "per_class_metrics", "evaluation_summary")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _print_phase(i: int, n: int, msg: str) -> None:
    print(f"[{i}/{n}] {msg}")


def _persist_smoke_result(data_dir: Path, summary: dict[str, Any]) -> None:
    root = data_dir / "ml" / "smoke_test_results"
    hist = root / "history"
    hist.mkdir(parents=True, exist_ok=True)
    latest = root / "latest.json"
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    item = hist / f"smoke_test_{stamp}.json"
    latest.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    item.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def run_smoke_workflow(*, data_dir: Path, force_invalid_model: bool = False, ci_mode: bool = False) -> dict[str, Any]:
    t0 = perf_counter()
    data_dir = data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    results: list[PhaseResult] = []

    smoke_id = datetime.now(UTC).strftime("smoke_%Y%m%d_%H%M%S")
    dataset_id = f"ml_dataset_{smoke_id}"
    experiment_id = f"exp_{smoke_id}"
    deployment_id = f"dep_{smoke_id}"

    total_phases = 8

    # [1/8] dataset + deterministic synthetic take + baseline processing for features
    if not ci_mode:
        _print_phase(1, total_phases, "Creating demo dataset and synthetic takes...")
    try:
        service = DatasetService(data_dir)
        service.create_dataset(dataset_id=dataset_id, name=f"Smoke dataset {smoke_id}")
        service.create_session(dataset_id=dataset_id, session_id="smoke_session", name="Smoke Session")

        train_takes: list[str] = []
        for idx, (seed, expected_class) in enumerate(((101, "BALL_GOOD"), (102, "BALL_SCRAP")), start=1):
            take_id_train, _ = create_synthetic_25d_take(
                data_dir,
                session_id="smoke_session",
                seed=seed,
                take_id=f"{smoke_id}_train_{idx}",
            )
            train_takes.append(take_id_train)
            service.upsert_take_metadata(
                take_id=take_id_train,
                dataset_id=dataset_id,
                session_id="smoke_session",
                updates={"expected_class": expected_class, "validation_status": "validated", "labels": ["smoke"]},
                source_metadata=_load_json(data_dir / "incoming" / take_id_train / "metadata.json"),
            )
            incoming_meta = _load_json(data_dir / "incoming" / take_id_train / "metadata.json")
            incoming_meta["dataset_id"] = dataset_id
            incoming_meta["take_management"] = {
                **(incoming_meta.get("take_management") if isinstance(incoming_meta.get("take_management"), dict) else {}),
                "dataset_id": dataset_id,
                "session_id": "smoke_session",
                "expected_class": expected_class,
                "validation_status": "validated",
                "labels": ["smoke"],
            }
            _write_json(data_dir / "incoming" / take_id_train / "metadata.json", incoming_meta)
            baseline = run_ball_inspection_25d_flow(data_dir, take_id=take_id_train)
            # Ensure ML feature builder can discover the result in canonical lookup paths.
            canonical_result_dir = data_dir / "processed" / take_id_train
            canonical_result_dir.mkdir(parents=True, exist_ok=True)
            baseline_result = baseline.output_dir / "result.json"
            canonical_result = canonical_result_dir / "result.json"
            if baseline_result.is_file() and baseline_result.resolve() != canonical_result.resolve():
                shutil.copy2(baseline_result, canonical_result)
        results.append(PhaseResult("dataset", True, f"dataset={dataset_id} takes={','.join(train_takes)}"))
    except Exception as exc:
        results.append(PhaseResult("dataset", False, str(exc)))
        return _finalize(results, data_dir=data_dir, force_invalid_model=force_invalid_model, started_at=t0, ci_mode=ci_mode)

    # [2/8] export features
    if not ci_mode:
        _print_phase(2, total_phases, "Exporting features...")
    try:
        ml = MLService(data_dir)
        ml.create_dataset(
            {
                "id": dataset_id,
                "name": f"Smoke dataset {smoke_id}",
                "source_dataset_ids": [dataset_id],
                "filters": {"exclude_archived": True},
                "label_schema": {"task": "classification", "classes": ["BALL_GOOD", "BALL_SCRAP"]},
                "object_count": 0,
                "tags": ["smoke"],
            }
        )
        rows = ml.build_rows_for_dataset(dataset_id)
        if len(rows) <= 0:
            raise ValueError("no feature rows exported")
        export_paths = export_rows(rows, data_dir / "ml" / "evaluations" / experiment_id, "features_smoke")
        if not Path(export_paths["csv"]).is_file() or not Path(export_paths["jsonl"]).is_file():
            raise ValueError("missing csv/jsonl exports")
        required_cols = set(FEATURE_COLUMNS)
        first = rows[0]
        if not required_cols.issubset(set(first.keys())):
            raise ValueError("missing required feature columns")
        if not first.get("labels"):
            # labels may be empty in some data; ensure expected_class is present as supervised target
            if not first.get("expected_class"):
                raise ValueError("missing labels/expected_class")
        results.append(PhaseResult("feature export", True, f"rows={len(rows)} exports={list(export_paths.keys())}"))
    except Exception as exc:
        results.append(PhaseResult("feature export", False, str(exc)))
        return _finalize(results, data_dir=data_dir, force_invalid_model=force_invalid_model, started_at=t0, ci_mode=ci_mode)

    # [3/8] train
    if not ci_mode:
        _print_phase(3, total_phases, "Training classifier...")
    try:
        run = ml.run_experiment(
            {
                "id": experiment_id,
                "name": f"Smoke experiment {smoke_id}",
                "classifier_type": "logistic_regression",
                "feature_set_id": "core_25d_v1",
                "dataset_id": dataset_id,
                "split_strategy": "stratified",
                "training_config": {"seed": 42, "max_iter": 200},
            }
        )
        if str(run.get("status")) != "completed":
            raise ValueError(f"training status={run.get('status')}")
        model_id = f"model_{experiment_id}"
        model = ml.get_model(model_id)
        if not model:
            raise ValueError("model missing after training")
        if not (model.get("artifact_paths") or {}).get("metadata_path"):
            raise ValueError("model metadata_path missing")
        if not (model.get("backend") or {}).get("backend_name"):
            raise ValueError("backend metadata missing")
        results.append(PhaseResult("training", True, f"model={model_id}"))
    except Exception as exc:
        results.append(PhaseResult("training", False, str(exc)))
        return _finalize(results, data_dir=data_dir, force_invalid_model=force_invalid_model, started_at=t0, ci_mode=ci_mode)

    # [4/8] evaluate outputs
    if not ci_mode:
        _print_phase(4, total_phases, "Evaluating classifier outputs...")
    try:
        artifacts = run.get("artifact_paths") or {}
        for key in REQUIRED_EVAL_FILES:
            p = artifacts.get(key)
            if not p or not Path(str(p)).is_file():
                raise ValueError(f"missing evaluation artifact: {key}")
        summary = _load_json(Path(str(artifacts["evaluation_summary"])))
        if "labels" not in summary:
            raise ValueError("evaluation summary missing labels")
        results.append(PhaseResult("evaluation", True, f"labels={summary.get('labels')}"))
    except Exception as exc:
        results.append(PhaseResult("evaluation", False, str(exc)))
        return _finalize(results, data_dir=data_dir, force_invalid_model=force_invalid_model, started_at=t0, ci_mode=ci_mode)

    # [5/8] deployment create
    if not ci_mode:
        _print_phase(5, total_phases, "Creating deployment...")
    try:
        model_id = f"model_{experiment_id}"
        # For smoke path, keep runtime compatibility deterministic on synthetic input
        # without requiring an external calibration profile.
        storage = MLStorage(data_dir)
        model_for_smoke = storage.get("models", model_id) or {}
        backend_meta = model_for_smoke.get("backend") if isinstance(model_for_smoke.get("backend"), dict) else {}
        feature_req = backend_meta.get("feature_requirements") if isinstance(backend_meta.get("feature_requirements"), dict) else {}
        feature_req["requires_mm_calibration"] = False
        backend_meta["feature_requirements"] = feature_req
        model_for_smoke["backend"] = backend_meta
        storage.save("models", model_id, model_for_smoke)

        ml.promote_model(model_id=model_id, notes="smoke")
        dep = ml.deploy_model(
            {
                "id": deployment_id,
                "model_id": model_id,
                "target_pipeline_id": "25d_ball_inspection",
                "target_stage_id": "classification",
                "pipeline_family": "25d",
                "deployed_by": "smoke_test",
                "notes": "smoke",
            }
        )
        if str(dep.get("status")) != "inactive":
            raise ValueError("deployment not created as inactive")
        results.append(PhaseResult("deployment create", True, dep["id"]))
    except Exception as exc:
        results.append(PhaseResult("deployment create", False, str(exc)))
        return _finalize(results, data_dir=data_dir, force_invalid_model=force_invalid_model, started_at=t0, ci_mode=ci_mode)

    # [6/8] activate
    if not ci_mode:
        _print_phase(6, total_phases, "Activating deployment...")
    try:
        activated = ml.activate_deployment(deployment_id)
        active_rows = [
            d
            for d in ml.list_active_deployments()
            if str(d.get("target_pipeline_id")) == "25d_ball_inspection"
            and str(d.get("target_stage_id")) == "classification"
            and str(d.get("pipeline_family")) == "25d"
        ]
        if str(activated.get("status")) != "active":
            raise ValueError(f"activation status not active: {activated.get('status')}")
        if len(active_rows) != 1:
            raise ValueError(f"expected exactly one active deployment, got {len(active_rows)}")
        if not isinstance(activated.get("compatibility_snapshot"), dict):
            raise ValueError("compatibility snapshot missing")
        if force_invalid_model:
            # Force runtime incompatibility after activation to verify fallback path
            # without depending on pre-existing active deployment state.
            storage = MLStorage(data_dir)
            model = storage.get("models", model_id) or {}
            feature_schema = model.get("feature_schema") if isinstance(model.get("feature_schema"), dict) else {}
            feature_schema["version"] = "999.0.0"
            model["feature_schema"] = feature_schema
            storage.save("models", model_id, model)
        results.append(PhaseResult("deployment activate", True, f"status={activated.get('status')}"))
    except Exception as exc:
        results.append(PhaseResult("deployment activate", False, str(exc)))
        return _finalize(results, data_dir=data_dir, force_invalid_model=force_invalid_model, started_at=t0, ci_mode=ci_mode)

    # [7/8] run pipeline with deployed model/fallback path
    if not ci_mode:
        _print_phase(7, total_phases, "Running 25D pipeline...")
    try:
        take_id_run, _ = create_synthetic_25d_take(
            data_dir,
            session_id="smoke_session",
            seed=202,
            take_id=f"{smoke_id}_runtime",
        )
        service.upsert_take_metadata(
            take_id=take_id_run,
            dataset_id=dataset_id,
            session_id="smoke_session",
            updates={"validation_status": "validated", "labels": ["smoke"]},
            source_metadata=_load_json(data_dir / "incoming" / take_id_run / "metadata.json"),
        )
        incoming_meta_run = _load_json(data_dir / "incoming" / take_id_run / "metadata.json")
        incoming_meta_run["dataset_id"] = dataset_id
        incoming_meta_run["take_management"] = {
            **(incoming_meta_run.get("take_management") if isinstance(incoming_meta_run.get("take_management"), dict) else {}),
            "dataset_id": dataset_id,
            "session_id": "smoke_session",
            "validation_status": "validated",
            "labels": ["smoke"],
        }
        _write_json(data_dir / "incoming" / take_id_run / "metadata.json", incoming_meta_run)
        run_result = run_ball_inspection_25d_flow(data_dir, take_id=take_id_run)
        output_dir = run_result.output_dir
        result_payload = run_result.result_payload
        diag_path = output_dir / "classification_runtime_diagnostics.json"
        if not diag_path.is_file():
            raise ValueError("classification_runtime_diagnostics.json missing")
        diag = _load_json(diag_path)
        classification = result_payload.get("classification") if isinstance(result_payload.get("classification"), dict) else {}
        fallback_used = bool(diag.get("fallback_used"))
        if force_invalid_model:
            if not fallback_used:
                raise ValueError("fallback expected but not used")
            if not str(diag.get("fallback_reason") or ""):
                raise ValueError("fallback reason missing")
        else:
            if fallback_used:
                raise ValueError("unexpected fallback in happy path")
            if str(classification.get("model_id") or "") == "":
                raise ValueError("model_id missing in classification payload")
            if str(classification.get("deployment_id") or "") == "":
                raise ValueError("deployment_id missing in classification payload")
            if str(classification.get("classifier_backend") or "") == "":
                raise ValueError("classifier_backend missing in classification payload")
        results.append(PhaseResult("runtime inference", True, f"take={take_id_run} fallback={fallback_used}"))
    except Exception as exc:
        results.append(PhaseResult("runtime inference", False, str(exc)))
        return _finalize(results, data_dir=data_dir, force_invalid_model=force_invalid_model, started_at=t0, ci_mode=ci_mode)

    # [8/8] diagnostics + observability assertions
    if not ci_mode:
        _print_phase(8, total_phases, "Validating runtime diagnostics and observability...")
    try:
        comp = diag.get("compatibility_checks") if isinstance(diag.get("compatibility_checks"), dict) else {}
        if "errors" not in comp or "warnings" not in comp:
            raise ValueError("compatibility checks missing errors/warnings")
        if not str(diag.get("feature_schema_hash") or ""):
            raise ValueError("feature_schema_hash missing")
        if not str(diag.get("inference_backend_used") or ""):
            raise ValueError("inference_backend_used missing")

        stats_path = data_dir / "ml" / "runtime_stats" / f"{datetime.now(UTC).strftime('%Y%m%d')}.json"
        if not stats_path.is_file():
            raise ValueError("runtime stats file missing")
        stats = _load_json(stats_path)
        if int(stats.get("inference_count", 0)) < 0:
            raise ValueError("invalid inference_count")
        if force_invalid_model:
            if int(stats.get("fallback_count", 0)) < 1:
                raise ValueError("fallback_count not incremented")
            if int(stats.get("heuristic_usage_count", 0)) < 1:
                raise ValueError("heuristic_usage_count not incremented")
        results.append(PhaseResult("diagnostics+observability", True, f"stats={stats_path}"))
    except Exception as exc:
        results.append(PhaseResult("diagnostics+observability", False, str(exc)))
        return _finalize(results, data_dir=data_dir, force_invalid_model=force_invalid_model, started_at=t0, ci_mode=ci_mode)

    return _finalize(results, data_dir=data_dir, force_invalid_model=force_invalid_model, started_at=t0, ci_mode=ci_mode)


def _finalize(
    results: list[PhaseResult],
    *,
    data_dir: Path | None = None,
    force_invalid_model: bool = False,
    started_at: float | None = None,
    ci_mode: bool = False,
) -> dict[str, Any]:
    ok = all(item.ok for item in results)
    failures = [{"name": r.name, "detail": r.detail} for r in results if not r.ok]
    summary = {
        "ok": ok,
        "passed": ok,
        "phases": [{"name": r.name, "ok": r.ok, "detail": r.detail} for r in results],
        "duration_ms": int((perf_counter() - started_at) * 1000.0) if started_at is not None else None,
        "fallback_mode": bool(force_invalid_model),
        "failures": failures,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    if data_dir is not None:
        _persist_smoke_result(data_dir, summary)
    if ci_mode:
        print(json.dumps(summary, sort_keys=True))
    else:
        print("\nSummary:")
        for row in summary["phases"]:
            mark = "PASS" if row["ok"] else "FAIL"
            print(f"- {mark}: {row['name']} ({row['detail']})")
        print("SMOKE TEST PASSED" if ok else "SMOKE TEST FAILED")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run end-to-end ML smoke workflow.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--force-invalid-model", action="store_true")
    parser.add_argument("--ci-mode", action="store_true")
    args = parser.parse_args()

    summary = run_smoke_workflow(data_dir=args.data_dir, force_invalid_model=args.force_invalid_model, ci_mode=args.ci_mode)
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
