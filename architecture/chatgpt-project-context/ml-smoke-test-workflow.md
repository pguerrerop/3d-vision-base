# ML Smoke-Test Workflow

## Purpose
`scripts/ml/run_ml_smoke_test.py` provides a deterministic end-to-end regression check for:
Dataset -> Feature Export -> Train -> Evaluate -> Deploy -> Run Pipeline -> Runtime Diagnostics.

## Workflow Phases
1. Create deterministic synthetic 25D dataset/take.
2. Export features and validate schema + required columns.
3. Train a lightweight classifier (logistic regression).
4. Validate evaluation artifacts (`confusion_matrix.csv`, `per_class_metrics.csv`, `evaluation_summary.json`).
5. Create deployment snapshot.
6. Activate deployment and assert single-active semantics for target tuple.
7. Run 25D pipeline on a new deterministic synthetic take.
8. Validate runtime diagnostics and operational observability counters.

## Modes
- Happy path (default): deployed model should run with `fallback_used = false`.
- Fallback path (`--force-invalid-model`): model compatibility is intentionally broken; pipeline must complete via heuristic fallback with diagnostics.

## Runtime Assertions
Smoke test validates:
- `classification_runtime_diagnostics.json` exists.
- Deployment resolution and compatibility checks are present.
- Feature schema hash + backend metadata + inference timing are present.
- Fallback fields are correct per mode.

## Observability Assertions
Smoke test validates `data/ml/runtime_stats/YYYYMMDD.json` updates:
- `inference_count`
- `fallback_count`
- `heuristic_usage_count`
- `incompatible_deployment_count`
- `average_inference_time_ms`

## Regression Contract
The smoke workflow does not add new ML capabilities. It validates current architecture behavior and operational safety semantics.
