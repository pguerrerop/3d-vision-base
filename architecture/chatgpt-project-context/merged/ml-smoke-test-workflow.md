# Merged: ML Smoke-Test Regression Validation

A deterministic smoke workflow is available at:
- `python scripts/ml/run_ml_smoke_test.py --data-dir data`
- `python scripts/ml/run_ml_smoke_test.py --data-dir data --force-invalid-model`

It validates:
- feature export contract
- training/evaluation artifacts
- deployment lifecycle activation semantics
- runtime inference and diagnostics artifact generation
- fallback behavior under incompatibility
- runtime observability counters

Pytest integration:
- `tests/ml/test_ml_smoke_workflow.py`

This is intended as a pre-demo/pre-refactor confidence gate for end-to-end ML operational health.
