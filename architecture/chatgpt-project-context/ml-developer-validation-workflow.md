# ML Developer Validation Workflow Integration

This pass integrates ML smoke testing into daily developer workflows and regression checks.

## Developer commands
- `make ml-smoke-test`
- `make ml-smoke-test-fallback`

## CI-friendly mode
`run_ml_smoke_test.py` supports `--ci-mode` and emits machine-readable JSON summary.

## Persisted smoke-test results
- Latest: `data/ml/smoke_test_results/latest.json`
- History: `data/ml/smoke_test_results/history/smoke_test_YYYYMMDD_HHMMSS.json`

## APIs
- `POST /api/ml/smoke-test/run`
- `GET /api/ml/smoke-test/latest`
- `GET /api/ml/smoke-test/history`
- `GET /api/ml/runtime-health`

## Studio integration
Classifiers includes `ML System Health` card showing latest smoke status, timing, failures, and runtime-health counters.
Dev-only controls allow manual smoke-test trigger.

## Observability model
Runtime stats include:
- `inference_count`
- `inference_success_count`
- `inference_failure_count`
- `fallback_count`
- `heuristic_usage_count`
- `incompatible_deployment_count`
- `deployment_resolution_failures`
- `calibration_compatibility_failures`
- `schema_mismatch_failures`
- `average_inference_time_ms`

## Guarantees preserved
- explicit deployment lifecycle
- immutable deployment snapshots
- deterministic runtime behavior
- heuristic fallback continuity
