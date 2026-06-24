# ML Developer Validation Workflow

## Quick commands
- Happy path smoke test:
  - `make ml-smoke-test`
- Fallback-path smoke test:
  - `make ml-smoke-test-fallback`
- Pytest ML suite:
  - `python -m pytest -q tests/ml/`

## Expected PASS behavior
Happy path should:
- train/evaluate successfully
- activate deployment explicitly
- run 25D inference without fallback
- produce runtime diagnostics artifact
- update runtime stats and smoke-test result history

Fallback mode should:
- intentionally force model incompatibility
- complete pipeline with heuristic fallback
- set `fallback_used=true` and `fallback_reason`

## Interpreting diagnostics
Check `classification_runtime_diagnostics.json` for:
- deployment resolution
- compatibility checks
- backend metadata
- schema hash
- inference timing
- fallback metadata

## Runtime health counters
Endpoint: `GET /api/ml/runtime-health`
- `inference_count`
- `fallback_rate`
- `heuristic_usage_rate`
- `incompatible_deployment_count`
- `average_inference_time_ms`
- latest smoke-test summary

## Pre-demo checklist
1. Run happy path smoke test.
2. Run fallback smoke test.
3. Confirm latest smoke-test status is `HEALTHY` in Classifiers > ML System Health.
4. Confirm runtime-health counters are non-anomalous.
5. Run `python -m pytest -q tests/ml/`.
