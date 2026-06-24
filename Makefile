.PHONY: ml-smoke-test ml-smoke-test-fallback

ml-smoke-test:
	@. .venv/bin/activate && python scripts/ml/run_ml_smoke_test.py --data-dir data

ml-smoke-test-fallback:
	@. .venv/bin/activate && python scripts/ml/run_ml_smoke_test.py --data-dir data --force-invalid-model
