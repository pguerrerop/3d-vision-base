-- P0 audit trail.  These records are authoritative operational evidence, not
-- derived UI state: a label correction and an evaluation must remain explainable
-- after the materialized CSV/JSON artifacts have been regenerated.
CREATE TABLE label_correction (
    id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    physical_object_id TEXT NOT NULL,
    actor TEXT NOT NULL,
    reason TEXT,
    before_json TEXT NOT NULL,
    after_json TEXT NOT NULL,
    affected_take_ids_json TEXT NOT NULL DEFAULT '[]',
    affected_ml_set_ids_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL
);
CREATE INDEX label_correction_object ON label_correction(dataset_id, physical_object_id, created_at DESC);

CREATE TABLE evaluation_run (
    id TEXT PRIMARY KEY,
    experiment_id TEXT,
    dataset_id TEXT NOT NULL,
    classifier_id TEXT NOT NULL,
    split_strategy TEXT NOT NULL,
    recipe_snapshot_json TEXT NOT NULL DEFAULT '{}',
    metrics_json TEXT NOT NULL,
    artifacts_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE INDEX evaluation_run_dataset ON evaluation_run(dataset_id, created_at DESC);
