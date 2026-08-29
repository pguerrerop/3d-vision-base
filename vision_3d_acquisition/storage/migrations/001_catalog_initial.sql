-- Catalog schema, stage 0.
--
-- Two classes of table live here and they carry different guarantees:
--
--   derived      take_index, take_modality, process_run
--                A projection of what is already on disk. Rebuildable in full
--                with `sensor-studio index rebuild --full`. Safe to delete.
--
--   authoritative dataset, experiment_session, take_metadata, take_label,
--                ml_set, ml_set_membership, physical_object,
--                physical_object_observation, object_annotation
--                Written by operators through the API. Stage 0 still imports
--                these from the JSON files under data/datasets/, which remain
--                the source of truth until stage 2 flips the direction.
--
-- No point cloud, PNG, full result.json, run artifact or feature CSV belongs
-- in this database. Columns hold paths, never bytes.

-- ---------------------------------------------------------------- authoritative

CREATE TABLE dataset (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT,
    notes       TEXT,
    tags_json   TEXT NOT NULL DEFAULT '[]',
    created_at  TEXT,
    updated_at  TEXT
);

CREATE TABLE experiment_session (
    dataset_id   TEXT NOT NULL REFERENCES dataset(id) ON DELETE CASCADE,
    id           TEXT NOT NULL,
    name         TEXT,
    session_type TEXT NOT NULL DEFAULT 'engineering',
    description  TEXT,
    notes        TEXT,
    tags_json    TEXT NOT NULL DEFAULT '[]',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at   TEXT,
    updated_at   TEXT,
    PRIMARY KEY (dataset_id, id)
);

-- Replaces datasets/<dataset>/sessions/<session>/takes/<take>/metadata.json
CREATE TABLE take_metadata (
    take_id               TEXT PRIMARY KEY,
    dataset_id            TEXT,
    session_id            TEXT,
    friendly_name         TEXT,
    notes                 TEXT,
    operator_notes        TEXT,
    session_notes         TEXT,
    validation_status     TEXT NOT NULL DEFAULT 'unreviewed',
    normalized_class      TEXT,
    normalization_version TEXT,
    expected_class        TEXT,
    expected_diameter_mm  REAL,
    expected_count        INTEGER,
    physical_object_id    TEXT,
    acquisition_group_id  TEXT,
    reference_type        TEXT,
    is_reference          INTEGER NOT NULL DEFAULT 0,
    is_golden_sample      INTEGER NOT NULL DEFAULT 0,
    archived              INTEGER NOT NULL DEFAULT 0,
    archived_at           TEXT,
    archived_reason       TEXT,
    created_at            TEXT,
    updated_at            TEXT
);

CREATE INDEX take_metadata_dataset ON take_metadata(dataset_id, session_id);
CREATE INDEX take_metadata_object ON take_metadata(physical_object_id);
CREATE INDEX take_metadata_validation ON take_metadata(validation_status);

-- tags / semantic_labels / superclass_labels / categories / labels are JSON
-- arrays on disk, filtered with .lower() in Python (api/filesystem.py:238-255).
CREATE TABLE take_label (
    take_id    TEXT NOT NULL REFERENCES take_metadata(take_id) ON DELETE CASCADE,
    kind       TEXT NOT NULL
               CHECK (kind IN ('tag', 'semantic', 'superclass', 'category', 'label')),
    value      TEXT NOT NULL,
    value_norm TEXT GENERATED ALWAYS AS (lower(value)) STORED,
    PRIMARY KEY (take_id, kind, value)
);

CREATE INDEX take_label_lookup ON take_label(kind, value_norm);

-- A take_id may have metadata under more than one session directory. The API
-- resolves that by taking the first membership it happens to iterate over
-- (datasets/service.py:908-929), so take_metadata records that same winner and
-- every other copy lands here instead of being silently dropped. A row in this
-- table is drift to be cleaned up, not a supported state.
CREATE TABLE take_metadata_conflict (
    take_id    TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    path       TEXT NOT NULL,
    PRIMARY KEY (take_id, dataset_id, session_id)
);

CREATE TABLE object_annotation (
    take_id              TEXT NOT NULL REFERENCES take_metadata(take_id) ON DELETE CASCADE,
    id                   TEXT NOT NULL,
    source_stage         TEXT,
    source_artifact_id   TEXT,
    candidate_id         TEXT,
    bbox_json            TEXT,
    centroid_json        TEXT,
    contour_ref          TEXT,
    labels_json          TEXT NOT NULL DEFAULT '[]',
    expected_class       TEXT,
    expected_diameter_mm REAL,
    notes                TEXT,
    validation_status    TEXT NOT NULL DEFAULT 'unreviewed',
    created_at           TEXT,
    updated_at           TEXT,
    PRIMARY KEY (take_id, id)
);

CREATE TABLE ml_set (
    dataset_id  TEXT NOT NULL,
    id          TEXT NOT NULL,
    name        TEXT,
    description TEXT,
    task_type   TEXT,
    notes       TEXT,
    created_at  TEXT,
    updated_at  TEXT,
    PRIMARY KEY (dataset_id, id)
);

-- Replaces ml_sets/<set>/memberships.json, rewritten whole on every edit today
-- (datasets/service.py:379).
CREATE TABLE ml_set_membership (
    dataset_id            TEXT NOT NULL,
    ml_set_id             TEXT NOT NULL,
    take_id               TEXT NOT NULL,
    split                 TEXT NOT NULL DEFAULT 'unassigned',
    "include"             INTEGER NOT NULL DEFAULT 1,
    default_trainable     INTEGER,
    trainable             INTEGER,
    physical_object_id    TEXT,
    expected_label        TEXT,
    expected_class        TEXT,
    expected_subclass     TEXT,
    raw_label             TEXT,
    label_policy          TEXT,
    review_required       INTEGER,
    normalization_version TEXT,
    source_row            TEXT,
    notes                 TEXT,
    measurements_mm_json  TEXT NOT NULL DEFAULT '{}',
    extra_fields_json     TEXT NOT NULL DEFAULT '{}',
    created_at            TEXT,
    updated_at            TEXT,
    PRIMARY KEY (dataset_id, ml_set_id, take_id),
    FOREIGN KEY (dataset_id, ml_set_id) REFERENCES ml_set(dataset_id, id) ON DELETE CASCADE
);

CREATE INDEX ml_set_membership_take ON ml_set_membership(take_id);
CREATE INDEX ml_set_membership_split ON ml_set_membership(dataset_id, ml_set_id, split);

CREATE TABLE physical_object (
    dataset_id             TEXT NOT NULL,
    id                     TEXT NOT NULL,
    raw_operator_label     TEXT,
    normalized_class       TEXT,
    superclass             TEXT,
    d1_mm                  REAL,
    d2_mm                  REAL,
    d3_mm                  REAL,
    diameter_mean_mm       REAL,
    diameter_range_mm      REAL,
    annotation_confidence  TEXT,
    needs_review           INTEGER NOT NULL DEFAULT 0,
    source_type            TEXT,
    source_row_index       INTEGER,
    notes                  TEXT,
    tags_json              TEXT NOT NULL DEFAULT '[]',
    source_session_ids_json TEXT NOT NULL DEFAULT '[]',
    created_at             TEXT,
    PRIMARY KEY (dataset_id, id)
);

CREATE TABLE physical_object_observation (
    dataset_id TEXT NOT NULL,
    object_id  TEXT NOT NULL,
    take_id    TEXT NOT NULL,
    PRIMARY KEY (dataset_id, object_id, take_id)
);

CREATE INDEX physical_object_observation_take ON physical_object_observation(take_id);

-- ---------------------------------------------------------------------- derived

-- Projection of incoming/<take>/metadata.json + processed/<take>/result.json.
CREATE TABLE take_index (
    take_id                TEXT PRIMARY KEY,
    status                 TEXT NOT NULL,
    has_ready              INTEGER NOT NULL DEFAULT 0,
    has_done               INTEGER NOT NULL DEFAULT 0,
    created_at             TEXT,
    processed_at           TEXT,
    decision               TEXT,
    engine                 TEXT,
    object_count           INTEGER,
    ball_count             INTEGER,
    plane_found            INTEGER,
    processing_time_ms     REAL,
    warning_count          INTEGER NOT NULL DEFAULT 0,
    calibration_id         TEXT,
    frame_count            INTEGER,
    frameset_id            TEXT,
    acquisition_session_id TEXT,
    processed_class_label  TEXT,
    processed_superclass   TEXT,
    latest_run_status      TEXT,
    thumbnail_path         TEXT,

    -- Serialized TakeSummary: the response is read back, not recomposed.
    summary_json           TEXT NOT NULL,

    -- Invalidation keys for the incremental reindex. The mtimes fold in the
    -- containing directory's mtime, so adding or removing an asset, a READY or
    -- a DONE marker invalidates the row too.
    metadata_mtime         REAL,
    metadata_size          INTEGER,
    result_mtime           REAL,
    result_size            INTEGER,
    status_mtime           REAL,
    status_size            INTEGER,
    management_hash        TEXT,
    projection_version     INTEGER NOT NULL,
    indexed_at             TEXT NOT NULL
);

CREATE INDEX take_index_created ON take_index(created_at DESC);
CREATE INDEX take_index_status ON take_index(status);

CREATE TABLE take_modality (
    take_id  TEXT NOT NULL REFERENCES take_index(take_id) ON DELETE CASCADE,
    modality TEXT NOT NULL,
    PRIMARY KEY (take_id, modality)
);

-- Replaces data/processes/index/runs.json.
CREATE TABLE process_run (
    run_id                 TEXT PRIMARY KEY,
    take_id                TEXT,
    pipeline_instance_id   TEXT,
    pipeline_id            TEXT,
    pipeline_family        TEXT NOT NULL,
    status                 TEXT NOT NULL,
    created_at             TEXT NOT NULL,
    path                   TEXT,
    recipe_id              TEXT,
    recipe_version_id      TEXT,
    config_snapshot_hash   TEXT,
    source_id              TEXT,
    acquisition_group_id   TEXT,
    calibration_profile_id TEXT,
    execution_mode         TEXT NOT NULL DEFAULT 'full_run',
    parent_run_id          TEXT,
    parent_take_id         TEXT,
    partial_rerun_plan_id  TEXT,
    boundary_stage_id      TEXT,
    boundary_unit_id       TEXT,
    selected_unit_id       TEXT,
    comparison_id          TEXT,
    summary_json           TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX process_run_take ON process_run(take_id, pipeline_family, created_at DESC);
CREATE INDEX process_run_parent ON process_run(parent_run_id);

-- last_full_scan_at, last_stale_scan_at, projection_version
CREATE TABLE index_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
