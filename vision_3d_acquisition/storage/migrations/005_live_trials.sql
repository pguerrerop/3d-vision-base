CREATE TABLE live_trial (
    id TEXT PRIMARY KEY,
    deployment_id TEXT NOT NULL,
    recipe_snapshot_json TEXT NOT NULL DEFAULT '{}',
    runtime_health_json TEXT NOT NULL DEFAULT '{}',
    notes TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    started_at TEXT NOT NULL,
    ended_at TEXT
);
CREATE INDEX live_trial_status ON live_trial(status, started_at DESC);
