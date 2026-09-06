-- Controlled vocabulary for physical-object labels.
--
-- Replaces the two drifting, contradictory sources that caused a real bug
-- (config/label_normalization/mining_balls_v1.json vs.
-- vision_3d_acquisition/ml/label_normalization.py disagreeing on "chica",
-- and "cadena" mapped to the wrong superclass): one normalized_class maps to
-- exactly one superclass, enforced by the primary key and read by both the
-- label-correction transaction and the correction UI's typeahead.
CREATE TABLE label_taxonomy (
    normalized_class TEXT PRIMARY KEY,
    superclass TEXT NOT NULL,
    is_uncertain INTEGER NOT NULL DEFAULT 0,
    notes TEXT,
    updated_at TEXT NOT NULL,
    updated_by TEXT
);
CREATE INDEX label_taxonomy_superclass ON label_taxonomy(superclass);
