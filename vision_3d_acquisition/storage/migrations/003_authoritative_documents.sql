-- Stage 2: dataset documents become rows instead of files.
--
-- Each table now stores the document verbatim in payload_json, with the existing
-- columns as a queryable projection of it. take_metadata already worked this way
-- and it is the pattern that makes the flip safe: reads return exactly what was
-- written, byte for byte, so moving the source of truth cannot silently drop a
-- field the schema does not know about. The columns exist to filter and join;
-- the document is what a caller gets back.

ALTER TABLE dataset ADD COLUMN payload_json TEXT NOT NULL DEFAULT '{}';
ALTER TABLE experiment_session ADD COLUMN payload_json TEXT NOT NULL DEFAULT '{}';
ALTER TABLE ml_set ADD COLUMN payload_json TEXT NOT NULL DEFAULT '{}';
ALTER TABLE ml_set_membership ADD COLUMN payload_json TEXT NOT NULL DEFAULT '{}';
ALTER TABLE physical_object ADD COLUMN payload_json TEXT NOT NULL DEFAULT '{}';

-- Ordering for the listings, which sort by created_at descending.
CREATE INDEX dataset_created ON dataset(created_at DESC);
CREATE INDEX experiment_session_created ON experiment_session(dataset_id, created_at DESC);
