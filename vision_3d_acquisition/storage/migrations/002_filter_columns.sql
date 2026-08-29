-- Columns the paged listing filters on that stage 0 did not project.
--
-- list_takes_paged reads some values from the acquisition metadata.json and
-- others from the dataset sidecar, and they are not always the same value the
-- TakeSummary ends up carrying. calibration_id is the clearest case: the filter
-- uses the sidecar's, falling back to the acquisition metadata's, while the
-- summary reports the one the processing result recorded. Projecting only the
-- summary's would quietly change which takes a filter returns.

ALTER TABLE take_index ADD COLUMN acquisition_created_at TEXT;
ALTER TABLE take_index ADD COLUMN acquisition_calibration_id TEXT;

-- Written by the bulk set_split action; absent from default_take_metadata, so
-- it only exists on sidecars that were explicitly given one.
ALTER TABLE take_metadata ADD COLUMN split TEXT;

-- The sidecar verbatim. Fields the listing reads but that were never promoted
-- to columns (object_count, latest_run_status, calibration_id, and whatever a
-- caller writes next) are read back with json_extract, so a sidecar key the
-- schema does not know about still filters correctly.
ALTER TABLE take_metadata ADD COLUMN payload_json TEXT NOT NULL DEFAULT '{}';

CREATE INDEX take_metadata_split ON take_metadata(split);
