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

-- session_id straight off the acquisition metadata. The existing
-- acquisition_session_id column carries the summary's, which falls back to the
-- processing result; the listing falls back to nothing, so it needs the raw one.
ALTER TABLE take_index ADD COLUMN metadata_session_id TEXT;

-- session_type exactly as the file has it, empty when the key is absent.
-- experiment_session.session_type carries the effective value, defaulting a
-- missing key to 'engineering' the way create_session would. The listing does
-- not apply that default: it compares against the raw value, so a session
-- written before the field existed does not match ?session_type=engineering.
-- Keeping both means the index can be faithful now and the inconsistency can be
-- decided on its own, instead of being silently settled by a schema default.
ALTER TABLE experiment_session ADD COLUMN session_type_raw TEXT NOT NULL DEFAULT '';

-- Written by the bulk set_split action; absent from default_take_metadata, so
-- it only exists on sidecars that were explicitly given one.
ALTER TABLE take_metadata ADD COLUMN split TEXT;

-- Sidecar fields the listing reads that are absent from default_take_metadata:
-- they only appear when something wrote them. Real columns rather than
-- json_extract, because the listing touches every filtered row and parsing a
-- ~700 byte document per row to read one key dominated the query.
ALTER TABLE take_metadata ADD COLUMN sidecar_calibration_id TEXT;
ALTER TABLE take_metadata ADD COLUMN sidecar_latest_run_status TEXT;
ALTER TABLE take_metadata ADD COLUMN sidecar_object_count INTEGER;

-- The sidecar verbatim, so a key the schema does not know about is still
-- available. Not read on the listing path; stage 2 migrates from it.
ALTER TABLE take_metadata ADD COLUMN payload_json TEXT NOT NULL DEFAULT '{}';

CREATE INDEX take_metadata_split ON take_metadata(split);
