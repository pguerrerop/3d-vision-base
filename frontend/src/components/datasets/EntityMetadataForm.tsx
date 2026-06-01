import { type ChangeEvent } from "react";

export type SessionForm = {
  name: string;
  notes: string;
  tags: string;
  description: string;
  acquisition_type: string;
  calibration_id: string;
  sensor_metadata: string;
  conveyor_metadata: string;
  lighting_metadata: string;
};

export default function EntityMetadataForm({ value, onChange }: { value: SessionForm; onChange: (next: SessionForm) => void }) {
  const set = (key: keyof SessionForm) => (event: ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    onChange({ ...value, [key]: event.target.value });
  };
  return (
    <section className="entity-section">
      <h4>Editable Metadata</h4>
      <label className="field-label">Name<input value={value.name} onChange={set("name")} /></label>
      <label className="field-label">Description<textarea value={value.description} onChange={set("description")} rows={3} /></label>
      <label className="field-label">Notes<textarea value={value.notes} onChange={set("notes")} rows={3} /></label>
      <label className="field-label">Tags (comma-separated)<input value={value.tags} onChange={set("tags")} /></label>
      <small>Example: wet_surface, benchmark, calibration_reference</small>
      <label className="field-label">Acquisition type<input value={value.acquisition_type} onChange={set("acquisition_type")} /></label>
      <label className="field-label">Calibration ID<input value={value.calibration_id} onChange={set("calibration_id")} /></label>
      <details className="entity-advanced-metadata">
        <summary>Advanced metadata</summary>
        <label className="field-label">Sensor metadata (JSON)<textarea value={value.sensor_metadata} onChange={set("sensor_metadata")} rows={3} /></label>
        <label className="field-label">Conveyor metadata (JSON)<textarea value={value.conveyor_metadata} onChange={set("conveyor_metadata")} rows={3} /></label>
        <label className="field-label">Lighting metadata (JSON)<textarea value={value.lighting_metadata} onChange={set("lighting_metadata")} rows={3} /></label>
      </details>
    </section>
  );
}
