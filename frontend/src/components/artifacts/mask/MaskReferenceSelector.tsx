import { describeReferenceCandidate } from "./maskReferenceResolution";
import type { MaskReferenceCandidate } from "./maskRenderingTypes";

type Props = {
  candidates: MaskReferenceCandidate[];
  selectedReferenceId?: string | null;
  defaultReferenceId?: string | null;
  onChange?: (referenceId: string) => void;
};

export default function MaskReferenceSelector({
  candidates,
  selectedReferenceId = null,
  defaultReferenceId = null,
  onChange,
}: Props) {
  if (!candidates.length) {
    return (
      <div className="mask-reference-selector readonly">
        <span>Reference</span>
        <strong>Unavailable</strong>
      </div>
    );
  }
  if (candidates.length === 1) {
    return (
      <div className="mask-reference-selector readonly">
        <span>Reference</span>
        <strong>{describeReferenceCandidate(candidates[0], selectedReferenceId, defaultReferenceId)}</strong>
      </div>
    );
  }
  return (
    <label className="mask-reference-selector">
      <span>Reference</span>
      <select
        aria-label="Mask visual reference"
        onChange={(event) => onChange?.(event.target.value)}
        value={selectedReferenceId ?? candidates[0]?.id ?? ""}
      >
        {candidates.map((candidate) => (
          <option key={candidate.id} value={candidate.id}>
            {describeReferenceCandidate(candidate, selectedReferenceId, defaultReferenceId)}
          </option>
        ))}
      </select>
    </label>
  );
}
