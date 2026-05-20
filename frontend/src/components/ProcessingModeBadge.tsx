import type { ProcessingMode } from "../api/client";

type Props = {
  mode?: ProcessingMode | null;
};

export default function ProcessingModeBadge({ mode }: Props) {
  if (!mode) {
    return <span className="processing-badge processing-unknown">Unprocessed</span>;
  }
  return <span className={`processing-badge processing-${mode}`}>{mode === "mock" ? "Mock processing" : "Real processing"}</span>;
}
